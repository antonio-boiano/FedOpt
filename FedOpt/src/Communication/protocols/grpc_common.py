#!/usr/bin/env python3
"""
Shared helpers and constants for gRPC protocol implementations.
"""

import json
import asyncio
import socket
import logging
import os
import sys
import importlib.util
from typing import Any, Optional, List, Tuple

import grpc

logger = logging.getLogger("FedOpt")

# gRPC configuration constants
TIMEOUT_GRPC = 120
RECONNECT_MAX = 60
# Make keepalive less aggressive to avoid ping timeouts on shaky links
KEEPALIVE_TIME = 600 * 1000  # 10 minutes in ms
KEEPALIVE_TIMEOUT = 60 * 1000  # 60 seconds to respond to ping
MIN_PING_INTERVAL = 600 * 1000  # server minimum interval between pings
MAX_PINGS_WITHOUT_DATA = 0
MAX_MESSAGE_SIZE = 16 * 1024 * 1024  # 16 MB
MAX_RETRY = 5
RETRY_BASE_DELAY = 1.0


def get_grpc_modules():
    """Load protobuf modules only from the canonical Communication/gRPC folder."""
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "gRPC"))
    pb2_path = os.path.join(base_dir, "FedOpt_pb2.py")
    pb2_grpc_path = os.path.join(base_dir, "FedOpt_pb2_grpc.py")

    if not os.path.exists(pb2_path) or not os.path.exists(pb2_grpc_path):
        raise ImportError(
            f"gRPC modules not found at {base_dir}. "
            "Regenerate with: python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. FedOpt.proto"
        )

    def _load(name: str, path: str):
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module {name} from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    pb2_module = _load("FedOpt_pb2", pb2_path)
    sys.modules["FedOpt_pb2"] = pb2_module  # ensure grpc module resolves the same copy
    pb2_grpc_module = _load("FedOpt_pb2_grpc", pb2_grpc_path)

    return pb2_module, pb2_grpc_module


def common_grpc_options() -> List[Tuple[str, Any]]:
    return [
        ("grpc.max_receive_message_length", MAX_MESSAGE_SIZE),
        ("grpc.max_send_message_length", MAX_MESSAGE_SIZE),
        ("grpc.keepalive_time_ms", KEEPALIVE_TIME),
        ("grpc.keepalive_timeout_ms", KEEPALIVE_TIMEOUT),
        ("grpc.keepalive_permit_without_calls", 1),
        ("grpc.http2.max_pings_without_data", MAX_PINGS_WITHOUT_DATA),
    ]


def server_grpc_options() -> List[Tuple[str, Any]]:
    return common_grpc_options() + [
        ("grpc.http2.min_recv_ping_interval_without_data_ms", MIN_PING_INTERVAL),
    ]


async def retry_rpc(name: str, fn):
    """Retry helper for transient gRPC failures."""
    delay = RETRY_BASE_DELAY
    for attempt in range(MAX_RETRY):
        try:
            return await fn()
        except grpc.aio.AioRpcError as exc:
            if exc.code() in (
                grpc.StatusCode.UNAVAILABLE,
                grpc.StatusCode.DEADLINE_EXCEEDED,
            ) and attempt < MAX_RETRY - 1:
                logger.warning(f"{name} failed with {exc.code().name}, retrying in {delay}s")
                await asyncio.sleep(delay)
                delay = min(delay * 2, RECONNECT_MAX)
                continue
            raise


def build_client_id(address: Optional[str], port: Optional[str], peer: str = "") -> str:
    if address and port:
        return f"{address}:{port}"
    if peer:
        return peer.replace("ipv4:", "").replace("ipv6:", "")
    return "unknown"


def serialize_message(pb2, message, address: str, port: Any):
    return pb2.Message(
        payload=json.dumps(message.to_dict()),
        address=address,
        port=str(port),
    )


def get_local_ip() -> str:
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("8.8.8.8", 80))
        ip = sock.getsockname()[0]
        sock.close()
        return ip
    except Exception:
        return "127.0.0.1"


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]
