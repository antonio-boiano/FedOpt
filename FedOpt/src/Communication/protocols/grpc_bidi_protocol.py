#!/usr/bin/env python3
"""
Bidirectional streaming gRPC protocol implementation for FL communication.
"""

import asyncio
import time
import logging
from contextlib import suppress
from typing import Dict, Any, Optional, List

import grpc
from google.protobuf import empty_pb2

from ..base import BaseClient, BaseServer, Message, MessageType, ClientInfo
from ..registry import register_protocol
from .grpc_common import (
    TIMEOUT_GRPC,
    common_grpc_options,
    server_grpc_options,
    retry_rpc,
    build_client_id,
    serialize_message,
    get_grpc_modules,
    get_local_ip,
    find_free_port,
)

logger = logging.getLogger("FedOpt")


class StreamConnection:
    """Holds streaming connection state."""

    def __init__(self, queue: asyncio.Queue, peer: str, address: str, port: str):
        self.queue = queue
        self.peer = peer
        self.address = address
        self.port = port
        self.closed = False


@register_protocol(
    "grpc-bidi",
    description="gRPC bidirectional streaming for FL communication",
    version="1.0",
    async_support=True,
)
@register_protocol(
    "grpc-stream",
    description="gRPC bidirectional streaming for FL communication",
    version="1.0",
    async_support=True,
)
class GRPCStreamClient(BaseClient):
    """gRPC-based FL client using a single bidirectional stream."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.my_ip = config.get("my_ip", get_local_ip())
        self.my_port = find_free_port()

        self.channel: Optional[grpc.aio.Channel] = None
        self.stub = None
        self.stream = None

        self.reader_task: Optional[asyncio.Task] = None
        self.stop_event = asyncio.Event()

    #     self._init_model_components()

    # def _init_model_components(self) -> None:
    #     try:
    #         from FedOpt.src.Federation.manager import model_manager
    #         from FedOpt.src.Optimizations.ModelPruning.prune import ModelPruning

    #         self.model_manager = model_manager(self.config)
    #         self.model_pruning = ModelPruning(
    #             self.config.get("prune_ratio", 0.1),
    #             self.model_manager.model,
    #         )
    #     except ImportError as exc:
    #         logger.warning(f"Could not initialize model components: {exc}")

    async def connect(self) -> None:
        await self._open_stream()
        self.reader_task = asyncio.create_task(self._read_stream())
        logger.info("gRPC streaming client connected")

    async def disconnect(self) -> None:
        self.stop_event.set()
        if self.stream:
            with suppress(Exception):
                await self.stream.done_writing()
        if self.reader_task:
            self.reader_task.cancel()
            with suppress(Exception):
                await self.reader_task
        if self.channel:
            await self.channel.close()
        logger.info("gRPC streaming client disconnected")

    async def _read_stream(self) -> None:
        try:
            while not self.stop_event.is_set():
                try:
                    response = await self.stream.read()
                    if response is None:
                        break
                    message = Message.from_dict(__import__("json").loads(response.payload))
                    msg_type = message.type.value if isinstance(message.type, MessageType) else message.type
                    if msg_type == "data":
                        receive_time = time.perf_counter()
                        await self._handle_data_message(message, receive_time)
                    elif msg_type == "end":
                        await self._handle_end_message(message)
                        self.stop_event.set()
                    else:
                        await self.handle_message(message)
                except grpc.aio.AioRpcError as exc:
                    if exc.code() in (grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.INTERNAL):
                        logger.warning(f"Stream interrupted ({exc.code().name}), attempting reconnect")
                        await self._reconnect()
                        continue
                    raise
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error(f"Stream reader error: {exc}")
        finally:
            self.stop_event.set()

    async def send_message(self, message: Message) -> None:
        await self._send_stream_message(message)

    async def _send_stream_message(self, message: Message) -> None:
        if not self.stream:
            raise ConnectionError("Streaming channel is not ready")
        FedOpt_pb2, _ = get_grpc_modules()
        grpc_msg = serialize_message(FedOpt_pb2, message, self.my_ip, self.my_port)
        async def _write():
            try:
                await self.stream.write(grpc_msg)
            except grpc.aio.AioRpcError as exc:
                if exc.code() in (grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.INTERNAL):
                    logger.warning(f"Write failed ({exc.code().name}), reconnecting")
                    await self._reconnect()
                    await self.stream.write(grpc_msg)
                else:
                    raise

        await retry_rpc("Stream->server", _write)

    async def receive_message(self) -> Message:
        raise NotImplementedError("Streaming client drives reads internally")

    async def start_listening(self) -> None:
        await self.stop_event.wait()

    async def _open_stream(self):
        _, FedOpt_pb2_grpc = get_grpc_modules()
        self.channel = grpc.aio.insecure_channel(
            f"{self.ip}:{self.port}", options=common_grpc_options()
        )
        self.stub = FedOpt_pb2_grpc.CommunicationStub(self.channel)
        self.stream = self.stub.Stream()
        await asyncio.wait_for(self.channel.channel_ready(), timeout=TIMEOUT_GRPC)
        # Re-send sync after (re)connecting so the server registers us again
        await self._send_stream_message(Message(type=MessageType.SYNC, payload=""))

    async def _reconnect(self):
        if self.channel:
            with suppress(Exception):
                await self.channel.close()
        await asyncio.sleep(1.0)
        await self._open_stream()


@register_protocol(
    "grpc-bidi",
    description="gRPC bidirectional streaming for FL communication",
    version="1.0",
    async_support=True,
)
@register_protocol(
    "grpc-stream",
    description="gRPC bidirectional streaming for FL communication",
    version="1.0",
    async_support=True,
)
class GRPCStreamServer(BaseServer):
    """gRPC-based FL server using bidirectional streams."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.server = None
        self.stream_clients: Dict[str, StreamConnection] = {}
    #     self._init_federation_components()

    # def _init_federation_components(self) -> None:
    #     try:
    #         from FedOpt.src.Federation.manager import federation_manager
    #         from FedOpt.src.Optimizations.DynamicSampling.dynamic_sampling import DynamicSampling
    #         from FedOpt.src.Optimizations.ModelPruning.prune import ModelPruning

    #         self.federation = federation_manager(self.config)
    #         self.dynamic_sampling = DynamicSampling(
    #             self.config.get("decay_rate", 0.1),
    #             self.client_round,
    #         )
    #         self.model_pruning = ModelPruning(
    #             self.config.get("prune_ratio", 0.1),
    #             self.federation.server_model.model,
    #         )
    #     except ImportError as exc:
    #         logger.warning(f"Could not initialize federation components: {exc}")

    async def start(self) -> None:
        FedOpt_pb2, FedOpt_pb2_grpc = get_grpc_modules()
        self.server = grpc.aio.server(options=server_grpc_options())
        FedOpt_pb2_grpc.add_CommunicationServicer_to_server(
            self._StreamServerServicer(self),
            self.server,
        )
        self.server.add_insecure_port(f"{self.ip}:{self.port}")
        await self.server.start()
        logger.info(f"gRPC streaming server started on {self.ip}:{self.port}")

    async def stop(self) -> None:
        if self.server:
            await self.server.stop(grace=5)
        self.stream_clients.clear()
        logger.info("gRPC streaming server stopped")

    async def send_to_client(self, client_id: str, message: Message) -> None:
        FedOpt_pb2, _ = get_grpc_modules()
        conn = self.stream_clients.get(client_id)
        if not conn or conn.closed:
            logger.warning(f"No active stream for client {client_id}")
            return
        grpc_msg = serialize_message(FedOpt_pb2, message, self.ip, self.port)
        await conn.queue.put(grpc_msg)
        self.send_times[client_id] = time.perf_counter()

    async def broadcast(self, message: Message, client_ids: Optional[List[str]] = None) -> None:
        targets = client_ids or list(self.stream_clients.keys())
        for client_id in targets:
            await self.send_to_client(client_id, message)

    async def _register_stream_client(
        self,
        client_id: str,
        client_ip: str,
        client_port: str,
        queue: asyncio.Queue,
        peer: str,
    ) -> None:
        if client_id in self.stream_clients:
            return

        self.client_index_counter += 1
        self.client_count += 1

        conn = StreamConnection(queue, peer, client_ip, client_port)
        self.stream_clients[client_id] = conn

        info = ClientInfo(
            client_id=client_id,
            index=self.client_index_counter,
            address=client_ip,
            port=int(client_port) if client_port else None,
            connection=conn,
        )
        self.clients[client_id] = info
        self.new_connections[client_id] = self.client_index_counter

        # Send index immediately over the stream
        index_msg = Message(type=MessageType.INDEX, payload=self.client_index_counter)
        FedOpt_pb2, _ = get_grpc_modules()
        await queue.put(serialize_message(FedOpt_pb2, index_msg, self.ip, self.port))
        logger.info(f"Streaming client registered: {client_id} (index {self.client_index_counter})")

    async def _cleanup_stream_client(self, client_id: str) -> None:
        conn = self.stream_clients.pop(client_id, None)
        if conn:
            conn.closed = True
        if client_id in self.clients:
            await self.remove_client(client_id)

    class _StreamServerServicer:
        """Implements the bidirectional Stream RPC."""

        def __init__(self, server_instance: "GRPCStreamServer"):
            self.server = server_instance

        async def SendToServer(self, request, context):
            logger.warning("Unary SendToServer invoked on streaming servicer - ignoring")
            return empty_pb2.Empty()

        async def SendToClient(self, request, context):
            logger.warning("Unary SendToClient invoked on streaming servicer - ignoring")
            return empty_pb2.Empty()

        async def Stream(self, request_iterator, context):
            FedOpt_pb2, _ = get_grpc_modules()
            send_queue: asyncio.Queue = asyncio.Queue()
            client_id: Optional[str] = None

            async def reader():
                nonlocal client_id
                async for request in request_iterator:
                    msg = Message.from_dict(__import__("json").loads(request.payload))
                    if client_id is None:
                        client_id = build_client_id(request.address, request.port, context.peer())
                        await self.server._register_stream_client(
                            client_id, request.address, request.port, send_queue, context.peer()
                        )
                    await self.server.handle_client_message(client_id, msg)
                await send_queue.put(None)

            reader_task = asyncio.create_task(reader())

            try:
                while True:
                    to_send = await send_queue.get()
                    if to_send is None:
                        break
                    yield to_send
            except asyncio.CancelledError:
                # Expected during shutdown; suppress noisy tracebacks
                pass
            finally:
                reader_task.cancel()
                with suppress(Exception):
                    await reader_task
                if client_id:
                    await self.server._cleanup_stream_client(client_id)
