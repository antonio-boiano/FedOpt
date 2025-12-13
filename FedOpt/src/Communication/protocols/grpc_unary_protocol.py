#!/usr/bin/env python3
"""
Unary gRPC protocol implementation for FL communication.
"""

import asyncio
import json
import time
import logging
from contextlib import suppress
from typing import Dict, Any, Optional, List, Set

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


class ClientConnection:
    """Holds a unary gRPC connection to a client."""

    def __init__(self, channel: grpc.aio.Channel, stub):
        self.channel = channel
        self.stub = stub

    async def close(self):
        await self.channel.close()


@register_protocol(
    "grpc",
    description="gRPC unary RPCs for FL communication",
    version="1.1",
    async_support=True,
)
class GRPCClient(BaseClient):
    """gRPC-based FL client using unary RPCs."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.my_ip = config.get("my_ip", get_local_ip())
        self.my_port = find_free_port()

        self.channel: Optional[grpc.aio.Channel] = None
        self.stub = None
        self.server = None

        self.pending_tasks: Set[asyncio.Task] = set()
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
        """Start local listener and connect to server."""
        FedOpt_pb2, FedOpt_pb2_grpc = get_grpc_modules()

        # Local server to receive SendToClient calls
        self.server = grpc.aio.server(options=server_grpc_options())
        FedOpt_pb2_grpc.add_CommunicationServicer_to_server(
            self._UnaryClientServicer(self),
            self.server,
        )
        self.server.add_insecure_port(f"{self.my_ip}:{self.my_port}")
        await self.server.start()

        # Outgoing channel to main server
        channel_options = common_grpc_options() + [
            ("grpc.enable_retries", 1),
            ("grpc.initial_reconnect_backoff_ms", 1000),
            ("grpc.max_reconnect_backoff_ms", 60000),
        ]
        self.channel = grpc.aio.insecure_channel(
            f"{self.ip}:{self.port}", options=channel_options
        )
        self.stub = FedOpt_pb2_grpc.CommunicationStub(self.channel)

        await asyncio.wait_for(self.channel.channel_ready(), timeout=TIMEOUT_GRPC)
        logger.info(f"gRPC unary client started on {self.my_ip}:{self.my_port}")

    async def disconnect(self) -> None:
        """Disconnect gracefully."""
        if self.pending_tasks:
            tasks = list(self.pending_tasks)
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

        if self.channel:
            await self.channel.close()

        if self.server:
            await self.server.stop(grace=0)
            await self.server.wait_for_termination()

        logger.info("gRPC unary client disconnected")

    async def send_message(self, message: Message) -> None:
        """Send message to the server."""
        if not self.stub:
            raise ConnectionError("gRPC stub is not initialized")

        FedOpt_pb2, _ = get_grpc_modules()
        grpc_msg = serialize_message(FedOpt_pb2, message, self.my_ip, self.my_port)

        await retry_rpc(
            "SendToServer", lambda: self.stub.SendToServer(grpc_msg, timeout=TIMEOUT_GRPC)
        )
        logger.debug(f"Sent message of type {message.type}")

    async def receive_message(self) -> Message:
        raise NotImplementedError("Unary gRPC uses service callbacks for incoming data")

    async def start_listening(self) -> None:
        try:
            await self.stop_event.wait()
        except asyncio.CancelledError:
            logger.debug("Listening cancelled")

    class _UnaryClientServicer:
        """Handles SendToClient calls on the client-side listener."""

        def __init__(self, client_instance: "GRPCClient"):
            self.client = client_instance

        async def SendToServer(self, request, context):
            logger.warning("SendToServer invoked on client listener - ignoring")
            return empty_pb2.Empty()

        async def Stream(self, request_iterator, context):
            logger.warning("Stream invoked on unary client listener - not supported")
            context.set_code(grpc.StatusCode.UNIMPLEMENTED)
            context.set_details("Use grpc-stream/grpc-bidi for streaming")
            async for _ in request_iterator:
                pass
            return None

        async def SendToClient(self, request, context):
            receive_time = time.perf_counter()
            try:
                message_dict = json.loads(request.payload)
                message = Message.from_dict(message_dict)

                msg_type = message.type.value if isinstance(message.type, MessageType) else message.type
                if msg_type == "data":
                    task = asyncio.create_task(
                        self.client._handle_data_message(message, receive_time)
                    )
                    task.add_done_callback(self._cleanup_task)
                    self.client.pending_tasks.add(task)
                elif msg_type == "end":
                    await self.client._handle_end_message(message)
                    self.client.stop_event.set()
                else:
                    await self.client.handle_message(message)
            except Exception as exc:
                logger.error(f"Error handling message on client: {exc}")
            return empty_pb2.Empty()

        def _cleanup_task(self, task: asyncio.Task):
            self.client.pending_tasks.discard(task)
            with suppress(Exception):
                task.result()


@register_protocol("grpc")
class GRPCServer(BaseServer):
    """gRPC-based FL server using unary RPCs."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.server = None
        self.client_connections: Dict[str, ClientConnection] = {}
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
            self._UnaryServerServicer(self),
            self.server,
        )
        self.server.add_insecure_port(f"{self.ip}:{self.port}")
        await self.server.start()
        logger.info(f"gRPC unary server started on {self.ip}:{self.port}")

    async def stop(self) -> None:
        if self.server:
            await self.server.stop(grace=5)
        for conn in list(self.client_connections.values()):
            with suppress(Exception):
                await conn.close()
        logger.info("gRPC unary server stopped")

    async def send_to_client(self, client_id: str, message: Message) -> None:
        FedOpt_pb2, _ = get_grpc_modules()
        connection = self.client_connections.get(client_id)
        if not connection:
            logger.warning(f"No connection for client {client_id}")
            return

        grpc_msg = serialize_message(FedOpt_pb2, message, self.ip, self.port)
        try:
            async def _write():
                try:
                    await connection.stub.SendToClient(grpc_msg, timeout=TIMEOUT_GRPC)
                except grpc.aio.AioRpcError as exc:
                    if exc.code() in (grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.INTERNAL):
                        logger.warning(f"Send to {client_id} failed ({exc.code().name}), reconnecting")
                        await self._reconnect_client(client_id)
                        new_conn = self.client_connections.get(client_id)
                        if new_conn:
                            await new_conn.stub.SendToClient(grpc_msg, timeout=TIMEOUT_GRPC)
                        else:
                            raise
                    else:
                        raise

            await retry_rpc(f"SendToClient[{client_id}]", _write)
            self.send_times[client_id] = time.perf_counter()
            logger.debug(f"Sent {message.type} to {client_id}")
        except grpc.aio.AioRpcError as exc:
            logger.warning(f"Failed to reach {client_id}: {exc.code().name}")

    async def broadcast(self, message: Message, client_ids: Optional[List[str]] = None) -> None:
        targets = client_ids or list(self.client_connections.keys())
        for client_id in targets:
            await self.send_to_client(client_id, message)

    async def _register_client_connection(
        self, client_id: str, client_ip: str, client_port: str
    ) -> None:
        """Create reverse channel to the client listener and store metadata."""
        FedOpt_pb2, FedOpt_pb2_grpc = get_grpc_modules()
        if client_id in self.client_connections:
            return

        channel = grpc.aio.insecure_channel(
            f"{client_ip}:{client_port}", options=common_grpc_options()
        )
        await asyncio.wait_for(channel.channel_ready(), timeout=TIMEOUT_GRPC)
        stub = FedOpt_pb2_grpc.CommunicationStub(channel)
        self.client_connections[client_id] = ClientConnection(channel, stub)

        self.client_index_counter += 1
        self.client_count += 1

        info = ClientInfo(
            client_id=client_id,
            index=self.client_index_counter,
            address=client_ip,
            port=int(client_port),
            connection=self.client_connections[client_id],
        )
        self.clients[client_id] = info
        self.new_connections[client_id] = self.client_index_counter

        # Send index assignment
        index_msg = Message(type=MessageType.INDEX, payload=self.client_index_counter)
        await self.send_to_client(client_id, index_msg)
        logger.info(f"Registered client {client_id} (index {self.client_index_counter})")

    async def _reconnect_client(self, client_id: str):
        """Attempt to recreate a channel/stub to a client listener."""
        info = self.clients.get(client_id)
        if not info or not info.address or not info.port:
            logger.warning(f"Cannot reconnect {client_id}: missing address/port")
            return
        old = self.client_connections.pop(client_id, None)
        if old:
            with suppress(Exception):
                await old.close()
        await self._register_client_connection(client_id, info.address, str(info.port))

    class _UnaryServerServicer:
        """Handles SendToServer calls arriving from clients."""

        def __init__(self, server_instance: "GRPCServer"):
            self.server = server_instance

        async def SendToServer(self, request, context):
            client_ip = request.address or "unknown"
            client_port = request.port or "0"
            client_id = build_client_id(client_ip, client_port, context.peer())

            try:
                import json

                message_dict = json.loads(request.payload)
                message = Message.from_dict(message_dict)

                if client_id not in self.server.clients:
                    await self.server._register_client_connection(client_id, client_ip, client_port)

                await self.server.handle_client_message(client_id, message)
            except Exception as exc:
                logger.error(f"Error processing message from {client_id}: {exc}")

            return empty_pb2.Empty()

        async def SendToClient(self, request, context):
            logger.warning("SendToClient invoked on server servicer - ignoring")
            return empty_pb2.Empty()

        async def Stream(self, request_iterator, context):
            logger.warning("Stream invoked on unary server servicer - not supported")
            context.set_code(grpc.StatusCode.UNIMPLEMENTED)
            context.set_details("Use grpc-stream/grpc-bidi for streaming")
            async for _ in request_iterator:
                pass
            return None
