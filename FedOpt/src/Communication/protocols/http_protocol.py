#!/usr/bin/env python3
"""
HTTP Protocol Implementation for FL Communication Layer.

This is a lightweight async HTTP transport built with aiohttp.
The server exposes /send for client uploads and pushes messages to clients
via their /recv endpoints.
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional, List

import aiohttp
from aiohttp import web

from ..base import BaseClient, BaseServer, Message, MessageType, ClientInfo
from ..registry import register_protocol

logger = logging.getLogger("FedOpt")


@register_protocol(
    "http",
    description="HTTP-based federated learning communication",
    version="1.0",
    async_support=True,
)
class HTTPClient(BaseClient):
    """HTTP-based FL client implementation."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.session: Optional[aiohttp.ClientSession] = None
        self.app: Optional[web.Application] = None
        self.runner: Optional[web.AppRunner] = None
        self.site: Optional[web.TCPSite] = None
        self.my_ip = config.get("my_ip", "127.0.0.1")
        self.my_port = config.get("my_port", 0)  # if 0, aiohttp picks a free port
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
        """Start local HTTP server and create session to contact main server."""
        self.session = aiohttp.ClientSession()
        self.app = web.Application()
        self.app.add_routes([web.post("/recv", self._handle_recv)])
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, self.my_ip, self.my_port)
        await self.site.start()

        if self.my_port == 0 and self.site:
            # fetch the bound port
            for sock in self.site._server.sockets:  # type: ignore[attr-defined]
                self.my_port = sock.getsockname()[1]
                break

        logger.info(f"HTTP client listening on {self.my_ip}:{self.my_port}")

    async def disconnect(self) -> None:
        if self.session:
            await self.session.close()
        if self.runner:
            await self.runner.cleanup()
        logger.info("HTTP client disconnected")

    async def send_message(self, message: Message) -> None:
        if not self.session:
            raise ConnectionError("HTTP session not initialized")
        msg_dict = message.to_dict()
        msg_dict.setdefault("address", self.my_ip)
        msg_dict.setdefault("port", self.my_port)
        payload = json.dumps(msg_dict)
        url = f"http://{self.ip}:{self.port}/send"
        async with self.session.post(url, data=payload) as resp:
            if resp.status >= 400:
                text = await resp.text()
                logger.error(f"HTTP send failed: {resp.status} {text}")

    async def receive_message(self) -> Message:
        raise NotImplementedError("HTTP uses /recv endpoint for incoming messages")

    async def start_listening(self) -> None:
        await self.stop_event.wait()

    async def _handle_recv(self, request: web.Request) -> web.Response:
        data = await request.text()
        try:
            message_dict = json.loads(data)
            message = Message.from_dict(message_dict)
            receive_time = asyncio.get_event_loop().time()

            msg_type = message.type.value if isinstance(message.type, MessageType) else message.type
            if msg_type == "data":
                await self._handle_data_message(message, receive_time)
            elif msg_type == "end":
                await self._handle_end_message(message)
                self.stop_event.set()
            else:
                await self.handle_message(message)
        except Exception as exc:
            logger.error(f"Error handling incoming HTTP message: {exc}")
        return web.Response(text="ok")


@register_protocol("http")
class HTTPServer(BaseServer):
    """HTTP-based FL server implementation."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.session: Optional[aiohttp.ClientSession] = None
        self.app: Optional[web.Application] = None
        self.runner: Optional[web.AppRunner] = None
        self.site: Optional[web.TCPSite] = None
        self._init_federation_components()

    def _init_federation_components(self) -> None:
        try:
            from FedOpt.src.Federation.manager import federation_manager
            from FedOpt.src.Optimizations.DynamicSampling.dynamic_sampling import DynamicSampling
            from FedOpt.src.Optimizations.ModelPruning.prune import ModelPruning

            self.federation = federation_manager(self.config)
            self.dynamic_sampling = DynamicSampling(
                self.config.get("decay_rate", 0.1),
                self.client_round,
            )
            self.model_pruning = ModelPruning(
                self.config.get("prune_ratio", 0.1),
                self.federation.server_model.model,
            )
        except ImportError as exc:
            logger.warning(f"Could not initialize federation components: {exc}")

    async def start(self) -> None:
        self.session = aiohttp.ClientSession()
        self.app = web.Application()
        self.app.add_routes([web.post("/send", self._handle_send)])
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, self.ip, self.port)
        await self.site.start()
        logger.info(f"HTTP server started on {self.ip}:{self.port}")

    async def stop(self) -> None:
        if self.session:
            await self.session.close()
        if self.runner:
            await self.runner.cleanup()
        logger.info("HTTP server stopped")

    async def send_to_client(self, client_id: str, message: Message) -> None:
        if not self.session:
            raise ConnectionError("HTTP session not initialized")
        client_info = self.clients.get(client_id)
        if not client_info or not client_info.address or not client_info.port:
            logger.warning(f"No client info for {client_id}")
            return
        url = f"http://{client_info.address}:{client_info.port}/recv"
        payload = json.dumps(message.to_dict())
        try:
            async with self.session.post(url, data=payload) as resp:
                if resp.status >= 400:
                    text = await resp.text()
                    logger.error(f"Failed to send to {client_id}: {resp.status} {text}")
            self.send_times[client_id] = asyncio.get_event_loop().time()
        except Exception as exc:
            logger.error(f"HTTP send to {client_id} failed: {exc}")

    async def broadcast(self, message: Message, client_ids: Optional[List[str]] = None) -> None:
        targets = client_ids or list(self.clients.keys())
        for client_id in targets:
            await self.send_to_client(client_id, message)

    async def _handle_send(self, request: web.Request) -> web.Response:
        data = await request.text()
        try:
            message_dict = json.loads(data)
            client_addr = message_dict.get("address") or (request.remote.split(":")[0] if request.remote else None)
            client_port = message_dict.get("port")
            client_id = f"{client_addr}:{client_port}" if client_addr and client_port else (request.remote or "unknown")

            message = Message.from_dict(message_dict)

            # Register client if not present
            if client_id not in self.clients:
                self.client_index_counter += 1
                self.client_count += 1
                self.clients[client_id] = ClientInfo(
                    client_id=client_id,
                    index=self.client_index_counter,
                    address=client_addr,
                    port=int(client_port) if client_port else None,
                )
                self.new_connections[client_id] = self.client_index_counter

            await self.handle_client_message(client_id, message)
        except Exception as exc:
            logger.error(f"Error handling HTTP send: {exc}")
        return web.Response(text="ok")
