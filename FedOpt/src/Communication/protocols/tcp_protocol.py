#!/usr/bin/env python3
"""
TCP Protocol Implementation for FL Communication Layer

This module provides TCP-based client and server implementations
using the abstract base classes. This serves as an example of how
to implement new protocols.
"""

import json
import time
import socket
import struct
import asyncio
import threading
from typing import Dict, Any, Optional, List
from collections import defaultdict
from copy import deepcopy

from ..base import BaseClient, BaseServer, Message, MessageType, ClientInfo
from ..registry import register_protocol

import logging
logger = logging.getLogger("FedOpt")


def recv_all(sock: socket.socket, length: int) -> bytes:
    """Receive exactly `length` bytes from socket."""
    data = bytearray()
    while len(data) < length:
        packet = sock.recv(length - len(data))
        if not packet:
            raise ConnectionResetError("Connection reset by peer")
        data.extend(packet)
    return bytes(data)


def send_message(sock: socket.socket, message: Message) -> None:
    """Send a message over a socket."""
    data = json.dumps(message.to_dict()).encode('utf-8')
    length = len(data)
    sock.sendall(struct.pack('!I', length))
    sock.sendall(data)


def receive_message(sock: socket.socket) -> Message:
    """Receive a message from a socket."""
    length_bytes = recv_all(sock, 4)
    length = struct.unpack('!I', length_bytes)[0]
    data = recv_all(sock, length)
    message_dict = json.loads(data.decode('utf-8'))
    return Message.from_dict(message_dict)


@register_protocol(
    "tcp",
    description="TCP-based federated learning communication",
    version="1.0",
    async_support=False  # TCP implementation uses threading
)
class TCPClient(BaseClient):
    """TCP-based FL client implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.socket: Optional[socket.socket] = None
        self._running = False
        
    #     # Initialize model components
    #     self._init_model_components()
    
    # def _init_model_components(self) -> None:
    #     """Initialize model manager and pruning components."""
    #     try:
    #         from FedOpt.src.Federation.manager import model_manager
    #         from FedOpt.src.Optimizations.ModelPruning.prune import ModelPruning
            
    #         self.model_manager = model_manager(self.config)
    #         self.model_pruning = ModelPruning(
    #             self.config.get("prune_ratio", 0.1),
    #             self.model_manager.model
    #         )
    #     except ImportError as e:
    #         logger.warning(f"Could not initialize model components: {e}")
    
    async def connect(self) -> None:
        """Connect to the TCP server."""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.ip, self.port))
        self._running = True
        logger.info(f"Connected to TCP server at {self.ip}:{self.port}")
    
    async def disconnect(self) -> None:
        """Disconnect from the TCP server."""
        self._running = False
        if self.socket:
            self.socket.close()
            self.socket = None
        logger.info("Disconnected from TCP server")
    
    async def send_message(self, message: Message) -> None:
        """Send a message to the server."""
        if self.socket:
            send_message(self.socket, message)
            logger.debug(f"Sent message of type {message.type}")
    
    async def receive_message(self) -> Message:
        """Receive a message from the server."""
        if self.socket:
            return receive_message(self.socket)
        raise ConnectionError("Not connected")
    
    async def start_listening(self) -> None:
        """Start listening for messages."""
        while self._running and not self.end_event.is_set():
            try:
                message = await self.receive_message()
                await self.handle_message(message)
                
                # Check for end message
                msg_type = message.type.value if isinstance(message.type, MessageType) else message.type
                if msg_type == "end":
                    await self.send_message(Message(type=MessageType.END))
                    break
                    
            except ConnectionResetError:
                logger.warning("Connection reset by server")
                break
            except Exception as e:
                logger.error(f"Error receiving message: {e}")
                break
            
            await asyncio.sleep(0.1)


@register_protocol("tcp")
class TCPServer(BaseServer):
    """TCP-based FL server implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.server_socket: Optional[socket.socket] = None
        self.client_sockets: Dict[str, socket.socket] = {}
        self.max_clients = config.get("server_config", {}).get("max_num_clients", 10)
        
        self._lock = threading.Lock()
        self._client_threads: List[threading.Thread] = []
        
        # Initialize federation components
        # self._init_federation_components()
    
    # def _init_federation_components(self) -> None:
    #     """Initialize federation manager and optimization components."""
    #     try:
    #         from FedOpt.src.Federation.manager import federation_manager
    #         from FedOpt.src.Optimizations.DynamicSampling.dynamic_sampling import DynamicSampling
    #         from FedOpt.src.Optimizations.ModelPruning.prune import ModelPruning
            
    #         self.federation = federation_manager(self.config)
    #         self.dynamic_sampling = DynamicSampling(
    #             self.config.get("decay_rate", 0.1),
    #             self.client_round
    #         )
    #         self.model_pruning = ModelPruning(
    #             self.config.get("prune_ratio", 0.1),
    #             self.federation.server_model.model
    #         )
    #     except ImportError as e:
    #         logger.warning(f"Could not initialize federation components: {e}")
    
    async def start(self) -> None:
        """Start the TCP server."""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.settimeout(0.2)
        self.server_socket.bind((self.ip, self.port))
        self.server_socket.listen(self.max_clients)
        
        logger.info(f"TCP server started on {self.ip}:{self.port}")
        
        # Start accept thread
        accept_thread = threading.Thread(
            target=self._accept_connections,
            daemon=True
        )
        accept_thread.start()
    
    async def stop(self) -> None:
        """Stop the TCP server."""
        self.is_running = False
        
        # Close all client connections
        for client_id, sock in list(self.client_sockets.items()):
            try:
                sock.close()
            except:
                pass
        
        # Close server socket
        if self.server_socket:
            self.server_socket.close()
        
        # Wait for client threads
        for thread in self._client_threads:
            thread.join(timeout=1)
        
        logger.info("TCP server stopped")
    
    async def send_to_client(self, client_id: str, message: Message) -> None:
        """Send a message to a specific client."""
        if client_id not in self.client_sockets:
            logger.warning(f"No socket for client {client_id}")
            return
        
        try:
            sock = self.client_sockets[client_id]
            send_message(sock, message)
            logger.debug(f"Sent message to client {client_id}")
        except Exception as e:
            logger.error(f"Failed to send to {client_id}: {e}")
    
    async def broadcast(self, message: Message, client_ids: Optional[List[str]] = None) -> None:
        """Broadcast a message to multiple clients."""
        targets = client_ids or list(self.clients.keys())
        
        for client_id in targets:
            await self.send_to_client(client_id, message)
    
    def _accept_connections(self) -> None:
        """Accept incoming connections in a separate thread."""
        while self.is_running and self.client_count < self.max_clients:
            try:
                client_socket, client_address = self.server_socket.accept()
                
                # Create client ID
                client_id = f"{client_address[0]}:{client_address[1]}"
                
                with self._lock:
                    self.client_count += 1
                    self.client_index_counter += 1
                    
                    self.client_sockets[client_id] = client_socket
                    self.clients[client_id] = ClientInfo(
                        client_id=client_id,
                        index=self.client_index_counter,
                        address=client_address[0],
                        port=client_address[1],
                        connection=client_socket
                    )
                    self.new_connections[client_id] = self.client_index_counter
                
                logger.info(f"New client connected: {client_id}")
                
                # Send index
                index_msg = Message(type=MessageType.INDEX, payload=self.client_index_counter)
                send_message(client_socket, index_msg)
                
                # Start client handler thread
                thread = threading.Thread(
                    target=self._handle_client,
                    args=(client_id, client_socket),
                    daemon=True
                )
                thread.start()
                self._client_threads.append(thread)
                
            except socket.timeout:
                continue
            except Exception as e:
                if self.is_running:
                    logger.error(f"Error accepting connection: {e}")
    
    def _handle_client(self, client_id: str, client_socket: socket.socket) -> None:
        """Handle communication with a single client."""
        try:
            while self.is_running:
                # Check if client is selected for this round
                if client_id in self.client_selected and client_id not in self.client_flag:
                    if self.round_type == "train":
                        # Create and send model data
                        with self._lock:
                            if self.model_data is None and self.federation:
                                server_model = self.federation.get_server_model()
                                if self.channel_indexes_pruned:
                                    server_model["pruning_info"] = self.channel_indexes_pruned
                                self.model_data = server_model
                        
                        send_time = time.perf_counter()
                        msg = Message(type=MessageType.DATA, payload=self.model_data)
                        send_message(client_socket, msg)
                        
                        # Wait for response
                        response = receive_message(client_socket)
                        self._process_response(client_id, response, send_time)
                        
                    elif self.round_type == "prune":
                        msg = Message(type=MessageType.PRUNE)
                        send_message(client_socket, msg)
                        
                        response = receive_message(client_socket)
                        self._process_response(client_id, response, 0)
                        
                    elif self.round_type == "end":
                        msg = Message(type=MessageType.END)
                        send_message(client_socket, msg)
                        
                        response = receive_message(client_socket)
                        if response.type == MessageType.END or response.type == "end":
                            break
                
                time.sleep(self.sleep_time)
                
        except Exception as e:
            logger.error(f"Error handling client {client_id}: {e}")
        finally:
            self._remove_client(client_id, client_socket)
    
    def _process_response(self, client_id: str, response: Message, send_time: float) -> None:
        """Process a response from a client."""
        receive_time = time.perf_counter()
        
        with self._lock:
            self.client_flag.append(client_id)
            self.client_responses[client_id] = response.payload
            
            if send_time > 0 and response.payload and "client_info" in response.payload:
                total_time = receive_time - send_time
                client_data = response.payload["client_info"]
                
                self.client_sel_data[client_id] = {
                    "communication_time": round(total_time - client_data.get("client_time", 0), 7),
                    "training_time": round(client_data.get("training_time", 0), 5),
                    "data_size": client_data.get("data_size", 0),
                    "accuracy": client_data.get("accuracy", 0)
                }
    
    def _remove_client(self, client_id: str, client_socket: socket.socket) -> None:
        """Remove a client from the server."""
        with self._lock:
            if client_id in self.client_selected:
                self.client_selected.remove(client_id)
            
            self.clients.pop(client_id, None)
            self.client_sockets.pop(client_id, None)
            self.client_count -= 1
        
        try:
            client_socket.close()
        except:
            pass
        
        logger.info(f"Client removed: {client_id}")
