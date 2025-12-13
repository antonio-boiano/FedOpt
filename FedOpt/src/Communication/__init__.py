#!/usr/bin/env python3
"""
FL Communication Layer Package

This package provides a generic, extensible communication layer for
federated learning. It supports multiple protocols through a plugin-based
architecture.

Supported Protocols:
- MQTT: Message Queue Telemetry Transport
- gRPC: Google Remote Procedure Call
- TCP: Transmission Control Protocol

Usage:
    from FedOpt.src.Communication import create_client, create_server
    
    # Create a client
    client = create_client("mqtt", config)
    await client.main()
    
    # Create a server  
    server = create_server("grpc", config)
    await server.main()
    
Adding New Protocols:
    from FedOpt.src.Communication.base import BaseClient, BaseServer
    from FedOpt.src.Communication.registry import register_protocol
    
    @register_protocol("myprotocol", description="My custom protocol")
    class MyClient(BaseClient):
        async def connect(self): ...
        async def disconnect(self): ...
        async def send_message(self, message): ...
        async def receive_message(self): ...
        async def start_listening(self): ...
    
    @register_protocol("myprotocol")
    class MyServer(BaseServer):
        async def start(self): ...
        async def stop(self): ...
        async def send_to_client(self, client_id, message): ...
        async def broadcast(self, message, client_ids=None): ...
"""

# Import base classes
from .base import (
    BaseClient,
    BaseServer,
    Message,
    MessageType,
    ClientInfo,
    ClientSelectionData,
)

# Import registry and factory
from .registry import (
    ProtocolRegistry,
    CommunicationFactory,
    register_protocol,
    create_client,
    create_server,
)

# Import protocol implementations (this registers them)
from . import protocols

# Legacy compatibility - CommunicationManager and utility functions
from .communication_manager import (
    CommunicationManager,
    # Utility functions for backward compatibility
    create_typed_message,
    recv_all,
    state_dict_to_json,
    json_to_state_dict,
    tensor_list_to_json,
    json_to_tensor_list,
    check_model_size,
    update_client_variables,
)

__all__ = [
    # Base classes
    'BaseClient',
    'BaseServer',
    'Message',
    'MessageType',
    'ClientInfo',
    'ClientSelectionData',
    # Registry and factory
    'ProtocolRegistry',
    'CommunicationFactory',
    'register_protocol',
    'create_client',
    'create_server',
    # Legacy compatibility
    'CommunicationManager',
    'create_typed_message',
    'recv_all',
    'state_dict_to_json',
    'json_to_state_dict',
    'tensor_list_to_json',
    'json_to_tensor_list',
    'check_model_size',
    'update_client_variables',
]
