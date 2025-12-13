#!/usr/bin/env python3
"""
Protocol Registry and Factory for FL Communication Layer

This module provides a registry pattern for managing communication protocols
and a factory for creating protocol instances.
"""

from typing import Dict, Type, Optional, Any, Callable
from dataclasses import dataclass, field
import logging

from .base import BaseClient, BaseServer

logger = logging.getLogger("FedOpt")


@dataclass
class ProtocolInfo:
    """Information about a registered protocol."""
    name: str
    client_class: Type[BaseClient]
    server_class: Type[BaseServer]
    description: str = ""
    version: str = "1.0"
    async_support: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class ProtocolRegistry:
    """
    Registry for communication protocols.
    
    Provides a centralized way to register and retrieve protocol implementations.
    """
    
    _instance: Optional['ProtocolRegistry'] = None
    _protocols: Dict[str, ProtocolInfo] = {}
    
    def __new__(cls) -> 'ProtocolRegistry':
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._protocols = {}
        return cls._instance
    
    @classmethod
    def register(
        cls,
        name: str,
        client_class: Type[BaseClient],
        server_class: Type[BaseServer],
        description: str = "",
        version: str = "1.0",
        async_support: bool = True,
        **metadata
    ) -> None:
        """
        Register a new protocol.
        
        Args:
            name: Protocol name (e.g., 'mqtt', 'grpc').
            client_class: Client implementation class.
            server_class: Server implementation class.
            description: Protocol description.
            version: Protocol version.
            async_support: Whether the protocol supports async.
            **metadata: Additional protocol metadata.
        """
        name = name.lower()
        
        # Check if already registered with same classes (avoid duplicate warnings on reimport)
        if name in cls._protocols:
            existing = cls._protocols[name]
            # Skip silently if same classes are being registered (reimport scenario)
            if (existing.client_class is client_class and 
                existing.server_class is server_class):
                return
            # Don't warn when just filling in placeholder classes
            is_placeholder_update = (
                existing.server_class.__name__ == 'BaseServer' or 
                existing.client_class.__name__ == 'BaseClient'
            )
            if not is_placeholder_update:
                logger.warning(f"Overwriting existing protocol: {name}")
        
        cls._protocols[name] = ProtocolInfo(
            name=name,
            client_class=client_class,
            server_class=server_class,
            description=description,
            version=version,
            async_support=async_support,
            metadata=metadata
        )
        
        logger.debug(f"Registered protocol: {name}")
    
    @classmethod
    def get(cls, name: str) -> Optional[ProtocolInfo]:
        """Get protocol info by name."""
        return cls._protocols.get(name.lower())
    
    @classmethod
    def get_client_class(cls, name: str) -> Optional[Type[BaseClient]]:
        """Get client class for a protocol."""
        info = cls.get(name)
        return info.client_class if info else None
    
    @classmethod
    def get_server_class(cls, name: str) -> Optional[Type[BaseServer]]:
        """Get server class for a protocol."""
        info = cls.get(name)
        return info.server_class if info else None
    
    @classmethod
    def list_protocols(cls) -> Dict[str, ProtocolInfo]:
        """List all registered protocols."""
        return cls._protocols.copy()
    
    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a protocol is registered."""
        return name.lower() in cls._protocols
    
    @classmethod
    def unregister(cls, name: str) -> bool:
        """Unregister a protocol."""
        name = name.lower()
        if name in cls._protocols:
            del cls._protocols[name]
            return True
        return False


def register_protocol(
    name: str,
    description: str = "",
    version: str = "1.0",
    async_support: bool = True,
    **metadata
) -> Callable:
    """
    Decorator for registering protocol classes.
    
    Usage:
        @register_protocol("myprotocol", description="My custom protocol")
        class MyProtocolClient(BaseClient):
            ...
        
        @register_protocol("myprotocol")
        class MyProtocolServer(BaseServer):
            ...
    """
    def decorator(cls: Type) -> Type:
        if issubclass(cls, BaseClient):
            existing = ProtocolRegistry.get(name)
            if existing:
                ProtocolRegistry.register(
                    name=name,
                    client_class=cls,
                    server_class=existing.server_class,
                    description=description or existing.description,
                    version=version,
                    async_support=async_support,
                    **{**existing.metadata, **metadata}
                )
            else:
                # Placeholder server until it's registered
                ProtocolRegistry.register(
                    name=name,
                    client_class=cls,
                    server_class=BaseServer,  # type: ignore
                    description=description,
                    version=version,
                    async_support=async_support,
                    **metadata
                )
        elif issubclass(cls, BaseServer):
            existing = ProtocolRegistry.get(name)
            if existing:
                ProtocolRegistry.register(
                    name=name,
                    client_class=existing.client_class,
                    server_class=cls,
                    description=description or existing.description,
                    version=version,
                    async_support=async_support,
                    **{**existing.metadata, **metadata}
                )
            else:
                ProtocolRegistry.register(
                    name=name,
                    client_class=BaseClient,  # type: ignore
                    server_class=cls,
                    description=description,
                    version=version,
                    async_support=async_support,
                    **metadata
                )
        else:
            raise TypeError(f"Class must inherit from BaseClient or BaseServer")
        
        return cls
    
    return decorator


class CommunicationFactory:
    """
    Factory for creating communication instances.
    
    Provides a unified interface for creating clients and servers
    regardless of the underlying protocol.
    """
    
    @staticmethod
    def create_client(protocol: str, config: Dict[str, Any]) -> BaseClient:
        """
        Create a client instance for the specified protocol.
        
        Args:
            protocol: Protocol name.
            config: Configuration dictionary.
            
        Returns:
            Client instance.
            
        Raises:
            ValueError: If protocol is not registered.
        """
        client_class = ProtocolRegistry.get_client_class(protocol)
        if client_class is None:
            available = list(ProtocolRegistry.list_protocols().keys())
            raise ValueError(
                f"Unknown protocol: {protocol}. Available: {available}"
            )
        
        return client_class(config)
    
    @staticmethod
    def create_server(protocol: str, config: Dict[str, Any]) -> BaseServer:
        """
        Create a server instance for the specified protocol.
        
        Args:
            protocol: Protocol name.
            config: Configuration dictionary.
            
        Returns:
            Server instance.
            
        Raises:
            ValueError: If protocol is not registered.
        """
        server_class = ProtocolRegistry.get_server_class(protocol)
        if server_class is None:
            available = list(ProtocolRegistry.list_protocols().keys())
            raise ValueError(
                f"Unknown protocol: {protocol}. Available: {available}"
            )
        
        return server_class(config)
    
    @staticmethod
    def create(mode: str, protocol: str, config: Dict[str, Any]):
        """
        Create either a client or server instance.
        
        Args:
            mode: Either 'client' or 'server'.
            protocol: Protocol name.
            config: Configuration dictionary.
            
        Returns:
            Client or Server instance.
        """
        mode = mode.lower()
        if mode == "client":
            return CommunicationFactory.create_client(protocol, config)
        elif mode == "server":
            return CommunicationFactory.create_server(protocol, config)
        else:
            raise ValueError(f"Invalid mode: {mode}. Use 'client' or 'server'.")


# Convenience functions
def create_client(protocol: str, config: Dict[str, Any]) -> BaseClient:
    """Create a client instance."""
    return CommunicationFactory.create_client(protocol, config)


def create_server(protocol: str, config: Dict[str, Any]) -> BaseServer:
    """Create a server instance."""
    return CommunicationFactory.create_server(protocol, config)
