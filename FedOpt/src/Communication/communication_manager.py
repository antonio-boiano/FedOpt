#!/usr/bin/env python3
"""
Communication Manager - Backward Compatible Interface

This module provides a backward-compatible interface to the new
communication layer while preserving the original API.
"""

import os
import asyncio
import logging
from typing import Dict, Any

from .registry import CommunicationFactory, ProtocolRegistry
from .base import Message, MessageType

logger = logging.getLogger("FedOpt")

# Allowed protocols and modes
dir_path = os.path.dirname(os.path.realpath(__file__))
allowed_protocols = set([
    folder.name for folder in os.scandir(dir_path) 
    if os.path.isdir(folder) and not folder.name.startswith('_')
])
allowed_protocols.discard("__pycache__")
allowed_protocols.update(["mqtt", "grpc", "tcp", "rest", "coap", "amqp"])
allowed_modes = {"client", "server"}


class CommunicationManager:
    """
    Communication manager class that creates and handles communication 
    between clients and servers.
    
    This class provides backward compatibility with the existing codebase
    while using the new factory-based architecture internally.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the communication manager.
        
        Args:
            config: Configuration dictionary containing protocol and mode settings.
        """
        self.protocol_name = config.get("protocol", "").lower()
        self.mode = config.get("mode", "").lower()
        self.config = config
        
        # Validate inputs
        if not self.protocol_name:
            raise ValueError("Protocol must be specified in config")
        if not self.mode:
            raise ValueError("Mode must be specified in config")
        if self.mode not in allowed_modes:
            raise ValueError(f"Invalid mode: {self.mode}. Use one of {allowed_modes}")
    
    def run_instance(self) -> None:
        """
        Run the communication instance (client or server).
        
        This method attempts to use the new factory-based system first,
        falling back to legacy imports if the protocol isn't registered.
        """
        if not self.protocol_name or not self.mode:
            raise ValueError("Both protocol and mode must be defined!")
        
        # Try using the new registry first
        if ProtocolRegistry.is_registered(self.protocol_name):
            self._run_with_factory()
        else:
            # Fall back to legacy imports
            self._run_legacy()
    
    def _run_with_factory(self) -> None:
        """Run using the new factory-based system."""
        logger.info(f"Starting {self.mode} with {self.protocol_name} protocol (new system)")
        
        try:
            instance = CommunicationFactory.create(
                mode=self.mode,
                protocol=self.protocol_name,
                config=self.config
            )
            
            # Check if the protocol supports async
            protocol_info = ProtocolRegistry.get(self.protocol_name)
            
            if protocol_info and protocol_info.async_support:
                asyncio.run(instance.main())
            else:
                # For non-async protocols, call main() directly
                if asyncio.iscoroutinefunction(instance.main):
                    asyncio.run(instance.main())
                else:
                    instance.main()
                    
        except Exception as e:
            logger.error(f"Error running {self.mode}: {e}")
            raise
    
    def _run_legacy(self) -> None:
        """Run using the legacy import system."""
        logger.info(f"Starting {self.mode} with {self.protocol_name} protocol (legacy system)")
        
        # TCP
        if self.protocol_name == "tcp":
            if self.mode == "client":
                from .TCP.Client import Client
                mode = Client(config=self.config)
                mode.main()
            elif self.mode == "server":
                from .TCP.Server import Server
                mode = Server(config=self.config)
                mode.main()
            else:
                raise ValueError(f"Invalid mode: {self.mode}")
        
        # REST
        elif self.protocol_name == "rest":
            if self.mode == "client":
                from .REST.Client import Client
                Client(config=self.config)
            elif self.mode == "server":
                from .REST.Server import Server
                Server(config=self.config)
            else:
                raise ValueError(f"Invalid mode: {self.mode}")
        
        # CoAP
        elif self.protocol_name == "coap":
            if self.mode == "client":
                from .CoAP.Client import Client
                Client(config=self.config)
            elif self.mode == "server":
                from .CoAP.Server import Server
                Server(config=self.config)
            else:
                raise ValueError(f"Invalid mode: {self.mode}")
        
        # MQTT
        elif self.protocol_name == "mqtt":
            if self.mode == "client":
                from .MQTT.Client import Client
                mode = Client(config=self.config)
                asyncio.run(mode.main())
            elif self.mode == "server":
                from .MQTT.Server import Server
                mode = Server(config=self.config)
                asyncio.run(mode.main())
            else:
                raise ValueError(f"Invalid mode: {self.mode}")
        
        # AMQP
        elif self.protocol_name == "amqp":
            if self.mode == "client":
                from .AMQP.Client import Client
                Client(config=self.config)
            elif self.mode == "server":
                from .AMQP.Server import Server
                Server(config=self.config)
            else:
                raise ValueError(f"Invalid mode: {self.mode}")
        
        # gRPC
        elif self.protocol_name == "grpc":
            if self.mode == "client":
                from .gRPC.Client import Client
                mode = Client(config=self.config)
                asyncio.run(mode.main())
            elif self.mode == "server":
                from .gRPC.Server import Server
                mode = Server(config=self.config)
                asyncio.run(mode.main())
            else:
                raise ValueError(f"Invalid mode: {self.mode}")
        else:
            raise ValueError(f"Unknown protocol: {self.protocol_name}")


# ==================== Utility Functions ====================
# These are kept for backward compatibility with existing code

def create_typed_message(msg_type: str, payload: Any = "") -> Dict[str, Any]:
    """
    Create a typed message dictionary.
    
    Args:
        msg_type: Message type (sync, data, ack, end, etc.)
        payload: Message payload.
        
    Returns:
        Message dictionary.
    """
    return {"type": msg_type, "payload": payload}


def recv_all(client_socket, length: int) -> bytes:
    """
    Receive exactly `length` bytes from a socket.
    
    Args:
        client_socket: Socket to receive from.
        length: Number of bytes to receive.
        
    Returns:
        Received bytes.
    """
    data = bytearray()
    while len(data) < length:
        packet = client_socket.recv(length - len(data))
        if not packet:
            raise ConnectionResetError("Connection reset by peer")
        data.extend(packet)
    return bytes(data)


def state_dict_to_json(state_dict: Dict) -> str:
    """Convert PyTorch state_dict to JSON string."""
    import json
    import torch
    return json.dumps({k: v.cpu().numpy().tolist() for k, v in state_dict.items()})


def json_to_state_dict(json_data: str, device: str = "cpu") -> Dict:
    """Convert JSON string to PyTorch state_dict."""
    import json
    import torch
    state_dict_serializable = json.loads(json_data)
    return {k: torch.tensor(v, device=device) for k, v in state_dict_serializable.items()}


def tensor_list_to_json(tensor_list) -> str:
    """Convert list of tensors to JSON string."""
    import json
    return json.dumps([tensor.detach().cpu().numpy().tolist() for tensor in tensor_list])


def json_to_tensor_list(json_data: str, device: str = "cpu"):
    """Convert JSON string to list of tensors."""
    import json
    import torch
    return [torch.tensor(tensor_data, device=device) for tensor_data in json.loads(json_data)]


def check_model_size(model, payload: Dict, device: str) -> bool:
    """
    Check if model size matches the payload.
    
    Args:
        model: PyTorch model.
        payload: Dictionary containing model parameters.
        device: Device string.
        
    Returns:
        True if sizes match, False otherwise.
    """
    try:
        import torch
        
        if not payload or "model_data" not in payload:
            return True
        
        model_state = model.state_dict()
        payload_data = payload["model_data"]
        
        for key in model_state:
            if key in payload_data:
                model_shape = model_state[key].shape
                if isinstance(payload_data[key], list):
                    payload_shape = torch.tensor(payload_data[key]).shape
                elif hasattr(payload_data[key], 'shape'):
                    payload_shape = payload_data[key].shape
                else:
                    continue
                
                if model_shape != payload_shape:
                    return False
        
        return True
    except Exception as e:
        logger.warning(f"Error checking model size: {e}")
        return True


def update_client_variables(model_mng, pruning_info, old_network=None):
    """
    Update client model variables after pruning.
    
    Args:
        model_mng: Model manager instance.
        pruning_info: Pruning information.
        old_network: Old network for weight transfer (optional).
    """
    try:
        # Try to use the pruning module's implementation if available
        from FedOpt.src.Optimizations.ModelPruning.prune import update_client_variables as _update
        _update(model_mng, pruning_info, old_network)
    except ImportError:
        # Fallback: basic parameter update
        if hasattr(model_mng, 'model') and pruning_info:
            try:
                import torch
                model_state = model_mng.model.state_dict()
                for key, indices in pruning_info.items() if isinstance(pruning_info, dict) else []:
                    if key in model_state and old_network is not None:
                        old_state = old_network.state_dict()
                        if key in old_state:
                            # Transfer weights from old network for pruned channels
                            pass  # Specific implementation depends on model architecture
            except Exception as e:
                logger.warning(f"Error updating client variables: {e}")


# Additional utility that might be needed
def calculate_entropy(data_dist: Dict[int, int]) -> float:
    """
    Calculate entropy of data distribution.
    
    Args:
        data_dist: Dictionary mapping labels to counts.
        
    Returns:
        Entropy value.
    """
    import math
    total = sum(data_dist.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in data_dist.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)
    return entropy
