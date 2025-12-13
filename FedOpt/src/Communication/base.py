#!/usr/bin/env python3
"""
Abstract Base Classes for Federated Learning Communication Layer

This module provides a generic abstraction layer for different communication protocols.
New protocols can be added by inheriting from BaseClient and BaseServer and implementing
the required abstract methods.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import time
import random
import logging
from copy import deepcopy
from collections import defaultdict

logger = logging.getLogger("FedOpt")


class MessageType(Enum):
    """Standard message types for FL communication."""
    SYNC = "sync"
    INDEX = "index"
    DATA = "data"
    PRUNE = "prune"
    END = "end"
    ACK = "ack"


@dataclass
class Message:
    """Standard message structure for FL communication."""
    type: MessageType
    payload: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value if isinstance(self.type, MessageType) else self.type,
            "payload": self.payload,
            **self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        msg_type = data.get("type", "")
        try:
            msg_type = MessageType(msg_type)
        except ValueError:
            pass  # Keep as string if not a known type
        
        payload = data.get("payload")
        metadata = {k: v for k, v in data.items() if k not in ("type", "payload")}
        return cls(type=msg_type, payload=payload, metadata=metadata)


@dataclass
class ClientInfo:
    """Information about a connected client."""
    client_id: str
    index: int
    address: Optional[str] = None
    port: Optional[int] = None
    connection: Any = None  # Protocol-specific connection object
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ClientSelectionData:
    """Data used for intelligent client selection."""
    communication_time: float
    training_time: float
    data_size: int
    data_entropy: float
    accuracy_delta: float


class BaseClient(ABC):
    """
    Abstract base class for FL clients.
    
    Subclasses must implement the protocol-specific methods for connection
    management and message passing.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the base client.
        
        Args:
            config: Configuration dictionary containing all client settings.
        """
        self.config = config
        self.ip = config.get("ip")
        self.port = config.get("port")
        self.device = config.get("device", "cpu")
        self.epochs = config.get("client_config", {}).get("epochs", 1)
        self.sleep_time = config.get("sleep_time", 1)
        self.client_selection_enabled = config.get("client_selection", False)
        
        # Client state
        self.index: Optional[int] = config.get("client_config", {}).get("index")
        self.client_id: Optional[str] = None
        self.is_running = False
        self.end_event = asyncio.Event()
        
        # Dataset information
        self.num_samples = 0
        self.data_dist: Dict[int, int] = {}
        
        # Model management - initialized by _init_model_components
        self.model_manager = None
        self.model_pruning = None
        
        # Hooks for custom behavior
        self._on_connect_hooks: List[Callable] = []
        self._on_message_hooks: List[Callable] = []
        self._on_disconnect_hooks: List[Callable] = []
        
        # Initialize components
        self._init_model_components()
    
    # ==================== Component Initialization ====================
    
    def _init_model_components(self) -> None:
        """
        Initialize model manager and pruning components.
        
        This is called automatically during __init__. Override if you need
        custom initialization logic.
        """
        try:
            from FedOpt.src.Federation.manager import model_manager
            from FedOpt.src.Optimizations.ModelPruning.prune import ModelPruning
            
            self.model_manager = model_manager(self.config)
            self.model_pruning = ModelPruning(
                self.config.get("prune_ratio", 0.1),
                self.model_manager.model
            )
        except ImportError as exc:
            logger.warning(f"Could not initialize model components: {exc}")
    
    # ==================== Abstract Methods ====================
    
    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the server."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the server."""
        pass
    
    @abstractmethod
    async def send_message(self, message: Message) -> None:
        """
        Send a message to the server.
        
        Args:
            message: The message to send.
        """
        pass
    
    @abstractmethod
    async def start_listening(self) -> None:
        """Start listening for incoming messages."""
        pass
    
    # ==================== Optional Methods ====================
    
    async def receive_message(self) -> Message:
        """
        Receive a message from the server (polling-based).
        
        This is optional and only needs to be implemented by protocols
        that use polling/pull-based message reception (e.g., TCP).
        
        Push/callback-based protocols (gRPC streaming, MQTT, HTTP webhooks)
        should call `handle_message` directly from their callbacks.
        
        Returns:
            The received message.
            
        Raises:
            NotImplementedError: If the protocol uses push-based messaging.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} uses push-based messaging. "
            "Messages are handled via callbacks, not polling."
        )
    
    # ==================== Common Implementation ====================
    
    async def main(self) -> None:
        """Main client loop."""
        try:
            self.is_running = True
            await self.connect()
            await self._run_hooks(self._on_connect_hooks)
            
            # Send initial sync message
            await self.send_message(Message(type=MessageType.SYNC, payload=""))
            logger.info("Sent synchronization message to server")
            
            await self.start_listening()
            
            # Wait until end event is set
            while not self.end_event.is_set():
                await asyncio.sleep(self.sleep_time)
                
        except KeyboardInterrupt:
            logger.warning("KeyboardInterrupt received, shutting down client")
        except Exception as e:
            logger.error(f"Client error: {e}")
            raise
        finally:
            self.is_running = False
            await self._run_hooks(self._on_disconnect_hooks)
            await self.disconnect()
    
    async def handle_message(self, message: Message) -> None:
        """
        Handle an incoming message based on its type.
        
        Args:
            message: The received message.
        """
        receive_time = time.perf_counter()
        await self._run_hooks(self._on_message_hooks, message)
        
        if message.type == MessageType.INDEX or message.type == "index":
            await self._handle_index_message(message)
        elif message.type == MessageType.DATA or message.type == "data":
            await self._handle_data_message(message, receive_time)
        elif message.type == MessageType.PRUNE or message.type == "prune":
            await self._handle_prune_message(message)
        elif message.type == MessageType.END or message.type == "end":
            await self._handle_end_message(message)
        else:
            logger.warning(f"Received unknown message type: {message.type}")
    
    async def _handle_index_message(self, message: Message) -> None:
        """Handle index assignment message."""
        if self.index is None:
            self.index = message.payload
            logger.info(f"Client index assigned: {self.index}")
        else:
            logger.info(f"Client index: {self.index}, server suggested: {message.payload}")
        self.analyze_dataset()
    
    async def _handle_data_message(self, message: Message, receive_time: float) -> None:
        """Handle model data message - train and respond."""
        if self.model_manager is None:
            logger.error("Model manager not initialized")
            return
        
        payload = message.payload
        
        # Check and adjust model size if necessary
        while not self._check_model_size(payload):
            logger.info("Adjusting model size to match server model")
            await self._adjust_model_size(payload)
        
        # Update local model with server parameters
        self.model_manager.set_client_model(payload)
        
        # Train the model
        training_time = self.model_manager.train(self.epochs, self.index or 0)
        accuracy = self.model_manager.evaluate()
        
        # Prepare response
        response_payload = self.model_manager.get_client_model()
        response_payload["client_time"] = time.perf_counter() - receive_time
        
        # Add client info if selection is enabled
        if self.client_selection_enabled:
            response_payload["client_info"] = {
                "training_time": training_time,
                "data_size": self.num_samples,
                "data_dist": self.data_dist,
                "accuracy": accuracy,
                "client_time": time.perf_counter() - receive_time
            }
        
        logger.info(f"Training complete. Accuracy: {accuracy:.4f}")
        await self.send_message(Message(type=MessageType.DATA, payload=response_payload))
    
    async def _handle_prune_message(self, message: Message) -> None:
        """Handle pruning request."""
        if self.model_pruning is None:
            logger.error("Model pruning not initialized")
            return
        
        result = self.model_pruning.client_fed_prune(self.model_manager)
        # Convert tensors to JSON-serializable format
        result_json = self._tensor_list_to_json(result)
        await self.send_message(Message(type=MessageType.PRUNE, payload=result_json))
        logger.info("Sent pruning data to server")
    
    async def _handle_end_message(self, message: Message) -> None:
        """Handle end of communication."""
        logger.info("Received END signal from server")
        self.end_event.set()
    
    def _check_model_size(self, payload: Dict[str, Any]) -> bool:
        """Check if local model size matches server model."""
        # This should be implemented based on your model structure
        # Default implementation assumes sizes match
        if self.model_manager is None:
            return True
        try:
            from FedOpt.src.Communication.communication import check_model_size
            return check_model_size(self.model_manager.model, payload, self.device)
        except ImportError:
            return True
    
    async def _adjust_model_size(self, payload: Dict[str, Any]) -> None:
        """Adjust local model to match server model size."""
        if self.model_pruning is None:
            return
        
        array_sum_of_kernel = self.model_pruning.client_fed_prune(self.model_manager)
        old_network = deepcopy(self.model_manager.model)
        channel_index_pruned = self.model_pruning.new_network(
            self.model_manager.model, array_sum_of_kernel
        )
        
        if "pruning_info" in payload:
            logger.debug("Using server pruning info")
            self._update_client_variables(payload["pruning_info"], old_network)
        else:
            logger.warning("No pruning info from server, using local indices")
            self._update_client_variables(channel_index_pruned, None)
    
    def _update_client_variables(self, pruning_info: Any, old_network: Any) -> None:
        """Update client model variables after pruning."""
        try:
            from FedOpt.src.Communication.communication import update_client_variables
            update_client_variables(self.model_manager, pruning_info, old_network)
        except ImportError:
            pass
    
    def _tensor_list_to_json(self, tensor_list: List) -> str:
        """Convert tensor list to JSON string."""
        try:
            from FedOpt.src.Communication.communication import tensor_list_to_json
            return tensor_list_to_json(tensor_list)
        except ImportError:
            import json
            return json.dumps([t.tolist() for t in tensor_list])
    
    def analyze_dataset(self) -> None:
        """Analyze the training dataset for client selection."""
        if self.model_manager is None or self.index is None:
            return
        
        try:
            dataset = self.model_manager.dataset
            loader_idx = self.index % dataset.num_parts
            
            label_occurrences = defaultdict(int)
            for _, labels in dataset.train_loader[loader_idx]:
                for label in labels:
                    label_occurrences[label.item()] += 1
            
            self.data_dist = dict(label_occurrences)
            self.num_samples = sum(self.data_dist.values())
            logger.debug(f"Dataset analysis: {self.num_samples} samples")
        except Exception as e:
            logger.warning(f"Failed to analyze dataset: {e}")
    
    # ==================== Hook Management ====================
    
    def add_on_connect_hook(self, hook: Callable) -> None:
        """Add a hook to be called on connection."""
        self._on_connect_hooks.append(hook)
    
    def add_on_message_hook(self, hook: Callable) -> None:
        """Add a hook to be called on message receipt."""
        self._on_message_hooks.append(hook)
    
    def add_on_disconnect_hook(self, hook: Callable) -> None:
        """Add a hook to be called on disconnection."""
        self._on_disconnect_hooks.append(hook)
    
    async def _run_hooks(self, hooks: List[Callable], *args, **kwargs) -> None:
        """Run all hooks in a list."""
        for hook in hooks:
            try:
                if asyncio.iscoroutinefunction(hook):
                    await hook(*args, **kwargs)
                else:
                    hook(*args, **kwargs)
            except Exception as e:
                logger.error(f"Hook error: {e}")


class BaseServer(ABC):
    """
    Abstract base class for FL servers.
    
    Subclasses must implement the protocol-specific methods for connection
    management and message passing.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the base server.
        
        Args:
            config: Configuration dictionary containing all server settings.
        """
        self.config = config
        self.protocol_name = config.get("protocol", "unknown")
        self.ip = config.get("ip", "0.0.0.0")
        self.port = config.get("port", 8080)
        self.sleep_time = config.get("sleep_time", 1)
        
        # Server configuration
        server_config = config.get("server_config", {})
        self.rounds_limit = server_config.get("rounds")
        self.max_time = server_config.get("max_time")
        self.max_accuracy = server_config.get("max_accuracy")
        self.client_round = server_config.get("client_round", 2)
        self.min_client_to_start = server_config.get("min_client_to_start", 2)
        
        # Log stopping condition status
        stop_conditions = []
        if self.rounds_limit and self.rounds_limit > 0:
            stop_conditions.append(f"rounds={self.rounds_limit}")
        if self.max_time and self.max_time > 0:
            stop_conditions.append(f"max_time={self.max_time}s")
        if self.max_accuracy and self.max_accuracy > 0:
            stop_conditions.append(f"max_accuracy={self.max_accuracy}")
        
        if stop_conditions:
            logger.info(f"Stopping conditions: {', '.join(stop_conditions)}")
        else:
            logger.warning("No stopping condition configured. Server will run indefinitely.")
        
        # Feature flags
        self.dyn_sampling = config.get("dyn_sampling", False)
        self.performance_selection = config.get("client_selection", False)
        self.pruning_flag = config.get("model_pruning", False)
        self.prune_interval = config.get("prune_interval", 10)
        self.synchronicity = config.get("synchronicity", 1)
        self.client_clustering = config.get("client_clustering", False)
        
        # Client management
        self.clients: Dict[str, ClientInfo] = {}
        self.client_count = 0
        self.client_index_counter = 0
        self.client_selected: List[str] = []
        self.client_responses: Dict[str, Any] = {}
        self.client_flag: List[str] = []
        self.client_sel_data: Dict[str, ClientSelectionData] = {}
        self.new_connections: Dict[str, int] = {}
        self.used_addresses: List[str] = []
        
        # Federation state
        self.current_round = 0
        self.round_type = "train"
        self.last_accuracy = 0.0
        self.accuracies: List[float] = []
        self.model_data: Optional[Dict[str, Any]] = None
        self.channel_indexes_pruned: List = []
        self.send_times: Dict[str, float] = {}
        self.is_running = False
        self.start_time: Optional[float] = None
        
        # Federation and optimization components - initialized by _init_federation_components
        self.federation = None
        self.dynamic_sampling = None
        self.model_pruning = None
        
        # Synchronization
        self.message_lock = asyncio.Lock()
        
        # Output configuration
        self.accuracy_file = config.get("accuracy_file", "accuracy.png")
        
        # Hooks
        self._on_client_connect_hooks: List[Callable] = []
        self._on_client_disconnect_hooks: List[Callable] = []
        self._on_round_start_hooks: List[Callable] = []
        self._on_round_end_hooks: List[Callable] = []
        
        # Initialize components
        self._init_federation_components()
    
    # ==================== Component Initialization ====================
    
    def _init_federation_components(self) -> None:
        """
        Initialize federation manager and optimization components.
        
        This is called automatically during __init__. Override if you need
        custom initialization logic.
        """
        try:
            from FedOpt.src.Federation.manager import federation_manager
            from FedOpt.src.Optimizations.DynamicSampling.dynamic_sampling import DynamicSampling
            from FedOpt.src.Optimizations.ModelPruning.prune import ModelPruning
            
            self.federation = federation_manager(self.config)
            self.dynamic_sampling = DynamicSampling(
                self.config.get("decay_rate", 0.1),
                self.client_round
            )
            self.model_pruning = ModelPruning(
                self.config.get("prune_ratio", 0.1),
                self.federation.server_model.model
            )
        except ImportError as exc:
            logger.warning(f"Could not initialize federation components: {exc}")
    
    # ==================== Abstract Methods ====================
    
    @abstractmethod
    async def start(self) -> None:
        """Start the server and begin listening for connections."""
        pass
    
    @abstractmethod
    async def stop(self) -> None:
        """Stop the server gracefully."""
        pass
    
    @abstractmethod
    async def send_to_client(self, client_id: str, message: Message) -> None:
        """
        Send a message to a specific client.
        
        Args:
            client_id: The client's identifier.
            message: The message to send.
        """
        pass
    
    @abstractmethod
    async def broadcast(self, message: Message, client_ids: Optional[List[str]] = None) -> None:
        """
        Broadcast a message to multiple clients.
        
        Args:
            message: The message to broadcast.
            client_ids: List of client IDs. If None, broadcast to all.
        """
        pass
    
    # ==================== Common Implementation ====================
    
    async def main(self) -> None:
        """Main server loop."""
        try:
            self.start_time = time.time()
            self.is_running = True
            
            await self.start()
            logger.info(f"Server started on {self.ip}:{self.port}")
            
            await self.federate()
            
        except KeyboardInterrupt:
            logger.warning("KeyboardInterrupt received, shutting down server")
        except asyncio.CancelledError:
            logger.warning("Server cancelled")
        except Exception as e:
            logger.error(f"Server error: {e}")
            raise
        finally:
            self.is_running = False
            await self.stop()
    
    async def handle_client_message(self, client_id: str, message: Message) -> None:
        """
        Handle an incoming message from a client.
        
        Args:
            client_id: The client's identifier.
            message: The received message.
        """
        receive_time = time.perf_counter()
        
        # Get the message type as string for comparison
        msg_type = message.type.value if isinstance(message.type, MessageType) else message.type
        logger.debug(f"handle_client_message: client={client_id}, type={msg_type}, "
                   f"client_in_registry={client_id in self.clients}")
        
        # Handle new client registration (only if not already registered by protocol-specific code)
        if client_id not in self.clients:
            logger.info(f"New client {client_id} not in registry, registering via base class")
            await self._register_client(client_id, message)
        
        # Only process data and prune messages for response collection
        if msg_type in ["data", "prune"]:
            self.client_flag.append(client_id)
            self.client_responses[client_id] = message.payload
            logger.info(f"Stored {msg_type} response from {client_id}. "
                       f"Total responses: {len(self.client_responses)}/{self.client_round}")
            
            # Calculate timing information
            if client_id in self.send_times:
                total_time = receive_time - self.send_times[client_id]
                
                if message.payload and "client_time" in message.payload:
                    comm_time = round(total_time - message.payload["client_time"], 6)
                    logger.info(f"Round {self.current_round}, client {client_id} comm time: {comm_time}")
                
                if message.payload and "client_info" in message.payload:
                    self._store_client_selection_data(client_id, message.payload, total_time)
            else:
                logger.warning(f"No send_time recorded for {client_id} - might be late or duplicate response")
    
    async def _register_client(self, client_id: str, message: Message) -> None:
        """Register a new client."""
        self.client_index_counter += 1
        self.client_count += 1
        
        client_info = ClientInfo(
            client_id=client_id,
            index=self.client_index_counter
        )
        self.clients[client_id] = client_info
        self.new_connections[client_id] = self.client_index_counter
        
        logger.info(f"New client registered: {client_id} (index={self.client_index_counter})")
        
        # Send index assignment
        await self.send_to_client(client_id, Message(
            type=MessageType.INDEX,
            payload=self.client_index_counter
        ))
        
        await self._run_hooks(self._on_client_connect_hooks, client_info)
    
    def _store_client_selection_data(self, client_id: str, payload: Dict, total_time: float) -> None:
        """Store client performance data for selection."""
        try:
            client_data = payload["client_info"]
            self.client_sel_data[client_id] = ClientSelectionData(
                communication_time=round(total_time - client_data.get("client_time", 0), 7),
                training_time=round(client_data.get("training_time", 0), 5),
                data_size=client_data.get("data_size", 0),
                data_entropy=self._calculate_entropy(client_data.get("data_dist", {})),
                accuracy_delta=round(client_data.get("accuracy", 0) - self.last_accuracy, 6)
            )
            logger.debug(f"Stored selection data for {client_id}")
        except Exception as e:
            logger.warning(f"Failed to store selection data: {e}")
    
    @staticmethod
    def _calculate_entropy(data_dist: Dict[int, int]) -> float:
        """Calculate entropy of data distribution."""
        try:
            from FedOpt.src.Optimizations.ClientSelection.client_selection import calculate_entropy
            return calculate_entropy(data_dist)
        except ImportError:
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
    
    async def federate(self) -> None:
        """Main federation loop."""
        self.current_round = 1
        
        # Wait for minimum clients
        logger.info(f"Waiting for {self.min_client_to_start} clients to connect...")
        wait_counter = 0
        while self.client_count < self.min_client_to_start or self.client_count < self.client_round:
            await asyncio.sleep(self.sleep_time)
            wait_counter += 1
            # Log every 5 seconds (25 iterations at 0.2s sleep)
            if wait_counter % 25 == 0:
                logger.info(f"Still waiting: {self.client_count}/{self.min_client_to_start} clients, "
                           f"registered: {list(self.clients.keys())}")
        
        logger.info(f"Minimum clients reached ({self.client_count} connected). Starting federation.")
        
        # First training round
        self.round_type = "train"
        logger.info("-- START Training --")
        self._select_clients(first_round=True)
        await self._send_model_to_selected_clients()
        
        loop_counter = 0
        while True:
            min_clients, num_processed = self._get_aggregation_params()
            num_responses = len(self.client_responses)
            
            # Log loop status periodically
            loop_counter += 1
            if loop_counter % 50 == 0:  # Log every 10 seconds (at 0.2s sleep)
                logger.info(f"Waiting for responses: {num_responses}/{min_clients}, "
                           f"selected={self.client_selected}, "
                           f"responded={list(self.client_responses.keys())}")
            
            if num_responses >= min_clients:
                logger.info(f"Received {num_responses} responses (needed {min_clients}). Starting aggregation.")
                await self._run_hooks(self._on_round_start_hooks, self.current_round)
                
                # Perform aggregation
                await self._perform_aggregation(num_processed)
                
                self.accuracies.append(self.last_accuracy)
                await self._run_hooks(self._on_round_end_hooks, self.current_round, self.last_accuracy)
                
                # Check stopping condition
                if self._check_stop_condition():
                    await self._finish_federation()
                    break
                
                # Prepare next round
                await self._prepare_next_round()
                loop_counter = 0  # Reset counter for new round
            
            await asyncio.sleep(self.sleep_time)
    
    def _get_aggregation_params(self) -> tuple:
        """Get aggregation parameters based on synchronicity mode."""
        if self.synchronicity == 1:
            # Synchronous: wait for client_round responses, process client_round models
            return self.client_round, self.client_round
        elif self.client_clustering:
            # Async with clustering: aggregate all received responses
            return 1, len(self.client_responses)
        else:
            # Async without clustering: aggregate one at a time
            return 1, 1
    
    async def _perform_aggregation(self, num_processed: int) -> None:
        """Perform model aggregation."""
        if self.federation is None:
            logger.error("Federation manager not initialized")
            return
        
        try:
            if self.synchronicity == 1:
                if self.round_type == "train":
                    logger.info(f"START synchronous federation #{self.current_round} "
                               f"with {len(self.client_responses)} client responses")
                    self.federation.apply(
                        aggregated_dict=self.client_responses,
                        num_clients=num_processed
                    )
                    self.last_accuracy = self.federation.server_model.evaluate()
                    logger.info(f"END Federation #{self.current_round}, accuracy: {self.last_accuracy:.4f}")
                elif self.round_type == "prune":
                    logger.info("START Model pruning aggregation")
                    if self.model_pruning:
                        self.channel_indexes_pruned = self.model_pruning.server_fed_prune(
                            self.federation, self.client_responses
                        )
                    logger.info("END Model pruning")
                    self.federation.server_model.evaluate()
                
                self._reset_round_data()
            else:
                # Asynchronous aggregation
                if self.round_type == "train":
                    logger.info(f"START asynchronous federation #{self.current_round}")
                    keys_subset = list(self.client_responses.keys())[:num_processed]
                    aggregated_dict = {k: self.client_responses[k] for k in keys_subset}
                    self.federation.apply(
                        aggregated_dict=aggregated_dict,
                        num_clients=num_processed + 1
                    )
                    self.last_accuracy = self.federation.server_model.evaluate()
                    logger.info(f"END Federation (clients: {keys_subset})")
                elif self.round_type == "prune":
                    raise Exception("Pruning is not supported in asynchronous mode")
                
                self.used_addresses = list(aggregated_dict.keys())
                self._reset_round_data_async()
        except Exception as e:
            logger.error(f"Error during aggregation: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def _check_stop_condition(self) -> bool:
        """Check if federation should stop."""
        # Check max_time first (if set and > 0)
        if self.max_time is not None and self.max_time > 0:
            return time.time() - self.start_time >= self.max_time
        
        # Check rounds_limit (if set and > 0)
        if self.rounds_limit is not None and self.rounds_limit > 0:
            return self.current_round >= self.rounds_limit
        
        # Check max_accuracy (if set and > 0)
        if self.max_accuracy is not None and self.max_accuracy > 0:
            return self.last_accuracy >= self.max_accuracy
        
        # No stopping condition defined - default to never stop (run indefinitely)
        # This allows manual stopping via keyboard interrupt
        logger.warning("No stopping condition defined (rounds, max_time, or max_accuracy). "
                      "Server will run indefinitely until manually stopped.")
        return False
    
    async def _prepare_next_round(self) -> None:
        """Prepare for the next federation round."""
        # Check if pruning is needed
        if (self.pruning_flag and 
            self.round_type != "prune" and
            self.current_round % self.prune_interval == 0 and
            self.model_pruning and
            len(self.model_pruning.prune_layers) > 0 and
            self.current_round > 20):
            
            self.round_type = "prune"
            logger.info("-- START Pruning --")
        else:
            self.current_round += 1
            self.round_type = "train"
            logger.info("-- START Training --")
            self._select_clients(first_round=False)
        
        # Send appropriate messages
        if self.round_type == "train":
            await self._send_model_to_selected_clients()
        elif self.round_type == "prune":
            await self._send_prune_commands()
    
    async def _send_model_to_selected_clients(self) -> None:
        """Send current model to selected clients."""
        await self._create_model_message()
        
        message = Message(type=MessageType.DATA, payload=self.model_data)
        
        # Filter to only send to clients that are actually registered
        valid_clients = [cid for cid in self.client_selected if cid in self.clients]
        if len(valid_clients) != len(self.client_selected):
            missing = set(self.client_selected) - set(valid_clients)
            logger.warning(f"Some selected clients not found in registry: {missing}")
            self.client_selected = valid_clients
        
        logger.info(f"Sending model to {len(self.client_selected)} clients: {self.client_selected}")
        
        for client_id in self.client_selected:
            self.send_times[client_id] = time.perf_counter()
            await self.send_to_client(client_id, message)
    
    async def _send_prune_commands(self) -> None:
        """Send prune commands to selected clients."""
        message = Message(type=MessageType.PRUNE)
        for client_id in self.client_selected:
            self.send_times[client_id] = time.perf_counter()
            await self.send_to_client(client_id, message)
        logger.debug("Sent prune commands to clients")
    
    async def _create_model_message(self) -> None:
        """Create the model data message."""
        async with self.message_lock:
            if self.model_data is None and self.federation:
                logger.debug("Creating model data message")
                server_model = self.federation.get_server_model()
                if self.channel_indexes_pruned:
                    server_model["pruning_info"] = self.channel_indexes_pruned
                self.model_data = server_model
    
    async def _finish_federation(self) -> None:
        """Finish the federation process."""
        self._save_accuracy_figure()
        
        # Send end message to all clients
        end_message = Message(type=MessageType.END)
        for client_id in self.clients:
            await self.send_to_client(client_id, end_message)
        
        self.client_flag = []
        logger.info("END of federation rounds!")
    
    def _save_accuracy_figure(self) -> None:
        """Save accuracy plot."""
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(8, 5))
            plt.plot(self.accuracies, linestyle='-', color='b', label="Test Accuracy", linewidth=1)
            plt.xlabel("Round")
            plt.ylabel("Accuracy")
            plt.title("Accuracy over Rounds")
            plt.legend()
            plt.grid(True)
            plt.ylim(0, 1.0)
            plt.savefig(self.accuracy_file)
            plt.close()
            logger.info(f"Saved accuracy figure to {self.accuracy_file}")
        except Exception as e:
            logger.warning(f"Failed to save accuracy figure: {e}")
    
    def _select_clients(self, first_round: bool = False) -> None:
        """Select clients for the current round."""
        self.client_selected = []
        
        if self.dyn_sampling and not first_round and self.dynamic_sampling:
            if self.synchronicity != 1:
                raise Exception("Dynamic sampling not supported in async mode")
            self.client_round = self.dynamic_sampling.number_of_clients(self.current_round)
        
        elif self.performance_selection:
            if self.synchronicity != 1:
                raise Exception("Performance selection not supported in async mode")
            self._performance_based_selection()
        
        else:
            if self.synchronicity == 1:
                self.client_flag = []
                available = list(self.clients.keys())
                self.client_selected = random.sample(
                    available, 
                    min(self.client_round, len(available))
                )
            else:
                # Async: select new connections + previously used
                self.client_selected = list(self.new_connections.keys())
                self.client_selected.extend([
                    addr for addr in self.clients.keys() 
                    if addr in self.used_addresses
                ])
                self.new_connections = {}
        
        logger.info(f"Selected clients: {self.client_selected}")
    
    def _performance_based_selection(self) -> None:
        """Select clients based on performance metrics."""
        try:
            from FedOpt.src.Optimizations.ClientSelection.client_selection import get_score_sorted_clients
            sorted_ids = get_score_sorted_clients(self.client_sel_data)
            
            if sorted_ids:
                all_clients = set(self.clients.keys())
                missing = list(all_clients - set(sorted_ids))
                self.client_selected = (missing + sorted_ids)[:self.client_round]
            else:
                self.client_selected = random.sample(
                    list(self.clients.keys()),
                    min(self.client_round, len(self.clients))
                )
        except ImportError:
            self.client_selected = random.sample(
                list(self.clients.keys()),
                min(self.client_round, len(self.clients))
            )
    
    def _reset_round_data(self) -> None:
        """Reset data after a synchronous round."""
        self.client_responses.clear()
        self.client_flag = []
        self.model_data = None
    
    def _reset_round_data_async(self) -> None:
        """Reset data after an asynchronous round."""
        self.client_responses = {
            k: v for k, v in self.client_responses.items() 
            if k not in self.used_addresses
        }
        self.client_flag = [
            k for k in self.client_flag 
            if k not in self.used_addresses
        ]
        self.model_data = None
    
    async def remove_client(self, client_id: str) -> None:
        """Remove a client from the server."""
        if client_id in self.client_selected:
            self.client_selected.remove(client_id)
        
        if client_id in self.clients:
            client_info = self.clients.pop(client_id)
            self.client_count -= 1
            await self._run_hooks(self._on_client_disconnect_hooks, client_info)
            logger.info(f"Client removed: {client_id}")
    
    # ==================== Hook Management ====================
    
    def add_on_client_connect_hook(self, hook: Callable) -> None:
        """Add a hook for client connection events."""
        self._on_client_connect_hooks.append(hook)
    
    def add_on_client_disconnect_hook(self, hook: Callable) -> None:
        """Add a hook for client disconnection events."""
        self._on_client_disconnect_hooks.append(hook)
    
    def add_on_round_start_hook(self, hook: Callable) -> None:
        """Add a hook for round start events."""
        self._on_round_start_hooks.append(hook)
    
    def add_on_round_end_hook(self, hook: Callable) -> None:
        """Add a hook for round end events."""
        self._on_round_end_hooks.append(hook)
    
    async def _run_hooks(self, hooks: List[Callable], *args, **kwargs) -> None:
        """Run all hooks in a list."""
        for hook in hooks:
            try:
                if asyncio.iscoroutinefunction(hook):
                    await hook(*args, **kwargs)
                else:
                    hook(*args, **kwargs)
            except Exception as e:
                logger.error(f"Hook error: {e}")