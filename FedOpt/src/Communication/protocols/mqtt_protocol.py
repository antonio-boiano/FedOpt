#!/usr/bin/env python3
"""
MQTT Protocol Implementation for FL Communication Layer

This module provides MQTT-based client and server implementations
using the abstract base classes.
"""

import json
import time
import socket
import random
import asyncio
import threading
import queue
from typing import Dict, Any, Optional, List

import paho.mqtt.client as mqtt

from ..base import BaseClient, BaseServer, Message, MessageType, ClientInfo
from ..registry import register_protocol, ProtocolRegistry

import logging
logger = logging.getLogger("FedOpt")


@register_protocol(
    "mqtt",
    description="MQTT-based federated learning communication",
    version="1.0",
    async_support=True
)
class MQTTClient(BaseClient):
    """MQTT-based FL client implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Generate client ID
        self.client_id = self._generate_client_id()
        logger.debug(f"Client ID: {self.client_id}")
        
        # MQTT client setup
        self.mqtt_client = mqtt.Client(client_id=self.client_id)
        self.mqtt_client.on_connect = self._on_connect
        self.mqtt_client.on_message = self._on_message
        
        # Message queue for async handling
        self._message_queue: queue.Queue = queue.Queue()
        self._worker_thread: Optional[threading.Thread] = None
        
        # Topics
        self._subscribe_topic = f"FedOpt/ModelUpdate/{self.client_id}"
        self._publish_topic = f"FedOpt/Model/{self.client_id}"
        
        # Initialize model components
        # self._init_model_components()
    
    def _generate_client_id(self) -> str:
        """Generate a unique client ID."""
        if self.index is not None:
            return str(self.index)

        localhost_range = ['127.0.0.1', 'localhost', '127.0.1.1', '::1', 
                          '0:0:0:0:0:0:0:1', '0:0:0:0:0:0:0:0', '::']
        
        try:
            my_ip = socket.gethostbyname(socket.gethostname())
        except:
            my_ip = '127.0.0.1'
        
        if my_ip in localhost_range:
            return str(random.randint(1, 10000))
        else:
            return my_ip.split('.')[-1]
    
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
        """Connect to the MQTT broker."""
        keepalive = 1000 if __debug__ else 60
        
        try:
            self.mqtt_client.connect(self.ip, self.port, keepalive)
            self.mqtt_client.loop_start()
            
            # Start worker thread for message processing
            self._worker_thread = threading.Thread(
                target=self._message_worker,
                name="MQTTClientWorker",
                daemon=True
            )
            self._worker_thread.start()
            
            logger.info(f"Connected to MQTT broker at {self.ip}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to connect to MQTT broker: {e}")
            raise
    
    async def disconnect(self) -> None:
        """Disconnect from the MQTT broker."""
        self.mqtt_client.disconnect()
        self.mqtt_client.loop_stop()
        logger.info("Disconnected from MQTT broker")
    
    async def send_message(self, message: Message) -> None:
        """Send a message to the server."""
        payload = json.dumps(message.to_dict())
        result = self.mqtt_client.publish(self._publish_topic, payload, qos=0)
        
        if result.rc != mqtt.MQTT_ERR_SUCCESS:
            logger.error(f"Failed to publish message: {result.rc}")
        else:
            logger.debug(f"Sent message of type {message.type}")
    
    async def receive_message(self) -> Message:
        """Receive a message (handled via callbacks)."""
        # Messages are handled via on_message callback
        raise NotImplementedError("MQTT uses callback-based message handling")
    
    async def start_listening(self) -> None:
        """Start listening for messages (already started in connect)."""
        # MQTT loop is already started in connect()
        pass
    
    def _on_connect(self, client, userdata, flags, rc):
        """MQTT connect callback."""
        if rc == 0:
            logger.debug(f"Connected to MQTT broker (rc={rc})")
            self.mqtt_client.subscribe(self._subscribe_topic)
        else:
            logger.error(f"Failed to connect to MQTT broker: rc={rc}")
    
    def _on_message(self, client, userdata, msg):
        """MQTT message callback - queue for async processing."""
        try:
            message_data = json.loads(msg.payload)
            self._message_queue.put((msg.topic, message_data))
        except Exception as e:
            logger.warning(f"Failed to parse message: {e}")
    
    def _message_worker(self) -> None:
        """Worker thread for processing messages."""
        while True:
            try:
                topic, message_data = self._message_queue.get(timeout=1)
                message = Message.from_dict(message_data)
                
                # Run async handler in event loop
                asyncio.run(self.handle_message(message))
                
            except queue.Empty:
                if self.end_event.is_set():
                    break
            except Exception as e:
                logger.error(f"Error processing message: {e}")
            finally:
                try:
                    self._message_queue.task_done()
                except:
                    pass


@register_protocol("mqtt")
class MQTTServer(BaseServer):
    """MQTT-based FL server implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # MQTT client setup
        self.mqtt_client = mqtt.Client()
        self.mqtt_client.on_connect = self._on_connect
        self.mqtt_client.on_message = self._on_message
        
        # Initialize federation components
    #     self._init_federation_components()
    
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
        """Start the MQTT server."""
        keepalive = 1000 if __debug__ else 60
        
        self.mqtt_client.connect(self.ip, self.port, keepalive)
        self.mqtt_client.subscribe("FedOpt/Model/#")
        self.mqtt_client.loop_start()
        
        logger.info(f"MQTT server started, listening on {self.ip}:{self.port}")
    
    async def stop(self) -> None:
        """Stop the MQTT server."""
        self.mqtt_client.disconnect()
        self.mqtt_client.loop_stop()
        logger.info("MQTT server stopped")
    
    async def send_to_client(self, client_id: str, message: Message) -> None:
        """Send a message to a specific client."""
        topic = f"FedOpt/ModelUpdate/{client_id}"
        payload = json.dumps(message.to_dict())
        
        self.mqtt_client.publish(topic, payload, qos=0)
        logger.debug(f"Sent message to client {client_id}")
    
    async def broadcast(self, message: Message, client_ids: Optional[List[str]] = None) -> None:
        """Broadcast a message to multiple clients."""
        targets = client_ids or list(self.clients.keys())
        
        for client_id in targets:
            await self.send_to_client(client_id, message)
    
    def _on_connect(self, client, userdata, flags, rc):
        """MQTT connect callback."""
        logger.debug(f"Server connected to broker (rc={rc})")
    
    def _on_message(self, client, userdata, msg):
        """MQTT message callback."""
        try:
            client_id = msg.topic.split('/')[-1]
            message_data = json.loads(msg.payload)
            message = Message.from_dict(message_data)
            
            logger.info(f"Received message from {client_id}: type={message.type}")
            
            # Handle message synchronously (MQTT runs in separate thread)
            asyncio.run(self.handle_client_message(client_id, message))
            
        except Exception as e:
            logger.error(f"Error handling message: {e}")
