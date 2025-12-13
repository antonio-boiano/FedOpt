# FL Communication Layer

A modular, clean communication layer for Federated Learning.

## Usage

### With CommunicationManager (recommended)
```python
from FedOpt.src.Communication import CommunicationManager

config = {
    "protocol": "grpc",  # or "grpc_async", "http"
    "mode": "server",    # or "client"
    "ip": "localhost",
    "port": 50051,
    # ... other config
}

manager = CommunicationManager(config)
manager.run_instance()
```

### Direct class usage
```python
from FedOpt.src.Communication.protocols.grpc_sync import GRPCServer
import asyncio

server = GRPCServer(config)
asyncio.run(server.main())
```

## Configuration

### Required
```yaml
ip: "localhost"
port: 50051
protocol: "grpc"
mode: "server"  # or "client"
```

### Server config
```yaml
server_config:
  rounds: 10              # Stop after N rounds
  max_time: 3600          # Stop after N seconds
  max_accuracy: 0.95      # Stop at accuracy
  client_round: 5         # Clients per round
  min_client_to_start: 5  # Wait for N clients
```

### Client config
```yaml
client_config:
  epochs: 5
  index: null  # Assigned by server
```


## Adding a New Protocol

### Step 1: Create Protocol File

Create a new file in `protocols/` directory:

```python
# protocols/myprotocol.py

from ..base import BaseClient, BaseServer, Message, MessageType
from ..registry import register_protocol

@register_protocol(
    "myprotocol",
    description="My custom protocol for FL",
    version="1.0",
    async_support=True
)
class MyProtocolClient(BaseClient):
    """Client implementation for my protocol."""
    
    def __init__(self, config):
        super().__init__(config)
        # Protocol-specific initialization
        self.connection = None
    
    async def connect(self):
        """Establish connection to the server."""
        # Your connection logic here
        pass
    
    async def disconnect(self):
        """Disconnect from the server."""
        # Your disconnection logic here
        pass
    
    async def send_message(self, message: Message):
        """Send a message to the server."""
        # Your send logic here
        pass
    
    async def receive_message(self) -> Message:
        """Receive a message from the server."""
        # Your receive logic here
        pass
    
    async def start_listening(self):
        """Start listening for incoming messages."""
        # Your listening logic here
        pass


@register_protocol("myprotocol")
class MyProtocolServer(BaseServer):
    """Server implementation for my protocol."""
    
    def __init__(self, config):
        super().__init__(config)
        # Protocol-specific initialization
    
    async def start(self):
        """Start the server."""
        # Your server start logic here
        pass
    
    async def stop(self):
        """Stop the server."""
        # Your server stop logic here
        pass
    
    async def send_to_client(self, client_id: str, message: Message):
        """Send a message to a specific client."""
        # Your send logic here
        pass
    
    async def broadcast(self, message: Message, client_ids=None):
        """Broadcast a message to multiple clients."""
        # Your broadcast logic here
        pass
```

### Step 2: Register Protocol

Add import to `protocols/__init__.py`:

```python
from .myprotocol import MyProtocolClient, MyProtocolServer
```

### Step 3: Use the Protocol

```python
from FedOpt.src.Communication import create_client, create_server

# Create instances
client = create_client("myprotocol", config)
server = create_server("myprotocol", config)

# Run
await client.main()
await server.main()
```

## Base Class Methods

### BaseClient

| Method | Required | Description |
|--------|----------|-------------|
| `connect()` | Yes | Establish connection to server |
| `disconnect()` | Yes | Clean up and disconnect |
| `send_message(message)` | Yes | Send a message to server |
| `receive_message()` | No | Receive a message from server |
| `start_listening()` | Yes | Start message listener |

### BaseServer

| Method | Required | Description |
|--------|----------|-------------|
| `start()` | Yes | Start the server |
| `stop()` | Yes | Stop the server |
| `send_to_client(client_id, message)` | Yes | Send to specific client |
| `broadcast(message, client_ids)` | Yes | Send to multiple clients |

## Message Types

Standard message types are defined in `MessageType` enum:

```python
class MessageType(Enum):
    SYNC = "sync"    # Initial synchronization
    INDEX = "index"  # Client index assignment
    DATA = "data"    # Model data exchange
    PRUNE = "prune"  # Pruning commands
    END = "end"      # End communication
    ACK = "ack"      # Acknowledgment
```

## Using Hooks

Both client and server support hooks for custom behavior:

```python
# Client hooks
client.add_on_connect_hook(lambda: print("Connected!"))
client.add_on_message_hook(lambda msg: print(f"Received: {msg.type}"))
client.add_on_disconnect_hook(lambda: print("Disconnected!"))

# Server hooks
server.add_on_client_connect_hook(lambda info: print(f"Client joined: {info.client_id}"))
server.add_on_client_disconnect_hook(lambda info: print(f"Client left: {info.client_id}"))
server.add_on_round_start_hook(lambda r: print(f"Round {r} starting"))
server.add_on_round_end_hook(lambda r, acc: print(f"Round {r} ended, accuracy: {acc}"))
```

## Example: HTTP/REST Protocol

Here's a complete example of implementing an HTTP-based protocol:

```python
import aiohttp
import json
from ..base import BaseClient, BaseServer, Message, MessageType
from ..registry import register_protocol

@register_protocol("http", description="HTTP-based FL communication")
class HTTPClient(BaseClient):
    def __init__(self, config):
        super().__init__(config)
        self.base_url = f"http://{self.ip}:{self.port}"
        self.session = None
    
    async def connect(self):
        self.session = aiohttp.ClientSession()
    
    async def disconnect(self):
        if self.session:
            await self.session.close()
    
    async def send_message(self, message: Message):
        async with self.session.post(
            f"{self.base_url}/message",
            json=message.to_dict()
        ) as resp:
            return await resp.json()
    
    async def receive_message(self) -> Message:
        async with self.session.get(f"{self.base_url}/message") as resp:
            data = await resp.json()
            return Message.from_dict(data)
    
    async def start_listening(self):
        while not self.end_event.is_set():
            try:
                message = await self.receive_message()
                await self.handle_message(message)
            except Exception as e:
                await asyncio.sleep(1)
```

