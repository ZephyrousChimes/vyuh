---
title: Traffic Env Environment Server
emoji: 🚦
colorFrom: red
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Traffic Env Environment

A traffic network environment that provides control of all the traffic signals to your agent.

There are multiple levels of grading, which correspond to more and more tedious traffic conditions. They range from simple rational impatient driver queues to priority vehicles that need routing at the earliest.

The environment understands that road networks and intersection abilities and semantics vary greatly across the world. To accomodate that, it has been kept agnostic.

The current implementation just implements on traffic intersection for roads that are left-aligned (India, Europe, etc.).

The transfer to other conventions and actual road network topologies is fairly straightforward, in the given framework.

## Quick Start

The simplest way to use the Traffic Env environment is through the `TrafficEnv` class:

```python
from traffic_env import TrafficAction, TrafficEnv

try:
    # Create environment from Docker image
    traffic_envenv = TrafficEnv.from_docker_image("traffic_env-env:latest")

    # Reset
    result = traffic_envenv.reset()
    print(f"Reset: {result.observation.echoed_message}")

    # Send multiple messages
    messages = ["Hello, World!", "Testing echo", "Final message"]

    for msg in messages:
        result = traffic_envenv.step(TrafficAction(message=msg))
        print(f"Sent: '{msg}'")
        print(f"  → Echoed: '{result.observation.echoed_message}'")
        print(f"  → Length: {result.observation.message_length}")
        print(f"  → Reward: {result.reward}")

finally:
    # Always clean up
    traffic_envenv.close()
```

That's it! The `TrafficEnv.from_docker_image()` method handles:
- Starting the Docker container
- Waiting for the server to be ready
- Connecting to the environment
- Container cleanup when you call `close()`

## Building the Docker Image

Before using the environment, you need to build the Docker image:

```bash
# From project root
docker build -t traffic_env-env:latest -f server/Dockerfile .
```

## Deploying to Hugging Face Spaces

You can easily deploy your OpenEnv environment to Hugging Face Spaces using the `openenv push` command:

```bash
# From the environment directory (where openenv.yaml is located)
openenv push

# Or specify options
openenv push --namespace my-org --private
```

The `openenv push` command will:
1. Validate that the directory is an OpenEnv environment (checks for `openenv.yaml`)
2. Prepare a custom build for Hugging Face Docker space (enables web interface)
3. Upload to Hugging Face (ensuring you're logged in)

### Prerequisites

- Authenticate with Hugging Face: The command will prompt for login if not already authenticated

### Options

- `--directory`, `-d`: Directory containing the OpenEnv environment (defaults to current directory)
- `--repo-id`, `-r`: Repository ID in format 'username/repo-name' (defaults to 'username/env-name' from openenv.yaml)
- `--base-image`, `-b`: Base Docker image to use (overrides Dockerfile FROM)
- `--private`: Deploy the space as private (default: public)

### Examples

```bash
# Push to your personal namespace (defaults to username/env-name from openenv.yaml)
openenv push

# Push to a specific repository
openenv push --repo-id my-org/my-env

# Push with a custom base image
openenv push --base-image ghcr.io/meta-pytorch/openenv-base:latest

# Push as a private space
openenv push --private

# Combine options
openenv push --repo-id my-org/my-env --base-image custom-base:latest --private
```

After deployment, your space will be available at:
`https://huggingface.co/spaces/<repo-id>`

The deployed space includes:
- **Web Interface** at `/web` - Interactive UI for exploring the environment
- **API Documentation** at `/docs` - Full OpenAPI/Swagger interface
- **Health Check** at `/health` - Container health monitoring
- **WebSocket** at `/ws` - Persistent session endpoint for low-latency interactions

## Environment Details

### Action
**TrafficAction**: Contains a single field
- `message` (str) - The message to echo back

### Observation
**TrafficObservation**: Contains the echo response and metadata
- `echoed_message` (str) - The message echoed back
- `message_length` (int) - Length of the message
- `reward` (float) - Reward based on message length (length × 0.1)
- `done` (bool) - Always False for echo environment
- `metadata` (dict) - Additional info like step count

### Reward
The reward is calculated as: `message_length × 0.1`
- "Hi" → reward: 0.2
- "Hello, World!" → reward: 1.3
- Empty message → reward: 0.0

## Advanced Usage

### Connecting to an Existing Server

If you already have a Traffic Env environment server running, you can connect directly:

```python
from traffic_env import TrafficEnv

# Connect to existing server
traffic_envenv = TrafficEnv(base_url="<ENV_HTTP_URL_HERE>")

# Use as normal
result = traffic_envenv.reset()
result = traffic_envenv.step(TrafficAction(message="Hello!"))
```

Note: When connecting to an existing server, `traffic_envenv.close()` will NOT stop the server.

### Using the Context Manager

The client supports context manager usage for automatic connection management:

```python
from traffic_env import TrafficAction, TrafficEnv

# Connect with context manager (auto-connects and closes)
with TrafficEnv(base_url="http://localhost:8000") as env:
    result = env.reset()
    print(f"Reset: {result.observation.echoed_message}")
    # Multiple steps with low latency
    for msg in ["Hello", "World", "!"]:
        result = env.step(TrafficAction(message=msg))
        print(f"Echoed: {result.observation.echoed_message}")
```

The client uses WebSocket connections for:
- **Lower latency**: No HTTP connection overhead per request
- **Persistent session**: Server maintains your environment state
- **Efficient for episodes**: Better for many sequential steps

### Concurrent WebSocket Sessions

The server supports multiple concurrent WebSocket connections. To enable this,
modify `server/app.py` to use factory mode:

```python
# In server/app.py - use factory mode for concurrent sessions
app = create_app(
    TrafficEnvironment,  # Pass class, not instance
    TrafficAction,
    TrafficObservation,
    max_concurrent_envs=4,  # Allow 4 concurrent sessions
)
```

Then multiple clients can connect simultaneously:

```python
from traffic_env import TrafficAction, TrafficEnv
from concurrent.futures import ThreadPoolExecutor

def run_episode(client_id: int):
    with TrafficEnv(base_url="http://localhost:8000") as env:
        result = env.reset()
        for i in range(10):
            result = env.step(TrafficAction(message=f"Client {client_id}, step {i}"))
        return client_id, result.observation.message_length

# Run 4 episodes concurrently
with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(run_episode, range(4)))
```

## Development & Testing

### Direct Environment Testing

Test the environment logic directly without starting the HTTP server:

```bash
# From the server directory
python3 server/traffic_env_environment.py
```

This verifies that:
- Environment resets correctly
- Step executes actions properly
- State tracking works
- Rewards are calculated correctly

### Running Locally

Run the server locally for development:

```bash
uvicorn server.app:app --reload
```

## Project Structure

```
traffic_env/
├── .dockerignore         # Docker build exclusions
├── __init__.py            # Module exports
├── README.md              # This file
├── openenv.yaml           # OpenEnv manifest
├── pyproject.toml         # Project metadata and dependencies
├── uv.lock                # Locked dependencies (generated)
├── client.py              # TrafficEnv client
├── models.py              # Action and Observation models
└── server/
    ├── __init__.py        # Server module exports
    ├── traffic_env_environment.py  # Core environment logic
    ├── app.py             # FastAPI application (HTTP + WebSocket endpoints)
    └── Dockerfile         # Container image definition
```


## The Traffic Model

We model traffic in two layers
- **The Control Plane**: A graph of nodes and roads connecting them as edges
- **The Data Plane**: A mini-graph at each node, where vertices are roads and edges being possible transfer from one road to another

This distinction helps separate global routing and local scheduling tasks.

## Observation Model

At each turn, the agent can see the entire road network state.
We have described below what a network state constitutes.

1) The Nodes
Nodes are possible points where you can control the traffic using a traffic signal.
It is of two types- JUNCTION and ENDPOINT.

JUNCTIONs are actual intersections where we can direct flow of traffic.

ENDPOINTs are sources and sinks of traffic. Roads beyond these nodes are NOT under our jurisdiction.

2) The Roads
Roads are edges between nodes in the control plane.
Roads are nodes in the data plane graph at each intersection.

Roads are the dynamic element in the simulation.

Traffic is loaded onto and out of roads in the form of 
`inflight` and `waiting` queue lengths. 

3) The Intersections
Intersection are the data plane graph representation.
Each node has its own Intersection.
An intersection contains an edge-list that tells our options to *route* traffic from one road to another road. The source road and target road are encapsulate in a Route object.

Each intersection has a Phase- the collection of routes that are opened at a time.

Intersection has a set of *safe* phases in the `phase_set` attribute. The agent is allowed only to choose from this phase set.

4) RoadNetwork

This is the Observation.
It has three things-
- The Node List
- The Road List (it is an edge list)
- The Intersection List

Looking at each road in the list, particularly its `inflight` and `waiting` attributes, one can get information of the moving traffic.

## Actions

The Action is simple- for each intersection, tell the phase it is in. Done.

```python
TrafficAction()
```

## Reward Model

### Grader 1: Do not attend to empty lanes instead of waiting lanes

### Grader 2: Grader 1 + Handle sudden surges

### Grader 3: Grader 2 + Handle priority vehicles

One can choose the level they want the environment to test them on.