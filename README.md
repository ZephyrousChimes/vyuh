---
title: Traffic Environment Server
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

# 🚦 Traffic Env

A reinforcement learning environment for traffic signal control at a 4-way intersection. Give an agent full visibility of the road network and let it learn to keep traffic flowing.

Built on [OpenEnv](https://github.com/meta-platforms/openenv), deployable locally or on Hugging Face Spaces.

---

## The Environment

Traffic is modelled in two layers:

- **Control Plane** — a graph of nodes and roads. Nodes are either `JUNCTION` (controllable intersections) or `ENDPOINT` (sources/sinks beyond our jurisdiction). Roads are edges between nodes.
- **Data Plane** — a mini-graph at each junction where roads are vertices and routes are edges. A *phase* is a set of routes that are simultaneously open.

The current implementation is a single 4-way intersection using a **left-hand traffic model** (India, UK, etc.) — left turns are always open.

### Observation

At each step the agent receives the full `RoadNetwork`:

| Field | Description |
|---|---|
| `nodes` | List of `JUNCTION` and `ENDPOINT` nodes |
| `roads` | Edges with `inflight` and `waiting` queue lengths |
| `intersections` | Data-plane graph with `routes`, `phase_set`, and active `phase` |

### Action

For each intersection, pick a phase from its `phase_set`:

| Phase | Opens |
|---|---|
| `ALL_RED` | Nothing — use for transitions |
| `NS_THROUGH` | North↔South through traffic + left turns |
| `EW_THROUGH` | East↔West through traffic + left turns |
| `N_RIGHT` | North right turn + left turns |
| `S_RIGHT` | South right turn + left turns |

```python
TrafficAction(decisions=[
    IntersectionPhaseDecision(
        intersection_id="intersection_center",
        phase_id="NS_THROUGH"
    )
])
```

### Reward

Each step returns a scalar reward combining three signals:

- **Pressure** — penalises total vehicles waiting across all inbound roads
- **Starvation** — quadratic penalty when any single road queue exceeds threshold
- **Throughput** — bonus for lanes that are fully clear

---

## Tasks

| Task | Description |
|---|---|
| `easy` | Uniform arrival rates across all roads |
| `medium` | Asymmetric flow — north road significantly busier |
| `hard` | Asymmetric flow + a mid-episode surge event on one road |

---

## Quick Start

### Connect to this Space

```python
from traffic_env import TrafficAction, TrafficEnv
from traffic_env.models import IntersectionPhaseDecision

async with TrafficEnv(base_url="https://etherealwhisper-traffic-env.hf.space") as env:
    result = await env.reset(task="easy")
    obs = result.observation

    # Inspect the network
    for road in obs.road_network.roads:
        print(f"{road.id}: waiting={road.waiting}, inflight={road.inflight}")

    # Take a step
    action = TrafficAction(decisions=[
        IntersectionPhaseDecision(
            intersection_id=obs.road_network.intersections[0].id,
            phase_id="NS_THROUGH"
        )
    ])
    result = await env.step(action)
    print(f"reward: {result.reward:.3f}")
```

### Run Locally with Docker

```bash
# Build
docker build -t traffic_env-env:latest -f server/Dockerfile .

# Run
docker run -p 8000:8000 traffic_env-env:latest
```

```python
async with TrafficEnv(base_url="http://localhost:8000") as env:
    result = await env.reset(task="medium")
```

### Run Locally with uvicorn

```bash
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
```

---

## Deploy Your Own Space

```bash
# From the environment directory
openenv push

# With options
openenv push --repo-id my-org/my-traffic-env --private
```

The deployed space exposes:

| Path | Description |
|---|---|
| `/web` | Interactive web UI |
| `/docs` | OpenAPI / Swagger docs |
| `/ws` | WebSocket endpoint |
| `/health` | Health check |

---

## Project Structure

```
traffic_env/
├── client.py                          # TrafficEnv WebSocket client
├── models.py                          # Action and Observation models
├── openenv.yaml                       # OpenEnv manifest
├── pyproject.toml                     # Project metadata
└── server/
    ├── app.py                         # FastAPI application
    ├── traffic_env_environment.py     # Core simulation logic
    └── Dockerfile
```