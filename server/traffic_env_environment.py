from uuid import uuid4
from typing import Optional, Dict, List
import random
import numpy as np

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import (
        TrafficAction, TrafficObservation,
        RoadNetwork, Node, Road, Route, Phase, Intersection,
        NodeType, IntersectionPhaseDecision
    )
except ImportError:
    from models import (
        TrafficAction, TrafficObservation,
        RoadNetwork, Node, Road, Route, Phase, Intersection,
        NodeType, IntersectionPhaseDecision
    )


# ---------------------------------------------------------------------------
# Sim constants
# ---------------------------------------------------------------------------

MAX_STEPS       = 50
DISCHARGE_RATE  = 5    # vehicles drained per open route per step
INFLIGHT_DELAY  = 2    # steps before inflight vehicles join waiting


# ---------------------------------------------------------------------------
# Network factory
# ---------------------------------------------------------------------------

def create_single_intersection_network(
    seed: Optional[int] = 42,
    base_arrival_rate: float = 0.5,
    surge_step: Optional[int] = None,
    surge_road_id: Optional[str] = None,
    surge_multiplier: float = 3.0,
) -> RoadNetwork:
    """
    One junction, four endpoint nodes (N/S/E/W).
    Left-hand traffic (Indian road model) — left turns always open.
    """
    rng = random.Random(seed)

    center = Node(id='node-center', type=NodeType.JUNCTION)
    north  = Node(id='node-north', type=NodeType.ENDPOINT)
    south  = Node(id='node-south', type=NodeType.ENDPOINT)
    east   = Node(id='node-east', type=NodeType.ENDPOINT)
    west   = Node(id='node-west', type=NodeType.ENDPOINT)

    # Inbound roads (endpoint -> center)
    nc = Road(id='road-north-to-center', src=north.id,  dst=center.id, base_arrival_rate=base_arrival_rate)
    sc = Road(id='road-south-to-center', src=south.id,  dst=center.id, base_arrival_rate=base_arrival_rate)
    ec = Road(id='road-east-to-center', src=east.id,   dst=center.id, base_arrival_rate=base_arrival_rate)
    wc = Road(id='road-west-to-center', src=west.id,   dst=center.id, base_arrival_rate=base_arrival_rate)

    # Outbound roads (center -> endpoint)
    cn = Road(id='road-center-to-north', src=center.id, dst=north.id,  base_arrival_rate=0.0)
    cs = Road(id='road-center-to-south', src=center.id, dst=south.id,  base_arrival_rate=0.0)
    ce = Road(id='road-center-to-east', src=center.id, dst=east.id,   base_arrival_rate=0.0)
    cw = Road(id='road-center-to-west', src=center.id, dst=west.id,   base_arrival_rate=0.0)

    # Seed initial inflight vehicles
    for road in [nc, sc, ec, wc]:
        road.inflight = rng.randint(2, 8)

    # Apply surge config if set
    if surge_road_id and surge_step is not None:
        for road in [nc, sc, ec, wc]:
            if road.id == surge_road_id:
                road.surge_multiplier = surge_multiplier
                road.surge_step       = surge_step

    # ---------------------------------------------------------------------------
    # Routes — all 16 inroad x outroad combinations
    # Left turns (sw, wn, ne, es) are always open in left-hand traffic
    # ---------------------------------------------------------------------------
    nn = Route(id='route-north-center-road-to-center-north-road', inroad=nc.id, outroad=cn.id)  # U-turn — excluded from phases
    ns = Route(id='route-north-center-road-to-center-south-road', inroad=nc.id, outroad=cs.id)  # through
    ne = Route(id='route-north-center-road-to-center-east-road', inroad=nc.id, outroad=ce.id)  # left turn (always open)
    nw = Route(id='route-north-center-road-to-center-west-road', inroad=nc.id, outroad=cw.id)  # right turn

    sn = Route(id='route-south-center-road-to-center-north-road', inroad=sc.id, outroad=cn.id)  # through
    ss = Route(id='route-south-center-road-to-center-south-road', inroad=sc.id, outroad=cs.id)  # U-turn — excluded
    se = Route(id='route-south-center-road-to-center-east-road', inroad=sc.id, outroad=ce.id)  # right turn
    sw = Route(id='route-south-center-road-to-center-west-road', inroad=sc.id, outroad=cw.id)  # left turn (always open)

    en = Route(id='route-east-center-road-to-center-north-road', inroad=ec.id, outroad=cn.id)  # right turn
    es = Route(id='route-east-center-road-to-center-south-road', inroad=ec.id, outroad=cs.id)  # left turn (always open)
    ee = Route(id='route-east-center-road-to-center-east-road', inroad=ec.id, outroad=ce.id)  # U-turn — excluded
    ew = Route(id='route-east-center-road-to-center-west-road', inroad=ec.id, outroad=cw.id)  # through

    wn = Route(id='route-west-center-road-to-center-north-road', inroad=wc.id, outroad=cn.id)  # left turn (always open)
    ws = Route(id='route-west-center-road-to-center-south-road', inroad=wc.id, outroad=cs.id)  # right turn
    we = Route(id='route-west-center-road-to-center-east-road', inroad=wc.id, outroad=ce.id)  # through
    ww = Route(id='route-west-center-road-to-center-west-road', inroad=wc.id, outroad=cw.id)  # U-turn — excluded

    # Always-open left turns
    left_turns = [ne.id, sw.id, es.id, wn.id]

    all_routes = [nn, ns, ne, nw, sn, ss, se, sw, en, es, ee, ew, wn, ws, we, ww]

    # ---------------------------------------------------------------------------
    # Phase set — 4 phases + 1 all-red transition
    # Each phase = left turns (always) + one direction's through + right turns
    # ---------------------------------------------------------------------------
    phase_all_red = Phase(id="ALL_RED",  routes=[])

    phase_ns = Phase(id="NS_THROUGH", routes=left_turns + [ns.id, sn.id, nw.id, se.id])
    phase_ew = Phase(id="EW_THROUGH", routes=left_turns + [ew.id, we.id, en.id, ws.id])
    phase_nr = Phase(id="N_RIGHT",    routes=left_turns + [nw.id])
    phase_sr = Phase(id="S_RIGHT",    routes=left_turns + [se.id])

    phase_set = [phase_all_red, phase_ns, phase_ew, phase_nr, phase_sr]

    intersection = Intersection(
        node=center.id,
        routes=all_routes,
        phase_set=phase_set,
        phase=phase_all_red.id,
    )

    return RoadNetwork(
        nodes=[center, north, south, east, west],
        roads=[nc, sc, ec, wc, cn, cs, ce, cw],
        intersections=[intersection],
    )


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class TrafficEnvironment(Environment):

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state  = State(episode_id=str(uuid4()), step_count=0)
        self._task   = "easy"
        self._rng    = np.random.default_rng(42)
        self.network: Optional[RoadNetwork] = None
        # fast lookup dicts — built on reset
        self._roads: Dict[str, Road] = {}
        self._routes: Dict[str, Route] = {}
        self._phases: Dict[str, Phase] = {}
        self._intersections: Dict[str, Intersection] = {}

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(self, task: str = "easy", seed: int = 42) -> TrafficObservation:
        self._task  = task
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._rng   = np.random.default_rng(seed)

        self.network = create_single_intersection_network(
            seed=seed,
            base_arrival_rate=0.5,
        )

        return TrafficObservation(
            done=False,
            reward=1.0,
            task='easy',
            step=0,
            road_network=self.network
        )

        self._task  = task
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._rng   = np.random.default_rng(seed)

        if task == "easy":
            self.network = create_single_intersection_network(
                seed=seed,
                base_arrival_rate=0.5,
            )
        elif task == "medium":
            # Asymmetric flow — north road much busier
            self.network = create_single_intersection_network(
                seed=seed,
                base_arrival_rate=0.3,
            )
            # Manually bump north road arrival rate
            for road in self.network.roads:
                if road.dst == self._get_center_id():
                    if self._get_node_direction(road.src) == "north":
                        road.base_arrival_rate = 1.5
        elif task == "hard":
            # Asymmetric + surge event mid-episode
            self.network = create_single_intersection_network(
                seed=seed,
                base_arrival_rate=0.4,
                surge_step=25,
                surge_multiplier=4.0,
            )

        self._build_lookup_dicts()
        return self._make_observation(reward=0.0, done=False)
    

    def step(self, action: TrafficAction) -> TrafficObservation:
        self._state.step_count += 1
        step = self._state.step_count

        # 1. Apply phase decisions
        self._apply_action(action)

        # 2. Drain waiting vehicles through open routes
        self._drain_queues()

        # 3. Move inflight vehicles into waiting
        self._inflight_to_waiting()

        # 4. Poisson arrivals + dropoffs on each road
        self._poisson_step(step)

        # 5. Compute reward
        reward = self._compute_reward()

        done = step >= MAX_STEPS
        return self._make_observation(reward=reward, done=done)

    @property
    def state(self) -> State:
        return self._state

    # ------------------------------------------------------------------
    # Sim steps
    # ------------------------------------------------------------------

    def _apply_action(self, action: TrafficAction):
        for decision in action.decisions:
            intersection = self._intersections.get(decision.intersection_id)
            if intersection is None:
                continue
            # Validate phase is in phase_set
            valid_ids = {p.id for p in intersection.phase_set}
            if decision.phase_id in valid_ids:
                intersection.phase = decision.phase_id

    def _drain_queues(self):
        """
        For each intersection, get open routes in current phase.
        Drain DISCHARGE_RATE vehicles from each inroad's waiting queue.
        Distribute drained vehicles uniformly across outroads.
        """
        for intersection in self.network.intersections:
            phase = self._phases.get(intersection.phase)
            if phase is None or not phase.routes:
                continue

            # Group open routes by inroad
            inroad_to_outroads: Dict[str, List[str]] = {}
            for route_id in phase.routes:
                route = self._routes.get(route_id)
                if route is None:
                    continue
                inroad_to_outroads.setdefault(route.inroad, []).append(route.outroad)

            for inroad_id, outroad_ids in inroad_to_outroads.items():
                inroad = self._roads.get(inroad_id)
                if inroad is None or inroad.waiting == 0:
                    continue

                drained = min(inroad.waiting, DISCHARGE_RATE)
                inroad.waiting -= drained

                # Distribute uniformly to outroads
                per_road = drained // len(outroad_ids)
                remainder = drained % len(outroad_ids)

                for i, outroad_id in enumerate(outroad_ids):
                    outroad = self._roads.get(outroad_id)
                    if outroad is not None:
                        extra = 1 if i < remainder else 0
                        outroad.inflight += per_road + extra

    def _inflight_to_waiting(self):
        """Move inflight vehicles into waiting on inbound roads."""
        for road in self.network.roads:
            if road.inflight > 0:
                # All inflight join waiting this step (no delay model yet)
                road.waiting += road.inflight
                road.inflight = 0

    def _poisson_step(self, step: int):
        """Poisson arrivals and random dropoffs on each road."""
        for road in self.network.roads:
            rate = road.base_arrival_rate

            # Surge event for hard task
            if hasattr(road, "surge_step") and step == road.surge_step:
                rate *= road.surge_multiplier

            # New arrivals
            arrivals = int(self._rng.poisson(rate))
            road.inflight += arrivals

            # Random dropoffs — some vehicles leave the road
            if road.waiting > 0:
                dropoff_rate = max(0.1, rate * 0.15)
                dropoffs = min(road.waiting, int(self._rng.poisson(dropoff_rate)))
                road.waiting = max(0, road.waiting - dropoffs)

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(self) -> float:
        """
        Three components, all normalized to [-1, 0]:

        1. Pressure     — sum of waiting vehicles across all inbound roads (negative)
        2. Starvation   — penalty if any road waiting > threshold
        3. Throughput   — bonus for roads that were drained this step (positive)
        """
        inbound_roads = [r for r in self.network.roads if self._is_inbound(r)]

        # 1. Pressure — negative, normalized
        total_waiting = sum(r.waiting for r in inbound_roads)
        pressure = -total_waiting / max(1, len(inbound_roads) * 20)

        # 2. Starvation — quadratic penalty for long queues
        STARVATION_THRESHOLD = 15
        starvation = 0.0
        for road in inbound_roads:
            if road.waiting > STARVATION_THRESHOLD:
                excess = road.waiting - STARVATION_THRESHOLD
                starvation -= (excess ** 2) / 500.0

        # 3. Throughput — reward roads that are clear
        clear_roads = sum(1 for r in inbound_roads if r.waiting == 0)
        throughput = clear_roads / max(1, len(inbound_roads)) * 0.2

        raw = pressure + starvation + throughput

        # Clamp to [-1, 1]
        return float(max(-1.0, min(1.0, raw)))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_lookup_dicts(self):
        self._roads         = {r.id: r   for r in self.network.roads}
        self._intersections = {i.id: i   for i in self.network.intersections}
        self._routes        = {}
        self._phases        = {}
        for intersection in self.network.intersections:
            for route in intersection.routes:
                self._routes[route.id] = route
            for phase in intersection.phase_set:
                self._phases[phase.id] = phase

    def _make_observation(self, reward: float, done: bool) -> TrafficObservation:
        return TrafficObservation(
            road_network=self.network,
            reward=reward,
            done=done,
            task=self._task,
            step=self._state.step_count,
        )

    def _is_inbound(self, road: Road) -> bool:
        """True if road leads into a junction node."""
        for node in self.network.nodes:
            if node.id == road.dst and node.type == NodeType.JUNCTION:
                return True
        return False

    def _get_center_id(self) -> str:
        for node in self.network.nodes:
            if node.type == NodeType.JUNCTION:
                return node.id
        return ""

    def _get_node_direction(self, node_id: str) -> str:
        """Best effort — returns endpoint position label if known."""
        # Not robust for general networks, fine for single intersection
        return ""