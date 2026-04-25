# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Traffic Signal Control Environment — OpenEnv Implementation.

A 4-way intersection modeled using the Cell Transmission Model (Daganzo 1994).
Vehicles arrive via Poisson process. Agent controls signal phases.
Three tasks of increasing difficulty: uniform flow, asymmetric flow,
and priority vehicle preemption.
"""

import math
import random
from uuid import uuid4
from typing import Optional, Any, List, Dict

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import TrafficAction, TrafficObservation, PriorityVehicleObservation
    from ..models import RoadObservation, CellObservation
    from ..models import IntersectionObservation
    from .simulation import Cell, Road, Intersection, Route
except ImportError:
    from models import TrafficAction, TrafficObservation, PriorityVehicleObservation
    from models import RoadObservation, CellObservation
    from models import IntersectionObservation
    from server.simulation import Cell, Road, Intersection, Route


# ─────────────────────────────────────────────────────────────────────────────
# TASK CONFIGS
# ─────────────────────────────────────────────────────────────────────────────

TASK_CONFIGS = {
    "easy": {
        "arrival_rates": {
            "north_in": 1.5,
            "south_in": 1.5,
            "east_in":  1.5,
            "west_in":  1.5,
        },
        "episode_length": 150,
        "priority_vehicle": None,
        "description": "Uniform Poisson flow. Learn basic phase rotation.",
    },
    "medium": {
        "arrival_rates": {
            "north_in": 3.0,
            "south_in": 3.0,
            "east_in":  0.5,
            "west_in":  0.5,
        },
        "episode_length": 150,
        "priority_vehicle": None,
        "description": "Asymmetric flow. N/S dominant. Minimize max wait time.",
    },
    "hard": {
        "arrival_rates": {
            "north_in": 2.0,
            "south_in": 2.0,
            "east_in":  2.0,
            "west_in":  2.0,
        },
        "episode_length": 200,
        "priority_vehicle": {
            "spawn_tick": 80,
            "approach":   "north_in",
        },
        "description": "Priority vehicle mid-episode. Clear path immediately.",
    },
}

REWARD_BASELINES = {"easy": -30.0, "medium": -40.0, "hard": -60.0}
REWARD_OPTIMAL   = 0.0

ALPHA_PRESSURE       = 1.0
BETA_FAIRNESS        = 0.5
GAMMA_WAIT           = 0.8
PRIORITY_WEIGHT      = 10.0
CLEARANCE_BONUS      = 50.0
STARVATION_THRESHOLD = 10.0


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def poisson_sample(lam: float) -> float:
    if lam <= 0:
        return 0.0
    L = math.exp(-lam)
    k, p = 0, 1.0
    while True:
        k += 1
        p *= random.random()
        if p <= L:
            return float(k - 1)


def jains_fairness(queues: List[float]) -> float:
    if not queues or all(q == 0 for q in queues):
        return 1.0
    n = len(queues)
    return (sum(queues) ** 2) / (n * sum(q ** 2 for q in queues) + 1e-9)


def make_road(road_id: str) -> Road:
    return Road(
        id=Road.RoadId(road_id),
        cells=[
            Cell(jam_cap=10.0, flow_cap=3.0, free_flow=1.0, shock_speed=0.5)
            for _ in range(8)
        ]
    )


# ─────────────────────────────────────────────────────────────────────────────
# ENVIRONMENT
# ─────────────────────────────────────────────────────────────────────────────

class TrafficEnvironment(Environment):
    """
    4-way intersection traffic signal control environment.

    Observation: per-cell densities on all approaches, current phase,
                 arrival rates, optional priority vehicle signal.

    Action: phase_routes — list of route id groups to activate.

    Reward:
        easy   — minimize pressure + maximize Jain's fairness
        medium — above + minimize Little's Law wait times
        hard   — above + heavy penalty for priority vehicle delay
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state             = State(episode_id=str(uuid4()), step_count=0)
        self._task              = "easy"
        self._intersection      = None
        self._all_roads         = {}
        self._arrival_rates     = {}
        self._episode_length    = 150
        self._priority_config   = None
        self._priority_active   = False
        self._priority_cleared  = False
        self._cumulative_reward = 0.0 
        self._reset_count       = 0

    # ── OpenEnv interface ────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = 42,
        episode_id: Optional[str] = None,
        task: str = "easy",
        **kwargs: Any,
    ) -> TrafficObservation:

        # Seed is set to 42 be default
        if seed is not None:
            random.seed(seed)

        # 
        self._task   = task if task in TASK_CONFIGS else "easy"
        config       = TASK_CONFIGS[self._task]

        self._state             = State(episode_id=episode_id or str(uuid4()), step_count=0)
        self._reset_count      += 1
        self._cumulative_reward = 0.0
        self._priority_active   = False
        self._priority_cleared  = False
        self._priority_config   = config["priority_vehicle"]
        self._arrival_rates     = config["arrival_rates"].copy()
        self._episode_length    = config["episode_length"]

        self._build_intersection()
        return self._observe(reward=0.0, done=False)


    def step(
        self,
        action: TrafficAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> TrafficObservation:
        
        if self._intersection is None:
            raise ValueError('Tried to step with None as self._intersection')
        
        self._state.step_count += 1
        tick = self._state.step_count

        # spawn priority vehicle
        if (
            self._priority_config
            and not self._priority_active
            and not self._priority_cleared
            and tick >= self._priority_config["spawn_tick"]
        ):
            self._priority_active = True
            road = self._all_roads[self._priority_config["approach"]]
            road.source.curr += 5.0

        default_action = TrafficAction(
            phase_routes = [
                [
                    'north_in_east_out', 'east_in_south_out',
                    'south_in_west_out', 'west_in_north_out',
                ]
            ]
        )

        # apply phase action
        if action.phase_routes:
            # route_ids = [r for group in action.phase_routes for r, *_ in group]
            route_ids = action.phase_routes[0]
            # TODO: Change the route_ids above... can take only one intersection
            self._intersection.set_phase(route_ids)

        # Poisson arrivals
        for road_id, rate in self._arrival_rates.items():
            road = self._all_roads.get(road_id)
            if road:
                road.source.curr += poisson_sample(rate)

        # step CTM
        self._intersection.step()

        # check priority clearance
        clearance_bonus = 0.0
        if self._priority_active and not self._priority_cleared:
            if self._priority_config is None:
                raise ValueError('[OBSERVE] self._priority_active is True but self._priority_config is None')
            
            road = self._all_roads[self._priority_config["approach"]]
            if road.cells[-1].curr < 1.0:
                self._priority_cleared = True
                self._priority_active  = False
                clearance_bonus        = CLEARANCE_BONUS

        reward = self._compute_reward(clearance_bonus)
        self._cumulative_reward += reward

        done = tick >= self._episode_length
        return self._observe(reward=reward, done=done)

    @property
    def state(self) -> State:
        return self._state

    # ── Sim construction ─────────────────────────────────────────────────────

    def _build_intersection(self):
        in_roads  = [make_road(id) for id in ('north_in','south_in','east_in','west_in')]
        out_roads = [make_road(id) for id in ('north_out','south_out','east_out','west_out')]

        self._all_roads = {r.id: r for r in in_roads + out_roads}

        routes = [
            Route(Route.RouteId('north_in_south_out'), in_roads[0], out_roads[1]),
            Route(Route.RouteId('north_in_east_out'),  in_roads[0], out_roads[2]),
            Route(Route.RouteId('north_in_west_out'),  in_roads[0], out_roads[3]),
            Route(Route.RouteId('south_in_north_out'), in_roads[1], out_roads[0]),
            Route(Route.RouteId('south_in_east_out'),  in_roads[1], out_roads[2]),
            Route(Route.RouteId('south_in_west_out'),  in_roads[1], out_roads[3]),
            Route(Route.RouteId('east_in_north_out'),  in_roads[2], out_roads[0]),
            Route(Route.RouteId('east_in_south_out'),  in_roads[2], out_roads[1]),
            Route(Route.RouteId('east_in_west_out'),   in_roads[2], out_roads[3]),
            Route(Route.RouteId('west_in_north_out'),  in_roads[3], out_roads[0]),
            Route(Route.RouteId('west_in_south_out'),  in_roads[3], out_roads[1]),
            Route(Route.RouteId('west_in_east_out'),   in_roads[3], out_roads[2]),
        ]

        conflicts = set([
            (str(routes[i].id), str(routes[j].id))
            for i, j in [
                (0, 3),  (0, 4),  (0, 6),  (0, 8),  (0, 10), (0, 11),
                (2, 3),  (2, 4),  (2, 6),  (2, 8),  (2, 10), (2, 11),
                (3, 0),  (3, 2),  (3, 6),  (3, 8),  (3, 10), (3, 11),
                (4, 0),  (4, 2),  (4, 6),  (4, 8),  (4, 10), (4, 11),
                (6, 0),  (6, 2),  (6, 3),  (6, 4),  (6, 10), (6, 11),
                (8, 0),  (8, 2),  (8, 3),  (8, 4),  (8, 10), (8, 11),
                (10, 0), (10, 2), (10, 3), (10, 4), (10, 6), (10, 8),
                (11, 0), (11, 2), (11, 3), (11, 4), (11, 6), (11, 8),
            ]
        ])

        self._intersection = Intersection(
            Intersection.IntersectionId('intersection'),
            inroads=in_roads,
            outroads=out_roads,
            routes=routes,
            conflicts=conflicts,
            current_phase=[routes[0], routes[3]],
        )
        self._intersection.time_in_phase = self._intersection.min_green_time

    # ── Reward ───────────────────────────────────────────────────────────────

    def _compute_reward(self, clearance_bonus: float = 0.0) -> float:
        if self._intersection is None:
            raise ValueError('Tried to compute reward with None as intersection')
        
        inroads       = self._intersection.inroads
        arrival_rates = list(self._arrival_rates.values())

        queues = [r.cells[-1].curr for r in inroads]
        waits  = [q / max(arrival_rates[i], 0.1) for i, q in enumerate(queues)]

        pressure   = sum(queues)
        fairness   = jains_fairness(queues)
        starvation = sum(math.exp(max(0.0, w - STARVATION_THRESHOLD)) for w in waits)

        if self._task == "easy":
            reward = -ALPHA_PRESSURE * pressure + BETA_FAIRNESS * fairness

        elif self._task == "medium":
            mean_wait = sum(waits) / len(waits)
            max_wait  = max(waits)
            reward = (
                -ALPHA_PRESSURE * pressure
                + BETA_FAIRNESS  * fairness
                - GAMMA_WAIT     * mean_wait
                - GAMMA_WAIT     * max_wait
                - 0.1            * starvation
            )

        else:  # hard
            mean_wait = sum(waits) / len(waits)
            max_wait  = max(waits)
            base = (
                -ALPHA_PRESSURE * pressure
                + BETA_FAIRNESS  * fairness
                - GAMMA_WAIT     * mean_wait
                - GAMMA_WAIT     * max_wait
                - 0.1            * starvation
            )
            if self._priority_active:
                if self._priority_config is None:
                    raise ValueError('[OBSERVE] self._priority_active is True but self._priority_config is None')
                
                approach = self._priority_config["approach"]
                road     = self._all_roads[approach]
                pv_wait  = road.cells[-1].curr / max(self._arrival_rates.get(approach, 1.0), 0.1)
                gamma    = 0.7
                reward   = (1 - gamma) * base - gamma * PRIORITY_WEIGHT * pv_wait
            else:
                reward = base

            reward += clearance_bonus

        return float(reward)

    # ── Observation ──────────────────────────────────────────────────────────

    def _observe(self, reward: float, done: bool) -> TrafficObservation:
        if self._intersection is None:
            raise ValueError('Tried to observe with None as self._intersecton')        

        road_obs = [
            RoadObservation(
                id=road.id,
                cells=[CellObservation(curr=c.curr) for c in road.cells]
            )
            for road in self._intersection.inroads + self._intersection.outroads
        ]

        ix = self._intersection
        
        intersection_obs = IntersectionObservation(
            id=str(ix.id),
            inroads=[r.id for r in ix.inroads],
            outroads=[r.id for r in ix.outroads],
            routes=[(r.id, r.inroad.id, r.outroad.id) for r in ix.routes],
            conflicts=list(ix.conflicts),
            current_phase=[r.id for r in ix.current_phase],
            time_in_phase=ix.time_in_phase,
            min_green_time=ix.min_green_time,
        )

        pv_obs = None
        if self._priority_active:
            if self._priority_config is None:
                raise ValueError('[OBSERVE] self._priority_active is True but self._priority_config is None')
            
            approach      = self._priority_config["approach"]
            road          = self._all_roads[approach]
            cells_to_stop = next(
                (i for i, c in enumerate(reversed(road.cells)) if c.curr > 0.5),
                0
            )
            pv_obs = PriorityVehicleObservation(
                approach=approach,
                cells_to_stop=cells_to_stop,
            )

        arrival_rates = [
            self._arrival_rates.get(r.id, 0.0)
            for r in ix.inroads
        ]

        baseline = REWARD_BASELINES[self._task]
        score    = max(0.0, min(1.0,
            (self._cumulative_reward - baseline * self._state.step_count)
            / (abs(baseline) * self._episode_length + 1e-9)
        ))

        return TrafficObservation(
            roads=road_obs,
            intersections=[intersection_obs],
            timestep=self._state.step_count,
            arrival_rates=arrival_rates,
            priority_vehicle=pv_obs,
            done=done,
            reward=reward,
            metadata={
                "task":              self._task,
                "cumulative_reward": self._cumulative_reward,
                "score":             score,
                "priority_active":   self._priority_active,
                "priority_cleared":  self._priority_cleared,
            }
        )