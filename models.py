from pydantic import BaseModel, Field
from typing import List, Tuple, NewType

class CellObservation(BaseModel):
    curr: float


class RoadObservation(BaseModel):
    id: str
    cells: List[CellObservation]


IntersectionId = NewType('IntersectionId', str)
RouteId = NewType('RouteId', str)


class IntersectionObservation(BaseModel):

    id: str

    inroads: List[str]
    outroads: List[str]
    routes: List[Tuple[str, str, str]]

    conflicts: List[Tuple[str, str]]

    current_phase: List[str]
    time_in_phase: float = Field(default=0.0)
    min_green_time: float = Field(default=0.0)


# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Traffic Env Environment.

The traffic_env environment is a simple test environment that echoes back messages.
"""

from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, Field
from typing import List, Tuple, Optional


class TrafficAction(Action):
    """
    Action for the Traffic Env environment 
    List of routes for each intersection.
    """

    phase_routes: List[List[str]]


class PriorityVehicleObservation(BaseModel):
    approach: str       # which inroad
    cells_to_stop: int  # how far from stop line


class TrafficObservation(Observation):
    roads: List[RoadObservation]
    intersections: List[IntersectionObservation]
    timestep: int                          # for Little's Law W = L/λ
    arrival_rates: List[float]             # per inroad, for Little's Law
    priority_vehicle: Optional[PriorityVehicleObservation] = None
