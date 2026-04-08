# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Traffic Env Environment.

The traffic_env environment is a simple test environment that echoes back messages.
"""

from uuid import uuid4
from openenv.core.env_server.types import Action, Observation
from pydantic import Field, BaseModel, model_validator
from enum import Enum
from typing import List, Tuple, Optional



class NodeType(str, Enum):
    ENDPOINT = "endpoint"
    JUNCTION = "junction"


# class Road(BaseModel):
#     """
#     Two opposing directions of roads, just by each other, are considered different roads. Moving from one to the other is considered a road change.
#     """
#     id: str
#     src: str
#     dst: str
#     wait_list: List[int]
    

# class Connection(BaseModel):
#     id: str = Field(default=str(uuid4()))
#     inroad: Road
#     outroad: Road


# class Phase(BaseModel):
#     id: str = Field(default=str(uuid4()))
#     connections: List[Connection] = Field(default=[])



    

# class Node(BaseModel):
#     id: str = Field(default=str(uuid4()))
#     type: NodeType
#     inroads: List[Road] = Field(default=[])
#     outroads: List[Road] = Field(default=[])


# class Intersection(BaseModel):
#     id: str = Field(default=str(uuid4()))
#     # inroads: List[Road] = Field(default=[])
#     # outroads: List[Road] = Field(default=[])
#     node: Node
#     connections: List[Connection] = Field(default=[])
#     valid_phases: List[Phase] = Field(default=[])
#     current_phase: Phase = Field(default=Phase())


# class RoadNetwork(BaseModel):
#     nodes: List[Node]


# Control Plane
class Node(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    type: NodeType


class Vehicle(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    speed: int
    priority: int = Field(default=0)


class Road(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    src: str
    dst: str
    inflight: int = Field(default=0)
    waiting: int = Field(default=0)
    base_arrival_rate: float = Field(default=0.5)
    surge_multiplier: float = Field(default=1.0)
    surge_step: Optional[int] = Field(default=None)



# Data Plane
class Route(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    inroad: str
    outroad: str


class Phase(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    routes: List[str]


class Intersection(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    node: str
    routes: List[Route]
    phase_set: List[Phase]
    phase: str

# Observation Package

class RoadNetwork(BaseModel):
    nodes: List[Node]
    roads: List[Road]
    intersections: List[Intersection]


class IntersectionPhaseDecision(BaseModel):
    intersection_id: str
    phase_id: str

class TrafficAction(Action):
    decisions: List[IntersectionPhaseDecision]



class TrafficObservation(Observation):
    """Observation from the Traffic Env environment - the entire road network state"""
    task: str = Field(default='easy')
    step: int = Field(default=0)
    road_network: RoadNetwork
