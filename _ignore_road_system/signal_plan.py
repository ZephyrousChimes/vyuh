from .road import RoadId

from pydantic import BaseModel
from typing import List, Tuple, NewType


SignalId = NewType('SignalId', str)
RouteId = NewType('RouteId', str)


class Route(BaseModel):
    id: RouteId
    in_road_id: RoadId
    out_road_id: RoadId


class SignalPlan(BaseModel):
    id: SignalId
    in_roads: List[RoadId]      # possible roads for incoming traffic
    out_roads: List[RoadId]     # possible roads for outgoing traffic
    routes: List[RouteId]
    conflicts: List[Tuple[RouteId, RouteId]]
    curr_routes: List[RouteId]
    
