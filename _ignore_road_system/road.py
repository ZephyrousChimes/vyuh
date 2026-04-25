from pydantic import BaseModel
from typing import List, NewType


RoadId = NewType('RoadId', str)


class Cell(BaseModel):
    sending_fn: float 
    receiving_fn: float
    flow: float
    vehicle_count: int


class Road(BaseModel):
    id: RoadId
    cells: List[Cell]
    num_lanes: int
    free_flow_speed: float
    backward_wave_speed: float
    jam_density: float
    capacity: float