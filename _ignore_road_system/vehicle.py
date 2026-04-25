from .road import RoadId

from pydantic import BaseModel
from typing import NewType 


VehicleId = NewType('VehicleId', str)


class Vehicle(BaseModel):
    id: VehicleId
    road: RoadId
    cell_idx: int
    cell_tick: float # contains ticks for which the vehicle has been in the current cell
