from pydantic import BaseModel
from typing import List


class CellObservation(BaseModel):
    curr: float


class RoadObservation(BaseModel):
    id: str
    cells: List[CellObservation]