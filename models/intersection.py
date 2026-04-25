from pydantic import BaseModel, Field
from typing import List, Tuple, NewType


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