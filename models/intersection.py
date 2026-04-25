from pydantic import BaseModel
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