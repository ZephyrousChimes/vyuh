from .road import Road, Cell
from ...models import IntersectionObservation

from dataclasses import dataclass
from itertools import combinations
from typing import List, Tuple, Set, NewType


@dataclass
class Route:
    RouteId = NewType('RouteId', str)

    id: RouteId
    inroad: Road
    outroad: Road


@dataclass
class Intersection:
    IntersectionId = NewType('IntersectionId', str)

    id: IntersectionId
    inroads: List[Road]
    outroads: List[Road]
    routes: List[Route]
    conflicts: Set[Tuple[Route.RouteId, Route.RouteId]]  

    current_phase: List[Route]
    time_in_phase: int = 0
    min_green_time: int = 7


    def set_phase(self, route_ids: List[str]) -> bool:
        """Validate and apply new phase. Returns False if invalid."""
        # enforce minimum green time
        if self.time_in_phase < self.min_green_time:
            return False

        # validate no conflicts
        for r1, r2 in combinations(route_ids, 2):
            if (r1, r2) in self.conflicts or (r2, r1) in self.conflicts:
                return False


        self.block_current_routes()
        self.allow_new_routes(route_ids)

        self.time_in_phase = 0
        return True


    def block_current_routes(self):
        # Just remove the phantom cells. Default is blocking.
        for route in self.current_phase:
            route.inroad.set_sink(None)
            route.outroad.set_source(None)
            

    def allow_new_routes(self, route_ids: List[str]):
        # Set the current phase using the route ids
        self.current_phase = [
            r for r in self.routes if r.id in route_ids
        ]

        # For each route, create a phantom cell and set the sink and source
        for route in self.current_phase:
            phantom = Cell(
                jam_cap=10.0,
                flow_cap=3.0,
                free_flow=1.0, 
                shock_speed=0.5,   
            )

            route.inroad.set_sink(phantom)
            route.outroad.set_source(phantom)
    

    def step(self):
        # step all roads — CTM handles everything
        for road in self.inroads:
            road.step()

        for road in self.outroads:
            road.step()

        self.time_in_phase += 1


    def observe(self) -> IntersectionObservation:
        return IntersectionObservation(
            id=str(self.id),
            inroads=[str(r.id) for r in self.inroads],
            outroads=[str(r.id) for r in self.outroads],
            routes=[(str(r.id), str(r.inroad.id), str(r.outroad.id))
                    for r in self.routes],
            conflicts=[(str(r1), str(r2)) for r1, r2 in self.conflicts],
            current_phase=[str(r.id) for r in self.current_phase]
        )
