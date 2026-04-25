from .road import Road, Cell
from models import IntersectionObservation

from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Tuple, Set, NewType


@dataclass
class Route:
    RouteId = NewType('RouteId', str)

    id: RouteId
    inroad: Road
    outroad: Road


@dataclass
class Intersection:
    IntersectionId = NewType('IntersectionId', str)
 
    id: str
    inroads: List[Road]
    outroads: List[Road]
    routes: List[Route]
    conflicts: Set[Tuple[str, str]]
 
    current_phase: List[Route] = field(default_factory=list)
    time_in_phase: int = 0
    min_green_time: int = 7
 
    # routing_table[inroad_id] = list of outroad_ids currently active
    # built fresh each phase change
    routing_table: Dict[str, List[Road]] = field(default_factory=dict)
 
    def set_phase(self, route_ids: List[str]) -> bool:
        if self.time_in_phase < self.min_green_time:
            return False
 
        for r1, r2 in combinations(route_ids, 2):
            if (r1, r2) in self.conflicts or (r2, r1) in self.conflicts:
                return False
 
        self.current_phase = [r for r in self.routes if r.id in route_ids]
        self.time_in_phase = 0
 
        # rebuild routing table — inroad → list of outroads
        self.routing_table = {}
        for route in self.current_phase:
            inid = route.inroad.id
            if inid not in self.routing_table:
                self.routing_table[inid] = []
            self.routing_table[inid].append(route.outroad)
 
        return True
 
    def step(self):
        # For each inroad with active routes —
        # read stop line demand, split evenly across outroads, inject
        for inroad_id, outroads in self.routing_table.items():
            inroad = next(r for r in self.inroads if r.id == inroad_id)
 
            available = inroad.stop_line_demand()
            if available <= 0 or not outroads:
                continue
 
            # clean split — equal fraction to each active outroad
            per_out = available / len(outroads)
 
            total_sent = 0.0
            for outroad in outroads:
                actual = min(per_out, outroad.cells[0].supply())
                outroad.source.curr += actual  # inject directly into source
                total_sent += actual
 
            # drain from inroad stop line
            inroad.cells[-1].curr -= total_sent
            inroad.cells[-1].curr = max(0.0, inroad.cells[-1].curr)
 
        # step all roads — CTM drains source naturally
        for road in self.inroads + self.outroads:
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
