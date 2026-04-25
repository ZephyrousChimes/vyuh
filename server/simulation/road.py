from dataclasses import dataclass, field
from typing import List, Optional, NewType

from models import RoadObservation, CellObservation


@dataclass
class Cell:
    jam_cap: float
    flow_cap: float
    free_flow: float
    shock_speed: float
    curr: float = 0.0

    def demand(self) -> float:
        return min(self.flow_cap, self.curr)

    def supply(self) -> float:
        return min(
            self.flow_cap,
            (self.shock_speed / self.free_flow) * (self.jam_cap - self.curr)
        )

    def update(self, inflow: float, outflow: float):
        self.curr += inflow - outflow
        self.curr = max(0.0, min(self.jam_cap, self.curr))


    def observe(self) -> CellObservation:
        return CellObservation(curr=self.curr)


@dataclass
class PhantomSource(Cell):
    """
    Default phantom source cell. No cap on jam. No cap on flow. The traffic generator will manually add vehicles.

    It simulates a parking space.
    """
    jam_cap: float = field(default=float('inf'), init=False)
    flow_cap: float = field(default=float('inf'), init=False)
    free_flow: float = field(default=1.0, init=False)
    shock_speed: float = field(default=1.0, init=False)

    def demand(self) -> float:
        return self.curr  # everything in source wants to move
 
    def supply(self) -> float:
        return float('inf')
 
    def update(self, inflow: float, outflow: float):
        self.curr -= outflow  # only drain, never receive
        self.curr = max(0.0, self.curr)


@dataclass
class PhantomSink(Cell):
    """
    Default phantom sink cell. 
    No cap on jam. 
    Full cap on flow. 
    """
    jam_cap: float = field(default=float('inf'), init=False)
    flow_cap: float = field(default=0.0, init=False)
    free_flow: float = field(default=1.0, init=False)
    shock_speed: float = field(default=1.0, init=False)

    def update(self, inflow: float, outflow: float):
        pass


@dataclass
class Road:
    RoadId = NewType('RoadId', str)
 
    id: str
    cells: List[Cell]
    source: Cell = field(default_factory=PhantomSource)
    sink: Cell   = field(default_factory=PhantomSink)
 
    def set_source(self, cell: Optional[Cell] = None):
        self.source = cell if cell is not None else PhantomSource()
 
    def set_sink(self, cell: Optional[Cell] = None):
        self.sink = cell if cell is not None else PhantomSink()
 
    def step(self) -> None:
        all_cells = [self.source] + self.cells + [self.sink]
        n = len(all_cells)
        inflows = [0.0] * n
 
        for i in range(1, n):
            inflows[i] = min(all_cells[i-1].demand(), all_cells[i].supply())
 
        for i in range(n):
            inflow  = inflows[i]
            outflow = inflows[i+1] if i+1 < n else 0.0
            all_cells[i].update(inflow, outflow)
 
    def stop_line_demand(self) -> float:
        return self.cells[-1].demand()
 
    def total_vehicles(self) -> float:
        return sum(c.curr for c in self.cells)

    def observe(self) -> RoadObservation:
        return RoadObservation(
            id=str(self.id),
            cells=[cell.observe() for cell in self.cells]
        )