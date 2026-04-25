from dataclasses import dataclass, field
from typing import List, Tuple, Optional, NewType, override

from models.road import RoadObservation

from ...models import CellObservation


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

    @override
    def update(self, inflow: float, outflow: float):
        pass


@dataclass
class Road:
    RoadId = NewType('RoadId', str)

    id: RoadId
    cells: List[Cell] 

    source: Cell = field(default_factory=PhantomSource)
    sink: Cell = field(default_factory=PhantomSink)


    def set_source(self, cell: Optional[Cell] = None):
        if cell is None:
            cell = PhantomSource()
        self.source = cell


    def set_sink(self, cell: Optional[Cell] = None):
        if cell is None:
            cell = PhantomSink()
        self.sink = cell


    def step(self) -> None:
        """Pure CTM across all cells including phantoms."""
        all_cells = [self.source] + self.cells + [self.sink]
        n = len(all_cells)
        inflows = [0.0] * n  # inflows[i] = inflow from all_cells[i-1] to all_cells[i]

        for i in range(1, n):
            inflows[i] = min(
                all_cells[i - 1].demand(),
                all_cells[i].supply()
            )

        for i in range(n):
            inflow = inflows[i] if i < n else 0.0
            outflow = inflows[i + 1] if i + 1 < n else 0.0
            all_cells[i].update(inflow, outflow)


    def observe(self) -> RoadObservation:
        return RoadObservation(
            id=str(self.id),
            cells=[cell.observe() for cell in self.cells]
        )