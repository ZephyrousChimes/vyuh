"""
intersection_v2.py — CTM Intersection with routing dict + terminal viz
Run: python intersection_v2.py
"""

import time
import os
import random
import math
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, NewType
from itertools import combinations


# ─────────────────────────────────────────────────────────────────────────────
# CTM CORE
# ─────────────────────────────────────────────────────────────────────────────

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


@dataclass
class PhantomSource(Cell):
    jam_cap: float   = field(default=float('inf'), init=False)
    flow_cap: float  = field(default=float('inf'), init=False)
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
    jam_cap: float   = field(default=float('inf'), init=False)
    flow_cap: float  = field(default=0.0, init=False)
    free_flow: float = field(default=1.0, init=False)
    shock_speed: float = field(default=1.0, init=False)

    def update(self, inflow: float, outflow: float):
        pass  # sink absorbs, state irrelevant


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


# ─────────────────────────────────────────────────────────────────────────────
# ROUTE + INTERSECTION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Route:
    RouteId = NewType('RouteId', str)
    id: str
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


# ─────────────────────────────────────────────────────────────────────────────
# FACTORY — standard 4-way intersection
# ─────────────────────────────────────────────────────────────────────────────

CELL_PARAMS = dict(jam_cap=10.0, flow_cap=3.0, free_flow=1.0, shock_speed=0.5)
NUM_CELLS   = 8

def make_road(road_id: str) -> Road:
    return Road(
        id=road_id,
        cells=[Cell(**CELL_PARAMS) for _ in range(NUM_CELLS)]
    )

def make_four_way() -> Intersection:
    # inroads — vehicles flowing toward intersection
    n_in = make_road('north_in')
    s_in = make_road('south_in')
    e_in = make_road('east_in')
    w_in = make_road('west_in')

    # outroads — vehicles flowing away from intersection
    n_out = make_road('north_out')
    s_out = make_road('south_out')
    e_out = make_road('east_out')
    w_out = make_road('west_out')

    # all 12 movements (no U-turns)
    routes = [
        Route('N_S', n_in, s_out),
        Route('N_E', n_in, e_out),
        Route('N_W', n_in, w_out),

        Route('S_N', s_in, n_out),
        Route('S_E', s_in, e_out),
        Route('S_W', s_in, w_out),

        Route('E_N', e_in, n_out),
        Route('E_S', e_in, s_out),
        Route('E_W', e_in, w_out),

        Route('W_N', w_in, n_out),
        Route('W_S', w_in, s_out),
        Route('W_E', w_in, e_out),
    ]

    # NEMA-inspired conflict pairs
    # opposing straights don't conflict, crossing movements do
    conflicts = {
        # N↔S straights vs E↔W straights
        ('N_S', 'E_W'), ('N_S', 'W_E'),
        ('S_N', 'E_W'), ('S_N', 'W_E'),
        # left turns conflict with opposing straight + crossing
        ('N_W', 'S_N'), ('N_W', 'E_W'), ('N_W', 'W_E'), ('N_W', 'E_S'),
        ('S_E', 'N_S'), ('S_E', 'E_W'), ('S_E', 'W_E'), ('S_E', 'W_N'),
        ('E_N', 'W_E'), ('E_N', 'N_S'), ('E_N', 'S_N'), ('E_N', 'N_W'),
        ('W_S', 'E_W'), ('W_S', 'N_S'), ('W_S', 'S_N'), ('W_S', 'S_E'),
        # crossing movements
        ('N_E', 'S_W'), ('N_E', 'W_E'), ('N_E', 'E_W'),
        ('S_W', 'N_E'), ('S_W', 'E_W'), ('S_W', 'W_E'),
    }

    intersection = Intersection(
        id='main',
        inroads=[n_in, s_in, e_in, w_in],
        outroads=[n_out, s_out, e_out, w_out],
        routes=routes,
        conflicts=conflicts,
    )

    # start with N↔S straight phase
    intersection.time_in_phase = 7  # allow immediate set
    intersection.set_phase(['N_S', 'S_N'])

    return intersection


# ─────────────────────────────────────────────────────────────────────────────
# TRAFFIC GENERATOR — Poisson arrivals
# ─────────────────────────────────────────────────────────────────────────────

def poisson_sample(lam: float) -> float:
    if lam <= 0:
        return 0.0
    # Knuth
    L = math.exp(-lam)
    k, p = 0, 1.0
    while True:
        k += 1
        p *= random.random()
        if p <= L:
            return float(k - 1)

def generate_arrivals(intersection: Intersection, rates: Dict[str, float]):
    for road in intersection.inroads:
        base = road.id.replace('_in', '')
        rate = rates.get(base, 0.0)
        road.source.curr += poisson_sample(rate)


# ─────────────────────────────────────────────────────────────────────────────
# TERMINAL VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"

def density_color(curr: float, jam_cap: float) -> str:
    if jam_cap == float('inf') or jam_cap == 0:
        return "\033[90m"
    r = curr / jam_cap
    if r == 0:    return "\033[90m"
    if r < 0.25:  return "\033[92m"
    if r < 0.5:   return "\033[93m"
    if r < 0.75:  return "\033[33m"
    return "\033[91m"

def cell_char(curr: float, jam_cap: float) -> str:
    col = density_color(curr, jam_cap)
    if jam_cap == float('inf') or curr == 0:
        chars = '░'
    else:
        r = curr / jam_cap
        chars = ['░','▒','▓','█'][min(3, int(r * 4))]
    return col + chars + RESET

def road_bar(road: Road, reverse: bool = False) -> str:
    cells = road.cells if not reverse else list(reversed(road.cells))
    return ''.join(cell_char(c.curr, c.jam_cap) for c in cells)

def render(intersection: Intersection, tick: int, 
           rates: Dict[str, float], phase_name: str,
           total_discharged: float):

    os.system('clear')

    n_in  = next(r for r in intersection.inroads  if 'north' in r.id)
    s_in  = next(r for r in intersection.inroads  if 'south' in r.id)
    e_in  = next(r for r in intersection.inroads  if 'east'  in r.id)
    w_in  = next(r for r in intersection.inroads  if 'west'  in r.id)
    n_out = next(r for r in intersection.outroads if 'north' in r.id)
    s_out = next(r for r in intersection.outroads if 'south' in r.id)

    W = NUM_CELLS

    # header
    print(f"\n  {BOLD}CTM FOUR-WAY INTERSECTION{RESET}  "
          f"tick={BOLD}{tick:>4}{RESET}  "
          f"phase={BOLD}{phase_name}{RESET}  "
          f"phase_tick={intersection.time_in_phase:>2}/"
          f"{intersection.min_green_time}  "
          f"discharged={total_discharged:.0f}")
    print()

    # pressure
    pressure = sum(r.stop_line_demand() for r in intersection.inroads)
    total_veh = sum(r.total_vehicles() for r in intersection.inroads)
    print(f"  pressure={pressure:.1f}  queued={total_veh:.1f}")
    print()

    pad = ' ' * (W + 4)

    # NORTH in (flows down, cell 0 at top, stop line at bottom)
    print(f"  {pad}  \033[94mNORTH IN  λ={rates.get('north',0):.1f}{RESET}")
    north_cells = list(reversed(n_in.cells))  # cell 0 at top
    for i, c in enumerate(north_cells):
        marker = '▼' if i == len(north_cells)-1 else ' '
        print(f"  {pad}  {marker}{cell_char(c.curr, c.jam_cap)}{RESET}  "
              f"\033[90m{c.curr:4.1f}{RESET}")
    print()

    # middle row: WEST | intersection box | EAST
    # west flows right (cell 0 at left, stop at right)
    # east flows left (stop line at left, cell 0 at right)

    box_lines = [
        f"┌{'─'*8}┐",
        f"│ {phase_name:<6} │",
        f"│ t={intersection.time_in_phase:<4} │",
        f"│        │",
        f"└{'─'*8}┘",
    ]

    west_cells = w_in.cells  # cell 0 at left
    east_cells = list(reversed(e_in.cells))  # stop line at left

    print(f"  \033[92mWEST IN  λ={rates.get('west',0):.1f}{RESET}")
    for i in range(max(len(west_cells), len(box_lines), len(east_cells))):
        # west cell
        if i < len(west_cells):
            wc = west_cells[i]
            w_str = f"{cell_char(wc.curr, wc.jam_cap)}\033[90m{wc.curr:3.1f}{RESET}"
        else:
            w_str = '    '

        # box
        box = box_lines[i] if i < len(box_lines) else ' ' * 10

        # east cell
        if i < len(east_cells):
            ec = east_cells[i]
            e_str = f"{cell_char(ec.curr, ec.jam_cap)}\033[90m{ec.curr:3.1f}{RESET}"
        else:
            e_str = '    '

        print(f"  {w_str} {box} {e_str}")

    print(f"  \033[33mEAST IN  λ={rates.get('east',0):.1f}{RESET}")
    print()

    # SOUTH in (flows up, stop line at top, cell 0 at bottom)
    print(f"  {pad}  \033[95mSOUTH IN  λ={rates.get('south',0):.1f}{RESET}")
    south_cells = s_in.cells  # stop line at index 0 visually (top)
    for i, c in enumerate(south_cells):
        marker = '▲' if i == 0 else ' '
        print(f"  {pad}  {marker}{cell_char(c.curr, c.jam_cap)}{RESET}  "
              f"\033[90m{c.curr:4.1f}{RESET}")
    print()

    # legend
    print(f"  \033[92m█{RESET} free  \033[93m█{RESET} mod  \033[33m█{RESET} heavy  "
          f"\033[91m█{RESET} jam  \033[90m░{RESET} empty")
    print()

    # phase info
    active = [r.id for r in intersection.current_phase]
    print(f"  active routes: {BOLD}{', '.join(active)}{RESET}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIOS
# ─────────────────────────────────────────────────────────────────────────────

PHASE_CYCLE = [
    (['N_S', 'S_N'],         'P1:NS'),
    (['E_W', 'W_E'],         'P2:EW'),
    (['N_W', 'S_E'],         'P3:LT'),
    (['E_N', 'W_S'],         'P4:LT'),
]

def run_scenario(name: str, rates: Dict[str, float],
                 steps: int = 60, delay: float = 0.15,
                 priority_tick: Optional[int] = None):

    intersection = make_four_way()
    tick = 0
    phase_idx = 0
    total_discharged = 0.0

    print(f"\n  Starting scenario: {BOLD}{name}{RESET}")
    time.sleep(0.8)

    for tick in range(steps):
        # generate arrivals
        generate_arrivals(intersection, rates)

        # auto cycle phases every min_green_time ticks
        if intersection.time_in_phase >= intersection.min_green_time:
            phase_idx = (phase_idx + 1) % len(PHASE_CYCLE)
            route_ids, _ = PHASE_CYCLE[phase_idx]
            intersection.set_phase(route_ids)

        # priority vehicle — burst on north at specified tick
        if priority_tick and tick == priority_tick:
            intersection.inroads[0].cells[0].curr = min(
                intersection.inroads[0].cells[0].curr + 8,
                intersection.inroads[0].cells[0].jam_cap
            )
            # force phase to clear north
            intersection.time_in_phase = intersection.min_green_time
            intersection.set_phase(['N_S', 'S_N'])
            phase_idx = 0

        before = sum(r.total_vehicles() for r in intersection.inroads)
        intersection.step()
        after  = sum(r.total_vehicles() for r in intersection.inroads)
        if before > after:
            total_discharged += (before - after)

        _, phase_name = PHASE_CYCLE[phase_idx]
        render(intersection, tick, rates, phase_name, total_discharged)
        time.sleep(delay)

    print(f"\n  Scenario complete. Total discharged: {total_discharged:.0f}")
    time.sleep(1.5)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    scenarios = [
        (
            "Uniform Flow — all approaches equal",
            {'north': 1.5, 'south': 1.5, 'east': 1.5, 'west': 1.5},
            50, 0.12, None
        ),
        (
            "Asymmetric — N/S dominant",
            {'north': 3.0, 'south': 3.0, 'east': 0.5, 'west': 0.5},
            50, 0.12, None
        ),
        (
            "Priority Vehicle — burst on NORTH at tick 20",
            {'north': 1.0, 'south': 1.0, 'east': 1.0, 'west': 1.0},
            60, 0.12, 20
        ),
        (
            "Heavy Load — near saturation",
            {'north': 3.5, 'south': 3.5, 'east': 3.5, 'west': 3.5},
            50, 0.10, None
        ),
    ]

    print(f"\n  {BOLD}CTM Four-Way Intersection Visualizer{RESET}")
    print(f"  Four scenarios. Cells fill with traffic, drain on green phase.")
    print(f"\n  Press Enter to start...")
    input()

    for name, rates, steps, delay, pv in scenarios:
        run_scenario(name, rates, steps, delay, pv)
        print(f"  Press Enter for next scenario...")
        input()

    print(f"\n  {BOLD}All scenarios complete.{RESET}\n")