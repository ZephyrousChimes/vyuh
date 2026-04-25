"""
road_test.py — CTM Road visualization
Run: python road_test.py
"""

import time
import os
from dataclasses import dataclass, field
from typing import List, NewType

RoadId = NewType('RoadId', str)

# ── CTM Core ──────────────────────────────────────────────────────────────────

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
class Road:
    id: RoadId
    cells: List[Cell]
    source_buffer: float = 0.0

    def inject(self, count: float):
        self.source_buffer += count

    def step(self) -> float:
        n = len(self.cells)
        flows = [0.0] * (n + 1)

        flows[0] = min(self.source_buffer, self.cells[0].supply())

        for i in range(n - 1):
            flows[i + 1] = min(
                self.cells[i].demand(),
                self.cells[i + 1].supply()
            )

        flows[n] = self.cells[-1].demand()

        self.source_buffer -= flows[0]
        for i in range(n):
            self.cells[i].update(flows[i], flows[i + 1])

        return flows[n]

    def outflow(self) -> float:
        return self.cells[-1].demand()


# ── Visualization ─────────────────────────────────────────────────────────────

def make_road(num_cells: int = 8) -> Road:
    cells = [
        Cell(
            jam_cap=10.0,
            flow_cap=3.0,
            free_flow=1.0,      # 1 cell per tick free flow
            shock_speed=0.5,    # backward wave half speed
        )
        for _ in range(num_cells)
    ]
    return Road(id=RoadId("R1"), cells=cells)


def density_bar(curr: float, jam_cap: float, width: int = 10) -> str:
    """Render a cell as a filled bar."""
    ratio = curr / jam_cap
    filled = int(ratio * width)
    empty = width - filled

    if ratio > 0.8:
        color = "\033[91m"   # red — near jam
    elif ratio > 0.5:
        color = "\033[93m"   # yellow — congested
    elif ratio > 0.0:
        color = "\033[92m"   # green — free flow
    else:
        color = "\033[90m"   # grey — empty

    reset = "\033[0m"
    bar = color + "█" * filled + "░" * empty + reset
    return f"{bar} {curr:4.1f}"


def render(road: Road, tick: int, outflow_total: float, scenario: str):
    os.system('clear')
    print(f"\033[1m  CTM Road Simulation — {scenario}\033[0m")
    print(f"  {'─' * 60}")
    print(f"  tick {tick:>3}   |   source buffer: {road.source_buffer:5.1f}   |   total discharged: {outflow_total:6.1f}")
    print(f"  {'─' * 60}")
    print()
    print("  DIRECTION OF TRAVEL  →  →  →  →  →  →  →  →  [INTERSECTION]")
    print()

    n = len(road.cells)
    print("  ", end="")
    for i, cell in enumerate(road.cells):
        label = f"C{i}"
        print(f"  {label:<4}", end="")
    print("   OUT")

    print("  ", end="")
    for cell in road.cells:
        bar = density_bar(cell.curr, cell.jam_cap, width=4)
        print(f"  {bar}", end="")
    arrow = "\033[96m" + f"  {road.outflow():4.1f}" + "\033[0m"
    print(arrow)

    print()
    print("  density (vehicles/cell):")
    print("  ", end="")
    for cell in road.cells:
        ratio = cell.curr / cell.jam_cap
        pct = int(ratio * 100)
        print(f"  {pct:>3}%  ", end="")
    print()

    print()
    print("  \033[92m█\033[0m free flow   \033[93m█\033[0m congested   \033[91m█\033[0m near jam   \033[90m░\033[0m empty")
    print(f"  {'─' * 60}")


def run_scenario(scenario: str, inject_fn, steps: int = 40, delay: float = 0.15):
    road = make_road(num_cells=8)
    total_outflow = 0.0

    for tick in range(steps):
        inject_fn(road, tick)
        outflow = road.step()
        total_outflow += outflow
        render(road, tick, total_outflow, scenario)
        time.sleep(delay)

    print(f"\n  Simulation complete. Total discharged: {total_outflow:.1f} vehicles")
    time.sleep(2)


# ── Scenarios ─────────────────────────────────────────────────────────────────

def scenario_steady(road: Road, tick: int):
    """Steady injection — 3 vehicles every tick."""
    if tick < 20:
        road.inject(3.0)


def scenario_burst(road: Road, tick: int):
    """Burst injection — 20 vehicles at tick 0, then nothing."""
    if tick == 0:
        road.inject(20.0)


def scenario_pulse(road: Road, tick: int):
    """Pulse — inject heavily, stop, inject again. Watch waves."""
    if tick < 8:
        road.inject(4.0)
    elif tick == 20:
        road.inject(15.0)


def scenario_priority(road: Road, tick: int):
    """Normal flow then priority vehicle burst clears everything."""
    if tick < 15:
        road.inject(2.0)
    if tick == 15:
        print("\n  \033[91m⚠  PRIORITY VEHICLE DETECTED — clearing path\033[0m")
        time.sleep(1)
        # priority vehicle enters, no more normal injection
        road.inject(1.0)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    scenarios = [
        ("Steady Inflow", scenario_steady),
        ("Burst — Wave Propagation", scenario_burst),
        ("Pulse — Stop and Go Waves", scenario_pulse),
        ("Priority Vehicle Clearance", scenario_priority),
    ]

    print("\033[1m  CTM Road Simulation\033[0m")
    print("  Four scenarios. Watch density waves propagate.\n")
    print("  Press Enter to start...")
    input()

    for name, fn in scenarios:
        print(f"\n  Starting: {name}")
        time.sleep(0.5)
        run_scenario(name, fn, steps=35, delay=0.12)
        print("  Press Enter for next scenario...")
        input()

    print("\n  All scenarios complete.\n")