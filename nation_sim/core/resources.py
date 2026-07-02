"""resource types, income classes, and per-capita demand profiles."""
from __future__ import annotations

import numpy as np

# the 9 tradeable resources
RESOURCES: tuple[str, ...] = (
    "food",
    "energy",
    "metal",
    "wood",
    "goods",
    "luxury",
    "tech",
    "fuel",
    "services",
)

N_RESOURCES: int = len(RESOURCES)
RESOURCE_IDX: dict[str, int] = {name: i for i, name in enumerate(RESOURCES)}

# income classes — composition shifts with welfare (engel/kuznets style)
CLASS_NAMES: tuple[str, ...] = ("poor", "middle", "rich")
N_CLASSES: int = len(CLASS_NAMES)

# per-capita monthly demand by (class, resource)
# poor spend more on staples, rich on luxury/tech/services — engel's law
PER_CAPITA_DEMAND_BY_CLASS: np.ndarray = np.array(
    [
        # food, energy, metal, wood, goods, luxury, tech, fuel, services
        [1.20, 0.50, 0.04, 0.05, 0.30, 0.05, 0.01, 0.40, 0.20],   # poor
        [1.00, 0.60, 0.05, 0.05, 0.40, 0.15, 0.02, 0.50, 0.50],   # middle
        [0.80, 0.70, 0.06, 0.05, 0.50, 0.40, 0.05, 0.60, 1.00],   # rich
    ],
    dtype=np.float64,
)

# class-averaged demand at mean welfare ≈ 0.55 (used for capacity calibration)
_CALIB_FRACTIONS = np.array([0.40, 0.40, 0.20])
PER_CAPITA_DEMAND: np.ndarray = (
    _CALIB_FRACTIONS[:, None] * PER_CAPITA_DEMAND_BY_CLASS
).sum(axis=0)

# how much unmet demand hurts welfare — staples matter most
DEMAND_WEIGHT: np.ndarray = np.array(
    [3.0, 2.0, 0.3, 0.3, 1.0, 0.5, 0.2, 1.5, 1.5],
    dtype=np.float64,
)

# monthly stockpile decay — services decay at 100% (can't store a haircut)
STOCK_DECAY_RATE: np.ndarray = np.array(
    [
        0.020,  # food    — perishable
        0.010,  # energy  — leakage
        0.008,  # metal   — minimal rust
        0.010,  # wood    — rot
        0.010,  # goods   — wear and tear
        0.008,  # luxury  — fashion cycles
        0.015,  # tech    — obsolescence
        0.018,  # fuel    — evaporation
        1.000,  # services — non-storable
    ],
    dtype=np.float64,
)

# life-critical resources — used by policies for priority trade decisions
IS_ESSENTIAL: np.ndarray = np.zeros(N_RESOURCES, dtype=bool)
IS_ESSENTIAL[[RESOURCE_IDX["food"], RESOURCE_IDX["energy"], RESOURCE_IDX["fuel"]]] = True

IS_CONSUMER: np.ndarray = PER_CAPITA_DEMAND > 0

# services can't be traded internationally
IS_TRADEABLE: np.ndarray = np.ones(N_RESOURCES, dtype=bool)
IS_TRADEABLE[RESOURCE_IDX["services"]] = False

IS_SERVICE: np.ndarray = np.zeros(N_RESOURCES, dtype=bool)
IS_SERVICE[RESOURCE_IDX["services"]] = True

assert PER_CAPITA_DEMAND.shape == (N_RESOURCES,)
assert PER_CAPITA_DEMAND_BY_CLASS.shape == (N_CLASSES, N_RESOURCES)
assert DEMAND_WEIGHT.shape == (N_RESOURCES,)
assert STOCK_DECAY_RATE.shape == (N_RESOURCES,)
