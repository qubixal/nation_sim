# trade surplus appreciates, deficit depreciates
# FX rate movement
from __future__ import annotations

import numpy as np

from .economy import compute_gdp
from .world import WorldState

FX_TRADE_SENS = 0.02
FX_NOISE_STD = 0.01
FX_FLOOR = 0.05
FX_CEIL = 20.0
FX_STEP_CLAMP = 0.10


def update_fx(world: WorldState, trade_value: np.ndarray, rng: np.random.Generator) -> None:
    #trade surplus → fx down (stronger)
    #deficit → fx up (weaker)

    gdp = compute_gdp(world).clip(min=1e-6)
    net_balance = trade_value.sum(axis=1)
    step = -FX_TRADE_SENS * (net_balance / gdp)
    step = step + rng.normal(0.0, FX_NOISE_STD, size=world.n_countries)
    step = step.clip(-FX_STEP_CLAMP, FX_STEP_CLAMP)
    world.fx = (world.fx * (1.0 + step)).clip(FX_FLOOR, FX_CEIL)
