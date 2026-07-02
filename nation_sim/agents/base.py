"""Policy interface + shared helpers for scripted and autonomous agents"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..core.resources import IS_ESSENTIAL, N_RESOURCES
from ..core.world import RESERVE_INCOME_PER_CAPITA, WorldState, debt_capacity

# months of base income at which reserve_health = 1.0
_HEALTH_MONTHS = 24.0


class Policy(ABC):
    """decides trade orders for a single country each tick"""

    def invest(self, country_idx: int, world: WorldState) -> None:
        """one-time investment before the first tick. default: no-op"""

    def adjust_tariffs(self, country_idx: int, world: WorldState) -> None:
        """per-tick tariff tweak before market clearing. default: no-op"""

    @abstractmethod
    def act(
        self, country_idx: int, world: WorldState, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """return (buy, sell, limit), each shape (R,).

        buy[r], sell[r]: non-negative quantities.
        limit[r]: price limit in ref currency; NaN = no limit.
                  ceiling if buying, floor if selling.
        """
        raise NotImplementedError


# ── shared helpers ────────────────────────────────────────────────────────────

def _debt_headroom(world: WorldState, i: int) -> float:
    """how much more debt country i can take on"""
    cap = float(debt_capacity(world)[i])
    return max(cap - float(world.debt[i]), 0.0)


def _reserve_health(reserves: float, pop: float) -> float:
    """0-1 score, 1.0 = plenty of reserves relative to income"""
    monthly_income = pop * RESERVE_INCOME_PER_CAPITA
    return float(np.clip(reserves / max(monthly_income * _HEALTH_MONTHS, 1.0), 0.0, 1.0))


def _is_emergency(welfare: float, health: float, w_thresh: float, h_thresh: float) -> bool:
    """true if welfare or reserve health is below threshold"""
    return welfare < w_thresh or health < h_thresh


def _allocate_budget(
    buy: np.ndarray,
    price: np.ndarray,
    reserves: float,
    emergency_borrow: float,
    max_reserve_spend: float,
) -> np.ndarray:
    """priority budget: essentials get funded first, then non-essentials with leftovers"""
    budget = max((reserves + emergency_borrow) * max_reserve_spend, 0.0)
    ess = IS_ESSENTIAL.astype(np.float64)
    ess_cost = float((buy * price * ess).sum())
    other_cost = float((buy * price * (1.0 - ess)).sum())

    if ess_cost > budget > 0:
        buy = np.where(IS_ESSENTIAL, buy * (budget / ess_cost), 0.0)
    elif ess_cost + other_cost > budget > 0:
        remaining = budget - ess_cost
        buy = np.where(IS_ESSENTIAL, buy, buy * (remaining / max(other_cost, 1e-6)))
    return buy


def batch_act(
    policies: list[Policy], world: WorldState, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """call each policy and stack into (N, R) arrays for market clearing"""
    N, R = world.n_countries, N_RESOURCES
    buy = np.zeros((N, R), dtype=np.float64)
    sell = np.zeros((N, R), dtype=np.float64)
    limit = np.full((N, R), np.nan, dtype=np.float64)
    for i, p in enumerate(policies):
        b, s, l = p.act(i, world, rng)
        buy[i] = b
        sell[i] = s
        limit[i] = l
    return buy, sell, limit
