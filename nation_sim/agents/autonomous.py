"""autonomous agents — invest, set tariffs, and trade each tick.

4 archetypes:
  GrowthAgent       — free trade, double down on what you're good at
  MercantilistAgent — tax competitors, sell everything
  WelfareAgent      — keep citizens happy, big essential buffers
  BalancedAgent     — adjust tariffs based on reserve flow
"""
from __future__ import annotations

import numpy as np

from ..core.economy import (
    CAPITAL_COST_PER_UNIT,
    country_demand,
)
from ..core.resources import (
    IS_ESSENTIAL,
    IS_TRADEABLE,
    N_RESOURCES,
    RESOURCE_IDX,
)
from ..core.world import WorldState
from .base import (
    Policy,
    _allocate_budget,
    _debt_headroom,
    _is_emergency,
    _reserve_health,
)

# fraction of starting reserves to invest upfront
INITIAL_INVESTMENT_FRACTION = 0.20

# tariff tweak per tick + max tariff
TARIFF_STEP = 0.005
MAX_TARIFF = 0.35


# ── shared helpers ────────────────────────────────────────────────────────────

def _invest_by_weights(i: int, world: WorldState, weights: np.ndarray) -> None:
    """spend a fraction of reserves on capacity, weighted by resource importance"""
    budget = float(world.reserves[i]) * INITIAL_INVESTMENT_FRACTION
    world.reserves[i] -= budget
    world.capacity[i] += budget * weights / CAPITAL_COST_PER_UNIT


def _trade(
    i: int,
    world: WorldState,
    essential_months: float,
    nonessential_months: float,
    price_ceiling: float,
    sell_floor: float,
    max_reserve_spend: float,
    emergency_borrow_frac: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """generic trade builder — essentials first, non-essentials after"""
    monthly  = country_demand(world)[i]
    reserves = float(world.reserves[i])
    welfare  = float(world.welfare[i])
    health   = _reserve_health(reserves, float(world.population[i]))
    in_emergency = _is_emergency(welfare, health, 0.70, 0.25)

    target = np.where(IS_ESSENTIAL, monthly * essential_months, monthly * nonessential_months)
    gap    = target - world.stockpile[i]
    buy    = np.where(IS_TRADEABLE, gap.clip(min=0.0), 0.0)
    sell   = np.where(IS_TRADEABLE, (-gap).clip(min=0.0), 0.0)

    buy  = np.where(world.price > price_ceiling, 0.0, buy)
    sell = np.where(world.price < sell_floor, 0.0, sell)
    limit = np.where(buy > 0, price_ceiling, np.where(sell > 0, sell_floor, np.nan))

    borrow = _debt_headroom(world, i) * emergency_borrow_frac if in_emergency else 0.0
    buy = _allocate_budget(buy, world.price, reserves, borrow, max_reserve_spend)

    return buy, sell, limit


# ── agent archetypes ──────────────────────────────────────────────────────────

class GrowthAgent(Policy):
    """free trade specialist — invests in comparative advantage, zeros tariffs.

    invest: into whatever resources we're best at
    tariffs: reduce every tick toward zero
    trade: buy what we lack, sell everything else
    """

    def invest(self, i: int, world: WorldState) -> None:
        world_avg = (world.capacity * world.tech).mean(axis=0) + 1e-9
        own = world.capacity[i] * world.tech[i]
        advantage = (own / world_avg) ** 2  # sharpen the signal
        _invest_by_weights(i, world, advantage / advantage.sum())

    def adjust_tariffs(self, i: int, world: WorldState) -> None:
        world.tariffs[i] = np.maximum(0.0, world.tariffs[i] - TARIFF_STEP)

    def act(
        self, i: int, world: WorldState, _rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return _trade(
            i, world,
            essential_months=5.0, nonessential_months=3.0,
            price_ceiling=6.0,    sell_floor=0.50,
            max_reserve_spend=0.35, emergency_borrow_frac=0.20,
        )


class MercantilistAgent(Policy):
    """protectionist exporter — taxes rivals, sells aggressively.

    invest: into highest-price (most profitable) resources
    tariffs: raise on stronger competitors, lower on weaker ones
    trade: dump surplus, import only essentials at crisis level
    """

    def invest(self, i: int, world: WorldState) -> None:
        w = world.price ** 2
        _invest_by_weights(i, world, w / w.sum())

    def adjust_tariffs(self, i: int, world: WorldState) -> None:
        perf_i = float(world.performance[i])
        for j in range(world.n_countries):
            if j == i:
                continue
            if world.performance[j] > perf_i:
                world.tariffs[i, j] = np.minimum(MAX_TARIFF, world.tariffs[i, j] + TARIFF_STEP)
            else:
                world.tariffs[i, j] = np.maximum(0.0, world.tariffs[i, j] - TARIFF_STEP * 0.5)

    def act(
        self, i: int, world: WorldState, _rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        monthly  = country_demand(world)[i]
        reserves = float(world.reserves[i])

        # sell everything above a thin buffer
        thin = monthly * 1.5
        sell = np.where(IS_TRADEABLE, (world.stockpile[i] - thin).clip(min=0.0), 0.0)

        # only buy essentials when critically low
        critical = monthly * 0.75
        buy = np.where(
            IS_ESSENTIAL & IS_TRADEABLE,
            (critical - world.stockpile[i]).clip(min=0.0),
            0.0,
        )
        buy   = np.where(world.price > 8.0, 0.0, buy)
        limit = np.where(buy > 0, 8.0, np.nan)

        budget = reserves * 0.20
        cost   = float((buy * world.price).sum())
        if cost > budget > 0:
            buy *= budget / cost

        return buy, sell, limit


class WelfareAgent(Policy):
    """welfare-maximiser — keeps citizens happy, big essential buffers.

    invest: food, energy, fuel, services
    tariffs: zero on essentials (cheap imports = happy people), small on non-essentials
    trade: huge essential buffer, high price ceiling, generous emergency borrowing
    """

    def invest(self, i: int, world: WorldState) -> None:
        w = np.zeros(N_RESOURCES, dtype=np.float64)
        for name in ("food", "energy", "fuel", "services"):
            w[RESOURCE_IDX[name]] = 1.0
        _invest_by_weights(i, world, w / w.sum())

    def adjust_tariffs(self, i: int, world: WorldState) -> None:
        for r in range(N_RESOURCES):
            if IS_ESSENTIAL[r]:
                world.tariffs[i, :, r] = np.maximum(0.0, world.tariffs[i, :, r] - TARIFF_STEP)
            else:
                world.tariffs[i, :, r] = np.minimum(0.10, world.tariffs[i, :, r] + TARIFF_STEP * 0.3)

    def act(
        self, i: int, world: WorldState, _rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return _trade(
            i, world,
            essential_months=8.0, nonessential_months=2.0,
            price_ceiling=10.0,   sell_floor=0.40,
            max_reserve_spend=0.40, emergency_borrow_frac=0.40,
        )


class BalancedAgent(Policy):
    """adaptive generalist — adjusts tariffs based on reserve flow.

    invest: even split across resources, slight tilt toward services
    tariffs: raise when reserves fall, lower when they rise
    trade: moderate buffers, standard ceiling
    """

    def __init__(self) -> None:
        self._last_reserves: float | None = None

    def invest(self, i: int, world: WorldState) -> None:
        self._last_reserves = None
        w = np.ones(N_RESOURCES, dtype=np.float64)
        w[RESOURCE_IDX["services"]] = 1.5
        _invest_by_weights(i, world, w / w.sum())

    def adjust_tariffs(self, i: int, world: WorldState) -> None:
        current = float(world.reserves[i])
        if self._last_reserves is None:
            self._last_reserves = current
            return
        delta = current - self._last_reserves
        self._last_reserves = current
        step = TARIFF_STEP * 0.5
        if delta < 0:
            world.tariffs[i] = np.minimum(MAX_TARIFF, world.tariffs[i] + step)
        else:
            world.tariffs[i] = np.maximum(0.0, world.tariffs[i] - step)

    def act(
        self, i: int, world: WorldState, _rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return _trade(
            i, world,
            essential_months=6.0, nonessential_months=3.0,
            price_ceiling=8.0,    sell_floor=0.50,
            max_reserve_spend=0.35, emergency_borrow_frac=0.30,
        )
