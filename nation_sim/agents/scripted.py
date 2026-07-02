from __future__ import annotations

import numpy as np

from ..core.economy import country_demand
from ..core.resources import IS_CONSUMER, IS_ESSENTIAL, IS_TRADEABLE, N_RESOURCES
from ..core.world import WorldState
from .base import (
    Policy,
    _allocate_budget,
    _debt_headroom,
    _is_emergency,
    _reserve_health,
)


class BufferStockPolicy(Policy):
    """safe default — keeps a buffer of essentials, scales back when broke.

    params:
      essential_months     : months of essentials to stockpile
      goods_months         : months of non-essentials to stockpile
      price_ceiling_multiple: max price we'll pay (× ref price)
      emergency_ceiling_multiple: ceiling when in emergency
      max_reserve_spend    : fraction of reserves we can spend per tick
      emergency_welfare    : welfare below this = emergency mode
      emergency_health     : health below this = emergency mode
    """

    def __init__(
        self,
        essential_months: float = 6.0,
        goods_months: float = 4.0,
        price_ceiling_multiple: float = 8.0,
        emergency_ceiling_multiple: float = 7.0,
        max_reserve_spend: float = 0.35,
        emergency_welfare: float = 0.70,
        emergency_health: float = 0.25,
    ) -> None:
        self.essential_months = essential_months
        self.goods_months = goods_months
        self.price_ceiling = price_ceiling_multiple
        self.emergency_ceiling = emergency_ceiling_multiple
        self.max_reserve_spend = max_reserve_spend
        self.emergency_welfare = emergency_welfare
        self.emergency_health = emergency_health

    def act(
        self, i: int, world: WorldState, _rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pop      = world.population[i]
        reserves = world.reserves[i]
        welfare  = world.welfare[i]
        monthly  = country_demand(world)[i]

        health = _reserve_health(reserves, pop)
        in_emergency = _is_emergency(welfare, health, self.emergency_welfare, self.emergency_health)

        # scale buffer targets by health, but keep a minimum so we always buy something
        eff_essential = max(1.5, self.essential_months * health)
        eff_goods     = max(0.5, self.goods_months * health)

        target = np.where(
            IS_ESSENTIAL,
            monthly * eff_essential,
            np.where(IS_CONSUMER, monthly * eff_goods, 0.0),
        )

        gap  = target - world.stockpile[i]
        buy  = gap.clip(min=0.0)
        sell = (-gap).clip(min=0.0)

        # services can't be traded
        buy  = np.where(IS_TRADEABLE, buy, 0.0)
        sell = np.where(IS_TRADEABLE, sell, 0.0)

        # essentials get a higher price ceiling when we're stressed
        ceiling = np.where(
            IS_ESSENTIAL & in_emergency,
            self.emergency_ceiling,
            self.price_ceiling,
        )
        buy   = np.where(world.price > ceiling, 0.0, buy)
        limit = np.where(buy > 0, ceiling, np.nan)

        # if we're really low on cash, cut non-essentials hard
        if reserves < 50:
            buy = np.where(IS_ESSENTIAL, buy, buy * 0.2)

        # fund essentials first, non-essentials get leftovers
        emergency_borrow = _debt_headroom(world, i) * 0.30 if in_emergency else 0.0
        buy = _allocate_budget(buy, world.price, reserves, emergency_borrow, self.max_reserve_spend)

        return buy, sell, limit


class AutarkicPolicy(Policy):
    #do nothing

    def act(
        self, _i: int, _world: WorldState, _rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        z = np.zeros(N_RESOURCES, dtype=np.float64)
        return z, z.copy(), np.full(N_RESOURCES, np.nan, dtype=np.float64)


class MercantilistPolicy(Policy):
    # exports everything, buys nothing

    def __init__(self, buffer_months: float = 1.5, critical_months: float = 1.0) -> None:
        self.buffer_months = buffer_months
        self.critical_months = critical_months

    def act(
        self, i: int, world: WorldState, _rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        monthly = country_demand(world)[i]
        thin_buffer = np.where(IS_CONSUMER, monthly * self.buffer_months, 0.0)
        sell = (world.stockpile[i] - thin_buffer).clip(min=0.0)

        critical = monthly * self.critical_months
        shortfall = (critical - world.stockpile[i]).clip(min=0.0)
        buy = np.where(IS_CONSUMER & IS_TRADEABLE, shortfall, 0.0)
        sell = np.where(IS_TRADEABLE, sell, 0.0)

        limit = np.full(N_RESOURCES, np.nan, dtype=np.float64)
        return buy, sell, limit


class PriceAwarePolicy(Policy):
    """price-signal agent with three modes (normal, emergency, crisis)

    normal:    buy essentials below ceiling, sell surplus above floor
    emergency: higher ceiling for essentials, reduced non-essentials
    crisis:    highest ceiling, dump non-essentials to rebuild reserves
    """

    def __init__(
        self,
        essential_months: float = 8.0,
        goods_months: float = 3.0,
        buy_ceiling: float = 2.5,
        emergency_ceiling: float = 6.0,
        crisis_ceiling: float = 12.0,
        sell_floor: float = 0.60,
        max_reserve_spend: float = 0.35,
        emergency_welfare: float = 0.65,
        crisis_welfare: float = 0.45,
        emergency_health: float = 0.25,
        crisis_health: float = 0.10,
    ) -> None:
        self.essential_months  = essential_months
        self.goods_months      = goods_months
        self.buy_ceiling       = buy_ceiling
        self.emergency_ceiling = emergency_ceiling
        self.crisis_ceiling    = crisis_ceiling
        self.sell_floor        = sell_floor
        self.max_reserve_spend = max_reserve_spend
        self.emergency_welfare = emergency_welfare
        self.crisis_welfare    = crisis_welfare
        self.emergency_health  = emergency_health
        self.crisis_health     = crisis_health

    def act(
        self, i: int, world: WorldState, _rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pop      = world.population[i]
        reserves = world.reserves[i]
        welfare  = world.welfare[i]
        monthly  = country_demand(world)[i]

        health = _reserve_health(reserves, pop)

        in_crisis    = _is_emergency(welfare, health, self.crisis_welfare, self.crisis_health)
        in_emergency = _is_emergency(welfare, health, self.emergency_welfare, self.emergency_health)

        # buffer targets — scaled by health, full at health=0.5
        eff_essential = max(2.0, self.essential_months * min(1.0, health * 2.0))
        eff_goods     = 0.0 if in_crisis else max(0.5, self.goods_months * health)

        target = np.where(
            IS_ESSENTIAL,
            monthly * eff_essential,
            np.where(IS_CONSUMER, monthly * eff_goods, 0.0),
        )

        gap      = target - world.stockpile[i]
        buy_raw  = gap.clip(min=0.0)
        sell_raw = (-gap).clip(min=0.0)

        # pick ceiling based on severity
        ceiling = np.where(
            IS_ESSENTIAL & in_crisis,
            self.crisis_ceiling,
            np.where(IS_ESSENTIAL & in_emergency, self.emergency_ceiling, self.buy_ceiling),
        )

        buy  = np.where(world.price > ceiling, 0.0, buy_raw)
        sell = np.where(world.price < self.sell_floor, 0.0, sell_raw)

        # services aren't tradeable
        buy  = np.where(IS_TRADEABLE, buy, 0.0)
        sell = np.where(IS_TRADEABLE, sell, 0.0)

        # in crisis, dump non-essentials to raise cash
        if in_crisis:
            non_ess = IS_CONSUMER & ~IS_ESSENTIAL
            surplus = (world.stockpile[i] - monthly * 0.5).clip(min=0.0)
            sell = np.maximum(sell, np.where(non_ess, surplus, 0.0))
            buy  = np.where(non_ess, 0.0, buy)

        limit = np.where(buy > 0, ceiling,
                         np.where(sell > 0, self.sell_floor, np.nan))

        # borrow more when things are dire
        borrow_frac = 0.50 if in_crisis else (0.30 if in_emergency else 0.0)
        emergency_borrow = _debt_headroom(world, i) * borrow_frac
        buy = _allocate_budget(buy, world.price, reserves, emergency_borrow, self.max_reserve_spend)

        return buy, sell, limit
