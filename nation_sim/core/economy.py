"""production, consumption, prices, population, tech, debt, investment."""
from __future__ import annotations

import numpy as np

from .resources import (
    DEMAND_WEIGHT,
    IS_CONSUMER,
    IS_SERVICE,
    N_CLASSES,
    N_RESOURCES,
    PER_CAPITA_DEMAND,
    PER_CAPITA_DEMAND_BY_CLASS,
    STOCK_DECAY_RATE,
)
from .world import (
    RESERVE_INCOME_PER_CAPITA,
    WorldState,
    debt_capacity,
)

POP_ALPHA = 0.7

PRICE_ELASTICITY = 0.08
PRICE_FLOOR = 0.20
PRICE_CEIL = 50.0
PRICE_STEP_CLAMP = 0.20

# production throttling — produce less when shelves are full
THROTTLE_TARGET_MONTHS = 6.0
THROTTLE_MAX_MONTHS = 18.0
MIN_PRODUCTION_UTIL = 0.10

# high price → ignore inventory throttle, run at full capacity (scarcity signal)
HIGH_PRICE_OVERRIDE_THRESHOLD = 2.0

# low price → idle capacity (deflationary surplus)
LOW_PRICE_THROTTLE_END = 0.60
LOW_PRICE_MIN_UTIL = 0.02

WELFARE_EMA_DECAY = 0.3
TECH_MONTHLY_GROWTH = 0.0008  # ≈ 1%/yr at welfare=1
TECH_MAX = 5.0

# demographic transition — growth peaks at intermediate welfare
# too poor = famine, too rich = low fertility
POP_PEAK_WELFARE = 0.60
POP_PEAK_ANNUAL_RATE = 0.010
POP_HIGH_WELFARE_DRAG = 0.020
POP_LOW_WELFARE_DRAG = 0.080
POP_LOW_WELFARE_THRESHOLD = 0.45

# operating cost per unit produced — the main reserve sink
PRODUCTION_COST_RATE = 0.005

# investment — small fraction of surplus reserves → new capacity
INVESTMENT_INTENSITY = 0.005
CAPITAL_COST_PER_UNIT = 800.0
SAFE_RESERVE_MONTHS = 12.0

# capacity depreciation (off by default — would race against tech growth)
CAPACITY_DECAY_RATE = 0.0

# migration — people move toward higher welfare, rich move fastest
CLASS_MOBILITY = np.array([0.0003, 0.0008, 0.0015])
MIGRATION_PULL_SCALE = 0.40
MAX_MIGRATION_FRACTION_PER_TICK = 0.02

# sovereign debt — borrow when broke, pay interest
DEBT_INTEREST_RATE = 0.004  # ≈ 5%/yr
DEBT_PAYDOWN_RATE = 0.05


def class_fractions(welfare: np.ndarray) -> np.ndarray:
    """return (..., 3) [poor, middle, rich] fractions per country.

    shifts with welfare: low = mostly poor, high = mostly rich.
    """
    w = np.asarray(welfare, dtype=np.float64)
    rich = 0.05 + 0.50 * w * w
    poor = (0.70 - 0.65 * w).clip(min=0.05)
    middle = 1.0 - rich - poor
    return np.stack([poor, middle, rich], axis=-1)


def country_demand(world: WorldState) -> np.ndarray:
    """total per-tick demand for each (country, resource), accounting for class mix."""
    cls = class_fractions(world.welfare)            # (N, 3)
    class_pop = world.population[:, None] * cls     # (N, 3) in millions
    demand = np.einsum("ic,cr->ir", class_pop, PER_CAPITA_DEMAND_BY_CLASS)
    return demand


def _utilization(world: WorldState) -> np.ndarray:
    """capacity utilisation in [LOW_PRICE_MIN_UTIL, 1.0].

    two signals:
      1. inventory buffer — produce less when shelves are stocked
      2. price — high price overrides to full util, low price idles capacity
    """
    demand_monthly = country_demand(world)
    months_buf = world.stockpile / np.clip(demand_monthly, 1e-6, None)
    cap_buf = world.stockpile / np.clip(world.capacity, 1e-6, None)
    buffer = np.where(IS_CONSUMER[None, :], months_buf, cap_buf)

    span = THROTTLE_MAX_MONTHS - THROTTLE_TARGET_MONTHS
    util = 1.0 - (buffer - THROTTLE_TARGET_MONTHS) / span * (1.0 - MIN_PRODUCTION_UTIL)
    util = util.clip(MIN_PRODUCTION_UTIL, 1.0)

    # high price = scarcity override for consumer resources
    scarce = (world.price[None, :] > HIGH_PRICE_OVERRIDE_THRESHOLD) & IS_CONSUMER[None, :]

    # low price = idle capacity
    low_factor = ((world.price - PRICE_FLOOR) /
                  (LOW_PRICE_THROTTLE_END - PRICE_FLOOR)).clip(0.0, 1.0)
    util_low = (util * low_factor[None, :]).clip(min=LOW_PRICE_MIN_UTIL)

    util = np.where(scarce, 1.0, util_low)
    # services are always full capacity (no stockpiling)
    util = np.where(IS_SERVICE[None, :], 1.0, util)
    return util


def produce(world: WorldState, rng: np.random.Generator, noise_std: float) -> np.ndarray:
    """add one month of production to stockpiles. returns (N, R) tensor."""
    N = world.n_countries
    pop_factor = np.power(world.population, POP_ALPHA)[:, None]
    noise = rng.normal(1.0, noise_std, size=(N, N_RESOURCES)).clip(0.2, 2.0)
    util = _utilization(world)
    production = world.capacity * pop_factor * world.tech * noise * util
    world.stockpile += production
    world.last_production = production
    return production


def charge_production_costs(world: WorldState, production: np.ndarray) -> None:
    """deduct operating costs from reserves — the main reserve sink."""
    cost = production.sum(axis=1) * PRODUCTION_COST_RATE
    world.reserves -= cost
    world.reserves.clip(min=0.0, out=world.reserves)


def consume(world: WorldState, demand: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """consume from stockpile up to demand. updates welfare EMA."""
    consumed = np.minimum(demand, world.stockpile)
    unmet = demand - consumed
    world.stockpile -= consumed

    weighted_demand = (demand * DEMAND_WEIGHT[None, :]).sum(axis=1)
    weighted_unmet = (unmet * DEMAND_WEIGHT[None, :]).sum(axis=1)
    tick_welfare = 1.0 - (weighted_unmet / np.clip(weighted_demand, 1e-9, None))
    tick_welfare = tick_welfare.clip(0.0, 1.0)
    world.welfare = (1.0 - WELFARE_EMA_DECAY) * world.welfare + WELFARE_EMA_DECAY * tick_welfare
    return consumed, unmet


def decay_stockpiles(world: WorldState) -> None:
    """monthly spoilage/wear/obsolescence on all stockpiles."""
    world.stockpile *= 1.0 - STOCK_DECAY_RATE[None, :]
    world.stockpile.clip(min=0.0, out=world.stockpile)


def decay_capacity(world: WorldState) -> None:
    """slow physical depreciation of production capacity."""
    world.capacity *= 1.0 - CAPACITY_DECAY_RATE


def update_prices(world: WorldState, production: np.ndarray, demand: np.ndarray) -> None:
    """spot-market price update based on flow supply vs demand + decay consumption.

    uses production flow (not stockpile) to avoid floor-lock.
    decay gives industrial resources a non-zero demand signal.
    """
    # only add decay demand for industrial resources (not consumer — they'd inflate from hoarded surplus)
    industrial_only = (~IS_CONSUMER).astype(np.float64)
    decay_demand = world.stockpile.sum(axis=0) * STOCK_DECAY_RATE * industrial_only
    supply = production.sum(axis=0) + 1e-6
    total_demand = demand.sum(axis=0) + decay_demand + 1e-6
    ratio = total_demand / supply
    step = ((ratio - 1.0) * PRICE_ELASTICITY).clip(-PRICE_STEP_CLAMP, PRICE_STEP_CLAMP)
    world.price = (world.price * (1.0 + step)).clip(PRICE_FLOOR, PRICE_CEIL)


def update_population(world: WorldState) -> None:
    """demographic transition model — growth peaks at mid welfare, drops at extremes."""
    high_welfare_drag = np.maximum(world.welfare - POP_PEAK_WELFARE, 0.0)
    low_welfare_drag = np.maximum(POP_LOW_WELFARE_THRESHOLD - world.welfare, 0.0)
    annual_rate = (
        POP_PEAK_ANNUAL_RATE
        - POP_HIGH_WELFARE_DRAG * high_welfare_drag
        - POP_LOW_WELFARE_DRAG * low_welfare_drag
    )
    monthly_rate = (annual_rate / 12.0).clip(-0.02, 0.02)
    world.population = (world.population * (1.0 + monthly_rate)).clip(0.1, 5_000.0)


def inject_reserve_income(world: WorldState) -> None:
    """add domestic-activity income to reserves.

    scaled by population × tech × welfare factor × class bonus.
    """
    avg_tech = world.tech.mean(axis=1)
    welfare_factor = 0.5 + 0.5 * world.welfare
    rich_share = class_fractions(world.welfare)[:, 2]
    class_bonus = 1.0 + 0.5 * rich_share
    income = (world.population
              * RESERVE_INCOME_PER_CAPITA
              * welfare_factor
              * avg_tech
              * class_bonus)
    world.reserves += income


def grow_tech(world: WorldState) -> None:
    """gradual tech improvement driven by welfare (proxy for R&D)."""
    monthly_gain = TECH_MONTHLY_GROWTH * world.welfare[:, None]
    world.tech = (world.tech * (1.0 + monthly_gain)).clip(0.3, TECH_MAX)


def migrate(world: WorldState) -> None:
    """move people toward higher-welfare countries. rich move fastest."""
    welfare = world.welfare
    diff = (welfare[None, :] - welfare[:, None]).clip(min=0.0)  # (origin, dest)
    np.fill_diagonal(diff, 0.0)

    cls = class_fractions(welfare)                              # (N, 3)
    mobility = (cls * CLASS_MOBILITY[None, :]).sum(axis=1)      # (N,) per-tick rate
    flow_rate = (mobility[:, None] * diff * MIGRATION_PULL_SCALE)
    # cap aggregate outflow per origin
    total_rate = flow_rate.sum(axis=1, keepdims=True)
    excess = (total_rate / MAX_MIGRATION_FRACTION_PER_TICK).clip(min=1.0)
    flow_rate = flow_rate / excess

    outflow = world.population[:, None] * flow_rate
    inflow = outflow.sum(axis=0)
    out_total = outflow.sum(axis=1)
    world.population = (world.population + inflow - out_total).clip(0.1, 5_000.0)


def service_debt(world: WorldState) -> None:
    """charge interest, repay from surplus, capitalise unpaid interest (capped)."""
    interest = world.debt * DEBT_INTEREST_RATE
    paid_from_reserves = np.minimum(interest, world.reserves)
    world.reserves -= paid_from_reserves
    unpaid = interest - paid_from_reserves

    # capitalise unpaid interest, capped at headroom
    cap = debt_capacity(world)
    headroom = (cap - world.debt).clip(min=0.0)
    capitalised = np.minimum(unpaid, headroom)
    world.debt += capitalised

    # voluntary paydown from surplus
    safe_reserve = world.population * RESERVE_INCOME_PER_CAPITA * SAFE_RESERVE_MONTHS
    surplus = (world.reserves - safe_reserve).clip(min=0.0)
    payment = np.minimum(surplus * DEBT_PAYDOWN_RATE, world.debt)
    world.reserves -= payment
    world.debt -= payment

    # never exceed capacity
    world.debt = np.minimum(world.debt, cap)


def invest_in_capacity(world: WorldState) -> None:
    """convert surplus reserves into production capacity.

    flows toward higher-priced resources (price² weighting).
    """
    safe_reserve = world.population * RESERVE_INCOME_PER_CAPITA * SAFE_RESERVE_MONTHS
    surplus = (world.reserves - safe_reserve).clip(min=0.0)
    spend = surplus * INVESTMENT_INTENSITY

    price_signal = world.price ** 2
    weights = price_signal / price_signal.sum()

    capacity_added = (spend[:, None] * weights[None, :]) / CAPITAL_COST_PER_UNIT
    world.capacity = world.capacity + capacity_added
    world.reserves -= spend


def compute_gdp(world: WorldState) -> np.ndarray:
    """flow GDP — market value of last tick's production."""
    return (world.last_production * world.price[None, :]).sum(axis=1)


# performance / stock-index dynamics
# composite of welfare + GDP/capita + reserve health - debt burden
# measured relative to world mean each tick
PERFORMANCE_GROWTH_GAIN = 0.003
PERFORMANCE_GROWTH_CLAMP = 0.005
PERFORMANCE_DRIFT = 0.0001
PERFORMANCE_NOISE_STD = 0.0010
PERFORMANCE_INDEX_FLOOR = 5.0
PERFORMANCE_INDEX_CEIL = 100_000.0


def update_performance(world: WorldState, rng: np.random.Generator) -> None:
    """update the per-country performance index.

    each dimension scored relative to world mean so median ≈ flat.
    outperformers compound up, underperformers compound down.
    """
    pop = np.maximum(world.population, 0.1)
    gdp = compute_gdp(world)
    gdp_pc = gdp / pop

    mean_welfare = max(float(world.welfare.mean()), 0.30)
    mean_gdp_pc = max(float(gdp_pc.mean()), 1e-6)

    welfare_rel = (world.welfare / mean_welfare).clip(0.2, 3.0)
    gdp_rel = (gdp_pc / mean_gdp_pc).clip(0.2, 5.0)

    safe_reserve = pop * RESERVE_INCOME_PER_CAPITA * SAFE_RESERVE_MONTHS
    reserve_health = (world.reserves / np.maximum(safe_reserve, 1.0)).clip(0.0, 1.0)

    annual_gdp = np.maximum(gdp * 12.0, 1.0)
    debt_burden = (world.debt / annual_gdp).clip(0.0, 1.0)

    score = (
        0.50 * welfare_rel
        + 0.30 * gdp_rel
        + 0.20 * reserve_health
        - 0.20 * debt_burden
    )

    growth = ((score - 1.0) * PERFORMANCE_GROWTH_GAIN
              ).clip(-PERFORMANCE_GROWTH_CLAMP, PERFORMANCE_GROWTH_CLAMP)
    noise = rng.normal(0.0, PERFORMANCE_NOISE_STD, size=world.n_countries)
    world.performance = (world.performance
                         * (1.0 + growth + PERFORMANCE_DRIFT + noise)
                        ).clip(PERFORMANCE_INDEX_FLOOR, PERFORMANCE_INDEX_CEIL)
