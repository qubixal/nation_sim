"""procedural world generation with named presets."""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ..core.economy import POP_ALPHA
from ..core.resources import N_RESOURCES, PER_CAPITA_DEMAND
from ..core.world import WorldState


@dataclass(frozen=True)
class WorldConfig:
    n_countries: int = 10
    preset: str = "balanced"
    seed: int | None = 0
    production_noise_std: float = 0.15
    tariff_density: float = 0.10       # fraction of (importer, exporter, resource) triples with a tariff
    tariff_range: tuple[float, float] = (0.05, 0.20)
    starting_buffer_months: float = 6.0  # match BufferStockPolicy target
    fx_spread: float = 0.30
    reference_reserves: float = 1_000.0
    trade_cost_base: float = 0.02      # minimum cost for any shipment
    trade_cost_distance: float = 0.20  # additional cost = distance × this


_PRESETS: dict[str, dict] = {
    "balanced": {},
    "mercantilist": {"tariff_density": 0.60, "tariff_range": (0.10, 0.35)},
    "free_trade": {"tariff_density": 0.0, "tariff_range": (0.0, 0.0)},
}


def _resolve(cfg: WorldConfig) -> WorldConfig:
    if cfg.preset not in _PRESETS:
        raise ValueError(f"Unknown preset {cfg.preset!r}. Known: {list(_PRESETS)}")
    overrides = _PRESETS[cfg.preset]
    return replace(cfg, **overrides) if overrides else cfg


def generate_world(cfg: WorldConfig | None = None) -> WorldState:
    """build a world from config — random population, capacity, tariffs, geography."""
    cfg = _resolve(cfg or WorldConfig())
    rng = np.random.default_rng(cfg.seed)
    N, R = cfg.n_countries, N_RESOURCES

    # lognormal population clipped to [5, 80]M — prevents structural lock-ins
    population = rng.lognormal(mean=2.5, sigma=1.1, size=N).clip(5.0, 80.0)

    # gamma-distributed capacity, scaled by demand fraction
    capacity = rng.gamma(shape=2.0, scale=1.0, size=(N, R))
    demand_scale = PER_CAPITA_DEMAND / PER_CAPITA_DEMAND.max()
    capacity *= demand_scale[None, :]
    capacity = np.maximum(capacity, demand_scale * 0.50)

    # normalize so total production at full util = 90% of total demand
    total_prod = (capacity * (population[:, None] ** POP_ALPHA)).sum(axis=0)
    total_demand = (PER_CAPITA_DEMAND[None, :] * population[:, None]).sum(axis=0)
    norm_scale = np.where(total_prod > 1e-6, 0.90 * total_demand / total_prod, 1.0)
    capacity *= norm_scale[None, :]

    tech = rng.normal(1.0, 0.2, size=(N, R)).clip(0.3, 2.0)

    demand_monthly = PER_CAPITA_DEMAND[None, :] * population[:, None]
    stockpile = demand_monthly * cfg.starting_buffer_months + 1.0

    price = np.ones(R, dtype=np.float64)
    fx = rng.lognormal(mean=0.0, sigma=cfg.fx_spread, size=N)
    reserves = np.full(N, cfg.reference_reserves, dtype=np.float64)

    # random tariffs based on preset
    tariffs = np.zeros((N, N, R), dtype=np.float64)
    if cfg.tariff_density > 0.0 and cfg.tariff_range[1] > 0.0:
        mask = rng.random((N, N, R)) < cfg.tariff_density
        values = rng.uniform(cfg.tariff_range[0], cfg.tariff_range[1], size=(N, N, R))
        tariffs = np.where(mask, values, 0.0)
        diag = np.arange(N)
        tariffs[diag, diag, :] = 0.0

    country_names = [f"C{i:02d}" for i in range(N)]
    welfare = np.ones(N, dtype=np.float64)

    # random positions on unit square → pairwise distance → trade cost
    position = rng.uniform(0.0, 1.0, size=(N, 2))
    diff = position[:, None, :] - position[None, :, :]
    distance = np.sqrt((diff ** 2).sum(axis=-1))
    distance /= max(distance.max(), 1e-6)
    trade_cost = cfg.trade_cost_base + cfg.trade_cost_distance * distance
    np.fill_diagonal(trade_cost, 0.0)

    world = WorldState(
        country_names=country_names,
        tick=0,
        population=population.astype(np.float64),
        stockpile=stockpile.astype(np.float64),
        capacity=capacity.astype(np.float64),
        tech=tech.astype(np.float64),
        price=price,
        fx=fx.astype(np.float64),
        reserves=reserves,
        tariffs=tariffs,
        welfare=welfare,
        last_production=np.zeros((N, N_RESOURCES), dtype=np.float64),
        position=position,
        distance=distance,
        trade_cost=trade_cost,
        debt=np.zeros(N, dtype=np.float64),
        performance=np.full(N, 100.0, dtype=np.float64),
    )
    world.check()
    return world
