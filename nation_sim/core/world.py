"""vectorized world state — all per-country / per-resource arrays."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .resources import N_RESOURCES

# domestic income per million people per tick
RESERVE_INCOME_PER_CAPITA = 0.08

# max debt as months of base income
DEBT_CAPACITY_MONTHS = 36.0


@dataclass
class WorldState:
    """all simulation state as numpy arrays.

    axis 0 = country, axis 1 = resource (for 2D arrays).
    """

    country_names: list[str]
    tick: int = 0

    population: np.ndarray = field(default_factory=lambda: np.zeros(0))

    # shape (N, R)
    stockpile: np.ndarray = field(default_factory=lambda: np.zeros((0, N_RESOURCES)))
    capacity: np.ndarray = field(default_factory=lambda: np.zeros((0, N_RESOURCES)))
    tech: np.ndarray = field(default_factory=lambda: np.zeros((0, N_RESOURCES)))

    # market state
    price: np.ndarray = field(default_factory=lambda: np.ones(N_RESOURCES))
    fx: np.ndarray = field(default_factory=lambda: np.ones(0))
    reserves: np.ndarray = field(default_factory=lambda: np.zeros(0))

    # ad-valorem tariffs — shape (N_importer, N_exporter, R)
    tariffs: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 0, N_RESOURCES))
    )

    welfare: np.ndarray = field(default_factory=lambda: np.ones(0))

    # last tick's production — used for flow GDP
    last_production: np.ndarray = field(
        default_factory=lambda: np.zeros((0, N_RESOURCES))
    )

    # geography — unit square positions, pairwise distance, trade cost
    position: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    distance: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    trade_cost: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))

    # sovereign debt
    debt: np.ndarray = field(default_factory=lambda: np.zeros(0))

    # performance index — starts at 100, compounds each tick
    performance: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def n_countries(self) -> int:
        return len(self.country_names)

    def check(self) -> None:
        """assert shape invariants — cheap guard against bad mutations."""
        N, R = self.n_countries, N_RESOURCES
        assert self.population.shape == (N,), self.population.shape
        assert self.stockpile.shape == (N, R), self.stockpile.shape
        assert self.capacity.shape == (N, R), self.capacity.shape
        assert self.tech.shape == (N, R), self.tech.shape
        assert self.price.shape == (R,), self.price.shape
        assert self.fx.shape == (N,), self.fx.shape
        assert self.reserves.shape == (N,), self.reserves.shape
        assert self.tariffs.shape == (N, N, R), self.tariffs.shape
        assert self.welfare.shape == (N,), self.welfare.shape
        assert self.last_production.shape == (N, R), self.last_production.shape
        assert self.position.shape == (N, 2), self.position.shape
        assert self.distance.shape == (N, N), self.distance.shape
        assert self.trade_cost.shape == (N, N), self.trade_cost.shape
        assert self.debt.shape == (N,), self.debt.shape
        assert self.performance.shape == (N,), self.performance.shape

    def __repr__(self) -> str:
        N = self.n_countries
        avg_w = float(self.welfare.mean()) if N else 0.0
        avg_r = float(self.reserves.mean()) if N else 0.0
        tot_p = float(self.population.sum()) if N else 0.0
        return (
            f"WorldState(tick={self.tick}, countries={N}, "
            f"avg_welfare={avg_w:.3f}, avg_reserves={avg_r:.1f}, "
            f"total_pop={tot_p:.1f}M)"
        )


def debt_capacity(world: WorldState) -> np.ndarray:
    """max allowed outstanding debt per country."""
    return world.population * RESERVE_INCOME_PER_CAPITA * DEBT_CAPACITY_MONTHS
