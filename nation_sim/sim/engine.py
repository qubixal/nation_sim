"""tick loop — wires economy, market, currency, and recorder together."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from ..agents.base import Policy, batch_act
from ..core import economy
from ..core.currency import update_fx
from ..core.market import clear_market
from ..core.world import WorldState
from ..recording.recorder import Recorder

logger = logging.getLogger(__name__)


@dataclass
class EngineConfig:
    production_noise_std: float = 0.15
    seed: int | None = 0

    def __post_init__(self) -> None:
        if self.production_noise_std < 0.0:
            raise ValueError(
                f"production_noise_std must be >= 0, got {self.production_noise_std}"
            )
        if self.production_noise_std > 1.0:
            raise ValueError(
                f"production_noise_std must be <= 1.0, got {self.production_noise_std}"
            )


# stochastic shock events — each tick, each shock is independently rolled
_SHOCK_DEFS = [
    {"name": "SUPPLY SHOCK",    "prob": 0.003,
     "msg": "Global production disruption — output halved this tick",
     "kind": "prod_all",       "mult": 0.50},
    {"name": "FINANCIAL PANIC", "prob": 0.002,
     "msg": "Market panic — reserves drained 20%",
     "kind": "reserves",       "mult": 0.80},
    {"name": "DROUGHT",         "prob": 0.002,
     "msg": "Agricultural drought — food & wood output severely reduced",
     "kind": "prod_resources",  "mult": 0.20, "resources": [0, 3]},
    {"name": "ENERGY CRISIS",   "prob": 0.002,
     "msg": "Energy supply disruption — energy & fuel output halved",
     "kind": "prod_resources",  "mult": 0.30, "resources": [1, 7]},
    {"name": "TECH COLLAPSE",   "prob": 0.001,
     "msg": "Technology disruption — global innovation output near zero this tick",
     "kind": "prod_resources",  "mult": 0.05, "resources": [6]},
]


class Engine:
    """one tick = one month. deterministic given (world, policies, seed)."""

    def __init__(
        self,
        world: WorldState,
        policies: list[Policy],
        cfg: EngineConfig | None = None,
        recorder: Recorder | None = None,
    ) -> None:
        if len(policies) != world.n_countries:
            raise ValueError(
                f"Need one policy per country; got {len(policies)} for {world.n_countries}."
            )
        self.world = world
        self.policies = policies
        self.cfg = cfg or EngineConfig()
        self.recorder = recorder
        self.rng = np.random.default_rng(self.cfg.seed)

        # one-time infrastructure investment
        for i, p in enumerate(self.policies):
            p.invest(i, world)
        world.check()
        logger.info(
            "Engine initialised: %d countries, seed=%s, noise=%.3f",
            world.n_countries, self.cfg.seed, self.cfg.production_noise_std,
        )

    def step(self) -> None:
        w = self.world
        production = economy.produce(w, self.rng, self.cfg.production_noise_std)

        # roll for shocks before consumption
        for shock in _SHOCK_DEFS:
            if self.rng.random() < shock["prob"]:
                kind = shock["kind"]
                mult = shock["mult"]
                if kind == "prod_all":
                    reduction = production * (1.0 - mult)
                    w.stockpile -= reduction
                    production -= reduction
                    w.last_production = production.copy()
                elif kind == "prod_resources":
                    for r in shock["resources"]:
                        reduction = production[:, r] * (1.0 - mult)
                        w.stockpile[:, r] -= reduction
                        production[:, r] -= reduction
                    w.last_production = production.copy()
                elif kind == "reserves":
                    w.reserves *= mult
                logger.info("Tick %d: %s — %s", w.tick, shock["name"], shock["msg"])
                if self.recorder is not None:
                    self.recorder.record_crash(w.tick, shock["name"], shock["msg"])

        economy.charge_production_costs(w, production)
        demand = economy.country_demand(w)
        consumed, _unmet = economy.consume(w, demand)
        economy.decay_stockpiles(w)
        # price signal uses higher of actual consumption vs 50% of desired demand
        price_demand = np.maximum(consumed, 0.50 * demand)
        economy.update_prices(w, production, price_demand)

        for i, p in enumerate(self.policies):
            p.adjust_tariffs(i, w)

        buy, sell, limit = batch_act(self.policies, w, self.rng)
        trade_value, shipments = clear_market(w, buy, sell, limit, self.rng)

        update_fx(w, trade_value, self.rng)
        economy.inject_reserve_income(w)
        economy.service_debt(w)
        economy.update_population(w)
        economy.migrate(w)
        economy.grow_tech(w)
        economy.decay_capacity(w)
        economy.invest_in_capacity(w)
        economy.update_performance(w, self.rng)

        w.tick += 1
        w.check()
        if self.recorder is not None:
            self.recorder.record(w, shipments=shipments)

    def run(self, n_ticks: int) -> None:
        for _ in range(n_ticks):
            self.step()
