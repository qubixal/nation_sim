"""head-to-head policy comparison on identical starting worlds.

runs each strategy on the same seed and prints a summary table.
usage:
    python -m nation_sim.scripts.compare_policies [--ticks 120] [--seed 42]
"""
from __future__ import annotations

import argparse

import pandas as pd

from nation_sim.agents import AUTONOMOUS_CYCLE, POLICY_REGISTRY
from nation_sim.sim.config import WorldConfig, generate_world
from nation_sim.sim.engine import Engine, EngineConfig

# all registered policies + a mixed autonomous strategy
STRATEGIES = {
    name: lambda n, cls=cls: [cls() for _ in range(n)]
    for name, cls in POLICY_REGISTRY.items()
}
STRATEGIES["autonomous_mix"] = (
    lambda n: [AUTONOMOUS_CYCLE[k % len(AUTONOMOUS_CYCLE)]() for k in range(n)]
)


def _run(name: str, factory, cfg: WorldConfig, n_ticks: int) -> dict:
    world = generate_world(cfg)
    engine = Engine(world, factory(world.n_countries), EngineConfig(seed=cfg.seed))
    engine.run(n_ticks)
    w = world
    collapsed = int((w.welfare < 0.5).sum())
    return {
        "strategy":       name,
        "stock_mean":     round(float(w.performance.mean()), 1),
        "stock_max":      round(float(w.performance.max()),  1),
        "welfare_mean":   round(float(w.welfare.mean()), 3),
        "welfare_min":    round(float(w.welfare.min()),  3),
        "collapsed_n":    collapsed,
        "pop_total_M":    round(float(w.population.sum()), 1),
        "reserves_mean":  round(float(w.reserves.mean()), 2),
        "price_food":     round(float(w.price[0]), 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks",    type=int, default=120)
    ap.add_argument("--seed",     type=int, default=42)
    ap.add_argument("--countries",type=int, default=10)
    ap.add_argument("--preset",   default="balanced")
    args = ap.parse_args()

    cfg = WorldConfig(n_countries=args.countries, preset=args.preset, seed=args.seed)

    print(f"Comparing policies: {args.ticks} ticks, {args.countries} countries, "
          f"preset={args.preset}, seed={args.seed}\n")

    rows = [_run(name, factory, cfg, args.ticks) for name, factory in STRATEGIES.items()]
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print("\nstock_mean/max = performance index (start=100); "
          "welfare_mean/min = EMA welfare; collapsed_n = countries with welfare < 0.5; "
          "reserves_mean = avg ref-currency reserves.")


if __name__ == "__main__":
    main()
