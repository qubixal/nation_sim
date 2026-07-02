# Quick launch script
from __future__ import annotations

import argparse
import time

from nation_sim.agents.autonomous import (
    BalancedAgent,
    GrowthAgent,
    MercantilistAgent,
    WelfareAgent,
)
from nation_sim.agents.scripted import BufferStockPolicy
from nation_sim.recording.recorder import Recorder
from nation_sim.sim.config import WorldConfig, generate_world
from nation_sim.sim.engine import Engine, EngineConfig

_AUTONOMOUS_CYCLE = [GrowthAgent, MercantilistAgent, WelfareAgent, BalancedAgent]


def _build_policies(agents_mode: str, n: int) -> list:
    if agents_mode == "buffer":
        return [BufferStockPolicy() for _ in range(n)]
    if agents_mode == "growth":
        return [GrowthAgent() for _ in range(n)]
    if agents_mode == "mercantilist":
        return [MercantilistAgent() for _ in range(n)]
    if agents_mode == "welfare":
        return [WelfareAgent() for _ in range(n)]
    if agents_mode == "balanced":
        return [BalancedAgent() for _ in range(n)]
    return [_AUTONOMOUS_CYCLE[k % len(_AUTONOMOUS_CYCLE)]() for k in range(n)]


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="nation_sim",
        description="Multi-nation economic simulator with live dashboard.",
    )
    ap.add_argument("--ticks", type=int, default=240,
                    help="Months to simulate (default: 240).")
    ap.add_argument("--preset", default="balanced",
                    choices=["balanced", "mercantilist", "free_trade"],
                    help="World preset (default: balanced).")
    ap.add_argument("--agents", default="autonomous",
                    choices=["autonomous", "buffer", "growth", "mercantilist",
                             "welfare", "balanced"],
                    help="Agent strategy (default: autonomous).")
    ap.add_argument("--seed", type=int, default=0,
                    help="Random seed (default: 0).")
    ap.add_argument("--countries", type=int, default=10,
                    help="Number of countries (default: 10).")
    ap.add_argument("--tick-delay-ms", type=int, default=200,
                    help="Delay between ticks in ms (default: 200). 0 = full speed.")
    ap.add_argument("--port", type=int, default=8050,
                    help="Dashboard port (default: 8050).")
    ap.add_argument("--no-dashboard", action="store_true",
                    help="Skip the live dashboard.")
    ap.add_argument("--no-browser", action="store_true",
                    help="Don't auto-open the browser.")
    ap.add_argument("--parquet", action="store_true",
                    help="Also write Parquet files alongside CSV.")
    args = ap.parse_args()

    world = generate_world(WorldConfig(
        n_countries=args.countries, preset=args.preset, seed=args.seed,
    ))
    policies = _build_policies(args.agents, world.n_countries)
    recorder = Recorder(
        "runs/latest", flush_every=30, write_csv=True, write_parquet=args.parquet,
    )
    engine = Engine(world, policies, EngineConfig(seed=args.seed), recorder=recorder)

    dashboard_url = None
    if not args.no_dashboard:
        from nation_sim.viz.dashboard import make_app, run_in_thread
        app = make_app(recorder)
        run_in_thread(app, port=args.port)
        dashboard_url = f"http://127.0.0.1:{args.port}"
        print(f"Dashboard live: {dashboard_url}")

        if not args.no_browser:
            import webbrowser
            webbrowser.open(dashboard_url)

        print(f"Waiting 2s for browser to connect...")
        time.sleep(2.0)

    print(f"Running {args.ticks} ticks ({args.preset}, {args.agents})...")
    delay_s = max(args.tick_delay_ms, 0) / 1000.0
    try:
        for _ in range(args.ticks):
            engine.step()
            if delay_s > 0:
                time.sleep(delay_s)
    finally:
        recorder.close()

    print(f"\nDone. {args.ticks} ticks, {world.n_countries} countries.")
    print(f"  CSV: {recorder.country_csv.resolve()}")

    if dashboard_url:
        print(f"\nDashboard live at {dashboard_url} — Ctrl+C to exit.")
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
