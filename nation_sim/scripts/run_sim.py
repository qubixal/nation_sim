"""end-to-end run: procedural world → sim → CSV + live dashboard."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from nation_sim.agents import AUTONOMOUS_CYCLE, POLICY_REGISTRY
from nation_sim.recording.recorder import Recorder
from nation_sim.sim.config import WorldConfig, generate_world
from nation_sim.sim.engine import Engine, EngineConfig


def _build_policies(agents_mode: str, n: int) -> list:
    if agents_mode == "autonomous":
        return [AUTONOMOUS_CYCLE[k % len(AUTONOMOUS_CYCLE)]() for k in range(n)]
    cls = POLICY_REGISTRY.get(agents_mode)
    if cls is None:
        raise ValueError(
            f"Unknown agent mode {agents_mode!r}. "
            f"Choose from: {list(POLICY_REGISTRY)} + autonomous"
        )
    return [cls() for _ in range(n)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="balanced",
                    choices=["balanced", "mercantilist", "free_trade"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--countries", type=int, default=10)
    ap.add_argument("--ticks", type=int, default=240)
    ap.add_argument("--out", type=Path, default=None,
                    help="Output directory (default: runs/<preset>).")
    ap.add_argument("--agents", default="autonomous",
                    choices=["autonomous"] + list(POLICY_REGISTRY.keys()),
                    help="Agent strategy mix (default: autonomous — cycles all 4 archetypes).")
    ap.add_argument("--tick-delay-ms", type=int, default=200,
                    help="Sleep between ticks for dashboard animation. 0 = full speed.")
    ap.add_argument("--parquet", action="store_true",
                    help="Also write Parquet alongside CSV (for ML pipelines).")
    ap.add_argument("--no-dashboard", action="store_true",
                    help="Skip the live dashboard.")
    ap.add_argument("--no-browser", action="store_true",
                    help="Don't auto-open the browser.")
    ap.add_argument("--start-delay-s", type=float, default=2.0,
                    help="Seconds to wait for browser to connect before ticking.")
    ap.add_argument("--port", type=int, default=8050)
    args = ap.parse_args()
    out_dir = args.out or Path("runs") / args.preset

    world = generate_world(WorldConfig(
        n_countries=args.countries, preset=args.preset, seed=args.seed,
    ))
    policies = _build_policies(args.agents, world.n_countries)
    recorder = Recorder(
        out_dir,
        flush_every=30,
        write_csv=True,
        write_parquet=args.parquet,
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

        if args.start_delay_s > 0:
            print(f"Waiting {args.start_delay_s:.1f}s for browser to connect...")
            time.sleep(args.start_delay_s)

    print(f"Ticking {args.ticks} months at {args.tick_delay_ms} ms/tick...")
    delay_s = max(args.tick_delay_ms, 0) / 1000.0
    try:
        for _ in range(args.ticks):
            engine.step()
            if delay_s > 0:
                time.sleep(delay_s)
    finally:
        recorder.close()

    print(
        f"\nDone. {args.ticks} ticks, {world.n_countries} countries, "
        f"preset={args.preset}, agents={args.agents}.\n"
        f"  CSV: {recorder.country_csv.resolve()}\n"
        f"       {recorder.resource_csv.resolve()}"
    )
    if args.parquet:
        print(f"  Parquet: {out_dir.resolve()}/*.parquet")

    if dashboard_url is not None:
        print(f"\nDashboard still live at {dashboard_url} — Ctrl+C to exit.")
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
