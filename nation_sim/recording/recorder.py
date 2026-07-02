"""recorder — CSV/Parquet output + in-memory ring buffers for the live dashboard.

two tables:
  country.csv  — one row per (tick, country)
  resource.csv — one row per (tick, country, resource)
"""
from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

from ..core.economy import class_fractions, compute_gdp
from ..core.resources import RESOURCES
from ..core.world import WorldState

# min shipment qty to show in the live trade feed
_TRADE_EVENT_THRESHOLD = 0.10


class Recorder:
    def __init__(
        self,
        out_dir: Path | str,
        flush_every: int = 30,
        keep_live_country_rows: int = 60_000,
        keep_live_resource_rows: int = 600_000,
        keep_live_trade_events: int = 500,
        write_csv: bool = True,
        write_parquet: bool = False,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.flush_every = flush_every
        self.write_csv = write_csv
        self.write_parquet = write_parquet

        self._country_buf: list[dict] = []
        self._resource_buf: list[dict] = []
        self._live_country: deque = deque(maxlen=keep_live_country_rows)
        self._live_resource: deque = deque(maxlen=keep_live_resource_rows)
        self._live_trades: deque = deque(maxlen=keep_live_trade_events)
        self._live_crashes: deque = deque(maxlen=20)
        self._flush_idx = 0
        # snapshot of slow-changing state for dashboard
        self.tariffs: np.ndarray | None = None
        self.country_names: list[str] = []

        self.country_csv = self.out_dir / "country.csv"
        self.resource_csv = self.out_dir / "resource.csv"

        # start fresh — don't mix with stale CSVs
        if self.write_csv:
            self.country_csv.unlink(missing_ok=True)
            self.resource_csv.unlink(missing_ok=True)

    def record(self, world: WorldState, shipments: np.ndarray | None = None) -> None:
        t = world.tick
        gdp = compute_gdp(world)
        cls = class_fractions(world.welfare)
        # stash for dashboard read access
        self.tariffs = world.tariffs
        self.country_names = list(world.country_names)
        for i, name in enumerate(world.country_names):
            row = {
                "tick": t,
                "country": name,
                "population": float(world.population[i]),
                "welfare": float(world.welfare[i]),
                "fx": float(world.fx[i]),
                "reserves": float(world.reserves[i]),
                "debt": float(world.debt[i]),
                "gdp": float(gdp[i]),
                "performance": float(world.performance[i]),
                "frac_poor": float(cls[i, 0]),
                "frac_middle": float(cls[i, 1]),
                "frac_rich": float(cls[i, 2]),
                "pos_x": float(world.position[i, 0]),
                "pos_y": float(world.position[i, 1]),
            }
            self._country_buf.append(row)
            self._live_country.append(row)
            for r, rname in enumerate(RESOURCES):
                rrow = {
                    "tick": t,
                    "country": name,
                    "resource": rname,
                    "stockpile": float(world.stockpile[i, r]),
                    "price": float(world.price[r]),
                }
                self._resource_buf.append(rrow)
                self._live_resource.append(rrow)

        if shipments is not None:
            self._record_trades(t, shipments, world)

        if t > 0 and t % self.flush_every == 0:
            self.flush()

    def _record_trades(
        self, tick: int, shipments: np.ndarray, world: WorldState
    ) -> None:
        """extract significant bilateral trades from the shipment matrix."""
        names = world.country_names
        price = world.price
        # only record big trades, largest first
        mask = shipments > _TRADE_EVENT_THRESHOLD
        idxs = np.argwhere(mask)
        if len(idxs) == 0:
            return
        volumes = shipments[mask]
        order = np.argsort(volumes)[::-1]
        for k in order:
            imp_i, exp_e, r = idxs[k]
            qty = float(shipments[imp_i, exp_e, r])
            self._live_trades.append(
                {
                    "tick": tick,
                    "buyer": names[imp_i],
                    "seller": names[exp_e],
                    "resource": RESOURCES[r],
                    "qty": round(qty, 2),
                    "price": round(float(price[r]), 3),
                    "value": round(qty * float(price[r]), 2),
                }
            )

    def flush(self) -> None:
        if self._country_buf:
            cdf = pd.DataFrame(self._country_buf)
            if self.write_csv:
                header = not self.country_csv.exists()
                cdf.to_csv(self.country_csv, mode="a", header=header, index=False)
            if self.write_parquet:
                cdf.to_parquet(
                    self.out_dir / f"country_{self._flush_idx:04d}.parquet",
                    index=False,
                )
            self._country_buf.clear()
        if self._resource_buf:
            rdf = pd.DataFrame(self._resource_buf)
            if self.write_csv:
                header = not self.resource_csv.exists()
                rdf.to_csv(self.resource_csv, mode="a", header=header, index=False)
            if self.write_parquet:
                rdf.to_parquet(
                    self.out_dir / f"resource_{self._flush_idx:04d}.parquet",
                    index=False,
                )
            self._resource_buf.clear()
        self._flush_idx += 1

    def close(self) -> None:
        self.flush()

    @property
    def live_country_rows(self) -> list[dict]:
        return list(self._live_country)

    @property
    def live_resource_rows(self) -> list[dict]:
        return list(self._live_resource)

    def record_crash(self, tick: int, name: str, msg: str) -> None:
        self._live_crashes.appendleft({"tick": tick, "name": name, "msg": msg})

    @property
    def live_trade_events(self) -> list[dict]:
        return list(self._live_trades)

    @property
    def live_crash_events(self) -> list[dict]:
        return list(self._live_crashes)
