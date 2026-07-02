"""Tunable simulation constants as a dataclass.

Instead of scattering 20+ magic numbers across economy.py, collect them in one
place so users can override them without editing source.  The defaults exactly
match the original hardcoded values for backward compatibility.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SimConfig:
    """All tuneable simulation parameters in one place.

    Every field's default matches the original hardcoded constant so existing
    behaviour is preserved when constructing ``SimConfig()`` with no arguments.
    """

    # ── price formation ────────────────────────────────────────────────────
    price_elasticity: float = 0.08
    price_floor: float = 0.20
    price_ceil: float = 50.0
    price_step_clamp: float = 0.20

    # ── production throttling ──────────────────────────────────────────────
    throttle_target_months: float = 6.0
    throttle_max_months: float = 18.0
    min_production_util: float = 0.10
    high_price_override_threshold: float = 2.0
    low_price_throttle_end: float = 0.60
    low_price_min_util: float = 0.02

    # ── welfare / tech / population ────────────────────────────────────────
    welfare_ema_decay: float = 0.3
    tech_monthly_growth: float = 0.0008
    tech_max: float = 5.0
    pop_alpha: float = 0.7
    pop_peak_welfare: float = 0.60
    pop_peak_annual_rate: float = 0.010
    pop_high_welfare_drag: float = 0.020
    pop_low_welfare_drag: float = 0.080
    pop_low_welfare_threshold: float = 0.45

    # ── finance ────────────────────────────────────────────────────────────
    production_cost_rate: float = 0.005
    investment_intensity: float = 0.005
    capital_cost_per_unit: float = 800.0
    safe_reserve_months: float = 12.0
    capacity_decay_rate: float = 0.0
    debt_interest_rate: float = 0.004
    debt_paydown_rate: float = 0.05

    # ── migration ──────────────────────────────────────────────────────────
    class_mobility: np.ndarray = field(
        default_factory=lambda: np.array([0.0003, 0.0008, 0.0015])
    )
    migration_pull_scale: float = 0.40
    max_migration_fraction_per_tick: float = 0.02

    # ── performance index ──────────────────────────────────────────────────
    performance_growth_gain: float = 0.003
    performance_growth_clamp: float = 0.005
    performance_drift: float = 0.0001
    performance_noise_std: float = 0.0010
    performance_index_floor: float = 5.0
    performance_index_ceil: float = 100_000.0
