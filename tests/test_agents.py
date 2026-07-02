"""
Tests that all 8 agent policies run without errors.
"""
from __future__ import annotations

import numpy as np
import pytest

from nation_sim.agents.autonomous import (
    BalancedAgent,
    GrowthAgent,
    MercantilistAgent,
    WelfareAgent,
)
from nation_sim.agents.scripted import (
    AutarkicPolicy,
    BufferStockPolicy,
    MercantilistPolicy,
    PriceAwarePolicy,
)
from nation_sim.core.resources import N_RESOURCES
from nation_sim.sim.config import WorldConfig, generate_world
from nation_sim.sim.engine import Engine, EngineConfig


_ALL_POLICIES = [
    ("AutarkicPolicy", AutarkicPolicy),
    ("BufferStockPolicy", BufferStockPolicy),
    ("MercantilistPolicy", MercantilistPolicy),
    ("PriceAwarePolicy", PriceAwarePolicy),
    ("GrowthAgent", GrowthAgent),
    ("MercantilistAgent", MercantilistAgent),
    ("WelfareAgent", WelfareAgent),
    ("BalancedAgent", BalancedAgent),
]


@pytest.fixture
def small_world():
    return generate_world(WorldConfig(n_countries=4, seed=99))


@pytest.mark.parametrize("name,policy_cls", _ALL_POLICIES, ids=[p[0] for p in _ALL_POLICIES])
def test_agent_runs_without_error(small_world, name, policy_cls):
    """Agent can run 20 ticks without raising."""
    policies = [policy_cls() for _ in range(small_world.n_countries)]
    engine = Engine(small_world, policies, EngineConfig(seed=42))
    engine.run(20)
    assert small_world.tick == 20


@pytest.mark.parametrize("name,policy_cls", _ALL_POLICIES, ids=[p[0] for p in _ALL_POLICIES])
def test_welfare_in_bounds(small_world, name, policy_cls):
    """Welfare stays in [0, 1] after simulation."""
    policies = [policy_cls() for _ in range(small_world.n_countries)]
    engine = Engine(small_world, policies, EngineConfig(seed=42))
    engine.run(20)
    assert small_world.welfare.min() >= 0.0
    assert small_world.welfare.max() <= 1.0


@pytest.mark.parametrize("name,policy_cls", _ALL_POLICIES, ids=[p[0] for p in _ALL_POLICIES])
def test_stockpile_non_negative(small_world, name, policy_cls):
    """Stockpiles never go negative."""
    policies = [policy_cls() for _ in range(small_world.n_countries)]
    engine = Engine(small_world, policies, EngineConfig(seed=42))
    engine.run(20)
    assert (small_world.stockpile >= 0.0).all()


@pytest.mark.parametrize("name,policy_cls", _ALL_POLICIES, ids=[p[0] for p in _ALL_POLICIES])
def test_population_non_negative(small_world, name, policy_cls):
    """Population never goes negative."""
    policies = [policy_cls() for _ in range(small_world.n_countries)]
    engine = Engine(small_world, policies, EngineConfig(seed=42))
    engine.run(20)
    assert (small_world.population >= 0.0).all()


@pytest.mark.parametrize("name,policy_cls", _ALL_POLICIES, ids=[p[0] for p in _ALL_POLICIES])
def test_shapes_preserved(small_world, name, policy_cls):
    """WorldState shape invariants hold after simulation."""
    policies = [policy_cls() for _ in range(small_world.n_countries)]
    engine = Engine(small_world, policies, EngineConfig(seed=42))
    engine.run(20)
    small_world.check()
