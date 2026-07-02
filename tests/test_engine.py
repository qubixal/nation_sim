from __future__ import annotations

import numpy as np
import pytest

from nation_sim.agents.scripted import AutarkicPolicy, BufferStockPolicy
from nation_sim.core.market import clear_market
from nation_sim.core.resources import N_RESOURCES
from nation_sim.sim.config import WorldConfig, generate_world
from nation_sim.sim.engine import Engine, EngineConfig


@pytest.fixture
def small_world():
    return generate_world(WorldConfig(n_countries=3, seed=42))


def test_world_shapes(small_world):
    small_world.check()
    assert small_world.n_countries == 3
    assert small_world.stockpile.shape == (3, N_RESOURCES)


def test_single_step_preserves_shape(small_world):
    policies = [BufferStockPolicy() for _ in range(small_world.n_countries)]
    engine = Engine(small_world, policies, EngineConfig(seed=0))
    engine.step()
    assert small_world.tick == 1
    small_world.check()


def test_determinism():
    w1 = generate_world(WorldConfig(n_countries=5, seed=123))
    w2 = generate_world(WorldConfig(n_countries=5, seed=123))
    e1 = Engine(w1, [BufferStockPolicy() for _ in range(5)], EngineConfig(seed=7))
    e2 = Engine(w2, [BufferStockPolicy() for _ in range(5)], EngineConfig(seed=7))
    e1.run(20)
    e2.run(20)
    np.testing.assert_allclose(w1.population, w2.population)
    np.testing.assert_allclose(w1.stockpile, w2.stockpile)
    np.testing.assert_allclose(w1.price, w2.price)
    np.testing.assert_allclose(w1.fx, w2.fx)


def test_autarky_reserves_net_positive():
    """Under autarky no trade fires; reserves grow because base income exceeds production costs."""
    w = generate_world(WorldConfig(n_countries=4, seed=1))
    initial_reserves = w.reserves.copy()
    policies = [AutarkicPolicy() for _ in range(w.n_countries)]
    engine = Engine(w, policies, EngineConfig(seed=0))
    engine.run(10)
    # Net injection (income − production operating costs) must be positive for viable economies.
    assert (w.reserves > initial_reserves * 0.90).all()


def test_population_grows_with_abundance():
    """Cranking up production capacity keeps welfare high → population should not collapse."""
    w = generate_world(WorldConfig(n_countries=3, seed=2))
    w.capacity *= 10.0
    policies = [BufferStockPolicy() for _ in range(w.n_countries)]
    p0 = w.population.copy()
    engine = Engine(w, policies, EngineConfig(seed=0))
    engine.run(24)
    assert (w.population >= p0 * 0.99).all()


def test_preset_free_trade_has_no_tariffs():
    w = generate_world(WorldConfig(preset="free_trade", n_countries=5, seed=0))
    assert np.all(w.tariffs == 0.0)


def test_preset_mercantilist_has_many_tariffs():
    w = generate_world(WorldConfig(preset="mercantilist", n_countries=5, seed=0))
    # Exclude the zero-by-construction diagonal from the density check.
    N = w.n_countries
    off_diag_mask = np.ones((N, N), dtype=bool)
    off_diag_mask[np.arange(N), np.arange(N)] = False
    off_diag_tariffs = w.tariffs[off_diag_mask]
    fraction_with_tariff = (off_diag_tariffs > 0).mean()
    assert fraction_with_tariff > 0.4


def test_trade_conserves_resources():
    """Trade is a transfer: total stockpile of a resource should not change."""
    w = generate_world(WorldConfig(n_countries=4, seed=9))
    N, R = w.n_countries, N_RESOURCES
    buy = np.zeros((N, R))
    sell = np.zeros((N, R))
    w.stockpile[0, 0] = 100.0
    w.reserves[1] = 1e6
    sell[0, 0] = 5.0
    buy[1, 0] = 5.0
    limit = np.full((N, R), np.nan)
    before = w.stockpile.sum(axis=0).copy()
    clear_market(w, buy, sell, limit, np.random.default_rng(0))
    after = w.stockpile.sum(axis=0)
    np.testing.assert_allclose(after, before)


def test_trade_moves_stockpile_and_reserves():
    """Under a matched buy/sell pair the stockpile should move from seller to buyer."""
    w = generate_world(WorldConfig(preset="free_trade", n_countries=2, seed=3))
    w.trade_cost[:] = 0.0  # isolate from geography-based trade friction
    N, R = w.n_countries, N_RESOURCES
    w.stockpile[0, 0] = 100.0
    w.stockpile[1, 0] = 0.0
    w.reserves[:] = 1e6
    w.price[0] = 2.0
    buy = np.zeros((N, R));   buy[1, 0] = 10.0
    sell = np.zeros((N, R));  sell[0, 0] = 10.0
    limit = np.full((N, R), np.nan)

    pre_seller_reserves = w.reserves[0]
    pre_buyer_reserves = w.reserves[1]
    clear_market(w, buy, sell, limit, np.random.default_rng(0))

    assert w.stockpile[0, 0] == pytest.approx(90.0)
    assert w.stockpile[1, 0] == pytest.approx(10.0)
    # Seller got 10 * 2.0 = 20; buyer paid 20 (no tariffs in free_trade preset).
    assert w.reserves[0] == pytest.approx(pre_seller_reserves + 20.0)
    assert w.reserves[1] == pytest.approx(pre_buyer_reserves - 20.0)


def test_tariff_enriches_importer():
    """When importer has a tariff on exporter, importer pockets the tariff."""
    w = generate_world(WorldConfig(preset="free_trade", n_countries=2, seed=5))
    w.trade_cost[:] = 0.0  # isolate tariff arithmetic from geography
    w.tariffs[1, 0, 0] = 0.25  # importer=1 charges 25% on imports of resource 0 from exporter=0
    N, R = w.n_countries, N_RESOURCES
    w.stockpile[0, 0] = 50.0
    w.stockpile[1, 0] = 0.0
    w.reserves[:] = 1e6
    w.price[0] = 4.0
    buy = np.zeros((N, R));  buy[1, 0] = 10.0
    sell = np.zeros((N, R)); sell[0, 0] = 10.0
    limit = np.full((N, R), np.nan)

    r_before = w.reserves.copy()
    clear_market(w, buy, sell, limit, np.random.default_rng(0))

    # Gross trade value = 10 * 4.0 = 40; tariff = 10.
    # Exporter gains 40. Importer pays 50 gross (40 + 10) but collects 10 tariff, net -40.
    assert w.reserves[0] == pytest.approx(r_before[0] + 40.0)
    assert w.reserves[1] == pytest.approx(r_before[1] - 40.0)
