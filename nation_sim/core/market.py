from __future__ import annotations

import numpy as np

from .resources import IS_TRADEABLE, N_RESOURCES
from .world import WorldState, debt_capacity


def _spendable(world: WorldState) -> np.ndarray:
    """reserves + remaining debt headroom — total budget for buyers."""
    headroom = (debt_capacity(world) - world.debt).clip(min=0.0)
    return world.reserves + headroom


def clear_market(
    world: WorldState,
    buy: np.ndarray,
    sell: np.ndarray,
    limit: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """resolve the central exchange for one tick.

    mutates stockpiles, reserves, and debt.
    returns:
      trade_value : (N, R) net export value (positive = net exporter)
      shipments   : (N, N, R) bilateral shipment quantities
    """
    N = world.n_countries
    R = N_RESOURCES
    price = world.price
    tariffs = world.tariffs
    trade_cost_pair = world.trade_cost

    # clip to what we actually have / can afford
    sell = np.minimum(sell, world.stockpile).clip(min=0.0)
    spendable = _spendable(world)
    max_afford = spendable[:, None] / price[None, :].clip(min=1e-6)
    buy = np.where(buy > max_afford, max_afford, buy).clip(min=0.0)

    # apply limit prices (ceiling for buyers, floor for sellers)
    has_limit = ~np.isnan(limit)
    if has_limit.any():
        too_expensive = has_limit & (price[None, :] > limit)
        too_cheap = has_limit & (price[None, :] < limit)
        buy = np.where(too_expensive, 0.0, buy)
        sell = np.where(too_cheap, 0.0, sell)

    trade_value = np.zeros((N, R), dtype=np.float64)
    shipments = np.zeros((N, N, R), dtype=np.float64)

    total_buy = buy.sum(axis=0)
    total_sell = sell.sum(axis=0)

    # only trade resources with both buyers and sellers
    active = IS_TRADEABLE & (total_buy > 1e-12) & (total_sell > 1e-12)
    active_idx = np.where(active)[0]

    if active_idx.size:
        matched = np.minimum(total_buy[active_idx], total_sell[active_idx])

        # proportional matching
        buy_share = buy[:, np.newaxis, active_idx] / total_buy[np.newaxis, np.newaxis, active_idx]
        sell_share = sell[np.newaxis, :, active_idx] / total_sell[np.newaxis, np.newaxis, active_idx]
        active_shipments = buy_share * sell_share * matched[np.newaxis, np.newaxis, :]
        shipments[:, :, active_idx] = active_shipments

        # tariffs + trade costs (importer pays both)
        tariff_active = tariffs[:, :, active_idx]
        gross = active_shipments * price[np.newaxis, np.newaxis, active_idx]
        tariff_rev = gross * tariff_active
        trade_cost_paid = gross * trade_cost_pair[:, :, np.newaxis]
        paid_by_importer = gross + tariff_rev + trade_cost_paid

        # stockpile transfers
        world.stockpile[:, active_idx] += active_shipments.sum(axis=1)
        world.stockpile[:, active_idx] -= active_shipments.sum(axis=0)

        # reserve flows
        importer_cost = paid_by_importer.sum(axis=1)
        tariff_collected = tariff_rev.sum(axis=1)
        exporter_revenue = gross.sum(axis=0)

        world.reserves -= importer_cost.sum(axis=1)
        world.reserves += exporter_revenue.sum(axis=1)
        world.reserves += tariff_collected.sum(axis=1)

        trade_value[:, active_idx] = exporter_revenue - (importer_cost - tariff_collected)

    # convert cash deficit into sovereign debt
    deficit = (-world.reserves).clip(min=0.0)
    headroom = (debt_capacity(world) - world.debt).clip(min=0.0)
    new_debt = np.minimum(deficit, headroom)
    world.debt += new_debt
    world.reserves += new_debt

    return trade_value, shipments
