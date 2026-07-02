# Agent policies init
from .autonomous import (
    BalancedAgent,
    GrowthAgent,
    MercantilistAgent,
    WelfareAgent,
)
from .base import Policy, batch_act
from .scripted import (
    AutarkicPolicy,
    BufferStockPolicy,
    MercantilistPolicy,
    PriceAwarePolicy,
)

# Canonical registry: maps CLI name → policy class.
POLICY_REGISTRY: dict[str, type[Policy]] = {
    "autarkic":        AutarkicPolicy,
    "buffer":          BufferStockPolicy,
    "mercantilist":    MercantilistPolicy,
    "price_aware":     PriceAwarePolicy,
    "growth":          GrowthAgent,
    "mercantilist_ag": MercantilistAgent,
    "welfare":         WelfareAgent,
    "balanced":        BalancedAgent,
}

# Cycling order for mixed autonomous simulation.
AUTONOMOUS_CYCLE: list[type[Policy]] = [
    GrowthAgent,
    MercantilistAgent,
    WelfareAgent,
    BalancedAgent,
]

__all__ = [
    "Policy",
    "batch_act",
    "AutarkicPolicy",
    "BufferStockPolicy",
    "MercantilistPolicy",
    "PriceAwarePolicy",
    "BalancedAgent",
    "GrowthAgent",
    "MercantilistAgent",
    "WelfareAgent",
    "POLICY_REGISTRY",
    "AUTONOMOUS_CYCLE",
]
