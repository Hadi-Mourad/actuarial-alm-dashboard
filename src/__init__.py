"""Stochastic Asset-Liability Management (ALM) & Portfolio Immunization toolkit."""

from .assets import Bond, Portfolio, default_bond_universe
from .immunization import (
    OBJECTIVES,
    ImmunizationError,
    ImmunizationResult,
    InfeasibleImmunizationError,
    immunize,
)
from .liabilities import CashFlowStream, generate_liabilities, summarize_liabilities
from .simulation import ShockConfig, run_monte_carlo

__all__ = [
    "Bond",
    "CashFlowStream",
    "ImmunizationError",
    "ImmunizationResult",
    "InfeasibleImmunizationError",
    "OBJECTIVES",
    "Portfolio",
    "ShockConfig",
    "default_bond_universe",
    "generate_liabilities",
    "immunize",
    "run_monte_carlo",
    "summarize_liabilities",
]
