"""Monte Carlo interest-rate simulation and surplus risk metrics.

Interest-rate shock models
--------------------------
Both models generate a *level* (parallel) shift ``dy_k`` of the flat yield curve
for scenario ``k = 1 ... N`` over a risk horizon ``H``:

1. **Gaussian** shift::

       dy ~ N(mu, sigma^2)

2. **Vasicek** (1-factor, mean-reverting short rate)::

       dr_t = kappa (theta - r_t) dt + sigma dW_t

   The transition density is known exactly (Ornstein-Uhlenbeck), so no time
   discretisation error is incurred by jumping straight to the horizon::

       r_H | r_0 ~ N( theta + (r_0 - theta) e^{-kappa H},
                      sigma^2 (1 - e^{-2 kappa H}) / (2 kappa) )

   and ``dy = r_H - r_0`` (the short rate is identified with the flat yield).

Optionally an independent **slope (twist)** factor ``s ~ N(0, sigma_s^2)`` is
added, giving the scenario spot curve::

       y_k(t) = y_0 + dy_k + s_k * (t - t_pivot) / 10

which pivots around ``t_pivot`` (default: the liability duration) and moves a
cash flow at maturity ``t`` by ``s_k`` basis points per ten years.  Parallel-only
shocks (``sigma_s = 0``) are exactly what Redington immunization protects
against; the twist term shows its limitation.

Risk metrics
------------
Surplus in scenario ``k`` is ``S_k = PV_A(y_k) - PV_L(y_k)``.  Risk is measured
on the *loss* relative to the initial surplus ``S_0``::

       Loss_k = S_0 - S_k

       VaR_a   = inf{ x : P(Loss <= x) >= a }          (empirical a-quantile)
       TVaR_a  = E[ Loss | Loss >= VaR_a ]              (conditional tail expectation)
       P(ruin) = P(S_k < 0)

A *negative* VaR/TVaR means that even in the tail the surplus is higher than
today, which is what a well-immunized (convex) portfolio delivers under
parallel shifts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .liabilities import CashFlowStream, FloatArray

ShockModel = Literal["gaussian", "vasicek"]

#: Lower clamp on simulated spot rates to keep ``(1 + r)^{-t}`` well defined.
RATE_FLOOR: float = -0.50
#: Relative tolerance (fraction of PV_L) below which a negative surplus is treated as zero.
INSOLVENCY_TOL_REL: float = 1e-9


# --------------------------------------------------------------------------- #
# Configuration + scenario generation
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ShockConfig:
    """Parameters of the interest-rate shock generator (all rates are decimals).

    Attributes
    ----------
    model:
        ``'gaussian'`` or ``'vasicek'``.
    n_sims:
        Number of Monte Carlo scenarios ``N``.
    mu:
        Mean parallel shift (Gaussian model only).
    sigma:
        Gaussian: standard deviation of the shift.  Vasicek: *instantaneous*
        volatility of the short rate (the horizon std is smaller).
    kappa, theta_offset:
        Vasicek speed of mean reversion and long-run mean expressed as an
        offset from the base yield ``theta = y_0 + theta_offset``.
    horizon:
        Risk horizon ``H`` in years (Vasicek model only).
    slope_sigma:
        Std-dev of the slope shock in rate per 10 years of maturity (0 disables).
    seed:
        RNG seed for reproducibility (``None`` for entropy).
    """

    model: ShockModel = "gaussian"
    n_sims: int = 2500
    mu: float = 0.0
    sigma: float = 0.01
    kappa: float = 0.25
    theta_offset: float = 0.0
    horizon: float = 1.0
    slope_sigma: float = 0.0
    seed: int | None = 42

    def __post_init__(self) -> None:
        if self.model not in ("gaussian", "vasicek"):
            raise ValueError("model must be 'gaussian' or 'vasicek'.")
        if self.n_sims < 10:
            raise ValueError("n_sims must be at least 10.")
        if self.sigma < 0 or self.slope_sigma < 0:
            raise ValueError("Volatilities must be non-negative.")
        if self.kappa < 0:
            raise ValueError("kappa must be non-negative.")
        if self.horizon <= 0:
            raise ValueError("horizon must be strictly positive.")


@dataclass(frozen=True)
class ShockScenarios:
    """Simulated shocks: parallel level shift and slope factor per scenario."""

    level: FloatArray
    slope: FloatArray


def vasicek_moments(
    r0: float, kappa: float, theta: float, sigma: float, horizon: float
) -> tuple[float, float]:
    r"""Exact mean and std-dev of the Vasicek short rate at ``horizon``.

    .. math::
        E[r_H] = \theta + (r_0 - \theta)e^{-\kappa H}, \qquad
        Var[r_H] = \frac{\sigma^2}{2\kappa}\left(1 - e^{-2\kappa H}\right)

    For ``kappa -> 0`` the variance tends to ``sigma^2 H`` (Brownian motion).
    """
    if kappa < 1e-10:
        return r0, float(sigma * np.sqrt(horizon))
    decay = np.exp(-kappa * horizon)
    mean = theta + (r0 - theta) * decay
    var = sigma**2 * (1.0 - np.exp(-2.0 * kappa * horizon)) / (2.0 * kappa)
    return float(mean), float(np.sqrt(var))


def effective_shock_std(config: ShockConfig, base_yield: float) -> float:
    """Standard deviation of the parallel shift ``dy`` implied by ``config``."""
    if config.model == "gaussian":
        return config.sigma
    _, std = vasicek_moments(
        base_yield, config.kappa, base_yield + config.theta_offset, config.sigma, config.horizon
    )
    return std


def simulate_shocks(config: ShockConfig, base_yield: float) -> ShockScenarios:
    """Draw ``config.n_sims`` yield-curve shocks according to ``config``."""
    rng = np.random.default_rng(config.seed)
    z_level = rng.standard_normal(config.n_sims)
    z_slope = rng.standard_normal(config.n_sims)

    if config.model == "gaussian":
        level = config.mu + config.sigma * z_level
    else:
        mean_r, std_r = vasicek_moments(
            base_yield, config.kappa, base_yield + config.theta_offset, config.sigma, config.horizon
        )
        level = (mean_r + std_r * z_level) - base_yield  # dy = r_H - r_0

    return ShockScenarios(level=level, slope=config.slope_sigma * z_slope)


def scenario_rate_matrix(
    times: FloatArray,
    base_yield: float,
    shocks: ShockScenarios,
    slope_pivot: float,
) -> FloatArray:
    r"""Spot-rate matrix ``(T, N)``: :math:`y_k(t) = y_0 + dy_k + s_k (t - t^*)/10`.

    Rates are floored at :data:`RATE_FLOOR`.
    """
    t = np.asarray(times, dtype=float)[:, None]
    rates = base_yield + shocks.level[None, :] + shocks.slope[None, :] * (t - slope_pivot) / 10.0
    return np.maximum(rates, RATE_FLOOR)


# --------------------------------------------------------------------------- #
# Risk metrics
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RiskMetrics:
    """Statistical summary of the simulated surplus distribution."""

    initial_surplus: float
    mean: float
    std: float
    minimum: float
    maximum: float
    var_95: float  # loss-based
    var_99: float
    tvar_95: float
    tvar_99: float
    prob_insolvency: float

    @property
    def surplus_at_var_95(self) -> float:
        """Surplus level corresponding to the 95 % VaR (``S_0 - VaR_95``)."""
        return self.initial_surplus - self.var_95

    @property
    def surplus_at_tvar_95(self) -> float:
        """Average surplus in the worst 5 % of scenarios (``S_0 - TVaR_95``)."""
        return self.initial_surplus - self.tvar_95

    @property
    def surplus_at_var_99(self) -> float:
        return self.initial_surplus - self.var_99


def value_at_risk(loss: FloatArray, level: float) -> float:
    r""":math:`VaR_a` = empirical ``a``-quantile of the loss distribution."""
    if not 0.0 < level < 1.0:
        raise ValueError("level must lie in (0, 1).")
    return float(np.quantile(loss, level, method="linear"))


def tail_value_at_risk(loss: FloatArray, level: float) -> float:
    r""":math:`TVaR_a = E[\,Loss \mid Loss \ge VaR_a\,]` (conditional tail expectation)."""
    var = value_at_risk(loss, level)
    tail = loss[loss >= var]
    return float(tail.mean())  # tail is never empty: it contains at least the max


def compute_risk_metrics(
    surplus: FloatArray, initial_surplus: float, pv_liabilities: float
) -> RiskMetrics:
    """Compute mean / std / VaR / TVaR / ruin probability of ``surplus``.

    Parameters
    ----------
    surplus:
        Simulated surplus ``S_k``, shape ``(N,)``.
    initial_surplus:
        Surplus ``S_0`` before any shock (reference for loss).
    pv_liabilities:
        Used to set the numerical tolerance for the insolvency test.
    """
    s = np.asarray(surplus, dtype=float)
    if s.ndim != 1 or s.size == 0:
        raise ValueError("surplus must be a non-empty 1-D array.")
    loss = initial_surplus - s
    tol = INSOLVENCY_TOL_REL * pv_liabilities
    return RiskMetrics(
        initial_surplus=float(initial_surplus),
        mean=float(s.mean()),
        std=float(s.std(ddof=1)),
        minimum=float(s.min()),
        maximum=float(s.max()),
        var_95=value_at_risk(loss, 0.95),
        var_99=value_at_risk(loss, 0.99),
        tvar_95=tail_value_at_risk(loss, 0.95),
        tvar_99=tail_value_at_risk(loss, 0.99),
        prob_insolvency=float(np.mean(s < -tol)),
    )


# --------------------------------------------------------------------------- #
# Monte Carlo driver
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SimulationResult:
    """Scenario-level output of :func:`run_monte_carlo`."""

    level_shifts: FloatArray
    slope_shifts: FloatArray
    asset_values: FloatArray
    liability_values: FloatArray
    surplus: FloatArray
    metrics: RiskMetrics


def run_monte_carlo(
    liabilities: CashFlowStream,
    assets: CashFlowStream,
    base_yield: float,
    config: ShockConfig,
    slope_pivot: float | None = None,
) -> SimulationResult:
    r"""Re-price assets and liabilities in every shocked scenario.

    .. math::
        S_k = PV_A(y_k) - PV_L(y_k), \qquad k = 1,\dots,N

    Parameters
    ----------
    liabilities, assets:
        Cash-flow streams (the asset stream is the aggregated portfolio flow).
    base_yield:
        Current flat yield ``y_0``.
    config:
        Shock generator settings.
    slope_pivot:
        Maturity (years) that is unaffected by the slope factor; defaults to
        the liability Macaulay duration.
    """
    pivot = liabilities.macaulay_duration(base_yield) if slope_pivot is None else slope_pivot
    shocks = simulate_shocks(config, base_yield)

    grid = np.arange(1, int(max(liabilities.times.max(), assets.times.max())) + 1, dtype=float)
    a = assets.on_grid(grid)
    l = liabilities.on_grid(grid)

    rates = scenario_rate_matrix(grid, base_yield, shocks, pivot)
    pv_a = np.asarray(a.present_value(rates))
    pv_l = np.asarray(l.present_value(rates))
    surplus = pv_a - pv_l

    s0 = float(a.present_value(base_yield) - l.present_value(base_yield))
    metrics = compute_risk_metrics(surplus, s0, float(l.present_value(base_yield)))
    return SimulationResult(
        level_shifts=shocks.level,
        slope_shifts=shocks.slope,
        asset_values=pv_a,
        liability_values=pv_l,
        surplus=surplus,
        metrics=metrics,
    )


# --------------------------------------------------------------------------- #
# Deterministic price-yield sensitivity
# --------------------------------------------------------------------------- #
def price_yield_curve(
    liabilities: CashFlowStream,
    assets: CashFlowStream,
    base_yield: float,
    bp_range: float = 300.0,
    n_points: int = 121,
) -> pd.DataFrame:
    """Value of assets / liabilities / surplus over ``y_0 +/- bp_range`` bps.

    Under Redington immunization the asset and liability curves are tangent at
    ``y_0`` (equal slope) and the asset curve lies above (greater curvature),
    so the surplus column is non-negative everywhere near ``y_0``.
    """
    grid = np.arange(1, int(max(liabilities.times.max(), assets.times.max())) + 1, dtype=float)
    a = assets.on_grid(grid)
    l = liabilities.on_grid(grid)

    shifts_bp = np.linspace(-bp_range, bp_range, n_points)
    yields = np.maximum(base_yield + shifts_bp / 1e4, RATE_FLOOR)
    pv_a = np.asarray(a.present_value(yields))
    pv_l = np.asarray(l.present_value(yields))
    return pd.DataFrame(
        {
            "shift_bp": shifts_bp,
            "yield": yields,
            "assets": pv_a,
            "liabilities": pv_l,
            "surplus": pv_a - pv_l,
        }
    )
