"""Fixed-income asset universe: pricing, duration and convexity.

Bonds pay an annual coupon ``c * F`` at ``t = 1 ... N`` and the face value
``F`` at maturity ``N``.  With a flat effective annual yield ``y`` and
``v = 1/(1+y)``::

    P(y) = F c * (1 - v^N) / y  +  F v^N                       (closed form)
         = sum_t CF_t v^t

    D_A  = (1/P) * sum_t t CF_t v^t                            (Macaulay)
    C_A  = (1/P) * sum_t t (t+1) CF_t v^(t+2)                  (= P'' / P)

A :class:`Portfolio` holds market-value weights ``w_i`` (``sum w_i = 1``).
Because value is additive, the portfolio's duration and convexity are the
*value-weighted* averages of the bonds' (for a flat yield curve)::

    D_P = sum_i w_i D_i          C_P = sum_i w_i C_i

which makes the Redington constraints *linear* in the weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from .liabilities import CashFlowStream, FloatArray

_ZERO_YIELD_TOL = 1e-10


# --------------------------------------------------------------------------- #
# Single bond
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Bond:
    """Annual-pay, fixed-coupon, bullet bond.

    Parameters
    ----------
    name:
        Display label.
    maturity:
        Integer maturity ``N`` in years.
    coupon_rate:
        Annual coupon rate ``c`` (decimal, e.g. ``0.04``).
    face_value:
        Redemption value ``F`` (default 100).
    """

    name: str
    maturity: int
    coupon_rate: float
    face_value: float = 100.0

    def __post_init__(self) -> None:
        if int(self.maturity) != self.maturity or self.maturity < 1:
            raise ValueError("maturity must be a positive integer number of years.")
        if self.coupon_rate < 0:
            raise ValueError("coupon_rate must be non-negative.")
        if self.face_value <= 0:
            raise ValueError("face_value must be strictly positive.")

    # -- cash flows --------------------------------------------------------- #
    def cash_flow_stream(self) -> CashFlowStream:
        """Cash flows per bond: coupons ``F c`` each year plus ``F`` at maturity."""
        times = np.arange(1, int(self.maturity) + 1, dtype=float)
        amounts = np.full_like(times, self.face_value * self.coupon_rate)
        amounts[-1] += self.face_value
        return CashFlowStream(times=times, amounts=amounts)

    # -- analytics ---------------------------------------------------------- #
    def price(self, y: float) -> float:
        r"""Closed-form price :math:`P = Fc\,\frac{1-v^N}{y} + F v^N`."""
        n, f, c = self.maturity, self.face_value, self.coupon_rate
        if abs(y) < _ZERO_YIELD_TOL:
            return float(f * (1.0 + c * n))
        v_n = (1.0 + y) ** (-n)
        return float(f * c * (1.0 - v_n) / y + f * v_n)

    def macaulay_duration(self, y: float) -> float:
        r"""Macaulay duration :math:`D_A = \frac{1}{P}\sum_t t\,CF_t v^t`."""
        return self.cash_flow_stream().macaulay_duration(y)

    def modified_duration(self, y: float) -> float:
        r""":math:`D_{mod} = D_A/(1+y)`."""
        return self.macaulay_duration(y) / (1.0 + y)

    def convexity(self, y: float) -> float:
        r"""Convexity :math:`C_A = P''/P = \frac{1}{P}\sum_t t(t+1)\,CF_t v^{t+2}`."""
        return self.cash_flow_stream().convexity(y)


def default_bond_universe(
    short_coupon: float = 0.030,
    mid_coupon: float = 0.040,
    long_coupon: float = 0.050,
) -> tuple[Bond, Bond, Bond]:
    """Three-instrument universe: 2-year, 10-year and 30-year bonds."""
    return (
        Bond("2Y Short", 2, short_coupon),
        Bond("10Y Intermediate", 10, mid_coupon),
        Bond("30Y Long", 30, long_coupon),
    )


def bond_analytics_table(bonds: tuple[Bond, ...], y: float) -> pd.DataFrame:
    """Tabulate price, durations and convexity of each bond at yield ``y``."""
    rows = [
        {
            "Bond": b.name,
            "Maturity (y)": b.maturity,
            "Coupon": b.coupon_rate,
            "Price": b.price(y),
            "Macaulay D": b.macaulay_duration(y),
            "Modified D": b.modified_duration(y),
            "Convexity": b.convexity(y),
        }
        for b in bonds
    ]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Portfolio
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Portfolio:
    """Long-only bond portfolio specified by market-value weights.

    Attributes
    ----------
    bonds:
        Instruments held.
    weights:
        Market-value weights ``w_i >= 0`` at the base yield, ``sum w_i = 1``.
    total_value:
        Total market value ``V`` at the base yield ``base_yield``.
    base_yield:
        Flat yield at which the weights / holdings were struck.

    The holdings (number of bonds, face value 100 each) are
    ``n_i = w_i V / P_i(y_0)``, so that ``sum_i n_i P_i(y_0) = V``.
    """

    bonds: tuple[Bond, ...]
    weights: FloatArray
    total_value: float
    base_yield: float
    units: FloatArray = field(init=False)

    def __post_init__(self) -> None:
        w = np.asarray(self.weights, dtype=float)
        if w.shape != (len(self.bonds),):
            raise ValueError("weights must have one entry per bond.")
        if np.any(w < -1e-9):
            raise ValueError("Short positions are not allowed (weights must be >= 0).")
        if not np.isclose(w.sum(), 1.0, atol=1e-6):
            raise ValueError("weights must sum to one.")
        if self.total_value <= 0:
            raise ValueError("total_value must be strictly positive.")
        w = np.clip(w, 0.0, None)
        prices = np.array([b.price(self.base_yield) for b in self.bonds])
        object.__setattr__(self, "weights", w)
        object.__setattr__(self, "units", w * self.total_value / prices)

    # -- cash flows --------------------------------------------------------- #
    def cash_flow_matrix(self, horizon: int | None = None) -> FloatArray:
        """Per-bond cash flows, shape ``(T, n_bonds)``, on grid ``1 ... T``."""
        t_max = max(int(horizon or 0), max(b.maturity for b in self.bonds))
        grid = np.arange(1, t_max + 1, dtype=float)
        cols = [b.cash_flow_stream().on_grid(grid).amounts * u for b, u in zip(self.bonds, self.units)]
        return np.column_stack(cols)

    def cash_flow_stream(self, horizon: int | None = None) -> CashFlowStream:
        """Aggregate portfolio cash-flow stream on an annual grid."""
        mat = self.cash_flow_matrix(horizon)
        grid = np.arange(1, mat.shape[0] + 1, dtype=float)
        return CashFlowStream(times=grid, amounts=mat.sum(axis=1))

    # -- analytics ---------------------------------------------------------- #
    def value(self, rates: ArrayLike) -> float | FloatArray:
        r"""Market value :math:`PV_A(r) = \sum_i n_i P_i(r)` (vectorised)."""
        return self.cash_flow_stream().present_value(rates)

    def macaulay_duration(self, y: float | None = None) -> float:
        r"""Portfolio Macaulay duration :math:`D_P = \sum_i w_i D_i`."""
        return self.cash_flow_stream().macaulay_duration(self.base_yield if y is None else y)

    def convexity(self, y: float | None = None) -> float:
        r"""Portfolio convexity :math:`C_P = \sum_i w_i C_i`."""
        return self.cash_flow_stream().convexity(self.base_yield if y is None else y)

    def allocation_table(self) -> pd.DataFrame:
        """Holdings table (weights, units, market values, bond analytics)."""
        y = self.base_yield
        prices = np.array([b.price(y) for b in self.bonds])
        return pd.DataFrame(
            {
                "Bond": [b.name for b in self.bonds],
                "Weight": self.weights,
                "Units (face 100)": self.units,
                "Market Value": self.units * prices,
                "Price": prices,
                "Macaulay D": [b.macaulay_duration(y) for b in self.bonds],
                "Convexity": [b.convexity(y) for b in self.bonds],
            }
        )


def build_portfolio(
    bonds: tuple[Bond, ...], weights: ArrayLike, total_value: float, base_yield: float
) -> Portfolio:
    """Create a :class:`Portfolio` (thin factory with a friendlier signature)."""
    return Portfolio(
        bonds=tuple(bonds),
        weights=np.asarray(weights, dtype=float),
        total_value=float(total_value),
        base_yield=float(base_yield),
    )


def equal_weight_portfolio(
    bonds: tuple[Bond, ...], total_value: float, base_yield: float
) -> Portfolio:
    """Naive benchmark: ``w_i = 1/n`` with no duration/convexity targeting."""
    n = len(bonds)
    return build_portfolio(bonds, np.full(n, 1.0 / n), total_value, base_yield)
