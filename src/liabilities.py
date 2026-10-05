"""Liability cash-flow generation and valuation.

This module also hosts the generic :class:`CashFlowStream` primitive that the
rest of the code base (assets, immunization, simulation) builds on, so every
module shares exactly one implementation of discounting, duration and
convexity.

Financial mathematics (flat effective annual yield ``y``)
---------------------------------------------------------
For a stream of cash flows ``C_t`` paid at times ``t = t_1 ... t_n``::

    PV(y)  = sum_t C_t (1+y)^(-t)

    Macaulay duration   D(y) = (1/PV) * sum_t t * C_t (1+y)^(-t)
    Modified duration   D_mod = D / (1+y)
    Convexity           C(y) = (1/PV) * sum_t t (t+1) * C_t (1+y)^(-t-2)

Differentiating the present value with respect to the yield gives::

    dPV/dy    = -sum_t t C_t (1+y)^(-t-1)   = -PV * D / (1+y)
    d2PV/dy2  =  sum_t t(t+1) C_t (1+y)^(-t-2) =  PV * C

so ``D`` is the (negative, scaled) first derivative of value and ``C`` is the
second derivative of value per unit of value.  These are exactly the two
quantities that Redington immunization needs to match / dominate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]

#: Yields at or below this level make ``(1 + y)`` non-positive and are rejected.
MIN_ALLOWED_RATE: float = -0.999


# --------------------------------------------------------------------------- #
# Low-level discounting helpers
# --------------------------------------------------------------------------- #
def discount_factors(times: ArrayLike, rates: ArrayLike) -> FloatArray:
    r"""Return discount factors :math:`v_t = (1 + r)^{-t}`.

    Parameters
    ----------
    times:
        1-D array of cash-flow times in years, shape ``(T,)``.
    rates:
        Effective annual rate(s).  Accepted shapes:

        * scalar          -> returns shape ``(T,)`` (flat curve)
        * ``(N,)``        -> returns shape ``(T, N)`` (N parallel scenarios)
        * ``(T, N)``      -> returns shape ``(T, N)`` (scenario-specific spot
          rate for every cash-flow time, i.e. a non-parallel curve)

    Raises
    ------
    ValueError
        If any rate is not strictly greater than ``-1`` (discount factor would
        be undefined).
    """
    t = np.asarray(times, dtype=float)
    r = np.asarray(rates, dtype=float)
    if not np.all(np.isfinite(r)):
        raise ValueError("Rates must be finite.")
    if np.any(r <= MIN_ALLOWED_RATE):
        raise ValueError(f"Rates must be > {MIN_ALLOWED_RATE:.3f} (so that 1 + r > 0).")

    if r.ndim == 0:
        return (1.0 + r) ** (-t)
    if r.ndim == 1:
        return (1.0 + r[None, :]) ** (-t[:, None])
    if r.ndim == 2:
        if r.shape[0] != t.shape[0]:
            raise ValueError("2-D rates must have shape (len(times), n_scenarios).")
        return (1.0 + r) ** (-t[:, None])
    raise ValueError("rates must be a scalar, a 1-D or a 2-D array.")


@dataclass(frozen=True)
class CashFlowStream:
    """A deterministic stream of cash flows ``(t_i, C_i)``.

    Attributes
    ----------
    times:
        Strictly positive payment times in years (not necessarily integers).
    amounts:
        Cash amounts paid at each time (same length as ``times``).
    """

    times: FloatArray
    amounts: FloatArray

    def __post_init__(self) -> None:
        t = np.asarray(self.times, dtype=float)
        c = np.asarray(self.amounts, dtype=float)
        if t.ndim != 1 or c.ndim != 1 or t.shape != c.shape:
            raise ValueError("times and amounts must be 1-D arrays of equal length.")
        if t.size == 0:
            raise ValueError("A cash-flow stream needs at least one cash flow.")
        if not (np.all(np.isfinite(t)) and np.all(np.isfinite(c))):
            raise ValueError("times and amounts must be finite.")
        if np.any(t <= 0):
            raise ValueError("All cash-flow times must be strictly positive.")
        object.__setattr__(self, "times", t)
        object.__setattr__(self, "amounts", c)

    # -- valuation ---------------------------------------------------------- #
    def present_value(self, rates: ArrayLike) -> float | FloatArray:
        r"""Present value :math:`PV = \sum_t C_t (1+r)^{-t}`.

        A scalar ``rates`` returns a ``float``; array-valued ``rates`` (see
        :func:`discount_factors`) returns one PV per scenario, shape ``(N,)``.
        """
        df = discount_factors(self.times, rates)
        pv = self.amounts @ df
        return float(pv) if np.ndim(pv) == 0 else np.asarray(pv, dtype=float)

    def macaulay_duration(self, y: float) -> float:
        r"""Macaulay duration :math:`D = \frac{1}{PV}\sum_t t\,C_t (1+y)^{-t}`."""
        df = discount_factors(self.times, y)
        pv = float(self.amounts @ df)
        self._require_positive_pv(pv)
        return float((self.times * self.amounts) @ df / pv)

    def modified_duration(self, y: float) -> float:
        r"""Modified duration :math:`D_{mod} = D / (1 + y)`."""
        return self.macaulay_duration(y) / (1.0 + y)

    def convexity(self, y: float) -> float:
        r"""Convexity :math:`C = \frac{1}{PV}\sum_t t(t+1)\,C_t (1+y)^{-t-2}`.

        Equivalent to :math:`PV''(y) / PV(y)`.
        """
        df = discount_factors(self.times, y)
        pv = float(self.amounts @ df)
        self._require_positive_pv(pv)
        return float((self.times * (self.times + 1.0) * self.amounts) @ df / (pv * (1.0 + y) ** 2))

    # -- utilities ---------------------------------------------------------- #
    def on_grid(self, grid: ArrayLike) -> "CashFlowStream":
        """Re-express the stream on a common time grid (zero-filling gaps).

        Cash flows whose time is not on the grid, or that fall beyond it, raise
        a ``ValueError`` so no cash flow is ever silently dropped.
        """
        g = np.asarray(grid, dtype=float)
        out = np.zeros_like(g)
        for t, c in zip(self.times, self.amounts):
            idx = np.flatnonzero(np.isclose(g, t))
            if idx.size == 0:
                raise ValueError(f"Cash flow at t={t} is not on the supplied grid.")
            out[idx[0]] += c
        return CashFlowStream(times=g, amounts=out)

    @property
    def total(self) -> float:
        """Undiscounted sum of all cash flows."""
        return float(self.amounts.sum())

    @staticmethod
    def _require_positive_pv(pv: float) -> None:
        if not np.isfinite(pv) or pv <= 0.0:
            raise ValueError("Present value must be strictly positive to define duration/convexity.")


# --------------------------------------------------------------------------- #
# Liability generators
# --------------------------------------------------------------------------- #
LiabilityKind = Literal["level", "escalating"]


def generate_level_annuity(horizon: int, annual_payment: float) -> CashFlowStream:
    r"""Level guaranteed annuity-immediate.

    Pays ``annual_payment`` at the end of each year ``t = 1 ... n``::

        PV = P * a_n(y) = P * (1 - (1+y)^-n) / y
    """
    return generate_liabilities(horizon, annual_payment, kind="level")


def generate_escalating_annuity(
    horizon: int, first_payment: float, escalation: float
) -> CashFlowStream:
    r"""Annuity whose payments grow at a cost-of-living rate ``g``.

    Defined-benefit style stream::

        C_t = P_1 * (1 + g)^(t-1),     t = 1 ... n
    """
    return generate_liabilities(horizon, first_payment, kind="escalating", escalation=escalation)


def generate_liabilities(
    horizon: int,
    annual_payment: float,
    kind: LiabilityKind = "level",
    escalation: float = 0.0,
) -> CashFlowStream:
    """Generate an ``horizon``-year liability outflow stream (paid year-end).

    Parameters
    ----------
    horizon:
        Number of annual payments ``n >= 1``.
    annual_payment:
        First (or only, for ``kind='level'``) annual payment, strictly positive.
    kind:
        ``'level'`` for a flat guaranteed annuity, ``'escalating'`` for a
        payment stream growing at ``escalation`` per year.
    escalation:
        Annual growth rate ``g`` of payments (ignored when ``kind='level'``).
    """
    if int(horizon) != horizon or horizon < 1:
        raise ValueError("horizon must be a positive integer number of years.")
    if not np.isfinite(annual_payment) or annual_payment <= 0:
        raise ValueError("annual_payment must be strictly positive.")
    if kind not in ("level", "escalating"):
        raise ValueError(f"Unknown liability kind: {kind!r}")
    if escalation <= -1.0:
        raise ValueError("escalation must be greater than -100%.")

    times = np.arange(1, int(horizon) + 1, dtype=float)
    growth = escalation if kind == "escalating" else 0.0
    amounts = annual_payment * (1.0 + growth) ** (times - 1.0)
    return CashFlowStream(times=times, amounts=amounts)


# --------------------------------------------------------------------------- #
# Convenience valuation wrappers + summary
# --------------------------------------------------------------------------- #
def liability_pv(liabilities: CashFlowStream, y: ArrayLike) -> float | FloatArray:
    r"""Liability present value :math:`PV_L = \sum_t L_t (1+y)^{-t}`."""
    return liabilities.present_value(y)


def liability_duration(liabilities: CashFlowStream, y: float) -> float:
    r"""Liability Macaulay duration :math:`D_L`."""
    return liabilities.macaulay_duration(y)


def liability_convexity(liabilities: CashFlowStream, y: float) -> float:
    r"""Liability convexity :math:`C_L = PV_L'' / PV_L`."""
    return liabilities.convexity(y)


@dataclass(frozen=True)
class LiabilitySummary:
    """Headline liability analytics at a given flat yield."""

    yield_rate: float
    present_value: float
    macaulay_duration: float
    modified_duration: float
    convexity: float
    dollar_duration: float  # PV * D_mac  (proportional to -dPV/dy * (1+y))
    dollar_convexity: float  # PV * C      (= d2PV/dy2)


def summarize_liabilities(liabilities: CashFlowStream, y: float) -> LiabilitySummary:
    """Compute :class:`LiabilitySummary` for ``liabilities`` at yield ``y``."""
    pv = float(liabilities.present_value(y))
    d_mac = liabilities.macaulay_duration(y)
    conv = liabilities.convexity(y)
    return LiabilitySummary(
        yield_rate=y,
        present_value=pv,
        macaulay_duration=d_mac,
        modified_duration=d_mac / (1.0 + y),
        convexity=conv,
        dollar_duration=pv * d_mac,
        dollar_convexity=pv * conv,
    )
