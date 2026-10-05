"""Redington immunization via constrained optimisation.

Theory
------
Let ``V_A(y)`` and ``V_L(y)`` be the market values of the assets and the
liabilities under a (parallel) change of the flat yield ``y``, and define the
surplus ``S(y) = V_A(y) - V_L(y)``.  A second-order Taylor expansion around the
current yield ``y0`` gives::

    S(y0 + dy) ~ S(y0) + S'(y0) dy + 1/2 S''(y0) dy^2

Redington's (1952) conditions make ``y0`` a *strict local minimum* of surplus,
i.e. surplus can only increase after a small parallel shift::

    (R1)  V_A(y0)  >= V_L(y0)                (initial solvency, S(y0) = target)
    (R2)  V_A'(y0)  = V_L'(y0)               (first-order / dollar-duration match)
    (R3)  V_A''(y0) > V_L''(y0)              (second-order / convexity dominance)

Using ``V' = -V D/(1+y)`` and ``V'' = V C`` (see :mod:`src.liabilities`) these
become, with a common ``y`` so that the ``(1+y)`` factors cancel::

    (R2)  V_A D_A = V_L D_L        <=>  D_A = D_L * V_L / V_A
    (R3)  V_A C_A > V_L C_L        <=>  C_A > C_L * V_L / V_A

When the target surplus is zero (``V_A = V_L``) these collapse to the familiar
``D_A = D_L`` and ``C_A > C_L``.  With a positive surplus buffer the *dollar*
(value-weighted) versions above are the mathematically correct conditions, and
that is what this module enforces.

Optimisation problem
--------------------
Decision variables are market-value weights ``w in R^n`` over the bond
universe.  Because portfolio duration/convexity are value-weighted averages,
all constraints are **linear** in ``w``::

    minimise    J(w)                                   (see :data:`OBJECTIVES`)
    subject to  sum_i w_i            = 1                (budget: V_A = V_L + S*)
                sum_i w_i D_i        = D_L V_L / V_A    (R2, duration match)
                sum_i w_i C_i       >= C_L V_L / V_A + eps   (R3, convexity)
                w_i >= 0                                 (no short selling)

Here the PV-matching condition ``PV_A = PV_L + S*`` is imposed *by
construction*: the portfolio is sized to ``V_A = V_L + S*`` and the budget
constraint ``sum w_i = 1`` then guarantees the invested market value equals
that amount.  (Writing ``sum_i n_i P_i = V_L + S*`` as a second equality would
only duplicate the budget row and make the Jacobian rank-deficient for SLSQP.)

Because the feasible set is a polytope, feasibility can be decided exactly with a
linear programme (maximise convexity subject to the equalities).  That LP is
used (i) to raise precise, user-friendly errors *before* running SLSQP and
(ii) as a guaranteed-feasible starting point / last-resort fallback.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import Bounds, LinearConstraint, BFGS, linprog, minimize

from .assets import Bond, Portfolio, build_portfolio
from .liabilities import CashFlowStream, FloatArray

Objective = Literal["cashflow_tracking", "transaction_cost", "convexity_match", "duration_gap"]

#: Human-readable names for every supported objective.
OBJECTIVES: dict[str, str] = {
    "cashflow_tracking": "Cash-flow tracking error  (min sum_t (A_t - L_t)^2)",
    "transaction_cost": "Transaction cost  (min sum_i c_i w_i)",
    "convexity_match": "Tight convexity  (min (C_A - C_L*)^2)",
    "duration_gap": "Squared duration gap + diversification",
}

_FEAS_TOL = 1e-6


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #
class ImmunizationError(Exception):
    """Base class for all immunization failures."""


class InfeasibleImmunizationError(ImmunizationError):
    """Raised when no long-only portfolio can satisfy the Redington conditions."""


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ImmunizationResult:
    """Outcome of :func:`immunize`.

    The ``*_gap`` fields that are suffixed ``_scaled`` are *dollar* gaps divided
    by ``PV_L`` (so they have units of years / years^2 and equal the plain
    ``D_A - D_L`` / ``C_A - C_L`` when the target surplus is zero).
    """

    portfolio: Portfolio
    weights: FloatArray
    objective: str
    objective_value: float
    success: bool
    message: str
    pv_assets: float
    pv_liabilities: float
    surplus: float
    duration_assets: float
    duration_liabilities: float
    convexity_assets: float
    convexity_liabilities: float
    duration_gap: float
    convexity_gap: float
    duration_gap_scaled: float
    convexity_gap_scaled: float
    redington_satisfied: bool
    warnings: list[str] = field(default_factory=list)
    diagnostics: dict[str, float | str | int] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def default_transaction_costs_bps(bonds: Sequence[Bond]) -> FloatArray:
    """Illustrative round-trip costs ``c_i = 2 + 0.9 * maturity`` bps of value.

    Longer, less liquid bonds have wider bid/ask spreads.
    """
    return np.array([2.0 + 0.9 * b.maturity for b in bonds], dtype=float)


def duration_capacity(bonds: Sequence[Bond], y: float) -> tuple[float, float]:
    """Smallest and largest achievable (long-only) portfolio Macaulay duration."""
    d = np.array([b.macaulay_duration(y) for b in bonds])
    return float(d.min()), float(d.max())


def _build_objective(
    objective: str,
    *,
    D: FloatArray,
    C: FloatArray,
    d_star: float,
    c_star: float,
    cost_bps: FloatArray,
    B_norm: FloatArray,
    L_norm: FloatArray,
) -> tuple[Callable[[FloatArray], float], Callable[[FloatArray], FloatArray]]:
    """Return ``(f, grad_f)`` for the chosen objective (all analytically smooth)."""
    n = D.size

    if objective == "cashflow_tracking":
        k = 100.0

        def f(w: FloatArray) -> float:
            r = B_norm @ w - L_norm
            return float(k * r @ r)

        def g(w: FloatArray) -> FloatArray:
            return 2.0 * k * B_norm.T @ (B_norm @ w - L_norm)

    elif objective == "transaction_cost":

        def f(w: FloatArray) -> float:
            return float(cost_bps @ w)

        def g(w: FloatArray) -> FloatArray:
            return cost_bps.copy()

    elif objective == "convexity_match":
        scale = max(abs(c_star), 1.0)

        def f(w: FloatArray) -> float:
            return float(100.0 * ((C @ w - c_star) / scale) ** 2)

        def g(w: FloatArray) -> FloatArray:
            return 200.0 * (C @ w - c_star) / scale**2 * C

    elif objective == "duration_gap":
        scale = max(abs(d_star), 1.0)
        ridge = 1e-2
        centre = np.full(n, 1.0 / n)

        def f(w: FloatArray) -> float:
            gap = (D @ w - d_star) / scale
            return float(100.0 * gap**2 + ridge * np.sum((w - centre) ** 2))

        def g(w: FloatArray) -> FloatArray:
            gap = (D @ w - d_star) / scale
            return 200.0 * gap / scale * D + 2.0 * ridge * (w - centre)

    else:
        raise ValueError(f"Unknown objective {objective!r}. Choose from {list(OBJECTIVES)}.")

    return f, g


def _is_feasible(
    w: FloatArray, D: FloatArray, C: FloatArray, d_star: float, c_star: float, margin: float
) -> bool:
    return bool(
        np.all(w >= -1e-9)
        and abs(w.sum() - 1.0) <= 1e-7
        and abs(D @ w - d_star) <= _FEAS_TOL
        and (C @ w - c_star) >= margin - _FEAS_TOL
    )


def _clean(w: NDArray[np.float64]) -> FloatArray:
    """Clip numerical dust below zero and renormalise to the simplex."""
    w = np.clip(np.asarray(w, dtype=float), 0.0, None)
    total = w.sum()
    return w / total if total > 0 else w


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def immunize(
    liabilities: CashFlowStream,
    bonds: Sequence[Bond],
    y: float,
    target_surplus: float = 0.0,
    objective: Objective = "cashflow_tracking",
    transaction_costs_bps: Sequence[float] | None = None,
    min_convexity_gap: float = 1e-3,
) -> ImmunizationResult:
    """Compute the Redington-immunizing long-only bond portfolio.

    Parameters
    ----------
    liabilities:
        Liability outflows.
    bonds:
        Investable universe (>= 2 instruments).
    y:
        Flat valuation yield (effective annual).
    target_surplus:
        Desired initial surplus ``S* >= 0``; assets are sized to ``PV_L + S*``.
    objective:
        One of :data:`OBJECTIVES`.
    transaction_costs_bps:
        Per-bond cost in basis points of traded value (only used by the
        ``transaction_cost`` objective).  Defaults to
        :func:`default_transaction_costs_bps`.
    min_convexity_gap:
        Strictness margin ``eps`` in ``C_A - C_L* >= eps`` (scaled years^2)
        that turns the strict inequality (R3) into a closed constraint.

    Raises
    ------
    InfeasibleImmunizationError
        If the target duration lies outside the span of the bond durations, or
        if the maximum attainable convexity cannot dominate the liabilities.
    ImmunizationError
        For invalid inputs or if no feasible solution can be produced.
    """
    bonds = tuple(bonds)
    n = len(bonds)
    if n < 2:
        raise ImmunizationError("At least two bonds are required to match duration and convexity.")
    if target_surplus < 0:
        raise ImmunizationError("target_surplus must be non-negative.")
    if min_convexity_gap < 0:
        raise ImmunizationError("min_convexity_gap must be non-negative.")
    if objective not in OBJECTIVES:
        raise ImmunizationError(f"Unknown objective {objective!r}.")

    # --- liability side ---------------------------------------------------- #
    try:
        pv_l = float(liabilities.present_value(y))
        d_l = liabilities.macaulay_duration(y)
        c_l = liabilities.convexity(y)
    except ValueError as exc:  # invalid yield / non-positive PV
        raise ImmunizationError(f"Cannot value liabilities: {exc}") from exc

    pv_a_target = pv_l + target_surplus  # (R1)  V_A = V_L + S*
    ratio = pv_l / pv_a_target  # V_L / V_A  in (0, 1]
    d_star = d_l * ratio  # (R2)  D_A = D_L V_L / V_A
    c_star = c_l * ratio  # (R3)  C_A > C_L V_L / V_A

    # --- asset side -------------------------------------------------------- #
    prices = np.array([b.price(y) for b in bonds])
    D = np.array([b.macaulay_duration(y) for b in bonds])
    C = np.array([b.convexity(y) for b in bonds])

    # --- exact feasibility screening --------------------------------------- #
    d_min, d_max = float(D.min()), float(D.max())
    if d_star < d_min - 1e-9 or d_star > d_max + 1e-9:
        side = "longer" if d_star > d_max else "shorter"
        raise InfeasibleImmunizationError(
            f"Target (surplus-adjusted) liability duration {d_star:.2f}y lies outside the asset "
            f"duration capacity [{d_min:.2f}y, {d_max:.2f}y]. The liabilities are {side} than any "
            f"long-only mix of the available bonds. Shorten/lengthen the liability horizon, change "
            f"the base yield, reduce the target surplus, or add a bond with a "
            f"{'longer' if side == 'longer' else 'shorter'} duration."
        )
    d_star = float(np.clip(d_star, d_min, d_max))

    lp = linprog(
        c=-C,  # maximise convexity
        A_eq=np.vstack([np.ones(n), D]),
        b_eq=np.array([1.0, d_star]),
        bounds=[(0.0, None)] * n,
        method="highs",
    )
    if not lp.success:
        raise ImmunizationError(f"Feasibility LP failed ({lp.message}).")
    w_max_convex = _clean(lp.x)
    c_max = float(C @ w_max_convex)
    if c_max < c_star + min_convexity_gap:
        raise InfeasibleImmunizationError(
            f"Convexity condition cannot be met: with duration matched at {d_star:.2f}y the most "
            f"convex long-only portfolio has C_A = {c_max:.1f}, but the liabilities require "
            f"C_A > {c_star + min_convexity_gap:.1f}. Add a more convex (longer / lower-coupon) "
            f"instrument or reduce the liability horizon."
        )

    # --- objective --------------------------------------------------------- #
    horizon = int(max(liabilities.times.max(), max(b.maturity for b in bonds)))
    grid = np.arange(1, horizon + 1, dtype=float)
    unit_cf = np.column_stack(
        [b.cash_flow_stream().on_grid(grid).amounts / p for b, p in zip(bonds, prices)]
    )  # cash flow per unit of market value invested, shape (T, n)
    B_norm = unit_cf * pv_a_target / pv_l  # asset CF / V_L when weight vector is w
    L_norm = liabilities.on_grid(grid).amounts / pv_l

    costs = (
        np.asarray(transaction_costs_bps, dtype=float)
        if transaction_costs_bps is not None
        else default_transaction_costs_bps(bonds)
    )
    if costs.shape != (n,) or np.any(costs < 0):
        raise ImmunizationError("transaction_costs_bps must be non-negative with one entry per bond.")

    f, grad = _build_objective(
        objective, D=D, C=C, d_star=d_star, c_star=c_star, cost_bps=costs, B_norm=B_norm, L_norm=L_norm
    )

    # --- solve: SLSQP -> trust-constr -> LP fallback ------------------------ #
    bounds_ = [(0.0, 1.0)] * n
    slsqp_constraints = [
        {"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones(n)},
        {"type": "eq", "fun": lambda w: D @ w - d_star, "jac": lambda w: D},
        {"type": "ineq", "fun": lambda w: C @ w - c_star - min_convexity_gap, "jac": lambda w: C},
    ]
    warnings: list[str] = []
    candidates: list[tuple[str, FloatArray, bool, str, int]] = []

    try:
        res = minimize(
            f, w_max_convex, jac=grad, method="SLSQP", bounds=bounds_,
            constraints=slsqp_constraints, options={"maxiter": 500, "ftol": 1e-12},
        )
        candidates.append(("SLSQP", _clean(res.x), bool(res.success), str(res.message), int(res.nit)))
    except (ValueError, FloatingPointError) as exc:  # pragma: no cover - defensive
        warnings.append(f"SLSQP raised {type(exc).__name__}: {exc}")

    chosen: tuple[str, FloatArray, bool, str, int] | None = None
    for cand in candidates:
        if cand[2] and _is_feasible(cand[1], D, C, d_star, c_star, min_convexity_gap):
            chosen = cand
            break

    if chosen is None:
        if candidates:
            warnings.append(f"SLSQP did not converge to a feasible point ({candidates[-1][3]}); trying trust-region.")
        try:
            tc_constraints = [
                LinearConstraint(np.ones((1, n)), 1.0, 1.0),
                LinearConstraint(D.reshape(1, -1), d_star, d_star),
                LinearConstraint(C.reshape(1, -1), c_star + min_convexity_gap, np.inf),
            ]
            res = minimize(
                f, w_max_convex, jac=grad, hess=BFGS(), method="trust-constr",
                bounds=Bounds(np.zeros(n), np.ones(n)), constraints=tc_constraints,
                options={"maxiter": 1000, "gtol": 1e-10, "xtol": 1e-12},
            )
            cand = ("trust-constr", _clean(res.x), bool(res.success), str(res.message), int(res.nit))
            if _is_feasible(cand[1], D, C, d_star, c_star, min_convexity_gap):
                chosen = cand
        except (ValueError, FloatingPointError) as exc:  # pragma: no cover - defensive
            warnings.append(f"trust-constr raised {type(exc).__name__}: {exc}")

    if chosen is None:
        warnings.append(
            "Both nonlinear solvers failed; returning the feasible maximum-convexity LP solution."
        )
        chosen = ("linprog-fallback", w_max_convex, False, "Fallback to feasibility LP vertex.", 0)

    method, w_opt, success, message, nit = chosen

    # --- assemble result (all analytics recomputed from the real cash flows) #
    portfolio = build_portfolio(bonds, w_opt, pv_a_target, y)
    pv_a = float(portfolio.value(y))
    d_a = portfolio.macaulay_duration(y)
    c_a = portfolio.convexity(y)

    dollar_dur_gap = pv_a * d_a - pv_l * d_l
    dollar_cvx_gap = pv_a * c_a - pv_l * c_l
    dur_gap_scaled = dollar_dur_gap / pv_l
    cvx_gap_scaled = dollar_cvx_gap / pv_l
    satisfied = bool(
        abs(dur_gap_scaled) < 1e-4
        and cvx_gap_scaled > 0
        and pv_a >= pv_l - 1e-6 * pv_l
    )
    if not satisfied:
        warnings.append("Redington conditions are not fully satisfied by the returned portfolio.")

    return ImmunizationResult(
        portfolio=portfolio,
        weights=portfolio.weights,
        objective=objective,
        objective_value=float(f(portfolio.weights)),
        success=success,
        message=message,
        pv_assets=pv_a,
        pv_liabilities=pv_l,
        surplus=pv_a - pv_l,
        duration_assets=d_a,
        duration_liabilities=d_l,
        convexity_assets=c_a,
        convexity_liabilities=c_l,
        duration_gap=d_a - d_l,
        convexity_gap=c_a - c_l,
        duration_gap_scaled=dur_gap_scaled,
        convexity_gap_scaled=cvx_gap_scaled,
        redington_satisfied=satisfied,
        warnings=warnings,
        diagnostics={
            "method": method,
            "iterations": nit,
            "target_duration": d_star,
            "target_convexity": c_star,
            "duration_capacity_min": d_min,
            "duration_capacity_max": d_max,
            "max_attainable_convexity": c_max,
        },
    )
