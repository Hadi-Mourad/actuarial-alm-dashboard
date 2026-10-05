"""Stochastic ALM & Portfolio Immunization Dashboard (Streamlit entry point).

Run with::

    streamlit run app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde

from src.assets import (
    Portfolio,
    bond_analytics_table,
    build_portfolio,
    default_bond_universe,
    equal_weight_portfolio,
)
from src.immunization import (
    OBJECTIVES,
    ImmunizationError,
    ImmunizationResult,
    InfeasibleImmunizationError,
    default_transaction_costs_bps,
    immunize,
)
from src.liabilities import CashFlowStream, generate_liabilities, summarize_liabilities
from src.simulation import (
    RiskMetrics,
    ShockConfig,
    SimulationResult,
    effective_shock_std,
    price_yield_curve,
    run_monte_carlo,
)

# --------------------------------------------------------------------------- #
# Page setup & styling
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Stochastic ALM & Immunization Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLOR_ASSET = "#1f77b4"
COLOR_LIABILITY = "#d62728"
COLOR_SURPLUS = "#2ca02c"
COLOR_BENCH = "#ff7f0e"
PLOT_TEMPLATE = "plotly_white"


def money(x: float, decimals: int = 0) -> str:
    """Format a currency amount, e.g. ``-$1,234``."""
    sign = "-" if x < 0 else ""
    return f"{sign}${abs(x):,.{decimals}f}"


# --------------------------------------------------------------------------- #
# Chart builders
# --------------------------------------------------------------------------- #
def chart_cash_flows(liabilities: CashFlowStream, assets: CashFlowStream) -> go.Figure:
    """Chart 1 - overlaid bars of asset vs. liability cash flows."""
    fig = go.Figure()
    fig.add_bar(
        x=assets.times, y=assets.amounts, name="Asset cash flows",
        marker_color=COLOR_ASSET, opacity=0.65,
        hovertemplate="Year %{x:.0f}<br>Assets: $%{y:,.0f}<extra></extra>",
    )
    fig.add_bar(
        x=liabilities.times, y=liabilities.amounts, name="Liability cash flows",
        marker_color=COLOR_LIABILITY, opacity=0.75,
        hovertemplate="Year %{x:.0f}<br>Liabilities: $%{y:,.0f}<extra></extra>",
    )
    fig.update_layout(
        template=PLOT_TEMPLATE, barmode="overlay", height=430,
        title="Asset vs. Liability Cash Flows",
        xaxis_title="Year", yaxis_title="Cash flow", yaxis_tickprefix="$", yaxis_tickformat=".3s",
        legend=dict(orientation="h", y=1.1, x=0), margin=dict(t=80, b=40),
    )
    return fig


def chart_price_yield(curve: pd.DataFrame, base_yield: float, target_surplus: float) -> go.Figure:
    """Chart 2 - price-yield curves (top) and the resulting surplus (bottom)."""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.66, 0.34],
        subplot_titles=("Market value vs. parallel yield shift", "Surplus = Assets - Liabilities"),
    )
    hover = "Shift %{x:+.0f} bp<br>%{customdata:.2%} yield<br>%{y:$,.0f}<extra>%{fullData.name}</extra>"
    fig.add_trace(
        go.Scatter(x=curve["shift_bp"], y=curve["assets"], name="Asset value", mode="lines",
                   line=dict(color=COLOR_ASSET, width=3), customdata=curve["yield"], hovertemplate=hover),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=curve["shift_bp"], y=curve["liabilities"], name="Liability value", mode="lines",
                   line=dict(color=COLOR_LIABILITY, width=3, dash="dash"), customdata=curve["yield"],
                   hovertemplate=hover),
        row=1, col=1,
    )
    i0 = int(np.argmin(np.abs(curve["shift_bp"].to_numpy())))
    fig.add_trace(
        go.Scatter(x=[curve["shift_bp"].iloc[i0]], y=[curve["liabilities"].iloc[i0]], mode="markers",
                   marker=dict(size=11, color="black", symbol="diamond"),
                   name=f"Tangency at y0 = {base_yield:.2%}", hoverinfo="skip"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=curve["shift_bp"], y=curve["surplus"], name="Surplus", mode="lines",
                   line=dict(color=COLOR_SURPLUS, width=2.5), fill="tozeroy",
                   fillcolor="rgba(44,160,44,0.18)", customdata=curve["yield"], hovertemplate=hover),
        row=2, col=1,
    )
    fig.add_hline(y=0, line_color="black", line_width=1, row=2, col=1)
    fig.add_hline(y=target_surplus, line_dash="dot", line_color="gray", row=2, col=1,
                  annotation_text="target surplus", annotation_position="top right")
    for r in (1, 2):
        fig.add_vline(x=0, line_dash="dot", line_color="gray", row=r, col=1)
    fig.update_yaxes(tickprefix="$", tickformat=".3s")
    fig.update_xaxes(title_text="Parallel yield shift (bp)", row=2, col=1)
    fig.update_layout(
        template=PLOT_TEMPLATE, height=680, title="Price-Yield Sensitivity (Redington Tangency & Convexity)",
        legend=dict(orientation="h", y=1.08, x=0), margin=dict(t=100, b=40),
    )
    return fig


def _kde_curve(values: np.ndarray, n: int = 400) -> tuple[np.ndarray, np.ndarray] | None:
    """Gaussian KDE on a grid, or ``None`` if the sample is (numerically) degenerate."""
    if np.ptp(values) < 1e-9 * max(1.0, float(np.abs(values).mean())):
        return None
    try:
        kde = gaussian_kde(values)
    except (np.linalg.LinAlgError, ValueError):
        return None
    pad = 0.05 * np.ptp(values)
    xs = np.linspace(values.min() - pad, values.max() + pad, n)
    return xs, kde(xs)


def chart_surplus_distribution(
    sim: SimulationResult,
    bench_sim: SimulationResult | None = None,
    bench_label: str = "",
) -> go.Figure:
    """Chart 3 - histogram + KDE of the simulated surplus with risk markers."""
    m: RiskMetrics = sim.metrics
    fig = go.Figure()
    fig.add_histogram(
        x=sim.surplus, histnorm="probability density", nbinsx=60, name="Immunized surplus",
        marker_color=COLOR_ASSET, opacity=0.55,
        hovertemplate="Surplus: $%{x:,.0f}<br>Density: %{y:.3e}<extra></extra>",
    )
    kde = _kde_curve(sim.surplus)
    if kde is not None:
        fig.add_scatter(x=kde[0], y=kde[1], mode="lines", name="KDE (immunized)",
                        line=dict(color="#0b3d91", width=2.5), hoverinfo="skip")
    if bench_sim is not None:
        fig.add_histogram(
            x=bench_sim.surplus, histnorm="probability density", nbinsx=60, name=f"{bench_label} surplus",
            marker_color=COLOR_BENCH, opacity=0.4,
        )
        bench_kde = _kde_curve(bench_sim.surplus)
        if bench_kde is not None:
            fig.add_scatter(x=bench_kde[0], y=bench_kde[1], mode="lines", name=f"KDE ({bench_label})",
                            line=dict(color="#b35806", width=2, dash="dot"), hoverinfo="skip")

    markers = [
        ("Mean", m.mean, "#2ca02c", "solid", 1.00),
        ("95% VaR", m.surplus_at_var_95, "#ff7f0e", "dash", 0.88),
        ("95% TVaR", m.surplus_at_tvar_95, "#d62728", "dash", 0.76),
    ]
    for label, x, color, dash, y_frac in markers:
        fig.add_vline(x=x, line_color=color, line_dash=dash, line_width=2)
        fig.add_annotation(
            x=x, y=y_frac, yref="paper", text=f"<b>{label}</b><br>{money(x)}", showarrow=False,
            xanchor="left", xshift=4, font=dict(color=color, size=12), bgcolor="rgba(255,255,255,0.75)",
        )
    fig.update_layout(
        template=PLOT_TEMPLATE, barmode="overlay", height=470,
        title=f"Monte Carlo Surplus Distribution (N = {sim.surplus.size:,})",
        xaxis_title="Surplus after shock", yaxis_title="Probability density",
        xaxis_tickprefix="$", xaxis_tickformat=".3s",
        legend=dict(orientation="h", y=1.12, x=0), margin=dict(t=90, b=40),
    )
    return fig


def chart_allocation(result: ImmunizationResult) -> go.Figure:
    """Donut chart of the optimal market-value weights."""
    names = [b.name for b in result.portfolio.bonds]
    fig = go.Figure(go.Pie(labels=names, values=result.weights, hole=0.55, sort=False,
                           textinfo="label+percent"))
    fig.update_layout(template=PLOT_TEMPLATE, height=320, showlegend=False, margin=dict(t=20, b=20, l=10, r=10))
    return fig


# --------------------------------------------------------------------------- #
# Sidebar controls
# --------------------------------------------------------------------------- #
st.title("Stochastic ALM & Portfolio Immunization Dashboard")
st.caption(
    "Redington immunization of a guaranteed-annuity liability with a 3-bond universe, stress-tested with "
    "Monte Carlo interest-rate shocks. Financial mathematics, optimisation, corporate finance and "
    "statistics in one view."
)

with st.sidebar:
    st.header("Liabilities")
    horizon = st.slider("Liability horizon (years)", 5, 40, 20, help="Number of annual guaranteed payments.")
    liability_kind_label = st.selectbox(
        "Liability structure", ["Level guaranteed annuity", "Escalating annuity (COLA)"]
    )
    annual_payment = st.number_input(
        "Annual payment ($)", min_value=10_000.0, max_value=1_000_000_000.0,
        value=1_000_000.0, step=100_000.0, format="%.0f",
    )
    escalation_pct = 0.0
    if liability_kind_label.startswith("Escalating"):
        escalation_pct = st.slider("Annual escalation (%)", 0.0, 5.0, 2.0, 0.25)

    st.header("Market")
    base_yield_pct = st.slider("Base yield rate (%)", 0.5, 10.0, 4.0, 0.25,
                               help="Flat effective annual yield used to value assets and liabilities.")

    st.header("Asset universe (coupon rates)")
    c1_pct = st.slider("2-year bond coupon (%)", 0.0, 10.0, 3.0, 0.25)
    c2_pct = st.slider("10-year bond coupon (%)", 0.0, 10.0, 4.0, 0.25)
    c3_pct = st.slider("30-year bond coupon (%)", 0.0, 10.0, 5.0, 0.25)

    st.header("Immunization")
    target_surplus_pct = st.slider(
        "Target surplus (% of PV of liabilities)", 0.0, 10.0, 1.0, 0.25,
        help="Assets are sized to PV_L x (1 + target surplus).",
    )
    objective = st.selectbox("Optimisation objective", list(OBJECTIVES), index=0,
                             format_func=lambda k: OBJECTIVES[k])
    with st.expander("Transaction costs (bps)"):
        _defaults = default_transaction_costs_bps(default_bond_universe())
        tc = [
            st.number_input(name, 0.0, 200.0, float(round(d, 1)), 0.5, key=f"tc_{i}")
            for i, (name, d) in enumerate(zip(["2Y bond", "10Y bond", "30Y bond"], _defaults))
        ]

    st.header("Monte Carlo")
    shock_model_label = st.radio("Rate model", ["Gaussian shift", "Vasicek (mean-reverting)"])
    sigma_bp = st.slider("Shock volatility sigma (bp)", 10, 300, 100, 5,
                         help="Gaussian: std-dev of the parallel shift. Vasicek: instantaneous short-rate volatility.")
    n_sims = st.slider("Simulations (N)", 500, 20_000, 2_500, 500)
    seed = int(st.number_input("Random seed", 0, 1_000_000, 42, 1))
    mu_bp, kappa, theta_off_bp, mc_horizon = 0, 0.25, 0, 1.0
    if shock_model_label.startswith("Gaussian"):
        mu_bp = st.slider("Mean shift mu (bp)", -100, 100, 0, 5)
    else:
        kappa = st.slider("Mean reversion speed kappa", 0.01, 2.0, 0.25, 0.01)
        theta_off_bp = st.slider("Long-run mean minus base yield (bp)", -200, 200, 0, 10)
        mc_horizon = st.slider("Risk horizon (years)", 0.25, 5.0, 1.0, 0.25)
    slope_bp = st.slider(
        "Slope (twist) volatility (bp per 10y)", 0, 150, 0, 5,
        help="0 = parallel shocks only (what Redington protects against). Increase to see the limits of immunization.",
    )
    benchmark_label = st.selectbox(
        "Compare against unhedged portfolio",
        ["None", "100% 2Y bond", "100% 10Y bond", "100% 30Y bond", "Equal-weight (1/3 each)"],
    )

# --------------------------------------------------------------------------- #
# Model construction
# --------------------------------------------------------------------------- #
y0 = base_yield_pct / 100.0
bonds = default_bond_universe(c1_pct / 100.0, c2_pct / 100.0, c3_pct / 100.0)
liab_kind = "escalating" if liability_kind_label.startswith("Escalating") else "level"
liabilities = generate_liabilities(horizon, annual_payment, liab_kind, escalation_pct / 100.0)
liab_summary = summarize_liabilities(liabilities, y0)
target_surplus = liab_summary.present_value * target_surplus_pct / 100.0

try:
    result = immunize(
        liabilities, bonds, y0, target_surplus=target_surplus, objective=objective,
        transaction_costs_bps=tc,
    )
except InfeasibleImmunizationError as exc:
    st.error(f"**Immunization infeasible.** {exc}")
    st.info(
        f"Liability analytics at {y0:.2%}: PV = {money(liab_summary.present_value)}, "
        f"D = {liab_summary.macaulay_duration:.2f} y, convexity = {liab_summary.convexity:.1f}."
    )
    st.dataframe(bond_analytics_table(bonds, y0), hide_index=True)
    st.stop()
except ImmunizationError as exc:
    st.error(f"**Optimisation failed.** {exc}")
    st.stop()

for msg in result.warnings:
    st.warning(msg)

portfolio = result.portfolio
asset_stream = portfolio.cash_flow_stream(horizon)

shock_cfg = ShockConfig(
    model="gaussian" if shock_model_label.startswith("Gaussian") else "vasicek",
    n_sims=n_sims, mu=mu_bp / 1e4, sigma=sigma_bp / 1e4, kappa=kappa, theta_offset=theta_off_bp / 1e4,
    horizon=mc_horizon, slope_sigma=slope_bp / 1e4, seed=seed,
)
sim = run_monte_carlo(liabilities, asset_stream, y0, shock_cfg)
metrics = sim.metrics

bench_sim: SimulationResult | None = None
bench_portfolio: Portfolio | None = None
if benchmark_label != "None":
    if benchmark_label.startswith("Equal"):
        bench_portfolio = equal_weight_portfolio(bonds, result.pv_assets, y0)
    else:
        idx = {"100% 2Y bond": 0, "100% 10Y bond": 1, "100% 30Y bond": 2}[benchmark_label]
        w = np.zeros(len(bonds))
        w[idx] = 1.0
        bench_portfolio = build_portfolio(bonds, w, result.pv_assets, y0)
    bench_sim = run_monte_carlo(liabilities, bench_portfolio.cash_flow_stream(horizon), y0, shock_cfg)

# --------------------------------------------------------------------------- #
# Metrics bar
# --------------------------------------------------------------------------- #
st.subheader("Portfolio Key Metrics")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("PV Liabilities (PV_L)", money(result.pv_liabilities),
          help="PV_L = sum L_t / (1+y)^t")
m2.metric("PV Assets (PV_A)", money(result.pv_assets),
          delta=f"Surplus {money(result.surplus)}", delta_color="off",
          help="PV_A = PV_L + target surplus (budget constraint).")
_dur_gap = 0.0 if abs(result.duration_gap_scaled) < 5e-5 else result.duration_gap_scaled  # avoid "-0.0000"
m3.metric("Duration gap", f"{_dur_gap:+.4f} yrs",
          delta=f"D_A {result.duration_assets:.3f} vs D_L {result.duration_liabilities:.3f}", delta_color="off",
          help="(PV_A*D_A - PV_L*D_L) / PV_L. Zero means Redington's first-order condition holds "
               "(equals D_A - D_L when the target surplus is 0).")
m4.metric("Convexity gap", f"{result.convexity_gap_scaled:+.2f}",
          delta="Redington satisfied" if result.redington_satisfied else "Condition violated",
          delta_color="normal" if result.redington_satisfied else "inverse",
          help="(PV_A*C_A - PV_L*C_L) / PV_L. Must be > 0 for the second-order Redington condition.")
m5.metric("95% TVaR (surplus loss)", money(metrics.tvar_95),
          delta=f"Tail surplus {money(metrics.surplus_at_tvar_95)}", delta_color="off",
          help="Mean loss, relative to the initial surplus, in the worst 5% of scenarios. "
               "Negative = surplus still rises in the tail.")

s1, s2, s3, s4, s5 = st.columns(5)
s1.metric("Mean surplus", money(metrics.mean))
s2.metric("Std dev of surplus", money(metrics.std))
s3.metric("95% VaR", money(metrics.var_95), help="95th percentile of loss = initial surplus - surplus.")
s4.metric("99% VaR", money(metrics.var_99))
s5.metric("P(Surplus < 0)", f"{metrics.prob_insolvency:.2%}", help="Empirical probability of insolvency.")

eff_std = effective_shock_std(shock_cfg, y0)
st.caption(
    f"Rate model: **{shock_model_label}** - effective std-dev of the parallel shift over the horizon = "
    f"**{eff_std * 1e4:.0f} bp**. Optimiser: `{result.diagnostics['method']}` "
    f"({'converged' if result.success else 'fallback'}), objective value {result.objective_value:.4g}."
)

# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
st.plotly_chart(chart_cash_flows(liabilities, asset_stream))

curve = price_yield_curve(liabilities, asset_stream, y0, bp_range=300.0, n_points=121)
st.plotly_chart(chart_price_yield(curve, y0, result.surplus))
st.caption(
    "At the base yield the asset curve is tangent to the liability curve (equal slope: dollar durations "
    "match) and lies above it (greater curvature: dollar convexity dominates), so the surplus panel is "
    "non-negative for parallel moves near y0. Larger moves or non-parallel shocks can still erode it."
)

st.plotly_chart(
    chart_surplus_distribution(
        sim, bench_sim, benchmark_label if bench_sim is not None else ""
    )
)

if bench_sim is not None and bench_portfolio is not None:
    bm = bench_sim.metrics
    cmp_df = pd.DataFrame(
        {
            "Metric": ["Mean surplus", "Std dev", "95% VaR (loss)", "99% VaR (loss)", "95% TVaR (loss)", "P(Surplus < 0)"],
            "Immunized": [metrics.mean, metrics.std, metrics.var_95, metrics.var_99, metrics.tvar_95, metrics.prob_insolvency],
            benchmark_label: [bm.mean, bm.std, bm.var_95, bm.var_99, bm.tvar_95, bm.prob_insolvency],
        }
    )
    st.dataframe(
        cmp_df, hide_index=True,
        column_config={
            "Immunized": st.column_config.NumberColumn(format="%.4g"),
            benchmark_label: st.column_config.NumberColumn(format="%.4g"),
        },
    )

# --------------------------------------------------------------------------- #
# Details
# --------------------------------------------------------------------------- #
st.subheader("Optimal Portfolio & Diagnostics")
left, right = st.columns([3, 2])
with left:
    alloc = portfolio.allocation_table()
    alloc["Weight"] = alloc["Weight"] * 100.0
    st.dataframe(
        alloc, hide_index=True,
        column_config={
            "Weight": st.column_config.NumberColumn("Weight (%)", format="%.2f"),
            "Units (face 100)": st.column_config.NumberColumn(format="localized"),
            "Market Value": st.column_config.NumberColumn(format="dollar"),
            "Price": st.column_config.NumberColumn(format="%.3f"),
            "Macaulay D": st.column_config.NumberColumn(format="%.3f"),
            "Convexity": st.column_config.NumberColumn(format="%.2f"),
        },
    )
    st.markdown("**Bond universe analytics**")
    st.dataframe(
        bond_analytics_table(bonds, y0), hide_index=True,
        column_config={
            "Coupon": st.column_config.NumberColumn(format="percent"),
            "Price": st.column_config.NumberColumn(format="%.3f"),
            "Macaulay D": st.column_config.NumberColumn(format="%.3f"),
            "Modified D": st.column_config.NumberColumn(format="%.3f"),
            "Convexity": st.column_config.NumberColumn(format="%.2f"),
        },
    )
with right:
    st.plotly_chart(chart_allocation(result))

with st.expander("Redington condition check & solver diagnostics"):
    diag = pd.DataFrame(
        {
            "Quantity": [
                "PV_A - PV_L (surplus)", "D_A (Macaulay)", "D_L (Macaulay)", "C_A", "C_L",
                "Dollar-duration gap / PV_L", "Dollar-convexity gap / PV_L",
                "Target D_A = D_L PV_L/PV_A", "Target C_A > C_L PV_L/PV_A",
                "Duration capacity of universe (min)", "Duration capacity of universe (max)",
                "Max attainable convexity at target duration", "Solver iterations",
            ],
            "Value": [
                result.surplus, result.duration_assets, result.duration_liabilities, result.convexity_assets,
                result.convexity_liabilities, result.duration_gap_scaled, result.convexity_gap_scaled,
                result.diagnostics["target_duration"], result.diagnostics["target_convexity"],
                result.diagnostics["duration_capacity_min"], result.diagnostics["duration_capacity_max"],
                result.diagnostics["max_attainable_convexity"], result.diagnostics["iterations"],
            ],
        }
    )
    st.dataframe(diag, hide_index=True, column_config={"Value": st.column_config.NumberColumn(format="%.6g")})
    st.caption(f"Solver message: {result.message}")

with st.expander("Methodology"):
    st.markdown(
        r"""
**Valuation** - $PV(y)=\sum_t C_t(1+y)^{-t}$, Macaulay duration $D=\tfrac1{PV}\sum_t tC_t(1+y)^{-t}$,
convexity $C=\tfrac1{PV}\sum_t t(t+1)C_t(1+y)^{-t-2}$.

**Redington immunization** - surplus $S(y)=V_A(y)-V_L(y)$ has a local minimum at $y_0$ when
$V_A=V_L$ (+ target surplus), $V_A'=V_L'$ and $V_A''>V_L''$.

**Optimisation** - SLSQP over long-only weights with linear budget, duration-match and convexity
constraints; an LP pre-check certifies feasibility and provides a fallback solution.

**Risk** - $\text{Loss}_k=S_0-S_k$, $\text{VaR}_\alpha$ = empirical $\alpha$-quantile of loss,
$\text{TVaR}_\alpha=E[\text{Loss}\mid\text{Loss}\ge\text{VaR}_\alpha]$, ruin probability $P(S<0)$.
"""
    )
