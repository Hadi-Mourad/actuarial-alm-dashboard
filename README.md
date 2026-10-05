# Stochastic ALM & Portfolio Immunization Dashboard

An interactive Streamlit app that immunizes a guaranteed-annuity liability with a three-bond portfolio
(Redington immunization), then stress-tests the result with Monte Carlo interest-rate shocks.

| Discipline | Where it shows up |
|---|---|
| Actuarial / financial mathematics | `liabilities.py`, `assets.py`: PV, Macaulay/modified duration, convexity |
| Calculus & optimisation | `immunization.py`: derivative-matching conditions solved with SLSQP |
| Corporate finance | Asset/liability balance sheet, surplus buffer, NPV of surplus under shocks |
| Probability & statistics | `simulation.py`: Gaussian / Vasicek shocks, VaR, TVaR, ruin probability, KDE |

## Quick start

**Windows (PowerShell)**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

The app opens at <http://localhost:8501>. Python 3.10+ is required (developed on 3.12).

## Project layout

```
app.py                 Streamlit UI (sidebar, metrics bar, three charts, diagnostics)
requirements.txt       Pinned dependencies
src/
  liabilities.py       CashFlowStream primitive, liability generators, PV / duration / convexity
  assets.py            Bond (closed-form price, D, C), 2Y/10Y/30Y universe, Portfolio
  immunization.py      Redington optimisation (LP feasibility pre-check + SLSQP + fallbacks)
  simulation.py        Shock models, scenario re-pricing, VaR / TVaR, price-yield sensitivity
```

## Methodology

All values use a flat effective annual yield `y` and annual end-of-year cash flows.

**Valuation**

```
PV(y) = sum_t C_t (1+y)^-t
D(y)  = (1/PV) sum_t t C_t (1+y)^-t              (Macaulay)
C(y)  = (1/PV) sum_t t(t+1) C_t (1+y)^-(t+2)     (= PV'' / PV)
PV'   = -PV * D / (1+y)
```

**Redington immunization.** With surplus `S(y) = V_A(y) - V_L(y)`, a second-order Taylor expansion
shows `y0` is a strict local minimum of surplus when

1. `V_A = V_L + S*` (initial solvency / target surplus),
2. `V_A' = V_L'`, i.e. `PV_A * D_A = PV_L * D_L` (dollar-duration match),
3. `V_A'' > V_L''`, i.e. `PV_A * C_A > PV_L * C_L` (dollar-convexity dominance).

For `S* = 0` these reduce to the textbook `D_A = D_L` and `C_A > C_L`. With a positive buffer the
value-weighted forms above are the correct ones, and they are what the optimiser enforces. The dashboard's
"Duration gap" and "Convexity gap" are these dollar gaps divided by `PV_L`.

**Optimisation.** Decision variables are market-value weights `w` over the three bonds. Portfolio
duration and convexity are value-weighted averages, so every constraint is linear:

```
min  J(w)
s.t. sum w_i = 1                        (PV matching: assets sized to PV_L + S*)
     sum w_i D_i  = D_L * PV_L / PV_A   (duration)
     sum w_i C_i >= C_L * PV_L / PV_A + eps   (convexity; eps turns ">" into a closed constraint)
     w_i >= 0                           (no short selling)
```

Selectable objectives: cash-flow tracking error, linear transaction cost, tight convexity, and squared
duration gap with a diversification tie-breaker. Because the feasible set is a polytope, a linear programme
(maximise convexity subject to the equalities) decides feasibility exactly *before* SLSQP runs. It also
provides a feasible starting point. If SLSQP fails, `trust-constr` is tried, then the LP vertex is returned
with a warning.

**Edge cases handled** (`InfeasibleImmunizationError` with an explanatory message):

* liability duration outside the span of bond durations (liabilities too long or too short);
* convexity condition unattainable with a long-only mix (e.g. very short horizons, or 35-40 years at
  very low yields where the 30-year bond is not long or convex enough);
* invalid inputs (negative surplus, non-positive PV, yields <= -100 %).

**Monte Carlo.** Each scenario shocks the curve and re-prices both sides:

* Gaussian: `dy ~ N(mu, sigma^2)`.
* Vasicek: `dr = kappa (theta - r) dt + sigma dW`, sampled with the exact transition density at the
  risk horizon, `dy = r_H - r_0`.
* Optional slope (twist) factor: `y_k(t) = y0 + dy_k + s_k (t - t*) / 10`, pivoting at the liability
  duration. Redington protects against *parallel* shifts only, so raising the slope volatility
  shows where immunization breaks down.

```
S_k    = PV_A(y_k) - PV_L(y_k)
Loss_k = S_0 - S_k
VaR_a  = a-quantile of Loss          TVaR_a = E[Loss | Loss >= VaR_a]
P(ruin) = P(S_k < 0)
```

A negative VaR or TVaR means the surplus is higher than today even in the tail. That is the expected
outcome for a convex, duration-matched portfolio under parallel shocks.

## Assumptions and limitations

* Flat yield curve and parallel shifts for immunization; the slope option is a stress, not a re-optimisation.
* Annual coupons and liability payments; bullet bonds with no default or credit risk.
* Simulated rates are floored at -50 % purely for numerical safety.
* Transaction costs are illustrative (`2 + 0.9 * maturity` bps by default and editable in the sidebar).
* Educational / analytical tool, not investment advice.
