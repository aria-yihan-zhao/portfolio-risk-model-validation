# Executive summary

## Scope

An equal-weighted portfolio of five U.S. industry return series was evaluated
from **2015-01-02** through
**2026-09-30**. One-day 99% VaR and Expected
Shortfall forecasts use a strictly lagged 500-day window.

## Portfolio context

- Annualized return: **14.20%**
- Annualized volatility: **17.10%**
- Maximum drawdown: **-34.05%**
- Worst daily return: **-11.55%**

These are descriptive sample statistics, not forecasts.

## Latest risk snapshot

- **Historical**: VaR 2.26%; ES 3.80%.
- **Gaussian**: VaR 2.09%; ES 2.40%.
- **Monte Carlo**: VaR 2.32%; ES 2.95%.
- **Filtered Historical**: VaR 1.40%; ES 2.47%.

## Out-of-sample validation

- **Historical**: 37 exceptions vs. 24.5 expected; fails Kupiec coverage (p=0.019) and fails independence (p<0.001).
- **Gaussian**: 59 exceptions vs. 24.5 expected; fails Kupiec coverage (p<0.001) and fails independence (p<0.001).
- **Monte Carlo**: 54 exceptions vs. 24.5 expected; fails Kupiec coverage (p<0.001) and fails independence (p<0.001).
- **Filtered Historical**: 30 exceptions vs. 24.5 expected; passes Kupiec coverage (p=0.283) and fails independence (p<0.001).

Passing a test means the sample does not provide sufficient evidence to reject
that property at 5%; it is not proof that the model is correct.

## Stress testing

The largest named deterministic shock is **broad equity selloff**,
with a **8.20%** one-day portfolio
loss. Under a 1.5x volatility and higher-correlation simulation, 99% VaR rises
to **3.41%** and ES to
**3.92%**.

## Decision use and limitations

The four-model comparison is intended for transparent benchmarking. The
results should not be interpreted as regulatory capital, an investment
recommendation, or evidence that daily research portfolios are tradable. See
[`../docs/model_validation.md`](../docs/model_validation.md) for limitations.
