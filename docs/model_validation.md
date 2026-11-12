# Model validation interpretation

This project uses multiple models because agreement and disagreement are both
informative. The historical estimator is assumption-light but responds slowly
when an old crisis observation leaves the rolling window. The Gaussian model is
stable and transparent but can understate heavy tails. The Student-t Monte Carlo
challenger explicitly allows heavy tails, but its fixed distributional family
and degrees of freedom remain modeling assumptions. The filtered historical
remediation adapts faster to current volatility and preserves the empirical
residual tail, but its 0.94 decay parameter remains a choice that requires
challenge.

## Decision rules

- **Coverage:** Kupiec p-value below 0.05 indicates that the exception frequency
  is inconsistent with the nominal rate.
- **Independence:** Christoffersen p-value below 0.05 indicates exception
  clustering and potential volatility-regime misspecification.
- **Expected Shortfall:** realized exception severity is compared with average
  forecast ES as a descriptive diagnostic; this is not a formal ES backtest.
- **Sensitivity:** a conclusion that changes materially across reasonable
  windows or confidence levels is treated as unstable.

## Observed validation results

The 2015–2026 backtest contains 2,453 one-day forecasts, so a correctly
calibrated 99% VaR model should produce approximately 24.5 exceptions.

| Model | Exceptions | Coverage | Independence |
|---|---:|---|---|
| Historical | 37 | Reject (p=0.019) | Reject (p<0.001) |
| Gaussian | 59 | Reject (p<0.001) | Reject (p<0.001) |
| Monte Carlo Student-t | 54 | Reject (p<0.001) | Reject (p<0.001) |
| Filtered historical | 30 | Do not reject (p=0.283) | Reject (p<0.001) |

Static-window models understate risk across changing volatility regimes. The
filtered historical remediation brings unconditional coverage into an
acceptable range, but it does not eliminate exception clustering. It should
therefore be described as a partial remediation rather than a fully validated
production model. A natural next extension would test a dynamic volatility
model and a conditional-coverage-aware escalation policy on a separately
locked evaluation period.

## Known limitations

- Fixed portfolio weights omit rebalancing and changing exposures.
- Daily close-to-close returns omit intraday and liquidity risk.
- Industry research portfolios are not directly tradable instruments.
- Transaction costs, nonlinear instruments, and regulatory capital rules are
  outside scope.
- The Monte Carlo model does not estimate dynamic volatility or tail dependence.
- Statistical non-rejection is not proof of model correctness.

The repository is designed as a transparent validation exercise, not a
production risk engine or investment recommendation.
