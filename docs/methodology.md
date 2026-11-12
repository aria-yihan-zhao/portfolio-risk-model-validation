# Methodology

## Portfolio

The analysis uses the value-weighted daily returns of five U.S. industry
portfolios: Consumer, Manufacturing, High Technology, Healthcare, and Other.
The modeled portfolio holds 20% in each series and is not re-optimized using
future observations. The industry series are research portfolios rather than
directly tradable securities.

## Forecast timing

Every forecast for date *t* uses only the preceding estimation window. The
realized return on *t* is kept outside estimation and is used solely to evaluate
the forecast. This prevents look-ahead leakage.

## Risk estimators

1. **Historical simulation** uses the empirical 1% return quantile and the mean
   return below that quantile.
2. **Gaussian parametric** uses the rolling sample mean and standard deviation
   with the analytical normal VaR and ES formulas.
3. **Monte Carlo challenger** simulates a Student-t distribution with six
   degrees of freedom. Its scale is adjusted so its variance matches the rolling
   sample variance. A fixed seed makes all results reproducible.
4. **Filtered historical remediation** uses a 0.94 EWMA variance filter,
   standardizes the historical returns, and rescales their empirical tail by
   the next-day volatility forecast. This preserves empirical tail shape while
   reacting faster to volatility regime changes.

VaR and ES are reported as positive one-day loss magnitudes. The primary
configuration uses a 500-observation estimation window and a 99% confidence
level (`alpha = 0.01`).

## Validation

The Kupiec test evaluates whether the observed exception rate is consistent
with the nominal tail probability. The Christoffersen independence test asks
whether exceptions cluster. Failing either test at 5% is treated as evidence
against the corresponding aspect of model adequacy, not as proof that another
model is correct.

The project also reports sensitivity across estimation windows and confidence
levels, deterministic scenario losses, and a joint volatility/correlation
stress. See [`model_validation.md`](model_validation.md) for interpretation.
