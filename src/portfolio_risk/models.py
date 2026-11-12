"""Value at Risk and Expected Shortfall estimators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class RiskEstimate:
    """Positive loss magnitudes for one-day VaR and Expected Shortfall."""

    var: float
    expected_shortfall: float


def _validate_inputs(returns: np.ndarray, alpha: float) -> np.ndarray:
    values = np.asarray(returns, dtype=float)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("returns must be a one-dimensional array with at least two values")
    if not np.isfinite(values).all():
        raise ValueError("returns must be finite")
    if not 0 < alpha < 0.5:
        raise ValueError("alpha must be between zero and 0.5")
    return values


def historical_var_es(returns: np.ndarray, alpha: float = 0.01) -> RiskEstimate:
    """Estimate tail risk from the empirical lower tail."""

    values = _validate_inputs(returns, alpha)
    quantile = float(np.quantile(values, alpha, method="linear"))
    tail = values[values <= quantile]
    return RiskEstimate(var=max(0.0, -quantile), expected_shortfall=max(0.0, -float(tail.mean())))


def gaussian_var_es(returns: np.ndarray, alpha: float = 0.01) -> RiskEstimate:
    """Estimate tail risk under a Gaussian return assumption."""

    values = _validate_inputs(returns, alpha)
    mean = float(values.mean())
    sigma = float(values.std(ddof=1))
    z_score = float(stats.norm.ppf(alpha))
    quantile = mean + sigma * z_score
    conditional_tail_mean = mean - sigma * float(stats.norm.pdf(z_score)) / alpha
    return RiskEstimate(
        var=max(0.0, -quantile),
        expected_shortfall=max(0.0, -conditional_tail_mean),
    )


def filtered_historical_var_es(
    returns: np.ndarray,
    alpha: float = 0.01,
    decay: float = 0.94,
) -> RiskEstimate:
    """Scale empirical residual tails by an EWMA volatility forecast."""

    values = _validate_inputs(returns, alpha)
    if not 0 < decay < 1:
        raise ValueError("decay must be between zero and one")
    initialization = min(50, len(values))
    initial_variance = float(values[:initialization].var(ddof=1))
    variance_forecast = np.empty(len(values) + 1)
    variance_forecast[0] = max(initial_variance, 1e-12)
    for index, value in enumerate(values):
        variance_forecast[index + 1] = (
            decay * variance_forecast[index] + (1 - decay) * value**2
        )
    residuals = values / np.sqrt(variance_forecast[:-1])
    residual_risk = historical_var_es(residuals, alpha)
    next_volatility = float(np.sqrt(variance_forecast[-1]))
    return RiskEstimate(
        var=residual_risk.var * next_volatility,
        expected_shortfall=residual_risk.expected_shortfall * next_volatility,
    )


def monte_carlo_var_es(
    returns: np.ndarray,
    alpha: float = 0.01,
    simulations: int = 5_000,
    degrees_of_freedom: int = 6,
    rng: np.random.Generator | None = None,
) -> RiskEstimate:
    """Estimate risk from a variance-matched Student-t Monte Carlo challenger."""

    values = _validate_inputs(returns, alpha)
    if simulations < 1_000:
        raise ValueError("simulations must be at least 1,000")
    if degrees_of_freedom <= 2:
        raise ValueError("degrees_of_freedom must exceed two for finite variance")
    generator = rng if rng is not None else np.random.default_rng(42)
    mean = float(values.mean())
    sigma = float(values.std(ddof=1))
    scale = sigma * np.sqrt((degrees_of_freedom - 2) / degrees_of_freedom)
    draws = mean + scale * generator.standard_t(degrees_of_freedom, size=simulations)
    return historical_var_es(draws, alpha=alpha)
