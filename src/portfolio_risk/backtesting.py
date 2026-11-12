"""Rolling risk forecasts and statistical exception tests."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from .models import filtered_historical_var_es, gaussian_var_es, historical_var_es, monte_carlo_var_es

MODEL_NAMES = ("historical", "gaussian", "monte_carlo", "filtered_historical")


def _bernoulli_log_likelihood(successes: int, trials: int, probability: float) -> float:
    if not 0 <= successes <= trials:
        raise ValueError("successes must be between zero and trials")
    if not 0 <= probability <= 1:
        raise ValueError("probability must be between zero and one")
    failures = trials - successes
    result = 0.0
    if successes:
        if probability == 0:
            return -math.inf
        result += successes * math.log(probability)
    if failures:
        if probability == 1:
            return -math.inf
        result += failures * math.log1p(-probability)
    return result


def kupiec_test(exceptions: np.ndarray, alpha: float) -> dict[str, float | int]:
    """Kupiec proportion-of-failures test for unconditional coverage."""

    observed = np.asarray(exceptions, dtype=bool)
    trials = int(len(observed))
    if trials == 0:
        raise ValueError("At least one exception indicator is required")
    failures = int(observed.sum())
    observed_rate = failures / trials
    null_ll = _bernoulli_log_likelihood(failures, trials, alpha)
    fitted_ll = _bernoulli_log_likelihood(failures, trials, observed_rate)
    statistic = max(0.0, -2.0 * (null_ll - fitted_ll))
    return {
        "observations": trials,
        "exceptions": failures,
        "expected_exceptions": trials * alpha,
        "exception_rate": observed_rate,
        "lr_statistic": statistic,
        "p_value": float(stats.chi2.sf(statistic, df=1)),
    }


def christoffersen_independence_test(exceptions: np.ndarray) -> dict[str, float | int]:
    """Test whether VaR exceptions are independent through time."""

    observed = np.asarray(exceptions, dtype=int)
    if len(observed) < 2:
        raise ValueError("At least two exception indicators are required")
    previous, current = observed[:-1], observed[1:]
    n00 = int(((previous == 0) & (current == 0)).sum())
    n01 = int(((previous == 0) & (current == 1)).sum())
    n10 = int(((previous == 1) & (current == 0)).sum())
    n11 = int(((previous == 1) & (current == 1)).sum())
    total_transitions = n00 + n01 + n10 + n11
    pooled = (n01 + n11) / total_transitions
    p01 = n01 / (n00 + n01) if n00 + n01 else 0.0
    p11 = n11 / (n10 + n11) if n10 + n11 else 0.0

    null_ll = _bernoulli_log_likelihood(n01 + n11, total_transitions, pooled)
    alternative_ll = _bernoulli_log_likelihood(n01, n00 + n01, p01)
    alternative_ll += _bernoulli_log_likelihood(n11, n10 + n11, p11)
    statistic = max(0.0, -2.0 * (null_ll - alternative_ll))
    return {
        "n00": n00,
        "n01": n01,
        "n10": n10,
        "n11": n11,
        "lr_statistic": statistic,
        "p_value": float(stats.chi2.sf(statistic, df=1)),
    }


def rolling_risk_forecasts(
    returns: pd.Series,
    window: int = 500,
    alpha: float = 0.01,
    simulations: int = 3_000,
    degrees_of_freedom: int = 6,
    seed: int = 42,
) -> pd.DataFrame:
    """Create strictly lagged one-day risk forecasts for three estimators."""

    if window < 100:
        raise ValueError("window must be at least 100 observations")
    if len(returns) <= window:
        raise ValueError("returns must contain more observations than the estimation window")
    if returns.isna().any():
        raise ValueError("returns cannot contain missing values")

    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | pd.Timestamp]] = []
    values = returns.to_numpy(dtype=float)
    for position in range(window, len(returns)):
        training = values[position - window : position]
        historical = historical_var_es(training, alpha)
        gaussian = gaussian_var_es(training, alpha)
        monte_carlo = monte_carlo_var_es(
            training,
            alpha,
            simulations=simulations,
            degrees_of_freedom=degrees_of_freedom,
            rng=rng,
        )
        filtered_historical = filtered_historical_var_es(training, alpha)
        rows.append(
            {
                "date": returns.index[position],
                "actual_return": values[position],
                "historical_var": historical.var,
                "historical_es": historical.expected_shortfall,
                "gaussian_var": gaussian.var,
                "gaussian_es": gaussian.expected_shortfall,
                "monte_carlo_var": monte_carlo.var,
                "monte_carlo_es": monte_carlo.expected_shortfall,
                "filtered_historical_var": filtered_historical.var,
                "filtered_historical_es": filtered_historical.expected_shortfall,
            }
        )
    result = pd.DataFrame(rows).set_index("date")
    for model in MODEL_NAMES:
        result[f"{model}_exception"] = result["actual_return"] < -result[f"{model}_var"]
    return result


def summarize_backtests(forecasts: pd.DataFrame, alpha: float = 0.01) -> pd.DataFrame:
    """Summarize exception frequency, clustering, and realized tail severity."""

    rows: list[dict[str, float | int | str]] = []
    for model in MODEL_NAMES:
        exceptions = forecasts[f"{model}_exception"].to_numpy(dtype=bool)
        kupiec = kupiec_test(exceptions, alpha)
        independence = christoffersen_independence_test(exceptions)
        tail_losses = -forecasts.loc[exceptions, "actual_return"]
        mean_es = float(forecasts[f"{model}_es"].mean())
        rows.append(
            {
                "model": model,
                **kupiec,
                "independence_lr": independence["lr_statistic"],
                "independence_p_value": independence["p_value"],
                "mean_forecast_es": mean_es,
                "mean_realized_exception_loss": float(tail_losses.mean()) if len(tail_losses) else np.nan,
                "coverage_pass_5pct": bool(kupiec["p_value"] >= 0.05),
                "independence_pass_5pct": bool(independence["p_value"] >= 0.05),
            }
        )
    return pd.DataFrame(rows).set_index("model")
