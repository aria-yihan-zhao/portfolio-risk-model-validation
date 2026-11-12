import numpy as np
import pandas as pd
import pytest

from portfolio_risk.backtesting import (
    christoffersen_independence_test,
    kupiec_test,
    rolling_risk_forecasts,
    summarize_backtests,
)


def test_kupiec_reports_expected_counts() -> None:
    exceptions = np.array([True] * 10 + [False] * 990)
    result = kupiec_test(exceptions, alpha=0.01)
    assert result["exceptions"] == 10
    assert result["expected_exceptions"] == pytest.approx(10)
    assert result["p_value"] == pytest.approx(1.0)


def test_independence_detects_clustered_pattern() -> None:
    independent_like = np.array(([False] * 49 + [True]) * 20)
    clustered = np.array([False] * 980 + [True] * 20)
    assert christoffersen_independence_test(clustered)["p_value"] < christoffersen_independence_test(independent_like)["p_value"]


def test_rolling_forecasts_are_strictly_out_of_sample() -> None:
    index = pd.date_range("2020-01-01", periods=130, freq="B")
    returns = pd.Series(np.linspace(-0.02, 0.02, len(index)), index=index)
    forecasts = rolling_risk_forecasts(returns, window=100, simulations=1_000)
    assert forecasts.index[0] == index[100]
    assert len(forecasts) == 30
    assert {"historical_exception", "gaussian_exception", "monte_carlo_exception"}.issubset(forecasts.columns)


def test_summary_contains_all_models() -> None:
    index = pd.date_range("2020-01-01", periods=130, freq="B")
    rng = np.random.default_rng(1)
    returns = pd.Series(rng.normal(0, 0.01, len(index)), index=index)
    forecasts = rolling_risk_forecasts(returns, window=100, simulations=1_000)
    summary = summarize_backtests(forecasts)
    assert set(summary.index) == {"historical", "gaussian", "monte_carlo", "filtered_historical"}
