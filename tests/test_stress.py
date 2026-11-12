import numpy as np
import pandas as pd
import pytest

from portfolio_risk.stress import correlation_volatility_stress, scenario_losses


def weights() -> pd.Series:
    return pd.Series(0.2, index=["Cnsmr", "Manuf", "HiTec", "Hlth", "Other"])


def test_named_scenarios_produce_positive_losses() -> None:
    result = scenario_losses(weights())
    assert len(result) == 4
    assert (result["portfolio_loss"] > 0).all()


def test_correlation_stress_returns_ordered_tail_metrics() -> None:
    rng = np.random.default_rng(3)
    returns = pd.DataFrame(
        rng.normal(0, 0.01, size=(600, 5)),
        columns=weights().index,
    )
    estimate = correlation_volatility_stress(returns, weights(), simulations=2_000)
    assert estimate.expected_shortfall >= estimate.var > 0


def test_stress_requires_amplified_volatility() -> None:
    frame = pd.DataFrame(np.eye(5), columns=weights().index)
    with pytest.raises(ValueError, match="exceed"):
        correlation_volatility_stress(frame, weights(), volatility_multiplier=1.0)
