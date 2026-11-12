import numpy as np
import pytest

from portfolio_risk.models import filtered_historical_var_es, gaussian_var_es, historical_var_es, monte_carlo_var_es


def test_historical_tail_loss_ordering() -> None:
    returns = np.array([-0.10, -0.05, -0.02, 0.0, 0.02, 0.03])
    estimate = historical_var_es(returns, alpha=0.2)
    assert estimate.var > 0
    assert estimate.expected_shortfall >= estimate.var


def test_gaussian_matches_closed_form() -> None:
    returns = np.array([-0.02, -0.01, 0.0, 0.01, 0.02])
    estimate = gaussian_var_es(returns, alpha=0.05)
    assert estimate.var > 0
    assert estimate.expected_shortfall > estimate.var


def test_monte_carlo_is_reproducible() -> None:
    returns = np.linspace(-0.03, 0.03, 500)
    first = monte_carlo_var_es(returns, simulations=2_000, rng=np.random.default_rng(7))
    second = monte_carlo_var_es(returns, simulations=2_000, rng=np.random.default_rng(7))
    assert first == second


def test_filtered_historical_responds_to_recent_volatility() -> None:
    volatile_block = np.tile(
        np.array([-0.08, 0.07, -0.06, 0.05, -0.07, 0.06, -0.05, 0.04, -0.04, 0.03]),
        2,
    )
    calm_then_volatile = np.concatenate([np.full(100, 0.001), volatile_block])
    volatile_then_calm = calm_then_volatile[::-1]
    recent_risk = filtered_historical_var_es(calm_then_volatile, alpha=0.05)
    stale_risk = filtered_historical_var_es(volatile_then_calm, alpha=0.05)
    assert recent_risk.var > stale_risk.var
    assert recent_risk.expected_shortfall > stale_risk.expected_shortfall


def test_rejects_invalid_alpha() -> None:
    with pytest.raises(ValueError, match="alpha"):
        historical_var_es(np.array([0.0, 0.1]), alpha=0.7)
