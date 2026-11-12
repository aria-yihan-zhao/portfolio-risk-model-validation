"""Deterministic and statistical portfolio stress tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .models import RiskEstimate, historical_var_es


SCENARIOS = {
    "broad_equity_selloff": {
        "Cnsmr": -0.08,
        "Manuf": -0.09,
        "HiTec": -0.10,
        "Hlth": -0.06,
        "Other": -0.08,
    },
    "technology_repricing": {
        "Cnsmr": -0.04,
        "Manuf": -0.05,
        "HiTec": -0.14,
        "Hlth": -0.03,
        "Other": -0.05,
    },
    "stagflation_shock": {
        "Cnsmr": -0.06,
        "Manuf": -0.09,
        "HiTec": -0.08,
        "Hlth": -0.04,
        "Other": -0.07,
    },
    "healthcare_drawdown": {
        "Cnsmr": -0.02,
        "Manuf": -0.02,
        "HiTec": -0.03,
        "Hlth": -0.12,
        "Other": -0.02,
    },
}


def scenario_losses(weights: pd.Series) -> pd.DataFrame:
    """Apply named one-day shocks and return positive portfolio losses."""

    if not abs(float(weights.sum()) - 1.0) < 1e-12:
        raise ValueError("weights must sum to one")
    rows = []
    for name, shocks in SCENARIOS.items():
        vector = pd.Series(shocks).reindex(weights.index)
        if vector.isna().any():
            raise ValueError(f"Scenario {name} is missing a portfolio component")
        portfolio_return = float(vector.dot(weights))
        rows.append(
            {
                "scenario": name,
                "portfolio_return": portfolio_return,
                "portfolio_loss": -portfolio_return,
            }
        )
    return pd.DataFrame(rows).set_index("scenario")


def correlation_volatility_stress(
    asset_returns: pd.DataFrame,
    weights: pd.Series,
    alpha: float = 0.01,
    volatility_multiplier: float = 1.5,
    correlation_blend: float = 0.35,
    simulations: int = 100_000,
    seed: int = 42,
) -> RiskEstimate:
    """Simulate a higher-volatility, higher-correlation Gaussian stress state."""

    if volatility_multiplier <= 1:
        raise ValueError("volatility_multiplier must exceed one")
    if not 0 <= correlation_blend <= 1:
        raise ValueError("correlation_blend must be between zero and one")
    covariance = asset_returns.cov().to_numpy(dtype=float)
    standard_deviations = np.sqrt(np.diag(covariance))
    correlation = covariance / np.outer(standard_deviations, standard_deviations)
    stressed_correlation = (1 - correlation_blend) * correlation + correlation_blend * np.ones_like(correlation)
    stressed_volatility = standard_deviations * volatility_multiplier
    stressed_covariance = np.outer(stressed_volatility, stressed_volatility) * stressed_correlation
    eigenvalues, eigenvectors = np.linalg.eigh(stressed_covariance)
    stable_covariance = eigenvectors @ np.diag(np.clip(eigenvalues, 1e-12, None)) @ eigenvectors.T

    rng = np.random.default_rng(seed)
    draws = rng.multivariate_normal(
        mean=asset_returns.mean().to_numpy(dtype=float),
        cov=stable_covariance,
        size=simulations,
    )
    portfolio_draws = draws @ weights.to_numpy(dtype=float)
    return historical_var_es(portfolio_draws, alpha=alpha)
