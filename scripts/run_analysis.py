#!/usr/bin/env python3
"""Run the end-to-end portfolio risk estimation and validation workflow."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from portfolio_risk.backtesting import rolling_risk_forecasts, summarize_backtests
from portfolio_risk.data import EXPECTED_COLUMNS, load_industry_returns, portfolio_returns, write_json
from portfolio_risk.models import filtered_historical_var_es, gaussian_var_es, historical_var_es, monte_carlo_var_es
from portfolio_risk.stress import correlation_volatility_stress, scenario_losses

ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = ROOT / "data" / "raw" / "5_Industry_Portfolios_Daily.csv"
FIGURES = ROOT / "reports" / "figures"
TABLES = ROOT / "reports" / "tables"
START_DATE = "2015-01-01"
WINDOW = 500
ALPHA = 0.01
SIMULATIONS = 3_000
WEIGHTS = pd.Series(0.20, index=EXPECTED_COLUMNS, name="weight")


def portfolio_context(returns: pd.Series) -> dict[str, float]:
    """Compute descriptive context without presenting it as a risk forecast."""

    growth = (1 + returns).cumprod()
    drawdown = growth / growth.cummax() - 1
    return {
        "annualized_return": float((1 + returns.mean()) ** 252 - 1),
        "annualized_volatility": float(returns.std(ddof=1) * np.sqrt(252)),
        "maximum_drawdown": float(drawdown.min()),
        "worst_daily_return": float(returns.min()),
    }


def latest_risk_snapshot(returns: pd.Series) -> pd.DataFrame:
    """Evaluate all models on the latest primary estimation window."""

    window = returns.iloc[-WINDOW:].to_numpy(dtype=float)
    estimates = {
        "historical": historical_var_es(window, ALPHA),
        "gaussian": gaussian_var_es(window, ALPHA),
        "monte_carlo": monte_carlo_var_es(
            window,
            ALPHA,
            simulations=100_000,
            rng=np.random.default_rng(42),
        ),
        "filtered_historical": filtered_historical_var_es(window, ALPHA),
    }
    return pd.DataFrame(
        [
            {
                "model": model,
                "var_99": estimate.var,
                "expected_shortfall_99": estimate.expected_shortfall,
            }
            for model, estimate in estimates.items()
        ]
    ).set_index("model")


def sensitivity_table(returns: pd.Series) -> pd.DataFrame:
    """Measure endpoint sensitivity to window length and confidence level."""

    rows: list[dict[str, float | int | str]] = []
    for window_size in (250, 500, 750):
        training = returns.iloc[-window_size:].to_numpy(dtype=float)
        for alpha in (0.05, 0.025, 0.01):
            estimates = {
                "historical": historical_var_es(training, alpha),
                "gaussian": gaussian_var_es(training, alpha),
                "monte_carlo": monte_carlo_var_es(
                    training,
                    alpha,
                    simulations=25_000,
                    rng=np.random.default_rng(1000 + window_size + int(alpha * 1000)),
                ),
                "filtered_historical": filtered_historical_var_es(training, alpha),
            }
            for model, estimate in estimates.items():
                rows.append(
                    {
                        "window": window_size,
                        "confidence_level": 1 - alpha,
                        "model": model,
                        "var": estimate.var,
                        "expected_shortfall": estimate.expected_shortfall,
                    }
                )
    return pd.DataFrame(rows)


def configure_plots() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 180})


def plot_rolling_backtest(forecasts: pd.DataFrame) -> None:
    visible = forecasts.loc["2019-01-01":].copy()
    losses = -visible["actual_return"] * 100
    figure, axis = plt.subplots(figsize=(13, 6.5))
    axis.plot(losses.index, losses, color="#9aa0a6", linewidth=0.8, alpha=0.75, label="Realized daily loss")
    colors = {
        "historical": "#1f77b4",
        "gaussian": "#ff7f0e",
        "monte_carlo": "#2ca02c",
        "filtered_historical": "#9467bd",
    }
    for model, color in colors.items():
        axis.plot(
            visible.index,
            visible[f"{model}_var"] * 100,
            linewidth=1.35,
            color=color,
            label=f"{model.replace('_', ' ').title()} 99% VaR",
        )
        exceptions = visible[f"{model}_exception"]
        axis.scatter(
            visible.index[exceptions],
            losses.loc[exceptions],
            s=16,
            color=color,
            alpha=0.8,
        )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_ylabel("One-day loss (%)")
    axis.set_title("Rolling Out-of-Sample 99% VaR Backtest")
    axis.legend(ncol=2, fontsize=10)
    figure.tight_layout()
    figure.savefig(FIGURES / "rolling_var_backtest.png")
    plt.close(figure)


def plot_backtest_diagnostics(summary: pd.DataFrame) -> None:
    labels = [index.replace("_", " ").title() for index in summary.index]
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    positions = np.arange(len(summary))
    axes[0].bar(positions, summary["exceptions"], color=["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"])
    axes[0].axhline(summary["expected_exceptions"].iloc[0], color="#8b0000", linestyle="--", label="Expected")
    axes[0].set_xticks(positions, labels, rotation=15)
    axes[0].set_ylabel("Exception count")
    axes[0].set_title("Observed vs. Expected Exceptions")
    axes[0].legend()

    width = 0.36
    axes[1].bar(positions - width / 2, summary["p_value"], width, label="Kupiec coverage")
    axes[1].bar(positions + width / 2, summary["independence_p_value"], width, label="Christoffersen independence")
    axes[1].axhline(0.05, color="#8b0000", linestyle="--", label="5% threshold")
    axes[1].set_xticks(positions, labels, rotation=15)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("p-value")
    axes[1].set_title("Statistical Validation Tests")
    axes[1].legend(fontsize=9)
    figure.tight_layout()
    figure.savefig(FIGURES / "backtest_diagnostics.png")
    plt.close(figure)


def plot_stress_scenarios(stresses: pd.DataFrame) -> None:
    ordered = stresses.sort_values("portfolio_loss", ascending=True)
    figure, axis = plt.subplots(figsize=(10, 5.5))
    axis.barh(
        [name.replace("_", " ").title() for name in ordered.index],
        ordered["portfolio_loss"] * 100,
        color="#b23a48",
    )
    axis.set_xlabel("One-day portfolio loss (%)")
    axis.set_title("Deterministic Stress-Scenario Losses")
    figure.tight_layout()
    figure.savefig(FIGURES / "stress_scenarios.png")
    plt.close(figure)


def plot_tail_distribution(returns: pd.Series, snapshot: pd.DataFrame) -> None:
    latest = returns.iloc[-WINDOW:] * 100
    figure, axis = plt.subplots(figsize=(10, 5.8))
    sns.histplot(latest, bins=55, stat="density", color="#6c8ebf", alpha=0.55, ax=axis)
    colors = {
        "historical": "#1f77b4",
        "gaussian": "#ff7f0e",
        "monte_carlo": "#2ca02c",
        "filtered_historical": "#9467bd",
    }
    for model, row in snapshot.iterrows():
        axis.axvline(-row["var_99"] * 100, color=colors[model], linewidth=2, label=f"{model.replace('_', ' ').title()} VaR")
    axis.set_xlabel("Daily portfolio return (%)")
    axis.set_title(f"Latest {WINDOW}-Day Return Distribution and 99% VaR")
    axis.legend(fontsize=10)
    figure.tight_layout()
    figure.savefig(FIGURES / "tail_distribution.png")
    plt.close(figure)


def write_executive_summary(
    returns: pd.Series,
    summary: pd.DataFrame,
    snapshot: pd.DataFrame,
    stresses: pd.DataFrame,
    statistical_stress: dict[str, float],
    context: dict[str, float],
) -> None:
    def format_p_value(value: float) -> str:
        return "<0.001" if value < 0.001 else f"={value:.3f}"

    validation_lines = []
    for model, row in summary.iterrows():
        coverage = "passes" if row["coverage_pass_5pct"] else "fails"
        independence = "passes" if row["independence_pass_5pct"] else "fails"
        validation_lines.append(
            f"- **{model.replace('_', ' ').title()}**: {int(row['exceptions'])} exceptions "
            f"vs. {row['expected_exceptions']:.1f} expected; {coverage} Kupiec coverage "
            f"(p{format_p_value(row['p_value'])}) and {independence} independence "
            f"(p{format_p_value(row['independence_p_value'])})."
        )
    latest_lines = [
        f"- **{model.replace('_', ' ').title()}**: VaR {row['var_99']:.2%}; ES {row['expected_shortfall_99']:.2%}."
        for model, row in snapshot.iterrows()
    ]
    worst_scenario = stresses["portfolio_loss"].idxmax()
    report = f"""# Executive summary

## Scope

An equal-weighted portfolio of five U.S. industry return series was evaluated
from **{returns.index.min().date().isoformat()}** through
**{returns.index.max().date().isoformat()}**. One-day 99% VaR and Expected
Shortfall forecasts use a strictly lagged {WINDOW}-day window.

## Portfolio context

- Annualized return: **{context['annualized_return']:.2%}**
- Annualized volatility: **{context['annualized_volatility']:.2%}**
- Maximum drawdown: **{context['maximum_drawdown']:.2%}**
- Worst daily return: **{context['worst_daily_return']:.2%}**

These are descriptive sample statistics, not forecasts.

## Latest risk snapshot

{chr(10).join(latest_lines)}

## Out-of-sample validation

{chr(10).join(validation_lines)}

Passing a test means the sample does not provide sufficient evidence to reject
that property at 5%; it is not proof that the model is correct.

## Stress testing

The largest named deterministic shock is **{worst_scenario.replace('_', ' ')}**,
with a **{stresses.loc[worst_scenario, 'portfolio_loss']:.2%}** one-day portfolio
loss. Under a 1.5x volatility and higher-correlation simulation, 99% VaR rises
to **{statistical_stress['var_99']:.2%}** and ES to
**{statistical_stress['expected_shortfall_99']:.2%}**.

## Decision use and limitations

The four-model comparison is intended for transparent benchmarking. The
results should not be interpreted as regulatory capital, an investment
recommendation, or evidence that daily research portfolios are tradable. See
[`../docs/model_validation.md`](../docs/model_validation.md) for limitations.
"""
    (ROOT / "reports" / "executive_summary.md").write_text(report, encoding="utf-8")


def main() -> None:
    if not RAW_DATA.exists():
        raise FileNotFoundError("Run scripts/download_data.py before the analysis.")
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    configure_plots()

    asset_returns = load_industry_returns(RAW_DATA, start_date=START_DATE)
    returns = portfolio_returns(asset_returns, WEIGHTS)
    forecasts = rolling_risk_forecasts(
        returns,
        window=WINDOW,
        alpha=ALPHA,
        simulations=SIMULATIONS,
        seed=42,
    )
    summary = summarize_backtests(forecasts, alpha=ALPHA)
    snapshot = latest_risk_snapshot(returns)
    sensitivity = sensitivity_table(returns)
    stresses = scenario_losses(WEIGHTS)
    statistical_stress_estimate = correlation_volatility_stress(
        asset_returns.iloc[-WINDOW:],
        WEIGHTS,
        alpha=ALPHA,
        simulations=100_000,
        seed=42,
    )
    statistical_stress = {
        "var_99": statistical_stress_estimate.var,
        "expected_shortfall_99": statistical_stress_estimate.expected_shortfall,
        "volatility_multiplier": 1.5,
        "correlation_blend": 0.35,
    }
    context = portfolio_context(returns)

    summary.to_csv(TABLES / "backtest_summary.csv", float_format="%.10f")
    snapshot.to_csv(TABLES / "latest_risk_snapshot.csv", float_format="%.10f")
    sensitivity.to_csv(TABLES / "sensitivity_analysis.csv", index=False, float_format="%.10f")
    stresses.to_csv(TABLES / "stress_scenarios.csv", float_format="%.10f")
    returns.nsmallest(10).rename("portfolio_return").to_csv(TABLES / "worst_days.csv", float_format="%.10f")
    write_json(statistical_stress, TABLES / "correlation_volatility_stress.json")
    write_json(context, TABLES / "portfolio_context.json")
    write_json(
        {
            "analysis_start": asset_returns.index.min().date().isoformat(),
            "analysis_end": asset_returns.index.max().date().isoformat(),
            "observations": int(len(asset_returns)),
            "backtest_observations": int(len(forecasts)),
            "window": WINDOW,
            "alpha": ALPHA,
            "monte_carlo_simulations_per_forecast": SIMULATIONS,
            "weights": {key: float(value) for key, value in WEIGHTS.items()},
        },
        TABLES / "analysis_manifest.json",
    )

    plot_rolling_backtest(forecasts)
    plot_backtest_diagnostics(summary)
    plot_stress_scenarios(stresses)
    plot_tail_distribution(returns, snapshot)
    write_executive_summary(returns, summary, snapshot, stresses, statistical_stress, context)

    print(f"Analysis window: {returns.index.min().date()} to {returns.index.max().date()}")
    print(f"Backtest observations: {len(forecasts):,}")
    print("\n99% VaR/ES snapshot:")
    print(snapshot.to_string(float_format=lambda value: f"{value:.4%}"))
    print("\nBacktest summary:")
    print(summary[["exceptions", "expected_exceptions", "p_value", "independence_p_value"]].to_string())


if __name__ == "__main__":
    main()
