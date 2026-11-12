from pathlib import Path

import pandas as pd
import pytest

from portfolio_risk.data import EXPECTED_COLUMNS, load_industry_returns, portfolio_returns


def test_loads_value_weighted_daily_section(tmp_path: Path) -> None:
    sample = """description\n Average Value Weighted Returns -- Daily\n,Cnsmr,Manuf,HiTec,Hlth,Other\n20240102,1.00,2.00,-1.00,0.50,0.25\n20240103,-2.00,1.00,0.00,1.50,0.50\n\n Average Equal Weighted Returns -- Daily\n"""
    path = tmp_path / "sample.csv"
    path.write_text(sample, encoding="utf-8")
    frame = load_industry_returns(path)
    assert list(frame.columns) == EXPECTED_COLUMNS
    assert frame.loc[pd.Timestamp("2024-01-02"), "Cnsmr"] == pytest.approx(0.01)
    assert len(frame) == 2


def test_portfolio_requires_aligned_weights() -> None:
    frame = pd.DataFrame([[0.01, 0.02]], columns=["a", "b"])
    with pytest.raises(ValueError, match="match"):
        portfolio_returns(frame, pd.Series({"b": 0.5, "a": 0.5}))


def test_equal_weight_portfolio_return() -> None:
    frame = pd.DataFrame([[0.01, 0.03]], columns=["a", "b"])
    result = portfolio_returns(frame, pd.Series({"a": 0.5, "b": 0.5}))
    assert result.iloc[0] == pytest.approx(0.02)
