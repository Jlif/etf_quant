import numpy as np
import pandas as pd
import pytest

from strategy.rotation import run


def _make_data(n=300, names=("沪深300ETF", "红利ETF")):
    idx = pd.date_range("2023-01-01", periods=n)
    close = pd.DataFrame(
        {name: np.linspace(100, 100 + i * 10, n) for i, name in enumerate(names)},
        index=idx,
    )
    open_ = close.copy()
    high = close * 1.01
    low = close * 0.99
    return {"close": close, "open": open_, "high": high, "low": low}


def test_adaptive_scoring_sector_requires_benchmark():
    data = _make_data()
    params = {
        "lookback": 20,
        "scoring": "momentum",
        "adaptive_scoring": True,
        "top_n": 1,
    }
    name_types = {"沪深300ETF": "行业股票", "红利ETF": "红利"}
    with pytest.raises(ValueError, match="params\\['benchmark'\\] 必须提供"):
        run(data, list(data["close"].columns), params, name_types=name_types)


def test_adaptive_scoring_sector_requires_benchmark_in_name_list():
    data = _make_data()
    params = {
        "lookback": 20,
        "scoring": "momentum",
        "adaptive_scoring": True,
        "benchmark": "中证500ETF",
        "top_n": 1,
    }
    name_types = {"沪深300ETF": "行业股票", "红利ETF": "红利"}
    with pytest.raises(ValueError, match="params\\['benchmark'\\]=.* 不在 name_list"):
        run(data, list(data["close"].columns), params, name_types=name_types)


def test_adaptive_scoring_uses_type_scores():
    data = _make_data()
    params = {
        "lookback": 20,
        "scoring": "momentum",
        "adaptive_scoring": True,
        "benchmark": "沪深300ETF",
        "top_n": 1,
    }
    name_types = {"沪深300ETF": "宽基", "红利ETF": "红利"}
    result = run(data, list(data["close"].columns), params, name_types=name_types)
    assert "轮动策略净值" in result.columns
    assert result["轮动策略净值"].iloc[-1] > 0


def test_adaptive_scoring_without_sector_allows_missing_benchmark():
    data = _make_data()
    params = {
        "lookback": 20,
        "scoring": "momentum",
        "adaptive_scoring": True,
        "top_n": 1,
    }
    name_types = {"沪深300ETF": "宽基", "红利ETF": "红利"}
    result = run(data, list(data["close"].columns), params, name_types=name_types)
    assert "轮动策略净值" in result.columns
    assert result["轮动策略净值"].iloc[-1] > 0
