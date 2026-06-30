import warnings

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


def test_adaptive_scoring_non_broad_index_does_not_require_252_days():
    """非宽基组合不应被强制要求 252 天预热期。"""
    n = 82  # 小于 252，但大于商品所需的 61
    data = _make_data(n=n, names=("红利ETF", "商品ETF"))
    # 给红利 ETF 一些涨幅，避免波动率为 0 导致得分 NaN
    data["close"]["红利ETF"] = np.linspace(100, 120, n)
    data["open"]["红利ETF"] = data["close"]["红利ETF"].copy()
    data["high"]["红利ETF"] = data["close"]["红利ETF"] * 1.01
    data["low"]["红利ETF"] = data["close"]["红利ETF"] * 0.99
    params = {
        "lookback": 20,
        "scoring": "momentum",
        "adaptive_scoring": True,
        "top_n": 1,
    }
    name_types = {"红利ETF": "红利", "商品ETF": "商品"}
    result = run(data, list(data["close"].columns), params, name_types=name_types)
    assert len(result) > 0
    assert "轮动策略净值" in result.columns
    assert result["轮动策略净值"].iloc[-1] > 0


def test_adaptive_scoring_unknown_type_uses_lookback():
    """未知/缺失类型应使用默认 lookback，而不是 252。"""
    n = 50  # 小于 252，但大于默认 lookback 20
    data = _make_data(n=n, names=("未知ETF", "缺失ETF"))
    # 给 ETF 一些涨幅，避免波动率为 0 导致得分 NaN
    for name in ("未知ETF", "缺失ETF"):
        data["close"][name] = np.linspace(100, 110, n)
        data["open"][name] = data["close"][name].copy()
        data["high"][name] = data["close"][name] * 1.01
        data["low"][name] = data["close"][name] * 0.99
    params = {
        "lookback": 20,
        "scoring": "momentum",
        "adaptive_scoring": True,
        "top_n": 1,
    }
    name_types = {"未知ETF": "外星ETF"}  # 缺失类型不传入
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = run(
            data,
            list(data["close"].columns),
            params,
            name_types=name_types,
        )
    assert len(result) > 0
    assert "轮动策略净值" in result.columns
    assert result["轮动策略净值"].iloc[-1] > 0
