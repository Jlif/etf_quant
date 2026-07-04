import numpy as np
import pandas as pd
import pytest

from core.scorer import adaptive_momentum_score, momentum_score


def _price_series(values):
    return pd.Series(values, index=pd.date_range("2024-01-01", periods=len(values)))


def test_unknown_type_falls_back_to_momentum():
    prices = _price_series([100.0, 101.0, 102.0, 103.0, 104.0] * 5)
    score = adaptive_momentum_score(prices, etf_type="未知类型", lookback=20)
    expected = momentum_score(prices, lookback=20)
    assert score == pytest.approx(expected)


def test_sector_residual_momentum_positive_when_outperforming():
    # ETF 持续上涨，benchmark 横盘
    etf = _price_series(np.linspace(100, 120, 62))
    benchmark = _price_series(np.linspace(100, 102, 62))
    score = adaptive_momentum_score(etf, etf_type="行业", benchmark_series=benchmark, lookback=20)
    assert score > 0


def test_dividend_risk_adjusted_score():
    prices = _price_series(np.linspace(100, 110, 41))
    score = adaptive_momentum_score(prices, etf_type="红利", lookback=40)
    assert score > 0


def test_free_cash_flow_uses_same_method_as_dividend():
    prices = _price_series(np.linspace(100, 110, 41))
    dividend = adaptive_momentum_score(prices, etf_type="红利", lookback=40)
    fcf = adaptive_momentum_score(prices, etf_type="自由现金流", lookback=40)
    assert fcf == pytest.approx(dividend)


def test_growth_momentum_score():
    # 近5日加速上涨的价格序列
    prices = _price_series(np.concatenate([np.linspace(100, 105, 16), np.linspace(105, 120, 5)]))
    score = adaptive_momentum_score(prices, etf_type="成长", lookback=20)
    assert score > 0


def test_commodity_trend_score():
    prices = _price_series(np.linspace(100, 120, 61))
    score = adaptive_momentum_score(prices, etf_type="商品", lookback=60)
    assert score > 0


def test_broad_breakout_score_binary():
    # 突破252日新高 -> 1.0
    at_high = _price_series([100.0] * 251 + [120.0])
    assert adaptive_momentum_score(at_high, etf_type="宽基", lookback=252) == pytest.approx(1.0)
    # 未突破 -> 0.0
    below_high = _price_series([100.0] * 251 + [99.0])
    assert adaptive_momentum_score(below_high, etf_type="宽基", lookback=252) == pytest.approx(0.0)
