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
    etf = _price_series(np.linspace(100, 120, 80))
    benchmark = _price_series(np.linspace(100, 102, 80))
    score = adaptive_momentum_score(etf, etf_type="行业股票", benchmark_series=benchmark, lookback=20)
    assert score > 0


def test_dividend_risk_adjusted_score():
    prices = _price_series(np.linspace(100, 110, 41))
    score = adaptive_momentum_score(prices, etf_type="红利", lookback=40)
    assert score > 0


def test_commodity_trend_score():
    prices = _price_series(np.linspace(100, 120, 61))
    score = adaptive_momentum_score(prices, etf_type="商品", lookback=60)
    assert score > 0


def test_broad_breakout_score_near_one_at_high():
    prices = _price_series([100.0] * 251 + [120.0])
    score = adaptive_momentum_score(prices, etf_type="宽基", lookback=252)
    assert score == pytest.approx(1.0)
