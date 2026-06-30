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
    score = adaptive_momentum_score(etf, etf_type="行业股票", benchmark_series=benchmark, lookback=20)
    assert score > 0


def test_dividend_risk_adjusted_score():
    prices = _price_series(np.linspace(100, 110, 61))
    score = adaptive_momentum_score(prices, etf_type="红利", lookback=60)
    assert score > 0


def test_value_types_use_sixty_day_risk_adjusted():
    prices = _price_series(np.linspace(100, 110, 61))
    for etf_type in ("红利", "自由现金流", "价值"):
        score = adaptive_momentum_score(prices, etf_type=etf_type, lookback=60)
        assert score > 0, etf_type


def test_value_factor_multiplier_scales_score():
    prices = _price_series(np.linspace(100, 110, 61))
    base = adaptive_momentum_score(prices, etf_type="价值", lookback=60)
    boosted = adaptive_momentum_score(
        prices, etf_type="价值", lookback=60, factor_multiplier=1.5
    )
    assert boosted == pytest.approx(base * 1.5)


def test_growth_momentum_score():
    # 近5日加速上涨的价格序列
    prices = _price_series(np.concatenate([np.linspace(100, 105, 16), np.linspace(105, 120, 5)]))
    score = adaptive_momentum_score(prices, etf_type="成长", lookback=20)
    assert score > 0


def test_commodity_trend_score():
    prices = _price_series(np.linspace(100, 120, 61))
    score = adaptive_momentum_score(prices, etf_type="商品", lookback=60)
    assert score > 0


def test_broad_breakout_score_near_one_at_high():
    prices = _price_series([100.0] * 251 + [120.0])
    score = adaptive_momentum_score(prices, etf_type="宽基", lookback=252)
    assert score == pytest.approx(1.0)
