import os
import shutil
import tempfile

import pandas as pd
import pytest

from core.orchestrator import fetch_pool_data, run_strategy
from utils.config import AppConfig, BacktestConfig, DataSourceConfig, PoolItem, StrategyConfig


class _MockDataSource:
    name = "mock"
    adjusted = True

    def __init__(self, data_map):
        self._data = data_map

    def fetch(self, code, start, end=None):
        df = self._data[code].copy()
        start_dt = pd.to_datetime(start)
        return df.loc[df.index >= start_dt]


def _make_ohlc(code, idx, price=100.0):
    return pd.DataFrame(
        {
            f"{code}_open": [price] * len(idx),
            f"{code}_high": [price * 1.01] * len(idx),
            f"{code}_low": [price * 0.99] * len(idx),
            f"{code}_close": [price] * len(idx),
        },
        index=idx,
    )


def test_dynamic_pool_effective_start_uses_earliest_etf():
    """dynamic_pool=true 时，effective_start 应从最早有数据的 ETF 开始。"""
    idx_early = pd.date_range("2020-01-02", periods=100)
    idx_late = pd.date_range("2024-11-19", periods=50)

    data_map = {
        "512040": _make_ohlc("512040", idx_early),
        "159361": _make_ohlc("159361", idx_late),
    }
    data_source = _MockDataSource(data_map)

    cache_dir = tempfile.mkdtemp()
    try:
        app_config = AppConfig(
            data_source=DataSourceConfig(provider="mock"),
            backtest=BacktestConfig(start_date="20200101", cache_dir=cache_dir),
        )
        strategy = StrategyConfig(
            name="test",
            mode="rotation",
            pool=[
                PoolItem(code="512040", name="价值100"),
                PoolItem(code="159361", name="A500"),
            ],
            params={"dynamic_pool": True, "lookback": 20},
        )

        result = fetch_pool_data(strategy, app_config, data_source)
        assert result["close"].index[0].strftime("%Y%m%d") == "20200102"
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)


def test_dynamic_pool_false_effective_start_uses_latest_etf():
    """dynamic_pool=false 时，effective_start 仍取所有 ETF 中最晚数据起始日。"""
    idx_early = pd.date_range("2020-01-02", periods=100)
    idx_late = pd.date_range("2024-11-19", periods=50)

    data_map = {
        "512040": _make_ohlc("512040", idx_early),
        "159361": _make_ohlc("159361", idx_late),
    }
    data_source = _MockDataSource(data_map)

    cache_dir = tempfile.mkdtemp()
    try:
        app_config = AppConfig(
            data_source=DataSourceConfig(provider="mock"),
            backtest=BacktestConfig(start_date="20200101", cache_dir=cache_dir),
        )
        strategy = StrategyConfig(
            name="test",
            mode="rotation",
            pool=[
                PoolItem(code="512040", name="价值100"),
                PoolItem(code="159361", name="A500"),
            ],
            params={"dynamic_pool": False, "lookback": 20},
        )

        result = fetch_pool_data(strategy, app_config, data_source)
        # 默认模式下，要等待所有 ETF 到齐并完成 lookback 预热
        assert result["close"].index[0].strftime("%Y%m%d") == "20241209"
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)


def test_fetch_pool_data_cutoff_date():
    """cutoff_date 参数应正确截断数据。"""
    idx = pd.date_range("2020-01-02", periods=100)
    data_map = {
        "512040": _make_ohlc("512040", idx),
    }
    data_source = _MockDataSource(data_map)

    cache_dir = tempfile.mkdtemp()
    try:
        app_config = AppConfig(
            data_source=DataSourceConfig(provider="mock"),
            backtest=BacktestConfig(start_date="20200101", cache_dir=cache_dir),
        )
        strategy = StrategyConfig(
            name="test",
            mode="rotation",
            pool=[PoolItem(code="512040", name="价值100")],
            params={"lookback": 20},
        )

        cutoff = pd.Timestamp("2020-03-15")
        result = fetch_pool_data(strategy, app_config, data_source, cutoff_date=cutoff)
        assert result["close"].index[-1].date() == cutoff.date()
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)


def test_fetch_pool_data_start_date_override():
    """start_date 参数应覆盖配置中的起始日，并尊重非 dynamic_pool 的预热窗口。"""
    idx = pd.date_range("2020-01-02", periods=100)
    data_map = {
        "512040": _make_ohlc("512040", idx),
    }
    data_source = _MockDataSource(data_map)

    cache_dir = tempfile.mkdtemp()
    try:
        app_config = AppConfig(
            data_source=DataSourceConfig(provider="mock"),
            backtest=BacktestConfig(start_date="20200101", cache_dir=cache_dir),
        )
        strategy = StrategyConfig(
            name="test",
            mode="rotation",
            pool=[PoolItem(code="512040", name="价值100")],
            params={"lookback": 20},
        )

        result = fetch_pool_data(
            strategy, app_config, data_source, start_date="20200301"
        )
        # start_date 被覆盖后，数据从 2020-03-01 开始；默认 momentum 需要 lookback+1=21 天预热
        assert result["close"].index[0].strftime("%Y%m%d") == "20200321"
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)


def test_run_strategy_passes_name_types_to_rotation():
    """run_strategy 应始终将 name_types 传给 rotation.run，使 adaptive_scoring 生效。"""
    idx = pd.date_range("2020-01-02", periods=120)
    # 让成长 ETF 价格单调上涨，确保能被选中
    prices = 100 + pd.Series(range(len(idx)), index=idx) * 0.1
    df = pd.DataFrame(
        {
            "510300_open": prices,
            "510300_high": prices * 1.01,
            "510300_low": prices * 0.99,
            "510300_close": prices,
        },
        index=idx,
    )
    data_map = {"510300": df}
    data_source = _MockDataSource(data_map)

    cache_dir = tempfile.mkdtemp()
    try:
        app_config = AppConfig(
            data_source=DataSourceConfig(provider="mock"),
            backtest=BacktestConfig(start_date="20200101", cache_dir=cache_dir),
        )
        strategy = StrategyConfig(
            name="test",
            mode="rotation",
            pool=[PoolItem(code="510300", name="成长ETF", type="成长")],
            params={
                "lookback": 20,
                "top_n": 1,
                "adaptive_scoring": True,
                "scoring": "momentum",
            },
        )

        result, name_list = run_strategy(strategy, app_config, data_source, silent=True)
        assert "轮动策略净值" in result.columns
        assert name_list == ["成长ETF"]
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)


def test_run_strategy_weighted_mode():
    """run_strategy 应支持 weighted 模式。"""
    idx = pd.date_range("2020-01-02", periods=100)
    data_map = {
        "510300": _make_ohlc("510300", idx, price=100.0),
        "510880": _make_ohlc("510880", idx, price=50.0),
    }
    data_source = _MockDataSource(data_map)

    cache_dir = tempfile.mkdtemp()
    try:
        app_config = AppConfig(
            data_source=DataSourceConfig(provider="mock"),
            backtest=BacktestConfig(start_date="20200101", cache_dir=cache_dir),
        )
        strategy = StrategyConfig(
            name="test",
            mode="weighted",
            pool=[
                PoolItem(code="510300", name="沪深300", weight=60),
                PoolItem(code="510880", name="红利ETF", weight=40),
            ],
            params={"rebalance_freq": 1},
        )

        result, name_list = run_strategy(strategy, app_config, data_source, silent=True)
        assert "轮动策略净值" in result.columns
        assert set(name_list) == {"沪深300", "红利ETF"}
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)
