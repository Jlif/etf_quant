import os
import shutil
import tempfile

import pandas as pd
import pytest

from main import fetch_pool_data
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
