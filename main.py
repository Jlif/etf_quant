#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETF 轮动策略回测系统入口

支持通过 config.yaml 配置策略参数、ETF 代码、仓位占比

用法:
    python main.py
    python main.py --config my_config.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings

import pandas as pd

# 确保本地模块可导入
sys.path.insert(0, os.path.dirname(__file__))

from data_source import get_data_source
from core import (
    performance_report,
    plot_nav_curves,
    plot_strategy_comparison,
    print_summary,
)
from strategy import rotation, weighted
from utils import load_config, AppConfig, StrategyConfig

warnings.filterwarnings("ignore", category=RuntimeWarning)


def detect_and_fix_price_jumps(
    prices: pd.Series,
    name: str,
    threshold: float = 0.30,
) -> pd.Series:
    """
    检测并修正价格序列中的异常复权跳空。

    yfinance 等数据源对国内 ETF 的复权处理偶尔出错，
    会出现单日涨跌幅远超正常范围（如 -50%）的虚假跳空。
    本函数把这些点当作"复权系数错误"，对前期价格做整体缩放，
    使修正后的序列保持连续。

    Parameters
    ----------
    prices : pd.Series
        收盘价序列
    name : str
        标的名称，用于日志
    threshold : float
        异常阈值，日收益率绝对值超过此值即认为异常

    Returns
    -------
    pd.Series
        修正后的价格序列
    """
    prices = prices.copy().sort_index()
    returns = prices.pct_change().dropna()

    fixed = prices.copy()
    # 从最早日期开始扫描，避免多次修正相互影响
    for date in returns.index:
        daily_ret = returns.loc[date]
        if abs(daily_ret) > threshold:
            prev_date = returns.index[returns.index.get_loc(date) - 1]
            factor = fixed.loc[date] / fixed.loc[prev_date]
            # 将 prev_date 及之前所有价格乘以 factor，使序列连续
            mask = fixed.index <= prev_date
            fixed.loc[mask] *= factor
            direction = "下跌" if daily_ret < 0 else "上涨"
            print(
                f"  [数据修正] {name} 在 {date.date()} 出现异常{direction} "
                f"({daily_ret:+.2%})，已整体缩放前期价格 (factor={factor:.4f})"
            )
            # 重新计算后续收益率
            returns = fixed.pct_change().dropna()

    return fixed


def fetch_pool_data(strategy: StrategyConfig, app_config: AppConfig, data_source):
    """获取策略候选池数据，自动对齐起始日期"""
    codes = [p.code for p in strategy.pool]
    names = {p.code: p.name for p in strategy.pool}
    cache_dir = app_config.backtest.cache_dir
    os.makedirs(cache_dir, exist_ok=True)

    # 策略级起始日优先于全局起始日
    target_start = strategy.start_date or app_config.backtest.start_date
    target_start_dt = pd.to_datetime(target_start)

    all_data = {}
    actual_starts = {}  # 记录每个 ETF 实际数据起始日
    for code in codes:
        name = names[code]
        cache_file = os.path.join(cache_dir, f"{code}_{data_source.name}.csv")

        if os.path.exists(cache_file):
            print(f"  [缓存] {code} ({name})")
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        else:
            print(f"  [下载] {code} ({name}) via {data_source.name}")
            df = data_source.fetch(code, target_start)
            df.to_csv(cache_file)

        all_data[name] = df[code]
        actual_starts[name] = df.index[0]

    # 异常复权跳空修正
    for name in all_data:
        all_data[name] = detect_and_fix_price_jumps(all_data[name], name)

    data = pd.DataFrame(all_data)

    # 策略实际起始日 = max(配置起始日, 所有 ETF 中最晚的数据起始日)
    latest_etf_start = max(actual_starts.values())
    effective_start = max(target_start_dt, latest_etf_start)

    if latest_etf_start > target_start_dt:
        print(f"  [注意] 配置起始日 {target_start_dt.date()} 早于部分 ETF 数据起始日")
        for name, st in actual_starts.items():
            if st > target_start_dt:
                print(f"         {name} 实际起始: {st.date()}")

    if effective_start != data.index[0]:
        print(f"  [调整] 策略实际起始日: {effective_start.date()}")
        data = data.loc[data.index >= effective_start]

    print(f"  时间范围: {data.index[0].date()} ~ {data.index[-1].date()}, 共 {len(data)} 条")
    return data


def run_strategy(strategy: StrategyConfig, app_config: AppConfig, data_source):
    """执行单个策略回测"""
    print(f"\n{'='*60}")
    print(f"【{strategy.name}】{strategy.description}")
    print(f"  模式: {strategy.mode} | 参数: {strategy.params}")
    print(f"{'='*60}")

    data = fetch_pool_data(strategy, app_config, data_source)
    name_list = data.columns.tolist()

    if strategy.mode == "rotation":
        result = rotation.run(data, name_list, strategy.params)
        # 以第一个标的为基准
        benchmark_col = f"{name_list[0]}净值"
        for name in name_list:
            result[f"{name}净值"] = result[name] / result[name].iloc[0]
    elif strategy.mode == "weighted":
        weights = {p.name: p.weight for p in strategy.pool}
        result = weighted.run(data, name_list, weights, strategy.params)
        for name in name_list:
            result[f"{name}净值"] = result[name] / result[name].iloc[0]
        benchmark_col = None
    else:
        raise ValueError(f"不支持的模式: {strategy.mode}")

    # 绩效报告
    performance_report(
        result["轮动策略净值"],
        benchmark=result[benchmark_col] if benchmark_col and benchmark_col in result.columns else None,
        title=f"{strategy.name}回测报告",
    )

    return result, name_list


def main():
    parser = argparse.ArgumentParser(description="ETF 轮动策略回测")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    args = parser.parse_args()

    # 加载配置
    app_config = load_config(args.config)
    print("="*60)
    print("ETF 轮动策略回测系统")
    print(f"数据源: {app_config.data_source.provider}")
    print(f"策略数: {len(app_config.strategies)}")
    print("="*60)

    # 初始化数据源
    data_source = get_data_source(
        name=app_config.data_source.provider,
        fallback=True,
    )

    # 运行启用的策略
    all_results = {}
    nav_series = {}
    for strategy in app_config.strategies:
        if not strategy.enabled:
            print(f"\n[跳过] 策略 '{strategy.name}' (enabled=false)")
            continue
        result, name_list = run_strategy(strategy, app_config, data_source)
        all_results[strategy.name] = (result, name_list)
        nav_series[strategy.name] = result["轮动策略净值"]

    # 可视化
    plot_nav_curves(
        {k: (v[0], v[1], "轮动策略净值") for k, v in all_results.items()}
    )
    plot_strategy_comparison(nav_series)
    print_summary(nav_series)


if __name__ == "__main__":
    main()
