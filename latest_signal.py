#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETF 最新信号打印工具

获取上一个交易日最新数据后，执行 print_latest_signal，
不生成回测报告、图表及其他文件。

用法:
    python latest_signal.py
    python latest_signal.py --config my_config.yaml
    python latest_signal.py --strategy "动量轮动策略"
    python latest_signal.py --today
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from data_source import get_data_source
from core.orchestrator import (
    compute_signal_start_date,
    fetch_pool_data,
    print_latest_signal,
    run_strategy,
)
from utils import load_config


def is_trading_day(date: datetime) -> bool:
    """判断是否为交易日（简单排除周末）"""
    return date.weekday() < 5


def get_last_trading_day(date: datetime | None = None) -> datetime:
    """获取上一个交易日"""
    if date is None:
        date = datetime.now()
    date -= timedelta(days=1)
    while not is_trading_day(date):
        date -= timedelta(days=1)
    return date


def main():
    parser = argparse.ArgumentParser(description="ETF 最新信号打印")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    parser.add_argument("--strategy", help="指定策略名称（默认运行所有启用策略）")
    parser.add_argument("--date", help="指定交易截止日 (YYYYMMDD)，默认上一个交易日")
    parser.add_argument(
        "--today",
        action="store_true",
        help="使用当天作为截止日并拉取最新行情数据（默认上一个交易日）",
    )
    args = parser.parse_args()

    # 解析指定日期
    cutoff_date = None
    if args.today:
        cutoff_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        print(f"[指定截止日] 当天: {cutoff_date.date()}")
    elif args.date:
        try:
            cutoff_date = datetime.strptime(args.date, "%Y%m%d")
            print(f"[指定截止日] {cutoff_date.date()}")
        except ValueError:
            print(f"错误: 日期格式无效 '{args.date}'，请使用 YYYYMMDD 格式")
            sys.exit(1)
    else:
        cutoff_date = get_last_trading_day()
        print(f"[默认截止日] 上一个交易日: {cutoff_date.date()}")

    app_config = load_config(args.config)
    enabled_strategies = [s for s in app_config.strategies if s.enabled]

    # 如果指定了策略名称，过滤
    if args.strategy:
        enabled_strategies = [s for s in enabled_strategies if s.name == args.strategy]
        if not enabled_strategies:
            print(f"错误: 未找到策略 '{args.strategy}'")
            sys.exit(1)

    print("="*60)
    print("ETF 最新信号系统")
    print(f"数据源: {app_config.data_source.provider}")
    print(f"策略数: {len(enabled_strategies)}")
    print("="*60)

    # 检查缓存
    cache_dir = app_config.backtest.cache_dir
    provider = app_config.data_source.provider
    required_codes = {
        p.code
        for s in enabled_strategies
        for p in s.pool
    }

    all_cached = all(
        os.path.exists(os.path.join(cache_dir, f"{code}_{provider}.csv"))
        for code in required_codes
    )
    skip_test = all_cached and not args.today
    if skip_test:
        print(f"[缓存] 所有 {len(required_codes)} 个 ETF 已缓存，跳过数据源连通性测试")
    elif args.today:
        print(f"[刷新] 将重新拉取 {len(required_codes)} 个 ETF 的最新行情数据")

    # 初始化数据源
    data_source = get_data_source(
        name=provider,
        fallback=True,
        skip_test=skip_test,
    )

    # 运行启用的策略
    for strategy in enabled_strategies:
        try:
            start_dt = compute_signal_start_date(strategy, cutoff_date)
            start_date = start_dt.strftime("%Y%m%d")
            min_bars = strategy.params.get("lookback", 20) + 5

            data = fetch_pool_data(
                strategy,
                app_config,
                data_source,
                include_today=args.today,
                cutoff_date=cutoff_date,
                start_date=start_date,
                min_bars=min_bars,
            )
            result, name_list = run_strategy(
                strategy,
                app_config,
                data_source,
                data=data,
            )

            last_quote_dates = {
                name: data["close"][name].last_valid_index().strftime("%Y-%m-%d")
                for name in name_list
            }

            print_latest_signal(strategy, result, name_list, last_quote_dates)
        except Exception as e:
            print(f"\n[错误] 策略 '{strategy.name}' 执行失败: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
