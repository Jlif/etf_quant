#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETF 轮动策略回测系统入口

支持通过 config.yaml 配置策略参数、ETF 代码、仓位占比

用法:
    python main.py
    python main.py --config my_config.yaml
    python main.py --today
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings

# 确保本地模块可导入
sys.path.insert(0, os.path.dirname(__file__))

from data_source import get_data_source
from core import (
    OUTPUT_DIR,
    plot_nav_curves,
    plot_strategy_comparison,
    print_summary,
)
from core.orchestrator import (
    clear_output_dir,
    report_strategy_result,
    run_strategy,
)
from utils import load_config

warnings.filterwarnings("ignore", category=RuntimeWarning)


def main():
    parser = argparse.ArgumentParser(description="ETF 轮动策略回测")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    parser.add_argument(
        "--today",
        action="store_true",
        help="拉取当天最新行情数据（默认使用缓存/历史数据）",
    )
    args = parser.parse_args()

    # 加载配置
    app_config = load_config(args.config)
    enabled_strategies = [s for s in app_config.strategies if s.enabled]
    print("="*60)
    print("ETF 轮动策略回测系统")
    print(f"数据源: {app_config.data_source.provider}")
    print(f"已启用策略数: {len(enabled_strategies)}")
    print("="*60)

    # 清空 output 目录
    clear_output_dir(OUTPUT_DIR)

    # 检查所有启用策略的 ETF 是否都已缓存
    cache_dir = app_config.backtest.cache_dir
    provider = app_config.data_source.provider
    required_codes = {
        p.code
        for s in enabled_strategies
        for p in s.pool
    }
    missing_codes = [
        code for code in required_codes
        if not os.path.exists(os.path.join(cache_dir, f"{code}_{provider}.csv"))
    ]
    skip_test = not missing_codes and not args.today
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
    all_results = {}
    nav_series = {}
    for strategy in app_config.strategies:
        if not strategy.enabled:
            print(f"\n[跳过] 策略 '{strategy.name}' (enabled=false)")
            continue
        result, name_list = run_strategy(strategy, app_config, data_source, include_today=args.today)
        all_results[strategy.name] = (result, name_list)
        nav_series[strategy.name] = result["轮动策略净值"]

        report_strategy_result(strategy, result, name_list)

    # 可视化
    plot_nav_curves(
        {k: (v[0], v[1], "轮动策略净值") for k, v in all_results.items()}
    )
    plot_strategy_comparison(nav_series)
    print_summary(nav_series)


if __name__ == "__main__":
    main()
