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

import pandas as pd

# 确保本地模块可导入
sys.path.insert(0, os.path.dirname(__file__))

from data_source import get_data_source
from core import (
    OUTPUT_DIR,
    performance_report,
    plot_nav_curves,
    plot_strategy_comparison,
    print_summary,
)
from core.orchestrator import (
    build_holding_df,
    print_holding_summary,
    print_position_contribution,
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
    output_dir = "./output"
    if os.path.exists(output_dir):
        for f in os.listdir(output_dir):
            os.remove(os.path.join(output_dir, f))

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

        # 绩效报告
        benchmark_col = f"{name_list[0]}净值"
        benchmark_series = result[benchmark_col] if benchmark_col in result.columns else None
        if benchmark_series is not None:
            benchmark_returns = benchmark_series.pct_change(fill_method=None).fillna(0)
            benchmark_returns.name = benchmark_col
        else:
            benchmark_returns = None

        strategy_returns = result["轮动策略净值"].pct_change(fill_method=None).fillna(0)
        strategy_returns.name = "轮动策略净值"

        holding_df = build_holding_df(result)
        performance_report(
            strategy_returns,
            benchmark=benchmark_returns,
            title=f"{strategy.name}回测报告",
            holding_df=holding_df,
        )

        # 输出各 ETF 持有天数与收益贡献占比（仅 rotation 策略）
        if strategy.mode == "rotation":
            print_position_contribution(strategy, result, name_list)

        # 输出 rotation 策略每日持仓记录 CSV
        if holding_df is not None and not holding_df.empty:
            safe_name = strategy.name.replace(" ", "_").replace(":", "_")
            csv_path = os.path.join(OUTPUT_DIR, f"{safe_name}_持仓记录.csv")
            holding_df.to_csv(csv_path, encoding="utf-8-sig")
            print(f"[持仓记录已保存] {csv_path}")
            print_holding_summary(holding_df, strategy.name)

    # 可视化
    plot_nav_curves(
        {k: (v[0], v[1], "轮动策略净值") for k, v in all_results.items()}
    )
    plot_strategy_comparison(nav_series)
    print_summary(nav_series)


if __name__ == "__main__":
    main()
