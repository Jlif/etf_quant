#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
参数扫描：测试不同 lookback 对轮动策略总收益、年化收益、最大回撤的影响。

用法:
    python param_sweep.py
    python param_sweep.py --lookback-start 5 --lookback-end 60 --lookback-step 5
    python param_sweep.py --config my_config.yaml --output output/lookback_sweep.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

# 确保本地模块可导入
sys.path.insert(0, os.path.dirname(__file__))

from data_source import get_data_source
from main import fetch_pool_data, run_strategy
from utils import StrategyConfig, load_config

# 无图形界面环境使用 Agg 后端
if os.environ.get("DISPLAY") is None and os.name != "nt":
    matplotlib.use("Agg")

plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


@dataclass
class SweepResult:
    lookback: int
    total_return: float
    cagr: float
    max_drawdown: float
    final_nav: float
    n_days: int


def compute_metrics(nav: pd.Series) -> tuple[float, float, float]:
    """从净值序列计算总收益、CAGR、最大回撤。"""
    total_return = nav.iloc[-1] / nav.iloc[0] - 1
    n_days = len(nav)
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (252 / n_days) - 1

    running_max = nav.cummax()
    drawdown = (nav - running_max) / running_max
    max_drawdown = drawdown.min()

    return total_return, cagr, max_drawdown


def sweep_lookback(
    strategy: StrategyConfig,
    app_config,
    data_source,
    lookback_values: list[int],
    data: dict | None = None,
) -> list[SweepResult]:
    """对指定 lookback 列表批量运行回测并收集指标。"""
    results: list[SweepResult] = []
    base_params = dict(strategy.params)

    for lookback in lookback_values:
        params = {**base_params, "lookback": lookback}
        strategy.params = params

        result, _ = run_strategy(strategy, app_config, data_source, silent=True, data=data)
        nav = result["轮动策略净值"]
        total_return, cagr, max_drawdown = compute_metrics(nav)

        results.append(
            SweepResult(
                lookback=lookback,
                total_return=total_return,
                cagr=cagr,
                max_drawdown=max_drawdown,
                final_nav=nav.iloc[-1],
                n_days=len(nav),
            )
        )

    # 恢复原始参数
    strategy.params = base_params
    return results


def print_results(results: list[SweepResult]) -> None:
    """打印参数扫描结果表格。"""
    print("\n" + "=" * 80)
    print(f"{'lookback':>10} {'总收益':>12} {'年化收益 (CAGR)':>18} {'最大回撤':>12} {'最终净值':>12}")
    print("-" * 80)
    for r in results:
        print(
            f"{r.lookback:>10} {r.total_return:>11.2%} {r.cagr:>17.2%} "
            f"{r.max_drawdown:>11.2%} {r.final_nav:>12.4f}"
        )
    print("=" * 80)


def save_results_csv(results: list[SweepResult], path: str) -> None:
    """保存结果到 CSV。"""
    df = pd.DataFrame(
        [
            {
                "lookback": r.lookback,
                "total_return": r.total_return,
                "cagr": r.cagr,
                "max_drawdown": r.max_drawdown,
                "final_nav": r.final_nav,
                "n_days": r.n_days,
            }
            for r in results
        ]
    )
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\n[参数扫描结果已保存] {path}")


def plot_results(results: list[SweepResult], path: str) -> None:
    """绘制 lookback 与各项指标的关系图。"""
    df = pd.DataFrame(
        [
            {
                "lookback": r.lookback,
                "total_return": r.total_return * 100,
                "cagr": r.cagr * 100,
                "max_drawdown": r.max_drawdown * 100,
            }
            for r in results
        ]
    )

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    axes[0].plot(df["lookback"], df["total_return"], marker="o", color="#1f77b4")
    axes[0].set_ylabel("总收益 (%)")
    axes[0].set_title("lookback 参数扫描")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df["lookback"], df["cagr"], marker="o", color="#2ca02c")
    axes[1].set_ylabel("年化收益 CAGR (%)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(df["lookback"], df["max_drawdown"], marker="o", color="#d62728")
    axes[2].set_ylabel("最大回撤 (%)")
    axes[2].set_xlabel("lookback (交易日)")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[参数扫描图表已保存] {path}")


def main():
    parser = argparse.ArgumentParser(description="lookback 参数扫描")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    parser.add_argument("--lookback-start", type=int, default=5, help="lookback 起始值")
    parser.add_argument("--lookback-end", type=int, default=60, help="lookback 结束值")
    parser.add_argument("--lookback-step", type=int, default=5, help="lookback 步长")
    parser.add_argument("--output", default="output/lookback_sweep.csv", help="CSV 输出路径")
    parser.add_argument("--chart", default="output/lookback_sweep.png", help="图表输出路径")
    parser.add_argument(
        "--today",
        action="store_true",
        help="拉取当天最新行情数据（默认使用缓存/历史数据）",
    )
    args = parser.parse_args()

    app_config = load_config(args.config)
    enabled_strategies = [s for s in app_config.strategies if s.enabled]
    if not enabled_strategies:
        print("没有启用的策略，请在 config.yaml 中将至少一个策略的 enabled 设为 true")
        return

    if len(enabled_strategies) > 1:
        print(f"检测到 {len(enabled_strategies)} 个启用策略，仅对第一个进行参数扫描")

    strategy = enabled_strategies[0]
    if strategy.mode != "rotation":
        print(f"参数扫描当前仅支持 rotation 策略，当前策略模式为 {strategy.mode}")
        return

    if "lookback" not in strategy.params:
        print("当前策略参数中不存在 lookback，无法扫描")
        return

    lookback_values = list(range(args.lookback_start, args.lookback_end + 1, args.lookback_step))
    print(f"\n开始扫描 lookback: {lookback_values}")
    print(f"策略: {strategy.name} | 标的池: {[p.code for p in strategy.pool]}")

    # 检查所有启用策略的 ETF 是否都已缓存，跳过数据源连通性测试
    cache_dir = app_config.backtest.cache_dir
    provider = app_config.data_source.provider
    required_codes = {p.code for s in enabled_strategies for p in s.pool}
    missing_codes = [
        code
        for code in required_codes
        if not os.path.exists(os.path.join(cache_dir, f"{code}_{provider}.csv"))
    ]
    skip_test = not missing_codes and not args.today

    data_source = get_data_source(
        name=provider,
        fallback=True,
        skip_test=skip_test,
    )

    # 预加载一次数据，避免每个 lookback 都重复读取缓存
    data = fetch_pool_data(strategy, app_config, data_source, include_today=args.today)
    results = sweep_lookback(strategy, app_config, data_source, lookback_values, data=data)
    print_results(results)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    save_results_csv(results, args.output)
    plot_results(results, args.chart)


if __name__ == "__main__":
    main()
