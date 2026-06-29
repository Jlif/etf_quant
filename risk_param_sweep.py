#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
风控参数网格扫描：测试不同 lookback、absolute_momentum_lookback、
volatility_lookback、target_volatility、trailing_stop_pct 组合的收益/回撤表现。

用法:
    python risk_param_sweep.py
    python risk_param_sweep.py --lookbacks 20,22,25 --vol-lookbacks 10,20,30
    python risk_param_sweep.py --output output/risk_sweep.csv
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import unicodedata

sys.path.insert(0, os.path.dirname(__file__))

from data_source import get_data_source
from main import fetch_pool_data, run_strategy
from param_sweep import compute_metrics
from utils import load_config


def _display_width(s: str) -> int:
    """计算字符串在终端中的显示宽度（中文等宽字符按 2 计）。"""
    return sum(
        2 if unicodedata.east_asian_width(ch) in ("F", "W") else 1
        for ch in str(s)
    )


def _rjust(s: str, width: int) -> str:
    """按显示宽度右对齐。"""
    return " " * max(0, width - _display_width(s)) + s


def _ljust(s: str, width: int) -> str:
    """按显示宽度左对齐。"""
    return s + " " * max(0, width - _display_width(s))


@dataclass
class SweepResult:
    lookback: int
    abs_momentum_lookback: int
    vol_lookback: int | None
    target_vol: float | None
    trailing_stop: float | None
    total_return: float
    cagr: float
    max_drawdown: float
    sharpe: float
    final_nav: float
    n_days: int


def _parse_float_list(s: str) -> list[float | None]:
    """解析逗号分隔的浮点数，'none' 转为 None。"""
    values = []
    for part in s.split(","):
        part = part.strip().lower()
        if part in ("", "none", "null"):
            values.append(None)
        else:
            values.append(float(part))
    return values


def _parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_optional_int_list(s: str) -> list[int | None]:
    """解析逗号分隔的整数，'none' 转为 None。"""
    values = []
    for part in s.split(","):
        part = part.strip().lower()
        if part in ("", "none", "null"):
            values.append(None)
        else:
            values.append(int(part))
    return values


def _compute_sharpe(nav: pd.Series) -> float:
    """从净值序列计算年化夏普比率（无风险利率假设为 0）。"""
    daily_returns = nav.pct_change().dropna()
    if daily_returns.std() == 0 or len(daily_returns) < 2:
        return 0.0
    return (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)


def sweep_risk_params(
    strategy,
    app_config,
    data_source,
    lookbacks: list[int],
    abs_lookbacks: list[int],
    vol_lookbacks: list[int | None],
    target_vols: list[float | None],
    trailing_stops: list[float | None],
    data: dict | None = None,
) -> list[SweepResult]:
    """对多组风控参数做网格扫描。"""
    results: list[SweepResult] = []
    base_params = dict(strategy.params)

    total_combos = (
        len(lookbacks)
        * len(abs_lookbacks)
        * len(vol_lookbacks)
        * len(target_vols)
        * len(trailing_stops)
    )
    print(f"总共 {total_combos} 种参数组合\n")

    for (
        lookback,
        abs_lb,
        vol_lb,
        target_vol,
        trailing_stop,
    ) in itertools.product(
        lookbacks, abs_lookbacks, vol_lookbacks, target_vols, trailing_stops
    ):
        params = {**base_params}
        params["lookback"] = lookback
        params["absolute_momentum_lookback"] = abs_lb
        params["absolute_momentum_filter"] = True

        if vol_lb is not None:
            params["volatility_lookback"] = vol_lb
        else:
            params.pop("volatility_lookback", None)

        if target_vol is not None:
            params["target_volatility"] = target_vol
        else:
            params.pop("target_volatility", None)

        if trailing_stop is not None:
            params["trailing_stop_pct"] = trailing_stop
        else:
            params.pop("trailing_stop_pct", None)

        strategy.params = params

        try:
            result, _ = run_strategy(
                strategy, app_config, data_source, silent=True, data=data
            )
            nav = result["轮动策略净值"]
            total_return, cagr, max_dd = compute_metrics(nav)
            sharpe = _compute_sharpe(nav)
            results.append(
                SweepResult(
                    lookback=lookback,
                    abs_momentum_lookback=abs_lb,
                    vol_lookback=vol_lb,
                    target_vol=target_vol,
                    trailing_stop=trailing_stop,
                    total_return=total_return,
                    cagr=cagr,
                    max_drawdown=max_dd,
                    sharpe=sharpe,
                    final_nav=nav.iloc[-1],
                    n_days=len(nav),
                )
            )
        except Exception as e:
            print(
                f"  [跳过] lookback={lookback}, abs_lb={abs_lb}, vol_lb={vol_lb}, "
                f"target_vol={target_vol}, trailing_stop={trailing_stop}: {e}"
            )

    strategy.params = base_params
    return results


def print_results(results: list[SweepResult], sort_by: str = "cagr") -> None:
    """打印扫描结果表格。"""
    sort_key = {
        "total": lambda r: r.total_return,
        "cagr": lambda r: r.cagr,
        "max_dd": lambda r: r.max_drawdown,
        "sharpe": lambda r: r.sharpe,
    }.get(sort_by, lambda r: r.cagr)

    sorted_results = sorted(results, key=sort_key, reverse=(sort_by != "max_dd"))

    # 定义列宽（显示宽度，中文按2计）
    col_widths = {
        "lookback": 10,
        "abs_lb": 8,
        "vol_lb": 8,
        "target_vol": 12,
        "stop": 10,
        "总收益": 12,
        "CAGR": 12,
        "最大回撤": 12,
        "夏普": 10,
        "最终净值": 12,
    }

    # 构建表头
    headers = [
        _rjust("lookback", col_widths["lookback"]),
        _rjust("abs_lb", col_widths["abs_lb"]),
        _rjust("vol_lb", col_widths["vol_lb"]),
        _rjust("target_vol", col_widths["target_vol"]),
        _rjust("stop", col_widths["stop"]),
        _rjust("总收益", col_widths["总收益"]),
        _rjust("CAGR", col_widths["CAGR"]),
        _rjust("最大回撤", col_widths["最大回撤"]),
        _rjust("夏普", col_widths["夏普"]),
        _rjust("最终净值", col_widths["最终净值"]),
    ]

    total_width = sum(col_widths.values()) + len(col_widths) - 1
    print("\n" + "=" * total_width)
    print(" ".join(headers))
    print("-" * total_width)
    for r in sorted_results:
        target_vol_str = f"{r.target_vol:.2%}" if r.target_vol is not None else "-"
        stop_str = f"{r.trailing_stop:.2%}" if r.trailing_stop is not None else "-"
        vol_lb_str = str(r.vol_lookback) if r.vol_lookback is not None else "-"
        total_return_str = f"{r.total_return:+.2%}"
        cagr_str = f"{r.cagr:+.2%}"
        max_dd_str = f"{r.max_drawdown:+.2%}"
        sharpe_str = f"{r.sharpe:.2f}"
        final_nav_str = f"{r.final_nav:.4f}"

        row = [
            _rjust(str(r.lookback), col_widths["lookback"]),
            _rjust(str(r.abs_momentum_lookback), col_widths["abs_lb"]),
            _rjust(vol_lb_str, col_widths["vol_lb"]),
            _rjust(target_vol_str, col_widths["target_vol"]),
            _rjust(stop_str, col_widths["stop"]),
            _rjust(total_return_str, col_widths["总收益"]),
            _rjust(cagr_str, col_widths["CAGR"]),
            _rjust(max_dd_str, col_widths["最大回撤"]),
            _rjust(sharpe_str, col_widths["夏普"]),
            _rjust(final_nav_str, col_widths["最终净值"]),
        ]
        print(" ".join(row))
    print("=" * total_width)


def save_results_csv(results: list[SweepResult], path: str) -> None:
    """保存结果到 CSV。"""
    df = pd.DataFrame(
        [
            {
                "lookback": r.lookback,
                "absolute_momentum_lookback": r.abs_momentum_lookback,
                "volatility_lookback": r.vol_lookback,
                "target_volatility": r.target_vol,
                "trailing_stop_pct": r.trailing_stop,
                "total_return": r.total_return,
                "cagr": r.cagr,
                "max_drawdown": r.max_drawdown,
                "sharpe": r.sharpe,
                "final_nav": r.final_nav,
                "n_days": r.n_days,
            }
            for r in results
        ]
    )
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\n[参数扫描结果已保存] {path}")


def main():
    parser = argparse.ArgumentParser(description="风控参数网格扫描")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    parser.add_argument(
        "--lookbacks",
        default="22",
        help="lookback 列表，逗号分隔",
    )
    parser.add_argument(
        "--abs-lookbacks",
        default="9",
        help="absolute_momentum_lookback 列表，逗号分隔",
    )
    parser.add_argument(
        "--vol-lookbacks",
        default="26",
        help="volatility_lookback 列表，逗号分隔，none 表示用默认值",
    )
    parser.add_argument(
        "--target-vols",
        default="0.07,0.08,0.09,0.1,0.11,0.12,0.13,0.14,0.15",
        help="target_volatility 列表，逗号分隔，none 表示不启用",
    )
    parser.add_argument(
        "--trailing-stops",
        default="0.07",
        help="trailing_stop_pct 列表，逗号分隔，none 表示不启用",
    )
    parser.add_argument(
        "--output",
        default="output/risk_param_sweep.csv",
        help="CSV 输出路径",
    )
    parser.add_argument(
        "--sort-by",
        default="cagr",
        choices=["total", "cagr", "max_dd", "sharpe"],
        help="结果排序依据",
    )
    parser.add_argument(
        "--today",
        action="store_true",
        help="拉取当天最新行情数据（默认使用缓存/历史数据）",
    )
    args = parser.parse_args()

    app_config = load_config(args.config)
    enabled_strategies = [s for s in app_config.strategies if s.enabled]
    if not enabled_strategies:
        print("没有启用的策略")
        return

    strategy = enabled_strategies[0]
    if strategy.mode != "rotation":
        print(f"仅支持 rotation 策略，当前为 {strategy.mode}")
        return

    lookbacks = _parse_int_list(args.lookbacks)
    abs_lookbacks = _parse_int_list(args.abs_lookbacks)
    vol_lookbacks = _parse_optional_int_list(args.vol_lookbacks)
    target_vols = _parse_float_list(args.target_vols)
    trailing_stops = _parse_float_list(args.trailing_stops)

    print(
        f"\n策略: {strategy.name}"
        f"\nlookback: {lookbacks}"
        f"\nabsolute_momentum_lookback: {abs_lookbacks}"
        f"\nvolatility_lookback: {vol_lookbacks}"
        f"\ntarget_volatility: {target_vols}"
        f"\ntrailing_stop_pct: {trailing_stops}\n"
    )

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

    data = fetch_pool_data(strategy, app_config, data_source, include_today=args.today)
    results = sweep_risk_params(
        strategy=strategy,
        app_config=app_config,
        data_source=data_source,
        lookbacks=lookbacks,
        abs_lookbacks=abs_lookbacks,
        vol_lookbacks=vol_lookbacks,
        target_vols=target_vols,
        trailing_stops=trailing_stops,
        data=data,
    )

    print_results(results, sort_by=args.sort_by)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    save_results_csv(results, args.output)


if __name__ == "__main__":
    main()
