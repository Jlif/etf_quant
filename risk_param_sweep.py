#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
风控参数网格扫描：基于三层风控体系（layer1/layer2/layer3）扫描参数组合，
按年化收益率（CAGR）倒序输出表现较好的组合。

用法:
    python risk_param_sweep.py
    python risk_param_sweep.py --lookbacks 20,22,25 --l3-target-vols 0.06,0.08,0.1
    python risk_param_sweep.py --output output/risk_sweep.csv --sort-by cagr
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
from core.orchestrator import fetch_pool_data, run_strategy
from utils import load_config


def compute_metrics(nav: pd.Series) -> tuple[float, float, float]:
    """从净值序列计算总收益、年化收益率（CAGR）和最大回撤。"""
    total_return = nav.iloc[-1] / nav.iloc[0] - 1.0
    n_years = len(nav) / 252.0
    cagr = (
        (nav.iloc[-1] / nav.iloc[0]) ** (1.0 / n_years) - 1.0
        if n_years > 0 and nav.iloc[0] > 0
        else 0.0
    )
    running_max = nav.expanding().max()
    drawdown = (nav - running_max) / running_max
    max_drawdown = drawdown.min()
    return total_return, cagr, max_drawdown



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
    l1_ma_lookback: int
    l1_drawdown_lookback: int
    l1_drawdown_threshold: float
    l2_atr_multiplier: float
    l2_atr_lookback: int
    l3_target_vol: float
    l3_vol_lookback: int
    l3_comfort_zone: float
    l3_caution_zone: float
    l3_caution_scale: float
    l3_transition_power: float | None
    total_return: float
    cagr: float
    max_drawdown: float
    sharpe: float
    final_nav: float
    n_days: int


def _parse_float_list(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_optional_float_list(s: str) -> list[float | None]:
    """解析可能包含 None 的浮点数列表，'none' 或空字符串表示 None。"""
    result: list[float | None] = []
    for part in s.split(","):
        part = part.strip().lower()
        if not part or part == "none" or part == "null":
            result.append(None)
        else:
            result.append(float(part))
    return result


def _compute_sharpe(nav: pd.Series) -> float:
    """从净值序列计算年化夏普比率（无风险利率假设为 0）。"""
    daily_returns = nav.pct_change().dropna()
    if daily_returns.std() == 0 or len(daily_returns) < 2:
        return 0.0
    return (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)


def _build_risk_control(
    l1_ma_lookback: int,
    l1_drawdown_lookback: int,
    l1_drawdown_threshold: float,
    l2_atr_multiplier: float,
    l2_atr_lookback: int,
    l3_target_vol: float,
    l3_vol_lookback: int,
    l3_comfort_zone: float,
    l3_caution_zone: float,
    l3_caution_scale: float,
    l3_transition_power: float | None,
) -> dict:
    """构造新的 risk_control 参数字典。"""
    layer3 = {
        "enabled": True,
        "target_vol": l3_target_vol,
        "vol_lookback": l3_vol_lookback,
        "comfort_zone": l3_comfort_zone,
        "caution_zone": l3_caution_zone,
        "caution_scale": l3_caution_scale,
    }
    if l3_transition_power is not None:
        layer3["transition_power"] = l3_transition_power
    return {
        "layer1": {
            "enabled": True,
            "ma_lookback": l1_ma_lookback,
            "drawdown_lookback": l1_drawdown_lookback,
            "drawdown_threshold": l1_drawdown_threshold,
        },
        "layer2": {
            "enabled": True,
            "atr_multiplier": l2_atr_multiplier,
            "atr_lookback": l2_atr_lookback,
        },
        "layer3": layer3,
    }


def sweep_risk_params(
    strategy,
    app_config,
    data_source,
    lookbacks: list[int],
    l1_ma_lookbacks: list[int],
    l1_drawdown_lookbacks: list[int],
    l1_drawdown_thresholds: list[float],
    l2_atr_multipliers: list[float],
    l2_atr_lookbacks: list[int],
    l3_target_vols: list[float],
    l3_vol_lookbacks: list[int],
    l3_comfort_zones: list[float],
    l3_caution_zones: list[float],
    l3_caution_scales: list[float],
    l3_transition_powers: list[float | None],
    data: dict | None = None,
) -> list[SweepResult]:
    """对多组三层风控参数做网格扫描。"""
    results: list[SweepResult] = []
    base_params = dict(strategy.params)

    total_combos = (
        len(lookbacks)
        * len(l1_ma_lookbacks)
        * len(l1_drawdown_lookbacks)
        * len(l1_drawdown_thresholds)
        * len(l2_atr_multipliers)
        * len(l2_atr_lookbacks)
        * len(l3_target_vols)
        * len(l3_vol_lookbacks)
        * len(l3_comfort_zones)
        * len(l3_caution_zones)
        * len(l3_caution_scales)
        * len(l3_transition_powers)
    )
    print(f"总共 {total_combos} 种参数组合\n")

    combos = itertools.product(
        lookbacks,
        l1_ma_lookbacks,
        l1_drawdown_lookbacks,
        l1_drawdown_thresholds,
        l2_atr_multipliers,
        l2_atr_lookbacks,
        l3_target_vols,
        l3_vol_lookbacks,
        l3_comfort_zones,
        l3_caution_zones,
        l3_caution_scales,
        l3_transition_powers,
    )

    for (
        lookback,
        l1_ma_lb,
        l1_dd_lb,
        l1_dd,
        l2_atr_mul,
        l2_atr_lb,
        l3_target_vol,
        l3_vol_lb,
        l3_comfort,
        l3_caution,
        l3_scale,
        l3_power,
    ) in combos:
        params = {**base_params}
        params["lookback"] = lookback

        # 清理旧风控参数，避免与三层风控冲突
        for old_key in (
            "absolute_momentum_filter",
            "absolute_momentum_cash",
            "absolute_momentum_lookback",
            "absolute_momentum_threshold",
            "target_volatility",
            "volatility_lookback",
            "trailing_stop_pct",
        ):
            params.pop(old_key, None)

        params["risk_control"] = _build_risk_control(
            l1_ma_lb,
            l1_dd_lb,
            l1_dd,
            l2_atr_mul,
            l2_atr_lb,
            l3_target_vol,
            l3_vol_lb,
            l3_comfort,
            l3_caution,
            l3_scale,
            l3_power,
        )

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
                    l1_ma_lookback=l1_ma_lb,
                    l1_drawdown_lookback=l1_dd_lb,
                    l1_drawdown_threshold=l1_dd,
                    l2_atr_multiplier=l2_atr_mul,
                    l2_atr_lookback=l2_atr_lb,
                    l3_target_vol=l3_target_vol,
                    l3_vol_lookback=l3_vol_lb,
                    l3_comfort_zone=l3_comfort,
                    l3_caution_zone=l3_caution,
                    l3_caution_scale=l3_scale,
                    l3_transition_power=l3_power,
                    total_return=total_return,
                    cagr=cagr,
                    max_drawdown=max_dd,
                    sharpe=sharpe,
                    final_nav=nav.iloc[-1],
                    n_days=len(nav),
                )
            )
        except Exception as e:
            power_str = "None" if l3_power is None else f"{l3_power:.1f}"
            print(
                f"  [跳过] lookback={lookback}, "
                f"l1=({l1_ma_lb},{l1_dd_lb},{l1_dd:.2%}), "
                f"l2=({l2_atr_mul},{l2_atr_lb}), "
                f"l3=({l3_target_vol:.2%},{l3_vol_lb},{l3_comfort:.2%},{l3_caution:.2%},{l3_scale},power={power_str}): {e}"
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

    # 默认按夏普倒序、再按 CAGR 倒序；最大回撤越小越好，所以升序
    if sort_by == "cagr":
        sorted_results = sorted(results, key=lambda r: (r.sharpe, r.cagr), reverse=True)
        sort_label = "夏普+CAGR"
    else:
        sorted_results = sorted(results, key=sort_key, reverse=(sort_by != "max_dd"))
        sort_label = sort_by

    col_widths = {
        "lookback": 10,
        "l1_ma": 8,
        "l1_dd_lb": 10,
        "l1_dd": 8,
        "l2_mul": 8,
        "l2_lb": 8,
        "l3_vol": 8,
        "l3_lb": 8,
        "l3_com": 8,
        "l3_cau": 8,
        "l3_scl": 8,
        "l3_pwr": 8,
        "总收益": 10,
        "CAGR": 10,
        "最大回撤": 10,
        "夏普": 8,
        "最终净值": 10,
    }

    headers = [
        _rjust("lookback", col_widths["lookback"]),
        _rjust("l1_ma", col_widths["l1_ma"]),
        _rjust("l1_dd_lb", col_widths["l1_dd_lb"]),
        _rjust("l1_dd", col_widths["l1_dd"]),
        _rjust("l2_mul", col_widths["l2_mul"]),
        _rjust("l2_lb", col_widths["l2_lb"]),
        _rjust("l3_vol", col_widths["l3_vol"]),
        _rjust("l3_lb", col_widths["l3_lb"]),
        _rjust("l3_com", col_widths["l3_com"]),
        _rjust("l3_cau", col_widths["l3_cau"]),
        _rjust("l3_scl", col_widths["l3_scl"]),
        _rjust("l3_pwr", col_widths["l3_pwr"]),
        _rjust("总收益", col_widths["总收益"]),
        _rjust("CAGR", col_widths["CAGR"]),
        _rjust("最大回撤", col_widths["最大回撤"]),
        _rjust("夏普", col_widths["夏普"]),
        _rjust("最终净值", col_widths["最终净值"]),
    ]

    total_width = sum(col_widths.values()) + len(col_widths) - 1
    print("\n" + "=" * total_width)
    print(f"按 {sort_label} {'倒序' if sort_by != 'max_dd' else '升序'} 排列")
    print("-" * total_width)
    print(" ".join(headers))
    print("-" * total_width)
    for r in sorted_results:
        row = [
            _rjust(str(r.lookback), col_widths["lookback"]),
            _rjust(str(r.l1_ma_lookback), col_widths["l1_ma"]),
            _rjust(str(r.l1_drawdown_lookback), col_widths["l1_dd_lb"]),
            _rjust(f"{r.l1_drawdown_threshold:.1%}", col_widths["l1_dd"]),
            _rjust(f"{r.l2_atr_multiplier:.1f}", col_widths["l2_mul"]),
            _rjust(str(r.l2_atr_lookback), col_widths["l2_lb"]),
            _rjust(f"{r.l3_target_vol:.1%}", col_widths["l3_vol"]),
            _rjust(str(r.l3_vol_lookback), col_widths["l3_lb"]),
            _rjust(f"{r.l3_comfort_zone:.1%}", col_widths["l3_com"]),
            _rjust(f"{r.l3_caution_zone:.1%}", col_widths["l3_cau"]),
            _rjust(f"{r.l3_caution_scale:.1f}", col_widths["l3_scl"]),
            _rjust("None" if r.l3_transition_power is None else f"{r.l3_transition_power:.1f}", col_widths["l3_pwr"]),
            _rjust(f"{r.total_return:+.2%}", col_widths["总收益"]),
            _rjust(f"{r.cagr:+.2%}", col_widths["CAGR"]),
            _rjust(f"{r.max_drawdown:+.2%}", col_widths["最大回撤"]),
            _rjust(f"{r.sharpe:.2f}", col_widths["夏普"]),
            _rjust(f"{r.final_nav:.4f}", col_widths["最终净值"]),
        ]
        print(" ".join(row))
    print("=" * total_width)


def save_results_csv(results: list[SweepResult], path: str) -> None:
    """保存结果到 CSV。"""
    df = pd.DataFrame(
        [
            {
                "lookback": r.lookback,
                "l1_ma_lookback": r.l1_ma_lookback,
                "l1_drawdown_lookback": r.l1_drawdown_lookback,
                "l1_drawdown_threshold": r.l1_drawdown_threshold,
                "l2_atr_multiplier": r.l2_atr_multiplier,
                "l2_atr_lookback": r.l2_atr_lookback,
                "l3_target_vol": r.l3_target_vol,
                "l3_vol_lookback": r.l3_vol_lookback,
                "l3_comfort_zone": r.l3_comfort_zone,
                "l3_caution_zone": r.l3_caution_zone,
                "l3_caution_scale": r.l3_caution_scale,
                "l3_transition_power": r.l3_transition_power,
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
    parser = argparse.ArgumentParser(description="三层风控参数网格扫描")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    parser.add_argument(
        "--lookbacks",
        default="20",
        help="策略 lookback 列表，逗号分隔",
    )
    parser.add_argument(
        "--l1-ma-lookbacks",
        default="12",
        help="Layer1 均线回望周期列表（标的级别，单只ETF价格均线），逗号分隔",
    )
    parser.add_argument(
        "--l1-drawdown-lookbacks",
        default="38",
        help="Layer1 回撤回望周期列表（标的级别，单只ETF回撤窗口），逗号分隔",
    )
    parser.add_argument(
        "--l1-drawdown-thresholds",
        default="0.12",
        help="Layer1 回撤阈值列表（标的级别，单只ETF从高点回撤），逗号分隔",
    )
    parser.add_argument(
        "--l2-atr-multipliers",
        default="2",
        help="Layer2 ATR 乘数列表，逗号分隔",
    )
    parser.add_argument(
        "--l2-atr-lookbacks",
        default="13",
        help="Layer2 ATR 回望周期列表，逗号分隔",
    )
    parser.add_argument(
        "--l3-target-vols",
        default="0.12",
        help="Layer3 目标波动率列表（标的级别，单只ETF目标波动率），逗号分隔",
    )
    parser.add_argument(
        "--l3-vol-lookbacks",
        default="75",
        help="Layer3 波动率回望周期列表，逗号分隔",
    )
    parser.add_argument(
        "--l3-comfort-zones",
        default="0.25",
        help="Layer3 舒适区波动率上限列表（标的级别），逗号分隔",
    )
    parser.add_argument(
        "--l3-caution-zones",
        default="0.4",
        help="Layer3 警惕区波动率上限列表（标的级别，行业ETF波动可达40-50%），逗号分隔",
    )
    parser.add_argument(
        "--l3-caution-scales",
        default="0.5",
        help="Layer3 警惕区仓位系数列表，逗号分隔（transition_power 非 None 时被忽略）",
    )
    parser.add_argument(
        "--l3-transition-powers",
        default="3",
        help="Layer3 平滑过渡幂指数列表，逗号分隔；'none' 表示不使用平滑过渡",
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
        help="结果排序依据，默认 CAGR 倒序",
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
    l1_ma_lookbacks = _parse_int_list(args.l1_ma_lookbacks)
    l1_drawdown_lookbacks = _parse_int_list(args.l1_drawdown_lookbacks)
    l1_drawdown_thresholds = _parse_float_list(args.l1_drawdown_thresholds)
    l2_atr_multipliers = _parse_float_list(args.l2_atr_multipliers)
    l2_atr_lookbacks = _parse_int_list(args.l2_atr_lookbacks)
    l3_target_vols = _parse_float_list(args.l3_target_vols)
    l3_vol_lookbacks = _parse_int_list(args.l3_vol_lookbacks)
    l3_comfort_zones = _parse_float_list(args.l3_comfort_zones)
    l3_caution_zones = _parse_float_list(args.l3_caution_zones)
    l3_caution_scales = _parse_float_list(args.l3_caution_scales)
    l3_transition_powers = _parse_optional_float_list(args.l3_transition_powers)

    total_combos = (
        len(lookbacks)
        * len(l1_ma_lookbacks)
        * len(l1_drawdown_lookbacks)
        * len(l1_drawdown_thresholds)
        * len(l2_atr_multipliers)
        * len(l2_atr_lookbacks)
        * len(l3_target_vols)
        * len(l3_vol_lookbacks)
        * len(l3_comfort_zones)
        * len(l3_caution_zones)
        * len(l3_caution_scales)
        * len(l3_transition_powers)
    )
    print(
        f"\n策略: {strategy.name}"
        f"\nlookback: {lookbacks}"
        f"\nLayer1 ma_lookback: {l1_ma_lookbacks}"
        f"\nLayer1 drawdown_lookback: {l1_drawdown_lookbacks}"
        f"\nLayer1 drawdown_threshold: {l1_drawdown_thresholds}"
        f"\nLayer2 atr_multiplier: {l2_atr_multipliers}"
        f"\nLayer2 atr_lookback: {l2_atr_lookbacks}"
        f"\nLayer3 target_vol: {l3_target_vols}"
        f"\nLayer3 vol_lookback: {l3_vol_lookbacks}"
        f"\nLayer3 comfort_zone: {l3_comfort_zones}"
        f"\nLayer3 caution_zone: {l3_caution_zones}"
        f"\nLayer3 caution_scale: {l3_caution_scales}"
        f"\nLayer3 transition_power: {l3_transition_powers}"
        f"\n总组合数: {total_combos}\n"
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
        l1_ma_lookbacks=l1_ma_lookbacks,
        l1_drawdown_lookbacks=l1_drawdown_lookbacks,
        l1_drawdown_thresholds=l1_drawdown_thresholds,
        l2_atr_multipliers=l2_atr_multipliers,
        l2_atr_lookbacks=l2_atr_lookbacks,
        l3_target_vols=l3_target_vols,
        l3_vol_lookbacks=l3_vol_lookbacks,
        l3_comfort_zones=l3_comfort_zones,
        l3_caution_zones=l3_caution_zones,
        l3_caution_scales=l3_caution_scales,
        l3_transition_powers=l3_transition_powers,
        data=data,
    )

    print_results(results, sort_by=args.sort_by)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    save_results_csv(results, args.output)


if __name__ == "__main__":
    main()
