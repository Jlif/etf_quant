#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
风控参数网格扫描 / 单参数敏感性扫描：基于三层风控体系（layer1/layer2/layer3）
扫描参数组合，支持全网格（grid）和单参数逐个（sequential）两种模式。

用法:
    python risk_param_sweep.py
    python risk_param_sweep.py --lookbacks 20,22,25 --l3-target-vols 0.06,0.08,0.1
    python risk_param_sweep.py --output output/risk_sweep.csv --sort-by cagr
    python risk_param_sweep.py --mode sequential --lookbacks 15,20,25 --l3-target-vols 0.15,0.25,0.35 --start-date 20200101 --end-date 20221231
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import yaml
from datetime import datetime
from dataclasses import dataclass

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from data_source import get_data_source
from core.orchestrator import fetch_pool_data, run_strategy
from core.metrics import compute_metrics, compute_sharpe
from utils import load_config
from utils.text import display_width, ljust, rjust


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


DEFAULTS = {
    "lookbacks": "20",
    "l1_ma_lookbacks": "13",
    "l1_drawdown_lookbacks": "41",
    "l1_drawdown_thresholds": "0.08",
    "l2_atr_multipliers": "2",
    "l2_atr_lookbacks": "10",
    "l3_target_vols": "0.25",
    "l3_vol_lookbacks": "100",
    "l3_comfort_zones": "0.25",
    "l3_caution_zones": "0.4",
    "l3_caution_scales": "0.5",
    "l3_transition_powers": "4",
    "mode": "grid",
    "output": "output/risk_param_sweep.csv",
    "sort_by": "cagr",
    "sharpe_threshold": None,
    "start_date": None,
    "end_date": None,
    "no_fetch": False,
    "today": False,
}


def load_sweep_config(path: str) -> dict:
    """加载扫描配置文件，扁平化 params 到顶层。"""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not raw:
        return {}
    cfg = dict(raw)
    params = cfg.pop("params", {})
    for k, v in params.items():
        cfg[k] = v
    return cfg


def _resolve_list(value, parser):
    """统一处理列表参数：配置文件里可能是 list，CLI 可能是逗号分隔字符串。"""
    if value is None:
        return parser("")
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return parser(value)
    return [value]


BASELINE_DEFAULTS = {
    "lookback": 20,
    "l1_ma_lookback": 13,
    "l1_drawdown_lookback": 41,
    "l1_drawdown_threshold": 0.08,
    "l2_atr_multiplier": 2.0,
    "l2_atr_lookback": 10,
    "l3_target_vol": 0.13,
    "l3_vol_lookback": 80,
    "l3_comfort_zone": 0.25,
    "l3_caution_zone": 0.4,
    "l3_caution_scale": 0.5,
    "l3_transition_power": 4,
}


_LEGACY_RISK_KEYS = (
    "absolute_momentum_filter",
    "absolute_momentum_cash",
    "absolute_momentum_lookback",
    "absolute_momentum_threshold",
    "target_volatility",
    "volatility_lookback",
    "trailing_stop_pct",
)


def _get_baseline_values(strategy) -> dict:
    """从当前策略参数中提取基线值，缺失项使用 BASELINE_DEFAULTS。"""
    base = dict(BASELINE_DEFAULTS)
    base["lookback"] = strategy.params.get("lookback", base["lookback"])
    rc = strategy.params.get("risk_control", {})
    if rc:
        l1 = rc.get("layer1", {})
        l2 = rc.get("layer2", {})
        l3 = rc.get("layer3", {})
        base["l1_ma_lookback"] = l1.get("ma_lookback", base["l1_ma_lookback"])
        base["l1_drawdown_lookback"] = l1.get("drawdown_lookback", base["l1_drawdown_lookback"])
        base["l1_drawdown_threshold"] = l1.get("drawdown_threshold", base["l1_drawdown_threshold"])
        base["l2_atr_multiplier"] = l2.get("atr_multiplier", base["l2_atr_multiplier"])
        base["l2_atr_lookback"] = l2.get("atr_lookback", base["l2_atr_lookback"])
        base["l3_target_vol"] = l3.get("target_vol", base["l3_target_vol"])
        base["l3_vol_lookback"] = l3.get("vol_lookback", base["l3_vol_lookback"])
        base["l3_comfort_zone"] = l3.get("comfort_zone", base["l3_comfort_zone"])
        base["l3_caution_zone"] = l3.get("caution_zone", base["l3_caution_zone"])
        base["l3_caution_scale"] = l3.get("caution_scale", base["l3_caution_scale"])
        if "transition_power" in l3:
            base["l3_transition_power"] = l3["transition_power"]
    return base


def _run_single(
    strategy,
    app_config,
    data_source,
    data: dict | None,
    *,
    lookback: int,
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
) -> SweepResult | None:
    """运行一组风控参数，返回结果；失败时打印并返回 None。"""
    base_params = dict(strategy.params)
    params = {**base_params}
    params["lookback"] = lookback

    # 清理旧风控参数
    for old_key in _LEGACY_RISK_KEYS:
        params.pop(old_key, None)

    params["risk_control"] = _build_risk_control(
        l1_ma_lookback,
        l1_drawdown_lookback,
        l1_drawdown_threshold,
        l2_atr_multiplier,
        l2_atr_lookback,
        l3_target_vol,
        l3_vol_lookback,
        l3_comfort_zone,
        l3_caution_zone,
        l3_caution_scale,
        l3_transition_power,
    )

    strategy.params = params
    try:
        result, _ = run_strategy(
            strategy, app_config, data_source, silent=True, data=data
        )
        nav = result["轮动策略净值"]
        total_return, cagr, max_dd = compute_metrics(nav)
        sharpe = compute_sharpe(nav)
        return SweepResult(
            lookback=lookback,
            l1_ma_lookback=l1_ma_lookback,
            l1_drawdown_lookback=l1_drawdown_lookback,
            l1_drawdown_threshold=l1_drawdown_threshold,
            l2_atr_multiplier=l2_atr_multiplier,
            l2_atr_lookback=l2_atr_lookback,
            l3_target_vol=l3_target_vol,
            l3_vol_lookback=l3_vol_lookback,
            l3_comfort_zone=l3_comfort_zone,
            l3_caution_zone=l3_caution_zone,
            l3_caution_scale=l3_caution_scale,
            l3_transition_power=l3_transition_power,
            total_return=total_return,
            cagr=cagr,
            max_drawdown=max_dd,
            sharpe=sharpe,
            final_nav=nav.iloc[-1],
            n_days=len(nav),
        )
    except Exception as e:
        power_str = "None" if l3_transition_power is None else f"{l3_transition_power:.1f}"
        print(
            f"  [跳过] lookback={lookback}, "
            f"l1=({l1_ma_lookback},{l1_drawdown_lookback},{l1_drawdown_threshold:.2%}), "
            f"l2=({l2_atr_multiplier},{l2_atr_lookback}), "
            f"l3=({l3_target_vol:.2%},{l3_vol_lookback},{l3_comfort_zone:.2%},{l3_caution_zone:.2%},{l3_caution_scale},power={power_str}): {e}"
        )
        return None
    finally:
        strategy.params = base_params


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
        for old_key in _LEGACY_RISK_KEYS:
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
            sharpe = compute_sharpe(nav)
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


def sweep_sequential(
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
) -> list[tuple[str, int | float | None, SweepResult]]:
    """逐参数单因子敏感性扫描：每次只变一个参数，其余固定在当前策略基线值。

    返回 [(参数名, 参数值, SweepResult), ...]，组合数仅等于各参数列表长度之和。
    """
    base = _get_baseline_values(strategy)
    param_specs = [
        ("lookback", lookbacks),
        ("l1_ma_lookback", l1_ma_lookbacks),
        ("l1_drawdown_lookback", l1_drawdown_lookbacks),
        ("l1_drawdown_threshold", l1_drawdown_thresholds),
        ("l2_atr_multiplier", l2_atr_multipliers),
        ("l2_atr_lookback", l2_atr_lookbacks),
        ("l3_target_vol", l3_target_vols),
        ("l3_vol_lookback", l3_vol_lookbacks),
        ("l3_comfort_zone", l3_comfort_zones),
        ("l3_caution_zone", l3_caution_zones),
        ("l3_caution_scale", l3_caution_scales),
        ("l3_transition_power", l3_transition_powers),
    ]

    results: list[tuple[str, int | float | None, SweepResult]] = []
    total_runs = sum(len(values) for _, values in param_specs if len(values) > 1)
    print(f"顺序单参数扫描：共 {total_runs} 次运行（基线值来自当前策略配置）\n")

    for name, values in param_specs:
        if len(values) <= 1:
            continue
        for v in values:
            kwargs = dict(base)
            kwargs[name] = v
            r = _run_single(strategy, app_config, data_source, data, **kwargs)
            if r is not None:
                results.append((name, v, r))
    return results


def print_sequential_results(
    results: list[tuple[str, int | float | None, SweepResult]],
    top_n: int = 5,
    sharpe_threshold: float | None = None,
) -> None:
    """打印单参数扫描结果，并按参数汇总最佳取值范围。"""
    if not results:
        print("无有效扫描结果")
        return

    grouped: dict[str, list[tuple[int | float | None, SweepResult]]] = {}
    for name, v, r in results:
        grouped.setdefault(name, []).append((v, r))

    total_width = 100
    print("\n" + "=" * total_width)
    print("单参数敏感性扫描结果")
    print("=" * total_width)

    for name, items in grouped.items():
        items_sorted = sorted(
            items, key=lambda x: (x[1].sharpe, x[1].cagr), reverse=True
        )
        print(f"\n【参数】{name}  共测试 {len(items)} 个取值")
        col_specs = [
            ("取值", 18, "l"),
            ("CAGR", 10, "r"),
            ("夏普", 10, "r"),
            ("最大回撤", 10, "r"),
            ("最终净值", 10, "r"),
        ]
        table_width = sum(w for _, w, _ in col_specs) + len(col_specs) - 1
        print("-" * table_width)
        header = " ".join(
            ljust(c, w) if a == "l" else rjust(c, w)
            for c, w, a in col_specs
        )
        print(header)
        print("-" * table_width)
        for v, r in items_sorted[:top_n]:
            row = " ".join([
                ljust(str(v), 18),
                rjust(f"{r.cagr:+.2%}", 10),
                rjust(f"{r.sharpe:.2f}", 10),
                rjust(f"{r.max_drawdown:+.2%}", 10),
                rjust(f"{r.final_nav:.4f}", 10),
            ])
            print(row)

        threshold = sharpe_threshold
        if threshold is None:
            threshold = max(r.sharpe for _, r in items) * 0.9
        good = [(v, r) for v, r in items if r.sharpe >= threshold]
        if not good:
            good = items_sorted[:3]
        good_sorted_by_value = sorted(
            good, key=lambda x: (x[0] is None, x[0] if x[0] is not None else 0)
        )
        values_only = [v for v, _ in good_sorted_by_value]
        print(f"\n  → {name} 的较优取值范围（夏普 >= {threshold:.2f}）：{values_only}")

    print("=" * total_width)


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
        rjust("lookback", col_widths["lookback"]),
        rjust("l1_ma", col_widths["l1_ma"]),
        rjust("l1_dd_lb", col_widths["l1_dd_lb"]),
        rjust("l1_dd", col_widths["l1_dd"]),
        rjust("l2_mul", col_widths["l2_mul"]),
        rjust("l2_lb", col_widths["l2_lb"]),
        rjust("l3_vol", col_widths["l3_vol"]),
        rjust("l3_lb", col_widths["l3_lb"]),
        rjust("l3_com", col_widths["l3_com"]),
        rjust("l3_cau", col_widths["l3_cau"]),
        rjust("l3_scl", col_widths["l3_scl"]),
        rjust("l3_pwr", col_widths["l3_pwr"]),
        rjust("总收益", col_widths["总收益"]),
        rjust("CAGR", col_widths["CAGR"]),
        rjust("最大回撤", col_widths["最大回撤"]),
        rjust("夏普", col_widths["夏普"]),
        rjust("最终净值", col_widths["最终净值"]),
    ]

    total_width = sum(col_widths.values()) + len(col_widths) - 1
    print("\n" + "=" * total_width)
    print(f"按 {sort_label} {'倒序' if sort_by != 'max_dd' else '升序'} 排列")
    print("-" * total_width)
    print(" ".join(headers))
    print("-" * total_width)
    for r in sorted_results:
        row = [
            rjust(str(r.lookback), col_widths["lookback"]),
            rjust(str(r.l1_ma_lookback), col_widths["l1_ma"]),
            rjust(str(r.l1_drawdown_lookback), col_widths["l1_dd_lb"]),
            rjust(f"{r.l1_drawdown_threshold:.1%}", col_widths["l1_dd"]),
            rjust(f"{r.l2_atr_multiplier:.1f}", col_widths["l2_mul"]),
            rjust(str(r.l2_atr_lookback), col_widths["l2_lb"]),
            rjust(f"{r.l3_target_vol:.1%}", col_widths["l3_vol"]),
            rjust(str(r.l3_vol_lookback), col_widths["l3_lb"]),
            rjust(f"{r.l3_comfort_zone:.1%}", col_widths["l3_com"]),
            rjust(f"{r.l3_caution_zone:.1%}", col_widths["l3_cau"]),
            rjust(f"{r.l3_caution_scale:.1f}", col_widths["l3_scl"]),
            rjust("None" if r.l3_transition_power is None else f"{r.l3_transition_power:.1f}", col_widths["l3_pwr"]),
            rjust(f"{r.total_return:+.2%}", col_widths["总收益"]),
            rjust(f"{r.cagr:+.2%}", col_widths["CAGR"]),
            rjust(f"{r.max_drawdown:+.2%}", col_widths["最大回撤"]),
            rjust(f"{r.sharpe:.2f}", col_widths["夏普"]),
            rjust(f"{r.final_nav:.4f}", col_widths["最终净值"]),
        ]
        print(" ".join(row))
    print("=" * total_width)


def save_results_csv(
    results: list[SweepResult] | list[tuple[str, int | float | None, SweepResult]],
    path: str,
    sequential: bool = False,
) -> None:
    """保存结果到 CSV。"""
    if sequential:
        rows = []
        for name, v, r in results:
            rows.append(
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
                    "vary_param": name,
                    "vary_value": v,
                }
            )
    else:
        rows = [
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
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\n[参数扫描结果已保存] {path}")


def main():
    parser = argparse.ArgumentParser(description="三层风控参数网格扫描")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    parser.add_argument(
        "--sweep-config",
        default=None,
        help="扫描专用配置文件路径（YAML），CLI 参数可覆盖其中值",
    )
    parser.add_argument(
        "--lookbacks",
        default=None,
        help="策略 lookback 列表，逗号分隔",
    )
    parser.add_argument(
        "--l1-ma-lookbacks",
        default=None,
        help="Layer1 均线回望周期列表（标的级别，单只ETF价格均线），逗号分隔",
    )
    parser.add_argument(
        "--l1-drawdown-lookbacks",
        default=None,
        help="Layer1 回撤回望周期列表（标的级别，单只ETF回撤窗口），逗号分隔",
    )
    parser.add_argument(
        "--l1-drawdown-thresholds",
        default=None,
        help="Layer1 回撤阈值列表（标的级别，单只ETF从高点回撤），逗号分隔",
    )
    parser.add_argument(
        "--l2-atr-multipliers",
        default=None,
        help="Layer2 ATR 乘数列表，逗号分隔",
    )
    parser.add_argument(
        "--l2-atr-lookbacks",
        default=None,
        help="Layer2 ATR 回望周期列表，逗号分隔",
    )
    parser.add_argument(
        "--l3-vol-lookbacks",
        default=None,
        help="Layer3 波动率回望周期列表，逗号分隔",
    )
    parser.add_argument(
        "--l3-target-vols",
        default=None,
        help="Layer3 目标波动率列表（标的级别，单只ETF目标波动率），逗号分隔",
    )
    parser.add_argument(
        "--l3-comfort-zones",
        default=None,
        help="Layer3 舒适区波动率上限列表（标的级别），逗号分隔",
    )
    parser.add_argument(
        "--l3-caution-zones",
        default=None,
        help="Layer3 警惕区波动率上限列表（标的级别，行业ETF波动可达40-50%），逗号分隔",
    )
    parser.add_argument(
        "--l3-caution-scales",
        default=None,
        help="Layer3 警惕区仓位系数列表，逗号分隔（transition_power 非 None 时被忽略）",
    )
    parser.add_argument(
        "--l3-transition-powers",
        default=None,
        help="Layer3 平滑过渡幂指数列表，逗号分隔；'none' 表示不使用平滑过渡",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="CSV 输出路径",
    )
    parser.add_argument(
        "--sort-by",
        default=None,
        choices=["total", "cagr", "max_dd", "sharpe"],
        help="结果排序依据，默认 CAGR 倒序",
    )
    parser.add_argument(
        "--mode",
        default=None,
        choices=["grid", "sequential"],
        help="扫描模式：grid 全网格（组合爆炸），sequential 单参数逐个扫描（效率高）",
    )
    parser.add_argument(
        "--sharpe-threshold",
        type=float,
        default=None,
        help="sequential 模式下判定较优取值的最小夏普阈值（默认取各参数最大夏普的 90%%）",
    )
    parser.add_argument(
        "--today",
        action="store_true",
        help="拉取当天最新行情数据（默认使用缓存/历史数据）",
    )
    parser.add_argument(
        "--no-fetch",
        action="store_true",
        help="强制使用本地缓存，不触发任何数据下载或更新（需先通过 main 下载好数据）",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="覆盖回测起始日 (YYYYMMDD)，默认使用配置中的 backtest.start_date",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="覆盖回测截止日 (YYYYMMDD)，默认使用配置中的 backtest.end_date",
    )
    args = parser.parse_args()

    # 加载扫描专用配置：CLI > sweep_config > DEFAULTS
    sweep_config = load_sweep_config(args.sweep_config) if args.sweep_config else {}
    cli_overrides = {}
    for k, v in vars(args).items():
        if k in ("no_fetch", "today"):
            if v:
                cli_overrides[k] = v
        elif v is not None:
            cli_overrides[k] = v
    merged = {**DEFAULTS, **sweep_config, **cli_overrides}

    app_config = load_config(args.config)
    enabled_strategies = [s for s in app_config.strategies if s.enabled]
    if not enabled_strategies:
        print("没有启用的策略")
        return

    strategy = enabled_strategies[0]
    if strategy.mode != "rotation":
        print(f"仅支持 rotation 策略，当前为 {strategy.mode}")
        return

    lookbacks = _resolve_list(merged["lookbacks"], _parse_int_list)
    l1_ma_lookbacks = _resolve_list(merged["l1_ma_lookbacks"], _parse_int_list)
    l1_drawdown_lookbacks = _resolve_list(merged["l1_drawdown_lookbacks"], _parse_int_list)
    l1_drawdown_thresholds = _resolve_list(merged["l1_drawdown_thresholds"], _parse_float_list)
    l2_atr_multipliers = _resolve_list(merged["l2_atr_multipliers"], _parse_float_list)
    l2_atr_lookbacks = _resolve_list(merged["l2_atr_lookbacks"], _parse_int_list)
    l3_target_vols = _resolve_list(merged["l3_target_vols"], _parse_float_list)
    l3_vol_lookbacks = _resolve_list(merged["l3_vol_lookbacks"], _parse_int_list)
    l3_comfort_zones = _resolve_list(merged["l3_comfort_zones"], _parse_float_list)
    l3_caution_zones = _resolve_list(merged["l3_caution_zones"], _parse_float_list)
    l3_caution_scales = _resolve_list(merged["l3_caution_scales"], _parse_float_list)
    _l3tp = merged["l3_transition_powers"]
    if isinstance(_l3tp, list):
        l3_transition_powers = _l3tp
    elif isinstance(_l3tp, str):
        l3_transition_powers = _parse_optional_float_list(_l3tp)
    else:
        l3_transition_powers = [_l3tp]

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
        f"\n"
    )

    cache_dir = app_config.backtest.cache_dir
    provider = app_config.data_source.provider
    required_codes = {p.code for s in enabled_strategies for p in s.pool}
    missing_codes = [
        code
        for code in required_codes
        if not os.path.exists(os.path.join(cache_dir, f"{code}_{provider}.csv"))
    ]
    skip_test = merged["no_fetch"] or (not missing_codes and not merged["today"])

    data_source = get_data_source(
        name=provider,
        fallback=True,
        skip_test=skip_test,
    )

    cutoff_date = datetime.strptime(merged["end_date"], "%Y%m%d") if merged["end_date"] else None
    data = fetch_pool_data(
        strategy,
        app_config,
        data_source,
        include_today=merged["today"],
        cutoff_date=cutoff_date,
        start_date=merged["start_date"],
        skip_download=merged["no_fetch"],
    )

    os.makedirs(os.path.dirname(merged["output"]) or ".", exist_ok=True)

    if merged["mode"] == "grid":
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
        print_results(results, sort_by=merged["sort_by"])
        save_results_csv(results, merged["output"])
    else:
        results = sweep_sequential(
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
        print_sequential_results(results, sharpe_threshold=merged["sharpe_threshold"])
        save_results_csv(results, merged["output"], sequential=True)


if __name__ == "__main__":
    main()
