#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ema_diff 评分参数网格扫描：对启用的 rotation 策略扫描 ema_fast / ema_slow 组合。

用法:
    python ema_param_sweep.py
    python ema_param_sweep.py --fasts 5,8,12 --slows 20,26,40
    python ema_param_sweep.py --sort-by sharpe --output output/ema_sweep.csv
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from data_source import get_data_source
from core.orchestrator import fetch_pool_data, run_strategy
from core.metrics import compute_metrics, compute_sharpe
from utils import load_config


def _parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description="ema_diff 快/慢线参数网格扫描")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    parser.add_argument("--fasts", default="5,8,10,12,15,20", help="快线周期列表，逗号分隔")
    parser.add_argument("--slows", default="20,26,30,40,50,60", help="慢线周期列表，逗号分隔")
    parser.add_argument("--sort-by", default="cagr", choices=["cagr", "sharpe", "max_drawdown", "total_return"])
    parser.add_argument("--output", default=None, help="结果 CSV 保存路径")
    args = parser.parse_args()

    app_config = load_config(args.config)
    strategy = next(
        (s for s in app_config.strategies if s.enabled and s.mode == "rotation"),
        None,
    )
    if strategy is None:
        print("未找到启用的 rotation 策略")
        sys.exit(1)

    data_source = get_data_source(
        name=app_config.data_source.provider,
        fallback=False,
        skip_test=True,
    )
    base_params = dict(strategy.params)
    strategy.params = {**base_params, "scoring": "ema_diff", "adaptive_scoring": False}
    data = fetch_pool_data(strategy, app_config, data_source, silent=True)

    fasts = _parse_int_list(args.fasts)
    slows = _parse_int_list(args.slows)
    combos = [(f, s) for f in fasts for s in slows if s > f]
    print(f"\n共 {len(combos)} 组参数（fast < slow）\n")

    rows = []
    for i, (fast, slow) in enumerate(combos, 1):
        strategy.params = {**base_params, "scoring": "ema_diff", "adaptive_scoring": False,
                           "ema_fast": fast, "ema_slow": slow}
        try:
            result, _ = run_strategy(strategy, app_config, data_source, silent=True, data=data)
            nav = result["轮动策略净值"]
            total_return, cagr, max_dd = compute_metrics(nav)
            sharpe = compute_sharpe(nav)
            rows.append({
                "ema_fast": fast, "ema_slow": slow,
                "total_return": total_return, "cagr": cagr,
                "max_drawdown": max_dd, "sharpe": sharpe,
                "final_nav": nav.iloc[-1], "n_days": len(nav),
            })
            print(f"  [{i}/{len(combos)}] fast={fast:>2} slow={slow:>2}  "
                  f"CAGR={cagr:+.2%}  回撤={max_dd:.2%}  Sharpe={sharpe:.2f}")
        except Exception as e:
            print(f"  [{i}/{len(combos)}] fast={fast:>2} slow={slow:>2}  失败: {e}")

    strategy.params = base_params

    if not rows:
        print("无有效结果")
        sys.exit(1)

    df = pd.DataFrame(rows)
    ascending = args.sort_by == "max_drawdown"
    df = df.sort_values(args.sort_by, ascending=ascending).reset_index(drop=True)

    print(f"\n{'='*70}")
    print(f"按 {args.sort_by} 排序的 Top10：")
    print(f"{'='*70}")
    top = df.head(10)
    for rank, r in top.iterrows():
        print(f"  {rank+1:>2}. fast={r.ema_fast:>2.0f} slow={r.ema_slow:>2.0f}  "
              f"CAGR={r.cagr:+.2%}  总收益={r.total_return:+.2%}  "
              f"回撤={r.max_drawdown:.2%}  Sharpe={r.sharpe:.2f}")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        df.to_csv(args.output, index=False)
        print(f"\n结果已保存: {args.output}")


if __name__ == "__main__":
    main()
