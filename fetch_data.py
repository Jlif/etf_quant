#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""独立数据拉取命令。

用法:
    python fetch_data.py
    python fetch_data.py --config my_config.yaml
    python fetch_data.py --today
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from data_source import get_data_source
from utils import load_config


def _merge_price_data(old_df: pd.DataFrame | None, new_df: pd.DataFrame) -> pd.DataFrame:
    """合并旧缓存与新下载数据，新数据覆盖重叠日期，结果按日期排序。"""
    if old_df is None or old_df.empty:
        return new_df.copy()
    merged = pd.concat([old_df, new_df])
    merged = merged[~merged.index.duplicated(keep="last")]
    return merged.sort_index()


def fetch_all_data(app_config, data_source, include_today: bool = False) -> None:
    """拉取所有启用策略 pool 中的 ETF 数据到缓存目录。"""
    enabled_strategies = [s for s in app_config.strategies if s.enabled]
    codes = sorted({p.code for s in enabled_strategies for p in s.pool})
    if not codes:
        print("[提示] 没有启用的策略或候选池为空")
        return

    cache_dir = app_config.backtest.cache_dir
    os.makedirs(cache_dir, exist_ok=True)
    target_start = app_config.backtest.start_date

    print("=" * 60)
    print("ETF 数据拉取")
    print(f"数据源: {data_source.name}")
    print(f"缓存目录: {cache_dir}")
    print(f"标的数量: {len(codes)}")
    print("=" * 60)

    for i, code in enumerate(codes):
        cache_file = os.path.join(cache_dir, f"{code}_{data_source.name}.csv")
        meta_file = cache_file + ".meta.json"
        action = "刷新" if include_today and os.path.exists(cache_file) else "下载"
        print(f"[{i + 1}/{len(codes)}] {action} {code} ...", flush=True)

        df_new = data_source.fetch(code, target_start, expect_today=include_today)
        if df_new is None or df_new.empty:
            print(f"  [警告] {code} 未返回数据")
            continue

        if os.path.exists(cache_file):
            df_old = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            df_result = _merge_price_data(df_old, df_new)
        else:
            df_result = df_new.copy()

        df_result.to_csv(cache_file)
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump({"adjusted": data_source.adjusted}, f)

        print(f"  [完成] {code} -> {cache_file} ({len(df_result)} 行)")

        if i < len(codes) - 1:
            time.sleep(1)

    print("[完成] 数据拉取结束")


def main():
    parser = argparse.ArgumentParser(description="拉取 ETF 历史数据到缓存")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    parser.add_argument(
        "--today",
        action="store_true",
        help="强制拉取到最新交易日（含当天）",
    )
    parser.add_argument(
        "--provider",
        default="akshare",
        help="数据源，默认 akshare",
    )
    args = parser.parse_args()

    app_config = load_config(args.config)
    data_source = get_data_source(name=args.provider, fallback=False, skip_test=False)
    fetch_all_data(app_config, data_source, include_today=args.today)


if __name__ == "__main__":
    main()
