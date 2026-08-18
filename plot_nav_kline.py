#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轮动策略组合净值 K 线图（最近 N 个交易日）

净值是日频单点数据，K 线构造：开=昨日净值，收=今日净值，高/低=两者极值。
用于观察组合近期波动，辅助止盈/止损决策。

用法:
    python plot_nav_kline.py                # 最近 60 个交易日
    python plot_nav_kline.py --days 90
    python plot_nav_kline.py --today        # 与 latest_signal.py --today 同口径
"""

from __future__ import annotations

import argparse
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle


def plot_nav_kline(df: pd.DataFrame, strategy_name: str, days: int = 120,
                   out: str = "output/nav_kline.png") -> None:
    """绘制组合净值 K 线图（近 days 个交易日），保存到 out 并打印简要统计。"""
    nav = df["轮动策略净值"].dropna().tail(days)
    daily_ret = df["轮动策略日收益率"].reindex(nav.index)

    # 构造 K 线 OHLC：开=昨收净值，收=当日净值
    close = nav.values
    open_ = np.concatenate([[nav.iloc[0]], close[:-1]])
    high = np.maximum(open_, close)
    low = np.minimum(open_, close)

    fig, ax = plt.subplots(figsize=(14, 7))
    up_color, down_color = "#e54545", "#2ca02c"  # 红涨绿跌（A股习惯）
    x = np.arange(len(nav))

    for i in range(len(nav)):
        color = up_color if close[i] >= open_[i] else down_color
        ax.plot([x[i], x[i]], [low[i], high[i]], color=color, linewidth=1)
        body_low, body_h = min(open_[i], close[i]), abs(close[i] - open_[i])
        ax.add_patch(Rectangle((x[i] - 0.35, body_low), 0.7, max(body_h, nav.max() * 1e-5),
                               facecolor=color, edgecolor=color))

    # 净值均线
    for w, c in [(5, "#ff7f0e"), (10, "#1f77b4"), (20, "#9467bd")]:
        ma = nav.rolling(w).mean()
        ax.plot(x, ma.values, color=c, linewidth=1.2, alpha=0.85, label=f"MA{w}")

    # 标注区间峰值与当前回撤
    peak_idx = int(np.argmax(close))
    cur_dd = close[-1] / close.max() - 1.0
    ax.axhline(close.max(), color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.annotate(f"区间峰值 {close.max():.3f} ({nav.index[peak_idx]:%m-%d})",
                xy=(peak_idx, close.max()), xytext=(peak_idx, close.max() * 1.01),
                fontsize=9, color="gray")

    total_ret = close[-1] / open_[0] - 1.0
    ax.set_title(
        f"{strategy_name} — 组合净值K线（近{len(nav)}个交易日）\n"
        f"区间收益 {total_ret:+.2%} | 当前净值 {close[-1]:.3f} | 距区间峰值 {cur_dd:+.2%}",
        fontsize=13,
    )
    ticks = np.linspace(0, len(nav) - 1, 12, dtype=int)
    ax.set_xticks(ticks)
    ax.set_xticklabels([nav.index[i].strftime("%m-%d") for i in ticks])
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    ax.set_xlim(-1, len(nav))
    fig.tight_layout()

    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[净值K线] 已保存: {out}")

    # 辅助决策的简要统计
    recent5 = close[-1] / close[-6] - 1.0 if len(nav) > 5 else np.nan
    recent20 = close[-1] / close[-21] - 1.0 if len(nav) > 20 else np.nan
    print(f"[净值K线] 近5日收益: {recent5:+.2%} | 近20日收益: {recent20:+.2%} | 距区间峰值: {cur_dd:+.2%}")
    print("最近10日净值:")
    for d, v, r in zip(nav.index[-10:], close[-10:], daily_ret.iloc[-10:]):
        print(f"  {d:%Y-%m-%d}  净值 {v:.4f}  日收益 {r:+.2%}")


def main():
    parser = argparse.ArgumentParser(description="轮动策略组合净值 K 线图")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--days", type=int, default=120, help="显示的交易日数量（默认120，约6个月）")
    parser.add_argument("--today", action="store_true", help="使用当天作为截止日")
    args = parser.parse_args()

    from data_source import get_data_source
    from core.orchestrator import compute_signal_start_date, fetch_pool_data, run_strategy
    from utils import load_config

    cutoff_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    app_config = load_config(args.config)
    strategy = next(s for s in app_config.strategies if s.enabled and s.mode == "rotation")

    data_source = get_data_source(name=app_config.data_source.provider, fallback=False, skip_test=True)
    start_date = compute_signal_start_date(strategy, cutoff_date).strftime("%Y%m%d")
    data = fetch_pool_data(
        strategy, app_config, data_source,
        include_today=args.today, cutoff_date=cutoff_date,
        start_date=start_date,
        min_bars=strategy.params.get("lookback", 20) + 5,
        skip_download=True,
    )
    df, name_list = run_strategy(strategy, app_config, data_source, data=data)
    plot_nav_kline(df, strategy.name, days=args.days)


if __name__ == "__main__":
    main()
