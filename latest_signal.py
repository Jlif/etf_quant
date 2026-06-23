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
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from data_source import get_data_source
from strategy import rotation, weighted
from utils import load_config, AppConfig, StrategyConfig


def is_trading_day(date: datetime) -> bool:
    """判断是否为交易日（简单排除周末）"""
    return date.weekday() < 5


def get_last_trading_day(date: datetime | None = None) -> datetime:
    """获取上一个交易日"""
    if date is None:
        date = datetime.now()
    # 回退一天，确保是上一个交易日
    date -= timedelta(days=1)
    # 如果回退后是周末，继续回退到最近的工作日
    while not is_trading_day(date):
        date -= timedelta(days=1)
    return date


def fetch_latest_data(strategy: StrategyConfig, app_config: AppConfig, data_source, cutoff_date: datetime | None = None):
    """
    获取策略候选池最新数据，智能更新缓存。

    与 main.py 的 fetch_pool_data 类似，但：
    1. 计算所需历史数据长度（lookback + 缓冲）
    2. 智能判断缓存是否需要更新
    3. 仅获取必要的历史数据
    4. 支持指定截止日，过滤掉截止日之后的数据
    """
    codes = [p.code for p in strategy.pool]
    names = {p.code: p.name for p in strategy.pool}
    cache_dir = app_config.backtest.cache_dir
    os.makedirs(cache_dir, exist_ok=True)

    # 计算所需数据起始日
    lookback = strategy.params.get("lookback", 20)
    buffer_days = 50  # 额外缓冲，确保有足够数据计算指标

    today = datetime.now()
    required_start = today - timedelta(days=lookback + buffer_days + 10)  # +10 应对节假日
    required_start_str = required_start.strftime("%Y%m%d")

    # 确定截止日（默认上一个交易日）
    if cutoff_date is None:
        cutoff_date = get_last_trading_day()

    all_close = {}
    all_open = {}
    all_high = {}
    all_low = {}

    for code in codes:
        name = names[code]
        cache_file = os.path.join(cache_dir, f"{code}_{data_source.name}.csv")
        meta_file = cache_file + ".meta.json"

        need_download = True
        df = None

        if os.path.exists(cache_file):
            # 检查缓存是否最新
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            if not df.empty:
                last_date = df.index[-1]
                last_trading = get_last_trading_day()

                # 如果缓存最新日期就是最近交易日，且数据足够，使用缓存
                if last_date.date() >= last_trading.date() and len(df) >= lookback + buffer_days:
                    print(f"  [缓存] {code} ({name}) 已是最新")
                    need_download = False
                else:
                    print(f"  [更新] {code} ({name}) 缓存过期或数据不足")
            else:
                print(f"  [更新] {code} ({name}) 缓存为空")

        if need_download:
            print(f"  [下载] {code} ({name}) via {data_source.name}")
            try:
                df = data_source.fetch(code, required_start_str)
                # 下载成功后，先过滤截止日，再保存缓存
                if df is not None and not df.empty:
                    df_before_cutoff = df[df.index.date <= cutoff_date.date()]
                    if len(df_before_cutoff) < len(df):
                        print(f"  [过滤] {code} ({name}) 截断至 {cutoff_date.date()}，原 {len(df)} 条 -> {len(df_before_cutoff)} 条")
                    df = df_before_cutoff
                # 确保缓存目录存在
                os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                df.to_csv(cache_file)
                with open(meta_file, "w", encoding="utf-8") as f:
                    json.dump({"adjusted": data_source.adjusted}, f)
            except Exception as e:
                print(f"  [警告] {code} ({name}) 下载失败: {e}")
                # 如果下载失败但缓存存在，回退使用缓存
                if os.path.exists(cache_file):
                    df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                    print(f"  [回退] 使用缓存数据: {code} ({name})")
                else:
                    raise

        # 过滤截止日之后的数据（对缓存数据也执行）
        if df is not None and not df.empty:
            df_before_cutoff = df[df.index.date <= cutoff_date.date()]
            if len(df_before_cutoff) < len(df):
                print(f"  [过滤] {code} ({name}) 截断至 {cutoff_date.date()}，原 {len(df)} 条 -> {len(df_before_cutoff)} 条")
            df = df_before_cutoff

        close_col = f"{code}_close"
        open_col = f"{code}_open"
        high_col = f"{code}_high"
        low_col = f"{code}_low"

        if close_col in df.columns:
            all_close[name] = df[close_col]
            all_open[name] = df[open_col]
            all_high[name] = df[high_col]
            all_low[name] = df[low_col]
        elif code in df.columns:
            # 兼容旧版单收盘价缓存
            all_close[name] = df[code]
            all_open[name] = df[code]
            all_high[name] = df[code]
            all_low[name] = df[code]
        else:
            raise ValueError(f"{cache_file} 中未找到 {close_col} 或 {code} 列")

    data_close = pd.DataFrame(all_close)
    data_open = pd.DataFrame(all_open)
    data_high = pd.DataFrame(all_high)
    data_low = pd.DataFrame(all_low)

    # 确保数据足够计算 lookback
    if len(data_close) < lookback + 5:
        raise ValueError(f"数据不足: 仅 {len(data_close)} 条，需要至少 {lookback + 5} 条")

    print(f"  时间范围: {data_close.index[0].date()} ~ {data_close.index[-1].date()}, 共 {len(data_close)} 条")

    return {
        "close": data_close,
        "open": data_open,
        "high": data_high,
        "low": data_low,
    }


def print_latest_signal(strategy: StrategyConfig, result: pd.DataFrame, name_list: list[str]):
    """
    打印最新交易信号，方便实盘操作。

    从 main.py 提取并复用，保持输出格式一致。
    """
    latest = result.iloc[-1]
    prev = result.iloc[-2] if len(result) > 1 else None

    print(f"\n{'='*60}")
    print(f"【今日交易信号】{strategy.name}")
    print(f"{'='*60}")
    print(f"信号日期: {result.index[-1].date()}")
    print(f"策略参数: lookback={strategy.params.get('lookback', 20)}日, top_n={strategy.params.get('top_n', 1)}")
    print(f"{'-'*60}")

    if strategy.mode == "rotation":
        # 动量轮动策略：显示排名和得分
        scoring = strategy.params.get("scoring", "momentum")
        prefix = "得分_" if scoring == "slope_r2" else "涨幅_"

        print(f"{'排名':<4} {'ETF名称':<20} {'代码':<10} {'周期动量得分':<12} {'建议仓位':<10}")
        print(f"{'-'*60}")

        # 按得分排序
        scores = []
        for name in name_list:
            score_col = f"{prefix}{name}"
            score = latest[score_col] if score_col in latest else 0
            weight = latest[f"权重_{name}"] if f"权重_{name}" in latest else 0
            # 获取代码
            code = next((p.code for p in strategy.pool if p.name == name), "")
            scores.append((name, code, score, weight))

        scores.sort(key=lambda x: x[2], reverse=True)

        for i, (name, code, score, weight) in enumerate(scores, 1):
            marker = "★" if weight > 0 else " "
            weight_pct = f"{weight*100:.0f}%" if weight > 0 else "0%"
            score_str = f"{score:+.2%}" if scoring == "momentum" else f"{score:.4f}"
            print(f"{marker}{i:<3} {name:<20} {code:<10} {score_str:<12} {weight_pct:<10}")

        # 显示持仓变化
        if prev is not None:
            print(f"\n{'-'*60}")
            print("持仓变化:")
            current_holdings = [name for name in name_list if latest[f"权重_{name}"] > 0]
            prev_holdings = [name for name in name_list if prev[f"权重_{name}"] > 0]

            added = set(current_holdings) - set(prev_holdings)
            removed = set(prev_holdings) - set(current_holdings)

            if added:
                for name in added:
                    code = next((p.code for p in strategy.pool if p.name == name), "")
                    print(f"  [买入] {name} ({code})")
            if removed:
                for name in removed:
                    code = next((p.code for p in strategy.pool if p.name == name), "")
                    print(f"  [卖出] {name} ({code})")
            if not added and not removed:
                print("  [维持] 持仓不变")

        # 打印风控触发原因（rotation 策略专用）
        risk_reason = latest.get("风控原因", "")
        if risk_reason:
            print(f"{'-'*60}")
            print("风控说明:")
            for reason in str(risk_reason).split(";"):
                reason = reason.strip()
                if reason:
                    print(f"  · {reason}")

        print(f"{'='*60}")
        print("操作建议:")
        holdings = [(name, latest[f"权重_{name}"]) for name in name_list if latest[f"权重_{name}"] > 0]
        for name, weight in holdings:
            code = next((p.code for p in strategy.pool if p.name == name), "")
            print(f"  持有 {name} ({code}): {weight*100:.0f}%")

    elif strategy.mode == "weighted":
        # 加权组合策略：显示目标权重
        print(f"{'ETF名称':<20} {'代码':<10} {'目标权重':<10} {'当前权重':<10}")
        print(f"{'-'*60}")

        weights = {p.name: p.weight for p in strategy.pool}
        for name in name_list:
            code = next((p.code for p in strategy.pool if p.name == name), "")
            target = weights.get(name, 0)
            print(f"{name:<20} {code:<10} {target:.0f}%")

        print(f"{'='*60}")
        print("操作建议: 按上述目标权重配置仓位")

    print(f"{'='*60}")


def run_latest_signal(strategy: StrategyConfig, app_config: AppConfig, data_source, cutoff_date: datetime | None = None):
    """
    执行单个策略的最新信号计算和打印。
    """
    print(f"\n{'='*60}")
    print(f"【{strategy.name}】{strategy.description}")
    print(f"  模式: {strategy.mode} | 参数: {strategy.params}")
    print(f"{'='*60}")

    data = fetch_latest_data(strategy, app_config, data_source, cutoff_date)
    name_list = data["close"].columns.tolist()

    if strategy.mode == "rotation":
        result = rotation.run(data, name_list, strategy.params)
    elif strategy.mode == "weighted":
        weights = {p.name: p.weight for p in strategy.pool}
        result = weighted.run(data, name_list, weights, strategy.params)
    else:
        raise ValueError(f"不支持的模式: {strategy.mode}")

    # 仅打印最新信号，不生成报告
    print_latest_signal(strategy, result, name_list)

    return result, name_list


def main():
    parser = argparse.ArgumentParser(description="ETF 最新信号打印")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    parser.add_argument("--strategy", help="指定策略名称（默认运行所有启用策略）")
    parser.add_argument("--date", help="指定交易截止日 (YYYYMMDD)，默认上一个交易日")
    args = parser.parse_args()

    # 解析指定日期
    cutoff_date = None
    if args.date:
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

    # 检查是否所有缓存都已存在
    all_cached = all(
        os.path.exists(os.path.join(cache_dir, f"{code}_{provider}.csv"))
        for code in required_codes
    )
    skip_test = all_cached
    if skip_test:
        print(f"[缓存] 所有 {len(required_codes)} 个 ETF 已缓存，跳过数据源连通性测试")

    # 初始化数据源
    data_source = get_data_source(
        name=provider,
        fallback=True,
        skip_test=skip_test,
    )

    # 运行启用的策略
    for strategy in enabled_strategies:
        try:
            run_latest_signal(strategy, app_config, data_source, cutoff_date)
        except Exception as e:
            print(f"\n[错误] 策略 '{strategy.name}' 执行失败: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
