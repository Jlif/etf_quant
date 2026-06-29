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
import json
import os
import sys
import warnings

import pandas as pd
import unicodedata

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
from strategy import rotation, weighted
from utils import load_config, AppConfig, StrategyConfig

warnings.filterwarnings("ignore", category=RuntimeWarning)


def _display_width(s: str) -> int:
    """计算字符串在终端中的显示宽度（中文等宽字符按 2 计）。"""
    return sum(
        2 if unicodedata.east_asian_width(ch) in ("F", "W") else 1
        for ch in str(s)
    )


def _ljust(s: str, width: int) -> str:
    """按显示宽度左对齐。"""
    return s + " " * max(0, width - _display_width(s))


def _rjust(s: str, width: int) -> str:
    """按显示宽度右对齐。"""
    return " " * max(0, width - _display_width(s)) + s


def detect_and_fix_price_jumps(
    prices: pd.Series,
    name: str,
    threshold: float = 0.30,
) -> pd.Series:
    """
    检测并修正价格序列中的异常复权跳空。

    yfinance 等数据源对国内 ETF 的复权处理偶尔出错，
    会出现单日涨跌幅远超正常范围（如 -50%）的虚假跳空。
    本函数把这些点当作"复权系数错误"，对前期价格做整体缩放，
    使修正后的序列保持连续。

    Parameters
    ----------
    prices : pd.Series
        收盘价序列
    name : str
        标的名称，用于日志
    threshold : float
        异常阈值，日收益率绝对值超过此值即认为异常

    Returns
    -------
    pd.Series
        修正后的价格序列
    """
    prices = prices.copy().sort_index()
    returns = prices.pct_change().dropna()

    fixed = prices.copy()
    # 从最早日期开始扫描，避免多次修正相互影响
    for date in returns.index:
        daily_ret = returns.loc[date]
        if abs(daily_ret) > threshold:
            prev_date = returns.index[returns.index.get_loc(date) - 1]
            factor = fixed.loc[date] / fixed.loc[prev_date]
            # 将 prev_date 及之前所有价格乘以 factor，使序列连续
            mask = fixed.index <= prev_date
            fixed.loc[mask] *= factor
            direction = "下跌" if daily_ret < 0 else "上涨"
            print(
                f"  [数据修正] {name} 在 {date.date()} 出现异常{direction} "
                f"({daily_ret:+.2%})，已整体缩放前期价格 (factor={factor:.4f})"
            )
            # 重新计算后续收益率
            returns = fixed.pct_change().dropna()

    return fixed


def fetch_pool_data(
    strategy: StrategyConfig,
    app_config: AppConfig,
    data_source,
    include_today: bool = False,
):
    """获取策略候选池数据，自动对齐起始日期。

    Parameters
    ----------
    include_today : bool
        为 True 时强制重新拉取数据，确保包含最新行情（含当天）。
    """
    codes = [p.code for p in strategy.pool]
    names = {p.code: p.name for p in strategy.pool}
    cache_dir = app_config.backtest.cache_dir
    os.makedirs(cache_dir, exist_ok=True)

    # 策略级起始日优先于全局起始日
    target_start = strategy.start_date or app_config.backtest.start_date
    target_start_dt = pd.to_datetime(target_start)

    all_close = {}
    all_open = {}
    all_high = {}
    all_low = {}
    actual_starts = {}  # 记录每个 ETF 实际数据起始日
    for code in codes:
        name = names[code]
        cache_file = os.path.join(cache_dir, f"{code}_{data_source.name}.csv")
        meta_file = cache_file + ".meta.json"

        if os.path.exists(cache_file) and not include_today:
            print(f"  [缓存] {code} ({name})")
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            close_col = f"{code}_close"
            if close_col not in df.columns:
                print(f"  [缓存格式旧] 重新下载 {code} ({name})")
                df = data_source.fetch(code, target_start)
                df.to_csv(cache_file)
                with open(meta_file, "w", encoding="utf-8") as f:
                    json.dump({"adjusted": data_source.adjusted}, f)
            elif os.path.exists(meta_file):
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                data_source.adjusted = meta.get("adjusted", True)
        else:
            action = "刷新" if include_today and os.path.exists(cache_file) else "下载"
            print(f"  [{action}] {code} ({name}) via {data_source.name}")
            df = data_source.fetch(code, target_start)
            df.to_csv(cache_file)
            with open(meta_file, "w", encoding="utf-8") as f:
                json.dump({"adjusted": data_source.adjusted}, f)

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
            print(f"  [缓存旧格式] 仅使用收盘价: {code} ({name})")
            all_close[name] = df[code]
            all_open[name] = df[code]
            all_high[name] = df[code]
            all_low[name] = df[code]
        else:
            raise ValueError(
                f"{cache_file} 中未找到 {close_col} 或 {code} 列"
            )
        actual_starts[name] = df.index[0]

    # 异常复权跳空修正（以 close 为准，同比例缩放 open/high/low）
    # 仅对明确为复权/已调整价格的数据源启用；未复权数据中的大跳空通常是
    # 真实拆股/分红，修正反而会扭曲历史收益。
    if data_source.adjusted:
        for name in all_close:
            fixed_close = detect_and_fix_price_jumps(all_close[name], name)
            ratio = fixed_close / all_close[name]
            all_close[name] = fixed_close
            all_open[name] = all_open[name] * ratio
            all_high[name] = all_high[name] * ratio
            all_low[name] = all_low[name] * ratio
    else:
        print("  [数据] 当前数据源返回未复权价格，跳过复权跳空修正")

    data_close = pd.DataFrame(all_close)
    data_open = pd.DataFrame(all_open)
    data_high = pd.DataFrame(all_high)
    data_low = pd.DataFrame(all_low)

    # 策略实际起始日 = max(配置起始日, 所有 ETF 中最晚的数据起始日)
    latest_etf_start = max(actual_starts.values())
    effective_start = max(target_start_dt, latest_etf_start)

    if latest_etf_start > target_start_dt:
        print(f"  [注意] 配置起始日 {target_start_dt.date()} 早于部分 ETF 数据起始日")
        for name, st in actual_starts.items():
            if st > target_start_dt:
                print(f"         {name} 实际起始: {st.date()}")

    if effective_start != data_close.index[0]:
        print(f"  [调整] 策略实际起始日: {effective_start.date()}")
        data_close = data_close.loc[data_close.index >= effective_start]
        data_open = data_open.loc[data_open.index >= effective_start]
        data_high = data_high.loc[data_high.index >= effective_start]
        data_low = data_low.loc[data_low.index >= effective_start]

    print(f"  时间范围: {data_close.index[0].date()} ~ {data_close.index[-1].date()}, 共 {len(data_close)} 条")
    return {
        "close": data_close,
        "open": data_open,
        "high": data_high,
        "low": data_low,
    }


def run_strategy(
    strategy: StrategyConfig,
    app_config: AppConfig,
    data_source,
    silent: bool = False,
    data: dict | None = None,
    include_today: bool = False,
):
    """执行单个策略回测

    Parameters
    ----------
    silent : bool
        为 True 时跳过报告、信号打印和文件输出，仅返回结果，
        用于参数扫描等批量场景。
    data : dict | None
        预加载的池数据（由 fetch_pool_data 返回）。传入后可避免重复读取缓存，
        在参数扫描等批量场景下减少 I/O 和日志输出。
    include_today : bool
        为 True 时强制重新拉取数据，确保包含最新行情（含当天）。
    """
    if not silent:
        print(f"\n{'='*60}")
        print(f"【{strategy.name}】{strategy.description}")
        print(f"  模式: {strategy.mode} | 参数: {strategy.params}")
        print(f"{'='*60}")

    if data is None:
        data = fetch_pool_data(strategy, app_config, data_source, include_today=include_today)
    name_list = data["close"].columns.tolist()

    if strategy.mode == "rotation":
        result = rotation.run(data, name_list, strategy.params)
        # 以第一个标的为基准（净值序列，从1开始）
        benchmark_col = f"{name_list[0]}净值"
        for name in name_list:
            result[f"{name}净值"] = result[name] / result[name].iloc[0]
    elif strategy.mode == "weighted":
        weights = {p.name: p.weight for p in strategy.pool}
        result = weighted.run(data, name_list, weights, strategy.params)
        for name in name_list:
            result[f"{name}净值"] = result[name] / result[name].iloc[0]
        benchmark_col = None
    else:
        raise ValueError(f"不支持的模式: {strategy.mode}")

    if silent:
        return result, name_list

    # 绩效报告 - benchmark需要传入日收益率序列，而不是净值序列
    benchmark_series = result[benchmark_col] if benchmark_col and benchmark_col in result.columns else None
    # 将净值序列转换为日收益率序列
    if benchmark_series is not None:
        benchmark_returns = benchmark_series.pct_change().fillna(0)
        benchmark_returns.name = benchmark_col  # 保持名称用于报告展示
    else:
        benchmark_returns = None

    # quantstats 在当前版本下对净值序列的累计收益计算有兼容性问题，
    # 直接传入日收益率序列可得到正确的滚动收益指标。
    strategy_returns = result["轮动策略净值"].pct_change().fillna(0)
    strategy_returns.name = "轮动策略净值"

    # 提前构建换仓记录，供 HTML 报告和 CSV 使用
    rebalance_df = None
    if strategy.mode == "rotation" and "换仓" in result.columns:
        rebalance_df = _build_rebalance_df(result)

    performance_report(
        strategy_returns,
        benchmark=benchmark_returns,
        title=f"{strategy.name}回测报告",
        rebalance_df=rebalance_df,
    )

    # 输出最新交易信号（用于实盘调仓）
    # print_latest_signal(strategy, result, name_list)

    # 输出各 ETF 持有天数与收益贡献占比（仅 rotation 策略）
    if strategy.mode == "rotation":
        print_position_contribution(strategy, result, name_list)

    # 输出轮动策略换仓记录 CSV
    if strategy.mode == "rotation" and "换仓" in result.columns:
        if rebalance_df is not None and not rebalance_df.empty:
            safe_name = strategy.name.replace(" ", "_").replace(":", "_")
            csv_path = os.path.join(OUTPUT_DIR, f"{safe_name}_换仓记录.csv")
            rebalance_df.to_csv(csv_path, encoding="utf-8-sig")
            print(f"[换仓记录已保存] {csv_path}")

    return result, name_list


def _build_rebalance_df(result: pd.DataFrame) -> pd.DataFrame | None:
    """从 rotation 策略结果中提取换仓记录 DataFrame。"""
    if "换仓" not in result.columns or "持仓" not in result.columns:
        return None

    changes = result[result["换仓"]].copy()
    if changes.empty:
        return None

    def _parse_holding_names(holding_str: str) -> set[str]:
        return {
            part.split("(")[0].strip()
            for part in str(holding_str).split("+")
            if part.strip()
        }

    prev_holdings = result["持仓"].shift(1)
    changes["调出"] = [
        "、".join(sorted(_parse_holding_names(prev) - _parse_holding_names(cur)))
        for prev, cur in zip(prev_holdings.loc[changes.index], changes["持仓"])
    ]
    changes["调入"] = [
        "、".join(sorted(_parse_holding_names(cur) - _parse_holding_names(prev)))
        for prev, cur in zip(prev_holdings.loc[changes.index], changes["持仓"])
    ]

    nav_col = "轮动策略净值"
    changes["换仓前净值"] = result[nav_col].shift(1).loc[changes.index]
    changes["换仓后净值"] = result[nav_col].loc[changes.index]

    rebalance_df = changes[["持仓", "调出", "调入", "换仓前净值", "换仓后净值"]]
    rebalance_df.index.name = "日期"
    return rebalance_df


def print_latest_signal(strategy: StrategyConfig, result: pd.DataFrame, name_list: list[str]):
    """打印最新交易信号，方便实盘操作"""
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

        header = (
            f"{_ljust('排名', 4)} "
            f"{_ljust('ETF名称', 20)} "
            f"{_ljust('代码', 10)} "
            f"{_ljust('周期动量得分', 12)} "
            f"{_ljust('建议仓位', 10)}"
        )
        print(header)
        print("-" * 60)

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
            rank_str = f"{marker}{i}"
            row = (
                f"{_ljust(rank_str, 4)} "
                f"{_ljust(name, 20)} "
                f"{_ljust(code, 10)} "
                f"{_ljust(score_str, 12)} "
                f"{_ljust(weight_pct, 10)}"
            )
            print(row)

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
        header = (
            f"{_ljust('ETF名称', 20)} "
            f"{_ljust('代码', 10)} "
            f"{_ljust('目标权重', 10)} "
            f"{_ljust('当前权重', 10)}"
        )
        print(header)
        print("-" * 54)

        weights = {p.name: p.weight for p in strategy.pool}
        for name in name_list:
            code = next((p.code for p in strategy.pool if p.name == name), "")
            target = weights.get(name, 0)
            row = (
                f"{_ljust(name, 20)} "
                f"{_ljust(code, 10)} "
                f"{_ljust(f'{target:.0f}%', 10)} "
                f"{_ljust('-', 10)}"
            )
            print(row)

        print(f"{'='*60}")
        print("操作建议: 按上述目标权重配置仓位")

    print(f"{'='*60}")


def print_position_contribution(strategy: StrategyConfig, result: pd.DataFrame, name_list: list[str]):
    """打印 rotation 策略各 ETF 的持有天数与收益贡献占比"""
    final_nav = result["轮动策略净值"].iloc[-1]
    total_return = final_nav - 1.0

    # 定义列宽（显示宽度，中文按2计）
    col_widths = {
        "ETF名称": 18,
        "代码": 10,
        "持有天数": 10,
        "累计贡献": 12,
        "贡献占比": 12,
    }

    total_width = sum(col_widths.values()) + len(col_widths) - 1

    print(f"\n{'='*total_width}")
    print(f"【持仓统计与收益贡献】{strategy.name}")
    print(f"{'='*total_width}")

    # 构建表头
    headers = [
        _ljust("ETF名称", col_widths["ETF名称"]),
        _ljust("代码", col_widths["代码"]),
        _ljust("持有天数", col_widths["持有天数"]),
        _rjust("累计贡献", col_widths["累计贡献"]),
        _rjust("贡献占比", col_widths["贡献占比"]),
    ]
    print(" ".join(headers))
    print("-" * total_width)

    rows = []
    total_contribution = 0.0
    for name in name_list:
        code = next((p.code for p in strategy.pool if p.name == name), "")
        hold_days = int((result[f"权重_{name}"] > 0).sum())
        contrib_col = f"贡献_日收益_{name}"
        if contrib_col in result.columns:
            contribution = result[contrib_col].sum()
        else:
            contribution = 0.0
        total_contribution += contribution
        rows.append((name, code, hold_days, contribution))

    # 按贡献占比排序
    abs_total = abs(total_contribution) if total_contribution != 0 else 1.0
    rows_with_ratio = [
        (name, code, hold_days, contribution, contribution / abs_total)
        for name, code, hold_days, contribution in rows
    ]
    rows_with_ratio.sort(key=lambda x: x[4], reverse=True)

    for name, code, hold_days, contribution, ratio in rows_with_ratio:
        contrib_str = f"{contribution:+.2%}"
        ratio_str = f"{ratio:+.1%}"
        row = [
            _ljust(name, col_widths["ETF名称"]),
            _ljust(code, col_widths["代码"]),
            _ljust(str(hold_days), col_widths["持有天数"]),
            _rjust(contrib_str, col_widths["累计贡献"]),
            _rjust(ratio_str, col_widths["贡献占比"]),
        ]
        print(" ".join(row))

    print("-" * total_width)
    print(f"合计贡献（简单加总口径）: {total_contribution:+.2%}")
    print(f"策略累计收益（复利口径）:   {total_return:+.2%}")
    print("  注：因复利再投资效应，简单加总的贡献合计会低于复利累计收益")
    print(f"{'='*total_width}")


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

    # 可视化
    plot_nav_curves(
        {k: (v[0], v[1], "轮动策略净值") for k, v in all_results.items()}
    )
    plot_strategy_comparison(nav_series)
    print_summary(nav_series)


if __name__ == "__main__":
    main()
