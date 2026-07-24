"""统一回测编排层

为 main.py、latest_signal.py、risk_param_sweep.py 提供一致的数据获取、
策略执行、信号打印与持仓记录逻辑。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from strategy import rotation, weighted
from strategy._common import required_window, weight_col
from utils import AppConfig, StrategyConfig
from utils.text import display_width, ljust, rjust
from core.report import OUTPUT_DIR, performance_report


def clear_output_dir(output_dir: str = OUTPUT_DIR) -> None:
    """清空输出目录。"""
    if os.path.exists(output_dir):
        for f in os.listdir(output_dir):
            os.remove(os.path.join(output_dir, f))


def calculate_benchmark_returns(
    result: pd.DataFrame, name_list: list[str]
) -> pd.Series | None:
    """从 rotation/weighted 结果中提取首个标的的日收益率序列作为 benchmark。"""
    benchmark_col = f"{name_list[0]}净值"
    benchmark_series = result[benchmark_col] if benchmark_col in result.columns else None
    if benchmark_series is None:
        return None
    returns = benchmark_series.pct_change(fill_method=None).fillna(0)
    returns.name = benchmark_col
    return returns


def save_holding_csv(
    holding_df: pd.DataFrame,
    strategy_name: str,
    output_dir: str = OUTPUT_DIR,
) -> str | None:
    """保存 rotation 策略的每日持仓记录 CSV。"""
    if holding_df is None or holding_df.empty:
        return None
    safe_name = strategy_name.replace(" ", "_").replace(":", "_")
    csv_path = os.path.join(output_dir, f"{safe_name}_持仓记录.csv")
    holding_df.to_csv(csv_path, encoding="utf-8-sig")
    return csv_path


def report_strategy_result(
    strategy: StrategyConfig,
    result: pd.DataFrame,
    name_list: list[str],
) -> None:
    """为单个策略输出绩效报告、持仓 CSV 与收益贡献统计。"""
    strategy_returns = result["轮动策略净值"].pct_change(fill_method=None).fillna(0)
    strategy_returns.name = "轮动策略净值"

    benchmark_returns = calculate_benchmark_returns(result, name_list)
    holding_df = build_holding_df(result)

    performance_report(
        strategy_returns,
        benchmark=benchmark_returns,
        title=f"{strategy.name}回测报告",
        holding_df=holding_df,
    )

    if strategy.mode == "rotation":
        print_position_contribution(strategy, result, name_list)

    csv_path = save_holding_csv(holding_df, strategy.name)
    if csv_path:
        print(f"[持仓记录已保存] {csv_path}")
        print_holding_summary(holding_df, strategy.name)


def detect_and_fix_price_jumps(
    prices: pd.Series,
    name: str,
    threshold: float = 0.30,
) -> pd.Series:
    """
    检测并修正价格序列中的异常复权跳空。

    部分数据源对国内 ETF 的复权处理偶尔出错，
    会出现单日涨跌幅远超正常范围（如 -50%）的虚假跳空。
    本函数把这些点当作"复权系数错误"，对前期价格做整体缩放，
    使修正后的序列保持连续。
    """
    prices = prices.copy().sort_index()
    returns = prices.pct_change(fill_method=None).dropna()

    fixed = prices.copy()
    for date in returns.index:
        daily_ret = returns.loc[date]
        if abs(daily_ret) > threshold:
            prev_date = returns.index[returns.index.get_loc(date) - 1]
            factor = fixed.loc[date] / fixed.loc[prev_date]
            mask = fixed.index <= prev_date
            fixed.loc[mask] *= factor
            direction = "下跌" if daily_ret < 0 else "上涨"
            print(
                f"  [数据修正] {name} 在 {date.date()} 出现异常{direction} "
                f"({daily_ret:+.2%})，已整体缩放前期价格 (factor={factor:.4f})"
            )
            returns = fixed.pct_change().dropna()

    return fixed


def compute_signal_start_date(
    strategy: StrategyConfig,
    cutoff_date: datetime,
    buffer_days: int = 20,
) -> datetime:
    """
    根据策略 lookback 与三层风控参数，估算获取最新信号所需的历史起始日。

    仅用于 latest_signal.py 等只需要最近一段数据的场景。
    """
    lookback = strategy.params.get("lookback", 20)

    risk_lookbacks = [lookback]
    risk_control = strategy.params.get("risk_control", {})
    if risk_control.get("layer1", {}).get("enabled"):
        risk_lookbacks.append(risk_control["layer1"].get("ma_lookback", 10))
    if risk_control.get("layer2", {}).get("enabled"):
        risk_lookbacks.append(risk_control["layer2"].get("atr_lookback", 14))
    if risk_control.get("layer3", {}).get("enabled"):
        risk_lookbacks.append(risk_control["layer3"].get("vol_lookback", 23))

    required_trading_days = max(risk_lookbacks) + buffer_days
    # 交易日约占日历日的 ~5/7，2 倍余量可覆盖周末和节假日
    calendar_days = int(required_trading_days * 2) + 5
    return cutoff_date - timedelta(days=calendar_days)


def _merge_price_data(old_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    """合并旧缓存与新下载数据，新数据覆盖重叠日期，结果按日期排序。"""
    if old_df is None or old_df.empty:
        return new_df.copy()
    if new_df is None or new_df.empty:
        return old_df.copy()
    # 去重：new_df 优先
    merged = pd.concat([old_df, new_df])
    merged = merged[~merged.index.duplicated(keep="last")]
    return merged.sort_index()


def _filter_by_cutoff(df: pd.DataFrame, cutoff_date: datetime) -> pd.DataFrame:
    """按 cutoff_date 截断数据。"""
    if df is None or df.empty:
        return df
    return df[df.index.date <= cutoff_date.date()]


def fetch_pool_data(
    strategy: StrategyConfig,
    app_config: AppConfig,
    data_source,
    *,
    include_today: bool = False,
    cutoff_date: datetime | None = None,
    start_date: str | None = None,
    min_bars: int | None = None,
    silent: bool = False,
    skip_download: bool = True,
) -> dict[str, pd.DataFrame]:
    """获取策略候选池 OHLC 数据，自动对齐起始日期并修正异常复权跳空。

    Parameters
    ----------
    include_today : bool
        为 True 时强制重新拉取数据，确保包含最新行情（含当天）。
    cutoff_date : datetime | None
        仅保留 <= cutoff_date 的数据，latest_signal.py 使用。
    start_date : str | None
        覆盖默认 target_start；latest_signal.py 用于只取最近必要历史。
        格式与 config 一致：YYYYMMDD。
    min_bars : int | None
        过滤后要求的最少条数，不足时抛 ValueError。
    silent : bool
        为 True 时抑制所有进度打印，risk_param_sweep 使用。
    skip_download : bool
        为 True 时强制使用本地缓存，不触发任何网络下载；缓存缺失则报错。
    """
    if skip_download:
        include_today = False
    codes = [p.code for p in strategy.pool]
    names = {p.code: p.name for p in strategy.pool}
    cache_dir = app_config.backtest.cache_dir
    os.makedirs(cache_dir, exist_ok=True)

    # 策略级起始日优先于全局起始日；start_date 参数优先级最高
    target_start = start_date or strategy.start_date or app_config.backtest.start_date
    target_start_dt = pd.to_datetime(target_start)

    # 回测截止日：显式 cutoff_date 参数 > 策略级 end_date > 全局 backtest.end_date
    if cutoff_date is None:
        end_date_str = strategy.end_date or app_config.backtest.end_date
        if end_date_str:
            cutoff_date = pd.to_datetime(end_date_str)

    if cutoff_date is not None and cutoff_date < target_start_dt:
        raise ValueError(
            f"回测截止日 {cutoff_date.date()} 早于起始日 {target_start_dt.date()}"
        )

    all_close = {}
    all_open = {}
    all_high = {}
    all_low = {}
    actual_starts = {}

    # 第一遍：检查缓存，记录需要下载的
    cached_dfs = {}       # code -> df（缓存足够）
    download_tasks = []   # [(code, cache_file, meta_file, df_cached), ...]
    
    for code in codes:
        name = names[code]
        cache_file = os.path.join(cache_dir, f"{code}_{data_source.name}.csv")
        meta_file = cache_file + ".meta.json"

        df = None
        cache_sufficient = False

        if os.path.exists(cache_file) and not include_today:
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            close_col = f"{code}_close"
            if close_col not in df.columns:
                if not silent:
                    print(f"  [缓存格式旧] 重新下载 {code} ({name})")
                df = None
            else:
                if os.path.exists(meta_file):
                    with open(meta_file, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    data_source.adjusted = meta.get("adjusted", True)

                cache_covers_start = df.index[0] <= target_start_dt + timedelta(days=30)
                if cutoff_date is not None:
                    df_before = _filter_by_cutoff(df, cutoff_date)
                    last_date = df_before.index[-1] if not df_before.empty else None
                    cache_reaches_cutoff = (
                        last_date is not None and last_date.date() >= cutoff_date.date()
                    )
                    enough_bars = min_bars is None or len(df_before) >= min_bars
                    if cache_covers_start and cache_reaches_cutoff and enough_bars:
                        if not silent:
                            print(f"  [缓存] {code} ({name}) 已是最新")
                        cached_dfs[code] = df_before
                        cache_sufficient = True
                    elif not cache_covers_start and not silent:
                        print(
                            f"  [更新] {code} ({name}) 缓存起始 {df.index[0].date()} "
                            f"晚于目标起始 {target_start_dt.date()}"
                        )
                    elif not cache_reaches_cutoff and not silent:
                        print(
                            f"  [更新] {code} ({name}) 缓存最新日期 {last_date.date()} "
                            f"早于截止日 {cutoff_date.date()}"
                        )
                    elif not enough_bars and not silent:
                        print(
                            f"  [更新] {code} ({name}) 缓存数据不足 "
                            f"({len(df_before)} < {min_bars})"
                        )
                else:
                    if cache_covers_start:
                        if not silent:
                            print(f"  [缓存] {code} ({name})")
                        cached_dfs[code] = df
                        cache_sufficient = True
                    elif not silent:
                        print(
                            f"  [更新] {code} ({name}) 缓存起始 {df.index[0].date()} "
                            f"晚于目标起始 {target_start_dt.date()}"
                        )

        if not cache_sufficient:
            if skip_download:
                if df is not None and not df.empty:
                    if not silent:
                        print(f"  [强制缓存] {code} ({name}) 跳过下载，使用现有缓存")
                    if cutoff_date is not None:
                        df = _filter_by_cutoff(df, cutoff_date)
                    cached_dfs[code] = df
                else:
                    raise ValueError(
                        f"{code} ({name}) 无本地缓存，"
                        f"请先运行 python fetch_data.py --config ... 拉取数据"
                    )
            else:
                download_tasks.append((code, cache_file, meta_file, df))
    
    # 并发下载需要更新的 ETF（最多 5 个并发）
    downloaded_dfs = {}
    if download_tasks:
        if not silent and len(download_tasks) > 1:
            print(f"  [并发下载] {len(download_tasks)} 个 ETF...")
        
        def _download_one(code, cache_file, meta_file, df_cached):
            """下载单个 ETF"""
            name = names[code]
            action = "刷新" if include_today and os.path.exists(cache_file) else "下载"
            if not silent:
                print(f"  [{action}] {code} ({name}) via {data_source.name}")
            try:
                df_new = data_source.fetch(code, target_start, expect_today=include_today)
                if os.path.exists(cache_file):
                    df_old = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                    df_result = _merge_price_data(df_old, df_new)
                else:
                    df_result = df_new.copy()
                df_result.to_csv(cache_file)
                with open(meta_file, "w", encoding="utf-8") as f:
                    json.dump({"adjusted": data_source.adjusted}, f)
                return (code, df_result, None)
            except Exception as e:
                return (code, df_cached, e)
        
        failed_downloads = []  # [(code, name, error_msg), ...]
        with ThreadPoolExecutor(max_workers=min(2, len(download_tasks))) as executor:
            futures = [
                executor.submit(_download_one, code, cache_file, meta_file, df_cached)
                for code, cache_file, meta_file, df_cached in download_tasks
            ]
            for future in as_completed(futures):
                code, df_result, error = future.result()
                name = names[code]
                if error:
                    if not silent:
                        error_msg = str(error)
                        if "RemoteDisconnected" in error_msg:
                            print(f"  [警告] {code} ({name}) 网络连接失败，尝试回退缓存")
                        elif "腾讯接口" in error_msg:
                            tencent_msg = error_msg.split("腾讯接口")[-1].strip()
                            print(f"  [警告] {code} ({name}) 腾讯接口{tencent_msg}，尝试回退缓存")
                        else:
                            print(f"  [警告] {code} ({name}) 下载失败: {error_msg[:100]}")
                    cache_file = next(cf for c, cf, _, _ in download_tasks if c == code)
                    if os.path.exists(cache_file):
                        df_result = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                        if not silent:
                            print(f"  [回退] 使用缓存数据: {code} ({name})")
                    else:
                        raise error
                    failed_downloads.append((code, name, str(error)[:100]))
                downloaded_dfs[code] = df_result

        if failed_downloads and not silent:
            max_code_width = max(len(c) for c, _, _ in failed_downloads)
            print(f"  [下载失败] {len(failed_downloads)} 个标的未能从数据源拉取到数据（已回退缓存）:")
            for code, name, err in failed_downloads:
                print(f"      {code.ljust(max_code_width)}  {name}  ({err})")

    # 第二遍：从缓存或下载结果中读取数据
    for code in codes:
        name = names[code]
        # 注意：不能用 `or`，因为 bool(DataFrame) 会抛
        # ValueError: truth value of a DataFrame is ambiguous
        df = cached_dfs.get(code)
        if df is None:
            df = downloaded_dfs.get(code)

        # 按 cutoff_date 过滤（仅对新下载的）
        if code in downloaded_dfs and cutoff_date is not None and df is not None and not df.empty:
            df_before = _filter_by_cutoff(df, cutoff_date)
            if not silent and len(df_before) < len(df):
                print(
                    f"  [过滤] {code} ({name}) 截断至 {cutoff_date.date()}，"
                    f"原 {len(df)} 条 -> {len(df_before)} 条"
                )
            df = df_before

        # cutoff 后该 ETF 可能在回测窗口内完全无数据（上市日晚于截止日）
        if df is None or df.empty:
            is_critical = (
                name == strategy.params.get("safe_haven")
                or name == strategy.params.get("benchmark")
            )
            if is_critical or not strategy.params.get("dynamic_pool", False):
                window_end = cutoff_date.date() if cutoff_date is not None else "最新"
                hint = "" if is_critical else "，或开启 params.dynamic_pool 自动排除未上市标的"
                raise ValueError(
                    f"{code} ({name}) 在回测窗口 "
                    f"{target_start_dt.date()} ~ {window_end} 内无数据；"
                    f"请调整 backtest.end_date / start_date{hint}"
                )
            if not silent:
                print(
                    f"  [剔除] {code} ({name}) 在回测窗口内无数据"
                    f"（上市日晚于截止日），已从候选池排除"
                )
            continue

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
            if not silent:
                print(f"  [缓存旧格式] 仅使用收盘价: {code} ({name})")
            all_close[name] = df[code]
            all_open[name] = df[code]
            all_high[name] = df[code]
            all_low[name] = df[code]
        else:
            raise ValueError(f"未找到 {close_col} 或 {code} 列")

        actual_starts[name] = df.index[0]

    # 异常复权跳空修正（以 close 为准，同比例缩放 open/high/low）
    if data_source.adjusted:
        for name in all_close:
            fixed_close = detect_and_fix_price_jumps(all_close[name], name)
            ratio = fixed_close / all_close[name]
            all_close[name] = fixed_close
            all_open[name] = all_open[name] * ratio
            all_high[name] = all_high[name] * ratio
            all_low[name] = all_low[name] * ratio
    else:
        if not silent:
            print("  [数据] 当前数据源返回未复权价格，跳过复权跳空修正")

    data_close = pd.DataFrame(all_close)
    data_open = pd.DataFrame(all_open)
    data_high = pd.DataFrame(all_high)
    data_low = pd.DataFrame(all_low)

    if data_close.empty:
        raise ValueError(
            "回测窗口内所有 ETF 均无数据，请检查 start_date / end_date 与候选池上市日"
        )

    latest_etf_start = max(actual_starts.values())
    earliest_etf_start = min(actual_starts.values())

    dynamic_pool = strategy.params.get("dynamic_pool", False)

    if not silent:
        max_name_width = max(display_width(name) for name in actual_starts)
        start_lines = "\n".join(
            f"      {ljust(name, max_name_width)} : {d.date()}"
            for name, d in sorted(actual_starts.items(), key=lambda x: x[1])
        )
        print("  [数据] 各 ETF 数据起始日期:")
        print(start_lines)

    if dynamic_pool:
        effective_start = max(target_start_dt, earliest_etf_start)
    else:
        name_types = {p.name: p.type for p in strategy.pool}
        lookback = strategy.params.get("lookback", 20)
        required_starts = {}
        for name in data_close.columns:
            window = required_window(name_types.get(name), lookback)
            etf_series = all_close[name]
            if len(etf_series) >= window:
                required_starts[name] = etf_series.index[window - 1]
            else:
                required_starts[name] = etf_series.index[-1]
        latest_required_start = max(required_starts.values())
        effective_start = max(target_start_dt, latest_etf_start, latest_required_start)

        if latest_required_start > max(target_start_dt, latest_etf_start) and not silent:
            print(
                f"  [注意] 部分 ETF 需要更长的预热期才能产生有效得分，"
                f"策略实际起始日调整为 {latest_required_start.date()}"
            )

    if effective_start != data_close.index[0]:
        if not silent:
            print(f"  [调整] 策略实际起始日: {effective_start.date()}")
        data_close = data_close.loc[data_close.index >= effective_start]
        data_open = data_open.loc[data_open.index >= effective_start]
        data_high = data_high.loc[data_high.index >= effective_start]
        data_low = data_low.loc[data_low.index >= effective_start]

    if min_bars is not None and len(data_close) < min_bars:
        raise ValueError(
            f"数据不足: 截断后仅 {len(data_close)} 条，需要至少 {min_bars} 条"
        )

    if not silent:
        print(
            f"  时间范围: {data_close.index[0].date()} ~ {data_close.index[-1].date()}, "
            f"共 {len(data_close)} 条"
        )

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
    *,
    data: dict[str, pd.DataFrame] | None = None,
    include_today: bool = False,
    cutoff_date: datetime | None = None,
    start_date: str | None = None,
    min_bars: int | None = None,
    silent: bool = False,
    skip_download: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """执行单个策略回测，返回结果 DataFrame 与标的名称列表。

    统一保证 rotation.run 始终收到 name_types，不生成报告、不打印信号、不保存文件。
    """
    if not silent:
        print(f"\n{'='*60}")
        print(f"【{strategy.name}】{strategy.description}")
        print(f"  模式: {strategy.mode} | 参数: {strategy.params}")
        print(f"{'='*60}")

    if data is None:
        data = fetch_pool_data(
            strategy,
            app_config,
            data_source,
            include_today=include_today,
            cutoff_date=cutoff_date,
            start_date=start_date,
            min_bars=min_bars,
            silent=silent,
            skip_download=skip_download,
        )
    name_list = data["close"].columns.tolist()

    if strategy.mode == "rotation":
        name_types = {p.name: p.type for p in strategy.pool}
        result = rotation.run(data, name_list, strategy.params, name_types=name_types)
        benchmark_col = f"{name_list[0]}净值"
        benchmark_name = name_list[0]
        for name in name_list:
            price = data["close"][name].copy()
            if name == benchmark_name:
                first_valid = price.first_valid_index()
                if first_valid is not None and first_valid != price.index[0]:
                    price.loc[:first_valid] = price.loc[first_valid]
            result[f"{name}净值"] = price / price.iloc[0]
    elif strategy.mode == "weighted":
        weights = {p.name: p.weight for p in strategy.pool}
        result = weighted.run(data, name_list, weights, strategy.params)
        for name in name_list:
            result[f"{name}净值"] = result[name] / result[name].iloc[0]
        benchmark_col = None
    else:
        raise ValueError(f"不支持的模式: {strategy.mode}")

    return result, name_list


def _calc_position(
    capital: float, weight_frac: float, price: float, lot_size: int = 100
) -> tuple[float, int]:
    """按资金与权重计算买入金额和股数（向下取整到 lot_size 的整数倍）。"""
    amount = capital * weight_frac
    if price <= 0 or not np.isfinite(price):
        return amount, 0
    shares = int((amount / price) // lot_size) * lot_size
    return amount, max(0, shares)


def print_latest_signal(
    strategy: StrategyConfig,
    result: pd.DataFrame,
    name_list: list[str],
    last_quote_dates: dict[str, str] | None = None,
    capital: float | None = None,
    lot_size: int = 100,
):
    """打印最新交易日的信号与操作建议（rotation / weighted）。"""
    latest = result.iloc[-1]

    print(f"\n{'='*60}")
    print(f"【今日交易信号】{strategy.name}")
    print(f"{'='*60}")
    print(f"信号日期: {result.index[-1].date()}")
    print(f"策略参数: lookback={strategy.params.get('lookback', 20)}日, top_n={strategy.params.get('top_n', 1)}")
    print(f"{'-'*60}")

    if strategy.mode == "rotation":
        scoring = strategy.params.get("scoring", "momentum")
        prefix = "得分_" if scoring == "slope_r2" else "涨幅_"
        lookback = strategy.params.get("lookback", 20)
        top_n = strategy.params.get("top_n", 1)

        # 计算每只 ETF 的 lookback 日年化波动率
        vol_map = {}
        for name in name_list:
            if name in result.columns:
                prices = result[name].dropna()
                returns = prices.pct_change(fill_method=None).dropna()
                if len(returns) >= 2:
                    vol = returns.tail(lookback).std() * np.sqrt(252)
                    vol_map[name] = vol
                else:
                    vol_map[name] = np.nan
            else:
                vol_map[name] = np.nan

        header = (
            f"{ljust('排名', 4)} "
            f"{ljust('ETF名称', 20)} "
            f"{ljust('代码', 10)} "
            f"{ljust('最新行情日', 12)} "
            f"{ljust('周期动量得分', 12)} "
            f"{ljust('波动率', 10)} "
            f"{ljust('目前持仓仓位', 12)} "
            f"{ljust('建议下日仓位', 12)} "
            f"{ljust('未入选原因', 12)}"
        )
        print(header)
        print("-" * 112)

        scores = []
        for name in name_list:
            score_col = f"{prefix}{name}"
            score = latest[score_col] if score_col in latest else np.nan
            # 目前持仓仓位：shift 后的实际持仓
            current_weight = latest[weight_col(name)] if weight_col(name) in latest else 0
            # 建议下一交易日仓位：按最新日信号重新计算后的原始权重
            signal_weight_col = f"信号权重_{name}"
            signal_weight = (
                latest[signal_weight_col]
                if signal_weight_col in latest
                else current_weight
            )
            code = next((p.code for p in strategy.pool if p.name == name), "")
            last_date = last_quote_dates.get(name, "-") if last_quote_dates else "-"
            vol = vol_map.get(name, np.nan)
            # 未入选原因对应“建议下一交易日仓位”的风控原因
            signal_reason_col = f"信号风控原因_{name}"
            per_etf_reason = str(
                latest.get(signal_reason_col)
                if signal_reason_col in latest
                else latest.get(f"风控原因_{name}", "")
            )
            if signal_weight > 0:
                reason = ""
            elif per_etf_reason:
                reason = per_etf_reason
            else:
                reason = f"未进top{top_n}"
            scores.append((name, code, last_date, score, vol, current_weight, signal_weight, reason))

        scores.sort(key=lambda x: x[3] if not pd.isna(x[3]) else -np.inf, reverse=True)

        for i, (name, code, last_date, score, vol, current_weight, signal_weight, reason) in enumerate(scores, 1):
            marker = "★" if signal_weight > 0 else " "
            current_weight_pct = f"{current_weight*100:.0f}%" if current_weight > 0 else "0%"
            signal_weight_pct = f"{signal_weight*100:.0f}%" if signal_weight > 0 else "0%"
            if pd.isna(score):
                score_str = "-"
            elif scoring == "momentum":
                score_str = f"{score:+.2%}"
            else:
                score_str = f"{score:.4f}"
            vol_str = f"{vol:.1%}" if not pd.isna(vol) else "-"
            rank_str = f"{marker}{i}"
            row = (
                f"{ljust(rank_str, 4)} "
                f"{ljust(name, 20)} "
                f"{ljust(code, 10)} "
                f"{ljust(last_date, 12)} "
                f"{ljust(score_str, 12)} "
                f"{ljust(vol_str, 10)} "
                f"{ljust(current_weight_pct, 12)} "
                f"{ljust(signal_weight_pct, 12)} "
                f"{ljust(reason, 12)}"
            )
            print(row)

        print(f"{'='*60}")
        print("操作建议:")
        signal_weight_col = f"信号权重_{name_list[0]}"
        has_signal_weight = signal_weight_col in result.columns
        weight_prefix = "信号权重_" if has_signal_weight else "权重_"
        holdings = [(name, latest[f"{weight_prefix}{name}"]) for name in name_list if latest[f"{weight_prefix}{name}"] > 0]
        for name, weight in holdings:
            code = next((p.code for p in strategy.pool if p.name == name), "")
            line = f"  持有 {name} ({code}): {weight*100:.0f}%"
            if capital:
                price = float(result[name].iloc[-1])
                amount, shares = _calc_position(capital, float(weight), price, lot_size)
                line += (
                    f" | 买入金额 ≈ {amount:,.0f} 元"
                    f" | 最新价 {price:.3f}"
                    f" | 买入 {shares} 股"
                )
            print(line)

    elif strategy.mode == "weighted":
        header = (
            f"{ljust('ETF名称', 20)} "
            f"{ljust('代码', 10)} "
            f"{ljust('目标权重', 10)} "
            f"{ljust('当前权重', 10)}"
        )
        print(header)
        print("-" * 54)

        weights = {p.name: p.weight for p in strategy.pool}
        for name in name_list:
            code = next((p.code for p in strategy.pool if p.name == name), "")
            target = weights.get(name, 0)
            row = (
                f"{ljust(name, 20)} "
                f"{ljust(code, 10)} "
                f"{ljust(f'{target:.0f}%', 10)} "
                f"{ljust('-', 10)}"
            )
            print(row)

        print(f"{'='*60}")
        print("操作建议: 按上述目标权重配置仓位")
        if capital:
            for name in name_list:
                weight_frac = weights.get(name, 0) / 100.0
                if weight_frac <= 0:
                    continue
                code = next((p.code for p in strategy.pool if p.name == name), "")
                price = float(result[name].iloc[-1])
                amount, shares = _calc_position(capital, weight_frac, price, lot_size)
                print(
                    f"  买入 {name} ({code}): {weight_frac*100:.0f}% | "
                    f"金额 ≈ {amount:,.0f} 元 | 最新价 {price:.3f} | 买入 {shares} 股"
                )

    print(f"{'='*60}")


def build_holding_df(result: pd.DataFrame) -> pd.DataFrame | None:
    """从 rotation 策略结果中提取每日持仓记录 DataFrame。"""
    if "持仓" not in result.columns:
        return None

    cols = ["持仓", "当天动量第一", "风控原因"]
    if "轮动策略净值" in result.columns:
        cols.append("轮动策略净值")
    if "轮动策略日收益率" in result.columns:
        cols.append("轮动策略日收益率")

    holding_df = result[cols].copy()
    holding_df.index.name = "日期"

    if "轮动策略日收益率" in holding_df.columns:
        holding_df["轮动策略日收益率"] = holding_df["轮动策略日收益率"].apply(
            lambda x: f"{x:+.2%}"
        )

    return holding_df


def print_holding_summary(holding_df: pd.DataFrame, strategy_name: str, tail_n: int = 10) -> None:
    """在终端打印最近若干条每日持仓记录摘要。"""
    print(f"\n{'='*80}")
    print(f"【每日持仓记录】{strategy_name}（最近 {tail_n} 条）")
    print(f"{'='*80}")
    print(holding_df.tail(tail_n).to_string())
    print(f"{'='*80}")


def print_position_contribution(strategy: StrategyConfig, result: pd.DataFrame, name_list: list[str]):
    """打印 rotation 策略各 ETF 的持有天数与收益贡献占比。"""
    final_nav = result["轮动策略净值"].iloc[-1]
    total_return = final_nav - 1.0

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

    headers = [
        ljust("ETF名称", col_widths["ETF名称"]),
        ljust("代码", col_widths["代码"]),
        ljust("持有天数", col_widths["持有天数"]),
        rjust("累计贡献", col_widths["累计贡献"]),
        rjust("贡献占比", col_widths["贡献占比"]),
    ]
    print(" ".join(headers))
    print("-" * total_width)

    rows = []
    total_contribution = 0.0
    for name in name_list:
        code = next((p.code for p in strategy.pool if p.name == name), "")
        hold_days = int((result[weight_col(name)] > 0).sum())
        contrib_col = f"贡献_日收益_{name}"
        if contrib_col in result.columns:
            contribution = result[contrib_col].sum()
        else:
            contribution = 0.0
        total_contribution += contribution
        rows.append((name, code, hold_days, contribution))

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
            ljust(name, col_widths["ETF名称"]),
            ljust(code, col_widths["代码"]),
            ljust(str(hold_days), col_widths["持有天数"]),
            rjust(contrib_str, col_widths["累计贡献"]),
            rjust(ratio_str, col_widths["贡献占比"]),
        ]
        print(" ".join(row))

    print("-" * total_width)
    print(f"合计贡献（简单加总口径）: {total_contribution:+.2%}")
    print(f"策略累计收益（复利口径）:   {total_return:+.2%}")
    print("  注：因复利再投资效应，简单加总的贡献合计会低于复利累计收益")
    print(f"{'='*total_width}")
