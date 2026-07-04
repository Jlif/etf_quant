"""轮动策略：每天选最强 top_n 个 ETF，等权分配"""

from __future__ import annotations

import pandas as pd
import numpy as np

from core.scorer import (
    adaptive_momentum_score,
    momentum_quality_score,
    momentum_score,
    slope_r2_score,
)
from .risk import (
    layer1_market_filter,
    layer2_atr_trailing_stop,
    layer3_vol_target_filter,
)


def _adaptive_window(etf_type: str | None, default_lookback: int) -> int:
    """Return the rolling window needed for an ETF type's adaptive scorer."""
    if etf_type == "行业":
        return 62
    if etf_type in {"红利", "自由现金流", "价值"}:
        return 41
    if etf_type == "成长":
        return 21
    if etf_type == "商品":
        return 61
    if etf_type == "宽基":
        return 252
    return default_lookback + 1


def run(
    data: dict[str, pd.DataFrame],
    name_list: list[str],
    params: dict,
    name_types: dict[str, str | None] | None = None,
) -> pd.DataFrame:
    """
    轮动策略回测

    T 日收盘后计算信号，确定 T+1 日持仓。
    - 新调入的标的：T+1 日开盘价买入，收益 = close_T+1 / open_T+1 - 1
    - 继续持有的标的：T 日收盘已持有，收益 = close_T+1 / close_T - 1
    - 调出的标的：T+1 日不再产生收益

    Parameters
    ----------
    data : dict[str, pd.DataFrame]
        包含 close/open/high/low 四个字典，其中 close 用于计算信号，
        open 用于计算新调入标的的开盘价成交收益。
    name_list : list[str]
        标的名称列表
    params : dict
        - lookback: 回望周期
        - scoring: "momentum" | "slope_r2"
        - top_n: 每天选前 N 个

    Returns
    -------
    pd.DataFrame
        包含回测结果的数据
    """
    lookback = params.get("lookback", 20)
    scoring = params.get("scoring", "momentum")
    top_n = params.get("top_n", 1)

    close_df = data["close"].copy()
    open_df = data["open"].copy()

    df = close_df.copy()

    # 1. 计算两种日收益率
    dynamic_pool = params.get("dynamic_pool", False)
    for name in name_list:
        # 新调入标的：开盘价买入，收盘价结算
        df[f"日收益率_再平衡_{name}"] = close_df[name] / open_df[name] - 1.0
        # 继续持有标的：前日收盘价已持有，当日收盘价结算
        df[f"日收益率_持有_{name}"] = close_df[name] / close_df[name].shift(1) - 1.0
        if dynamic_pool:
            # 未上市/无数据期间，该 ETF 不会获得权重，收益列填 0
            # 避免最终 dropna 把策略前期的有效交易日误删
            missing_mask = close_df[name].isna()
            df.loc[missing_mask, f"日收益率_再平衡_{name}"] = 0.0
            df.loc[missing_mask, f"日收益率_持有_{name}"] = 0.0

    # 2. 计算得分（基于收盘价）
    if params.get("adaptive_scoring"):
        benchmark_name = params.get("benchmark")
        type_map = name_types or {}
        benchmark_series = close_df[benchmark_name] if benchmark_name else None
        for name in name_list:
            etf_type = type_map.get(name)
            window = _adaptive_window(etf_type, lookback)
            df[f"自适应得分_{name}"] = df[name].rolling(window).apply(
                lambda x: adaptive_momentum_score(
                    x,
                    etf_type=etf_type,
                    benchmark_series=benchmark_series,
                    lookback=lookback,
                )
            )
        signal_cols = [f"自适应得分_{v}" for v in name_list]
        prefix = "自适应得分_"
    elif scoring == "slope_r2":
        for name in name_list:
            df[f"得分_{name}"] = df[name].rolling(lookback).apply(
                lambda x: slope_r2_score(x, lookback)
            )
        signal_cols = [f"得分_{v}" for v in name_list]
        prefix = "得分_"
    elif scoring == "momentum_quality":
        for name in name_list:
            df[f"质量_{name}"] = df[name].rolling(lookback).apply(
                lambda x: momentum_quality_score(x, lookback)
            )
        signal_cols = [f"质量_{v}" for v in name_list]
        prefix = "质量_"
    else:
        for name in name_list:
            df[f"涨幅_{name}"] = df[name] / df[name].shift(lookback + 1) - 1.0
        signal_cols = [f"涨幅_{v}" for v in name_list]
        prefix = "涨幅_"

    # dynamic_pool 模式下，先根据原始数据计算每只 ETF 的最早可参与交易日，
    # 然后再 dropna；避免先 dropna 后用截断后的 series.index[window-1] 导致 eligible_start 被推迟。
    if dynamic_pool:
        type_map = name_types or {}
        eligible_start_map = {}
        for name in name_list:
            etf_type = type_map.get(name)
            if params.get("adaptive_scoring"):
                window = _adaptive_window(etf_type, lookback)
            elif scoring == "slope_r2" or scoring == "momentum_quality":
                window = lookback
            else:
                window = lookback + 1
            series = df[name]
            if len(series) >= window:
                eligible_start_map[name] = series.index[window - 1]
            else:
                eligible_start_map[name] = None

    if params.get("adaptive_scoring") or dynamic_pool:
        df = df.dropna(subset=signal_cols, how="all")
    else:
        df = df.dropna()

    if df.empty:
        raise ValueError(
            "计算得分后没有剩余有效数据。可能原因：\n"
            "1. 数据长度不足（例如 宽基 需要 252 个交易日）；\n"
            "2. 所有 ETF 在计算窗口内价格无变化（波动率为 0）；\n"
            "3. 配置了无法识别的 type，导致回退为默认 momentum 但数据仍不足。\n"
            f"当前时间范围: {data['close'].index[0].date()} ~ {data['close'].index[-1].date()}, "
            f"共 {len(data['close'])} 条"
        )

    # 计算每日候选池可用性（dynamic_pool 模式下，未上市/未预热的 ETF 不纳入排名）
    if dynamic_pool:
        eligible_df = pd.DataFrame(False, index=df.index, columns=name_list)
        for name in name_list:
            start = eligible_start_map[name]
            if start is not None:
                eligible_df[name] = df.index >= start
    else:
        eligible_df = pd.DataFrame(True, index=df.index, columns=name_list)

    # 3. 生成每日权重：top_n 等权，其余为 0
    eligible_signal_df = df[signal_cols].where(eligible_df.values)
    rank_df = eligible_signal_df.rank(axis=1, ascending=False, method="first", na_option="keep")
    for name in name_list:
        col = f"{prefix}{name}"
        df[f"权重_{name}"] = (rank_df[col] <= top_n).astype(float) / top_n

    # dynamic_pool：可选 ETF 不足 top_n 时，剩余仓位优先填充已就绪的 safe_haven
    if dynamic_pool:
        weight_cols = [f"权重_{n}" for n in name_list]
        for date in df.index:
            eligible_names = eligible_df.columns[eligible_df.loc[date]].tolist()
            selected_count = int((df.loc[date, weight_cols] > 0).sum())
            if selected_count < top_n:
                fill_weight = (top_n - selected_count) / top_n
                safe_haven = params.get("safe_haven")
                if safe_haven and safe_haven in eligible_names:
                    df.loc[date, f"权重_{safe_haven}"] += fill_weight
            # 归一化，确保总权重为 1.0
            total_weight = df.loc[date, weight_cols].sum()
            if total_weight > 0:
                df.loc[date, weight_cols] /= total_weight

    # 初始化风控原因列，用于记录每个过滤器对持仓的调整
    df["风控原因"] = ""
    # 每只 ETF 单独记录被哪一层风控清零/压缩，供最新信号打印使用
    for name in name_list:
        df[f"风控原因_{name}"] = ""

    # 3.05–3.7 三层风控系统
    risk_control = params.get("risk_control", {})

    # 3.05 第一层：组合趋势/回撤过滤
    layer1 = risk_control.get("layer1", {})
    if layer1.get("enabled", False):
        safe_haven = params.get("safe_haven")
        if not safe_haven:
            raise ValueError("开启 risk_control.layer1 时必须配置 safe_haven")
        ma_lookback = layer1.get("ma_lookback", 20)
        drawdown_threshold = layer1.get("drawdown_threshold", 0.05)
        drawdown_lookback = layer1.get("drawdown_lookback", 252)

        weight_cols = [f"权重_{n}" for n in name_list]
        weights_df = df[weight_cols].copy()
        pre_weights = weights_df.copy()
        adjusted_weights = layer1_market_filter(
            weights_df=weights_df,
            close_df=close_df,
            ma_lookback=ma_lookback,
            drawdown_threshold=drawdown_threshold,
            safe_haven=safe_haven,
            drawdown_lookback=drawdown_lookback,
        )
        for name in name_list:
            df[f"权重_{name}"] = adjusted_weights[f"权重_{name}"]

        # 记录触发第一层的风控日（组合级字符串）
        risk_cols = [c for c in weight_cols if c != f"权重_{safe_haven}"]
        triggered = (pre_weights[risk_cols].sum(axis=1) > 0) & (
            df[f"权重_{safe_haven}"] >= 0.999
        )
        if triggered.any():
            df.loc[triggered, "风控原因"] += (
                f"第一层: 组合趋势/回撤过滤(跌破{ma_lookback}日均线或"
                f"{drawdown_lookback}日高点回撤>{drawdown_threshold:.1%}); "
            )
            # 记录每只被清零的风险资产
            for name in name_list:
                if name == safe_haven:
                    continue
                etf_triggered = triggered & (pre_weights[f"权重_{name}"] > 0) & (
                    df[f"权重_{name}"] == 0
                )
                if etf_triggered.any():
                    df.loc[etf_triggered, f"风控原因_{name}"] = "L1"

    # 3.6 第二层：ATR 跟踪止损拦截
    layer2 = risk_control.get("layer2", {})
    if layer2.get("enabled", False):
        safe_haven = params.get("safe_haven")
        atr_multiplier = layer2.get("atr_multiplier", 3.0)
        atr_lookback = layer2.get("atr_lookback", 14)

        weight_cols = [f"权重_{n}" for n in name_list]
        weights_df = df[weight_cols].copy()
        pre_weights = weights_df.copy()
        adjusted_weights = layer2_atr_trailing_stop(
            weights_df=weights_df,
            close_df=close_df,
            high_df=data["high"],
            low_df=data["low"],
            atr_multiplier=atr_multiplier,
            atr_lookback=atr_lookback,
            safe_haven=safe_haven,
        )
        for name in name_list:
            df[f"权重_{name}"] = adjusted_weights[f"权重_{name}"]

        risk_names = [n for n in name_list if n != safe_haven]
        for name in risk_names:
            triggered = (pre_weights[f"权重_{name}"] > 0) & (df[f"权重_{name}"] == 0)
            if triggered.any():
                df.loc[triggered, "风控原因"] += (
                    f"{name}: ATR跟踪止损(回落>{atr_multiplier}*ATR); "
                )
                df.loc[triggered, f"风控原因_{name}"] = "L2"

    # 3.7 第三层：目标波动率平准（非线性）
    layer3 = risk_control.get("layer3", {})
    if layer3.get("enabled", False):
        safe_haven = params.get("safe_haven")
        target_vol = layer3["target_vol"]
        vol_lookback = layer3.get("vol_lookback", 20)
        comfort_zone = layer3.get("comfort_zone", 0.15)
        caution_zone = layer3.get("caution_zone", 0.25)
        caution_scale = layer3.get("caution_scale", 0.5)
        transition_power = layer3.get("transition_power")

        weight_cols = [f"权重_{n}" for n in name_list]
        weights_df = df[weight_cols].copy()
        pre_weights = weights_df.copy()
        adjusted_weights = layer3_vol_target_filter(
            weights_df=weights_df,
            close_df=close_df,
            target_vol=target_vol,
            vol_lookback=vol_lookback,
            comfort_zone=comfort_zone,
            caution_zone=caution_zone,
            caution_scale=caution_scale,
            safe_haven=safe_haven,
            transition_power=transition_power,
        )
        for name in name_list:
            df[f"权重_{name}"] = adjusted_weights[f"权重_{name}"]

        # 区分“警惕区（仓位压缩）”和“熔断区（强制清仓）”，写出明确的风控原因
        risk_cols = [f"权重_{n}" for n in name_list if n != safe_haven]
        pre_risk_weight = pre_weights[risk_cols].sum(axis=1)
        post_risk_weight = df[risk_cols].sum(axis=1)

        panic = (pre_risk_weight > 0) & (post_risk_weight == 0)
        if panic.any():
            df.loc[panic, "风控原因"] += (
                f"Layer3: 波动率熔断(组合波动≥{caution_zone:.1%})，风险资产强制清仓; "
            )
            for name in name_list:
                if name == safe_haven:
                    continue
                etf_panic = panic & (pre_weights[f"权重_{name}"] > 0) & (
                    df[f"权重_{name}"] == 0
                )
                if etf_panic.any():
                    df.loc[etf_panic, f"风控原因_{name}"] = "L3"

        caution = (pre_risk_weight > post_risk_weight) & (post_risk_weight > 0)
        if caution.any():
            if transition_power is not None:
                df.loc[caution, "风控原因"] += (
                    f"Layer3: 波动率警惕(组合波动{comfort_zone:.1%}~{caution_zone:.1%})，"
                    f"仓位平滑压缩(power={transition_power}); "
                )
            else:
                df.loc[caution, "风控原因"] += (
                    f"Layer3: 波动率警惕(组合波动{comfort_zone:.1%}~{caution_zone:.1%})，"
                    f"仓位压缩为{caution_scale:.0%}; "
                )

    # 4. 持仓权重前移1天（T日收盘后信号决定T+1日持仓）
    for name in name_list:
        df[f"权重_{name}"] = df[f"权重_{name}"].shift(1).fillna(0.0)

    # 风控原因也随持仓前移1天，使其与风控实际生效当日的持仓对齐，
    # 避免“当天显示触发止损但当天持仓仍是旧仓位”的误解。
    df["风控原因"] = df["风控原因"].shift(1).fillna("")
    for name in name_list:
        df[f"风控原因_{name}"] = df[f"风控原因_{name}"].shift(1).fillna("")

    # 权重为 0 时，该 ETF 的收益贡献应为 0；把收益列中的 NaN 填 0
    # 避免 0 * NaN = NaN 污染策略日收益率。
    for name in name_list:
        df[f"日收益率_再平衡_{name}"] = df[f"日收益率_再平衡_{name}"].fillna(0.0)
        df[f"日收益率_持有_{name}"] = df[f"日收益率_持有_{name}"].fillna(0.0)

    # dynamic_pool 模式下，原始价格列可能包含未上市 ETF 的 NaN，
    # 但 plot_nav_curves 等后续流程需要这些列。dropna 时只检查权重/收益列，
    # 不检查原始价格列，避免误删策略前期的有效交易日。
    check_cols = (
        [f"权重_{n}" for n in name_list]
        + [f"日收益率_再平衡_{n}" for n in name_list]
        + [f"日收益率_持有_{n}" for n in name_list]
    )
    df = df.dropna(subset=check_cols)

    # 5. 计算策略日收益率
    #    区分新调入（按开盘价成交）和继续持有（按前日收盘价持有）
    df["轮动策略日收益率"] = 0.0
    for name in name_list:
        weight_col = f"权重_{name}"
        prev_weight = df[weight_col].shift(1).fillna(0)

        # 新调入：今日权重 > 0 且 昨日权重 == 0
        is_entry = (df[weight_col] > 0) & (prev_weight == 0)
        # 继续持有：今日权重 > 0 且 昨日权重 > 0
        is_hold = (df[weight_col] > 0) & (prev_weight > 0)

        df.loc[is_entry, "轮动策略日收益率"] += (
            df.loc[is_entry, f"日收益率_再平衡_{name}"] * df.loc[is_entry, weight_col]
        )
        df.loc[is_hold, "轮动策略日收益率"] += (
            df.loc[is_hold, f"日收益率_持有_{name}"] * df.loc[is_hold, weight_col]
        )

        # 记录每只 ETF 每日的加权收益贡献，用于后续归因统计
        df[f"贡献_日收益_{name}"] = 0.0
        df.loc[is_entry, f"贡献_日收益_{name}"] = (
            df.loc[is_entry, f"日收益率_再平衡_{name}"] * df.loc[is_entry, weight_col]
        )
        df.loc[is_hold, f"贡献_日收益_{name}"] = (
            df.loc[is_hold, f"日收益率_持有_{name}"] * df.loc[is_hold, weight_col]
        )

    if df.empty:
        raise ValueError(
            "持仓权重前移后没有剩余有效数据。可能原因：\n"
            "1. 所有 ETF 在首个交易日即无有效得分；\n"
            "2. 风控过滤器（如绝对动量过滤）清空了所有持仓。"
        )

    df.loc[df.index[0], "轮动策略日收益率"] = 0.0

    # 5.1 交易成本（滑点 + 手续费）
    # transaction_cost 为单边成本，例如 0.0015 表示千分之 1.5。
    # 换手率 = sum(|今日权重 - 昨日权重|) / 2，双边成本 = 换手率 * 2 * 单边成本。
    transaction_cost = params.get("transaction_cost", 0.0)
    if transaction_cost > 0:
        weight_cols = [f"权重_{n}" for n in name_list]
        turnover = np.zeros(len(df))
        for i in range(1, len(df)):
            turnover[i] = np.sum(np.abs(df.iloc[i][weight_cols].values - df.iloc[i - 1][weight_cols].values)) / 2.0
        df["换手率"] = turnover
        df["交易成本"] = df["换手率"] * transaction_cost * 2.0
        df["轮动策略日收益率"] -= df["交易成本"]

    df["轮动策略净值"] = (1.0 + df["轮动策略日收益率"]).cumprod()

    # 记录每日主持仓信号（权重最大的那个）
    weight_cols = [f"权重_{n}" for n in name_list]
    df["信号"] = df[weight_cols].idxmax(axis=1).str.replace("权重_", "")

    # 记录每日动量排名第一的 ETF（基于信号日得分）
    df["当天动量第一"] = df[signal_cols].idxmax(axis=1).str.replace(prefix, "")

    # 记录每日完整持仓组合，并标记换仓日
    def _format_holding(row: pd.Series) -> str:
        holdings = [(n, row[f"权重_{n}"]) for n in name_list if row[f"权重_{n}"] > 0]
        holdings.sort(key=lambda x: x[1], reverse=True)
        if not holdings:
            return "空仓"
        return "+".join(f"{n}({w * 100:.0f}%)" for n, w in holdings)

    df["持仓"] = df.apply(_format_holding, axis=1)
    df["换仓"] = df["持仓"] != df["持仓"].shift(1)
    df.loc[df.index[0], "换仓"] = False

    return df
