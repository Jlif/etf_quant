"""风险控制过滤器集合

为轮动策略提供可选的风控层，所有过滤器接收权重 DataFrame 和价格 DataFrame，
返回调整后的权重 DataFrame。不配置时不改变原始行为。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def absolute_momentum_filter(
    weights_df: pd.DataFrame,
    close_df: pd.DataFrame,
    lookback: int,
    threshold: float,
    safe_haven: str,
) -> pd.DataFrame:
    """
    绝对动量过滤（双重动量防守层）。

    对权重大于 0 的标的，检查其过去 lookback 天的绝对收益率。
    若绝对收益率 <= threshold，则将该标的在对应日期的权重切换到 safe_haven。

    Parameters
    ----------
    weights_df : pd.DataFrame
        列名为 "权重_{标的名称}" 的每日权重表。
    close_df : pd.DataFrame
        列名为标的名称的收盘价表，索引与 weights_df 对齐。
    lookback : int
        绝对动量回望周期。收益率计算方式为 close_t / close_{t-lookback-1} - 1，
        与 rotation 策略中的动量口径保持一致。
    threshold : float
        触发切换的阈值，绝对收益率 <= threshold 时切换。
    safe_haven : str
        避风港标的名称，必须存在于 close_df 列中。

    Returns
    -------
    pd.DataFrame
        调整后的权重表（副本）。
    """
    if safe_haven not in close_df.columns:
        raise ValueError(f"safe_haven '{safe_haven}' 不在 close_df 列中")

    safe_weight_col = f"权重_{safe_haven}"
    if safe_weight_col not in weights_df.columns:
        raise ValueError(f"safe_haven '{safe_haven}' 不在 weights_df 列中")

    adjusted = weights_df.copy()

    # 计算每只标的的绝对动量
    abs_return = close_df / close_df.shift(lookback + 1) - 1.0

    # 计算每只标的的绝对动量
    abs_return = close_df / close_df.shift(lookback + 1) - 1.0

    for name in close_df.columns:
        if name == safe_haven:
            continue

        weight_col = f"权重_{name}"
        if weight_col not in adjusted.columns:
            continue

        # 该标的被选中且绝对动量不达标的日期
        failing = (adjusted[weight_col] > 0) & (abs_return[name] <= threshold)
        if not failing.any():
            continue

        # 将失败仓位累加到避风港，并清空原标的权重
        adjusted.loc[failing, safe_weight_col] += adjusted.loc[failing, weight_col]
        adjusted.loc[failing, weight_col] = 0.0

    return adjusted


def volatility_target_filter(
    weights_df: pd.DataFrame,
    close_df: pd.DataFrame,
    target_vol: float,
    vol_lookback: int,
    safe_haven: str | None = None,
) -> pd.DataFrame:
    """
    目标波动率控制。

    计算组合过去 vol_lookback 天的实际年化波动率，若超过 target_vol，
    则按比例降低风险资产仓位，释放出的部分切换为 safe_haven（或空仓）。

    Parameters
    ----------
    weights_df : pd.DataFrame
        列名为 "权重_{标的名称}" 的每日权重表。
    close_df : pd.DataFrame
        列名为标的名称的收盘价表，索引与 weights_df 对齐。
    target_vol : float
        目标年化波动率，例如 0.10 表示 10%。
    vol_lookback : int
        计算实际波动率的滚动窗口（交易日）。
    safe_haven : str | None
        避风港标的名称，必须存在于 close_df 列中。若为 None，
        则减仓后剩余部分空仓（权重和 < 1）。

    Returns
    -------
    pd.DataFrame
        调整后的权重表（副本）。
    """
    if target_vol <= 0:
        raise ValueError("target_vol 必须大于 0")

    adjusted = weights_df.copy()
    weight_cols = [c for c in adjusted.columns if c.startswith("权重_")]
    name_list = [c.replace("权重_", "") for c in weight_cols]

    # 组合日收益率
    returns = close_df.pct_change(fill_method=None)
    portfolio_returns = pd.Series(0.0, index=adjusted.index)
    for name in name_list:
        portfolio_returns += adjusted[f"权重_{name}"] * returns[name]

    # 滚动年化波动率（min_periods=1 避免数据不足时全部返回 NaN，导致 fillna 成 1.0 失效）
    realized_vol = portfolio_returns.rolling(vol_lookback, min_periods=1).std() * np.sqrt(252)

    # 缩放因子：实际波动率越高，仓位越低
    scale = (target_vol / realized_vol).fillna(1.0)
    scale = scale.clip(upper=1.0)

    # 风险资产 = 除 safe_haven 外的标的
    if safe_haven and safe_haven in name_list:
        risk_names = [n for n in name_list if n != safe_haven]
    else:
        risk_names = name_list

    # 缩放风险资产权重
    for name in risk_names:
        adjusted[f"权重_{name}"] *= scale

    # 如果有 safe_haven，用其承接释放的仓位，保持总权重为 1
    if safe_haven and safe_haven in name_list:
        safe_col = f"权重_{safe_haven}"
        adjusted[safe_col] = 1.0 - adjusted[[f"权重_{n}" for n in risk_names]].sum(axis=1)

    return adjusted


def trailing_stop_filter(
    weights_df: pd.DataFrame,
    close_df: pd.DataFrame,
    stop_pct: float,
    safe_haven: str | None = None,
) -> pd.DataFrame:
    """
    移动止损过滤器。

    跟踪每只标的自进入持仓以来的最高价（high water mark）。
    若当前收盘价从最高价回撤超过 stop_pct，强制清仓并转入 safe_haven。

    Parameters
    ----------
    weights_df : pd.DataFrame
        列名为 "权重_{标的名称}" 的每日权重表（信号日权重，尚未 shift）。
    close_df : pd.DataFrame
        列名为标的名称的收盘价表，索引与 weights_df 对齐。
    stop_pct : float
        触发止损的回撤比例，例如 0.08 表示 8%。
    safe_haven : str | None
        止损后资金去向。若为 None，则该部分空仓。

    Returns
    -------
    pd.DataFrame
        调整后的权重表（副本）。
    """
    if not 0 < stop_pct < 1:
        raise ValueError("stop_pct 应在 (0, 1) 之间")

    adjusted = weights_df.copy()
    weight_cols = [c for c in adjusted.columns if c.startswith("权重_")]
    name_list = [c.replace("权重_", "") for c in weight_cols]

    # 不对 safe_haven 本身做止损
    risk_names = [n for n in name_list if n != safe_haven]

    # 记录每只标的的 high water mark
    hwm = pd.DataFrame(np.nan, index=adjusted.index, columns=risk_names)

    for i, date in enumerate(adjusted.index):
        for name in risk_names:
            weight_col = f"权重_{name}"
            current_weight = adjusted.loc[date, weight_col]
            current_price = close_df.loc[date, name]

            prev_weight = adjusted.iloc[i - 1][weight_col] if i > 0 else 0.0

            # 新进入持仓：重置 HWM
            if current_weight > 0 and prev_weight == 0:
                hwm.loc[date, name] = current_price
            # 继续持仓：更新 HWM
            elif current_weight > 0 and prev_weight > 0:
                prev_hwm = hwm.iloc[i - 1][name]
                hwm.loc[date, name] = max(prev_hwm, current_price)
            # 未持仓：HWM 清空
            else:
                hwm.loc[date, name] = np.nan

            # 检查是否触发移动止损
            if current_weight > 0 and not pd.isna(hwm.loc[date, name]):
                if current_price < hwm.loc[date, name] * (1 - stop_pct):
                    if safe_haven and safe_haven in name_list:
                        safe_col = f"权重_{safe_haven}"
                        adjusted.loc[date, safe_col] += current_weight
                    adjusted.loc[date, weight_col] = 0.0
                    hwm.loc[date, name] = np.nan

    return adjusted


def layer1_market_filter(
    weights_df: pd.DataFrame,
    close_df: pd.DataFrame,
    benchmark_series: pd.Series | None,
    ma_lookback: int,
    drawdown_threshold: float,
    safe_haven: str,
) -> pd.DataFrame:
    """
    第一层：时序动量过滤。

    当市场代理价格跌破 N 日均线，或从前期高点回撤超过阈值时，
    强制清空所有风险资产仓位，全部转入 safe_haven。

    Parameters
    ----------
    weights_df : pd.DataFrame
        列名为 "权重_{标的名称}" 的每日权重表。
    close_df : pd.DataFrame
        收盘价表，用于构建 pool 等权市场代理（benchmark 未配置时）。
    benchmark_series : pd.Series | None
        外部基准序列（如 params.benchmark 对应的收盘价）。若为空则使用 pool 等权价格。
    ma_lookback : int
        均线回望周期。
    drawdown_threshold : float
        回撤阈值，例如 0.05 表示回撤 5% 即触发。
    safe_haven : str
        避风港标的名称。

    Returns
    -------
    pd.DataFrame
        调整后的权重表（副本）。
    """
    adjusted = weights_df.copy()
    weight_cols = [c for c in adjusted.columns if c.startswith("权重_")]

    if benchmark_series is not None:
        market = benchmark_series.reindex(adjusted.index)
    else:
        market = close_df.mean(axis=1)

    ma = market.rolling(ma_lookback).mean()
    peak = market.expanding().max()
    drawdown = (market - peak) / peak

    triggered = (market < ma) | (drawdown < -drawdown_threshold)
    if not triggered.any():
        return adjusted

    risk_cols = [c for c in weight_cols if c != f"权重_{safe_haven}"]
    safe_col = f"权重_{safe_haven}"

    # 触发日：风险资产全部转给 safe_haven
    for col in risk_cols:
        adjusted.loc[triggered, safe_col] += adjusted.loc[triggered, col]
        adjusted.loc[triggered, col] = 0.0

    # 归一化，避免浮点误差导致权重和不为 1
    total = adjusted[weight_cols].sum(axis=1)
    adjusted = adjusted.div(total, axis=0).fillna(0.0)
    return adjusted


def layer2_atr_trailing_stop(
    weights_df: pd.DataFrame,
    close_df: pd.DataFrame,
    high_df: pd.DataFrame,
    low_df: pd.DataFrame,
    atr_multiplier: float,
    atr_lookback: int,
    safe_haven: str | None,
) -> pd.DataFrame:
    """
    第二层：ATR 跟踪止损拦截。

    对每只风险资产维护持仓期间最高价（high water mark）。
    若收盘价从 HWM 回落超过 atr_multiplier * ATR，则清仓并转入 safe_haven。

    Parameters
    ----------
    weights_df : pd.DataFrame
        列名为 "权重_{标的名称}" 的每日权重表。
    close_df : pd.DataFrame
        收盘价表。
    high_df : pd.DataFrame
        最高价表。
    low_df : pd.DataFrame
        最低价表。
    atr_multiplier : float
        ATR 乘数，例如 3.0 表示回落 3 倍 ATR 触发止损。
    atr_lookback : int
        ATR 回望周期。
    safe_haven : str | None
        止损后资金去向。若为 None 则该部分空仓。

    Returns
    -------
    pd.DataFrame
        调整后的权重表（副本）。
    """
    adjusted = weights_df.copy()
    weight_cols = [c for c in adjusted.columns if c.startswith("权重_")]
    name_list = [c.replace("权重_", "") for c in weight_cols]
    risk_names = [n for n in name_list if n != safe_haven]

    # True Range
    tr = pd.DataFrame(index=adjusted.index, columns=risk_names)
    for name in risk_names:
        tr[name] = pd.concat(
            [
                high_df[name] - low_df[name],
                (high_df[name] - close_df[name].shift(1)).abs(),
                (low_df[name] - close_df[name].shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
    atr = tr.rolling(atr_lookback).mean()

    hwm = pd.DataFrame(np.nan, index=adjusted.index, columns=risk_names)

    for i, date in enumerate(adjusted.index):
        for name in risk_names:
            weight_col = f"权重_{name}"
            current_weight = adjusted.loc[date, weight_col]
            current_price = close_df.loc[date, name]

            prev_weight = adjusted.iloc[i - 1][weight_col] if i > 0 else 0.0

            if current_weight > 0 and prev_weight == 0:
                hwm.loc[date, name] = current_price
            elif current_weight > 0 and prev_weight > 0:
                prev_hwm = hwm.iloc[i - 1][name]
                hwm.loc[date, name] = max(prev_hwm, current_price)
            else:
                hwm.loc[date, name] = np.nan

            if current_weight > 0 and not pd.isna(hwm.loc[date, name]):
                current_atr = atr.loc[date, name]
                if pd.isna(current_atr):
                    continue
                if current_price < hwm.loc[date, name] - atr_multiplier * current_atr:
                    if safe_haven and safe_haven in name_list:
                        safe_col = f"权重_{safe_haven}"
                        adjusted.loc[date, safe_col] += current_weight
                    adjusted.loc[date, weight_col] = 0.0
                    hwm.loc[date, name] = np.nan

    # 归一化
    total = adjusted[weight_cols].sum(axis=1)
    adjusted = adjusted.div(total, axis=0).fillna(0.0)
    return adjusted


def layer3_vol_target_filter(
    weights_df: pd.DataFrame,
    close_df: pd.DataFrame,
    target_vol: float,
    vol_lookback: int,
    comfort_zone: float,
    caution_zone: float,
    caution_scale: float,
    safe_haven: str | None,
) -> pd.DataFrame:
    """
    第三层：目标波动率平准（非线性）。

    计算组合 EWMA 年化波动率，按分段函数缩放风险资产仓位：
    - 波动率 < comfort_zone：线性缩放（target_vol / realized_vol），上限 1.0
    - comfort_zone <= 波动率 < caution_zone：线性结果 * caution_scale
    - 波动率 >= caution_zone：强制清零

    Parameters
    ----------
    weights_df : pd.DataFrame
        列名为 "权重_{标的名称}" 的每日权重表。
    close_df : pd.DataFrame
        收盘价表。
    target_vol : float
        目标年化波动率，例如 0.10 表示 10%。
    vol_lookback : int
        EWMA 波动率的 span 参数。
    comfort_zone : float
        舒适区波动率上限。
    caution_zone : float
        警惕区波动率上限。
    caution_scale : float
        警惕区仓位缩放系数。
    safe_haven : str | None
        减仓后承接资金的去向。若为 None 则空仓。

    Returns
    -------
    pd.DataFrame
        调整后的权重表（副本）。
    """
    if target_vol <= 0:
        raise ValueError("target_vol 必须大于 0")
    if not (0 < comfort_zone < caution_zone):
        raise ValueError("必须满足 0 < comfort_zone < caution_zone")

    adjusted = weights_df.copy()
    weight_cols = [c for c in adjusted.columns if c.startswith("权重_")]
    name_list = [c.replace("权重_", "") for c in weight_cols]

    returns = close_df.pct_change(fill_method=None)
    portfolio_returns = pd.Series(0.0, index=adjusted.index)
    for name in name_list:
        portfolio_returns += adjusted[f"权重_{name}"] * returns[name]

    ewma_vol = portfolio_returns.ewm(span=vol_lookback).std() * np.sqrt(252)
    linear_scale = (target_vol / ewma_vol).fillna(1.0).clip(upper=1.0)

    scale = pd.Series(np.nan, index=ewma_vol.index)
    mask_comfort = ewma_vol < comfort_zone
    mask_caution = (ewma_vol >= comfort_zone) & (ewma_vol < caution_zone)
    mask_panic = ewma_vol >= caution_zone

    scale[mask_comfort] = linear_scale[mask_comfort]
    scale[mask_caution] = linear_scale[mask_caution] * caution_scale
    scale[mask_panic] = 0.0
    scale = scale.fillna(1.0)

    if safe_haven and safe_haven in name_list:
        risk_names = [n for n in name_list if n != safe_haven]
    else:
        risk_names = name_list

    for name in risk_names:
        adjusted[f"权重_{name}"] *= scale

    if safe_haven and safe_haven in name_list:
        safe_col = f"权重_{safe_haven}"
        adjusted[safe_col] = 1.0 - adjusted[[f"权重_{n}" for n in risk_names]].sum(axis=1)

    return adjusted
