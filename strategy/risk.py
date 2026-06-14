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

    # 滚动年化波动率
    realized_vol = portfolio_returns.rolling(vol_lookback).std() * np.sqrt(252)

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
