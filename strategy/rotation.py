"""轮动策略：每天选最强 top_n 个 ETF，等权分配"""

from __future__ import annotations

import pandas as pd

from core.scorer import momentum_score, slope_r2_score


def run(
    data: pd.DataFrame,
    name_list: list[str],
    params: dict,
) -> pd.DataFrame:
    """
    轮动策略回测

    Parameters
    ----------
    data : pd.DataFrame
        收盘价数据
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

    df = data.copy()

    # 1. 计算日收益率
    for name in name_list:
        df[f"日收益率_{name}"] = df[name] / df[name].shift(1) - 1.0

    # 2. 计算得分
    if scoring == "slope_r2":
        for name in name_list:
            df[f"得分_{name}"] = df[name].rolling(lookback).apply(
                lambda x: slope_r2_score(x, lookback)
            )
        signal_cols = [f"得分_{v}" for v in name_list]
        prefix = "得分_"
    else:
        for name in name_list:
            df[f"涨幅_{name}"] = df[name] / df[name].shift(lookback + 1) - 1.0
        signal_cols = [f"涨幅_{v}" for v in name_list]
        prefix = "涨幅_"

    df = df.dropna()

    # 3. 生成每日权重：top_n 等权，其余为 0
    rank_df = df[signal_cols].rank(axis=1, ascending=False, method="first")
    for name in name_list:
        col = f"{prefix}{name}"
        df[f"权重_{name}"] = (rank_df[col] <= top_n).astype(float) / top_n

    # 4. 持仓权重前移1天（T日收益由T-1日持仓产生）
    for name in name_list:
        df[f"权重_{name}"] = df[f"权重_{name}"].shift(1)

    df = df.dropna()

    # 5. 计算策略日收益率
    df["轮动策略日收益率"] = 0.0
    for name in name_list:
        df["轮动策略日收益率"] += df[f"日收益率_{name}"] * df[f"权重_{name}"]

    df.loc[df.index[0], "轮动策略日收益率"] = 0.0
    df["轮动策略净值"] = (1.0 + df["轮动策略日收益率"]).cumprod()

    # 记录每日主持仓信号（权重最大的那个）
    weight_cols = [f"权重_{n}" for n in name_list]
    df["信号"] = df[weight_cols].idxmax(axis=1).str.replace("权重_", "")

    return df
