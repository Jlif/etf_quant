"""加权组合策略：按配置权重持有，定期再平衡"""

from __future__ import annotations

import pandas as pd


def run(
    data: pd.DataFrame,
    name_list: list[str],
    weights: dict[str, float],
    params: dict,
) -> pd.DataFrame:
    """
    加权组合策略回测

    Parameters
    ----------
    data : pd.DataFrame
        收盘价数据
    name_list : list[str]
        标的名称列表
    weights : dict[str, float]
        各标的权重（百分数，如 25 表示 25%）
    params : dict
        - rebalance_freq: 再平衡频率（1=每日，5=每5日，20=每月）

    Returns
    -------
    pd.DataFrame
        包含回测结果的数据
    """
    rebalance_freq = params.get("rebalance_freq", 1)

    df = data.copy()

    # 1. 计算日收益率
    for name in name_list:
        df[f"日收益率_{name}"] = df[name] / df[name].shift(1) - 1.0

    df = df.dropna()

    # 2. 权重归一化（百分数 -> 小数）
    w = {name: weights.get(name, 0) / 100.0 for name in name_list}

    # 3. 计算策略日收益率
    # 固定权重组合：每日收益 = sum(日收益率_i * w_i)
    # 当 rebalance_freq > 1 时，需要考虑再平衡前的漂移
    if rebalance_freq == 1:
        # 每日再平衡，直接加权
        df["轮动策略日收益率"] = sum(df[f"日收益率_{name}"] * w[name] for name in name_list)
    else:
        # 按 rebalance_freq 再平衡
        df["轮动策略日收益率"] = 0.0
        current_weights = {name: w[name] for name in name_list}

        for i in range(len(df)):
            idx = df.index[i]
            if i > 0:
                # 计算今日收益
                daily_return = sum(
                    df.loc[idx, f"日收益率_{name}"] * current_weights[name]
                    for name in name_list
                )
                df.loc[idx, "轮动策略日收益率"] = daily_return

                # 更新权重（因价格变化导致权重漂移）
                for name in name_list:
                    current_weights[name] *= (1 + df.loc[idx, f"日收益率_{name}"])
                total = sum(current_weights.values())
                current_weights = {k: v / total for k, v in current_weights.items()}

            # 再平衡日：恢复到目标权重
            if (i % rebalance_freq) == 0:
                current_weights = {name: w[name] for name in name_list}

    df.loc[df.index[0], "轮动策略日收益率"] = 0.0
    df["轮动策略净值"] = (1.0 + df["轮动策略日收益率"]).cumprod()

    # 记录持仓（固定权重）
    df["信号"] = "/".join(f"{n}={w[n]}%" for n in name_list)

    return df
