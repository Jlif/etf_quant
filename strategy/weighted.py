"""加权组合策略：按配置权重持有，定期再平衡"""

from __future__ import annotations

import pandas as pd


def run(
    data: dict[str, pd.DataFrame],
    name_list: list[str],
    weights: dict[str, float],
    params: dict,
) -> pd.DataFrame:
    """
    加权组合策略回测

    再平衡日：原持仓先经历隔夜跳空，再按目标权重在开盘价重新配置；
    非再平衡日：按当前漂移权重持有到收盘价。

    Parameters
    ----------
    data : dict[str, pd.DataFrame]
        包含 close/open/high/low 四个字典，close 用于非再平衡收益和隔夜跳空，
        open 用于再平衡日开盘价成交。
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

    close_df = data["close"].copy()
    open_df = data["open"].copy()

    df = close_df.copy()

    # 1. 计算三种日收益率
    for name in name_list:
        # 再平衡日：开盘价买入/卖出，收盘价结算
        df[f"日收益率_再平衡_{name}"] = close_df[name] / open_df[name] - 1.0
        # 隔夜跳空：前日收盘价 -> 当日开盘价
        df[f"日收益率_隔夜_{name}"] = open_df[name] / close_df[name].shift(1) - 1.0
        # 非再平衡日：前日收盘价 -> 当日收盘价
        df[f"日收益率_漂移_{name}"] = close_df[name] / close_df[name].shift(1) - 1.0

    df = df.dropna()

    # 2. 权重归一化（百分数 -> 小数）
    w = {name: weights.get(name, 0) / 100.0 for name in name_list}

    # 3. 计算策略日收益率
    if rebalance_freq == 1:
        # 每日再平衡：开盘即按目标权重配置
        df["轮动策略日收益率"] = sum(df[f"日收益率_再平衡_{name}"] * w[name] for name in name_list)
    else:
        # 按 rebalance_freq 再平衡
        df["轮动策略日收益率"] = 0.0
        current_weights = {name: w[name] for name in name_list}

        for i in range(len(df)):
            idx = df.index[i]
            is_rebalance = (i % rebalance_freq) == 0
            if i > 0:
                if is_rebalance:
                    # 再平衡日：原持仓承受隔夜跳空，然后按目标权重在开盘价重新配置
                    overnight_return = sum(
                        df.loc[idx, f"日收益率_隔夜_{name}"] * current_weights[name]
                        for name in name_list
                    )
                    intraday_return = sum(
                        df.loc[idx, f"日收益率_再平衡_{name}"] * w[name]
                        for name in name_list
                    )
                    daily_return = (1.0 + overnight_return) * (1.0 + intraday_return) - 1.0
                else:
                    # 非再平衡日：按当前漂移权重持有
                    daily_return = sum(
                        df.loc[idx, f"日收益率_漂移_{name}"] * current_weights[name]
                        for name in name_list
                    )
                df.loc[idx, "轮动策略日收益率"] = daily_return

                # 更新权重（因价格变化导致权重漂移）
                if is_rebalance:
                    current_weights = {name: w[name] for name in name_list}
                else:
                    for name in name_list:
                        current_weights[name] *= (1 + df.loc[idx, f"日收益率_漂移_{name}"])
                    total = sum(current_weights.values())
                    current_weights = {k: v / total for k, v in current_weights.items()}

            # 再平衡日初始化目标权重
            if is_rebalance:
                current_weights = {name: w[name] for name in name_list}

    df.loc[df.index[0], "轮动策略日收益率"] = 0.0
    df["轮动策略净值"] = (1.0 + df["轮动策略日收益率"]).cumprod()

    # 记录持仓（固定权重）
    df["信号"] = "/".join(f"{n}={w[n]}%" for n in name_list)

    return df
