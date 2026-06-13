"""轮动策略：每天选最强 top_n 个 ETF，等权分配"""

from __future__ import annotations

import pandas as pd

from core.scorer import momentum_score, slope_r2_score


def run(
    data: dict[str, pd.DataFrame],
    name_list: list[str],
    params: dict,
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
    for name in name_list:
        # 新调入标的：开盘价买入，收盘价结算
        df[f"日收益率_再平衡_{name}"] = close_df[name] / open_df[name] - 1.0
        # 继续持有标的：前日收盘价已持有，当日收盘价结算
        df[f"日收益率_持有_{name}"] = close_df[name] / close_df[name].shift(1) - 1.0

    # 2. 计算得分（基于收盘价）
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

    # 4. 持仓权重前移1天（T日收盘后信号决定T+1日持仓）
    for name in name_list:
        df[f"权重_{name}"] = df[f"权重_{name}"].shift(1)

    df = df.dropna()

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

    df.loc[df.index[0], "轮动策略日收益率"] = 0.0
    df["轮动策略净值"] = (1.0 + df["轮动策略日收益率"]).cumprod()

    # 记录每日主持仓信号（权重最大的那个）
    weight_cols = [f"权重_{n}" for n in name_list]
    df["信号"] = df[weight_cols].idxmax(axis=1).str.replace("权重_", "")

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
