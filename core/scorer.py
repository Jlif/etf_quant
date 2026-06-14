"""评分器：用于计算 ETF 强弱得分"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def momentum_score(srs: pd.Series, lookback: int) -> float:
    """
    N 日涨幅得分
    score = (今日收盘价 / N+1 日前收盘价) - 1
    """
    if srs.shape[0] < lookback + 1:
        return np.nan
    return srs.iloc[-1] / srs.iloc[-(lookback + 1)] - 1.0


def slope_r2_score(srs: pd.Series, lookback: int) -> float:
    """
    斜率 * R^2 得分
    对收盘价序列归一化后做线性回归
    """
    if srs.shape[0] < lookback:
        return np.nan
    x = np.arange(1, lookback + 1)
    y = srs.values / srs.values[0]
    lr = LinearRegression().fit(x.reshape(-1, 1), y)
    slope = lr.coef_[0]
    r_squared = lr.score(x.reshape(-1, 1), y)
    return 10000 * slope * r_squared


def momentum_quality_score(srs: pd.Series, lookback: int) -> float:
    """
    动量质量得分：涨得稳的 ETF 得更高分。

    score = 区间总收益 / 区间日收益波动率

    该指标惩罚那些靠突发暴涨暴跌堆砌出来的“假动量”，
    优先选择稳步上涨的标的。
    """
    if srs.shape[0] < 3:
        return np.nan

    total_return = srs.iloc[-1] / srs.iloc[0] - 1.0
    daily_returns = srs.pct_change().dropna()
    if len(daily_returns) < 2:
        return np.nan

    volatility = daily_returns.std()
    if volatility == 0 or pd.isna(volatility):
        return -np.inf

    return total_return / volatility
