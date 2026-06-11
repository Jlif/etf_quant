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
