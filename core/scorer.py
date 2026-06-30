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


def _regression_slope(srs: pd.Series, lookback: int) -> float:
    """对序列最近 lookback 个点做线性回归，返回斜率。"""
    if srs.shape[0] < lookback:
        return np.nan
    x = np.arange(lookback).reshape(-1, 1)
    y = srs.iloc[-lookback:].values / srs.iloc[-lookback]
    lr = LinearRegression().fit(x, y)
    return lr.coef_[0]


def _residual_momentum_score(
    srs: pd.Series,
    benchmark_series: pd.Series,
    lookback: int = 20,
) -> float:
    """
    行业股票：相对 benchmark 的残差动量 × 动量加速度修正。
    """
    if srs.shape[0] < max(lookback, 60) + 1:
        return np.nan

    # 对齐日期
    aligned = pd.concat([srs, benchmark_series], axis=1).dropna()
    if aligned.shape[0] < lookback + 1:
        return np.nan
    aligned.columns = ["etf", "bench"]

    # 滚动收益率残差
    etf_ret = aligned["etf"].pct_change().dropna()
    bench_ret = aligned["bench"].pct_change().dropna()
    if len(etf_ret) < lookback or len(bench_ret) < lookback:
        return np.nan

    recent_etf = etf_ret.iloc[-lookback:].values.reshape(-1, 1)
    recent_bench = bench_ret.iloc[-lookback:].values.reshape(-1, 1)
    lr = LinearRegression().fit(recent_bench, recent_etf)
    residual = float(recent_etf[-1, 0] - lr.predict(recent_bench[-1].reshape(1, -1))[0, 0])

    # 动量加速度 = 短期斜率 / 中期斜率
    slope_20 = _regression_slope(aligned["etf"], 20)
    slope_60 = _regression_slope(aligned["etf"], 60)
    if pd.isna(slope_20) or pd.isna(slope_60) or slope_60 <= 0:
        accel = 1.0
    else:
        accel = slope_20 / slope_60

    return residual * accel


def _risk_adjusted_momentum_score(srs: pd.Series, lookback: int) -> float:
    """
    红利 / 商品：区间收益 / 区间年化波动率。
    """
    if srs.shape[0] < lookback + 1:
        return np.nan
    total_return = srs.iloc[-1] / srs.iloc[-(lookback + 1)] - 1.0
    daily_returns = srs.pct_change().dropna().iloc[-lookback:]
    if len(daily_returns) < 2:
        return np.nan
    vol = daily_returns.std() * np.sqrt(252)
    if vol == 0 or pd.isna(vol):
        return np.nan
    return total_return / vol


def _trend_momentum_score(srs: pd.Series, lookback: int) -> float:
    """
    商品中长周期趋势：风险调整收益（与红利共用实现）。
    """
    return _risk_adjusted_momentum_score(srs, lookback)


def _breakout_score(srs: pd.Series, lookback: int = 252) -> float:
    """
    宽基：当前价 / 过去 lookback 日最高价。
    """
    if srs.shape[0] < lookback:
        return np.nan
    highest = srs.iloc[-lookback:].max()
    if highest == 0 or pd.isna(highest):
        return np.nan
    return srs.iloc[-1] / highest


def adaptive_momentum_score(
    srs: pd.Series,
    etf_type: str | None,
    benchmark_series: pd.Series | None = None,
    lookback: int = 20,
) -> float:
    """
    根据 ETF 类型计算自适应动量得分。

    Parameters
    ----------
    srs : pd.Series
        收盘价序列
    etf_type : str | None
        ETF 类型，如 "行业股票", "红利", "商品", "宽基"
    benchmark_series : pd.Series | None
        行业股票残差动量所需的基准序列
    lookback : int
        默认回望周期，不同类型内部可能使用固定周期

    Returns
    -------
    float
        得分，数据不足时返回 np.nan
    """
    if etf_type == "行业股票":
        if benchmark_series is None:
            raise ValueError("行业股票动量需要提供 benchmark_series")
        return _residual_momentum_score(srs, benchmark_series, lookback=20)
    elif etf_type == "红利":
        return _risk_adjusted_momentum_score(srs, lookback=40)
    elif etf_type == "商品":
        return _trend_momentum_score(srs, lookback=60)
    elif etf_type == "宽基":
        return _breakout_score(srs, lookback=252)
    else:
        if etf_type:
            print(f"[自适应动量] 未识别类型 \"{etf_type}\"，退化为默认 momentum 得分")
        return momentum_score(srs, lookback)
