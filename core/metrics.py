"""核心绩效指标计算"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_metrics(nav: pd.Series) -> tuple[float, float, float]:
    """从净值序列计算总收益、年化收益率（CAGR）和最大回撤。"""
    total_return = nav.iloc[-1] / nav.iloc[0] - 1.0
    n_years = len(nav) / 252.0
    cagr = (
        (nav.iloc[-1] / nav.iloc[0]) ** (1.0 / n_years) - 1.0
        if n_years > 0 and nav.iloc[0] > 0
        else 0.0
    )
    running_max = nav.expanding().max()
    drawdown = (nav - running_max) / running_max
    max_drawdown = drawdown.min()
    return total_return, cagr, max_drawdown


def compute_sharpe(nav: pd.Series) -> float:
    """从净值序列计算年化夏普比率（无风险利率假设为 0）。"""
    daily_returns = nav.pct_change().dropna()
    if daily_returns.std() == 0 or len(daily_returns) < 2:
        return 0.0
    return (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)


def annualized_volatility(returns: pd.Series) -> float:
    """日收益序列的年化波动率。"""
    return returns.std() * np.sqrt(252)
