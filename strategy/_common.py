"""策略模块内部共享工具"""

from __future__ import annotations

import pandas as pd

WEIGHT_PREFIX = "权重_"


def weight_col(name: str) -> str:
    """返回标的对应的权重列名。"""
    return f"{WEIGHT_PREFIX}{name}"


def parse_weight_cols(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """从 DataFrame 中解析权重列，返回 (weight_cols, name_list)。"""
    weight_cols = [c for c in df.columns if c.startswith(WEIGHT_PREFIX)]
    name_list = [c.replace(WEIGHT_PREFIX, "") for c in weight_cols]
    return weight_cols, name_list


def risk_names(name_list: list[str], safe_haven: str | None) -> list[str]:
    """返回排除 safe_haven 后的风险标的列表。"""
    return [n for n in name_list if n != safe_haven]


def required_window(etf_type: str | None, default_lookback: int) -> int:
    """根据 ETF 类型返回自适应评分所需的预热窗口长度。"""
    if etf_type == "行业":
        return 62
    if etf_type in {"红利", "自由现金流", "价值"}:
        return 41
    if etf_type == "成长":
        return 21
    if etf_type == "商品":
        return 61
    if etf_type == "宽基":
        return 252
    return default_lookback + 1
