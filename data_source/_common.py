"""数据源模块内部共享工具"""

from __future__ import annotations

import pandas as pd


def is_sz_stock(code: str) -> bool:
    """判断 ETF 代码是否属于深交所（首位为 0/1/2/3）。"""
    return code[0] in ("0", "1", "2", "3") if code else False


def filter_date_range(
    df: pd.DataFrame,
    start_dt: pd.Timestamp | None = None,
    end_dt: pd.Timestamp | None = None,
    name: str = "",
) -> pd.DataFrame:
    """按 [start_dt, end_dt] 过滤数据，并处理空数据 / 起始日晚于目标的情况。"""
    if start_dt is not None:
        df = df[df.index >= start_dt]
    if end_dt is not None:
        df = df[df.index <= end_dt]

    if df.empty:
        end_str = end_dt.date() if end_dt is not None else "今"
        start_str = start_dt.date() if start_dt is not None else "起始"
        raise RuntimeError(f"{name} 在 {start_str} ~ {end_str} 范围内无数据")

    if start_dt is not None and df.index[0] > start_dt:
        print(
            f"  [{name}] 数据仅回溯至 {df.index[0].date()}，"
            f"无法覆盖 {start_dt.date()}"
        )

    return df


def rename_ohlc(df: pd.DataFrame, code: str, names: dict[str, str]) -> pd.DataFrame:
    """
    将数据源原始 OHLC 列重命名为 {code}_open/high/low/close。

    Parameters
    ----------
    names : dict[str, str]
        {"open": "原始开盘列名", "high": "原始最高列名", ...}
    """
    return df[[names["open"], names["high"], names["low"], names["close"]]].rename(
        columns={
            names["open"]: f"{code}_open",
            names["high"]: f"{code}_high",
            names["low"]: f"{code}_low",
            names["close"]: f"{code}_close",
        }
    )
