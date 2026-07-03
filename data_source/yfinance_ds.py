"""Yahoo Finance 数据源"""

from __future__ import annotations

import pandas as pd

from .base import BaseDataSource


class YFinanceDataSource(BaseDataSource):
    """Yahoo Finance 数据源"""

    name = "yfinance"

    def _ticker(self, code: str) -> str:
        # A 股代码以 0/1/2/3 开头多为深交所，其余（5/6/9 等）多为上交所
        first = code[0] if code else ""
        suffix = ".SZ" if first in ("0", "1", "2", "3") else ".SS"
        return code + suffix

    def fetch(self, code: str, start: str, end: str | None = None) -> pd.DataFrame:
        import yfinance as yf

        ticker = self._ticker(code)
        start_fmt = pd.to_datetime(start).strftime("%Y-%m-%d")
        end_fmt = pd.to_datetime(end).strftime("%Y-%m-%d") if end else None

        df = yf.download(
            ticker,
            start=start_fmt,
            end=end_fmt,
            progress=False,
            auto_adjust=True,
        )
        if df.empty:
            raise ValueError(f"未获取到 {code} ({ticker}) 的数据")

        if isinstance(df.columns, pd.MultiIndex):
            # yfinance 单 ticker 返回 MultiIndex 列 (Price, Ticker)，取第一层 flatten
            df.columns = df.columns.get_level_values(0)

        df = df[["Open", "High", "Low", "Close"]].rename(
            columns={
                "Open": f"{code}_open",
                "High": f"{code}_high",
                "Low": f"{code}_low",
                "Close": f"{code}_close",
            }
        )
        df.index = pd.to_datetime(df.index)
        return df
