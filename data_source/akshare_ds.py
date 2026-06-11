"""东方财富数据源 (akshare)"""

from __future__ import annotations

import pandas as pd

from .base import BaseDataSource


class AkshareDataSource(BaseDataSource):
    """东方财富数据源"""

    name = "akshare"

    def fetch(self, code: str, start: str, end: str | None = None) -> pd.DataFrame:
        import akshare as ak

        df = ak.fund_etf_hist_em(
            symbol=code,
            period="daily",
            start_date=start,
            end_date=end or pd.Timestamp.now().strftime("%Y%m%d"),
            adjust="qfq",
        )

        df["日期"] = pd.to_datetime(df["日期"])
        df = df.set_index("日期").sort_index()
        return df[["收盘"]].rename(columns={"收盘": code})
