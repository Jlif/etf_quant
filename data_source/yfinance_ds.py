"""yfinance 数据源（Yahoo Finance）

作为 akshare 的替代/ fallback 数据源，适合：
- 国内网络 akshare 不稳定时
- Apple Silicon / Python 3.13 下 py_mini_racer 不兼容导致 akshare 新浪源失效时

标的代码规则：
- 深市 ETF（0/1/2/3 开头）加 .SZ，如 159915.SZ
- 其他（沪市等）加 .SS，如 510300.SS
"""

from __future__ import annotations

import pandas as pd

from .base import BaseDataSource


class YFinanceDataSource(BaseDataSource):
    """Yahoo Finance 数据源：自动为国内 ETF 添加 .SS/.SZ 后缀。"""

    name = "yfinance"

    @staticmethod
    def _to_yf_symbol(code: str) -> str:
        """根据 ETF 代码判断交易所并加 Yahoo Finance 后缀。"""
        first = code[0] if code else ""
        # 0/1/2/3 开头归为深交所，其余（5/6/9 等）归上交所
        if first in ("0", "1", "2", "3"):
            return f"{code}.SZ"
        return f"{code}.SS"

    def fetch(
        self,
        code: str,
        start: str,
        end: str | None = None,
        expect_today: bool = False,
    ) -> pd.DataFrame:
        import yfinance as yf

        symbol = self._to_yf_symbol(code)
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end) if end else pd.Timestamp.now()

        ticker = yf.Ticker(symbol)
        # yfinance 的 end 参数是排他的，调整到次日以包含 end 当天
        hist = ticker.history(
            start=start_dt.strftime("%Y-%m-%d"),
            end=(end_dt + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            auto_adjust=True,
        )

        if hist.empty:
            raise RuntimeError(f"yfinance 无法获取 {code} ({symbol}) 数据")

        hist.index = pd.to_datetime(hist.index)
        # Yahoo Finance 返回的索引带有时区，统一转为无时区以便和本地日期比较
        hist.index = hist.index.tz_localize(None)
        hist = hist[~hist.index.duplicated(keep="first")].sort_index()

        # 过滤到请求范围
        hist = hist[hist.index >= start_dt]
        if end:
            hist = hist[hist.index <= end_dt]

        if hist.empty:
            raise RuntimeError(
                f"yfinance 在 {start} ~ {end or '今'} 范围内无 {code} 数据"
            )

        if hist.index[0] > start_dt:
            print(
                f"  [yfinance] {code} 数据仅回溯至 {hist.index[0].date()}，"
                f"无法覆盖 {start_dt.date()}"
            )

        self.adjusted = True  # auto_adjust=True 返回前复权价格
        return hist[["Open", "High", "Low", "Close"]].rename(
            columns={
                "Open": f"{code}_open",
                "High": f"{code}_high",
                "Low": f"{code}_low",
                "Close": f"{code}_close",
            }
        )
