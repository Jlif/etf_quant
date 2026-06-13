"""akshare 数据源（东方财富 → 腾讯 fallback）"""

from __future__ import annotations

import time

import pandas as pd
import requests

from .base import BaseDataSource


class AkshareDataSource(BaseDataSource):
    """akshare 数据源：优先东方财富，失败时回退到腾讯（前复权）。"""

    name = "akshare"

    @staticmethod
    def _to_exchange_symbol(code: str) -> str:
        """根据 ETF 代码判断交易所并加前缀（sh510300 / sz159915）。"""
        first = code[0] if code else ""
        # 0/1/2/3 开头归为深交所，其余（5/6/9 等）归上交所
        if first in ("0", "1", "2", "3"):
            return f"sz{code}"
        return f"sh{code}"

    def _fetch_em(self, code: str, start: str, end: str | None) -> pd.DataFrame:
        """东方财富数据源（前复权）。"""
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
        self.adjusted = True
        return df[["开盘", "最高", "最低", "收盘"]].rename(
            columns={
                "开盘": f"{code}_open",
                "最高": f"{code}_high",
                "最低": f"{code}_low",
                "收盘": f"{code}_close",
            }
        )

    def _fetch_tencent(self, code: str, start: str, end: str | None) -> pd.DataFrame:
        """腾讯数据源（东方财富失败时 fallback，按年份分段获取前复权数据）。"""
        symbol = self._to_exchange_symbol(code)
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end) if end else pd.Timestamp.now()

        all_rows: list[list] = []
        start_year = start_dt.year
        end_year = end_dt.year
        for year in range(start_year, end_year + 1):
            year_start = f"{year}-01-01"
            year_end = f"{year}-12-31"
            url = (
                "http://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
                f"?param={symbol},day,{year_start},{year_end},1000,qfq"
            )
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            payload = r.json()

            if payload.get("code") != 0 or not payload.get("data"):
                raise RuntimeError(f"腾讯接口返回错误: {payload.get('msg')}")

            symbol_data = payload["data"].get(symbol, {})
            # 该年份可能尚未上市，返回空数据，跳过即可
            raw = symbol_data.get("qfqday", [])
            if not raw:
                continue

            all_rows.extend(raw)
            # 温和限速，避免对腾讯接口造成压力
            if year < end_year:
                time.sleep(0.05)

        # 腾讯格式：[date, open, close, high, low, volume]
        df = pd.DataFrame(all_rows, columns=["date", "open", "close", "high", "low", "volume"])
        df["date"] = pd.to_datetime(df["date"])
        df = df.drop_duplicates("date").sort_values("date").set_index("date")
        df = df[["open", "high", "low", "close"]].astype(float).rename(
            columns={
                "open": f"{code}_open",
                "high": f"{code}_high",
                "low": f"{code}_low",
                "close": f"{code}_close",
            }
        )

        # 按请求范围过滤
        df = df[df.index >= start_dt]
        if end:
            df = df[df.index <= end_dt]

        if df.empty:
            raise RuntimeError(f"腾讯接口在 {start} ~ {end or '今'} 范围内无数据")

        if df.index[0] > start_dt:
            print(
                f"  [腾讯] {code} 数据仅回溯至 {df.index[0].date()}，"
                f"无法覆盖 {start_dt.date()}"
            )

        self.adjusted = True
        return df

    def fetch(self, code: str, start: str, end: str | None = None) -> pd.DataFrame:
        try:
            return self._fetch_em(code, start, end)
        except Exception as e:
            print(f"  [akshare] 东方财富获取 {code} 失败: {e}，尝试腾讯...")
        return self._fetch_tencent(code, start, end)
