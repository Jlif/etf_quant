"""akshare 数据源（东方财富 → 腾讯 fallback）"""

from __future__ import annotations

import time

import pandas as pd
import requests

from .base import BaseDataSource


class AkshareDataSource(BaseDataSource):
    """akshare 数据源：优先东方财富，失败后回退腾讯/新浪，最后 yfinance。"""

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

    def _fetch_sina(self, code: str, start: str, end: str | None) -> pd.DataFrame:
        """新浪财经数据源（未复权），作为腾讯失败/滞后的 fallback。"""
        import akshare as ak

        symbol = self._to_exchange_symbol(code)
        df = ak.fund_etf_hist_sina(symbol=symbol)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        df = df[["open", "high", "low", "close"]].astype(float).rename(
            columns={
                "open": f"{code}_open",
                "high": f"{code}_high",
                "low": f"{code}_low",
                "close": f"{code}_close",
            }
        )

        start_dt = pd.to_datetime(start)
        df = df[df.index >= start_dt]
        if end:
            end_dt = pd.to_datetime(end)
            df = df[df.index <= end_dt]

        if df.empty:
            raise RuntimeError(f"新浪接口在 {start} ~ {end or '今'} 范围内无数据")

        if df.index[0] > start_dt:
            print(
                f"  [新浪] {code} 数据仅回溯至 {df.index[0].date()}，"
                f"无法覆盖 {start_dt.date()}"
            )

        # 新浪返回未复权价格
        self.adjusted = False
        return df

    def fetch(self, code: str, start: str, end: str | None = None) -> pd.DataFrame:
        try:
            return self._fetch_em(code, start, end)
        except Exception as e:
            error_msg = str(e)
            # 简化网络错误信息
            if "RemoteDisconnected" in error_msg or "Connection aborted" in error_msg:
                print(f"  [akshare] 东方财富获取 {code} 网络连接失败，尝试腾讯...")
            else:
                print(f"  [akshare] 东方财富获取 {code} 失败: {error_msg[:80]}...，尝试腾讯...")

        # 第一层 fallback：腾讯
        df = None
        try:
            df = self._fetch_tencent(code, start, end)
            if not self._is_stale(df, end):
                return df
            print(f"  [akshare] 腾讯数据滞后至 {df.index[-1].date()}，尝试新浪...")
        except Exception as e:
            print(f"  [akshare] 腾讯获取 {code} 失败: {e}，尝试新浪...")

        # 第二层 fallback：新浪
        try:
            df = self._fetch_sina(code, start, end)
            if not self._is_stale(df, end):
                print(f"  [akshare] 已使用新浪数据，最新 {df.index[-1].date()}")
                return df
            print(f"  [akshare] 新浪数据滞后至 {df.index[-1].date()}，尝试 yfinance...")
        except Exception as e:
            print(f"  [akshare] 新浪获取 {code} 失败: {e}，尝试 yfinance...")

        # 第三层 fallback：yfinance（用于国内接口均缺失/滞后的标的，如 159915）
        try:
            from .yfinance_ds import YFinanceDataSource

            yf_ds = YFinanceDataSource()
            df_yf = yf_ds.fetch(code, start, end)
            if not df_yf.empty:
                print(f"  [akshare] 已使用 yfinance 数据，最新 {df_yf.index[-1].date()}")
                self.adjusted = yf_ds.adjusted
                return df_yf
        except Exception as e:
            print(f"  [akshare] yfinance 获取 {code} 失败: {e}")

        if df is not None:
            return df
        raise RuntimeError(f"无法通过东方财富/腾讯/新浪/yfinance 获取 {code} 数据")

    def _is_stale(self, df: pd.DataFrame, end: str | None = None) -> bool:
        """判断 df 的最新日期是否早于预期截止日（默认今天）。"""
        if df.empty:
            return True
        latest = df.index[-1].date()
        expected = pd.to_datetime(end).date() if end else pd.Timestamp.now().date()
        return latest < expected
