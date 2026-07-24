"""akshare 数据源（东方财富 → 腾讯 fallback）"""

from __future__ import annotations

import time
import threading
from datetime import timedelta

import pandas as pd
import requests

from .base import BaseDataSource
from ._common import filter_date_range, is_sz_stock, rename_ohlc

# ---------------------------------------------------------------------------
# mini_racer (V8) 并发安全：akshare 的新浪接口 fund_etf_hist_sina 内部使用
# py_mini_racer 执行 JS 解密。V8 的 AddressPoolManager 在多线程同时初始化
# isolate 时存在竞态，会触发致命崩溃：
#     [FATAL:address_pool_manager.cc(67)] Check failed: !pool->IsInitialized().
# 由于 orchestrator 会用 ThreadPoolExecutor 并发下载多个 ETF，必须用一个
# 进程级锁串行化 mini_racer 的使用（已验证可消除该崩溃）。
# 注意：东方财富 fund_etf_hist_em 与腾讯接口均为纯 HTTP，不涉及 mini_racer，
# 仍可并发，不受此锁影响。
# ---------------------------------------------------------------------------
_MINI_RACER_LOCK = threading.Lock()

# _is_stale 容忍天数（未传 --today 时生效）：覆盖周末、节假日（春节/国庆最长约
# 9 天休市）以及当日数据尚未生成的情况——默认只要求最近一个交易日的数据，
# 避免因当天数据未出而误判滞后、无谓地切换数据源。
STALE_TOLERANCE_DAYS = 9


class AkshareDataSource(BaseDataSource):
    """akshare 数据源：优先东方财富，失败后回退腾讯/新浪。"""

    name = "akshare"

    @staticmethod
    def _to_exchange_symbol(code: str) -> str:
        """根据 ETF 代码判断交易所并加前缀（sh510300 / sz159915）。"""
        return f"sz{code}" if is_sz_stock(code) else f"sh{code}"

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
        return rename_ohlc(
            df,
            code,
            {"open": "开盘", "high": "最高", "low": "最低", "close": "收盘"},
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
            r = requests.get(url, timeout=8)
            r.raise_for_status()
            payload = r.json()

            if payload.get("code") != 0 or not payload.get("data"):
                raise RuntimeError(f"腾讯接口返回错误: {payload.get('msg')}")

            symbol_data = payload["data"].get(symbol, {})
            # 该年份可能尚未上市，返回空数据，跳过即可。
            # 新上市 ETF（尚无分红除权）腾讯不返回 qfqday，只有 day，
            # 此时未复权与前复权等价，回退使用 day。
            raw = symbol_data.get("qfqday") or symbol_data.get("day", [])
            if not raw:
                continue

            all_rows.extend(raw)
            # 温和限速，避免对腾讯接口造成压力
            if year < end_year:
                time.sleep(0.01)  # 减少限速延迟

        # 腾讯格式：[date, open, close, high, low, volume]
        df = pd.DataFrame(all_rows, columns=["date", "open", "close", "high", "low", "volume"])
        df["date"] = pd.to_datetime(df["date"])
        df = df.drop_duplicates("date").sort_values("date").set_index("date")
        df = df[["open", "high", "low", "close"]].astype(float)

        df = filter_date_range(df, start_dt=start_dt, end_dt=end_dt, name=f"腾讯 {code}")

        self.adjusted = True
        return rename_ohlc(
            df,
            code,
            {"open": "open", "high": "high", "low": "low", "close": "close"},
        )

    def _fetch_sina(self, code: str, start: str, end: str | None) -> pd.DataFrame:
        """新浪财经数据源（未复权），作为腾讯失败/滞后的 fallback。"""
        import akshare as ak

        symbol = self._to_exchange_symbol(code)
        # fund_etf_hist_sina 内部用 py_mini_racer 执行 JS 解密，并发会触发 V8
        # AddressPoolManager 崩溃，故用进程级锁串行化该调用。
        with _MINI_RACER_LOCK:
            df = ak.fund_etf_hist_sina(symbol=symbol)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        df = df[["open", "high", "low", "close"]].astype(float)

        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end) if end else None
        df = filter_date_range(df, start_dt=start_dt, end_dt=end_dt, name=f"新浪 {code}")

        # 新浪返回未复权价格
        self.adjusted = False
        return rename_ohlc(
            df,
            code,
            {"open": "open", "high": "high", "low": "low", "close": "close"},
        )

    def fetch(self, code: str, start: str, end: str | None = None, expect_today: bool = False) -> pd.DataFrame:
        # 主数据源：东方财富（前复权），成功直接返回，不做滞后判定
        try:
            return self._fetch_em(code, start, end)
        except Exception as e:
            error_msg = str(e)
            # 简化网络错误信息
            if "RemoteDisconnected" in error_msg or "Connection aborted" in error_msg:
                print(f"  [akshare] 东方财富获取 {code} 网络连接失败，尝试腾讯...")
            else:
                print(f"  [akshare] 东方财富获取 {code} 失败: {error_msg[:80]}...，尝试腾讯...")

        # 回退链：腾讯(前复权) -> 新浪(未复权)
        # expect_today=False 时 _is_stale 容忍若干天，默认只要求最近一个交易日的数据，
        # 避免因当天数据未出而误判滞后、无谓切换数据源。
        fallback_df = None        # 兜底数据（首个成功获取者，优先复权的腾讯）
        fallback_adjusted = None  # 对应的复权状态
        for label, fetcher in (("腾讯", self._fetch_tencent), ("新浪", self._fetch_sina)):
            try:
                df_cand = fetcher(code, start, end)
            except Exception as e:
                print(f"  [akshare] {label}获取 {code} 失败: {e}，尝试下一个数据源...")
                continue
            if not self._is_stale(df_cand, end, expect_today=expect_today):
                print(f"  [akshare] 已使用{label}数据，最新 {df_cand.index[-1].date()}")
                return df_cand
            # 数据滞后（通常是当日数据尚未生成）：留作兜底。
            # 首个成功者优先（腾讯复权优先于新浪未复权），避免被未复权数据覆盖。
            if fallback_df is None:
                fallback_df = df_cand
                fallback_adjusted = self.adjusted
            print(f"  [akshare] {label}数据最新 {df_cand.index[-1].date()}（可能为最近交易日），尝试下一个...")

        if fallback_df is not None:
            self.adjusted = fallback_adjusted
            print(f"  [akshare] 采用兜底数据，最新 {fallback_df.index[-1].date()}")
            return fallback_df
        raise RuntimeError(f"无法通过东方财富/腾讯/新浪 获取 {code} 数据")

    def _is_stale(self, df: pd.DataFrame, end: str | None = None, expect_today: bool = False) -> bool:
        """判断 df 的最新日期是否滞后。

        - expect_today=False（默认，未传 --today）：只要求最近一个交易日的数据，
          容忍 STALE_TOLERANCE_DAYS 天（覆盖周末/节假日/当日尚未出数据）。
        - expect_today=True（传了 --today）：严格要求当日数据，不容忍。
        """
        if df.empty:
            return True
        latest = df.index[-1].date()
        expected = pd.to_datetime(end).date() if end else pd.Timestamp.now().date()
        tolerance = timedelta(days=0) if expect_today else timedelta(days=STALE_TOLERANCE_DAYS)
        return latest < expected - tolerance
