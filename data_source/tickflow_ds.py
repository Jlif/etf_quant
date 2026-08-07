"""tickflow 数据源（免费服务，历史日K，前复权）"""

from __future__ import annotations

import pandas as pd

from .base import BaseDataSource
from ._common import filter_date_range, is_sz_stock, rename_ohlc

# 单次请求最大K线数（tickflow 上限 10000）
MAX_COUNT = 10000


class TickFlowDataSource(BaseDataSource):
    """tickflow 数据源：TickFlow.free() 免费服务，仅历史日K（盘中不更新）。"""

    name = "tickflow"

    def __init__(self) -> None:
        from tickflow import TickFlow

        self._tf = TickFlow.free()

    @staticmethod
    def _to_symbol(code: str) -> str:
        """ETF 代码加市场后缀（510300 -> 510300.SH, 159915 -> 159915.SZ）。"""
        return f"{code}.SZ" if is_sz_stock(code) else f"{code}.SH"

    def fetch(self, code: str, start: str, end: str | None = None, expect_today: bool = False) -> pd.DataFrame:
        # tickflow 的 start_time 为排他边界（K线时间戳为交易日 00:00，
        # start_time 需严格小于首根K线），故往前挪一天，多拉的由 filter_date_range 裁掉
        start_dt = pd.to_datetime(start)
        start_ms = int((start_dt - pd.Timedelta(days=1)).timestamp() * 1000)
        kwargs: dict = {"period": "1d", "start_time": start_ms, "count": MAX_COUNT, "as_dataframe": True}
        end_dt = pd.to_datetime(end) if end else None
        if end_dt is not None:
            # end_time 同理按排他边界处理，+1 天保证包含 end 当天
            kwargs["end_time"] = int((end_dt + pd.Timedelta(days=1)).timestamp() * 1000)

        # 默认 forward（比例前复权），与 akshare qfq 语义一致
        df = self._tf.klines.get(self._to_symbol(code), **kwargs)
        if df is None or df.empty:
            raise RuntimeError(f"tickflow 返回 {code} 为空")

        df = df.copy()
        df["date"] = pd.to_datetime(df["trade_date"])
        df = df.drop_duplicates("date").set_index("date").sort_index()

        df = filter_date_range(df, start_dt=start_dt, end_dt=end_dt, name=f"tickflow {code}")

        self.adjusted = True
        return rename_ohlc(
            df,
            code,
            {"open": "open", "high": "high", "low": "low", "close": "close"},
        )
