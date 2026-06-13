"""数据源抽象基类"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BaseDataSource(ABC):
    """数据源抽象基类"""

    name: str = "base"
    adjusted: bool = True
    """最近一次 fetch 返回的价格是否为复权/已调整价格。"""

    @abstractmethod
    def fetch(self, code: str, start: str, end: str | None = None) -> pd.DataFrame:
        """
        获取单个 ETF 历史数据

        Returns
        -------
        pd.DataFrame
            索引为日期, 包含以 code 命名的 OHLC 四列:
            {code}_open, {code}_high, {code}_low, {code}_close
        """
        ...
