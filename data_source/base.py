"""数据源抽象基类"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BaseDataSource(ABC):
    """数据源抽象基类"""

    name: str = "base"

    @abstractmethod
    def fetch(self, code: str, start: str, end: str | None = None) -> pd.DataFrame:
        """
        获取单个 ETF 历史数据

        Returns
        -------
        pd.DataFrame
            索引为日期, 至少包含一列以 code 命名的收盘价
        """
        ...
