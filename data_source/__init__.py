"""数据源模块"""

from .base import BaseDataSource
from .akshare_ds import AkshareDataSource
from .yfinance_ds import YFinanceDataSource

BUILTIN_SOURCES: dict[str, type[BaseDataSource]] = {
    "akshare": AkshareDataSource,
    "yfinance": YFinanceDataSource,
}


def get_data_source(
    name: str | None = None,
    fallback: bool = True,
) -> BaseDataSource:
    """
    获取数据源实例

    Parameters
    ----------
    name : str | None
        指定数据源名称, None 时自动探测
    fallback : bool
        失败时是否自动回退
    """
    if name is None:
        # 默认优先级: akshare -> yfinance
        candidates = ["akshare", "yfinance"]
    else:
        candidates = [name]

    last_error = None
    for candidate in candidates:
        cls = BUILTIN_SOURCES.get(candidate)
        if cls is None:
            continue

        try:
            instance = cls()
            _test = instance.fetch("510300", "20240101", "20240110")
            if not _test.empty:
                print(f"[数据源] 使用 {candidate}")
                return instance
        except Exception as e:
            last_error = e
            print(f"[数据源] {candidate} 不可用: {e}")
            if not fallback:
                raise
            continue

    raise RuntimeError(
        f"所有数据源均不可用. 最后一次错误: {last_error}\n"
        f"请检查网络连接或数据源配置."
    )
