"""数据源模块"""

from .base import BaseDataSource
from .akshare_ds import AkshareDataSource

BUILTIN_SOURCES: dict[str, type[BaseDataSource]] = {
    "akshare": AkshareDataSource,
}


def get_data_source(
    name: str | None = None,
    fallback: bool = True,
    skip_test: bool = False,
) -> BaseDataSource:
    """
    获取数据源实例

    Parameters
    ----------
    name : str | None
        指定数据源名称, None 时自动探测
    fallback : bool
        失败时是否自动回退
    skip_test : bool
        是否跳过连通性测试。当本地缓存已存在时，可跳过测试以避免触发网络请求。
    """
    if name is None:
        # 默认数据源: akshare
        candidates = ["akshare"]
    else:
        candidates = [name]

    last_error = None
    for candidate in candidates:
        cls = BUILTIN_SOURCES.get(candidate)
        if cls is None:
            continue

        try:
            instance = cls()
            if skip_test:
                print(f"[数据源] 使用 {candidate}（跳过连通性测试）")
                return instance

            # 连通性测试仅做轻量探测；失败时若允许 fallback 仍返回实例，
            # 由调用方在有缓存时避免重复请求。
            try:
                _test = instance.fetch("510300", "20240101", "20240110")
                close_col = f"510300_close"
                if close_col in _test.columns and not _test.empty:
                    print(f"[数据源] 使用 {candidate}")
                    return instance
            except Exception as e:
                last_error = e
                print(f"[数据源] {candidate} 连通性测试失败: {e}")
                if fallback:
                    print(f"[数据源] 仍使用 {candidate}（依赖本地缓存）")
                    return instance
                raise
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
