"""配置加载与校验"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import yaml


@dataclass
class DataSourceConfig:
    provider: Literal["akshare", "yfinance"] = "yfinance"


@dataclass
class BacktestConfig:
    start_date: str = "20130729"
    cache_dir: str = "./data_cache"


@dataclass
class PoolItem:
    code: str
    name: str
    weight: float = 0.0  # 百分数，如 25 表示 25%
    type: str | None = None


@dataclass
class StrategyConfig:
    name: str
    description: str = ""
    mode: Literal["rotation", "weighted"] = "rotation"
    pool: list[PoolItem] = field(default_factory=list)
    params: dict = field(default_factory=dict)
    start_date: str | None = None  # 策略级起始日, None 则使用全局 backtest.start_date
    enabled: bool = True  # 是否启用该策略


@dataclass
class AppConfig:
    data_source: DataSourceConfig = field(default_factory=DataSourceConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    strategies: list[StrategyConfig] = field(default_factory=list)


def load_config(path: str = "config.yaml") -> AppConfig:
    """从 YAML 文件加载配置"""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    ds = raw.get("data_source", {})
    bt = raw.get("backtest", {})

    strategies = []
    for s in raw.get("strategies", []):
        pool = [PoolItem(**item) for item in s.get("pool", [])]
        # 校验权重
        if s.get("mode") == "weighted":
            total = sum(p.weight for p in pool)
            if abs(total - 100) > 0.1:
                raise ValueError(
                    f'策略 "{s["name"]}" 权重之和为 {total}%，必须等于 100%'
                )

        # 校验 adaptive_scoring 的 benchmark 参数
        params = s.get("params", {})
        if params.get("adaptive_scoring"):
            has_sector = any(p.type == "行业股票" for p in pool)
            if has_sector:
                benchmark = params.get("benchmark")
                if not benchmark:
                    raise ValueError(
                        f'策略 "{s["name"]}" 开启 adaptive_scoring 且包含行业股票时，'
                        f'必须在 params 中配置 benchmark'
                    )
                pool_names = {p.name for p in pool}
                if benchmark not in pool_names:
                    raise ValueError(
                        f'策略 "{s["name"]}" 的 benchmark "{benchmark}" 不在 pool 中，'
                        f'可用的标的: {sorted(pool_names)}'
                    )

        # 校验需要 safe_haven 的风控参数
        needs_safe_haven = (
            params.get("absolute_momentum_filter", False)
            or params.get("target_volatility") is not None
            or params.get("trailing_stop_pct") is not None
        )
        if needs_safe_haven:
            safe_haven = params.get("safe_haven")
            if not safe_haven:
                raise ValueError(
                    f'策略 "{s["name"]}" 开启风控过滤器时必须配置 safe_haven'
                )
            pool_names = {p.name for p in pool}
            if safe_haven not in pool_names:
                raise ValueError(
                    f'策略 "{s["name"]}" 的 safe_haven "{safe_haven}" 不在 pool 中，'
                    f'可用的标的: {sorted(pool_names)}'
                )
        strategies.append(
            StrategyConfig(
                name=s["name"],
                description=s.get("description", ""),
                mode=s["mode"],
                pool=pool,
                params=s.get("params", {}),
                start_date=s.get("start_date"),
                enabled=s.get("enabled", True),
            )
        )

    return AppConfig(
        data_source=DataSourceConfig(
            provider=ds.get("provider", "yfinance"),
        ),
        backtest=BacktestConfig(
            start_date=bt.get("start_date", "20130729"),
            cache_dir=bt.get("cache_dir", "./data_cache"),
        ),
        strategies=strategies,
    )
