# 自适应 ETF 动量评分实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 rotation 策略中引入基于 ETF 类型的自适应动量得分，同时保持对旧配置的向后兼容。

**Architecture:** 在 `core/scorer.py` 中新增 `adaptive_momentum_score` 及按类型分发的辅助函数；在 `strategy/rotation.py` 中根据 `params.adaptive_scoring` 选择统一得分或自适应得分；在 `utils/config.py` 中扩展 `PoolItem` 并校验 `benchmark`；在 `main.py` 中把 type 信息传入策略；最后更新 `config.yaml` 示例。

**Tech Stack:** Python 3, pandas, numpy, scikit-learn (LinearRegression), PyYAML

## Global Constraints

- `adaptive_scoring` 默认 false，未配置时保持原有 `scoring` 行为。
- 未识别 `type` 退化为 `momentum_score`，并打印警告。
- 当 `adaptive_scoring=true` 且 pool 中存在 `type: "行业股票"` 时，`benchmark` 必填且必须存在于 pool 的 name 中。
- 所有新增函数必须保留类型注解，与现有代码风格一致。
- 每次任务完成后必须提交一次 commit。

---

## 文件变更清单

| 文件 | 责任 |
|---|---|
| `utils/config.py` | `PoolItem` 增加 `type` 字段；校验 `adaptive_scoring` 与 `benchmark` 的合法性 |
| `core/scorer.py` | 新增 `adaptive_momentum_score` 及行业/红利/商品/宽基辅助函数 |
| `strategy/rotation.py` | 根据 `adaptive_scoring` 选择得分计算方式 |
| `main.py` | 将 pool 中的 `type` 映射传入 `rotation.run()` |
| `config.yaml` | 更新启用策略示例，加入 `type` 和 `adaptive_scoring` |

---

### Task 1: 扩展配置模型并校验 benchmark

**Files:**
- Modify: `utils/config.py:22-27`
- Modify: `utils/config.py:56-84`
- Test: `tests/test_config.py`（如不存在则创建）

**Interfaces:**
- Consumes: YAML 配置中的 `pool[].type`、`params.adaptive_scoring`、`params.benchmark`
- Produces: `PoolItem(type: str | None)`、`StrategyConfig` 加载时校验 `benchmark`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_config.py
import pytest
from utils.config import load_config, PoolItem


def test_pool_item_has_type():
    item = PoolItem(code="510300", name="沪深300ETF", type="宽基")
    assert item.type == "宽基"


def test_adaptive_scoring_requires_benchmark(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("""
data_source:
  provider: akshare
backtest:
  start_date: "20210101"
strategies:
  - name: "测试策略"
    mode: rotation
    pool:
      - { code: "510300", name: "沪深300ETF", type: "宽基" }
      - { code: "512690", name: "酒ETF", type: "行业股票" }
    params:
      adaptive_scoring: true
      top_n: 1
""", encoding="utf-8")
    with pytest.raises(ValueError, match="benchmark"):
        load_config(str(config))


def test_benchmark_must_exist_in_pool(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("""
data_source:
  provider: akshare
backtest:
  start_date: "20210101"
strategies:
  - name: "测试策略"
    mode: rotation
    pool:
      - { code: "512690", name: "酒ETF", type: "行业股票" }
    params:
      adaptive_scoring: true
      benchmark: "沪深300ETF"
      top_n: 1
""", encoding="utf-8")
    with pytest.raises(ValueError, match="benchmark"):
        load_config(str(config))
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
python -m pytest tests/test_config.py -v
```

Expected: FAIL（`type` 参数不被接受，或 `adaptive_scoring` 未触发校验）

- [ ] **Step 3: 修改 `PoolItem` 和 `load_config` 校验逻辑**

```python
# utils/config.py
@dataclass
class PoolItem:
    code: str
    name: str
    weight: float = 0.0
    type: str | None = None
```

在 `load_config` 的策略校验段新增：

```python
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
```

- [ ] **Step 4: 运行测试，确认通过**

```bash
python -m pytest tests/test_config.py -v
```

Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add utils/config.py tests/test_config.py
git commit -m "feat(config): add PoolItem.type and adaptive_scoring benchmark validation

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: 实现自适应动量得分函数

**Files:**
- Modify: `core/scorer.py`
- Test: `tests/test_scorer.py`（如不存在则创建）

**Interfaces:**
- Consumes: `pd.Series` 价格序列、类型字符串、可选 benchmark 序列
- Produces: `adaptive_momentum_score(...)` 返回 float；`_residual_momentum_score`、`_risk_adjusted_momentum_score`、`_trend_momentum_score`、`_breakout_score`、`_regression_slope` 为内部辅助函数

- [ ] **Step 1: 写失败测试**

```python
# tests/test_scorer.py
import numpy as np
import pandas as pd
import pytest

from core.scorer import adaptive_momentum_score, momentum_score


def _price_series(values):
    return pd.Series(values, index=pd.date_range("2024-01-01", periods=len(values)))


def test_unknown_type_falls_back_to_momentum():
    prices = _price_series([100.0, 101.0, 102.0, 103.0, 104.0] * 5)
    score = adaptive_momentum_score(prices, etf_type="未知类型", lookback=20)
    expected = momentum_score(prices, lookback=20)
    assert score == pytest.approx(expected)


def test_sector_residual_momentum_positive_when_outperforming():
    # ETF 持续上涨，benchmark 横盘
    etf = _price_series(np.linspace(100, 120, 80))
    benchmark = _price_series(np.linspace(100, 102, 80))
    score = adaptive_momentum_score(etf, etf_type="行业股票", benchmark_series=benchmark, lookback=20)
    assert score > 0


def test_dividend_risk_adjusted_score():
    prices = _price_series(np.linspace(100, 110, 40))
    score = adaptive_momentum_score(prices, etf_type="红利", lookback=40)
    assert score > 0


def test_commodity_trend_score():
    prices = _price_series(np.linspace(100, 120, 60))
    score = adaptive_momentum_score(prices, etf_type="商品", lookback=60)
    assert score > 0


def test_broad_breakout_score_near_one_at_high():
    prices = _price_series([100.0] * 251 + [120.0])
    score = adaptive_momentum_score(prices, etf_type="宽基", lookback=252)
    assert score == pytest.approx(1.0)
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
python -m pytest tests/test_scorer.py -v
```

Expected: FAIL（`adaptive_momentum_score` 未定义）

- [ ] **Step 3: 实现 `core/scorer.py`**

在 `core/scorer.py` 末尾追加：

```python

def _regression_slope(srs: pd.Series, lookback: int) -> float:
    """对序列最近 lookback 个点做线性回归，返回斜率。"""
    if srs.shape[0] < lookback:
        return np.nan
    x = np.arange(lookback).reshape(-1, 1)
    y = srs.iloc[-lookback:].values / srs.iloc[-lookback]
    lr = LinearRegression().fit(x, y)
    return lr.coef_[0]


def _residual_momentum_score(
    srs: pd.Series,
    benchmark_series: pd.Series,
    lookback: int = 20,
) -> float:
    """
    行业股票：相对 benchmark 的残差动量 × 动量加速度修正。
    """
    if srs.shape[0] < max(lookback, 60) + 1:
        return np.nan

    # 对齐日期
    aligned = pd.concat([srs, benchmark_series], axis=1).dropna()
    if aligned.shape[0] < lookback + 1:
        return np.nan
    aligned.columns = ["etf", "bench"]

    # 滚动收益率残差
    etf_ret = aligned["etf"].pct_change().dropna()
    bench_ret = aligned["bench"].pct_change().dropna()
    if len(etf_ret) < lookback or len(bench_ret) < lookback:
        return np.nan

    recent_etf = etf_ret.iloc[-lookback:].values.reshape(-1, 1)
    recent_bench = bench_ret.iloc[-lookback:].values.reshape(-1, 1)
    lr = LinearRegression().fit(recent_bench, recent_etf)
    residual = float(recent_etf[-1] - lr.predict(recent_bench[-1].reshape(1, -1))[0])

    # 动量加速度 = 短期斜率 / 中期斜率
    slope_20 = _regression_slope(aligned["etf"], 20)
    slope_60 = _regression_slope(aligned["etf"], 60)
    if pd.isna(slope_20) or pd.isna(slope_60) or slope_60 <= 0:
        accel = 1.0
    else:
        accel = slope_20 / slope_60

    return residual * accel


def _risk_adjusted_momentum_score(srs: pd.Series, lookback: int) -> float:
    """
    红利 / 商品：区间收益 / 区间年化波动率。
    """
    if srs.shape[0] < lookback + 1:
        return np.nan
    total_return = srs.iloc[-1] / srs.iloc[-(lookback + 1)] - 1.0
    daily_returns = srs.pct_change().dropna().iloc[-lookback:]
    if len(daily_returns) < 2:
        return np.nan
    vol = daily_returns.std() * np.sqrt(252)
    if vol == 0 or pd.isna(vol):
        return np.nan
    return total_return / vol


def _trend_momentum_score(srs: pd.Series, lookback: int) -> float:
    """
    商品中长周期趋势：风险调整收益（与红利共用实现）。
    """
    return _risk_adjusted_momentum_score(srs, lookback)


def _breakout_score(srs: pd.Series, lookback: int = 252) -> float:
    """
    宽基：当前价 / 过去 lookback 日最高价。
    """
    if srs.shape[0] < lookback:
        return np.nan
    highest = srs.iloc[-lookback:].max()
    if highest == 0 or pd.isna(highest):
        return np.nan
    return srs.iloc[-1] / highest


def adaptive_momentum_score(
    srs: pd.Series,
    etf_type: str | None,
    benchmark_series: pd.Series | None = None,
    lookback: int = 20,
) -> float:
    """
    根据 ETF 类型计算自适应动量得分。

    Parameters
    ----------
    srs : pd.Series
        收盘价序列
    etf_type : str | None
        ETF 类型，如 "行业股票", "红利", "商品", "宽基"
    benchmark_series : pd.Series | None
        行业股票残差动量所需的基准序列
    lookback : int
        默认回望周期，不同类型内部可能使用固定周期

    Returns
    -------
    float
        得分，数据不足时返回 np.nan
    """
    if etf_type == "行业股票":
        if benchmark_series is None:
            raise ValueError("行业股票动量需要提供 benchmark_series")
        return _residual_momentum_score(srs, benchmark_series, lookback=20)
    elif etf_type == "红利":
        return _risk_adjusted_momentum_score(srs, lookback=40)
    elif etf_type == "商品":
        return _trend_momentum_score(srs, lookback=60)
    elif etf_type == "宽基":
        return _breakout_score(srs, lookback=252)
    else:
        if etf_type:
            print(f"[自适应动量] 未识别类型 \"{etf_type}\"，退化为默认 momentum 得分")
        return momentum_score(srs, lookback)
```

- [ ] **Step 4: 运行测试，确认通过**

```bash
python -m pytest tests/test_scorer.py -v
```

Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add core/scorer.py tests/test_scorer.py
git commit -m "feat(scorer): add adaptive momentum score by ETF type

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: 在 rotation 策略中接入自适应得分

**Files:**
- Modify: `strategy/rotation.py:11-77`
- Test: `tests/test_rotation.py`（如不存在则创建）

**Interfaces:**
- Consumes: `data`（含 close/open/high/low）、`name_list`、`params`、新增可选 `name_types: dict[str, str | None]`
- Produces: 当 `adaptive_scoring=true` 时，使用 `adaptive_momentum_score` 计算每日得分

- [ ] **Step 1: 写失败测试**

```python
# tests/test_rotation.py
import numpy as np
import pandas as pd

from strategy.rotation import run


def _make_data(n=300, names=("沪深300ETF", "红利ETF")):
    idx = pd.date_range("2023-01-01", periods=n)
    close = pd.DataFrame({name: np.linspace(100, 100 + i * 10, n) for i, name in enumerate(names)}, index=idx)
    open_ = close.copy()
    high = close * 1.01
    low = close * 0.99
    return {"close": close, "open": open_, "high": high, "low": low}


def test_adaptive_scoring_uses_type_scores():
    data = _make_data()
    params = {
        "lookback": 20,
        "scoring": "momentum",
        "adaptive_scoring": True,
        "benchmark": "沪深300ETF",
        "top_n": 1,
    }
    name_types = {"沪深300ETF": "宽基", "红利ETF": "红利"}
    result = run(data, list(data["close"].columns), params, name_types=name_types)
    assert "轮动策略净值" in result.columns
    assert result["轮动策略净值"].iloc[-1] > 0
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
python -m pytest tests/test_rotation.py::test_adaptive_scoring_uses_type_scores -v
```

Expected: FAIL（`run` 不接受 `name_types`）

- [ ] **Step 3: 修改 `strategy/rotation.py`**

修改函数签名：

```python
def run(
    data: dict[str, pd.DataFrame],
    name_list: list[str],
    params: dict,
    name_types: dict[str, str | None] | None = None,
) -> pd.DataFrame:
```

在导入处增加：

```python
from core.scorer import (
    adaptive_momentum_score,
    momentum_quality_score,
    momentum_score,
    slope_r2_score,
)
```

替换现有得分计算段（约第 58-76 行）为：

```python
    # 2. 计算得分（基于收盘价）
    if params.get("adaptive_scoring"):
        benchmark_name = params.get("benchmark")
        benchmark_series = close_df[benchmark_name] if benchmark_name else None
        type_map = name_types or {}
        for name in name_list:
            etf_type = type_map.get(name)
            df[f"自适应得分_{name}"] = df[name].rolling(max(lookback, 252)).apply(
                lambda x: adaptive_momentum_score(
                    x,
                    etf_type=etf_type,
                    benchmark_series=benchmark_series,
                    lookback=lookback,
                )
            )
        signal_cols = [f"自适应得分_{v}" for v in name_list]
        prefix = "自适应得分_"
    elif scoring == "slope_r2":
        ...  # 保持原逻辑
```

- [ ] **Step 4: 运行测试，确认通过**

```bash
python -m pytest tests/test_rotation.py -v
```

Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add strategy/rotation.py tests/test_rotation.py
git commit -m "feat(rotation): wire adaptive momentum scoring into rotation strategy

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: 在 main.py 中传递 type 映射

**Files:**
- Modify: `main.py:260-264`

**Interfaces:**
- Consumes: `strategy.pool` 中的 `PoolItem.type`
- Produces: 调用 `rotation.run(..., name_types={name: type})`

- [ ] **Step 1: 修改 `run_strategy` 中的 rotation 调用**

```python
    if strategy.mode == "rotation":
        name_types = {p.name: p.type for p in strategy.pool}
        result = rotation.run(data, name_list, strategy.params, name_types=name_types)
```

- [ ] **Step 2: 运行回测验证**

```bash
python main.py
```

Expected: 回测成功完成，输出报告和图表

- [ ] **Step 3: 提交**

```bash
git add main.py
git commit -m "feat(main): pass ETF type mapping to rotation strategy

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 5: 更新 config.yaml 示例

**Files:**
- Modify: `config.yaml:72-106`

**Interfaces:**
- 无新增代码接口，仅配置示例

- [ ] **Step 1: 为启用策略的 pool 项添加 type**

将当前启用的 "动量轮动策略" 改为：

```yaml
  - name: "动量轮动策略"
    description: "A股风格轮动，自适应动量得分"
    enabled: true
    mode: rotation
    pool:
      - { code: "512040", name: "价值100ETF富国", type: "红利" }
      - { code: "510300", name: "沪深300ETF", type: "宽基" }
      - { code: "515180", name: "红利ETF易方达", type: "红利" }
      - { code: "588080", name: "科创50ETF易方达", type: "宽基" }
      - { code: "159915", name: "创业板ETF", type: "宽基" }
      - { code: "511260", name: "10年国债ETF国泰", type: "债券" }
      - { code: "513500", name: "标普500ETF博时", type: "宽基" }
      - { code: "518880", name: "黄金ETF", type: "商品" }
    params:
      lookback: 22
      scoring: momentum
      adaptive_scoring: true
      benchmark: "沪深300ETF"
      top_n: 1
      absolute_momentum_filter: true
      absolute_momentum_lookback: 9
      absolute_momentum_threshold: 0.0
      safe_haven: "10年国债ETF国泰"
      target_volatility: 0.08
      volatility_lookback: 26
      trailing_stop_pct: 0.07
```

- [ ] **Step 2: 运行回测验证**

```bash
python main.py
```

Expected: 回测成功完成，输出报告和图表

- [ ] **Step 3: 提交**

```bash
git add config.yaml
git commit -m "chore(config): update example strategy with adaptive scoring and ETF types

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 6: 端到端验证与回归测试

**Files:**
- 运行命令验证

- [ ] **Step 1: 运行全部单元测试**

```bash
python -m pytest tests/ -v
```

Expected: 全部通过

- [ ] **Step 2: 运行回测并检查输出**

```bash
python main.py
```

Expected:
- 无报错
- `output/` 目录生成 HTML 报告和 PNG 图表
- 日志中出现 `[自适应动量] 未识别类型 "债券"，退化为默认 momentum 得分` 的警告

- [ ] **Step 3: 关闭 adaptive_scoring 做兼容性验证**

临时将 `config.yaml` 中 `adaptive_scoring` 改为 false，运行：

```bash
python main.py
```

Expected: 回测成功，结果与修改前一致

验证完成后恢复 `adaptive_scoring: true`。

- [ ] **Step 4: 提交最终验证结果**

```bash
git add -A
git commit -m "test: end-to-end validation for adaptive momentum

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**
- ✅ PoolItem.type — Task 1
- ✅ params.adaptive_scoring / benchmark — Task 1
- ✅ 行业股票残差动量 — Task 2
- ✅ 红利风险调整动量 — Task 2
- ✅ 商品中长周期趋势 — Task 2
- ✅ 宽基箱体突破 — Task 2
- ✅ 未识别类型 fallback — Task 2
- ✅ rotation 策略接入 — Task 3
- ✅ main.py 传递 type — Task 4
- ✅ config.yaml 示例 — Task 5
- ✅ 兼容性验证 — Task 6

**Placeholder scan:** 无 TBD/TODO/"实现 later"/"适当处理"。

**Type consistency：**
- `PoolItem.type: str | None` 在 Task 1 定义，Task 4 使用。
- `adaptive_momentum_score(srs, etf_type, benchmark_series, lookback)` 在 Task 2 定义，Task 3 调用。
- `rotation.run(..., name_types=...)` 在 Task 3 定义，Task 4 调用。

---

## 执行方式选择

Plan complete and saved to `docs/superpowers/plans/2026-06-30-adaptive-momentum.md`.

**Two execution options:**

1. **Subagent-Driven (recommended)** - Dispatch a fresh subagent per task, review between tasks, fast iteration
2. **Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
