# 自适应 ETF 动量评分设计

**日期**: 2026-06-30  
**主题**: 为 rotation 策略引入基于 ETF 类型的自适应动量得分  
**状态**: 已批准，待实现

---

## 背景

当前 `rotation` 策略对所有 ETF 使用同一种得分（`momentum`、`slope_r2` 或 `momentum_quality`）。不同 ETF 类别（宽基、行业、红利、商品等）的收益特征差异较大，统一评分会导致某些类别在截面排名中系统性吃亏或占优。本设计引入按 ETF 类型构造不同动量得分的机制，同时保持对旧配置的向后兼容。

---

## 目标

1. 允许在 `config.yaml` 的 pool 项中为每个 ETF 标注 `type`。
2. 当策略开启 `adaptive_scoring` 时，按类型使用不同的动量算法。
3. 未识别类型退化为普通 `momentum` 得分，并打印警告。
4. 不开启 `adaptive_scoring` 时，行为与现有实现完全一致。

---

## 配置变更

### `PoolItem` 新增字段

```yaml
pool:
  - { code: "510300", name: "沪深300ETF", type: "宽基" }
  - { code: "510880", name: "红利ETF", type: "红利" }
  - { code: "518880", name: "黄金ETF", type: "商品" }
  - { code: "512690", name: "酒ETF", type: "行业股票" }
```

- `type` 为可选字符串。
- 当 `adaptive_scoring` 为 false 或未配置时，`type` 不参与任何逻辑，仅用于日志/报告展示（可选）。

### `StrategyConfig.params` 新增参数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `adaptive_scoring` | bool | false | 是否启用类型化自适应动量 |
| `benchmark` | str | None | 行业股票残差动量所需的基准名称，必须对应 pool 中某个 name |

示例：

```yaml
params:
  lookback: 22
  scoring: momentum
  adaptive_scoring: true
  benchmark: "沪深300ETF"
  top_n: 1
```

---

## 动量算法

所有算法均基于收盘价序列 `price_series: pd.Series`，返回单个浮点数得分。

### 1. 行业股票

**思想**：衡量 ETF 相对基准的超额收益，并用短期趋势斜率相对于中期趋势斜率的比率修正，奖励动量在加速的标的。

步骤：

1. 计算 ETF 与 benchmark 的 20 日滚动收益率。
2. 对 ETF 收益率序列和 benchmark 收益率序列做线性回归，取残差作为超额收益。
3. 计算 20 日价格回归斜率 `slope_20` 和 60 日价格回归斜率 `slope_60`。
4. 加速度系数 `accel = slope_20 / slope_60`（若 `slope_60 <= 0`，则 `accel = 1.0`）。
5. `score = 残差收益 × accel`。

### 2. 红利

**思想**：红利类资产波动低、收益平滑，用风险调整收益才能在截面排名中脱颖而出。

```
score = 过去40日累计收益 / 过去40日年化波动率
年化波动率 = 日收益标准差 × sqrt(252)
```

### 3. 商品

**思想**：商品适合捕捉中长周期趋势，同时用波动率惩罚高波动标的。

```
score = 过去60日累计收益 / 过去60日年化波动率
年化波动率 = 日收益标准差 × sqrt(252)
```

### 4. 宽基

**思想**：宽基指数适合看长期箱体突破，用当前价相对 252 日最高价的连续比率表示突破强度。

```
score = 当前收盘价 / 过去252日最高价
```

结果落在 `[0, 1]` 区间，越接近 1 表示越接近新高。

### 5. 未识别类型

当 `type` 缺失或不在内置类型表中时：

- 退化为 `momentum_score(price_series, lookback)`。
- 打印警告：`[自适应动量] 未识别类型 "{etf_type}"，退化为默认 momentum 得分`。

---

## 模块设计

### `core/scorer.py`

新增函数：

```python
def adaptive_momentum_score(
    price_series: pd.Series,
    etf_type: str | None,
    benchmark_series: pd.Series | None = None,
    lookback: int = 20,
) -> float:
    """根据 ETF 类型计算自适应动量得分。"""
```

新增辅助函数：

```python
def _residual_momentum_score(price_series, benchmark_series, lookback)
def _risk_adjusted_momentum_score(price_series, lookback)
def _trend_momentum_score(price_series, lookback)
def _breakout_score(price_series, lookback)
def _regression_slope(price_series, lookback)
```

### `strategy/rotation.py`

在 `run()` 中新增分支：

```python
if params.get("adaptive_scoring"):
    # 读取每个 name 的 type
    # 读取 benchmark
    # 对每个 name 调用 adaptive_momentum_score
    # 统一排名选 top_n
else:
    # 保持原有 scoring 逻辑
```

### `utils/config.py`

- `PoolItem` 增加 `type: str | None = None`。
- 当 `adaptive_scoring=true` 且 pool 中存在 `type: "行业股票"` 的 ETF 时，校验 `benchmark` 是否已配置且存在于 pool 的 name 中。

### `main.py`

- 将 `strategy.pool` 中的 `type` 信息传递给 `rotation.run()`，使策略内部可以按 name 查找 type。
- 保持 `weighted` 策略完全不变。

### `config.yaml`

更新当前启用的策略示例，加入 `type` 和 `adaptive_scoring`。

---

## 数据流

```
config.yaml
    ↓
utils/config.py (PoolItem.type, params.adaptive_scoring, params.benchmark)
    ↓
main.py (fetch_pool_data, run_strategy)
    ↓
strategy/rotation.py
    ↓
core/scorer.py (adaptive_momentum_score)
    ↓
core/report.py (报告与图表，逻辑不变)
```

---

## 错误处理

| 场景 | 行为 |
|---|---|
| `adaptive_scoring=true` 但 `benchmark` 缺失，且 pool 中有行业股票 | 配置加载时报错 |
| `benchmark` 配置的值不在 pool 的 name 中 | 配置加载时报错 |
| `type` 缺失 | 退化为 `momentum_score`，打印警告 |
| `type` 不在内置类型表中 | 退化为 `momentum_score`，打印警告 |
| 某类型所需数据不足（如宽基不足 252 日） | 返回 `np.nan`，该日不参与排名 |

---

## 测试与验证

1. 单元测试：对 `adaptive_momentum_score` 的每种类型构造已知价格序列，验证输出符号和相对大小。
2. 配置测试：验证开启 `adaptive_scoring` 后缺少 `benchmark` 会报错。
3. 回测测试：运行 `python main.py`，确认回测通过且输出报告正常。
4. 兼容性测试：关闭 `adaptive_scoring` 时，结果与修改前一致。

---

## 兼容性

- 旧配置不写 `type` 和 `adaptive_scoring` 时，行为不变。
- `weighted` 策略不受影响。
- 报告、图表、换仓记录等下游逻辑不变。

---

## 后续可扩展点

- 支持用户通过配置文件自定义类型到得分函数的映射。
- 支持更多内置类型（如债券、QDII、REITs）。
- 将 `benchmark` 扩展为多个基准，分别用于不同行业 ETF。
