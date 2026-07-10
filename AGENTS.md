# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## 项目概述

这是一个基于 Python 的 ETF 量化回测系统，支持两种策略模式：
- **rotation（轮动策略）**：每日根据动量或斜率*R² 得分选出最强 top_n 个 ETF，等权分配
- **weighted（加权组合策略）**：按配置权重持有，定期再平衡

## 常用命令

```bash
# 安装依赖
pip install -r requirements.txt

# 运行回测（默认使用 config.yaml）
python main.py

# 使用自定义配置文件
python main.py --config my_config.yaml
```

## 架构说明

### 数据流

```
config.yaml → utils/config.py (AppConfig) → main.py
                                              ↓
                    data_source/ (akshare | yfinance) → data_cache/ (CSV 缓存)
                                              ↓
                    strategy/ (rotation | weighted) → core/report.py → output/ (图表 + HTML 报告)
```

### 核心模块

- **`data_source/`**：数据源抽象层
  - `BaseDataSource` 定义 `fetch(code, start, end)` 接口，返回以日期为索引、code 为列名的收盘价 DataFrame
  - `get_data_source()` 支持自动探测和 fallback（默认 akshare → yfinance），会先用 510300 做连通性测试
  - `YFinanceDataSource` 对深市 ETF（如 159915）自动加 `.SZ` 后缀，其余加 `.SS`

- **`strategy/`**：策略实现
  - `rotation.run(data, name_list, params)`：每日计算得分（`momentum` 或 `slope_r2`），选 top_n 等权，持仓权重前移1天（T日收益由T-1日持仓产生）
    - `dynamic_pool`（可选，默认 false）：为 true 时，每日仅将已满足预热窗口的 ETF 纳入轮动候选池；当可选 ETF 不足 `top_n` 时，剩余仓位优先分配给已就绪的 `safe_haven`，否则空仓。整体回测起始日不再被最晚预热 ETF 拖后。
  - `weighted.run(data, name_list, weights, params)`：权重以百分数传入（如 25 表示 25%），`rebalance_freq=1` 为每日再平衡，>1 时模拟权重漂移后再平衡

- **`core/`**：评分与报告
  - `scorer.py`：`momentum_score`（N日涨幅）和 `slope_r2_score`（线性回归斜率 × R²）
  - `report.py`：基于 quantstats 生成 HTML 报告（含中文指标翻译），并输出 PNG 净值曲线到 `output/` 目录

- **`utils/config.py`**：配置加载与校验
  - `load_config()` 从 YAML 解析为 `AppConfig` dataclass
  - `weighted` 模式会自动校验权重之和是否为 100%
  - 策略级 `start_date` 优先于全局 `backtest.start_date`
  - 可选 `backtest.end_date`（策略级 `end_date` 可覆盖）作为回测截止日，留空则一直用到最新数据，便于"训练期调参、留样期验证"

### 关键设计

- **数据缓存**：下载的 ETF 数据按 `data_cache/{code}_{provider}.csv` 缓存，不会自动过期，如需更新需手动删除
- **回测截止日**：`backtest.end_date`（可选，YYYYMMDD）为全局回测截止日；策略级 `end_date` 可覆盖。`fetch_pool_data()` 在显式 `cutoff_date` 缺失时回退到该值，按 `<= end_date` 截断数据
- **复权跳空修正**：`main.py` 中的 `detect_and_fix_price_jumps()` 会检测日收益率绝对值超过 30% 的异常点（yfinance 对国内 ETF 复权偶尔出错），通过整体缩放前期价格修复
- **策略实际起始日**：取 `max(配置起始日, 所有 ETF 中最晚的数据起始日)`，并在日志中提示
- **输出目录**：每次运行会清空 `output/` 目录，生成 HTML 报告和 PNG 图表

### 配置示例（config.yaml）

```yaml
data_source:
  provider: akshare       # akshare | yfinance

backtest:
  start_date: "20160701"
  cache_dir: "./data_cache"

strategies:
  - name: "动量轮动策略"
    enabled: true
    mode: rotation
    pool:
      - { code: "512040", name: "价值100ETF富国" }
      - { code: "510300", name: "沪深300ETF" }
    params:
      lookback: 20
      scoring: momentum      # momentum | slope_r2
      top_n: 1

  - name: "再平衡组合"
    enabled: true
    mode: weighted
    pool:
      - { code: "510880", name: "红利ETF", weight: 20 }
      - { code: "512040", name: "价值100ETF富国", weight: 50 }
    params:
      rebalance_freq: 60
```

## 开发注意事项

- 数据源依赖 akshare/yfinance，国内网络建议优先使用 akshare
- matplotlib 在无图形界面环境自动使用 Agg 后端，中文字体回退顺序：SimHei → Arial Unicode MS → DejaVu Sans
- 新增数据源需继承 `BaseDataSource` 并在 `data_source/__init__.py` 的 `BUILTIN_SOURCES` 中注册
- 新增策略模式需在 `strategy/` 下实现，并在 `main.py` 的 `run_strategy()` 中添加分支
