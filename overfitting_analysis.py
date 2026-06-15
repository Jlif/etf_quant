#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
过拟合检测脚本：多角度评估 rotation 策略参数是否过度拟合历史数据。

支持的检测维度：
1. 参数敏感性 / 参数平原分析
2. 样本内 / 样本外测试
3. 蒙特卡洛置换检验
4. 收益贡献集中度检查

用法：
    python overfitting_analysis.py
    python overfitting_analysis.py --config my_config.yaml --mc-trials 1000
    python overfitting_analysis.py --skip-tests mc,contrib
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Literal

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from data_source import get_data_source
from main import fetch_pool_data, run_strategy
from param_sweep import compute_metrics
from utils import AppConfig, StrategyConfig, load_config

warnings.filterwarnings("ignore", category=RuntimeWarning)

if os.environ.get("DISPLAY") is None and os.name != "nt":
    matplotlib.use("Agg")

plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


@dataclass
class TestResult:
    name: str
    verdict: Literal["PASS", "WARN", "FAIL"]
    metrics: dict = field(default_factory=dict)
    message: str = ""


def _compute_sharpe(nav: pd.Series) -> float:
    """从净值序列计算年化夏普（无风险利率假设为 0）。"""
    daily_returns = nav.ffill().pct_change().dropna()
    if daily_returns.std() == 0 or len(daily_returns) < 2:
        return 0.0
    return (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)


def _format_params(params: dict) -> str:
    """把参数字典格式化为可读字符串。"""
    parts = []
    for k, v in sorted(params.items()):
        if isinstance(v, float):
            parts.append(f"{k}={v:.2%}" if v < 1 else f"{k}={v:.2f}")
        else:
            parts.append(f"{k}={v}")
    return ", ".join(parts)


def load_data_once(config_path: str):
    """加载配置并一次性获取数据。"""
    app_config = load_config(config_path)
    enabled_strategies = [s for s in app_config.strategies if s.enabled]
    if not enabled_strategies:
        raise ValueError("没有启用的策略")
    if len(enabled_strategies) > 1:
        print(f"检测到 {len(enabled_strategies)} 个启用策略，仅对第一个进行过拟合检测")

    strategy = enabled_strategies[0]
    if strategy.mode != "rotation":
        raise ValueError(f"仅支持 rotation 策略，当前为 {strategy.mode}")

    cache_dir = app_config.backtest.cache_dir
    provider = app_config.data_source.provider
    required_codes = {p.code for s in enabled_strategies for p in s.pool}
    missing_codes = [
        code
        for code in required_codes
        if not os.path.exists(os.path.join(cache_dir, f"{code}_{provider}.csv"))
    ]
    skip_test = not missing_codes

    data_source = get_data_source(name=provider, fallback=True, skip_test=skip_test)
    data = fetch_pool_data(strategy, app_config, data_source)
    return app_config, strategy, data_source, data


def run_with_params(
    strategy: StrategyConfig,
    app_config: AppConfig,
    data_source,
    data: dict,
    params: dict,
) -> pd.DataFrame | None:
    """用指定参数运行策略，返回结果 DataFrame；失败返回 None。"""
    base_params = dict(strategy.params)
    strategy.params = {**base_params, **params}
    try:
        result, _ = run_strategy(
            strategy, app_config, data_source, silent=True, data=data
        )
        strategy.params = base_params
        return result
    except Exception as e:
        print(f"    [参数组合跳过] {_format_params(params)}: {e}")
        strategy.params = base_params
        return None


def slice_data_by_date(data: dict, start_date, end_date) -> dict:
    """按日期范围切分 data dict。"""
    mask = (data["close"].index >= start_date) & (data["close"].index <= end_date)
    return {
        "close": data["close"].loc[mask].copy(),
        "open": data["open"].loc[mask].copy(),
        "high": data["high"].loc[mask].copy(),
        "low": data["low"].loc[mask].copy(),
    }


def metrics_from_result(result: pd.DataFrame | None) -> dict:
    """从结果中提取关键指标。"""
    if result is None or result.empty or "轮动策略净值" not in result.columns:
        return {}
    nav = result["轮动策略净值"]
    total_return, cagr, max_dd = compute_metrics(nav)
    return {
        "total_return": total_return,
        "cagr": cagr,
        "max_drawdown": max_dd,
        "sharpe": _compute_sharpe(nav),
        "final_nav": nav.iloc[-1],
        "n_days": len(nav),
    }


# ---------------------------------------------------------------------------
# Test 1: 参数敏感性 / 参数平原分析
# ---------------------------------------------------------------------------

def test_parameter_plain(
    strategy: StrategyConfig,
    app_config: AppConfig,
    data_source,
    data: dict,
    grid_size: int = 200,
) -> TestResult:
    """扫描参数邻域，判断当前参数是否位于尖锐峰值。"""
    current_params = dict(strategy.params)
    base = {
        "lookback": current_params.get("lookback", 20),
        "absolute_momentum_lookback": current_params.get(
            "absolute_momentum_lookback", 20
        ),
        "volatility_lookback": current_params.get("volatility_lookback", 20),
        "target_volatility": current_params.get("target_volatility", 0.10),
        "trailing_stop_pct": current_params.get("trailing_stop_pct", 0.10),
    }

    # 只扫描实际启用了的参数
    enabled_params = {}
    if "lookback" in current_params:
        enabled_params["lookback"] = base["lookback"]
    if current_params.get("absolute_momentum_filter"):
        enabled_params["absolute_momentum_lookback"] = base["absolute_momentum_lookback"]
    if current_params.get("target_volatility") is not None:
        enabled_params["volatility_lookback"] = base["volatility_lookback"]
        enabled_params["target_volatility"] = base["target_volatility"]
    if current_params.get("trailing_stop_pct") is not None:
        enabled_params["trailing_stop_pct"] = base["trailing_stop_pct"]

    if not enabled_params:
        return TestResult(
            name="参数敏感性分析",
            verdict="PASS",
            metrics={},
            message="未启用可调风控参数，无需参数平原分析",
        )

    records = []
    rng = np.random.default_rng(42)

    # 关键参数对（固定其余）做 2-D 网格扫描，用于热力图
    param_keys = list(enabled_params.keys())
    pair_results = []
    for i in range(len(param_keys)):
        for j in range(i + 1, len(param_keys)):
            k1, k2 = param_keys[i], param_keys[j]
            grid1 = _param_grid_values(k1, enabled_params[k1])
            grid2 = _param_grid_values(k2, enabled_params[k2])
            fixed = {k: v for k, v in enabled_params.items() if k not in (k1, k2)}
            for v1, v2 in itertools.product(grid1, grid2):
                params = {**fixed, k1: v1, k2: v2}
                result = run_with_params(
                    strategy, app_config, data_source, data, params
                )
                m = metrics_from_result(result)
                if m:
                    pair_results.append(
                        {**params, **m, "pair": f"{k1}_vs_{k2}"}
                    )

    # 局部随机邻域采样，计算 ruggedness
    for _ in range(grid_size):
        params = dict(enabled_params)
        for key in enabled_params:
            params[key] = _perturb_param(key, enabled_params[key], rng)
        result = run_with_params(strategy, app_config, data_source, data, params)
        m = metrics_from_result(result)
        if m:
            dist = _param_distance(params, enabled_params)
            records.append({**params, **m, "distance_from_current": dist})

    df = pd.DataFrame(records)
    pair_df = pd.DataFrame(pair_results)

    if df.empty:
        return TestResult(
            name="参数敏感性分析",
            verdict="WARN",
            metrics={},
            message="参数扫描未产生有效结果，无法判断",
        )

    sharpe_cv = df["sharpe"].std() / abs(df["sharpe"].mean()) if df["sharpe"].mean() != 0 else 999
    current_result = run_with_params(
        strategy, app_config, data_source, data, enabled_params
    )
    current_metrics = metrics_from_result(current_result)
    current_sharpe = current_metrics.get("sharpe", 0)
    rank_pct = (df["sharpe"] > current_sharpe).mean()

    # 保存结果
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(os.path.join(OUTPUT_DIR, "overfit_param_sensitivity.csv"), index=False, encoding="utf-8-sig")
    if not pair_df.empty:
        pair_df.to_csv(
            os.path.join(OUTPUT_DIR, "overfit_param_pairs.csv"), index=False, encoding="utf-8-sig"
        )

    # 绘图
    _plot_param_sensitivity(df, enabled_params)
    if not pair_df.empty:
        _plot_param_heatmaps(pair_df, enabled_params)

    metrics = {
        "sharpe_cv": sharpe_cv,
        "current_sharpe": current_sharpe,
        "rank_percentile": rank_pct,
        "n_combos": len(df),
    }

    if sharpe_cv > 0.6 or rank_pct > 0.95:
        verdict = "FAIL"
        message = f"参数 landscape 很陡峭 (CV={sharpe_cv:.2f})，当前参数位于前 {rank_pct:.1%}，过拟合风险高"
    elif sharpe_cv > 0.3 or rank_pct > 0.80:
        verdict = "WARN"
        message = f"参数略有崎岖 (CV={sharpe_cv:.2f})，当前参数位于前 {rank_pct:.1%}"
    else:
        verdict = "PASS"
        message = f"参数平原较平坦 (CV={sharpe_cv:.2f})，当前参数位于前 {rank_pct:.1%}"

    return TestResult(
        name="参数敏感性分析",
        verdict=verdict,
        metrics=metrics,
        message=message,
    )


def _param_grid_values(key: str, current):
    """生成关键参数对的网格值。"""
    if key in ("lookback", "absolute_momentum_lookback", "volatility_lookback"):
        vals = [max(3, int(current * f)) for f in [0.7, 0.85, 1.0, 1.15, 1.3]]
        return sorted(set(vals))
    else:
        return [current * f for f in [0.7, 0.85, 1.0, 1.15, 1.3]]


def _perturb_param(key: str, current, rng: np.random.Generator):
    """在参数当前值附近随机扰动。"""
    if key in ("lookback", "absolute_momentum_lookback", "volatility_lookback"):
        lo, hi = max(3, int(current * 0.6)), int(current * 1.4)
        return int(rng.integers(lo, hi + 1))
    else:
        return current * rng.uniform(0.6, 1.4)


def _param_distance(params: dict, current: dict) -> float:
    """归一化参数距离。"""
    dist = 0.0
    for key in current:
        if current[key] == 0:
            continue
        if key in ("lookback", "absolute_momentum_lookback", "volatility_lookback"):
            dist += abs(params[key] - current[key]) / current[key]
        else:
            dist += abs(params[key] - current[key]) / current[key]
    return dist / len(current) if current else 0.0


def _plot_param_sensitivity(df: pd.DataFrame, current_params: dict):
    """绘制每个参数与 Sharpe 的关系。"""
    param_keys = [k for k in current_params if k in df.columns]
    if not param_keys:
        return
    n = len(param_keys)
    fig, axes = plt.subplots(n, 1, figsize=(8, 3 * n), squeeze=False)
    for idx, key in enumerate(param_keys):
        ax = axes[idx, 0]
        ax.scatter(df[key], df["sharpe"], alpha=0.4, s=20)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.axvline(current_params[key], color="red", linestyle="--", linewidth=1.5, label="当前参数")
        ax.set_xlabel(key)
        ax.set_ylabel("Sharpe")
        ax.set_title(f"{key} 对 Sharpe 的影响")
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "overfit_param_sensitivity.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_param_heatmaps(pair_df: pd.DataFrame, current_params: dict):
    """绘制关键参数对的 2-D 热力图。"""
    pairs = pair_df["pair"].unique()[:4]  # 最多画 4 张
    n = len(pairs)
    if n == 0:
        return
    cols = 2
    rows = (n + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(12, 5 * rows), squeeze=False)
    for idx, pair in enumerate(pairs):
        ax = axes[idx // cols, idx % cols]
        sub = pair_df[pair_df["pair"] == pair]
        k1, k2 = pair.split("_vs_")
        pivot = sub.pivot_table(index=k1, columns=k2, values="sharpe", aggfunc="mean")
        im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", origin="lower")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_yticks(range(len(pivot.index)))
        ax.set_xticklabels([f"{v:.3g}" for v in pivot.columns], rotation=45)
        ax.set_yticklabels([f"{v:.3g}" for v in pivot.index])
        ax.set_xlabel(k2)
        ax.set_ylabel(k1)
        ax.set_title(f"{k1} vs {k2} 的 Sharpe 热力图")
        # 标记当前参数位置
        if current_params.get(k1) in pivot.index and current_params.get(k2) in pivot.columns:
            y = list(pivot.index).index(current_params[k1])
            x = list(pivot.columns).index(current_params[k2])
            ax.plot(x, y, "r*", markersize=15)
        plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "overfit_param_heatmaps.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Test 2: 样本内 / 样本外测试
# ---------------------------------------------------------------------------

def test_in_sample_out_of_sample(
    strategy: StrategyConfig,
    app_config: AppConfig,
    data_source,
    data: dict,
    train_ratio: float = 0.7,
) -> TestResult:
    """按时间切分数据，训练集选最优参数，测试集评估。"""
    dates = data["close"].index
    split_idx = int(len(dates) * train_ratio)
    train_start, train_end = dates[0], dates[split_idx - 1]
    test_start, test_end = dates[split_idx], dates[-1]

    train_data = slice_data_by_date(data, train_start, train_end)
    test_data = slice_data_by_date(data, test_start, test_end)

    current_params = dict(strategy.params)
    enabled_params = _extract_tunable_params(current_params)

    # 在训练集上扫描参数组合（简化网格）
    param_grids = _build_param_grids(enabled_params, coarse=True)
    records = []
    best_params = None
    best_sharpe = -np.inf

    for params in param_grids:
        result = run_with_params(strategy, app_config, data_source, train_data, params)
        m = metrics_from_result(result)
        if m:
            records.append({"period": "train", **params, **m})
            if m["sharpe"] > best_sharpe:
                best_sharpe = m["sharpe"]
                best_params = params

    if best_params is None:
        return TestResult(
            name="样本内外测试",
            verdict="WARN",
            metrics={},
            message="训练集上未找到有效参数组合",
        )

    # 当前参数在训练集和测试集上的表现
    current_train = metrics_from_result(
        run_with_params(strategy, app_config, data_source, train_data, current_params)
    )
    current_test = metrics_from_result(
        run_with_params(strategy, app_config, data_source, test_data, current_params)
    )

    # 最优参数在训练集和测试集上的表现
    best_train = metrics_from_result(
        run_with_params(strategy, app_config, data_source, train_data, best_params)
    )
    best_test = metrics_from_result(
        run_with_params(strategy, app_config, data_source, test_data, best_params)
    )

    records.append({"period": "test", **best_params, **best_test})
    for prefix, params in [("current", current_params), ("best_is", best_params)]:
        for period, metrics_dict in [("train", best_train if prefix == "best_is" else current_train),
                                      ("test", best_test if prefix == "best_is" else current_test)]:
            records.append({
                "param_set": prefix,
                "period": period,
                **params,
                **metrics_dict,
            })

    df = pd.DataFrame(records)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(os.path.join(OUTPUT_DIR, "overfit_is_os.csv"), index=False, encoding="utf-8-sig")

    # 计算当前参数的样本外衰减
    is_sharpe = current_train.get("sharpe", 0)
    os_sharpe = current_test.get("sharpe", 0)
    is_cagr = current_train.get("cagr", 0)
    os_cagr = current_test.get("cagr", 0)

    sharpe_deg = 1 - os_sharpe / is_sharpe if is_sharpe != 0 else 0
    cagr_deg = 1 - os_cagr / is_cagr if is_cagr != 0 else 0

    metrics = {
        "train_start": train_start.date(),
        "train_end": train_end.date(),
        "test_start": test_start.date(),
        "test_end": test_end.date(),
        "is_sharpe": is_sharpe,
        "os_sharpe": os_sharpe,
        "sharpe_degradation": sharpe_deg,
        "cagr_degradation": cagr_deg,
        "best_is_sharpe": best_sharpe,
        "best_is_params": best_params,
    }

    if sharpe_deg > 0.60 or os_sharpe <= 0:
        verdict = "FAIL"
        message = f"样本外 Sharpe 大幅衰减：IS={is_sharpe:.2f}, OS={os_sharpe:.2f}, 衰减 {sharpe_deg:.1%}"
    elif sharpe_deg > 0.30:
        verdict = "WARN"
        message = f"样本外 Sharpe 有所衰减：IS={is_sharpe:.2f}, OS={os_sharpe:.2f}, 衰减 {sharpe_deg:.1%}"
    else:
        verdict = "PASS"
        message = f"样本外表现稳健：IS={is_sharpe:.2f}, OS={os_sharpe:.2f}, 衰减 {sharpe_deg:.1%}"

    # 绘图
    _plot_is_os_scatter(records, current_params, best_params)

    return TestResult(
        name="样本内外测试",
        verdict=verdict,
        metrics=metrics,
        message=message,
    )


def _extract_tunable_params(params: dict) -> dict:
    """提取实际可调的参数子集。"""
    enabled = {"lookback": params.get("lookback", 20)}
    if params.get("absolute_momentum_filter"):
        enabled["absolute_momentum_lookback"] = params.get(
            "absolute_momentum_lookback", 20
        )
    if params.get("target_volatility") is not None:
        enabled["volatility_lookback"] = params.get("volatility_lookback", 20)
        enabled["target_volatility"] = params.get("target_volatility", 0.10)
    if params.get("trailing_stop_pct") is not None:
        enabled["trailing_stop_pct"] = params.get("trailing_stop_pct", 0.10)
    return enabled


def _build_param_grids(enabled_params: dict, coarse: bool = False) -> list[dict]:
    """生成参数扫描网格。"""
    grids = {}
    for key, current in enabled_params.items():
        if key in ("lookback", "absolute_momentum_lookback", "volatility_lookback"):
            if coarse:
                grids[key] = sorted({max(3, int(current * f)) for f in [0.8, 1.0, 1.2]})
            else:
                grids[key] = sorted({max(3, int(current * f)) for f in [0.7, 0.85, 1.0, 1.15, 1.3]})
        else:
            if coarse:
                grids[key] = [current * f for f in [0.8, 1.0, 1.2]]
            else:
                grids[key] = [current * f for f in [0.7, 0.85, 1.0, 1.15, 1.3]]

    keys = list(grids.keys())
    values = [grids[k] for k in keys]
    combos = []
    for combo in itertools.product(*values):
        combos.append(dict(zip(keys, combo)))
    return combos


def _plot_is_os_scatter(records, current_params, best_params):
    """绘制样本内 vs 样本外 Sharpe 散点图。"""
    train_rows = [r for r in records if r.get("period") == "train" and "sharpe" in r]
    test_rows = [r for r in records if r.get("period") == "test" and "sharpe" in r]
    if len(train_rows) < 2:
        return

    # 按参数组合匹配 train/test
    by_params = {}
    for r in train_rows + test_rows:
        key = tuple(sorted((k, r[k]) for k in current_params if k in r))
        by_params.setdefault(key, {}).setdefault(r["period"], r)

    is_sharpes, os_sharpes = [], []
    for key, pair in by_params.items():
        if "train" in pair and "test" in pair:
            is_sharpes.append(pair["train"]["sharpe"])
            os_sharpes.append(pair["test"]["sharpe"])

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(is_sharpes, os_sharpes, alpha=0.5, s=40, label="参数组合")
    ax.plot([-2, 3], [-2, 3], "k--", linewidth=1, label="IS=OS")

    # 当前参数
    current_key = tuple(sorted((k, current_params[k]) for k in current_params))
    if current_key in by_params:
        cp = by_params[current_key]
        if "train" in cp and "test" in cp:
            ax.scatter(
                cp["train"]["sharpe"],
                cp["test"]["sharpe"],
                color="red",
                s=120,
                marker="*",
                label="当前参数",
            )

    # 最优 IS 参数
    best_key = tuple(sorted((k, best_params[k]) for k in best_params))
    if best_key in by_params:
        bp = by_params[best_key]
        if "train" in bp and "test" in bp:
            ax.scatter(
                bp["train"]["sharpe"],
                bp["test"]["sharpe"],
                color="orange",
                s=120,
                marker="s",
                label="最优 IS 参数",
            )

    ax.set_xlabel("样本内 Sharpe")
    ax.set_ylabel("样本外 Sharpe")
    ax.set_title("样本内 vs 样本外 Sharpe")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "overfit_is_os_scatter.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Test 3: 蒙特卡洛置换检验
# ---------------------------------------------------------------------------

def generate_null_data(
    data: dict,
    method: Literal["shuffle", "block"] = "shuffle",
    block_size: int | None = None,
    seed: int | None = None,
) -> dict:
    """生成打乱后的价格数据。"""
    rng = np.random.default_rng(seed)
    null_data = {}
    for key in data:
        null_data[key] = pd.DataFrame(index=data[key].index, columns=data[key].columns)

    for name in data["close"].columns:
        close = data["close"][name]
        returns = close.ffill().pct_change().dropna().values
        n = len(close)

        if method == "shuffle":
            perm = rng.permutation(len(returns))
            shuffled_returns = returns[perm]
        else:  # block
            bs = block_size or max(int(current_params.get("lookback", 20)), 20)
            perm, shuffled_returns = _block_bootstrap(returns, bs, rng)

        new_close = [close.iloc[0]]
        for r in shuffled_returns:
            new_close.append(new_close[-1] * (1 + r))
        null_data["close"][name] = new_close

        # 保持原始 OHLC / Close 比例关系
        for key in ("open", "high", "low"):
            ratio = (data[key][name] / close).iloc[1:].values
            null_data[key][name] = [data[key][name].iloc[0]] + list(
                np.array(new_close[1:]) * ratio[perm]
            )

    return null_data


def _block_bootstrap(returns: np.ndarray, block_size: int, rng: np.random.Generator):
    """循环块 Bootstrap。"""
    n = len(returns)
    n_blocks = int(np.ceil(n / block_size))
    starts = rng.integers(0, n, size=n_blocks)
    indices = []
    for s in starts:
        indices.extend([(s + i) % n for i in range(block_size)])
    indices = np.array(indices[:n])
    return indices, returns[indices]


def test_monte_carlo(
    strategy: StrategyConfig,
    app_config: AppConfig,
    data_source,
    data: dict,
    n_trials: int = 500,
    method: Literal["shuffle", "block"] = "shuffle",
    block_size: int | None = None,
) -> TestResult:
    """蒙特卡洛置换检验：打乱收益率后比较真实策略表现。"""
    current_params = dict(strategy.params)
    if method == "block" and block_size is None:
        block_size = max(int(current_params.get("lookback", 20)), 20)

    real_result = run_with_params(strategy, app_config, data_source, data, current_params)
    real_metrics = metrics_from_result(real_result)
    real_sharpe = real_metrics.get("sharpe", 0)
    real_cagr = real_metrics.get("cagr", 0)

    records = []
    count_better_sharpe = 0
    count_better_cagr = 0

    for trial in range(n_trials):
        null_data = generate_null_data(data, method=method, block_size=block_size, seed=trial)
        result = run_with_params(strategy, app_config, data_source, null_data, current_params)
        m = metrics_from_result(result)
        if not m:
            continue
        records.append({"trial": trial, **m})
        if m["sharpe"] >= real_sharpe:
            count_better_sharpe += 1
        if m["cagr"] >= real_cagr:
            count_better_cagr += 1

        if (trial + 1) % 100 == 0 or trial == n_trials - 1:
            print(f"      MC 进度: {trial + 1}/{n_trials}")

    df = pd.DataFrame(records)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(os.path.join(OUTPUT_DIR, "overfit_monte_carlo.csv"), index=False, encoding="utf-8-sig")

    sharpe_p = (count_better_sharpe + 1) / (len(records) + 1)
    cagr_p = (count_better_cagr + 1) / (len(records) + 1)

    metrics = {
        "real_sharpe": real_sharpe,
        "real_cagr": real_cagr,
        "sharpe_p_value": sharpe_p,
        "cagr_p_value": cagr_p,
        "n_trials": len(records),
        "method": method,
    }

    if sharpe_p < 0.05:
        verdict = "PASS"
        message = f"真实 Sharpe 显著优于随机分布 (p={sharpe_p:.3f})"
    elif sharpe_p < 0.10:
        verdict = "WARN"
        message = f"真实 Sharpe 勉强优于随机分布 (p={sharpe_p:.3f})"
    else:
        verdict = "FAIL"
        message = f"真实 Sharpe 与随机分布无显著差异 (p={sharpe_p:.3f})，可能过拟合"

    # 绘图
    _plot_mc_distribution(df["sharpe"].dropna(), real_sharpe, "Sharpe")
    _plot_mc_distribution(df["cagr"].dropna(), real_cagr, "CAGR")

    return TestResult(
        name="蒙特卡洛置换检验",
        verdict=verdict,
        metrics=metrics,
        message=message,
    )


def _plot_mc_distribution(null_values: pd.Series, real_value: float, metric_name: str):
    """绘制蒙特卡洛零分布与真实值。"""
    if null_values.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(null_values, bins=40, color="steelblue", edgecolor="white", alpha=0.7)
    ax.axvline(real_value, color="red", linestyle="--", linewidth=2, label=f"真实 {metric_name}: {real_value:.3f}")
    ax.set_xlabel(metric_name)
    ax.set_ylabel("频数")
    ax.set_title(f"蒙特卡洛置换检验：{metric_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, f"overfit_monte_carlo_{metric_name.lower()}.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


# ---------------------------------------------------------------------------
# Test 4: 收益贡献集中度检查
# ---------------------------------------------------------------------------

def test_contribution_concentration(
    result: pd.DataFrame, name_list: list[str]
) -> TestResult:
    """检查收益是否过度集中于少数 ETF 或少数交易日。"""
    contrib_cols = [c for c in result.columns if c.startswith("贡献_日收益_")]
    if not contrib_cols:
        return TestResult(
            name="收益贡献集中度",
            verdict="WARN",
            metrics={},
            message="结果中缺少贡献列，无法进行集中度分析",
        )

    records = []
    total_contribution = 0.0
    for name in name_list:
        col = f"贡献_日收益_{name}"
        if col not in result.columns:
            continue
        contrib = result[col].sum()
        hold_days = int((result[f"权重_{name}"] > 0).sum())
        total_contribution += contrib
        records.append(
            {
                "etf_name": name,
                "total_contribution": contrib,
                "hold_days": hold_days,
                "top_day_contribution": result[col].max(),
            }
        )

    df = pd.DataFrame(records)
    if total_contribution == 0 or df.empty:
        return TestResult(
            name="收益贡献集中度",
            verdict="WARN",
            metrics={},
            message="总贡献为 0，无法计算集中度",
        )

    df["contribution_pct"] = df["total_contribution"] / total_contribution
    df = df.sort_values("contribution_pct", ascending=False)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(os.path.join(OUTPUT_DIR, "overfit_contribution.csv"), index=False, encoding="utf-8-sig")

    top2_ratio = df["contribution_pct"].head(2).sum()
    top1_ratio = df["contribution_pct"].iloc[0]

    # 日贡献 Gini 系数
    daily_total = result[[c for c in result.columns if c.startswith("贡献_日收益_")]].sum(axis=1)
    daily_total = daily_total[daily_total != 0].abs().sort_values()
    n = len(daily_total)
    gini = 0.0
    if n > 0:
        cumsum = daily_total.cumsum().values
        gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n if cumsum[-1] != 0 else 0

    metrics = {
        "top1_ratio": top1_ratio,
        "top2_ratio": top2_ratio,
        "gini": gini,
        "total_contribution": total_contribution,
    }

    if top2_ratio > 0.85 or top1_ratio > 0.70:
        verdict = "FAIL"
        message = f"收益高度集中：Top-1 占比 {top1_ratio:.1%}，Top-2 占比 {top2_ratio:.1%}"
    elif top2_ratio > 0.70 or top1_ratio > 0.50:
        verdict = "WARN"
        message = f"收益相对集中：Top-1 占比 {top1_ratio:.1%}，Top-2 占比 {top2_ratio:.1%}"
    else:
        verdict = "PASS"
        message = f"收益来源较分散：Top-1 占比 {top1_ratio:.1%}，Top-2 占比 {top2_ratio:.1%}"

    # 绘图
    _plot_contribution_pie(df)

    return TestResult(
        name="收益贡献集中度",
        verdict=verdict,
        metrics=metrics,
        message=message,
    )


def _plot_contribution_pie(df: pd.DataFrame):
    """绘制收益贡献饼图。"""
    fig, ax = plt.subplots(figsize=(8, 8))
    labels = df["etf_name"].tolist()
    sizes = df["total_contribution"].clip(lower=0).tolist()
    if sum(sizes) == 0:
        plt.close(fig)
        return
    ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
    ax.set_title("各 ETF 收益贡献占比")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "overfit_contribution_pie.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def print_report(results: list[TestResult], current_params: dict):
    """打印最终检测报告。"""
    print("\n" + "=" * 60)
    print("【过拟合检测报告】")
    print("=" * 60)
    print(f"当前参数:")
    for line in _format_params(current_params).split(", "):
        print(f"  {line}")
    print("-" * 60)

    for idx, r in enumerate(results, 1):
        symbol = {"PASS": "✓", "WARN": "△", "FAIL": "✗"}.get(r.verdict, "?")
        print(f"[{idx}] {r.name:<18} ... {r.verdict}  {symbol}")
        print(f"    {r.message}")
        if r.metrics:
            for k, v in list(r.metrics.items())[:4]:
                if isinstance(v, float):
                    print(f"    · {k}: {v:.4f}")
                else:
                    print(f"    · {k}: {v}")

    counts = {"PASS": 0, "WARN": 0, "FAIL": 0}
    for r in results:
        counts[r.verdict] = counts.get(r.verdict, 0) + 1

    print("-" * 60)
    if counts["FAIL"] > 0:
        overall = "FAIL"
        overall_msg = f"存在 {counts['FAIL']} 项 FAIL，参数过拟合风险较高"
    elif counts["WARN"] > 0:
        overall = "WARN"
        overall_msg = f"{counts['WARN']} 项 WARN，建议进一步检查"
    else:
        overall = "PASS"
        overall_msg = "所有检测通过，参数相对稳健"

    print(f"综合 verdict: {overall} — {overall_msg}")
    print(f"详细结果已保存至 {OUTPUT_DIR}/overfit_*.csv")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="ETF 轮动策略过拟合检测")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    parser.add_argument(
        "--train-ratio", type=float, default=0.7, help="样本内数据占比"
    )
    parser.add_argument("--mc-trials", type=int, default=500, help="蒙特卡洛试验次数")
    parser.add_argument(
        "--mc-method", choices=["shuffle", "block"], default="shuffle", help="MC 方法"
    )
    parser.add_argument("--block-size", type=int, default=None, help="块 Bootstrap 块大小")
    parser.add_argument(
        "--param-grid-size", type=int, default=200, help="参数敏感性随机采样数"
    )
    parser.add_argument(
        "--skip-tests",
        default="",
        help="跳过的测试，逗号分隔：param,isos,mc,contrib",
    )
    parser.add_argument("--plot", action="store_true", default=True, help="生成图表")
    parser.add_argument("--no-plot", dest="plot", action="store_false", help="不生成图表")
    args = parser.parse_args()

    skip = {s.strip().lower() for s in args.skip_tests.split(",") if s.strip()}

    print("=" * 60)
    print("ETF 轮动策略过拟合检测")
    print("=" * 60)

    app_config, strategy, data_source, data = load_data_once(args.config)
    current_params = dict(strategy.params)

    results: list[TestResult] = []

    if "param" not in skip:
        print("\n[1/4] 参数敏感性分析...")
        results.append(
            test_parameter_plain(
                strategy, app_config, data_source, data, grid_size=args.param_grid_size
            )
        )

    if "isos" not in skip:
        print("\n[2/4] 样本内外测试...")
        results.append(
            test_in_sample_out_of_sample(
                strategy, app_config, data_source, data, train_ratio=args.train_ratio
            )
        )

    if "mc" not in skip:
        print(f"\n[3/4] 蒙特卡洛置换检验（{args.mc_trials} 次）...")
        results.append(
            test_monte_carlo(
                strategy,
                app_config,
                data_source,
                data,
                n_trials=args.mc_trials,
                method=args.mc_method,
                block_size=args.block_size,
            )
        )

    if "contrib" not in skip:
        print("\n[4/4] 收益贡献集中度检查...")
        real_result = run_with_params(
            strategy, app_config, data_source, data, current_params
        )
        if real_result is not None:
            name_list = data["close"].columns.tolist()
            results.append(test_contribution_concentration(real_result, name_list))
        else:
            results.append(
                TestResult(
                    name="收益贡献集中度",
                    verdict="WARN",
                    message="当前参数运行失败，无法检查贡献集中度",
                )
            )

    print_report(results, current_params)


if __name__ == "__main__":
    main()
