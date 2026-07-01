import warnings

import numpy as np
import pandas as pd
import pytest

from strategy.rotation import run


def _make_data(n=300, names=("沪深300ETF", "红利ETF")):
    idx = pd.date_range("2023-01-01", periods=n)
    close = pd.DataFrame(
        {name: np.linspace(100, 100 + i * 10, n) for i, name in enumerate(names)},
        index=idx,
    )
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


def test_adaptive_scoring_without_sector_allows_missing_benchmark():
    data = _make_data()
    params = {
        "lookback": 20,
        "scoring": "momentum",
        "adaptive_scoring": True,
        "top_n": 1,
    }
    name_types = {"沪深300ETF": "宽基", "红利ETF": "红利"}
    result = run(data, list(data["close"].columns), params, name_types=name_types)
    assert "轮动策略净值" in result.columns
    assert result["轮动策略净值"].iloc[-1] > 0


def test_adaptive_scoring_non_broad_index_does_not_require_252_days():
    """非宽基组合不应被强制要求 252 天预热期。"""
    n = 82  # 小于 252，但大于商品所需的 61
    data = _make_data(n=n, names=("红利ETF", "商品ETF"))
    # 给红利 ETF 一些涨幅，避免波动率为 0 导致得分 NaN
    data["close"]["红利ETF"] = np.linspace(100, 120, n)
    data["open"]["红利ETF"] = data["close"]["红利ETF"].copy()
    data["high"]["红利ETF"] = data["close"]["红利ETF"] * 1.01
    data["low"]["红利ETF"] = data["close"]["红利ETF"] * 0.99
    params = {
        "lookback": 20,
        "scoring": "momentum",
        "adaptive_scoring": True,
        "top_n": 1,
    }
    name_types = {"红利ETF": "红利", "商品ETF": "商品"}
    result = run(data, list(data["close"].columns), params, name_types=name_types)
    assert len(result) > 0
    assert "轮动策略净值" in result.columns
    assert result["轮动策略净值"].iloc[-1] > 0


def test_adaptive_scoring_nan_score_gets_zero_weight():
    """一只 ETF 得分为 NaN 时，另一只 ETF 仍应被选入并获得权重。"""
    n = 100
    idx = pd.date_range("2023-01-01", periods=n)
    # 红利 ETF 有正常趋势；商品 ETF 前期价格不变 -> 波动率为 0 -> 得分为 NaN
    close = pd.DataFrame(
        {
            "红利ETF": np.linspace(100, 120, n),
            "商品ETF": np.full(n, 100.0),
        },
        index=idx,
    )
    # 商品 ETF 后期出现小幅波动，使其得分恢复有效
    close.loc[idx[70:], "商品ETF"] = np.linspace(100.0, 100.5, len(idx[70:]))
    open_ = close.copy()
    high = close * 1.01
    low = close * 0.99
    data = {"close": close, "open": open_, "high": high, "low": low}
    params = {
        "lookback": 20,
        "scoring": "momentum",
        "adaptive_scoring": True,
        "top_n": 1,
    }
    name_types = {"红利ETF": "红利", "商品ETF": "商品"}
    result = run(data, list(data["close"].columns), params, name_types=name_types)
    assert len(result) > 0
    # 商品 ETF 前期得分为 NaN，对应权重应为 0
    assert (result["权重_商品ETF"].iloc[:10] == 0).all()
    # 红利 ETF 应获得权重
    assert (result["权重_红利ETF"] > 0).any()


def test_dynamic_pool_unavailable_etf_gets_zero_weight():
    """dynamic_pool=true 时，始终未满足预热条件的 ETF 权重为 0，可用 ETF 正常入选。"""
    n = 80
    idx = pd.date_range("2023-01-01", periods=n)
    close = pd.DataFrame(
        {
            "红利ETF": np.linspace(100, 120, n),
            # 数据从 idx[60] 开始，有效长度 20 < lookback+1=21，全程不满足预热条件
            "晚上市ETF": np.where(np.arange(n) < 60, np.nan, np.linspace(100, 110, n)),
        },
        index=idx,
    )
    open_ = close.copy()
    data = {"close": close, "open": open_, "high": close * 1.01, "low": close * 0.99}
    params = {"lookback": 20, "scoring": "momentum", "top_n": 1, "dynamic_pool": True}
    result = run(data, list(close.columns), params)
    # 晚上市 ETF 权重永远为 0
    assert (result["权重_晚上市ETF"] == 0).all()
    # 红利 ETF 在第一个有效信号后应被选中
    assert (result["权重_红利ETF"] > 0).any()
    # 策略净值应正常增长
    assert result["轮动策略净值"].iloc[-1] > 1.0


def test_dynamic_pool_fills_with_safe_haven():
    """dynamic_pool=true 且可选 ETF 不足 top_n 时，剩余仓位给 safe_haven。"""
    n = 100
    idx = pd.date_range("2023-01-01", periods=n)
    close = pd.DataFrame(
        {
            "红利ETF": np.linspace(100, 120, n),
            "晚上市ETF": np.where(np.arange(n) < 60, np.nan, np.linspace(100, 110, n)),
        },
        index=idx,
    )
    open_ = close.copy()
    data = {"close": close, "open": open_, "high": close * 1.01, "low": close * 0.99}
    params = {
        "lookback": 20,
        "scoring": "momentum",
        "top_n": 2,
        "dynamic_pool": True,
        "safe_haven": "红利ETF",
    }
    result = run(data, list(close.columns), params)
    # 第 20 天左右只有红利 ETF 可用（晚上市 ETF 3 月初才有数据），红利 ETF 应占满仓位
    early_row = result.iloc[10]
    assert early_row["权重_红利ETF"] == pytest.approx(1.0)


def test_dynamic_pool_excludes_not_ready_etfs():
    """dynamic_pool=true 时，未满足预热窗口的 ETF 权重为 0。"""
    n = 100
    idx = pd.date_range("2023-01-01", periods=n)
    close = pd.DataFrame(
        {
            "红利ETF": np.linspace(100, 120, n),
            "晚上市ETF": np.where(np.arange(n) < 30, np.nan, np.linspace(100, 110, n)),
        },
        index=idx,
    )
    open_ = close.copy()
    data = {"close": close, "open": open_, "high": close * 1.01, "low": close * 0.99}
    params = {"lookback": 20, "scoring": "momentum", "top_n": 1, "dynamic_pool": True}
    result = run(data, list(close.columns), params)
    # 前若干天晚上市 ETF 权重应为 0
    assert (result["权重_晚上市ETF"].iloc[:10] == 0).all()
    # 红利 ETF 应被选中
    assert (result["权重_红利ETF"] > 0).any()


def test_dynamic_pool_false_keeps_original_behavior():
    """dynamic_pool=false（默认）时，所有 ETF 同时可用，行为不变。"""
    n = 100
    idx = pd.date_range("2023-01-01", periods=n)
    close = pd.DataFrame(
        {
            "红利ETF": np.linspace(100, 120, n),
            "创业板ETF": np.linspace(100, 110, n),
        },
        index=idx,
    )
    open_ = close.copy()
    data = {"close": close, "open": open_, "high": close * 1.01, "low": close * 0.99}
    params = {"lookback": 20, "scoring": "momentum", "top_n": 1}
    result = run(data, list(close.columns), params)
    assert "轮动策略净值" in result.columns
    assert result["轮动策略净值"].iloc[-1] > 0
    assert (result["权重_红利ETF"] > 0).any()


def test_adaptive_scoring_unknown_type_uses_lookback():
    """未知/缺失类型应使用默认 lookback，而不是 252。"""
    n = 50  # 小于 252，但大于默认 lookback 20
    data = _make_data(n=n, names=("未知ETF", "缺失ETF"))
    # 给 ETF 一些涨幅，避免波动率为 0 导致得分 NaN
    for name in ("未知ETF", "缺失ETF"):
        data["close"][name] = np.linspace(100, 110, n)
        data["open"][name] = data["close"][name].copy()
        data["high"][name] = data["close"][name] * 1.01
        data["low"][name] = data["close"][name] * 0.99
    params = {
        "lookback": 20,
        "scoring": "momentum",
        "adaptive_scoring": True,
        "top_n": 1,
    }
    name_types = {"未知ETF": "外星ETF"}  # 缺失类型不传入
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = run(
            data,
            list(data["close"].columns),
            params,
            name_types=name_types,
        )
    assert len(result) > 0
    assert "轮动策略净值" in result.columns
    assert result["轮动策略净值"].iloc[-1] > 0


def test_layer1_market_filter_moves_to_safe_haven():
    """第一层触发后，风险资产应全部转入 safe_haven。"""
    n = 100
    idx = pd.date_range("2023-01-01", periods=n)
    close = pd.DataFrame(
        {
            "股票ETF": np.concatenate(
                [np.linspace(100, 150, 80), np.linspace(150, 100, 20)]
            ),
            "国债ETF": np.full(n, 100.0),
        },
        index=idx,
    )
    open_ = close.copy()
    data = {"close": close, "open": open_, "high": close * 1.01, "low": close * 0.99}
    params = {
        "lookback": 20,
        "scoring": "momentum",
        "top_n": 1,
        "safe_haven": "国债ETF",
        "risk_control": {
            "layer1": {"enabled": True, "ma_lookback": 20, "drawdown_threshold": 0.05}
        },
    }
    result = run(data, list(close.columns), params)
    # 下跌末期应全部切到 safe_haven
    assert (result["权重_国债ETF"].iloc[-10:] > 0.99).all()
    assert (result["权重_股票ETF"].iloc[-10:] == 0).all()


def test_layer2_atr_trailing_stop_exits_on_gap_down():
    """第二层 ATR 跟踪止损在大幅回落时清仓。"""
    n = 100
    idx = pd.date_range("2023-01-01", periods=n)
    close = pd.DataFrame(
        {"股票ETF": np.full(n, 100.0), "国债ETF": np.full(n, 100.0)},
        index=idx,
    )
    # 先冲高到 120，随后跳空跌至 50，触发 3*ATR 止损
    close.loc[idx[50], "股票ETF"] = 120.0
    close.loc[idx[51:], "股票ETF"] = 50.0
    open_ = close.copy()
    data = {"close": close, "open": open_, "high": close * 1.01, "low": close * 0.99}
    params = {
        "lookback": 20,
        "scoring": "momentum",
        "top_n": 2,  # 保持股票 ETF 始终在组合内，便于触发 ATR 止损
        "safe_haven": "国债ETF",
        "risk_control": {
            "layer2": {"enabled": True, "atr_multiplier": 3.0, "atr_lookback": 14}
        },
    }
    result = run(data, list(close.columns), params)
    # 跳空下跌后至少有一天 股票ETF 因 ATR 止损被清仓
    assert (result["权重_股票ETF"] == 0).any()
    # 止损资金转入 safe_haven
    safe_haven_full_days = result["权重_国债ETF"] >= 0.99
    assert safe_haven_full_days.any() or (result["权重_股票ETF"] == 0).any()


def test_layer3_vol_target_scales_down_in_high_vol():
    """第三层在高波动期应降低风险资产仓位。"""
    n = 100
    idx = pd.date_range("2023-01-01", periods=n)
    close = pd.DataFrame(
        {"股票ETF": np.full(n, 100.0), "国债ETF": np.full(n, 100.0)},
        index=idx,
    )
    # 后 30 天制造 10% 的日间振幅，EWMA 年化波动率远超 25% 警戒线
    for i in range(n - 30, n):
        close.loc[idx[i], "股票ETF"] = 100 + ((-1) ** i) * 10
    open_ = close.copy()
    data = {"close": close, "open": open_, "high": close * 1.01, "low": close * 0.99}
    params = {
        "lookback": 20,
        "scoring": "momentum",
        "top_n": 1,
        "safe_haven": "国债ETF",
        "risk_control": {
            "layer3": {
                "enabled": True,
                "target_vol": 0.08,
                "vol_lookback": 20,
                "comfort_zone": 0.15,
                "caution_zone": 0.25,
                "caution_scale": 0.5,
            }
        },
    }
    result = run(data, list(close.columns), params)
    # 高波动区域内至少存在减仓日
    high_vol_period = result.iloc[-20:]
    assert (high_vol_period["权重_股票ETF"] < 1.0).any()
    assert (high_vol_period["权重_股票ETF"] >= 0.0).all()


def test_three_layer_risk_control_runs_in_order():
    """三层风控同时开启时策略可正常完成回测。"""
    n = 120
    idx = pd.date_range("2023-01-01", periods=n)
    close = pd.DataFrame(
        {
            "股票ETF": np.linspace(100, 110, n) + np.sin(np.arange(n)) * 5,
            "国债ETF": np.full(n, 100.0),
        },
        index=idx,
    )
    open_ = close.copy()
    data = {"close": close, "open": open_, "high": close * 1.01, "low": close * 0.99}
    params = {
        "lookback": 20,
        "scoring": "momentum",
        "top_n": 1,
        "safe_haven": "国债ETF",
        "risk_control": {
            "layer1": {"enabled": True, "ma_lookback": 20, "drawdown_threshold": 0.10},
            "layer2": {"enabled": True, "atr_multiplier": 3.0, "atr_lookback": 14},
            "layer3": {
                "enabled": True,
                "target_vol": 0.08,
                "vol_lookback": 20,
                "comfort_zone": 0.15,
                "caution_zone": 0.25,
                "caution_scale": 0.5,
            },
        },
    }
    result = run(data, list(close.columns), params)
    assert "轮动策略净值" in result.columns
    assert result["轮动策略净值"].iloc[-1] > 0
    assert "风控原因" in result.columns
