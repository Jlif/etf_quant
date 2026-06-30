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
