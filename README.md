# etf_quant

基于 Python 的 ETF 量化回测系统。

## 快速开始

```bash
pip install -r requirements.txt

# 1. 拉取数据（优先 akshare 东财接口，标的间限速 1 秒）
python fetch_data.py --config config.yaml

# 2. 运行回测（只读本地缓存，不触发网络请求）
python main.py --config config.yaml

# 3. 查看最新信号
python latest_signal.py --config config.yaml
```

详见 [CLAUDE.md](CLAUDE.md)。
