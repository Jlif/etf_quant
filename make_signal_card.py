#!/Users/jiangchen/.pyenv/versions/3.13.2/bin/python3
# -*- coding: utf-8 -*-
"""
ETF 调仓信号卡片生成器（定时任务专用）

一条命令完成：抓数据 → 算信号 → 渲染 HTML → 截图出图。
stdout 只输出一行 PNG 绝对路径，把 agent 侧的 token 消耗压到最低。

用法:
    ./make_signal_card.py --today
    python make_signal_card.py --today

退出码非 0 时错误信息走 stderr。
"""

from __future__ import annotations

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
PYENV_PY = "/Users/jiangchen/.pyenv/versions/3.13.2/bin/python3"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def ensure_interpreter() -> None:
    """本机默认 python 缺依赖，必要时用 pyenv 解释器重启自己。"""
    try:
        import pandas  # noqa: F401
        return
    except ImportError:
        pass
    if os.path.realpath(sys.executable) == os.path.realpath(PYENV_PY):
        return
    if os.path.exists(PYENV_PY):
        os.execv(PYENV_PY, [PYENV_PY, os.path.abspath(__file__)] + sys.argv[1:])
    sys.stderr.write(f"[错误] 缺少 pandas，且未找到 {PYENV_PY}\n")
    sys.exit(1)


ensure_interpreter()

import argparse  # noqa: E402
import contextlib  # noqa: E402
import io  # noqa: E402
from datetime import datetime  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, ROOT)
os.chdir(ROOT)

CSS = """
  * { margin:0; padding:0; box-sizing:border-box; }
  body {
    background:#eef1f5;
    font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif;
    width:1240px; padding:28px; color:#1f2733;
  }
  .card { background:#fff; border-radius:16px; box-shadow:0 4px 24px rgba(20,30,50,.10); overflow:hidden; }
  .hd { background:linear-gradient(135deg,#1e3a5f 0%,#2c5282 100%); color:#fff; padding:26px 32px 22px; }
  .hd h1 { font-size:27px; font-weight:700; letter-spacing:.5px; }
  .hd .sub { margin-top:8px; font-size:14px; opacity:.82; }
  .hd .tags { margin-top:14px; }
  .tag { display:inline-block; background:rgba(255,255,255,.16); border:1px solid rgba(255,255,255,.25);
         border-radius:20px; padding:4px 13px; font-size:12.5px; margin-right:8px; }
  .stats { display:flex; border-bottom:1px solid #e8ecf1; background:#fafbfd; }
  .stat { flex:1; padding:16px 12px; text-align:center; border-right:1px solid #eef1f5; }
  .stat:last-child { border-right:none; }
  .stat .k { font-size:12px; color:#7a8798; margin-bottom:6px; }
  .stat .v { font-size:20px; font-weight:700; font-variant-numeric:tabular-nums; }
  .up { color:#d32029; } .down { color:#0f9960; } .flat { color:#4a5568; }
  .sec-title { padding:18px 32px 10px; font-size:15px; font-weight:700; color:#1e3a5f;
               display:flex; align-items:center; gap:9px; }
  .sec-title::before { content:""; width:4px; height:15px; background:#2c5282; border-radius:2px; }
  table { width:100%; border-collapse:collapse; font-size:12.5px; }
  thead th { background:#f4f6f9; color:#5a6779; font-weight:600; font-size:11.5px;
             padding:9px 6px; text-align:center; border-bottom:1px solid #e3e8ef; white-space:nowrap; }
  tbody td { padding:8px 6px; text-align:center; border-bottom:1px solid #f0f3f7;
             font-variant-numeric:tabular-nums; white-space:nowrap; }
  tbody tr.pick { background:#fff8e6; }
  .rank { color:#8a95a5; font-weight:600; }
  .star { color:#e8a33d; font-weight:700; }
  .name { text-align:left; padding-left:14px; font-weight:600; color:#263241; }
  .code { color:#8a95a5; font-size:11.5px; }
  .pos { font-weight:700; color:#2c5282; }
  .hold { color:#a0aab8; }
  .reason { font-size:11px; color:#98a3b3; }
  .dd-bad { color:#c0392b; font-weight:600; background:#fdecec; border-radius:4px; padding:1px 5px; }
  .dd-mid { color:#c07a1e; background:#fdf4e3; border-radius:4px; padding:1px 5px; }
  .actions { padding:6px 32px 24px; }
  .act-grid { display:flex; gap:12px; }
  .act { flex:1; border-radius:10px; padding:14px 16px; border:1px solid #e3e8ef; background:#fbfcfe; }
  .act.buy { border-color:#f3c9c9; background:#fef7f7; }
  .act.sell { border-color:#c6e6d5; background:#f5fbf8; }
  .act.hold { border-color:#cdd9e8; background:#f7f9fc; }
  .act .t { font-size:12px; font-weight:700; margin-bottom:9px; }
  .act.buy .t { color:#d32029; } .act.sell .t { color:#0f9960; } .act.hold .t { color:#2c5282; }
  .act .r { font-size:13px; color:#2b3543; margin-bottom:6px; line-height:1.5; }
  .act .r:last-child { margin-bottom:0; }
  .act .r b { color:#1f2733; }
  .act .r span { color:#7a8798; font-size:12px; }
  .ft { padding:16px 32px 22px; border-top:1px solid #eef1f5; }
  .src { font-size:11.5px; color:#8a95a5; line-height:1.7; }
  .disc { margin-top:9px; font-size:11.5px; color:#98a3b3; line-height:1.7; }
"""

DISCLAIMER = (
    "免责声明：以上内容基于公开数据和量化分析，仅供参考，不构成投资建议。市场有风险，投资需谨慎。"
    "任何投资决策应结合个人风险承受能力、资金状况和投资目标独立判断，必要时咨询持牌专业机构。"
    "过往表现不预示未来收益。"
)


def pct(x, digits: int = 1, sign: bool = False) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "-"
    fmt = f"{{:+.{digits}%}}" if sign else f"{{:.{digits}%}}"
    return fmt.format(x)


def cls_of(x: float, neutral: str = "flat") -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return neutral
    return "up" if x > 0 else ("down" if x < 0 else neutral)


def dd_cell(dd) -> str:
    if dd is None or (isinstance(dd, float) and pd.isna(dd)):
        return "-"
    if dd <= -0.20:
        return f'<span class="dd-bad">{dd:.1%}</span>'
    if dd <= -0.10:
        return f'<span class="dd-mid">{dd:.1%}</span>'
    return f"{dd:.1%}"


def run_fetch(today: bool) -> None:
    cmd = [sys.executable, os.path.join(ROOT, "fetch_data.py")]
    if today:
        cmd.append("--today")
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(f"[错误] fetch_data.py 失败:\n{proc.stdout}\n{proc.stderr}\n")
        sys.exit(proc.returncode)


def build_rows(strategy, result, name_list, last_quote_dates):
    """提取卡片所需的结构化信号数据（口径与 print_latest_signal 一致）。"""
    latest = result.iloc[-1]
    params = strategy.params
    scoring = params.get("scoring", "momentum")
    prefix = {
        "slope_r2": "得分_", "momentum_quality": "质量_", "ema_diff": "趋势_",
    }.get(scoring, "涨幅_")
    lookback = params.get("lookback", 20)
    top_n = params.get("top_n", 1)

    layer1 = params.get("risk_control", {}).get("layer1", {})
    l1_enabled = layer1.get("enabled", False)
    ma_lb = layer1.get("ma_lookback", 20)
    dd_lb = layer1.get("drawdown_lookback", 252)

    layer3 = params.get("risk_control", {}).get("layer3", {})
    l3_enabled = layer3.get("enabled", False)
    vol_lb = layer3.get("vol_lookback", 20)

    vol_map, dd_map, ma_map = {}, {}, {}
    for name in name_list:
        if name not in result.columns:
            continue
        prices = result[name].dropna()
        if prices.empty:
            continue
        returns = prices.pct_change(fill_method=None).dropna()
        if len(returns) >= 2:
            if l3_enabled:
                vol_map[name] = returns.ewm(span=vol_lb).std().iloc[-1] * np.sqrt(252)
            else:
                vol_map[name] = returns.tail(lookback).std() * np.sqrt(252)
        if l1_enabled:
            cur = prices.iloc[-1]
            dd_map[name] = cur / prices.tail(dd_lb).max() - 1
            if len(prices) >= ma_lb:
                ma = prices.tail(ma_lb).mean()
                ma_map[name] = (cur, ma, cur / ma - 1)

    rows = []
    for name in name_list:
        score_col = f"{prefix}{name}"
        score = latest[score_col] if score_col in latest else np.nan
        code = next((p.code for p in strategy.pool if p.name == name), "")
        last_date = last_quote_dates.get(name, "-") if last_quote_dates else "-"
        cur_w = float(latest[f"权重_{name}"]) if f"权重_{name}" in latest else 0.0
        sig_col = f"信号权重_{name}"
        sig_w = float(latest[sig_col]) if sig_col in latest else cur_w
        rsn_col = f"信号风控原因_{name}"
        raw_reason = str(latest.get(rsn_col) if rsn_col in latest else latest.get(f"风控原因_{name}", ""))
        reason = "" if sig_w > 0 else (raw_reason if raw_reason and raw_reason != "nan" else f"未进top{top_n}")
        rows.append({
            "name": name, "code": code, "date": last_date,
            "score": score, "vol": vol_map.get(name, np.nan),
            "dd": dd_map.get(name, np.nan), "ma": ma_map.get(name),
            "cur": cur_w, "sig": sig_w, "reason": reason,
        })

    rows.sort(key=lambda r: r["score"] if not pd.isna(r["score"]) else -np.inf, reverse=True)
    return rows, {"lookback": lookback, "top_n": top_n, "ma_lb": ma_lb, "dd_lb": dd_lb,
                  "vol_lb": vol_lb, "l1": l1_enabled, "l3": l3_enabled}


def build_actions(rows):
    buy, sell, hold = [], [], []
    for r in rows:
        cur, sig = r["cur"], r["sig"]
        label = f'<b>{r["name"]}</b> <span>{r["code"]}</span><br>{cur*100:.0f}% → <b>{sig*100:.0f}%</b>'
        if sig > cur + 1e-9:
            buy.append(label)
        elif sig < cur - 1e-9:
            sell.append(label)
        elif sig > 0:
            hold.append(label)
    turnover = 0.5 * sum(abs(r["sig"] - r["cur"]) for r in rows)
    return buy, sell, hold, turnover


def render_html(strategy, result, rows, meta, nav_stats, signal_date):
    buy, sell, hold, turnover = build_actions(rows)
    lookback, top_n = meta["lookback"], meta["top_n"]
    picked = [r for r in rows if r["sig"] > 0]

    stat = lambda k, v, c: f'<div class="stat"><div class="k">{k}</div><div class="v {c}">{v}</div></div>'
    c_day = cls_of(nav_stats["daily"])
    stats = "".join([
        stat("组合净值", f'{nav_stats["nav"]:.4f}', "flat"),
        stat("当日收益", pct(nav_stats["daily"], 2, True), c_day),
        stat("近5日", pct(nav_stats["r5"], 2, True), cls_of(nav_stats["r5"])),
        stat("近20日", pct(nav_stats["r20"], 2, True), cls_of(nav_stats["r20"])),
        stat("距区间峰值", pct(nav_stats["dd"], 2, True), cls_of(nav_stats["dd"])),
        stat("候选池 / 入选", f"{len(rows)} / {len(picked)}", "flat"),
    ])

    trs = []
    for i, r in enumerate(rows, 1):
        star = r["sig"] > 0
        rank = f'<span class="star">★{i}</span>' if star else str(i)
        if r["ma"]:
            cur, ma, dev = r["ma"]
            ma_cell = f'{cur:.3f} / {ma:.3f} &nbsp;<span class="{cls_of(dev)}">{dev:+.1%}</span>'
        else:
            ma_cell = "-"
        trs.append(
            f'<tr class="{"pick" if star else ""}">'
            f'<td class="rank">{rank}</td>'
            f'<td class="name">{r["name"]}</td>'
            f'<td class="code">{r["code"]}</td>'
            f'<td>{r["date"]}</td>'
            f'<td class="{cls_of(r["score"])}">{pct(r["score"], 2, True)}</td>'
            f'<td>{pct(r["vol"])}</td>'
            f'<td>{dd_cell(r["dd"])}</td>'
            f'<td>{ma_cell}</td>'
            f'<td class="{"pos" if r["cur"] > 0 else "hold"}">{r["cur"]*100:.0f}%</td>'
            f'<td class="{"pos" if r["sig"] > 0 else "hold"}">{r["sig"]*100:.0f}%</td>'
            f'<td class="reason">{r["reason"] or "—"}</td>'
            f"</tr>"
        )

    def act_block(kind, title, items):
        if not items:
            return f'<div class="act {kind}"><div class="t">{title}</div><div class="r">—</div></div>'
        body = "".join(f'<div class="r">{x}</div>' for x in items)
        return f'<div class="act {kind}"><div class="t">{title}</div>{body}</div>'

    actions = (
        act_block("buy", "▲ 买入 / 加仓", buy)
        + act_block("sell", "▼ 卖出 / 清仓", sell)
        + act_block("hold", "● 维持不动", hold)
    )

    n_bars = len(result)
    start = result.index[0].strftime("%Y-%m-%d")
    src = (
        f"数据来源：TickFlow 日K（本地策略脚本 latest_signal.py 口径）｜数据时点：{signal_date} 收盘"
        f"｜策略区间：{start} ~ {signal_date}（{n_bars} 个交易日）"
    )

    return (
        "<!DOCTYPE html><html lang=\"zh-CN\"><head><meta charset=\"UTF-8\">"
        f"<style>{CSS}</style></head><body><div class=\"card\">"
        f'<div class="hd"><h1>ETF 动量轮动 · 今日调仓信号</h1>'
        f'<div class="sub">信号日期 {signal_date} ｜ 基于当日收盘数据生成 · 盘后</div>'
        f'<div class="tags"><span class="tag">lookback {lookback}日</span>'
        f'<span class="tag">top_n {top_n}</span>'
        f'<span class="tag">周度调仓 · 日度风控</span>'
        f'<span class="tag">换手率 {turnover*100:.0f}%</span></div></div>'
        f'<div class="stats">{stats}</div>'
        f'<div class="sec-title">全池排名与信号明细</div>'
        f'<table><thead><tr>'
        f'<th style="width:52px">排名</th>'
        f'<th style="text-align:left;padding-left:14px">ETF名称</th>'
        f'<th style="width:58px">代码</th>'
        f'<th style="width:88px">行情日</th>'
        f'<th style="width:80px">动量{lookback}日</th>'
        f'<th style="width:92px">波动率EWMA{meta["vol_lb"]}</th>'
        f'<th style="width:78px">回撤{meta["dd_lb"]}日</th>'
        f'<th style="width:150px">现价 / MA{meta["ma_lb"]}</th>'
        f'<th style="width:66px">当前仓位</th>'
        f'<th style="width:66px">明日仓位</th>'
        f"<th>未入选原因</th></tr></thead>"
        f'<tbody>{"".join(trs)}</tbody></table>'
        f'<div class="sec-title">明日操作建议（{len(buy)+len(sell)} 笔变动，换手率 {turnover*100:.0f}%）</div>'
        f'<div class="actions"><div class="act-grid">{actions}</div></div>'
        f'<div class="ft"><div class="src">{src}</div><div class="disc">{DISCLAIMER}</div></div>'
        f"</div></body></html>"
    )


def crop_bottom(png_path: str) -> None:
    """截掉 Chrome 高窗口留下的底部空白，让卡片高度自适应。"""
    try:
        from PIL import Image
    except ImportError:
        return
    im = Image.open(png_path).convert("RGB")
    w, h = im.size
    bg = im.getpixel((2, 2))
    px = im.load()
    last = h - 1
    while last > 0:
        row = [px[x, last] for x in range(0, w, 8)]
        if any(abs(c[0] - bg[0]) + abs(c[1] - bg[1]) + abs(c[2] - bg[2]) > 12 for c in row):
            break
        last -= 1
    if last + 20 < h:
        im.crop((0, 0, w, min(h, last + 20))).save(png_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="ETF 调仓信号卡片生成器")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--strategy", help="指定策略名称（默认取第一个启用的 rotation 策略）")
    parser.add_argument("--date", help="指定交易截止日 (YYYYMMDD)")
    parser.add_argument("--today", action="store_true", help="使用当天作为截止日")
    parser.add_argument("--out", help="输出 PNG 路径")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="该信号日期的卡片若已生成则跳过（供定时任务避免节假日重复推送同一张旧信号）",
    )
    args = parser.parse_args()

    from data_source import get_data_source
    from core.orchestrator import compute_signal_start_date, fetch_pool_data, run_strategy
    from utils import load_config

    run_fetch(args.today)

    if args.today:
        cutoff = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    elif args.date:
        cutoff = datetime.strptime(args.date, "%Y%m%d")
    else:
        cutoff = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    app_config = load_config(args.config)
    strategies = [s for s in app_config.strategies if s.enabled and s.mode == "rotation"]
    if args.strategy:
        strategies = [s for s in strategies if s.name == args.strategy]
    if not strategies:
        sys.stderr.write("[错误] 未找到启用的 rotation 策略\n")
        sys.exit(1)
    strategy = strategies[0]

    # 计算过程全程静音：stdout 只保留最后那行 PNG 路径
    with contextlib.redirect_stdout(io.StringIO()):
        data_source = get_data_source(name=app_config.data_source.provider, fallback=False, skip_test=True)
        start_date = compute_signal_start_date(strategy, cutoff).strftime("%Y%m%d")
        data = fetch_pool_data(
            strategy, app_config, data_source,
            include_today=args.today, cutoff_date=cutoff, start_date=start_date,
            min_bars=strategy.params.get("lookback", 20) + 5, skip_download=True,
        )
        result, name_list = run_strategy(strategy, app_config, data_source, data=data)

    last_quote_dates = {
        name: data["close"][name].last_valid_index().strftime("%Y-%m-%d") for name in name_list
    }
    rows, meta = build_rows(strategy, result, name_list, last_quote_dates)

    nav = result["轮动策略净值"].dropna()
    nav_stats = {
        "nav": float(nav.iloc[-1]),
        "daily": float(result["轮动策略日收益率"].iloc[-1]),
        "r5": float(nav.iloc[-1] / nav.iloc[-6] - 1) if len(nav) > 5 else float("nan"),
        "r20": float(nav.iloc[-1] / nav.iloc[-21] - 1) if len(nav) > 20 else float("nan"),
        "dd": float(nav.iloc[-1] / nav.max() - 1),
    }

    signal_date = result.index[-1].strftime("%Y-%m-%d")
    os.makedirs("output", exist_ok=True)
    html_path = os.path.join(ROOT, "output", f"signal_card_{signal_date}.html")
    png_path = args.out or os.path.join(ROOT, "output", f"signal_card_{signal_date}.png")

    if args.skip_existing and os.path.exists(png_path):
        sys.stderr.write(
            f"[跳过] {signal_date} 的卡片已存在（非交易日或当日已出过图），本次不重复推送\n"
        )
        sys.exit(0)

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(render_html(strategy, result, rows, meta, nav_stats, signal_date))

    cmd = [
        CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars",
        "--force-device-scale-factor=2", "--window-size=1240,3200",
        f"--screenshot={png_path}", html_path,
    ]
    proc = subprocess.run(cmd, cwd=os.path.dirname(png_path), capture_output=True, text=True)
    if proc.returncode != 0 or not os.path.exists(png_path):
        sys.stderr.write(f"[错误] 截图失败:\n{proc.stderr}\n")
        sys.exit(1)
    crop_bottom(png_path)

    print(os.path.abspath(png_path))


if __name__ == "__main__":
    main()
