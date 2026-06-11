"""绩效报告与可视化"""

from __future__ import annotations

import os

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import quantstats as qstat

# 无图形界面环境使用 Agg 后端，避免 plt.show() 阻塞
if os.environ.get("DISPLAY") is None and os.name != "nt":
    matplotlib.use("Agg")

plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 输出目录
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def performance_report(
    nav: pd.Series,
    benchmark: pd.Series | None = None,
    title: str = "策略回测报告",
) -> str:
    """输出 quantstats 回测报告并保存 HTML 到 output 目录（不自动打开浏览器）"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

    # 使用 metrics 输出文字指标（避免 quantstats 图表兼容性问题）
    qstat.reports.metrics(nav, benchmark=benchmark, display=True)

    safe_title = title.replace(" ", "_").replace(":", "_")
    filename = f"{safe_title}.html"
    filepath = os.path.join(OUTPUT_DIR, filename)
    try:
        # output 传字符串路径，quantstats 会写入该文件
        qstat.reports.html(
            nav, benchmark=benchmark, title=title,
            output=filepath,
        )
        print(f"\n[报告已保存] {filepath}")
    except Exception as e:
        print(f"\n[警告] HTML 报告生成失败: {e}")
    return filepath


def plot_nav_curves(
    results: dict[str, tuple[pd.DataFrame, list[str], str]],
    save_path: str | None = None,
) -> None:
    """
    绘制各策略的净值曲线（含标的对比）

    Parameters
    ----------
    results : dict
        {策略名: (data_df, name_list, strategy_nav_col)}
    save_path : str | None
        图片保存路径，None 则保存到 output 目录
    """
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "轮动策略净值对比.png")

    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(8 * n, 6))
    if n == 1:
        axes = [axes]

    for ax, (strategy_name, (data, name_list, nav_col)) in zip(axes, results.items()):
        for name in name_list:
            nav = data[name] / data[name].iloc[0]
            ax.plot(nav.index, nav.values, linestyle="--", alpha=0.7, label=name)

        ax.plot(
            data[nav_col].index,
            data[nav_col].values,
            linestyle="-",
            color="#FF8124",
            linewidth=2,
            label="策略净值",
        )
        ax.set_xlabel("日期")
        ax.set_ylabel("净值")
        ax.legend(loc="upper left")
        ax.set_title(strategy_name)
        ax.grid(True, alpha=0.3)

    plt.suptitle("轮动策略净值曲线对比", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[图表已保存] {save_path}")


def plot_strategy_comparison(
    nav_dict: dict[str, pd.Series],
    save_path: str | None = None,
    title: str = "多策略净值对比",
) -> None:
    """将多个策略的净值放在一张图中对比"""
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "策略净值对比.png")

    fig, ax = plt.subplots(figsize=(15, 7))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for i, (name, nav) in enumerate(nav_dict.items()):
        ax.plot(nav.index, nav.values, linestyle="-", color=colors[i % len(colors)],
                linewidth=2, label=name)

    ax.set_xlabel("日期")
    ax.set_ylabel("净值")
    ax.legend(loc="upper left")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[图表已保存] {save_path}")


def print_summary(results: dict[str, pd.Series]) -> None:
    """打印各策略最终净值对比表"""
    print("\n" + "="*60)
    print("【最终净值对比】")
    print("="*60)
    summary = pd.DataFrame([
        {
            "策略": name,
            "最终净值": round(nav.iloc[-1], 4),
            "回测起点": str(nav.index[0]),
            "回测终点": str(nav.index[-1]),
        }
        for name, nav in results.items()
    ])
    print(summary.to_string(index=False))
    print("="*60)
