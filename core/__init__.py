"""核心模块"""

from .scorer import momentum_score, slope_r2_score
from .report import (
    OUTPUT_DIR,
    performance_report,
    plot_nav_curves,
    plot_strategy_comparison,
    print_summary,
)

__all__ = [
    "momentum_score",
    "slope_r2_score",
    "OUTPUT_DIR",
    "performance_report",
    "plot_nav_curves",
    "plot_strategy_comparison",
    "print_summary",
]
