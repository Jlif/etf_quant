"""核心模块"""

from .scorer import momentum_score, slope_r2_score
from .report import (
    OUTPUT_DIR,
    performance_report,
    plot_nav_curves,
    plot_strategy_comparison,
    print_summary,
)
from .orchestrator import (
    build_holding_df,
    calculate_benchmark_returns,
    clear_output_dir,
    compute_signal_start_date,
    detect_and_fix_price_jumps,
    fetch_pool_data,
    print_holding_summary,
    print_latest_signal,
    print_position_contribution,
    report_strategy_result,
    run_strategy,
    save_holding_csv,
)

__all__ = [
    "momentum_score",
    "slope_r2_score",
    "OUTPUT_DIR",
    "performance_report",
    "plot_nav_curves",
    "plot_strategy_comparison",
    "print_summary",
    "build_holding_df",
    "calculate_benchmark_returns",
    "clear_output_dir",
    "compute_signal_start_date",
    "detect_and_fix_price_jumps",
    "fetch_pool_data",
    "print_holding_summary",
    "print_latest_signal",
    "print_position_contribution",
    "report_strategy_result",
    "run_strategy",
    "save_holding_csv",
]
