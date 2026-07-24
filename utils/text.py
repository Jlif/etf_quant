"""终端文本对齐工具"""

from __future__ import annotations

import unicodedata


def display_width(s: str) -> int:
    """计算字符串在终端中的显示宽度（中文等宽字符按 2 计）。"""
    return sum(
        2 if unicodedata.east_asian_width(ch) in ("F", "W") else 1
        for ch in str(s)
    )


def ljust(s: str, width: int) -> str:
    """按显示宽度左对齐。"""
    return s + " " * max(0, width - display_width(s))


def rjust(s: str, width: int) -> str:
    """按显示宽度右对齐。"""
    return " " * max(0, width - display_width(s)) + s
