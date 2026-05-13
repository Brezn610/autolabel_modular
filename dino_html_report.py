"""兼容入口：等价于 `python -m autolabel_modular.reports.dino_html_report`。"""
from __future__ import annotations

from autolabel_modular.reports.dino_html_report import build_dino_html_report, main, parse_args

__all__ = ["build_dino_html_report", "main", "parse_args"]

if __name__ == "__main__":
    raise SystemExit(main())
