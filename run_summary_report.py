"""兼容入口：python -m autolabel_modular.run_summary_report"""
from __future__ import annotations

from autolabel_modular.reports.run_summary_html import main

if __name__ == "__main__":
    raise SystemExit(main())
