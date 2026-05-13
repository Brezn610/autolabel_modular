"""兼容入口：请优先使用 `python -m autolabel_modular`。"""
from __future__ import annotations

from autolabel_modular.cli.main import main, parse_args

if __name__ == "__main__":
    raise SystemExit(main())
