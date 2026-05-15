"""兼容 `python -m src.train` 的训练入口。"""

from __future__ import annotations

from .training.cli import main


if __name__ == '__main__':
    raise SystemExit(main())
