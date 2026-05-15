"""兼容 `python -m src.bc_pretrain` 的 BC 入口。"""

from __future__ import annotations

from .training.bc_pretrain import main


if __name__ == '__main__':
    raise SystemExit(main())
