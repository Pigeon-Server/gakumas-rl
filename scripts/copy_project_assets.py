from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Copy project assets into standalone/gakumas_rl/assets')
    parser.add_argument('--source-root', type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument('--dest-root', type=Path, default=Path(__file__).resolve().parents[1])
    return parser.parse_args()


def copy_tree(source: Path, target: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(f'Source not found: {source}')
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target, dirs_exist_ok=True, ignore=shutil.ignore_patterns('.git', '__pycache__'))
    print(f'copied: {source} -> {target}')


def main() -> int:
    args = parse_args()
    mappings = [
        (
            args.source_root / 'assets' / 'gakumasu-diff',
            args.dest_root / 'assets' / 'gakumasu-diff',
        ),
        (
            args.source_root / 'assets' / 'GakumasTranslationData' / 'local-files' / 'masterTrans',
            args.dest_root / 'assets' / 'GakumasTranslationData' / 'local-files' / 'masterTrans',
        ),
    ]
    for source, target in mappings:
        copy_tree(source, target)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
