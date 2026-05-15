"""pytest 测试引导。

确保子仓 `train/gakumas_rl` 的 `src` 包优先于外层工作区同名目录，
避免测试收集阶段误导入外层仓库模块。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / 'src'

project_root_text = str(PROJECT_ROOT)
if project_root_text in sys.path:
    sys.path.remove(project_root_text)
sys.path.insert(0, project_root_text)


def _force_local_src_package() -> None:
    """强制把 `src` 绑定到当前子仓目录。"""

    existing = sys.modules.get('src')
    if existing is not None:
        module_file = vars(existing).get('__file__', '') or ''
        if module_file.startswith(str(SRC_ROOT)):
            return
        sys.modules.pop('src', None)
    init_path = SRC_ROOT / '__init__.py'
    spec = importlib.util.spec_from_file_location(
        'src',
        init_path,
        submodule_search_locations=[str(SRC_ROOT)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f'无法为本地 src 包构造导入规格: {init_path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules['src'] = module
    spec.loader.exec_module(module)


_force_local_src_package()
