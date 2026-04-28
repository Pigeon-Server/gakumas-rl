"""Jinja2 渲染层 — 加载 prompts/ 目录下的模板并提供渲染 API。"""

from __future__ import annotations

import functools
from typing import Any, Iterable

import jinja2

from .state_snapshot import _short_effect_type, _short_category

# ---------------------------------------------------------------------------
# Jinja2 Environment（单例）
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _get_env() -> jinja2.Environment:
    env = jinja2.Environment(
        loader=jinja2.PackageLoader('gakumas_rl.prompts', ''),
        keep_trailing_newline=False,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    # 注册自定义 filter
    env.filters['short_effect_type'] = _short_effect_type
    env.filters['short_category'] = _short_category
    return env


# ---------------------------------------------------------------------------
# 公开 API
# ---------------------------------------------------------------------------

def render(template_name: str, **kwargs: Any) -> str:
    """渲染指定模板，返回字符串。"""
    tmpl = _get_env().get_template(template_name)
    return tmpl.render(**kwargs)


def load_system_prompt(
    *,
    extra_sections: Iterable[str] | None = None,
    base_prompt: str | None = None,
) -> str:
    """渲染 system.jinja2，并可选追加额外策略段落。"""

    prompt = base_prompt if base_prompt is not None else render('system.jinja2')
    if not extra_sections:
        return prompt
    sections = [str(section).strip() for section in extra_sections if str(section).strip()]
    if not sections:
        return prompt
    return prompt.rstrip() + '\n\n## 补充策略资料\n' + '\n\n'.join(sections)
