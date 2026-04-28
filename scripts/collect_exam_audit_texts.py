from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gakumas_rl.data import MasterDataRepository


HELP_TABLES = ('HelpCategory', 'HelpContent', 'Tips', 'Terms')
DICTIONARY_TABLES = (
    'ProduceDescription',
    'ProduceDescriptionExamEffect',
    'ProduceDescriptionProduceExamEffectType',
    'ProduceDescriptionProduceCardGrowEffect',
    'ProduceDescriptionProduceCardGrowEffectType',
    'EffectGroup',
)
TEXT_TABLES = (
    'ProduceItem',
    'ProduceCard',
    'ProduceDrink',
    'ProduceExamEffect',
    'ProduceCardSearch',
    'ProduceCardStatusEnchant',
    'ProduceExamStatusEnchant',
    'ProduceExamTrigger',
    'ProduceExamGimmickEffectGroup',
)
UNKNOWN_SUFFIX = '_Unknown'
THEME_RULES = {
    'next_turn_and_timing': {
        'label': '次回合 / 时机',
        'keywords': ('次回合', '下回合', '下一回合', '回合开始时', '回合结束后', '回合追加'),
        'effect_types': ('ProduceExamEffectType_ExamEffectTimer',),
    },
    'play_window_and_turn_flow': {
        'label': '出牌次数 / 回合流转',
        'keywords': ('技能卡使用数', '出牌次数', '额外回合', '回合追加', '结束回合'),
        'effect_types': ('ProduceExamEffectType_ExamPlayableValueAdd',),
    },
    'move_hold_and_return': {
        'label': '移动 / 保留 / 回手',
        'keywords': (
            '保留',
            '移到手牌',
            '移入手牌',
            '移至手牌',
            '移至弃牌',
            '移至牌库',
            '移至保留',
            '返回手牌',
            '加入手牌',
            '从牌堆',
            '从弃牌',
        ),
        'effect_types': ('ProduceExamEffectType_ExamCardMove',),
    },
    'grow_and_upgrade': {
        'label': '成长 / 强化 / 升级',
        'keywords': ('成长', '强化', '升级'),
        'effect_types': (
            'ProduceExamEffectType_ExamAddGrowEffect',
            'ProduceExamEffectType_ExamCardUpgrade',
        ),
    },
    'search_draw_and_force_play': {
        'label': '检索 / 抽牌 / 强制打出',
        'keywords': ('检索', '抽', '移动到手牌', '使用它', '使用那张卡'),
        'effect_types': (
            'ProduceExamEffectType_ExamCardDraw',
            'ProduceExamEffectType_ExamForcePlayCardSearch',
        ),
        'tables': ('ProduceCardSearch',),
    },
    'cost_and_stamina_override': {
        'label': '消耗 / 体力改写',
        'keywords': ('体力消耗', '消耗的体力变为0', '消耗增加', '消耗减少', '无额外消耗', '元气'),
        'effect_types': (),
    },
    'full_power_and_preservation': {
        'label': '全力 / 温存 / 强气',
        'keywords': ('全力', '温存', '强气'),
        'effect_types': (
            'ProduceExamEffectType_ExamFullPower',
            'ProduceExamEffectType_ExamPreservation',
        ),
    },
}


@dataclass(frozen=True)
class TablePayload:
    table: str
    entries: list[dict[str, Any]]
    matched_master_rows: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Collect help text and exam-related master text into structured audit artifacts.'
    )
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output-json', type=Path, default=Path('docs/exam_audit_texts.json'))
    parser.add_argument('--output-md', type=Path, default=Path('docs/exam_audit_texts.md'))
    parser.add_argument(
        '--fragments',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Include simplified fragment metadata in JSON output.',
    )
    return parser.parse_args()


def _read_localization_rows(localization_dir: Path, table_name: str) -> list[dict[str, Any]]:
    path = localization_dir / f'{table_name}.json'
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding='utf-8'))
    return list(payload.get('data', []))


def _clean_text(text: str) -> str:
    value = text.replace('\r\n', '\n').replace('\r', '\n')
    value = value.replace('<nobr>', '').replace('</nobr>', '')
    value = re.sub(r'<br\\s*/?>', '\n', value, flags=re.IGNORECASE)
    value = re.sub(r'<[^>]+>', '', value)
    value = value.replace('&nbsp;', ' ')
    value = re.sub(r'\n{3,}', '\n\n', value)
    return value.strip()


def _normalize_enum(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    if not value or value.endswith(UNKNOWN_SUFFIX):
        return None
    return value


def _compact_dict(payload: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in payload.items():
        if value in (None, '', [], {}, ()):
            continue
        result[key] = value
    return result


def _render_fragments(fragments: list[dict[str, Any]] | None) -> str:
    if not fragments:
        return ''
    pieces: list[str] = []
    for fragment in fragments:
        if isinstance(fragment, dict):
            cleaned = _clean_text(str(fragment.get('text') or ''))
        else:
            cleaned = _clean_text(str(fragment))
        if cleaned:
            pieces.append(cleaned)
    return ''.join(pieces).strip()


def _summarize_fragments(fragments: list[dict[str, Any]] | None) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    summarized: list[dict[str, Any]] = []
    refs: dict[str, set[str]] = {
        'exam_effect_types': set(),
        'card_categories': set(),
        'move_positions': set(),
        'step_types': set(),
        'description_types': set(),
        'origin_exam_effect_ids': set(),
        'origin_exam_trigger_ids': set(),
        'origin_card_status_enchant_ids': set(),
    }
    for fragment in fragments or []:
        if not isinstance(fragment, dict):
            cleaned = _clean_text(str(fragment))
            if cleaned:
                summarized.append({'text': cleaned})
            continue
        entry = _compact_dict(
            {
                'text': _clean_text(str(fragment.get('text') or '')),
                'produceDescriptionType': _normalize_enum(fragment.get('produceDescriptionType')),
                'examDescriptionType': _normalize_enum(fragment.get('examDescriptionType')),
                'examEffectType': _normalize_enum(fragment.get('examEffectType')),
                'produceCardGrowEffectType': _normalize_enum(fragment.get('produceCardGrowEffectType')),
                'produceCardCategory': _normalize_enum(fragment.get('produceCardCategory')),
                'produceCardMovePositionType': _normalize_enum(fragment.get('produceCardMovePositionType')),
                'produceStepType': _normalize_enum(fragment.get('produceStepType')),
                'targetId': fragment.get('targetId') or None,
                'originProduceExamEffectId': fragment.get('originProduceExamEffectId') or None,
                'originProduceExamTriggerId': fragment.get('originProduceExamTriggerId') or None,
                'originProduceCardStatusEnchantId': fragment.get('originProduceCardStatusEnchantId') or None,
            }
        )
        if entry:
            summarized.append(entry)
        if value := entry.get('examEffectType'):
            refs['exam_effect_types'].add(value)
        if value := entry.get('produceCardCategory'):
            refs['card_categories'].add(value)
        if value := entry.get('produceCardMovePositionType'):
            refs['move_positions'].add(value)
        if value := entry.get('produceStepType'):
            refs['step_types'].add(value)
        if value := entry.get('produceDescriptionType'):
            refs['description_types'].add(value)
        if value := entry.get('originProduceExamEffectId'):
            refs['origin_exam_effect_ids'].add(value)
        if value := entry.get('originProduceExamTriggerId'):
            refs['origin_exam_trigger_ids'].add(value)
        if value := entry.get('originProduceCardStatusEnchantId'):
            refs['origin_card_status_enchant_ids'].add(value)
    sorted_refs = {key: sorted(values) for key, values in refs.items() if values}
    return summarized, sorted_refs


def _select_master_fields(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if not row:
        return None
    selected: dict[str, Any] = {}
    for key, value in row.items():
        if key == 'produceDescriptions':
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            if value in ('', None):
                continue
            selected[key] = value
            continue
        if isinstance(value, list) and all(isinstance(item, (str, int, float, bool)) for item in value):
            if value:
                selected[key] = value
    return selected or None


def _collect_text_table(
    repo: MasterDataRepository,
    localization_dir: Path,
    table_name: str,
    include_fragments: bool,
) -> TablePayload:
    localization_rows = _read_localization_rows(localization_dir, table_name)
    try:
        master_index = repo.load_table(table_name)
    except FileNotFoundError:
        master_index = None
    entries: list[dict[str, Any]] = []
    matched = 0
    for row in localization_rows:
        item_id = str(row.get('id') or '')
        master_row = master_index.first(item_id) if master_index is not None else None
        if master_row is not None:
            matched += 1
        fragments = row.get('produceDescriptions') if isinstance(row.get('produceDescriptions'), list) else None
        summarized_fragments, fragment_refs = _summarize_fragments(fragments)
        entry = _compact_dict(
            {
                'source_table': table_name,
                'id': item_id,
                'name': row.get('name') or None,
                'order': row.get('order') if 'order' in row else None,
                'priority': row.get('priority') if 'priority' in row else None,
                'text': _render_fragments(fragments),
                'fragment_refs': fragment_refs or None,
                'fragments': summarized_fragments if include_fragments else None,
                'master': _select_master_fields(master_row),
            }
        )
        entries.append(entry)
    return TablePayload(table=table_name, entries=entries, matched_master_rows=matched)


def _collect_help(localization_dir: Path) -> dict[str, Any]:
    categories = _read_localization_rows(localization_dir, 'HelpCategory')
    contents = _read_localization_rows(localization_dir, 'HelpContent')
    tips = _read_localization_rows(localization_dir, 'Tips')
    terms = _read_localization_rows(localization_dir, 'Terms')
    contents_by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in contents:
        contents_by_category[str(row.get('helpCategoryId') or '')].append(
            _compact_dict(
                {
                    'id': row.get('id'),
                    'order': row.get('order'),
                    'name': row.get('name'),
                }
            )
        )
    category_entries = []
    for category in categories:
        category_entries.append(
            _compact_dict(
                {
                    'id': category.get('id'),
                    'order': category.get('order'),
                    'name': category.get('name'),
                    'texts': [_clean_text(str(text)) for text in category.get('texts', []) if _clean_text(str(text))],
                    'entries': contents_by_category.get(str(category.get('id') or ''), []),
                }
            )
        )
    tip_entries = [
        _compact_dict(
            {
                'id': row.get('id'),
                'title': row.get('title'),
                'description': _clean_text(str(row.get('description') or '')),
            }
        )
        for row in tips
    ]
    term_entries = [
        _compact_dict(
            {
                'type': row.get('type'),
                'name': row.get('name'),
            }
        )
        for row in terms
    ]
    return {
        'categories': category_entries,
        'tips': tip_entries,
        'terms': term_entries,
    }


def _collect_dictionaries(localization_dir: Path) -> dict[str, list[dict[str, Any]]]:
    payload: dict[str, list[dict[str, Any]]] = {}
    for table_name in DICTIONARY_TABLES:
        rows = _read_localization_rows(localization_dir, table_name)
        payload[table_name] = [_compact_dict(row) for row in rows]
    return payload


def _matches_theme(entry: dict[str, Any], table_name: str, theme_rule: dict[str, Any]) -> bool:
    allowed_tables = tuple(theme_rule.get('tables') or ())
    if allowed_tables and table_name not in allowed_tables:
        return False
    haystack = '\n'.join(
        value
        for value in (
            str(entry.get('name') or ''),
            str(entry.get('text') or ''),
            ' '.join(entry.get('fragment_refs', {}).get('exam_effect_types', [])),
        )
        if value
    )
    if any(keyword in haystack for keyword in theme_rule.get('keywords', ())):
        return True
    effect_types = set(entry.get('fragment_refs', {}).get('exam_effect_types', []))
    master_effect_type = (entry.get('master') or {}).get('effectType')
    if isinstance(master_effect_type, str):
        effect_types.add(master_effect_type)
    return any(effect_type in effect_types for effect_type in theme_rule.get('effect_types', ()))


def _build_theme_index(text_tables: dict[str, TablePayload]) -> dict[str, dict[str, Any]]:
    themes: dict[str, dict[str, Any]] = {}
    for theme_id, rule in THEME_RULES.items():
        matched_entries: list[dict[str, Any]] = []
        for table_name, payload in text_tables.items():
            for entry in payload.entries:
                if _matches_theme(entry, table_name, rule):
                    matched_entries.append(
                        _compact_dict(
                            {
                                'source_table': table_name,
                                'id': entry.get('id'),
                                'name': entry.get('name'),
                                'text': entry.get('text'),
                                'effect_types': entry.get('fragment_refs', {}).get('exam_effect_types')
                                or ((entry.get('master') or {}).get('effectType') and [(entry.get('master') or {}).get('effectType')])
                                or None,
                            }
                        )
                    )
        themes[theme_id] = {
            'label': rule['label'],
            'count': len(matched_entries),
            'entries': matched_entries,
        }
    return themes


def _build_effect_type_index(text_tables: dict[str, TablePayload]) -> dict[str, list[str]]:
    mapping: dict[str, set[str]] = defaultdict(set)
    for table_name, payload in text_tables.items():
        for entry in payload.entries:
            item_id = str(entry.get('id') or '')
            refs = entry.get('fragment_refs', {})
            for effect_type in refs.get('exam_effect_types', []):
                mapping[effect_type].add(f'{table_name}:{item_id}')
            master_effect_type = (entry.get('master') or {}).get('effectType')
            if isinstance(master_effect_type, str):
                mapping[master_effect_type].add(f'{table_name}:{item_id}')
    return {effect_type: sorted(item_ids) for effect_type, item_ids in sorted(mapping.items())}


def _build_markdown(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append('# Exam Audit Text Dump')
    lines.append('')
    lines.append(f"- Generated at: `{payload['generated_at']}`")
    lines.append(f"- Root: `{payload['root']}`")
    lines.append(f"- Localization dir: `{payload['localization_dir']}`")
    lines.append(f"- Master assets dir: `{payload['assets_dir']}`")
    lines.append('')
    lines.append('## Help Categories')
    lines.append('')
    for category in payload['help']['categories']:
        lines.append(f"### {category.get('name') or category.get('id')}")
        lines.append(f"- id: `{category.get('id')}`")
        for text in category.get('texts', []):
            lines.append(f"- {text}")
        entries = category.get('entries', [])
        if entries:
            lines.append(f"- entries: {', '.join(entry.get('name') or entry.get('id') or '' for entry in entries)}")
        lines.append('')
    lines.append('## Table Summary')
    lines.append('')
    for table_name, table_payload in payload['text_tables'].items():
        lines.append(
            f"- `{table_name}`: {table_payload['count']} rows, matched master rows: {table_payload['matched_master_rows']}"
        )
    lines.append('')
    lines.append('## Theme Index')
    lines.append('')
    for theme_id, theme_payload in payload['indexes']['themes'].items():
        lines.append(f"### {theme_payload['label']} ({theme_payload['count']})")
        for entry in theme_payload['entries'][:80]:
            name = entry.get('name')
            title = f"{entry['source_table']}:{entry['id']}"
            if name:
                title = f"{title} {name}"
            text = entry.get('text') or ''
            lines.append(f"- `{title}`: {text}")
        if theme_payload['count'] > 80:
            lines.append(f"- ... truncated, see JSON for full list (`{theme_id}`)")
        lines.append('')
    lines.append('## Tips')
    lines.append('')
    for tip in payload['help']['tips']:
        title = tip.get('title') or tip.get('id')
        description = tip.get('description') or ''
        lines.append(f"- `{tip.get('id')}` {title}: {description}")
    lines.append('')
    return '\n'.join(lines).strip() + '\n'


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    repo = MasterDataRepository(root_dir=root)
    localization_dir = repo.localization_dir
    text_tables: dict[str, TablePayload] = {}
    for table_name in TEXT_TABLES:
        text_tables[table_name] = _collect_text_table(repo, localization_dir, table_name, args.fragments)
    payload = {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'root': str(root),
        'localization_dir': str(localization_dir),
        'assets_dir': str(repo.assets_dir),
        'help': _collect_help(localization_dir),
        'dictionaries': _collect_dictionaries(localization_dir),
        'text_tables': {
            table_name: {
                'count': len(table_payload.entries),
                'matched_master_rows': table_payload.matched_master_rows,
                'entries': table_payload.entries,
            }
            for table_name, table_payload in text_tables.items()
        },
        'indexes': {
            'themes': _build_theme_index(text_tables),
            'exam_effect_types': _build_effect_type_index(text_tables),
            'table_counts': {table_name: len(table_payload.entries) for table_name, table_payload in text_tables.items()},
        },
    }
    output_json = args.output_json if args.output_json.is_absolute() else root / args.output_json
    output_md = args.output_md if args.output_md.is_absolute() else root / args.output_md
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    output_md.write_text(_build_markdown(payload), encoding='utf-8')
    print(f'wrote json: {output_json}')
    print(f'wrote markdown: {output_md}')
    print('table counts:')
    for table_name, count in payload['indexes']['table_counts'].items():
        print(f'  - {table_name}: {count}')
    theme_counts = Counter({theme_id: theme_payload['count'] for theme_id, theme_payload in payload['indexes']['themes'].items()})
    print('theme counts:')
    for theme_id, count in theme_counts.items():
        print(f'  - {theme_id}: {count}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
