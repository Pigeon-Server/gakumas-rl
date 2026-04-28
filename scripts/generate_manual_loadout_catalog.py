"""Generate static catalog data for the manual exam setup WebUI."""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gakumas_rl.data import MasterDataRepository
from gakumas_rl.service import SCENARIO_ALIASES


def _clean_text(value: str) -> str:
    return ' '.join(str(value or '').replace('<nobr>', '').replace('</nobr>', '').split()).strip()


def _description_text(localization_row: dict[str, Any] | None) -> str:
    if not localization_row:
        return ''
    lines: list[str] = []
    for item in localization_row.get('produceDescriptions', []) or []:
        text = _clean_text(item.get('text') or '')
        if text and text not in lines:
            lines.append(text)
    return ' / '.join(lines[:6])


def _short_enum(value: str | None) -> str:
    raw = str(value or '').strip()
    if not raw:
        return ''
    for prefix in (
        'ProduceExamEffectType_',
        'ProduceCardCategory_',
        'ProduceCardRarity_',
        'ProducePlanType_',
        'IdolCardRarity_',
    ):
        if raw.startswith(prefix):
            return raw[len(prefix):]
    return raw


def _display_card_name(value: str) -> str:
    text = _clean_text(value)
    stripped = re.sub(r'[+＋]+$', '', text).strip()
    return stripped or text


def build_catalog() -> dict[str, Any]:
    repository = MasterDataRepository()
    cards: list[dict[str, Any]] = []
    card_groups: list[dict[str, Any]] = []
    for card_id, rows in sorted(repository.produce_cards.by_id.items()):
        if not card_id:
            continue
        visible_rows = [
            row
            for row in sorted(rows, key=lambda item: int(item.get('upgradeCount') or 0))
            if not row.get('libraryHidden')
        ]
        if not visible_rows:
            continue
        base_row = visible_rows[0]
        base_name = _display_card_name(repository.card_name(base_row))
        base_raw_name = _display_card_name(repository.raw_card_name(base_row))
        loc = repository.produce_card_localization.get(card_id, {})
        variants: list[dict[str, Any]] = []
        group_effect_types: list[str] = []
        for row in visible_rows:
            if row.get('libraryHidden'):
                continue
            upgrade_count = int(row.get('upgradeCount') or 0)
            effect_types = [_short_enum(value) for value in repository.card_axis_effect_types(row)]
            for effect_type in effect_types:
                if effect_type not in group_effect_types:
                    group_effect_types.append(effect_type)
            variant = {
                'key': f'{card_id}@{upgrade_count}',
                'card_id': card_id,
                'upgrade_count': upgrade_count,
                'name': base_name,
                'raw_name': repository.raw_card_name(row),
                'base_raw_name': base_raw_name,
                'category': _short_enum(row.get('category')),
                'rarity': _short_enum(row.get('rarity')),
                'plan_type': _short_enum(row.get('planType')),
                'effect_types': effect_types,
                'evaluation': float(row.get('evaluation') or 0.0),
                'description': _description_text(loc),
                'image_path': f'skill_card/{card_id}_{upgrade_count}.png',
            }
            cards.append(variant)
            variants.append(variant)
        card_groups.append(
            {
                'key': card_id,
                'card_id': card_id,
                'name': base_name,
                'raw_name': base_raw_name,
                'category': _short_enum(base_row.get('category')),
                'rarity': _short_enum(base_row.get('rarity')),
                'plan_type': _short_enum(base_row.get('planType')),
                'effect_types': group_effect_types,
                'description': _description_text(loc),
                'variants': variants,
            }
        )
    drinks: list[dict[str, Any]] = []
    for row in sorted(repository.produce_drinks.rows, key=lambda item: repository.drink_name(item)):
        if row.get('libraryHidden'):
            continue
        drink_id = str(row.get('id') or '')
        loc = repository.produce_drink_localization.get(drink_id, {})
        drinks.append(
            {
                'id': drink_id,
                'name': repository.drink_name(row),
                'raw_name': repository.raw_drink_name(row),
                'rarity': _short_enum(row.get('rarity')),
                'plan_type': _short_enum(row.get('planType')),
                'effect_types': [_short_enum(value) for value in repository.drink_axis_effect_types(row)],
                'description': _description_text(loc),
            }
        )
    items: list[dict[str, Any]] = []
    for row in sorted(repository.produce_items.rows, key=lambda item: repository.item_name(item)):
        item_id = str(row.get('id') or '')
        if not item_id:
            continue
        loc = repository.produce_item_localization.get(item_id, {})
        items.append(
            {
                'id': item_id,
                'name': repository.item_name(row),
                'raw_name': repository.raw_item_name(row),
                'description': _description_text(loc),
                'is_exam_effect': bool(row.get('isExamEffect')),
                'is_challenge': bool(row.get('isChallenge')),
            }
        )
    idol_rows = [
        row
        for row in repository.load_table('IdolCard').rows
        if str(row.get('id') or '')
        and str(row.get('characterId') or '')
        and str(row.get('produceCardId') or '')
    ]
    idols = sorted(
        [
            {
                'id': str(row.get('id') or ''),
                'name': str((repository.load_localization('IdolCard').get(str(row.get('id') or ''), {}) or {}).get('name') or row.get('name') or row.get('id')),
                'character_id': str(row.get('characterId') or ''),
                'rarity': _short_enum(row.get('rarity')),
                'plan_type': _short_enum(row.get('planType')),
                'exam_effect_type': _short_enum(row.get('examEffectType')),
            }
            for row in idol_rows
        ],
        key=lambda item: (item['name'], item['id']),
    )
    scenarios = []
    for alias, scenario_id in sorted(SCENARIO_ALIASES.items()):
        scenario = repository.build_scenario(scenario_id)
        scenarios.append(
            {
                'alias': alias,
                'id': scenario.scenario_id,
                'name': repository.produce_name(scenario.scenario_id),
                'stages': list(scenario.audition_sequence),
            }
        )
    return {
        'generated_at': Path(__file__).resolve().as_posix(),
        'scenarios': scenarios,
        'idols': idols,
        'cards': cards,
        'card_groups': card_groups,
        'drinks': drinks,
        'items': items,
    }


def main() -> int:
    output = ROOT / 'gakumas_rl' / 'webapp' / 'manual_loadout_catalog.js'
    catalog = build_catalog()
    output.write_text(
        'window.GAKUMAS_MANUAL_LOADOUT_CATALOG = ' + json.dumps(catalog, ensure_ascii=False) + ';\n',
        encoding='utf-8',
    )
    print(f'Wrote {output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
