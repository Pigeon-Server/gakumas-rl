"""Generate effect coverage matrix for items, drinks, and challenge items."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gakumas_rl.data import MasterDataRepository
from gakumas_rl.produce_item_interpreter import ProduceItemInterpreter, SUPPORTED_RUNTIME_PRODUCE_ITEM_PHASES, SUPPORTED_SCENARIO_TAGS

DOC_PATH = ROOT / 'docs' / 'ITEM_DRINK_CHALLENGE_COVERAGE_MATRIX.md'
UNIMPLEMENTED_DOC_PATH = ROOT / 'docs' / 'ITEM_DRINK_CHALLENGE_UNIMPLEMENTED.md'

_STATUS_ORDER = {'已实现': 0, '部分实现': 1, '未实现': 2}

PARTIAL_PRODUCE_EFFECT_TYPES = {
    'ProduceEffectType_ProduceReward',
    'ProduceEffectType_ProduceRewardSet',
    'ProduceEffectType_ExamStatusEnchant',
    'ProduceEffectType_ExamPermanentLessonStatusEnchant',
}

PARTIAL_EXAM_EFFECT_TYPES = {
    'ProduceExamEffectType_ExamCardCreateSearch',
    'ProduceExamEffectType_ExamCardMove',
    'ProduceExamEffectType_ExamForcePlayCardSearch',
    'ProduceExamEffectType_ExamSearchPlayCardStaminaConsumptionChange',
}


def _function_tokens(source_path: Path, func_name: str, prefix: str) -> set[str]:
    text = source_path.read_text(encoding='utf-8')
    match = re.search(rf"def {re.escape(func_name)}\(", text)
    if not match:
        return set()
    start = match.start()
    next_match = re.search(r'\n    def ', text[match.end():])
    end = match.end() + next_match.start() if next_match else len(text)
    body = text[start:end]
    return set(re.findall(rf'{re.escape(prefix)}[A-Za-z0-9_]+', body))


def _file_tokens(source_path: Path, prefix: str) -> set[str]:
    text = source_path.read_text(encoding='utf-8')
    return set(re.findall(rf'{re.escape(prefix)}[A-Za-z0-9_]+', text))


HANDLED_PRODUCE_EFFECT_TYPES = _function_tokens(ROOT / 'gakumas_rl' / 'produce_runtime.py', '_apply_produce_effect', 'ProduceEffectType_')
if "'128'" in (ROOT / 'gakumas_rl' / 'produce_runtime.py').read_text(encoding='utf-8'):
    HANDLED_PRODUCE_EFFECT_TYPES.add('128')
HANDLED_EXAM_EFFECT_TYPES = _file_tokens(ROOT / 'gakumas_rl' / 'exam_runtime.py', 'ProduceExamEffectType_')


def _markdown_escape(value: object) -> str:
    return str(value).replace('|', r'\|').replace('\n', ' ')


def _status_label(statuses: list[str]) -> str:
    if statuses and all(status == '已实现' for status in statuses):
        return '已实现'
    if '部分实现' in statuses or '已实现' in statuses:
        return '部分实现'
    return '未实现'


def _classify_item_effect(
    item_row: dict,
    item_effect_row: dict,
    interpreter: ProduceItemInterpreter,
    produce_effect_map: dict[str, dict],
) -> tuple[str, str]:
    trigger = interpreter.resolve_item(str(item_row.get('id') or ''))
    trigger_spec = trigger.trigger if trigger is not None else None
    if trigger_spec is not None:
        if trigger_spec.phase_type not in SUPPORTED_RUNTIME_PRODUCE_ITEM_PHASES:
            return '未实现', f'phase {trigger_spec.phase_type} 当前 runtime 未分发'
        if trigger_spec.scenario_tag not in SUPPORTED_SCENARIO_TAGS:
            return '未实现', f'scenario tag {trigger_spec.scenario_tag} 未支持'
    if int(item_row.get('fireInterval') or 0) > 0:
        fire_interval_note = '；fireInterval 已做近似实现'
    else:
        fire_interval_note = ''
    effect_type = str(item_effect_row.get('effectType') or '')
    if effect_type == 'ProduceItemEffectType_ExamStatusEnchant':
        return ('部分实现' if fire_interval_note else '已实现'), f'结构化附魔已接入{fire_interval_note}'
    if effect_type != 'ProduceItemEffectType_ProduceEffect':
        return '未实现', f'未知 wrapper type {effect_type}'
    produce_effect = produce_effect_map.get(str(item_effect_row.get('produceEffectId') or '')) or {}
    produce_effect_type = str(produce_effect.get('produceEffectType') or '')
    if produce_effect_type not in HANDLED_PRODUCE_EFFECT_TYPES:
        return '未实现', f'{produce_effect_type} 未进入 _apply_produce_effect'
    if (
        produce_effect_type in {'ProduceEffectType_ProduceReward', 'ProduceEffectType_ProduceRewardSet'}
        and str(produce_effect.get('produceResourceType') or '') == 'ProduceResourceType_ProduceItem'
        and not produce_effect.get('produceRewards')
    ):
        return '部分实现', '直填 ProduceItem 选择池缺少结构化池定义'
    if produce_effect_type in PARTIAL_PRODUCE_EFFECT_TYPES:
        return '部分实现', f'{produce_effect_type} 仍有 battle scope/奖励池近似'
    if fire_interval_note:
        return '部分实现', fire_interval_note.lstrip('；')
    return '已实现', f'{produce_effect_type} 已接入'


def _classify_drink_effect(drink_effect_row: dict, exam_effect_map: dict[str, dict]) -> tuple[str, str, str]:
    exam_effect = exam_effect_map.get(str(drink_effect_row.get('produceExamEffectId') or '')) or {}
    exam_effect_type = str(exam_effect.get('effectType') or '')
    if exam_effect_type not in HANDLED_EXAM_EFFECT_TYPES:
        return '未实现', exam_effect_type, f'{exam_effect_type} 未进入 _apply_exam_effect'
    if exam_effect_type in PARTIAL_EXAM_EFFECT_TYPES:
        return '部分实现', exam_effect_type, f'{exam_effect_type} 仍在审计清单中'
    return '已实现', exam_effect_type, f'{exam_effect_type} 已接入'


def _classify_challenge_item(
    item_row: dict,
    interpreter: ProduceItemInterpreter,
    item_effect_map: dict[str, dict],
    produce_effect_map: dict[str, dict],
) -> tuple[str, str]:
    statuses: list[str] = []
    reasons: list[str] = []
    trigger = interpreter.resolve_item(str(item_row.get('id') or ''))
    trigger_spec = trigger.trigger if trigger is not None else None
    if trigger_spec is not None:
        if trigger_spec.phase_type not in SUPPORTED_RUNTIME_PRODUCE_ITEM_PHASES:
            return '未实现', f'phase {trigger_spec.phase_type} 当前 runtime 未分发'
        if trigger_spec.scenario_tag not in SUPPORTED_SCENARIO_TAGS:
            return '未实现', f'scenario tag {trigger_spec.scenario_tag} 未支持'
    for item_effect_id in item_row.get('produceItemEffectIds', []) or []:
        status, reason = _classify_item_effect(item_row, item_effect_map[str(item_effect_id)], interpreter, produce_effect_map)
        statuses.append(status)
        reasons.append(reason)
    item_status = _status_label(statuses)
    if int(item_row.get('fireInterval') or 0) > 0 and item_status == '已实现':
        item_status = '部分实现'
        reasons.append('fireInterval 为近似实现')
    if not reasons:
        reasons.append('无可解析 effect')
    return item_status, '；'.join(dict.fromkeys(reasons))


def main() -> None:
    repo = MasterDataRepository(root_dir=ROOT)
    interpreter = ProduceItemInterpreter(repo)
    item_rows = repo.load_table('ProduceItem').rows
    item_effect_rows = repo.load_table('ProduceItemEffect').rows
    drink_effect_rows = repo.load_table('ProduceDrinkEffect').rows
    produce_effect_map = {str(row.get('id') or ''): row for row in repo.load_table('ProduceEffect').rows}
    item_effect_map = {str(row.get('id') or ''): row for row in item_effect_rows}
    exam_effect_map = repo.exam_effect_map

    item_effect_records = []
    for item_row in item_rows:
        for item_effect_id in item_row.get('produceItemEffectIds', []) or []:
            item_effect_row = item_effect_map.get(str(item_effect_id))
            if not item_effect_row:
                continue
            status, reason = _classify_item_effect(item_row, item_effect_row, interpreter, produce_effect_map)
            produce_effect = produce_effect_map.get(str(item_effect_row.get('produceEffectId') or '')) or {}
            trigger = interpreter.resolve_item(str(item_row.get('id') or ''))
            item_effect_records.append(
                {
                    'item_id': str(item_row.get('id') or ''),
                    'item_effect_id': str(item_effect_row.get('id') or ''),
                    'item_effect_type': str(item_effect_row.get('effectType') or ''),
                    'produce_effect_type': str(produce_effect.get('produceEffectType') or ''),
                    'trigger_phase': trigger.trigger.phase_type if trigger and trigger.trigger else 'static',
                    'status': status,
                    'reason': reason,
                }
            )

    effect_type_summary: dict[str, Counter] = defaultdict(Counter)
    for record in item_effect_records:
        effect_type_summary[record['item_effect_type']][record['status']] += 1

    drink_effect_records = []
    drinks_by_effect: dict[str, list[str]] = defaultdict(list)
    for drink_row in repo.produce_drinks.rows:
        for effect_id in drink_row.get('produceDrinkEffectIds', []) or []:
            drinks_by_effect[str(effect_id)].append(str(drink_row.get('id') or ''))
    for row in drink_effect_rows:
        status, exam_effect_type, reason = _classify_drink_effect(row, exam_effect_map)
        drink_effect_records.append(
            {
                'id': str(row.get('id') or ''),
                'status': status,
                'exam_effect_type': exam_effect_type,
                'reason': reason,
                'drinks': ', '.join(sorted(drinks_by_effect.get(str(row.get('id') or ''), []))) or '-',
            }
        )

    challenge_records = []
    for item_row in item_rows:
        if not item_row.get('isChallenge'):
            continue
        status, reason = _classify_challenge_item(item_row, interpreter, item_effect_map, produce_effect_map)
        trigger = interpreter.resolve_item(str(item_row.get('id') or ''))
        trigger_phase = trigger.trigger.phase_type if trigger and trigger.trigger else 'static'
        challenge_records.append(
            {
                'id': str(item_row.get('id') or ''),
                'name': str(item_row.get('name') or ''),
                'trigger_phase': trigger_phase,
                'status': status,
                'reason': reason,
            }
        )

    challenge_counter = Counter(record['status'] for record in challenge_records)
    drink_counter = Counter(record['status'] for record in drink_effect_records)
    unimplemented_item_effect_records = [record for record in item_effect_records if record['status'] == '未实现']
    unimplemented_drink_effect_records = [record for record in drink_effect_records if record['status'] == '未实现']
    unimplemented_challenge_records = [record for record in challenge_records if record['status'] == '未实现']

    lines: list[str] = []
    lines.append('# Item / Drink / Challenge Coverage Matrix')
    lines.append('')
    lines.append('本表基于当前代码自动生成，判定依据是主数据、`produce item interpreter` 支持面，以及 `produce_runtime.py` / `exam_runtime.py` 的实际 handler。')
    lines.append('')
    lines.append('## Summary')
    lines.append('')
    lines.append(f"- `ProduceItemEffectType`: {sum(effect_type_summary[key]['已实现'] for key in effect_type_summary)} 行已实现，{sum(effect_type_summary[key]['部分实现'] for key in effect_type_summary)} 行部分实现，{sum(effect_type_summary[key]['未实现'] for key in effect_type_summary)} 行未实现")
    lines.append(f"- `ProduceDrinkEffect`: {drink_counter['已实现']} 行已实现，{drink_counter['部分实现']} 行部分实现，{drink_counter['未实现']} 行未实现")
    lines.append(f"- `challenge item`: {challenge_counter['已实现']} 个已实现，{challenge_counter['部分实现']} 个部分实现，{challenge_counter['未实现']} 个未实现")
    lines.append('')
    lines.append('## ProduceItemEffectType')
    lines.append('')
    lines.append('| ProduceItemEffectType | 已实现 | 部分实现 | 未实现 | 总计 | 总体判定 |')
    lines.append('| --- | ---: | ---: | ---: | ---: | --- |')
    for effect_type in sorted(effect_type_summary):
        counter = effect_type_summary[effect_type]
        overall = _status_label(
            ['已实现'] * counter['已实现']
            + ['部分实现'] * counter['部分实现']
            + ['未实现'] * counter['未实现']
        )
        total = counter['已实现'] + counter['部分实现'] + counter['未实现']
        lines.append(
            f"| {_markdown_escape(effect_type)} | {counter['已实现']} | {counter['部分实现']} | {counter['未实现']} | {total} | {overall} |"
        )
    lines.append('')
    lines.append('## ProduceDrinkEffect')
    lines.append('')
    lines.append('| ProduceDrinkEffect | ProduceExamEffectType | 状态 | 关联饮料 | 说明 |')
    lines.append('| --- | --- | --- | --- | --- |')
    for record in sorted(drink_effect_records, key=lambda item: (_STATUS_ORDER[item['status']], item['id'])):
        lines.append(
            f"| {_markdown_escape(record['id'])} | {_markdown_escape(record['exam_effect_type'])} | {record['status']} | {_markdown_escape(record['drinks'])} | {_markdown_escape(record['reason'])} |"
        )
    lines.append('')
    lines.append('## Challenge Item')
    lines.append('')
    lines.append('| challenge item | 名称 | trigger phase | 状态 | 说明 |')
    lines.append('| --- | --- | --- | --- | --- |')
    for record in sorted(challenge_records, key=lambda item: (_STATUS_ORDER[item['status']], item['id'])):
        lines.append(
            f"| {_markdown_escape(record['id'])} | {_markdown_escape(record['name'])} | {_markdown_escape(record['trigger_phase'])} | {record['status']} | {_markdown_escape(record['reason'])} |"
        )
    lines.append('')
    lines.append('## Notes')
    lines.append('')
    lines.append('- `部分实现` 主要来自三类问题：lesson battle 仍是抽象步骤、`fireInterval` 只是按 phase 次数近似、以及 `ProduceReward(Set)` 的部分 item 选择池缺少结构化来源。')
    if any(counter['未实现'] for counter in effect_type_summary.values()) or drink_counter['未实现'] or challenge_counter['未实现']:
        lines.append('- `未实现` 主要集中在当前 runtime 没有 phase 的路径，以及尚未接入的 effect handler。')
    else:
        lines.append('- 当前自动覆盖统计里，`未实现` 项已经清零；剩余缺口都归在 `部分实现`。')
    lines.append('- 饮料侧的 `ExamCardCreateSearch / ExamCardMove / ExamForcePlayCardSearch / ExamSearchPlayCardStaminaConsumptionChange` 仍在审计清单里，所以保守标为 `部分实现`。')
    DOC_PATH.write_text('\n'.join(lines) + '\n', encoding='utf-8')

    unimplemented_lines: list[str] = []
    unimplemented_lines.append('# Unimplemented Item / Drink / Challenge List')
    unimplemented_lines.append('')
    unimplemented_lines.append('本清单由覆盖矩阵脚本自动生成，只保留当前判定为 `未实现` 的项。')
    unimplemented_lines.append('')
    unimplemented_lines.append('## Summary')
    unimplemented_lines.append('')
    unimplemented_lines.append(f'- `ProduceItemEffect`: {len(unimplemented_item_effect_records)} 行')
    unimplemented_lines.append(f'- `ProduceDrinkEffect`: {len(unimplemented_drink_effect_records)} 行')
    unimplemented_lines.append(f'- `challenge item`: {len(unimplemented_challenge_records)} 个')
    unimplemented_lines.append('')
    unimplemented_lines.append('## ProduceItemEffect')
    unimplemented_lines.append('')
    if unimplemented_item_effect_records:
        unimplemented_lines.append('| item | item effect id | wrapper type | produce effect type | trigger phase | 说明 |')
        unimplemented_lines.append('| --- | --- | --- | --- | --- | --- |')
        for record in sorted(unimplemented_item_effect_records, key=lambda item: (item['item_id'], item['item_effect_id'])):
            unimplemented_lines.append(
                f"| {_markdown_escape(record['item_id'])} | {_markdown_escape(record['item_effect_id'])} | {_markdown_escape(record['item_effect_type'])} | {_markdown_escape(record['produce_effect_type'])} | {_markdown_escape(record['trigger_phase'])} | {_markdown_escape(record['reason'])} |"
            )
    else:
        unimplemented_lines.append('无。')
    unimplemented_lines.append('')
    unimplemented_lines.append('## ProduceDrinkEffect')
    unimplemented_lines.append('')
    if unimplemented_drink_effect_records:
        unimplemented_lines.append('| ProduceDrinkEffect | ProduceExamEffectType | 关联饮料 | 说明 |')
        unimplemented_lines.append('| --- | --- | --- | --- |')
        for record in sorted(unimplemented_drink_effect_records, key=lambda item: item['id']):
            unimplemented_lines.append(
                f"| {_markdown_escape(record['id'])} | {_markdown_escape(record['exam_effect_type'])} | {_markdown_escape(record['drinks'])} | {_markdown_escape(record['reason'])} |"
            )
    else:
        unimplemented_lines.append('无。')
    unimplemented_lines.append('')
    unimplemented_lines.append('## Challenge Item')
    unimplemented_lines.append('')
    if unimplemented_challenge_records:
        unimplemented_lines.append('| challenge item | 名称 | trigger phase | 说明 |')
        unimplemented_lines.append('| --- | --- | --- | --- |')
        for record in sorted(unimplemented_challenge_records, key=lambda item: item['id']):
            unimplemented_lines.append(
                f"| {_markdown_escape(record['id'])} | {_markdown_escape(record['name'])} | {_markdown_escape(record['trigger_phase'])} | {_markdown_escape(record['reason'])} |"
            )
    else:
        unimplemented_lines.append('无。')
    UNIMPLEMENTED_DOC_PATH.write_text('\n'.join(unimplemented_lines) + '\n', encoding='utf-8')
    print(DOC_PATH)
    print(UNIMPLEMENTED_DOC_PATH)


if __name__ == '__main__':
    main()
