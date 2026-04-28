"""共享局面描述模块 — 供 llm_reward 和 llm_player 复用。

提供两层 API：
- extract_state_context / extract_action_list_context — 纯数据提取，返回 dict
- build_state_snapshot / format_action_list — 渲染为字符串（通过 Jinja2 模板）
"""

from __future__ import annotations

import copy
import re
from collections import Counter
from typing import Any

from .exam_runtime import LESSON_EFFECT_TYPES

# ---------------------------------------------------------------------------
# 效果类型 → 中文短名映射
# ---------------------------------------------------------------------------

EFFECT_TYPE_SHORT_NAMES: dict[str, str] = {
    'ProduceExamEffectType_ExamLesson': '打分',
    'ProduceExamEffectType_ExamLessonFix': '固定打分',
    'ProduceExamEffectType_ExamReview': '集中',
    'ProduceExamEffectType_ExamBlock': '元気',
    'ProduceExamEffectType_ExamBlockFix': '元気',
    'ProduceExamEffectType_ExamParameterBuff': '好調',
    'ProduceExamEffectType_ExamLessonBuff': 'パラメータ上昇量増加',
    'ProduceExamEffectType_ExamCardPlayAggressive': '好印象',
    'ProduceExamEffectType_ExamConcentration': '強気',
    'ProduceExamEffectType_ExamPreservation': '温存',
    'ProduceExamEffectType_ExamFullPowerPoint': '全力値',
    'ProduceExamEffectType_ExamGimmickSleepy': '弱気',
    'ProduceExamEffectType_ExamGimmickSlump': 'スランプ',
    'ProduceExamEffectType_ExamPanic': 'パニック',
    'ProduceExamEffectType_ExamStaminaConsumptionDown': '体力消耗减',
    'ProduceExamEffectType_ExamStaminaConsumptionAdd': '体力消耗增',
    'ProduceExamEffectType_ExamPlayableValueAdd': 'スキルカード使用数+',
    'ProduceExamEffectType_ExamCardDraw': '抽牌',
    'ProduceExamEffectType_ExamStaminaRecovery': '回体',
    'ProduceExamEffectType_ExamEnthusiastic': '熱意',
    'ProduceExamEffectType_ExamAntiDebuff': '免疫负面',
    'ProduceExamEffectType_ExamBlockValueMultiple': '元気倍率',
    'ProduceExamEffectType_ExamLessonValueMultiple': '打分倍率',
    'ProduceExamEffectType_ExamActiveSkillForbidden': 'アクティブ封印',
    'ProduceExamEffectType_ExamMentalSkillForbidden': 'メンタル封印',
    'ProduceExamEffectType_ExamParameterBuffMultiplePerTurn': '絶好調',
    'ProduceExamEffectType_ExamParameterBuffPerSearchCount': '好調(按检索)',
    'ProduceExamEffectType_ExamParameterBuffDependLessonBuff': '好調(按パラメータ上昇量増加)',
    'ProduceExamEffectType_ExamCardSearch': '检索',
    'ProduceExamEffectType_ExamCardMove': '移牌',
    'ProduceExamEffectType_ExamCardUpgrade': '强化',
    'ProduceExamEffectType_ExamCardGenerate': '生成',
    'ProduceExamEffectType_ExamCardRemove': '除外',
    'ProduceExamEffectType_ExamStatusEnchant': '附魔',
    'ProduceExamEffectType_ExamTrigger': '触发器',
    'ProduceExamEffectType_ExamExtraTurn': '追加回合',
    'ProduceExamEffectType_ExamOverPreservation': '超温存',
    'ProduceExamEffectType_ExamLessonDependStamina': '打分(按体力)',
    'ProduceExamEffectType_ExamLessonFullPowerPoint': '打分(按全力値)',
    'ProduceExamEffectType_ExamLessonDependAggressiveAndSearchCount': '打分(好印象+检索)',
    'ProduceExamEffectType_ExamLessonDependReviewAndSearchCount': '打分(集中+检索)',
    'ProduceExamEffectType_ExamMultipleEnthusiasticLesson': '打分(熱意倍率)',
    'ProduceExamEffectType_ExamMultipleConcentrationLesson': '打分(強気倍率)',
    'ProduceExamEffectType_ExamMultipleFullPowerLesson': '打分(全力倍率)',
}

CATEGORY_SHORT: dict[str, str] = {
    'ProduceCardCategory_ActiveSkill': 'アクティブ',
    'ProduceCardCategory_MentalSkill': 'メンタル',
    'ProduceCardCategory_Trouble': 'トラブル',
}

RARITY_SHORT: dict[str, str] = {
    'IdolCardRarity_N': 'N',
    'IdolCardRarity_R': 'R',
    'IdolCardRarity_Sr': 'SR',
    'IdolCardRarity_SSR': 'SSR',
    'IdolCardRarity_Ssr': 'SSR',
}

PLAN_TYPE_DISPLAY: dict[str, tuple[str, str]] = {
    'ProducePlanType_Plan1': ('好調 / 集中', '「好調」や「集中」を活用して育成するプランです。'),
    'ProducePlanType_Plan2': ('好印象 / やる気', '「好印象」や「やる気」を活用して育成するプランです。'),
    'ProducePlanType_Plan3': ('全力 / 強気 / 温存', '「全力」「強気」「温存」の指針を活用して育成するプランです。'),
    'ProducePlanType_Common': ('', ''),
}

STANCE_NAMES: dict[str, str] = {
    'neutral': '中立',
    'concentration': '強気',
    'preservation': '温存',
    'full_power': '全力',
}

BATTLE_KIND_NAMES: dict[str, str] = {
    'exam': '試験',
    'audition': 'オーディション',
    'lesson': 'レッスン',
}

CLEAR_STATE_NAMES: dict[str, str] = {
    'ongoing': '進行中',
    'cleared': '目標達成',
    'perfect': 'パーフェクト',
}

RESOURCE_SHORT_NAMES: dict[str, str] = {
    'parameter_buff': '好調',
    'parameter_buff_multiple_per_turn': '絶好調',
    'review': '集中',
    'aggressive': '好印象',
    'block': '元気',
    'lesson_buff': 'パラメータ上昇量増加',
    'enthusiastic': '熱意',
    'full_power_point': '全力値',
    'anti_debuff': '免疫负面',
    'sleepy': '弱気',
    'panic': 'パニック',
    'slump': 'スランプ',
    'active_skill_forbidden': 'アクティブ封印',
    'mental_skill_forbidden': 'メンタル封印',
}

CARD_DESTINATION_NAMES: dict[str, str] = {
    'deck_first': '牌堆顶',
    'deck_last': '牌堆底',
    'deck_random': '牌堆',
    'grave': '弃牌',
    'hand': '回手',
    'hold': '保留',
    'lost': '除外',
}

SOURCE_NAMES: dict[str, str] = {
    'card': '卡牌',
    'drink': 'Pドリンク',
    'produce': '培养来源',
    'produce_item': 'Pアイテム',
    'gimmick': '応援/トラブル',
    'test': '测试',
}

TURN_COLOR_DISPLAY_NAMES: dict[str, str] = {
    'Vocal': 'ボーカル',
    'Dance': 'ダンス',
    'Visual': 'ビジュアル',
}

TRIGGER_PHASE_NAMES: dict[str, str] = {
    'ProduceExamPhaseType_ExamStartTurn': '回合开始',
    'ProduceExamPhaseType_ExamEndTurn': '回合结束',
    'ProduceExamPhaseType_ExamTurnSkip': '空过回合',
    'ProduceExamPhaseType_ExamTurnTimer': '指定回合',
    'ProduceExamPhaseType_ExamTurnInterval': '回合间隔',
    'ProduceExamPhaseType_StartExamPlay': '开始出牌流程',
    'ProduceExamPhaseType_StartPlay': '开始出牌',
    'ProduceExamPhaseType_ExamCardPlay': '出牌时',
    'ProduceExamPhaseType_ExamCardPlayAfter': '出牌后',
    'ProduceExamPhaseType_ExamSearchCardPlay': '检索牌出牌',
    'ProduceExamPhaseType_ExamStatusChange': '状态变化',
    'ProduceExamPhaseType_ExamStaminaReduceCard': '出牌扣体力',
    'ProduceExamPhaseType_ExamBuffConsume': '消耗强化状态',
    'ProduceExamPhaseType_ExamPlayTurnCountInterval': '本回合出牌间隔',
    'ProduceExamPhaseType_ExamPlayCountInterval': '累计出牌间隔',
    'ProduceExamPhaseType_ExamPlayCountIntervalAfter': '累计出牌后间隔',
}

PREVIEW_SHARED_FIELDS: set[str] = {
    'repository',
    'scenario',
    'loadout',
    'reward_mode',
    'stage_type',
    'exam_setting',
    'card_searches',
    'grow_effects',
    'card_status_enchants',
    'exam_gimmicks',
    'random_pools',
    'card_pools',
}

PREVIEW_RESOURCE_KEYS: tuple[str, ...] = (
    'parameter_buff',
    'parameter_buff_multiple_per_turn',
    'review',
    'aggressive',
    'block',
    'lesson_buff',
    'enthusiastic',
    'full_power_point',
    'anti_debuff',
    'sleepy',
    'panic',
    'slump',
    'active_skill_forbidden',
    'mental_skill_forbidden',
)


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------

def _short_effect_type(effect_type: str) -> str:
    """把长 enum 映射为中文短名，尽量避免把内部 id 暴露给 LLM。"""
    if effect_type in EFFECT_TYPE_SHORT_NAMES:
        return EFFECT_TYPE_SHORT_NAMES[effect_type]
    raw = str(effect_type or '')
    if not raw or raw.isdigit():
        return ''
    if raw.startswith('ProduceExamEffectType_'):
        tail = raw.removeprefix('ProduceExamEffectType_')
        heuristics = [
            ('ExamLesson', '打分'),
            ('ExamBlock', '元気'),
            ('ExamReview', '集中'),
            ('ExamCardPlayAggressive', '好印象'),
            ('ExamParameterBuff', '好調'),
            ('ExamLessonBuff', 'パラメータ上昇量増加'),
            ('ExamEnthusiastic', '熱意'),
            ('ExamFullPowerPoint', '全力値'),
            ('ExamConcentration', '強気'),
            ('ExamPreservation', '温存'),
            ('Search', '检索'),
            ('UseCardCount', 'スキルカード使用数'),
            ('PlayCardCount', 'スキルカード使用数'),
            ('PlayCount', 'スキルカード使用数'),
            ('Consumption', '消耗'),
            ('Stamina', '体力'),
            ('Multiple', '倍率'),
            ('Depend', '按条件'),
        ]
        parts: list[str] = []
        for needle, label in heuristics:
            if needle in tail and label not in parts:
                parts.append(label)
        if parts:
            return '·'.join(parts[:4])
        cleaned = re.sub(r'^Exam', '', tail)
        cleaned = re.sub(r'([a-z])([A-Z])', r'\1 \2', cleaned).strip()
        return cleaned or ''
    parts = raw.rsplit('_', 1)
    return parts[-1] if len(parts) > 1 else raw


def _short_category(category: str) -> str:
    return CATEGORY_SHORT.get(category, category.rsplit('_', 1)[-1])


def _format_number(value: Any) -> str:
    numeric = float(value or 0)
    if abs(numeric - round(numeric)) < 1e-6:
        return str(int(round(numeric)))
    return f'{numeric:.1f}'.rstrip('0').rstrip('.')


def _signed_number(value: float) -> str:
    return f'+{_format_number(value)}' if value >= 0 else f'-{_format_number(abs(value))}'


def _clean_description_text(text: str) -> str:
    cleaned = (
        str(text or '')
        .replace('<nobr>', '')
        .replace('</nobr>', '')
        .replace('<br>', '；')
        .replace('<br/>', '；')
        .replace('<br />', '；')
        .replace('\t', ' ')
    )
    cleaned = re.sub(r'\s+', ' ', cleaned)
    cleaned = re.sub(r'\s*([,，。；：、）])', r'\1', cleaned)
    cleaned = re.sub(r'([（(])\s+', r'\1', cleaned)
    cleaned = re.sub(r'（\s*）', '', cleaned)
    cleaned = re.sub(r'\(\s*\)', '', cleaned)
    cleaned = re.sub(r'[；]{2,}', '；', cleaned)
    return cleaned.strip()


def _humanize_runtime_text(text: str) -> str:
    cleaned = _clean_description_text(text)
    if not cleaned:
        return ''
    replacements = (
        ('干劲', 'やる気'),
        ('好调', '好調'),
        ('绝好调', '絶好調'),
        ('元气', '元気'),
        ('强气', '強気'),
        ('弱气', '弱気'),
        ('随机', 'パニック'),
        ('低谷', 'スランプ'),
        ('热意', '熱意'),
        ('全力值', '全力値'),
        ('数值变为打分上升', '转为打分'),
        ('数值变为好印象增加', '转为好印象'),
        ('体力回复', '回复体力'),
        ('技能卡出牌次数+', 'スキルカード使用数+'),
        ('技能卡使用数追加+', 'スキルカード使用数+'),
        ('回合追加+', '额外回合+'),
        ('课程结束为止，', '直到课程结束，'),
        ('以后，', '之后，'),
    )
    for before, after in replacements:
        cleaned = cleaned.replace(before, after)
    cleaned = re.sub(r'每使用([0-9]+)次技能卡，', r'每使用\1次技能卡时，', cleaned)
    cleaned = re.sub(r'，([^，；]+?)的([0-9]+%)转为(打分|好印象)', r'，将\1的\2转为\3', cleaned)
    cleaned = re.sub(r'[；]{2,}', '；', cleaned)
    return cleaned.strip('； ')


def _description_text(entries: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None) -> str:
    if not entries:
        return ''
    parts: list[str] = []
    for raw_entry in entries:
        entry = raw_entry or {}
        text = _clean_description_text(str(entry.get('text') or ''))
        if not text:
            continue
        desc_type = str(entry.get('produceDescriptionType') or '')
        if parts:
            prev = parts[-1]
            if (
                desc_type in {'ProduceDescriptionType_ProduceDescription', 'ProduceDescriptionType_ProduceCard'}
                and prev[-1] not in '，。；：、（('
            ):
                parts.append('；')
        parts.append(text)
    combined = ''.join(parts)
    combined = re.sub(r'([0-9一二三四五六七八九十])课程中仅限', r'\1；课程中仅限', combined)
    combined = re.sub(r'([0-9一二三四五六七八九十])回合(课程中仅限)', r'\1回合；\2', combined)
    combined = re.sub(r'([0-9一二三四五六七八九十])次(课程中仅限)', r'\1次；\2', combined)
    return _clean_description_text(combined)


def _localized_name(repo: Any, table_name: str, item_id: str, fallback_name: str = '') -> str:
    if repo is None or not item_id:
        return fallback_name or item_id
    row = repo.load_localization(table_name).get(str(item_id), {})
    return str(row.get('name') or fallback_name or item_id)


def _raw_name(item: dict[str, Any] | None, item_id: str = '') -> str:
    row = item or {}
    return str(row.get('name') or item_id or row.get('id') or '')


def _localized_description(repo: Any, table_name: str, item_id: str, fallback_entries: Any = None) -> str:
    if repo is None or not item_id:
        return _description_text(fallback_entries)
    row = repo.load_localization(table_name).get(str(item_id), {})
    entries = row.get('produceDescriptions') or fallback_entries
    return _description_text(entries)


def _localized_humanized_description(repo: Any, table_name: str, item_id: str, fallback_entries: Any = None) -> str:
    return _humanize_runtime_text(_localized_description(repo, table_name, item_id, fallback_entries))


def _raw_card_name(repo: Any, card: dict[str, Any] | None, fallback_id: str = '') -> str:
    row = card or {}
    if repo is not None and row:
        return str(getattr(repo, 'raw_card_name')(row))
    return _raw_name(row, fallback_id)


def _raw_drink_name(repo: Any, drink: dict[str, Any] | None, fallback_id: str = '') -> str:
    row = drink or {}
    if repo is not None and row:
        return str(getattr(repo, 'raw_drink_name')(row))
    return _raw_name(row, fallback_id)


def _raw_produce_item_name(item: dict[str, Any] | None, fallback_id: str = '') -> str:
    return _raw_name(item or {}, fallback_id)


def _unique_texts(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value or '').strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _friendly_rarity_label(rarity: str) -> str:
    raw = str(rarity or '')
    if not raw:
        return ''
    if raw in RARITY_SHORT:
        return RARITY_SHORT[raw]
    if raw.startswith('IdolCardRarity_'):
        return raw.removeprefix('IdolCardRarity_').upper()
    return raw


def _friendly_plan_type(plan_type: str) -> dict[str, str]:
    raw = str(plan_type or '')
    label, hint = PLAN_TYPE_DISPLAY.get(raw, ('', ''))
    return {'label': label, 'hint': hint}


def _effect_description(repo: Any, effect: dict[str, Any] | None) -> str:
    effect = effect or {}
    effect_id = str(effect.get('id') or '')
    return _localized_humanized_description(repo, 'ProduceExamEffect', effect_id, effect.get('produceDescriptions'))


def _card_description(card: Any, repo: Any) -> str:
    base_card = getattr(card, 'base_card', {}) or {}
    return _localized_humanized_description(
        repo,
        'ProduceCard',
        str(base_card.get('id') or ''),
        base_card.get('produceDescriptions'),
    )


def _enchant_description(repo: Any, enchant_id: str, fallback_row: dict[str, Any] | None = None) -> str:
    row = fallback_row or {}
    return _localized_humanized_description(
        repo,
        'ProduceExamStatusEnchant',
        str(enchant_id or ''),
        row.get('produceDescriptions'),
    )


def _source_label(source: str) -> str:
    if source.startswith('enchant:'):
        return '附魔触发'
    return SOURCE_NAMES.get(source, source)


def _card_destination_label(destination: str) -> str:
    return CARD_DESTINATION_NAMES.get(destination, destination)


def _current_stance_label(stance: str, level: int) -> str:
    stance_name = STANCE_NAMES.get(stance, stance)
    if stance == 'neutral':
        return '中立'
    return f'{stance_name} Lv{level}'


def _preview_new_card_summary(cards: list[Any], repo: Any) -> str:
    if not cards:
        return ''
    names = Counter(_raw_card_name(repo, card.base_card, str(card.card_id)) for card in cards)
    parts = [
        f'{name}x{count}' if count > 1 else name
        for name, count in names.items()
    ]
    return '、'.join(parts[:3]) + (f' 等{len(cards)}张' if len(parts) > 3 else '')


def _effect_limit_text(effect: dict[str, Any]) -> str:
    turns = int(effect.get('effectTurn') or 0)
    count = int(effect.get('effectCount') or 0)
    parts: list[str] = []
    if turns > 0:
        parts.append(f'{turns}回合')
    elif turns < 0:
        parts.append('永久')
    if count > 0:
        parts.append(f'{count}次')
    return ' '.join(parts)


def _initial_status_text(effect_turn: Any, effect_count: Any) -> str:
    turns = None if effect_turn is None else int(effect_turn)
    count = None if effect_count is None else int(effect_count)
    parts: list[str] = []
    if turns is not None:
        if turns < 0:
            parts.append('永久')
        elif turns > 0:
            parts.append(f'{turns}回合')
    if count is not None and count > 0:
        parts.append(f'{count}次')
    return '，'.join(parts)


def _trigger_summary(repo: Any, trigger_id: str) -> str:
    if repo is None or not trigger_id:
        return ''
    trigger = repo.exam_trigger_map.get(str(trigger_id))
    if not trigger:
        return trigger_id
    phase_types = [TRIGGER_PHASE_NAMES.get(str(value), str(value).rsplit('_', 1)[-1]) for value in trigger.get('phaseTypes', []) if value]
    phase_values = [int(value) for value in trigger.get('phaseValues', []) if int(value or 0) > 0]
    if len(phase_types) == 1 and phase_types[0] == '指定回合' and phase_values:
        return '第' + '/'.join(str(value) for value in phase_values) + '回合'
    parts: list[str] = []
    if phase_types:
        parts.append('/'.join(phase_types))
    if phase_values:
        parts.append('值=' + '/'.join(str(value) for value in phase_values))
    return ' '.join(parts) if parts else trigger_id


def _search_target_label(rt: Any, repo: Any, search_id: str) -> str:
    if not search_id or rt is None:
        return '命中的技能卡'
    search = rt.card_searches.first(str(search_id))
    if not search:
        return '命中的技能卡'
    produce_card_ids = [str(value) for value in search.get('produceCardIds', []) if value]
    if len(produce_card_ids) == 1 and repo is not None:
        card_row = repo.produce_cards.first(produce_card_ids[0])
        if card_row:
            return f'「{_raw_card_name(repo, card_row, produce_card_ids[0])}」'
    categories = {str(value) for value in search.get('cardCategories', []) if value}
    if categories == {'ProduceCardCategory_ActiveSkill'}:
        return '命中的アクティブスキルカード'
    if categories == {'ProduceCardCategory_MentalSkill'}:
        return '命中的メンタルスキルカード'
    if categories == {'ProduceCardCategory_Trouble'}:
        return '命中的トラブルカード'
    return '命中的技能卡'


def _card_effect_summary(rt: Any, card: Any, repo: Any) -> str:
    if repo is None:
        return ''
    summaries: list[str] = []
    for resolved in rt._resolved_card_play_effects(card):
        effect = repo.exam_effect_map.get(str(resolved.get('effect_id') or ''))
        if effect:
            summaries.append(_describe_exam_effect(rt, effect, repo, source='card'))
    for effect_id in card.base_card.get('moveProduceExamEffectIds', []):
        effect = repo.exam_effect_map.get(str(effect_id))
        if effect:
            summaries.append(_describe_exam_effect(rt, effect, repo, source='card'))
    unique = _unique_texts(summaries)
    return '；'.join(unique[:5])


def _enchant_title(rt: Any, repo: Any, enchant_id: str, enchant_row: dict[str, Any] | None = None, source: str = '') -> str:
    row = enchant_row or (repo.exam_status_enchant_map.get(str(enchant_id)) if repo else None) or {}
    description = _enchant_description(repo, str(enchant_id), row)
    if description:
        return description
    effect_summaries = _unique_texts([
        _describe_exam_effect(rt, effect, repo, source=source)
        for effect_id in row.get('produceExamEffectIds', [])
        if (effect := repo.exam_effect_map.get(str(effect_id))) is not None
    ]) if repo else []
    trigger_text = _trigger_summary(repo, str(row.get('produceExamTriggerId') or ''))
    body = '；'.join(effect_summaries[:3])
    if trigger_text and body:
        return f'{trigger_text}：{body}'
    return body or '附魔效果'


def _runtime_enchant_status_text(instances: list[Any]) -> str:
    if not instances:
        return '当前未在场（已触发完或已移除）'
    first = instances[0]
    parts = ['当前在场']
    if len(instances) > 1:
        parts.append(f'{len(instances)}层')
    if first.remaining_turns is not None:
        parts.append(f'剩余{first.remaining_turns}回合')
    if first.remaining_count is not None:
        parts.append(f'剩余{first.remaining_count}次')
    return '，'.join(parts)


def _effect_runtime_status_text(effect: dict[str, Any], remaining_turns: Any, remaining_count: Any) -> str:
    parts: list[str] = []
    raw_turn = effect.get('effectTurn')
    if remaining_turns is not None and int(remaining_turns) > 0:
        parts.append(f'剩余{int(remaining_turns)}回合')
    elif raw_turn is not None and int(raw_turn) < 0:
        parts.append('永久')
    if remaining_count is not None and int(remaining_count) > 0:
        parts.append(f'剩余{int(remaining_count)}次')
    return '，'.join(parts) if parts else '当前生效'


def _enchant_runtime_status_text(remaining_turns: Any, remaining_count: Any) -> str:
    parts = ['当前在场']
    if remaining_turns is not None and int(remaining_turns) > 0:
        parts.append(f'剩余{int(remaining_turns)}回合')
    if remaining_count is not None and int(remaining_count) > 0:
        parts.append(f'剩余{int(remaining_count)}次')
    return '，'.join(parts)


def _clone_runtime_for_preview(rt: Any) -> Any:
    clone = copy.copy(rt)
    for key, value in vars(rt).items():
        if key in PREVIEW_SHARED_FIELDS:
            setattr(clone, key, value)
        else:
            setattr(clone, key, copy.deepcopy(value))
    return clone


def _capture_preview_state(rt: Any) -> dict[str, Any]:
    known_uids = {
        card.uid
        for zone in (rt.hand, rt.deck, rt.grave, rt.hold, rt.lost, rt.playing)
        for card in zone
    }
    return {
        'score': float(rt.score),
        'stamina': float(rt.stamina),
        'resources': {key: float(rt.resources.get(key, 0)) for key in PREVIEW_RESOURCE_KEYS},
        'stance': rt.stance,
        'stance_level': int(getattr(rt, 'stance_level', 0)),
        'play_limit': int(getattr(rt, 'play_limit', 0)),
        'extra_turns': int(getattr(rt, 'extra_turns', 0)),
        'hand_uids': {card.uid for card in rt.hand},
        'known_uids': known_uids,
        'active_effect_uids': {eff.uid for eff in rt.active_effects},
        'active_enchant_uids': {ench.uid for ench in rt.active_enchants},
    }


def _describe_exam_effect(rt: Any, effect: dict[str, Any], repo: Any, source: str = '') -> str:
    effect_type = str(effect.get('effectType') or '')
    short = _short_effect_type(effect_type)
    from_card = source == 'card'

    if effect_type in LESSON_EFFECT_TYPES or effect_type == 'ProduceExamEffectType_ExamMultipleLessonBuffLesson':
        return f'打分+{_format_number(rt._resolve_lesson_effect_value(effect, from_card=from_card))}'

    if effect_type == 'ProduceExamEffectType_ExamParameterBuff':
        delta = rt._parameter_buff_gain_value(effect)
        if from_card:
            delta = rt._adjust_direct_gain(delta, add_grow_type='ProduceCardGrowEffectType_ParameterBuffTurnAdd')
        return f'好調+{_format_number(delta)}ターン'

    if effect_type == 'ProduceExamEffectType_ExamCardPlayAggressive':
        delta = rt._direct_value(effect)
        if from_card:
            delta = rt._adjust_direct_gain(delta, add_grow_type='ProduceCardGrowEffectType_AggressiveAdd')
        return f'好印象+{_format_number(delta)}'

    if effect_type in {'ProduceExamEffectType_ExamReview', 'ProduceExamEffectType_ExamReviewAdditive'}:
        delta = rt._direct_value(effect)
        if from_card:
            delta = rt._adjust_direct_gain(delta, add_grow_type='ProduceCardGrowEffectType_ReviewAdd')
        return f'集中+{_format_number(delta)}'

    if effect_type == 'ProduceExamEffectType_ExamLessonBuff':
        delta = rt._direct_value(effect)
        if from_card:
            delta = rt._adjust_direct_gain(delta, add_grow_type='ProduceCardGrowEffectType_LessonBuffAdd')
        return f'パラメータ上昇量増加+{_format_number(delta)}'

    if effect_type == 'ProduceExamEffectType_ExamFullPowerPoint':
        delta = rt._direct_value(effect)
        if from_card:
            delta = rt._adjust_direct_gain(
                delta,
                add_grow_type='ProduceCardGrowEffectType_FullPowerPointAdd',
                reduce_grow_type='ProduceCardGrowEffectType_FullPowerPointReduce',
            )
        return f'全力値+{_format_number(delta)}'

    if effect_type in {'ProduceExamEffectType_ExamBlock', 'ProduceExamEffectType_ExamBlockFix'}:
        delta = rt._raw_effect_value(effect)
        if from_card:
            delta = rt._adjust_direct_gain(
                delta,
                add_grow_type='ProduceCardGrowEffectType_BlockAdd',
                reduce_grow_type='ProduceCardGrowEffectType_BlockReduce',
            )
        delta = rt._apply_scalar_modifiers('ProduceExamEffectType_ExamBlock', delta)
        return f'元気+{_format_number(delta)}'

    if effect_type == 'ProduceExamEffectType_ExamReviewDependExamCardPlayAggressive':
        delta = rt.resources['aggressive'] * rt._ratio_value(effect)
        if from_card:
            delta = rt._adjust_direct_gain(delta, add_grow_type='ProduceCardGrowEffectType_ReviewAdd')
        return f'集中+{_format_number(delta)}(按好印象)'

    if effect_type == 'ProduceExamEffectType_ExamReviewDependExamBlock':
        delta = rt.resources['block'] * rt._ratio_value(effect)
        if from_card:
            delta = rt._adjust_direct_gain(delta, add_grow_type='ProduceCardGrowEffectType_ReviewAdd')
        suffix = ' 并清空元気' if float(effect.get('effectValue2') or 0) >= 1000 else ''
        return f'集中+{_format_number(delta)}(按元気){suffix}'

    if effect_type == 'ProduceExamEffectType_ExamBlockDependExamReview':
        delta = rt.resources['review'] * rt._ratio_value(effect)
        if from_card:
            delta = rt._adjust_direct_gain(
                delta,
                add_grow_type='ProduceCardGrowEffectType_BlockAdd',
                reduce_grow_type='ProduceCardGrowEffectType_BlockReduce',
            )
        delta = rt._apply_scalar_modifiers('ProduceExamEffectType_ExamBlock', delta)
        return f'元気+{_format_number(delta)}(按集中)'

    if effect_type == 'ProduceExamEffectType_ExamBlockAddMultipleAggressive':
        delta = max(rt._raw_effect_value(effect), 0.0) + rt.resources['aggressive'] * (float(effect.get('effectValue2') or 0) / 1000.0)
        if from_card:
            delta = rt._adjust_direct_gain(
                delta,
                add_grow_type='ProduceCardGrowEffectType_BlockAdd',
                reduce_grow_type='ProduceCardGrowEffectType_BlockReduce',
            )
        delta = rt._apply_scalar_modifiers('ProduceExamEffectType_ExamBlock', delta)
        return f'元気+{_format_number(delta)}(含好印象加成)'

    if effect_type == 'ProduceExamEffectType_ExamCardDraw':
        return f'抽{_format_number(max(1.0, round(rt._direct_value(effect))))}张'

    if effect_type in {'ProduceExamEffectType_ExamStaminaRecover', 'ProduceExamEffectType_ExamStaminaRecoverFix'}:
        return f'回体+{_format_number(rt._direct_value(effect))}'

    if effect_type == 'ProduceExamEffectType_ExamStaminaRecoverMultiple':
        return f'回体+{_format_number(rt.max_stamina * rt._ratio_value(effect))}'

    if effect_type in {'ProduceExamEffectType_ExamStaminaDamage', 'ProduceExamEffectType_ExamStaminaReduce', 'ProduceExamEffectType_ExamStaminaReduceFix'}:
        return f'体力-{_format_number(rt._direct_value(effect))}'

    if effect_type == 'ProduceExamEffectType_ExamConcentration':
        return f'进入強気Lv{int(max(rt._raw_effect_value(effect), 1.0))}'

    if effect_type == 'ProduceExamEffectType_ExamPreservation':
        return f'进入温存Lv{int(max(rt._raw_effect_value(effect), 1.0))}'

    if effect_type == 'ProduceExamEffectType_ExamOverPreservation':
        return '进入超温存(Lv3)'

    if effect_type == 'ProduceExamEffectType_ExamFullPower':
        return '进入全力'

    if effect_type == 'ProduceExamEffectType_ExamExtraTurn':
        return '+1额外回合'

    if effect_type == 'ProduceExamEffectType_ExamAntiDebuff':
        return f'免疫负面+{_format_number(rt._count_value(effect))}次'

    if effect_type == 'ProduceExamEffectType_ExamDebuffRecover':
        return '清除负面'

    if effect_type == 'ProduceExamEffectType_ExamPlayableValueAdd':
        return f'スキルカード使用数+{_format_number(effect.get("effectValue1") or effect.get("effectCount") or 1)}'

    if effect_type == 'ProduceExamEffectType_ExamItemFireLimitAdd':
        return f'Pアイテム触发次数+{_format_number(rt._direct_value(effect))}'

    if effect_type == 'ProduceExamEffectType_ExamStatusEnchant':
        enchant_id = str(effect.get('produceExamStatusEnchantId') or '')
        limit = _effect_limit_text(effect)
        enchant_row = repo.exam_status_enchant_map.get(enchant_id) if repo else None
        title = _enchant_title(rt, repo, enchant_id, enchant_row, source=source)
        return f'挂附魔:{title}' + (f' ({limit})' if limit else '')

    if effect_type == 'ProduceExamEffectType_ExamEffectTimer':
        delay = max(int(effect.get('effectValue1') or 1), 1)
        return f'{delay}回合后触发链式效果'

    if effect_type == 'ProduceExamEffectType_ExamCardCreateId':
        card_row = repo.produce_cards.first(str(effect.get('targetProduceCardId') or '')) if repo else None
        card_name = _raw_card_name(repo, card_row, str(effect.get('targetProduceCardId') or ''))
        return f'生成卡:{card_name}'

    if effect_type == 'ProduceExamEffectType_ExamCardCreateSearch':
        return '按检索条件生成牌'

    if effect_type == 'ProduceExamEffectType_ExamCardMove':
        return f'移牌到{_card_destination_label(str(effect.get("movePositionType") or "").rsplit("_", 1)[-1].lower())}'

    if effect_type == 'ProduceExamEffectType_ExamCardUpgrade':
        return '强化命中的牌'

    if effect_type == 'ProduceExamEffectType_ExamForcePlayCardSearch':
        return '强制打出命中的牌'

    if effect_type == 'ProduceExamEffectType_ExamSearchPlayCardStaminaConsumptionChange':
        target = _search_target_label(rt, repo, str(effect.get('produceCardSearchId') or ''))
        stamina_value = float(effect.get('effectValue1') or 0)
        if abs(stamina_value) < 1e-6:
            return f'{target}体力消耗改为0'
        return f'{target}体力消耗改为{_format_number(stamina_value)}'

    ratio = rt._ratio_value(effect)
    raw = float(effect.get('effectValue1') or effect.get('effectValue2') or effect.get('effectCount') or 0)
    limit = _effect_limit_text(effect)
    if ratio > 0 and 'Multiple' in effect_type:
        value_text = f'+{_format_number(ratio * 100)}%'
    elif ratio > 0 and 'Down' in effect_type:
        value_text = f'-{_format_number(ratio * 100)}%'
    elif raw > 0:
        value_text = _format_number(raw)
    else:
        value_text = ''
    parts = [short]
    if value_text:
        parts.append(value_text)
    if limit:
        parts.append(limit)
    return ' '.join(parts)


def _effect_display_text(rt: Any, effect: dict[str, Any], repo: Any, source: str = '') -> str:
    effect_type = str(effect.get('effectType') or '')
    if effect_type == 'ProduceExamEffectType_ExamStaminaConsumptionAdd':
        return '技能卡体力消耗翻倍'
    if effect_type == 'ProduceExamEffectType_ExamStaminaConsumptionDown':
        return '技能卡体力消耗减半'
    if effect_type == 'ProduceExamEffectType_ExamStaminaConsumptionAddFix':
        return f'技能卡体力消耗+{_format_number(rt._raw_effect_value(effect))}'
    if effect_type == 'ProduceExamEffectType_ExamStaminaConsumptionDownFix':
        return f'技能卡体力消耗-{_format_number(rt._raw_effect_value(effect))}'
    if effect_type == 'ProduceExamEffectType_ExamSearchPlayCardStaminaConsumptionChange':
        target = _search_target_label(rt, repo, str(effect.get('produceCardSearchId') or ''))
        stamina_value = float(effect.get('effectValue1') or 0)
        if abs(stamina_value) < 1e-6:
            return f'{target}体力消耗改为0'
        return f'{target}体力消耗改为{_format_number(stamina_value)}'
    if effect_type == 'ProduceExamEffectType_ExamPlayableValueAdd':
        return f'スキルカード使用数+{_format_number(effect.get("effectValue1") or effect.get("effectCount") or 1)}'
    description = _humanize_runtime_text(_effect_description(repo, effect))
    if description:
        return description
    return _humanize_runtime_text(_describe_exam_effect(rt, effect, repo, source=source))


def _enchant_display_text(ench: Any, rt: Any, repo: Any, enchant_row: dict[str, Any] | None = None) -> str:
    description = _humanize_runtime_text(_enchant_description(repo, str(ench.enchant_id), enchant_row))
    if description:
        return description
    title = _humanize_runtime_text(_enchant_title(rt, repo, str(ench.enchant_id), enchant_row, source=str(ench.source or '')))
    if title:
        return title
    effect_summaries = _unique_texts([
        _humanize_runtime_text(_describe_exam_effect(rt, effect, repo, source=ench.source))
        for effect_id in (enchant_row or {}).get('produceExamEffectIds', [])
        if (effect := repo.exam_effect_map.get(str(effect_id))) is not None
    ]) if repo else []
    return '；'.join(effect_summaries[:3]) if effect_summaries else '附魔效果'


def _preview_action_delta(before: dict[str, Any], after_rt: Any, repo: Any) -> str:
    parts: list[str] = []
    score_delta = float(after_rt.score) - float(before['score'])
    if abs(score_delta) > 1e-6:
        parts.append(f'打分{_signed_number(score_delta)}')

    stamina_delta = float(after_rt.stamina) - float(before['stamina'])
    if abs(stamina_delta) > 1e-6:
        parts.append(f'体力{_signed_number(stamina_delta)}')

    for key in PREVIEW_RESOURCE_KEYS:
        delta = float(after_rt.resources.get(key, 0)) - float(before['resources'].get(key, 0))
        if abs(delta) > 1e-6:
            parts.append(f'{RESOURCE_SHORT_NAMES.get(key, key)}{_signed_number(delta)}')

    if after_rt.stance != before['stance'] or int(after_rt.stance_level) != int(before['stance_level']):
        parts.append(f'指针->{_current_stance_label(after_rt.stance, int(after_rt.stance_level))}')

    play_limit_delta = int(after_rt.play_limit) - int(before['play_limit'])
    if play_limit_delta != 0:
        parts.append(f'スキルカード使用上限{_signed_number(play_limit_delta)}')

    extra_turn_delta = int(after_rt.extra_turns) - int(before['extra_turns'])
    if extra_turn_delta != 0:
        parts.append(f'额外回合{_signed_number(extra_turn_delta)}')

    new_hand_cards = [card for card in after_rt.hand if card.uid not in before['hand_uids']]
    if new_hand_cards:
        parts.append(f'补牌:{_preview_new_card_summary(new_hand_cards, repo)}')

    created_cards = [
        card
        for zone in (after_rt.hand, after_rt.deck, after_rt.grave, after_rt.hold, after_rt.lost)
        for card in zone
        if card.uid not in before['known_uids']
    ]
    if created_cards:
        parts.append(f'生成:{_preview_new_card_summary(created_cards, repo)}')

    new_effects = [eff for eff in after_rt.active_effects if eff.uid not in before['active_effect_uids']]
    if new_effects:
        summaries = [
            _describe_exam_effect(after_rt, eff.effect, repo, source=eff.source)
            for eff in new_effects[:2]
        ]
        parts.append('新增持续:' + '；'.join(summaries))

    new_enchants = [ench for ench in after_rt.active_enchants if ench.uid not in before['active_enchant_uids']]
    if new_enchants:
        enchant_summaries = [
            _enchant_effect_summary(ench, after_rt, repo)
            for ench in new_enchants[:2]
        ]
        parts.append('新增附魔:' + '；'.join(enchant_summaries))

    return '；'.join(parts[:10]) if parts else '无显著变化'


def _card_cost_summary(rt: Any, card: Any) -> str:
    cost_parts: list[str] = []
    base_cost, force_cost = rt._card_stamina_components(card)
    if base_cost > 0:
        cost_parts.append(f'体力{_format_number(base_cost)}')
    if force_cost > 0:
        cost_parts.append(f'直伤体力{_format_number(force_cost)}')
    for resource_key, amount in rt._card_resource_costs(card).items():
        if amount <= 0:
            continue
        cost_parts.append(f'{RESOURCE_SHORT_NAMES.get(resource_key, resource_key)}{_format_number(amount)}')
    return ', '.join(cost_parts) if cost_parts else '无额外消耗'


def _card_preview_summary(rt: Any, card: Any, repo: Any) -> str:
    preview = _clone_runtime_for_preview(rt)
    preview_card = next((candidate for candidate in preview.hand if candidate.uid == card.uid), None)
    if preview_card is None:
        return '无法预演'
    before = _capture_preview_state(preview)
    preview._play_card(preview_card)
    return _preview_action_delta(before, preview, repo)


def _drink_preview_summary(rt: Any, drink_index: int, repo: Any) -> str:
    preview = _clone_runtime_for_preview(rt)
    if drink_index < 0 or drink_index >= len(preview.drinks):
        return '无法预演'
    before = _capture_preview_state(preview)
    preview._use_drink(drink_index)
    return _preview_action_delta(before, preview, repo)


def _is_offensive_card(card: Any, repo: Any) -> bool:
    """粗略判断一张牌是否属于考试中的火力牌。"""

    if repo is None:
        return False
    effect_types = repo.card_exam_effect_types(card.base_card)
    return any(effect_type in LESSON_EFFECT_TYPES for effect_type in effect_types)


def _count_offensive_cards(cards: list[Any], repo: Any) -> int:
    return sum(1 for card in cards if _is_offensive_card(card, repo))


# ---------------------------------------------------------------------------
# 格式化函数
# ---------------------------------------------------------------------------

def format_card(card: Any, repo: Any) -> str:
    """格式化单张手牌。"""
    name = _raw_card_name(repo, card.base_card, str(card.card_id))
    upgrade = f'[+{card.upgrade_count}]' if card.upgrade_count > 0 else ''
    category = _short_category(str(card.base_card.get('category') or ''))
    stamina = card.base_card.get('stamina') or 0
    description = _card_description(card, repo)
    return f'- {name}{upgrade} ({category}) 体力:{stamina} 说明:{description or "见当前结算"}'


def _extract_zone_card_data(card: Any, repo: Any) -> dict[str, Any]:
    """为牌库/弃牌/保留/除外等非手牌区域提取简明卡片信息。"""

    name = _raw_card_name(repo, card.base_card, str(card.card_id))
    category = _short_category(str(card.base_card.get('category') or ''))
    description = _humanize_runtime_text(_card_description(card, repo))
    return {
        'name': name,
        'upgrade_count': int(getattr(card, 'upgrade_count', 0) or 0),
        'category': category,
        'description': description or '无额外说明',
    }


def format_deck_summary(rt: Any, repo: Any) -> str:
    """按 category 分组统计牌库组成。"""
    if not rt.deck:
        return '(空)'
    cat_counts: Counter[str] = Counter()
    total_stamina = 0.0
    for card in rt.deck:
        cat = _short_category(str(card.base_card.get('category') or ''))
        cat_counts[cat] += 1
        total_stamina += float(card.base_card.get('stamina') or 0)
    n = len(rt.deck)
    cat_str = ', '.join(f'{cat}×{count}' for cat, count in cat_counts.most_common())
    avg_stamina = total_stamina / n
    return f'分类: {cat_str} | 平均体力:{avg_stamina:.1f}'


def format_effect(eff: Any) -> str:
    """格式化单个活跃效果。"""
    effect_type = str(eff.effect.get('effectType') or '')
    short = _short_effect_type(effect_type)
    parts = [f'- {short}']
    if eff.remaining_turns is not None:
        parts.append(f'剩余{eff.remaining_turns}回合')
    if eff.remaining_count is not None:
        parts.append(f'剩余{eff.remaining_count}次')
    if eff.source:
        parts.append(f'来源:{eff.source}')
    return ' '.join(parts)


def format_enchant(ench: Any) -> str:
    """格式化单个活跃附魔。"""
    parts = [f'- {ench.enchant_id}']
    if ench.remaining_turns is not None:
        parts.append(f'剩余{ench.remaining_turns}回合')
    if ench.remaining_count is not None:
        parts.append(f'剩余{ench.remaining_count}次')
    if ench.source:
        parts.append(f'来源:{ench.source}')
    return ' '.join(parts)


def format_drink(drink: dict, repo: Any) -> str:
    """格式化单瓶饮料。"""
    name = _raw_drink_name(repo, drink, str(drink.get('id') or ''))
    description = _localized_humanized_description(
        repo,
        'ProduceDrink',
        str(drink.get('id') or ''),
        drink.get('produceDescriptions'),
    )
    return f'- {name} 说明:{description or "见当前结算"}'


def format_gimmicks(rt: Any, repo: Any) -> str:
    """格式化考场机制（仅未来回合）。"""
    future = [
        row for row in rt.gimmick_rows
        if int(row.get('startTurn') or 0) > rt.turn
    ]
    if not future:
        return '无未来机制'
    future.sort(key=lambda r: int(r.get('startTurn') or 0))
    parts = []
    for row in future[:5]:
        turn = int(row.get('startTurn') or 0)
        effect_id = str(row.get('produceExamEffectId') or '')
        effect = repo.exam_effect_map.get(effect_id) if repo else None
        summary = _describe_exam_effect(rt, effect, repo, source='gimmick') if effect else effect_id
        parts.append(f'- 第{turn}回合: {summary}')
    if len(future) > 5:
        parts.append(f'...及其他{len(future) - 5}个')
    return '\n'.join(parts)


# ---------------------------------------------------------------------------
# 纯数据提取 — 返回模板变量 dict
# ---------------------------------------------------------------------------

def _extract_card_data(card: Any, repo: Any, rt: Any) -> dict:
    """提取单张手牌的数据。"""
    name = _raw_card_name(repo, card.base_card, str(card.card_id))
    category = _short_category(str(card.base_card.get('category') or ''))
    move_destination = _card_destination_label(rt._card_move_destination(card))
    return {
        'name': name,
        'upgrade_count': card.upgrade_count,
        'category': category,
        'playable': bool(rt._can_play_card(card)),
        'description': _card_description(card, repo),
        'effect_summary': _card_effect_summary(rt, card, repo),
        'cost_summary': _card_cost_summary(rt, card),
        'preview_summary': _card_preview_summary(rt, card, repo),
        'move_destination': move_destination,
    }


def _extract_effect_data(eff: Any, rt: Any, repo: Any) -> dict:
    """提取单个活跃效果的数据。"""
    effect_type = str(eff.effect.get('effectType') or '')
    return {
        'short_name': _short_effect_type(effect_type),
        'remaining_turns': eff.remaining_turns,
        'remaining_count': eff.remaining_count,
        'source': eff.source,
        'source_label': _source_label(str(eff.source or '')),
        'summary': _describe_exam_effect(rt, eff.effect, repo, source=str(eff.source or '')),
        'description': _effect_description(repo, eff.effect),
        'display_text': _effect_display_text(rt, eff.effect, repo, source=str(eff.source or '')),
        'status_text': _effect_runtime_status_text(eff.effect, eff.remaining_turns, eff.remaining_count),
    }


def _enchant_effect_summary(ench: Any, rt: Any, repo: Any) -> str:
    enchant_row = repo.exam_status_enchant_map.get(str(ench.enchant_id)) if repo else None
    return _enchant_title(rt, repo, str(ench.enchant_id), enchant_row, source=str(ench.source or ''))


def _extract_enchant_data(ench: Any, rt: Any, repo: Any) -> dict:
    """提取单个活跃附魔的数据。"""
    enchant_row = repo.exam_status_enchant_map.get(str(ench.enchant_id)) if repo else None
    effect_summaries = _unique_texts([
        _describe_exam_effect(rt, effect, repo, source=ench.source)
        for effect_id in (enchant_row or {}).get('produceExamEffectIds', [])
        if (effect := repo.exam_effect_map.get(str(effect_id)))
    ])
    return {
        'title': _enchant_title(rt, repo, str(ench.enchant_id), enchant_row, source=str(ench.source or '')),
        'description': _enchant_description(repo, str(ench.enchant_id), enchant_row),
        'display_text': _enchant_display_text(ench, rt, repo, enchant_row),
        'remaining_turns': ench.remaining_turns,
        'remaining_count': ench.remaining_count,
        'status_text': _enchant_runtime_status_text(ench.remaining_turns, ench.remaining_count),
        'source': ench.source,
        'source_label': _source_label(str(ench.source or '')),
        'trigger': _trigger_summary(repo, str((enchant_row or {}).get('produceExamTriggerId') or '')),
        'effects': effect_summaries,
    }


def _extract_drink_data(drink: dict, drink_index: int, repo: Any, rt: Any) -> dict:
    """提取单瓶饮料的数据。"""
    name = _raw_drink_name(repo, drink, str(drink.get('id') or ''))
    effect_types = repo.drink_exam_effect_types(drink) if repo else []
    effect_names = _unique_texts([_short_effect_type(str(et)) for et in effect_types])
    return {
        'name': name,
        'effects': '；'.join(effect_names) if effect_names else '无',
        'available': not drink.get('_consumed'),
        'description': _localized_humanized_description(
            repo,
            'ProduceDrink',
            str(drink.get('id') or ''),
            drink.get('produceDescriptions'),
        ),
        'preview_summary': _drink_preview_summary(rt, drink_index, repo) if not drink.get('_consumed') else '已使用',
    }


def _extract_produce_item_context(rt: Any, repo: Any) -> dict | None:
    loadout = getattr(rt, 'loadout', None)
    if loadout is None or not getattr(loadout, 'produce_item_id', ''):
        return None
    item_id = str(loadout.produce_item_id)
    item_row = repo.load_table('ProduceItem').first(item_id) if repo else {}
    specs = getattr(loadout, 'exam_status_enchant_specs', ()) or ()
    enchant_entries: list[dict[str, Any]] = []
    active_item_enchants = [
        enchant for enchant in getattr(rt, 'active_enchants', [])
        if str(getattr(enchant, 'source', '')) == 'produce_item'
    ]
    for spec in specs:
        enchant_row = repo.exam_status_enchant_map.get(str(spec.enchant_id)) if repo else None
        effect_summaries = _unique_texts([
            _describe_exam_effect(rt, effect, repo, source='produce_item')
            for effect_id in (enchant_row or {}).get('produceExamEffectIds', [])
            if (effect := repo.exam_effect_map.get(str(effect_id)))
        ])
        matching_instances = [
            enchant for enchant in active_item_enchants
            if str(getattr(enchant, 'enchant_id', '')) == str(spec.enchant_id)
        ]
        enchant_entries.append({
            'description': _enchant_description(repo, str(spec.enchant_id), enchant_row) or _enchant_title(rt, repo, str(spec.enchant_id), enchant_row, source='produce_item'),
            'trigger': _trigger_summary(repo, str((enchant_row or {}).get('produceExamTriggerId') or '')),
            'initial_status': _initial_status_text(spec.effect_turn, spec.effect_count),
            'effects': effect_summaries,
            'runtime_status': _runtime_enchant_status_text(matching_instances),
        })
    return {
        'id': item_id,
        'name': _raw_produce_item_name(item_row, item_id),
        'description': _localized_humanized_description(
            repo,
            'ProduceItem',
            item_id,
            (item_row or {}).get('produceDescriptions'),
        ),
        'static_note': '以下是该道具开场装载到考试中的附魔；当前是否仍在场，以各条“当前状态”为准。',
        'enchants': enchant_entries,
    }


def _extract_loadout_context(rt: Any, repo: Any) -> dict | None:
    loadout = getattr(rt, 'loadout', None)
    if loadout is None:
        return None
    parameter_stats = getattr(rt, 'parameter_stats', [0.0, 0.0, 0.0])
    plan_type = str(loadout.stat_profile.plan_type or 'ProducePlanType_Common')
    plan_display = _friendly_plan_type(plan_type)
    return {
        'idol_name': str(loadout.metadata.get('idol_name') or loadout.idol_card_id),
        'rarity': _friendly_rarity_label(str(loadout.metadata.get('rarity') or '')),
        'producer_level': int(loadout.producer_level),
        'idol_rank': int(loadout.idol_rank),
        'dearness_level': int(loadout.dearness_level),
        'exam_effect_type_label': plan_display['label'],
        'exam_effect_type_hint': plan_display['hint'],
        'score_bonus_multiplier': _format_number(getattr(rt, 'score_bonus_multiplier', loadout.exam_score_bonus_multiplier)),
        'stats': {
            'vocal': _format_number(parameter_stats[0] if len(parameter_stats) > 0 else loadout.stat_profile.vocal),
            'dance': _format_number(parameter_stats[1] if len(parameter_stats) > 1 else loadout.stat_profile.dance),
            'visual': _format_number(parameter_stats[2] if len(parameter_stats) > 2 else loadout.stat_profile.visual),
            'stamina': _format_number(loadout.stat_profile.stamina),
        },
        'produce_item': _extract_produce_item_context(rt, repo),
    }


def extract_state_context(rt: Any, repo: Any, last_action_label: str = '') -> dict:
    """从 ExamRuntime 提取完整局面数据，返回模板变量 dict。"""
    target = float(rt._target_score() if hasattr(rt, '_target_score') else (rt.profile.get('base_score') or 1))
    remaining = max(rt.max_turns - rt.turn + 1, 0)
    ratio_val = rt.score / target if target > 0 else 0.0
    battle_kind = str(getattr(rt, 'battle_kind', 'exam') or 'exam')
    clear_state = str(getattr(rt, 'clear_state', 'ongoing') or 'ongoing')
    lesson_target_remaining = float(rt._lesson_target_remaining() if hasattr(rt, '_lesson_target_remaining') else 0.0)
    lesson_perfect_remaining = float(rt._lesson_perfect_remaining() if hasattr(rt, '_lesson_perfect_remaining') else 0.0)
    lesson_perfect_enabled = float(rt._current_perfect_target() if hasattr(rt, '_current_perfect_target') else 0.0) > 0
    turn_color = str(getattr(rt, 'current_turn_color', '') or '')
    turn_color_label = str(rt.turn_color_label() if hasattr(rt, 'turn_color_label') else '')
    fan_votes = float(getattr(rt, 'fan_votes', 0.0) or 0.0)
    fan_vote_baseline = float(rt._reported_fan_vote_baseline() if hasattr(rt, '_reported_fan_vote_baseline') else 0.0)
    fan_vote_requirement = float(rt._reported_fan_vote_requirement() if hasattr(rt, '_reported_fan_vote_requirement') else 0.0)

    res = rt.resources

    # 指针
    stance_name = STANCE_NAMES.get(rt.stance, rt.stance)
    if rt.stance == 'neutral':
        stance_desc = '中立 (无加成)'
    else:
        stance_desc = f'{stance_name} Lv{rt.stance_level}'

    # 负面状态
    negatives = []
    if res.get('sleepy', 0) > 0:
        negatives.append(f'弱気={res["sleepy"]:.0f}')
    if res.get('panic', 0) > 0:
        negatives.append(f'パニック={res["panic"]:.0f}')
    if res.get('slump', 0) > 0:
        negatives.append(f'スランプ={res["slump"]:.0f}')
    if res.get('active_skill_forbidden', 0) > 0:
        negatives.append('アクティブ封印')
    if res.get('mental_skill_forbidden', 0) > 0:
        negatives.append('メンタル封印')

    drink_inventory = list(rt.drinks)
    available_drinks = [
        (index, drink)
        for index, drink in enumerate(drink_inventory)
        if not drink.get('_consumed')
    ]
    deck_cards = list(rt.deck)
    grave_cards = list(rt.grave)
    hold_cards = list(rt.hold)
    offensive_counts = {
        'hand': _count_offensive_cards(rt.hand, repo),
        'deck': _count_offensive_cards(deck_cards, repo),
        'grave': _count_offensive_cards(grave_cards, repo),
        'hold': _count_offensive_cards(hold_cards, repo),
    }
    reshuffle_hint = ''
    if len(deck_cards) <= 2 and grave_cards:
        reshuffle_hint = f'牌库仅剩{len(deck_cards)}张；下次抽牌大概率会把弃牌堆洗回。'
    elif offensive_counts['deck'] <= 0 and offensive_counts['grave'] > 0:
        reshuffle_hint = '当前牌库几乎没有火力牌，后续主要依赖洗回弃牌堆后的再抽。'

    tc = rt.total_counters

    return {
        'turn': rt.turn,
        'max_turns': rt.max_turns,
        'remaining': remaining,
        'battle_kind': battle_kind,
        'battle_kind_label': '训练' if battle_kind == 'exam' and str(getattr(rt, 'reward_mode', '') or '') == 'clear' else BATTLE_KIND_NAMES.get(battle_kind, battle_kind),
        'clear_state': clear_state,
        'clear_state_label': CLEAR_STATE_NAMES.get(clear_state, clear_state),
        'lesson_cleared': bool(getattr(rt, 'lesson_cleared', False)),
        'lesson_target_remaining': _format_number(lesson_target_remaining),
        'lesson_perfect_remaining': _format_number(lesson_perfect_remaining),
        'lesson_perfect_enabled': lesson_perfect_enabled,
        'score': f'{rt.score:.0f}',
        'target': f'{target:.0f}',
        'ratio': f'{ratio_val:.0%}',
        'stamina': f'{rt.stamina:.0f}',
        'max_stamina': f'{rt.max_stamina:.0f}',
        'last_action_label': last_action_label,
        'play_limit_remaining': max(int(rt.play_limit) - int(rt.turn_counters['play_count']), 0),
        'play_limit_total': int(rt.play_limit),
        'turn_color': turn_color,
        'turn_color_label': turn_color_label,
        'turn_color_display_label': TURN_COLOR_DISPLAY_NAMES.get(turn_color_label, turn_color_label),
        'fan_votes': _format_number(fan_votes),
        'fan_vote_baseline': _format_number(fan_vote_baseline),
        'fan_vote_requirement': _format_number(fan_vote_requirement),
        'score_bonus_multiplier': _format_number(getattr(rt, 'score_bonus_multiplier', 1.0)),
        'parameter_stats': {
            'vocal': _format_number(rt.parameter_stats[0] if len(rt.parameter_stats) > 0 else 0),
            'dance': _format_number(rt.parameter_stats[1] if len(rt.parameter_stats) > 1 else 0),
            'visual': _format_number(rt.parameter_stats[2] if len(rt.parameter_stats) > 2 else 0),
        },
        'loadout': _extract_loadout_context(rt, repo),
        'hand': [_extract_card_data(c, repo, rt) for c in rt.hand],
        'deck_count': len(rt.deck),
        'deck_summary': format_deck_summary(rt, repo),
        'deck_cards': [_extract_zone_card_data(c, repo) for c in deck_cards],
        'grave_cards': [_extract_zone_card_data(c, repo) for c in grave_cards],
        'hold_cards': [_extract_zone_card_data(c, repo) for c in hold_cards],
        'lost_cards': [_extract_zone_card_data(c, repo) for c in rt.lost],
        'zone_counts': {
            'deck': len(deck_cards),
            'grave': len(grave_cards),
            'hold': len(hold_cards),
            'lost': len(rt.lost),
        },
        'offensive_counts': offensive_counts,
        'reshuffle_hint': reshuffle_hint,
        'resources': {
            'parameter_buff': f'{res["parameter_buff"]:.0f}',
            'review': f'{res["review"]:.0f}',
            'aggressive': f'{res["aggressive"]:.0f}',
            'block': f'{res["block"]:.0f}',
            'lesson_buff': f'{res["lesson_buff"]:.0f}',
            'enthusiastic': f'{res["enthusiastic"]:.0f}',
            'full_power_point': f'{res["full_power_point"]:.0f}',
        },
        'stance_desc': stance_desc,
        'negatives': ', '.join(negatives) if negatives else '',
        'active_effects': [_extract_effect_data(e, rt, repo) for e in rt.active_effects],
        'active_enchants': [_extract_enchant_data(e, rt, repo) for e in rt.active_enchants],
        'drinks': [_extract_drink_data(drink, index, repo, rt) for index, drink in available_drinks],
        'available_drink_count': len(available_drinks),
        'used_drink_count': len(drink_inventory) - len(available_drinks),
        'drink_total_count': len(drink_inventory),
        'gimmicks': format_gimmicks(rt, repo),
        'total_counters': {
            'play_count': tc['play_count'],
            'stamina_spent': f'{tc["stamina_spent"]:.0f}',
            'block_consumed': f'{tc["block_consumed"]:.0f}',
        },
    }


def _get_runtime_from_env(env: Any) -> Any:
    current = env
    while hasattr(current, 'env'):
        if hasattr(current, 'runtime'):
            return current.runtime
        current = current.env
    return getattr(current, 'runtime', None)


def _get_repository_from_env(env: Any) -> Any:
    current = env
    while hasattr(current, 'env'):
        if hasattr(current, 'repository'):
            return current.repository
        current = current.env
    return getattr(current, 'repository', None)


def _llm_action_label(env: Any, candidate: Any) -> str:
    kind = str(getattr(candidate, 'kind', '') or '')
    payload = getattr(candidate, 'payload', {}) or {}
    runtime = _get_runtime_from_env(env)
    repo = _get_repository_from_env(env)
    if runtime is None:
        return str(getattr(candidate, 'label', '') or '')
    if kind == 'card':
        _raw_uid = payload.get('uid')
        uid = int(_raw_uid) if _raw_uid is not None else -1
        card = next((item for item in getattr(runtime, 'hand', []) if int(getattr(item, 'uid', -2)) == uid), None)
        if card is not None:
            return f'{_raw_card_name(repo, card.base_card, str(card.card_id))}[{int(getattr(card, "upgrade_count", 0) or 0)}]'
    if kind == 'drink':
        _raw_idx = payload.get('index')
        index = int(_raw_idx) if _raw_idx is not None else -1
        drinks = list(getattr(runtime, 'drinks', []))
        if 0 <= index < len(drinks):
            return _raw_drink_name(repo, drinks[index], str(drinks[index].get('id') or ''))
    return str(getattr(candidate, 'label', '') or '')


def action_label_for_llm(env: Any, action_index: int) -> str:
    """为 llm prompt 生成动作标签，优先使用主数据库原始日文名。"""
    candidates = getattr(env, '_candidates', [])
    if 0 <= int(action_index) < len(candidates):
        return _llm_action_label(env, candidates[int(action_index)])
    return ''


def extract_action_list_context(env: Any) -> list[dict]:
    """提取合法动作列表数据。"""
    actions = []
    candidates = getattr(env, '_candidates', [])
    for idx, candidate in enumerate(candidates):
        if not candidate.payload.get('available', False):
            continue
        actions.append({
            'index': idx,
            'kind': candidate.kind,
            'label': _llm_action_label(env, candidate),
        })
    return actions


# ---------------------------------------------------------------------------
# 完整局面描述（通过 Jinja2 模板渲染）
# ---------------------------------------------------------------------------

def build_state_snapshot(rt: Any, repo: Any, last_action_label: str = '') -> str:
    """从 ExamRuntime 提取完整局面描述。"""
    from .prompt_renderer import render
    ctx = extract_state_context(rt, repo, last_action_label)
    return render('state_snapshot.jinja2', **ctx)


# ---------------------------------------------------------------------------
# 合法动作列表格式化（供 llm_player 使用）
# ---------------------------------------------------------------------------

def format_action_list(env: Any) -> str:
    """格式化当前环境的合法动作列表。"""
    from .prompt_renderer import render
    actions = extract_action_list_context(env)
    # 渲染时需要 snapshot 占位（action_select 模板中包含 snapshot）
    # 这里只返回动作列表部分，保持向后兼容
    lines = ['## 合法动作']
    for a in actions:
        if a['kind'] == 'end_turn':
            lines.append(f'{a["index"]}: 结束回合')
        elif a['kind'] == 'drink':
            lines.append(f'{a["index"]}: [饮料] {a["label"]}')
        else:
            lines.append(f'{a["index"]}: {a["label"]} ({a["kind"]})')
    return '\n'.join(lines)
