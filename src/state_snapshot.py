"""考试运行时的可读局面摘要工具。"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from .constants.game.action_types import EXAM_ACTION_CARD, EXAM_ACTION_DRINK, EXAM_ACTION_END_TURN
from .constants.game.resource_keys import (
    RESOURCE_AGGRESSIVE,
    RESOURCE_BLOCK,
    RESOURCE_CONCENTRATION,
    RESOURCE_ENTHUSIASTIC,
    RESOURCE_FULL_POWER_POINT,
    RESOURCE_REVIEW,
)
from .repository.master_data import MasterDataRepository
from .simulation.exam.runtime import ExamRuntime, RuntimeCard, TimedExamEffect, TriggeredEnchant


_TURN_COLOR_DISPLAY_LABELS = {
    'vocal': 'ボーカル',
    'dance': 'ダンス',
    'visual': 'ビジュアル',
}

_SOURCE_LABELS = {
    'card': 'スキルカード',
    'drink': 'Pドリンク',
    'produce_item': 'Pアイテム',
    'support_card': 'サポートカード',
    'gimmick': '応援/トラブル',
}

_EFFECT_SENTENCES = {
    'ProduceExamEffectType_ExamStaminaConsumptionAdd': '技能卡体力消耗翻倍',
    'ProduceExamEffectType_ExamSearchPlayCardStaminaConsumptionChange': '命中的技能卡体力消耗改为0',
}

_INTERNAL_TOKEN_PATTERN = re.compile(
    r'ProduceExamEffectType_[A-Za-z0-9_]+|'
    r'IdolCardRarity_[A-Za-z0-9_]+|'
    r'ExamBlockDepend[A-Za-z0-9_]*|'
    r'enchant-pitem_[A-Za-z0-9_-]+'
)


@dataclass(frozen=True)
class CardPromptContext:
    """可供调试和提示词复用的手牌/牌库卡牌摘要。"""

    id: str
    name: str
    upgrade_count: int
    description: str
    effect_summary: str
    cost_summary: str
    preview_summary: str
    available: bool = True


@dataclass(frozen=True)
class DrinkPromptContext:
    """可供调试和提示词复用的 P 饮料摘要。"""

    id: str
    name: str
    description: str
    effect_summary: str
    preview_summary: str
    available: bool


@dataclass(frozen=True)
class ProduceItemPromptContext:
    """可供调试和提示词复用的 P 道具摘要。"""

    id: str
    name: str
    description: str
    enchants: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class LoadoutPromptContext:
    """可供调试和提示词复用的编成摘要。"""

    idol_card_id: str
    produce_item: ProduceItemPromptContext | None


@dataclass(frozen=True)
class StatePromptContext:
    """结构化局面摘要，输出前会转换为普通 dict 兼容旧测试。"""

    battle_kind: str
    clear_state: str
    lesson_cleared: bool
    lesson_target_remaining: str
    lesson_perfect_remaining: str
    turn_color: str
    turn_color_label: str
    turn_color_display_label: str
    score: float
    stamina: float
    max_stamina: float
    turn: int
    max_turns: int
    used_drink_count: int
    available_drink_count: int
    drink_total_count: int
    resources: dict[str, float]
    hand: list[CardPromptContext]
    deck_cards: list[CardPromptContext]
    drinks: list[DrinkPromptContext]
    loadout: LoadoutPromptContext | None
    active_effects: list[str]
    active_enchants: list[str]


@dataclass(frozen=True)
class ActionPromptContext:
    """动作列表中的单个动作摘要。"""

    index: int
    label: str
    kind: str
    available: bool


def _sanitize_user_text(text: str) -> str:
    """统一修正历史中文俗称，避免摘要混用非游戏原文。"""

    normalized = (
        str(text or '')
        .replace('干劲', 'やる気')
        .replace('好调', '好調')
        .replace('元气', '元気')
        .replace('P饮料', 'Pドリンク')
        .replace('P道具/饰品', 'Pアイテム')
        .replace('パラメータ上昇量増加', 'パラメータ補正')
        .replace('<nobr>', '')
        .replace('</nobr>', '')
    )
    return _INTERNAL_TOKEN_PATTERN.sub('効果', normalized)


def _description_text(row: dict[str, Any]) -> str:
    """从主数据行中提取可读描述文本。"""

    texts: list[str] = []
    for key in ('produceDescriptions', 'descriptions'):
        values = row.get(key, [])
        if isinstance(values, list):
            for item in values:
                if isinstance(item, dict):
                    text = str(item.get('text') or item.get('description') or '')
                else:
                    text = str(item or '')
                if text:
                    texts.append(_sanitize_user_text(text))
    for key in ('description', 'produceDescription'):
        value = str(row.get(key) or '')
        if value:
            texts.append(_sanitize_user_text(value))
    return ' / '.join(texts)


def _effect_axis_summary(effect_types: list[str]) -> str:
    """把效果轴标签拼成面向人读的短摘要。"""

    clean = [str(value) for value in effect_types if value]
    return ' / '.join(clean)


def _card_context(runtime: ExamRuntime, card: RuntimeCard) -> CardPromptContext:
    """构造一张运行时卡牌的结构化摘要。"""

    repository = runtime.repository
    raw_name = str(card.base_card.get('name') or card.card_id)
    effect_summary = _effect_axis_summary(repository.card_axis_effect_types(card.base_card))
    description = _description_text(card.base_card)
    cost = float(card.base_card.get('stamina') or card.base_card.get('forceStamina') or 0.0)
    cost_summary = f'体力消耗 {cost:g}' if cost else '无体力消耗'
    preview_summary = effect_summary or description or '无追加说明'
    return CardPromptContext(
        id=str(card.card_id),
        name=raw_name,
        upgrade_count=int(card.upgrade_count),
        description=description,
        effect_summary=effect_summary,
        cost_summary=cost_summary,
        preview_summary=preview_summary,
        available=runtime._can_play_card(card),
    )


def _drink_context(runtime: ExamRuntime, drink: dict[str, Any]) -> DrinkPromptContext:
    """构造一瓶 P 饮料的结构化摘要。"""

    repository = runtime.repository
    raw_name = str(drink.get('name') or drink.get('id') or '')
    effect_summary = _effect_axis_summary(repository.drink_axis_effect_types(drink))
    description = _description_text(drink)
    preview_summary = effect_summary or description or '无追加说明'
    available = (not bool(drink.get('_consumed'))) and runtime._can_use_drink(drink)
    return DrinkPromptContext(
        id=str(drink.get('id') or ''),
        name=raw_name,
        description=description,
        effect_summary=effect_summary,
        preview_summary=preview_summary,
        available=available,
    )


def _produce_item_context(runtime: ExamRuntime, repository: MasterDataRepository) -> ProduceItemPromptContext | None:
    """构造当前 P 道具摘要。"""

    if runtime.loadout is None or not runtime.loadout.produce_item_id:
        return None
    item_id = str(runtime.loadout.produce_item_id)
    row = repository.load_table('ProduceItem').first(item_id)
    if row is None:
        return ProduceItemPromptContext(id=item_id, name=item_id, description='', enchants=[])
    enchants: list[str] = []
    for enchant_id in runtime.loadout.exam_status_enchant_ids:
        enchant_row = repository.exam_status_enchant_map.get(str(enchant_id))
        if enchant_row is not None:
            description = _description_text(enchant_row)
            if description:
                enchants.append(description)
    return ProduceItemPromptContext(
        id=item_id,
        name=str(row.get('name') or item_id),
        description=_description_text(row),
        enchants=enchants,
    )


def _lesson_remaining_text(value: float) -> str:
    """把课程剩余目标格式化为无多余小数的文本。"""

    bounded = max(float(value), 0.0)
    return str(int(bounded)) if abs(bounded - int(bounded)) <= 1e-9 else f'{bounded:.2f}'


def _effect_status(remaining_turns: int | None, remaining_count: int | None) -> str:
    """格式化持续效果的剩余回合和次数。"""

    parts: list[str] = []
    if remaining_turns is None:
        parts.append('永久')
    else:
        parts.append(f'剩余{int(remaining_turns)}回合')
    if remaining_count is not None:
        parts.append(f'剩余{int(remaining_count)}次')
    return '，'.join(parts)


def _source_label(source: str) -> str:
    """把运行时来源字段转换为可读来源。"""

    return _SOURCE_LABELS.get(str(source or ''), str(source or '不明'))


def _active_effect_sentence(effect: TimedExamEffect) -> str:
    """把活跃持续效果格式化为自然语言句子。"""

    effect_type = str(effect.effect.get('effectType') or '')
    sentence = _EFFECT_SENTENCES.get(effect_type)
    if sentence is None:
        sentence = _sanitize_user_text(_description_text(effect.effect) or '未命名效果')
    status = _effect_status(effect.remaining_turns, effect.remaining_count)
    return f'效果：{sentence}；状态：{status}；来源：{_source_label(effect.source)}'


def _active_enchant_sentence(repository: MasterDataRepository, enchant: TriggeredEnchant) -> str:
    """把活跃附魔格式化为自然语言句子。"""

    row = repository.exam_status_enchant_map.get(str(enchant.enchant_id), {})
    description = _sanitize_user_text(_description_text(row)).replace('数值变为打分上升', '转为打分')
    if not description:
        description = '未命名附魔'
    status = _effect_status(enchant.remaining_turns, enchant.remaining_count)
    return f'附魔：{description}；状态：{status}；来源：{_source_label(enchant.source)}'


def extract_state_context(runtime: ExamRuntime, repository: MasterDataRepository) -> dict[str, Any]:
    """提取可测试、可渲染的结构化考试局面上下文。"""

    battle_kind = str(runtime.battle_kind)
    lesson_target_remaining = runtime._lesson_target_remaining()
    lesson_perfect_remaining = runtime._lesson_perfect_remaining()
    available_drinks = [_drink_context(runtime, drink) for drink in runtime.drinks if not bool(drink.get('_consumed'))]
    item_context = _produce_item_context(runtime, repository)
    loadout_context = None
    if runtime.loadout is not None:
        loadout_context = LoadoutPromptContext(
            idol_card_id=str(runtime.loadout.idol_card_id),
            produce_item=item_context,
        )
    context = StatePromptContext(
        battle_kind=battle_kind,
        clear_state=str(runtime.clear_state),
        lesson_cleared=bool(runtime.lesson_cleared),
        lesson_target_remaining=_lesson_remaining_text(lesson_target_remaining),
        lesson_perfect_remaining=_lesson_remaining_text(lesson_perfect_remaining),
        turn_color=str(runtime.current_turn_color or ''),
        turn_color_label=runtime.turn_color_label(),
        turn_color_display_label=_TURN_COLOR_DISPLAY_LABELS.get(str(runtime.current_turn_color or ''), ''),
        score=float(runtime.score),
        stamina=float(runtime.stamina),
        max_stamina=float(runtime.max_stamina),
        turn=int(runtime.turn),
        max_turns=int(runtime.max_turns),
        used_drink_count=sum(1 for drink in runtime.drinks if bool(drink.get('_consumed'))),
        available_drink_count=len(available_drinks),
        drink_total_count=len(runtime.drinks),
        resources={
            '好調': float(runtime.resources[RESOURCE_REVIEW]),
            '好印象': float(runtime.resources[RESOURCE_AGGRESSIVE]),
            'やる気': float(runtime.resources[RESOURCE_ENTHUSIASTIC]),
            '元気': float(runtime.resources[RESOURCE_BLOCK]),
            '集中': float(runtime.resources[RESOURCE_CONCENTRATION]),
            '全力値': float(runtime.resources[RESOURCE_FULL_POWER_POINT]),
        },
        hand=[_card_context(runtime, card) for card in runtime.hand],
        deck_cards=[_card_context(runtime, card) for card in runtime.deck],
        drinks=available_drinks,
        loadout=loadout_context,
        active_effects=[_active_effect_sentence(effect) for effect in runtime.active_effects],
        active_enchants=[_active_enchant_sentence(repository, enchant) for enchant in runtime.active_enchants],
    )
    return asdict(context)


def _clear_state_label(context: dict[str, Any]) -> str:
    """把清课状态转换为摘要显示文本。"""

    if context['battle_kind'] != 'lesson':
        return ''
    if context['clear_state'] == 'perfect':
        return 'PERFECT'
    if context['lesson_cleared']:
        return '目標達成'
    return '未達成'


def _format_resources(resources: dict[str, float]) -> str:
    """格式化主要考试资源。"""

    return ' / '.join(f'{name}={value:g}' for name, value in resources.items())


def build_state_snapshot(runtime: ExamRuntime, repository: MasterDataRepository) -> str:
    """构造面向调试和提示词审查的可读局面快照。"""

    context = extract_state_context(runtime, repository)
    lines: list[str] = [
        '### 当前结算:',
        f"模式: {'レッスン' if context['battle_kind'] == 'lesson' else '試験'}",
        f"分数: {context['score']:g}",
        f"体力: {context['stamina']:g}/{context['max_stamina']:g}",
        f"当前回合颜色: {context['turn_color_display_label'] or '-'}",
        f"状态: {_format_resources(context['resources'])}",
        '本回合剩余スキルカード使用数: '
        f"{max(int(runtime.play_limit) - int(runtime.turn_counters['play_count']), 0)}",
    ]
    if context['battle_kind'] == 'lesson':
        lines.extend(
            [
                f"课程状态: {_clear_state_label(context)}",
                f"目标剩余: {context['lesson_target_remaining']}",
                f"パーフェクト剩余: {context['lesson_perfect_remaining']}",
            ]
        )
    loadout = context['loadout']
    if loadout and loadout['produce_item']:
        item = loadout['produce_item']
        lines.extend(
            [
                f"Pアイテム: {item['name']}",
                f"Pアイテム说明: {item['description']}",
                'Pアイテム附带附魔（开场装载）: '
                + (' / '.join(item['enchants']) if item['enchants'] else 'なし'),
            ]
        )
    lines.append(f"Pドリンク库存: {context['available_drink_count']}/{context['drink_total_count']}")
    for drink in context['drinks']:
        lines.append(f"- {drink['name']}: {drink['preview_summary']}；说明: {drink['description']}")
    lines.append('### 手牌')
    for card in context['hand']:
        lines.append(
            f"- {card['name']}[{card['upgrade_count']}]: "
            f"卡面说明: {card['description'] or card['effect_summary']}; "
            f"消耗: {card['cost_summary']}; 预览: {card['preview_summary']}"
        )
    lines.append('### 牌库顺序明细（顶 -> 底）')
    for index, card in enumerate(context['deck_cards'], start=1):
        lines.append(f"{index}. {card['name']}[{card['upgrade_count']}] 卡面说明: {card['description'] or card['effect_summary']}")
    lines.extend(
        [
            'おすすめ効果: 好調 / 集中',
            '「好調」や「集中」を活用して育成するプランです。',
            f"参数面板: ボーカル={runtime.parameter_stats[0]:g} / ダンス={runtime.parameter_stats[1]:g} / ビジュアル={runtime.parameter_stats[2]:g}",
        ]
    )
    if runtime.loadout is not None:
        profile = runtime.loadout.stat_profile
        lines.append(f'基础属性: ボーカル={profile.vocal:g} / ダンス={profile.dance:g} / ビジュアル={profile.visual:g}')
    else:
        lines.append('基础属性: ボーカル=0 / ダンス=0 / ビジュアル=0')
    if context['active_effects']:
        lines.append('### 活跃效果')
        lines.extend(context['active_effects'])
    if context['active_enchants']:
        lines.append('### 活跃附魔')
        lines.extend(context['active_enchants'])
    return '\n'.join(_sanitize_user_text(line) for line in lines)


def action_label_for_llm(env: Any, action_index: int) -> str:
    """返回动作槽在可读摘要中应展示的标签。"""

    candidate = env._candidates[int(action_index)]
    kind = str(candidate.kind)
    payload = dict(candidate.payload)
    if kind == EXAM_ACTION_CARD:
        uid = int(payload.get('uid') or -1)
        for card in env.runtime.hand:
            if int(card.uid) == uid:
                return f'{card.base_card.get("name") or card.card_id}[{int(card.upgrade_count)}]'
    if kind == EXAM_ACTION_DRINK:
        index = int(payload.get('index') or 0)
        if 0 <= index < len(env.runtime.drinks):
            drink = env.runtime.drinks[index]
            return str(drink.get('name') or drink.get('id') or '')
    if kind == EXAM_ACTION_END_TURN and env.runtime.battle_kind == 'lesson':
        return 'SKIP'
    return str(candidate.label)


def extract_action_list_context(env: Any) -> list[dict[str, Any]]:
    """提取环境当前动作槽的可读列表上下文。"""

    actions: list[ActionPromptContext] = []
    mask = env.action_masks()
    for index, candidate in enumerate(env._candidates):
        actions.append(
            ActionPromptContext(
                index=index,
                label=action_label_for_llm(env, index),
                kind=str(candidate.kind),
                available=bool(mask[index]),
            )
        )
    return [asdict(action) for action in actions]
