"""Shared parsers and helpers for initial deck guarantee / force-card constraints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping


def _alias_key(value: str) -> str:
    return ''.join(ch for ch in str(value or '').strip().lower() if ch not in {' ', '-', '_'})


_EXAM_EFFECT_ALIASES = {
    _alias_key('score'): 'ProduceExamEffectType_Score',
    _alias_key('scoring'): 'ProduceExamEffectType_Score',
    _alias_key('打分'): 'ProduceExamEffectType_Score',
    _alias_key('review'): 'ProduceExamEffectType_Review',
    _alias_key('好印象'): 'ProduceExamEffectType_Review',
    _alias_key('block'): 'ProduceExamEffectType_Block',
    _alias_key('genki'): 'ProduceExamEffectType_Block',
    _alias_key('元气'): 'ProduceExamEffectType_Block',
    _alias_key('元気'): 'ProduceExamEffectType_Block',
    _alias_key('aggressive'): 'ProduceExamEffectType_Aggressive',
    _alias_key('yaruki'): 'ProduceExamEffectType_Aggressive',
    _alias_key('干劲'): 'ProduceExamEffectType_Aggressive',
    _alias_key('干勁'): 'ProduceExamEffectType_Aggressive',
    _alias_key('やる気'): 'ProduceExamEffectType_Aggressive',
    _alias_key('concentration'): 'ProduceExamEffectType_Concentration',
    _alias_key('strong'): 'ProduceExamEffectType_Concentration',
    _alias_key('强气'): 'ProduceExamEffectType_Concentration',
    _alias_key('強気'): 'ProduceExamEffectType_Concentration',
    _alias_key('fullpower'): 'ProduceExamEffectType_FullPowerPoint',
    _alias_key('fullpowerpoint'): 'ProduceExamEffectType_FullPowerPoint',
    _alias_key('全力'): 'ProduceExamEffectType_FullPowerPoint',
    _alias_key('全力pt'): 'ProduceExamEffectType_FullPowerPoint',
    _alias_key('parameterbuff'): 'ProduceExamEffectType_ParameterBuff',
    _alias_key('goodcondition'): 'ProduceExamEffectType_ParameterBuff',
    _alias_key('好调'): 'ProduceExamEffectType_ParameterBuff',
    _alias_key('好調'): 'ProduceExamEffectType_ParameterBuff',
    _alias_key('lessonbuff'): 'ProduceExamEffectType_LessonBuff',
    _alias_key('lesson'): 'ProduceExamEffectType_LessonBuff',
    _alias_key('preservation'): 'ProduceExamEffectType_Preservation',
    _alias_key('温存'): 'ProduceExamEffectType_Preservation',
    _alias_key('overpreservation'): 'ProduceExamEffectType_OverPreservation',
    _alias_key('余温存'): 'ProduceExamEffectType_OverPreservation',
    _alias_key('enthusiastic'): 'ProduceExamEffectType_Enthusiastic',
    _alias_key('熱意'): 'ProduceExamEffectType_Enthusiastic',
    _alias_key('sleepy'): 'ProduceExamEffectType_Sleepy',
    _alias_key('眠气'): 'ProduceExamEffectType_Sleepy',
    _alias_key('眠気'): 'ProduceExamEffectType_Sleepy',
    _alias_key('panic'): 'ProduceExamEffectType_Panic',
    _alias_key('パニック'): 'ProduceExamEffectType_Panic',
}

_FORCE_AXIS_ALIASES = {
    _alias_key('好调'): 'ProduceExamEffectType_ParameterBuff',
    _alias_key('好調'): 'ProduceExamEffectType_ParameterBuff',
    _alias_key('parameterbuff'): 'ProduceExamEffectType_ParameterBuff',
    _alias_key('ProduceExamEffectType_ParameterBuff'): 'ProduceExamEffectType_ParameterBuff',
    _alias_key('produceexameffecttype_examparameterbuff'): 'ProduceExamEffectType_ParameterBuff',
    _alias_key('集中'): 'ProduceExamEffectType_LessonBuff',
    _alias_key('lessonbuff'): 'ProduceExamEffectType_LessonBuff',
    _alias_key('ProduceExamEffectType_LessonBuff'): 'ProduceExamEffectType_LessonBuff',
    _alias_key('produceexameffecttype_examlessonbuff'): 'ProduceExamEffectType_LessonBuff',
    _alias_key('好印象'): 'ProduceExamEffectType_Review',
    _alias_key('review'): 'ProduceExamEffectType_Review',
    _alias_key('ProduceExamEffectType_Review'): 'ProduceExamEffectType_Review',
    _alias_key('produceexameffecttype_examreview'): 'ProduceExamEffectType_Review',
    _alias_key('干劲'): 'ProduceExamEffectType_Aggressive',
    _alias_key('干勁'): 'ProduceExamEffectType_Aggressive',
    _alias_key('やる気'): 'ProduceExamEffectType_Aggressive',
    _alias_key('yaruki'): 'ProduceExamEffectType_Aggressive',
    _alias_key('aggressive'): 'ProduceExamEffectType_Aggressive',
    _alias_key('ProduceExamEffectType_Aggressive'): 'ProduceExamEffectType_Aggressive',
    _alias_key('produceexameffecttype_examcardplayaggressive'): 'ProduceExamEffectType_Aggressive',
    _alias_key('强气'): 'ProduceExamEffectType_Concentration',
    _alias_key('強気'): 'ProduceExamEffectType_Concentration',
    _alias_key('concentration'): 'ProduceExamEffectType_Concentration',
    _alias_key('ProduceExamEffectType_Concentration'): 'ProduceExamEffectType_Concentration',
    _alias_key('produceexameffecttype_examconcentration'): 'ProduceExamEffectType_Concentration',
    _alias_key('全力'): 'ProduceExamEffectType_FullPowerPoint',
    _alias_key('fullpower'): 'ProduceExamEffectType_FullPowerPoint',
    _alias_key('fullpowerpoint'): 'ProduceExamEffectType_FullPowerPoint',
    _alias_key('ProduceExamEffectType_FullPowerPoint'): 'ProduceExamEffectType_FullPowerPoint',
    _alias_key('produceexameffecttype_examfullpower'): 'ProduceExamEffectType_FullPowerPoint',
    _alias_key('produceexameffecttype_examfullpowerpoint'): 'ProduceExamEffectType_FullPowerPoint',
    _alias_key('温存'): 'ProduceExamEffectType_Preservation',
    _alias_key('preservation'): 'ProduceExamEffectType_Preservation',
    _alias_key('ProduceExamEffectType_Preservation'): 'ProduceExamEffectType_Preservation',
    _alias_key('余温存'): 'ProduceExamEffectType_OverPreservation',
    _alias_key('overpreservation'): 'ProduceExamEffectType_OverPreservation',
    _alias_key('ProduceExamEffectType_OverPreservation'): 'ProduceExamEffectType_OverPreservation',
}

_CANONICAL_EFFECT_TYPES = (
    'ProduceExamEffectType_Score',
    'ProduceExamEffectType_Review',
    'ProduceExamEffectType_Block',
    'ProduceExamEffectType_Aggressive',
    'ProduceExamEffectType_Concentration',
    'ProduceExamEffectType_FullPowerPoint',
    'ProduceExamEffectType_ParameterBuff',
    'ProduceExamEffectType_LessonBuff',
    'ProduceExamEffectType_Preservation',
    'ProduceExamEffectType_OverPreservation',
    'ProduceExamEffectType_Enthusiastic',
    'ProduceExamEffectType_Sleepy',
    'ProduceExamEffectType_Panic',
)

_EFFECT_AXIS_MARKERS = {
    'ProduceExamEffectType_Score': (
        'ProduceExamEffectType_Score',
        'ProduceExamEffectType_ExamLesson',
    ),
    'ProduceExamEffectType_Review': (
        'ProduceExamEffectType_Review',
        'ProduceExamEffectType_ExamReview',
    ),
    'ProduceExamEffectType_Block': (
        'ProduceExamEffectType_Block',
        'ProduceExamEffectType_ExamBlock',
    ),
    'ProduceExamEffectType_Aggressive': (
        'ProduceExamEffectType_Aggressive',
        'ProduceExamEffectType_ExamAggressive',
        'ProduceExamEffectType_ExamCardPlayAggressive',
    ),
    'ProduceExamEffectType_Concentration': (
        'ProduceExamEffectType_Concentration',
        'ProduceExamEffectType_ExamConcentration',
    ),
    'ProduceExamEffectType_FullPowerPoint': (
        'ProduceExamEffectType_FullPowerPoint',
        'ProduceExamEffectType_ExamFullPowerPoint',
    ),
    'ProduceExamEffectType_ParameterBuff': (
        'ProduceExamEffectType_ParameterBuff',
        'ProduceExamEffectType_ExamParameterBuff',
    ),
    'ProduceExamEffectType_LessonBuff': (
        'ProduceExamEffectType_LessonBuff',
        'ProduceExamEffectType_ExamLessonBuff',
    ),
    'ProduceExamEffectType_Preservation': (
        'ProduceExamEffectType_Preservation',
        'ProduceExamEffectType_ExamPreservation',
    ),
    'ProduceExamEffectType_OverPreservation': (
        'ProduceExamEffectType_OverPreservation',
        'ProduceExamEffectType_ExamOverPreservation',
    ),
    'ProduceExamEffectType_Enthusiastic': (
        'ProduceExamEffectType_Enthusiastic',
        'ProduceExamEffectType_ExamEnthusiastic',
    ),
    'ProduceExamEffectType_Sleepy': (
        'ProduceExamEffectType_Sleepy',
        'ProduceExamEffectType_ExamSleepy',
    ),
    'ProduceExamEffectType_Panic': (
        'ProduceExamEffectType_Panic',
        'ProduceExamEffectType_ExamPanic',
    ),
}

UNSCOPED_FORCE_CARD_GROUP = '__unscoped__'


def normalize_exam_effect_type(value: str) -> str:
    """Resolve a CLI/user-facing token into a canonical `ProduceExamEffectType_*`."""

    normalized = str(value or '').strip()
    if not normalized:
        raise ValueError('exam effect type cannot be empty')
    if normalized in _CANONICAL_EFFECT_TYPES:
        return normalized
    if normalized.startswith('ProduceExamEffectType_'):
        for canonical, markers in _EFFECT_AXIS_MARKERS.items():
            if any(marker in normalized for marker in markers):
                return canonical
    canonical = _EXAM_EFFECT_ALIASES.get(_alias_key(normalized))
    if canonical is not None:
        return canonical
    raise ValueError(f'Unknown exam effect type alias: {value}')


def short_effect_type_name(value: str) -> str:
    """Return the short suffix without the `ProduceExamEffectType_` prefix."""

    normalized = str(value or '').strip()
    if normalized.startswith('ProduceExamEffectType_'):
        return normalized[len('ProduceExamEffectType_'):]
    return normalized


def normalize_force_axis_type(value: str) -> str:
    """Resolve a user-facing idol-axis token used by `--force-card`."""

    normalized = str(value or '').strip()
    if not normalized:
        raise ValueError('force-card axis cannot be empty')
    canonical = _FORCE_AXIS_ALIASES.get(_alias_key(normalized))
    if canonical is not None:
        return canonical
    raise ValueError(
        f'Unknown force-card axis alias: {value}. '
        'Expected one of: 好调, 集中, 好印象, 干劲, 强气, 全力, 温存, 余温存.'
    )


def summarize_exam_effect_axes(effect_types: Iterable[str] | None) -> tuple[str, ...]:
    """Project raw master-data effect types onto user-facing axis buckets."""

    if effect_types is None:
        return ()
    raw_values = [str(value or '').strip() for value in effect_types if str(value or '').strip()]
    if not raw_values:
        return ()
    resolved: list[str] = []
    seen: set[str] = set()
    for canonical in _CANONICAL_EFFECT_TYPES:
        markers = _EFFECT_AXIS_MARKERS.get(canonical, ())
        if any(any(marker in raw_value for marker in markers) for raw_value in raw_values):
            seen.add(canonical)
            resolved.append(canonical)
    for raw_value in raw_values:
        try:
            canonical = normalize_exam_effect_type(raw_value)
        except ValueError:
            continue
        if canonical in seen:
            continue
        seen.add(canonical)
        resolved.append(canonical)
    return tuple(resolved)


def normalize_guaranteed_effect_counts(
    raw: Mapping[str, int] | Iterable[str] | None,
) -> dict[str, int]:
    """Normalize either a mapping or repeated CLI specs into canonical counts."""

    if raw is None:
        return {}
    normalized: dict[str, int] = {}
    if isinstance(raw, Mapping):
        items = raw.items()
    else:
        parsed_items: list[tuple[str, int]] = []
        for spec in raw:
            text = str(spec or '').strip()
            if not text:
                continue
            separator = '=' if '=' in text else ':'
            if separator not in text:
                raise ValueError(f'Invalid guarantee-card-effect spec: {text}. Expected effect=count.')
            effect_token, count_token = text.split(separator, 1)
            parsed_items.append((effect_token.strip(), int(count_token.strip())))
        items = parsed_items
    for effect_token, count in items:
        canonical = normalize_exam_effect_type(str(effect_token))
        numeric = int(count)
        if numeric < 0:
            raise ValueError(f'Guaranteed card count cannot be negative: {effect_token}={count}')
        if numeric <= 0:
            continue
        normalized[canonical] = normalized.get(canonical, 0) + numeric
    return normalized


def _coerce_force_card_value(raw_value: Any) -> list[str]:
    if raw_value is None:
        return []
    if isinstance(raw_value, str):
        text = raw_value.strip()
        return [text] if text else []
    if isinstance(raw_value, Iterable):
        values: list[str] = []
        for item in raw_value:
            text = str(item or '').strip()
            if text:
                values.append(text)
        return values
    text = str(raw_value).strip()
    return [text] if text else []


def _load_force_card_mapping_spec(spec: str) -> Mapping[str, Any]:
    text = str(spec or '').strip()
    if not text:
        return {}
    if text.startswith('@'):
        payload = json.loads(Path(text[1:]).read_text(encoding='utf-8'))
    elif text.startswith('{'):
        payload = json.loads(text)
    else:
        separator = '=' if '=' in text else ':'
        if separator not in text:
            return {UNSCOPED_FORCE_CARD_GROUP: [text]}
        axis_token, card_tokens = text.split(separator, 1)
        payload = {
            axis_token.strip(): [token.strip() for token in card_tokens.split(',') if token.strip()],
        }
    if not isinstance(payload, Mapping):
        raise ValueError(f'Invalid force-card payload: expected object mapping, got {type(payload).__name__}')
    return payload


def normalize_forced_card_groups(
    raw: Mapping[str, Any] | Iterable[Any] | str | None,
) -> dict[str, tuple[str, ...]]:
    """Normalize axis-scoped force-card specs into canonical effect groups."""

    if raw is None:
        return {}
    grouped: dict[str, list[str]] = {}
    if isinstance(raw, Mapping):
        specs: list[Mapping[str, Any]] = [raw]
    elif isinstance(raw, str):
        specs = [_load_force_card_mapping_spec(raw)]
    else:
        specs = []
        for item in raw:
            if isinstance(item, Mapping):
                specs.append(item)
                continue
            spec = str(item or '').strip()
            if not spec:
                continue
            specs.append(_load_force_card_mapping_spec(spec))
    for mapping in specs:
        for raw_axis, raw_values in mapping.items():
            axis_token = str(raw_axis or '').strip()
            canonical_axis = (
                UNSCOPED_FORCE_CARD_GROUP
                if not axis_token or axis_token == UNSCOPED_FORCE_CARD_GROUP
                else normalize_force_axis_type(axis_token)
            )
            values = _coerce_force_card_value(raw_values)
            if not values:
                continue
            grouped.setdefault(canonical_axis, []).extend(values)
    return {axis: tuple(values) for axis, values in grouped.items()}
