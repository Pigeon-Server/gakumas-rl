"""Manual exam setup jsonl loader for real in-game deck / drink / item overrides."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import json
from pathlib import Path
from typing import Any, Iterable

from .data import MasterDataRepository


@dataclass(frozen=True)
class ManualDeckCardSpec:
    """One manual card entry, optionally repeated by `count`."""

    card_id: str
    upgrade_count: int = 0
    count: int = 1


@dataclass(frozen=True)
class ManualExamSetupRecord:
    """One jsonl record describing a real exam opening setup."""

    label: str = ''
    scenario: str = ''
    idol_card_id: str = ''
    stage_type: str = ''
    deck: tuple[ManualDeckCardSpec, ...] = ()
    drinks: tuple[str, ...] = ()
    produce_items: tuple[str, ...] = ()
    notes: str = ''
    metadata: dict[str, Any] = field(default_factory=dict)
    source_path: str = ''
    line_number: int = 0


@dataclass(frozen=True)
class ResolvedManualExamSetup:
    """Repository-resolved runtime payload derived from one manual record."""

    record: ManualExamSetupRecord
    deck_rows: tuple[dict[str, Any], ...]
    drink_rows: tuple[dict[str, Any], ...]
    produce_item_ids: tuple[str, ...]


def _read_card_spec(raw: Any, *, source_path: str, line_number: int) -> ManualDeckCardSpec:
    if isinstance(raw, str):
        card_id = raw.strip()
        if not card_id:
            raise ValueError(f'{source_path}:{line_number} card id cannot be empty')
        return ManualDeckCardSpec(card_id=card_id)
    if not isinstance(raw, dict):
        raise TypeError(f'{source_path}:{line_number} deck entry must be a string or object')
    card_id = str(raw.get('card_id') or raw.get('id') or '').strip()
    if not card_id:
        raise ValueError(f'{source_path}:{line_number} deck entry is missing card_id')
    upgrade_count = int(raw.get('upgrade_count') or raw.get('upgradeCount') or 0)
    count = int(raw.get('count') or 1)
    if count <= 0:
        raise ValueError(f'{source_path}:{line_number} deck entry count must be positive: {card_id}')
    if upgrade_count < 0:
        raise ValueError(f'{source_path}:{line_number} upgrade_count cannot be negative: {card_id}')
    return ManualDeckCardSpec(card_id=card_id, upgrade_count=upgrade_count, count=count)


def _read_string_list(raw: Any, *, source_path: str, line_number: int, field_name: str) -> tuple[str, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise TypeError(f'{source_path}:{line_number} field {field_name} must be a list')
    values: list[str] = []
    for item in raw:
        value = str(item or '').strip()
        if not value:
            continue
        values.append(value)
    return tuple(values)


def _record_from_payload(payload: dict[str, Any], *, source_path: str, line_number: int) -> ManualExamSetupRecord:
    raw_deck = payload.get('deck')
    if raw_deck is None:
        raw_deck = payload.get('cards')
    if raw_deck is None:
        raise ValueError(f'{source_path}:{line_number} is missing deck/cards')
    if not isinstance(raw_deck, list):
        raise TypeError(f'{source_path}:{line_number} field deck/cards must be a list')
    deck = tuple(_read_card_spec(item, source_path=source_path, line_number=line_number) for item in raw_deck)
    if not deck:
        raise ValueError(f'{source_path}:{line_number} deck/cards cannot be empty')
    drinks = _read_string_list(
        payload.get('drinks'),
        source_path=source_path,
        line_number=line_number,
        field_name='drinks',
    )
    produce_items = _read_string_list(
        payload.get('produce_items', payload.get('items')),
        source_path=source_path,
        line_number=line_number,
        field_name='produce_items/items',
    )
    metadata = payload.get('metadata') if isinstance(payload.get('metadata'), dict) else {}
    return ManualExamSetupRecord(
        label=str(payload.get('label') or payload.get('name') or '').strip(),
        scenario=str(payload.get('scenario') or '').strip(),
        idol_card_id=str(payload.get('idol_card_id') or '').strip(),
        stage_type=str(payload.get('stage_type') or '').strip(),
        deck=deck,
        drinks=drinks,
        produce_items=produce_items,
        notes=str(payload.get('notes') or '').strip(),
        metadata=dict(metadata),
        source_path=source_path,
        line_number=line_number,
    )


@dataclass(frozen=True)
class ManualExamSetupDataset:
    """All records loaded from one or more jsonl files."""

    records: tuple[ManualExamSetupRecord, ...]

    def sample(
        self,
        rng,
        *,
        scenario_tokens: Iterable[str],
        fixed_idol_card_id: str = '',
    ) -> ManualExamSetupRecord:
        candidates: list[ManualExamSetupRecord] = []
        normalized_scenarios = {str(token or '').strip() for token in scenario_tokens if str(token or '').strip()}
        normalized_fixed_idol = str(fixed_idol_card_id or '').strip()
        for record in self.records:
            if normalized_scenarios and record.scenario and record.scenario not in normalized_scenarios:
                continue
            if normalized_fixed_idol and record.idol_card_id and record.idol_card_id != normalized_fixed_idol:
                continue
            candidates.append(record)
        if not candidates:
            raise RuntimeError(
                'No manual exam setup record matched '
                f'scenarios={sorted(normalized_scenarios)} idol_card_id={normalized_fixed_idol or "<any>"}'
            )
        return candidates[int(rng.integers(0, len(candidates)))]

    def resolve(
        self,
        repository: MasterDataRepository,
        record: ManualExamSetupRecord,
    ) -> ResolvedManualExamSetup:
        deck_rows: list[dict[str, Any]] = []
        for card_spec in record.deck:
            rows = repository.produce_cards.all(card_spec.card_id)
            if not rows:
                raise KeyError(
                    f'Unknown manual deck card: {card_spec.card_id} '
                    f'({record.source_path}:{record.line_number})'
                )
            card_row = next(
                (
                    row
                    for row in rows
                    if int(row.get('upgradeCount') or 0) == int(card_spec.upgrade_count)
                ),
                None,
            )
            if card_row is None:
                raise KeyError(
                    f'Unknown card variant: {card_spec.card_id}@{card_spec.upgrade_count} '
                    f'({record.source_path}:{record.line_number})'
                )
            for _ in range(card_spec.count):
                deck_rows.append(dict(card_row))
        drink_rows: list[dict[str, Any]] = []
        for drink_id in record.drinks:
            drink_row = repository.produce_drinks.first(drink_id)
            if drink_row is None:
                raise KeyError(
                    f'Unknown manual drink id: {drink_id} ({record.source_path}:{record.line_number})'
                )
            drink_rows.append(dict(drink_row))
        produce_item_table = repository.load_table('ProduceItem')
        for item_id in record.produce_items:
            if produce_item_table.first(item_id) is None:
                raise KeyError(
                    f'Unknown manual produce item id: {item_id} '
                    f'({record.source_path}:{record.line_number})'
                )
        return ResolvedManualExamSetup(
            record=record,
            deck_rows=tuple(deck_rows),
            drink_rows=tuple(drink_rows),
            produce_item_ids=tuple(record.produce_items),
        )


@lru_cache(maxsize=16)
def _load_manual_exam_setup_dataset_cached(paths: tuple[str, ...]) -> ManualExamSetupDataset:
    records: list[ManualExamSetupRecord] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f'Manual exam setup file not found: {path}')
        with path.open('r', encoding='utf-8') as handle:
            for line_number, line in enumerate(handle, start=1):
                text = line.strip()
                if not text:
                    continue
                payload = json.loads(text)
                if not isinstance(payload, dict):
                    raise TypeError(f'{path}:{line_number} must be a JSON object per line')
                records.append(
                    _record_from_payload(payload, source_path=str(path), line_number=line_number)
                )
    return ManualExamSetupDataset(records=tuple(records))


def load_manual_exam_setup_dataset(paths: Iterable[str | Path] | None) -> ManualExamSetupDataset | None:
    """Load and cache one or more manual exam setup jsonl files."""

    if paths is None:
        return None
    normalized_paths = tuple(str(Path(path)) for path in paths if str(path or '').strip())
    if not normalized_paths:
        return None
    return _load_manual_exam_setup_dataset_cached(normalized_paths)
