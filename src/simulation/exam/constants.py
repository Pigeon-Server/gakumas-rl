"""考试模拟中与主数据枚举相连的共享常量。"""

from .ids import ExamEffect, ExamPhase, GrowEffect

CARD_ZONE_MAP = {
    'ProduceCardPositionType_Deck': 'deck',
    'ProduceCardPositionType_DeckAll': 'deck',
    'ProduceCardPositionType_DeckGrave': 'deck_grave',
    'ProduceCardPositionType_Hand': 'hand',
    'ProduceCardPositionType_Hold': 'hold',
    'ProduceCardPositionType_Lost': 'lost',
    'ProduceCardPositionType_NotLost': 'not_lost',
    'ProduceCardPositionType_Playing': 'playing',
    'ProduceCardPositionType_Target': 'target',
}

MOVE_POSITION_MAP = {
    'ProduceCardMovePositionType_DeckFirst': 'deck_first',
    'ProduceCardMovePositionType_DeckLast': 'deck_last',
    'ProduceCardMovePositionType_DeckRandom': 'deck_random',
    'ProduceCardMovePositionType_Grave': 'grave',
    'ProduceCardMovePositionType_Hand': 'hand',
    'ProduceCardMovePositionType_Hold': 'hold',
    'ProduceCardMovePositionType_Lost': 'lost',
    'ProduceCardMovePositionType_Unknown': 'grave',
}

PHASE_TURN_VALUE = {
    ExamPhase.START_TURN,
    ExamPhase.END_TURN,
    ExamPhase.END_TURN_INTERVAL,
    ExamPhase.TURN_INTERVAL,
    ExamPhase.TURN_TIMER,
}

SCALAR_RESOURCE_TYPES = {
    ExamEffect.CARD_PLAY_AGGRESSIVE: 'aggressive',
    ExamEffect.REVIEW: 'review',
    ExamEffect.LESSON_BUFF: 'lesson_buff',
    ExamEffect.FULL_POWER_POINT: 'full_power_point',
}

DURATION_RESOURCE_TYPES = {
    ExamEffect.PARAMETER_BUFF: 'parameter_buff',
}

COST_RESOURCE_MAP = {
    'ExamCostType_ExamCardPlayAggressive': 'aggressive',
    'ExamCostType_ExamReview': 'review',
    'ExamCostType_ExamParameterBuff': 'parameter_buff',
    'ExamCostType_ExamParameterBuffMultiplePerTurn': 'parameter_buff_multiple_per_turn',
    'ExamCostType_ExamLessonBuff': 'lesson_buff',
    'ExamCostType_ExamFullPowerPoint': 'full_power_point',
}

GROW_EFFECT_COST_RESOURCE_MAP = {
    GrowEffect.COST_ADD: 'stamina',
    GrowEffect.COST_REDUCE: 'stamina',
    GrowEffect.COST_AGGRESSIVE_ADD: 'aggressive',
    GrowEffect.COST_AGGRESSIVE_REDUCE: 'aggressive',
    GrowEffect.COST_REVIEW_ADD: 'review',
    GrowEffect.COST_REVIEW_REDUCE: 'review',
    GrowEffect.COST_PARAMETER_BUFF_ADD: 'parameter_buff',
    GrowEffect.COST_PARAMETER_BUFF_REDUCE: 'parameter_buff',
    GrowEffect.COST_PARAMETER_BUFF_MULTIPLE_PER_TURN_REDUCE: 'parameter_buff_multiple_per_turn',
    GrowEffect.COST_LESSON_BUFF_ADD: 'lesson_buff',
    GrowEffect.COST_LESSON_BUFF_REDUCE: 'lesson_buff',
    GrowEffect.COST_FULL_POWER_POINT_ADD: 'full_power_point',
    GrowEffect.COST_FULL_POWER_POINT_REDUCE: 'full_power_point',
    GrowEffect.COST_PENETRATE_ADD: 'penetrate',
    GrowEffect.COST_PENETRATE_REDUCE: 'penetrate',
}

LESSON_EFFECT_TYPES = {
    ExamEffect.LESSON_FIX,
    ExamEffect.LESSON_DEPEND_REVIEW,
    ExamEffect.LESSON_DEPEND_AGGRESSIVE,
    ExamEffect.LESSON_DEPEND_BLOCK,
    ExamEffect.LESSON_DEPEND_PARAMETER_BUFF,
    ExamEffect.LESSON_DEPEND_PLAY_CARD_COUNT_SUM,
    ExamEffect.LESSON_DEPEND_STAMINA_CONSUMPTION_SUM,
    ExamEffect.LESSON_DEPEND_BLOCK_CONSUMPTION_SUM,
    ExamEffect.LESSON_DEPEND_BLOCK_AND_SEARCH_COUNT,
    ExamEffect.LESSON_PER_SEARCH_COUNT,
    ExamEffect.LESSON_FULL_POWER_POINT,
    ExamEffect.LESSON_ADD_MULTIPLE_LESSON_BUFF,
    ExamEffect.MULTIPLE_LESSON_BUFF_LESSON,
    ExamEffect.LESSON_ADD_MULTIPLE_PARAMETER_BUFF,
}

STANCE_PHASES = {
    ExamPhase.STANCE_CHANGE_CONCENTRATION: 'concentration',
    ExamPhase.STANCE_CHANGE_FULL_POWER: 'full_power',
    ExamPhase.STANCE_CHANGE_PRESERVATION: 'preservation',
}

FULL_POWER_POINT_THRESHOLD = 10.0
HOLD_CARD_LIMIT = 2
STATUS_CHANGE_TRIGGER_ORIGINS = {'card', 'drink'}

NEGATIVE_TIMED_EFFECT_TYPES = {
    ExamEffect.BLOCK_ADD_DOWN,
    ExamEffect.BLOCK_RESTRICTION,
    ExamEffect.GIMMICK_LESSON_DEBUFF,
    ExamEffect.GIMMICK_PARAMETER_DEBUFF,
    ExamEffect.GIMMICK_SLUMP,
    ExamEffect.PANIC,
    ExamEffect.STANCE_LOCK,
    ExamEffect.STAMINA_RECOVER_RESTRICTION,
}

ANTI_DEBUFF_EFFECT_TYPES = {
    ExamEffect.BLOCK_ADD_DOWN,
    ExamEffect.BLOCK_RESTRICTION,
    ExamEffect.GIMMICK_LESSON_DEBUFF,
    ExamEffect.GIMMICK_PARAMETER_DEBUFF,
    ExamEffect.GIMMICK_PLAY_CARD_LIMIT,
    ExamEffect.GIMMICK_SLEEPY,
    ExamEffect.GIMMICK_SLUMP,
    ExamEffect.GIMMICK_START_TURN_CARD_DRAW_DOWN,
    ExamEffect.PANIC,
    ExamEffect.STANCE_LOCK,
    ExamEffect.STAMINA_RECOVER_RESTRICTION,
}
