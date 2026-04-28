"""独立可打包的 Gakumas RL 训练与模拟模块。"""

from __future__ import annotations

from .backends import TrainingResult, TrainingSpec, run_training
from .data import EffectTaxonomy, MasterDataRepository, ScenarioSpec
from .idol_config import build_idol_loadout
from .loadout import DeckArchetype, IdolLoadout, IdolStatProfile, ProduceSkillEffect
from .service import LoadoutConfig, SCENARIO_ALIASES, loadout_summary, simulate_exam, simulate_planning

__version__ = '0.1.0'

__all__ = [
    'DeckArchetype',
    'EffectTaxonomy',
    'GakumasExamEnv',
    'GakumasPlanningEnv',
    'IdolLoadout',
    'IdolStatProfile',
    'LoadoutConfig',
    'MasterDataRepository',
    'ProduceSkillEffect',
    'SCENARIO_ALIASES',
    'ScenarioSpec',
    'TrainingResult',
    'TrainingSpec',
    'build_idol_loadout',
    'loadout_summary',
    'run_training',
    'simulate_exam',
    'simulate_planning',
]


def __getattr__(name: str):
    """按需导入依赖 `gymnasium` 的环境封装。"""

    if name in {'GakumasExamEnv', 'GakumasPlanningEnv'}:
        from .envs import GakumasExamEnv, GakumasPlanningEnv

        return {
            'GakumasExamEnv': GakumasExamEnv,
            'GakumasPlanningEnv': GakumasPlanningEnv,
        }[name]
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
