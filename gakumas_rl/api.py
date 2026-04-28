"""FastAPI 调试接口，暴露无状态的配装与模拟能力。"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel, Field

from .service import LoadoutConfig, SCENARIO_ALIASES, get_repository, loadout_summary, simulate_exam, simulate_planning

router = APIRouter(prefix='/api/gakumas_rl', tags=['gakumas_rl'])


class LoadoutRequest(BaseModel):
    """通用配装请求参数。"""

    scenario: str = 'nia_master'
    idol_card_id: str = ''
    producer_level: int = 35
    idol_rank: int = 0
    dearness_level: int = 0
    use_after_item: bool | None = None
    produce_card_conversion_after_ids: list[str] = Field(default_factory=list)
    exam_score_bonus_multiplier: float | None = None
    fan_votes: float | None = None
    assist_mode: bool = False
    auto_support_cards: bool = True
    support_card_ids: list[str] = Field(default_factory=list)
    support_card_level: int | None = None

    def to_config(self) -> LoadoutConfig:
        """把 API 请求模型转换成内部 loadout 配置。"""

        return LoadoutConfig(
            idol_card_id=self.idol_card_id,
            producer_level=self.producer_level,
            idol_rank=self.idol_rank,
            dearness_level=self.dearness_level,
            use_after_item=self.use_after_item,
            produce_card_conversion_after_ids=tuple(self.produce_card_conversion_after_ids),
            exam_score_bonus_multiplier=self.exam_score_bonus_multiplier,
            fan_votes=self.fan_votes,
            assist_mode=self.assist_mode,
            auto_support_cards=self.auto_support_cards,
            support_card_ids=tuple(self.support_card_ids),
            support_card_level=self.support_card_level,
        )


class PlanningSimulationRequest(LoadoutRequest):
    """培育规划模拟请求。"""

    actions: list[int] = Field(default_factory=list)
    auto_policy: str | None = None
    auto_steps: int = 0
    seed: int | None = None


class ExamSimulationRequest(LoadoutRequest):
    """考试战斗模拟请求。"""

    stage_type: str | None = None
    actions: list[int] = Field(default_factory=list)
    auto_policy: str | None = None
    auto_steps: int = 0
    seed: int | None = None
    focus_effect_type: str | None = None
    exam_reward_mode: str = 'score'


class ApiResponse(BaseModel):
    """统一 API 响应结构。"""

    status: bool
    message: str = 'OK'
    data: dict[str, Any] | list[Any] | None = None


def _wrap(data: dict[str, Any] | list[Any] | None = None, message: str = 'OK') -> dict[str, Any]:
    """构造统一的成功响应体。"""

    return {'status': True, 'message': message, 'data': data}


@router.get('/scenarios')
def list_scenarios() -> dict[str, Any]:
    """列出支持的场景别名与原始场景 ID。"""

    repository = get_repository()
    supported = repository.list_supported_scenarios()
    return _wrap(
        {
            'aliases': dict(SCENARIO_ALIASES),
            'supported': supported,
        }
    )


@router.post('/loadout')
def resolve_loadout(payload: LoadoutRequest) -> dict[str, Any]:
    """解析偶像卡配置并返回初始卡组摘要。"""

    try:
        return _wrap(loadout_summary(payload.scenario, payload.to_config()))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post('/planning/simulate')
def run_planning(payload: PlanningSimulationRequest) -> dict[str, Any]:
    """执行无状态的培育流程模拟。"""

    try:
        return _wrap(
            simulate_planning(
                payload.scenario,
                actions=payload.actions,
                auto_policy=payload.auto_policy,
                auto_steps=payload.auto_steps,
                seed=payload.seed,
                loadout_config=payload.to_config(),
            )
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post('/exam/simulate')
def run_exam(payload: ExamSimulationRequest) -> dict[str, Any]:
    """执行无状态的考试战斗模拟。"""

    try:
        return _wrap(
            simulate_exam(
                payload.scenario,
                stage_type=payload.stage_type,
                actions=payload.actions,
                auto_policy=payload.auto_policy,
                auto_steps=payload.auto_steps,
                seed=payload.seed,
                loadout_config=payload.to_config(),
                focus_effect_type=payload.focus_effect_type,
                exam_reward_mode=payload.exam_reward_mode,
            )
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def register_rl_routes(app: FastAPI) -> None:
    """把 RL 调试接口挂载到已有 FastAPI 应用。"""

    app.include_router(router)


def create_app() -> FastAPI:
    """创建仅包含 RL 路由的独立 FastAPI 应用。"""

    app = FastAPI()
    register_rl_routes(app)
    return app
