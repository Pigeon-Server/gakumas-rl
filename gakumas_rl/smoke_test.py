"""快速烟雾测试，验证两类环境都能完成 reset 和单步执行。"""

from __future__ import annotations

import json

from .data import MasterDataRepository
from .envs import GakumasExamEnv, GakumasPlanningEnv


def run_smoke() -> dict:
    """构造两个环境并返回首步观测与一步动作后的摘要。"""

    repository = MasterDataRepository()
    planning_scenario = repository.build_scenario('produce-003')
    exam_scenario = repository.build_scenario('produce-005')

    planning_env = GakumasPlanningEnv(repository, planning_scenario)
    exam_env = GakumasExamEnv(repository, exam_scenario)

    planning_obs, planning_info = planning_env.reset(seed=7)
    exam_obs, exam_info = exam_env.reset(seed=7)

    planning_action = next(idx for idx, flag in enumerate(planning_obs['action_mask']) if flag > 0.5)
    exam_action = next(idx for idx, flag in enumerate(exam_obs['action_mask']) if flag > 0.5)

    planning_step = planning_env.step(planning_action)
    exam_step = exam_env.step(exam_action)

    return {
        'planning': {
            'global_shape': list(planning_obs['global'].shape),
            'action_shape': list(planning_obs['action_features'].shape),
            'labels': planning_info['action_labels'],
            'reward': planning_step[1],
        },
        'exam': {
            'global_shape': list(exam_obs['global'].shape),
            'action_shape': list(exam_obs['action_features'].shape),
            'labels': exam_info['action_labels'],
            'reward': exam_step[1],
        },
    }


if __name__ == '__main__':
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2))
