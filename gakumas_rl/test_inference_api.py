"""推理服务 HTTP API 回归测试。"""

from __future__ import annotations

from fastapi.testclient import TestClient

from gakumas_rl.inference_api import _resolve_server_config, create_inference_app
from gakumas_rl.inference_service import InferenceResponse


class _DummyInferenceService:
    def __init__(self) -> None:
        self.last_request = None

    def predict(self, request):
        self.last_request = request
        return InferenceResponse(
            action_index=1,
            action_id='produce_card:card_beta:0',
            db_id='card_beta',
            action_kind='card',
            confidence=1.0,
            value_estimate=1.75,
        )

    def get_info(self):
        return {
            'service': 'gakumas_rl_inference',
            'backend_type': 'ppo',
            'checkpoint_path': 'runs/fixed/model.zip',
        }


def test_resolve_server_config_reads_startup_env(monkeypatch) -> None:
    """服务启动配置应从环境变量读取固定模型，而不是走 HTTP 热加载。"""

    monkeypatch.setenv('GAKUMAS_RL_INFERENCE_BACKEND', 'ppo')
    monkeypatch.setenv('GAKUMAS_RL_INFERENCE_CHECKPOINT', 'runs/fixed/model.zip')
    monkeypatch.setenv('GAKUMAS_RL_INFERENCE_DEVICE', 'cpu')

    backend, checkpoint, device = _resolve_server_config()

    assert backend == 'ppo'
    assert checkpoint == 'runs/fixed/model.zip'
    assert device == 'cpu'


def test_inference_api_uses_fixed_startup_service(monkeypatch) -> None:
    """predict/info 只依赖启动期固定的服务实例。"""

    dummy_service = _DummyInferenceService()
    monkeypatch.setattr(
        'gakumas_rl.inference_api._build_service',
        lambda **_: dummy_service,
    )

    app = create_inference_app(checkpoint_path='runs/fixed/model.zip')
    client = TestClient(app)

    info_response = client.get('/api/inference/info')
    predict_response = client.post(
        '/api/inference/predict',
        json={
            'vocal': 450,
            'dance': 420,
            'visual': 390,
            'stamina': 8,
            'max_stamina': 15,
            'score': 820,
            'target_score': 2000,
            'turn': 3,
            'max_turns': 9,
            'legal_actions': [
                {
                    'index': 0,
                    'action_id': 'produce_card:card_alpha:0',
                    'db_id': 'card_alpha',
                    'kind': 'card',
                    'available': True,
                },
                {
                    'index': 1,
                    'action_id': 'produce_card:card_beta:0',
                    'db_id': 'card_beta',
                    'kind': 'card',
                    'available': True,
                },
            ],
        },
    )

    assert info_response.status_code == 200
    assert info_response.json()['status'] == 'ready'
    assert predict_response.status_code == 200
    assert predict_response.json() == {
        'action_index': 1,
        'action_id': 'produce_card:card_beta:0',
        'db_id': 'card_beta',
        'action_kind': 'card',
        'confidence': 1.0,
        'value_estimate': 1.75,
        'policy_probs': None,
    }
    assert dummy_service.last_request is not None
    assert dummy_service.last_request.legal_actions[1]['db_id'] == 'card_beta'
    assert client.post('/api/inference/load_model').status_code == 404
    assert client.post('/api/inference/unload').status_code == 404
