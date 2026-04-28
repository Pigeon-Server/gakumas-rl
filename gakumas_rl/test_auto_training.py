"""自动训练与早停策略的轻量回归测试。"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType
import numpy as np

from gakumas_rl.auto_training import (
    AutoTrainingConfig,
    DynamicEvaluationScheduler,
    EarlyStoppingCallback,
    ExamRandomizationCurriculumConfig,
    estimate_total_timesteps,
    extract_evaluation_stats,
    make_json_serializable,
    save_training_metadata,
)
from gakumas_rl.backends import (
    TrainingResult,
    TrainingSpec,
    _apply_exam_randomization_flags,
    _apply_exam_randomization_flags_to_rllib_algo,
    _coerce_rllib_checkpoint_path,
    _evaluation_env_config,
    _generate_demo_artifacts,
    _maybe_generate_demo_artifacts,
    _resolve_exam_randomization_flags,
    _resolve_rllib_runtime_config,
    _safe_save_training_metadata,
    _save_rllib_checkpoint,
    run_training,
)


def test_early_stopping_respects_min_training_steps() -> None:
    """达到最少训练步数前，不应累计早停耐心值。"""

    callback = EarlyStoppingCallback(
        patience=2,
        min_delta=0.05,
        min_training_steps=100,
        verbose=False,
        window_size=1,
    )

    assert callback.on_evaluation(20, 1.0, 0.1) is False
    assert callback.on_evaluation(60, 1.0, 0.1) is False
    assert callback.wait == 0

    assert callback.on_evaluation(120, 1.0, 0.1) is False
    assert callback.wait == 1

    assert callback.on_evaluation(180, 1.0, 0.1) is True
    assert callback.should_stop is True
    assert callback.stopped_step == 180


def test_estimate_total_timesteps_uses_exam_reward_mode() -> None:
    """不同考试奖励模式应得到不同的默认训练步数估算。"""

    score_steps = estimate_total_timesteps(
        {'mode': 'exam', 'scenario': 'nia_master', 'exam_reward_mode': 'score'},
        min_timesteps=100_000,
        max_timesteps=5_000_000,
    )
    clear_steps = estimate_total_timesteps(
        {'mode': 'exam', 'scenario': 'nia_master', 'exam_reward_mode': 'clear'},
        min_timesteps=100_000,
        max_timesteps=5_000_000,
    )

    assert score_steps == 780_000
    assert clear_steps == 468_000
    assert score_steps > clear_steps


def test_estimate_total_timesteps_supports_lesson_mode() -> None:
    """独立 lesson 模式应有单独的自动训练步数基线。"""

    lesson_steps = estimate_total_timesteps(
        {'mode': 'lesson', 'scenario': 'nia_master'},
        min_timesteps=100_000,
        max_timesteps=5_000_000,
    )

    assert lesson_steps == 390_000


def test_estimate_total_timesteps_penalizes_exam_randomization() -> None:
    """训练环境越随机，自动训练估算步数应越高。"""

    baseline_steps = estimate_total_timesteps(
        {'mode': 'exam', 'scenario': 'nia_master', 'exam_reward_mode': 'score'},
        min_timesteps=100_000,
        max_timesteps=5_000_000,
    )
    randomized_steps = estimate_total_timesteps(
        {
            'mode': 'exam',
            'scenario': 'nia_master',
            'exam_reward_mode': 'score',
            'exam_randomize_context': True,
            'exam_randomize_stage_type': True,
            'exam_randomize_use_after_item': True,
            'exam_starting_stamina_mode': 'random',
        },
        min_timesteps=100_000,
        max_timesteps=5_000_000,
    )

    assert baseline_steps == 780_000
    assert randomized_steps == 1_135_134
    assert randomized_steps > baseline_steps


def test_evaluation_env_config_keeps_generalization_axes_but_disables_noise() -> None:
    """评估环境应保留通用模型要覆盖的轴，但关闭高噪声抖动。"""

    source = {
        'mode': 'exam',
        'scenario': 'nia_master',
        'exam_randomize_context': True,
        'exam_randomize_stage_type': True,
        'exam_randomize_use_after_item': True,
        'exam_stat_jitter_ratio': 0.2,
        'exam_score_bonus_jitter_ratio': 0.1,
        'exam_starting_stamina_mode': 'random',
    }

    resolved = _evaluation_env_config(source, seed=1000)

    assert resolved['seed'] == 1000
    assert resolved['exam_randomize_context'] is False
    assert resolved['exam_randomize_stage_type'] is True
    assert resolved['exam_randomize_use_after_item'] is True
    assert resolved['exam_stat_jitter_ratio'] == 0.0
    assert resolved['exam_score_bonus_jitter_ratio'] == 0.0
    assert resolved['exam_starting_stamina_mode'] == 'full'
    assert source['exam_randomize_context'] is True
    assert source['exam_starting_stamina_mode'] == 'random'


def test_exam_randomization_curriculum_opens_axes_by_progress() -> None:
    """课程式随机化应按训练进度逐步打开 stage_type / use_after_item。"""

    env_config = {
        'mode': 'exam',
        'scenario': 'nia_master',
        'exam_randomize_stage_type': True,
        'exam_randomize_use_after_item': True,
    }
    curriculum = ExamRandomizationCurriculumConfig(
        enabled=True,
        stage_type_start_ratio=0.30,
        use_after_item_start_ratio=0.60,
    )

    start_flags = _resolve_exam_randomization_flags(env_config, curriculum, current_step=0, total_timesteps=900_000)
    middle_flags = _resolve_exam_randomization_flags(env_config, curriculum, current_step=270_000, total_timesteps=900_000)
    late_flags = _resolve_exam_randomization_flags(env_config, curriculum, current_step=540_000, total_timesteps=900_000)

    assert _apply_exam_randomization_flags(env_config, start_flags)['exam_randomize_stage_type'] is False
    assert _apply_exam_randomization_flags(env_config, start_flags)['exam_randomize_use_after_item'] is False
    assert middle_flags['exam_randomize_stage_type'] is True
    assert middle_flags['exam_randomize_use_after_item'] is False
    assert late_flags['exam_randomize_stage_type'] is True
    assert late_flags['exam_randomize_use_after_item'] is True


def test_dynamic_scheduler_evaluates_more_frequently_near_convergence() -> None:
    """动态调度应在训练后期和收敛阶段更频繁评估。"""

    config = AutoTrainingConfig(
        full_auto=True,
        timesteps_per_eval=50_000,
        min_timesteps=100_000,
        dynamic_eval_schedule=True,
        min_eval_interval=10_000,
        max_eval_interval=100_000,
        target_num_evaluations=12,
    )
    scheduler = DynamicEvaluationScheduler(
        config=config,
        estimated_total_timesteps=600_000,
        rollout_steps=2_048,
    )

    early_interval = scheduler.interval_for(20_000, [])
    stable_history = [
        {'step': 200_000, 'mean_reward': 5.00, 'std_reward': 0.2},
        {'step': 250_000, 'mean_reward': 5.01, 'std_reward': 0.2},
        {'step': 300_000, 'mean_reward': 5.00, 'std_reward': 0.2},
        {'step': 350_000, 'mean_reward': 5.00, 'std_reward': 0.2},
        {'step': 400_000, 'mean_reward': 5.01, 'std_reward': 0.2},
    ]
    late_interval = scheduler.interval_for(520_000, stable_history)

    assert early_interval > late_interval
    assert late_interval >= 10_240


def test_extract_evaluation_stats_supports_rllib_payload() -> None:
    """RLlib 的评估结果应能被规范成统一的 reward 统计。"""

    payload = {
        'evaluation': {
            'env_runners': {
                'episode_return_mean': 6.25,
                'episode_return_std': 0.75,
            },
        },
    }

    mean_reward, std_reward = extract_evaluation_stats(payload)

    assert mean_reward == 6.25
    assert std_reward == 0.75


def test_make_json_serializable_handles_numpy_scalars_and_paths() -> None:
    """训练元数据里的 numpy 标量和 Path 应能递归转成 JSON 原生类型。"""

    payload = {
        'flag': np.bool_(True),
        'value': np.float64(1.25),
        'count': np.int64(7),
        'path': Path('runs/example'),
        'nested': {'array': np.array([1, 2, 3], dtype=np.int64)},
    }

    normalized = make_json_serializable(payload)

    assert normalized == {
        'flag': True,
        'value': 1.25,
        'count': 7,
        'path': str(Path('runs/example')),
        'nested': {'array': [1, 2, 3]},
    }
    json.dumps(normalized, ensure_ascii=False)


def test_save_training_metadata_accepts_numpy_bool_summary() -> None:
    """保存训练元数据时不应因 numpy.bool_ 之类的标量报错。"""

    with TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)
        save_training_metadata(
            run_dir,
            config={
                'backend': 'rllib',
                'evaluation_summary': {
                    'converged': np.bool_(True),
                    'best_reward': np.float64(4.5),
                },
            },
        )

        payload = json.loads((run_dir / 'training_metadata.json').read_text(encoding='utf-8'))
        assert payload['config']['evaluation_summary'] == {
            'converged': True,
            'best_reward': 4.5,
        }


def test_safe_save_training_metadata_does_not_raise_on_failure() -> None:
    """元数据保存失败时，训练后端应记录 warning 而不是直接崩溃。"""

    import gakumas_rl.backends as backends_module

    original = backends_module.save_training_metadata

    def _boom(run_dir: Path, config: dict[str, object], early_stopping_summary: dict[str, object] | None = None) -> None:
        raise TypeError('Object of type bool is not JSON serializable')

    try:
        backends_module.save_training_metadata = _boom
        with TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            metadata_log = run_dir / 'artifacts.jsonl'
            _safe_save_training_metadata(run_dir, metadata_log, {'backend': 'rllib'})
            entries = [json.loads(line) for line in metadata_log.read_text(encoding='utf-8').splitlines() if line.strip()]
            assert entries[-1]['kind'] == 'warning'
            assert 'metadata_save_failed' in entries[-1]['warning']
    finally:
        backends_module.save_training_metadata = original


def test_resolve_rllib_runtime_config_scales_batch_with_parallelism_and_gpu() -> None:
    """RLlib 自动推导的 batch 配置应随并行度和 GPU 提升。"""

    runtime = _resolve_rllib_runtime_config(
        TrainingSpec(
            backend='rllib',
            env_config={'mode': 'exam', 'scenario': 'nia_master'},
            rollout_steps=512,
            rllib_num_workers=4,
            rllib_num_envs_per_worker=2,
            rllib_num_gpus=1.0,
        )
    )

    assert runtime.num_env_runners == 4
    assert runtime.num_envs_per_env_runner == 2
    assert runtime.rollout_fragment_length == 512
    assert runtime.train_batch_size >= 4_096
    assert runtime.train_batch_size % (runtime.rollout_fragment_length * runtime.num_envs_per_env_runner * runtime.num_env_runners) == 0
    assert 1 <= runtime.minibatch_size <= runtime.train_batch_size


def test_resolve_rllib_runtime_config_respects_explicit_overrides() -> None:
    """用户显式传入的 RLlib batch 参数应优先于自动推导。"""

    runtime = _resolve_rllib_runtime_config(
        TrainingSpec(
            backend='rllib',
            env_config={'mode': 'exam', 'scenario': 'nia_master'},
            rollout_steps=512,
            rllib_num_workers=2,
            rllib_train_batch_size=3_072,
            rllib_minibatch_size=768,
            rllib_num_epochs=5,
            rllib_rollout_fragment_length=256,
        )
    )

    assert runtime.rollout_fragment_length == 256
    assert runtime.train_batch_size == 3_072
    assert runtime.minibatch_size == 768
    assert runtime.num_epochs == 5


def test_apply_exam_randomization_flags_to_rllib_algo_updates_all_worker_envs() -> None:
    """RLlib curriculum 应能把随机化开关广播到所有训练 worker 的环境。"""

    class FakeEnv:
        def __init__(self) -> None:
            self.calls: list[tuple[bool, bool]] = []

        def update_episode_randomization(
            self,
            *,
            randomize_stage_type: bool | None = None,
            randomize_use_after_item: bool | None = None,
        ) -> None:
            self.calls.append((bool(randomize_stage_type), bool(randomize_use_after_item)))

    class WrappedEnv:
        def __init__(self, env: FakeEnv) -> None:
            self.unwrapped = env

    class FakeWorker:
        def __init__(self, envs: list[WrappedEnv]) -> None:
            self._envs = envs

        def foreach_env(self, fn):
            return [fn(env) for env in self._envs]

    class FakeWorkerGroup:
        def __init__(self, workers: list[FakeWorker]) -> None:
            self._workers = workers

        def foreach_worker(self, fn):
            return [fn(worker) for worker in self._workers]

    class FakeAlgo:
        def __init__(self, workers: FakeWorkerGroup) -> None:
            self.workers = workers

    envs = [FakeEnv(), FakeEnv(), FakeEnv()]
    algo = FakeAlgo(
        FakeWorkerGroup(
            [
                FakeWorker([WrappedEnv(envs[0]), WrappedEnv(envs[1])]),
                FakeWorker([WrappedEnv(envs[2])]),
            ]
        )
    )

    updated = _apply_exam_randomization_flags_to_rllib_algo(
        algo,
        {
            'exam_randomize_stage_type': True,
            'exam_randomize_use_after_item': False,
        },
    )

    assert updated == 3
    assert envs[0].calls == [(True, False)]
    assert envs[1].calls == [(True, False)]
    assert envs[2].calls == [(True, False)]


def test_apply_exam_randomization_flags_to_rllib_algo_updates_all_env_runners() -> None:
    """新 API stack 下也应能把 curriculum 广播到 env_runner_group。"""

    class FakeEnv:
        def __init__(self) -> None:
            self.calls: list[tuple[bool, bool]] = []

        def update_episode_randomization(
            self,
            *,
            randomize_stage_type: bool | None = None,
            randomize_use_after_item: bool | None = None,
        ) -> None:
            self.calls.append((bool(randomize_stage_type), bool(randomize_use_after_item)))

    class FakeEnvRunner:
        def __init__(self, envs: list[FakeEnv]) -> None:
            self.envs = envs

    class FakeEnvRunnerGroup:
        def __init__(self, env_runners: list[FakeEnvRunner]) -> None:
            self._env_runners = env_runners

        def foreach_env_runner(self, fn, **kwargs):
            return [fn(env_runner) for env_runner in self._env_runners]

    class FakeAlgo:
        def __init__(self, env_runner_group: FakeEnvRunnerGroup) -> None:
            self.env_runner_group = env_runner_group

    envs = [FakeEnv(), FakeEnv()]
    algo = FakeAlgo(FakeEnvRunnerGroup([FakeEnvRunner([envs[0]]), FakeEnvRunner([envs[1]])]))

    updated = _apply_exam_randomization_flags_to_rllib_algo(
        algo,
        {
            'exam_randomize_stage_type': False,
            'exam_randomize_use_after_item': True,
        },
    )

    assert updated == 2
    assert envs[0].calls == [(False, True)]
    assert envs[1].calls == [(False, True)]


def test_run_training_allows_rllib_curriculum_dispatch() -> None:
    """RLlib backend 现已允许直接走课程式考试随机化主线。"""

    import gakumas_rl.backends as backends_module

    original = backends_module.run_rllib_training
    result = TrainingResult(
        backend='rllib',
        run_dir=Path('runs/fake_rllib'),
        latest_checkpoint=None,
        total_timesteps=0,
        evaluation_log=None,
        metadata_log=None,
    )
    captured: list[TrainingSpec] = []

    def fake_run_rllib_training(spec: TrainingSpec) -> TrainingResult:
        captured.append(spec)
        return result

    try:
        backends_module.run_rllib_training = fake_run_rllib_training
        actual = run_training(
            TrainingSpec(
                backend='rllib',
                env_config={'mode': 'exam', 'scenario': 'nia_master'},
                exam_randomization_curriculum=ExamRandomizationCurriculumConfig(
                    enabled=True,
                    stage_type_start_ratio=0.10,
                    use_after_item_start_ratio=0.25,
                ),
            )
        )
    finally:
        backends_module.run_rllib_training = original

    assert actual is result
    assert captured and captured[0].backend == 'rllib'


def test_coerce_rllib_checkpoint_path_supports_training_result_like_object() -> None:
    """兼容新版 Ray 的 TrainingResult(checkpoint.path=...) 返回值。"""

    class FakeCheckpoint:
        path = 'runs/rllib_exam/checkpoints/checkpoint_00065536'

    class FakeTrainingResult:
        checkpoint = FakeCheckpoint()

    checkpoint_path = _coerce_rllib_checkpoint_path(FakeTrainingResult())

    assert checkpoint_path == Path('runs/rllib_exam/checkpoints/checkpoint_00065536')


def test_coerce_rllib_checkpoint_path_supports_file_uri() -> None:
    """save_to_path 可能返回 file:// URI，应转换回本地路径。"""

    with TemporaryDirectory() as tmpdir:
        checkpoint_path = (Path(tmpdir) / 'checkpoints' / 'checkpoint_00002048').resolve()
        actual_path = _coerce_rllib_checkpoint_path(checkpoint_path.as_uri())

        assert actual_path == checkpoint_path


def test_save_rllib_checkpoint_uses_save_to_path_when_available() -> None:
    """支持 save_to_path 的 RLlib 版本应写到显式 checkpoint_<step> 目录。"""

    class FakeAlgo:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def save_to_path(self, path: str) -> str:
            self.calls.append(path)
            return path

    with TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir) / 'checkpoints'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        algo = FakeAlgo()

        checkpoint_path = _save_rllib_checkpoint(algo, checkpoint_dir, 2048)

        expected_path = (checkpoint_dir / 'checkpoint_00002048').resolve()
        assert checkpoint_path == expected_path
        assert algo.calls == [expected_path.as_uri()]


def test_save_rllib_checkpoint_falls_back_to_save_for_old_api_stack() -> None:
    """old API stack 的 save_to_path 不可用时，应自动回退到 algo.save()."""

    class FakeCheckpoint:
        def __init__(self, path: str) -> None:
            self.path = path

    class FakeTrainingResult:
        def __init__(self, path: str) -> None:
            self.checkpoint = FakeCheckpoint(path)

    class FakeAlgo:
        def __init__(self) -> None:
            self.save_to_path_calls: list[str] = []
            self.save_calls: list[str] = []

        def save_to_path(self, path: str) -> str:
            self.save_to_path_calls.append(path)
            raise RuntimeError('Algorithm.get_state() not supported on the old API stack! Use Algorithm.__getstate__() instead.')

        def save(self, checkpoint_dir: str) -> FakeTrainingResult:
            self.save_calls.append(checkpoint_dir)
            return FakeTrainingResult(checkpoint_dir)

    with TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir) / 'checkpoints'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        algo = FakeAlgo()

        checkpoint_path = _save_rllib_checkpoint(algo, checkpoint_dir, 4096)

        expected_path = (checkpoint_dir / 'checkpoint_00004096').resolve()
        assert checkpoint_path == expected_path
        assert algo.save_to_path_calls == [expected_path.as_uri()]
        assert algo.save_calls == [str(expected_path)]


def test_generate_demo_artifacts_for_exam_training() -> None:
    """考试训练结束后应尝试生成 demo 回放文件并记录到 metadata。"""

    fake_module = ModuleType('gakumas_rl.demo_exam')

    def fake_run_demo(args):
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text('<html></html>', encoding='utf-8')
        json_path = output_path.with_suffix('.json')
        json_path.write_text(json.dumps({'summary': {'scenario': args.scenario}}), encoding='utf-8')
        return {'artifacts': {'html': str(output_path), 'json': str(json_path)}}

    fake_module.run_demo = fake_run_demo
    original_module = sys.modules.get('gakumas_rl.demo_exam')
    sys.modules['gakumas_rl.demo_exam'] = fake_module
    try:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint_path = root / 'checkpoints' / 'step_123.zip'
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path.write_text('checkpoint', encoding='utf-8')
            metadata_log = root / 'artifacts.jsonl'
            html_path, json_path = _maybe_generate_demo_artifacts(
                TrainingSpec(
                    backend='sb3',
                    env_config={'mode': 'exam', 'scenario': 'nia_master', 'exam_reward_mode': 'score'},
                    seed=42,
                ),
                checkpoint_path,
                root,
                metadata_log,
            )
            assert html_path is not None and html_path.exists()
            assert json_path is not None and json_path.exists()
            payload = json.loads(metadata_log.read_text(encoding='utf-8').strip())
            assert payload['kind'] == 'replay'
            assert payload['step'] == 123
    finally:
        if original_module is None:
            sys.modules.pop('gakumas_rl.demo_exam', None)
        else:
            sys.modules['gakumas_rl.demo_exam'] = original_module


def test_generate_periodic_test_report_uses_explicit_step_and_kind() -> None:
    """周期性测试报告应按指定 step 和 kind 输出。"""

    fake_module = ModuleType('gakumas_rl.demo_exam')

    def fake_run_demo(args):
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text('<html></html>', encoding='utf-8')
        json_path = output_path.with_suffix('.json')
        json_path.write_text(json.dumps({'summary': {'scenario': args.scenario}}), encoding='utf-8')
        return {'artifacts': {'html': str(output_path), 'json': str(json_path)}}

    fake_module.run_demo = fake_run_demo
    original_module = sys.modules.get('gakumas_rl.demo_exam')
    sys.modules['gakumas_rl.demo_exam'] = fake_module
    try:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint_path = root / 'checkpoints' / 'step_456.zip'
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path.write_text('checkpoint', encoding='utf-8')
            metadata_log = root / 'artifacts.jsonl'
            html_path, json_path = _generate_demo_artifacts(
                TrainingSpec(
                    backend='rllib',
                    env_config={'mode': 'exam', 'scenario': 'nia_master', 'exam_reward_mode': 'score'},
                    seed=7,
                ),
                checkpoint_path,
                root,
                metadata_log,
                step=2048,
                artifact_kind='test_report',
            )
            assert html_path is not None and html_path.name == 'test_report_rllib_nia_master_2048.html'
            assert json_path is not None and json_path.name == 'test_report_rllib_nia_master_2048.json'
            payload = json.loads(metadata_log.read_text(encoding='utf-8').strip())
            assert payload['kind'] == 'test_report'
            assert payload['step'] == 2048
    finally:
        if original_module is None:
            sys.modules.pop('gakumas_rl.demo_exam', None)
        else:
            sys.modules['gakumas_rl.demo_exam'] = original_module
