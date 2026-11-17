"""Tests for episode outcome evidence and outcome-aware mask calibration.

Everything here is synthetic: scripted Gymnasium environments for the
rollout evidence in `eval_utils`, and hand-built evidence dictionaries for the
pure calibration functions in `mask_calibration`. No expert is trained.
"""

import copy
import dataclasses
import json
import math
import signal
from typing import Dict, List, Optional, Sequence

import gymnasium as gym
import numpy as np
import pytest
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from imitation.experiments.agnostic import mask_calibration as mc
from imitation.experiments.agnostic import pilot, random_grid, restriction, tasks
from imitation.experiments.ftrl import env_utils, eval_utils

# ---------------------------------------------------------------------------
# Scripted environments and policies for rollout evidence
# ---------------------------------------------------------------------------


class _ScriptedEnv(gym.Env):
    """One-dimensional env whose start state comes from its seeded RNG.

    The episode ends on its own after ``length`` steps when ``terminates`` is
    True; otherwise it only ends through an outer TimeLimit. Every step pays
    -1 except the terminating step, which pays ``terminal_reward``.
    """

    metadata: Dict = {"render_modes": []}

    def __init__(
        self,
        length: int = 3,
        terminates: bool = True,
        terminal_reward: float = 1.0,
        info_success: Optional[bool] = None,
    ):
        self.observation_space = gym.spaces.Box(0.0, 1e6, (1,), np.float32)
        self.action_space = gym.spaces.Discrete(2)
        self.length = length
        self.terminates = terminates
        self.terminal_reward = terminal_reward
        self.info_success = info_success
        self._t = 0
        self._start = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._t = 0
        self._start = int(self.np_random.integers(0, 1_000_000))
        return np.array([self._start], dtype=np.float32), {}

    def step(self, action):
        self._t += 1
        done = self.terminates and self._t >= self.length
        reward = self.terminal_reward if done else -1.0
        info = {}
        if done and self.info_success is not None:
            info["is_success"] = self.info_success
        obs = np.array([self._start + self._t], dtype=np.float32)
        return obs, reward, done, False, info


class _ConstPolicy:
    def __init__(self, action: int = 0):
        self.action = action

    def predict(self, obs, deterministic=True):
        return np.full(len(obs), self.action, dtype=np.int64), None


def _venv(raw_flags: bool = False, time_limit: Optional[int] = None, **kwargs):
    def make():
        env: gym.Env = _ScriptedEnv(**kwargs)
        if time_limit is not None:
            env = gym.wrappers.TimeLimit(env, max_episode_steps=time_limit)
        if raw_flags:
            env = eval_utils.RawTerminationInfoWrapper(env)
        return Monitor(env)

    return DummyVecEnv([make])


def _direct_digest(seed: int) -> str:
    obs, _ = _ScriptedEnv().reset(seed=seed)
    return eval_utils.observation_digest(obs)


# ---------------------------------------------------------------------------
# EvalResult backward compatibility
# ---------------------------------------------------------------------------


def test_eval_result_old_positional_construction_leaves_new_fields_none():
    res = eval_utils.EvalResult(1.0, [1.0], 3, None, None, None)
    assert res.episode_lengths is None
    assert res.episode_terminated is None
    assert res.episode_truncated is None
    assert res.episode_successes is None
    assert res.evaluation_episode_seeds is None
    assert res.initial_state_digests is None
    assert res.success_rule is None
    assert res.termination_flag_source is None
    assert res.reset_accounting is None


def test_default_rollout_keeps_old_behavior_and_adds_evidence():
    res = eval_utils.eval_policy_rollout(
        _ConstPolicy(), _venv(length=3), n_episodes=4, env_id="unknown-v0"
    )
    assert res.episode_returns == [-1.0] * 4  # -1 -1 +1
    assert res.n_steps == 12
    assert res.evaluation_episode_seeds is None
    assert res.episode_lengths == [3, 3, 3, 3]
    assert res.episode_terminated == [True] * 4
    assert res.episode_truncated == [False] * 4
    assert res.episode_successes is None  # unknown env and no is_success
    assert res.success_rule is None
    assert len(res.initial_state_digests) == 4
    assert res.reset_accounting == {
        "explicit_resets": 1,
        "auto_resets": 4,
        "discarded_auto_resets": 1,
    }


def test_returns_are_not_clipped():
    res = eval_utils.eval_policy_rollout(
        _ConstPolicy(),
        _venv(length=2, terminal_reward=-5000.0),
        n_episodes=2,
        env_id="unknown-v0",
    )
    assert res.episode_returns == [-5001.0, -5001.0]


# ---------------------------------------------------------------------------
# Paired per-episode reset seeds
# ---------------------------------------------------------------------------


def test_episode_seeds_give_identical_initial_states_across_policies():
    seeds = [11, 12, 13, 14]
    a = eval_utils.eval_policy_rollout(
        _ConstPolicy(0), _venv(length=2), 4, episode_seeds=seeds, env_id="x-v0"
    )
    venv_b = _venv(length=5)
    # Advance the hidden RNG so an auto-reset sequence would diverge.
    venv_b.reset()
    for _ in range(7):
        venv_b.step(np.zeros(1, dtype=np.int64))
    b = eval_utils.eval_policy_rollout(
        _ConstPolicy(1), venv_b, 4, episode_seeds=seeds, env_id="x-v0"
    )
    assert a.evaluation_episode_seeds == seeds
    assert b.evaluation_episode_seeds == seeds
    assert a.initial_state_digests == b.initial_state_digests
    assert a.initial_state_digests == [_direct_digest(s) for s in seeds]
    # Episode lengths differ, so pairing is by seed, not by RNG coincidence.
    assert a.episode_lengths == [2] * 4 and b.episode_lengths == [5] * 4


def test_episode_seeds_reset_accounting_counts_discarded_auto_resets():
    res = eval_utils.eval_policy_rollout(
        _ConstPolicy(), _venv(length=2), 3, episode_seeds=[1, 2, 3], env_id="x-v0"
    )
    assert res.n_steps == 6  # resets are never counted as steps
    assert res.reset_accounting == {
        "explicit_resets": 3,
        "auto_resets": 3,
        "discarded_auto_resets": 3,
    }


@pytest.mark.parametrize(
    "seeds",
    [[1, 2], [1, 2, 3, 4], [1, 1, 2], [1, -2, 3], [1, True, 3], [1, 2.0, 3]],
)
def test_episode_seeds_must_be_exact_distinct_nonnegative_ints(seeds):
    with pytest.raises(ValueError, match="episode_seeds"):
        eval_utils.eval_policy_rollout(
            _ConstPolicy(), _venv(), 3, episode_seeds=seeds, env_id="x-v0"
        )


def test_episode_seeds_on_real_cartpole_pair_initial_states():
    seeds = [100, 101, 102]
    results = []
    for action in (0, 1):
        venv = env_utils.make_env("CartPole-v1", n_envs=1, rng=np.random.default_rng(0))
        results.append(
            eval_utils.eval_policy_rollout(
                _ConstPolicy(action), venv, 3, episode_seeds=seeds
            )
        )
    a, b = results
    assert a.initial_state_digests == b.initial_state_digests
    assert a.success_rule == "survived_to_time_limit"  # env id from the spec
    assert a.episode_successes == [False] * 3  # constant pushes fall over
    assert a.episode_terminated == [True] * 3
    assert all(length < 500 for length in a.episode_lengths)


# ---------------------------------------------------------------------------
# Termination and truncation evidence
# ---------------------------------------------------------------------------


def test_timeout_is_inferred_truncated_from_sb3_info():
    res = eval_utils.eval_policy_rollout(
        _ConstPolicy(),
        _venv(terminates=False, time_limit=4),
        2,
        env_id="x-v0",
    )
    assert res.episode_truncated == [True, True]
    assert res.episode_terminated == [False, False]
    assert res.episode_lengths == [4, 4]
    assert res.termination_flag_source == eval_utils.FLAGS_SB3_EXCLUSIVE


def test_termination_at_cap_is_terminated_under_sb3_exclusive_flags():
    res = eval_utils.eval_policy_rollout(
        _ConstPolicy(), _venv(length=4, time_limit=4), 2, env_id="x-v0"
    )
    assert res.episode_terminated == [True, True]
    assert res.episode_truncated == [False, False]


def test_raw_flags_wrapper_reports_both_flags():
    res = eval_utils.eval_policy_rollout(
        _ConstPolicy(),
        _venv(raw_flags=True, length=4, time_limit=4),
        2,
        env_id="x-v0",
    )
    assert res.episode_terminated == [True, True]
    assert res.episode_truncated == [True, True]
    assert res.termination_flag_source == eval_utils.FLAGS_RAW


def test_cliff_goal_on_last_step_is_success_not_timeout_failure():
    # 200 steps, the goal is reached on the 200th: return near -200 but
    # terminated is True, so it is a success.
    res = eval_utils.eval_policy_rollout(
        _ConstPolicy(),
        _venv(length=200, time_limit=200, terminal_reward=-1.0),
        1,
        env_id="CliffWalking-v0",
    )
    assert res.episode_returns == [-200.0]
    assert res.episode_successes == [True]
    stall = eval_utils.eval_policy_rollout(
        _ConstPolicy(),
        _venv(terminates=False, time_limit=200),
        1,
        env_id="CliffWalking-v0",
    )
    assert stall.episode_returns == [-200.0]
    assert stall.episode_successes == [False]


def test_info_is_success_takes_precedence_over_env_rule():
    res = eval_utils.eval_policy_rollout(
        _ConstPolicy(),
        _venv(length=2, terminal_reward=5.0, info_success=False),
        2,
        env_id="FrozenLake-v1",
    )
    assert res.episode_successes == [False, False]
    assert res.success_rule == eval_utils.IS_SUCCESS_RULE.name


def test_episode_evidence_is_json_serializable():
    res = eval_utils.eval_policy_rollout(
        _ConstPolicy(), _venv(length=2), 2, episode_seeds=[3, 4], env_id="Taxi-v3"
    )
    evidence = eval_utils.episode_evidence(res)
    assert json.loads(json.dumps(evidence)) == evidence
    assert evidence["evaluation_episode_seeds"] == [3, 4]
    assert evidence["episode_successes"] == [True, True]
    assert evidence["success_rule"] == "goal_terminated"


# ---------------------------------------------------------------------------
# Success rules (pure)
# ---------------------------------------------------------------------------


def _success(env_id, terminated=True, truncated=False, reward=0.0, ret=0.0, info=None):
    return eval_utils.episode_success(
        env_id,
        terminated=terminated,
        truncated=truncated,
        terminal_reward=reward,
        episode_return=ret,
        terminal_info=info,
    )


@pytest.mark.parametrize(
    "env_id, kwargs, expected",
    [
        ("FrozenLake-v1", dict(reward=1.0, ret=1.0), True),
        ("FrozenLake-v1", dict(reward=0.0), False),
        ("FrozenLake-v1", dict(terminated=False, truncated=True), False),
        ("FrozenLakeVariedStart-v0", dict(reward=1.0, ret=1.0), True),
        ("agnostic/FrozenLakeVariedStart-v0", dict(reward=0.0), False),
        ("FrozenLakeSlippery-v0", dict(reward=1.0, ret=1.0), True),
        ("FrozenLakeSlippery-v0", dict(reward=0.0), False),
        ("FrozenLakeSlippery-v0", dict(terminated=False, truncated=True), False),
        ("CliffWalking-v0", dict(reward=-1.0, ret=-200.0), True),
        ("CliffWalking-v0", dict(terminated=True, truncated=True, ret=-200.0), True),
        ("CliffWalking-v0", dict(terminated=False, truncated=True, ret=-200.0), False),
        (
            "agnostic/CliffWalking-v0",
            dict(terminated=False, truncated=True, ret=-200.0),
            False,
        ),
        # The withdrawn varied-start Cliff variant has no rule (unknown).
        ("CliffWalkingVariedStart-v0", dict(reward=-1.0, ret=-200.0), None),
        ("Taxi-v3", dict(reward=20.0), True),
        ("Taxi-v3", dict(terminated=False, truncated=True), False),
        ("Acrobot-v1", dict(), True),
        ("MountainCar-v0", dict(terminated=False, truncated=True, ret=-200.0), False),
        ("CartPole-v1", dict(terminated=False, truncated=True, ret=500.0), True),
        ("CartPole-v1", dict(terminated=True, truncated=False, ret=37.0), False),
        ("LunarLander-v2", dict(reward=100.0, ret=230.0), True),
        ("LunarLander-v2", dict(reward=-100.0, ret=-150.0), False),
        ("LunarLander-v2", dict(terminated=False, truncated=True), False),
        ("Blackjack-v1", dict(reward=1.0, ret=1.0), True),
        ("Blackjack-v1", dict(reward=0.0, ret=0.0), False),
        ("Unknown-v9", dict(reward=1.0, ret=1.0), None),
        ("Unknown-v9", dict(info={"is_success": True}), True),
        ("FrozenLake-v1", dict(reward=1.0, info={"is_success": False}), False),
    ],
)
def test_episode_success_rules(env_id, kwargs, expected):
    assert _success(env_id, **kwargs) is expected


def test_success_rule_metadata():
    assert eval_utils.success_rule_for("Unknown-v9") is None
    blackjack = eval_utils.success_rule_for("Blackjack-v1")
    assert blackjack.name == "positive_return"
    assert blackjack.used_for_calibration is False
    for env_id in (
        "CartPole-v1",
        "FrozenLake-v1",
        "FrozenLakeVariedStart-v0",
        "CliffWalking-v0",
        "Taxi-v3",
        "Acrobot-v1",
        "MountainCar-v0",
        "LunarLander-v2",
    ):
        rule = eval_utils.success_rule_for(env_id)
        assert rule.used_for_calibration is True
        assert rule.description


def test_frozenlake_slippery_success_rule_is_registered_like_frozenlake():
    assert "FrozenLakeSlippery-v0" in eval_utils.SUCCESS_RULES
    slippery = eval_utils.success_rule_for("FrozenLakeSlippery-v0")
    assert slippery is eval_utils.success_rule_for("FrozenLake-v1")
    assert slippery.name == "terminal_reward_positive"
    assert mc.success_used("FrozenLakeSlippery-v0") is True
    assert mc.goal_objective("FrozenLakeSlippery-v0") is True
    assert mc.criterion_for("FrozenLakeSlippery-v0") == mc.NORMALIZED_RETURN


def test_frozenlake_slippery_rollout_records_terminal_reward_success():
    venv = env_utils.make_env(
        env_utils.FROZENLAKE_SLIPPERY, n_envs=1, rng=np.random.default_rng(0)
    )
    res = eval_utils.eval_policy_rollout(
        _ConstPolicy(2), venv, 20, episode_seeds=list(range(20))
    )
    assert res.success_rule == "terminal_reward_positive"
    assert res.episode_successes == [r > 0 for r in res.episode_returns]


# ---------------------------------------------------------------------------
# Calibration evidence builders
# ---------------------------------------------------------------------------

SEEDS = (0, 1, 2)
HASH = "a" * 64
EXPERT_HASH = "b" * 64


def _row(
    method: str,
    arm: str,
    seed: int,
    returns: Sequence[float],
    successes: Optional[Sequence[bool]] = None,
    truncated: Optional[Sequence[bool]] = None,
    eval_seeds: Optional[Sequence[int]] = None,
) -> Dict:
    n = len(returns)
    truncated = list(truncated) if truncated is not None else [False] * n
    row = {
        "method": method,
        "arm": arm,
        "seed": seed,
        "final": True,
        # The production row flag (`run_experiment._compute_round_eval`).
        "evaluation_deterministic": True,
        "episode_returns": [float(r) for r in returns],
        "episode_lengths": [10] * n,
        "episode_terminated": [not t for t in truncated],
        "episode_truncated": truncated,
        "episode_successes": None if successes is None else list(successes),
        "evaluation_episode_seeds": (
            list(eval_seeds) if eval_seeds is not None else list(range(1000, 1000 + n))
        ),
    }
    if arm == "masked":
        row["mask_sha256"] = HASH
    return row


def _evidence(
    env_name: str,
    rows: List[Dict],
    expert: float,
    random: float,
    pairing: str = "paired",
    seeds: Sequence[int] = SEEDS,
) -> Dict:
    return {
        "env_name": env_name,
        "mask": "test_mask",
        "mask_sha256": HASH,
        "witness_verified": True,
        "deterministic_evaluation": True,
        "selection_seeds": list(seeds),
        "pairing": pairing,
        "full_control": {"verified": True},
        "reference": {
            "expert_return": expert,
            "random_return": random,
            "uncertainty": None,
        },
        "rows": rows,
    }


def _binary(n_success: int, n: int = 100) -> List[float]:
    return [1.0] * n_success + [0.0] * (n - n_success)


def _frozen_varied(masked_success: Dict[str, int], full_success: int = 90) -> Dict:
    rows = []
    for seed in SEEDS:
        for method in mc.REQUIRED_METHODS:
            full = _binary(full_success)
            masked = _binary(masked_success[method])
            rows.append(_row(method, "full", seed, full, [r > 0 for r in full]))
            rows.append(_row(method, "masked", seed, masked, [r > 0 for r in masked]))
    return _evidence("FrozenLakeVariedStart-v0", rows, expert=1.0, random=0.0)


ACCEPTABLE = {"ftl": 60, "bc": 40, "bc_iid": 50, "bc_prefix": 45}


def _blackjack(full: Sequence[float], masked: Sequence[float], **kw) -> Dict:
    seeds = kw.pop("seeds", SEEDS)
    rows = []
    for seed in seeds:
        for method in mc.REQUIRED_METHODS:
            rows.append(_row(method, "full", seed, full))
            rows.append(_row(method, "masked", seed, masked))
    return _evidence("Blackjack-v1", rows, seeds=seeds, **kw)


# ---------------------------------------------------------------------------
# Protocol constants and the effect gate
# ---------------------------------------------------------------------------


def test_frozen_protocol_constants():
    assert mc.PROTOCOL == "agnostic-mask-calibration/6"
    assert mc.SEEDED_RANDOM_BINNING == "seeded_random"
    assert mc.REJECT_REASONS == {
        mc.INSUFFICIENT_EFFECT,
        mc.FLOOR_COLLAPSE,
        mc.MASKED_NEAR_REFERENCE,
        mc.NO_BETTER_THAN_RANDOM,
        mc.SCIENTIFICALLY_INELIGIBLE,
    }
    assert mc.GOAL_SUCCESS_ENVS == {"CliffWalking-v0"}
    assert mc.REFERENCE_TYPES == ("random_policy",)
    assert not hasattr(mc, "STALLED_REFERENCE_ENVS")
    assert mc.ELIGIBILITY_REQUIRED_ENVS == {"CliffWalking-v0"}
    assert mc.MIN_NORM_GAP == 0.10
    assert mc.SIGNIFICANCE_SE == 2.0
    assert mc.NEAR_OPTIMAL == 0.95
    assert mc.MIN_EPISODES == 100
    assert mc.REQUIRED_METHODS == ("ftl", "bc", "bc_iid", "bc_prefix")


def test_effect_gate_equality_at_min_gap_is_inclusive():
    assert mc.effect_gate(0.10, raw_gap=0.5, raw_se=0.1) is None


def test_effect_gate_rejects_original_blackjack_gap():
    assert mc.effect_gate(0.099068, raw_gap=0.5, raw_se=0.01) == mc.INSUFFICIENT_EFFECT


def test_effect_gate_two_se_is_strict():
    assert mc.effect_gate(0.2, raw_gap=0.2, raw_se=0.1) == mc.INCONCLUSIVE_NOISE
    assert mc.effect_gate(0.2, raw_gap=0.2000001, raw_se=0.1) is None


def test_effect_gate_insufficient_takes_precedence_over_noise():
    assert mc.effect_gate(0.05, raw_gap=0.05, raw_se=1.0) == mc.INSUFFICIENT_EFFECT


# ---------------------------------------------------------------------------
# calibrate_mask decisions
# ---------------------------------------------------------------------------


def test_frozen_varied_start_mixed_rewards_accepted_for_all_methods():
    report = mc.calibrate_mask(_frozen_varied(ACCEPTABLE))
    assert report["decision"] == mc.ACCEPTED, report["reasons"]
    assert report["reasons"] == []
    assert set(report["per_method"]) == set(mc.REQUIRED_METHODS)
    assert report["passing_methods"] == list(mc.REQUIRED_METHODS)
    # Most robust is the highest-return masked method, not BC.
    assert report["most_robust_method"] == "ftl"
    assert report["scientific_eligibility"]["status"] == mc.ELIGIBILITY_UNKNOWN
    assert report["benign_agnostic"] is False
    ftl = report["per_method"]["ftl"]
    assert ftl["effect"] == pytest.approx(0.30)
    assert ftl["normalized_gap"] == pytest.approx(0.30)
    assert ftl["raw_gap"] > mc.SIGNIFICANCE_SE * ftl["raw_se"]
    assert ftl["masked"]["success_rate"] == pytest.approx(0.60)
    assert report["pairing"] == "paired"
    assert "optimal" in report["normalization_note"]


def test_robust_ftl_does_not_disqualify_an_informative_mask():
    report = mc.calibrate_mask(_frozen_varied({**ACCEPTABLE, "ftl": 85}))
    assert report["decision"] == mc.ACCEPTED, report["messages"]
    assert report["reasons"] == []
    assert report["most_robust_method"] == "ftl"
    assert report["per_method"]["bc"]["gate"] is None
    assert report["per_method"]["ftl"]["gate"] == mc.INSUFFICIENT_EFFECT
    assert report["passing_methods"] == ["bc", "bc_iid", "bc_prefix"]
    assert "insufficient_effect:ftl" in report["flags"]


def test_one_significant_method_with_others_null_is_accepted_with_flags():
    report = mc.calibrate_mask(
        _frozen_varied({"ftl": 90, "bc": 40, "bc_iid": 88, "bc_prefix": 89})
    )
    assert report["decision"] == mc.ACCEPTED, report["messages"]
    assert report["passing_methods"] == ["bc"]
    for method in ("ftl", "bc_iid", "bc_prefix"):
        assert f"insufficient_effect:{method}" in report["flags"]


def test_no_method_passing_with_all_insufficient_is_rejected():
    report = mc.calibrate_mask(
        _frozen_varied({m: 85 for m in mc.REQUIRED_METHODS}),
    )
    assert report["decision"] == mc.REJECTED
    assert report["reasons"] == [mc.INSUFFICIENT_EFFECT]
    assert report["passing_methods"] == []


def test_no_method_passing_with_some_noisy_is_inconclusive():
    rng = np.random.default_rng(7)
    full = [1.0] * 62 + [-1.0] * 38
    noisy = [1.0] * 50 + [-1.0] * 50  # gap 0.12, not significant unpaired
    rows = []
    for method in mc.REQUIRED_METHODS:
        masked = noisy if method == "bc" else [r - 0.05 for r in full]
        rows.append(_row(method, "full", 0, full))
        rows.append(_row(method, "masked", 0, list(rng.permutation(masked))))
    ev = _evidence(
        "Blackjack-v1", rows, expert=1.0, random=-1.0, pairing="unpaired", seeds=(0,)
    )
    report = mc.calibrate_mask(ev)
    assert report["decision"] == mc.INCONCLUSIVE
    assert report["reasons"] == [mc.INCONCLUSIVE_NOISE]
    assert "inconclusive_noise:bc" in report["flags"]
    assert "insufficient_effect:ftl" in report["flags"]


def test_blackjack_original_gap_is_rejected():
    rng = np.random.default_rng(0)
    full = list(rng.normal(-0.1, 0.5, 100))
    masked = [r - 0.099068 for r in full]  # expert-random range is 1.0
    report = mc.calibrate_mask(_blackjack(full, masked, expert=0.0, random=-1.0))
    assert report["decision"] == mc.REJECTED
    assert report["reasons"] == [mc.INSUFFICIENT_EFFECT]
    assert report["per_method"]["bc"]["normalized_gap"] == pytest.approx(0.099068)


def test_blackjack_needs_no_success_evidence_and_accepts_clear_gap():
    rng = np.random.default_rng(1)
    full = list(rng.normal(0.0, 0.5, 100))
    masked = [r - 0.3 for r in full]
    report = mc.calibrate_mask(_blackjack(full, masked, expert=0.0, random=-1.0))
    assert report["decision"] == mc.ACCEPTED, report["reasons"]


def test_blackjack_constant_masked_returns_is_floor():
    rng = np.random.default_rng(2)
    full = list(rng.normal(0.0, 0.5, 100))
    report = mc.calibrate_mask(_blackjack(full, [-1.0] * 100, expert=0.0, random=-1.0))
    assert report["decision"] == mc.REJECTED
    assert mc.FLOOR_COLLAPSE in report["reasons"]


def test_noise_is_inconclusive_not_rejected():
    full = [1.0] * 62 + [-1.0] * 38
    masked = [1.0] * 50 + [-1.0] * 50
    report = mc.calibrate_mask(
        _blackjack(
            full, masked, expert=1.0, random=-1.0, pairing="unpaired", seeds=(0,)
        )
    )
    bc = report["per_method"]["bc"]
    assert bc["normalized_gap"] == pytest.approx(0.12)
    assert report["decision"] == mc.INCONCLUSIVE
    assert report["reasons"] == [mc.INCONCLUSIVE_NOISE]
    assert report["pairing"] == "unpaired"


def test_all_masked_near_reference_is_rejected():
    rng = np.random.default_rng(3)
    full = list(rng.normal(0.0, 0.01, 100))
    masked = [r - 0.04 for r in full]  # normalized about 0.96
    report = mc.calibrate_mask(_blackjack(full, masked, expert=0.0, random=-1.0))
    assert report["decision"] == mc.REJECTED
    assert mc.MASKED_NEAR_REFERENCE in report["reasons"]
    assert mc.INSUFFICIENT_EFFECT in report["reasons"]
    assert "reference_uncertainty_unavailable" in report["flags"]
    # No eligibility certificate, so not marked benign.
    assert report["benign_agnostic"] is False
    # Every task except CliffWalking judges near-reference on normalized return.
    assert report["near_reference_criterion"] == mc.NORMALIZED_RETURN
    near = report["near_reference"]
    assert near["criterion"] == mc.NORMALIZED_RETURN
    assert near["all_near"] is True
    assert near["masked"]["bc"] == pytest.approx(0.96, abs=0.01)
    assert mc.REFERENCE_COMPRESSION not in report["flags"]


def test_all_near_reference_rejects_even_when_one_method_passes():
    rng = np.random.default_rng(8)
    full = list(rng.normal(0.2, 0.01, 100))  # normalized about 1.2
    rows = []
    for seed in SEEDS:
        for method in mc.REQUIRED_METHODS:
            drop = 0.15 if method == "bc" else 0.01  # bc ends near 1.05
            rows.append(_row(method, "full", seed, full))
            rows.append(_row(method, "masked", seed, [r - drop for r in full]))
    report = mc.calibrate_mask(_evidence("Blackjack-v1", rows, expert=0.0, random=-1.0))
    assert report["passing_methods"] == ["bc"]
    assert report["decision"] == mc.REJECTED
    assert report["reasons"] == [mc.MASKED_NEAR_REFERENCE]


def test_one_method_below_near_reference_is_not_all_near():
    report = mc.calibrate_mask(
        _frozen_varied({"ftl": 96, "bc": 40, "bc_iid": 97, "bc_prefix": 98}, 100)
    )
    assert mc.MASKED_NEAR_REFERENCE not in report["reasons"]
    assert report["decision"] == mc.ACCEPTED, report["messages"]


def _cliff_certificate(**overrides) -> Dict:
    """Row-pair certificate: best in class -15 against the full optimum -13."""
    cert = {
        "verified": True,
        "sha256": "e" * 64,
        "mask_sha256": HASH,
        "metric": "return",
        "policy_class": mc.DETERMINISTIC_MEMORYLESS,
        "method": "exhaustive_search",
        "full_optimum": -13.0,
        "best_in_class": -15.0,
        "lower_reference": -200.0,
        "goal_feasible": True,
        "ordinary_binning": True,
    }
    cert.update(overrides)
    return cert


# Column pairs: no deterministic memoryless policy reaches the goal.
COLUMN_PAIRS = dict(best_in_class=-200.0, goal_feasible=False)


def _cliff(
    masked_fn, full_fn=None, expert=-13.0, random=-2113.0, eligibility="row_pairs"
) -> Dict:
    """``*_fn(method) -> (returns, successes, truncated)``."""

    def solved(method):
        return [-13.0] * 100, [True] * 100, [False] * 100

    full_fn = full_fn or solved
    rows = []
    for seed in SEEDS:
        for method in mc.REQUIRED_METHODS:
            for arm, fn in (("full", full_fn), ("masked", masked_fn)):
                ret, suc, trunc = fn(method)
                rows.append(_row(method, arm, seed, ret, suc, trunc))
    ev = _evidence("CliffWalking-v0", rows, expert=expert, random=random)
    if eligibility == "row_pairs":
        ev["eligibility"] = _cliff_certificate()
    elif eligibility is not None:
        ev["eligibility"] = eligibility
    return ev


def _stall(method):
    return [-200.0] * 100, [False] * 100, [True] * 100


def test_cliff_stalling_mask_is_floor_and_compression_is_only_a_flag():
    # -2113 is the uniform-random reference, the only accepted one: a stalled
    # -200 normalizes to 0.911 although it never reaches the goal.
    report = mc.calibrate_mask(_cliff(_stall))
    bc = report["per_method"]["bc"]
    assert bc["masked"]["normalized_mean"] == pytest.approx(0.911, abs=1e-3)
    assert report["decision"] == mc.REJECTED
    assert mc.FLOOR_COLLAPSE in report["reasons"]
    assert "normalization_compressed:bc" in report["flags"]
    assert mc.SUCCESS_RATE_PRESENTATION_REQUIRED in report["flags"]


def test_cliff_full_arm_stalling_fails_control_qualification():
    report = mc.calibrate_mask(_cliff(_stall, full_fn=_stall))
    assert report["decision"] != mc.ACCEPTED
    assert mc.CONTROL_UNQUALIFIED in report["reasons"]
    assert report["control"]["qualified"] is False


CLIFF_PARTIAL = {"ftl": 50, "bc": 30, "bc_iid": 40, "bc_prefix": 35}


def _cliff_partial(method):
    k = CLIFF_PARTIAL[method]
    returns = [-13.0] * k + [-113.0] * (100 - k)  # failures fell off the cliff
    successes = [True] * k + [False] * (100 - k)
    truncated = [False] * k + [True] * (100 - k)
    return returns, successes, truncated


def test_cliff_partial_success_uses_goal_success_criterion():
    report = mc.calibrate_mask(_cliff(_cliff_partial))
    assert report["decision"] == mc.ACCEPTED, report["reasons"]
    assert report["criterion"] == "goal_success"
    ftl = report["per_method"]["ftl"]
    assert ftl["effect"] == pytest.approx(0.50)
    assert report["most_robust_method"] == "ftl"
    assert report["scientific_eligibility"]["status"] == mc.ELIGIBILITY_ELIGIBLE


def test_cliff_near_reference_is_judged_on_goal_success_not_compressed_return():
    # Root reproduction: full -13, uniform random -2113, partial success 30 to
    # 50% with raw failures at -113. Every normalized mean is above 0.95 only
    # because the random reference is far below; goal success is 0.3 to 0.5.
    report = mc.calibrate_mask(_cliff(_cliff_partial))
    assert report["passing_methods"] == list(mc.REQUIRED_METHODS)
    assert report["decision"] == mc.ACCEPTED, report["messages"]
    assert mc.MASKED_NEAR_REFERENCE not in report["reasons"]
    assert report["near_reference_criterion"] == mc.GOAL_SUCCESS
    near = report["near_reference"]
    assert near["criterion"] == mc.GOAL_SUCCESS
    assert near["threshold"] == mc.NEAR_OPTIMAL
    assert near["all_near"] is False
    assert near["masked"] == {m: k / 100 for m, k in CLIFF_PARTIAL.items()}
    # The compressed normalized means are retained as a flag, not a reason.
    assert near["normalized_all_near"] is True
    assert mc.REFERENCE_COMPRESSION in report["flags"]
    for method, k in CLIFF_PARTIAL.items():
        masked = report["per_method"][method]["masked"]
        expected = (k * -13.0 + (100 - k) * -113.0) / 100
        assert masked["normalized_mean"] == pytest.approx(
            (expected + 2113.0) / 2100.0,
        )
        assert masked["normalized_mean"] > mc.NEAR_OPTIMAL
    assert report["reference"]["random_return"] == -2113.0
    assert report["reference"]["reference_type"] == mc.REFERENCE_RANDOM


def test_cliff_all_goal_success_at_least_near_optimal_is_rejected():
    def k95(method):
        return (
            [-13.0] * 95 + [-113.0] * 5,
            [True] * 95 + [False] * 5,
            [False] * 95 + [True] * 5,
        )

    report = mc.calibrate_mask(_cliff(k95))
    assert report["decision"] == mc.REJECTED
    assert mc.MASKED_NEAR_REFERENCE in report["reasons"]  # equality is near
    assert report["near_reference"]["all_near"] is True
    assert mc.REFERENCE_COMPRESSION not in report["flags"]


def test_cliff_one_method_below_near_optimal_success_is_not_all_near():
    def masked(method):
        k = 60 if method == "bc" else 96
        return (
            [-13.0] * k + [-113.0] * (100 - k),
            [True] * k + [False] * (100 - k),
            [False] * k + [True] * (100 - k),
        )

    report = mc.calibrate_mask(_cliff(masked))
    assert mc.MASKED_NEAR_REFERENCE not in report["reasons"]
    assert report["decision"] == mc.ACCEPTED, report["messages"]
    assert report["passing_methods"] == ["bc"]


def test_cliff_success_reduction_below_min_gap_is_insufficient():
    def near_full(method):
        k = 95
        return (
            [-13.0] * k + [-113.0] * (100 - k),
            [True] * k + [False] * (100 - k),
            [False] * k + [True] * (100 - k),
        )

    report = mc.calibrate_mask(_cliff(near_full))
    assert report["decision"] == mc.REJECTED
    assert mc.INSUFFICIENT_EFFECT in report["reasons"]


def test_cliff_all_masked_zero_goal_is_floor_even_with_varied_returns():
    def falls(method):
        returns = [-113.0] * 50 + [-213.0] * 50
        return returns, [False] * 100, [True] * 100

    report = mc.calibrate_mask(_cliff(falls))
    assert report["decision"] == mc.REJECTED
    assert mc.FLOOR_COLLAPSE in report["reasons"]


def test_mountaincar_all_timeouts_is_floor_without_compression():
    rows = []
    for seed in SEEDS:
        for method in mc.REQUIRED_METHODS:
            rows.append(
                _row(method, "full", seed, [-110.0] * 50 + [-120.0] * 50, [True] * 100)
            )
            rows.append(
                _row(
                    method, "masked", seed, [-200.0] * 100, [False] * 100, [True] * 100
                )
            )
    report = mc.calibrate_mask(
        _evidence("MountainCar-v0", rows, expert=-110.0, random=-200.0)
    )
    assert report["decision"] == mc.REJECTED
    assert mc.FLOOR_COLLAPSE in report["reasons"]
    assert not [f for f in report["flags"] if f.startswith(mc.NORMALIZATION_COMPRESSED)]
    assert report["per_method"]["bc"]["masked"]["normalized_mean"] == pytest.approx(0.0)


def test_frozenlake_all_zero_masked_is_floor():
    rows = []
    for seed in SEEDS:
        for method in mc.REQUIRED_METHODS:
            rows.append(_row(method, "full", seed, [1.0] * 100, [True] * 100))
            rows.append(_row(method, "masked", seed, [0.0] * 100, [False] * 100))
    report = mc.calibrate_mask(_evidence("FrozenLake-v1", rows, expert=1.0, random=0.0))
    assert report["decision"] == mc.REJECTED
    assert mc.FLOOR_COLLAPSE in report["reasons"]


# ---------------------------------------------------------------------------
# Correction: floor is aggregate, compression is goal-objective only
# ---------------------------------------------------------------------------


def test_one_method_floor_is_a_flag_when_others_are_informative():
    report = mc.calibrate_mask(_frozen_varied({**ACCEPTABLE, "bc": 0}))
    assert report["decision"] == mc.ACCEPTED, report["reasons"]
    assert report["per_method"]["bc"]["masked"]["floor"] is True
    assert "masked_floor:bc" in report["flags"]
    assert "masked_floor:ftl" not in report["flags"]


def test_all_four_methods_at_floor_is_rejected():
    report = mc.calibrate_mask(
        _frozen_varied({m: 0 for m in mc.REQUIRED_METHODS}),
    )
    assert report["decision"] == mc.REJECTED
    assert mc.FLOOR_COLLAPSE in report["reasons"]
    for method in mc.REQUIRED_METHODS:
        assert f"masked_floor:{method}" in report["flags"]


def _two_arm(env_name, full_fn, masked_fn, expert, random) -> Dict:
    """``*_fn(method) -> (returns, successes, truncated)`` for every seed."""
    rows = []
    for seed in SEEDS:
        for method in mc.REQUIRED_METHODS:
            for arm, fn in (("full", full_fn), ("masked", masked_fn)):
                ret, suc, trunc = fn(method)
                rows.append(_row(method, arm, seed, ret, suc, trunc))
    return _evidence(env_name, rows, expert=expert, random=random)


def _cart_full_500(method):
    return [500.0] * 100, [True] * 100, [True] * 100


def _cart_masked_partial(method):
    returns = list(np.linspace(100.0, 250.0, 100))
    return returns, [False] * 100, [False] * 100  # every pole fell


def test_cartpole_partial_masked_returns_are_not_compressed_failure():
    report = mc.calibrate_mask(
        _two_arm(
            "CartPole-v1",
            _cart_full_500,
            _cart_masked_partial,
            expert=500.0,
            random=22.0,
        ),
    )
    assert report["decision"] == mc.ACCEPTED, report["messages"]
    bc = report["per_method"]["bc"]
    assert bc["masked"]["success_rate"] == 0.0
    assert bc["masked"]["normalized_mean"] == pytest.approx((175.0 - 22.0) / 478.0)
    assert bc["effect"] > mc.MIN_NORM_GAP
    assert report["per_method"]["ftl"]["full"]["success_rate"] == 1.0


def test_lunar_partial_shaped_masked_return_is_not_auto_rejected():
    rng = np.random.default_rng(4)

    def full(method):
        return list(rng.normal(230.0, 20.0, 100)), [True] * 100, [False] * 100

    def masked(method):
        return list(rng.normal(60.0, 60.0, 100)), [False] * 100, [False] * 100

    report = mc.calibrate_mask(
        _two_arm("LunarLander-v2", full, masked, expert=230.0, random=-180.0),
    )
    assert report["decision"] == mc.ACCEPTED, report["messages"]
    assert not [f for f in report["flags"] if f.startswith(mc.NORMALIZATION_COMPRESSED)]


def test_cartpole_partial_full_arm_is_flagged_not_unqualified():
    def full(method):
        return list(np.linspace(300.0, 450.0, 100)), [False] * 100, [False] * 100

    def masked(method):
        return list(np.linspace(50.0, 150.0, 100)), [False] * 100, [False] * 100

    report = mc.calibrate_mask(
        _two_arm("CartPole-v1", full, masked, expert=500.0, random=22.0),
    )
    assert mc.CONTROL_UNQUALIFIED not in report["reasons"]
    assert "full_arm_zero_success:ftl" in report["flags"]
    # Competence of a return-objective full arm rests on upstream qualification.
    ev = _two_arm("CartPole-v1", full, masked, expert=500.0, random=22.0)
    ev["full_control"] = {"verified": False}
    assert mc.CONTROL_UNQUALIFIED in mc.calibrate_mask(ev)["reasons"]


def test_cartpole_full_survival_is_not_flagged():
    report = mc.calibrate_mask(
        _two_arm(
            "CartPole-v1",
            _cart_full_500,
            _cart_masked_partial,
            expert=500.0,
            random=22.0,
        ),
    )
    assert not [f for f in report["flags"] if f.startswith("full_arm_zero_success")]


def _cart_masked_constant_195(method):
    return [195.0] * 100, [False] * 100, [False] * 100  # every pole fell at 195


def test_cartpole_constant_partial_balancing_is_not_a_floor():
    # Root reproduction: qualified full arm constant 500, every masked method
    # constant 195. Constancy alone is not a floor on a return objective.
    report = mc.calibrate_mask(
        _two_arm(
            "CartPole-v1",
            _cart_full_500,
            _cart_masked_constant_195,
            expert=500.0,
            random=22.0,
        ),
    )
    assert report["decision"] == mc.ACCEPTED, report["messages"]
    assert mc.FLOOR_COLLAPSE not in report["reasons"]
    assert report["passing_methods"] == list(mc.REQUIRED_METHODS)
    for method in mc.REQUIRED_METHODS:
        masked = report["per_method"][method]["masked"]
        assert masked["constant_returns"] is True
        assert masked["floor"] is None  # unsupported, never fabricated
        assert masked["floor_basis"] is None
        assert f"masked_floor:{method}" not in report["flags"]
        assert f"{mc.DEGENERATE_SE}:{method}" in report["flags"]
        assert report["per_method"][method]["effect"] == pytest.approx(305 / 478)
        assert masked["random_comparison"] == mc.ABOVE_RANDOM
    assert mc.FLOOR_UNKNOWN in report["flags"]
    assert mc.NO_BETTER_THAN_RANDOM not in report["reasons"]


@pytest.mark.parametrize("masked_return", [22.0, 9.0])
def test_cartpole_all_masked_at_or_below_random_is_rejected(masked_return):
    # Every masked method normalizes to 0 (exactly the uniform-random return)
    # or below it. The effect gate passes, yet the mask is uninformative. The
    # floor stays unknown: this is a random-reference comparison, not a floor.
    def masked(method):
        return [masked_return] * 100, [False] * 100, [False] * 100

    report = mc.calibrate_mask(
        _two_arm("CartPole-v1", _cart_full_500, masked, expert=500.0, random=22.0),
    )
    assert report["passing_methods"] == list(mc.REQUIRED_METHODS)
    assert report["decision"] == mc.REJECTED
    assert report["reasons"] == [mc.NO_BETTER_THAN_RANDOM]
    assert mc.FLOOR_COLLAPSE not in report["reasons"]
    assert mc.FLOOR_UNKNOWN in report["flags"]
    for method in mc.REQUIRED_METHODS:
        stats = report["per_method"][method]["masked"]
        assert stats["floor"] is None
        assert stats["normalized_mean"] <= 0.0
        assert stats["random_comparison"] == mc.AT_OR_BELOW_RANDOM
        assert f"{mc.AT_OR_BELOW_RANDOM}:{method}" in report["flags"]


def test_lunar_all_masked_below_random_is_rejected():
    rng = np.random.default_rng(9)

    def full(method):
        return list(rng.normal(230.0, 20.0, 100)), [True] * 100, [False] * 100

    def masked(method):
        return list(rng.normal(-300.0, 40.0, 100)), [False] * 100, [False] * 100

    report = mc.calibrate_mask(
        _two_arm("LunarLander-v2", full, masked, expert=230.0, random=-180.0),
    )
    assert report["decision"] == mc.REJECTED
    assert report["reasons"] == [mc.NO_BETTER_THAN_RANDOM]


def test_cartpole_masked_indistinct_from_random_is_inconclusive():
    # bc is 1 return above random with an SE near 1.3: not clearly better.
    def masked(method):
        if method == "bc":
            return [1.0, 45.0] * 50, [False] * 100, [False] * 100
        return [22.0] * 100, [False] * 100, [False] * 100

    report = mc.calibrate_mask(
        _two_arm("CartPole-v1", _cart_full_500, masked, expert=500.0, random=22.0),
    )
    assert report["decision"] == mc.INCONCLUSIVE
    assert report["reasons"] == [mc.INCONCLUSIVE_NOISE]
    bc = report["per_method"]["bc"]["masked"]
    assert bc["normalized_mean"] > 0.0
    assert bc["normalized_mean"] <= mc.SIGNIFICANCE_SE * bc["normalized_se"]
    assert bc["random_comparison"] == mc.INDISTINCT_FROM_RANDOM
    assert f"{mc.INDISTINCT_FROM_RANDOM}:bc" in report["flags"]


def test_cartpole_one_masked_method_above_random_is_enough():
    def masked(method):
        if method == "ftl":
            return [195.0] * 100, [False] * 100, [False] * 100
        return [22.0] * 100, [False] * 100, [False] * 100

    report = mc.calibrate_mask(
        _two_arm("CartPole-v1", _cart_full_500, masked, expert=500.0, random=22.0),
    )
    assert report["decision"] == mc.ACCEPTED, report["messages"]
    assert f"{mc.AT_OR_BELOW_RANDOM}:bc" in report["flags"]
    assert f"{mc.AT_OR_BELOW_RANDOM}:ftl" not in report["flags"]


def test_random_comparison_is_computed_on_every_arm():
    report = mc.calibrate_mask(_frozen_varied({**ACCEPTABLE, "bc": 0}))
    assert report["decision"] == mc.ACCEPTED, report["messages"]
    bc = report["per_method"]["bc"]
    assert bc["masked"]["random_comparison"] == mc.AT_OR_BELOW_RANDOM
    assert bc["full"]["random_comparison"] == mc.ABOVE_RANDOM
    assert f"{mc.AT_OR_BELOW_RANDOM}:bc" in report["flags"]
    assert report["per_method"]["ftl"]["masked"]["random_comparison"] == (
        mc.ABOVE_RANDOM
    )


def _pm_one(n_win: int, n: int) -> List[float]:
    return [1.0] * n_win + [-1.0] * (n - n_win)


def test_blackjack_collapse_to_random_is_rejected():
    # Root reproduction of review F1: full 4750 +1 / 5250 -1 (mean -0.05),
    # every masked method 3000 +1 / 7000 -1 (mean -0.4), references expert
    # -0.05 and random -0.39. Not a lowest-payout floor, yet no better than
    # uniform random: /4 accepted it.
    report = mc.calibrate_mask(
        _blackjack(
            _pm_one(4750, 10000), _pm_one(3000, 10000), expert=-0.05, random=-0.39
        ),
    )
    assert report["decision"] == mc.REJECTED, report["messages"]
    assert report["reasons"] == [mc.NO_BETTER_THAN_RANDOM]
    assert report["passing_methods"] == list(mc.REQUIRED_METHODS)
    for method in mc.REQUIRED_METHODS:
        masked = report["per_method"][method]["masked"]
        assert masked["floor"] is False
        assert masked["floor_basis"] == mc.FLOOR_LOWEST_PAYOUT
        assert masked["normalized_mean"] == pytest.approx(-0.01 / 0.34)
        assert masked["random_comparison"] == mc.AT_OR_BELOW_RANDOM
        assert f"{mc.AT_OR_BELOW_RANDOM}:{method}" in report["flags"]
    assert mc.FLOOR_COLLAPSE not in report["reasons"]


def test_slippery_success_at_or_below_random_is_rejected():
    # A few masked successes (at or below the uniform value) are not a goal
    # floor, but they are no better than random.
    rows = []
    for seed in SEEDS:
        for method in mc.REQUIRED_METHODS:
            full, masked = _binary(74), _binary(1)
            rows.append(_row(method, "full", seed, full, [r > 0 for r in full]))
            rows.append(_row(method, "masked", seed, masked, [r > 0 for r in masked]))
    report = mc.calibrate_mask(
        _evidence("FrozenLakeSlippery-v0", rows, expert=0.74, random=0.014),
    )
    assert report["decision"] == mc.REJECTED, report["messages"]
    assert report["reasons"] == [mc.NO_BETTER_THAN_RANDOM]
    for method in mc.REQUIRED_METHODS:
        masked = report["per_method"][method]["masked"]
        assert masked["floor"] is False
        assert masked["random_comparison"] == mc.AT_OR_BELOW_RANDOM


def test_known_floor_indistinct_from_random_is_only_a_flag():
    # bc is above random but within 2 SE; the rest are at or below. On a
    # known-floor environment this is a flag, never a new inconclusive stop.
    full = _pm_one(48, 100)
    rows = []
    for seed in SEEDS:
        for method in mc.REQUIRED_METHODS:
            masked = _pm_one(31 if method == "bc" else 30, 100)
            rows.append(_row(method, "full", seed, full))
            rows.append(_row(method, "masked", seed, masked))
    report = mc.calibrate_mask(
        _evidence("Blackjack-v1", rows, expert=-0.05, random=-0.39),
    )
    bc = report["per_method"]["bc"]["masked"]
    assert bc["normalized_mean"] > 0.0
    assert bc["random_comparison"] == mc.INDISTINCT_FROM_RANDOM
    assert f"{mc.INDISTINCT_FROM_RANDOM}:bc" in report["flags"]
    assert f"{mc.AT_OR_BELOW_RANDOM}:ftl" in report["flags"]
    assert report["decision"] == mc.ACCEPTED, report["messages"]
    assert report["reasons"] == []


def test_goal_success_env_keeps_its_goal_criterion_below_random():
    # CliffWalking: returns below the uniform-random reference with partial
    # goal success. The comparison is reported, but only goal success judges.
    def masked(method, seed):
        returns = [-13.0] * 50 + [-5000.0] * 50
        return returns, [True] * 50 + [False] * 50, [False] * 50 + [True] * 50

    report = mc.calibrate_mask(_cliff_seeded(masked, digests=False))
    assert report["decision"] == mc.ACCEPTED, report["messages"]
    assert mc.NO_BETTER_THAN_RANDOM not in report["reasons"]
    for method in mc.REQUIRED_METHODS:
        stats = report["per_method"][method]["masked"]
        assert stats["random_comparison"] == mc.AT_OR_BELOW_RANDOM
        assert f"{mc.AT_OR_BELOW_RANDOM}:{method}" in report["flags"]


def test_cartpole_constant_masked_still_fails_insufficient_effect():
    def masked(method):
        return [480.0] * 100, [False] * 100, [False] * 100

    report = mc.calibrate_mask(
        _two_arm("CartPole-v1", _cart_full_500, masked, expert=500.0, random=22.0),
    )
    assert report["decision"] == mc.REJECTED
    assert report["reasons"] == [mc.INSUFFICIENT_EFFECT, mc.MASKED_NEAR_REFERENCE]


def test_lunar_constant_masked_return_is_not_a_floor():
    def full(method):
        return [230.0, 240.0] * 50, [True] * 100, [False] * 100

    def masked(method):
        return [-100.0] * 100, [False] * 100, [False] * 100

    report = mc.calibrate_mask(
        _two_arm("LunarLander-v2", full, masked, expert=235.0, random=-180.0),
    )
    assert mc.FLOOR_COLLAPSE not in report["reasons"]
    assert report["decision"] == mc.ACCEPTED, report["messages"]
    assert report["per_method"]["bc"]["masked"]["floor"] is None


@pytest.mark.parametrize("payout", [0.0, 1.0])
def test_blackjack_constant_non_lowest_payout_is_not_a_floor(payout):
    rng = np.random.default_rng(5)
    full = list(rng.normal(0.0, 0.5, 100))
    report = mc.calibrate_mask(
        _blackjack(full, [payout] * 100, expert=0.0, random=-1.0),
    )
    assert mc.FLOOR_COLLAPSE not in report["reasons"]
    bc = report["per_method"]["bc"]["masked"]
    assert bc["floor"] is False
    assert bc["floor_basis"] == mc.FLOOR_LOWEST_PAYOUT


def test_blackjack_lowest_payout_floor_is_recorded_with_its_basis():
    rng = np.random.default_rng(2)
    full = list(rng.normal(0.0, 0.5, 100))
    report = mc.calibrate_mask(_blackjack(full, [-1.0] * 100, expert=0.0, random=-1.0))
    bc = report["per_method"]["bc"]["masked"]
    assert bc["floor"] is True
    assert bc["floor_basis"] == mc.FLOOR_LOWEST_PAYOUT
    assert mc.FLOOR_UNKNOWN not in report["flags"]


def test_blackjack_full_arm_at_lowest_payout_is_unqualified():
    report = mc.calibrate_mask(
        _blackjack([-1.0] * 100, [-1.0] * 100, expert=0.0, random=-1.0)
    )
    assert mc.CONTROL_UNQUALIFIED in report["reasons"]


def test_goal_only_floors_keep_their_goal_failure_basis():
    report = mc.calibrate_mask(
        _frozen_varied({m: 0 for m in mc.REQUIRED_METHODS}),
    )
    assert report["per_method"]["bc"]["masked"]["floor_basis"] == mc.FLOOR_GOAL_FAILURE
    assert mc.FLOOR_UNKNOWN not in report["flags"]


def test_taxi_like_compression_is_a_flag_not_a_rejection():
    def full(method):
        return [6.0, 8.0, 10.0, 8.0] * 25, [True] * 100, [False] * 100

    def masked(method):
        if method == "bc":  # times out every episode, illegal-action penalties
            return [-200.0, -209.0] * 50, [False] * 100, [True] * 100
        return (
            [8.0] * 50 + [-200.0] * 50,
            [True] * 50 + [False] * 50,
            [False] * 50 + [True] * 50,
        )

    report = mc.calibrate_mask(
        _two_arm("Taxi-v3", full, masked, expert=8.0, random=-770.0),
    )
    assert report["decision"] == mc.ACCEPTED, report["messages"]
    assert report["reasons"] == []
    assert "normalization_compressed:bc" in report["flags"]
    assert mc.SUCCESS_RATE_PRESENTATION_REQUIRED in report["flags"]
    assert "masked_floor:bc" in report["flags"]
    bc = report["per_method"]["bc"]
    assert bc["masked"]["normalized_mean"] > 0.7
    assert bc["masked"]["success_rate"] == 0.0  # what the success panel shows


def test_compression_alone_never_rejects_but_all_floor_still_does():
    def full(method):
        return [6.0, 8.0, 10.0, 8.0] * 25, [True] * 100, [False] * 100

    def masked(method):  # every method compressed and at the floor
        return [-200.0] * 100, [False] * 100, [True] * 100

    report = mc.calibrate_mask(
        _two_arm("Taxi-v3", full, masked, expert=8.0, random=-770.0),
    )
    assert report["decision"] == mc.REJECTED
    assert mc.FLOOR_COLLAPSE in report["reasons"]
    for method in mc.REQUIRED_METHODS:
        assert f"normalization_compressed:{method}" in report["flags"]


def test_acrobot_all_timeouts_for_all_methods_is_floor():
    def full(method):
        return [-80.0, -90.0] * 50, [True] * 100, [False] * 100

    def masked(method):
        return [-500.0] * 100, [False] * 100, [True] * 100

    report = mc.calibrate_mask(
        _two_arm("Acrobot-v1", full, masked, expert=-85.0, random=-499.5),
    )
    assert report["decision"] == mc.REJECTED
    assert mc.FLOOR_COLLAPSE in report["reasons"]
    assert not [f for f in report["flags"] if f.startswith(mc.NORMALIZATION_COMPRESSED)]


def test_goal_objective_full_arm_without_success_is_still_unqualified():
    def full(method):
        return [-500.0, -499.0] * 50, [False] * 100, [True] * 100

    report = mc.calibrate_mask(
        _two_arm("Acrobot-v1", full, full, expert=-85.0, random=-499.5),
    )
    assert mc.CONTROL_UNQUALIFIED in report["reasons"]


# ---------------------------------------------------------------------------
# CliffWalking-v0 with the uniform-random reference and eligibility
# ---------------------------------------------------------------------------

CLIFF = "CliffWalking-v0"
CLIFF_RANDOM = -2113.0
START_DIGEST = "f" * 64
_ROW_PAIRS = object()


def _cliff_solved(method, seed):
    return [-13.0] * 100, [True] * 100, [False] * 100


def _cliff_seeded(
    masked_fn, full_fn=_cliff_solved, digests=True, eligibility=_ROW_PAIRS
) -> Dict:
    """Uniform-random reference; ``*_fn(method, seed) -> (ret, suc, trunc)``."""
    rows = []
    for seed in SEEDS:
        for method in mc.REQUIRED_METHODS:
            for arm, fn in (("full", full_fn), ("masked", masked_fn)):
                ret, suc, trunc = fn(method, seed)
                row = _row(method, arm, seed, ret, suc, trunc)
                if digests:
                    row["initial_state_digests"] = [START_DIGEST] * len(ret)
                rows.append(row)
    ev = _evidence(CLIFF, rows, expert=-13.0, random=CLIFF_RANDOM)
    if eligibility is _ROW_PAIRS:
        ev["eligibility"] = _cliff_certificate()
    elif eligibility is not None:
        ev["eligibility"] = eligibility
    return ev


def _cliff_half(method, seed):
    returns = [-15.0] * 50 + [-200.0] * 50
    return returns, [True] * 50 + [False] * 50, [False] * 50 + [True] * 50


def _cliff_stall(method, seed):
    return _stall(method)


def _cliff_row_pair_detour(method, seed):
    return [-15.0] * 100, [True] * 100, [False] * 100


def _cliff_seed_split(method, seed):
    """Deterministic single start: seeds 0 and 1 reach the goal, seed 2 stalls."""
    return _cliff_row_pair_detour(method, seed) if seed < 2 else _stall(method)


def test_cliff_random_reference_half_goals_is_accepted():
    ev = _cliff_seeded(_cliff_half, digests=False)
    snapshot = copy.deepcopy(ev)
    report = mc.calibrate_mask(ev)
    assert report["decision"] == mc.ACCEPTED, report["messages"]
    assert report["criterion"] == mc.GOAL_SUCCESS
    for method in mc.REQUIRED_METHODS:
        stats = report["per_method"][method]
        assert stats["masked"]["raw_mean"] == pytest.approx(-107.5)
        assert stats["effect"] == pytest.approx(0.5)
        # Compressed above 0.95, so only the goal-success scale can judge it.
        assert stats["masked"]["normalized_mean"] == pytest.approx(2005.5 / 2100.0)
        assert stats["degenerate_se"] is False
        assert stats["masked"]["random_comparison"] == mc.ABOVE_RANDOM
    assert mc.REFERENCE_COMPRESSION in report["flags"]
    # Reference values are reported as given, never rewritten.
    assert report["reference"]["random_return"] == CLIFF_RANDOM
    assert report["reference"]["expert_return"] == -13.0
    assert report["reference"]["reference_type"] == "random_policy"
    assert report["scientific_eligibility"]["status"] == mc.ELIGIBILITY_ELIGIBLE
    assert ev == snapshot


def test_cliff_deterministic_single_start_flags_degenerate_se():
    report = mc.calibrate_mask(_cliff_seeded(_cliff_seed_split))
    assert report["decision"] == mc.ACCEPTED, report["messages"]
    assert report["single_start"] is True
    assert "deterministic_single_start" in report["flags"]
    for method in mc.REQUIRED_METHODS:
        stats = report["per_method"][method]
        assert f"degenerate_se:{method}" in report["flags"]
        assert stats["degenerate_se"] is True
        assert stats["success_se"] == 0.0
        assert stats["gate_zero_se_seeds"] == 3
        assert stats["effective_sample_size"] == 3
        assert stats["effect"] == pytest.approx(1 / 3)
        # Between-seed spread is reported; the frozen gate is unchanged.
        assert stats["gate_between_seed_se"] == pytest.approx(1 / 3)
        masked = stats["masked"]
        assert masked["seed_success_rates"] == [1.0, 1.0, 0.0]
        assert masked["seeds_with_success"] == 2
        assert masked["seed_success_summary"] == "2/3 seeds with success"
        assert stats["full"]["seed_success_summary"] == "3/3 seeds with success"


def test_single_start_is_not_imputed_without_digests():
    report = mc.calibrate_mask(_cliff_seeded(_cliff_seed_split, digests=False))
    assert report["single_start"] is None
    assert "deterministic_single_start" not in report["flags"]
    assert "degenerate_se:bc" in report["flags"]  # measured, not imputed


def test_varied_starts_are_not_single_start():
    ev = _cliff_seeded(_cliff_half)
    ev["rows"][0]["initial_state_digests"][1] = "0" * 64
    report = mc.calibrate_mask(ev)
    assert report["single_start"] is False
    assert "deterministic_single_start" not in report["flags"]


def test_cliff_row_pairs_all_near_reference_is_rejected_but_benign():
    report = mc.calibrate_mask(_cliff_seeded(_cliff_row_pair_detour))
    assert report["decision"] == mc.REJECTED
    assert mc.MASKED_NEAR_REFERENCE in report["reasons"]
    assert mc.SCIENTIFICALLY_INELIGIBLE not in report["reasons"]
    eligibility = report["scientific_eligibility"]
    assert eligibility["status"] == mc.ELIGIBILITY_ELIGIBLE
    assert eligibility["class_gap"] == pytest.approx(2.0 / 187.0)
    assert eligibility["small_class_gap"] is True
    assert "small_class_gap" in report["flags"]
    assert report["benign_agnostic"] is True
    assert report["benign_note"] == mc.BENIGN_NOTE
    # The outcome report is retained in full.
    bc = report["per_method"]["bc"]
    assert bc["masked"]["normalized_mean"] == pytest.approx(2098.0 / 2100.0)
    assert bc["masked"]["success_rate"] == 1.0
    # Near on the goal-success scale, so the normalized agreement is no flag.
    assert report["near_reference"]["criterion"] == mc.GOAL_SUCCESS
    assert mc.REFERENCE_COMPRESSION not in report["flags"]


def test_cliff_column_pairs_goal_infeasible_is_rejected():
    ev = _cliff_seeded(_cliff_stall, eligibility=_cliff_certificate(**COLUMN_PAIRS))
    report = mc.calibrate_mask(ev)
    assert report["decision"] == mc.REJECTED
    assert mc.SCIENTIFICALLY_INELIGIBLE in report["reasons"]
    assert mc.FLOOR_COLLAPSE in report["reasons"]
    assert report["scientific_eligibility"]["status"] == mc.ELIGIBILITY_GOAL_INFEASIBLE
    assert report["benign_agnostic"] is False
    # Ineligibility does not depend on outcomes: it holds without any rows.
    ev["rows"] = []
    report = mc.calibrate_mask(ev)
    assert report["decision"] == mc.REJECTED
    assert mc.SCIENTIFICALLY_INELIGIBLE in report["reasons"]
    assert mc.MISSING_OUTCOMES in report["reasons"]


def test_cliff_requires_an_eligibility_certificate():
    ev = _cliff_seeded(_cliff_half, eligibility=None)
    report = _assert_not_accepted(ev, mc.ELIGIBILITY_UNVERIFIED)
    assert report["scientific_eligibility"]["required"] is True
    assert report["scientific_eligibility"]["status"] == mc.ELIGIBILITY_UNKNOWN


@pytest.mark.parametrize(
    "overrides",
    [
        {"verified": False},
        {"policy_class": "stochastic_memoryless"},
        {"goal_feasible": None},
        {"ordinary_binning": None},
    ],
)
def test_cliff_certificate_must_verify_ordinary_deterministic_feasibility(overrides):
    cert = _cliff_certificate(**overrides)
    for key in [k for k, v in overrides.items() if v is None]:
        cert.pop(key)
    ev = _cliff_seeded(_cliff_half, eligibility=cert)
    report = _assert_not_accepted(ev, mc.ELIGIBILITY_UNVERIFIED)
    assert report["scientific_eligibility"]["status"] == mc.ELIGIBILITY_UNVERIFIED


def test_offset_binning_is_rejected_not_searched():
    cert = _cliff_certificate(best_in_class=-13.0, ordinary_binning=False)
    report = mc.calibrate_mask(_cliff_seeded(_cliff_half, eligibility=cert))
    assert report["decision"] == mc.REJECTED
    assert report["scientific_eligibility"]["status"] == mc.ELIGIBILITY_NOT_ORDINARY


def test_cliff_one_stalled_method_is_a_flag():
    def masked(method, seed):
        return _stall(method) if method == "bc" else _cliff_half(method, seed)

    report = mc.calibrate_mask(_cliff_seeded(masked))
    assert report["decision"] == mc.ACCEPTED, report["messages"]
    assert "masked_floor:bc" in report["flags"]


def test_cliff_robust_ftl_with_stalled_others_is_accepted_with_flags():
    def masked(method, seed):
        return _cliff_solved(method, seed) if method == "ftl" else _stall(method)

    report = mc.calibrate_mask(_cliff_seeded(masked))
    assert report["decision"] == mc.ACCEPTED, report["messages"]
    assert report["passing_methods"] == ["bc", "bc_iid", "bc_prefix"]
    assert "insufficient_effect:ftl" in report["flags"]
    assert "no_partial_goal_success" in report["flags"]
    assert "masked_floor:bc" in report["flags"]


def test_reference_type_defaults_to_random_policy():
    report = mc.calibrate_mask(_accepted())
    assert report["reference"]["reference_type"] == mc.REFERENCE_RANDOM
    ev = _cliff_seeded(_cliff_half)
    assert "reference_type" not in ev["reference"]
    assert mc.calibrate_mask(ev)["reference"]["reference_type"] == "random_policy"
    ev["reference"]["reference_type"] = mc.REFERENCE_RANDOM  # explicit is fine
    assert mc.calibrate_mask(ev)["decision"] == mc.ACCEPTED


@pytest.mark.parametrize("ref_type", ["stalled_policy", "made_up", None])
@pytest.mark.parametrize(
    "make",
    [lambda: _accepted(), lambda: _cliff_seeded(_cliff_half)],
    ids=["frozen", "cliff"],
)
def test_only_uniform_random_reference_is_accepted(make, ref_type):
    # A stalled-policy reference is refused on every MDP, CliffWalking
    # included; it is never read as a uniform-random return.
    ev = make()
    ev["reference"]["reference_type"] = ref_type
    report = _assert_not_accepted(ev, mc.INVALID_EVIDENCE)
    assert report["per_method"] == {}
    assert any("uniform-random" in m for m in report["messages"])


def test_withdrawn_cliff_variant_is_not_in_the_gate():
    assert mc.criterion_for("CliffWalkingVariedStart-v0") == mc.NORMALIZED_RETURN
    ev = _cliff_seeded(_cliff_half)
    ev["env_name"] = "CliffWalkingVariedStart-v0"
    report = _assert_not_accepted(ev, mc.MISSING_OUTCOMES)
    # The variant is not registered, so it has no success rule either.
    assert "CliffWalkingVariedStart-v0" not in eval_utils.SUCCESS_RULES
    assert mc.success_used("CliffWalkingVariedStart-v0") is None
    assert mc.MISSING_OUTCOMES in report["reasons"]


# ---------------------------------------------------------------------------
# Scientific eligibility on other environments
# ---------------------------------------------------------------------------


def _generic_certificate(**overrides) -> Dict:
    cert = {
        "verified": True,
        "sha256": "e" * 64,
        "mask_sha256": HASH,
        "metric": "goal_success",
        "policy_class": mc.DETERMINISTIC_MEMORYLESS,
        "method": "exact_enumeration",
        "full_optimum": 0.9,
        "best_in_class": 0.5,
        "lower_reference": 0.0,
    }
    cert.update(overrides)
    return cert


def test_eligible_certificate_is_recorded_without_changing_detection():
    ev = _accepted()
    ev["eligibility"] = _generic_certificate()
    report = mc.calibrate_mask(ev)
    assert report["decision"] == mc.ACCEPTED, report["messages"]
    eligibility = report["scientific_eligibility"]
    assert eligibility["status"] == mc.ELIGIBILITY_ELIGIBLE
    assert eligibility["required"] is False
    assert eligibility["class_gap"] == pytest.approx(0.4 / 0.9)
    assert eligibility["certificate"]["sha256"] == "e" * 64


@pytest.mark.parametrize(
    "overrides, status",
    [
        ({"best_in_class": 0.9}, mc.ELIGIBILITY_REALIZABLE),
        ({"best_in_class": 0.0}, mc.ELIGIBILITY_AT_FLOOR),
        ({"goal_feasible": False}, mc.ELIGIBILITY_GOAL_INFEASIBLE),
    ],
)
def test_ineligible_certificate_is_rejected(overrides, status):
    ev = _accepted()
    ev["eligibility"] = _generic_certificate(**overrides)
    report = mc.calibrate_mask(ev)
    assert report["decision"] == mc.REJECTED
    assert report["reasons"] == [mc.SCIENTIFICALLY_INELIGIBLE]
    assert report["scientific_eligibility"]["status"] == status


def test_missing_or_unverified_certificate_does_not_block_a_continuous_task():
    ev = _two_arm(
        "CartPole-v1", _cart_full_500, _cart_masked_partial, expert=500.0, random=22.0
    )
    assert mc.calibrate_mask(ev)["decision"] == mc.ACCEPTED
    ev["eligibility"] = _generic_certificate(
        verified=False,
        metric="return",
        full_optimum=500.0,
        best_in_class=300.0,
        lower_reference=22.0,
    )
    report = mc.calibrate_mask(ev)
    assert report["decision"] == mc.ACCEPTED, report["messages"]
    assert report["scientific_eligibility"]["status"] == mc.ELIGIBILITY_UNVERIFIED
    # The positive witness requirement remains.
    ev["witness_verified"] = False
    _assert_not_accepted(ev, mc.WITNESS_UNVERIFIED)


@pytest.mark.parametrize(
    "overrides",
    [
        {"sha256": "not-a-hash"},
        {"mask_sha256": "c" * 64},
        {"metric": "normalized_return"},
        {"policy_class": ""},
        {"best_in_class": 0.95},
        {"lower_reference": 0.9},
        {"full_optimum": 1.5},
        {"best_in_class": math.nan},
        {"goal_feasible": "yes"},
        {"verified": 1},
    ],
)
def test_malformed_certificate_is_invalid(overrides):
    ev = _accepted()
    ev["eligibility"] = _generic_certificate(**overrides)
    report = _assert_not_accepted(ev, mc.INVALID_EVIDENCE)
    assert report["scientific_eligibility"]["status"] == mc.ELIGIBILITY_INVALID


def test_non_mapping_certificate_is_invalid():
    ev = _accepted()
    ev["eligibility"] = ["eligible"]
    _assert_not_accepted(ev, mc.INVALID_EVIDENCE)


# ---------------------------------------------------------------------------
# Deterministic evaluation declaration
# ---------------------------------------------------------------------------


def test_absent_deterministic_declaration_is_missing_not_imputed():
    ev = _accepted()
    ev.pop("deterministic_evaluation")
    report = _assert_not_accepted(ev, mc.MISSING_OUTCOMES)
    assert report["deterministic_evaluation"] is None
    assert report["per_method"] == {}


@pytest.mark.parametrize("value", [False, "true", 1])
def test_non_deterministic_declaration_is_invalid(value):
    ev = _accepted()
    ev["deterministic_evaluation"] = value
    _assert_not_accepted(ev, mc.INVALID_EVIDENCE)


def test_row_level_stochastic_evaluation_is_invalid():
    ev = _accepted()
    ev["rows"][0]["deterministic_evaluation"] = True
    assert mc.calibrate_mask(ev)["decision"] == mc.ACCEPTED
    ev["rows"][1]["deterministic_evaluation"] = False
    _assert_not_accepted(ev, mc.INVALID_EVIDENCE)


def test_production_row_flag_absent_is_missing_not_imputed():
    # Blocker: a copied production row without its recorded flag used to pass.
    ev = _accepted()
    ev["rows"][3].pop("evaluation_deterministic")
    report = _assert_not_accepted(ev, mc.MISSING_OUTCOMES)
    assert report["per_method"] == {}
    assert any("evaluation_deterministic" in m for m in report["messages"])
    ev["rows"][3]["evaluation_deterministic"] = None
    _assert_not_accepted(ev, mc.MISSING_OUTCOMES)


@pytest.mark.parametrize("value", [False, "true", 1, np.True_])
def test_production_row_flag_not_true_is_invalid(value):
    ev = _accepted()
    ev["rows"][3]["evaluation_deterministic"] = value
    report = _assert_not_accepted(ev, mc.INVALID_EVIDENCE)
    assert report["per_method"] == {}


def test_legacy_row_alias_cannot_stand_in_for_the_production_flag():
    ev = _accepted()
    row = ev["rows"][3]
    row["deterministic_evaluation"] = True  # consistent alias is retained
    assert mc.calibrate_mask(ev)["decision"] == mc.ACCEPTED
    row.pop("evaluation_deterministic")  # the alias alone is not the flag
    _assert_not_accepted(ev, mc.MISSING_OUTCOMES)


@pytest.mark.parametrize("actual, alias", [(True, False), (False, True)])
def test_conflicting_row_alias_is_invalid(actual, alias):
    ev = _accepted()
    ev["rows"][3]["evaluation_deterministic"] = actual
    ev["rows"][3]["deterministic_evaluation"] = alias
    _assert_not_accepted(ev, mc.INVALID_EVIDENCE)


def test_evidence_level_declaration_does_not_satisfy_the_row_flag():
    ev = _accepted()
    assert ev["deterministic_evaluation"] is True
    for row in ev["rows"]:
        row.pop("evaluation_deterministic")
    _assert_not_accepted(ev, mc.MISSING_OUTCOMES)


def test_production_round_eval_rolls_out_deterministically(monkeypatch):
    from imitation.experiments.ftrl import run_experiment as rex

    calls = []

    def spy(policy, venv, n_episodes, deterministic=None, **kwargs):
        calls.append(deterministic)
        seeds = kwargs.get("episode_seeds")
        return eval_utils.EvalResult(
            mean_return=1.0,
            episode_returns=[1.0],
            n_steps=1,
            rollout_batch=eval_utils.RolloutBatch(
                obs=np.zeros((1, 1)),
                expert_actions=np.zeros(1, dtype=np.int64),
                learner_actions=np.zeros(1, dtype=np.int64),
            ),
            current_round_disagreement=0.0,
            current_round_ce=0.0,
            evaluation_episode_seeds=None if seeds is None else list(seeds),
        )

    monkeypatch.setattr(eval_utils, "eval_policy_rollout", spy)
    monkeypatch.setattr(eval_utils, "compute_sampled_action_ce", lambda *a, **k: 0.0)
    baselines = {"expert_return": 1.0, "random_return": 0.0}
    rex._compute_round_eval(object(), object(), None, baselines)
    rex._compute_round_eval(object(), object(), None, baselines, eval_seed=5)
    assert calls == [True, True]


# ---------------------------------------------------------------------------
# Evidence schema failures
# ---------------------------------------------------------------------------


def _accepted() -> Dict:
    return _frozen_varied(ACCEPTABLE)


def _assert_not_accepted(evidence: Dict, reason: str, decision: str = mc.INCONCLUSIVE):
    report = mc.calibrate_mask(evidence)
    assert report["decision"] == decision
    assert reason in report["reasons"], report
    return report


def test_missing_fourth_method_is_missing_outcomes():
    ev = _accepted()
    ev["rows"] = [r for r in ev["rows"] if r["method"] != "bc_prefix"]
    report = _assert_not_accepted(ev, mc.MISSING_OUTCOMES)
    assert report["per_method"] == {}


def test_missing_seed_row_is_missing_outcomes():
    ev = _accepted()
    ev["rows"] = [
        r for r in ev["rows"] if not (r["seed"] == 2 and r["arm"] == "masked")
    ]
    _assert_not_accepted(ev, mc.MISSING_OUTCOMES)


def test_placeholder_row_is_missing_outcomes():
    ev = _accepted()
    ev["rows"][0]["final"] = False
    _assert_not_accepted(ev, mc.MISSING_OUTCOMES)


def test_legacy_row_without_outcome_fields_is_missing_not_pass():
    ev = _accepted()
    for row in ev["rows"]:
        for key in ("episode_terminated", "episode_truncated", "episode_successes"):
            row.pop(key)
    _assert_not_accepted(ev, mc.MISSING_OUTCOMES)


def test_fewer_than_min_episodes_is_missing_outcomes():
    ev = _accepted()
    row = ev["rows"][0]
    for key in mc.EPISODE_FIELDS:
        if row.get(key) is not None:
            row[key] = row[key][:99]
    _assert_not_accepted(ev, mc.MISSING_OUTCOMES)


def test_mask_hash_mismatch_is_invalid():
    ev = _accepted()
    ev["rows"][1]["mask_sha256"] = "c" * 64
    _assert_not_accepted(ev, mc.INVALID_EVIDENCE)


def test_malformed_mask_hash_is_invalid():
    ev = _accepted()
    ev["mask_sha256"] = "not-a-hash"
    _assert_not_accepted(ev, mc.INVALID_EVIDENCE)


def test_expert_checkpoint_hash_must_be_shared():
    ev = _accepted()
    ev["full_control"] = {
        "checkpoint_sha256": EXPERT_HASH,
        "return_quality_passed": True,
    }
    for row in ev["rows"]:
        row["expert_sha256"] = EXPERT_HASH
    assert mc.calibrate_mask(ev)["decision"] == mc.ACCEPTED
    ev["rows"][3]["expert_sha256"] = "d" * 64
    _assert_not_accepted(ev, mc.INVALID_EVIDENCE)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_return_is_invalid(bad):
    ev = _accepted()
    ev["rows"][0]["episode_returns"][5] = bad
    _assert_not_accepted(ev, mc.INVALID_EVIDENCE)


def test_paired_seed_mismatch_is_invalid():
    ev = _accepted()
    ev["rows"][1]["evaluation_episode_seeds"][0] = 999_999
    _assert_not_accepted(ev, mc.INVALID_EVIDENCE)


def test_paired_requires_evaluation_seeds():
    ev = _accepted()
    ev["rows"][1]["evaluation_episode_seeds"] = None
    _assert_not_accepted(ev, mc.MISSING_OUTCOMES)


def test_unpaired_does_not_need_matching_seeds():
    ev = _accepted()
    ev["pairing"] = "unpaired"
    for row in ev["rows"]:
        row["evaluation_episode_seeds"] = None
    report = mc.calibrate_mask(ev)
    assert report["decision"] == mc.ACCEPTED
    assert report["pairing"] == "unpaired"


def test_undeclared_pairing_defaults_to_unpaired():
    ev = _accepted()
    ev.pop("pairing")
    assert mc.calibrate_mask(ev)["pairing"] == "unpaired"


def test_paired_and_unpaired_standard_errors_differ():
    paired = mc.calibrate_mask(_accepted())
    ev = _accepted()
    ev["pairing"] = "unpaired"
    unpaired = mc.calibrate_mask(ev)
    p, u = paired["per_method"]["bc"], unpaired["per_method"]["bc"]
    assert p["raw_gap"] == pytest.approx(u["raw_gap"])
    assert p["raw_se"] != pytest.approx(u["raw_se"])


def test_unexpected_seed_is_invalid():
    ev = _accepted()
    extra = copy.deepcopy(ev["rows"][0])
    extra["seed"] = 77
    ev["rows"].append(extra)
    _assert_not_accepted(ev, mc.INVALID_EVIDENCE)


def test_duplicate_row_is_invalid():
    ev = _accepted()
    ev["rows"].append(copy.deepcopy(ev["rows"][0]))
    _assert_not_accepted(ev, mc.INVALID_EVIDENCE)


def test_nan_nll_present_is_invalid_but_nll_not_required():
    ev = _accepted()
    ev["rows"][0]["nll"] = 0.3
    assert mc.calibrate_mask(ev)["decision"] == mc.ACCEPTED
    ev["rows"][0]["nll"] = math.nan
    _assert_not_accepted(ev, mc.INVALID_EVIDENCE)


def test_reference_uncertainty_key_must_be_explicit():
    ev = _accepted()
    ev["reference"].pop("uncertainty")
    _assert_not_accepted(ev, mc.INVALID_EVIDENCE)


def test_reference_uncertainty_is_reported_separately():
    base = mc.calibrate_mask(_accepted())
    assert base["reference"]["uncertainty"] == "unavailable"
    ev = _accepted()
    ev["reference"]["uncertainty"] = {"expert_return_se": 0.01, "random_return_se": 0.0}
    report = mc.calibrate_mask(ev)
    assert report["reference"]["uncertainty"] == {
        "expert_return_se": 0.01,
        "random_return_se": 0.0,
    }
    assert "reference_uncertainty_unavailable" not in report["flags"]
    assert report["per_method"]["bc"]["raw_se"] == base["per_method"]["bc"]["raw_se"]


def test_reference_must_be_ordered_and_finite():
    ev = _accepted()
    ev["reference"]["expert_return"] = 0.0  # equals random
    _assert_not_accepted(ev, mc.INVALID_EVIDENCE)


def test_unverified_control_is_unqualified():
    ev = _accepted()
    ev["full_control"] = {"verified": False}
    report = _assert_not_accepted(ev, mc.CONTROL_UNQUALIFIED)
    assert report["control"]["qualified"] is False


def test_control_qualified_by_shared_checkpoint_quality():
    ev = _accepted()
    ev["full_control"] = {
        "checkpoint_sha256": EXPERT_HASH,
        "return_quality_passed": True,
    }
    assert mc.calibrate_mask(ev)["decision"] == mc.ACCEPTED
    ev["full_control"]["return_quality_passed"] = False
    _assert_not_accepted(ev, mc.CONTROL_UNQUALIFIED)


def test_unverified_witness_is_inconclusive():
    ev = _accepted()
    ev["witness_verified"] = False
    _assert_not_accepted(ev, mc.WITNESS_UNVERIFIED)


def test_unknown_environment_is_missing_outcomes():
    ev = _accepted()
    ev["env_name"] = "Unknown-v9"
    _assert_not_accepted(ev, mc.MISSING_OUTCOMES)


def test_calibrate_mask_does_not_mutate_input():
    ev = _accepted()
    snapshot = copy.deepcopy(ev)
    mc.calibrate_mask(ev)
    assert ev == snapshot


def test_report_is_json_serializable():
    report = mc.calibrate_mask(_accepted())
    assert json.loads(json.dumps(report)) == report


# ---------------------------------------------------------------------------
# Surveillance over legacy and new results
# ---------------------------------------------------------------------------


def test_missing_outcome_fields_flags_legacy_rows():
    legacy = {"method": "bc", "arm": "masked", "seed": 0, "episode_returns": [0.0]}
    missing = mc.missing_outcome_fields(legacy, "FrozenLake-v1")
    assert set(missing) == {
        "episode_lengths",
        "episode_terminated",
        "episode_truncated",
        "episode_successes",
    }
    assert "episode_successes" not in mc.missing_outcome_fields(legacy, "Blackjack-v1")


def test_survey_selects_unique_accepted_mask_and_flags_legacy():
    accepted = _accepted()
    accepted["mask"] = "frozen_good"
    rejected = _frozen_varied({m: 88 for m in mc.REQUIRED_METHODS})
    rejected["mask"] = "frozen_weak"
    legacy = _cliff(_stall)
    legacy["mask"] = "cliffwalking_column_hidden"
    for row in legacy["rows"]:
        row.pop("episode_successes")
    summary = mc.survey_calibrations([accepted, rejected, legacy])

    frozen = summary["environments"]["FrozenLakeVariedStart-v0"]
    assert frozen["selected_mask"] == "frozen_good"
    assert frozen["status"] == "selected"
    assert [c["mask"] for c in frozen["candidates"]] == ["frozen_good", "frozen_weak"]

    cliff = summary["environments"]["CliffWalking-v0"]
    assert cliff["selected_mask"] is None
    assert cliff["status"] == "none_accepted"
    assert cliff["candidates"][0]["decision"] == mc.INCONCLUSIVE
    assert {"env_name": "CliffWalking-v0", "mask": "cliffwalking_column_hidden"} in (
        summary["missing_outcomes"]
    )
    assert json.loads(json.dumps(summary)) == summary


def test_survey_does_not_promote_among_multiple_accepted():
    a = _accepted()
    a["mask"] = "m1"
    b = _accepted()
    b["mask"] = "m2"
    env = mc.survey_calibrations([a, b])["environments"]["FrozenLakeVariedStart-v0"]
    assert env["selected_mask"] is None
    assert env["status"] == "multiple_accepted_needs_review"
    assert env["accepted"] == ["m1", "m2"]


# ---------------------------------------------------------------------------
# Seeded random binning certificates (protocol /4, unchanged in /5 and /6)
# ---------------------------------------------------------------------------

RANDOM_RID = tasks.random_binning_id(CLIFF, 20, tasks.RANDOM_BINNING_SEED)
OTHER_RID = tasks.random_binning_id(CLIFF, 16, tasks.RANDOM_BINNING_SEED)
SLIPPERY_RID = tasks.random_binning_id(
    env_utils.FROZENLAKE_SLIPPERY, 6, tasks.RANDOM_BINNING_SEED
)


def _refuse_env(*args, **kwargs):
    raise AssertionError("metadata validation must not construct an environment")


@pytest.fixture
def no_env(monkeypatch):
    """Environment and expert constructors raise; each test is bounded to 60s."""
    for module, name in (
        (gym, "make"),
        (env_utils, "make_env"),
        (tasks, "_raw_env"),
        (tasks, "_reachable"),
        (tasks, "reachable_nonterminal"),
        (tasks, "alias_table"),
        (tasks, "describe"),
    ):
        monkeypatch.setattr(module, name, _refuse_env)

    def _timeout(signum, frame):
        raise TimeoutError("test exceeded its 60 s bound")

    previous = signal.signal(signal.SIGALRM, _timeout)
    signal.alarm(60)
    yield
    signal.alarm(0)
    signal.signal(signal.SIGALRM, previous)


def _static_binning(rid: str = RANDOM_RID, restriction_sha: str = HASH) -> Dict:
    """Binding built independently from the static `tasks` registry."""
    spec = tasks.RANDOM_ALIAS_SPECS[rid]
    return {
        "kind": "seeded_random",
        "generator": spec.generator,
        "restriction_id": rid,
        "env_name": spec.env_name,
        "partition_seed": spec.seed,
        "n_bins": spec.n_bins,
        "partition_sha256": tasks.random_partition_sha256(rid),
        "frozen_table_sha256": tasks.FROZEN_TABLE_SHA256[rid],
        "restriction_sha256": restriction_sha,
    }


def _controller_cliff_certificate(**overrides) -> Dict:
    """`certify_cliff` output as the private controller emits it today."""
    cert = {
        "verified": True,
        "metric": "return",
        "policy_class": mc.DETERMINISTIC_MEMORYLESS,
        "method": "branch_and_bound_simple_paths_deterministic_dynamics",
        "goal_feasible": True,
        "binning_kind": "seeded_random",
        "full_optimum": -13.0,
        "best_in_class": -15.0,
        "lower_reference": -200.0,
        "mask_sha256": HASH,
        "sha256": "e" * 64,
    }
    cert.update(overrides)
    return cert


def _random_cliff_certificate(**overrides) -> Dict:
    """The /4 shape: the controller certificate plus its static binding."""
    cert = _controller_cliff_certificate(
        binning=_static_binning(),
        lower_reference_kind="stall_benchmark",
    )
    cert.update(overrides)
    return cert


def _random_cliff(masked_fn=_cliff_half, cert=None, binning=None) -> Dict:
    ev = _cliff_seeded(
        masked_fn,
        eligibility=_random_cliff_certificate() if cert is None else cert,
    )
    ev["mask"] = RANDOM_RID
    ev["binning"] = _static_binning() if binning is None else binning
    return ev


def _invalid_eligibility(ev: Dict) -> Dict:
    report = _assert_not_accepted(ev, mc.INVALID_EVIDENCE)
    assert report["scientific_eligibility"]["status"] == mc.ELIGIBILITY_INVALID
    assert report["per_method"] == {}
    return report


def test_controller_certificate_without_binding_fails_closed(no_env):
    # /3 rejected this certificate as unverified for lacking ordinary_binning;
    # /4 refuses it as invalid: binning_kind alone is never trusted.
    ev = _random_cliff(cert=_controller_cliff_certificate())
    report = _invalid_eligibility(ev)
    assert any("binning" in m for m in report["messages"])


def test_seeded_random_cliff_certificate_with_static_binding_is_accepted(no_env):
    ev = _random_cliff()
    snapshot = copy.deepcopy(ev)
    report = mc.calibrate_mask(ev)
    assert report["protocol"] == "agnostic-mask-calibration/6"
    assert report["decision"] == mc.ACCEPTED, report["messages"]
    eligibility = report["scientific_eligibility"]
    assert eligibility["status"] == mc.ELIGIBILITY_ELIGIBLE
    assert eligibility["binning_kind"] == mc.SEEDED_RANDOM_BINNING
    assert eligibility["class_gap"] == pytest.approx(2.0 / 187.0)
    assert report["binning"] == _static_binning()
    assert ev == snapshot


def test_lower_benchmark_is_labelled_apart_from_normalization(no_env):
    report = mc.calibrate_mask(_random_cliff())
    lower = report["scientific_eligibility"]["lower_benchmark"]
    assert lower["value"] == -200.0
    assert lower["kind"] == mc.LOWER_STALL_BENCHMARK
    assert "not the normalization reference" in lower["role"]
    # Normalization stays uniform random (0) to expert (1); -200 is not used.
    assert report["reference"]["random_return"] == CLIFF_RANDOM
    assert report["reference"]["reference_type"] == mc.REFERENCE_RANDOM
    bc = report["per_method"]["bc"]["masked"]
    assert bc["normalized_mean"] == pytest.approx((-107.5 + 2113.0) / 2100.0)


def test_legacy_ordinary_certificate_stays_valid_and_is_labelled(no_env):
    report = mc.calibrate_mask(_cliff_seeded(_cliff_half))
    assert report["decision"] == mc.ACCEPTED, report["messages"]
    eligibility = report["scientific_eligibility"]
    assert eligibility["binning_kind"] == mc.LEGACY_ORDINARY_BINNING
    assert eligibility["lower_benchmark"]["kind"] is None  # unlabelled legacy
    assert report["binning"] is None


def test_legacy_ordinary_false_without_kind_is_still_rejected(no_env):
    cert = _cliff_certificate(ordinary_binning=False)
    report = mc.calibrate_mask(_cliff_seeded(_cliff_half, eligibility=cert))
    assert report["decision"] == mc.REJECTED
    assert report["scientific_eligibility"]["status"] == mc.ELIGIBILITY_NOT_ORDINARY


def test_seeded_random_with_explicit_ordinary_false_is_permitted(no_env):
    ev = _random_cliff(cert=_random_cliff_certificate(ordinary_binning=False))
    report = mc.calibrate_mask(ev)
    assert report["decision"] == mc.ACCEPTED, report["messages"]
    assert report["scientific_eligibility"]["status"] == mc.ELIGIBILITY_ELIGIBLE


@pytest.mark.parametrize("value", [True, 1, "false", None])
def test_seeded_random_contradicting_ordinary_flag_is_invalid(no_env, value):
    ev = _random_cliff(cert=_random_cliff_certificate(ordinary_binning=value))
    _invalid_eligibility(ev)


@pytest.mark.parametrize(
    "kind",
    [
        "ordinary",
        "seeded-random",
        "SEEDED_RANDOM",
        "",
        None,
        1,
        True,
        ["seeded_random"],
    ],
)
def test_unknown_or_mutated_binning_kind_is_invalid(no_env, kind):
    ev = _random_cliff(cert=_random_cliff_certificate(binning_kind=kind))
    _invalid_eligibility(ev)


def _splice(**changes) -> Dict:
    binning = _static_binning()
    for key, value in changes.items():
        if value is _DROP:
            binning.pop(key)
        else:
            binning[key] = value
    return binning


_DROP = object()
_SPLICES = {
    "other_count_id": dict(restriction_id=OTHER_RID),
    "other_env_id": dict(restriction_id=SLIPPERY_RID),
    "unregistered_id": dict(restriction_id="cliffwalking_random_bins21_seed20261003"),
    "id_not_str": dict(restriction_id=[RANDOM_RID]),
    "env": dict(env_name="Taxi-v3"),
    "env_not_str": dict(env_name=None),
    "seed": dict(partition_seed=20261004),
    "seed_str": dict(partition_seed="20261003"),
    "seed_float": dict(partition_seed=20261003.0),
    "k": dict(n_bins=16),
    "k_float": dict(n_bins=20.0),
    "k_bool": dict(n_bins=True),
    "descriptor": dict(partition_sha256=tasks.random_partition_sha256(OTHER_RID)),
    "descriptor_malformed": dict(partition_sha256="A" * 64),
    "table": dict(frozen_table_sha256=tasks.FROZEN_TABLE_SHA256[OTHER_RID]),
    "restriction": dict(restriction_sha256="c" * 64),
    "generator": dict(generator="random_state_binning/2"),
    "kind": dict(kind="ordinary"),
    "extra": dict(note="trusted"),
    "missing_descriptor": dict(partition_sha256=_DROP),
    "missing_seed": dict(partition_seed=_DROP),
    "missing_k": dict(n_bins=_DROP),
    "missing_restriction": dict(restriction_sha256=_DROP),
}


@pytest.mark.parametrize("name", sorted(_SPLICES))
def test_spliced_binning_metadata_fails_closed(no_env, name):
    # Cert and evidence carry the same spliced binding, so only the static
    # registry and the evidence mask can catch it.
    binning = _splice(**_SPLICES[name])
    cert = _random_cliff_certificate(binning=copy.deepcopy(binning))
    _invalid_eligibility(_random_cliff(cert=cert, binning=binning))


def test_certificate_binding_must_equal_the_evidence_binding(no_env):
    ev = _random_cliff()
    ev["eligibility"]["binning"] = _static_binning(OTHER_RID)
    _invalid_eligibility(ev)


def test_evidence_mask_rid_must_be_the_bound_restriction(no_env):
    ev = _random_cliff()
    ev["mask"] = OTHER_RID
    _invalid_eligibility(ev)


def test_random_evidence_without_binding_is_invalid(no_env):
    ev = _random_cliff()
    del ev["binning"]
    report = _assert_not_accepted(ev, mc.INVALID_EVIDENCE)
    assert any("binning" in m for m in report["messages"])


@pytest.mark.parametrize("bad", [["seeded_random"], "seeded_random", 1])
def test_non_mapping_evidence_binding_is_invalid(no_env, bad):
    ev = _random_cliff(binning=bad)
    _assert_not_accepted(ev, mc.INVALID_EVIDENCE)


def test_legacy_certificate_spliced_onto_random_evidence_is_invalid(no_env):
    ev = _random_cliff(cert=_cliff_certificate())
    _invalid_eligibility(ev)


def test_random_certificate_spliced_onto_ordinary_evidence_is_invalid(no_env):
    ev = _cliff_seeded(_cliff_half, eligibility=_random_cliff_certificate())
    _invalid_eligibility(ev)


def test_binding_on_ordinary_evidence_is_invalid(no_env):
    ev = _accepted()
    ev["binning"] = _static_binning()
    _assert_not_accepted(ev, mc.INVALID_EVIDENCE)


def test_random_certificate_mask_hash_must_match_restriction_hash(no_env):
    ev = _random_cliff()
    ev["eligibility"]["mask_sha256"] = "c" * 64
    _invalid_eligibility(ev)


@pytest.mark.parametrize(
    "overrides, pop",
    [
        ({"verified": False}, None),
        ({"policy_class": "stochastic_memoryless"}, None),
        ({}, "goal_feasible"),
    ],
)
def test_random_cliff_keeps_class_and_feasibility_requirements(no_env, overrides, pop):
    cert = _random_cliff_certificate(**overrides)
    if pop:
        cert.pop(pop)
    report = _assert_not_accepted(_random_cliff(cert=cert), mc.ELIGIBILITY_UNVERIFIED)
    assert report["scientific_eligibility"]["status"] == mc.ELIGIBILITY_UNVERIFIED


def test_random_cliff_goal_infeasible_is_rejected(no_env):
    cert = _random_cliff_certificate(best_in_class=-200.0, goal_feasible=False)
    report = mc.calibrate_mask(_random_cliff(_cliff_stall, cert=cert))
    assert report["decision"] == mc.REJECTED
    assert report["scientific_eligibility"]["status"] == mc.ELIGIBILITY_GOAL_INFEASIBLE


@pytest.mark.parametrize("kind", [_DROP, "uniform_random_policy", None, 3, ""])
def test_random_certificate_needs_a_known_lower_benchmark_label(no_env, kind):
    cert = _random_cliff_certificate()
    if kind is _DROP:
        cert.pop("lower_reference_kind")
    else:
        cert["lower_reference_kind"] = kind
    _invalid_eligibility(_random_cliff(cert=cert))


def test_lower_benchmark_label_on_legacy_certificate_is_checked(no_env):
    cert = _cliff_certificate(lower_reference_kind=mc.LOWER_STALL_BENCHMARK)
    report = mc.calibrate_mask(_cliff_seeded(_cliff_half, eligibility=cert))
    assert report["decision"] == mc.ACCEPTED, report["messages"]
    assert report["scientific_eligibility"]["lower_benchmark"]["kind"] == (
        mc.LOWER_STALL_BENCHMARK
    )
    cert["lower_reference_kind"] = "normalization"
    _invalid_eligibility(_cliff_seeded(_cliff_half, eligibility=cert))


def test_public_binding_helper_matches_the_static_registry(no_env):
    assert mc.seeded_random_binning(RANDOM_RID, HASH) == _static_binning()
    for rid in ("cliffwalking_row_pairs", "nope", None, 3):
        with pytest.raises(ValueError):
            mc.seeded_random_binning(rid, HASH)
    with pytest.raises(ValueError):
        mc.seeded_random_binning(RANDOM_RID, "not-a-hash")


# ---------------------------------------------------------------------------
# Random-offset grid binding (protocol /6)
# ---------------------------------------------------------------------------

GRID_ENV = "CartPole-v1"
GRID_RID = tasks.random_grid_id(GRID_ENV, 32, tasks.RANDOM_GRID_SEED)
OTHER_GRID_RID = tasks.random_grid_id(GRID_ENV, 16, tasks.RANDOM_GRID_SEED)
ACRO_GRID_RID = tasks.random_grid_id("Acrobot-v1", 32, tasks.RANDOM_GRID_SEED)
# Digest of the restriction record the runs carry, computed independently.
GRID_SHA = pilot.config_digest(restriction.describe_restriction(GRID_RID))


def test_frozen_grid_binding_constants():
    assert mc.SEEDED_RANDOM_GRID == "seeded_random_grid"
    assert mc.BINNING_KINDS == ("seeded_random",)  # no grid certificate kind
    assert mc.GRID_BINNING_FIELDS == (
        "kind",
        "generator",
        "restriction_id",
        "env_name",
        "seed",
        "n_bins",
        "grid_descriptor_sha256",
        "restriction_sha256",
    )
    assert len(tasks.RANDOM_GRID_SPECS) == 32
    assert tasks.RANDOM_GRID_COUNTS == (2, 4, 8, 16, 32, 64, 128, 256)


def _static_grid_binning(rid: str = GRID_RID, restriction_sha: str = GRID_SHA) -> Dict:
    """Grid binding built independently from the static registries."""
    spec = tasks.RANDOM_GRID_SPECS[rid]
    return {
        "kind": "seeded_random_grid",
        "generator": spec.generator,
        "restriction_id": rid,
        "env_name": spec.env_name,
        "seed": spec.seed,
        "n_bins": spec.n_bins,
        "grid_descriptor_sha256": random_grid.FROZEN_GRID_SHA256[rid],
        "restriction_sha256": restriction_sha,
    }


def _grid(masked_fn=_cart_masked_partial, binning=None, sha: str = GRID_SHA) -> Dict:
    ev = _two_arm(GRID_ENV, _cart_full_500, masked_fn, expert=500.0, random=22.0)
    ev.update(mask=GRID_RID, mask_sha256=sha)
    for row in ev["rows"]:
        if row["arm"] == "masked":
            row["mask_sha256"] = sha
    ev["binning"] = (
        _static_grid_binning(restriction_sha=sha) if binning is None else binning
    )
    return ev


def test_grid_with_static_binding_is_accepted_with_unknown_eligibility(no_env):
    ev = _grid()
    snapshot = copy.deepcopy(ev)
    report = mc.calibrate_mask(ev)
    assert report["protocol"] == "agnostic-mask-calibration/6"
    assert report["decision"] == mc.ACCEPTED, report["messages"]
    assert report["binning"] == _static_grid_binning()
    eligibility = report["scientific_eligibility"]
    # Empirical screen only: no ceiling, no certificate, never a known class gap.
    assert eligibility["status"] == mc.ELIGIBILITY_UNKNOWN
    assert eligibility["required"] is False
    assert eligibility["certificate"] is None
    assert eligibility["class_gap"] is None
    assert ev == snapshot


def test_grid_binding_needs_no_environment(no_env, monkeypatch):
    monkeypatch.setattr(random_grid, "describe_box", _refuse_env)
    assert mc.seeded_random_grid_binning(GRID_RID, GRID_SHA) == _static_grid_binning()


def test_grid_constants_are_unchanged_under_the_binding(no_env):
    masked = lambda m: ([10.0] * 100, [False] * 100, [False] * 100)  # noqa: E731
    report = mc.calibrate_mask(_grid(masked))
    assert report["decision"] == mc.REJECTED
    assert mc.NO_BETTER_THAN_RANDOM in report["reasons"]
    assert report["constants"]["min_norm_gap"] == 0.10
    assert report["constants"]["significance_se"] == 2.0
    assert report["constants"]["near_optimal"] == 0.95
    assert report["constants"]["min_episodes"] == 100


def test_grid_evidence_without_binding_is_invalid(no_env):
    ev = _grid()
    del ev["binning"]
    report = _assert_not_accepted(ev, mc.INVALID_EVIDENCE)
    assert any("binning" in m for m in report["messages"])


def _grid_splice(**changes) -> Dict:
    binning = _static_grid_binning()
    for key, value in changes.items():
        if value is _DROP:
            binning.pop(key)
        else:
            binning[key] = value
    return binning


_GRID_SPLICES = {
    "other_count_id": dict(restriction_id=OTHER_GRID_RID),
    "other_env_id": dict(restriction_id=ACRO_GRID_RID),
    "unregistered_id": dict(restriction_id="cartpole_random_grid3_seed20261003"),
    "tabular_id": dict(restriction_id=RANDOM_RID),
    "id_not_str": dict(restriction_id=[GRID_RID]),
    "env": dict(env_name="Acrobot-v1"),
    "env_not_str": dict(env_name=None),
    "seed": dict(seed=20261004),
    "seed_str": dict(seed="20261003"),
    "seed_float": dict(seed=20261003.0),
    "seed_bool": dict(seed=True),
    "k": dict(n_bins=16),
    "k_axis_cells": dict(n_bins=33),
    "k_float": dict(n_bins=32.0),
    "k_bool": dict(n_bins=True),
    "descriptor": dict(
        grid_descriptor_sha256=random_grid.FROZEN_GRID_SHA256[OTHER_GRID_RID],
    ),
    "descriptor_malformed": dict(grid_descriptor_sha256="A" * 64),
    "restriction": dict(restriction_sha256="c" * 64),
    "restriction_other_k": dict(
        restriction_sha256=pilot.config_digest(
            restriction.describe_restriction(OTHER_GRID_RID),
        ),
    ),
    "generator": dict(generator="random_offset_grid/2"),
    "tabular_kind": dict(kind="seeded_random"),
    "kind_other": dict(kind="seeded_random_grid/2"),
    "tabular_field": dict(partition_seed=20261003),
    "extra": dict(note="trusted"),
    "missing_descriptor": dict(grid_descriptor_sha256=_DROP),
    "missing_seed": dict(seed=_DROP),
    "missing_k": dict(n_bins=_DROP),
    "missing_restriction": dict(restriction_sha256=_DROP),
    "missing_generator": dict(generator=_DROP),
}


@pytest.mark.parametrize("name", sorted(_GRID_SPLICES))
def test_spliced_grid_binding_fails_closed(no_env, name):
    report = _assert_not_accepted(
        _grid(binning=_grid_splice(**_GRID_SPLICES[name])),
        mc.INVALID_EVIDENCE,
    )
    assert report["per_method"] == {}


def test_grid_restriction_digest_must_be_the_rebuilt_record(no_env):
    # Mask, every masked row and the binding agree on a foreign digest; only
    # the rebuilt restriction record can catch it.
    report = _assert_not_accepted(_grid(sha="c" * 64), mc.INVALID_EVIDENCE)
    assert any("rebuilt restriction record" in m for m in report["messages"])


def test_grid_masked_row_digest_must_equal_the_mask(no_env):
    ev = _grid()
    ev["rows"][1]["mask_sha256"] = "c" * 64
    assert ev["rows"][1]["arm"] == "masked"
    _assert_not_accepted(ev, mc.INVALID_EVIDENCE)


def test_grid_evidence_mask_must_be_the_bound_grid(no_env):
    ev = _grid()
    ev["mask"] = OTHER_GRID_RID
    _assert_not_accepted(ev, mc.INVALID_EVIDENCE)
    ev = _grid()
    ev["env_name"] = "Acrobot-v1"
    _assert_not_accepted(ev, mc.INVALID_EVIDENCE)


def test_tampered_frozen_grid_descriptor_digest_fails_closed(no_env, monkeypatch):
    monkeypatch.setitem(random_grid.FROZEN_GRID_SHA256, GRID_RID, "d" * 64)
    report = _assert_not_accepted(
        _grid(binning=_grid_splice(grid_descriptor_sha256="d" * 64)),
        mc.INVALID_EVIDENCE,
    )
    assert any("frozen" in m for m in report["messages"])


@pytest.mark.parametrize("field", ["seed", "n_bins", "generator", "env_name"])
def test_tampered_grid_spec_fails_closed(no_env, monkeypatch, field):
    spec = tasks.RANDOM_GRID_SPECS[GRID_RID]
    changed = dict(
        seed=spec.seed + 1,
        n_bins=16,
        generator="random_offset_grid/2",
        env_name="Acrobot-v1",
    )[field]
    random_grid.get_grid(GRID_RID)  # cached before the splice
    monkeypatch.setitem(
        tasks.RANDOM_GRID_SPECS,
        GRID_RID,
        dataclasses.replace(spec, **{field: changed}),
    )
    # The binding agrees with the spliced registry; the frozen grid does not.
    binning = _static_grid_binning()
    assert binning[field] == changed
    _assert_not_accepted(_grid(binning=binning), mc.INVALID_EVIDENCE)


@pytest.mark.parametrize(
    "cert",
    [
        dict(_random_cliff_certificate(), mask_sha256=GRID_SHA),
        dict(_cliff_certificate(), mask_sha256=GRID_SHA, metric="return"),
        dict(
            _controller_cliff_certificate(mask_sha256=GRID_SHA, verified=False),
            binning_kind="seeded_random_grid",
            binning=_static_grid_binning(),
            lower_reference_kind="uniform_random_exact_value",
        ),
    ],
)
def test_any_certificate_on_a_grid_mask_is_invalid(no_env, cert):
    ev = _grid()
    ev["eligibility"] = cert
    report = _invalid_eligibility(ev)
    assert any("carries no certificate" in m for m in report["messages"])


def test_grid_binding_on_tabular_or_ordinary_masks_is_invalid(no_env):
    ev = _random_cliff(binning=_static_grid_binning())
    _assert_not_accepted(ev, mc.INVALID_EVIDENCE)
    ev = _accepted()
    ev["binning"] = _static_grid_binning()
    _assert_not_accepted(ev, mc.INVALID_EVIDENCE)


def test_tabular_binding_on_a_grid_mask_is_invalid(no_env):
    _assert_not_accepted(_grid(binning=_static_binning()), mc.INVALID_EVIDENCE)


def test_seeded_random_and_legacy_paths_are_unaffected_by_grids(no_env):
    report = mc.calibrate_mask(_random_cliff())
    assert report["decision"] == mc.ACCEPTED, report["messages"]
    assert report["binning"]["kind"] == "seeded_random"
    report = mc.calibrate_mask(_cliff_seeded(_cliff_half))
    assert report["decision"] == mc.ACCEPTED, report["messages"]
    assert (
        report["scientific_eligibility"]["binning_kind"] == mc.LEGACY_ORDINARY_BINNING
    )


def test_public_grid_binding_helper_matches_the_static_registry(no_env):
    for rid in tasks.RANDOM_GRID_SPECS:
        sha = pilot.config_digest(restriction.describe_restriction(rid))
        assert mc.seeded_random_grid_binning(rid, sha) == _static_grid_binning(rid, sha)
        assert mc.restriction_record_sha256(rid) == sha
    for rid in (RANDOM_RID, "cartpole_random_grid3_seed20261003", "nope", None, 3):
        with pytest.raises(ValueError):
            mc.seeded_random_grid_binning(rid, GRID_SHA)
    for sha in ("not-a-hash", "c" * 64, HASH):
        with pytest.raises(ValueError):
            mc.seeded_random_grid_binning(GRID_RID, sha)
