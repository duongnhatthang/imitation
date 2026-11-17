"""Architecture-bound one-hidden-layer teacher preparation and loading.

No teacher is trained: preparation tests inject a trainer that saves an exact
hand-built PPO policy, and the trainer test replaces ``PPO.learn`` with a
no-op so only construction, convergence bookkeeping and saving run. Clocks
are explicit, so the October 7 hard cutoff of the legacy runners is not hit.
"""

import datetime
import json
import pathlib

import numpy as np
import pytest
import torch as th
from stable_baselines3 import PPO

from imitation.experiments.agnostic import classical, pilot, run_classical, task_experts
from imitation.experiments.ftrl import env_utils, expert_training

UTC = datetime.timezone.utc
FAR = datetime.datetime(2026, 10, 7, 1, 0, tzinfo=UTC)
PRIMARY = ("Acrobot-v1", "CartPole-v1", "MountainCar-v0", "LunarLander-v2")
ONE_HIDDEN = [64]


class Clock:
    """Deterministic clock before the legacy hard cutoff."""

    def __init__(self):
        self.now = datetime.datetime(2026, 10, 1, tzinfo=UTC)

    def __call__(self):
        self.now += datetime.timedelta(milliseconds=1)
        return self.now


def save_cart_expert(path, hidden=(64,)):
    """Exact balancing CartPole expert with the given actor hidden widths.

    The first hidden layer passes ``0.1 * obs`` through Tanh, later hidden
    layers pass the first four units on, and action 1 scores
    ``10 * [1, 1, 10, 10] . h``; deterministic returns are 500.
    """
    venv = env_utils.make_env("CartPole-v1", 1, np.random.default_rng(0))
    model = PPO(
        "MlpPolicy",
        venv,
        policy_kwargs=dict(net_arch=list(hidden)),
        device="cpu",
        seed=0,
    )
    with th.no_grad():
        for layer in model.policy.mlp_extractor.policy_net:
            if isinstance(layer, th.nn.Linear):
                layer.weight.zero_()
                layer.bias.zero_()
                for i in range(4):
                    layer.weight[i, i] = 0.1 if layer.in_features == 4 else 1.0
        model.policy.action_net.weight.zero_()
        model.policy.action_net.bias.zero_()
        model.policy.action_net.weight[1, :4] = th.tensor([10.0, 10.0, 100.0, 100.0])
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    venv.close()


def injected_trainer(hidden, seen=None):
    """Trainer replacement that records its call and saves an exact expert."""

    def trainer(env, cache_dir, rng, seed, convergence_override=None, **kwargs):
        if seen is not None:
            seen.append(kwargs)
        save_cart_expert(pathlib.Path(cache_dir) / env / "model.zip", hidden)

    return trainer


def legacy_trainer(hidden):
    """Trainer with the legacy signature: no architecture keyword."""

    def trainer(env, cache_dir, rng, seed, convergence_override=None):
        save_cart_expert(pathlib.Path(cache_dir) / env / "model.zip", hidden)

    return trainer


def prepare(out, trainer, actor_net_arch=None, env_name="CartPole-v1"):
    kwargs = {} if actor_net_arch is None else {"actor_net_arch": actor_net_arch}
    return run_classical.prepare_expert(
        env_name,
        out,
        seed=0,
        qualification_seed=1234,
        eval_episodes=2,
        deadline=FAR,
        trainer=trainer,
        clock=Clock(),
        **kwargs,
    )


def _record(prep):
    return json.loads((pathlib.Path(prep) / pilot.PREPARATION_FILE).read_text())


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("env_name", run_classical.PREPARATION_ENVS)
def test_default_preparation_config_is_unchanged(env_name):
    kwargs = dict(
        seed=0,
        qualification_seed=1234,
        eval_episodes=100,
        convergence_override=None,
    )
    default = run_classical.build_preparation_config(env_name, **kwargs)
    explicit_none = run_classical.build_preparation_config(
        env_name,
        actor_net_arch=None,
        **kwargs,
    )
    assert default == explicit_none
    assert pilot.canonical_json(default) == pilot.canonical_json(explicit_none)
    assert default["expert"]["net_arch"] == [64, 64]
    assert "architecture" not in default["expert"]
    assert sorted(default["expert"]) == [
        "algorithm",
        "device",
        "net_arch",
        "policy",
        "ppo_kwargs",
        "ppo_n_envs",
        "trainer",
        "training_rng",
    ]


@pytest.mark.parametrize("env_name", PRIMARY)
def test_one_hidden_config_binds_architecture_and_capacity_seeds(env_name):
    config = run_classical.build_preparation_config(
        env_name,
        seed=71000,
        qualification_seed=72000,
        eval_episodes=100,
        convergence_override=None,
        actor_net_arch=(64,),
    )
    expert = config["expert"]
    assert expert["net_arch"] == ONE_HIDDEN
    assert expert["architecture"] == {
        "schema": run_classical.ARCHITECTURE_SCHEMA,
        "pi": ONE_HIDDEN,
        "vf": ONE_HIDDEN,
        "activation_fn": "torch.nn.modules.activation.Tanh",
        "features_extractor": (
            "stable_baselines3.common.torch_layers.FlattenExtractor"
        ),
        "action_rule": "deterministic predict (argmax)",
    }
    legacy = run_classical.build_preparation_config(
        env_name,
        seed=71000,
        qualification_seed=72000,
        eval_episodes=100,
        convergence_override=None,
    )
    # Thresholds and convergence settings are the frozen defaults.
    for key in ("threshold", "convergence_config", "convergence_override"):
        assert config[key] == legacy[key]
    assert pilot.config_digest(config) != pilot.config_digest(legacy)


@pytest.mark.parametrize("arch", [[64, 64], [32], [], [64, 0], [True], [64.0]])
def test_only_the_declared_one_hidden_architecture_is_accepted(arch):
    with pytest.raises(ValueError, match="actor_net_arch"):
        run_classical.build_preparation_config(
            "CartPole-v1",
            seed=0,
            qualification_seed=1234,
            eval_episodes=2,
            convergence_override=None,
            actor_net_arch=arch,
        )


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("net_arch,expected", [(None, [64, 64]), ([64], [64])])
def test_trainer_builds_the_requested_architecture_without_learning(
    tmp_path,
    monkeypatch,
    net_arch,
    expected,
):
    calls = []
    monkeypatch.setattr(PPO, "learn", lambda self, *a, **k: calls.append(a))
    convergence = dict(
        chunk_timesteps=1,
        max_timesteps=10,
        min_timesteps=1,
        threshold=-1e9,
        patience=1,
        self_ce_eps=1e9,
    )
    kwargs = {} if net_arch is None else {"net_arch": net_arch}
    policy = expert_training.train_classical_expert_until_converged(
        "CartPole-v1",
        tmp_path,
        np.random.default_rng(0),
        0,
        convergence_override=convergence,
        **kwargs,
    )
    assert calls  # every chunk went through the replaced learn
    saved = PPO.load(tmp_path / "CartPole-v1" / "model.zip", device="cpu")
    for model_policy in (policy, saved.policy):
        architecture = expert_training.policy_architecture(model_policy)
        assert architecture["pi"] == expected and architecture["vf"] == expected
        assert architecture["activation_fn"] == "torch.nn.modules.activation.Tanh"


# ---------------------------------------------------------------------------
# Preparation and loading
# ---------------------------------------------------------------------------


def test_one_hidden_preparation_records_and_loads_verified_architecture(tmp_path):
    seen = []
    prep = tmp_path / "prep"
    assert prepare(prep, injected_trainer((64,), seen), (64,)) == 0
    assert seen == [{"net_arch": ONE_HIDDEN}]
    record = _record(prep)
    assert record["status"] == "complete" and record["qualified"] is True
    assert record["config"]["expert"]["architecture"]["pi"] == ONE_HIDDEN
    # Top-level, so a campaign manifest can pin it with expected_fields.
    assert record["actor_net_arch"] == ONE_HIDDEN
    actual = record["checkpoint"]["architecture"]
    assert actual["pi"] == ONE_HIDDEN and actual["vf"] == ONE_HIDDEN
    assert actual["features_dim"] == 4 and actual["n_actions"] == 2
    model, identity = task_experts.load_qualified_expert(prep, "CartPole-v1")
    assert identity["architecture"]["claimed"] == (
        record["config"]["expert"]["architecture"]
    )
    assert identity["architecture"]["actual"] == actual
    assert expert_training.policy_architecture(model.policy) == actual


def test_architecture_mismatch_after_training_fails_the_preparation(tmp_path):
    prep = tmp_path / "prep"
    code = prepare(prep, injected_trainer((64, 64)), (64,))
    record = _record(prep)
    assert code == run_classical.EXIT_NOT_QUALIFIED
    assert record["status"] == "failed" and record["qualified"] is False
    assert "architecture" in record["error"]["message"]
    with pytest.raises(classical.Refused):
        task_experts.load_qualified_expert(prep, "CartPole-v1")


def test_swapped_checkpoint_with_consistent_hashes_refuses_on_architecture(
    tmp_path,
):
    prep = tmp_path / "prep"
    assert prepare(prep, injected_trainer((64,)), (64,)) == 0
    record = _record(prep)
    checkpoint = prep / record["checkpoint"]["path"]
    save_cart_expert(checkpoint, hidden=(64, 64))
    model = PPO.load(checkpoint, device="cpu")
    digest = pilot.sha256_file(checkpoint)
    record["checkpoint"].update(
        sha256=digest,
        policy_state_sha256=pilot.policy_state_sha256(model.policy),
    )
    record["expert_sha256"] = digest
    (prep / pilot.PREPARATION_FILE).write_text(json.dumps(record))
    with pytest.raises(classical.Refused, match="architecture"):
        task_experts.load_qualified_expert(prep, "CartPole-v1")


def test_tampered_architecture_claim_refuses(tmp_path):
    prep = tmp_path / "prep"
    assert prepare(prep, injected_trainer((64,)), (64,)) == 0
    record = _record(prep)
    record["config"]["expert"]["architecture"]["activation_fn"] = "torch.nn.ReLU"
    record["config_sha256"] = pilot.config_digest(record["config"])
    (prep / pilot.PREPARATION_FILE).write_text(json.dumps(record))
    with pytest.raises(classical.Refused):
        task_experts.load_qualified_expert(prep, "CartPole-v1")


def test_legacy_preparations_keep_their_bytes_and_loading(tmp_path):
    prep = tmp_path / "legacy"
    assert prepare(prep, legacy_trainer((64, 64))) == 0
    record = _record(prep)
    assert "architecture" not in record["config"]["expert"]
    assert "architecture" not in record["checkpoint"]
    assert "actor_net_arch" not in record
    assert record["config"] == run_classical.build_preparation_config(
        "CartPole-v1",
        seed=0,
        qualification_seed=1234,
        eval_episodes=2,
        convergence_override=None,
    )
    _, identity = task_experts.load_qualified_expert(prep, "CartPole-v1")
    assert "architecture" not in identity


def test_cli_forwards_the_explicit_architecture(tmp_path):
    seen = []
    out = tmp_path / "cli"
    code = run_classical.main(
        [
            "prepare-expert",
            "--env",
            "CartPole-v1",
            "--output-dir",
            str(out),
            "--seed",
            "0",
            "--qualification-seed",
            "1234",
            "--eval-episodes",
            "2",
            "--deadline",
            "2099-01-01T00:00:00Z",
            "--actor-net-arch",
            "64",
        ],
        trainer=injected_trainer((64,), seen),
    )
    assert code == 0 and seen == [{"net_arch": ONE_HIDDEN}]
    assert _record(out)["config"]["expert"]["net_arch"] == ONE_HIDDEN
    assert (
        run_classical._parse_args(
            [
                "prepare-expert",
                "--env",
                "CartPole-v1",
                "--output-dir",
                str(tmp_path / "x"),
                "--seed",
                "0",
                "--qualification-seed",
                "1",
                "--deadline",
                "2099-01-01T00:00:00Z",
            ],
        ).actor_net_arch
        is None
    )
    bad = run_classical.main(
        [
            "prepare-expert",
            "--env",
            "CartPole-v1",
            "--output-dir",
            str(tmp_path / "bad"),
            "--seed",
            "0",
            "--qualification-seed",
            "1234",
            "--deadline",
            "2099-01-01T00:00:00Z",
            "--actor-net-arch",
            "64",
            "64",
        ],
        trainer=injected_trainer((64, 64)),
    )
    assert bad == run_classical.EXIT_USAGE
    assert not (tmp_path / "bad").exists()
