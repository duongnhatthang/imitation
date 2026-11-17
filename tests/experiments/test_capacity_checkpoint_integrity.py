"""Checkpoint integrity of the one-hidden capacity family.

Regressions for the independent review findings, every one on real SB3
policies and real ``policy.save`` files:

* C1: the constructor observation and action spaces must equal the spec's
  canonical description (type, shape, dtype, exact Box bounds, Discrete n and
  start), with the spec digest left unchanged by the mutation.
* C2: the writer binds the exact bytes it wrote; a consumer refuses any
  checkpoint whose bytes or state differ from that binding, including a valid
  finite actor substituted after the writer emitted it.
* C3: every state tensor must match the constructed policy's key, shape and
  dtype and every floating tensor must be finite before anything is loaded.
"""

import copy
import hashlib
import io
import pathlib

import numpy as np
import pytest
import torch as th
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy

from imitation.experiments.agnostic import capacity_policy as cp
from imitation.experiments.agnostic import restriction_pilot as rp
from imitation.experiments.ftrl import policy_utils, run_experiment

from .test_capacity_policy import OBS, _image_teacher, _images, _step, _vector_teacher

LOADERS = {
    "generic": policy_utils.load_policy_checkpoint,
    "policy_from_saved": lambda path: cp.policy_from_saved(
        th.load(path, weights_only=False),
    ),
}


def _student(image=False, width=4):
    teacher = _image_teacher() if image else _vector_teacher()
    student = cp.make_capacity_policy(
        teacher, hidden_width=width, initialization_seed=4100
    )
    _step(student, obs=_images(3) if image else OBS)
    return student


def _write(tmp_path, saved, name="mutated.pt"):
    path = tmp_path / name
    th.save(saved, path)
    return path


def _sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def _trusted(path, binding, **overrides):
    kwargs = dict(
        expected_spec_sha256=binding["spec_sha256"],
        expected_file_sha256=binding["file_sha256"],
        expected_state_sha256=binding["state_sha256"],
    )
    kwargs.update(overrides)
    return cp.load_checkpoint(path, **kwargs)


# ---------------------------------------------------------------------------
# C1: observation and action interface
# ---------------------------------------------------------------------------


def _image_low_element(data):
    space = data["observation_space"]
    low = space.low.copy()
    low[0, 8, 8] = 1
    data["observation_space"] = spaces.Box(low, space.high, dtype=space.dtype)


INTERFACE_MUTATIONS = {
    "discrete_observations": lambda d: d.update(observation_space=spaces.Discrete(4)),
    "reshaped_observations": lambda d: d.update(
        observation_space=spaces.Box(-1, 1, (2, 2), dtype=np.float32),
    ),
    "changed_box_bounds": lambda d: d.update(
        observation_space=spaces.Box(-100, 100, (4,), dtype=np.float32),
    ),
    "one_changed_bound": lambda d: d.update(
        observation_space=spaces.Box(
            np.array([-1, -1, -1, -2], dtype=np.float32),
            np.ones(4, dtype=np.float32),
            dtype=np.float32,
        ),
    ),
    "float64_observations": lambda d: d.update(
        observation_space=spaces.Box(-1, 1, (4,), dtype=np.float64),
    ),
    "multibinary_observations": lambda d: d.update(
        observation_space=spaces.MultiBinary(4),
    ),
    "changed_action_start": lambda d: d.update(
        action_space=spaces.Discrete(2, start=10),
    ),
    "changed_action_count": lambda d: d.update(action_space=spaces.Discrete(3)),
}


@pytest.mark.parametrize("loader", sorted(LOADERS))
@pytest.mark.parametrize("name", sorted(INTERFACE_MUTATIONS))
def test_changed_constructor_interface_refuses_with_unchanged_spec(
    tmp_path,
    loader,
    name,
):
    student = _student()
    student.save(tmp_path / "student.pt")
    saved = th.load(tmp_path / "student.pt", weights_only=False)
    INTERFACE_MUTATIONS[name](saved["data"])
    if name == "changed_action_count":
        # Keep the state loadable, so only the interface check can refuse.
        saved["state_dict"]["action_net.weight"] = th.zeros(3, 4)
        saved["state_dict"]["action_net.bias"] = th.zeros(3)
    assert saved["data"]["capacity_spec_sha256"] == student.capacity_spec_sha256
    assert cp.spec_sha256(saved["data"]["capacity_spec"]) == (
        student.capacity_spec_sha256
    )
    path = _write(tmp_path, saved)
    with pytest.raises(ValueError):
        LOADERS[loader](path)


def test_one_changed_image_bound_refuses(tmp_path):
    student = _student(image=True)
    student.save(tmp_path / "student.pt")
    saved = th.load(tmp_path / "student.pt", weights_only=False)
    _image_low_element(saved["data"])
    path = _write(tmp_path, saved)
    with pytest.raises(ValueError, match="observation_space"):
        policy_utils.load_policy_checkpoint(path)


def test_interface_refusal_does_not_depend_on_the_file_digest(tmp_path):
    """A writer binding of mutated bytes still cannot pass the interface check."""
    student = _student()
    binding = student.save_bound(tmp_path / "student.pt")
    saved = th.load(tmp_path / "student.pt", weights_only=False)
    INTERFACE_MUTATIONS["changed_box_bounds"](saved["data"])
    path = _write(tmp_path, saved)
    with pytest.raises(ValueError, match="observation_space"):
        _trusted(path, binding, expected_file_sha256=_sha(path))


def test_spec_binds_exact_observation_and_action_semantics():
    teacher = _vector_teacher()
    student = cp.make_capacity_policy(teacher, hidden_width=8, initialization_seed=1)
    observation = student.capacity_spec["encoder"]["observation_space"]
    space = teacher.observation_space
    assert observation["type"] == "gymnasium.spaces.box.Box"
    assert observation["shape"] == [4] and observation["dtype"] == "float32"
    for bound in ("low", "high"):
        digest = hashlib.sha256()
        array = np.ascontiguousarray(getattr(space, bound))
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(str(tuple(array.shape)).encode("utf-8"))
        digest.update(array.tobytes())
        assert observation[f"{bound}_sha256"] == digest.hexdigest()
    action = student.capacity_spec["teacher"]["action_space"]
    assert action == {
        "type": "gymnasium.spaces.discrete.Discrete",
        "n": 2,
        "start": 0,
        "dtype": "int64",
    }


@pytest.mark.parametrize(
    "observation,action,needle",
    [
        (spaces.MultiBinary(4), spaces.Discrete(2), "observation"),
        (spaces.Discrete(4), spaces.Discrete(2), "observation"),
        (
            spaces.Box(-1, 1, (4,), dtype=np.float32),
            spaces.Discrete(2, start=1),
            "start",
        ),
    ],
)
def test_unsupported_teacher_interfaces_refuse(observation, action, needle):
    teacher = ActorCriticPolicy(
        observation,
        action,
        lr_schedule=lambda _: 1e-3,
        net_arch={"pi": [64], "vf": [64]},
    )
    with pytest.raises(ValueError, match=needle):
        cp.make_capacity_policy(teacher, hidden_width=4, initialization_seed=1)


# ---------------------------------------------------------------------------
# C2: writer binding of the trained checkpoint bytes
# ---------------------------------------------------------------------------


def test_writer_binds_the_exact_bytes_it_wrote(tmp_path):
    student = _student(image=True)
    path = tmp_path / "student.pt"
    binding = student.save_bound(path)
    assert binding == {
        "schema": cp.CHECKPOINT_BINDING_SCHEMA,
        "path": str(path),
        "file_sha256": _sha(path),
        "file_bytes": path.stat().st_size,
        "state_sha256": cp.state_sha256(student),
        "spec_sha256": student.capacity_spec_sha256,
    }
    restored = _trusted(path, binding)
    assert cp.state_sha256(restored) == binding["state_sha256"]
    images = _images(4, seed=9)
    student.eval()
    assert th.equal(
        student.get_distribution(images).distribution.logits,
        restored.get_distribution(images).distribution.logits,
    )
    # Plain save writes the same checkpoint format.
    student.save(tmp_path / "plain.pt")
    assert cp.state_sha256(policy_utils.load_policy_checkpoint(tmp_path / "plain.pt"))


def test_valid_actor_replaced_after_writer_emission_refuses(tmp_path):
    """The reviewer's actor swap: finite, float32, correct shape and spec."""
    student = _student()
    path = tmp_path / "student.pt"
    binding = student.save_bound(path)
    saved = th.load(path, weights_only=False)
    saved["state_dict"]["action_net.bias"] = th.tensor([100.0, -100.0])
    th.save(saved, path)
    swapped = policy_utils.load_policy_checkpoint(path)  # structurally valid
    assert th.isfinite(swapped.action_net.bias).all()
    assert swapped.capacity_spec_sha256 == binding["spec_sha256"]
    with pytest.raises(ValueError, match="writer"):
        _trusted(path, binding)
    with pytest.raises(ValueError, match="writer"):
        rp._verified_checkpoint(
            path,
            {"spec_sha256": binding["spec_sha256"]},
            binding,
            path,
        )


def test_state_digest_is_checked_after_the_file_digest(tmp_path):
    student = _student()
    path = tmp_path / "student.pt"
    binding = student.save_bound(path)
    with pytest.raises(ValueError, match="state"):
        _trusted(path, binding, expected_state_sha256="0" * 64)
    with pytest.raises(ValueError, match="spec"):
        _trusted(path, binding, expected_spec_sha256="0" * 64)


@pytest.mark.parametrize("value", [None, "", "0" * 63, "G" * 64, 7])
def test_trusted_loader_requires_a_well_formed_writer_digest(tmp_path, value):
    student = _student()
    path = tmp_path / "student.pt"
    binding = student.save_bound(path)
    with pytest.raises(ValueError, match="digest"):
        _trusted(path, binding, expected_file_sha256=value)
    with pytest.raises(TypeError):
        cp.load_checkpoint(path, expected_spec_sha256=binding["spec_sha256"])


def test_trusted_loader_hashes_and_parses_one_blob(tmp_path):
    student = _student()
    blob = io.BytesIO()
    binding = student.save_bound(tmp_path / "student.pt")
    blob.write((tmp_path / "student.pt").read_bytes())
    blob.seek(0)
    restored = _trusted(blob, binding)
    assert cp.state_sha256(restored) == binding["state_sha256"]


def _consumer_case(tmp_path, change):
    student = _student()
    path = tmp_path / "round-00003.pt"
    binding = student.save_bound(path)
    capacity = {"spec_sha256": student.capacity_spec_sha256}
    expected = path
    path, capacity, binding, expected = change(path, capacity, dict(binding), expected)
    return path, capacity, binding, expected


def _copy_elsewhere(path, capacity, binding, expected):
    other = path.parent / "round-00006.pt"
    other.write_bytes(path.read_bytes())
    return other, capacity, binding, other


CONSUMER_REFUSALS = {
    "missing_binding": lambda p, c, b, e: (p, c, None, e),
    "unknown_schema": lambda p, c, b, e: (p, c, {**b, "schema": "other/1"}, e),
    "other_spec": lambda p, c, b, e: (p, {"spec_sha256": "0" * 64}, b, e),
    "binding_other_spec": lambda p, c, b, e: (p, c, {**b, "spec_sha256": "1" * 64}, e),
    "binding_other_path": lambda p, c, b, e: (p, c, {**b, "path": str(p) + "x"}, e),
    "unexpected_round_path": lambda p, c, b, e: (p, c, b, p.parent / "round-1.pt"),
    "bytes_at_another_path": _copy_elsewhere,
    "missing_file_digest": lambda p, c, b, e: (
        p,
        c,
        {k: v for k, v in b.items() if k != "file_sha256"},
        e,
    ),
    "missing_state_digest": lambda p, c, b, e: (
        p,
        c,
        {k: v for k, v in b.items() if k != "state_sha256"},
        e,
    ),
}


@pytest.mark.parametrize("name", sorted(CONSUMER_REFUSALS))
def test_runner_consumer_refuses_unbound_or_misassociated_checkpoints(tmp_path, name):
    path, capacity, binding, expected = _consumer_case(
        tmp_path,
        CONSUMER_REFUSALS[name],
    )
    with pytest.raises(ValueError):
        rp._verified_checkpoint(path, capacity, binding, expected)


def test_runner_consumer_accepts_the_writer_binding(tmp_path):
    path, capacity, binding, expected = _consumer_case(
        tmp_path,
        lambda p, c, b, e: (p, c, b, e),
    )
    fields = rp._verified_checkpoint(path, capacity, binding, expected)
    assert fields == {
        "checkpoint_capacity_spec_sha256": capacity["spec_sha256"],
        "checkpoint_binding": binding,
    }
    assert fields["checkpoint_binding"] is not binding


def test_run_experiment_records_bindings_for_capacity_and_nothing_for_legacy(
    tmp_path,
):
    config = rp._experiment_config("ftl", 3, 6, 3, tmp_path, "CartPole-v1")
    legacy = policy_utils.create_linear_policy(_vector_teacher(hidden=(64, 64)))
    fields = run_experiment._checkpoint_fields(config, legacy, 2)
    assert fields == {"checkpoint": run_experiment._save_policy(config, legacy, 2)}
    assert fields["checkpoint"].endswith("_CartPole-v1_seed3/round-00002.pt")
    student = _student()
    fields = run_experiment._checkpoint_fields(config, student, 2)
    assert sorted(fields) == ["checkpoint", "checkpoint_binding"]
    assert fields["checkpoint_binding"]["path"] == fields["checkpoint"]
    assert fields["checkpoint_binding"]["file_sha256"] == _sha(fields["checkpoint"])
    assert fields["checkpoint"] == str(run_experiment._policy_path(config, 2))


# ---------------------------------------------------------------------------
# C3: tensor keys, shapes, dtypes and finiteness before loading
# ---------------------------------------------------------------------------


def _fill(key, value):
    def change(state):
        state[key] = th.full_like(state[key], value)

    return change


def _cast(key, dtype):
    def change(state):
        state[key] = state[key].to(dtype)

    return change


STATE_MUTATIONS = {
    "nan_actor_weight": _fill("action_net.weight", float("nan")),
    "inf_actor_bias": _fill("action_net.bias", float("inf")),
    "neg_inf_hidden_weight": _fill("mlp_extractor.policy_net.0.weight", float("-inf")),
    "nan_hidden_bias": _fill("mlp_extractor.policy_net.0.bias", float("nan")),
    "int64_actor_weight": _cast("action_net.weight", th.int64),
    "float64_actor_weight": _cast("action_net.weight", th.float64),
    "float16_hidden_bias": _cast("mlp_extractor.policy_net.0.bias", th.float16),
    "float_batch_counter": _cast(
        "features_extractor.cnn.1.num_batches_tracked",
        th.float32,
    ),
    "reshaped_actor_bias": lambda s: s.update(
        {"action_net.bias": s["action_net.bias"].reshape(1, -1)},
    ),
    "missing_actor_key": lambda s: s.pop("action_net.bias"),
    "extra_key": lambda s: s.update({"action_net.extra": th.zeros(1)}),
    "non_tensor": lambda s: s.update({"action_net.bias": [0.0, 0.0, 0.0]}),
}


@pytest.mark.parametrize("loader", sorted(LOADERS))
@pytest.mark.parametrize("name", sorted(STATE_MUTATIONS))
def test_malformed_state_tensors_refuse_before_loading(
    tmp_path,
    monkeypatch,
    loader,
    name,
):
    student = _student(image=True)
    student.save(tmp_path / "student.pt")
    saved = th.load(tmp_path / "student.pt", weights_only=False)
    STATE_MUTATIONS[name](saved["state_dict"])
    path = _write(tmp_path, saved)
    calls = []
    original = cp.CapacityActorCriticPolicy.load_state_dict

    def spy(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(cp.CapacityActorCriticPolicy, "load_state_dict", spy)
    with pytest.raises(ValueError):
        LOADERS[loader](path)
    assert not calls


def test_writer_bound_nan_actor_still_refuses(tmp_path):
    """A writer digest of malformed bytes is not a validity certificate."""
    student = _student()
    with th.no_grad():
        student.action_net.weight.fill_(float("nan"))
    path = tmp_path / "student.pt"
    binding = student.save_bound(path)
    with pytest.raises(ValueError, match="finite"):
        _trusted(path, binding)


def test_valid_roundtrip_keeps_integer_buffers_and_exact_dtypes(tmp_path):
    student = _student(image=True)
    path = tmp_path / "student.pt"
    binding = student.save_bound(path)
    restored = _trusted(path, binding)
    state = restored.state_dict()
    counter = "features_extractor.cnn.1.num_batches_tracked"
    assert state[counter].dtype == th.int64
    assert {k: (v.dtype, tuple(v.shape)) for k, v in state.items()} == {
        k: (v.dtype, tuple(v.shape)) for k, v in student.state_dict().items()
    }
    assert all(th.equal(state[k], v) for k, v in student.state_dict().items())


def test_frozen_alias_mutation_still_refuses(tmp_path):
    student = _student(image=True)
    student.save(tmp_path / "student.pt")
    saved = th.load(tmp_path / "student.pt", weights_only=False)
    key = "pi_features_extractor.linear.0.weight"
    saved["state_dict"][key] = copy.deepcopy(saved["state_dict"][key]) + 1
    path = _write(tmp_path, saved)
    with pytest.raises(ValueError, match="frozen"):
        policy_utils.load_policy_checkpoint(path)
