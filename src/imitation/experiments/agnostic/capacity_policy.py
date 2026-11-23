"""One-hidden-layer capacity students on a shared frozen teacher encoder.

The approved capacity study compares a qualified teacher
``F(x) -> hidden 64 -> categorical logits`` with students
``F(x) -> hidden w -> categorical logits`` for ``w`` in `STUDENT_WIDTHS`, all
strictly narrower than the teacher. ``F`` is the teacher's own features
extractor (``Flatten`` for the vector tasks, the qualified CNN for images),
copied byte for byte, frozen and kept in eval mode. Nothing of the teacher's
actor (hidden layer or logits) enters the student, and no teacher optimizer
state exists to transfer.

Students (`CapacityActorCriticPolicy`, built by `make_capacity_policy`):

* The actor hidden and output layers are fresh, trainable and the only
  parameters an optimizer may own (`bc_optimizer_parameters`, used by
  ``bc.BC``). The critic is kept for SB3 compatibility, frozen and excluded.
* Every draw of torch randomness happens on a forked CPU generator seeded from
  a named stream of ``initialization_seed``
  (``SeedSequence(seed, spawn_key=(CAPACITY_SPAWN_KEY, purpose, draw))``), so
  building a student of any width, or reinitializing one, leaves every global
  RNG exactly as it was. Draw 0 initializes the actor at construction; cold
  refits (`cold_refit_reinitialize`, used by ``ftrl.FTRLTrainer``) take draws
  1, 2, ... with SB3's default actor initialization.
* A serialized spec (`POLICY_SCHEMA`) binds the family, widths, teacher
  architecture and state digest, the exact observation and action spaces
  (type, shape, dtype, Box bound bytes, Discrete n and start), encoder
  class/configuration/digest, actual parameter counts and initialization
  rule. Its digest, the frozen state, the structure and every state tensor's
  key, shape, dtype and finiteness are checked on construction and on every
  checkpoint load (`policy_utils.load_policy_checkpoint` dispatches here).
* The spec is constant over training, so it cannot identify a trained
  checkpoint. `CapacityActorCriticPolicy.save_bound` writes a checkpoint and
  returns the writer's binding of the exact bytes it wrote and of the full
  state it serialized (`CHECKPOINT_BINDING_SCHEMA`); `load_checkpoint` refuses
  any file whose bytes or state differ from that binding or whose spec
  differs from the recorded one.

Width ratios are not parameter ratios: the trainable decision head has
``d*w + w*a + w + a`` parameters for ``d`` encoder features and ``a`` actions;
both are recorded. A narrower head is a capacity stress conditional on ``F``,
not a certificate that the teacher is outside the student class.
"""

import contextlib
import copy
import dataclasses
import hashlib
import io
import json
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy

from imitation.experiments.agnostic import pilot

LEARNER_FAMILY = "capacity-one-hidden"
# /2 binds the exact observation and action spaces; /1 specs are refused.
POLICY_SCHEMA = "agnostic-capacity-one-hidden-policy/2"
CHECKPOINT_BINDING_SCHEMA = "agnostic-capacity-checkpoint-binding/1"
TEACHER_WIDTH = 64
TEACHER_WIDTHS = (TEACHER_WIDTH,)
# Strictly narrower than the teacher; width 64 is a teacher reference only.
STUDENT_WIDTHS = (32, 16, 8, 4)
ACTION_RULE = "categorical logits; deterministic predict takes the argmax"
# "CAP" in ASCII: separates these streams from every one-element spawn key the
# data jobs and phase streams derive from the same seed.
CAPACITY_SPAWN_KEY = 0x434150
_PURPOSES = ("construction", "actor")
INITIALIZATION_RULE = (
    "Construction runs on a forked CPU torch generator seeded from "
    "SeedSequence(seed, spawn_key=(CAPACITY_SPAWN_KEY, 0, 0)); then the encoder "
    "is replaced by the teacher's bytes and the actor takes draw 0. Draw k "
    "builds Linear(d, w) and Linear(w, a) on a forked CPU generator seeded "
    "from SeedSequence(seed, spawn_key=(CAPACITY_SPAWN_KEY, 1, k)) and, with "
    "SB3 ortho_init, applies orthogonal gains sqrt(2) and 0.01 with zero "
    "biases. Cold refit number k uses draw k. Global RNG states are restored."
)
FREEZE_RULE = {
    "encoder": "teacher features extractor bytes; requires_grad False; eval "
    "mode under policy.train()",
    "critic": "frozen and excluded from every optimizer; unused",
    "optimizer": "actor hidden layer and action_net only",
    "teacher": "never modified; no teacher actor weights or optimizer state "
    "are copied",
}
_ACTOR_PREFIXES = ("mlp_extractor.policy_net.", "action_net.")
_ENCODER_PREFIXES = (
    "features_extractor.",
    "pi_features_extractor.",
    "vf_features_extractor.",
)
_TEACHER_CLASSES = (ActorCriticPolicy, ActorCriticCnnPolicy)
_HEX = set("0123456789abcdef")


# ---------------------------------------------------------------------------
# Digests and descriptions
# ---------------------------------------------------------------------------


def _array_sha256(array: np.ndarray) -> str:
    """Digest of an array's dtype, shape and C-order bytes."""
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(str(tuple(array.shape)).encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _entries_sha256(items: Iterable[Tuple[str, th.Tensor]]) -> str:
    """`pilot.policy_state_sha256` over the given ``(name, tensor)`` entries."""
    digest = hashlib.sha256()
    for name, tensor in sorted(items, key=lambda item: item[0]):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(str(tuple(array.shape)).encode("utf-8"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _is_actor(name: str) -> bool:
    return name.startswith(_ACTOR_PREFIXES)


def state_sha256(module: th.nn.Module) -> str:
    """Digest of every parameter and buffer (equals `pilot.policy_state_sha256`)."""
    return _entries_sha256(module.state_dict().items())


def encoder_sha256(policy: th.nn.Module) -> str:
    """Digest of the features-extractor state, every SB3 alias included."""
    state = policy.state_dict()
    return _entries_sha256(
        (k, v) for k, v in state.items() if k.startswith(_ENCODER_PREFIXES)
    )


def _frozen_entries(state) -> Iterator[Tuple[str, th.Tensor]]:
    return ((k, v) for k, v in state.items() if not _is_actor(k))


def frozen_sha256(policy: th.nn.Module) -> str:
    """Digest of everything except the student actor: encoder and critic."""
    return _entries_sha256(_frozen_entries(policy.state_dict()))


def spec_sha256(spec: Dict[str, Any]) -> str:
    """Canonical digest of a capacity spec."""
    return pilot.config_digest(spec)


def _qualified(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def _jsonable(value: Any) -> Any:
    """``value`` if it is canonical JSON already, else its ``repr``."""
    try:
        return json.loads(pilot.canonical_json(value))
    except (TypeError, ValueError):
        return repr(value)


def _numel(module: th.nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def _hidden_layers(net: th.nn.Module) -> List[th.nn.Module]:
    return list(net)


def _plain_int(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def space_description(space: Any, role: str) -> Dict[str, Any]:
    """Canonical description of a supported observation or action space.

    Observations must be a plain ``Box``: its exact low and high bytes are
    digested, since a repr may abbreviate them. Actions must be a plain
    ``Discrete`` starting at 0; SB3 categorical heads ignore ``start``, so any
    other start would silently reinterpret the actions.

    Raises:
        ValueError: For any other space.
    """
    if role == "observation" and type(space) is spaces.Box:
        bounds = (np.asarray(space.low), np.asarray(space.high))
        if any(b.dtype != space.dtype or b.shape != space.shape for b in bounds):
            raise ValueError(f"Inconsistent observation Box bounds: {space!r}")
        return {
            "type": _qualified(spaces.Box),
            "shape": [int(n) for n in space.shape],
            "dtype": str(space.dtype),
            "low_sha256": _array_sha256(bounds[0]),
            "high_sha256": _array_sha256(bounds[1]),
        }
    if role == "action" and type(space) is spaces.Discrete:
        if int(space.start) != 0:
            raise ValueError(f"Discrete actions must start at 0: {space!r}")
        return {
            "type": _qualified(spaces.Discrete),
            "n": int(space.n),
            "start": int(space.start),
            "dtype": str(space.dtype),
        }
    supported = {"observation": "a Box", "action": "a Discrete"}.get(role)
    if supported is None:
        raise ValueError(f"Unknown space role {role!r}")
    raise ValueError(f"The capacity {role} space must be {supported}: {space!r}")


def _described(space: Any, role: str) -> Optional[Dict[str, Any]]:
    try:
        return space_description(space, role)
    except ValueError:
        return None


def _check_teacher_width(teacher_hidden_width: Any) -> int:
    if not _plain_int(teacher_hidden_width) or teacher_hidden_width not in (
        TEACHER_WIDTHS
    ):
        raise ValueError(
            f"teacher_hidden_width must be one of {TEACHER_WIDTHS}: "
            f"{teacher_hidden_width!r}",
        )
    return int(teacher_hidden_width)


def _check_widths(hidden_width: Any, teacher_hidden_width: Any) -> Tuple[int, int]:
    _check_teacher_width(teacher_hidden_width)
    if not _plain_int(hidden_width) or hidden_width not in STUDENT_WIDTHS:
        raise ValueError(
            f"student hidden_width must be one of {STUDENT_WIDTHS}: {hidden_width!r}",
        )
    if not int(hidden_width) < int(teacher_hidden_width):
        raise ValueError(
            f"student hidden_width {hidden_width} is not narrower than the "
            f"teacher's {teacher_hidden_width}",
        )
    return int(hidden_width), int(teacher_hidden_width)


def _check_seed(seed: Any) -> int:
    if not _plain_int(seed) or seed < 0:
        raise ValueError(
            f"initialization_seed must be a non-negative integer: {seed!r}"
        )
    return int(seed)


def teacher_architecture(
    expert: ActorCriticPolicy,
    teacher_hidden_width: int = TEACHER_WIDTH,
) -> Dict[str, Any]:
    """Describe a compatible teacher, or refuse it.

    A compatible teacher is a plain SB3 ``ActorCriticPolicy`` (or its CNN
    alias) with a ``Box`` observation space, a ``Discrete`` action space
    starting at 0 (`space_description`), no gSDE, and an actor of exactly one
    hidden ``Linear(d, teacher_hidden_width)`` followed by its activation,
    then ``action_net = Linear(teacher_hidden_width, a)``.

    Returns:
        JSON-compatible actor description with the teacher's state digest.

    Raises:
        ValueError: For any other policy, including linear or two-hidden heads.
    """
    teacher_width = _check_teacher_width(teacher_hidden_width)
    if type(expert) not in _TEACHER_CLASSES:
        raise ValueError(
            "The teacher must be a plain SB3 ActorCriticPolicy, got "
            f"{type(expert).__name__}",
        )
    if not isinstance(expert.action_space, spaces.Discrete):
        raise ValueError(
            f"Capacity students need a Discrete action space: {expert.action_space}",
        )
    observation_space = space_description(expert.observation_space, "observation")
    action_space = space_description(expert.action_space, "action")
    if expert.use_sde:
        raise ValueError("The teacher uses gSDE; only categorical heads qualify")
    d = int(expert.features_dim)
    a = int(expert.action_space.n)
    layers = _hidden_layers(expert.mlp_extractor.policy_net)
    linears = [m for m in layers if isinstance(m, th.nn.Linear)]
    widths = [m.out_features for m in linears]
    if (
        widths != [teacher_width]
        or len(layers) != 2
        or layers[0] is not linears[0]
        or layers[0].in_features != d
        or not isinstance(layers[1], expert.activation_fn)
    ):
        raise ValueError(
            f"The teacher actor must be exactly Linear({d}, {teacher_width}) then "
            f"its activation; found hidden widths {widths} in {layers}",
        )
    head = expert.action_net
    if (
        not isinstance(head, th.nn.Linear)
        or head.in_features != teacher_width
        or head.out_features != a
    ):
        raise ValueError(f"The teacher action_net is not Linear({teacher_width}, {a})")
    critic = [
        m.out_features
        for m in expert.mlp_extractor.value_net
        if isinstance(m, th.nn.Linear)
    ]
    return {
        "policy_class": _qualified(type(expert)),
        "actor_hidden_widths": [teacher_width],
        "critic_hidden_widths": critic,
        "activation_fn": _qualified(expert.activation_fn),
        "features_dim": d,
        "n_actions": a,
        "observation_space": observation_space,
        "action_space": action_space,
        "action_distribution": "categorical",
        "action_rule": ACTION_RULE,
        "ortho_init": bool(expert.ortho_init),
        "share_features_extractor": bool(expert.share_features_extractor),
        "decision_head_parameters": d * teacher_width
        + teacher_width * a
        + teacher_width
        + a,
        "policy_state_sha256": state_sha256(expert),
    }


def _encoder_description(expert: ActorCriticPolicy) -> Dict[str, Any]:
    return {
        "boundary": "teacher features extractor before the actor MLP; no actor "
        "hidden layer or logits",
        "class": _qualified(expert.features_extractor_class),
        "kwargs": _jsonable(expert.features_extractor_kwargs),
        "features_dim": int(expert.features_dim),
        "observation_space": space_description(
            expert.observation_space,
            "observation",
        ),
        "normalize_images": bool(expert.normalize_images),
        "share_features_extractor": bool(expert.share_features_extractor),
        "parameters": _numel(expert.pi_features_extractor),
        "state_sha256": encoder_sha256(expert),
    }


# ---------------------------------------------------------------------------
# Named initialization streams
# ---------------------------------------------------------------------------


def _stream_seed(seed: int, purpose: str, draw: int) -> int:
    sequence = np.random.SeedSequence(
        int(seed),
        spawn_key=(CAPACITY_SPAWN_KEY, _PURPOSES.index(purpose), int(draw)),
    )
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


@contextlib.contextmanager
def _seeded_cpu_rng(seed: int) -> Iterator[None]:
    """Run with the CPU torch generator at ``seed``, restoring it afterwards."""
    with th.random.fork_rng(devices=[]):
        th.random.default_generator.manual_seed(seed)
        yield


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class CapacityRunner:
    """Learner selection a `restriction_pilot.RunnerSpec` carries.

    ``student_width`` is None for the data job, which every width shares.
    """

    family: str = LEARNER_FAMILY
    teacher_width: int = TEACHER_WIDTH
    student_width: Optional[int] = None

    def __post_init__(self):
        if self.family != LEARNER_FAMILY:
            raise ValueError(f"Unknown learner family {self.family!r}")
        _check_teacher_width(self.teacher_width)
        if self.student_width is not None:
            _check_widths(self.student_width, self.teacher_width)


def _validated_spec(spec: Any, digest: Any) -> Dict[str, Any]:
    if not isinstance(spec, dict):
        raise ValueError("capacity_spec must be a dict")
    if digest != spec_sha256(spec):
        raise ValueError("capacity_spec does not match its digest")
    if spec.get("schema") != POLICY_SCHEMA:
        raise ValueError(f"Unknown capacity spec schema {spec.get('schema')!r}")
    if spec.get("learner_family") != LEARNER_FAMILY:
        raise ValueError(f"Unknown learner family {spec.get('learner_family')!r}")
    _check_widths(spec.get("hidden_width"), spec.get("teacher_hidden_width"))
    _check_seed((spec.get("initialization") or {}).get("seed"))
    for key in ("teacher", "encoder", "student", "parameters"):
        if not isinstance(spec.get(key), dict):
            raise ValueError(f"capacity_spec lacks {key}")
    return copy.deepcopy(spec)


class CapacityActorCriticPolicy(ActorCriticPolicy):
    """Student with a frozen teacher encoder and a fresh one-hidden actor.

    Build students with `make_capacity_policy`; this constructor rebuilds the
    structure a spec declares (used by checkpoint loading), refuses any other
    structure, freezes everything except the actor and initializes the actor
    with draw 0. The encoder holds the teacher's bytes only once loaded.
    """

    def __init__(
        self,
        *args: Any,
        capacity_spec: Dict[str, Any],
        capacity_spec_sha256: str,
        share_features_extractor: bool = True,
        **kwargs: Any,
    ):
        spec = _validated_spec(capacity_spec, capacity_spec_sha256)
        seed = int(spec["initialization"]["seed"])
        with _seeded_cpu_rng(_stream_seed(seed, "construction", 0)):
            super().__init__(
                *args,
                share_features_extractor=share_features_extractor,
                **kwargs,
            )
        self.capacity_spec = spec
        self.capacity_spec_sha256 = str(capacity_spec_sha256)
        self.initialization_seed = seed
        self.initialization_draws = 0
        self._check_structure()
        for name, parameter in self.named_parameters():
            parameter.requires_grad = _is_actor(name)
        self._check_counts()
        self._write_actor(self._fresh_actor(0))
        self.train(self.training)

    # -- structure -----------------------------------------------------------

    def _check_structure(self) -> None:
        spec = self.capacity_spec
        teacher, encoder = spec["teacher"], spec["encoder"]
        width = int(spec["hidden_width"])
        layers = _hidden_layers(self.mlp_extractor.policy_net)
        observation = _described(self.observation_space, "observation")
        action = _described(self.action_space, "action")
        checks = {
            "observation_space": observation is not None
            and observation
            == encoder.get("observation_space")
            == teacher.get("observation_space"),
            "action_space": action is not None
            and action == teacher.get("action_space")
            and action["n"] == teacher.get("n_actions"),
            "net_arch": self.net_arch
            == {"pi": [width], "vf": list(spec["student"].get("critic_hidden_widths"))},
            "actor_layers": len(layers) == 2
            and isinstance(layers[0], th.nn.Linear)
            and layers[0].out_features == width
            and isinstance(layers[1], self.activation_fn),
            "action_net": isinstance(self.action_net, th.nn.Linear)
            and self.action_net.in_features == width,
            "activation_fn": _qualified(self.activation_fn)
            == teacher.get("activation_fn")
            == spec["student"].get("activation_fn"),
            "ortho_init": bool(self.ortho_init) == teacher.get("ortho_init"),
            "use_sde": not self.use_sde,
            "encoder_class": _qualified(self.features_extractor_class)
            == encoder.get("class"),
            "encoder_kwargs": _jsonable(self.features_extractor_kwargs)
            == encoder.get("kwargs"),
            "features_dim": int(self.features_dim)
            == encoder.get("features_dim")
            == teacher.get("features_dim"),
            "normalize_images": bool(self.normalize_images)
            == encoder.get("normalize_images"),
            "share_features_extractor": bool(self.share_features_extractor)
            == encoder.get("share_features_extractor"),
        }
        failed = sorted(k for k, ok in checks.items() if not ok)
        if failed:
            raise ValueError(f"Capacity policy structure fails checks {failed}")

    def _check_counts(self) -> None:
        d = int(self.features_dim)
        w = int(self.capacity_spec["hidden_width"])
        a = int(self.action_space.n)
        actual = {
            "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "encoder": _numel(self.pi_features_extractor),
            "critic_frozen": _numel(self.mlp_extractor.value_net)
            + _numel(self.value_net),
            "implementation_total": _numel(self),
        }
        actual["total_excluding_critic"] = actual["encoder"] + actual["trainable"]
        claimed = self.capacity_spec["parameters"]
        if actual["trainable"] != d * w + w * a + w + a or any(
            claimed.get(k) != v for k, v in actual.items()
        ):
            raise ValueError(
                f"Capacity parameter counts {actual} differ from the spec {claimed}",
            )

    def _encoders(self) -> List[th.nn.Module]:
        names = ("features_extractor", "pi_features_extractor", "vf_features_extractor")
        return [getattr(self, n) for n in names if getattr(self, n, None) is not None]

    def train(self, mode: bool = True) -> "CapacityActorCriticPolicy":
        """Set the training mode; the frozen encoder always stays in eval mode."""
        super().train(mode)
        for extractor in self._encoders():
            extractor.train(False)
        return self

    # -- actor ---------------------------------------------------------------

    def _actor_modules(self) -> Tuple[th.nn.Linear, th.nn.Linear]:
        return self.mlp_extractor.policy_net[0], self.action_net

    def _fresh_actor(self, draw: int) -> Dict[str, th.Tensor]:
        hidden, head = self._actor_modules()
        with _seeded_cpu_rng(_stream_seed(self.initialization_seed, "actor", draw)):
            fresh_hidden = th.nn.Linear(hidden.in_features, hidden.out_features)
            fresh_head = th.nn.Linear(head.in_features, head.out_features)
            if self.ortho_init:
                self.init_weights(fresh_hidden, gain=np.sqrt(2))
                self.init_weights(fresh_head, gain=0.01)
        return {
            "mlp_extractor.policy_net.0.weight": fresh_hidden.weight.detach(),
            "mlp_extractor.policy_net.0.bias": fresh_hidden.bias.detach(),
            "action_net.weight": fresh_head.weight.detach(),
            "action_net.bias": fresh_head.bias.detach(),
        }

    def _write_actor(self, values: Dict[str, th.Tensor]) -> None:
        parameters = dict(self.named_parameters())
        if sorted(values) != sorted(n for n in parameters if _is_actor(n)):
            raise RuntimeError("Fresh actor does not cover the actor parameters")
        with th.no_grad():
            for name, value in values.items():
                parameters[name].copy_(value)

    def bc_optimizer_parameters(self) -> List[th.nn.Parameter]:
        """Parameters a BC optimizer may own: the actor hidden and output layer."""
        return [p for module in self._actor_modules() for p in module.parameters()]

    def cold_refit_reinitialize(self) -> None:
        """Replace the actor with the next draw of its named stream.

        The encoder and critic are untouched and no global RNG advances. The
        caller clears its optimizer state.
        """
        self.initialization_draws += 1
        self._write_actor(self._fresh_actor(self.initialization_draws))

    # -- provenance ----------------------------------------------------------

    def verify_frozen_state(self) -> None:
        """Refuse a policy whose frozen state or freezing differs from its spec.

        Raises:
            ValueError: If the encoder or critic bytes, the requires_grad flags
                or the encoder mode differ from the spec.
        """
        spec = self.capacity_spec
        claimed = spec.get("frozen_state_sha256")
        if not (
            isinstance(claimed, str) and len(claimed) == 64 and set(claimed) <= _HEX
        ):
            raise ValueError("capacity_spec lacks its frozen state digest")
        if frozen_sha256(self) != claimed:
            raise ValueError(
                "The frozen state (encoder and critic) differs from the spec"
            )
        if encoder_sha256(self) != spec["encoder"].get("state_sha256"):
            raise ValueError("The frozen encoder state differs from the teacher's")
        flags = {n: p.requires_grad for n, p in self.named_parameters()}
        if flags != {n: _is_actor(n) for n in flags}:
            raise ValueError("Only the actor may be trainable; frozen flags changed")
        if any(m.training for e in self._encoders() for m in e.modules()):
            raise ValueError("The frozen encoder is not in eval mode")

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            share_features_extractor=self.share_features_extractor,
            capacity_spec=copy.deepcopy(self.capacity_spec),
            capacity_spec_sha256=self.capacity_spec_sha256,
        )
        return data

    def save_bound(self, path, *, compact=False) -> Dict[str, Any]:
        """Write the ``policy.save`` checkpoint and bind the bytes written.

        The checkpoint is serialized once in memory, hashed and written
        unchanged, so the binding describes this writer's bytes rather than a
        later read of the file; ``state_sha256`` digests the very state dict
        that was serialized (every parameter and buffer).

        Returns:
            A `CHECKPOINT_BINDING_SCHEMA` record of the path, file SHA256 and
            size, full state digest and spec digest.
        """
        state = self.state_dict()
        from imitation.experiments.ftrl import policy_utils

        blob = policy_utils.save_policy_checkpoint(path, self, compact=compact)
        return {
            "schema": CHECKPOINT_BINDING_SCHEMA,
            "path": str(path),
            "file_sha256": hashlib.sha256(blob).hexdigest(),
            "file_bytes": len(blob),
            "state_sha256": _entries_sha256(state.items()),
            "spec_sha256": self.capacity_spec_sha256,
        }

    def save(self, path) -> None:
        """Save as `save_bound` does, discarding the binding."""
        self.save_bound(path)


def make_capacity_policy(
    expert_policy: ActorCriticPolicy,
    *,
    hidden_width: int,
    initialization_seed: int,
    teacher_hidden_width: int = TEACHER_WIDTH,
) -> CapacityActorCriticPolicy:
    """Build a fresh student of ``hidden_width`` on the teacher's frozen encoder.

    Args:
        expert_policy: Qualified teacher (`teacher_architecture`); not modified.
        hidden_width: One of `STUDENT_WIDTHS`, below ``teacher_hidden_width``.
        initialization_seed: Seed of the named initialization streams.
        teacher_hidden_width: Declared teacher width, one of `TEACHER_WIDTHS`.

    Returns:
        The student, on the teacher's device, with a finalized spec.

    Raises:
        ValueError: For an incompatible teacher, an undeclared width or seed.
    """
    width, teacher_width = _check_widths(hidden_width, teacher_hidden_width)
    seed = _check_seed(initialization_seed)
    teacher = teacher_architecture(expert_policy, teacher_width)
    encoder = _encoder_description(expert_policy)
    d, a = teacher["features_dim"], teacher["n_actions"]
    head = d * width + width * a + width + a
    teacher_critic = _numel(expert_policy.mlp_extractor.value_net) + _numel(
        expert_policy.value_net,
    )
    parameters = {
        "trainable": head,
        "decision_head_formula": "d*w + w*a + w + a",
        "encoder": encoder["parameters"],
        "total_excluding_critic": encoder["parameters"] + head,
        "critic_frozen": teacher_critic,
        "implementation_total": _numel(expert_policy)
        - teacher["decision_head_parameters"]
        + head,
        "teacher_decision_head": teacher["decision_head_parameters"],
    }
    spec = {
        "schema": POLICY_SCHEMA,
        "learner_family": LEARNER_FAMILY,
        "hidden_width": width,
        "teacher_hidden_width": teacher_width,
        "student_widths_declared": list(STUDENT_WIDTHS),
        "teacher": teacher,
        "encoder": encoder,
        "student": {
            "actor_hidden_widths": [width],
            "critic_hidden_widths": teacher["critic_hidden_widths"],
            "activation_fn": teacher["activation_fn"],
            "action_rule": ACTION_RULE,
        },
        "parameters": parameters,
        "width_ratio": width / teacher_width,
        "parameter_ratio": head / teacher["decision_head_parameters"],
        "initialization": {
            "seed": seed,
            "spawn_key": [CAPACITY_SPAWN_KEY],
            "rule": INITIALIZATION_RULE,
        },
        "freeze": FREEZE_RULE,
        "frozen_state_sha256": None,
    }
    params = expert_policy._get_constructor_parameters()
    params["net_arch"] = {"pi": [width], "vf": list(teacher["critic_hidden_widths"])}
    params["share_features_extractor"] = expert_policy.share_features_extractor
    policy = CapacityActorCriticPolicy(
        **params,
        capacity_spec=spec,
        capacity_spec_sha256=spec_sha256(spec),
    )
    teacher_state = expert_policy.state_dict()
    state = policy.state_dict()
    encoder_keys = sorted(k for k in state if k.startswith(_ENCODER_PREFIXES))
    if encoder_keys != sorted(
        k for k in teacher_state if k.startswith(_ENCODER_PREFIXES)
    ):
        raise ValueError("Student and teacher encoders have different state keys")
    for key in encoder_keys:
        state[key] = teacher_state[key].detach().cpu().clone()
    policy.load_state_dict(state)
    spec["frozen_state_sha256"] = frozen_sha256(policy)
    policy.capacity_spec = spec
    policy.capacity_spec_sha256 = spec_sha256(spec)
    policy.verify_frozen_state()
    policy = policy.to(expert_policy.device)
    policy.train(expert_policy.training)
    return policy


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------


def _check_state_tensors(policy: th.nn.Module, state: Dict[str, Any]) -> None:
    """Refuse a state unlike ``policy``'s before anything is loaded or cast.

    Keys, shapes and dtypes must equal those of the constructed policy (so
    integer buffers such as batch-norm counters stay integers and no float
    is silently converted), and every floating tensor must be finite.

    Raises:
        ValueError: On the first difference.
    """
    expected = policy.state_dict()
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(str(k) for k in set(state) - set(expected))
    if missing or unexpected:
        raise ValueError(
            f"Capacity checkpoint state keys differ: missing {missing}, "
            f"unexpected {unexpected}",
        )
    for key, reference in expected.items():
        value = state[key]
        if not isinstance(value, th.Tensor) or value.layout != th.strided:
            raise ValueError(f"Capacity checkpoint state {key} is not a dense tensor")
        if value.dtype != reference.dtype or value.shape != reference.shape:
            raise ValueError(
                f"Capacity checkpoint state {key} is {value.dtype} "
                f"{tuple(value.shape)}, not {reference.dtype} "
                f"{tuple(reference.shape)}",
            )
        if (value.is_floating_point() or value.is_complex()) and not bool(
            th.isfinite(value).all(),
        ):
            raise ValueError(f"Capacity checkpoint state {key} is not finite")


def policy_from_saved(saved: Dict[str, Any], device: str = "cpu"):
    """Rebuild a student from ``policy.save`` contents, failing closed.

    Raises:
        ValueError: For an unknown or mutated spec, structure, observation or
            action space, frozen state (aliases included), or state keys,
            shapes, dtypes or non-finite values.
    """
    data = saved.get("data")
    state = saved.get("state_dict")
    if not isinstance(data, dict) or not isinstance(state, dict):
        raise ValueError("Not a policy checkpoint")
    try:
        policy = CapacityActorCriticPolicy(**data)
    except TypeError as exc:
        raise ValueError(f"Capacity checkpoint constructor data rejected: {exc}")
    _check_state_tensors(policy, state)
    claimed = policy.capacity_spec.get("frozen_state_sha256")
    if _entries_sha256(_frozen_entries(state)) != claimed:
        raise ValueError("The checkpoint's frozen state differs from its spec")
    try:
        policy.load_state_dict(state)
    except RuntimeError as exc:
        raise ValueError(f"Capacity checkpoint state rejected: {exc}") from None
    policy.verify_frozen_state()
    return policy.to(device)


def _check_digest(name: str, value: Any) -> None:
    if not (isinstance(value, str) and len(value) == 64 and set(value) <= _HEX):
        raise ValueError(f"{name} must be a SHA256 hex digest: {value!r}")


def load_checkpoint(
    path,
    device: str = "cpu",
    *,
    expected_spec_sha256: str,
    expected_file_sha256: str,
    expected_state_sha256: str,
) -> CapacityActorCriticPolicy:
    """Load a trusted student checkpoint bound to its writer's record.

    The expected digests come from the binding `save_bound` returned when the
    file was written. The file is read once; those bytes are hashed and then
    parsed, so the checked and the loaded bytes are the same.

    Raises:
        ValueError: If a digest is malformed, the bytes differ from the
            writer's, the file is not a capacity checkpoint, fails any
            `policy_from_saved` check, or carries another spec or state.
    """
    for name, value in (
        ("expected_spec_sha256", expected_spec_sha256),
        ("expected_file_sha256", expected_file_sha256),
        ("expected_state_sha256", expected_state_sha256),
    ):
        _check_digest(name, value)
    if hasattr(path, "read"):
        blob = path.read()
    else:
        with open(path, "rb") as f:
            blob = f.read()
    actual = hashlib.sha256(blob).hexdigest()
    if actual != expected_file_sha256:
        raise ValueError(
            f"{path} bytes have SHA256 {actual}, not the writer's "
            f"{expected_file_sha256}",
        )
    saved = th.load(io.BytesIO(blob), map_location=device, weights_only=False)
    from imitation.experiments.ftrl import policy_utils

    saved = policy_utils.read_policy_checkpoint(path, device, saved=saved)
    if not isinstance(saved, dict) or "capacity_spec" not in (saved.get("data") or {}):
        raise ValueError(f"{path} is not a capacity policy checkpoint")
    policy = policy_from_saved(saved, device)
    if policy.capacity_spec_sha256 != expected_spec_sha256:
        raise ValueError(
            f"{path} has capacity spec {policy.capacity_spec_sha256}, not the "
            f"recorded {expected_spec_sha256}",
        )
    if state_sha256(policy) != expected_state_sha256:
        raise ValueError(f"{path} loads a state that differs from the writer's")
    return policy
