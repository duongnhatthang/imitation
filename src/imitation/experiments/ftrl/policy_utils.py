"""Policy creation utilities for FTRL experiments.

Provides end-to-end and linear policy modes for comparing FTL vs FTRL.
Linear mode freezes the feature extractor + MLP hidden layers from an
expert and only trains the final action_net layer.
"""

import hashlib
import io
import pathlib

import torch as th
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy

COMPACT_SCHEMA = "trainable-policy-checkpoint/1"


def _state_digest(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def save_policy_checkpoint(path, policy, *, compact=False):
    """Write policy weights, sharing frozen state for compact round snapshots.

    Returns the exact bytes written, for callers that bind checkpoint identity.
    Optimizer and training-resumption state are outside this policy format.
    """
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in policy.state_dict().items()}
    data = policy._get_constructor_parameters()
    trainable = {
        n for n, p in policy.named_parameters(remove_duplicate=False) if p.requires_grad
    }
    if compact:
        # Buffers can change during training even when they have no gradient.
        round_keys = trainable | {
            n for n, _ in policy.named_buffers(remove_duplicate=False)
        }
        round_keys.intersection_update(state)
        frozen = {k: v for k, v in state.items() if k not in round_keys}
        base_path = path.parent / f"frozen-{_state_digest(frozen)}.pt"
        if not base_path.exists():
            th.save(
                {
                    "state_dict": frozen,
                    "data": data,
                    "policy_class": type(policy),
                    "trainable": sorted(trainable),
                    "round_keys": sorted(round_keys),
                },
                base_path,
            )
        base_digest = hashlib.sha256(base_path.read_bytes()).hexdigest()
        saved = {
            "schema": COMPACT_SCHEMA,
            "state_dict": {k: v for k, v in state.items() if k in round_keys},
            "frozen_file": base_path.name,
            "frozen_sha256": base_digest,
        }
    else:
        saved = {"state_dict": state, "data": data}
        if "capacity_spec" not in data:
            saved["policy_class"] = type(policy)
    buffer = io.BytesIO()
    th.save(saved, buffer)
    blob = buffer.getvalue()
    path.write_bytes(blob)
    return blob


def read_policy_checkpoint(path, device="cpu", *, saved=None):
    """Read a full checkpoint or resolve a compact checkpoint's frozen sibling."""
    if saved is None:
        saved = th.load(path, map_location=device, weights_only=False)
    if not isinstance(saved, dict):
        raise ValueError("Not a policy checkpoint")
    if saved.get("schema") != COMPACT_SCHEMA:
        return saved
    if hasattr(path, "read"):
        raise ValueError("A compact checkpoint needs a path to its frozen state")
    base_path = pathlib.Path(path).parent / saved["frozen_file"]
    blob = base_path.read_bytes()
    if hashlib.sha256(blob).hexdigest() != saved["frozen_sha256"]:
        raise ValueError("Compact checkpoint frozen state digest mismatch")
    base = th.load(io.BytesIO(blob), map_location=device, weights_only=False)
    head = saved["state_dict"]
    if set(head) != set(base["round_keys"]) or set(head) & set(base["state_dict"]):
        raise ValueError("Compact checkpoint trainable state keys differ")
    return dict(base, state_dict={**base["state_dict"], **head})


def load_policy_checkpoint(path, device="cpu"):
    """Load trusted full or compact experiment policy checkpoints.

    Constructor metadata and frozen state live once beside compact heads.
    Capacity checkpoints additionally validate their architecture and spec.
    """
    saved = read_policy_checkpoint(path, device)
    if "capacity_spec" in saved["data"]:
        from imitation.experiments.agnostic import capacity_policy

        return capacity_policy.policy_from_saved(saved, device)
    policy = saved.get("policy_class", ActorCriticPolicy)(**saved["data"])
    policy.load_state_dict(saved["state_dict"])
    if "trainable" in saved:
        for name, param in policy.named_parameters():
            param.requires_grad = name in saved["trainable"]
    return policy.to(device)


def create_end_to_end_policy(
    obs_space: spaces.Space,
    act_space: spaces.Space,
) -> ActorCriticPolicy:
    """Create a fresh [64,64] MLP policy for end-to-end training.

    Args:
        obs_space: Observation space.
        act_space: Action space.

    Returns:
        An untrained ActorCriticPolicy with [64,64] architecture.
    """
    return ActorCriticPolicy(
        obs_space,
        act_space,
        lr_schedule=lambda _: 1e-3,
        net_arch=[64, 64],
    )


def freeze_feature_layers(policy: ActorCriticPolicy) -> None:
    """Freeze all parameters except action_net.

    Sets requires_grad=False on all parameters whose name does not
    start with "action_net".

    Args:
        policy: The policy to partially freeze.
    """
    for name, param in policy.named_parameters():
        if not name.startswith("action_net"):
            param.requires_grad = False


def reinitialize_action_net(policy: ActorCriticPolicy) -> None:
    """Reinitialize action_net parameters with Xavier uniform / zeros.

    Args:
        policy: The policy whose action_net to reinitialize.
    """
    for name, param in policy.named_parameters():
        if name.startswith("action_net"):
            if param.dim() >= 2:
                th.nn.init.xavier_uniform_(param)
            elif param.dim() == 1:
                th.nn.init.zeros_(param)


def create_linear_policy(
    expert_policy: ActorCriticPolicy,
) -> ActorCriticPolicy:
    """Create a linear-mode policy by cloning expert and resetting action_net.

    Deep-copies the expert's entire policy, freezes all parameters except
    action_net, and reinitializes action_net. Works with any architecture
    (MLP, CNN, etc.) without needing to know internals.

    Args:
        expert_policy: Trained expert policy to clone features from.

    Returns:
        A policy with frozen hidden layers and fresh action_net.
    """
    import copy

    policy = copy.deepcopy(expert_policy)
    freeze_feature_layers(policy)
    reinitialize_action_net(policy)
    return policy
