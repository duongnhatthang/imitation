# Phase 1: FTRL Algorithm - Research

**Researched:** 2026-03-19
**Domain:** FTRL algorithm extension of DAgger (imitation learning, PyTorch)
**Confidence:** HIGH — all findings come from direct source inspection of the existing codebase

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
None — all implementation choices are at Claude's discretion. The algorithm formulation is
fully specified by REQUIREMENTS.md (ALGO-01 through ALGO-06) and the paper's Eq. 6.

### Claude's Discretion
All implementation choices: class structure, file layout, test strategy, gradient accumulation
approach, anchor snapshot mechanism.

### Deferred Ideas (OUT OF SCOPE)
None stated.
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| ALGO-01 | `FTRLDAggerTrainer` subclasses `SimpleDAggerTrainer` | `SimpleDAggerTrainer` is at `dagger.py:552`; `train()` and `extend_and_update()` are the override points |
| ALGO-02 | Proximal term `(1/(2*eta_t))||w - w_t||^2` centered on current weights w_t | Requires saving a frozen param snapshot at round start; computed via `sum(||p - p_anchor||^2)` over policy parameters |
| ALGO-03 | Linear correction `-⟨w, Σ∇l_i(w_t)⟩` using gradients of past losses at w_t | Requires one backward pass over all accumulated data at round start with `retain_graph=False`; gradient stored as a flat vector or per-parameter tensors |
| ALGO-04 | Before each round: save w_t snapshot and compute gradient of accumulated data at w_t; both fixed during round | `extend_and_update()` override runs the snapshot/gradient step BEFORE calling `bc_trainer.train()` |
| ALGO-05 | FTRL degenerates to FTL (plain DAgger) when alpha → infinity | Proximal coefficient = `1/(2*eta_t)` where `eta_t = 1 / (alpha * Sigma_sigma_i)`; as alpha → inf the coefficient → inf, collapsing the solution to the proximal anchor w_t, which is not FTL degeneracy. See critical note below. |
| ALGO-06 | Smoke test: CartPole, 5 rounds, no error, reward >= BC | `seals/CartPole-v0` fixtures already exist in `tests/algorithms/conftest.py`; `cartpole_expert_policy` fixture available |
</phase_requirements>

---

## Summary

Phase 1 delivers `FTRLDAggerTrainer` and its supporting `FTRLLossCalculator` as a new file
`src/imitation/algorithms/ftrl.py`, plus a two-line backward-compatible change to `BC.__init__`
that allows injection of a custom `loss_calculator`. All infrastructure (optimizer, training loop,
rollout collection, beta schedule) is inherited unchanged from `SimpleDAggerTrainer` and `BC`.

The core algorithmic work is confined to three operations that happen at the boundary between
DAgger rounds: (1) snapshot the current policy weights as the anchor w_t, (2) run one forward+backward
pass over the entire accumulated dataset at w_t to compute Sigma_grad (the sum of gradients of past
losses at current weights), and (3) inject an `FTRLLossCalculator` into the BC inner loop that adds
the proximal term and linear correction to each batch's loss. The BC training loop, optimizer, and
data loading are untouched.

The critical correctness risk is the sign and centering of the FTRL loss terms. The proximal term
must be `(1/(2*eta_t)) * ||w - w_t||^2` (centered on w_t, NOT on zero), and the linear correction
is `- dot(w, Sigma_grad)` which is already a dot product in parameter space, not a squared norm.
Using `l2_weight` on the existing `BehaviorCloningLossCalculator` (which computes `||w||^2`, not
`||w - w_t||^2`) would silently produce wrong results.

**Primary recommendation:** Implement `FTRLLossCalculator` as a standalone callable (matching the
`BehaviorCloningLossCalculator` protocol) injected into `BC` via a new `loss_calculator` constructor
parameter. Implement `FTRLDAggerTrainer` by overriding only `extend_and_update` to insert the
pre-round snapshot/gradient computation before delegating to the parent's training loop.

---

## Standard Stack

### Core (all existing — no new dependencies)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyTorch | existing (`th`) | Gradient computation, param snapshots, parameter iteration | Native framework — `policy.parameters()` gives iterator |
| stable-baselines3 | existing (`~2.2.1`) | `policies.ActorCriticPolicy` interface | Policy class already used by BC |
| imitation.algorithms.bc | existing | Inner training loop, optimizer, data loading | Direct parent's parent — `BC.train()` is the inner loop |
| imitation.algorithms.dagger | existing | `SimpleDAggerTrainer` superclass, round logic, rollout collection | Direct parent |

### No new installations needed
All dependencies for Phase 1 are already present. No `pip install` required.

---

## Architecture Patterns

### Recommended File Layout
```
src/imitation/algorithms/
├── ftrl.py                  # NEW: FTRLLossCalculator + FTRLDAggerTrainer
├── bc.py                    # MODIFIED: 2-line change to __init__ only
├── dagger.py                # UNCHANGED
tests/algorithms/
├── test_ftrl.py             # NEW: unit tests + smoke test
├── conftest.py              # UNCHANGED (fixtures already sufficient)
```

### Pattern 1: Loss Calculator Injection into BC

`BC.__init__` currently hardcodes:
```python
# bc.py line 369 (CURRENT — must change):
self.loss_calculator = BehaviorCloningLossCalculator(ent_weight, l2_weight)
```

The two-line change makes `loss_calculator` an optional constructor argument:
```python
# bc.py (AFTER CHANGE):
def __init__(
    self,
    *,
    # ... existing params ...
    loss_calculator: Optional[Callable] = None,  # line 1 added to signature
):
    # ... existing body ...
    self.loss_calculator = loss_calculator or BehaviorCloningLossCalculator(ent_weight, l2_weight)  # line 2 changed
```

This is fully backward-compatible: no existing callers break, `loss_calculator=None` preserves
the current default behavior. The `BC.train()` loop at `bc.py:495` already calls
`self.loss_calculator(self.policy, obs_tensor, acts)` — no further changes needed there.

### Pattern 2: FTRLLossCalculator Protocol

The `FTRLLossCalculator` must match the existing callable protocol:
```python
# Source: bc.py lines 100-156 (BehaviorCloningLossCalculator.__call__ signature)
def __call__(
    self,
    policy: policies.ActorCriticPolicy,
    obs: Union[types.AnyTensor, types.DictObs, Dict[str, np.ndarray], Dict[str, th.Tensor]],
    acts: Union[th.Tensor, np.ndarray],
) -> BCTrainingMetrics:
    ...
```

`FTRLLossCalculator` adds two terms to the BC base loss:
```python
# ftrl.py (conceptual sketch — not final code):
@dataclasses.dataclass(frozen=True)
class FTRLLossCalculator:
    bc_loss_calculator: BehaviorCloningLossCalculator
    anchor_params: List[th.Tensor]   # frozen snapshot of w_t (detached)
    sigma_grad: List[th.Tensor]      # frozen Sigma_{i<t} grad l_i(w_t) (detached)
    eta_t: float                     # learning rate = 1 / (alpha * Sigma sigma_i)

    def __call__(self, policy, obs, acts) -> BCTrainingMetrics:
        bc_metrics = self.bc_loss_calculator(policy, obs, acts)

        # Proximal term: (1 / (2 * eta_t)) * ||w - w_t||^2
        proximal = sum(
            th.sum((p - a.detach()) ** 2)
            for p, a in zip(policy.parameters(), self.anchor_params)
        ) / (2.0 * self.eta_t)

        # Linear correction: -<w, sigma_grad>  (dot product in param space)
        linear_correction = -sum(
            th.sum(p * g.detach())
            for p, g in zip(policy.parameters(), self.sigma_grad)
        )

        total_loss = bc_metrics.loss + proximal + linear_correction
        # Return BCTrainingMetrics with updated loss field
        ...
```

### Pattern 3: FTRLDAggerTrainer Override Point

`SimpleDAggerTrainer.train()` calls `self.extend_and_update(bc_train_kwargs)` once per round.
`extend_and_update()` calls `self._try_load_demos()` then `self.bc_trainer.train()`.

`FTRLDAggerTrainer` overrides `extend_and_update` to insert snapshot/gradient computation
**before** calling `super().extend_and_update()`:

```python
# ftrl.py (conceptual sketch):
class FTRLDAggerTrainer(SimpleDAggerTrainer):
    def __init__(self, *, alpha: float, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self._sigma_sigma_i = 0.0  # accumulated sum of per-round sigma_i

    def extend_and_update(self, bc_train_kwargs=None):
        # Step 1: snapshot current weights as w_t (BEFORE this round's training)
        anchor = [p.detach().clone() for p in self.bc_trainer.policy.parameters()]

        # Step 2: compute sigma_grad = Sigma_{i=1}^{t-1} grad l_i(w_t)
        #   (only meaningful for round >= 1; for round 0 it is zero)
        sigma_grad = self._compute_sigma_grad()

        # Step 3: compute eta_t from accumulated sigma_i
        sigma_i = self._compute_current_sigma()
        self._sigma_sigma_i += sigma_i
        eta_t = 1.0 / (self.alpha * self._sigma_sigma_i) if self._sigma_sigma_i > 0 else float('inf')

        # Step 4: inject FTRLLossCalculator into bc_trainer for this round
        self.bc_trainer.loss_calculator = FTRLLossCalculator(
            bc_loss_calculator=BehaviorCloningLossCalculator(
                ent_weight=..., l2_weight=0.0
            ),
            anchor_params=anchor,
            sigma_grad=sigma_grad,
            eta_t=eta_t,
        )

        # Step 5: delegate to parent (loads demos, calls bc_trainer.train())
        return super().extend_and_update(bc_train_kwargs)
```

### Pattern 4: Computing sigma_grad

The gradient Sigma_{i=1}^{t-1} grad l_i(w_t) must be evaluated at the current weights w_t.
This requires running the entire accumulated dataset through the policy in forward+backward mode,
but WITHOUT updating parameters:

```python
# Computing sigma_grad over accumulated dataset (conceptual):
def _compute_sigma_grad(self) -> List[th.Tensor]:
    policy = self.bc_trainer.policy
    policy.zero_grad()
    # iterate over self.bc_trainer._demo_data_loader (already set by _try_load_demos)
    for batch in self.bc_trainer._demo_data_loader:
        metrics = bc_loss_calculator(policy, batch["obs"], batch["acts"])
        metrics.loss.backward()
    # Collect accumulated gradients as the snapshot
    grads = [p.grad.detach().clone() if p.grad is not None
             else th.zeros_like(p)
             for p in policy.parameters()]
    policy.zero_grad()
    return grads
```

IMPORTANT: `_try_load_demos()` must be called before `_compute_sigma_grad()` so the data
loader is populated. The override in `extend_and_update` must call `_try_load_demos()` itself
(or duplicate its logic) before computing the gradient. Alternatively, call `_try_load_demos()`
at the start of the override and pass the now-loaded loader.

### Anti-Patterns to Avoid

- **Using `l2_weight` in `BehaviorCloningLossCalculator` for the proximal term:** `l2_weight`
  computes `||w||^2` (weight decay), not `||w - w_t||^2`. This is mathematically wrong for FTRL.
- **Using `LpRegularizer` or `WeightDecayRegularizer`:** Same problem — both implement `||w||^p`
  regularization, not anchor-centered proximal terms.
- **Sharing the `FTRLLossCalculator` instance across rounds:** `anchor_params` and `sigma_grad`
  must be fresh each round. Create a new `FTRLLossCalculator` at the start of each round.
- **Calling `optimizer.step()` during sigma_grad computation:** The gradient accumulation pass
  computes gradients at w_t for storage only; `optimizer.step()` must NOT be called, and
  `zero_grad()` must be called before and after.
- **Forgetting `detach()` on anchor/grad tensors:** These tensors must not participate in the
  computational graph of the inner training loop or autograd will incorrectly compute higher-order
  derivatives.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Inner training loop | Custom epoch/batch iterator | `BC.train()` | Already handles minibatch accumulation, logging, progress bar |
| Rollout collection | Custom trajectory collector | `SimpleDAggerTrainer.train()` / `create_trajectory_collector()` | Already handles beta schedule, demo saving, round counting |
| Policy parameter iteration | Manual weight access | `policy.parameters()` (SB3 / PyTorch standard) | Works with any SB3 ActorCriticPolicy including CNN policies |
| Gradient zeroing | Manual `.grad = None` | `optimizer.zero_grad()` or `policy.zero_grad()` | Handles all parameter groups |
| Data loading | Custom DataLoader | Existing `DAggerTrainer._try_load_demos()` mechanism | Already handles shuffle, drop_last, collate_fn |

---

## Common Pitfalls

### Pitfall 1: Proximal Term Centered on Zero Instead of w_t
**What goes wrong:** `||w||^2` is computed instead of `||w - w_t||^2`. Algorithm no longer
matches Eq. 6. Results will differ from DAgger even with alpha → infinity in ways that are not
obviously wrong in aggregate metrics.
**Why it happens:** `BehaviorCloningLossCalculator`'s `l2_weight` does exactly this, and it's
tempting to reuse it.
**How to avoid:** Always compute `(p - anchor.detach()) ** 2` explicitly. Log `proximal_term`
value and verify it starts near zero at round start (since w is close to anchor) and grows
as training proceeds.
**Warning signs:** Proximal term is nonzero at round start (before any gradient steps); test
for ALGO-05 degeneracy fails.

### Pitfall 2: ALGO-05 Degeneracy Test Misunderstanding
**What goes wrong:** The FTL degeneracy test (ALGO-05) is written incorrectly.
**Explanation from Eq. 6:** As alpha → infinity, `eta_t = 1 / (alpha * Sigma sigma_i) → 0`,
making the proximal coefficient `1/(2*eta_t) → infinity`. This means the proximal term
dominates, forcing `w ≈ w_t` (the anchor). This is NOT plain DAgger — it means the policy
barely moves from its initialization each round.
**Correct interpretation:** FTRL degenerates to FTL (DAgger) when `alpha → 0` (no
regularization, proximal coefficient → 0), not when alpha → infinity. OR the requirement
means FTRL with a large proximal weight should produce the same result as using no proximal
term — read REQUIREMENTS.md ALGO-05 carefully: "FTRL degenerates to FTL when alpha → infinity."
This implies the paper's parameterization may define alpha as the INVERSE of the proximal
weight (i.e., large alpha = weak regularization). Verify the paper's Eq. 6 parameterization
before writing this test.
**How to avoid:** Re-read Proposition 4.1 carefully. The test should assert that with
alpha=1e8 (or equivalent "no regularization"), training curves are statistically close to
`SimpleDAggerTrainer` curves on the same seed.
**Warning signs:** Degeneracy test fails even with alpha=1e-8 or passes trivially.

### Pitfall 3: sigma_grad Computed AFTER _try_load_demos
**What goes wrong:** `_compute_sigma_grad()` is called before `_try_load_demos()` populates
`bc_trainer._demo_data_loader`. The data loader is None or stale, gradient is computed on
wrong/empty data.
**How to avoid:** In `extend_and_update` override: call `_try_load_demos()` first, then
compute anchor/sigma_grad, then call `bc_trainer.train()` (not `super().extend_and_update()`
since that would call `_try_load_demos()` again needlessly). Or call `_try_load_demos()`
and then `super().extend_and_update()` while ensuring sigma_grad uses the post-load loader.
**Warning signs:** sigma_grad is all zeros after round 1.

### Pitfall 4: Round 0 Edge Case for sigma_grad
**What goes wrong:** On round 0 (first round), there are no past losses to accumulate gradients
from. Code crashes with empty iterator or divides by zero.
**How to avoid:** Explicitly handle `self.round_num == 0` by initializing sigma_grad to
all-zero tensors of the same shape as policy parameters.
**Warning signs:** IndexError or NaN loss on first round.

### Pitfall 5: eta_t Parameterization (alpha vs sigma_i)
**What goes wrong:** sigma_i (per-round gradient norm or step size) is hardcoded to 1.0
or computed incorrectly, making eta_t meaningless.
**What sigma_i is:** In Lavington et al., `sigma_i` is the local smoothness / Lipschitz
constant of the loss at round i. A practical proxy is the gradient norm at w_t for round i's
data. A simpler approach used in practice is `sigma_i = 1` (constant, equivalent to
eta_t = 1 / (alpha * t)).
**How to avoid:** Start with `sigma_i = 1` for simplicity (eta_t = 1/(alpha*t)), document
the choice, and verify the degeneracy test still works.
**Warning signs:** eta_t is NaN or inf on round 1; proximal term is 0 or inf for all rounds.

### Pitfall 6: Gradient Accumulation During sigma_grad Leaves Stale Gradients
**What goes wrong:** After computing sigma_grad via backward passes, `policy.parameters()`
still have `.grad` set from the computation. The first call to `loss.backward()` inside
`BC.train()` then accumulates onto these stale gradients rather than computing fresh ones.
**How to avoid:** Call `self.bc_trainer.optimizer.zero_grad()` (or `policy.zero_grad()`)
immediately after extracting sigma_grad and before returning from the pre-round computation.
**Warning signs:** Loss diverges on first batch of round 1+; optimizer behaves erratically.

---

## Code Examples

### BC Constructor Change (the "two-line change")
```python
# Source: src/imitation/algorithms/bc.py lines 274-369 (current state)
# BEFORE (line 369):
self.loss_calculator = BehaviorCloningLossCalculator(ent_weight, l2_weight)

# AFTER — add to __init__ signature:
loss_calculator: Optional[Callable[..., BCTrainingMetrics]] = None,

# AFTER — replace line 369:
self.loss_calculator = loss_calculator or BehaviorCloningLossCalculator(ent_weight, l2_weight)
```

### Parameter Snapshot Pattern
```python
# Detached clone of current policy weights — serves as frozen anchor w_t
anchor_params: List[th.Tensor] = [
    p.detach().clone() for p in self.bc_trainer.policy.parameters()
]
```

### Proximal Term Computation
```python
# Source: derived from REQUIREMENTS.md ALGO-02 and bc.py BehaviorCloningLossCalculator pattern
# (1/(2*eta_t)) * ||w - w_t||^2
proximal_term = sum(
    th.sum((p - a) ** 2)
    for p, a in zip(policy.parameters(), self.anchor_params)
) / (2.0 * self.eta_t)
```

### Linear Correction Computation
```python
# Source: derived from REQUIREMENTS.md ALGO-03 and PROJECT.md Eq. 6
# - <w, Sigma_grad>  — note negative sign
linear_correction = -sum(
    th.sum(p * g)
    for p, g in zip(policy.parameters(), self.sigma_grad)
)
```

### Existing Fixture for Smoke Test (ALGO-06)
```python
# Source: tests/algorithms/conftest.py lines 22-28 and 50-70
# cartpole_expert_policy and cartpole_venv fixtures already exist:
@pytest.fixture
def cartpole_expert_policy(cartpole_venv: VecEnv) -> BasePolicy:
    return serialize.load_policy("ppo-huggingface", cartpole_venv, env_name="seals/CartPole-v0")

# cartpole_bc_trainer fixture already exists with batch_size=50
# FTRLDAggerTrainer smoke test can parallel the existing SimpleDAggerTrainer test pattern
```

---

## State of the Art

| Old Approach | Current Approach | Impact on This Phase |
|--------------|------------------|---------------------|
| Separate `loss_calculator` constructor param | Hardcoded `BehaviorCloningLossCalculator` in `BC.__init__` | Need to add the param — two-line change |
| N/A | `BehaviorCloningLossCalculator` as `@dataclasses.dataclass(frozen=True)` | `FTRLLossCalculator` should match this pattern |
| N/A | `SimpleDAggerTrainer.extend_and_update()` delegating to `BC.train()` | Override point for pre-round FTRL setup |

**Key confirmed facts (from direct source inspection):**
- `BC.__init__` does NOT currently accept `loss_calculator` — the two-line change is needed
- `BC.train()` calls `self.loss_calculator(self.policy, obs_tensor, acts)` at line 495 — injection will work
- `BCTrainingMetrics` is a frozen dataclass — `FTRLLossCalculator` should return this type (or a compatible subclass with extra fields logged separately)
- `SimpleDAggerTrainer.train()` calls `self.extend_and_update(bc_train_kwargs)` at line 693 — the override point is confirmed
- `_try_load_demos()` at line 423 sets `self.bc_trainer` data loader and increments `_last_loaded_round`
- `round_num` is incremented inside `extend_and_update` (line 495) — the override must be aware of this

---

## Open Questions

1. **Alpha parameterization direction (ALGO-05 degeneracy test)**
   - What we know: REQUIREMENTS.md says "FTRL degenerates to FTL when alpha → infinity"
   - What's unclear: In Eq. 6, large proximal coefficient forces w ≈ w_t (not FTL behavior). "Alpha → infinity" producing FTL behavior requires alpha to be in the DENOMINATOR of the proximal coefficient (large alpha = weak regularization).
   - Recommendation: Parameterize as `eta_t = alpha / (Sigma sigma_i)` so large alpha = large learning rate = weak proximal = FTL. This makes ALGO-05 testable. Confirm against the paper before finalizing.

2. **sigma_i computation: constant 1 vs gradient norm**
   - What we know: REQUIREMENTS.md does not specify how to compute sigma_i per round
   - What's unclear: Whether to use constant sigma_i=1 (eta_t = alpha/t) or gradient-norm-based sigma_i
   - Recommendation: Use sigma_i=1 for Phase 1 (well-defined, matches many practical FTRL implementations). Document as a hyperparameter for Phase 3+ tuning.

3. **BCTrainingMetrics return from FTRLLossCalculator**
   - What we know: `BC.train()` expects `loss_calculator` to return `BCTrainingMetrics`; the logging code at line 472 logs all fields of this dataclass
   - What's unclear: Whether to subclass `BCTrainingMetrics` to add `proximal_term` and `linear_correction` for logging, or to return a plain `BCTrainingMetrics` with only `loss` updated
   - Recommendation: Return a `BCTrainingMetrics` with the combined `loss` only; log `proximal_term` and `linear_correction` separately via the FTRL trainer's logger to avoid touching the BCLogger code.

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (configured in `setup.cfg [tool:pytest]`) |
| Config file | `setup.cfg` — `[tool:pytest]` section |
| Quick run command | `pytest tests/algorithms/test_ftrl.py -x -q` |
| Full suite command | `pytest tests/algorithms/ -x -q` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| ALGO-01 | `FTRLDAggerTrainer` can be imported and instantiated | unit | `pytest tests/algorithms/test_ftrl.py::test_ftrl_instantiation -x` | Wave 0 |
| ALGO-02 | Proximal term uses `||w - w_t||^2` not `||w||^2` | unit | `pytest tests/algorithms/test_ftrl.py::test_proximal_term_centering -x` | Wave 0 |
| ALGO-03 | Linear correction gradient computed at current weights | unit | `pytest tests/algorithms/test_ftrl.py::test_sigma_grad_at_current_weights -x` | Wave 0 |
| ALGO-04 | Anchor + gradient computed before round's optimization | unit | `pytest tests/algorithms/test_ftrl.py::test_anchor_frozen_during_round -x` | Wave 0 |
| ALGO-05 | FTL degeneracy: large alpha produces DAgger-like curves | integration | `pytest tests/algorithms/test_ftrl.py::test_ftl_degeneracy -x` | Wave 0 |
| ALGO-06 | CartPole smoke test: 5 rounds, no error, reward >= BC | smoke | `pytest tests/algorithms/test_ftrl.py::test_cartpole_smoke -x -m "not expensive"` | Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest tests/algorithms/test_ftrl.py -x -q`
- **Per wave merge:** `pytest tests/algorithms/ -x -q`
- **Phase gate:** `pytest tests/algorithms/ -x -q` green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/algorithms/test_ftrl.py` — covers ALGO-01 through ALGO-06 (does not exist yet)

*(All required fixtures are already present in `tests/algorithms/conftest.py` and `tests/conftest.py` — `cartpole_venv`, `cartpole_expert_policy`, `cartpole_bc_trainer`, `rng`, `custom_logger` are all available without modification.)*

---

## Sources

### Primary (HIGH confidence — direct source inspection)
- `src/imitation/algorithms/bc.py` — BC constructor (no existing `loss_calculator` param confirmed at line 274-369); `train()` loop calls `self.loss_calculator` at line 495; `BehaviorCloningLossCalculator` protocol at lines 94-156; `BCTrainingMetrics` dataclass at lines 81-91
- `src/imitation/algorithms/dagger.py` — `SimpleDAggerTrainer` at line 552; `train()` calls `extend_and_update` at line 693; `extend_and_update` loads demos then calls `bc_trainer.train()` at lines 455-497; `round_num` increment at line 495; `_try_load_demos` at line 423
- `tests/algorithms/conftest.py` — `cartpole_expert_policy`, `cartpole_venv`, `cartpole_bc_trainer` fixtures confirmed available
- `tests/algorithms/test_dagger.py` — `_build_simple_dagger_trainer` pattern at lines 224-249 (template for FTRL smoke test)
- `.planning/research/SUMMARY.md` — prior project-level research confirming key architectural decisions
- `.planning/PROJECT.md` — FTRL Eq. 6 formulation and parameterization from Lavington et al. 2022

### Secondary (MEDIUM confidence)
- Lavington et al. (2022) "Improved Policy Optimization for Online Imitation Learning" — Proposition 4.1 Eq. 6 (accessed via PROJECT.md description; full PDF not directly verified)

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all from direct source inspection; no external dependencies needed
- Architecture: HIGH — BC constructor, training loop, and DAgger override points all confirmed via source reading
- Pitfalls: HIGH — most pitfalls derived from direct source inspection (e.g., `l2_weight` computes `||w||^2` confirmed at bc.py:138-145); ALGO-05 parameterization direction is MEDIUM pending paper verification

**Research date:** 2026-03-19
**Valid until:** 2026-06-19 (stable library — BC/DAgger source unlikely to change)
