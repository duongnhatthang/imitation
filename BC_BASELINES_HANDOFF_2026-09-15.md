# BC baselines campaign, what changed and how to launch it

Branch `feature/bc-iid-meeting-followups`. Covers meeting items 1, 2, 3 and 4.
Item 2's analysis is in `BC_IID_VS_BC_DAGGER_DIAGNOSIS_2026-09-15.md`.

## The three BC variants

All three run through the **same** warm-started FTRL trainer as FTL, with the
same L2 schedule (0), the same batch size, the same aggregation and the same
on-policy evaluation. The only thing that differs is which transitions arrive
at round t. That is deliberate: holding the optimizer fixed is what turns the
comparison into a measurement of the sampling effect rather than a confound.

| algo | round-t training set | expert episodes behind N labels |
|---|---|---|
| `bc_prefix` | first t transitions of the chronological offline dataset, byte-for-byte fixed BC's artifact, consumed in order | `⌈N/H⌉` |
| `bc_pool` | t uniform draws from that **same** fixed pool (this is the removed `bc_dagger`) | `⌈N/H⌉` |
| `bc_iid` | one uniform state from each of t freshly collected expert episodes | `N` |

`bc_prefix → bc_pool` isolates ordering and time-truncation within a fixed
pool; `bc_pool → bc_iid` isolates how many independent episodes the labels come
from. The diagnosis says nearly all of the observed gap is in the second step,
this measures it instead of asserting it.

A regression test pins that `bc_prefix` and fixed BC resolve to the same expert
dataset `sha256`, so "identical training data, consumed sequentially" is a
checked property rather than an intention.

## Item 1: m > 1 sampling

`_uniform_round_demos` now draws `m` **distinct trajectories** uniformly without
replacement and one uniformly random state from each, instead of pooling the
round's transitions and drawing `m` from that pool. `trajectories_per_round` is
raised to `samples_per_round` automatically. Applies to FTL, FTRL and BC-iid.

**The m = 1 path is bit-for-bit unchanged, RNG stream included.** When the round
holds exactly `m` trajectories there is nothing to choose between them, so no
draw is taken. Verified by rerunning `ftl`, `ftrl`, `bc_iid` and `bc` on
CartPole against the pre-change tree: every logged metric is identical. So this
change cannot account for any difference between existing m = 1 campaigns, and
the new run's `ftl`/`ftrl`/`bc_iid`/`bc` cells stay comparable with
`20260914-bc-iid-v2`.

## Item 3: visitation overlap

`render_coverage_diff` compared `{ftl, ftrl}` against `bc` (fixed). It now
compares `ftl` against `bc_iid` by default (`COVERAGE_DIFF_SIDES`), with both
sides parameterised. The point: BC-iid's training data *is* d^{π^E}, so a region
FTL reaches and BC-iid does not is coverage that interaction bought, not
coverage a larger offline dataset would have supplied. Needs no new training,
it reads the scratch demos the sweep already writes.

## Reproducibility

- `_seed_everything(seed)` at the top of `run_single` pins Python's `random` and
  numpy's legacy global RNG alongside the existing per-cell `Generator` and
  torch seeding.
- Expert training uses a pinned `EXPERT_TRAINING_SEED = 0` on its own RNG
  stream. Previously the shared per-env expert was trained with whichever
  cell's seed happened to miss the cache first, and that miss also perturbed
  that cell's RNG. An existing `expert_cache` is unaffected, only envs without
  a cached expert change.
- `protocol_version` 3 → 4, so resume will not mix old and new cells.

## Launching on the execution host

No new driver. The algo set is now a campaign parameter (`EXP_ALGOS` in
`experiments/paths.sh`, overridable with `ALGOS`), threaded through the existing
`run.sh` -> `run_bc_iid_experiments.sh` -> `run_learning_curves.sh` path.
`ENVS="CartPole-v1 Taxi-v3"` narrows the classical sweep the same way.

Run everything from this worktree. `experiments/sync_results.sh` and
`experiments/setup_server.sh` are gitignored, so they are not in a fresh
worktree; they have been copied across from the main checkout already. If you
make another worktree later, copy them again or `push` will fail.

### Do not push while 20260914-bc-iid-v2 is running

`protocol_version` went 3 -> 4, and `run_experiment` refuses to resume results
written under a different protocol. Pushing this branch onto the server while
v2 still has cells to finish would leave that campaign unable to resume its own
work. `run.sh start` also refuses outright while a driver is alive.

### Sequence

```bash
cd /path/to/imitation

# 1. where is v2?  (its four-algo set is needed for correct denominators)
RUN_ID=20260914-bc-iid-v2 ALGOS="ftl ftrl bc bc_iid" ./run.sh status

# 2. once v2 has finished the stages you want, stop it and take its plots
RUN_ID=20260914-bc-iid-v2 ./run.sh stop     # prompts; completed cells are kept
RUN_ID=20260914-bc-iid-v2 ./run.sh pull     # -> experiments/plots/20260914-bc-iid-v2/

# 3. push this branch, then start
./run.sh push
./run.sh start                              # RUN_ID=20260916-bc-baselines-v1

# 4. monitor
./run.sh watch                              # every 10 min until Ctrl-C
./run.sh status                             # or a single check
./run.sh runs                               # every run directory on the server
./run.sh pull                               # plots for the current RUN_ID
```

`push` sends the branch's tracked files (`git ls-files`) to `~/imitation` on the
server, so `run.sh` and `RUN.md` now travel with it. `start` activates the
conda env, preflights the imports, then detaches the driver; it prints the algo
set, the seed count and the classical cell total before doing so.

240 classical cells (8 envs x 6 algos x 5 seeds) and 280 Atari cells. That is
1.5x the four-algo campaign; at v2's rate (~4 500 rounds/h on 9 workers)
classical training is roughly 50 h of wall clock, Atari considerably more.

### Cheaper first pass

If 50 h is too much before seeing whether the decomposition works, the four
environments that span the pool-size range are enough to confirm or kill it:

```bash
ENVS="CartPole-v1 LunarLander-v2 Taxi-v3 Acrobot-v1" ./run.sh start
```

CartPole's pool is 2 expert episodes and Acrobot's is ~12, so the prediction is
a large `bc_pool` -> `bc_iid` gap on CartPole and almost none on Acrobot. Note
that `./run.sh status` still reports cell totals for the full group, so the
count will stop short of 240 -- that is the narrowing, not a stall.

### GPUs and workers

**The classical stage is CPU-only on purpose**, so 0% GPU utilisation during
`classical_training` is correct rather than a fault. `run_learning_curves.sh`
passes `--n-gpus 0`, and `run_single` selects a device with
`use_gpu = is_atari(env) and _WORKER_GPU_ID is not None`. The policies here are
linear maps on toy MDPs -- CartPole's is a 4x2 matrix -- where a host-to-device
round trip costs more than the matmul saves. Moving classical to GPU would make
it slower.

The lever for the classical stage is **CPU workers**. The driver used to force
8 regardless of the node; it now lets `run_learning_curves.sh` size itself to
`nproc - 2`, capped at `WORKER_CAP` (default 32). `CLASSICAL_WORKERS=N` or an
exported `N_WORKERS=N` still overrides. `./run.sh status` now prints

```
cpu               <workers> workers / <cores> cores   load <1m/5m/15m>
```

so the headroom is visible before deciding. Classical throughput is roughly
approximately linear in worker count for independent cells.

The Atari stage is where GPUs matter. `run.sh start` masks the cards with
`GPUS` (see RUN.md for local configuration) and sets
`N_GPUS` to match, and `run_experiment` hands worker *w* the visible device
*w % N_GPUS*.

That handout had a race. The parent fills a `multiprocessing.Queue` and the
pool initializer read it with `get_nowait()`, so a worker that started before
the feeder thread had delivered got `None` and ran the whole sweep on CPU,
silently -- the same "silently misplaced worker" failure the GPU mask is trying
to prevent, arriving by a different route. It now blocks up to 30s, logs
`worker <pid> -> cuda:<n>`, and warns on every fallback instead of swallowing
the exception. Each result JSON also records the `device` the cell ran on, so
"did this actually use the GPUs" is answerable from the artifacts.

`ATARI_WORKERS` defaults to 4. Under the three-card mask that puts two workers
on one card and one on each of the others; set it to a multiple of the visible
card count for an even spread, or higher to put several cells on each card if
memory allows.

### Watching without an SSH session

Each check overwrites `runstate/latest.txt` and appends to
`runstate/history.tsv` in the main checkout. Leaving `./run.sh watch` running in
a terminal lets anything that can read that folder -- including a Claude session
linked to this Mac -- follow the run without an SSH session of its own.

## Verification done locally

- 26 new tests in `tests/experiments/test_bc_baselines.py`, all passing.
- `tests/experiments/test_result_layout.py` passes except one pre-existing
  failure (the Atari disk guard, which also fails on the unmodified tree here).
- Six-algo CartPole smoke: all six cells complete, `bc_prefix` and fixed BC
  share an expert-dataset sha, the offline baselines report zero trajectories
  collected per round and a constant collection cost, and `plot_results`
  renders all six series.
- `plot_results`, `plot_tsne_coverage` and `tsne_coverage` test failures in this
  sandbox are pandas/scikit-learn version skew and reproduce identically on the
  unmodified tree; they need re-running in the server's `imitation` conda env.

## Still open

- The fairness caveat for the paper: BC-iid spends one full expert episode per
  label (~500k expert env steps for 1000 CartPole labels) while BC-prefix,
  BC-pool and fixed BC spend ~1000 in total. Matched on the x-axis, 500× apart
  on demonstration effort. FTL is in BC-iid's position, so FTL vs BC-iid is a
  fair pairing; BC-iid vs fixed BC is not. The learning-curve caption should say
  so.
- Atari was not smoke-tested here (no GPU, and `box2d-py` does not build in the
  sandbox, so LunarLander was skipped locally too).
