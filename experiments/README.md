Experiment scripts are compatible with Linux and macOS.

## (macOS only) macOS compatibility setup

macOS to install some GNU-compatible binaries before all experiments scripts will work.

```
brew install coreutils gnu-getopt parallel
```

## Scripts

### FTL, FTRL, BC-iid, and fixed BC

The learning-curve and coverage sweeps run `ftl`, `ftrl`, `bc_iid`, and `bc`.
BC-iid replaces the former `bc_dagger` growing-dataset baseline.

BC-iid collects fresh complete trajectories using the same expert policy as BC,
with deterministic expert actions throughout every round. It uniformly samples
`--samples-per-round` distinct transitions from the pooled round trajectories
(default 1), accumulates them, and trains with FTL's zero-L2 objective, optimizer,
warm-start setting, and early-stopping rules. Evaluation uses the current learned
policy: once before training, then after the first, scheduled, and final rounds.

`--trajectories-per-round` (alias `--traj-per-round`) sets the minimum number of
complete trajectories per round, default 1. Extra trajectories are collected
when needed to supply the retained sample count. The learning-curve x-axis counts
retained training observations, not all environment steps or expert queries.
FTL/FTRL retain their configured expert-to-learner mixing schedule.

Fixed BC defaults to `prefix`: it keeps the first transitions in temporal order
from its expert dataset. FTL/FTRL and BC-iid default to `uniform`.
`--subsample-strategy uniform` explicitly selects uniform sampling for BC too;
`--subsample-strategy prefix` is available for runs that exclude BC-iid.
BC-iid always requires uniform sampling. Training minibatches may still shuffle;
`prefix` describes data selection, not optimizer minibatch order.

Example BC-iid run with one retained sample per round:

```bash
python -m imitation.experiments.ftrl.run_experiment \
    --envs CartPole-v1 --algos bc_iid --seeds 1 \
    --samples-per-round 1 --traj-per-round 1 --n-rounds 100 \
    --output-dir experiments/learning_curves/bc_iid_v1
```

Use a fresh output directory when comparing the new baselines. Existing
`bc_dagger` result files represent a different method and are not renamed or
reused as BC-iid results.

### Phase 1: Generate expert demonstrations from models.

Run `experiments/rollouts_from_policies.sh`. (Rollouts saved in `output/train_experts/`).
Demonstrations are used in Phase 2 for imitation learning.

### Phase 2: Train imitation learning.

Run `experiments/imit_benchmark.sh --run_name RUN_NAME`. To choose AIRL or GAIL, add the `--airl` and `--gail` flags (default is GAIL).

To analyze these results, run `python -m imitation.scripts.analyze with run_name=RUN_NAME`. Analysis can be run even while training is midway (will only show completed imitation learner's results). [Example output.](https://gist.github.com/shwang/4049cd4fb5cab72f2eeb7f3d15a7ab47)

### Phase 3: Transfer learning.

Run `experiments/transfer_learn_benchmark.sh`. To choose AIRL or GAIL, add the `--airl` and `--gail` flags (default is GAIL). Transfer rewards are loaded from `data/reward_models`.

### Coverage / recoverability / runtime (post-hoc analysis)

After a classical sweep (e.g. `run_learning_curves.sh`), visualize state coverage
and the DAgger recoverability constant, and summarize wall-clock:

```
./experiments/run_coverage_analysis.sh          # CartPole Phase 1
```

- **t-SNE coverage** (`plot_tsne_coverage`): one shared embedding per env, one panel
  per algorithm, colored by data-arrival round. Override the projection with
  `--perplexity`/`--tsne-seed`; embeddings are cached next to the PNG.
- **Recoverability** (`plot_recoverability`): distribution of
  mu(s)=max_a Q-min_a Q from a separately-trained DQN reference expert (the figure
  states this provenance and shows DQN vs PPO return). Interactive IL benefits when
  mu(s) << J.
- **Runtime** (`aggregate_runtime`): `runtime.csv` + grouped bar chart from the
  per-run `elapsed_seconds` already stored in result JSONs.

#### Atari + all-classical (Phase 2)

Recoverability dispatches by env family: exact `env.P` for toy-text (FrozenLake,
CliffWalking, Taxi), pretrained `sb3/dqn-<Game>` for Atari, per-env tuned DQN for
continuous classical (CartPole/Acrobot/MountainCar/LunarLander). Blackjack skips μ.
t-SNE uses the expert CNN features for Atari, StandardScaler otherwise.

Generate viz for an existing sweep:
```
./experiments/gen_coverage_viz.sh experiments/learning_curves/classical \
    CartPole-v1 Acrobot-v1 MountainCar-v0 LunarLander-v2 FrozenLake-v1 CliffWalking-v0 Taxi-v3 Blackjack-v1
./experiments/gen_coverage_viz.sh experiments/learning_curves/atari \
    PongNoFrameskip-v4 BreakoutNoFrameskip-v4 QbertNoFrameskip-v4 SeaquestNoFrameskip-v4
```

## Hyperparameter tuning

Add a named config containing the hyperparameter search space and other settings to `src/imitation/scripts/config/parallel.py`. (`def example_cartpole_rl():` is an example).

Run your hyperparameter tuning experiment using `python -m imitation.scripts.parallel with YOUR_NAMED_CONFIG inner_run_name=RUN_NAME`.

Analyze imitation learning experiments using `python -m imitation.scripts.analyze with run_name=RUN_NAME source_dir=~/ray_results`.

View Stable Baselines training stats on TensorBoard (available for regular RL, imitation learning, and transfer learning) using `tensorboard --log_dir ~/ray_results`. To view only a subset of TensorBoard training progress use `imitation.scripts.analyze gather_tb_directories with source_dir=~/ray_results run_name=RUN_NAME`.
