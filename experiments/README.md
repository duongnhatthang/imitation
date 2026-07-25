Experiment scripts are compatible with Linux and macOS.

## (macOS only) macOS compatibility setup

macOS to install some GNU-compatible binaries before all experiments scripts will work.

```
brew install coreutils gnu-getopt parallel
```

## Scripts

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

## Hyperparameter tuning

Add a named config containing the hyperparameter search space and other settings to `src/imitation/scripts/config/parallel.py`. (`def example_cartpole_rl():` is an example).

Run your hyperparameter tuning experiment using `python -m imitation.scripts.parallel with YOUR_NAMED_CONFIG inner_run_name=RUN_NAME`.

Analyze imitation learning experiments using `python -m imitation.scripts.analyze with run_name=RUN_NAME source_dir=~/ray_results`.

View Stable Baselines training stats on TensorBoard (available for regular RL, imitation learning, and transfer learning) using `tensorboard --log_dir ~/ray_results`. To view only a subset of TensorBoard training progress use `imitation.scripts.analyze gather_tb_directories with source_dir=~/ray_results run_name=RUN_NAME`.
