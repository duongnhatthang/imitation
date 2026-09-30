# Running and monitoring an experiment campaign

Activate the experiment Python environment, then run
`bash experiments/run_bc_iid_experiments.sh` inside tmux. The default campaign
runs classical environments with `ftl ftrl bc bc_iid`, cold refits, and zero
expert mixing for FTL/FTRL. BC-iid always collects expert trajectories.

Defaults are five seeds, 1,000 classical rounds, one retained sample from each
of one complete trajectory per round, and no outer early stopping.
`bc_prefix` and `bc_pool` remain explicit opt-in algorithms.

```bash
EXP_RUN_ID=classical-cold-start-v1 bash experiments/run_bc_iid_experiments.sh
ENVS="CartPole-v1 Acrobot-v1" CLASSICAL_ROUNDS=100 SEEDS=3 \
  EXP_RUN_ID=classical-pilot bash experiments/run_bc_iid_experiments.sh
```

Use a fresh run ID for changed settings. Existing results carry configuration
and source fingerprints and must not be relabeled as results from this protocol.
The manifest records cold starts, beta, algorithms, environment filters, and
whether Atari was requested.

Atari is deferred by default. Explicitly opt in with `RUN_ATARI=1`.
`ENVS` filters the classical stage; `ATARI_ENVS` separately filters Atari.
All available GPUs are eligible by default, with an optional `GPUS` mask in
the remote launcher. No device is excluded by index.

## Optional remote launcher

`run.sh` reads connection settings from the environment or the gitignored
`experiments/.local/runner.env`. Set `HOST`, and optionally `REMOTE_REPO`,
`CONDA_ENV`, and `RUN_ID` there. Keep machine-specific configuration private.
The private `experiments/sync_results.sh` handles code and plot transfers.

```bash
./run.sh status
./run.sh start
./run.sh watch
./run.sh cells
./run.sh stop
./run.sh push
./run.sh pull
```

`start` activates Conda, validates imports, detects visible devices, and starts
a named tmux session. It refuses to launch while a campaign is already active.
`RUN_ATARI`, `ATARI_ENVS`, `ENVS`, `ALGOS`, `SEEDS`, and `CLASSICAL_WORKERS`
are forwarded to the campaign. `stop` asks before stopping active workers.

Status totals use the saved manifest, including environment selection and
whether Atari is enabled. A cell is one environment, algorithm, and seed.
A result JSON appears after that cell finishes; intermediate round directories
show progress before completion. Local snapshots are written to `runstate/`,
which is ignored by Git. Timing estimates require actual observed progress.

Results and logs belong under `experiments/runs/<run-id>/`; public plots are
mirrored separately under `experiments/plots/<run-id>/`. Never delete remote
results as part of code synchronization. Review cell completeness and recorded
settings before interpreting a campaign as evidence.
