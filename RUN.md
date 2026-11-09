# Running and watching the BC-iid campaign

Everything is `./run.sh`. Run it from this directory.

```bash
./run.sh status          # one check
./run.sh watch           # check every 10 min, keep going (Ctrl-C to stop)
./run.sh start           # start, or resume where it stopped
./run.sh stop            # stop; completed cells are kept
./run.sh runs            # every run directory on the server, with cell counts
./run.sh push            # push tracked code up
./run.sh pull            # pull plots back
```

Set `HOST` to your SSH alias in the environment or in the ignored
`experiments/.local/run-host.conf` shell file. That file may also set `GPUS`;
command-line environment values override both settings.

Defaults: `RUN_ID=20261109-bc-baselines-eval50-v1`,
`CONDA_ENV=imitation`, atari 200 rounds, classical 1000 rounds, and
`ALGOS="ftl ftrl bc bc_iid bc_prefix bc_pool"`. Override on the command line,
e.g. `RUN_ID=20260920-bc-iid-v3 ./run.sh start`.

**Watching an older run needs its algo set too.** The cell totals are derived
from `ALGOS` and `SEEDS`, so the four-algo campaign is

```bash
RUN_ID=20260914-bc-iid-v2 ALGOS="ftl ftrl bc bc_iid" ./run.sh status
```

`status` now reads the algorithm set and seed count from the run's own
`manifest.json` first, so this is only needed for a run whose manifest is
missing; without either, the totals fall back to the four-algo 160/140.

`push` and `pull` delegate to `experiments/sync_results.sh`, per CLAUDE.md,
no raw rsync, and nothing that can delete on the server.

## What "progress" means

`run_experiment` writes one JSON per *cell*, one (environment, algorithm, seed)
triple, at `<data dir>/<env>/<algo>_<policy-mode>_seed<N>.json`.

| stage | cells | from |
|---|---|---|
| classical | 8 x N x 5 | 8 classical envs x N algos x 5 seeds |
| atari | 7 x N x 5 | 7 atari-zoo envs x N algos x 5 seeds |

With the default six algorithms that is 240 classical and 210 atari cells
(7 x 6 x 5, not 280).
The six are `ftl`, `ftrl`, `bc` (fixed, single-round), `bc_iid`, `bc_prefix`
and `bc_pool`; the last three differ only in where round t's transitions come
from, which is what the comparison is measuring. See
`BC_BASELINES_HANDOFF_2026-09-15.md`.

A cell's JSON appears only when that cell finishes all its rounds, so the cell
count moves in jumps. Between jumps, `atari rounds seen` counts `round-NNN`
directories under `data/atari/scratch/` and shows that work is still happening.
`newest result ... min old` is the stall detector: if it climbs past a couple of
hours while `worker processes` is 0, the run has died.

The ETA is measured, not assumed: it is the cell-completion rate over the rows
in `runstate/history.tsv`, so it needs about 30 minutes of watching before it
says anything.

## Reading progress without SSH

Each check overwrites `runstate/latest.txt` and appends a row to
`runstate/history.tsv`. Anything that can read this folder, including a Claude
session linked to this Mac, can follow the run from those two files without an
SSH session of its own. Leave `./run.sh watch` running in a terminal for that.

## Starting and resuming

`start` does three things the earlier scripts did not:

1. **Activates conda.** `ssh host 'cmd'` gives a non-login shell with no conda
   on `PATH`, so `python` resolves to the system interpreter and the driver dies
   in its first python block. That is precisely what the three orphan run
   directories dated 2026-09-14 are, all three read
   `failed: starting (exit 1)` with zero cells.
2. **Preflights.** It imports torch, gymnasium, stable_baselines3 and
   `imitation.experiments.ftrl.run_experiment` and prints the CUDA device count
   *before* detaching, so a broken environment fails in one second with a
   readable message instead of silently in a background job.
3. **Sets `EXP_RUN_ID`.** `paths.sh` defaults it to a fresh UTC timestamp, so
   without it every launch opens a new empty run. With it, a restart continues
   the same run and completed cells are skipped.

## GPU selection

`GPUS` lists the physical cards the sweep may use. It defaults to `0` unless
set in the environment or the local config. Set it to the devices available
for your run.

`start` exports `CUDA_VISIBLE_DEVICES=$GPUS` and sets `N_GPUS` to the number of
cards in that list. Both are needed: `run_experiment` gives worker *w* the
device index `w % n_gpus` counting from 0 over the **visible** devices, and
`run_atari_curves.sh` defaults `N_GPUS` to 4, which under a three-card mask
would address a device that does not exist. The preflight prints the visible
device count so a mismatch shows up before the driver detaches.

```bash
GPUS=0,1 ./run.sh start        # use two available cards
GPUS=0 ./run.sh start          # pin one card
```

There is no mid-run tolerance for a card dying: a CUDA error propagates out of
the worker, the sweep exits non-zero and the stage is recorded as `failed:`.
What limits the damage is that finished cells are already on disk and `start`
resumes past them, so a fault costs only the cells in flight. Masking the bad
card out is what prevents the fault, not recovering from it.

## Two things not to do

* Never `rsync --delete` into the server. `./run.sh push` takes explicit paths
  and does not delete; results live only on the server, and `experiments/runs/`
  is not in this repo, so a mirroring sync removes them.
* Don't throttle or reset the GPUs while a sweep is running.

See `_archive_2026-09-14/README.md` for what the previous scripts got wrong.
