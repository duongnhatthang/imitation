"""Sacred entry point for a single Atari experiment run.

Runs ONE (algorithm, game, seed) combination end-to-end and writes Sacred
output to an isolated FileStorageObserver directory:
    {output_dir}/{algo}/{game}/{seed}/

Usage (Sacred CLI syntax):
    python experiments/run_atari_experiment.py with algo=dagger game=Pong seed=0

Usage (pre-parser + Sacred combined, e.g. from run_atari_benchmark.sh):
    python experiments/run_atari_experiment.py --algo dagger --game Pong --seed 0 \\
        with algo=dagger game=Pong seed=0 n_rounds=20 total_timesteps=500000

IMPORTANT: Pre-parser defaults MUST match Sacred config defaults to prevent silent
directory misalignment. If a user passes only Sacred `with algo=X` syntax (no
--algo flag), the pre-parser uses its default, which must equal the Sacred default.
Both sets of defaults: algo="dagger", game="Pong", seed=0, output_dir="output/sacred".
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path (same pattern as atari_smoke.py).
# This allows both `python experiments/run_atari_experiment.py` and
# `python -m experiments.run_atari_experiment` to resolve all imports.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import sacred
from sacred.observers import FileStorageObserver

ex = sacred.Experiment("atari_run")


@ex.config
def cfg():
    # CRITICAL: These defaults MUST match the pre-parser defaults in __main__
    # to prevent silent observer-path misalignment when only Sacred `with` syntax
    # is used (no --algo/--game/--seed flags). See module docstring.
    algo = "dagger"          # One of: bc, dagger, ftrl
    game = "Pong"            # One of the 7 ATARI_GAMES keys
    seed = 0
    n_rounds = 20
    total_timesteps = 500000
    n_envs = 8
    alpha = 1.0              # FTRL only — ignored for bc/dagger
    output_dir = "output/sacred"


@ex.main
def run(algo, game, seed, n_rounds, total_timesteps, n_envs, alpha, output_dir, _run):
    """Run a single (algo, game, seed) experiment and return normalized score.

    Args:
        algo: Algorithm name: "bc", "dagger", or "ftrl".
        game: Atari game display name, e.g. "Pong".
        seed: Random seed.
        n_rounds: Number of DAgger/FTRL rounds (ignored for BC).
        total_timesteps: Total environment timesteps (DAgger/FTRL only).
        n_envs: Number of parallel environments.
        alpha: FTRL step-size scale (FTRL only).
        output_dir: Base directory for Sacred FileStorageObserver output.
        _run: Sacred run object for metric logging.

    Returns:
        Dict with "normalized_score" key.
    """
    # Import here (inside Sacred main) so Sacred captures the experiment source
    from experiments.atari_smoke import load_random_baselines, run_bc, run_dagger
    from experiments.atari_helpers import ATARI_GAMES
    import numpy as np

    game_id = ATARI_GAMES[game]
    baselines = load_random_baselines()
    baseline_entry = baselines[game]
    # baselines values may be dicts ({"mean": ..., "std": ...}) or plain floats
    random_score = (
        baseline_entry["mean"] if isinstance(baseline_entry, dict) else float(baseline_entry)
    )

    rng = np.random.default_rng(seed)

    if algo == "bc":
        # BC is a single-round algorithm; log result as round-0 entry (INFRA-03,
        # Pitfall 6: BC has no DAgger rounds, so we treat it as round 0).
        norm_score = run_bc(game, game_id, seed, n_envs, random_score, rng)
        _run.log_scalar("normalized_score", norm_score, step=0)
    elif algo in ("dagger", "ftrl"):
        use_ftrl = (algo == "ftrl")
        norm_score = run_dagger(
            game,
            game_id,
            seed,
            n_envs,
            n_rounds,
            total_timesteps,
            random_score,
            rng,
            use_ftrl=use_ftrl,
            alpha=alpha,
        )
        _run.log_scalar("normalized_score", norm_score)
    else:
        raise ValueError(f"Unknown algo '{algo}'. Must be one of: bc, dagger, ftrl")

    if norm_score is None:
        print(f"WARNING: {algo} on {game} seed={seed} returned None (run failed internally)")
    else:
        print(f"[{algo}/{game}/seed={seed}] normalized_score={norm_score:.4f}")

    return {"normalized_score": norm_score}


if __name__ == "__main__":
    # Pre-parse --algo, --game, --seed, --output-dir BEFORE Sacred consumes sys.argv,
    # so we can build the observer directory path.
    #
    # CRITICAL: Pre-parser defaults MUST match Sacred @ex.config defaults above.
    # If Sacred `with algo=X` is used without --algo, the pre-parser sees its
    # default. The directory will be wrong if the two differ.
    #   Pre-parser defaults: algo="dagger", game="Pong", seed=0, output_dir="output/sacred"
    #   Sacred config defaults: algo="dagger", game="Pong", seed=0, output_dir="output/sacred"
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--algo", default="dagger")
    pre.add_argument("--game", default="Pong")
    pre.add_argument("--seed", type=int, default=0)
    pre.add_argument("--output-dir", default="output/sacred")
    known, _ = pre.parse_known_args()

    # Build isolated observer path: {output_dir}/{algo}/{game}/{seed}
    # Each (algo, game, seed) triple gets its own directory, so Sacred run IDs
    # (assigned per directory) never collide across parallel runs (INFRA-04).
    obs_path = f"{known.output_dir}/{known.algo}/{known.game}/{known.seed}"
    ex.observers.append(FileStorageObserver(obs_path))

    ex.run_commandline()
