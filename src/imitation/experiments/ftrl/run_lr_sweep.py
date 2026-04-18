"""Learning-rate calibration sweep for FTL+DAgger.

Finds the best learning rate per environment by sweeping LR values and
detecting when the disagreement rate saturates (T_sat).  Results are saved
as a calibration JSON that ``run_experiment.py`` can load to pick the best
LR for each env automatically.

Usage:
    # CartPole smoke test
    python -m imitation.experiments.ftrl.run_lr_sweep \
        --envs CartPole-v1 --seeds 3 --n-workers 1

    # All classical MDPs on 4 GPUs
    python -m imitation.experiments.ftrl.run_lr_sweep \
        --env-group classical --n-workers 4 --n-gpus 4
"""

import argparse
import copy
import json
import logging
import multiprocessing
import os
import pathlib
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from imitation.experiments.ftrl import env_utils
from imitation.experiments.ftrl.run_experiment import (
    ExperimentConfig,
    _run_single_wrapper,
    _worker_init,
    resolve_envs,
)

logger = logging.getLogger(__name__)

# LR values to sweep — biased upward since default 1e-3 is too small.
DEFAULT_LR_VALUES = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]

# Sweep uses more rounds than production to give slower LRs time to converge.
DEFAULT_N_ROUNDS = 60
DEFAULT_SAMPLES_PER_ROUND = 50  # small for speed; saturation is LR-dependent


def detect_t_sat(
    disagreement_rates: List[Optional[float]],
    rounds: List[int],
    patience: int = 5,
    rel_delta: float = 0.02,
) -> Tuple[Optional[int], Optional[float]]:
    """Detect the round at which disagreement rate saturates.

    Saturation is defined as the first eval point after which the metric
    does not improve by more than ``rel_delta`` (relative) for
    ``patience`` consecutive eval points.

    Returns:
        (t_sat_round, saturated_value) or (None, None) if not saturated.
    """
    # Filter to evaluated rounds only (non-None disagreement_rate)
    vals = [
        (r, d)
        for r, d in zip(rounds, disagreement_rates)
        if d is not None
    ]
    if len(vals) < patience + 1:
        return None, None

    for i in range(len(vals) - patience):
        base_round, base_val = vals[i]
        # Check if all subsequent `patience` points are within rel_delta
        saturated = True
        for j in range(1, patience + 1):
            _, later_val = vals[i + j]
            if base_val > 0 and abs(later_val - base_val) / base_val > rel_delta:
                saturated = False
                break
        if saturated:
            return base_round, base_val

    return None, None


def detect_best_value(
    disagreement_rates: List[Optional[float]],
) -> Optional[float]:
    """Return the minimum (best) disagreement rate observed."""
    valid = [d for d in disagreement_rates if d is not None]
    return min(valid) if valid else None


def analyze_sweep_results(
    results: List[Dict[str, Any]],
    lr_values: List[float],
) -> Dict[str, Any]:
    """Analyze sweep results and pick the best LR per environment.

    Best LR = lowest final disagreement rate.  Ties broken by faster T_sat.

    Returns a calibration dict keyed by env name.
    """
    # Group by env
    by_env: Dict[str, Dict[float, List[Dict]]] = {}
    for r in results:
        if "error" in r:
            continue
        env = r["env"]
        lr = r["config"]["learning_rate"]
        by_env.setdefault(env, {}).setdefault(lr, []).append(r)

    calibration: Dict[str, Any] = {}
    for env_name, lr_results in sorted(by_env.items()):
        lr_summaries = []
        for lr in sorted(lr_results.keys()):
            seed_results = lr_results[lr]
            # Aggregate across seeds
            all_disagree = []
            all_t_sat = []
            all_best = []
            for sr in seed_results:
                per_round = sr["per_round"]
                rounds = [p["round"] for p in per_round]
                disagree = [p.get("disagreement_rate") for p in per_round]
                t_sat, sat_val = detect_t_sat(disagree, rounds)
                best_val = detect_best_value(disagree)
                all_disagree.append(disagree)
                all_t_sat.append(t_sat)
                all_best.append(best_val)

            # Mean best disagreement across seeds
            valid_best = [b for b in all_best if b is not None]
            mean_best = float(np.mean(valid_best)) if valid_best else None
            # Median T_sat (None if majority didn't saturate)
            valid_t_sat = [t for t in all_t_sat if t is not None]
            median_t_sat = (
                int(np.median(valid_t_sat))
                if len(valid_t_sat) > len(all_t_sat) / 2
                else None
            )
            lr_summaries.append({
                "lr": lr,
                "mean_best_disagreement": mean_best,
                "median_t_sat": median_t_sat,
                "n_seeds": len(seed_results),
                "n_saturated": len(valid_t_sat),
                "per_seed_best": all_best,
                "per_seed_t_sat": all_t_sat,
            })

        # Pick best LR: lowest mean_best_disagreement, tie-break by earliest T_sat
        ranked = sorted(
            lr_summaries,
            key=lambda s: (
                s["mean_best_disagreement"] if s["mean_best_disagreement"] is not None else 1e9,
                s["median_t_sat"] if s["median_t_sat"] is not None else 1e9,
            ),
        )
        best = ranked[0]
        calibration[env_name] = {
            "best_lr": best["lr"],
            "best_disagreement": best["mean_best_disagreement"],
            "best_t_sat": best["median_t_sat"],
            "all_lr_results": lr_summaries,
        }

    return calibration


def print_sweep_summary(calibration: Dict[str, Any]) -> None:
    """Print a human-readable summary table."""
    print("\n" + "=" * 90)
    print("LR SWEEP RESULTS")
    print("=" * 90)

    for env_name, cal in sorted(calibration.items()):
        print(f"\n{'─' * 90}")
        print(f"  {env_name}")
        print(f"  Best LR: {cal['best_lr']:.0e}  |  "
              f"Best disagreement: {cal['best_disagreement']:.4f}  |  "
              f"T_sat: {cal['best_t_sat']}")
        print(f"  {'LR':<10} {'Best Disagree':<18} {'T_sat':<10} "
              f"{'Saturated':<12} {'Seeds'}")
        print(f"  {'─' * 70}")
        for s in cal["all_lr_results"]:
            best_d = f"{s['mean_best_disagreement']:.4f}" if s["mean_best_disagreement"] is not None else "N/A"
            t_sat = str(s["median_t_sat"]) if s["median_t_sat"] is not None else "N/A"
            print(f"  {s['lr']:<10.0e} {best_d:<18} {t_sat:<10} "
                  f"{s['n_saturated']}/{s['n_seeds']:<10} {s['n_seeds']}")

    print("\n" + "=" * 90)


def main():
    parser = argparse.ArgumentParser(
        description="LR calibration sweep for FTL+DAgger",
    )
    parser.add_argument("--envs", nargs="+", default=None)
    parser.add_argument(
        "--env-group", type=str, default=None,
        choices=list(env_utils.ENV_GROUPS.keys()),
    )
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--n-rounds", type=int, default=DEFAULT_N_ROUNDS)
    parser.add_argument(
        "--samples-per-round", type=int, default=DEFAULT_SAMPLES_PER_ROUND,
    )
    parser.add_argument(
        "--lr-values", type=float, nargs="+", default=DEFAULT_LR_VALUES,
        help="Learning rates to sweep",
    )
    parser.add_argument("--bc-n-epochs", type=int, default=20)
    parser.add_argument(
        "--policy-mode", choices=["end_to_end", "linear"], default="linear",
    )
    parser.add_argument("--eval-interval", type=int, default=1,
                        help="Evaluate every N rounds (default: 1 for dense tracking)")
    parser.add_argument(
        "--output-dir", type=str, default="experiments/lr_sweep",
    )
    parser.add_argument(
        "--expert-cache-dir", type=str, default="experiments/expert_cache",
    )
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--n-gpus", type=int, default=0)
    parser.add_argument(
        "--calibration-file", type=str,
        default="experiments/lr_calibration.json",
        help="Path to save/update calibration results",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    envs = resolve_envs(args.env_group, args.envs)
    logger.info(
        f"LR sweep: {len(envs)} envs × {len(args.lr_values)} LRs × "
        f"{args.seeds} seeds = {len(envs) * len(args.lr_values) * args.seeds} configs"
    )
    logger.info(f"LR values: {args.lr_values}")
    logger.info(f"Rounds: {args.n_rounds}, samples/round: {args.samples_per_round}")

    # Build configs: FTL only, one per (env, lr, seed)
    all_configs = []
    for env_name in envs:
        for lr in args.lr_values:
            for seed in range(args.seeds):
                all_configs.append(
                    ExperimentConfig(
                        algo="ftl",
                        env_name=env_name,
                        seed=seed,
                        policy_mode=args.policy_mode,
                        n_rounds=args.n_rounds,
                        samples_per_round=args.samples_per_round,
                        l2_lambda=0.0,
                        l2_decay=False,
                        warm_start=True,
                        beta_rampdown=15,
                        bc_n_epochs=args.bc_n_epochs,
                        eval_interval=args.eval_interval,
                        output_dir=pathlib.Path(args.output_dir),
                        expert_cache_dir=pathlib.Path(args.expert_cache_dir),
                        learning_rate=lr,
                    )
                )

    # Use result_name_override to separate files by LR value.
    for cfg in all_configs:
        object.__setattr__(
            cfg, "result_name_override", f"ftl_lr{cfg.learning_rate:.0e}"
        )

    # Filter already-done configs
    from imitation.experiments.ftrl.run_experiment import _is_already_done

    total_requested = len(all_configs)
    configs = [c for c in all_configs if not _is_already_done(c)]
    skipped = total_requested - len(configs)
    logger.info(
        f"Running {len(configs)} new experiments "
        f"({skipped} already cached, {total_requested} total)"
    )

    if not configs:
        logger.info("All experiments already cached.")
    else:
        # Set GPU env var
        os.environ["FTRL_N_GPUS"] = str(args.n_gpus)

        start = time.time()
        if args.n_workers <= 1:
            _worker_init(multiprocessing.Queue())
            results = [_run_single_wrapper(c) for c in configs]
        else:
            ctx = multiprocessing.get_context("spawn")
            gpu_queue = ctx.Queue()
            if args.n_gpus > 0:
                for w in range(args.n_workers):
                    gpu_queue.put(w % args.n_gpus)
            else:
                for w in range(args.n_workers):
                    gpu_queue.put(None)

            with ctx.Pool(
                args.n_workers,
                initializer=_worker_init,
                initargs=(gpu_queue,),
            ) as pool:
                done = 0
                results = []
                for r in pool.imap_unordered(_run_single_wrapper, configs):
                    done += 1
                    elapsed = (time.time() - start) / 60
                    rate = done / elapsed if elapsed > 0 else 0
                    eta = (len(configs) - done) / rate if rate > 0 else 0
                    logger.info(
                        f"Progress: {done}/{len(configs)} done | "
                        f"elapsed {elapsed:.1f}m | "
                        f"ETA {eta:.1f}m"
                    )
                    results.append(r)

        elapsed = time.time() - start
        n_errors = sum(1 for r in results if "error" in r)
        logger.info(
            f"Done: {len(results)} completed, {n_errors} errors, "
            f"{elapsed:.0f}s total"
        )
        for r in results:
            if "error" in r:
                logger.error(f"  FAILED: {r}")

    # Load ALL results (cached + new) for analysis
    all_results = []
    output_dir = pathlib.Path(args.output_dir)
    for env_name in envs:
        env_dir = output_dir / env_name.replace("/", "_")
        if not env_dir.exists():
            continue
        for json_file in sorted(env_dir.glob("ftl_lr*.json")):
            with open(json_file) as f:
                all_results.append(json.load(f))

    if not all_results:
        logger.error("No results found!")
        return

    # Analyze and save calibration
    calibration = analyze_sweep_results(all_results, args.lr_values)
    print_sweep_summary(calibration)

    # Save/update calibration file
    cal_path = pathlib.Path(args.calibration_file)
    cal_path.parent.mkdir(parents=True, exist_ok=True)

    # Merge with existing calibration if present
    existing = {}
    if cal_path.exists():
        with open(cal_path) as f:
            existing = json.load(f)

    existing.update(calibration)

    with open(cal_path, "w") as f:
        json.dump(existing, f, indent=2)
    logger.info(f"Calibration saved to {cal_path}")


if __name__ == "__main__":
    main()
