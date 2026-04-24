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

# LR values to sweep — covers the 1e-1..3 range chosen for the 2-D sweep
# of classical MDPs (CartPole, Blackjack, FrozenLake).
DEFAULT_LR_VALUES = [1e-1, 3e-1, 1.0, 3.0]

# Observations-per-update values to sweep.
DEFAULT_SAMPLES_PER_ROUND_VALUES = [1, 15, 30, 45]

# Total observation budget per cell. n_rounds is derived from this.
DEFAULT_TOTAL_OBS = 1000


def detect_t_sat(
    values: List[Optional[float]],
    n_observations: List[int],
    smooth_window: int = 20,
    rel_change_threshold: float = 0.30,
    metric_direction: str = "down",
) -> Tuple[Optional[int], Optional[float]]:
    """Detect the observation count at which a curve saturates (flattens).

    Strategy (backward, relative-change on sliding windows):
      1. Smooth the series with a moving average to remove noise.
      2. For each position, compute the relative change across a window:
         ``|max - min| / mean`` over the next ``smooth_window`` points.
      3. Walk backward from the end; a point is "flat" if the relative
         change in its window is below ``rel_change_threshold``.
      4. T_sat = earliest contiguous flat point.

    Convergence gate, direction-aware:
      - ``metric_direction="down"``: reject if ``final_mean > 0.8 * initial_mean``
        (disagreement-rate style — must have fallen meaningfully).
      - ``metric_direction="up"``: reject if ``final_mean < 1.2 * initial_mean``
        (normalized-return style — must have risen meaningfully).

    Args:
        values: Per-round metric values (None = not eval'd).
        n_observations: Per-round cumulative observation counts.
        smooth_window: Moving-average window for smoothing. May be reduced
            automatically when the number of valid points is small.
        rel_change_threshold: Maximum ``(max-min)/mean`` within a window
            to consider it flat.
        metric_direction: "down" or "up".

    Returns:
        (t_sat_obs, saturated_value) or (None, None) if not saturated.
    """
    vals = [
        (obs, d)
        for obs, d in zip(n_observations, values)
        if d is not None
    ]
    if len(vals) < smooth_window + 2:
        return None, None

    obs_arr = np.array([o for o, _ in vals], dtype=float)
    raw_arr = np.array([d for _, d in vals], dtype=float)

    initial_mean = float(np.mean(raw_arr[: min(3, len(raw_arr))]))
    final_mean = float(np.mean(raw_arr[-smooth_window:]))

    if metric_direction == "down":
        # Disagreement-rate style: must have fallen meaningfully.
        if final_mean > 0.8 * initial_mean:
            return None, None
    elif metric_direction == "up":
        # Normalized-return style: no convergence gate. A curve that saturates
        # at a low level is still saturated; a curve that starts near the
        # ceiling (fast learning) is still saturated. The flatness check
        # below handles both cases; only genuinely-oscillating curves return
        # None via the ``earliest_flat >= n - 2`` trailing check.
        pass
    else:
        raise ValueError(
            f"metric_direction must be 'up' or 'down', got {metric_direction!r}"
        )

    # Smooth
    kernel = np.ones(smooth_window) / smooth_window
    pad_l = smooth_window // 2
    pad_r = smooth_window - 1 - pad_l
    padded = np.pad(raw_arr, (pad_l, pad_r), mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")

    n = len(smoothed)
    min_start = smooth_window

    def _is_flat(idx: int) -> bool:
        end = min(idx + smooth_window, n)
        window = smoothed[idx:end]
        wmean = float(np.mean(window))
        if abs(wmean) < 1e-8:
            return True
        rel_change = (float(np.max(window)) - float(np.min(window))) / abs(wmean)
        return rel_change < rel_change_threshold

    earliest_flat = n - 1
    for i in range(n - 1, min_start - 1, -1):
        if _is_flat(i):
            earliest_flat = i
        else:
            break

    if earliest_flat >= n - 2 or earliest_flat < min_start:
        return None, None

    sat_obs = int(obs_arr[earliest_flat])
    sat_val = float(np.mean(smoothed[earliest_flat:]))
    return sat_obs, sat_val


def detect_best_value(
    disagreement_rates: List[Optional[float]],
) -> Optional[float]:
    """Return the minimum (best) disagreement rate observed."""
    valid = [d for d in disagreement_rates if d is not None]
    return min(valid) if valid else None


def analyze_sweep_results(
    results: List[Dict[str, Any]],
    lr_values: List[float],
    saturation_metric: str = "normalized_return",
) -> Dict[str, Any]:
    """Analyze sweep results and pick the best LR per environment.

    The saturation detector runs on the chosen ``saturation_metric``.
    "normalized_return" → direction="up"; "disagreement_rate" → direction="down".

    Best LR = lowest final disagreement rate on a per-seed basis (unchanged).
    Ties broken by faster T_sat on the chosen saturation metric.
    """
    direction = "up" if saturation_metric == "normalized_return" else "down"

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
            all_best = []
            for sr in seed_results:
                disagree = [p.get("disagreement_rate") for p in sr["per_round"]]
                all_best.append(detect_best_value(disagree))

            valid_best = [b for b in all_best if b is not None]
            mean_best = float(np.mean(valid_best)) if valid_best else None

            by_round: Dict[int, List[float]] = {}
            by_round_obs: Dict[int, List[int]] = {}
            for sr in seed_results:
                for p in sr["per_round"]:
                    r = p["round"]
                    d = p.get(saturation_metric)
                    if d is not None:
                        by_round.setdefault(r, []).append(d)
                        by_round_obs.setdefault(r, []).append(p["n_observations"])

            rounds_sorted = sorted(by_round.keys())
            mean_vals = [float(np.mean(by_round[r])) for r in rounds_sorted]
            mean_obs = [int(np.mean(by_round_obs[r])) for r in rounds_sorted]
            n_points = len(mean_vals)
            window = max(5, min(20, n_points // 3)) if n_points else 20
            t_sat, sat_val = detect_t_sat(
                mean_vals, mean_obs,
                smooth_window=window,
                metric_direction=direction,
            )

            lr_summaries.append({
                "lr": lr,
                "mean_best_disagreement": mean_best,
                "t_sat": t_sat,
                "sat_val": round(sat_val, 6) if sat_val is not None else None,
                "n_seeds": len(seed_results),
                "per_seed_best": all_best,
            })

        all_best_d = [
            s["mean_best_disagreement"]
            for s in lr_summaries
            if s["mean_best_disagreement"] is not None
        ]
        best_d_overall = min(all_best_d) if all_best_d else 1e9
        near_best_threshold = max(best_d_overall * 2, best_d_overall + 0.01)

        competitive = [
            s for s in lr_summaries
            if s["mean_best_disagreement"] is not None
            and s["mean_best_disagreement"] <= near_best_threshold
        ]
        if not competitive:
            competitive = lr_summaries

        ranked = sorted(
            competitive,
            key=lambda s: (
                s["t_sat"] if s["t_sat"] is not None else 1e9,
                s["mean_best_disagreement"] if s["mean_best_disagreement"] is not None else 1e9,
            ),
        )
        full_ranked = sorted(
            lr_summaries,
            key=lambda s: (
                s["t_sat"] if s["t_sat"] is not None else 1e9,
                s["mean_best_disagreement"] if s["mean_best_disagreement"] is not None else 1e9,
            ),
        )
        best = ranked[0]
        calibration[env_name] = {
            "best_lr": best["lr"],
            "best_disagreement": best["mean_best_disagreement"],
            "best_t_sat": best["t_sat"],
            "near_best_threshold": round(near_best_threshold, 6),
            "all_lr_results": lr_summaries,
            "ranked": full_ranked,
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
              f"T_sat: {cal['best_t_sat']}  |  "
              f"Near-best threshold: {cal['near_best_threshold']:.4f}")

        # Table sorted by LR value
        print(f"\n  By LR value:")
        print(f"  {'LR':<10} {'Best Disagree':<18} {'T_sat (obs)':<14} "
              f"{'Sat Level':<14} {'Seeds'}")
        print(f"  {'─' * 70}")
        for s in cal["all_lr_results"]:
            best_d = f"{s['mean_best_disagreement']:.4f}" if s["mean_best_disagreement"] is not None else "N/A"
            t_sat = str(s["t_sat"]) if s["t_sat"] is not None else "N/A"
            sat_v = f"{s['sat_val']:.4f}" if s.get("sat_val") is not None else "N/A"
            competitive = (
                s["mean_best_disagreement"] is not None
                and s["mean_best_disagreement"] <= cal["near_best_threshold"]
            )
            marker = " *" if competitive else ""
            print(f"  {s['lr']:<10.0e} {best_d:<18} {t_sat:<14} "
                  f"{sat_v:<14} {s['n_seeds']}{marker}")

        # Ranking: sorted by T_sat (fastest to reach good performance)
        print(f"\n  Ranking (fastest T_sat among near-best, marked with *):")
        print(f"  {'Rank':<6} {'LR':<10} {'T_sat (obs)':<14} "
              f"{'Best Disagree':<18} {'Competitive'}")
        print(f"  {'─' * 70}")
        for rank, s in enumerate(cal["ranked"], 1):
            best_d = f"{s['mean_best_disagreement']:.4f}" if s["mean_best_disagreement"] is not None else "N/A"
            t_sat = str(s["t_sat"]) if s["t_sat"] is not None else "N/A"
            competitive = (
                s["mean_best_disagreement"] is not None
                and s["mean_best_disagreement"] <= cal["near_best_threshold"]
            )
            tag = "yes *" if competitive else "no"
            winner = " << BEST" if rank == 1 and s["lr"] == cal["best_lr"] else ""
            print(f"  {rank:<6} {s['lr']:<10.0e} {t_sat:<14} "
                  f"{best_d:<18} {tag}{winner}")

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
    parser.add_argument(
        "--total-obs", type=int, default=DEFAULT_TOTAL_OBS,
        help="Total observation budget per cell. n_rounds = total_obs // samples_per_round",
    )
    parser.add_argument(
        "--samples-per-round-values", type=int, nargs="+",
        default=DEFAULT_SAMPLES_PER_ROUND_VALUES,
        help="Observations-per-update values to sweep (heatmap X axis)",
    )
    parser.add_argument(
        "--lr-values", type=float, nargs="+", default=DEFAULT_LR_VALUES,
        help="Learning rates to sweep (heatmap Y axis)",
    )
    parser.add_argument(
        "--saturation-metric",
        choices=["normalized_return", "disagreement_rate"],
        default="normalized_return",
        help="Which per-round metric to run saturation detection on",
    )
    parser.add_argument(
        "--subsample-strategy", choices=["uniform", "prefix"],
        default="uniform",
        help="Per-round data selection strategy passed to ExperimentConfig",
    )
    parser.add_argument("--bc-n-epochs", type=int, default=20)
    parser.add_argument(
        "--policy-mode", choices=["end_to_end", "linear"], default="linear",
    )
    parser.add_argument("--eval-interval", type=int, default=1,
                        help="(unused in 2-D sweep; see --eval-points-per-cell)")
    parser.add_argument(
        "--eval-points-per-cell", type=int, default=60,
        help=(
            "Target number of eval points per cell. Per-cell eval_interval "
            "is derived as max(1, n_rounds // this). 60 is enough for the "
            "saturation detector and keeps sp=1 wall-time manageable."
        ),
    )
    parser.add_argument(
        "--output-dir", type=str, default="experiments/lr_obs_sweep",
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
        f"2-D sweep: {len(envs)} envs × {len(args.lr_values)} LRs × "
        f"{len(args.samples_per_round_values)} sample counts × "
        f"{args.seeds} seeds = "
        f"{len(envs) * len(args.lr_values) * len(args.samples_per_round_values) * args.seeds} configs"
    )
    logger.info(f"LR values: {args.lr_values}")
    logger.info(f"Samples-per-round values: {args.samples_per_round_values}")
    logger.info(f"Total obs per cell: {args.total_obs}")
    logger.info(f"Saturation metric: {args.saturation_metric}")

    # Build configs: FTL only, one per (env, lr, sp, seed)
    all_configs = []
    for env_name in envs:
        for lr in args.lr_values:
            for sp in args.samples_per_round_values:
                n_rounds = max(1, args.total_obs // sp)
                # Cap eval points per cell so sp=1 (1000 rounds) doesn't
                # dominate wall-clock. 60 eval points is enough for the
                # saturation detector (window=5..20 on 60 points is fine).
                cell_eval_interval = max(1, n_rounds // args.eval_points_per_cell)
                for seed in range(args.seeds):
                    all_configs.append(
                        ExperimentConfig(
                            algo="ftl",
                            env_name=env_name,
                            seed=seed,
                            policy_mode=args.policy_mode,
                            n_rounds=n_rounds,
                            samples_per_round=sp,
                            l2_lambda=0.0,
                            l2_decay=False,
                            warm_start=True,
                            beta_rampdown=min(15, max(1, n_rounds // 4)),
                            bc_n_epochs=args.bc_n_epochs,
                            eval_interval=cell_eval_interval,
                            output_dir=pathlib.Path(args.output_dir),
                            expert_cache_dir=pathlib.Path(args.expert_cache_dir),
                            learning_rate=lr,
                            subsample_strategy=args.subsample_strategy,
                            bc_batch_size=min(32, sp),
                        )
                    )

    # Filename override: include sp so (lr, sp) cells don't clobber each other.
    for cfg in all_configs:
        object.__setattr__(
            cfg,
            "result_name_override",
            f"ftl_lr{cfg.learning_rate:.0e}_sp{cfg.samples_per_round:02d}",
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
    calibration = analyze_sweep_results(
        all_results, args.lr_values, saturation_metric=args.saturation_metric,
    )
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
