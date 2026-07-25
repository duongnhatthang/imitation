# tests/experiments/test_aggregate_runtime.py
import json

import numpy as np
import pandas as pd

from imitation.experiments.ftrl import aggregate_runtime


def _write_result(root, env, algo, seed, elapsed):
    d = root / env
    d.mkdir(parents=True, exist_ok=True)
    payload = {
        "algo": algo,
        "env": env,
        "seed": seed,
        "policy_mode": "end_to_end",
        "elapsed_seconds": elapsed,
        "baselines": {"expert_return": 500.0},
        "per_round": [],
    }
    (d / f"{algo}_end_to_end_seed{seed}.json").write_text(json.dumps(payload))


def test_collect_runtimes_parses_and_groups(tmp_path):
    _write_result(tmp_path, "CartPole-v1", "ftrl", 0, 120.0)
    _write_result(tmp_path, "CartPole-v1", "bc", 0, 30.0)
    _write_result(tmp_path, "CartPole-v1", "ftrl", 1, 140.0)
    df = aggregate_runtime.collect_runtimes(tmp_path)
    assert set(df.columns) >= {"env", "algo", "seed", "elapsed_seconds"}
    assert len(df) == 3
    ftrl_mean = df[df.algo == "ftrl"]["elapsed_seconds"].mean()
    assert ftrl_mean == 130.0


def test_write_csv_and_plot(tmp_path):
    _write_result(tmp_path, "CartPole-v1", "ftrl", 0, 120.0)
    df = aggregate_runtime.collect_runtimes(tmp_path)
    csv_path = tmp_path / "runtime.csv"
    png_path = tmp_path / "runtime.png"
    aggregate_runtime.write_runtime_csv(df, csv_path)
    aggregate_runtime.plot_runtime_bar(df, png_path)
    assert csv_path.exists()
    assert png_path.exists()
    reloaded = pd.read_csv(csv_path)
    assert len(reloaded) == 1
