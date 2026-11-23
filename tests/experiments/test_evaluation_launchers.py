"""Capture real shell entry-point commands without launching experiments."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from imitation.experiments.ftrl import run_experiment

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "script", ["run_learning_curves.sh", "run_atari_curves.sh", "run_coverage_full.sh"]
)
@pytest.mark.parametrize("custom", [False, True])
def test_shell_sweeps_use_resolved_evaluation_defaults(
    tmp_path, monkeypatch, script, custom
):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    capture = tmp_path / "command.json"
    fake = bindir / "python"
    fake.write_text(
        f"#!{sys.executable}\nimport json,os,sys\n"
        "with open(os.environ['CAPTURE_COMMAND'],'w') as f: json.dump(sys.argv[1:],f)\n"
        "raise SystemExit(71)\n"
    )
    fake.chmod(0o755)
    sleep = bindir / "sleep"
    sleep.write_text("#!/bin/sh\nexec /bin/sleep 0.01\n")
    sleep.chmod(0o755)
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith("EXP_")
        and k not in ("EVAL_INTERVAL", "EVAL_INITIAL_ROUNDS", "EVAL_EPISODES")
    }
    env.update(
        PATH=str(bindir) + os.pathsep + env["PATH"],
        CAPTURE_COMMAND=str(capture),
        EXP_RUN_ID="probe",
        EXP_RUNS_DIR=str(tmp_path / "runs"),
        N_WORKERS="1",
        N_GPUS="0",
        ENVS="CartPole-v1",
        ALGOS="ftl",
    )
    extra = (
        [
            "--eval-interval",
            "13",
            "--eval-initial-rounds",
            "7",
            "--eval-episodes",
            "3",
            "--eval-only-checkpoints",
            "--full-checkpoints",
        ]
        if custom
        else []
    )
    result = subprocess.run(
        ["bash", str(ROOT / "experiments" / script), *extra],
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 71, result.stdout + result.stderr
    command = json.loads(capture.read_text())
    assert command[:2] == ["-m", "imitation.experiments.ftrl.run_experiment"]
    resolved = {}

    def capture_config(args):
        resolved.update(vars(args))
        raise SystemExit(0)

    monkeypatch.setattr(run_experiment, "build_configs", capture_config)
    monkeypatch.setattr(sys, "argv", ["run_experiment", *command[2:]])
    with pytest.raises(SystemExit) as stop:
        run_experiment.main()
    assert stop.value.code == 0
    assert resolved["eval_interval"] == (13 if custom else 20)
    assert resolved["eval_initial_rounds"] == (7 if custom else 60)
    assert resolved["eval_episodes"] == (3 if custom else 10)
    assert resolved["save_every_round"] is (not custom)
    assert resolved["compact_checkpoints"] is (not custom)
