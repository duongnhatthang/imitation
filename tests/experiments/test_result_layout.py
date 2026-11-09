"""Launcher path contracts without running learning experiments."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_paths_share_one_named_run(tmp_path):
    env = dict(os.environ, EXP_RUN_ID="trial", EXP_RUNS_DIR=str(tmp_path))
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source experiments/paths.sh; printf "%s\\n" "$EXP_LC_CLASSICAL" "$EXP_LC_ATARI" "$EXP_PLOTS_CLASSICAL" "$EXP_PLOTS_ATARI" "$EXP_LOG_DIR"',
        ],
        cwd=ROOT,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )
    assert result.stdout.splitlines() == [
        str(tmp_path / "trial/data/classical"),
        str(tmp_path / "trial/data/atari"),
        str(tmp_path / "trial/plots/classical"),
        str(tmp_path / "trial/plots/atari"),
        str(tmp_path / "trial/logs"),
    ]


@pytest.mark.parametrize(
    "script,group",
    [("run_learning_curves.sh", "classical"), ("run_atari_curves.sh", "atari")],
)
def test_launchers_train_and_plot_in_same_run(tmp_path, script, group):
    project = tmp_path / "project"
    (project / "experiments").mkdir(parents=True)
    for name in ("paths.sh", script):
        shutil.copy2(ROOT / "experiments" / name, project / "experiments" / name)
    bindir = tmp_path / "bin"
    bindir.mkdir()
    calls = tmp_path / "calls.jsonl"
    fake = bindir / "python"
    fake.write_text(
        '#!/usr/bin/env python3\nimport json,os,sys\nfrom pathlib import Path\nwith open(os.environ["TEST_CALLS"],"a") as f: f.write(json.dumps(sys.argv[1:])+"\\n")\nif "--output-dir" in sys.argv: Path(sys.argv[sys.argv.index("--output-dir")+1]).mkdir(parents=True,exist_ok=True)\n'
    )
    fake.chmod(0o755)
    env = dict(
        os.environ,
        PATH=str(bindir) + os.pathsep + os.environ["PATH"],
        EXP_RUN_ID="trial",
        EXP_RUNS_DIR=str(tmp_path / "runs"),
        TEST_CALLS=str(calls),
        N_WORKERS="1",
    )
    result = subprocess.run(
        ["bash", str(project / "experiments" / script)],
        cwd=project,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    commands = [json.loads(line) for line in calls.read_text().splitlines()]
    train, plot = commands

    def option(cmd, flag):
        return cmd[cmd.index(flag) + 1]

    root = tmp_path / "runs/trial"
    assert option(train, "--output-dir") == str(root / "data" / group)
    assert option(plot, "--results-dir") == str(root / "data" / group)
    assert option(plot, "--output-dir") == str(
        root / "plots" / group / "learning_curves"
    )
    assert "--flat-output" in plot
    assert (root / "logs" / f"{group}.log").is_file()


@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("run_atari", [False, True])
def test_driver_runs_classical_before_atari_and_stops_on_failure(
    tmp_path, fail, run_atari
):
    project = tmp_path / "project"
    (project / "experiments").mkdir(parents=True)
    for name in ("paths.sh", "run_bc_iid_experiments.sh"):
        shutil.copy2(ROOT / "experiments" / name, project / "experiments" / name)
    for name, stage in [
        ("run_learning_curves.sh", "classical"),
        ("run_atari_curves.sh", "atari"),
        ("gen_coverage_viz.sh", "coverage"),
    ]:
        (project / "experiments" / name).write_text(
            f'#!/bin/bash\necho {stage} >> "$TEST_EVENTS"\n'
            + (
                'test "$ENVS" = PongNoFrameskip-v4 || exit 9\n'
                if stage == "atari"
                else ""
            )
            + (
                'if [ "${FAIL_CLASSICAL:-0}" = 1 ]; then exit 7; fi\n'
                if stage == "classical"
                else ""
            )
        )
    bindir = tmp_path / "bin"
    bindir.mkdir()
    fake = bindir / "python"
    fake.write_text(
        '#!/usr/bin/env python3\nimport os,sys\nif sys.argv[1:]==["-"]: os.execv(sys.executable,[sys.executable,"-"])\nif "-c" in sys.argv: print("CartPole-v1")\n'
    )
    fake.chmod(0o755)
    events = tmp_path / "events"
    env = dict(
        os.environ,
        PATH=str(bindir) + os.pathsep + os.environ["PATH"],
        EXP_RUN_ID="trial",
        EXP_RUNS_DIR=str(tmp_path / "runs"),
        TEST_EVENTS=str(events),
        FAIL_CLASSICAL=str(int(fail)),
        RUN_ATARI=str(int(run_atari)),
        ENVS="CartPole-v1",
        ATARI_ENVS="PongNoFrameskip-v4",
    )
    result = subprocess.run(
        ["bash", str(project / "experiments/run_bc_iid_experiments.sh")],
        cwd=project,
        env=env,
        capture_output=True,
        text=True,
    )
    expected = ["classical"] if fail else ["classical", "coverage"]
    if run_atari and not fail:
        expected += ["atari", "coverage"]
    assert events.read_text().splitlines() == expected, result.stdout + result.stderr
    assert result.returncode == (7 if fail else 0), result.stdout + result.stderr
    status = (tmp_path / "runs/trial/status.txt").read_text()
    assert ("failed" in status) if fail else status.strip() == "complete"
    manifest = json.loads((tmp_path / "runs/trial/manifest.json").read_text())
    assert manifest["run_atari"] is run_atari
    assert manifest["warm_start"] is False
    assert manifest["dagger_beta"] == 0
    assert manifest["algorithms"] == ["ftl", "ftrl", "bc", "bc_iid"]


def test_coverage_uses_run_plots_seed_count_and_shared_cache(tmp_path):
    project = tmp_path / "project"
    (project / "experiments").mkdir(parents=True)
    shutil.copy2(ROOT / "experiments/gen_coverage_viz.sh", project / "experiments")
    bindir = tmp_path / "bin"
    bindir.mkdir()
    calls = tmp_path / "calls.jsonl"
    fake = bindir / "python"
    fake.write_text(
        '#!/usr/bin/env python3\nimport json,os,sys\nwith open(os.environ["TEST_CALLS"],"a") as f: f.write(json.dumps(sys.argv[1:])+"\\n")\n'
    )
    fake.chmod(0o755)
    run = tmp_path / "runs/trial"
    env = dict(
        os.environ,
        PATH=str(bindir) + os.pathsep + os.environ["PATH"],
        TEST_CALLS=str(calls),
        SEEDS="1",
        EXP_EXPERT_CACHE="custom-cache",
    )
    for key in ("COVERAGE_SEEDS", "EXPERT_CACHE", "PLOT_ROOT"):
        env.pop(key, None)
    subprocess.run(
        [
            "bash",
            str(project / "experiments/gen_coverage_viz.sh"),
            str(run / "data/classical"),
            "CartPole-v1",
        ],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    commands = [json.loads(line) for line in calls.read_text().splitlines()]
    assert len(commands) == 2
    coverage, runtime = commands
    assert coverage[coverage.index("--seed") + 1] == "0"
    assert coverage[coverage.index("--expert-cache") + 1] == "custom-cache"
    assert coverage[coverage.index("--output-dir") + 1] == str(
        run / "plots/classical/coverage"
    )
    assert runtime[runtime.index("--output-dir") + 1] == str(
        run / "plots/classical/runtime"
    )
