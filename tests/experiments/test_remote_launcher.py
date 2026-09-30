"""Exercise the remote launcher through a local transport substitute."""

import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("refuse", [False, True])
@pytest.mark.parametrize("custom_settings", [False, True])
@pytest.mark.parametrize("remote_repo", ["imitation", "imitation workspace"])
def test_start_uses_tmux_and_preserves_campaign_settings(
    tmp_path, refuse, custom_settings, remote_repo
):
    project = tmp_path / "project"
    project.mkdir()
    shutil.copy2(ROOT / "run.sh", project / "run.sh")
    remote = tmp_path / "remote"
    experiments = remote / remote_repo / "experiments"
    experiments.mkdir(parents=True)
    conda = remote / "miniconda3/etc/profile.d/conda.sh"
    conda.parent.mkdir(parents=True)
    conda.write_text("conda() { return 0; }\n")
    bindir = tmp_path / "bin"
    bindir.mkdir()
    scripts = {
        # OpenSSH joins command arguments, then the remote shell parses them.
        # Executing argv directly would hide lost empty and multiword values.
        "ssh": """#!/bin/bash
while [ "$1" = -o ]; do shift 2; done
shift
remote_command="$*"
unset ALGOS SEEDS ENVS CLASSICAL_WORKERS ATARI_ENVS RUN_ATARI GPUS CUDA_VISIBLE_DEVICES
exec bash -c "$remote_command"
""",
        "pgrep": '#!/bin/bash\n[ "$REFUSE" = 1 ]\n',
        "sleep": "#!/bin/bash\nexit 0\n",
        "python": '#!/bin/bash\nif [ "$1" = -c ]; then echo 2; else cat >/dev/null; fi\n',
        "tmux": """#!/bin/bash
case "$1" in
  has-session) test -f "$TMUX_CAPTURE" ;;
  new-session) printf '%s\\n' "${@: -1}" > "$TMUX_CAPTURE" ;;
  display-message) echo 123 ;;
esac
""",
    }
    for name, body in scripts.items():
        path = bindir / name
        path.write_text(body)
        path.chmod(0o755)
    capture = tmp_path / "tmux-command"
    env = dict(
        os.environ,
        HOME=str(remote),
        PATH=str(bindir) + os.pathsep + os.environ["PATH"],
        HOST="test-host",
        RUN_ID="test-run",
        REMOTE_REPO=remote_repo,
        RUN_ATARI="0",
        GPUS="",
        CUDA_VISIBLE_DEVICES="1",
        TMUX_CAPTURE=str(capture),
        REFUSE=str(int(refuse)),
    )
    for key in ("ALGOS", "SEEDS", "ENVS", "CLASSICAL_WORKERS", "ATARI_ENVS"):
        env.pop(key, None)
    if custom_settings:
        env.update(
            ENVS="CartPole-v1 Acrobot-v1",
            ALGOS="ftl bc bc_iid",
            SEEDS="2",
            CLASSICAL_WORKERS="4",
            ATARI_ENVS="PongNoFrameskip-v4 BreakoutNoFrameskip-v4",
        )
    result = subprocess.run(
        ["bash", str(project / "run.sh"), "start"],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == (3 if refuse else 0), result.stdout + result.stderr
    if refuse:
        assert not capture.exists()
        return
    command = shlex.split(capture.read_text())
    assert command[:3] == ["env", "-u", "CUDA_VISIBLE_DEVICES"]
    assert "RUN_ATARI=0" in command
    if custom_settings:
        assert "ENVS=CartPole-v1 Acrobot-v1" in command
        assert "ALGOS=ftl bc bc_iid" in command
        assert "SEEDS=2" in command
        assert "CLASSICAL_WORKERS=4" in command
        assert "ATARI_ENVS=PongNoFrameskip-v4 BreakoutNoFrameskip-v4" in command
    else:
        assert not any(
            arg.startswith(("ALGOS=", "SEEDS=", "ENVS=", "CLASSICAL_WORKERS="))
            for arg in command
        )
        assert "ATARI_ENVS=" in command
    assert "N_GPUS=2" in command
    assert not any(arg.startswith("CUDA_VISIBLE_DEVICES=") for arg in command)
    assert str(experiments / "run_bc_iid_experiments.sh") in command


@pytest.mark.parametrize("conda_available", [False, True])
@pytest.mark.parametrize("stage", ["classical_training", "complete"])
def test_status_uses_conda_and_reports_classical_progress(
    tmp_path, conda_available, stage
):
    project = tmp_path / "project"
    project.mkdir()
    shutil.copy2(ROOT / "run.sh", project / "run.sh")
    remote = tmp_path / "remote"
    run = remote / "imitation workspace/experiments/runs/test-run"
    run.mkdir(parents=True)
    (run / "status.txt").write_text(stage)
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "algorithms": ["ftl", "bc_iid"],
                "seeds": 3,
                "classical_envs": ["CartPole-v1"],
                "run_atari": False,
            }
        )
    )
    bindir = tmp_path / "bin"
    bindir.mkdir()
    conda_bin = tmp_path / "conda-bin"
    conda_bin.mkdir()
    used_conda = tmp_path / "used-conda-python"
    used_system = tmp_path / "used-system-python"
    scripts = {
        bindir
        / "ssh": '#!/bin/bash\nwhile [ "$1" = -o ]; do shift 2; done\nshift\nexec bash -c "$*"\n',
        bindir / "conda": "#!/bin/bash\nexit 1\n",
        bindir / "pgrep": "#!/bin/bash\nexit 1\n",
        bindir / "nvidia-smi": "#!/bin/bash\nexit 1\n",
        bindir / "tmux": "#!/bin/bash\nexit 1\n",
        bindir / "python3": '#!/bin/bash\ntouch "$SYSTEM_MARKER"\nexit 92\n',
        conda_bin / "python": '#!/bin/bash\ntouch "$CONDA_MARKER"\nexec '
        + shlex.quote(sys.executable)
        + ' "$@"\n',
    }
    for path, body in scripts.items():
        path.write_text(body)
        path.chmod(0o755)
    if conda_available:
        conda = remote / "miniconda3/etc/profile.d/conda.sh"
        conda.parent.mkdir(parents=True)
        conda.write_text(
            'conda() { [ "$1" = activate ] && [ "$2" = requested-env ] || return 1; '
            'export PATH="$CONDA_TEST_BIN:$PATH"; }\n'
        )
    env = dict(
        os.environ,
        HOME=str(remote),
        PATH=str(bindir) + os.pathsep + os.environ["PATH"],
        HOST="test-host",
        RUN_ID="test-run",
        REMOTE_REPO="imitation workspace",
        CONDA_ENV="requested-env",
        RUN_ATARI="0",
        CONDA_TEST_BIN=str(conda_bin),
        CONDA_MARKER=str(used_conda),
        SYSTEM_MARKER=str(used_system),
    )
    result = subprocess.run(
        ["bash", str(project / "run.sh"), "status"],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert used_conda.exists() is conda_available
    assert not used_system.exists()
    if conda_available:
        assert "classical cells   0 / 6" in result.stdout
    assert "atari sweep complete" not in result.stdout
    if stage == "complete":
        assert "campaign complete" in result.stdout
    else:
        assert "campaign complete" not in result.stdout
