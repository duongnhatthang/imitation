"""The review must select current figures without duplicating every seed."""

import importlib.util
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "compile_results_pdf",
    ROOT / "experiments/compile_results_pdf.py",
)
compiler = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compiler)


def test_default_coverage_launcher_paths_are_included(tmp_path):
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; printf "%s" "$EXP_PLOTS_COVERAGE"',
            "paths-test",
            str(ROOT / "experiments/paths.sh"),
        ],
        env={"PATH": os.environ["PATH"], "EXP_RUN_ROOT": str(tmp_path)},
        capture_output=True,
        text=True,
        check=True,
    )
    plots = Path(result.stdout)
    for relative in (
        "learning_curves/CartPole-v1.png",
        "coverage/CartPole-v1_seed0.png",
    ):
        path = plots / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    pages = compiler.figure_pages(tmp_path / "plots")
    assert [(section, family) for section, family, _ in pages] == [
        ("Learning curves", "classical"),
        ("Coverage", "classical"),
    ]


def test_current_selection_keeps_width_overview_and_one_coverage_seed(tmp_path):
    names = [
        "classical/learning_curves/student-width/Task/Task__all-widths.png",
        "classical/learning_curves/student-width/Task/Task__student-width-32.png",
        "classical/coverage/Task/Task_seed3100.png",
        "classical/coverage/Task/Task_seed3100_coverage_diff.png",
        "classical/coverage/Task/Task_seed3101.png",
        "classical/coverage/Task/Task_seed3101_coverage_diff.png",
        "archive/classical/learning_curves/old.png",
    ]
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    pages = compiler.figure_pages(tmp_path)
    assert [(section, len(images)) for section, _, images in pages] == [
        ("Learning curves", 1),
        ("Coverage", 2),
    ]
    included = {
        p.relative_to(tmp_path).as_posix() for _, _, paths in pages for p in paths
    }
    assert included == {names[0], names[2], names[3]}


def test_pdf_is_repeatable_and_preserves_previous_bytes(tmp_path):
    import hashlib

    from PIL import Image

    image = tmp_path / "classical/learning_curves/Task.png"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (240, 120), "blue").save(image)
    output = compiler.build_report(tmp_path, "trial")
    before = output.read_bytes()
    assert before.startswith(b"%PDF")
    compiler.build_report(tmp_path, "trial")
    assert output.read_bytes() == before
    assert not list(tmp_path.glob("archive/**/*.pdf"))
    Image.new("RGB", (240, 120), "red").save(image)
    compiler.build_report(tmp_path, "trial")
    archive = (
        tmp_path
        / "archive/previous-publications"
        / hashlib.sha256(before).hexdigest()
        / output.name
    )
    assert archive.read_bytes() == before
    assert output.read_bytes() != before
