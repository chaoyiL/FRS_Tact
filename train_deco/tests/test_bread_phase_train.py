import os
from pathlib import Path
import subprocess

from train_deco.bread_phase.train import build_training_argv


def test_bread_training_argv_forces_single_two_phase_model_and_bread_augmentation():
    argv = build_training_argv([
        "--dataset-manifest",
        "/tmp/bread.json",
        "--run-id",
        "bread-test",
    ])

    assert "--bread-phase" in argv
    assert "--use-task-condition" in argv
    assert argv[argv.index("--augmentation-identity-probability") + 1] == "0.25"
    assert argv[argv.index("--augmentation-low-light-probability") + 1] == "0.0"
    assert argv[argv.index("--augmentation-mild-probability") + 1] == "0.75"
    brightness = argv.index("--augmentation-mild-brightness-range")
    assert argv[brightness + 1 : brightness + 3] == ["0.8", "1.2"]


def test_bread_launcher_dry_run_is_independent_and_prints_expected_contract():
    root = Path(__file__).parents[2]
    launcher = root / "train_deco" / "scripts" / "train_bread_phase.sh"
    result = subprocess.run(
        [
            "bash",
            str(launcher),
            "--manifest",
            "/tmp/bread.json",
            "--dry-run",
        ],
        cwd=root,
        env={**os.environ, "RUN_ID": "bread-phase-test"},
        text=True,
        capture_output=True,
        check=True,
    )

    assert "-m train_deco.bread_phase.train" in result.stdout
    assert "--augmentation-mild-brightness-range 0.8 1.2" in result.stdout
    assert "--use-task-condition" in result.stdout
    assert "--action-chunk-size 32" in result.stdout
    assert "train_deco/scripts/train.sh" not in result.stdout
