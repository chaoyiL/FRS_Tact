import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEQUENCE = ROOT / "scripts" / "server_ljl_rdp_sequence.sh"
BREAD = ROOT / "scripts" / "server_ljl_bread_dual.sh"


def _fake_code_dir(tmp_path: Path, *, fail_press: bool = False) -> tuple[Path, Path]:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    log = tmp_path / "calls.log"
    failure = "" if not fail_press else 'if [[ "${2:-}" == "press" ]]; then exit 9; fi\n'
    single = scripts / "server_ljl_single_right.sh"
    single.write_text(
        "#!/usr/bin/env bash\n"
        "printf 'single|%s|%s|%s|%s|%s|%s|%s\\n' \"$1\" \"$2\" "
        "\"${EXPERIMENT_ID:-}\" \"${BASELINE_JSON-unset}\" \"${RDP_DIR:-}\" "
        "\"${GPU_ID:-}\" \"${DRY_RUN:-}\" "
        '>> "${RDP_SEQUENCE_LOG:?}"\n'
        + failure,
        encoding="utf-8",
    )
    bread = scripts / "server_ljl_bread_dual.sh"
    bread.write_text(
        "#!/usr/bin/env bash\n"
        "printf 'bread|%s|%s|%s|%s|%s|%s|%s\\n' \"$1\" "
        "\"${BREAD_EXPERIMENT_ID:-}\" \"${BREAD_RESUME:-}\" "
        "\"${BASELINE_JSON-unset}\" \"${BREAD_RDP_CODE_DIR:-}\" "
        "\"${GPU_ID:-}\" \"${DRY_RUN:-}\" "
        '>> "${RDP_SEQUENCE_LOG:?}"\n',
        encoding="utf-8",
    )
    single.chmod(0o755)
    bread.chmod(0o755)
    return tmp_path, log


def _run_sequence(code_dir: Path, log: Path, *args: str, **overrides: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(
        {
            "RDP_SEQUENCE_CODE_DIR": str(code_dir),
            "RDP_SEQUENCE_LOG": str(log),
            "BASELINE_JSON": "must-be-unset.json",
            **overrides,
        }
    )
    return subprocess.run(
        ["bash", str(SEQUENCE), *args],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_generic_launcher_requires_deployable_checkpoints_and_auto_calibrates():
    script = (ROOT / "scripts" / "train_pick_tube_single_gpu.sh").read_text(encoding="utf-8")

    assert "AT/LDP will auto-calibrate on the first valid deployment validation" in script
    assert "AT_CKPT=${AT_CKPT:-${AT_DIR}/checkpoints/deployable.ckpt}" in script
    assert "AT checkpoint must be checkpoints/deployable.ckpt; latest is recovery-only" in script
    assert "AT deployable checkpoint was not produced" in script
    assert "AT deployable checkpoint not found" in script
    assert "${LDP_DIR}/checkpoints/deployable.ckpt" in script
    assert "LDP deployable checkpoint was not produced" in script


def test_bread_help_only_allows_deployable_at_checkpoint_for_ldp():
    result = subprocess.run(
        ["bash", str(BREAD), "help"],
        cwd=ROOT,
        env={**os.environ, "BREAD_RDP_CODE_DIR": str(ROOT)},
        text=True,
        capture_output=True,
        check=False,
    )
    source = BREAD.read_text(encoding="utf-8")

    assert result.returncode == 0, result.stderr
    assert "AT deployable.ckpt" in result.stdout
    assert "latest.ckpt（仅恢复用）" in result.stdout
    assert "AT latest.ckpt" not in result.stdout
    assert "AT latest.ckpt" not in source


def test_sequence_runs_insert_press_bread_in_order_with_shared_ids_and_unset_baseline(tmp_path):
    code_dir, log = _fake_code_dir(tmp_path)

    result = _run_sequence(
        code_dir,
        log,
        RDP_SEQUENCE_ID="sequence-42",
        RESUME="false",
        GPU_ID="7",
        DRY_RUN="0",
    )

    assert result.returncode == 0, result.stderr
    assert log.read_text(encoding="utf-8").splitlines() == [
        f"single|train|insert|sequence-42|unset|{code_dir}|7|0",
        f"single|train|press|sequence-42|unset|{code_dir}|7|0",
        f"bread|train|sequence-42|false|unset|{code_dir}|7|0",
    ]
    assert "RDP sequence completed" in result.stdout


def test_sequence_honors_explicit_child_id_overrides(tmp_path):
    code_dir, log = _fake_code_dir(tmp_path)

    result = _run_sequence(
        code_dir,
        log,
        "all",
        RDP_SEQUENCE_ID="sequence-default",
        EXPERIMENT_ID="single-override",
        BREAD_EXPERIMENT_ID="bread-override",
    )

    assert result.returncode == 0, result.stderr
    assert log.read_text(encoding="utf-8").splitlines() == [
        f"single|all|insert|single-override|unset|{code_dir}||",
        f"single|all|press|single-override|unset|{code_dir}||",
        f"bread|all|bread-override|true|unset|{code_dir}||",
    ]


def test_sequence_fails_fast_and_rejects_invalid_stage(tmp_path):
    code_dir, log = _fake_code_dir(tmp_path, fail_press=True)

    failed = _run_sequence(code_dir, log, "doctor", RDP_SEQUENCE_ID="sequence-failure")

    assert failed.returncode == 9
    assert log.read_text(encoding="utf-8").splitlines() == [
        f"single|doctor|insert|sequence-failure|unset|{code_dir}||",
        f"single|doctor|press|sequence-failure|unset|{code_dir}||",
    ]

    log.unlink()
    invalid = _run_sequence(code_dir, log, "invalid")
    assert invalid.returncode == 2
    assert not log.exists()


def test_sequence_labels_dry_run_summary_and_inherits_child_environment(tmp_path):
    code_dir, log = _fake_code_dir(tmp_path)

    result = _run_sequence(
        code_dir,
        log,
        "train",
        RDP_SEQUENCE_ID="sequence-dry-run",
        GPU_ID="3",
        DRY_RUN="1",
    )

    assert result.returncode == 0, result.stderr
    assert "RDP sequence dry run completed" in result.stdout
    assert "RDP sequence completed" not in result.stdout
    assert log.read_text(encoding="utf-8").splitlines() == [
        f"single|train|insert|sequence-dry-run|unset|{code_dir}|3|1",
        f"single|train|press|sequence-dry-run|unset|{code_dir}|3|1",
        f"bread|train|sequence-dry-run|true|unset|{code_dir}|3|1",
    ]
