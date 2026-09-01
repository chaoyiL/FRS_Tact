import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEQUENCE = ROOT / "scripts" / "server_ljl_rdp_sequence.sh"


def _fake_code_dir(tmp_path: Path, *, fail_press: bool = False) -> tuple[Path, Path]:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    log = tmp_path / "calls.log"
    failure = "" if not fail_press else 'if [[ "${2:-}" == "press" ]]; then exit 9; fi\n'
    single = scripts / "server_ljl_single_right.sh"
    single.write_text(
        "#!/usr/bin/env bash\n"
        "printf 'single|%s|%s|%s|%s|%s\\n' \"$1\" \"$2\" "
        "\"${EXPERIMENT_ID:-}\" \"${BASELINE_JSON-unset}\" \"${RDP_DIR:-}\" "
        '>> "${RDP_SEQUENCE_LOG:?}"\n'
        + failure,
        encoding="utf-8",
    )
    bread = scripts / "server_ljl_bread_dual.sh"
    bread.write_text(
        "#!/usr/bin/env bash\n"
        "printf 'bread|%s|%s|%s|%s|%s\\n' \"$1\" "
        "\"${BREAD_EXPERIMENT_ID:-}\" \"${BREAD_RESUME:-}\" "
        "\"${BASELINE_JSON-unset}\" \"${BREAD_RDP_CODE_DIR:-}\" "
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


def test_sequence_runs_insert_press_bread_in_order_with_shared_ids_and_unset_baseline(tmp_path):
    code_dir, log = _fake_code_dir(tmp_path)

    result = _run_sequence(
        code_dir,
        log,
        RDP_SEQUENCE_ID="sequence-42",
        RESUME="false",
    )

    assert result.returncode == 0, result.stderr
    assert log.read_text(encoding="utf-8").splitlines() == [
        f"single|train|insert|sequence-42|unset|{code_dir}",
        f"single|train|press|sequence-42|unset|{code_dir}",
        f"bread|train|sequence-42|false|unset|{code_dir}",
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
        f"single|all|insert|single-override|unset|{code_dir}",
        f"single|all|press|single-override|unset|{code_dir}",
        f"bread|all|bread-override|true|unset|{code_dir}",
    ]


def test_sequence_fails_fast_and_rejects_invalid_stage(tmp_path):
    code_dir, log = _fake_code_dir(tmp_path, fail_press=True)

    failed = _run_sequence(code_dir, log, "doctor", RDP_SEQUENCE_ID="sequence-failure")

    assert failed.returncode == 9
    assert log.read_text(encoding="utf-8").splitlines() == [
        f"single|doctor|insert|sequence-failure|unset|{code_dir}",
        f"single|doctor|press|sequence-failure|unset|{code_dir}",
    ]

    log.unlink()
    invalid = _run_sequence(code_dir, log, "invalid")
    assert invalid.returncode == 2
    assert not log.exists()
