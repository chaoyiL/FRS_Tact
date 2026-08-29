import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_rdpctl_plan_prints_four_gpu_pipeline_without_writing_state(tmp_path):
    profile = tmp_path / "profile.yaml"
    output_root = tmp_path / "outputs"
    profile.write_text(
        f"""gpu_ids: [0, 1, 2, 3]
paths:
  python: .venv/bin/python
  accelerate: .venv/bin/accelerate
  jax_python: .venv-jax/bin/python
  dataset_root: {tmp_path / 'datasets'}
  encoder_dir: data/encoder_ckpt_0809
  tactile_cache_root: {tmp_path / 'cache'}
  tactile_pca_path: {tmp_path / 'pca.npz'}
  dataset_path: {tmp_path / 'rdp_zarr'}
  output_root: {output_root}
""",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "rdpctl.py"),
            "plan",
            "--profile",
            str(profile),
            "--run-id",
            "picktube6-p30-test",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "precompute:pick_tube_01" in result.stdout
    assert "CUDA_VISIBLE_DEVICES=0" in result.stdout
    assert "precompute:pick_tube_06" in result.stdout
    assert "convert_pick_tube_lerobot_to_rdp_zarr.py" in result.stdout
    assert "train_pick_tube_single_gpu.sh all" in result.stdout
    assert not output_root.exists()


def test_rdpctl_rejects_wandb_incompatible_run_id():
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "rdpctl.py"),
            "plan",
            "--run-id",
            "x" * 65,
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "1-64 characters" in result.stderr


def test_rdpctl_rejects_incomplete_profile(tmp_path):
    profile = tmp_path / "profile.yaml"
    profile.write_text("gpu_ids: [0]\npaths:\n  python: .venv/bin/python\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "rdpctl.py"),
            "plan",
            "--profile",
            str(profile),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "missing required keys" in result.stderr
