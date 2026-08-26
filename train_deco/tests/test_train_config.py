from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest

from train_deco.train import build_argument_parser, validate_stage_arguments


def _parse(*arguments: str):
    return build_argument_parser().parse_args(arguments)


def test_stage1_cli_defaults_remain_vision_only() -> None:
    args = _parse()

    assert args.stage == 1
    assert args.stage1_checkpoint is None
    assert args.tactile_encoder_checkpoint is None
    assert args.tactile_encoder_cache == "checkpoints/deco/tactile_encoder_cache"
    assert args.tactile_adapter_rank == 32
    assert args.resume_from is None
    assert args.epochs == 100
    validate_stage_arguments(args)


def test_fresh_stage2_requires_stage1_and_tactile_initialization_paths() -> None:
    for arguments, message in (
        (("--stage", "2"), "stage1-checkpoint"),
        (
            ("--stage", "2", "--stage1-checkpoint", "stage1.pt"),
            "tactile-encoder-checkpoint",
        ),
    ):
        with pytest.raises(ValueError, match=message):
            validate_stage_arguments(_parse(*arguments))

    args = _parse(
        "--stage",
        "2",
        "--stage1-checkpoint",
        "stage1.pt",
        "--tactile-encoder-checkpoint",
        "encoder",
        "--dataset-format",
        "lerobot-v21",
    )
    validate_stage_arguments(args)


def test_stage1_checkpoint_and_exact_resume_are_mutually_exclusive() -> None:
    args = _parse(
        "--stage",
        "2",
        "--stage1-checkpoint",
        "stage1.pt",
        "--resume",
        "stage2.pt",
        "--dataset-format",
        "lerobot-v21",
    )

    with pytest.raises(ValueError, match="mutually exclusive"):
        validate_stage_arguments(args)


def test_stage2_exact_resume_needs_no_fresh_initialization_paths() -> None:
    args = _parse(
        "--stage",
        "2",
        "--resume",
        "stage2.pt",
        "--dataset-format",
        "lerobot-v21",
    )

    assert args.resume_from == "stage2.pt"
    validate_stage_arguments(args)


def test_stage2_resume_rejects_the_preprocessed_backend() -> None:
    args = _parse(
        "--stage",
        "2",
        "--resume",
        "stage2.pt",
        "--dataset-format",
        "preprocessed",
    )

    with pytest.raises(ValueError, match="lerobot-v21"):
        validate_stage_arguments(args)


def test_stage2_reference_config_declares_required_initialization_contract() -> None:
    config = (
        Path(__file__).parents[1] / "configs" / "train_deco.yaml"
    ).read_text(encoding="utf-8")

    assert "stage: 2" in config
    assert "stage1_checkpoint: checkpoints/deco/image_aug/deco_stage1_latest.pt" in config
    assert "tactile_encoder_checkpoint: checkpoints/encoder/encoder_ckpt_0824" in config
    assert "tactile_encoder_cache: checkpoints/deco/tactile_encoder_cache" in config
    assert "tactile_adapter_rank: 32" in config


def test_stage2_shell_mode_passes_source_paths_without_hardcoded_artifact() -> None:
    launcher = Path(__file__).parents[1] / "scripts" / "train_deco.sh"
    result = subprocess.run(
        [
            "bash",
            str(launcher),
            "--mode",
            "local-stage2",
            "--stage1-checkpoint",
            "/tmp/stage1.pt",
            "--tactile-encoder-checkpoint",
            "/tmp/encoder",
            "--dry-run",
        ],
        cwd=Path(__file__).parents[2],
        env={**os.environ, "RUN_ID": "stage2_contract"},
        text=True,
        capture_output=True,
        check=True,
    )

    assert "--stage 2" in result.stdout
    assert "--stage1-checkpoint /tmp/stage1.pt" in result.stdout
    assert "--tactile-encoder-checkpoint /tmp/encoder" in result.stdout
    assert ".safetensors" not in result.stdout
    assert "--epochs 50" in result.stdout


def test_server_stage2_shell_mode_uses_ddp_and_source_directory() -> None:
    launcher = Path(__file__).parents[1] / "scripts" / "train_deco.sh"
    result = subprocess.run(
        [
            "bash",
            str(launcher),
            "--mode",
            "server-stage2",
            "--stage1-checkpoint",
            "/tmp/stage1.pt",
            "--tactile-encoder-checkpoint",
            "/tmp/encoder",
            "--dry-run",
        ],
        cwd=Path(__file__).parents[2],
        env={**os.environ, "RUN_ID": "stage2_server_contract"},
        text=True,
        capture_output=True,
        check=True,
    )

    assert "torchrun" in result.stdout
    assert "--nproc_per_node=2" in result.stdout
    assert "--stage 2" in result.stdout
    assert "--tactile-encoder-checkpoint /tmp/encoder" in result.stdout
    assert "--epochs 50" in result.stdout

