import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from omegaconf import OmegaConf

from reactive_diffusion_policy.common.artifact_manifest import (
    ArtifactManifest,
    build_normalizer_cache_signature,
    load_normalizer_cache,
    normalizer_identity_digest,
    save_normalizer_cache,
    sha256_file,
    stable_json_digest,
)
from reactive_diffusion_policy.model.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)
from reactive_diffusion_policy.workspace.base_workspace import BaseWorkspace
from reactive_diffusion_policy.workspace.train_at_workspace import TrainATWorkspace
from reactive_diffusion_policy.workspace.train_diffusion_unet_image_workspace import (
    TrainDiffusionUnetImageWorkspace,
)


def _dataset_manifest(**overrides):
    value = {
        "action_representation_version": 2,
        "action_contract": "bimanual_relative_pose20d_v2",
        "normalizer_version": "zero_centered_v2",
        "dataset_digest": "d" * 64,
        "pca_sha256": "p" * 64,
        "tactile_cache_sha256": "t" * 64,
        "converter_git_commit": "converter-commit",
    }
    value.update(overrides)
    return value


def _dataset(**manifest_overrides):
    return SimpleNamespace(
        artifact_manifest=_dataset_manifest(**manifest_overrides),
        train_mask=np.asarray([True, False, True], dtype=bool),
    )


def _cfg(**overrides):
    value = {
        "horizon": 29,
        "n_obs_steps": 2,
        "dataset_obs_steps": 4,
        "dataset_obs_temporal_downsample_ratio": 2,
        "n_latency_steps": 0,
        "n_action_steps": 26,
        "use_latent_action_before_vq": False,
        "artifact_git_commit": "training-commit",
        "task": {
            "dataset": {
                "seed": 42,
                "val_ratio": 0.1,
                "max_train_episodes": None,
                "action_normalizer_version": "zero_centered_v2",
            }
        },
    }
    value.update(overrides)
    return OmegaConf.create(value)


def _normalizer(*, action_offset=0.0, latent_offset=0.0):
    normalizer = LinearNormalizer()
    normalizer["action"] = SingleFieldLinearNormalizer.create_identity()
    normalizer["latent_action"] = SingleFieldLinearNormalizer.create_identity()
    with torch.no_grad():
        normalizer.params_dict["action"]["offset"].fill_(action_offset)
        normalizer.params_dict["latent_action"]["offset"].fill_(latent_offset)
    return normalizer


def test_stable_hashes_are_content_and_order_stable(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"same bytes")

    assert sha256_file(artifact) == sha256_file(artifact)
    assert stable_json_digest({"b": [2, 1], "a": 3}) == stable_json_digest(
        {"a": 3, "b": [2, 1]}
    )
    assert stable_json_digest({"value": 1}) != stable_json_digest({"value": 2})


def test_artifact_manifest_round_trip_and_missing_v2_field_rejection() -> None:
    manifest = ArtifactManifest.from_dict(
        {
            "schema_version": 2,
            "dataset_digest": "d" * 64,
            "split_digest": "s" * 64,
            "action_representation_version": 2,
            "action_contract": "bimanual_relative_pose20d_v2",
            "normalizer_version": "zero_centered_v2",
            "normalizer_sha256": "n" * 64,
            "pca_sha256": "p" * 64,
            "tactile_cache_sha256": "t" * 64,
            "git_commit": "training-commit",
        },
        role="AT",
    )

    assert ArtifactManifest.from_dict(manifest.to_dict(), role="AT") == manifest
    with pytest.raises(ValueError, match="tactile_cache_sha256"):
        ArtifactManifest.from_dict(
            {
                key: value
                for key, value in manifest.to_dict().items()
                if key != "tactile_cache_sha256"
            },
            role="AT",
        )


@pytest.mark.parametrize(
    ("mutation", "expected_key"),
    [
        (
            lambda cfg, dataset, at: setattr(
                dataset,
                "artifact_manifest",
                _dataset_manifest(dataset_digest="x" * 64),
            ),
            "dataset",
        ),
        (
            lambda cfg, dataset, at: setattr(
                dataset, "train_mask", np.asarray([True, True, False])
            ),
            "split",
        ),
        (lambda cfg, dataset, at: at.write_bytes(b"different AT"), "at"),
        (
            lambda cfg, dataset, at: setattr(
                dataset,
                "artifact_manifest",
                _dataset_manifest(pca_sha256="x" * 64),
            ),
            "pca",
        ),
        (
            lambda cfg, dataset, at: setattr(
                dataset,
                "artifact_manifest",
                _dataset_manifest(action_representation_version=3),
            ),
            "action",
        ),
    ],
)
def test_normalizer_cache_signature_invalidates_every_bound_identity(
    tmp_path: Path, mutation, expected_key: str
) -> None:
    at_path = tmp_path / "at.ckpt"
    at_path.write_bytes(b"AT checkpoint")
    cfg = _cfg()
    dataset = _dataset()
    original = build_normalizer_cache_signature(cfg, dataset, at_path)

    mutation(cfg, dataset, at_path)
    changed = build_normalizer_cache_signature(cfg, dataset, at_path)

    assert changed != original, expected_key


def test_normalizer_cache_reuse_requires_exact_canonical_signature(tmp_path: Path) -> None:
    normalizer_path = tmp_path / "normalizer.pkl"
    at_path = tmp_path / "at.ckpt"
    at_path.write_bytes(b"AT checkpoint")
    signature = build_normalizer_cache_signature(_cfg(), _dataset(), at_path)
    expected = _normalizer()

    save_normalizer_cache(normalizer_path, expected, signature)

    loaded = load_normalizer_cache(normalizer_path, signature)
    assert normalizer_identity_digest(loaded) == normalizer_identity_digest(expected)
    meta_path = normalizer_path.with_name("normalizer.meta.json")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    assert metadata["signature"] == signature
    assert metadata["normalizer_sha256"] == sha256_file(normalizer_path)
    assert metadata["action_normalizer_sha256"] == normalizer_identity_digest(
        expected
    )
    mismatch = json.loads(json.dumps(signature))
    mismatch["temporal"]["horizon"] += 1
    assert load_normalizer_cache(normalizer_path, mismatch) is None


def test_normalizer_cache_rejects_mutated_pickle_even_when_signature_matches(
    tmp_path: Path,
) -> None:
    normalizer_path = tmp_path / "normalizer.pkl"
    at_path = tmp_path / "at.ckpt"
    at_path.write_bytes(b"AT checkpoint")
    signature = build_normalizer_cache_signature(_cfg(), _dataset(), at_path)
    save_normalizer_cache(normalizer_path, _normalizer(), signature)

    normalizer_path.write_bytes(b"mutated after metadata publication")

    assert load_normalizer_cache(normalizer_path, signature) is None


def test_missing_cache_metadata_never_reuses_pickle(tmp_path: Path) -> None:
    normalizer_path = tmp_path / "normalizer.pkl"
    normalizer_path.write_bytes(b"stale pickle")
    at_path = tmp_path / "at.ckpt"
    at_path.write_bytes(b"AT checkpoint")

    assert (
        load_normalizer_cache(
            normalizer_path,
            build_normalizer_cache_signature(_cfg(), _dataset(), at_path),
        )
        is None
    )


@pytest.mark.parametrize(
    ("workspace_class", "role", "uses_at"),
    [
        (TrainATWorkspace, "AT", False),
        (TrainDiffusionUnetImageWorkspace, "LDP", True),
    ],
)
def test_training_workspace_checkpoint_cfg_carries_artifact_manifest(
    tmp_path: Path, workspace_class, role: str, uses_at: bool
) -> None:
    at_path = tmp_path / "at.ckpt"
    at_path.write_bytes(b"AT checkpoint")
    normalizer_path = tmp_path / "normalizer.pkl"
    normalizer_path.write_bytes(b"normalizer")
    normalizer = _normalizer()
    signature = build_normalizer_cache_signature(
        _cfg(), _dataset(), at_path if uses_at else None
    )
    cfg = OmegaConf.create({"training": {"use_ema": False}})
    workspace = workspace_class.__new__(workspace_class)
    BaseWorkspace.__init__(workspace, cfg, output_dir=str(tmp_path))
    workspace.model = nn.Linear(1, 1)
    workspace.global_step = 0
    workspace.optimizer_step = 0
    workspace.epoch = 0

    workspace.bind_checkpoint_artifacts(
        signature,
        normalizer=normalizer,
        normalizer_path=normalizer_path,
        role=role,
    )
    checkpoint = tmp_path / f"{role.lower()}.ckpt"
    workspace.save_checkpoint(path=checkpoint, use_thread=False)
    payload = torch.load(checkpoint, weights_only=False)

    artifacts = OmegaConf.to_container(payload["cfg"].artifacts, resolve=True)
    assert artifacts["dataset_digest"] == "d" * 64
    assert artifacts["normalizer_sha256"] == normalizer_identity_digest(normalizer)
    if uses_at:
        assert artifacts["at_sha256"] == sha256_file(at_path)
        assert artifacts["latent_target_mode"] == "posterior_mode_post_vq"
    else:
        assert "at_sha256" not in artifacts


def test_at_and_ldp_share_action_normalizer_identity_despite_different_caches(
    tmp_path: Path,
) -> None:
    at_checkpoint = tmp_path / "at-source.ckpt"
    at_checkpoint.write_bytes(b"AT checkpoint")
    artifact_values = []
    for role, at_path, cache_bytes, normalizer in (
        ("AT", None, b"AT normalizer cache", _normalizer(latent_offset=1.0)),
        (
            "LDP",
            at_checkpoint,
            b"LDP normalizer cache with latent statistics",
            _normalizer(latent_offset=2.0),
        ),
    ):
        normalizer_path = tmp_path / f"{role.lower()}-normalizer.pkl"
        normalizer_path.write_bytes(cache_bytes)
        workspace = BaseWorkspace(
            OmegaConf.create({"training": {"use_ema": False}}),
            output_dir=str(tmp_path),
        )
        workspace.bind_checkpoint_artifacts(
            build_normalizer_cache_signature(_cfg(), _dataset(), at_path),
            normalizer=normalizer,
            normalizer_path=normalizer_path,
            role=role,
        )
        artifact_values.append(workspace.cfg.artifacts.normalizer_sha256)

    assert artifact_values[0] == artifact_values[1]


def test_shared_normalizer_identity_hashes_action_parameters_only() -> None:
    first = _normalizer(action_offset=0.0, latent_offset=1.0)
    same_action = _normalizer(action_offset=0.0, latent_offset=9.0)
    changed_action = _normalizer(action_offset=0.25, latent_offset=1.0)

    assert normalizer_identity_digest(first) == normalizer_identity_digest(
        same_action
    )
    assert normalizer_identity_digest(first) != normalizer_identity_digest(
        changed_action
    )
