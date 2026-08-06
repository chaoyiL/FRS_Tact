from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from safetensors.numpy import save_file as save_safetensors_file

from lerobot.policies.smolvla_jax.validation import CheckpointContract
from tools.publish_smolvla_checkpoint import (
    INFERENCE_FILENAMES,
    SIDECAR_FILENAMES,
    build_inference_bundle,
    contract_from_training_yaml,
    publish_bundle,
    repair_sidecars,
    resolve_dataset_revisions,
)

VT_CONTRACT = CheckpointContract(
    state_dim=20,
    action_dim=20,
    chunk_size=20,
    image_keys=("observation.images.camera1", "observation.images.camera2"),
    tactile_keys=(
        "observation.images.tactile_left_0",
        "observation.images.tactile_right_0",
        "observation.images.tactile_left_1",
        "observation.images.tactile_right_1",
    ),
    tactile_embedding_dim=512,
    tactile_num_tokens=4,
    lora_rank=16,
    vlm_lora_target_modules=("q_proj", "v_proj"),
)


def _json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _weight_tensors() -> dict[str, np.ndarray]:
    tensors: dict[str, np.ndarray] = {
        "model.vlm_with_expert.vlm.model.text_model.embed_tokens.weight": np.zeros(
            (2, 960), dtype=np.float16
        ),
        "model.tactile_encoder.params/conv/kernel": np.zeros((1,), dtype=np.float32),
        "model.tactile_proj.weight": np.zeros((960, 512), dtype=np.float16),
        "model.tactile_proj.bias": np.zeros((960,), dtype=np.float16),
    }
    for target in ("q_proj", "v_proj"):
        prefix = f"model.vlm_with_expert.vlm.model.text_model.layers.0.self_attn.{target}"
        out_dim = 960 if target == "q_proj" else 480
        tensors[f"{prefix}.weight"] = np.zeros((out_dim, 960), dtype=np.float16)
        tensors[f"{prefix}.lora_a"] = np.zeros((16, 960), dtype=np.float16)
        tensors[f"{prefix}.lora_b"] = np.zeros((out_dim, 16), dtype=np.float16)
        tensors[f"{prefix}.lora_scale"] = np.asarray(1.0, dtype=np.float32)
    return tensors


def _valid_checkpoint(path: Path) -> Path:
    path.mkdir(parents=True)
    features = {
        "observation.state": {"type": "STATE", "shape": [20]},
        **{key: {"type": "VISUAL", "shape": [3, 512, 512]} for key in VT_CONTRACT.image_keys},
    }
    _json(
        path / "config.json",
        {
            "chunk_size": 20,
            "n_action_steps": 5,
            "num_vlm_layers": 1,
            "input_features": features,
            "output_features": {"action": {"type": "ACTION", "shape": [20]}},
            "use_tactile_encoder": True,
            "tactile_keys": list(VT_CONTRACT.tactile_keys),
            "tactile_embedding_dim": 512,
            "tactile_num_tokens": 4,
            "lora_rank": 16,
            "vlm_lora_target_modules": ["q_proj", "v_proj"],
            "module_modes": {
                "vision": "frozen",
                "connector": "frozen",
                "vlm_text": "lora",
                "expert": "full",
                "action": "full",
                "state_proj": "full",
                "tactile_proj": "full",
            },
        },
    )
    normalizer_features = {
        "observation.state": {"type": "STATE", "shape": [20]},
        "action": {"type": "ACTION", "shape": [20]},
        **{key: {"type": "VISUAL", "shape": [3, 512, 512]} for key in VT_CONTRACT.image_keys},
    }
    _json(
        path / "policy_preprocessor.json",
        {
            "steps": [
                {
                    "registry_name": "normalizer_processor",
                    "config": {"features": normalizer_features},
                    "state_file": "policy_preprocessor_step_5_normalizer_processor.safetensors",
                }
            ]
        },
    )
    _json(
        path / "policy_postprocessor.json",
        {
            "steps": [
                {
                    "registry_name": "unnormalizer_processor",
                    "config": {"features": {"action": {"type": "ACTION", "shape": [20]}}},
                    "state_file": "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
                }
            ]
        },
    )
    save_safetensors_file(
        {
            "observation.state.mean": np.zeros(20, dtype=np.float32),
            "observation.state.std": np.ones(20, dtype=np.float32),
            "action.mean": np.zeros(20, dtype=np.float32),
            "action.std": np.ones(20, dtype=np.float32),
        },
        path / "policy_preprocessor_step_5_normalizer_processor.safetensors",
    )
    save_safetensors_file(
        {
            "action.mean": np.zeros(20, dtype=np.float32),
            "action.std": np.ones(20, dtype=np.float32),
        },
        path / "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    )
    save_safetensors_file(_weight_tensors(), path / "model.safetensors")
    (path / "training_state.msgpack").write_bytes(b"secret optimizer state")
    (path / "dataset-cache.arrow").write_bytes(b"cache")
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_bundle_has_exact_allowlist_and_provenance(tmp_path: Path) -> None:
    source = _valid_checkpoint(tmp_path / "checkpoint")
    datasets = ({"repo_id": "owner/data", "revision": "a" * 40, "frames": 12},)

    bundle = build_inference_bundle(
        source,
        tmp_path / "bundle",
        expected=VT_CONTRACT,
        dataset_revisions=datasets,
    )

    assert {path.name for path in bundle.iterdir()} == set(INFERENCE_FILENAMES)
    assert not (bundle / "training_state.msgpack").exists()
    manifest = json.loads((bundle / "conversion_manifest.json").read_text())
    assert manifest["contract"] == {
        "state_dim": 20,
        "action_dim": 20,
        "chunk_size": 20,
        "image_keys": list(VT_CONTRACT.image_keys),
        "tactile_keys": list(VT_CONTRACT.tactile_keys),
        "tactile_embedding_dim": 512,
        "tactile_num_tokens": 4,
        "lora_rank": 16,
        "vlm_lora_target_modules": ["q_proj", "v_proj"],
    }
    assert manifest["datasets"] == list(datasets)
    assert manifest["source_weight_sha256"] == _sha256(source / "model.safetensors")
    assert set(manifest["files"]) == set(INFERENCE_FILENAMES) - {"conversion_manifest.json"}
    for filename, metadata in manifest["files"].items():
        assert metadata["sha256"] == _sha256(bundle / filename)
        assert metadata["size"] == (bundle / filename).stat().st_size


def test_bundle_copies_model_into_an_independent_file(tmp_path: Path) -> None:
    source = _valid_checkpoint(tmp_path / "checkpoint")
    bundle = build_inference_bundle(source, tmp_path / "bundle", expected=VT_CONTRACT)
    bundled_hash = _sha256(bundle / "model.safetensors")

    assert (source / "model.safetensors").stat().st_ino != (bundle / "model.safetensors").stat().st_ino
    with (source / "model.safetensors").open("ab") as file:
        file.write(b"source changed")
    assert _sha256(bundle / "model.safetensors") == bundled_hash


def test_without_model_bundle_is_explicitly_not_publishable(tmp_path: Path) -> None:
    source = _valid_checkpoint(tmp_path / "checkpoint")
    bundle = build_inference_bundle(
        source,
        tmp_path / "sidecars",
        expected=VT_CONTRACT,
        include_model=False,
    )
    assert not (bundle / "model.safetensors").exists()
    with pytest.raises(ValueError, match="not publishable without model.safetensors"):
        publish_bundle(
            bundle,
            repo_id="owner/model",
            expected=VT_CONTRACT,
            api=RecordingApi(_sha256(source / "model.safetensors")),
        )


@pytest.mark.parametrize("relative", ["checkpoint.incomplete", "checkpoint.incomplete/child"])
def test_bundle_refuses_incomplete_source(tmp_path: Path, relative: str) -> None:
    source = _valid_checkpoint(tmp_path / relative)
    with pytest.raises(ValueError, match="incomplete"):
        build_inference_bundle(source, tmp_path / "bundle", expected=VT_CONTRACT)


def test_bundle_refuses_existing_destination_and_invalid_source(tmp_path: Path) -> None:
    source = _valid_checkpoint(tmp_path / "checkpoint")
    destination = tmp_path / "bundle"
    destination.mkdir()
    with pytest.raises(FileExistsError):
        build_inference_bundle(source, destination, expected=VT_CONTRACT)

    destination.rmdir()
    (source / "config.json").unlink()
    with pytest.raises(ValueError, match="checkpoint validation failed"):
        build_inference_bundle(source, destination, expected=VT_CONTRACT)
    assert not destination.exists()


class RecordingApi:
    def __init__(
        self,
        weight_sha: str,
        *,
        mutate_after_commit: str | None = None,
        repo_sha: str = "a" * 40,
        lfs: bool = True,
    ) -> None:
        self.weight_sha = weight_sha
        self.mutate_after_commit = mutate_after_commit
        self.repo_sha = repo_sha
        self.lfs = lfs
        self.operations: list[object] = []
        self.commit_kwargs: dict[str, object] = {}
        self.path_revisions: list[str | None] = []
        self.repo_revisions: list[str | None] = []
        self.operation_sources: list[str] = []

    def repo_info(self, *args: object, revision: str | None = None, **kwargs: object) -> object:
        self.repo_revisions.append(revision)
        return SimpleNamespace(sha=self.repo_sha)

    def get_paths_info(self, *args: object, **kwargs: object) -> list[object]:
        self.path_revisions.append(kwargs.get("revision"))
        return [
            SimpleNamespace(
                path="model.safetensors",
                lfs={"sha256": self.weight_sha} if self.lfs else None,
                blob_id=self.weight_sha,
            )
        ]

    def create_commit(self, *, operations: list[object], **kwargs: object) -> object:
        self.operations = operations
        self.commit_kwargs = kwargs
        self.operation_sources = [str(operation.path_or_fileobj) for operation in operations]
        assert all(Path(path).is_file() for path in self.operation_sources)
        if self.mutate_after_commit is not None:
            self.weight_sha = self.mutate_after_commit
        self.repo_sha = "b" * 40
        return SimpleNamespace(oid=self.repo_sha, commit_url="https://huggingface.co/owner/model/commit/b")


def test_sidecar_publish_never_uploads_weight_and_preserves_remote_hash(tmp_path: Path) -> None:
    source = _valid_checkpoint(tmp_path / "checkpoint")
    bundle = build_inference_bundle(source, tmp_path / "bundle", expected=VT_CONTRACT)
    expected_sha = _sha256(source / "model.safetensors")
    api = RecordingApi(expected_sha)

    result = publish_bundle(
        bundle,
        repo_id="owner/model",
        expected=VT_CONTRACT,
        api=api,
        sidecars_only=True,
    )

    uploaded = {operation.path_in_repo for operation in api.operations}
    assert uploaded == set(SIDECAR_FILENAMES) | {"conversion_manifest.json"}
    assert "model.safetensors" not in uploaded
    assert result["weight_sha256_before"] == expected_sha
    assert result["weight_sha256_after"] == expected_sha
    assert api.commit_kwargs["parent_commit"] == "a" * 40
    assert api.path_revisions == ["a" * 40, "b" * 40]
    assert all(not path.startswith(str(bundle)) for path in api.operation_sources)


def test_publish_validates_manifest_against_payload_and_external_contract(tmp_path: Path) -> None:
    source = _valid_checkpoint(tmp_path / "checkpoint")
    expected_sha = _sha256(source / "model.safetensors")
    bundle = build_inference_bundle(source, tmp_path / "bundle", expected=VT_CONTRACT)
    manifest_path = bundle / "conversion_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"] = {}
    _json(manifest_path, manifest)
    with pytest.raises(ValueError, match="manifest files must exactly match"):
        publish_bundle(
            bundle,
            repo_id="owner/model",
            expected=VT_CONTRACT,
            api=RecordingApi(expected_sha),
        )

    bundle = build_inference_bundle(source, tmp_path / "bundle-2", expected=VT_CONTRACT)
    manifest_path = bundle / "conversion_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["contract"]["state_dim"] = 6
    _json(manifest_path, manifest)
    with pytest.raises(ValueError, match="manifest contract does not match expected contract"):
        publish_bundle(
            bundle,
            repo_id="owner/model",
            expected=VT_CONTRACT,
            api=RecordingApi(expected_sha),
        )


def test_publish_refuses_symlinked_payload(tmp_path: Path) -> None:
    source = _valid_checkpoint(tmp_path / "checkpoint")
    expected_sha = _sha256(source / "model.safetensors")
    bundle = build_inference_bundle(source, tmp_path / "bundle", expected=VT_CONTRACT)
    external = tmp_path / "external-config.json"
    (bundle / "config.json").replace(external)
    (bundle / "config.json").symlink_to(external)

    with pytest.raises(ValueError, match="symbolic links"):
        publish_bundle(
            bundle,
            repo_id="owner/model",
            expected=VT_CONTRACT,
            api=RecordingApi(expected_sha),
        )


def test_publish_refuses_invalid_or_unexpected_bundle(tmp_path: Path) -> None:
    source = _valid_checkpoint(tmp_path / "checkpoint")
    bundle = build_inference_bundle(source, tmp_path / "bundle", expected=VT_CONTRACT)
    expected_sha = _sha256(source / "model.safetensors")
    (bundle / "config.json").unlink()
    with pytest.raises(ValueError, match="missing inference files"):
        publish_bundle(
            bundle,
            repo_id="owner/model",
            expected=VT_CONTRACT,
            api=RecordingApi(expected_sha),
        )

    bundle = build_inference_bundle(source, tmp_path / "bundle-2", expected=VT_CONTRACT)
    (bundle / "training_state.msgpack").write_bytes(b"forbidden")
    with pytest.raises(ValueError, match="unexpected files"):
        publish_bundle(
            bundle,
            repo_id="owner/model",
            expected=VT_CONTRACT,
            api=RecordingApi(expected_sha),
        )


def test_publish_refuses_remote_weight_change_before_or_after_commit(tmp_path: Path) -> None:
    source = _valid_checkpoint(tmp_path / "checkpoint")
    bundle = build_inference_bundle(source, tmp_path / "bundle", expected=VT_CONTRACT)
    expected_sha = _sha256(source / "model.safetensors")

    with pytest.raises(ValueError, match="remote model.safetensors SHA-256"):
        publish_bundle(
            bundle,
            repo_id="owner/model",
            expected=VT_CONTRACT,
            api=RecordingApi("0" * 64),
        )
    with pytest.raises(RuntimeError, match="changed during publication"):
        publish_bundle(
            bundle,
            repo_id="owner/model",
            expected=VT_CONTRACT,
            api=RecordingApi(expected_sha, mutate_after_commit="f" * 64),
        )


def test_publish_pins_nonmain_parent_and_checks_new_commit(tmp_path: Path) -> None:
    source = _valid_checkpoint(tmp_path / "checkpoint")
    expected_sha = _sha256(source / "model.safetensors")
    bundle = build_inference_bundle(source, tmp_path / "bundle", expected=VT_CONTRACT)
    api = RecordingApi(expected_sha, repo_sha="c" * 40)

    publish_bundle(
        bundle,
        repo_id="owner/model",
        expected=VT_CONTRACT,
        api=api,
        revision="repair-branch",
    )

    assert api.repo_revisions == ["repair-branch"]
    assert api.commit_kwargs["revision"] == "repair-branch"
    assert api.commit_kwargs["parent_commit"] == "c" * 40
    assert api.path_revisions == ["c" * 40, "b" * 40]


def test_publish_relies_on_parent_commit_to_atomically_reject_branch_move(tmp_path: Path) -> None:
    class MovingBranchApi(RecordingApi):
        def create_commit(self, *, operations: list[object], **kwargs: object) -> object:
            assert kwargs["parent_commit"] == "a" * 40
            raise RuntimeError("parent commit does not match branch head")

    source = _valid_checkpoint(tmp_path / "checkpoint")
    bundle = build_inference_bundle(source, tmp_path / "bundle", expected=VT_CONTRACT)
    api = MovingBranchApi(_sha256(source / "model.safetensors"))

    with pytest.raises(RuntimeError, match="parent commit"):
        publish_bundle(
            bundle,
            repo_id="owner/model",
            expected=VT_CONTRACT,
            api=api,
            revision="repair-branch",
        )


def test_publish_refuses_non_lfs_remote_weight(tmp_path: Path) -> None:
    source = _valid_checkpoint(tmp_path / "checkpoint")
    expected_sha = _sha256(source / "model.safetensors")
    bundle = build_inference_bundle(source, tmp_path / "bundle", expected=VT_CONTRACT)

    with pytest.raises(ValueError, match="no LFS SHA-256"):
        publish_bundle(
            bundle,
            repo_id="owner/model",
            expected=VT_CONTRACT,
            api=RecordingApi(expected_sha, lfs=False),
        )


class DatasetApi:
    def __init__(self, *, head: str, modified: datetime, commits: list[object]) -> None:
        self.head = head
        self.modified = modified
        self.commits = commits

    def dataset_info(self, repo_id: str, *, revision: str | None = None) -> object:
        if revision is not None and revision != self.head:
            return SimpleNamespace(sha=revision, last_modified=self.modified)
        return SimpleNamespace(sha=self.head, last_modified=self.modified)

    def list_repo_commits(
        self,
        repo_id: str,
        *,
        repo_type: str,
        revision: str | None = None,
    ) -> list[object]:
        return self.commits


class RepairApi(DatasetApi):
    def __init__(self, *, weight_sha: str, weight_time: datetime, dataset_sha: str) -> None:
        super().__init__(
            head=dataset_sha,
            modified=weight_time - timedelta(days=1),
            commits=[
                SimpleNamespace(
                    commit_id=dataset_sha,
                    created_at=weight_time - timedelta(days=1),
                )
            ],
        )
        self.weight_sha = weight_sha
        self.weight_time = weight_time

    def get_paths_info(self, *args: object, **kwargs: object) -> list[object]:
        return [
            SimpleNamespace(
                path="model.safetensors",
                lfs={"sha256": self.weight_sha},
                last_commit=SimpleNamespace(date=self.weight_time),
            )
        ]


def test_dataset_revision_proof_accepts_explicit_sha_or_historical_head() -> None:
    weight_time = datetime(2026, 1, 2, tzinfo=UTC)
    head = "a" * 40
    api = DatasetApi(
        head=head,
        modified=weight_time - timedelta(days=1),
        commits=[SimpleNamespace(commit_id=head, created_at=weight_time - timedelta(days=1))],
    )
    explicit_sha = "b" * 40
    explicit_api = DatasetApi(
        head=explicit_sha,
        modified=weight_time - timedelta(days=2),
        commits=[SimpleNamespace(commit_id=explicit_sha, created_at=weight_time - timedelta(days=2))],
    )
    explicit = resolve_dataset_revisions(
        [{"repo_id": "owner/explicit", "revision": "b" * 40}],
        model_weight_uploaded_at=weight_time,
        api=explicit_api,
    )
    inferred = resolve_dataset_revisions(
        [{"repo_id": "owner/head"}], model_weight_uploaded_at=weight_time, api=api
    )
    assert explicit[0]["revision"] == "b" * 40
    assert inferred[0]["revision"] == head
    assert inferred[0]["revision_proof"] == "repository head predates model weight"


def test_dataset_revision_proof_rejects_mutable_or_late_history() -> None:
    weight_time = datetime(2026, 1, 2, tzinfo=UTC)
    head = "a" * 40
    late = weight_time + timedelta(days=1)
    api = DatasetApi(
        head=head,
        modified=late,
        commits=[SimpleNamespace(commit_id=head, created_at=late)],
    )
    with pytest.raises(ValueError, match="cannot prove training-time dataset revision"):
        resolve_dataset_revisions([{"repo_id": "owner/data"}], model_weight_uploaded_at=weight_time, api=api)


def test_explicit_dataset_revision_must_predate_model_weight() -> None:
    weight_time = datetime(2026, 1, 2, tzinfo=UTC)
    requested = "b" * 40
    late = weight_time + timedelta(days=1)
    api = DatasetApi(
        head=requested,
        modified=late,
        commits=[SimpleNamespace(commit_id=requested, created_at=late)],
    )

    with pytest.raises(ValueError, match="explicit dataset revision.*postdates model weight"):
        resolve_dataset_revisions(
            [{"repo_id": "owner/data", "revision": requested}],
            model_weight_uploaded_at=weight_time,
            api=api,
        )


def test_training_yaml_contract_is_authoritative(tmp_path: Path) -> None:
    yaml_path = tmp_path / "train.yaml"
    yaml_path.write_text(
        """model:
  state_dim: 20
  action_dim: 20
  chunk_size: 20
  image_keys: [observation.images.camera1, observation.images.camera2]
  use_tactile_encoder: true
  tactile_keys: [t0, t1, t2, t3]
  tactile_embedding_dim: 512
  tactile_num_tokens: 4
  lora_rank: 16
  vlm_lora_target_modules: [q_proj, v_proj]
""",
        encoding="utf-8",
    )
    contract = contract_from_training_yaml(yaml_path)
    assert contract.state_dim == 20
    assert contract.action_dim == 20
    assert contract.chunk_size == 20
    assert contract.tactile_num_tokens == 4


def test_repair_uses_metadata_only_stats_and_records_immutable_provenance(tmp_path: Path) -> None:
    snapshot = _valid_checkpoint(tmp_path / "snapshot")
    weight_sha = _sha256(snapshot / "model.safetensors")
    dataset_sha = "d" * 40
    weight_time = datetime(2026, 1, 2, tzinfo=UTC)
    api = RepairApi(weight_sha=weight_sha, weight_time=weight_time, dataset_sha=dataset_sha)
    config = tmp_path / "train.yaml"
    config.write_text(
        """datasets:
  - repo_id: owner/data
    action_key: actions
model:
  state_dim: 20
  action_dim: 20
  chunk_size: 20
  image_keys: [observation.images.camera1, observation.images.camera2]
  use_tactile_encoder: true
  tactile_encoder_path: /not/read/by/repair
  tactile_keys: [observation.images.tactile_left_0, observation.images.tactile_right_0, observation.images.tactile_left_1, observation.images.tactile_right_1]
  tactile_embedding_dim: 512
  tactile_num_tokens: 4
  lora_rank: 16
  vlm_lora_target_modules: [q_proj, v_proj]
  module_modes:
    vision: frozen
    connector: frozen
    vlm_text: lora
    expert: full
    action: full
    state_proj: full
    tactile_proj: full
""",
        encoding="utf-8",
    )
    metadata = SimpleNamespace(
        total_frames=17,
        features={"observation.state": {}, "actions": {}},
        stats={
            "observation.state": {
                "mean": np.arange(20, dtype=np.float32),
                "std": np.ones(20, dtype=np.float32),
            },
            "actions": {
                "mean": np.arange(20, dtype=np.float32) + 10,
                "std": np.ones(20, dtype=np.float32) * 2,
            },
        },
    )
    loads: list[tuple[str, str]] = []

    def metadata_loader(repo_id: str, revision: str) -> object:
        loads.append((repo_id, revision))
        return metadata

    output = repair_sidecars(
        repo_id="owner/model",
        training_config=config,
        output=tmp_path / "repair",
        expected_weight_sha256=weight_sha,
        api=api,
        snapshot_resolver=lambda repo_id, revision: snapshot,
        metadata_loader=metadata_loader,
    )

    assert loads == [("owner/data", dataset_sha)]
    assert _sha256(output / "model.safetensors") == weight_sha
    manifest = json.loads((output / "conversion_manifest.json").read_text())
    assert manifest["datasets"] == [
        {
            "action_key": "actions",
            "frames": 17,
            "repo_id": "owner/data",
            "revision": dataset_sha,
            "revision_proof": "repository head predates model weight",
        }
    ]


def test_current_unpinned_training_yaml_fails_closed_without_weight_history(tmp_path: Path) -> None:
    class UnprovenApi:
        def get_paths_info(self, *args: object, **kwargs: object) -> list[object]:
            return [
                SimpleNamespace(
                    path="model.safetensors",
                    lfs={"sha256": "9" * 64},
                    last_commit=None,
                )
            ]

    output = tmp_path / "repair"
    with pytest.raises(ValueError, match="model weight upload time is unavailable"):
        repair_sidecars(
            repo_id="owner/model",
            training_config="configs/train_vtsmolvla_jax.yaml",
            output=output,
            api=UnprovenApi(),
            snapshot_resolver=lambda repo_id, revision: pytest.fail("must fail before download"),
            metadata_loader=lambda repo_id, revision: pytest.fail("must fail before metadata"),
        )
    assert not output.exists()


def test_cli_help_and_invalid_validate_return_json(tmp_path: Path) -> None:
    help_result = subprocess.run(
        [sys.executable, "tools/publish_smolvla_checkpoint.py", "--help"],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
        check=False,
    )
    assert help_result.returncode == 0
    for command in ("validate", "bundle", "repair-sidecars", "publish"):
        assert command in help_result.stdout

    result = subprocess.run(
        [sys.executable, "tools/publish_smolvla_checkpoint.py", "validate", str(tmp_path)],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert payload["issues"]
