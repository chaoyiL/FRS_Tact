import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "deploy_pick_tube_rdp", ROOT / "deploy_pick_tube_rdp.py"
)
deploy = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(deploy)


def test_default_deploy_config_accepts_legacy_checkpoint_pairs() -> None:
    config = deploy.load_config(ROOT / "configs" / "deploy_pick_tube_rdp.yaml")

    assert config["model"]["artifact_verification"] == "legacy-compatible"


def test_server_config_negotiates_low_latency_rdp_step_v2() -> None:
    config = deploy.load_config(ROOT / "configs" / "deploy_pick_tube_rdp.yaml")

    payload = deploy.build_server_config(config)

    assert payload["execution_protocol"] == "rdp_step_v2"
    assert payload["policy_type"] == "rdp"
    assert payload["observation_profile"] == "rdp_vitac_224"
    assert payload["task"] == 0
    assert payload["control_frequency"] == 30.0
    assert payload["steps_per_inference"] == 1
    assert payload["action_horizon"] == 1


class FakeTactileEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.last_means = None

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        self.last_means = images.mean(dim=(1, 2, 3)).detach().cpu()
        return self.last_means.to(images.device)[:, None].repeat(1, 512)


class FakePolicy:
    def __init__(
        self, components_per_arm: int, state_dim: int = 20, action_dim: int = 20
    ) -> None:
        self.components_per_arm = components_per_arm
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.slow_calls = 0
        self.fast_history_lengths = []
        self.slow_observation_states = []

    def predict_action(self, obs_dict, **kwargs):
        assert tuple(obs_dict["camera1"].shape) == (1, 2, 3, 224, 224)
        assert tuple(obs_dict["camera2"].shape) == (1, 2, 3, 224, 224)
        assert tuple(obs_dict["observation_state"].shape) == (1, 2, self.state_dim)
        tactile_dim = self.components_per_arm * 2
        assert tuple(obs_dict["tactile_embedding"].shape) == (1, 2, tactile_dim)
        torch.testing.assert_close(
            obs_dict["tactile_embedding"][0, :, : self.components_per_arm],
            torch.full((2, self.components_per_arm), 1.0 / 255.0),
        )
        torch.testing.assert_close(
            obs_dict["tactile_embedding"][0, :, self.components_per_arm :],
            torch.full((2, self.components_per_arm), 3.0 / 255.0),
        )
        assert kwargs["return_latent_action"] is True
        self.slow_observation_states.append(
            obs_dict["observation_state"][0, :, 0].detach().cpu().tolist()
        )
        self.slow_calls += 1
        return {"action": torch.zeros((1, 29, 128))}

    def predict_from_latent_action(
        self,
        latent_action,
        extended_obs,
        extended_obs_last_step,
        dataset_obs_temporal_downsample_ratio,
    ):
        assert tuple(latent_action.shape) == (1, 128)
        assert dataset_obs_temporal_downsample_ratio == 2
        history_length = extended_obs["tactile_embedding"].shape[1]
        assert extended_obs["tactile_embedding"].shape[2] == self.components_per_arm * 2
        assert extended_obs_last_step == history_length
        self.fast_history_lengths.append(history_length)
        return {"action": torch.full((1, history_length, self.action_dim), float(history_length))}


def observation(step: int = 0) -> dict:
    result = {
        "observation.images.camera0": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation.images.camera1": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation.state": np.full(20, step, dtype=np.float32),
    }
    for value, key in enumerate(deploy.TACTILE_KEYS, start=1):
        result[key] = np.full((224, 224, 3), value, dtype=np.uint8)
    return result


def tactile_cfg(obs_dim: int, extended_dim: int | None = None):
    return OmegaConf.create(
        {
            "release_validation": {
                "passed": True,
                "score": 0.1,
                "deployment_slow_update_interval": 16,
                "phase_start": 3,
            },
            "n_obs_steps": 2,
            "dataset_obs_temporal_downsample_ratio": 2,
            "shape_meta": {
                "obs": {
                    "tactile_embedding": {"shape": [obs_dim]},
                    "observation_state": {"shape": [20]},
                },
                "extended_obs": {
                    "tactile_embedding": {
                        "shape": [obs_dim if extended_dim is None else extended_dim]
                    }
                },
                "action": {"shape": [20]},
            }
        }
    )


def state_action_cfg(state_dim: int, action_dim: int):
    return OmegaConf.create(
        {
            "shape_meta": {
                "obs": {"observation_state": {"shape": [state_dim]}},
                "action": {"shape": [action_dim]},
            }
        }
    )


class FakeAT:
    def __init__(self) -> None:
        self.normalizer = None

    def set_normalizer(self, normalizer) -> None:
        self.normalizer = normalizer


class LoadedFakePolicy:
    def __init__(self) -> None:
        self.at = FakeAT()
        self.normalizer = object()
        self.num_inference_steps = None
        self.eval_called = False
        self.device = None

    def eval(self):
        self.eval_called = True
        return self

    def to(self, device):
        self.device = device
        return self


def artifact_manifest(**overrides):
    value = {
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
    }
    value.update(overrides)
    return value


def policy_cfg(tactile_dim: int, artifacts=None):
    cfg = tactile_cfg(tactile_dim)
    cfg._target_ = "tests.FakeWorkspace"
    cfg.policy = {
        "at": {},
        "obs_encoder": {"random_transforms": []},
    }
    cfg.training = {"use_ema": True}
    if artifacts is not None:
        cfg.artifacts = artifacts
    return cfg


def payload(cfg):
    return {"cfg": cfg}


def release_cfg(**release_validation):
    evidence = {"phase_start": 3}
    evidence.update(release_validation)
    return OmegaConf.create(
        {
            "n_obs_steps": 2,
            "dataset_obs_temporal_downsample_ratio": 2,
            "release_validation": evidence,
        }
    )


def test_validate_release_qualification_accepts_matching_slow16_evidence() -> None:
    evidence = {
        "passed": True,
        "score": 0.0125,
        "deployment_slow_update_interval": 16,
        "phase_start": 3,
    }

    deploy.validate_release_qualification(
        release_cfg(**evidence),
        release_cfg(**evidence),
        slow_update_interval=16,
        ldp_checkpoint=Path("ldp/deployable.ckpt"),
        at_checkpoint=Path("at/deployable.ckpt"),
    )


@pytest.mark.parametrize("invalid", [16.9, 16.0, "16", True])
@pytest.mark.parametrize(
    "field", ["n_obs_steps", "dataset_obs_temporal_downsample_ratio"]
)
def test_validate_release_qualification_rejects_non_integer_phase_contract_values(
    field, invalid
) -> None:
    evidence = {
        "passed": True,
        "score": 0.1,
        "deployment_slow_update_interval": 16,
        "phase_start": 3,
    }
    ldp_cfg = release_cfg(**evidence)
    ldp_cfg[field] = invalid

    with pytest.raises(ValueError, match=field):
        deploy.validate_release_qualification(
            ldp_cfg,
            release_cfg(**evidence),
            slow_update_interval=16,
            ldp_checkpoint=Path("ldp/deployable.ckpt"),
            at_checkpoint=Path("at/deployable.ckpt"),
        )


@pytest.mark.parametrize(
    ("ldp_start", "at_start", "evidence_start"),
    [(3, 5, 3), (5, 5, 5), (3, 3, None), (3, 3, 5)],
)
def test_validate_release_qualification_rejects_phase_start_mismatch(
    ldp_start, at_start, evidence_start
) -> None:
    evidence = {
        "passed": True,
        "score": 0.1,
        "deployment_slow_update_interval": 16,
        "phase_start": evidence_start,
    }
    ldp_cfg = release_cfg(**evidence)
    at_cfg = release_cfg(**evidence)
    ldp_cfg.n_obs_steps = ldp_start + 1
    ldp_cfg.dataset_obs_temporal_downsample_ratio = 1
    at_cfg.n_obs_steps = at_start + 1
    at_cfg.dataset_obs_temporal_downsample_ratio = 1
    if evidence_start is None:
        del ldp_cfg.release_validation.phase_start

    with pytest.raises(ValueError, match="phase_start"):
        deploy.validate_release_qualification(
            ldp_cfg,
            at_cfg,
            slow_update_interval=16,
            ldp_checkpoint=Path("ldp/deployable.ckpt"),
            at_checkpoint=Path("at/deployable.ckpt"),
        )


@pytest.mark.parametrize(
    ("role", "release_validation", "message"),
    [
        pytest.param("LDP", None, "release_validation", id="missing-evidence"),
        pytest.param(
            "AT",
            {"passed": False, "score": 0.1, "deployment_slow_update_interval": 16},
            "passed",
            id="failed-gate",
        ),
        pytest.param(
            "AT",
            {"passed": 1, "score": 0.1, "deployment_slow_update_interval": 16},
            "passed",
            id="integer-passed",
        ),
        pytest.param(
            "AT",
            {"passed": "true", "score": 0.1, "deployment_slow_update_interval": 16},
            "passed",
            id="string-passed",
        ),
        pytest.param(
            "LDP",
            {"passed": True, "score": math.nan, "deployment_slow_update_interval": 16},
            "score",
            id="nan-score",
        ),
        pytest.param(
            "AT",
            {"passed": True, "score": math.inf, "deployment_slow_update_interval": 16},
            "score",
            id="infinite-score",
        ),
        pytest.param(
            "LDP",
            {"passed": True, "score": 0.1, "deployment_slow_update_interval": 15},
            "interval",
            id="release-interval-mismatch",
        ),
        pytest.param(
            "LDP",
            {"passed": True, "score": 0.1, "deployment_slow_update_interval": True},
            "interval",
            id="boolean-release-interval",
        ),
        pytest.param(
            "LDP",
            {"passed": True, "score": 0.1, "deployment_slow_update_interval": "16"},
            "interval",
            id="string-release-interval",
        ),
    ],
)
def test_validate_release_qualification_rejects_incomplete_or_invalid_evidence(
    role, release_validation, message
) -> None:
    evidence = {
        "passed": True,
        "score": 0.1,
        "deployment_slow_update_interval": 16,
        "phase_start": 3,
    }
    ldp_cfg = release_cfg(**evidence)
    at_cfg = release_cfg(**evidence)
    if release_validation is None:
        if role == "LDP":
            ldp_cfg = OmegaConf.create({})
        else:
            at_cfg = OmegaConf.create({})
    elif role == "LDP":
        ldp_cfg = release_cfg(**release_validation)
    else:
        at_cfg = release_cfg(**release_validation)

    with pytest.raises(ValueError, match=message):
        deploy.validate_release_qualification(
            ldp_cfg,
            at_cfg,
            slow_update_interval=16,
            ldp_checkpoint=Path("ldp/deployable.ckpt"),
            at_checkpoint=Path("at/deployable.ckpt"),
        )


@pytest.mark.parametrize("slow_update_interval", [15, True, "16"])
def test_validate_release_qualification_rejects_runtime_interval_other_than_sixteen(
    slow_update_interval,
) -> None:
    evidence = {
        "passed": True,
        "score": 0.1,
        "deployment_slow_update_interval": 16,
        "phase_start": 3,
    }

    with pytest.raises(ValueError, match="slow_update_interval"):
        deploy.validate_release_qualification(
            release_cfg(**evidence),
            release_cfg(**evidence),
            slow_update_interval=slow_update_interval,
            ldp_checkpoint=Path("ldp.ckpt"),
            at_checkpoint=Path("at.ckpt"),
        )


@pytest.mark.parametrize(
    ("ldp_checkpoint", "at_checkpoint"),
    [
        pytest.param(Path("latest.ckpt"), Path("deployable.ckpt"), id="ldp-latest"),
        pytest.param(Path("deployable.ckpt"), Path("latest.ckpt"), id="at-latest"),
    ],
)
def test_validate_release_qualification_rejects_recovery_checkpoints(
    ldp_checkpoint: Path, at_checkpoint: Path
) -> None:
    evidence = {
        "passed": True,
        "score": 0.1,
        "deployment_slow_update_interval": 16,
    }

    with pytest.raises(ValueError, match="deployable.ckpt"):
        deploy.validate_release_qualification(
            release_cfg(**evidence),
            release_cfg(**evidence),
            slow_update_interval=16,
            ldp_checkpoint=ldp_checkpoint,
            at_checkpoint=at_checkpoint,
        )


def test_unqualified_override_allows_legacy_checkpoints_without_release_evidence() -> None:
    legacy_cfg = OmegaConf.create(
        {
            "n_obs_steps": 2,
            "dataset_obs_temporal_downsample_ratio": 2,
        }
    )

    with pytest.warns(UserWarning, match="release qualification checks are disabled"):
        deploy.validate_release_qualification(
            legacy_cfg,
            legacy_cfg,
            slow_update_interval=16,
            ldp_checkpoint=Path("ldp/old-model.ckpt"),
            at_checkpoint=Path("at/old-model.ckpt"),
            allow_unqualified_checkpoint=True,
        )


def test_unqualified_override_allows_failed_at_and_ldp_evidence() -> None:
    failed = {
        "passed": False,
        "score": 5.48,
        "deployment_slow_update_interval": 15,
        "phase_start": 1,
    }

    with pytest.warns(UserWarning, match="release qualification checks are disabled"):
        deploy.validate_release_qualification(
            release_cfg(**failed),
            release_cfg(**failed),
            slow_update_interval=16,
            ldp_checkpoint=Path("ldp/latest.ckpt"),
            at_checkpoint=Path("at/latest.ckpt"),
            allow_unqualified_checkpoint=True,
        )


def test_unqualified_override_keeps_runtime_slow16_contract() -> None:
    with pytest.raises(ValueError, match="slow_update_interval"):
        deploy.validate_release_qualification(
            OmegaConf.create({}),
            OmegaConf.create({}),
            slow_update_interval=15,
            ldp_checkpoint=Path("ldp/latest.ckpt"),
            at_checkpoint=Path("at/latest.ckpt"),
            allow_unqualified_checkpoint=True,
        )


def test_unqualified_override_requires_manual_runtime_start() -> None:
    deploy.validate_unqualified_runtime_mode(
        allow_unqualified_checkpoint=True,
        auto_start=False,
    )

    with pytest.raises(ValueError, match="auto_start"):
        deploy.validate_unqualified_runtime_mode(
            allow_unqualified_checkpoint=True,
            auto_start=True,
        )


def test_parse_args_exposes_explicit_unqualified_checkpoint_switch() -> None:
    arguments = deploy.parse_args(["--allow-unqualified-checkpoint"])

    assert arguments.allow_unqualified_checkpoint is True


def test_prepare_inference_config_drops_training_only_color_jitter() -> None:
    config = OmegaConf.create(
        {
            "policy": {
                "obs_encoder": {
                    "random_transforms": [
                        {"type": "RandomCrop", "ratio": 0.9},
                        {
                            "type": "ColorJitter",
                            "brightness": 0.25,
                            "contrast": 0.25,
                            "saturation": 0.15,
                            "hue": 0.03,
                        },
                    ]
                }
            }
        }
    )

    deploy.prepare_inference_config(config)

    assert OmegaConf.to_container(
        config.policy.obs_encoder.random_transforms,
        resolve=True,
    ) == [{"type": "RandomCrop", "ratio": 0.9}]


@pytest.mark.parametrize("tactile_dim", [16, 30, 60])
def test_validate_tactile_dimensions_accepts_matching_artifacts(
    tactile_dim: int,
) -> None:
    deploy.validate_tactile_dimensions(
        tactile_dim,
        tactile_cfg(tactile_dim),
        tactile_cfg(tactile_dim),
        Path("ldp.ckpt"),
        Path("at.ckpt"),
    )


def test_validate_tactile_dimensions_reports_every_source() -> None:
    with pytest.raises(ValueError) as error:
        deploy.validate_tactile_dimensions(
            16,
            tactile_cfg(16, extended_dim=30),
            tactile_cfg(60),
            Path("ldp.ckpt"),
            Path("at.ckpt"),
        )

    message = str(error.value)
    assert "PCA output=16D" in message
    assert "LDP obs (ldp.ckpt)=16D" in message
    assert "LDP extended_obs (ldp.ckpt)=30D" in message
    assert "AT obs (at.ckpt)=60D" in message
    assert "AT extended_obs (at.ckpt)=60D" in message


def test_tactile_dim_reports_missing_checkpoint_field() -> None:
    with pytest.raises(ValueError, match="LDP checkpoint is missing"):
        deploy._tactile_dim(OmegaConf.create({}), "LDP", "obs")


@pytest.mark.parametrize("role", ["LDP", "AT"])
@pytest.mark.parametrize(
    "invalid_payload",
    [
        pytest.param(None, id="non-mapping-payload"),
        pytest.param({}, id="missing-cfg"),
        pytest.param({"cfg": None}, id="none-cfg"),
        pytest.param({"cfg": 16}, id="scalar-cfg"),
        pytest.param({"cfg": []}, id="list-cfg"),
    ],
)
def test_load_policy_reports_checkpoint_payload_cfg_errors(role, invalid_payload, monkeypatch) -> None:
    ldp_checkpoint = Path("ldp.ckpt")
    at_checkpoint = Path("at.ckpt")
    valid_payload = payload(policy_cfg(16))

    def load_payload(path, current_role):
        if current_role == role:
            return invalid_payload
        return valid_payload

    monkeypatch.setattr(deploy, "_load_checkpoint_payload", load_payload)

    with pytest.raises(ValueError) as error:
        deploy.load_policy(
            ldp_checkpoint,
            at_checkpoint,
            torch.device("cpu"),
            num_inference_steps=8,
            tactile_embedding_dim=16,
            slow_update_interval=16,
        )

    message = str(error.value)
    assert role in message
    assert str(ldp_checkpoint if role == "LDP" else at_checkpoint) in message
    assert "cfg" in message


def test_load_policy_reports_unresolved_checkpoint_metadata(monkeypatch) -> None:
    ldp_checkpoint = Path("ldp.ckpt")
    at_checkpoint = Path("at.ckpt")
    ldp_cfg = policy_cfg(16)
    ldp_cfg.shape_meta.obs.tactile_embedding.shape = ["${missing_dimension}"]

    monkeypatch.setattr(
        deploy,
        "_load_checkpoint_payload",
        lambda path, role: payload(ldp_cfg if role == "LDP" else policy_cfg(16)),
    )

    with pytest.raises(ValueError) as error:
        deploy.load_policy(
            ldp_checkpoint,
            at_checkpoint,
            torch.device("cpu"),
            num_inference_steps=8,
            tactile_embedding_dim=16,
            slow_update_interval=16,
        )

    message = str(error.value)
    assert "LDP" in message
    assert str(ldp_checkpoint) in message
    assert "shape_meta.obs.tactile_embedding.shape" in message
    assert error.value.__cause__ is not None


def test_load_policy_validates_matching_payloads_before_workspace_construction(monkeypatch) -> None:
    ldp_checkpoint = Path("ldp/deployable.ckpt")
    at_checkpoint = Path("at/deployable.ckpt")
    ldp_cfg_mapping = OmegaConf.to_container(policy_cfg(30), resolve=False)
    at_cfg_mapping = OmegaConf.to_container(tactile_cfg(30), resolve=False)
    ldp_payload = payload(ldp_cfg_mapping)
    at_payload = payload(at_cfg_mapping)
    payload_calls = []
    validated = False
    qualification_validated = False
    workspace_instances = []
    original_validate = deploy.validate_tactile_dimensions

    def load_payload(path, role):
        payload_calls.append((path, role))
        return ldp_payload if role == "LDP" else at_payload

    def validate(*args):
        nonlocal validated
        assert OmegaConf.is_config(args[1])
        assert OmegaConf.is_config(args[2])
        assert OmegaConf.to_container(args[1], resolve=False) == ldp_cfg_mapping
        assert OmegaConf.to_container(args[2], resolve=False) == at_cfg_mapping
        original_validate(*args)
        validated = True

    def validate_qualification(*args, **kwargs):
        nonlocal qualification_validated
        assert validated
        assert kwargs["slow_update_interval"] == 16
        qualification_validated = True

    class FakeWorkspace:
        def __init__(self, cfg):
            assert validated
            assert qualification_validated
            self.cfg = cfg
            self.ema_model = LoadedFakePolicy()
            self.model = LoadedFakePolicy()
            self.normalizer = object()
            self.loaded_payload = None
            workspace_instances.append(self)

        def load_payload(self, loaded_payload):
            self.loaded_payload = loaded_payload

    monkeypatch.setattr(deploy, "_load_checkpoint_payload", load_payload)
    monkeypatch.setattr(deploy, "validate_tactile_dimensions", validate)
    monkeypatch.setattr(deploy, "validate_release_qualification", validate_qualification)
    monkeypatch.setattr(deploy.hydra.utils, "get_class", lambda target: FakeWorkspace)

    with pytest.warns(UserWarning, match="legacy"):
        policy, cfg = deploy.load_policy(
            ldp_checkpoint,
            at_checkpoint,
            torch.device("cpu"),
            num_inference_steps=7,
            tactile_embedding_dim=30,
            slow_update_interval=16,
            artifact_verification="legacy-compatible",
        )

    assert payload_calls == [(ldp_checkpoint, "LDP"), (at_checkpoint, "AT")]
    assert len(workspace_instances) == 1
    assert workspace_instances[0].loaded_payload is ldp_payload
    assert policy is workspace_instances[0].ema_model
    assert cfg.at_load_dir == str(at_checkpoint)
    assert policy.at.normalizer is policy.normalizer
    assert policy.num_inference_steps == 7
    assert policy.eval_called
    assert policy.device == torch.device("cpu")


@pytest.mark.parametrize(
    ("pca_dim", "ldp_obs", "ldp_extended_obs", "at_obs", "at_extended_obs", "source"),
    [
        pytest.param(16, 30, 30, 30, 30, "PCA output", id="pca"),
        pytest.param(30, 16, 30, 30, 30, "LDP obs", id="ldp-obs"),
        pytest.param(30, 30, 16, 30, 30, "LDP extended_obs", id="ldp-extended-obs"),
        pytest.param(30, 30, 30, 16, 30, "AT obs", id="at-obs"),
        pytest.param(30, 30, 30, 30, 16, "AT extended_obs", id="at-extended-obs"),
    ],
)
def test_load_policy_rejects_each_mismatched_tactile_dimension_before_workspace(
    monkeypatch,
    pca_dim,
    ldp_obs,
    ldp_extended_obs,
    at_obs,
    at_extended_obs,
    source,
) -> None:
    ldp_checkpoint = Path("ldp.ckpt")
    at_checkpoint = Path("at.ckpt")
    ldp_payload = payload(policy_cfg(ldp_obs))
    ldp_payload["cfg"].shape_meta.extended_obs.tactile_embedding.shape = [ldp_extended_obs]
    at_payload = payload(tactile_cfg(at_obs, at_extended_obs))

    monkeypatch.setattr(
        deploy,
        "_load_checkpoint_payload",
        lambda path, role: ldp_payload if role == "LDP" else at_payload,
    )
    monkeypatch.setattr(
        deploy.hydra.utils,
        "get_class",
        lambda target: pytest.fail("workspace construction must follow validation"),
    )

    with pytest.raises(ValueError, match=source):
        deploy.load_policy(
            ldp_checkpoint,
            at_checkpoint,
            torch.device("cpu"),
            num_inference_steps=8,
            tactile_embedding_dim=pca_dim,
            slow_update_interval=16,
            artifact_verification="legacy-compatible",
        )


def _install_fake_policy_load(monkeypatch, ldp_cfg, at_cfg):
    class FakeWorkspace:
        def __init__(self, cfg):
            self.ema_model = LoadedFakePolicy()
            self.model = LoadedFakePolicy()

        def load_payload(self, loaded_payload):
            pass

    monkeypatch.setattr(
        deploy,
        "_load_checkpoint_payload",
        lambda path, role: payload(ldp_cfg if role == "LDP" else at_cfg),
    )
    monkeypatch.setattr(deploy.hydra.utils, "get_class", lambda target: FakeWorkspace)


def test_load_policy_strict_accepts_exact_v2_bundle(tmp_path, monkeypatch) -> None:
    at_path = tmp_path / "deployable.ckpt"
    at_path.write_bytes(b"AT checkpoint")
    pca_path = tmp_path / "pca.npz"
    pca_path.write_bytes(b"PCA artifact")
    at_artifacts = artifact_manifest(pca_sha256=deploy.sha256_file(pca_path))
    ldp_artifacts = artifact_manifest(
        normalizer_sha256="n" * 64,
        pca_sha256=deploy.sha256_file(pca_path),
        at_sha256=deploy.sha256_file(at_path),
        latent_target_mode="posterior_mode_post_vq",
    )
    _install_fake_policy_load(
        monkeypatch,
        policy_cfg(30, ldp_artifacts),
        policy_cfg(30, at_artifacts),
    )

    policy, _ = deploy.load_policy(
        tmp_path / "ldp" / "deployable.ckpt",
        at_path,
        torch.device("cpu"),
        num_inference_steps=8,
        tactile_embedding_dim=30,
        slow_update_interval=16,
        artifact_verification="strict",
        tactile_pca_path=pca_path,
    )

    assert policy.eval_called


@pytest.mark.parametrize(
    ("ldp_change", "message"),
    [
        ({"at_sha256": "x" * 64}, "AT"),
        ({"pca_sha256": "x" * 64}, "PCA"),
        ({"normalizer_sha256": "x" * 64}, "normalizer"),
    ],
)
def test_load_policy_strict_rejects_same_dimension_different_artifact(
    tmp_path, monkeypatch, ldp_change, message
) -> None:
    at_path = tmp_path / "at.ckpt"
    at_path.write_bytes(b"AT checkpoint")
    pca_path = tmp_path / "pca.npz"
    pca_path.write_bytes(b"PCA artifact")
    pca_sha256 = deploy.sha256_file(pca_path)
    at_artifacts = artifact_manifest(pca_sha256=pca_sha256)
    ldp_artifacts = artifact_manifest(
        **{
            "normalizer_sha256": "n" * 64,
            "pca_sha256": pca_sha256,
            "at_sha256": deploy.sha256_file(at_path),
            "latent_target_mode": "posterior_mode_post_vq",
            **ldp_change,
        }
    )
    _install_fake_policy_load(
        monkeypatch,
        policy_cfg(30, ldp_artifacts),
        policy_cfg(30, at_artifacts),
    )

    with pytest.raises(ValueError, match=message):
        deploy.load_policy(
            tmp_path / "ldp.ckpt",
            at_path,
            torch.device("cpu"),
            8,
            30,
            slow_update_interval=16,
            artifact_verification="strict",
            tactile_pca_path=pca_path,
        )


def test_load_policy_strict_rejects_v1_v2_mixing(tmp_path, monkeypatch) -> None:
    at_path = tmp_path / "at.ckpt"
    at_path.write_bytes(b"AT checkpoint")
    _install_fake_policy_load(
        monkeypatch,
        policy_cfg(
            30,
            artifact_manifest(
                at_sha256=deploy.sha256_file(at_path),
                latent_target_mode="posterior_mode_post_vq",
            ),
        ),
        policy_cfg(30),
    )

    with pytest.raises(ValueError, match="AT.*artifacts"):
        deploy.load_policy(
            tmp_path / "ldp.ckpt",
            at_path,
            torch.device("cpu"),
            8,
            30,
            slow_update_interval=16,
            artifact_verification="strict",
        )


def test_load_policy_strict_rejects_missing_v2_metadata(monkeypatch) -> None:
    _install_fake_policy_load(monkeypatch, policy_cfg(30), policy_cfg(30))

    with pytest.raises(ValueError, match="LDP.*artifacts"):
        deploy.load_policy(
            Path("ldp.ckpt"),
            Path("at.ckpt"),
            torch.device("cpu"),
            8,
            30,
            slow_update_interval=16,
            artifact_verification="strict",
        )


def test_legacy_metadata_requires_explicit_compatibility_mode(monkeypatch) -> None:
    _install_fake_policy_load(monkeypatch, policy_cfg(30), policy_cfg(30))

    with pytest.warns(UserWarning, match="legacy"):
        policy, _ = deploy.load_policy(
            Path("ldp/deployable.ckpt"),
            Path("at/deployable.ckpt"),
            torch.device("cpu"),
            8,
            30,
            slow_update_interval=16,
            artifact_verification="legacy-compatible",
        )
    assert policy.eval_called

    with pytest.raises(ValueError, match="artifact_verification"):
        deploy.load_policy(
            Path("ldp.ckpt"),
            Path("at.ckpt"),
            torch.device("cpu"),
            8,
            30,
            slow_update_interval=16,
            artifact_verification="off",
        )


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param(16, id="scalar"),
        pytest.param("7", id="string"),
        pytest.param([16.9], id="fractional"),
        pytest.param([True], id="boolean"),
        pytest.param([16, 30], id="multiple_items"),
        pytest.param([0], id="zero"),
        pytest.param([-1], id="negative"),
    ],
)
def test_tactile_dim_rejects_malformed_shapes(shape) -> None:
    cfg = OmegaConf.create(
        {"shape_meta": {"obs": {"tactile_embedding": {"shape": shape}}}}
    )

    with pytest.raises(ValueError):
        deploy._tactile_dim(cfg, "LDP", "obs")


@pytest.mark.parametrize(
    "profile",
    [deploy.DUAL_ARM_20X20, deploy.SINGLE_RIGHT_ARM_7X10],
    ids=["dual_arm", "single_right"],
)
def test_state_action_dimensions_accept_matching_ldp_and_at(profile) -> None:
    deploy.validate_state_action_dimensions(
        profile,
        state_action_cfg(profile.state_dim, profile.action_dim),
        state_action_cfg(profile.state_dim, profile.action_dim),
        Path("ldp.ckpt"),
        Path("at.ckpt"),
    )


@pytest.mark.parametrize(
    ("ldp_state", "ldp_action", "at_state", "at_action", "source"),
    [
        pytest.param(20, 10, 7, 10, "LDP state", id="ldp-state"),
        pytest.param(7, 20, 7, 10, "LDP action", id="ldp-action"),
        pytest.param(7, 10, 20, 10, "AT state", id="at-state"),
        pytest.param(7, 10, 7, 20, "AT action", id="at-action"),
    ],
)
def test_state_action_dimensions_reject_mismatching_ldp_or_at(
    ldp_state, ldp_action, at_state, at_action, source
) -> None:
    with pytest.raises(ValueError, match=source):
        deploy.validate_state_action_dimensions(
            deploy.SINGLE_RIGHT_ARM_7X10,
            state_action_cfg(ldp_state, ldp_action),
            state_action_cfg(at_state, at_action),
            Path("ldp.ckpt"),
            Path("at.ckpt"),
        )


def test_single_right_runtime_projects_state_and_expands_wire_action() -> None:
    policy = FakePolicy(15, state_dim=7, action_dim=10)
    encoder = FakeTactileEncoder()
    means = np.zeros((2, 1024), dtype=np.float32)
    components = np.zeros((2, 15, 1024), dtype=np.float32)
    components[:, np.arange(15), np.arange(15)] = 1.0
    tactile_pca = deploy.BimanualTactilePCA(means, components)
    runtime = deploy.PickTubeRDPRuntime(
        policy,
        encoder,
        torch.device("cpu"),
        tactile_pca,
        slow_update_interval=5,
        dataset_obs_temporal_downsample_ratio=2,
        n_obs_steps=2,
        profile=deploy.SINGLE_RIGHT_ARM_7X10,
    )

    bridge_observation = observation()
    bridge_observation["observation.state"] = np.arange(20, dtype=np.float32)
    policy_action, slow_update = runtime.predict(bridge_observation)
    wire_action = deploy.wire_action_for_profile(
        policy_action, bridge_observation, deploy.SINGLE_RIGHT_ARM_7X10
    )

    assert slow_update is True
    assert policy_action.shape == (1, 10)
    assert policy_action.dtype == np.float32
    assert policy.slow_observation_states == [[7.0, 7.0]]
    assert wire_action.shape == (1, 20)
    np.testing.assert_array_equal(wire_action[:, :3], 0)
    np.testing.assert_array_equal(wire_action[:, 3:9], [[1, 0, 0, 0, 1, 0]])
    np.testing.assert_array_equal(
        wire_action[:, 9], bridge_observation["observation.state"][6]
    )
    np.testing.assert_array_equal(wire_action[:, 10:], policy_action)


@pytest.mark.parametrize("components_per_arm", [8, 15, 30])
def test_runtime_updates_slow_plan_every_five_steps_and_decodes_every_step(
    components_per_arm: int,
) -> None:
    policy = FakePolicy(components_per_arm)
    encoder = FakeTactileEncoder()
    means = np.zeros((2, 1024), dtype=np.float32)
    components = np.zeros((2, components_per_arm, 1024), dtype=np.float32)
    components[:, np.arange(components_per_arm), np.arange(components_per_arm)] = 1.0
    tactile_pca = deploy.BimanualTactilePCA(means, components)
    runtime = deploy.PickTubeRDPRuntime(
        policy,
        encoder,
        torch.device("cpu"),
        tactile_pca,
        slow_update_interval=5,
        dataset_obs_temporal_downsample_ratio=2,
        n_obs_steps=2,
    )

    slow_updates = []
    actions = []
    for step in range(7):
        action, slow_update = runtime.predict(observation(step))
        actions.append(action)
        slow_updates.append(slow_update)

    assert slow_updates == [True, False, False, False, False, True, False]
    assert policy.slow_calls == 2
    assert policy.slow_observation_states == [[0.0, 0.0], [3.0, 5.0]]
    assert policy.fast_history_lengths == [4, 5, 6, 7, 8, 4, 5]
    assert all(action.shape == (1, 20) and action.dtype == np.float32 for action in actions)
    np.testing.assert_allclose(encoder.last_means, np.arange(1, 5) / 255.0)


def test_runtime_slow16_retains_the_deployed_phase_history_sequence() -> None:
    policy = FakePolicy(15)
    encoder = FakeTactileEncoder()
    means = np.zeros((2, 1024), dtype=np.float32)
    components = np.zeros((2, 15, 1024), dtype=np.float32)
    components[:, np.arange(15), np.arange(15)] = 1.0
    tactile_pca = deploy.BimanualTactilePCA(means, components)
    runtime = deploy.PickTubeRDPRuntime(
        policy,
        encoder,
        torch.device("cpu"),
        tactile_pca,
        slow_update_interval=16,
        dataset_obs_temporal_downsample_ratio=2,
        n_obs_steps=2,
    )

    slow_updates = []
    for step in range(17):
        _, slow_update = runtime.predict(observation(step))
        slow_updates.append(slow_update)

    assert slow_updates == [True] + [False] * 15 + [True]
    assert policy.fast_history_lengths == list(range(4, 20)) + [4]
