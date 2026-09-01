from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "deploy_pick_tube_rdp_right.yaml"
LAUNCHER_PATH = ROOT / "scripts" / "start_pick_tube_rdp_right.sh"


def test_right_config_uses_downloaded_single_right_bundle() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    baseline = yaml.safe_load(
        (ROOT / "configs" / "deploy_pick_tube_rdp.yaml").read_text(encoding="utf-8")
    )

    assert config["model"] == {
        "ldp_checkpoint": (
            "/home/typhon/FRS_Tact/checkpoints/model/rdp/rdp_0831/ldp/latest.ckpt"
        ),
        "at_checkpoint": (
            "/home/typhon/FRS_Tact/checkpoints/model/rdp/rdp_0831/at/latest.ckpt"
        ),
        "tactile_encoder_dir": (
            "/home/typhon/FRS_Tact/checkpoints/encoder/encoder_ckpt_0824"
        ),
        "tactile_pca_path": (
            "/home/typhon/FRS_Tact/checkpoints/model/rdp/rdp_0831/pca/"
            "tactile_pca_insert_01_02_encoder0824_2x15.npz"
        ),
        "state_action_profile": "single-right-arm-7x10",
        "artifact_verification": "legacy-compatible",
        "device": "cuda:0",
        "num_inference_steps": 8,
    }
    assert config["connection"] == baseline["connection"]
    assert config["control"] == baseline["control"]
    assert config["runtime"] == baseline["runtime"]


def test_right_launcher_selects_right_config_and_reuses_client() -> None:
    launcher = LAUNCHER_PATH.read_text(encoding="utf-8")

    assert "deploy_pick_tube_rdp_right.yaml" in launcher
    assert "RDP_DEPLOY_CONFIG" in launcher
    assert 'start_pick_tube_rdp_client.sh" "$@"' in launcher
