import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "run_rdp_counterfactual_replay.py"
SPEC = importlib.util.spec_from_file_location("rdp_counterfactual_replay", SCRIPT_PATH)
replay = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(replay)


def test_main_passes_raw_control_slow_update_interval_to_policy_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class StopAfterPolicyLoad(Exception):
        pass

    captured = {}
    fake_deploy = ModuleType("deploy_pick_tube_rdp")

    def load_policy(*args, **kwargs):
        captured.update(kwargs)
        raise StopAfterPolicyLoad

    fake_deploy.load_policy = load_policy
    fake_deploy.PickTubeRDPRuntime = object
    monkeypatch.setitem(sys.modules, "deploy_pick_tube_rdp", fake_deploy)

    from reactive_diffusion_policy.model.tactile_pca import BimanualTactilePCA

    monkeypatch.setattr(
        BimanualTactilePCA,
        "from_npz",
        classmethod(lambda cls, path, device: type("FakePCA", (), {"output_dim": 30})()),
    )
    monkeypatch.setattr(
        replay,
        "parse_args",
        lambda: argparse.Namespace(
            config=tmp_path / "config.yaml",
            failure_trial=tmp_path / "failure",
            success_trial=tmp_path / "success",
            steps=1,
            warmup_runs=0,
            seeds=1,
            repeat_seed=7,
            repeats=1,
            device="cpu",
            output=tmp_path / "report.json",
            raw_output=tmp_path / "raw.npz",
        ),
    )
    monkeypatch.setattr(
        replay,
        "load_config",
        lambda path: {
            "model": {
                "ldp_checkpoint": "ldp/deployable.ckpt",
                "at_checkpoint": "at/deployable.ckpt",
                "tactile_pca_path": "pca.npz",
                "tactile_encoder_dir": "encoder",
            },
            "control": {"slow_update_interval": 16},
        },
    )

    with pytest.raises(StopAfterPolicyLoad):
        replay.main()

    assert captured["slow_update_interval"] == 16
