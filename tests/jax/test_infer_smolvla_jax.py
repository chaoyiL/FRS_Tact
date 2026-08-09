import json

from tools import infer_smolvla_jax as infer


def test_policy_type_is_selected_from_checkpoint_config(tmp_path) -> None:
    from train_smolvla import JaxSmolVLAPolicy
    from train_vtsmolvla import VTJaxSmolVLAPolicy

    visual = tmp_path / "visual"
    visual.mkdir()
    (visual / "config.json").write_text(json.dumps({"use_tactile_encoder": False}))
    tactile = tmp_path / "tactile"
    tactile.mkdir()
    (tactile / "config.json").write_text(json.dumps({"use_tactile_encoder": True}))

    assert infer._policy_type_from_snapshot(visual) is JaxSmolVLAPolicy
    assert infer._policy_type_from_snapshot(tactile) is VTJaxSmolVLAPolicy
