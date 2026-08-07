def test_visual_package_is_discoverable():
    import importlib.util

    assert importlib.util.find_spec("train_smolvla") is not None


def test_visual_package_does_not_load_tactile_modules():
    import sys

    import train_smolvla

    assert "tactile_encoder" not in sys.modules
    assert "train_vtsmolvla" not in sys.modules


def test_visual_config_has_no_tactile_fields():
    from dataclasses import fields

    from train_smolvla import JaxSmolVLAConfig

    assert not {field.name for field in fields(JaxSmolVLAConfig) if "tactile" in field.name}
