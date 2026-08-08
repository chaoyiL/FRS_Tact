def test_vt_package_is_discoverable():
    import importlib.util

    assert importlib.util.find_spec("train_vtsmolvla") is not None


def test_vt_config_extends_visual_config():
    from train_smolvla import JaxSmolVLAConfig
    from train_vtsmolvla import VTSmolVLAConfig

    config = VTSmolVLAConfig(
        tactile_keys=("observation.images.tactile",),
        tactile_num_tokens=1,
    )

    assert isinstance(config, JaxSmolVLAConfig)


def test_vt_runtime_types_extend_visual_primitives():
    from train_smolvla.data import LeRobotJaxDataLoader
    from train_smolvla.modeling import JaxSmolVLA
    from train_smolvla.policy import JaxSmolVLAPolicy
    from train_smolvla.preprocessing import JaxSmolVLAPreprocessor
    from train_smolvla.training import JaxSmolVLATrainer
    from train_vtsmolvla import (
        VTJaxSmolVLA,
        VTJaxSmolVLAPolicy,
        VTJaxSmolVLAPreprocessor,
        VTJaxSmolVLATrainer,
        VTLeRobotJaxDataLoader,
    )

    assert issubclass(VTJaxSmolVLA, JaxSmolVLA)
    assert issubclass(VTJaxSmolVLAPolicy, JaxSmolVLAPolicy)
    assert issubclass(VTJaxSmolVLATrainer, JaxSmolVLATrainer)
    assert issubclass(VTLeRobotJaxDataLoader, LeRobotJaxDataLoader)
    assert issubclass(VTJaxSmolVLAPreprocessor, JaxSmolVLAPreprocessor)
