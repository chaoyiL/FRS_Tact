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


def test_vt_model_preserves_explicit_tactile_call_interfaces():
    import inspect

    from train_vtsmolvla import VTJaxSmolVLA

    for method_name in (
        "embed_prefix",
        "flow_velocity",
        "build_prefix_context",
        "sample_actions",
    ):
        parameters = inspect.signature(getattr(VTJaxSmolVLA, method_name)).parameters
        assert "tactile_images" in parameters, method_name
        assert "tactile_embeddings" in parameters, method_name
        assert "tactile_masks" in parameters, method_name
    sample_parameters = inspect.signature(VTJaxSmolVLA.sample_actions).parameters
    for parameter in (
        "noise",
        "num_steps",
        "previous_chunk",
        "inference_delay",
        "execution_horizon",
    ):
        assert parameter in sample_parameters
