def test_visual_package_is_discoverable():
    import importlib.util

    assert importlib.util.find_spec("train_smolvla") is not None
