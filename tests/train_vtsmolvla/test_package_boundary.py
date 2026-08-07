def test_vt_package_is_discoverable():
    import importlib.util

    assert importlib.util.find_spec("train_vtsmolvla") is not None
