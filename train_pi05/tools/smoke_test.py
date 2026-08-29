"""Cheap migration checks that do not initialize the multi-billion parameter model."""

from openpi.training import config as training_config


def main() -> None:
    expected_configs = {
        "pi05_single",
        "pi05_bi",
        "pi05_bi_no_state",
    }

    missing = expected_configs.difference(training_config._CONFIGS_DICT)
    if missing:
        raise RuntimeError(f"Missing migrated configs: {sorted(missing)}")
    print("pure-vision pi0.5 smoke test passed")


if __name__ == "__main__":
    main()
