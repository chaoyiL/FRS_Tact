from train_deco.metrics import wandb_payload


def test_train_step_payload_uses_train_namespace() -> None:
    payload = wandb_payload(
        {
            "event": "train_step",
            "epoch": 2,
            "global_step": 17,
            "loss": 0.25,
            "velocity_mae": 0.5,
            "gate_values": {"layer0": 0.1},
        }
    )

    assert payload["epoch"] == 2
    assert payload["global_step"] == 17
    assert payload["train/loss"] == 0.25
    assert payload["train/velocity_mae"] == 0.5
    assert payload["train/gate_values/layer0"] == 0.1


def test_epoch_payload_flattens_validation_metrics() -> None:
    payload = wandb_payload(
        {
            "event": "epoch",
            "epoch": 3,
            "global_step": 40,
            "train": {"loss": 0.3},
            "val_unseen": {"loss": 0.2, "velocity_mae": 0.4},
        }
    )

    assert payload["train/loss"] == 0.3
    assert payload["val_unseen/loss"] == 0.2
    assert payload["val_unseen/velocity_mae"] == 0.4
