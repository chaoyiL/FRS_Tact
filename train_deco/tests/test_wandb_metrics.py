import sys
from types import SimpleNamespace

from train_deco.metrics import WandbMetricsLogger, wandb_payload


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


def test_wandb_resume_status_is_only_queried_for_training_resume(monkeypatch, tmp_path) -> None:
    init_options: list[dict] = []
    run = SimpleNamespace(define_metric=lambda *args, **kwargs: None)

    def fake_init(**kwargs):
        init_options.append(kwargs)
        return run

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=fake_init))
    arguments = {
        "project": "deco-stage2",
        "entity": None,
        "name": "insert-stage2",
        "run_id": "insert-stage2",
        "group": None,
        "tags": [],
        "mode": "online",
        "output_dir": tmp_path,
        "config": {},
    }

    WandbMetricsLogger(**arguments)
    WandbMetricsLogger(**arguments, resume="allow")

    assert "resume" not in init_options[0]
    assert init_options[1]["resume"] == "allow"
