from __future__ import annotations

from train_encoder.train import parse_args
from train_encoder.utils.tasks import task_repo_ids


def test_default_config_has_four_tasks_with_300_training_episodes_each():
    args = parse_args([])
    assert [task["name"] for task in args.tasks] == ["pick", "insert", "bread", "two_tubes"]
    assert all(task["num_train_episodes"] == 300 for task in args.tasks)
    assert all(task["num_val_episodes"] == 35 for task in args.tasks)
    assert [task["sides"] for task in args.tasks] == [
        ["right"],
        ["right"],
        ["left", "right"],
        ["left", "right"],
    ]
    assert args.dataset_repo_ids == task_repo_ids(args.tasks)
    assert args.learning_rate == 3e-4
    assert args.dataset_repo_ids == [
        "KaiyueChen/pick_01",
        "KaiyueChen/insert_01",
        "KaiyueChen/bread_03",
        "KaiyueChen/two_tubes_01",
    ]
    assert args.dataset_manifest.name == "manifest.json"


def test_cli_override_keeps_legacy_dataset_mode(capsys):
    args = parse_args(["--dataset-repo-id", "org/custom", "--epochs", "2"])
    assert args.dataset_repo_ids == ["org/custom"]
    assert args.tasks is None
    assert args.dataset_manifest is None
    assert args.epochs == 2
    assert "legacy" in capsys.readouterr().out
