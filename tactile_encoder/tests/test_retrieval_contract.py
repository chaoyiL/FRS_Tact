from __future__ import annotations

from tactile_encoder.evaluate_retrieval import build_parser
from tactile_encoder.utils.data import FutureRecord
from tactile_encoder.utils.data import future_records_digest


def test_split_cli_defaults_defer_to_checkpoint_metadata() -> None:
    args = build_parser().parse_args(
        [
            "--checkpoint-dir",
            "checkpoint",
            "--dataset-repo-id",
            "dataset",
            "--output-dir",
            "output",
        ]
    )

    assert args.split_seed is None
    assert args.val_fraction is None
    assert args.frame_stride is None
    assert args.future_offset is None
    assert args.eval_mask_seed is None


def test_future_records_digest_covers_split_membership() -> None:
    train = FutureRecord(1, 2, 0, "train")
    val = FutureRecord(1, 2, 0, "val")

    assert future_records_digest((train,)) != future_records_digest((val,))
