import unittest

from train_encoder.train import _early_stop_state_after_eval
from train_encoder.train import _val_improved


class EarlyStopTrackingTests(unittest.TestCase):
    def test_first_eval_always_counts_as_improvement_from_sentinel(self) -> None:
        track_r, track_rank, last_epoch, since = _early_stop_state_after_eval(
            epoch=2,
            val_recall1=0.74,
            val_mean_rank=1.75,
            track_recall1=-1.0,
            track_mean_rank=float("inf"),
            last_improve_epoch=0,
        )
        self.assertEqual(track_r, 0.74)
        self.assertEqual(last_epoch, 2)
        self.assertEqual(since, 0)

    def test_stale_disk_best_does_not_block_counter(self) -> None:
        """Regression: old best/ on disk must not leave best_epoch=0 forever."""

        track_r, track_rank, last_epoch, since = (-1.0, float("inf"), 0, 0)
        for epoch, recall in ((2, 0.74), (4, 0.73), (6, 0.72), (8, 0.71)):
            track_r, track_rank, last_epoch, since = _early_stop_state_after_eval(
                epoch=epoch,
                val_recall1=recall,
                val_mean_rank=2.0,
                track_recall1=track_r,
                track_mean_rank=track_rank,
                last_improve_epoch=last_epoch,
            )
        self.assertEqual(last_epoch, 2)
        self.assertEqual(since, 6)

    def test_mean_rank_tiebreak(self) -> None:
        self.assertTrue(
            _val_improved(0.8, 1.5, best_recall1=0.8, best_mean_rank=2.0)
        )
        self.assertFalse(
            _val_improved(0.8, 2.5, best_recall1=0.8, best_mean_rank=2.0)
        )


if __name__ == "__main__":
    unittest.main()
