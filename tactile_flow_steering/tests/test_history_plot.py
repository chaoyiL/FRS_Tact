from __future__ import annotations

import csv
import pathlib
import tempfile
import unittest

from tactile_flow_steering.utils.history_plot import HISTORY_FIELDS
from tactile_flow_steering.utils.history_plot import plot_training_history


class HistoryPlotTest(unittest.TestCase):
    def test_plots_vla_baseline_repair_and_gate_quantiles(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            history = root / "history.csv"
            row = dict.fromkeys(HISTORY_FIELDS, "")
            row.update(
                {
                    "epoch": 5,
                    "train_flow_loss": 0.3,
                    "val_flow_loss": 0.2,
                    "val_mse_gt_high_w": 0.15,
                    "val_mse_gt_low_w": 0.12,
                    "val_mse_pred_high_w": 0.10,
                    "val_mse_pred_low_w": 0.05,
                    "val_mse_vla_gt_high_w": 0.20,
                    "val_mse_vla_gt_low_w": 0.13,
                    "val_gt_gain_high_w": 0.05,
                    "val_gt_gain_low_w": 0.01,
                    "val_relative_gt_error_high_w": 0.75,
                    "val_relative_gt_error_low_w": 0.92,
                    "val_gate_w_p10": 0.1,
                    "val_gate_w_p50": 0.5,
                    "val_gate_w_p90": 0.9,
                    "val_tactile_change_p10": 0.1,
                    "val_tactile_change_p50": 0.5,
                    "val_tactile_change_p90": 0.9,
                    "val_n_high_w": 40,
                    "val_n_low_w": 60,
                }
            )
            with history.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=HISTORY_FIELDS)
                writer.writeheader()
                writer.writerow(row)

            output = plot_training_history(history)
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
