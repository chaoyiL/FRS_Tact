from __future__ import annotations

import csv
import pathlib
import tempfile
import unittest
from unittest import mock

import train_frs.utils.history_plot as history_plot_module
from train_frs.utils.history_plot import HISTORY_FIELDS, plot_training_history


class HistoryPlotTest(unittest.TestCase):
    def test_keeps_five_panels_before_first_validation_epoch(self):
        with tempfile.TemporaryDirectory() as directory:
            history = pathlib.Path(directory) / "history.csv"
            row = dict.fromkeys(HISTORY_FIELDS, "")
            row.update(
                {
                    "epoch": 1,
                    "train_loss_total": 0.3,
                    "train_loss_gt_fm": 0.10,
                    "train_loss_vla_fm": 0.05,
                    "train_loss_decode": 0.06,
                    "train_loss_rank": 0.04,
                    "train_loss_repair": 0.05,
                    "train_flow_loss": 0.3,
                }
            )
            with history.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=HISTORY_FIELDS)
                writer.writeheader()
                writer.writerow(row)

            with mock.patch.object(
                history_plot_module.plt,
                "subplots",
                wraps=history_plot_module.plt.subplots,
            ) as subplots:
                output = plot_training_history(history)

            self.assertEqual(subplots.call_args.args[:2], (5, 1))
            self.assertTrue(output.is_file())

    def test_plots_vla_baseline_repair_and_gate_quantiles(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            history = root / "history.csv"
            row = dict.fromkeys(HISTORY_FIELDS, "")
            row.update(
                {
                    "epoch": 5,
                    "train_loss_total": 0.3,
                    "train_loss_gt_fm": 0.10,
                    "train_loss_vla_fm": 0.05,
                    "train_loss_decode": 0.06,
                    "train_loss_rank": 0.04,
                    "train_loss_repair": 0.05,
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
                    "val_rank_satisfied_high_frac": 0.8,
                    "val_rank_satisfied_low_frac": 0.9,
                    "val_repair_satisfied_high_frac": 0.7,
                    "val_gate_w_p10": 0.1,
                    "val_gate_w_p50": 0.5,
                    "val_gate_w_p90": 0.9,
                    "val_tactile_change_p10": 0.1,
                    "val_tactile_change_p50": 0.5,
                    "val_tactile_change_p90": 0.9,
                    "val_n_high_w": 40,
                    "val_n_low_w": 60,
                    "checkpoint_selection_key": "1,0.0906691381335,-0.015307482332,0.184984356165",
                    "checkpoint_selection_feasible": 0,
                }
            )
            for index, bin_id in enumerate(("00_01", "01_03", "03_05", "05_07", "07_09", "09_10")):
                row[f"val_gate_bin_{bin_id}_n"] = 10 + index
                row[f"val_gate_bin_{bin_id}_mse_gt"] = 0.20 - 0.01 * index
                row[f"val_gate_bin_{bin_id}_mse_pred"] = 0.05 + 0.02 * index
            with history.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=HISTORY_FIELDS)
                writer.writeheader()
                writer.writerow(row)

            with mock.patch.object(
                history_plot_module.plt,
                "subplots",
                wraps=history_plot_module.plt.subplots,
            ) as subplots:
                output = plot_training_history(history)
            self.assertEqual(subplots.call_args.args[:2], (6, 1))
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
