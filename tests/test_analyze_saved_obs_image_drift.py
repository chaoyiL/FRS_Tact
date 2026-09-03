from __future__ import annotations

import importlib.util
import io
import json
import math
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "analyze_saved_obs_image_drift.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("analyze_saved_obs_image_drift", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _encoded_image(rgb: tuple[int, int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(np.full((8, 8, 3), rgb, dtype=np.uint8), mode="RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def _write_saved_image(path: Path, rgb: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.open(io.BytesIO(_encoded_image(rgb))).save(path, format="PNG")


def _write_episode(path: Path, episode_id: int, camera0, camera1) -> None:
    image_type = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    table = pa.table(
        {
            "observation.images.camera0": pa.array(
                [{"bytes": _encoded_image(rgb), "path": None} for rgb in camera0], type=image_type
            ),
            "observation.images.camera1": pa.array(
                [{"bytes": _encoded_image(rgb), "path": None} for rgb in camera1], type=image_type
            ),
            "episode_index": pa.array([episode_id] * len(camera0), type=pa.int64()),
            "frame_index": pa.array(range(len(camera0)), type=pa.int64()),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def _all_numbers_finite(value) -> bool:
    if isinstance(value, dict):
        return all(_all_numbers_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_all_numbers_finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def test_histogram_distance_is_zero_for_identity_and_finite_for_disjoint_inputs():
    module = _load_module()
    first = np.array([1.0, 0.0, 0.0])
    second = np.array([0.0, 0.0, 1.0])

    assert module.jensen_shannon_distance(first, first) == 0.0
    distance = module.jensen_shannon_distance(first, second)
    assert math.isfinite(distance)
    assert distance > 0.9


def test_saved_frames_use_numeric_step_order_and_swap_legacy_red_blue_once(tmp_path):
    module = _load_module()
    saved_obs = tmp_path / "saved"
    _write_saved_image(saved_obs / "step_10" / "camera0_rgb.jpg", (10, 20, 30))
    _write_saved_image(saved_obs / "step_2" / "camera0_rgb.jpg", (40, 50, 60))
    _write_saved_image(saved_obs / "step_2" / "camera1_rgb.jpg", (70, 80, 90))

    normal = module.load_saved_frames(saved_obs, legacy_saved_rgb_swap=False)
    legacy = module.load_saved_frames(saved_obs, legacy_saved_rgb_swap=True)

    assert [(item["step_index"], item["camera"]) for item in normal] == [
        (2, "camera0"),
        (2, "camera1"),
        (10, "camera0"),
    ]
    np.testing.assert_array_equal(normal[0]["image"][0, 0], [40, 50, 60])
    np.testing.assert_array_equal(legacy[0]["image"][0, 0], [60, 50, 40])


def test_analyze_samples_each_episode_and_keeps_camera_baselines_separate(tmp_path):
    module = _load_module()
    episode7 = tmp_path / "episode_000007.parquet"
    episode8 = tmp_path / "episode_000008.parquet"
    _write_episode(episode7, 7, [(240, 0, 0)] * 4, [(0, 0, 240)] * 4)
    _write_episode(episode8, 8, [(200, 0, 0)] * 5, [(0, 0, 200)] * 5)
    saved_obs = tmp_path / "saved"
    _write_saved_image(saved_obs / "step_0" / "camera0_rgb.jpg", (220, 0, 0))
    _write_saved_image(saved_obs / "step_0" / "camera1_rgb.jpg", (0, 0, 220))

    summary = module.analyze(
        [episode7, episode8],
        saved_obs,
        legacy_saved_bgr=False,
        hf_revision="immutable-revision",
        frames_per_episode=2,
    )

    assert summary["reference"]["episode_ids"] == [7, 8]
    assert summary["reference"]["total_frames"] == 9
    assert summary["reference"]["sampled_frames_per_camera"] == 4
    assert summary["deployment"]["total_steps"] == 1
    assert summary["deployment"]["frames_per_camera"] == {"camera0": 1, "camera1": 1}
    assert summary["cameras"]["camera0"]["reference"]["metrics_mean"]["red_mean"] > 0.8
    assert summary["cameras"]["camera0"]["reference"]["metrics_mean"]["blue_mean"] < 0.01
    assert summary["cameras"]["camera1"]["reference"]["metrics_mean"]["blue_mean"] > 0.8
    assert summary["cameras"]["camera1"]["reference"]["metrics_mean"]["red_mean"] < 0.01
    assert len(summary["cameras"]["camera0"]["reference_episode_summaries"]) == 2


def test_write_report_emits_finite_json_and_frame_csv(tmp_path):
    module = _load_module()
    episode = tmp_path / "episode_000003.parquet"
    _write_episode(episode, 3, [(100, 110, 120)] * 3, [(130, 140, 150)] * 3)
    saved_obs = tmp_path / "saved"
    _write_saved_image(saved_obs / "step_12" / "camera0_rgb.jpg", (100, 110, 120))
    _write_saved_image(saved_obs / "step_12" / "camera1_rgb.jpg", (130, 140, 150))

    summary = module.analyze(
        [episode], saved_obs, legacy_saved_bgr=False, hf_revision="rev", frames_per_episode=0
    )
    output_dir = tmp_path / "report"
    module.write_report(summary, output_dir)

    loaded = json.loads((output_dir / "summary.json").read_text())
    assert _all_numbers_finite(loaded)
    assert loaded["reference"]["sampled_frames_per_camera"] == 3
    csv_lines = (output_dir / "frame_metrics.csv").read_text().splitlines()
    assert len(csv_lines) == 1 + 3 * 2 + 1 * 2


def test_analyzer_source_has_no_model_framework_or_deployment_imports():
    source = MODULE_PATH.read_text()
    forbidden = ("import torch", "import jax", "openpi", "remote_client", "robot_bridge")
    assert not any(token in source for token in forbidden)
