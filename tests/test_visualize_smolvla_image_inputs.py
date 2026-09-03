from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "visualize_smolvla_image_inputs.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("visualize_smolvla_image_inputs", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _images(values: list[tuple[int, int, int]]) -> np.ndarray:
    return np.stack([np.full((8, 8, 3), value, dtype=np.uint8) for value in values])


def test_select_even_indices_includes_endpoints():
    module = _load_module()

    assert module.select_even_indices(13, 6).tolist() == [0, 2, 5, 7, 10, 12]


def test_write_comparison_creates_readable_png(tmp_path, monkeypatch):
    module = _load_module()
    training = SimpleNamespace(
        root=tmp_path / "training",
        parquet_paths=(),
        states=np.zeros((3, 20), dtype=np.float32),
        actions=np.zeros((3, 20), dtype=np.float32),
        episode_indices=np.array([25, 26, 27]),
        frame_indices=np.array([0, 1, 2]),
        camera0_rgb=_images([(220, 30, 20), (180, 40, 30), (140, 50, 40)]),
        camera1_rgb=_images([(20, 220, 30), (30, 180, 40), (40, 140, 50)]),
    )
    saved = [
        SimpleNamespace(
            step=step,
            timestamp=float(step),
            left_pose=np.zeros(6),
            left_gripper=0.1,
            camera0_rgb=_images([(20 + step, 30, 220)])[0],
            right_pose=np.zeros(6),
            right_gripper=0.1,
            camera1_rgb=_images([(30, 20 + step, 220)])[0],
        )
        for step in (0, 10, 20)
    ]
    monkeypatch.setattr(module, "load_training_parquets", lambda _: training)
    monkeypatch.setattr(module, "load_saved_observations", lambda _: saved)

    result = module.write_image_input_comparison(
        tmp_path / "training",
        tmp_path / "obs",
        tmp_path / "comparison.png",
        sample_count=2,
    )

    with Image.open(result) as image:
        assert image.format == "PNG"
        assert image.width > image.height > 0
