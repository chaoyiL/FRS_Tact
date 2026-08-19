from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx, traverse_util

from train_pi05_frs.utils.checkpoint import load_checkpoint, save_checkpoint
from train_pi05_frs.utils.model import DecoderConfig, TactileConditionedFlowDecoder


REPO_ROOT = Path(__file__).resolve().parents[2]
DEPLOY_PYTHON = Path(
    os.environ.get(
        "DEPLOY_PI05_PYTHON",
        "/home/typhon/FRS_Tact/deploy_pi05/.venv/bin/python",
    )
)
LEGACY_NONE_SLOT = (68, "str:tactile_gru/str:cell/str:dense_h/str:bias")


DEPLOY_LOAD_SCRIPT = r'''
import hashlib
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx, traverse_util

from deploy_pi05.frs_inference.decoder_checkpoint import load_checkpoint


def path_name(path):
    return "/".join(f"{type(part).__name__}:{part}" for part in path)


def digest_model(model):
    flat = traverse_util.flatten_dict(nnx.state(model, nnx.Param).to_pure_dict())
    digest = hashlib.sha256()
    for path, leaf in sorted(flat.items(), key=lambda item: path_name(item[0])):
        if leaf is None:
            continue
        array = np.ascontiguousarray(jax.device_get(leaf))
        digest.update(path_name(path).encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(repr(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


checkpoint_dir = Path(sys.argv[1])
model, metadata = load_checkpoint(checkpoint_dir)
config = model.config
x_t = jnp.ones((2, config.action_horizon, config.action_dim), dtype=jnp.float32)
t = jnp.asarray([0.25, 0.75], dtype=jnp.float32)
tactile = jnp.ones(
    (2, config.tactile_window, config.num_tactile_tokens, config.resnet_embedding_dim),
    dtype=jnp.float32,
)
state = (
    jnp.ones((2, config.state_dim), dtype=jnp.float32)
    if config.state_conditioning
    else None
)
output = model(x_t, t, tactile, state=state)
print(json.dumps({
    "digest": digest_model(model),
    "finite": bool(jnp.all(jnp.isfinite(output))),
    "parameter_paths": metadata["parameter_paths"],
}, sort_keys=True))
'''


def _path_name(path: tuple[object, ...]) -> str:
    return "/".join(f"{type(part).__name__}:{part}" for part in path)


def _flat_parameters(model: TactileConditionedFlowDecoder):
    return traverse_util.flatten_dict(nnx.state(model, nnx.Param).to_pure_dict())


def _numeric_parameters(model: TactileConditionedFlowDecoder):
    return {
        path: leaf for path, leaf in _flat_parameters(model).items() if leaf is not None
    }


def _model_digest(model: TactileConditionedFlowDecoder) -> str:
    digest = hashlib.sha256()
    for path, leaf in sorted(
        _numeric_parameters(model).items(), key=lambda item: _path_name(item[0])
    ):
        array = np.ascontiguousarray(jax.device_get(leaf))
        digest.update(_path_name(path).encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(repr(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _rewrite_as_legacy_full_checkpoint(
    checkpoint_dir: Path,
    model: TactileConditionedFlowDecoder,
) -> list[tuple[int, str]]:
    metadata_path = checkpoint_dir / "checkpoint.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with np.load(checkpoint_dir / "params.npz", allow_pickle=False) as archive:
        filtered_arrays = {
            path: np.asarray(archive[f"p{index:05d}"])
            for index, path in enumerate(metadata["parameter_paths"])
        }

    ordered_full = sorted(_flat_parameters(model).items(), key=lambda item: _path_name(item[0]))
    none_slots = [
        (index, _path_name(path))
        for index, (path, value) in enumerate(ordered_full)
        if value is None
    ]
    assert none_slots == [LEGACY_NONE_SLOT]
    legacy_arrays = {
        f"p{index:05d}": filtered_arrays[_path_name(path)]
        for index, (path, value) in enumerate(ordered_full)
        if value is not None
    }
    np.savez(checkpoint_dir / "params.npz", **legacy_arrays)
    metadata["version"] = 2
    metadata.pop("generation", None)
    metadata.pop("files", None)
    metadata["parameter_paths"] = [_path_name(path) for path, _ in ordered_full]
    metadata_path.write_text(json.dumps(metadata, sort_keys=True), encoding="utf-8")
    return none_slots


class DeploymentCheckpointCompatibilityTest(unittest.TestCase):
    def make_model(self) -> TactileConditionedFlowDecoder:
        return TactileConditionedFlowDecoder(
            DecoderConfig(
                action_dim=3,
                action_horizon=6,
                tactile_window=3,
                gru_hidden_dim=8,
                resnet_embedding_dim=4,
                model_dim=16,
                depth=2,
                num_heads=4,
                state_conditioning=True,
                state_dim=5,
            ),
            rngs=nnx.Rngs(917),
        )

    def assert_all_numeric_leaves_equal(self, expected, actual) -> None:
        expected_flat = _numeric_parameters(expected)
        actual_flat = _numeric_parameters(actual)
        self.assertEqual(set(expected_flat), set(actual_flat))
        for path in sorted(expected_flat, key=_path_name):
            self.assertTrue(
                np.array_equal(
                    np.asarray(jax.device_get(expected_flat[path])),
                    np.asarray(jax.device_get(actual_flat[path])),
                ),
                msg=f"parameter differs: {_path_name(path)}",
            )

    def test_training_checkpoint_loads_in_deployment_runtime_for_both_path_formats(self):
        self.assertTrue(DEPLOY_PYTHON.is_file(), f"missing deployment Python: {DEPLOY_PYTHON}")
        for path_format in ("filtered", "legacy_full"):
            with self.subTest(path_format=path_format), tempfile.TemporaryDirectory() as directory:
                checkpoint_dir = Path(directory) / "output" / "last"
                model = self.make_model()
                save_checkpoint(
                    checkpoint_dir,
                    model,
                    epoch=7,
                    metrics={"val_mse": 0.125},
                )
                if path_format == "legacy_full":
                    none_slots = _rewrite_as_legacy_full_checkpoint(checkpoint_dir, model)
                    with np.load(checkpoint_dir / "params.npz", allow_pickle=False) as archive:
                        self.assertEqual(none_slots, [LEGACY_NONE_SLOT])
                        self.assertNotIn("p00068", archive.files)
                        self.assertIn("p00069", archive.files)

                restored, metadata = load_checkpoint(checkpoint_dir)
                self.assert_all_numeric_leaves_equal(model, restored)

                result = subprocess.run(
                    [str(DEPLOY_PYTHON), "-c", DEPLOY_LOAD_SCRIPT, str(checkpoint_dir)],
                    cwd=REPO_ROOT,
                    env={
                        **os.environ,
                        "JAX_PLATFORMS": "cpu",
                        "PYTHONDONTWRITEBYTECODE": "1",
                        "PYTHONPATH": f"{REPO_ROOT / 'deploy_pi05/src'}:{REPO_ROOT}",
                    },
                    check=True,
                    text=True,
                    capture_output=True,
                )
                payload = json.loads(result.stdout.strip().splitlines()[-1])
                self.assertEqual(payload["digest"], _model_digest(model))
                self.assertEqual(payload["parameter_paths"], metadata["parameter_paths"])
                self.assertTrue(payload["finite"])


if __name__ == "__main__":
    unittest.main()
