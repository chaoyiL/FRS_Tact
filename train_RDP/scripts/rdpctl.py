#!/usr/bin/env python3
"""Recipe-driven preparation and training controller for pick_tube RDP."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_DATASETS = tuple(f"pick_tube_{index:02d}" for index in range(1, 7))
DEFAULT_RECIPE = ROOT / "recipes" / "pick_tube_six30.yaml"
DEFAULT_PROFILE = ROOT / "profiles" / "rtxpro6000x4.yaml"
REQUIRED_PROFILE_PATHS = {
    "python",
    "accelerate",
    "jax_python",
    "dataset_root",
    "encoder_dir",
    "tactile_cache_root",
    "tactile_pca_path",
    "dataset_path",
    "output_root",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("plan", "doctor", "prepare", "train", "run", "resume", "status")
    )
    parser.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--run-id")
    parser.add_argument("--baseline-json", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite-precompute", action="store_true")
    parser.add_argument("--overwrite-conversion", action="store_true")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"configuration must contain a mapping: {path}")
    return value


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def display_command(command: list[str], env: dict[str, str] | None = None) -> str:
    prefix = ""
    if env:
        prefix = " ".join(
            f"{key}={shlex.quote(str(value))}" for key, value in sorted(env.items())
        ) + " "
    return prefix + shlex.join(str(value) for value in command)


def has_v2_manifest(zarr_path: Path) -> bool:
    attrs_path = zarr_path / "meta" / ".zattrs"
    if not attrs_path.is_file():
        return False
    try:
        attrs = json.loads(attrs_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(attrs.get("v2_manifest_json"))


def validate_v2_zarr_scope(
    zarr_path: Path, *, datasets: tuple[str, ...], tactile_dim: int
) -> None:
    """Reject a complete but semantically different Zarr before reuse/training."""
    attrs_path = zarr_path / "meta" / ".zattrs"
    try:
        attrs = json.loads(attrs_path.read_text(encoding="utf-8"))
        manifest_raw = attrs["v2_manifest_json"]
        manifest = json.loads(manifest_raw) if isinstance(manifest_raw, str) else manifest_raw
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid v2 Zarr metadata: {attrs_path}") from error
    if tuple(attrs.get("dataset_names", ())) != datasets:
        raise ValueError(
            f"Zarr dataset_names do not match the six-dataset recipe: {zarr_path}"
        )
    if not isinstance(manifest, dict):
        raise ValueError(f"v2 manifest must be a mapping: {attrs_path}")
    if manifest.get("action_representation_version") != 2:
        raise ValueError(f"Zarr is not action-contract-v2: {zarr_path}")
    if int(manifest.get("pca_output_dim", -1)) != tactile_dim:
        raise ValueError(
            f"Zarr tactile dimension does not match recipe PCA{tactile_dim}: {zarr_path}"
        )


class Pipeline:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.recipe = load_yaml(args.recipe.resolve())
        self.profile = load_yaml(args.profile.resolve())
        self.datasets = tuple(str(value) for value in self.recipe.get("datasets", ()))
        if self.datasets != EXPECTED_DATASETS:
            raise ValueError(
                "this controller is intentionally restricted to "
                f"{EXPECTED_DATASETS}, got {self.datasets}"
            )
        profile_paths = self.profile.get("paths")
        if not isinstance(profile_paths, dict):
            raise ValueError("profile.paths must be a mapping")
        missing_paths = sorted(REQUIRED_PROFILE_PATHS - profile_paths.keys())
        if missing_paths:
            raise ValueError(
                "profile.paths is missing required keys: " + ", ".join(missing_paths)
            )
        raw_gpu_ids = self.profile.get("gpu_ids")
        if not isinstance(raw_gpu_ids, list) or not raw_gpu_ids:
            raise ValueError("profile.gpu_ids must be a non-empty list")
        try:
            numeric_gpu_ids = [int(value) for value in raw_gpu_ids]
        except (TypeError, ValueError) as error:
            raise ValueError("profile.gpu_ids must contain non-negative integers") from error
        if any(value < 0 for value in numeric_gpu_ids) or len(set(numeric_gpu_ids)) != len(
            numeric_gpu_ids
        ):
            raise ValueError("profile.gpu_ids must contain unique non-negative integers")
        self.gpu_ids = tuple(str(value) for value in numeric_gpu_ids)
        self.paths = {key: resolve_path(value) for key, value in profile_paths.items()}
        if args.dataset_root is not None:
            self.paths["dataset_root"] = args.dataset_root.expanduser().resolve()
        if args.dataset_path is not None:
            self.paths["dataset_path"] = args.dataset_path.expanduser().resolve()
        if args.output_root is not None:
            self.paths["output_root"] = args.output_root.expanduser().resolve()
        self.run_id = args.run_id or str(self.recipe["name"])
        if not 1 <= len(self.run_id) <= 64:
            raise ValueError(
                f"run-id must contain 1-64 characters for W&B tags, got {len(self.run_id)}"
            )
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", self.run_id) is None:
            raise ValueError(
                "run-id must start with an alphanumeric character and contain only "
                "letters, numbers, dot, underscore, or hyphen"
            )
        data = self.recipe.get("data", {})
        if int(data.get("tactile_dim", -1)) != 2 * int(
            data.get("components_per_arm", -1)
        ):
            raise ValueError("data.tactile_dim must equal 2 * components_per_arm")
        self._validate_recipe_settings()
        training_gpu_id = int(self.recipe["training"]["gpu_id"])
        if str(training_gpu_id) not in self.gpu_ids:
            raise ValueError(
                f"training.gpu_id={training_gpu_id} is not listed in profile.gpu_ids"
            )
        self.dry_run = bool(args.dry_run or args.command == "plan")
        if args.baseline_json is not None:
            baseline_path = args.baseline_json.expanduser().resolve()
            if not baseline_path.is_file():
                raise FileNotFoundError(f"baseline JSON not found: {baseline_path}")
            baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
            for key in (
                "val_active_left_translation_mae_mm",
                "val_active_left_rotation_mae_deg",
            ):
                value = baseline.get(key) if isinstance(baseline, dict) else None
                if not isinstance(value, (int, float)) or float(value) <= 0:
                    raise ValueError(f"baseline field {key!r} must be a positive number")
        elif args.command in ("plan", "train", "run", "resume"):
            print(
                "warning: no frozen-v1 baseline JSON was provided; training can run, "
                "but its checkpoints remain non-deployable and cannot enter top-k",
                file=sys.stderr,
            )
        self.pipeline_dir = self.paths["output_root"] / "_pipeline" / self.run_id
        self.state_path = self.pipeline_dir / "run_state.json"
        self.resolved_config = {
            "recipe": self.recipe,
            "profile": self.profile,
            "paths": {key: str(value) for key, value in self.paths.items()},
        }
        self.config_digest = hashlib.sha256(
            json.dumps(
                self.resolved_config, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        self.state = self._load_state()

    def _validate_recipe_settings(self) -> None:
        positive_fields = {
            "precompute.batch_size": self.recipe.get("precompute", {}).get("batch_size"),
            "precompute.workers_per_gpu": self.recipe.get("precompute", {}).get(
                "workers_per_gpu"
            ),
            "precompute.prefetch_factor": self.recipe.get("precompute", {}).get(
                "prefetch_factor"
            ),
            "precompute.flush_every": self.recipe.get("precompute", {}).get(
                "flush_every"
            ),
            "pca.batch_size": self.recipe.get("pca", {}).get("batch_size"),
            "conversion.num_workers": self.recipe.get("conversion", {}).get(
                "num_workers"
            ),
            "conversion.rgb_chunk_frames": self.recipe.get("conversion", {}).get(
                "rgb_chunk_frames"
            ),
            "training.at_epochs": self.recipe.get("training", {}).get("at_epochs"),
            "training.ldp_epochs": self.recipe.get("training", {}).get("ldp_epochs"),
            "training.at_batch": self.recipe.get("training", {}).get("at_batch"),
            "training.ldp_batch": self.recipe.get("training", {}).get("ldp_batch"),
            "training.num_workers": self.recipe.get("training", {}).get("num_workers"),
            "training.checkpoint_every": self.recipe.get("training", {}).get(
                "checkpoint_every"
            ),
        }
        for name, value in positive_fields.items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        training = self.recipe.get("training", {})
        for name in ("periodic_keep", "topk_keep"):
            value = training.get(name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"training.{name} must be a non-negative integer")
        conversion = self.recipe.get("conversion", {})
        compression_level = conversion.get("compression_level")
        if (
            not isinstance(compression_level, int)
            or isinstance(compression_level, bool)
            or not 0 <= compression_level <= 9
        ):
            raise ValueError("conversion.compression_level must be an integer from 0 to 9")

    def _load_state(self) -> dict[str, Any]:
        if self.state_path.is_file():
            value = json.loads(self.state_path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                stored_digest = value.get("config_digest")
                if (
                    stored_digest
                    and stored_digest != self.config_digest
                    and self.args.command != "status"
                ):
                    raise ValueError(
                        "recipe/profile paths changed for an existing run-id; "
                        "use a new run-id or restore the original configuration"
                    )
                return value
        return {
            "run_id": self.run_id,
            "recipe": str(self.args.recipe.resolve()),
            "profile": str(self.args.profile.resolve()),
            "git_commit": self._git_commit(),
            "datasets": list(self.datasets),
            "resolved": self.resolved_config,
            "config_digest": self.config_digest,
            "stages": {},
        }

    @staticmethod
    def _git_commit() -> str:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or "unknown"

    def _save_state(self) -> None:
        if not self.dry_run:
            atomic_write_json(self.state_path, self.state)

    def _mark(self, stage: str, status: str, **extra: Any) -> None:
        record = dict(self.state["stages"].get(stage, {}))
        record.update(extra)
        record["status"] = status
        record[f"{status}_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.state["stages"][stage] = record
        self._save_state()

    def run_command(
        self,
        stage: str,
        command: list[str | Path],
        *,
        env: dict[str, str] | None = None,
    ) -> None:
        rendered = [str(value) for value in command]
        print(f"[{stage}] {display_command(rendered, env)}", flush=True)
        if self.dry_run:
            return
        running_record: dict[str, Any] = {"command": rendered}
        if env:
            running_record["environment"] = {
                key: str(value) for key, value in sorted(env.items())
            }
        self._mark(stage, "running", **running_record)
        process_env = os.environ.copy()
        if env:
            process_env.update({key: str(value) for key, value in env.items()})
        try:
            subprocess.run(rendered, cwd=ROOT, env=process_env, check=True)
        except BaseException as error:
            self._mark(stage, "failed", error=repr(error))
            raise
        self._mark(stage, "completed")

    def _require_file(self, path: Path, label: str, *, executable: bool = False) -> None:
        valid = path.is_file() and (not executable or os.access(path, os.X_OK))
        if not valid:
            suffix = " executable" if executable else " file"
            raise FileNotFoundError(f"{label}{suffix} not found: {path}")

    def _print_host_probe(self) -> None:
        print(f"host_cpu_count={os.cpu_count()}")
        for path in (
            Path("/sys/fs/cgroup/cpu.max"),
            Path("/sys/fs/cgroup/cpuset.cpus.effective"),
        ):
            if path.is_file():
                print(f"{path.name}={path.read_text(encoding='utf-8').strip()}")
        try:
            findmnt = subprocess.run(
                [
                    "findmnt",
                    "-T",
                    str(self.paths["dataset_path"]),
                    "-o",
                    "TARGET,SOURCE,FSTYPE,OPTIONS",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError:
            findmnt = None
        if findmnt is not None and findmnt.stdout.strip():
            print(findmnt.stdout.strip())

    def doctor(self, *, prepare: bool, train: bool) -> None:
        if self.dry_run:
            print("[doctor] dry-run: path and GPU probes are deferred", flush=True)
            return
        self._print_host_probe()
        if prepare:
            self._require_file(self.paths["python"], "data Python", executable=True)
            self._require_file(self.paths["jax_python"], "JAX Python", executable=True)
            self._require_file(self.paths["encoder_dir"] / "checkpoint.json", "encoder checkpoint")
            for dataset in self.datasets:
                self._require_file(
                    self.paths["dataset_root"] / dataset / "meta" / "info.json",
                    f"{dataset} metadata",
                )
            self.run_command(
                "doctor_jax",
                [
                    self.paths["jax_python"],
                    "-c",
                    (
                        "import jax; devices=jax.devices('gpu'); "
                        f"assert len(devices)=={len(self.gpu_ids)}, devices; "
                        "print('jax_gpu_devices=', devices)"
                    ),
                ],
                env={
                    "CUDA_VISIBLE_DEVICES": ",".join(self.gpu_ids),
                    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                },
            )
            self.run_command(
                "doctor_data",
                [
                    self.paths["python"],
                    "-c",
                    (
                        "import numpy, pyarrow, sklearn, zarr; "
                        "print('data_environment=ok')"
                    ),
                ],
            )
        if train:
            self._require_file(self.paths["python"], "training Python", executable=True)
            self._require_file(self.paths["accelerate"], "Accelerate", executable=True)
            self.run_command(
                "doctor_training",
                [
                    self.paths["python"],
                    "-c",
                    (
                        "import accelerate, diffusers, hydra, wandb, torch; "
                        "import torch.nn.functional as F; "
                        "assert torch.cuda.is_available(); "
                        f"assert torch.cuda.device_count()=={len(self.gpu_ids)}, "
                        "torch.cuda.device_count(); "
                        "x=torch.randn(32,128,device='cuda'); "
                        "w=torch.randn(64,128,device='cuda'); "
                        "y=F.linear(x,w); torch.cuda.synchronize(); "
                        "assert torch.isfinite(y).all(); print('cuda_linear_smoke=ok')"
                    ),
                ],
                env={"CUDA_VISIBLE_DEVICES": ",".join(self.gpu_ids)},
            )
            config_dir = ROOT / "reactive_diffusion_policy" / "config"
            compose_source = (
                "from hydra import compose, initialize_config_dir; "
                "from omegaconf import OmegaConf; "
                "OmegaConf.register_new_resolver('eval', eval, replace=True); "
                f"ctx=initialize_config_dir(version_base=None, config_dir={str(config_dir)!r}); "
                "ctx.__enter__(); "
                "a=compose(config_name='train_pick_tube_at_workspace'); "
                "l=compose(config_name='train_pick_tube_ldp_workspace'); "
                "OmegaConf.resolve(a); OmegaConf.resolve(l); "
                "ctx.__exit__(None,None,None); print('hydra_resolve=ok')"
            )
            self.run_command(
                "doctor_hydra", [self.paths["python"], "-c", compose_source]
            )
        self._mark("doctor", "completed")

    def precompute(self) -> None:
        settings = self.recipe["precompute"]
        gpu_ids = list(self.gpu_ids)
        commands = []
        for index, dataset in enumerate(self.datasets):
            cache_dir = self.paths["tactile_cache_root"] / "KaiyueChen" / dataset
            if (
                cache_dir.joinpath("embeddings.npy").is_file()
                and cache_dir.joinpath("metadata.json").is_file()
                and not self.args.overwrite_precompute
            ):
                print(f"[precompute:{dataset}] verified cache exists, skipping")
                continue
            command = [
                self.paths["jax_python"],
                ROOT / "precompute_pick_tube_v21_tactile_embeddings.py",
                "--dataset-root",
                self.paths["dataset_root"],
                "--datasets",
                dataset,
                "--cache-root",
                self.paths["tactile_cache_root"],
                "--encoder-path",
                self.paths["encoder_dir"],
                "--batch-size",
                str(settings["batch_size"]),
                "--num-workers",
                str(settings["workers_per_gpu"]),
                "--prefetch-factor",
                str(settings["prefetch_factor"]),
                "--flush-every",
                str(settings["flush_every"]),
            ]
            if self.args.overwrite_precompute:
                command.append("--overwrite")
            commands.append((dataset, gpu_ids[index % len(gpu_ids)], [str(v) for v in command]))
        for dataset, gpu_id, command in commands:
            print(
                f"[precompute:{dataset}] "
                + display_command(
                    command,
                    {
                        "CUDA_VISIBLE_DEVICES": gpu_id,
                        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                    },
                ),
                flush=True,
            )
        if self.dry_run or not commands:
            if not self.dry_run:
                self._mark("precompute", "completed", reused=not commands)
            return
        self._mark("precompute", "running")
        pending = list(commands)
        active: dict[subprocess.Popen, tuple[str, str]] = {}
        available_gpus = list(gpu_ids)
        try:
            while pending or active:
                while pending and available_gpus:
                    dataset, _, command = pending.pop(0)
                    gpu_id = available_gpus.pop(0)
                    env = os.environ.copy()
                    env.update(
                        {
                            "CUDA_VISIBLE_DEVICES": gpu_id,
                            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                        }
                    )
                    process = subprocess.Popen(command, cwd=ROOT, env=env)
                    active[process] = (dataset, gpu_id)
                completed = None
                for process in active:
                    if process.poll() is not None:
                        completed = process
                        break
                if completed is None:
                    time.sleep(0.25)
                    continue
                dataset, gpu_id = active.pop(completed)
                available_gpus.append(gpu_id)
                if completed.returncode != 0:
                    raise subprocess.CalledProcessError(completed.returncode, completed.args)
                print(f"[precompute:{dataset}] completed on GPU {gpu_id}", flush=True)
        except BaseException as error:
            for process in active:
                process.terminate()
            for process in active:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            self._mark("precompute", "failed", error=repr(error))
            raise
        cache_metrics = {}
        for dataset in self.datasets:
            metadata_path = (
                self.paths["tactile_cache_root"]
                / "KaiyueChen"
                / dataset
                / "metadata.json"
            )
            if metadata_path.is_file():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                cache_metrics[dataset] = metadata.get("timings", {})
        self._mark("precompute", "completed", metrics=cache_metrics)

    def prepare(self) -> None:
        self.precompute()
        pca_settings = self.recipe["pca"]
        if self.paths["tactile_pca_path"].is_file():
            print(f"[pca] existing artifact, skipping: {self.paths['tactile_pca_path']}")
            self._mark("pca", "completed", reused=True)
        else:
            self.run_command(
                "pca",
                [
                    self.paths["python"],
                    ROOT / "fit_pick_tube_tactile_pca.py",
                    "--tactile-cache-root",
                    self.paths["tactile_cache_root"],
                    "--output",
                    self.paths["tactile_pca_path"],
                    "--components-per-arm",
                    str(self.recipe["data"]["components_per_arm"]),
                    "--batch-size",
                    str(pca_settings["batch_size"]),
                    "--datasets",
                    *self.datasets,
                ],
            )
        conversion = self.recipe["conversion"]

        def conversion_command(
            output_dir: Path, *, max_episodes: int | None = None, overwrite: bool = False
        ) -> list[str | Path]:
            command: list[str | Path] = [
                self.paths["python"],
                ROOT / "convert_pick_tube_lerobot_to_rdp_zarr.py",
                "--dataset-root",
                self.paths["dataset_root"],
                "--tactile-cache-root",
                self.paths["tactile_cache_root"],
                "--output-dir",
                output_dir,
                "--tactile-pca-path",
                self.paths["tactile_pca_path"],
                "--datasets",
                *self.datasets,
                "--dataset-repeats",
                *self.recipe["data"]["dataset_repeats"],
                "--num-workers",
                str(conversion["num_workers"]),
                "--rgb-chunk-frames",
                str(conversion["rgb_chunk_frames"]),
                "--compressor",
                str(conversion["compressor"]),
                "--compression-level",
                str(conversion["compression_level"]),
            ]
            if max_episodes is not None:
                command.extend(["--max-episodes-per-dataset", str(max_episodes)])
            if overwrite:
                command.append("--overwrite")
            return command

        zarr_path = self.paths["dataset_path"] / "replay_buffer.zarr"
        if zarr_path.is_dir() and not has_v2_manifest(zarr_path) and not self.args.overwrite_conversion:
            raise ValueError(
                f"incomplete Zarr found at {zarr_path}; rerun with --overwrite-conversion"
            )
        if has_v2_manifest(zarr_path) and not self.args.overwrite_conversion:
            validate_v2_zarr_scope(
                zarr_path,
                datasets=self.datasets,
                tactile_dim=int(self.recipe["data"]["tactile_dim"]),
            )
            print(f"[convert] existing Zarr, validating instead of replacing: {zarr_path}")
            self._mark("convert", "completed", reused=True)
        else:
            benchmark_dir = (
                self.paths["dataset_path"].parent
                / "_rdp_benchmarks"
                / self.run_id
            )
            benchmark_zarr = benchmark_dir / "replay_buffer.zarr"
            if has_v2_manifest(benchmark_zarr):
                validate_v2_zarr_scope(
                    benchmark_zarr,
                    datasets=self.datasets,
                    tactile_dim=int(self.recipe["data"]["tactile_dim"]),
                )
                print(f"[convert_smoke] existing benchmark, reusing: {benchmark_dir}")
                self._mark("convert_smoke", "completed", reused=True)
            else:
                self.run_command(
                    "convert_smoke",
                    conversion_command(
                        benchmark_dir,
                        max_episodes=1,
                        overwrite=benchmark_zarr.is_dir(),
                    ),
                )
                benchmark_metrics = benchmark_dir / "conversion_metrics.json"
                if not self.dry_run and benchmark_metrics.is_file():
                    self.state["stages"]["convert_smoke"]["metrics"] = json.loads(
                        benchmark_metrics.read_text(encoding="utf-8")
                    )
                    self._save_state()
            self.run_command(
                "convert",
                conversion_command(
                    self.paths["dataset_path"],
                    overwrite=self.args.overwrite_conversion,
                ),
            )
            metrics_path = self.paths["dataset_path"] / "conversion_metrics.json"
            if not self.dry_run and metrics_path.is_file():
                self.state["stages"]["convert"]["metrics"] = json.loads(
                    metrics_path.read_text(encoding="utf-8")
                )
                self._save_state()
        self.run_command(
            "validate_data",
            ["bash", ROOT / "scripts" / "setup_pick_tube_data.sh", "validate"],
            env={
                "PYTHON_BIN": str(self.paths["python"]),
                "DATASET_PATH": str(self.paths["dataset_path"]),
            },
        )

    def train(self) -> None:
        if not self.dry_run and not (self.paths["dataset_path"] / "replay_buffer.zarr").is_dir():
            raise FileNotFoundError(
                f"prepared dataset not found: {self.paths['dataset_path'] / 'replay_buffer.zarr'}"
            )
        if not self.dry_run:
            validate_v2_zarr_scope(
                self.paths["dataset_path"] / "replay_buffer.zarr",
                datasets=self.datasets,
                tactile_dim=int(self.recipe["data"]["tactile_dim"]),
            )
        training = self.recipe["training"]
        baseline = self.args.baseline_json
        env = {
            "PYTHON_BIN": str(self.paths["python"]),
            "ACCELERATE_BIN": str(self.paths["accelerate"]),
            "DATASET_PATH": str(self.paths["dataset_path"]),
            "OUTPUT_ROOT": str(self.paths["output_root"]),
            "RUN_ID": self.run_id,
            "GPU_ID": str(training["gpu_id"]),
            "LOGGING_MODE": str(training["logging_mode"]),
            "MIXED_PRECISION": str(training["mixed_precision"]),
            "TACTILE_DIM": str(self.recipe["data"]["tactile_dim"]),
            "AT_EPOCHS": str(training["at_epochs"]),
            "LDP_EPOCHS": str(training["ldp_epochs"]),
            "AT_BATCH": str(training["at_batch"]),
            "LDP_BATCH": str(training["ldp_batch"]),
            "NUM_WORKERS": str(training["num_workers"]),
            "AT_CHECKPOINT_EVERY": str(training["checkpoint_every"]),
            "LDP_CHECKPOINT_EVERY": str(training["checkpoint_every"]),
            "AT_PERIODIC_KEEP": str(training["periodic_keep"]),
            "LDP_PERIODIC_KEEP": str(training["periodic_keep"]),
            "AT_CHECKPOINT_KEEP": str(training["topk_keep"]),
            "LDP_CHECKPOINT_KEEP": str(training["topk_keep"]),
            "RESUME": str(bool(training["resume"])).lower(),
            "VALIDATE_DATASET": "0",
            "BASELINE_JSON": str(baseline.resolve()) if baseline else "",
            "DRY_RUN": "1" if self.dry_run else "0",
        }
        self.run_command(
            "train_at_ldp",
            ["bash", ROOT / "scripts" / "train_pick_tube_single_gpu.sh", "all"],
            env=env,
        )

    def status(self) -> None:
        print(json.dumps(self.state, indent=2, sort_keys=True))


def main() -> None:
    args = parse_args()
    pipeline = Pipeline(args)
    command = args.command
    if command == "status":
        pipeline.status()
        return
    if command == "doctor":
        pipeline.doctor(prepare=True, train=True)
        return
    if command == "prepare":
        pipeline.doctor(prepare=True, train=False)
        pipeline.prepare()
        return
    if command == "train":
        pipeline.doctor(prepare=False, train=True)
        pipeline.train()
        return
    if command in ("plan", "run", "resume"):
        pipeline.doctor(prepare=True, train=True)
        pipeline.prepare()
        pipeline.train()
        return
    raise AssertionError(f"unhandled command: {command}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        print(f"rdpctl: {error}", file=sys.stderr)
        raise SystemExit(2) from error
