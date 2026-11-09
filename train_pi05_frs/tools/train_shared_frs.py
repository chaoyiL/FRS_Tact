"""Sequential native Pi0.5 cache generation followed by one masked shared FRS run."""

from __future__ import annotations

import argparse
import copy
import math
import os
import subprocess
import sys
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path

import yaml

from train_pi05_frs.tools.train_frs import preflight_environment

ROOT = Path(__file__).resolve().parents[2]
TRAIN_ROOT = ROOT / "train_pi05_frs"
DEFAULT_CONFIG = TRAIN_ROOT / "configs/train_pi05_frs_shared_four_tasks.yaml"


def load_shared_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict) or not isinstance(config.get("sources"), list) or not config["sources"]:
        raise ValueError("Shared config must contain a non-empty sources list")
    for field in ("workspace", "output", "tactile_encoder"):
        value = config[field]
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise ValueError(f"Shared {field} must be an absolute path")
    names, repos = [], []
    for source in config["sources"]:
        name = source["name"]
        if not name or any(
            c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name
        ):
            raise ValueError("Task names must use letters, digits, underscores or hyphens")
        names.append(name)
        repos.append(source["repo_id"])
        if source["native_action_dim"] not in (10, 20, 32):
            raise ValueError(f"Unsupported native_action_dim for {name}")
        wrists = source["active_wrists"]
        if not wrists or len(set(wrists)) != len(wrists) or set(wrists) - {"left", "right"}:
            raise ValueError(f"Invalid active_wrists for {name}")
        if source["native_action_dim"] == 10 and wrists != ["right"]:
            raise ValueError("Native 10D models must use only the right wrist")
    if len(set(names)) != len(names) or len(set(repos)) != len(repos):
        raise ValueError("Shared task names and repo IDs must be unique")
    for field in ("num_train_episodes", "num_val_episodes", "action_horizon"):
        if type(config[field]) is not int or config[field] <= 0:
            raise ValueError(f"{field} must be a positive integer")
    if type(config["seed"]) is not int or config["seed"] < 0:
        raise ValueError("seed must be a non-negative integer")
    training = config["training"]
    if training["loss_mode"] not in ("gt", "bimanual_gated"):
        raise ValueError("Shared decoder uses flow matching, not a learned residual head")
    for field in (
        "batch_size",
        "epochs",
        "model_dim",
        "depth",
        "num_heads",
        "mlp_ratio",
        "tactile_window",
        "history_stride",
        "eval_every",
        "validation_steps",
        "aux_decode_steps",
    ):
        if type(training[field]) is not int or training[field] <= 0:
            raise ValueError(f"training.{field} must be a positive integer")
    if training["model_dim"] % training["num_heads"]:
        raise ValueError("model_dim must be divisible by num_heads")
    if type(training["balance_tasks"]) is not bool:
        raise ValueError("balance_tasks must be boolean")
    if training.get("state_conditioning", False):
        raise ValueError("Shared FRS does not mix native 7D/20D state vectors")
    for key in ("learning_rate", "grad_clip_norm", "gate_temperature"):
        if not math.isfinite(float(training[key])) or training[key] <= 0:
            raise ValueError(f"training.{key} must be finite and positive")
    for key in (
        "weight_decay",
        "aux_decode_weight",
        "low_gate_safety_weight",
        "low_gate_safety_margin",
        "rank_weight",
        "rank_margin",
        "repair_weight",
        "repair_margin",
    ):
        if not math.isfinite(float(training[key])) or training[key] < 0:
            raise ValueError(f"training.{key} must be finite and non-negative")
    if type(training["warmup_epochs"]) is not int or training["warmup_epochs"] < 0:
        raise ValueError("warmup_epochs must be a non-negative integer")
    if not 0 <= training["min_lr_ratio"] <= 1 or training["decode_solver"] not in ("euler", "fireflow"):
        raise ValueError("Invalid shared LR ratio / decode solver")
    if (
        training["gate_temperature"] <= 0
        or not 0 <= training["low_gate_threshold"] < training["high_gate_threshold"] <= 1
    ):
        raise ValueError("Invalid Gate thresholds/temperature")
    return config


def resolve_checkpoint(source: dict) -> Path:
    if source.get("checkpoint"):
        checkpoint = Path(source["checkpoint"]).expanduser()
    else:
        directory = ROOT / "checkpoints/model"
        candidates = []
        for candidate in directory.glob("*"):
            if source["name"] in candidate.name.lower() or (
                source["name"] == "bread" and "task2" in candidate.name.lower()
            ):
                candidates.extend(
                    p.parent
                    for p in candidate.rglob("params")
                    if p.is_dir()
                    and any(
                        (p / marker).exists()
                        for marker in ("_CHECKPOINT_METADATA", "_METADATA", "manifest.ocdbt")
                    )
                )
        candidates = sorted(set(candidates))
        if len(candidates) != 1:
            raise ValueError(
                f"Set sources[{source['name']}].checkpoint to your existing Pi0.5 path; discovery found {candidates}"
            )
        checkpoint = candidates[0]
    if not (checkpoint / "params").is_dir() and (checkpoint / "checkpoint/params").is_dir():
        checkpoint = checkpoint / "checkpoint"
    return checkpoint.resolve()


def native_config(config: dict, source: dict, *, check_paths: bool) -> tuple[dict, dict]:
    from train_pi05_frs.tools.train_frs import source_cache_dir, validate_config

    single = source["native_action_dim"] == 10
    template = (
        TRAIN_ROOT
        / "configs"
        / ("train_pi05_frs_right.yaml" if single else "train_pi05_frs_bimanual_gated.yaml")
    )
    native = copy.deepcopy(yaml.safe_load(template.read_text()))
    checkpoint = resolve_checkpoint(source)
    asset = source.get("norm_stats_asset_id")
    if not asset:
        files = list((checkpoint / "assets").glob("*/norm_stats.json"))
        if len(files) != 1:
            raise ValueError(
                f"Set norm_stats_asset_id for {source['name']}; assets contains {len(files)} candidates"
            )
        asset = files[0].parent.name
    work = Path(config["workspace"])
    action_root, tactile_root = work / "action_cache" / source["name"], work / "tactile_embeddings"
    native["checkpoint"] = str(checkpoint)
    native["allow_download"] = False
    native["datasets"] = [
        {"repo_id": source["repo_id"], "root": source["dataset_root"], "action_key": "action"}
    ]
    native["norm_stats"] = {"dir": str(checkpoint / "assets"), "asset_id": asset, "use_quantile_norm": True}
    native["model"].update(
        action_dim=source["native_action_dim"],
        action_horizon=config["action_horizon"],
        tactile_encoder_path=config["tactile_encoder"],
        state_conditioning=False,
    )
    native["model"]["camera_map"] = (
        {"right_wrist_0_rgb": "observation.images.camera1"}
        if single
        else {
            "left_wrist_0_rgb": "observation.images.camera0",
            "right_wrist_0_rgb": "observation.images.camera1",
        }
    )
    native["action_cache"].update(config["cache"])
    native["action_cache"].pop("tactile_batch_size", None)
    native["action_cache"].update(
        root=str(action_root),
        split_seed=config["seed"],
        num_train_episodes=config["num_train_episodes"],
        num_val_episodes=config["num_val_episodes"],
        max_episodes=None,
        max_samples=None,
    )
    native["tactile_embedding_cache"].update(
        root=str(tactile_root), precompute_batch_size=config["cache"]["tactile_batch_size"]
    )
    native["frs_training"]["output"] = str(work / "unused_native_runs" / source["name"])
    validate_config(native, check_paths=check_paths)
    record = {
        "name": source["name"],
        "repo_id": source["repo_id"],
        "dataset_root": source["dataset_root"],
        "active_wrists": source["active_wrists"],
        "action_cache": str(source_cache_dir(str(action_root), source["repo_id"])),
        "tactile_cache": str(tactile_root.joinpath(*source["repo_id"].split("/"))),
        "checkpoint": str(checkpoint),
        "norm_stats_asset_id": asset,
    }
    return native, record


class _Tee:
    def __init__(self, console, log):
        self.console, self.log = console, log

    def write(self, value):
        self.console.write(value)
        self.log.write(value)
        self.flush()
        return len(value)

    def flush(self):
        self.console.flush()
        self.log.flush()

    def isatty(self):
        return False


def _execute(config, prepared, args):
    records = []
    directory = Path(config["workspace"]) / "native_configs"
    for source, native, record in prepared:
        records.append(record)
        if not (args.train_only or args.evaluate_only):
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"{source['name']}.yaml"
            path.write_text(yaml.safe_dump(native, sort_keys=False))
            for module, extra in (
                ("prepare_frs_pi05_cache", ["--checkpoint-smoke"]),
                ("precompute_tactile_embeddings", []),
                ("prepare_frs_pi05_cache", []),
            ):
                print(f"task={source['name']} stage={module} {extra}", flush=True)
                command = [
                    sys.executable,
                    "-m",
                    f"train_pi05_frs.tools.{module}",
                    "--config",
                    str(path),
                    *extra,
                ]
                with subprocess.Popen(
                    command,
                    cwd=TRAIN_ROOT,
                    env=os.environ.copy(),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    errors="replace",
                ) as process:
                    for line in process.stdout:
                        sys.stdout.write(line)
                    status = process.wait()
                    if status:
                        raise subprocess.CalledProcessError(status, command)
    if args.prepare_task:
        print(
            f"Prepared frozen caches for {args.prepare_task}; shared training still uses ALL configured tasks."
        )
        return
    from train_pi05_frs.shared_training import train_shared

    train_shared(
        config, records, resume=args.resume, evaluate_only=args.evaluate_only, evaluate_task=args.task
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--prepare-task", help="Prepare only one task's frozen caches; does not train a task-specific FRS"
    )
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--task", help="Evaluation report task filter, e.g. insert")
    args = parser.parse_args(argv)
    if args.task and not args.evaluate_only:
        parser.error("--task filters only --evaluate-only; shared training always uses all sources")
    if args.prepare_task and (args.train_only or args.resume or args.evaluate_only):
        parser.error("--prepare-task cannot be combined with training/evaluation flags")
    if args.resume and args.evaluate_only:
        parser.error("--resume and --evaluate-only are mutually exclusive")
    config = load_shared_config(args.config.resolve())
    preflight_environment()
    if args.prepare_task and args.prepare_task not in [s["name"] for s in config["sources"]]:
        parser.error("Unknown --prepare-task")
    # Validate all requested inputs before starting a GPU/cache mutation.
    requested = [s for s in config["sources"] if args.prepare_task in (None, s["name"])]
    prepared = [(source, *native_config(config, source, check_paths=True)) for source in requested]
    if args.check:
        for source, _, record in prepared:
            print(
                f"task={source['name']} checkpoint={record['checkpoint']} native_action_dim={source['native_action_dim']} active={source['active_wrists']}"
            )
        print(
            f"Shared FRS: 20D / 4 tactile tokens / {config['num_train_episodes']} train + {config['num_val_episodes']} val episodes per task"
        )
        return
    directory = Path(config["workspace"]) / "logs"
    directory.mkdir(parents=True, exist_ok=True)
    log_path = directory / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.log"
    with (
        log_path.open("w") as stream,
        redirect_stdout(_Tee(sys.stdout, stream)),
        redirect_stderr(_Tee(sys.stderr, stream)),
    ):
        print(f"Shared FRS pipeline log: {log_path}", flush=True)
        try:
            _execute(config, prepared, args)
        except Exception:
            import traceback

            traceback.print_exc()
            raise


if __name__ == "__main__":
    main()
