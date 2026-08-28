"""Quantify and visualize the Bread normal-to-deployment observation gap."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from scipy.spatial.transform import Rotation
from sklearn.decomposition import PCA

from deploy_deco.domain_gap import (
    basic_image_metrics,
    cosine_knn_distances,
    monotonic_state_match,
    relative_bimanual_state,
    right_world_z_delta,
    rotation_geodesic_degrees,
    standardized_state_summary,
)
from deploy_deco.config import load_config
from deploy_deco.policy import DECOPolicy
from train_deco.input_adapter import letterbox_and_normalize
from train_deco.model_factory import build_model


ROOT = Path("/home/typhon/FRS_Tact")
SERVER = Path("/home/typhon/vb3_robot_server")
PARQUETS = (
    ROOT / "data/episode_000034.parquet",
    ROOT / "data/episode_000035.parquet",
)
RUN_SPECS = (
    (
        "eval_obs_20260828_172311",
        SERVER / "action_debug_logs/20260828_172311_930128/chunk_trace.jsonl",
    ),
    (
        "eval_obs_20260828_172525",
        SERVER / "action_debug_logs/20260828_172525_481578/chunk_trace.jsonl",
    ),
)
CHECKPOINT = ROOT / "checkpoints/model/deco_0828/bread/deco_stage1_best.pt"
TORCHSCRIPT = ROOT / "checkpoints/model/deco_0828/bread/deco_stage1_best_gpu.ts"
DEFAULT_OUTPUT = ROOT / "deploy_deco/offline_results/bread_domain_gap"

STATE_NAMES = (
    "L rel x", "L rel y", "L rel z", "L rel rx", "L rel ry", "L rel rz", "L grip",
    "R rel x", "R rel y", "R rel z", "R rel rx", "R rel ry", "R rel rz", "R grip",
    "L/R x", "L/R y", "L/R z", "L/R rx", "L/R ry", "L/R rz",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_directory(path: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(str(child.relative_to(path)).encode())
        with child.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def decode_cell(cell) -> np.ndarray:
    payload = cell.as_py()
    with Image.open(io.BytesIO(payload["bytes"])) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def load_reference() -> list[dict]:
    episodes = []
    columns = [
        "observation.images.camera0",
        "observation.images.camera1",
        "observation.state",
        "actions",
        "timestamp",
    ]
    for path in PARQUETS:
        table = pq.read_table(path, columns=columns)
        states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)[:-1]
        actions = np.asarray(table["actions"].to_pylist(), dtype=np.float32)[:-1]
        episodes.append(
            {
                "episode": int(path.stem.split("_")[-1]),
                "path": path,
                "table": table.slice(0, len(states)),
                "states": states,
                "actions": actions,
                "timestamps": np.asarray(table["timestamp"].to_numpy(), dtype=float)[:-1],
            }
        )
    return episodes


def load_json_array(path: Path) -> np.ndarray:
    return np.asarray(json.loads(path.read_text()), dtype=np.float32)


def load_real_runs() -> list[dict]:
    runs = []
    for run_name, trace_path in RUN_SPECS:
        directory = SERVER / "eval_obs_data" / run_name
        step_dirs = sorted(directory.glob("step_*"))
        first = step_dirs[0]
        start_poses = [
            np.concatenate(
                [
                    load_json_array(first / f"robot{robot}_eef_pos.json")[-1],
                    load_json_array(first / f"robot{robot}_eef_rot_axis_angle.json")[-1],
                ]
            )
            for robot in range(2)
        ]
        rows = []
        states = []
        for step_dir in step_dirs:
            env_obs = {}
            for robot in range(2):
                for suffix in ("eef_pos", "eef_rot_axis_angle", "gripper_width"):
                    key = f"robot{robot}_{suffix}"
                    env_obs[key] = load_json_array(step_dir / f"{key}.json")
            current_poses = [
                np.concatenate(
                    [env_obs[f"robot{robot}_eef_pos"][-1], env_obs[f"robot{robot}_eef_rot_axis_angle"][-1]]
                )
                for robot in range(2)
            ]
            state = relative_bimanual_state(
                current_poses[0],
                current_poses[1],
                left_gripper=float(env_obs["robot0_gripper_width"][-1, 0]),
                right_gripper=float(env_obs["robot1_gripper_width"][-1, 0]),
                left_start=start_poses[0],
                right_start=start_poses[1],
            )
            images = [
                np.asarray(Image.open(step_dir / f"camera{camera}_rgb.jpg").convert("RGB"))
                for camera in range(2)
            ]
            rows.append(
                {
                    "step": int(step_dir.name.removeprefix("step_")),
                    "step_dir": step_dir,
                    "env_obs": env_obs,
                    "images": images,
                }
            )
            states.append(state)
        chunks = [json.loads(line) for line in trace_path.read_text().splitlines()]
        if len(chunks) != len(rows):
            raise ValueError(f"{run_name}: observation/chunk count mismatch")
        if [int(chunk["obs_seq"]) for chunk in chunks] != list(range(1, len(rows) + 1)):
            raise ValueError(f"{run_name}: chunk obs_seq is not contiguous from 1")
        if any(len(chunk["selected_raw_actions"]) != 24 for chunk in chunks):
            raise ValueError(f"{run_name}: expected exactly 24 selected actions per chunk")
        runs.append(
            {
                "name": run_name,
                "directory": directory,
                "trace_path": trace_path,
                "rows": rows,
                "states": np.asarray(states, dtype=np.float32),
                "start_poses": start_poses,
                "chunks": chunks,
            }
        )
    return runs


def load_visual_model(device: torch.device):
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = build_model(checkpoint["config"], load_backbone=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval().to(device)
    config = checkpoint["config"]
    stats = checkpoint["stats"]
    epoch = int(checkpoint["epoch"])
    del checkpoint
    gc.collect()
    return model, config, stats, epoch


def extract_embeddings(model, image_pairs, *, image_size: int, device, batch_size=48):
    embeddings = []
    with torch.inference_mode():
        for start in range(0, len(image_pairs), batch_size):
            batch = np.stack(image_pairs[start : start + batch_size], axis=0)
            tensor = torch.from_numpy(batch).permute(0, 1, 4, 2, 3).float().div_(255.0).to(device)
            normalized = letterbox_and_normalize(tensor, image_size)
            cameras = []
            for camera in range(2):
                feature = model.img_head(model.img_encoder(normalized[:, camera]))
                if tuple(feature.shape[1:]) != (512, 8, 8):
                    raise ValueError(f"unexpected visual feature shape {tuple(feature.shape)}")
                cameras.append(feature.mean(dim=(-2, -1)))
            embedding = torch.stack(cameras, dim=1)
            embedding = torch.nn.functional.normalize(embedding, dim=-1)
            embeddings.append(embedding.cpu().numpy())
    return np.concatenate(embeddings, axis=0)


def percentile(values, baseline) -> np.ndarray:
    baseline = np.asarray(baseline)
    return np.asarray([100.0 * np.mean(baseline <= value) for value in values])


def pose_matrix(pose6d) -> np.ndarray:
    pose6d = np.asarray(pose6d, dtype=float)
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_rotvec(pose6d[3:]).as_matrix()
    matrix[:3, 3] = pose6d[:3]
    return matrix


def world_right_z_delta(policy, images, state, seed, env_obs) -> tuple[float, np.ndarray]:
    observation = {
        "observation.images.camera0": images[0],
        "observation.images.camera1": images[1],
        "observation.state": np.asarray(state, dtype=np.float32),
    }
    prediction = policy.predict(observation, seed=seed)
    right_pose = np.concatenate(
        [
            np.asarray(env_obs["robot1_eef_pos"])[-1],
            np.asarray(env_obs["robot1_eef_rot_axis_angle"])[-1],
        ]
    )
    return right_world_z_delta(prediction[:24], right_pose=right_pose), prediction


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def main(output: Path, device_name: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    deploy_config_path = ROOT / "deploy_deco/configs/deploy_deco.yaml"
    deploy_config = load_config(deploy_config_path)
    configured_seed = int(deploy_config.get("seed", 0))
    configured_warmups = int(deploy_config["runtime"].get("warmup_runs", 1))
    if int(deploy_config["control"]["steps_per_inference"]) != 24:
        raise ValueError("recorded Bread runs require steps_per_inference=24")
    reference = load_reference()
    real_runs = load_real_runs()
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for model-aware Bread domain-gap analysis")
    model, config, stats, epoch = load_visual_model(device)
    mean = np.asarray(stats["observation_mean"], dtype=np.float64)
    std = np.asarray(stats["observation_std"], dtype=np.float64)

    reference_pairs = []
    reference_labels = []
    basic_rows = []
    for episode in reference:
        table = episode["table"]
        for frame in range(len(episode["states"])):
            pair = [decode_cell(table[f"observation.images.camera{camera}"][frame]) for camera in range(2)]
            reference_pairs.append(pair)
            reference_labels.append((episode["episode"], frame))
            for camera, image in enumerate(pair):
                basic_rows.append(
                    {
                        "source": "reference",
                        "sequence": f"episode_{episode['episode']:06d}",
                        "index": frame,
                        "camera": camera,
                        **basic_image_metrics(image),
                    }
                )
    reference_embeddings = extract_embeddings(
        model, reference_pairs, image_size=int(config["image_size"]), device=device
    )

    real_pairs = []
    real_labels = []
    for run in real_runs:
        for row in run["rows"]:
            real_pairs.append(row["images"])
            real_labels.append((run["name"], row["step"]))
            for camera, image in enumerate(row["images"]):
                basic_rows.append(
                    {
                        "source": "real",
                        "sequence": run["name"],
                        "index": row["step"],
                        "camera": camera,
                        **basic_image_metrics(image),
                    }
                )
    real_embeddings = extract_embeddings(
        model, real_pairs, image_size=int(config["image_size"]), device=device
    )
    del model
    torch.cuda.empty_cache() if device.type == "cuda" else None

    episode_slices = []
    offset = 0
    for episode in reference:
        episode_slices.append(slice(offset, offset + len(episode["states"])))
        offset += len(episode["states"])
    episode_sample_indices = [
        np.arange(slc.start, slc.stop, 24, dtype=int) for slc in episode_slices
    ]
    sampled_reference_indices = np.concatenate(episode_sample_indices)
    real_slices = []
    offset = 0
    for run in real_runs:
        real_slices.append(slice(offset, offset + len(run["states"])))
        offset += len(run["states"])

    visual = {"cameras": {}, "joint": {}}
    feature_sets = {}
    for camera in range(2):
        ref_feature = reference_embeddings[:, camera]
        real_feature = real_embeddings[:, camera]
        baseline = np.concatenate(
            [
                cosine_knn_distances(ref_feature[episode_sample_indices[0]], ref_feature[episode_sample_indices[1]], k=5),
                cosine_knn_distances(ref_feature[episode_sample_indices[1]], ref_feature[episode_sample_indices[0]], k=5),
            ]
        )
        real_knn = cosine_knn_distances(real_feature, ref_feature[sampled_reference_indices], k=5)
        feature_sets[f"camera{camera}"] = (ref_feature, real_feature, baseline, real_knn)
        visual["cameras"][str(camera)] = {
            "reference_cross_episode_knn": {
                "mean": float(baseline.mean()),
                "median": float(np.median(baseline)),
                "p95": float(np.quantile(baseline, 0.95)),
                "max": float(baseline.max()),
            },
            "real_to_reference_knn": {
                "mean": float(real_knn.mean()),
                "median": float(np.median(real_knn)),
                "min": float(real_knn.min()),
                "max": float(real_knn.max()),
                "mean_ratio_to_baseline": float(real_knn.mean() / baseline.mean()),
                "fraction_above_baseline_p95": float(np.mean(real_knn > np.quantile(baseline, 0.95))),
                "fraction_above_baseline_max": float(np.mean(real_knn > baseline.max())),
                "percentiles": percentile(real_knn, baseline),
            },
        }
    ref_joint = np.concatenate([reference_embeddings[:, 0], reference_embeddings[:, 1]], axis=1)
    real_joint = np.concatenate([real_embeddings[:, 0], real_embeddings[:, 1]], axis=1)
    ref_joint /= np.linalg.norm(ref_joint, axis=1, keepdims=True)
    real_joint /= np.linalg.norm(real_joint, axis=1, keepdims=True)
    joint_baseline = np.concatenate(
        [
            cosine_knn_distances(ref_joint[episode_sample_indices[0]], ref_joint[episode_sample_indices[1]], k=5),
            cosine_knn_distances(ref_joint[episode_sample_indices[1]], ref_joint[episode_sample_indices[0]], k=5),
        ]
    )
    joint_real_knn = cosine_knn_distances(real_joint, ref_joint[sampled_reference_indices], k=5)
    feature_sets["joint"] = (ref_joint, real_joint, joint_baseline, joint_real_knn)
    visual["joint"] = {
        "reference_cross_episode_knn": {
            "mean": float(joint_baseline.mean()),
            "median": float(np.median(joint_baseline)),
            "p95": float(np.quantile(joint_baseline, 0.95)),
            "max": float(joint_baseline.max()),
        },
        "real_to_reference_knn": {
            "mean": float(joint_real_knn.mean()),
            "median": float(np.median(joint_real_knn)),
            "min": float(joint_real_knn.min()),
            "max": float(joint_real_knn.max()),
            "mean_ratio_to_baseline": float(joint_real_knn.mean() / joint_baseline.mean()),
            "fraction_above_baseline_p95": float(np.mean(joint_real_knn > np.quantile(joint_baseline, 0.95))),
            "fraction_above_baseline_max": float(np.mean(joint_real_knn > joint_baseline.max())),
            "percentiles": percentile(joint_real_knn, joint_baseline),
        },
    }

    ref_states = np.concatenate([episode["states"] for episode in reference])
    ref_state_z = (ref_states - mean) / np.maximum(std, 1e-4)
    state_cross = np.concatenate(
        [
            np.min(
                np.linalg.norm(
                    ref_state_z[episode_sample_indices[0], None]
                    - ref_state_z[episode_sample_indices[1]][None], axis=2
                ), axis=1
            ) / np.sqrt(20),
            np.min(
                np.linalg.norm(
                    ref_state_z[episode_sample_indices[1], None]
                    - ref_state_z[episode_sample_indices[0]][None], axis=2
                ), axis=1
            ) / np.sqrt(20),
        ]
    )
    state_report = {"runs": {}, "cross_episode_knn_rms": {
        "mean": float(state_cross.mean()), "median": float(np.median(state_cross)),
        "p95": float(np.quantile(state_cross, 0.95)), "max": float(state_cross.max())}}
    match_weights = np.array([1, 1, 1, .25, .25, .25, 4, 1, 1, 1, .25, .25, .25, 4, 1, 1, 1, .25, .25, .25])
    for run in real_runs:
        normalized = (run["states"] - mean) / np.maximum(std, 1e-4)
        candidates = []
        for episode in reference:
            ref_normalized = (episode["states"] - mean) / np.maximum(std, 1e-4)
            indices, costs = monotonic_state_match(
                normalized,
                ref_normalized,
                weights=match_weights,
                max_reference_step=48,
            )
            candidates.append((float(costs.mean()), episode, indices, costs))
        _, matched_episode, matched_indices, matched_costs = min(candidates, key=lambda item: item[0])
        run["matched_episode"] = matched_episode
        run["matched_indices"] = matched_indices
        run["matched_costs"] = matched_costs
        summary = standardized_state_summary(ref_states, run["states"], mean=mean, std=std)
        state_knn = np.min(
            np.linalg.norm(
                normalized[:, None] - ref_state_z[sampled_reference_indices][None], axis=2
            ), axis=1
        ) / np.sqrt(20)
        run["state_knn"] = state_knn
        state_report["runs"][run["name"]] = {
            "max_abs_zscore": summary["max_abs_zscore"],
            "values_over_3sigma": summary["values_over_threshold"],
            "rms_zscore": summary["rms_zscore"],
            "wasserstein_by_dimension": summary["wasserstein_by_dimension"],
            "state_knn_rms": state_knn,
            "state_knn_mean": float(state_knn.mean()),
            "state_knn_mean_ratio_to_baseline": float(state_knn.mean() / state_cross.mean()),
            "state_knn_fraction_above_baseline_p95": float(np.mean(state_knn > np.quantile(state_cross, 0.95))),
            "state_knn_fraction_above_baseline_max": float(np.mean(state_knn > state_cross.max())),
            "state_knn_percentile": percentile(state_knn, state_cross),
            "matched_episode": matched_episode["episode"],
            "candidate_episode_matches": {
                str(candidate[1]["episode"]): {
                    "mean_cost": candidate[0],
                    "frames": candidate[2],
                    "costs": candidate[3],
                }
                for candidate in candidates
            },
            "matched_frames": matched_indices,
            "matched_costs": matched_costs,
            "query_zscore": summary["query_zscore"],
        }

    # State-matched visual distances separate task-phase mismatch from appearance shift.
    normalized_episodes = [
        (episode["states"] - mean) / np.maximum(std, 1e-4) for episode in reference
    ]
    ep0_to_ep1, _ = monotonic_state_match(
        normalized_episodes[0], normalized_episodes[1], weights=match_weights,
        max_reference_step=48,
    )
    ep1_to_ep0, _ = monotonic_state_match(
        normalized_episodes[1], normalized_episodes[0], weights=match_weights,
        max_reference_step=48,
    )
    for name, (ref_feature, real_feature, _, _) in feature_sets.items():
        stage_baseline = np.concatenate(
            [
                1.0 - np.sum(
                    ref_feature[episode_slices[0]]
                    * ref_feature[episode_slices[1]][ep0_to_ep1],
                    axis=1,
                ),
                1.0 - np.sum(
                    ref_feature[episode_slices[1]]
                    * ref_feature[episode_slices[0]][ep1_to_ep0],
                    axis=1,
                ),
            ]
        )
        real_stage = []
        for run, real_slice in zip(real_runs, real_slices):
            episode_number = run["matched_episode"]["episode"]
            episode_index = 0 if episode_number == reference[0]["episode"] else 1
            matched_reference = ref_feature[episode_slices[episode_index]][run["matched_indices"]]
            real_stage.extend(
                1.0
                - np.sum(real_feature[real_slice] * matched_reference, axis=1)
            )
        real_stage = np.asarray(real_stage)
        target = visual["joint"] if name == "joint" else visual["cameras"][name.removeprefix("camera")]
        target["state_matched_cosine"] = {
            "reference_cross_episode_mean": float(stage_baseline.mean()),
            "reference_cross_episode_p95": float(np.quantile(stage_baseline, 0.95)),
            "real_mean": float(real_stage.mean()),
            "real_median": float(np.median(real_stage)),
            "real_mean_ratio_to_baseline": float(real_stage.mean() / stage_baseline.mean()),
            "real_fraction_above_baseline_p95": float(
                np.mean(real_stage > np.quantile(stage_baseline, 0.95))
            ),
            "real_values": real_stage,
        }

    start_gap = {"arms": {}}
    for robot, label in ((0, "left"), (1, "right")):
        first, second = (run["start_poses"][robot] for run in real_runs)
        start_gap["arms"][label] = {
            "translation_mm": float(np.linalg.norm(second[:3] - first[:3]) * 1000.0),
            "rotation_degrees": rotation_geodesic_degrees(first[3:], second[3:]),
            "first_pose": first,
            "second_pose": second,
        }
    bilateral = []
    for run in real_runs:
        bilateral.append(np.linalg.inv(pose_matrix(run["start_poses"][1])) @ pose_matrix(run["start_poses"][0]))
    start_gap["bilateral_relative"] = {
        "translation_mm": float(np.linalg.norm(bilateral[1][:3, 3] - bilateral[0][:3, 3]) * 1000.0),
        "rotation_degrees": rotation_geodesic_degrees(
            Rotation.from_matrix(bilateral[0][:3, :3]).as_rotvec(),
            Rotation.from_matrix(bilateral[1][:3, :3]).as_rotvec(),
        ),
    }

    basic_summary = {}
    basic_lookup = {
        (row["sequence"], row["index"], row["camera"]): row for row in basic_rows
    }
    for camera in range(2):
        basic_summary[str(camera)] = {}
        for metric in ("luma_mean", "luma_std", "saturation_mean", "laplacian_variance", "edge_density"):
            ref_values = np.array([row[metric] for row in basic_rows if row["source"] == "reference" and row["camera"] == camera])
            real_values = np.array([row[metric] for row in basic_rows if row["source"] == "real" and row["camera"] == camera])
            basic_summary[str(camera)][metric] = {
                "reference_mean": float(ref_values.mean()),
                "real_mean": float(real_values.mean()),
                "relative_change_percent": float((real_values.mean() / ref_values.mean() - 1.0) * 100.0),
            }
        basic_summary[str(camera)]["state_matched"] = {}
        for metric in ("luma_mean", "luma_std", "saturation_mean", "laplacian_variance", "edge_density"):
            matched_reference_values = []
            matched_real_values = []
            for run in real_runs:
                reference_sequence = f"episode_{run['matched_episode']['episode']:06d}"
                for row, matched_index in zip(run["rows"], run["matched_indices"]):
                    matched_reference_values.append(
                        basic_lookup[(reference_sequence, int(matched_index), camera)][metric]
                    )
                    matched_real_values.append(
                        basic_lookup[(run["name"], row["step"], camera)][metric]
                    )
            ref_mean = float(np.mean(matched_reference_values))
            real_mean = float(np.mean(matched_real_values))
            basic_summary[str(camera)]["state_matched"][metric] = {
                "reference_mean": ref_mean,
                "real_mean": real_mean,
                "relative_change_percent": float((real_mean / ref_mean - 1.0) * 100.0),
            }

    policy = DECOPolicy(TORCHSCRIPT, device=device_name, verify_hash=True)
    counterfactual = {"runs": {}}
    for run in real_runs:
        rows = []
        episode = run["matched_episode"]
        table = episode["table"]
        for order, (real_row, ref_index) in enumerate(zip(run["rows"], run["matched_indices"])):
            normal_images = [decode_cell(table[f"observation.images.camera{camera}"][int(ref_index)]) for camera in range(2)]
            normal_state = episode["states"][int(ref_index)]
            real_images = real_row["images"]
            real_state = run["states"][order]
            seed = configured_seed + configured_warmups + order
            combinations = {}
            predictions = {}
            for name, images, state in (
                ("real_image_step0_proxy_state", real_images, real_state),
                ("normal_image_normal_state", normal_images, normal_state),
                ("real_image_normal_state", real_images, normal_state),
                ("normal_image_step0_proxy_state", normal_images, real_state),
            ):
                delta_z, prediction = world_right_z_delta(
                    policy, images, state, seed, real_row["env_obs"]
                )
                combinations[name] = delta_z
                predictions[name] = prediction
            logged = np.asarray(run["chunks"][order]["selected_raw_actions"], dtype=np.float32)
            replay = predictions["real_image_step0_proxy_state"][:24]
            right_pose = np.concatenate(
                [
                    real_row["env_obs"]["robot1_eef_pos"][-1],
                    real_row["env_obs"]["robot1_eef_rot_axis_angle"][-1],
                ]
            )
            replay_errors = {
                "left_translation_mae_mm": float(np.abs(replay[:, :3] - logged[:, :3]).mean() * 1000.0),
                "right_translation_mae_mm": float(np.abs(replay[:, 10:13] - logged[:, 10:13]).mean() * 1000.0),
                "left_rotation6d_mae": float(np.abs(replay[:, 3:9] - logged[:, 3:9]).mean()),
                "right_rotation6d_mae": float(np.abs(replay[:, 13:19] - logged[:, 13:19]).mean()),
                "left_gripper_mae_mm": float(np.abs(replay[:, 9] - logged[:, 9]).mean() * 1000.0),
                "right_gripper_mae_mm": float(np.abs(replay[:, 19] - logged[:, 19]).mean() * 1000.0),
                "composed_right_world_z_error_mm": float(
                    combinations["real_image_step0_proxy_state"]
                    - right_world_z_delta(logged, right_pose=right_pose)
                ),
            }
            rows.append(
                {
                    "step": real_row["step"],
                    "seed": seed,
                    "matched_episode": episode["episode"],
                    "matched_frame": int(ref_index),
                    "world_right_z24_mm": combinations,
                    "approximate_replay_errors": replay_errors,
                    "visual_replacement_effect_mm": combinations["normal_image_step0_proxy_state"] - combinations["real_image_step0_proxy_state"],
                    "state_replacement_effect_mm": combinations["real_image_normal_state"] - combinations["real_image_step0_proxy_state"],
                }
            )
        counterfactual["runs"][run["name"]] = rows

    grasp_transition_rows = []
    for run in real_runs:
        widths = np.asarray(
            [float(row["env_obs"]["robot1_gripper_width"][-1, 0]) for row in run["rows"]]
        )
        close_index = int(np.flatnonzero(widths <= 0.075)[0])
        selected = sorted({close_index, min(len(widths) - 1, close_index + 1)})
        for index in selected:
            row = counterfactual["runs"][run["name"]][index]
            grasp_transition_rows.append({"run": run["name"], **row})
    counterfactual["grasp_transition_summary"] = {
        "selection": "first observation at right gripper <= 0.075 m and its successor",
        "rows": grasp_transition_rows,
        "visual_replacement_effect_mm": {
            "mean": float(np.mean([row["visual_replacement_effect_mm"] for row in grasp_transition_rows])),
            "median": float(np.median([row["visual_replacement_effect_mm"] for row in grasp_transition_rows])),
            "positive_fraction": float(np.mean([row["visual_replacement_effect_mm"] > 0 for row in grasp_transition_rows])),
        },
        "state_replacement_effect_mm": {
            "mean": float(np.mean([row["state_replacement_effect_mm"] for row in grasp_transition_rows])),
            "median": float(np.median([row["state_replacement_effect_mm"] for row in grasp_transition_rows])),
            "positive_fraction": float(np.mean([row["state_replacement_effect_mm"] > 0 for row in grasp_transition_rows])),
        },
    }

    # Per-frame CSV: real and reference basic metrics plus real OOD columns.
    visual_offsets = {label: index for index, label in enumerate(real_labels)}
    metrics_path = output / "metrics.csv"
    extra_fields = ["feature_knn_cosine", "feature_percentile", "state_knn_rms", "state_percentile", "matched_episode", "matched_frame"]
    with metrics_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(basic_rows[0]) + extra_fields)
        writer.writeheader()
        run_lookup = {run["name"]: run for run in real_runs}
        for row in basic_rows:
            enriched = dict(row)
            enriched.update({field: "" for field in extra_fields})
            if row["source"] == "real":
                flat_index = visual_offsets[(row["sequence"], row["index"])]
                run = run_lookup[row["sequence"]]
                local_index = next(i for i, item in enumerate(run["rows"]) if item["step"] == row["index"])
                camera_key = f"camera{row['camera']}"
                baseline = feature_sets[camera_key][2]
                distance = feature_sets[camera_key][3][flat_index]
                enriched.update(
                    {
                        "feature_knn_cosine": float(distance),
                        "feature_percentile": float(percentile([distance], baseline)[0]),
                        "state_knn_rms": float(run["state_knn"][local_index]),
                        "state_percentile": float(percentile([run["state_knn"][local_index]], state_cross)[0]),
                        "matched_episode": run["matched_episode"]["episode"],
                        "matched_frame": int(run["matched_indices"][local_index]),
                    }
                )
            writer.writerow(enriched)

    # Raw visual metric distributions, separated by camera and sequence.
    sequences = [
        f"episode_{episode['episode']:06d}" for episode in reference
    ] + [run["name"] for run in real_runs]
    sequence_labels = ["ep34", "ep35", "172311", "172525"]
    sequence_colors = ["#4c78a8", "#72b7b2", "#e45756", "#f2cf5b"]
    distribution_metrics = (
        ("luma_mean", "mean luma"),
        ("luma_std", "luma contrast"),
        ("saturation_mean", "mean saturation"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for camera in range(2):
        for column, (metric, title) in enumerate(distribution_metrics):
            axis = axes[camera, column]
            values = [
                [
                    row[metric]
                    for row in basic_rows
                    if row["camera"] == camera and row["sequence"] == sequence
                ]
                for sequence in sequences
            ]
            boxes = axis.boxplot(values, tick_labels=sequence_labels, patch_artist=True, showfliers=False)
            for box, color in zip(boxes["boxes"], sequence_colors):
                box.set_facecolor(color)
                box.set_alpha(.65)
            axis.set_title(f"camera{camera}: {title}")
            axis.grid(axis="y", alpha=.2)
    fig.savefig(output / "basic_visual_distribution.png", dpi=180)
    plt.close(fig)

    # Reference 20D distributions with real observations overlaid in model z-space.
    fig, axis = plt.subplots(figsize=(16, 7))
    reference_by_dimension = [ref_state_z[:, index] for index in range(20)]
    boxes = axis.boxplot(
        reference_by_dimension,
        positions=np.arange(20),
        widths=.55,
        patch_artist=True,
        showfliers=False,
    )
    for box in boxes["boxes"]:
        box.set_facecolor("#4c78a8")
        box.set_alpha(.35)
    for offset_value, run, color in zip((-.13, .13), real_runs, ("#e45756", "#f2cf5b")):
        zscore = np.asarray(state_report["runs"][run["name"]]["query_zscore"])
        for dimension in range(20):
            axis.scatter(
                np.full(len(zscore), dimension + offset_value),
                zscore[:, dimension],
                s=18,
                alpha=.8,
                color=color,
                label=run["name"][-6:] if dimension == 0 else None,
            )
    axis.axhline(3, color="black", ls=":", lw=1)
    axis.axhline(-3, color="black", ls=":", lw=1)
    axis.set_xticks(range(20), STATE_NAMES, rotation=55, ha="right")
    axis.set_ylabel("model-space z-score")
    axis.set_title("Reference distributions and step0-proxy real states")
    axis.grid(axis="y", alpha=.2)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / "state_distribution.png", dpi=180)
    plt.close(fig)

    # PCA visualization.
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for axis, (name, (ref_feature, real_feature, _, _)) in zip(axes, feature_sets.items()):
        pca = PCA(n_components=min(10, ref_feature.shape[1])).fit(ref_feature)
        ref_2d = pca.transform(ref_feature)[:, :2]
        real_2d = pca.transform(real_feature)[:, :2]
        for episode, slc, color in zip(reference, episode_slices, ("#4c78a8", "#72b7b2")):
            axis.scatter(ref_2d[slc, 0], ref_2d[slc, 1], s=7, alpha=.35, color=color, label=f"episode {episode['episode']}")
        for run, slc, color in zip(real_runs, real_slices, ("#e45756", "#f2cf5b")):
            axis.plot(real_2d[slc, 0], real_2d[slc, 1], "o-", ms=5, lw=1.5, color=color, label=run["name"][-6:])
        explained = 100.0 * pca.explained_variance_ratio_[:2].sum()
        axis.set_title(f"{name}: reference PCA (PC1+PC2={explained:.1f}%)")
        axis.set_xlabel("PC1")
        axis.set_ylabel("PC2")
        axis.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "visual_embedding_pca.png", dpi=180)
    plt.close(fig)

    # OOD distance over real steps.
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False)
    for axis, (name, (_, _, baseline, real_knn)) in zip(axes, feature_sets.items()):
        for run, slc, color in zip(real_runs, real_slices, ("#e45756", "#f2cf5b")):
            axis.plot([row["step"] for row in run["rows"]], real_knn[slc], "o-", color=color, label=run["name"][-6:])
        axis.axhline(np.median(baseline), color="black", ls="--", lw=1, label="reference median")
        axis.axhline(np.quantile(baseline, .95), color="black", ls=":", lw=1.5, label="reference p95")
        axis.fill_between(axis.get_xlim(), 0, np.quantile(baseline, .95), color="#4c78a8", alpha=.08)
        axis.set_ylabel("5-NN cosine distance")
        axis.set_title(name)
        axis.grid(alpha=.2)
    axes[-1].set_xlabel("deployment control step")
    axes[0].legend(ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "visual_distance_timeline.png", dpi=180)
    plt.close(fig)

    # State z-score heatmaps.
    fig, axes = plt.subplots(2, 1, figsize=(15, 6), constrained_layout=True)
    for axis, run in zip(axes, real_runs):
        zscore = np.asarray(state_report["runs"][run["name"]]["query_zscore"])
        image = axis.imshow(zscore.T, aspect="auto", cmap="coolwarm", vmin=-3, vmax=3)
        axis.set_xticks(range(len(run["rows"])), [row["step"] for row in run["rows"]])
        axis.set_yticks(range(20), STATE_NAMES, fontsize=7)
        axis.set_title(run["name"])
        axis.set_xlabel("deployment control step")
    fig.colorbar(image, ax=axes, label="model-space z-score", shrink=.8)
    fig.savefig(output / "state_zscore_heatmap.png", dpi=180)
    plt.close(fig)

    # Start-pose 3D axes and gap bars.
    fig = plt.figure(figsize=(18, 6))
    axis3d = fig.add_subplot(131, projection="3d")
    colors = (("#4c78a8", "#72b7b2"), ("#e45756", "#f2cf5b"))
    for run_index, run in enumerate(real_runs):
        for robot, label in ((0, "L"), (1, "R")):
            pose = run["start_poses"][robot]
            rotation = Rotation.from_rotvec(pose[3:]).as_matrix()
            point = pose[:3]
            color = colors[run_index][robot]
            axis3d.scatter(*point, color=color, s=45, label=f"{run['name'][-6:]} {label}")
            for local_axis, axis_color in enumerate(("r", "g", "b")):
                vector = rotation[:, local_axis] * .035
                axis3d.quiver(*point, *vector, color=axis_color, alpha=.65)
    axis3d.set_xlabel("x [m]")
    axis3d.set_ylabel("y [m]")
    axis3d.set_zlabel("z [m]")
    axis3d.set_title("First persisted absolute poses (not pre-loop starts)")
    axis3d.legend(fontsize=8)
    labels = ["left", "right", "bilateral"]
    translation_values = [start_gap["arms"]["left"]["translation_mm"], start_gap["arms"]["right"]["translation_mm"], start_gap["bilateral_relative"]["translation_mm"]]
    rotation_values = [start_gap["arms"]["left"]["rotation_degrees"], start_gap["arms"]["right"]["rotation_degrees"], start_gap["bilateral_relative"]["rotation_degrees"]]
    for subplot, values, title, xlabel, color in (
        (132, translation_values, "Translation repeatability gap", "distance [mm]", "#4c78a8"),
        (133, rotation_values, "Rotation repeatability gap", "SO(3) geodesic [degree]", "#e45756"),
    ):
        axis_bar = fig.add_subplot(subplot)
        axis_bar.barh(range(3), values, color=color)
        axis_bar.set_yticks(range(3), labels)
        axis_bar.invert_yaxis()
        axis_bar.set_xlabel(xlabel)
        for index, value in enumerate(values):
            axis_bar.text(value, index, f" {value:.2f}", va="center")
        axis_bar.set_title(title)
        axis_bar.grid(axis="x", alpha=.2)
    fig.tight_layout()
    fig.savefig(output / "start_pose_comparison.png", dpi=180)
    plt.close(fig)

    # Phase distance heatmap for each run and camera against the selected episode.
    fig, axes = plt.subplots(2, 2, figsize=(15, 8), constrained_layout=True)
    for row_index, (run, real_slice) in enumerate(zip(real_runs, real_slices)):
        ref_slice = episode_slices[0] if run["matched_episode"]["episode"] == reference[0]["episode"] else episode_slices[1]
        for camera in range(2):
            ref_feature = reference_embeddings[ref_slice, camera]
            query_feature = real_embeddings[real_slice, camera]
            matrix = 1.0 - query_feature @ ref_feature.T
            axis = axes[row_index, camera]
            shown = axis.imshow(
                matrix, aspect="auto", cmap="magma", origin="lower", vmin=.10, vmax=.55
            )
            axis.plot(run["matched_indices"], range(len(run["rows"])), color="cyan", marker="o", ms=3, lw=1, label="state match")
            axis.set_title(f"{run['name'][-6:]} cam{camera} vs ep{run['matched_episode']['episode']}")
            axis.set_xlabel("reference frame")
            axis.set_ylabel("real observation index")
            fig.colorbar(shown, ax=axis, label="cosine distance")
    fig.savefig(output / "phase_distance_heatmap.png", dpi=180)
    plt.close(fig)

    # Counterfactual action sensitivity.
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), constrained_layout=True)
    keys = (
        ("real_image_step0_proxy_state", "real image + step0-proxy state", "#e45756"),
        ("normal_image_step0_proxy_state", "normal image + step0-proxy state", "#4c78a8"),
        ("real_image_normal_state", "real image + normal state", "#f2cf5b"),
        ("normal_image_normal_state", "normal image + normal state", "#72b7b2"),
    )
    for axis, run in zip(axes, real_runs):
        rows = counterfactual["runs"][run["name"]]
        steps = [row["step"] for row in rows]
        for key, label, color in keys:
            axis.plot(steps, [row["world_right_z24_mm"][key] for row in rows], "o-", label=label, color=color)
        axis.axhline(0.0, color="black", lw=1)
        axis.set_title(run["name"])
        axis.set_ylabel("predicted open-loop right world-z / 24 actions [mm]")
        axis.grid(alpha=.2)
        axis.text(
            .01, .03,
            "Sensitivity probe only: hybrid image/state pairs are non-physical; not executed motion.",
            transform=axis.transAxes, fontsize=8,
            bbox={"facecolor": "white", "alpha": .75, "edgecolor": "none"},
        )
    axes[-1].set_xlabel("deployment control step")
    axes[0].legend(ncol=2, fontsize=8)
    fig.savefig(output / "counterfactual_action.png", dpi=180)
    plt.close(fig)

    # Matched frame contact sheet: up to four evenly spaced observations per run.
    fig, axes = plt.subplots(8, 4, figsize=(12, 21))
    for block, run in enumerate(real_runs):
        picks = np.unique(np.linspace(0, len(run["rows"]) - 1, 4, dtype=int))
        while len(picks) < 4:
            picks = np.append(picks, picks[-1])
        episode = run["matched_episode"]
        for column, local_index in enumerate(picks[:4]):
            ref_index = int(run["matched_indices"][local_index])
            for camera in range(2):
                ref_image = decode_cell(episode["table"][f"observation.images.camera{camera}"][ref_index])
                real_image = run["rows"][local_index]["images"][camera]
                top = block * 4 + camera * 2
                axes[top, column].imshow(ref_image)
                axes[top, column].set_title(f"ep{episode['episode']} f{ref_index} cam{camera}", fontsize=8)
                axes[top + 1, column].imshow(real_image)
                axes[top + 1, column].set_title(f"{run['name'][-6:]} s{run['rows'][local_index]['step']} cam{camera}", fontsize=8)
                axes[top, column].axis("off")
                axes[top + 1, column].axis("off")
    fig.tight_layout()
    fig.savefig(output / "matched_frames.png", dpi=160)
    plt.close(fig)

    report = {
        "method": {
            "reference_terminal_row_excluded": True,
            "step0_proxy_state_anchor": "step_000000 proxy; true pre-loop episode_start_pose was not persisted",
            "visual_encoder": "Bread fine-tuned ResNet34 img_head GAP, per-camera L2 normalized",
            "visual_knn_k": 5,
            "state_std_floor": 1e-4,
            "counterfactual_prefix_steps": 24,
            "reference_baseline_stride": 24,
            "phase_match_max_reference_step": 48,
            "seed_assumption": "current deploy YAML; run-specific config was not persisted",
            "statistics_warning": "descriptive effects from 2 episodes and 2 runs; frame percentiles are not independent-sample significance",
            "counterfactual_warning": "hybrid image/state pairs are sensitivity probes, not physically valid closed-loop trajectories",
        },
        "inputs": {
            "checkpoint": str(CHECKPOINT),
            "checkpoint_sha256": sha256_file(CHECKPOINT),
            "torchscript": str(TORCHSCRIPT),
            "torchscript_sha256": sha256_file(TORCHSCRIPT),
            "torchscript_sidecar_sha256": sha256_file(Path(f"{TORCHSCRIPT}.json")),
            "deploy_config_sha256": sha256_file(deploy_config_path),
            "analysis_code_sha256": sha256_file(Path(__file__)),
            "metric_code_sha256": sha256_file(ROOT / "deploy_deco/domain_gap.py"),
            "checkpoint_epoch": epoch,
            "parquets": {str(path): sha256_file(path) for path in PARQUETS},
            "real_runs": {str(run["directory"]): sha256_directory(run["directory"]) for run in real_runs},
            "chunk_traces": {str(run["trace_path"]): sha256_file(run["trace_path"]) for run in real_runs},
        },
        "sample_counts": {
            "reference_frames": len(reference_pairs),
            "reference_baseline_stride": 24,
            "reference_baseline_frames": int(len(sampled_reference_indices)),
            "real_frames": len(real_pairs),
            "reference_by_episode": {str(ep["episode"]): len(ep["states"]) for ep in reference},
            "real_by_run": {run["name"]: len(run["states"]) for run in real_runs},
        },
        "first_persisted_pose_real_run_gap": start_gap,
        "state": state_report,
        "basic_visual": basic_summary,
        "model_visual": visual,
        "counterfactual_action": counterfactual,
    }
    (output / "summary.json").write_text(json.dumps(json_ready(report), indent=2) + "\n")
    print(json.dumps(json_ready({
        "sample_counts": report["sample_counts"],
        "first_persisted_pose_real_run_gap": start_gap,
        "basic_visual": basic_summary,
        "model_visual": visual,
        "approximate_replay_max_right_translation_mae_mm": max(
            row["approximate_replay_errors"]["right_translation_mae_mm"]
            for rows in counterfactual["runs"].values() for row in rows
        ),
        "output": str(output),
    }), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    arguments = parser.parse_args()
    main(arguments.output, arguments.device)
