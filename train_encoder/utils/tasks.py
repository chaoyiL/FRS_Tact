"""Task-level episode budgets and wrist selection for encoder pretraining."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def normalize_tasks(tasks: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(tasks, (list, tuple)) or not tasks:
        raise ValueError("tasks must be a non-empty list of task mappings.")
    result = []
    names: set[str] = set()
    repos: set[str] = set()
    for task in tasks:
        name = str(task["name"])
        repo_ids = task["dataset_repo_ids"]
        if isinstance(repo_ids, str):
            repo_ids = [repo_ids]
        repo_ids = [str(repo).strip() for repo in repo_ids]
        sides = list(task["sides"])
        train_count = task["num_train_episodes"]
        val_count = task["num_val_episodes"]
        if not name or name in names:
            raise ValueError(f"Task name must be non-empty and unique: {name!r}.")
        if (
            not repo_ids
            or any(not repo or repo in repos for repo in repo_ids)
            or len(set(repo_ids)) != len(repo_ids)
        ):
            raise ValueError(f"Task {name!r}: repo ids must be non-empty and unique across tasks.")
        if not sides or len(set(sides)) != len(sides) or set(sides) - {"left", "right"}:
            raise ValueError(f"Task {name!r}: sides must contain left and/or right without duplicates.")
        if any(type(count) is not int or count <= 0 for count in (train_count, val_count)):
            raise ValueError(f"Task {name!r}: episode budgets must be positive integers.")
        names.add(name)
        repos.update(repo_ids)
        result.append(
            {
                "name": name,
                "dataset_repo_ids": repo_ids,
                "sides": sides,
                "num_train_episodes": train_count,
                "num_val_episodes": val_count,
            }
        )
    return result


def task_repo_ids(tasks: Sequence[Mapping[str, Any]]) -> list[str]:
    return [repo for task in normalize_tasks(tasks) for repo in task["dataset_repo_ids"]]


def sample_task_episodes(candidates, *, train_count: int, val_count: int, seed: int, task_index: int):
    count = train_count + val_count
    if len(candidates) < count:
        raise ValueError(
            f"needs {count} episodes ({train_count} train + {val_count} val), "
            f"only {len(candidates)} available."
        )
    rng = np.random.default_rng(np.random.SeedSequence([seed, task_index]))
    selected = rng.permutation(candidates)[:count]
    return sorted(int(e) for e in selected[:train_count]), sorted(int(e) for e in selected[train_count:])


def select_task_episodes(dataset: Any, tasks, *, seed: int):
    """Select disjoint train/val episodes per task without replacement."""
    groups = dataset.episodes_by_repo()
    train, val, side_ids, selections = [], [], {}, []
    for task_i, task in enumerate(normalize_tasks(tasks)):
        missing = set(task["dataset_repo_ids"]) - groups.keys()
        if missing:
            raise ValueError(f"Task {task['name']!r}: datasets not loaded: {sorted(missing)}.")
        candidates = [episode for repo in task["dataset_repo_ids"] for episode in groups[repo]]
        train_count, val_count = task["num_train_episodes"], task["num_val_episodes"]
        prepared = getattr(dataset, "prepared_splits", {}).get(task["name"])
        if getattr(dataset, "prepared_splits", {}) and prepared is None:
            raise ValueError(f"Task {task['name']!r}: no matching task in the local manifest.")
        if prepared is not None:
            if (
                prepared["seed"] != seed
                or prepared["repo_ids"] != task["dataset_repo_ids"]
                or prepared["sides"] != task["sides"]
            ):
                raise ValueError(f"Task {task['name']!r}: local manifest and task configuration differ.")
            task_train, task_val = prepared["train"], prepared["val"]
            if len(task_train) != train_count or len(task_val) != val_count:
                raise ValueError(f"Task {task['name']!r}: local manifest episode budgets differ.")
        else:
            try:
                task_train, task_val = sample_task_episodes(
                    candidates, train_count=train_count, val_count=val_count, seed=seed, task_index=task_i
                )
            except ValueError as error:
                raise ValueError(f"Task {task['name']!r} {error}") from error
        train.extend(task_train)
        val.extend(task_val)
        for episode in (*task_train, *task_val):
            side_ids[episode] = tuple(i for i, side in enumerate(("left", "right")) if side in task["sides"])
        selections.append({**task, "train_episodes": task_train, "val_episodes": task_val})
    return tuple(sorted(train)), tuple(sorted(val)), side_ids, selections
