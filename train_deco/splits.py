import random


def split_episodes(episodes, val_fraction: float, seed: int):
    episodes = list(episodes)
    if len(episodes) < 2:
        raise ValueError(
            "DECO-C03: train/validation split requires at least 2 successful episodes. "
            f"found_episodes={len(episodes)}"
        )
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(
            "DECO-C03: VAL_FRACTION must be between 0 and 1. "
            f"val_fraction={val_fraction}"
        )
    random.Random(seed).shuffle(episodes)
    val_count = max(1, min(len(episodes) - 1, round(len(episodes) * val_fraction)))
    train_episodes = episodes[val_count:]
    val_episodes = episodes[:val_count]
    overlap = set(train_episodes).intersection(val_episodes)
    if overlap:
        raise ValueError(
            "DECO-C03: train and validation episode sets must be disjoint. "
            f"overlap_count={len(overlap)}, examples={sorted(overlap)[:3]}"
        )
    return train_episodes, val_episodes
