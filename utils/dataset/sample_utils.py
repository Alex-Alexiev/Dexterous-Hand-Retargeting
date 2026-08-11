from typing import Any, Dict, Iterator, Tuple

import numpy as np

from utils.dataset.dataset import DexYCBVideoDataset


def sample_random_valid_frame(
    dataset: DexYCBVideoDataset, rng: np.random.Generator
) -> Tuple[int, int, Dict[str, Any]]:
    for data_id in rng.permutation(len(dataset)):
        data_id = int(data_id)
        sample = dataset[data_id]
        valid_frame_ids = np.flatnonzero(np.abs(sample["hand_pose"]).sum(axis=(1, 2)) > 1e-5)
        if valid_frame_ids.size > 0:
            frame_id = int(rng.choice(valid_frame_ids))
            return data_id, frame_id, sample
    raise ValueError("Could not find a DexYCB trajectory with a valid MANO frame.")


def sample_random_valid_trajectory(
    dataset: DexYCBVideoDataset, rng: np.random.Generator
) -> Tuple[int, Dict[str, Any]]:
    for data_id, sample in iter_random_valid_trajectories(dataset, rng):
        return data_id, sample
    raise ValueError("Could not find a DexYCB trajectory with at least 3 valid MANO frames.")


def iter_random_valid_trajectories(
    dataset: DexYCBVideoDataset, rng: np.random.Generator
) -> Iterator[Tuple[int, Dict[str, Any]]]:
    for data_id in rng.permutation(len(dataset)):
        data_id = int(data_id)
        sample = dataset[data_id]
        valid_frame_ids = np.flatnonzero(np.abs(sample["hand_pose"]).sum(axis=(1, 2)) > 1e-5)
        if valid_frame_ids.size >= 3:
            yield data_id, sample
