from pathlib import Path

import tyro

from utils.dataset.dataset import DexYCBVideoDataset
from utils.dataset.retargeted_grasp_dataset import (
    build_retargeted_phase_datapoints,
    build_retargeting_resources,
    load_generation_config,
    resolve_generation_paths,
    save_retargeted_grasp_dataset,
    select_trajectory_indices,
)


def main(config: str) -> None:
    repo_root = Path(__file__).resolve().parent
    config = load_generation_config(config, repo_root)
    dexycb_dir, output_dir, urdf_path, mapping_path, reference_pose_path = (
        resolve_generation_paths(config, repo_root)
    )

    if not dexycb_dir.is_dir():
        raise ValueError(f"DexYCB directory does not exist: {dexycb_dir}")
    if not urdf_path.is_file():
        raise ValueError(f"URDF does not exist: {urdf_path}")
    if not mapping_path.is_file():
        raise ValueError(f"Mapping does not exist: {mapping_path}")

    dataset = DexYCBVideoDataset(str(dexycb_dir), hand_type=config.hand_type)
    if len(dataset) == 0:
        raise ValueError("DexYCBVideoDataset is empty with the current config.")

    selected_indices = select_trajectory_indices(len(dataset), config)
    resources = build_retargeting_resources(urdf_path, mapping_path, reference_pose_path)

    all_datapoints = []
    skipped_trajectories = []
    processed_count = 0
    for ordinal, data_id in enumerate(selected_indices.tolist(), start=1):
        sample = dataset[int(data_id)]
        try:
            datapoints = build_retargeted_phase_datapoints(
                sample=sample,
                data_id=int(data_id),
                config=config,
                resources=resources,
            )
        except Exception as exc:
            skipped_trajectories.append(
                {
                    "trajectory_id": int(data_id),
                    "capture_name": str(sample.get("capture_name", f"traj_{data_id}")),
                    "reason": str(exc),
                }
            )
            print(
                f"[skip {ordinal}/{len(selected_indices)}] trajectory={data_id} "
                f"capture={sample.get('capture_name', 'unknown')} reason={exc}"
            )
            continue

        all_datapoints.extend(datapoints)
        processed_count += 1
        print(
            f"[done {ordinal}/{len(selected_indices)}] trajectory={data_id} "
            f"capture={sample['capture_name']} datapoints={len(datapoints)}"
        )

    data_file = save_retargeted_grasp_dataset(
        output_dir,
        config=config,
        resources=resources,
        datapoints=all_datapoints,
        skipped_trajectories=skipped_trajectories,
        num_selected_trajectories=len(selected_indices),
        num_processed_trajectories=processed_count,
    )
    print(
        f"Saved retargeted grasp dataset with {len(all_datapoints)} datapoints "
        f"from {processed_count}/{len(selected_indices)} trajectories to {data_file}."
    )


if __name__ == "__main__":
    tyro.cli(main)
