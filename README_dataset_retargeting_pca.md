# Retargeted Grasp Dataset, Retargeting Pipeline, and PCA Analysis

This document summarizes the dataset used in this repository, how the human-hand trajectories are retargeted to the `robotis_5f_hand`, and what the current PCA experiments show. It is written as a report-ready technical note so it can be expanded into a longer project report later.

## 1. Overview

The workflow in this repository has three main stages:

1. Load human hand-object interaction trajectories from DexYCB.
2. Convert selected grasp phases into robot joint configurations (`qpos`) for the `robotis_5f_hand`.
3. Fit a PCA model to the retargeted robot joint vectors to study how much of the motion can be explained in a lower-dimensional latent space.

The relevant entry points are:

- Dataset generation: `generate_retargeted_grasp_dataset.py`
- Interactive retargeting viewer: `custom_hand_retargeting.py`
- PCA training/export: `train_pca.py`
- PCA latent viewer: `view_pca_latent_viser.py`

## 2. Source Dataset

### 2.1 Dataset origin

The source data is loaded from DexYCB through `DexYCBVideoDataset` in `utils/dataset/dataset.py`. Each trajectory sample contains:

- `hand_pose`: MANO hand pose sequence
- `object_pose`: per-frame YCB object poses
- `extrinsics`: camera extrinsics
- `ycb_ids`: object identity labels
- `hand_shape`: MANO shape parameters
- `object_mesh_file`: mesh path for each object
- `capture_name`: capture identifier

The current dataset-generation config is `configs/dataset/retargeted_grasps_test_config.yaml`.

### 2.2 Dataset configuration used

The saved retargeted dataset in `assets/datasets/retargeted_grasps_test` was generated with:

- `dexycb_dir: assets/datasets/dexycb`
- `hand_type: right`
- `retarget_config_path: configs/retargeting/robotis_5f.yaml`
- `trajectory_fraction: 1.0`
- `max_trajectories: 100000000000000000`
- `seed: 0`
- `trajectory_sample_spacing_m: 0.01`
- `inflection_stride: 4`
- `approach_offset_m: 0.05`
- `lift_offset_m: 0.05`
- `refine_window_m: 0.10`
- `refine_resample_spacing_m: 0.005`

### 2.3 Final retargeted dataset statistics

From `assets/datasets/retargeted_grasps_test/metadata.json`, the current dataset contains:

- `100` selected trajectories
- `100` successfully processed trajectories
- `0` skipped trajectories
- `300` total datapoints
- robot: `robotis_5f_hand`
- hand side: `right`

Because each processed trajectory contributes three grasp-phase snapshots, the final dataset has:

- `100` approach states
- `100` grasp states
- `100` lift states

## 3. Grasp-Phase Extraction

The retargeted dataset is not built from every frame. Instead, each DexYCB trajectory is reduced to three representative grasp-phase frames:

- `approach`
- `grasp`
- `lift`

This is implemented in `utils/dataset/grasp_phase_utils.py` and `utils/dataset/retargeted_grasp_dataset.py`.

### 3.1 Wrist-trajectory analysis

The pipeline first reconstructs MANO joint positions in world coordinates and tracks the wrist trajectory. Phase selection then works as follows:

1. Resample the wrist trajectory by arc length.
2. Compute local motion-direction changes using an `inflection_stride` of `4`.
3. Find the strongest turning point in the middle part of the trajectory.
4. Refine that turning point inside a local arc-length window.
5. Use the refined turning point as the `grasp` frame.
6. Choose `approach` and `lift` at fixed arc-length offsets of `-0.05 m` and `+0.05 m` from the grasp point.

This makes the dataset phase-based rather than frame-dense, which is useful for building compact posture datasets for robot grasp analysis.

### 3.2 Grasped-object selection

The grasped object is chosen automatically. At the reference frame, the pipeline:

- takes the MANO fingertip positions,
- transforms every candidate object mesh into world coordinates,
- computes fingertip-to-object distances,
- selects the object with the lowest mean fingertip distance.

This gives each datapoint a single target object identity and pose.

## 4. Retargeting Method

The retargeting code lives primarily in `retargeting/custom_hand_retargeting_fn.py`.

### 4.1 Robot model and mapping

The robot being controlled is `robotis_5f_hand`, loaded from:

- URDF: `assets/robots/robotis_5f_hand/robotis_5f_hand.urdf`
- MANO-link mapping: `assets/robots/robotis_5f_hand/robotis_5f_right_mano_link_mapping.json`

The robot hand uses 20 joint coordinates:

- 4 thumb joints
- 4 index joints
- 4 middle joints
- 4 ring joints
- 4 little-finger joints

The mapping file assigns MANO keypoints to robot links plus local offsets on those links. This defines which parts of the robot should match which parts of the MANO hand.

### 4.2 Optimization target

The main retargeting path does **not** directly solve for fingertip position matching. Instead, it matches **relative orientation structure** along the MANO kinematic tree.

In outline:

1. Compute local coordinate frames for each MANO point.
2. Compute analogous local frames for the mapped robot points.
3. For each mapped parent-child edge, compare the MANO relative rotation to the robot relative rotation.
4. Minimize the stacked orientation residuals with joint-limit bounds.
5. Add a regularization term toward a fixed reference robot pose.

The optimizer is `scipy.optimize.least_squares` with:

- method: `trf`
- maximum function evaluations: `80`
- joint-limit bounds from the robot model
- regularization weight: `1e-3`

This means the solver is finding a robot posture whose articulated finger orientations resemble the MANO hand as closely as possible while staying close to a nominal hand pose and within kinematic limits.

### 4.3 Root pose handling

The retargeting step solves only the robot finger joint configuration `retargeted_qpos`. After that, the code computes an aligned MANO-style root pose for the retargeted robot hand. As a result, each datapoint stores:

- the robot joint vector,
- the human MANO root pose,
- an aligned retargeted root pose,
- the target object pose.

This separation is useful because it lets downstream models study finger articulation independently from global hand placement.

## 5. Saved Dataset Format

The retargeted dataset is saved as:

- `assets/datasets/retargeted_grasps_test/data.npz`
- `assets/datasets/retargeted_grasps_test/metadata.json`
- `assets/datasets/retargeted_grasps_test/config_resolved.yaml`

Each datapoint in `data.npz` contains the following fields:

- `trajectory_id`
- `capture_name`
- `frame_id`
- `phase`
- `phase_index`
- `human_mano`
- `human_shape`
- `camera_transform`
- `human_mano_root_translation`
- `human_mano_root_orientation_wxyz`
- `retargeted_qpos`
- `retargeted_mano_root_translation`
- `retargeted_mano_root_orientation_wxyz`
- `object_identifier`
- `object_name`
- `object_mesh_file`
- `object_translation`
- `object_orientation_wxyz`

The most important field for the PCA study is `retargeted_qpos`, which is a 20-dimensional robot joint vector.

## 6. PCA Analysis

### 6.1 Goal

The PCA experiment asks a simple question:

> How many dimensions are needed to represent the retargeted 20-DoF hand postures with acceptable reconstruction error?

This provides a linear baseline against which nonlinear latent models such as IRVAE can later be compared.

### 6.2 Method

The PCA implementation is in `utils/training/pca.py`, and the experiment driver is `train_pca.py`.

The current setup:

- input dimension: `20`
- train/validation split: `90% / 10%`
- split sizes: `270` train, `30` validation
- centering: enabled
- fitting method: SVD on the centered training-set `qpos`

The model computes:

- dataset mean joint vector
- top `z_dim` principal directions
- explained variance ratio
- latent codes for all samples
- reconstruction statistics on train, validation, and full data

Encoding and decoding are:

- encode: `z = (x - mean) @ components^T`
- decode: `x_hat = z @ components + mean`

### 6.3 Saved PCA artifacts

Each PCA run saves:

- `pca_model.npz`
- `latent_stats.npz`
- `train_latent_stats.npz`
- `validation_latent_stats.npz`
- `reconstruction_stats.npz`
- `reconstruction_stats_training.npz`
- `reconstruction_stats_validation.npz`
- `summary.json`
- `manifest.json`
- `robot_metadata.json`

Two saved runs currently exist in the repository:

- `results/retargeted_grasps/pca/qpos_pca_z3`
- `results/retargeted_grasps/pca/qpos_pca_z5`

## 7. PCA Results

### 7.1 `z_dim = 3`

From `results/retargeted_grasps/pca/qpos_pca_z3/summary.json`:

- explained variance ratio:
  - PC1: `0.4216`
  - PC2: `0.2077`
  - PC3: `0.1235`
- cumulative explained variance: `0.7527`
- validation mean L2 reconstruction error: `0.6749`
- validation std L2: `0.3218`
- validation p95 L2: `1.1939`
- validation p99 L2: `1.3891`
- validation mean MSE: `0.0280`

Interpretation:

- A 3D linear latent space captures about `75.3%` of the posture variance.
- This is already fairly compact, but reconstruction error is still noticeable.
- A 3D latent representation is likely useful for visualization and coarse control structure, but it loses some articulation detail.

### 7.2 `z_dim = 5`

From `results/retargeted_grasps/pca/qpos_pca_z5/summary.json`:

- explained variance ratio:
  - PC1: `0.4169`
  - PC2: `0.2085`
  - PC3: `0.1229`
  - PC4: `0.0854`
  - PC5: `0.0447`
- cumulative explained variance: `0.8783`
- validation mean L2 reconstruction error: `0.4335`
- validation std L2: `0.2769`
- validation p95 L2: `0.9128`
- validation p99 L2: `1.3424`
- validation mean MSE: `0.0132`

Interpretation:

- Increasing the latent dimension from 3 to 5 raises explained variance from about `75.3%` to `87.8%`.
- Validation reconstruction error drops substantially:
  - mean L2 decreases from `0.6749` to `0.4335`
  - mean MSE decreases from `0.0280` to `0.0132`
- This suggests that a 5D linear latent space preserves much more of the hand posture detail while still reducing the original 20D joint space by 75%.

### 7.3 Compact comparison

| Run | Latent dim | Cumulative explained variance | Val mean L2 | Val p95 L2 | Val mean MSE |
| --- | --- | --- | --- | --- | --- |
| `qpos_pca_z3` | 3 | `0.7527` | `0.6749` | `1.1939` | `0.0280` |
| `qpos_pca_z5` | 5 | `0.8783` | `0.4335` | `0.9128` | `0.0132` |

## 8. Main Takeaways

The current experiments support the following points:

1. The repository builds a clean, phase-based grasp-posture dataset by reducing DexYCB trajectories to approach, grasp, and lift snapshots.
2. Retargeting is performed through MANO-to-robot structural orientation matching, not simple fingertip position IK.
3. The resulting dataset for `robotis_5f_hand` currently contains `300` labeled robot-hand postures.
4. The retargeted posture vectors are highly compressible.
5. A 3D PCA latent captures the dominant posture structure, while a 5D PCA latent provides a notably better linear reconstruction.

## 9. Reproduction Commands

Generate the retargeted dataset:

```bash
python3 generate_retargeted_grasp_dataset.py configs/dataset/retargeted_grasps_test_config.yaml
```

Run PCA with 3 latent dimensions:

```bash
python3 train_pca.py --config configs/train/pca_test.yaml --run qpos_pca_z3
```

Run PCA with 5 latent dimensions:

```bash
python3 train_pca.py --config configs/train/pca_z5.yaml --run qpos_pca_z5
```

Inspect the latent space interactively:

```bash
python3 view_pca_latent_viser.py --model results/retargeted_grasps/pca/qpos_pca_z3/pca_model.npz
```

## 10. Useful Report Angles

If this README is used as the basis for a longer report, the most natural sections would be:

- motivation for reducing dexterous hand posture dimensionality,
- description of DexYCB as the human-hand source domain,
- phase-based dataset extraction rather than frame-dense extraction,
- MANO-to-robot retargeting through structural orientation matching,
- PCA as a linear baseline for posture compression,
- comparison between low-dimensional reconstructions and the original joint space,
- discussion of why a nonlinear model such as IRVAE may outperform PCA.
