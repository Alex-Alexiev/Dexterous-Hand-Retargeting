import argparse
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from utils.system.path_utils import resolve_repo_path
from utils.training.pca import decode_pca, load_pca_model


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_PCA_DIR = "results/retargeted_grasps/pca/qpos_pca_z5"
DEFAULT_OUTPUT = "results/retargeted_grasps/pca/qpos_pca_z5/pca_star_figure.png"
BACKGROUND = (246, 244, 238)
LINE_COLOR = (66, 74, 86)
NEGATIVE_COLOR = (214, 92, 73)
POSITIVE_COLOR = (59, 128, 99)
TEXT_COLOR = (34, 38, 46)
HAND_BASE_COLOR = np.asarray([110.0, 168.0, 122.0], dtype=np.float32)


def _normalize(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        return np.zeros_like(vec)
    return vec / norm


def _rpy_matrix(rpy: Sequence[float]) -> np.ndarray:
    roll, pitch, yaw = [float(v) for v in rpy]
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.asarray([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float32)
    ry = np.asarray([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float32)
    rz = np.asarray([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float32)
    return rz @ ry @ rx


def _axis_angle_matrix(axis: Sequence[float], angle: float) -> np.ndarray:
    axis = _normalize(np.asarray(axis, dtype=np.float32))
    if float(np.linalg.norm(axis)) < 1e-8 or abs(angle) < 1e-8:
        return np.eye(3, dtype=np.float32)
    x, y, z = axis
    c = math.cos(angle)
    s = math.sin(angle)
    one_c = 1.0 - c
    return np.asarray(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=np.float32,
    )


def _make_transform(rotation: np.ndarray, translation: Sequence[float]) -> np.ndarray:
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = rotation.astype(np.float32)
    transform[:3, 3] = np.asarray(translation, dtype=np.float32)
    return transform


def _transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def _load_ascii_stl(mesh_path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: List[List[float]] = []
    faces: List[List[int]] = []
    current: List[int] = []
    with mesh_path.open("r", errors="ignore") as f:
        for line in f:
            stripped = line.strip()
            if not stripped.startswith("vertex "):
                continue
            _, x, y, z = stripped.split()
            vertices.append([float(x), float(y), float(z)])
            current.append(len(vertices) - 1)
            if len(current) == 3:
                faces.append(current)
                current = []
    if not vertices or not faces:
        raise ValueError(f"Failed to parse ASCII STL mesh: {mesh_path}")
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int32)


def _load_binary_stl(mesh_path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = mesh_path.read_bytes()
    if len(data) < 84:
        raise ValueError(f"Binary STL is too small: {mesh_path}")
    tri_count = int(np.frombuffer(data[80:84], dtype=np.uint32)[0])
    expected = 84 + tri_count * 50
    if expected != len(data):
        raise ValueError(f"Binary STL size mismatch for {mesh_path}: expected {expected}, got {len(data)}")

    vertices = np.empty((tri_count * 3, 3), dtype=np.float32)
    faces = np.arange(tri_count * 3, dtype=np.int32).reshape(tri_count, 3)
    offset = 84
    for tri_idx in range(tri_count):
        record = data[offset : offset + 50]
        tri_vertices = np.frombuffer(record[12:48], dtype="<f4").reshape(3, 3)
        vertices[tri_idx * 3 : tri_idx * 3 + 3] = tri_vertices
        offset += 50
    return vertices, faces


def _load_stl(mesh_path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        return _load_binary_stl(mesh_path)
    except Exception:
        return _load_ascii_stl(mesh_path)


def _downsample_faces(faces: np.ndarray, max_faces: int) -> np.ndarray:
    if faces.shape[0] <= max_faces:
        return faces
    indices = np.linspace(0, faces.shape[0] - 1, num=max_faces, dtype=np.int32)
    return faces[indices]


@dataclass
class JointSpec:
    name: str
    parent: str
    child: str
    joint_type: str
    origin_tf: np.ndarray
    axis: np.ndarray
    limit_lower: float
    limit_upper: float


class SimpleUrdfHand:
    def __init__(self, urdf_path: Path, *, max_faces_per_mesh: int = 700):
        self.urdf_path = urdf_path
        self.mesh_dir = urdf_path.parent
        tree = ET.parse(urdf_path)
        root = tree.getroot()

        self.joints_by_parent: Dict[str, List[JointSpec]] = {}
        self.parent_joint_by_link: Dict[str, str] = {}
        self.joints_by_name: Dict[str, JointSpec] = {}
        self.joint_order: List[str] = []
        self.joint_limits_lower: List[float] = []
        self.joint_limits_upper: List[float] = []

        for joint_elem in root.findall("joint"):
            name = joint_elem.attrib["name"]
            joint_type = joint_elem.attrib["type"]
            parent = joint_elem.find("parent").attrib["link"]
            child = joint_elem.find("child").attrib["link"]
            origin_elem = joint_elem.find("origin")
            xyz = [0.0, 0.0, 0.0]
            rpy = [0.0, 0.0, 0.0]
            if origin_elem is not None:
                xyz = [float(v) for v in origin_elem.attrib.get("xyz", "0 0 0").split()]
                rpy = [float(v) for v in origin_elem.attrib.get("rpy", "0 0 0").split()]
            axis_elem = joint_elem.find("axis")
            axis = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
            if axis_elem is not None:
                axis = np.asarray([float(v) for v in axis_elem.attrib.get("xyz", "0 0 1").split()], dtype=np.float32)

            lower = -np.inf
            upper = np.inf
            limit_elem = joint_elem.find("limit")
            if limit_elem is not None:
                lower = float(limit_elem.attrib.get("lower", "-inf"))
                upper = float(limit_elem.attrib.get("upper", "inf"))

            spec = JointSpec(
                name=name,
                parent=parent,
                child=child,
                joint_type=joint_type,
                origin_tf=_make_transform(_rpy_matrix(rpy), xyz),
                axis=axis,
                limit_lower=lower,
                limit_upper=upper,
            )
            self.joints_by_parent.setdefault(parent, []).append(spec)
            self.parent_joint_by_link[child] = name
            self.joints_by_name[name] = spec
            if joint_type == "revolute":
                self.joint_order.append(name)
                self.joint_limits_lower.append(lower)
                self.joint_limits_upper.append(upper)

        self.joint_limits_lower = np.asarray(self.joint_limits_lower, dtype=np.float32)
        self.joint_limits_upper = np.asarray(self.joint_limits_upper, dtype=np.float32)

    def clamp_qpos(self, qpos: np.ndarray) -> np.ndarray:
        qpos = np.asarray(qpos, dtype=np.float32).copy()
        return np.clip(qpos, self.joint_limits_lower, self.joint_limits_upper)

    def _link_transforms(self, qpos_by_name: Dict[str, float]) -> Dict[str, np.ndarray]:
        transforms: Dict[str, np.ndarray] = {"world": np.eye(4, dtype=np.float32)}
        stack = ["world"]
        while stack:
            parent = stack.pop()
            parent_tf = transforms[parent]
            for joint in self.joints_by_parent.get(parent, []):
                child_tf = parent_tf @ joint.origin_tf
                if joint.joint_type == "revolute":
                    angle = float(qpos_by_name.get(joint.name, 0.0))
                    joint_rot = _axis_angle_matrix(joint.axis, angle)
                    child_tf = child_tf @ _make_transform(joint_rot, [0.0, 0.0, 0.0])
                transforms[joint.child] = child_tf
                stack.append(joint.child)
        return transforms

    def gather_joint_positions(self, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        qpos = np.asarray(qpos, dtype=np.float32)
        qpos_by_name = {name: float(qpos[idx]) for idx, name in enumerate(self.joint_order)}
        link_tf = self._link_transforms(qpos_by_name)

        joint_names = ["palm"] + list(self.joints_by_name.keys())
        points: List[np.ndarray] = [link_tf.get("palm", np.eye(4, dtype=np.float32))[:3, 3]]
        point_index_by_name = {"palm": 0}

        for joint_name, joint in self.joints_by_name.items():
            parent_tf = link_tf[joint.parent]
            joint_tf = parent_tf @ joint.origin_tf
            point_index_by_name[joint_name] = len(points)
            points.append(joint_tf[:3, 3].astype(np.float32))

        edges: List[Tuple[int, int]] = []
        for joint_name, joint in self.joints_by_name.items():
            if joint.parent == "palm":
                parent_index = point_index_by_name["palm"]
            else:
                parent_joint_name = self.parent_joint_by_link.get(joint.parent)
                if parent_joint_name is None:
                    continue
                parent_index = point_index_by_name[parent_joint_name]
            edges.append((parent_index, point_index_by_name[joint_name]))

        return (
            np.asarray(points, dtype=np.float32),
            np.asarray(edges, dtype=np.int32),
            np.asarray(joint_names, dtype=object),
        )


def _view_rotation() -> np.ndarray:
    return _rpy_matrix((math.radians(18.0), math.radians(-8.0), math.radians(38.0)))


def _render_pose_image(
    renderer: SimpleUrdfHand,
    qpos: np.ndarray,
    *,
    size: int,
    bounds_center: np.ndarray,
    bounds_extent: float,
    accent_color: tuple[int, int, int],
) -> Image.Image:
    points_world, edges, _ = renderer.gather_joint_positions(qpos)
    if points_world.shape[0] == 0:
        return Image.new("RGB", (size, size), BACKGROUND)

    view_rot = _view_rotation()
    centered = points_world - bounds_center.reshape(1, 3)
    points_cam = centered @ view_rot.T

    scale = 0.84 * size / max(bounds_extent, 1e-6)
    projected = points_cam[:, :2] * scale
    projected[:, 0] += size * 0.5
    projected[:, 1] = size * 0.5 - projected[:, 1]

    image = Image.new("RGB", (size, size), BACKGROUND)
    draw = ImageDraw.Draw(image)

    if edges.shape[0] > 0:
        edge_depth = points_cam[edges][:, :, 2].mean(axis=1)
        order = np.argsort(edge_depth)
        for idx in order:
            i0, i1 = edges[idx]
            draw.line(
                [tuple(projected[i0].tolist()), tuple(projected[i1].tolist())],
                fill=(82, 88, 96),
                width=max(2, int(size * 0.016)),
            )

    point_depth = points_cam[:, 2]
    order = np.argsort(point_depth)
    root_radius = max(4, int(size * 0.030))
    joint_radius = max(3, int(size * 0.022))
    for idx in order:
        x, y = projected[idx]
        radius = root_radius if idx == 0 else joint_radius
        fill = (255, 251, 245) if idx == 0 else accent_color
        outline = (64, 68, 76)
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=fill,
            outline=outline,
            width=2,
        )
    return image


def _load_font(size: int):
    for name in ("DejaVuSans.ttf", "Arial.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _rounded_panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], fill: tuple[int, int, int], outline: tuple[int, int, int]) -> None:
    draw.rounded_rectangle(box, radius=16, fill=fill, outline=outline, width=2)


def build_star_figure(
    *,
    pca_dir: Path,
    output_path: Path,
    scale: float,
    image_size: int,
    max_faces_per_mesh: int,
) -> Path:
    metadata_path = pca_dir / "robot_metadata.json"
    latent_stats_path = pca_dir / "latent_stats.npz"
    model_path = pca_dir / "pca_model.npz"

    pca_model = load_pca_model(model_path)
    with metadata_path.open("r") as f:
        import json

        metadata = json.load(f)
    with np.load(latent_stats_path, allow_pickle=False) as payload:
        latent_std = payload["std"].astype(np.float32)

    urdf_path = Path(metadata["urdf_path"])
    renderer = SimpleUrdfHand(urdf_path, max_faces_per_mesh=max_faces_per_mesh)
    z_dim = int(pca_model["z_dim"])
    if z_dim != 5:
        raise ValueError(f"This figure generator expects z_dim=5, got {z_dim}.")

    poses = []
    labels = []
    for axis in range(z_dim):
        delta = float(scale) * float(latent_std[axis])
        latent_neg = np.zeros((z_dim,), dtype=np.float32)
        latent_pos = np.zeros((z_dim,), dtype=np.float32)
        latent_neg[axis] = -delta
        latent_pos[axis] = delta
        q_neg = renderer.clamp_qpos(decode_pca(pca_model, latent_neg).reshape(-1))
        q_pos = renderer.clamp_qpos(decode_pca(pca_model, latent_pos).reshape(-1))
        poses.append((q_neg, q_pos))
        labels.append((f"PC{axis + 1} -", f"PC{axis + 1} +", delta))

    all_points = []
    for q_neg, q_pos in poses:
        pts_neg, _, _ = renderer.gather_joint_positions(q_neg)
        pts_pos, _, _ = renderer.gather_joint_positions(q_pos)
        if pts_neg.size:
            all_points.append(pts_neg)
        if pts_pos.size:
            all_points.append(pts_pos)
    stacked = np.concatenate(all_points, axis=0)
    bounds_min = stacked.min(axis=0)
    bounds_max = stacked.max(axis=0)
    bounds_center = 0.5 * (bounds_min + bounds_max)
    bounds_extent = float(np.max(bounds_max - bounds_min))

    renders: List[tuple[Image.Image, Image.Image]] = []
    for q_neg, q_pos in poses:
        renders.append(
            (
                _render_pose_image(
                    renderer,
                    q_neg,
                    size=image_size,
                    bounds_center=bounds_center,
                    bounds_extent=bounds_extent,
                    accent_color=NEGATIVE_COLOR,
                ),
                _render_pose_image(
                    renderer,
                    q_pos,
                    size=image_size,
                    bounds_center=bounds_center,
                    bounds_extent=bounds_extent,
                    accent_color=POSITIVE_COLOR,
                ),
            )
        )

    canvas_size = 1900
    canvas = Image.new("RGB", (canvas_size, canvas_size), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    title_font = _load_font(50)
    subtitle_font = _load_font(24)
    label_font = _load_font(28)
    small_font = _load_font(22)

    title = "PCA Axis Star for Robotis 5F Hand"
    subtitle = f"Each axis shows mean ± {scale:.2f}σ for one PCA dimension from qpos_pca_z5"
    draw.text((canvas_size // 2, 70), title, fill=TEXT_COLOR, anchor="mm", font=title_font)
    draw.text((canvas_size // 2, 118), subtitle, fill=(92, 96, 104), anchor="mm", font=subtitle_font)

    center = np.asarray([canvas_size * 0.5, canvas_size * 0.55], dtype=np.float32)
    arm_radius = 520.0
    inset_offset = 155.0
    negative_insets = []
    positive_insets = []
    for axis in range(5):
        theta = -math.pi / 2.0 + axis * (2.0 * math.pi / 5.0)
        direction = np.asarray([math.cos(theta), math.sin(theta)], dtype=np.float32)
        tip = center + arm_radius * direction
        neg_center = center + (arm_radius - inset_offset) * direction
        pos_center = center + (arm_radius + inset_offset) * direction
        draw.line([tuple(center), tuple(tip)], fill=LINE_COLOR, width=6)
        draw.ellipse(
            [tip[0] - 10, tip[1] - 10, tip[0] + 10, tip[1] + 10],
            fill=LINE_COLOR,
        )
        negative_insets.append((neg_center, direction))
        positive_insets.append((pos_center, direction))

        axis_label = f"PC{axis + 1}"
        label_pos = center + (arm_radius - 300.0) * direction
        draw.text(tuple(label_pos), axis_label, fill=TEXT_COLOR, anchor="mm", font=label_font)

    central_box = (
        int(center[0] - 210),
        int(center[1] - 88),
        int(center[0] + 210),
        int(center[1] + 88),
    )
    _rounded_panel(draw, central_box, fill=(255, 252, 247), outline=(188, 182, 171))
    draw.text((center[0], center[1] - 20), "Latent Mean", fill=TEXT_COLOR, anchor="mm", font=label_font)
    draw.text(
        (center[0], center[1] + 24),
        "Five radial axes, rendered at ± along one PC",
        fill=(94, 98, 108),
        anchor="mm",
        font=small_font,
    )

    panel_fill = (255, 252, 247)
    panel_outline = (196, 190, 181)
    panel_size = image_size + 26
    for axis, ((neg_img, pos_img), (neg_label, pos_label, delta)) in enumerate(zip(renders, labels)):
        for center_pos, image, label, accent, sign in (
            (negative_insets[axis][0], neg_img, neg_label, NEGATIVE_COLOR, "-"),
            (positive_insets[axis][0], pos_img, pos_label, POSITIVE_COLOR, "+"),
        ):
            x0 = int(round(center_pos[0] - panel_size / 2))
            y0 = int(round(center_pos[1] - panel_size / 2))
            x1 = x0 + panel_size
            y1 = y0 + panel_size
            _rounded_panel(draw, (x0, y0, x1, y1), fill=panel_fill, outline=panel_outline)
            canvas.paste(image, (x0 + 13, y0 + 13))
            banner_h = 34
            draw.rounded_rectangle(
                (x0 + 16, y0 + 14, x0 + 124, y0 + 14 + banner_h),
                radius=10,
                fill=accent,
            )
            draw.text((x0 + 70, y0 + 31), label, fill=(255, 255, 255), anchor="mm", font=small_font)
            draw.text(
                (x0 + panel_size / 2, y1 + 18),
                f"{sign}{delta:.3f}",
                fill=(88, 92, 100),
                anchor="mm",
                font=small_font,
            )

    footer = f"URDF: {urdf_path.name} | PCA run: {pca_dir.name}"
    draw.text((canvas_size // 2, canvas_size - 42), footer, fill=(102, 106, 114), anchor="mm", font=small_font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pca-dir", default=DEFAULT_PCA_DIR)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--image-size", type=int, default=210)
    parser.add_argument("--max-faces-per-mesh", type=int, default=700)
    args = parser.parse_args()

    pca_dir = resolve_repo_path(args.pca_dir, REPO_ROOT)
    output_path = resolve_repo_path(args.output, REPO_ROOT)
    result = build_star_figure(
        pca_dir=pca_dir,
        output_path=output_path,
        scale=args.scale,
        image_size=args.image_size,
        max_faces_per_mesh=args.max_faces_per_mesh,
    )
    print(f"Saved PCA star figure to {result}")


if __name__ == "__main__":
    main()
