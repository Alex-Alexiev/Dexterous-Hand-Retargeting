import tempfile
from pathlib import Path

from dex_retargeting import yourdfpy as urdf


def resolve_visual_urdf(urdf_path: Path) -> Path:
    if "glb" in urdf_path.stem:
        return urdf_path
    glb_urdf_path = urdf_path.with_stem(urdf_path.stem + "_glb")
    return glb_urdf_path if glb_urdf_path.is_file() else urdf_path


def build_temp_urdf(urdf_path: Path, add_dummy_free_joints: bool) -> Path:
    robot_urdf = urdf.URDF.load(
        str(urdf_path),
        add_dummy_free_joints=add_dummy_free_joints,
        build_scene_graph=False,
    )
    temp_dir = Path(tempfile.mkdtemp(prefix="dex_retargeting-custom-retarget-viser-"))
    temp_path = temp_dir / urdf_path.name
    robot_urdf.write_xml_file(str(temp_path))
    return temp_path
