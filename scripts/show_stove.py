import argparse
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import trimesh


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLATE_STL = REPO_ROOT / "LIBERO" / "libero" / "libero" / "assets" / "articulated_objects" / "flat_stove" / "stove_burner" / "burnerplate.stl"
DEFAULT_MERGED_OBJ = REPO_ROOT / "SDF" / "assets" / "Libero-merge" / "stove_burner_combined_p_scaled.obj"

BURNER_BODY_POS = np.array([0.15, 0.0, 0.0], dtype=np.float64)
BOX_SIZE = np.array([0.095, 0.095, 0.02], dtype=np.float64)
BOX_POS = np.array([0.0, 0.0, 0.0], dtype=np.float64)
PLATE_POS = np.array([0.0, 0.0, 0.025], dtype=np.float64)
PLATE_SCALE = np.array([0.55, 0.55, 0.45], dtype=np.float64)


def load_mesh_vertices(mesh_path: Path):
    mesh = trimesh.load_mesh(str(mesh_path), force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(g for g in mesh.geometry.values()))
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    if verts.ndim != 2 or verts.shape[1] != 3 or verts.shape[0] == 0:
        raise RuntimeError(f"Mesh has no valid vertices: {mesh_path}")
    return verts


def compute_original_stove_stats(stl_path: Path, orig_scale: float):
    scale = float(orig_scale)
    plate_verts = load_mesh_vertices(stl_path)
    plate_verts = plate_verts * (PLATE_SCALE.reshape(1, 3) * scale)
    plate_verts = plate_verts + ((BURNER_BODY_POS + PLATE_POS) * scale).reshape(1, 3)

    box_center = (BURNER_BODY_POS + BOX_POS) * scale
    box_size = BOX_SIZE * scale
    signs = np.array([
        [-1, -1, -1],
        [-1, -1,  1],
        [-1,  1, -1],
        [-1,  1,  1],
        [ 1, -1, -1],
        [ 1, -1,  1],
        [ 1,  1, -1],
        [ 1,  1,  1],
    ], dtype=np.float64)
    box_corners = box_center.reshape(1, 3) + signs * box_size.reshape(1, 3)

    verts = np.vstack([plate_verts, box_corners])
    bounds = np.stack([verts.min(axis=0), verts.max(axis=0)], axis=0)
    centroid = verts.mean(axis=0)
    return {
        "bounds": bounds,
        "centroid": centroid,
        "bottom_z": float(bounds[0, 2]),
    }


def compute_mesh_stats(mesh_path: Path, scale: float):
    verts = load_mesh_vertices(mesh_path) * float(scale)
    bounds = np.stack([verts.min(axis=0), verts.max(axis=0)], axis=0)
    centroid = verts.mean(axis=0)
    return {
        "bounds": bounds,
        "centroid": centroid,
        "bottom_z": float(bounds[0, 2]),
    }


def make_xml(stl_path: Path, obj_path: Path, orig_pos, merged_pos, orig_scale: float, merged_scale: float):
    plate_scale = PLATE_SCALE * float(orig_scale)
    burner_body_pos = BURNER_BODY_POS * float(orig_scale)
    box_size = BOX_SIZE * float(orig_scale)
    box_pos = BOX_POS * float(orig_scale)
    plate_pos = PLATE_POS * float(orig_scale)
    return f"""
<mujoco model="show_stove">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.01" gravity="0 0 -9.81"/>
  <visual>
    <headlight diffuse="0.8 0.8 0.8" ambient="0.25 0.25 0.25" specular="0.1 0.1 0.1"/>
    <rgba haze="0.15 0.2 0.25 1"/>
  </visual>
  <asset>
    <mesh name="burnerplate" file="{stl_path.as_posix()}" scale="{plate_scale[0]:.6f} {plate_scale[1]:.6f} {plate_scale[2]:.6f}"/>
    <mesh name="stove_merged" file="{obj_path.as_posix()}" scale="{merged_scale:.6f} {merged_scale:.6f} {merged_scale:.6f}"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.20 0.22 0.24" rgb2="0.14 0.16 0.18" width="512" height="512"/>
    <material name="grid_mat" texture="grid" texrepeat="6 6" reflectance="0.05"/>
  </asset>
  <worldbody>
    <light pos="0 0 2.5" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="2 2 0.05" material="grid_mat"/>
    <camera name="overview" pos="0 -1.3 0.6" xyaxes="1 0 0 0 0.35 0.94"/>

    <body name="orig_root" pos="{orig_pos[0]:.6f} {orig_pos[1]:.6f} {orig_pos[2]:.6f}">
      <body name="orig_burner" pos="{burner_body_pos[0]:.6f} {burner_body_pos[1]:.6f} {burner_body_pos[2]:.6f}">
        <geom name="orig_box" type="box" size="{box_size[0]:.6f} {box_size[1]:.6f} {box_size[2]:.6f}" pos="{box_pos[0]:.6f} {box_pos[1]:.6f} {box_pos[2]:.6f}" rgba="0.55 0.55 0.58 1" contype="0" conaffinity="0"/>
        <geom name="orig_plate" type="mesh" mesh="burnerplate" pos="{plate_pos[0]:.6f} {plate_pos[1]:.6f} {plate_pos[2]:.6f}" rgba="0.92 0.40 0.25 1" contype="0" conaffinity="0"/>
      </body>
    </body>

    <body name="merged_root" pos="{merged_pos[0]:.6f} {merged_pos[1]:.6f} {merged_pos[2]:.6f}">
      <geom name="merged_geom" type="mesh" mesh="stove_merged" rgba="0.25 0.55 0.95 0.85" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>
"""


def main():
    parser = argparse.ArgumentParser(description="Show original flat stove burner (box + burnerplate STL) and merged OBJ in MuJoCo without knob and without any rotation.")
    parser.add_argument("--stl", type=Path, default=DEFAULT_PLATE_STL)
    parser.add_argument("--obj", type=Path, default=DEFAULT_MERGED_OBJ)
    parser.add_argument("--separation", type=float, default=0.6, help="X-axis separation between original and merged stove.")
    parser.add_argument("--y-offset", type=float, default=0.0, help="Shared Y offset.")
    parser.add_argument("--orig-scale", type=float, default=1.0, help="Uniform scale for the original stove definition.")
    parser.add_argument("--merged-scale", type=float, default=1.0, help="Uniform scale for the merged OBJ.")
    parser.add_argument("--ground-gap", type=float, default=0.02, help="Lift each stove so its local min-z sits this far above the ground plane.")
    args = parser.parse_args()

    stl_path = args.stl.expanduser().resolve()
    obj_path = args.obj.expanduser().resolve()
    if not stl_path.exists():
        raise FileNotFoundError(stl_path)
    if not obj_path.exists():
        raise FileNotFoundError(obj_path)

    orig_stats = compute_original_stove_stats(stl_path, args.orig_scale)
    merged_stats = compute_mesh_stats(obj_path, args.merged_scale)

    orig_pos = np.array([-0.5 * args.separation, args.y_offset, args.ground_gap - orig_stats["bottom_z"]], dtype=np.float64)
    merged_pos = np.array([0.5 * args.separation, args.y_offset, args.ground_gap - merged_stats["bottom_z"]], dtype=np.float64)

    print("[ORIGINAL STOVE]")
    print(f"  stl: {stl_path}")
    print(f"  scale: {args.orig_scale}")
    print(f"  centroid(local): {orig_stats['centroid'].tolist()}")
    print(f"  bounds(local): {orig_stats['bounds'].tolist()}")
    print(f"  body pos(world): {orig_pos.tolist()}")
    print("[MERGED STOVE]")
    print(f"  obj: {obj_path}")
    print(f"  scale: {args.merged_scale}")
    print(f"  centroid(local): {merged_stats['centroid'].tolist()}")
    print(f"  bounds(local): {merged_stats['bounds'].tolist()}")
    print(f"  body pos(world): {merged_pos.tolist()}")
    print("[NOTE] Both stoves are shown with identity rotation. The original stove uses the XML-defined box shell plus burnerplate STL.")

    xml = make_xml(stl_path, obj_path, orig_pos, merged_pos, args.orig_scale, args.merged_scale)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data, show_left_ui=True, show_right_ui=True) as viewer:
        viewer.cam.lookat[:] = np.array([0.15, 0.0, 0.05], dtype=np.float64)
        viewer.cam.distance = 1.3
        viewer.cam.azimuth = 90.0
        viewer.cam.elevation = -22.0
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()
