import argparse
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import trimesh


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STL = REPO_ROOT / "LIBERO/libero/libero/assets/articulated_objects/flat_stove/stove_burner/burnerplate.stl"
DEFAULT_OBJ = REPO_ROOT / "SDF/assets/Libero-merge/stove_burner_combined_p_scaled.obj"


def load_mesh_stats(mesh_path: Path, scale: float = 1.0):
    mesh = trimesh.load_mesh(str(mesh_path), force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(g for g in mesh.geometry.values()))
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    if verts.ndim != 2 or verts.shape[1] != 3 or verts.shape[0] == 0:
        raise RuntimeError(f"Mesh has no valid vertices: {mesh_path}")
    verts = verts * float(scale)
    bounds = np.stack([verts.min(axis=0), verts.max(axis=0)], axis=0).astype(np.float64)
    centroid = verts.mean(axis=0)
    return {
        "bounds": bounds,
        "centroid": centroid,
        "bottom_z": float(bounds[0, 2]),
    }


def make_xml(stl_path: Path, obj_path: Path, stl_pos, obj_pos, stl_scale: float, obj_scale: float):
    return f"""
<mujoco model="show_micro">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.01" gravity="0 0 -9.81"/>
  <visual>
    <headlight diffuse="0.8 0.8 0.8" ambient="0.25 0.25 0.25" specular="0.1 0.1 0.1"/>
    <rgba haze="0.15 0.2 0.25 1"/>
  </visual>
  <asset>
    <mesh name="micro_stl" file="{stl_path.as_posix()}" scale="{stl_scale:.6f} {stl_scale:.6f} {stl_scale:.6f}"/>
    <mesh name="micro_obj" file="{obj_path.as_posix()}" scale="{obj_scale:.6f} {obj_scale:.6f} {obj_scale:.6f}"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.20 0.22 0.24" rgb2="0.14 0.16 0.18" width="512" height="512"/>
    <material name="grid_mat" texture="grid" texrepeat="6 6" reflectance="0.05"/>
  </asset>
  <worldbody>
    <light pos="0 0 2.5" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="2 2 0.05" material="grid_mat"/>
    <camera name="overview" pos="0 -1.2 0.55" xyaxes="1 0 0 0 0.35 0.94"/>

    <body name="stl_body" pos="{stl_pos[0]:.6f} {stl_pos[1]:.6f} {stl_pos[2]:.6f}">
      <geom name="stl_geom" type="mesh" mesh="micro_stl" rgba="0.90 0.35 0.30 1" contype="0" conaffinity="0"/>
    </body>

    <body name="obj_body" pos="{obj_pos[0]:.6f} {obj_pos[1]:.6f} {obj_pos[2]:.6f}">
      <geom name="obj_geom" type="mesh" mesh="micro_obj" rgba="0.25 0.55 0.95 0.85" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>
"""


def main():
    parser = argparse.ArgumentParser(description="Show original microwave STL and merged OBJ in MuJoCo without any rotation.")
    parser.add_argument("--stl", type=Path, default=DEFAULT_STL)
    parser.add_argument("--obj", type=Path, default=DEFAULT_OBJ)
    parser.add_argument("--separation", type=float, default=0.8, help="X-axis separation between the two meshes.")
    parser.add_argument("--y-offset", type=float, default=0.0, help="Shared Y offset.")
    parser.add_argument("--stl-scale", type=float, default=0.55, help="Uniform scale applied to the STL mesh.")
    parser.add_argument("--obj-scale", type=float, default=1.0, help="Uniform scale applied to the OBJ mesh.")
    parser.add_argument("--ground-gap", type=float, default=0.02, help="Lift each mesh so its local min-z sits this far above the ground plane.")
    args = parser.parse_args()

    stl_path = args.stl.expanduser().resolve()
    obj_path = args.obj.expanduser().resolve()
    if not stl_path.exists():
        raise FileNotFoundError(stl_path)
    if not obj_path.exists():
        raise FileNotFoundError(obj_path)

    stl_stats = load_mesh_stats(stl_path, scale=args.stl_scale)
    obj_stats = load_mesh_stats(obj_path, scale=args.obj_scale)

    stl_pos = np.array([-0.5 * args.separation, args.y_offset, args.ground_gap - stl_stats["bottom_z"]], dtype=np.float64)
    obj_pos = np.array([0.5 * args.separation, args.y_offset, args.ground_gap - obj_stats["bottom_z"]], dtype=np.float64)

    print("[STL]")
    print(f"  path: {stl_path}")
    print(f"  centroid(local): {stl_stats['centroid'].tolist()}")
    print(f"  bounds(local): {stl_stats['bounds'].tolist()}")
    print(f"  scale: {args.stl_scale}")
    print(f"  body pos(world): {stl_pos.tolist()}")
    print("[OBJ]")
    print(f"  path: {obj_path}")
    print(f"  centroid(local): {obj_stats['centroid'].tolist()}")
    print(f"  bounds(local): {obj_stats['bounds'].tolist()}")
    print(f"  scale: {args.obj_scale}")
    print(f"  body pos(world): {obj_pos.tolist()}")
    print("[NOTE] Both meshes are shown with identity rotation. Any orientation mismatch comes from the mesh local frames themselves.")

    xml = make_xml(stl_path, obj_path, stl_pos, obj_pos, args.stl_scale, args.obj_scale)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data, show_left_ui=True, show_right_ui=True) as viewer:
        viewer.cam.lookat[:] = np.array([0.0, 0.0, 0.12], dtype=np.float64)
        viewer.cam.distance = 1.2
        viewer.cam.azimuth = 90.0
        viewer.cam.elevation = -20.0
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()
