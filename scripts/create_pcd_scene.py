"""
Create scene point clouds by merging object point clouds in LIBERO HDF5 datasets.

Supports:
- Single HDF5 input
- Directory input (recursive over *.hdf5)
- Selecting first N episodes per HDF5 via --episodes
- Optional table / countertop point cloud augmentation

Output NPZ keys:
  scene_pcd      : (T, N_max, 3) float32, padded with zeros when needed
  frame_points   : (T,) int32, valid point count per frame
  episode        : str
  object_names   : (K,) str
  source_hdf5    : str
"""

import argparse
import json
import re
from pathlib import Path

import h5py
import numpy as np

try:
    from create_pcd_dataset import (
        LiberoPoseExtractor,
        _mesh_from_mujoco_model,
        find_bddl_file,
        sample_points_from_mesh,
        sample_points_from_primitive,
        transform_point_cloud,
    )
except ImportError:
    from scripts.create_pcd_dataset import (
        LiberoPoseExtractor,
        _mesh_from_mujoco_model,
        find_bddl_file,
        sample_points_from_mesh,
        sample_points_from_primitive,
        transform_point_cloud,
    )


TABLE_ENTITY_NAME = "scene_table"
FLOOR_ENTITY_NAME = "scene_floor"
MJ_GEOM_TYPE_TO_PRIMITIVE = {2: "sphere", 3: "capsule", 4: "ellipsoid", 5: "cylinder", 6: "box"}
DEFAULT_VOXEL_SIZE = 0.005
DEFAULT_SURFACE_OVERSAMPLE = 2.0
DEFAULT_MIN_SAMPLES_PER_GEOM = 256
DEFAULT_MAX_SAMPLES_PER_GEOM = 2000000
TABLE_BODY_EXACT = {
    "table",
    "main_table",
    "kitchen_table",
    "study_table",
    "living_room_table",
    "living_room_table_col",
    "coffee_table",
    "coffee_table_col",
    "countertop",
}


def sorted_demo_keys(data_group):
    def key_fn(name):
        if name.startswith("demo_"):
            tail = name.split("_", 1)[1]
            if tail.isdigit():
                return int(tail)
        return name

    return sorted(list(data_group.keys()), key=key_fn)


def voxel_downsample_points(points, voxel_size):
    pts = np.asarray(points, dtype=np.float32)
    if pts.size == 0 or voxel_size is None or voxel_size <= 0:
        return pts
    coords = np.floor(pts / float(voxel_size)).astype(np.int64)
    _, inv = np.unique(coords, axis=0, return_inverse=True)
    counts = np.bincount(inv).astype(np.float32)
    sums = np.zeros((counts.shape[0], 3), dtype=np.float32)
    np.add.at(sums, inv, pts)
    return (sums / np.maximum(counts[:, None], 1.0)).astype(np.float32)


def estimate_surface_sample_count(
    surface_area,
    voxel_size,
    oversample=DEFAULT_SURFACE_OVERSAMPLE,
    min_samples=DEFAULT_MIN_SAMPLES_PER_GEOM,
    max_samples=DEFAULT_MAX_SAMPLES_PER_GEOM,
):
    if voxel_size <= 0:
        return int(max(min_samples, 1))
    area = float(max(surface_area, 1e-10))
    estimate = int(np.ceil((area / (float(voxel_size) * float(voxel_size))) * float(oversample)))
    return int(np.clip(estimate, int(min_samples), int(max_samples)))


def primitive_surface_area(geom_type, size):
    s = np.asarray(size, dtype=np.float64)
    if geom_type == "box":
        a, b, c = 2.0 * s[0], 2.0 * s[1], 2.0 * s[2]
        return 2.0 * (a * b + b * c + a * c)
    if geom_type == "sphere":
        r = float(s[0])
        return 4.0 * np.pi * r * r
    if geom_type == "cylinder":
        r, half_h = float(s[0]), float(s[1])
        h = 2.0 * half_h
        return 2.0 * np.pi * r * (r + h)
    if geom_type == "capsule":
        r, half_h = float(s[0]), float(s[1])
        h = 2.0 * half_h
        return 2.0 * np.pi * r * h + 4.0 * np.pi * r * r
    if geom_type == "ellipsoid":
        a, b, c = float(s[0]), float(s[1]), float(s[2])
        p = 1.6075
        return 4.0 * np.pi * (((a * b) ** p + (a * c) ** p + (b * c) ** p) / 3.0) ** (1.0 / p)
    return 0.0


def infer_task_name_from_dataset_path(hdf5_path):
    stem = Path(hdf5_path).stem
    for suffix in ["_demo_pcd", "_demo", "_pcd"]:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem


def find_bddl_from_hdf5(hdf5_path):
    bddl_name = ""
    with h5py.File(hdf5_path, "r") as f:
        if "data" in f and "env_args" in f["data"].attrs:
            try:
                env_args = json.loads(f["data"].attrs["env_args"])
                bddl_name = env_args.get("env_kwargs", {}).get("bddl_file_name", "")
            except Exception:
                bddl_name = ""

    if bddl_name:
        bddl_path = find_bddl_file(bddl_name)
        if bddl_path:
            return bddl_path

    task_name = infer_task_name_from_dataset_path(hdf5_path)
    guessed = find_bddl_file(f"{task_name}.bddl")
    if guessed:
        return guessed

    tokens = set(re.split(r"[^A-Za-z0-9]+", task_name.lower()))
    tokens = {t for t in tokens if t}
    bddl_root = Path(__file__).resolve().parents[1] / "libero" / "libero" / "bddl_files"
    best_path = None
    best_score = -1
    for cand in bddl_root.rglob("*.bddl"):
        cand_tokens = set(re.split(r"[^A-Za-z0-9]+", cand.stem.lower()))
        score = len(tokens.intersection(cand_tokens))
        if score > best_score:
            best_score = score
            best_path = cand

    return str(best_path) if best_path is not None else None


def select_table_bodies(body_names):
    selected = []
    for name in sorted(body_names):
        lname = name.lower()
        if lname in TABLE_BODY_EXACT:
            selected.append(name)
            continue
        if lname.endswith("_table") or lname.endswith("_table_col"):
            selected.append(name)
            continue
        if lname.endswith("countertop"):
            selected.append(name)
            continue

    return selected


def select_floor_bodies(body_names):
    selected = []
    for name in sorted(body_names):
        lname = name.lower()
        if lname in {"floor", "ground", "room_floor"}:
            selected.append(name)
            continue
        if lname.endswith("_floor") or lname.startswith("floor_"):
            selected.append(name)
            continue
    return selected


def get_body_poses_from_current_sim(pose_extractor):
    poses = {}
    sim_data = pose_extractor.env.sim.data
    for body_name, body_idx in pose_extractor.body_name_to_idx.items():
        pos = sim_data.body_xpos[body_idx].copy()
        quat = sim_data.body_xquat[body_idx].copy()  # wxyz
        poses[body_name] = (pos, quat)
    return poses


def get_geom_name(model, geom_id):
    try:
        adr = int(model.name_geomadr[geom_id])
    except Exception:
        return ""
    if adr <= 0:
        return ""
    try:
        return model.names[adr:].split(b"\x00")[0].decode("utf-8")
    except Exception:
        return ""


def is_table_leg_geom(geom_name):
    lname = geom_name.lower()
    return "leg" in lname and "table" in lname


def sample_points_from_plane(size, num_points, scale_xy=1.0, max_half_xy=None):
    if num_points <= 0:
        return None
    sx = float(size[0]) if len(size) > 0 else 1.0
    sy = float(size[1]) if len(size) > 1 else sx
    if sx <= 1e-6:
        sx = 1.0
    if sy <= 1e-6:
        sy = 1.0

    scale_xy = float(max(scale_xy, 1e-4))
    sx *= scale_xy
    sy *= scale_xy

    if max_half_xy is not None and float(max_half_xy) > 0:
        cap = float(max_half_xy)
        sx = min(sx, cap)
        sy = min(sy, cap)

    x = np.random.uniform(-sx, sx, size=(num_points, 1))
    y = np.random.uniform(-sy, sy, size=(num_points, 1))
    z = np.zeros((num_points, 1), dtype=np.float32)
    return np.concatenate([x, y, z], axis=1).astype(np.float32)


def build_table_points_from_model(
    hdf5_path,
    episode_key,
    voxel_size,
    surface_max_samples,
    remove_table_legs=True,
    table_top_z_band=0.06,
):
    bddl_path = find_bddl_from_hdf5(hdf5_path)
    if not bddl_path:
        print("  Warning: Cannot resolve BDDL for scene; skip table point cloud")
        return None

    pose_extractor = LiberoPoseExtractor()
    if not pose_extractor.initialize(bddl_path):
        print("  Warning: Cannot initialize scene model; skip table point cloud")
        return None

    try:
        first_state = None
        with h5py.File(hdf5_path, "r") as f:
            state_path = f"data/{episode_key}/states"
            if state_path in f and len(f[state_path]) > 0:
                first_state = np.asarray(f[state_path][0])

        if first_state is not None:
            poses = pose_extractor.extract_poses(first_state)
        else:
            poses = get_body_poses_from_current_sim(pose_extractor)

        model = pose_extractor.model
        body_names = list(pose_extractor.body_name_to_idx.keys())
        table_bodies = select_table_bodies(body_names)
        if not table_bodies:
            print("  Info: No table-like bodies found in scene")
            return None

        table_geom_refs = []
        skipped_leg_geoms = 0
        for body_name in table_bodies:
            body_idx = pose_extractor.body_name_to_idx[body_name]
            geom_ids = np.where(np.asarray(model.geom_bodyid) == body_idx)[0].tolist()
            if not geom_ids:
                continue

            visual_ids = [gid for gid in geom_ids if int(model.geom_group[gid]) == 1]
            active_ids = visual_ids if visual_ids else geom_ids
            for gid in active_ids:
                geom_name = get_geom_name(model, gid)
                if remove_table_legs and is_table_leg_geom(geom_name):
                    skipped_leg_geoms += 1
                    continue
                table_geom_refs.append((body_name, gid, geom_name))

        if not table_geom_refs:
            print("  Info: Table bodies found but no valid geoms for sampling")
            return None

        sampled_world = []

        for body_name, gid, _ in table_geom_refs:
            gtype = int(model.geom_type[gid]) if hasattr(model, "geom_type") else -1
            pts = None

            if gtype == 7 and hasattr(model, "geom_dataid"):
                mesh_id = int(model.geom_dataid[gid])
                if mesh_id >= 0:
                    mesh = _mesh_from_mujoco_model(model, mesh_id)
                    if mesh is not None:
                        n = estimate_surface_sample_count(
                            mesh.area, voxel_size, max_samples=surface_max_samples
                        )
                        pts = sample_points_from_mesh(mesh, n)
            elif gtype in MJ_GEOM_TYPE_TO_PRIMITIVE:
                size = np.asarray(model.geom_size[gid], dtype=np.float64)
                primitive = MJ_GEOM_TYPE_TO_PRIMITIVE[gtype]
                area = primitive_surface_area(primitive, size)
                n = estimate_surface_sample_count(
                    area, voxel_size, max_samples=surface_max_samples
                )
                pts = sample_points_from_primitive(primitive, size, n)

            if pts is None:
                continue

            geom_pos = np.asarray(model.geom_pos[gid], dtype=np.float64)
            geom_quat = np.asarray(model.geom_quat[gid], dtype=np.float64)
            pts_body = transform_point_cloud(pts, geom_pos, geom_quat)

            if body_name not in poses:
                continue
            body_pos, body_quat = poses[body_name]
            pts_world = transform_point_cloud(pts_body, body_pos, body_quat)
            sampled_world.append(pts_world)

        if not sampled_world:
            print("  Info: Table geoms sampled but produced no points")
            return None

        table_world = np.vstack(sampled_world).astype(np.float32)

        # Fallback for monolithic table meshes: keep only top slab points by z-band.
        if remove_table_legs and table_world.shape[0] > 0:
            z_max = float(table_world[:, 2].max())
            z_thresh = z_max - max(float(table_top_z_band), 1e-4)
            top_mask = table_world[:, 2] >= z_thresh
            top_points = table_world[top_mask]
            min_keep = 64
            if top_points.shape[0] >= min_keep:
                table_world = top_points

        table_world = voxel_downsample_points(table_world, voxel_size)
        print(
            f"  Added table points from bodies={table_bodies}, "
            f"geoms={len(table_geom_refs)}, skip_legs={skipped_leg_geoms}, "
            f"points={table_world.shape[0]}, voxel={voxel_size}"
        )
        return table_world
    finally:
        pose_extractor.close()



def build_floor_points_from_model(
    hdf5_path,
    episode_key,
    voxel_size,
    surface_max_samples,
    floor_scale=0.35,
    floor_max_xy=1.2,
):
    bddl_path = find_bddl_from_hdf5(hdf5_path)
    if not bddl_path:
        print("  Warning: Cannot resolve BDDL for scene; skip floor point cloud")
        return None

    pose_extractor = LiberoPoseExtractor()
    if not pose_extractor.initialize(bddl_path):
        print("  Warning: Cannot initialize scene model; skip floor point cloud")
        return None

    try:
        first_state = None
        with h5py.File(hdf5_path, "r") as f:
            state_path = f"data/{episode_key}/states"
            if state_path in f and len(f[state_path]) > 0:
                first_state = np.asarray(f[state_path][0])

        if first_state is not None:
            poses = pose_extractor.extract_poses(first_state)
        else:
            poses = get_body_poses_from_current_sim(pose_extractor)

        model = pose_extractor.model
        body_names = list(pose_extractor.body_name_to_idx.keys())
        floor_bodies = select_floor_bodies(body_names)
        if not floor_bodies:
            print("  Info: No floor-like bodies found in scene")
            return None

        floor_geom_refs = []
        for body_name in floor_bodies:
            body_idx = pose_extractor.body_name_to_idx[body_name]
            geom_ids = np.where(np.asarray(model.geom_bodyid) == body_idx)[0].tolist()
            if not geom_ids:
                continue

            visual_ids = [gid for gid in geom_ids if int(model.geom_group[gid]) == 1]
            active_ids = visual_ids if visual_ids else geom_ids
            for gid in active_ids:
                floor_geom_refs.append((body_name, gid, get_geom_name(model, gid)))

        if not floor_geom_refs:
            print("  Info: Floor bodies found but no valid geoms for sampling")
            return None

        sampled_world = []

        for body_name, gid, _ in floor_geom_refs:
            gtype = int(model.geom_type[gid]) if hasattr(model, "geom_type") else -1
            pts = None

            if gtype == 0:
                size = np.asarray(model.geom_size[gid], dtype=np.float64)
                sx = float(size[0]) if len(size) > 0 else 1.0
                sy = float(size[1]) if len(size) > 1 else sx
                sx = max(sx, 1e-6) * max(float(floor_scale), 1e-4)
                sy = max(sy, 1e-6) * max(float(floor_scale), 1e-4)
                if floor_max_xy is not None and float(floor_max_xy) > 0:
                    sx = min(sx, float(floor_max_xy))
                    sy = min(sy, float(floor_max_xy))
                plane_area = (2.0 * sx) * (2.0 * sy)
                n = estimate_surface_sample_count(
                    plane_area, voxel_size, max_samples=surface_max_samples
                )
                pts = sample_points_from_plane(
                    size,
                    n,
                    scale_xy=floor_scale,
                    max_half_xy=floor_max_xy,
                )
            elif gtype == 7 and hasattr(model, "geom_dataid"):
                mesh_id = int(model.geom_dataid[gid])
                if mesh_id >= 0:
                    mesh = _mesh_from_mujoco_model(model, mesh_id)
                    if mesh is not None:
                        n = estimate_surface_sample_count(
                            mesh.area, voxel_size, max_samples=surface_max_samples
                        )
                        pts = sample_points_from_mesh(mesh, n)
            elif gtype in MJ_GEOM_TYPE_TO_PRIMITIVE:
                size = np.asarray(model.geom_size[gid], dtype=np.float64)
                primitive = MJ_GEOM_TYPE_TO_PRIMITIVE[gtype]
                area = primitive_surface_area(primitive, size)
                n = estimate_surface_sample_count(
                    area, voxel_size, max_samples=surface_max_samples
                )
                pts = sample_points_from_primitive(primitive, size, n)

            if pts is None:
                continue

            geom_pos = np.asarray(model.geom_pos[gid], dtype=np.float64)
            geom_quat = np.asarray(model.geom_quat[gid], dtype=np.float64)
            pts_body = transform_point_cloud(pts, geom_pos, geom_quat)

            if body_name not in poses:
                continue
            body_pos, body_quat = poses[body_name]
            pts_world = transform_point_cloud(pts_body, body_pos, body_quat)
            sampled_world.append(pts_world)

        if not sampled_world:
            print("  Info: Floor geoms sampled but produced no points")
            return None

        floor_world = np.vstack(sampled_world).astype(np.float32)
        floor_world = voxel_downsample_points(floor_world, voxel_size)
        print(
            f"  Added floor points from bodies={floor_bodies}, "
            f"geoms={len(floor_geom_refs)}, points={floor_world.shape[0]}, "
            f"floor_scale={floor_scale}, floor_max_xy={floor_max_xy}, voxel={voxel_size}"
        )
        return floor_world
    finally:
        pose_extractor.close()


def merge_scene_point_clouds(
    hdf5_path,
    episode_key,
    drop_zero_points=False,
    table_world=None,
    floor_world=None,
    voxel_size=DEFAULT_VOXEL_SIZE,
):
    with h5py.File(hdf5_path, "r") as f:
        if "data" not in f:
            raise KeyError("Invalid dataset: missing 'data' group")

        ep_path = f"data/{episode_key}"
        if ep_path not in f:
            raise KeyError(f"Episode '{episode_key}' not found in dataset")

        ep_group = f[ep_path]
        if "object_pcds" not in ep_group:
            raise KeyError(f"Episode '{episode_key}' does not contain 'object_pcds'")

        pcd_group = ep_group["object_pcds"]
        object_names = sorted(list(pcd_group.keys()))
        if not object_names:
            raise ValueError(f"Episode '{episode_key}' has empty 'object_pcds'")

        object_trajs = {name: np.asarray(pcd_group[name]) for name in object_names}

    num_frames = None
    for name, arr in object_trajs.items():
        if arr.ndim != 3 or arr.shape[-1] != 3:
            raise ValueError(f"object_pcds/{name} has invalid shape {arr.shape}, expected (T, N, 3)")
        if num_frames is None:
            num_frames = arr.shape[0]
        elif arr.shape[0] != num_frames:
            raise ValueError(
                f"Trajectory length mismatch: '{name}' has {arr.shape[0]} frames, expected {num_frames}"
            )

    merged_frames = []
    frame_counts = np.zeros((num_frames,), dtype=np.int32)

    for t in range(num_frames):
        parts = []
        for name in sorted(object_trajs.keys()):
            pts = object_trajs[name][t].astype(np.float32, copy=False)
            if drop_zero_points and pts.shape[0] > 0:
                valid_mask = ~np.all(np.isclose(pts, 0.0), axis=1)
                pts = pts[valid_mask]
            parts.append(pts)

        if table_world is not None:
            parts.append(table_world)
        if floor_world is not None:
            parts.append(floor_world)

        merged = np.concatenate(parts, axis=0) if parts else np.zeros((0, 3), dtype=np.float32)
        if voxel_size is not None and voxel_size > 0 and merged.shape[0] > 0:
            merged = voxel_downsample_points(merged, voxel_size)
        merged_frames.append(merged)
        frame_counts[t] = merged.shape[0]

    max_points = int(frame_counts.max()) if num_frames > 0 else 0
    scene_pcd = np.zeros((num_frames, max_points, 3), dtype=np.float32)
    for t, pts in enumerate(merged_frames):
        n = pts.shape[0]
        if n > 0:
            scene_pcd[t, :n] = pts

    out_names = sorted(object_trajs.keys())
    if table_world is not None:
        out_names.append(TABLE_ENTITY_NAME)
    if floor_world is not None:
        out_names.append(FLOOR_ENTITY_NAME)

    return scene_pcd, frame_counts, out_names


def save_scene_npz(output_path, scene_pcd, frame_counts, episode, object_names, source_hdf5):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        scene_pcd=scene_pcd,
        frame_points=frame_counts,
        episode=np.array(episode),
        object_names=np.array(object_names),
        source_hdf5=np.array(str(source_hdf5)),
    )


def build_output_path(input_hdf5, dataset_root, output_dir, episode_key):
    input_path = Path(input_hdf5)
    if dataset_root is not None:
        rel = input_path.relative_to(dataset_root)
        return Path(output_dir) / rel.parent / f"{rel.stem}_{episode_key}_scene.npz"
    return Path(output_dir) / f"{input_path.stem}_{episode_key}_scene.npz"


def select_episode_keys(hdf5_path, episodes):
    with h5py.File(hdf5_path, "r") as f:
        if "data" not in f:
            return []
        keys = sorted_demo_keys(f["data"])

    if episodes <= 0:
        return keys
    return keys[: min(len(keys), episodes)]


def process_hdf5_file(
    hdf5_path,
    output_dir,
    dataset_root,
    episodes,
    drop_zero_points,
    include_table,
    include_floor,
    floor_scale,
    floor_max_xy,
    remove_table_legs,
    table_top_z_band,
    voxel_size,
    surface_max_samples,
    overwrite,
):
    episode_keys = select_episode_keys(hdf5_path, episodes)
    if not episode_keys:
        print(f"Skip (no valid trajectories): {hdf5_path}")
        return 0

    table_world = None
    if include_table:
        table_world = build_table_points_from_model(
            hdf5_path,
            episode_keys[0],
            voxel_size,
            surface_max_samples,
            remove_table_legs=remove_table_legs,
            table_top_z_band=table_top_z_band,
        )

    floor_world = None
    if include_floor and table_world is None:
        floor_world = build_floor_points_from_model(
            hdf5_path,
            episode_keys[0],
            voxel_size,
            surface_max_samples,
            floor_scale=floor_scale,
            floor_max_xy=floor_max_xy,
        )

    saved = 0
    for ep_key in episode_keys:
        out_path = build_output_path(hdf5_path, dataset_root, output_dir, ep_key)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.exists() and not overwrite:
            print(f"  Skip existing: {out_path}")
            continue

        scene_pcd, frame_counts, object_names = merge_scene_point_clouds(
            hdf5_path,
            episode_key=ep_key,
            drop_zero_points=drop_zero_points,
            table_world=table_world,
            floor_world=floor_world,
            voxel_size=voxel_size,
        )

        save_scene_npz(
            output_path=out_path,
            scene_pcd=scene_pcd,
            frame_counts=frame_counts,
            episode=ep_key,
            object_names=object_names,
            source_hdf5=hdf5_path,
        )

        print(
            f"  Saved {out_path.name}: frames={scene_pcd.shape[0]}, "
            f"max_points={scene_pcd.shape[1]}, objects={len(object_names)}, voxel={voxel_size}, "
            f"frame_points[min/mean/max]={int(frame_counts.min())}/"
            f"{float(frame_counts.mean()):.1f}/{int(frame_counts.max())}"
        )
        saved += 1

    return saved


def process_dataset(
    dataset_path,
    output_dir,
    episodes,
    drop_zero_points,
    include_table,
    include_floor,
    floor_scale,
    floor_max_xy,
    remove_table_legs,
    table_top_z_band,
    voxel_size,
    surface_max_samples,
    overwrite,
):
    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if dataset_path.is_file():
        if dataset_path.suffix.lower() != ".hdf5":
            raise ValueError(f"Input file is not .hdf5: {dataset_path}")
        print(f"Processing single file: {dataset_path}")
        return process_hdf5_file(
            hdf5_path=str(dataset_path),
            output_dir=output_dir,
            dataset_root=None,
            episodes=episodes,
            drop_zero_points=drop_zero_points,
            include_table=include_table,
            include_floor=include_floor,
            floor_scale=floor_scale,
            floor_max_xy=floor_max_xy,
            remove_table_legs=remove_table_legs,
            table_top_z_band=table_top_z_band,
            voxel_size=voxel_size,
            surface_max_samples=surface_max_samples,
            overwrite=overwrite,
        )

    if dataset_path.is_dir():
        hdf5_files = sorted(dataset_path.rglob("*.hdf5"))
        print(f"Found {len(hdf5_files)} HDF5 files under: {dataset_path}")
        total_saved = 0
        for p in hdf5_files:
            print(f"\nProcessing: {p}")
            try:
                total_saved += process_hdf5_file(
                    hdf5_path=str(p),
                    output_dir=output_dir,
                    dataset_root=dataset_path,
                    episodes=episodes,
                    drop_zero_points=drop_zero_points,
                    include_table=include_table,
                    include_floor=include_floor,
                    floor_scale=floor_scale,
                    floor_max_xy=floor_max_xy,
                    remove_table_legs=remove_table_legs,
                    table_top_z_band=table_top_z_band,
                    voxel_size=voxel_size,
                    surface_max_samples=surface_max_samples,
                    overwrite=overwrite,
                )
            except Exception as e:
                print(f"  Error: {e}")
        return total_saved

    raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")


def main():
    parser = argparse.ArgumentParser(description="Merge object point clouds into scene point clouds")
    parser.add_argument("--dataset", type=str, required=True, help="Path to HDF5 file or directory")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for scene NPZ files")
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="For each HDF5, process first N episodes (<=0 means all)",
    )
    parser.add_argument(
        "--drop-zero-points",
        action="store_true",
        help="Drop all-zero padded points before merging each frame",
    )
    parser.add_argument(
        "--no-table",
        action="store_true",
        help="Disable adding scene table / countertop points",
    )
    parser.add_argument(
        "--no-floor",
        action="store_true",
        help="Disable adding floor points when table is absent",
    )
    parser.add_argument(
        "--floor-scale",
        type=float,
        default=0.35,
        help="Scale factor for floor XY sampling range (smaller -> less floor area)",
    )
    parser.add_argument(
        "--floor-max-xy",
        type=float,
        default=0.5,
        help="Max floor half-size in XY (meters) after scaling; <=0 disables cap",
    )
    parser.add_argument(
        "--keep-table-legs",
        action="store_true",
        help="Keep table legs (default removes legs and keeps tabletop)",
    )
    parser.add_argument(
        "--table-top-z-band",
        type=float,
        default=0.02,
        help="Top z-band (meters) kept when removing table legs from monolithic meshes",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=DEFAULT_VOXEL_SIZE,
        help="Voxel size in meters for scene/object/table/floor voxel sampling",
    )
    parser.add_argument(
        "--surface-max-samples",
        type=int,
        default=DEFAULT_MAX_SAMPLES_PER_GEOM,
        help="Max pre-voxel samples per table/floor geom (higher = more sensitive for small voxel)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output npz files")

    args = parser.parse_args()
    if args.voxel_size <= 0:
        raise ValueError("--voxel-size must be > 0")
    if args.surface_max_samples <= 0:
        raise ValueError("--surface-max-samples must be > 0")

    total = process_dataset(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        episodes=args.episodes,
        drop_zero_points=args.drop_zero_points,
        include_table=not args.no_table,
        include_floor=not args.no_floor,
        floor_scale=args.floor_scale,
        floor_max_xy=args.floor_max_xy,
        remove_table_legs=not args.keep_table_legs,
        table_top_z_band=args.table_top_z_band,
        voxel_size=args.voxel_size,
        surface_max_samples=args.surface_max_samples,
        overwrite=args.overwrite,
    )

    print(f"\nDone. Saved {total} scene trajectories to: {args.output_dir}")


if __name__ == "__main__":
    main()
