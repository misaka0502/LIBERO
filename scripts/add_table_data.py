import argparse
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]


def decode_h5_text(x: object) -> str:
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="ignore")
    if isinstance(x, np.bytes_):
        return bytes(x).decode("utf-8", errors="ignore")
    return str(x)


def discover_hdf5_files(dataset_path: Path) -> List[Path]:
    if dataset_path.is_file():
        return [dataset_path]
    return sorted(p for p in dataset_path.rglob("*.hdf5") if p.is_file())


def parse_xyz(text: Optional[str], default: Sequence[float] = (0.0, 0.0, 0.0)) -> np.ndarray:
    if not text:
        return np.asarray(default, dtype=np.float64)
    vals = [float(v) for v in str(text).split()]
    if len(vals) != 3:
        return np.asarray(default, dtype=np.float64)
    return np.asarray(vals, dtype=np.float64)


def parse_quat_wxyz(text: Optional[str]) -> np.ndarray:
    if not text:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    vals = [float(v) for v in str(text).split()]
    if len(vals) != 4:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    q = np.asarray(vals, dtype=np.float64)
    n = np.linalg.norm(q)
    if n <= 1e-12:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / n


def quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    out = np.asarray(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )
    n = np.linalg.norm(out)
    if n <= 1e-12:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return out / n


def quat_to_rotmat_wxyz(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def transform_point(point: np.ndarray, pos: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    return quat_to_rotmat_wxyz(quat_wxyz) @ point + pos


def geom_top_center(
    geom_type: str,
    size_text: Optional[str],
    geom_world_pos: np.ndarray,
    geom_world_quat: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    geom_type = str(geom_type or "").lower()
    size_vals = [float(v) for v in str(size_text or "").split()] if size_text else []
    if geom_type == "plane":
        return geom_world_pos.copy(), geom_world_quat.copy()
    if geom_type == "box" and len(size_vals) >= 3:
        hx, hy, hz = float(size_vals[0]), float(size_vals[1]), float(size_vals[2])
        R = quat_to_rotmat_wxyz(geom_world_quat)
        world_z_half = abs(R[2, 0]) * hx + abs(R[2, 1]) * hy + abs(R[2, 2]) * hz
        top_pos = np.asarray(
            [geom_world_pos[0], geom_world_pos[1], geom_world_pos[2] + world_z_half],
            dtype=np.float64,
        )
        return top_pos, geom_world_quat.copy()
    if geom_type == "cylinder" and len(size_vals) >= 2:
        local_top = np.asarray([0.0, 0.0, float(size_vals[1])], dtype=np.float64)
        return transform_point(local_top, geom_world_pos, geom_world_quat), geom_world_quat.copy()
    return geom_world_pos.copy(), geom_world_quat.copy()


def score_scene_surface_geom(name: str, material: str) -> int:
    name_l = (name or "").lower()
    mat_l = (material or "").lower()
    if "table_collision" in name_l:
        return 100
    if "table_visual" in name_l:
        return 90
    if "table" in name_l:
        return 80
    if mat_l == "table_texture":
        return 70
    if name_l == "floor":
        return 40
    if mat_l == "floorplane":
        return 30
    if any(k in name_l for k in ("ground", "workspace", "arena_floor")):
        return 20
    return -1


def infer_table_pose_from_model_xml(model_xml: str) -> Optional[Tuple[np.ndarray, np.ndarray, str]]:
    if not model_xml:
        return None
    root = ET.fromstring(model_xml)
    best: Optional[Tuple[int, np.ndarray, np.ndarray, str]] = None

    def visit_body(body_elem: ET.Element, parent_pos: np.ndarray, parent_quat: np.ndarray) -> None:
        nonlocal best
        body_name_l = (body_elem.get("name") or "").lower()
        body_pos = parse_xyz(body_elem.get("pos"))
        body_quat = parse_quat_wxyz(body_elem.get("quat"))
        world_quat = quat_mul_wxyz(parent_quat, body_quat)
        world_pos = transform_point(body_pos, parent_pos, parent_quat)

        for geom in body_elem.findall("geom"):
            score = score_scene_surface_geom(geom.get("name", ""), geom.get("material", ""))
            # Scenes like LIVING_ROOM / STUDY have table bodies whose collision box
            # geoms carry no name or material.  Score them the same way as a named
            # table geom so they beat the floor fallback.
            inferred_from_body = False
            if score < 0 and geom.get("type", "").lower() == "box":
                if any(k in body_name_l for k in ("table", "desk")):
                    score = 65
                    inferred_from_body = True
            if score < 0:
                continue
            geom_pos = parse_xyz(geom.get("pos"))
            geom_quat = parse_quat_wxyz(geom.get("quat"))
            geom_world_quat = quat_mul_wxyz(world_quat, geom_quat)
            geom_world_pos = transform_point(geom_pos, world_pos, world_quat)
            top_center, top_quat = geom_top_center(
                geom_type=geom.get("type", "box"),
                size_text=geom.get("size"),
                geom_world_pos=geom_world_pos,
                geom_world_quat=geom_world_quat,
            )
            # For geoms inferred via body name the geom itself may be arbitrarily
            # rotated, but the table surface is always horizontal.  Force identity
            # so the pose orientation matches every other correctly-handled scene.
            if inferred_from_body:
                top_quat = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
            label = geom.get("name") or geom.get("material") or "scene_surface"
            cand = (score, top_center, top_quat, label)
            if best is None or score > best[0]:
                best = cand

        for child in body_elem.findall("body"):
            visit_body(child, world_pos, world_quat)

    worldbody = root.find("worldbody")
    if worldbody is None:
        return None

    identity_quat = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    zero = np.asarray([0.0, 0.0, 0.0], dtype=np.float64)

    for geom in worldbody.findall("geom"):
        score = score_scene_surface_geom(geom.get("name", ""), geom.get("material", ""))
        if score < 0:
            continue
        geom_pos = parse_xyz(geom.get("pos"))
        top_center, top_quat = geom_top_center(
            geom_type=geom.get("type", "box"),
            size_text=geom.get("size"),
            geom_world_pos=geom_pos,
            geom_world_quat=identity_quat,
        )
        label = geom.get("name") or geom.get("material") or "scene_surface"
        cand = (score, top_center, top_quat, label)
        if best is None or score > best[0]:
            best = cand

    for body in worldbody.findall("body"):
        visit_body(body, zero, identity_quat)

    if best is None:
        return None
    _, center, quat, label = best
    return center.astype(np.float32), quat.astype(np.float32), label


def ensure_pose_dataset(group: h5py.Group, entity_name: str, pose_seq: np.ndarray, overwrite: bool) -> bool:
    if entity_name in group:
        if not overwrite:
            return False
        del group[entity_name]
    group.create_dataset(entity_name, data=pose_seq.astype(np.float32), compression="gzip")
    return True


def resolve_npz_path(path: Optional[str]) -> Optional[Path]:
    if not path:
        return None
    p = Path(path)
    if p.is_absolute() and p.is_file():
        return p
    candidates = [
        REPO_ROOT / path,
        REPO_ROOT / "SDF" / path,
        REPO_ROOT / "SDF/data" / Path(path).name,
        REPO_ROOT / "SDF/data/Geom" / Path(path).name,
    ]
    for cand in candidates:
        if cand.is_file():
            return cand.resolve()
    return None


def load_latent_object_bounds(
    latent_ckpt_path: Optional[str],
) -> Tuple[
    Dict[int, Tuple[float, float, float, float, float, float]],
    Dict[str, int],
    Dict[int, Optional[Path]],
]:
    bounds_by_id: Dict[int, Tuple[float, float, float, float, float, float]] = {}
    object_id_by_npz_name: Dict[str, int] = {}
    npz_path_by_id: Dict[int, Optional[Path]] = {}
    if not latent_ckpt_path:
        return bounds_by_id, object_id_by_npz_name, npz_path_by_id
    ckpt = torch.load(str(latent_ckpt_path), map_location="cpu")
    object_meta = ckpt.get("object_meta")
    if not isinstance(object_meta, list):
        return bounds_by_id, object_id_by_npz_name, npz_path_by_id
    for i, item in enumerate(object_meta):
        if not isinstance(item, dict):
            continue
        object_id = int(item.get("object_id", i))
        bounds = item.get("bounds")
        npz_path = item.get("npz_path")
        if isinstance(bounds, (list, tuple)) and len(bounds) == 6:
            bounds_by_id[object_id] = tuple(float(v) for v in bounds)
        if isinstance(npz_path, str) and npz_path:
            object_id_by_npz_name[os.path.basename(npz_path.replace("\\", "/"))] = object_id
            npz_path_by_id[object_id] = resolve_npz_path(npz_path)
    return bounds_by_id, object_id_by_npz_name, npz_path_by_id


def infer_local_top_center_from_bounds(
    bounds: Optional[Tuple[float, float, float, float, float, float]],
) -> np.ndarray:
    if bounds is None:
        return np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
    xmin, xmax, ymin, ymax, _zmin, zmax = bounds
    return np.asarray(
        [
            0.5 * (float(xmin) + float(xmax)),
            0.5 * (float(ymin) + float(ymax)),
            float(zmax),
        ],
        dtype=np.float32,
    )


def infer_local_top_center_from_npz(npz_path: Optional[Path]) -> Optional[np.ndarray]:
    if npz_path is None or (not npz_path.is_file()):
        return None
    try:
        data = np.load(str(npz_path), allow_pickle=True)
        if "surface_points" not in data:
            return None
        sp = np.asarray(data["surface_points"], dtype=np.float32)
        if sp.ndim != 2 or sp.shape[1] != 3 or sp.shape[0] == 0:
            return None
        z_top = float(sp[:, 2].max())
        tol = max(1e-4, 1e-3 * max(1.0, abs(z_top)))
        top_mask = sp[:, 2] >= (z_top - tol)
        top_pts = sp[top_mask] if np.any(top_mask) else sp[np.argmax(sp[:, 2]) : np.argmax(sp[:, 2]) + 1]
        center_xy = top_pts[:, :2].mean(axis=0)
        return np.asarray([float(center_xy[0]), float(center_xy[1]), z_top], dtype=np.float32)
    except Exception:
        return None


def infer_centered_local_top_center_from_npz(npz_path: Optional[Path]) -> Optional[np.ndarray]:
    if npz_path is None or (not npz_path.is_file()):
        return None
    try:
        data = np.load(str(npz_path), allow_pickle=True)
        if "surface_points" not in data:
            return None
        sp = np.asarray(data["surface_points"], dtype=np.float32)
        if sp.ndim != 2 or sp.shape[1] != 3 or sp.shape[0] == 0:
            return None
        raw_top = infer_local_top_center_from_npz(npz_path)
        if raw_top is None:
            return None
        center = sp.mean(axis=0)
        return raw_top - center.astype(np.float32)
    except Exception:
        return None


def update_mapping_group(
    mapping_group: h5py.Group,
    *,
    entity_name: str,
    target_npz_name: str,
    target_object_id: int,
    category_name: str,
    overwrite: bool,
) -> bool:
    keys = list(mapping_group.keys())
    if not keys:
        raise RuntimeError("meta/latent_mapping is empty; cannot infer schema")

    entity_names = [decode_h5_text(x) for x in np.asarray(mapping_group["entity_names"])]
    existing_idx = entity_names.index(entity_name) if entity_name in entity_names else None
    if existing_idx is not None and not overwrite:
        return False

    string_fill = {
        "alignment_mode": "table_surface_pose",
        "body_names": entity_name,
        "categories": category_name,
        "entity_names": entity_name,
        "instances": entity_name,
        "match_kind": "manual_scene_surface",
        "npz_mapping_mode": "manual_scene_surface",
        "parts": "",
        "pose_source_names": entity_name,
        "target_npz_names": target_npz_name,
    }
    numeric_fill = {
        "npz_to_body_rot_wxyz": np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "npz_to_body_trans_local": np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        "pose_align_error": np.float32(0.0),
        "pose_offset_local": np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        "pose_rot_offset_wxyz": np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "target_object_ids": np.int32(target_object_id),
    }

    for key in keys:
        old = np.asarray(mapping_group[key])
        if existing_idx is None:
            if old.dtype.kind in ("O", "S", "U"):
                new = np.concatenate([old.astype(object), np.asarray([string_fill[key]], dtype=object)], axis=0)
            else:
                fill = numeric_fill[key]
                fill_arr = np.asarray(fill, dtype=old.dtype)
                if old.ndim == 1:
                    new = np.concatenate([old, fill_arr.reshape(1)], axis=0)
                else:
                    new = np.concatenate([old, fill_arr.reshape(1, *old.shape[1:])], axis=0)
        else:
            new = old.copy()
            if old.dtype.kind in ("O", "S", "U"):
                new = new.astype(object)
                new[existing_idx] = string_fill[key]
            else:
                fill = np.asarray(numeric_fill[key], dtype=old.dtype)
                if old.ndim == 1:
                    new[existing_idx] = fill.reshape(()).item() if fill.ndim == 0 else fill.reshape(-1)[0]
                else:
                    new[existing_idx] = fill.reshape(old.shape[1:])
        del mapping_group[key]
        if new.dtype.kind in ("O", "U"):
            dt = h5py.string_dtype(encoding="utf-8")
            mapping_group.create_dataset(key, data=new.astype(dt), dtype=dt)
        else:
            mapping_group.create_dataset(key, data=new)
    return True


def update_motion_labels(demo_group: h5py.Group, entity_name: str, overwrite: bool) -> None:
    motion_group = demo_group.get("object_motion_labels")
    if not isinstance(motion_group, h5py.Group):
        return
    if entity_name in motion_group:
        if not overwrite:
            return
        del motion_group[entity_name]
    motion_group.create_dataset(entity_name, data=np.asarray(False, dtype=np.bool_))


def process_hdf5_file(
    path: Path,
    *,
    entity_name: str,
    target_npz_name: str,
    target_object_id: int,
    local_top_center_raw: np.ndarray,
    local_top_center_centered: np.ndarray,
    overwrite: bool,
    dry_run: bool,
) -> Dict[str, int]:
    stats = {"files": 1, "demos": 0, "pose_groups_written": 0, "mapping_updated": 0, "motion_labels_updated": 0}
    mode = "r" if dry_run else "r+"
    with h5py.File(path, mode) as f:
        if "data" not in f or "meta/latent_mapping" not in f:
            return stats
        data_group = f["data"]
        mapping_group = f["meta/latent_mapping"]

        mapping_done = False
        for demo_name in sorted(data_group.keys()):
            demo_group = data_group[demo_name]
            stats["demos"] += 1
            model_xml = demo_group.attrs.get("model_file")
            inferred = infer_table_pose_from_model_xml(model_xml)
            if inferred is None:
                continue
            top_center, top_quat, surface_name = inferred

            pose_groups: Dict[str, h5py.Group] = {}
            for pose_group_name in ("object_poses_npz", "object_poses_aligned", "object_poses"):
                group = demo_group.get(pose_group_name)
                if isinstance(group, h5py.Group):
                    pose_groups[pose_group_name] = group
            if not pose_groups:
                continue

            num_steps = None
            for group in pose_groups.values():
                for ds_name in group.keys():
                    ds = group[ds_name]
                    if getattr(ds, "ndim", 0) == 2 and ds.shape[1] >= 7:
                        num_steps = int(ds.shape[0])
                        break
                if num_steps is not None:
                    break
            if num_steps is None or num_steps <= 0:
                continue

            rotmat = quat_to_rotmat_wxyz(top_quat.astype(np.float64))
            origin_world_raw = top_center.astype(np.float32) - (
                rotmat @ local_top_center_raw.astype(np.float64)
            ).astype(np.float32)
            origin_world_centered = top_center.astype(np.float32) - (
                rotmat @ local_top_center_centered.astype(np.float64)
            ).astype(np.float32)
            pose7_raw = np.concatenate([origin_world_raw.reshape(3), top_quat.reshape(4)], axis=0).astype(np.float32)
            pose7_centered = np.concatenate([origin_world_centered.reshape(3), top_quat.reshape(4)], axis=0).astype(
                np.float32
            )
            pose_seq_raw = np.repeat(pose7_raw[None, :], num_steps, axis=0)
            pose_seq_centered = np.repeat(pose7_centered[None, :], num_steps, axis=0)

            if not dry_run:
                has_npz_group = "object_poses_npz" in pose_groups
                for pose_group_name, group in pose_groups.items():
                    if pose_group_name == "object_poses_npz":
                        pose_seq = pose_seq_centered
                    elif pose_group_name == "object_poses_aligned":
                        pose_seq = pose_seq_centered
                    elif pose_group_name == "object_poses" and (not has_npz_group):
                        # Legacy v3-style datasets use object_poses_aligned and may
                        # fall back to object_poses; keep both on the same centered frame.
                        pose_seq = pose_seq_centered
                    else:
                        pose_seq = pose_seq_raw
                    wrote = ensure_pose_dataset(group, entity_name, pose_seq, overwrite=overwrite)
                    stats["pose_groups_written"] += int(wrote)
                update_motion_labels(demo_group, entity_name, overwrite=overwrite)
                if "object_motion_labels" in demo_group:
                    stats["motion_labels_updated"] += 1
            if not mapping_done:
                if not dry_run:
                    wrote_mapping = update_mapping_group(
                        mapping_group,
                        entity_name=entity_name,
                        target_npz_name=target_npz_name,
                        target_object_id=target_object_id,
                        category_name="table",
                        overwrite=overwrite,
                    )
                    stats["mapping_updated"] += int(wrote_mapping)
                mapping_done = True

            print(
                f"[INFO] {path.name}:{demo_name} surface={surface_name} "
                f"top=({top_center[0]:.4f}, {top_center[1]:.4f}, {top_center[2]:.4f}) "
                f"raw_top=({local_top_center_raw[0]:.4f}, {local_top_center_raw[1]:.4f}, {local_top_center_raw[2]:.4f}) "
                f"centered_top=({local_top_center_centered[0]:.4f}, {local_top_center_centered[1]:.4f}, {local_top_center_centered[2]:.4f}) "
                f"pose_raw=({pose7_raw[0]:.4f}, {pose7_raw[1]:.4f}, {pose7_raw[2]:.4f}) "
                f"pose_npz=({pose7_centered[0]:.4f}, {pose7_centered[1]:.4f}, {pose7_centered[2]:.4f})"
            )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Add scene table/ground poses into LIBERO PCD datasets.")
    parser.add_argument("--dataset", required=True, help="Target HDF5 file or dataset directory.")
    parser.add_argument("--entity-name", default="scene_table", help="Entity name to write into object pose groups.")
    parser.add_argument("--target-npz-name", default="table--sdfae_labels.npz", help="Latent-mapping target npz basename.")
    parser.add_argument("--target-object-id", type=int, default=5, help="Object id in latent checkpoint metadata.")
    parser.add_argument(
        "--latent-ckpt",
        default=str(REPO_ROOT / "SDF/output/20260329_141021_sdf_vnad/checkpoints/sdf_vnad_best.pt"),
        help="Optional latent checkpoint to infer the table local top-center from bounds.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing scene table entries.")
    parser.add_argument("--dry-run", action="store_true", help="Inspect only, do not modify files.")
    args = parser.parse_args()

    dataset_path = Path(args.dataset).expanduser().resolve()
    files = discover_hdf5_files(dataset_path)
    if not files:
        raise FileNotFoundError(f"No .hdf5 files found under {dataset_path}")

    bounds_by_id, object_id_by_npz_name, npz_path_by_id = load_latent_object_bounds(args.latent_ckpt)
    target_object_id = int(args.target_object_id)
    target_npz_name = str(args.target_npz_name)
    if target_npz_name in object_id_by_npz_name:
        target_object_id = int(object_id_by_npz_name[target_npz_name])
    local_top_center_raw = infer_local_top_center_from_npz(npz_path_by_id.get(target_object_id))
    if local_top_center_raw is None:
        local_top_center_raw = infer_local_top_center_from_bounds(bounds_by_id.get(target_object_id))
    local_top_center_centered = infer_centered_local_top_center_from_npz(npz_path_by_id.get(target_object_id))
    if local_top_center_centered is None:
        local_top_center_centered = local_top_center_raw.copy()
    print(
        f"[INFO] target_npz={target_npz_name} target_object_id={target_object_id} "
        f"local_top_raw=({local_top_center_raw[0]:.4f}, {local_top_center_raw[1]:.4f}, {local_top_center_raw[2]:.4f}) "
        f"local_top_centered=({local_top_center_centered[0]:.4f}, {local_top_center_centered[1]:.4f}, {local_top_center_centered[2]:.4f})"
    )

    total = {"files": 0, "demos": 0, "pose_groups_written": 0, "mapping_updated": 0, "motion_labels_updated": 0}
    for fp in files:
        stats = process_hdf5_file(
            fp,
            entity_name=str(args.entity_name),
            target_npz_name=target_npz_name,
            target_object_id=target_object_id,
            local_top_center_raw=local_top_center_raw,
            local_top_center_centered=local_top_center_centered,
            overwrite=bool(args.overwrite),
            dry_run=bool(args.dry_run),
        )
        for key, value in stats.items():
            total[key] += int(value)

    print(
        "[DONE] files={files} demos={demos} pose_groups_written={pose_groups_written} "
        "mapping_updated={mapping_updated} motion_labels_updated={motion_labels_updated}".format(**total)
    )


if __name__ == "__main__":
    main()
