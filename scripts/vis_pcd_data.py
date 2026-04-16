"""
Visualize point cloud data from LIBERO dataset HDF5 files.

Usage:
    python vis_pcd_data.py --dataset <hdf5_path>
    python vis_pcd_data.py --dataset <hdf5_path> --frame 10
    python vis_pcd_data.py --dataset <hdf5_path> --verify-distance --draw-nearest-segment
"""

import argparse
from pathlib import Path
import colorsys
import h5py
import numpy as np
import open3d as o3d
import sys

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover
    cKDTree = None


def _decode_h5_strings(values):
    out = []
    for v in np.asarray(values):
        if isinstance(v, bytes):
            out.append(v.decode("utf-8"))
        elif isinstance(v, np.bytes_):
            out.append(v.decode("utf-8"))
        else:
            out.append(str(v))
    return out


def _entity_instance_name(entity_name):
    name = str(entity_name)
    if "__" in name:
        return name.split("__", 1)[0]
    return name


def quat_wxyz_to_rotmat(quat_wxyz):
    q = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        q = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    else:
        q = q / n
    w, x, y, z = q
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def transform_points_wxyz(points, pos, quat_wxyz):
    rot = quat_wxyz_to_rotmat(quat_wxyz)
    pts = np.asarray(points, dtype=np.float64)
    pos = np.asarray(pos, dtype=np.float64).reshape(1, 3)
    return ((rot @ pts.T).T + pos).astype(np.float32, copy=False)


def inverse_transform_points_wxyz(points, pos, quat_wxyz):
    rot = quat_wxyz_to_rotmat(quat_wxyz)
    pts = np.asarray(points, dtype=np.float64)
    pos = np.asarray(pos, dtype=np.float64).reshape(1, 3)
    return ((pts - pos) @ rot).astype(np.float32, copy=False)


def normalize_quaternion_wxyz(quat_wxyz):
    q = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / n


def quat_mul_wxyz(q1_wxyz, q2_wxyz):
    w1, x1, y1, z1 = normalize_quaternion_wxyz(q1_wxyz)
    w2, x2, y2, z2 = normalize_quaternion_wxyz(q2_wxyz)
    out = np.asarray(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )
    return normalize_quaternion_wxyz(out)


def quat_conj_wxyz(quat_wxyz):
    w, x, y, z = normalize_quaternion_wxyz(quat_wxyz)
    return np.asarray([w, -x, -y, -z], dtype=np.float64)


def gripper_pose_quat_offset_wxyz(part_name):
    part_l = str(part_name).strip().lower()
    if part_l == 'right_finger':
        return np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def overlap_pose_quat_wxyz(entity_name, quat_wxyz):
    return normalize_quaternion_wxyz(quat_wxyz).astype(np.float32)


def load_pose_data(hdf5_path, episode=None, prefer_aligned=True, prefer_npz=False):
    with h5py.File(hdf5_path, "r") as f:
        demo_keys = list(f["data"].keys())
        if episode is None:
            episode = demo_keys[0]
        else:
            ep_str = str(episode)
            if ep_str in demo_keys:
                episode = ep_str
            elif ep_str.isdigit() and f"demo_{ep_str}" in demo_keys:
                episode = f"demo_{ep_str}"
            else:
                return None

        npz_group_path = f"data/{episode}/object_poses_npz"
        aligned_group_path = f"data/{episode}/object_poses_aligned"
        raw_group_path = f"data/{episode}/object_poses"
        pose_group = None
        if prefer_npz:
            if npz_group_path in f:
                pose_group = f[npz_group_path]
            elif raw_group_path in f:
                pose_group = f[raw_group_path]
            elif aligned_group_path in f:
                pose_group = f[aligned_group_path]
        elif prefer_aligned:
            if aligned_group_path in f:
                pose_group = f[aligned_group_path]
            elif raw_group_path in f:
                pose_group = f[raw_group_path]
        else:
            if raw_group_path in f:
                pose_group = f[raw_group_path]
            elif aligned_group_path in f:
                pose_group = f[aligned_group_path]
        if pose_group is None:
            return None
        out = {}
        for name in pose_group.keys():
            arr = np.asarray(pose_group[name], dtype=np.float32)
            if arr.ndim == 2 and arr.shape[1] >= 7:
                out[name] = arr[:, :7]
        return out


def _axis_aligned_rotation_candidates():
    mats = []
    eye = np.eye(3, dtype=np.float32)
    import itertools
    for perm in itertools.permutations(range(3)):
        base = eye[list(perm)]
        for signs in itertools.product([-1.0, 1.0], repeat=3):
            mat = (np.asarray(signs, dtype=np.float32).reshape(3, 1) * base).astype(np.float32)
            if np.linalg.det(mat) > 0.5:
                mats.append(mat)
    return mats


_AXIS_ROT_CANDIDATES = _axis_aligned_rotation_candidates()


def align_local_template_to_reference(local_points, reference_local_points):
    pts = np.asarray(local_points, dtype=np.float32)
    ref = np.asarray(reference_local_points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] == 0:
        return pts
    if ref.ndim != 2 or ref.shape[1] != 3 or ref.shape[0] == 0:
        return pts
    ref_extent = ref.max(axis=0) - ref.min(axis=0)
    ref_mean = ref.mean(axis=0)
    best_pts = pts
    best_score = None
    for rot in _AXIS_ROT_CANDIDATES:
        cand = (pts @ rot.T).astype(np.float32, copy=False)
        cand_extent = cand.max(axis=0) - cand.min(axis=0)
        score = float(np.linalg.norm(cand_extent - ref_extent))
        if best_score is None or score < best_score:
            best_score = score
            best_pts = cand
    offset = ref_mean - best_pts.mean(axis=0)
    return (best_pts + offset.reshape(1, 3)).astype(np.float32, copy=False)


def build_local_templates_from_reference_frame(pcd_data, pose_data, ref_frame=0):
    templates = {}
    if pose_data is None:
        return templates
    for obj_name, traj in pcd_data.items():
        pose_traj = pose_data.get(obj_name, None)
        if pose_traj is None or ref_frame >= pose_traj.shape[0]:
            continue
        if ref_frame >= traj.shape[0]:
            continue
        pts_ref = np.asarray(traj[ref_frame], dtype=np.float32)
        pose_ref = np.asarray(pose_traj[ref_frame], dtype=np.float32)
        templates[obj_name] = inverse_transform_points_wxyz(
            pts_ref,
            pose_ref[:3],
            pose_ref[3:7],
        )
    return templates


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SDF_DATA_ROOT = PROJECT_ROOT / "SDF" / "data"
_NPZ_BASENAME_CACHE = None
_OBJECT_META_CACHE = {}
_XML_PARTS_CACHE = {}


def _lazy_xml_helpers():
    from create_pcd_dataset_vnvae import find_object_xml, parse_object_parts_from_xml

    return find_object_xml, parse_object_parts_from_xml


def _canon_name(text):
    return ''.join(ch for ch in str(text).lower() if ch.isalnum())


def _load_xml_parts_for_category(category_name):
    key = str(category_name or '')
    if key in _XML_PARTS_CACHE:
        return _XML_PARTS_CACHE[key]
    if not key:
        _XML_PARTS_CACHE[key] = None
        return None
    try:
        find_object_xml, parse_object_parts_from_xml = _lazy_xml_helpers()
        xml_path = find_object_xml(key)
        if xml_path is None:
            _XML_PARTS_CACHE[key] = None
            return None
        parts = parse_object_parts_from_xml(xml_path)
        _XML_PARTS_CACHE[key] = parts if isinstance(parts, dict) else None
        return _XML_PARTS_CACHE[key]
    except Exception:
        _XML_PARTS_CACHE[key] = None
        return None


def _build_npz_basename_cache():
    global _NPZ_BASENAME_CACHE
    if _NPZ_BASENAME_CACHE is not None:
        return _NPZ_BASENAME_CACHE
    cache = {}
    if SDF_DATA_ROOT.is_dir():
        for p in SDF_DATA_ROOT.rglob("*.npz"):
            cache.setdefault(p.name, []).append(p)
    _NPZ_BASENAME_CACHE = cache
    return cache


def _pick_npz_candidate(name, candidates):
    if not candidates:
        return None
    basename = str(name or '').lower()
    preferred_tags = []
    if 'combined' in basename:
        preferred_tags.append('libero_merge')
    preferred_tags.extend(['libero_repair', 'libero_ch', 'libero', 'libero_norm'])
    lowered = [str(c).lower() for c in candidates]
    for tag in preferred_tags:
        for cand, low in zip(candidates, lowered):
            if f'/{tag}/' in low:
                return cand
    return sorted(candidates)[0]


def _resolve_npz_ref(npz_ref):
    if not npz_ref:
        return None
    p = Path(str(npz_ref))
    if p.is_file():
        return p
    candidates = [
        PROJECT_ROOT / 'SDF' / str(npz_ref),
        PROJECT_ROOT / str(npz_ref),
        SDF_DATA_ROOT / p.name,
    ]
    for cand in candidates:
        if cand.is_file():
            return cand
    cache = _build_npz_basename_cache()
    return _pick_npz_candidate(p.name, cache.get(p.name, []))


def _load_npz_meta(npz_path):
    resolved = _resolve_npz_ref(npz_path)
    if resolved is None:
        return None
    try:
        data = np.load(str(resolved), allow_pickle=True)
        meta_raw = data.get("meta", None)
        if meta_raw is None:
            return None
        if isinstance(meta_raw, np.ndarray) and meta_raw.shape == ():
            meta_raw = meta_raw.item()
        if isinstance(meta_raw, bytes):
            meta_raw = meta_raw.decode("utf-8")
        if isinstance(meta_raw, str):
            return json.loads(meta_raw)
        if isinstance(meta_raw, dict):
            return meta_raw
    except Exception:
        return None
    return None


def _load_object_meta_from_ckpt(latent_ckpt):
    key = str(latent_ckpt or '')
    if not key:
        return None
    if key in _OBJECT_META_CACHE:
        return _OBJECT_META_CACHE[key]
    resolved = Path(key)
    if not resolved.is_file():
        _OBJECT_META_CACHE[key] = None
        return None
    try:
        import torch
        ckpt = torch.load(str(resolved), map_location='cpu', weights_only=False)
    except Exception:
        _OBJECT_META_CACHE[key] = None
        return None
    meta = ckpt.get('object_meta', None)
    _OBJECT_META_CACHE[key] = meta if isinstance(meta, list) else None
    return _OBJECT_META_CACHE[key]


def _default_gripper_npz_name(entity_name):
    name = str(entity_name)
    if name.endswith('__hand'):
        return 'hand--sdfae_labels.npz'
    if name.endswith('__left_finger') or name.endswith('__right_finger'):
        return 'finger--sdfae_labels.npz'
    return ''


def resolve_target_npz_path(npz_name='', object_id=None, latent_ckpt=None, entity_name=''):
    object_meta = _load_object_meta_from_ckpt(latent_ckpt)
    if object_meta is not None and object_id is not None:
        try:
            oid = int(object_id)
        except Exception:
            oid = None
        if oid is not None:
            for item in object_meta:
                if int(item.get('object_id', -1)) == oid:
                    resolved = _resolve_npz_ref(item.get('npz_path', ''))
                    if resolved is not None:
                        return resolved
    direct = _resolve_npz_ref(npz_name)
    if direct is not None:
        return direct
    fallback_name = _default_gripper_npz_name(entity_name)
    if fallback_name:
        return _resolve_npz_ref(fallback_name)
    return None


def load_latent_mapping_info(hdf5_path):
    mapping = {}
    latent_ckpt = ''
    with h5py.File(hdf5_path, 'r') as f:
        if 'meta' not in f or 'latent_mapping' not in f['meta']:
            return mapping, latent_ckpt
        g = f['meta/latent_mapping']
        names = _decode_h5_strings(g['entity_names'][...]) if 'entity_names' in g else []
        npz_names = _decode_h5_strings(g['target_npz_names'][...]) if 'target_npz_names' in g else [''] * len(names)
        object_ids = np.asarray(g['target_object_ids'][...], dtype=np.int32) if 'target_object_ids' in g else np.full((len(names),), -1, dtype=np.int32)
        pose_offsets = np.asarray(g['pose_offset_local'][...], dtype=np.float32) if 'pose_offset_local' in g else np.zeros((len(names), 3), dtype=np.float32)
        body_names = _decode_h5_strings(g['body_names'][...]) if 'body_names' in g else [''] * len(names)
        categories = _decode_h5_strings(g['categories'][...]) if 'categories' in g else [''] * len(names)
        parts = _decode_h5_strings(g['parts'][...]) if 'parts' in g else [''] * len(names)
        match_kind = _decode_h5_strings(g['match_kind'][...]) if 'match_kind' in g else [''] * len(names)
        latent_ckpt = str(g.attrs.get('latent_ckpt', '') or '')
        for idx, name in enumerate(names):
            mapping[str(name)] = {
                'npz_name': str(npz_names[idx]) if idx < len(npz_names) else '',
                'object_id': int(object_ids[idx]) if idx < len(object_ids) else -1,
                'pose_offset_local': pose_offsets[idx].astype(np.float32, copy=False) if idx < len(pose_offsets) else np.zeros((3,), dtype=np.float32),
                'body_name': str(body_names[idx]) if idx < len(body_names) else '',
                'category': str(categories[idx]) if idx < len(categories) else '',
                'part': str(parts[idx]) if idx < len(parts) else '',
                'match_kind': str(match_kind[idx]) if idx < len(match_kind) else '',
            }
    return mapping, latent_ckpt


def load_npz_surface_points(npz_path, num_points=None):
    if npz_path is None:
        return None
    try:
        data = np.load(str(npz_path))
    except Exception:
        return None
    pts = np.asarray(data.get('surface_points', None), dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] == 0:
        return None
    if num_points is not None and num_points > 0 and pts.shape[0] != int(num_points):
        count = int(num_points)
        if pts.shape[0] > count:
            idx = np.linspace(0, pts.shape[0] - 1, count, dtype=np.int64)
            pts = pts[idx]
        else:
            reps = int(np.ceil(float(count) / float(pts.shape[0])))
            pts = np.tile(pts, (reps, 1))[:count]
    return pts.astype(np.float32, copy=False)


def _select_xml_mesh_item(items, npz_path):
    if not items:
        return None
    mesh_items = [item for item in items if isinstance(item, dict) and item.get('kind') == 'mesh']
    if not mesh_items:
        return None
    if len(mesh_items) == 1:
        return mesh_items[0]
    meta = _load_npz_meta(npz_path)
    asset_path = str(meta.get('asset_path', '') or '') if isinstance(meta, dict) else ''
    asset_stem = _canon_name(Path(asset_path).stem) if asset_path else ''
    if asset_stem:
        scored = []
        for item in mesh_items:
            mesh_stem = _canon_name(Path(str(item.get('mesh_path', ''))).stem)
            score = 0
            if mesh_stem == asset_stem:
                score += 10
            elif mesh_stem and (mesh_stem in asset_stem or asset_stem in mesh_stem):
                score += 5
            scored.append((score, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        if scored and scored[0][0] > 0:
            return scored[0][1]
    return mesh_items[0]


def sample_local_points_from_npz_via_xml(npz_path, category_name, body_name, num_points=None, unit_sphere_norm=False):
    resolved = _resolve_npz_ref(npz_path)
    if resolved is None:
        return None
    pts = load_npz_surface_points(resolved, num_points=num_points)
    if pts is None:
        return None
    parts = _load_xml_parts_for_category(category_name)
    if not parts:
        return None
    items = parts.get(str(body_name), None)
    if not items:
        body_canon = _canon_name(body_name)
        for key, cand_items in parts.items():
            if _canon_name(key) == body_canon:
                items = cand_items
                break
    if not items:
        return None
    item = _select_xml_mesh_item(items, resolved)
    if item is None:
        return None
    pts_body = transform_points_wxyz(pts, item.get('pos', [0.0, 0.0, 0.0]), item.get('quat', [1.0, 0.0, 0.0, 0.0]))
    center = pts_body.mean(axis=0, dtype=np.float32)
    pts_body = (pts_body - center.reshape(1, 3)).astype(np.float32, copy=False)
    if unit_sphere_norm:
        scale = float(np.linalg.norm(pts_body, axis=1).max()) if pts_body.shape[0] > 0 else 1.0
        if scale > 1e-8:
            pts_body = (pts_body / scale).astype(np.float32, copy=False)
    return pts_body.astype(np.float32, copy=False)


def sample_local_points_from_npz(npz_path, num_points=None, unit_sphere_norm=False):
    if npz_path is None:
        return None
    try:
        data = np.load(str(npz_path))
    except Exception:
        return None
    pts = np.asarray(data.get('surface_points', None), dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] == 0:
        return None
    center = pts.mean(axis=0, dtype=np.float32)
    pts = (pts - center.reshape(1, 3)).astype(np.float32, copy=False)
    if unit_sphere_norm:
        scale = float(np.linalg.norm(pts, axis=1).max()) if pts.shape[0] > 0 else 1.0
        if scale > 1e-8:
            pts = (pts / scale).astype(np.float32, copy=False)
    if num_points is not None and num_points > 0 and pts.shape[0] != int(num_points):
        count = int(num_points)
        if pts.shape[0] > count:
            idx = np.linspace(0, pts.shape[0] - 1, count, dtype=np.int64)
            pts = pts[idx]
        else:
            reps = int(np.ceil(float(count) / float(pts.shape[0])))
            pts = np.tile(pts, (reps, 1))[:count]
    return pts.astype(np.float32, copy=False)


def build_local_templates(
    pcd_data,
    pose_data,
    hdf5_path,
    ref_frame=0,
    template_source='ref',
    npz_unit_sphere_norm=False,
):
    templates = {}
    sources = {}
    reference_templates = build_local_templates_from_reference_frame(
        pcd_data,
        pose_data,
        ref_frame=ref_frame,
    )
    if str(template_source) == 'ref':
        for obj_name, reference_local in reference_templates.items():
            templates[obj_name] = np.asarray(reference_local, dtype=np.float32)
            sources[obj_name] = 'ref-frame-local'
        return templates, sources

    mapping_info, latent_ckpt = load_latent_mapping_info(hdf5_path)
    for obj_name, traj in pcd_data.items():
        target_count = int(np.asarray(traj[ref_frame]).shape[0]) if ref_frame < traj.shape[0] else None
        info = mapping_info.get(obj_name, {})
        match_kind = str(info.get('match_kind', ''))
        reference_local = reference_templates.get(obj_name, None)
        if match_kind.startswith('combined') and reference_local is not None:
            # Combined objects must also be validated against the latent NPZ/SDF
            # canonical frame. Do not silently fall back to frame-0 HDF5 points.
            pass
        npz_path = resolve_target_npz_path(
            npz_name=info.get('npz_name', ''),
            object_id=info.get('object_id', None),
            latent_ckpt=latent_ckpt,
            entity_name=obj_name,
        )
        pts_local = sample_local_points_from_npz(
            npz_path,
            num_points=target_count,
            unit_sphere_norm=bool(npz_unit_sphere_norm),
        )
        if pts_local is None:
            if reference_local is not None:
                templates[obj_name] = np.asarray(reference_local, dtype=np.float32)
                sources[obj_name] = 'ref-fallback-missing-npz'
            continue
        templates[obj_name] = pts_local.astype(np.float32, copy=False)
        tag = 'npz-unit-sphere' if npz_unit_sphere_norm else 'npz-centered'
        sources[obj_name] = f'{tag}:{Path(npz_path).name}'
    return templates, sources


def load_pcd_data(hdf5_path, episode=None, frame=None):
    """Load point cloud and optional distance data from HDF5 file."""
    with h5py.File(hdf5_path, "r") as f:
        demo_keys = list(f["data"].keys())
        if episode is None:
            episode = demo_keys[0]
        else:
            ep_str = str(episode)
            if ep_str in demo_keys:
                episode = ep_str
            elif ep_str.isdigit() and f"demo_{ep_str}" in demo_keys:
                episode = f"demo_{ep_str}"
            else:
                print(f"Episode '{episode}' not found. Available examples: {demo_keys[:5]}")
                return None, None, None, None

        if "object_pcds" not in f[f"data/{episode}"]:
            print(f"No object_pcds found in {episode}")
            return None, None, None, None

        pcd_group = f[f"data/{episode}/object_pcds"]
        object_names = list(pcd_group.keys())

        pcd_data = {obj_name: pcd_group[obj_name][:] for obj_name in object_names}

        distance_data = None
        dist_group_path = f"data/{episode}/object_distances"
        if dist_group_path in f:
            dist_group = f[dist_group_path]
            if "values" in dist_group and "entity_names" in dist_group:
                distance_data = {
                    "values": np.asarray(dist_group["values"]),
                    "entity_names": _decode_h5_strings(dist_group["entity_names"]),
                }

        num_frames = pcd_data[object_names[0]].shape[0]

        if frame is not None:
            if frame >= num_frames:
                print(f"Frame {frame} out of range (max: {num_frames - 1})")
                return None, None, None, None
            for obj_name in object_names:
                pcd_data[obj_name] = pcd_data[obj_name][frame]
            current_frame = frame
        else:
            current_frame = 0

        return pcd_data, current_frame, num_frames, distance_data


def _query_nearest(points, tree):
    try:
        dists, idx = tree.query(points, k=1, workers=-1)
    except TypeError:
        dists, idx = tree.query(points, k=1)
    dists = np.asarray(dists, dtype=np.float64).reshape(-1)
    idx = np.asarray(idx, dtype=np.int64).reshape(-1)
    return dists, idx


def compute_pairwise_distance_matrix(pcd_data, entity_names, frame, return_pairs=False):
    n = len(entity_names)
    out = np.zeros((n, n), dtype=np.float32)
    pair_points = {} if return_pairs else None

    frame_points = [np.asarray(pcd_data[name][frame], dtype=np.float32) for name in entity_names]
    trees = [cKDTree(pts) for pts in frame_points] if cKDTree is not None else None

    for i in range(n):
        out[i, i] = 0.0
        pts_i = frame_points[i]
        for j in range(i + 1, n):
            pts_j = frame_points[j]
            if pts_i.shape[0] == 0 or pts_j.shape[0] == 0:
                d = np.nan
                pa = np.zeros((3,), dtype=np.float32)
                pb = np.zeros((3,), dtype=np.float32)
            elif trees is not None:
                dists, nn_idx = _query_nearest(pts_i, trees[j])
                min_idx = int(np.argmin(dists))
                d = float(dists[min_idx])
                pa = pts_i[min_idx]
                pb = pts_j[int(nn_idx[min_idx])]
            else:
                diff = pts_i[:, None, :] - pts_j[None, :, :]
                d2 = np.einsum("ijk,ijk->ij", diff, diff)
                flat = int(np.argmin(d2))
                ii, jj = np.unravel_index(flat, d2.shape)
                d = float(np.sqrt(d2[ii, jj]))
                pa = pts_i[ii]
                pb = pts_j[jj]

            d = 0.0 if not np.isfinite(d) else float(d)
            out[i, j] = d
            out[j, i] = d
            if return_pairs:
                pair_points[(i, j)] = (
                    np.asarray(pa, dtype=np.float32),
                    np.asarray(pb, dtype=np.float32),
                    float(d),
                )

    return out, pair_points


def _nearest_points_between_entities(pcd_data, entity_a, entity_b, frame):
    pts_a = np.asarray(pcd_data[entity_a][frame], dtype=np.float32)
    pts_b = np.asarray(pcd_data[entity_b][frame], dtype=np.float32)
    if pts_a.shape[0] == 0 or pts_b.shape[0] == 0:
        return {
            "name_i": entity_a,
            "name_j": entity_b,
            "point_i": None,
            "point_j": None,
            "distance": float("nan"),
            "rank": 1,
        }

    if cKDTree is not None:
        tree_b = cKDTree(pts_b)
        dists, nn_idx = _query_nearest(pts_a, tree_b)
        min_idx = int(np.argmin(dists))
        pa = pts_a[min_idx]
        pb = pts_b[int(nn_idx[min_idx])]
        d = float(dists[min_idx])
    else:
        diff = pts_a[:, None, :] - pts_b[None, :, :]
        d2 = np.einsum("ijk,ijk->ij", diff, diff)
        flat = int(np.argmin(d2))
        ia, ib = np.unravel_index(flat, d2.shape)
        pa = pts_a[ia]
        pb = pts_b[ib]
        d = float(np.sqrt(d2[ia, ib]))

    return {
        "name_i": entity_a,
        "name_j": entity_b,
        "point_i": np.asarray(pa, dtype=np.float32),
        "point_j": np.asarray(pb, dtype=np.float32),
        "distance": d,
        "rank": 1,
    }


def _ranked_pair_from_matrix(entity_names, dist_matrix, pair_points, rank=1, cross_instance_only=True):
    n = len(entity_names)
    if n < 2:
        return None
    triu = np.triu_indices(n, k=1)
    vals = np.asarray(dist_matrix[triu], dtype=np.float64)
    if vals.size == 0:
        return None

    candidates = []
    for k in range(vals.size):
        i = int(triu[0][k])
        j = int(triu[1][k])
        if cross_instance_only and (_entity_instance_name(entity_names[i]) == _entity_instance_name(entity_names[j])):
            continue
        candidates.append((float(vals[k]), i, j, k))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0])
    ridx = max(0, min(int(rank) - 1, len(candidates) - 1))
    _d, i, j, kidx = candidates[ridx]
    pa, pb, d = pair_points.get((i, j), (None, None, float(vals[kidx])))
    return {
        "i": i,
        "j": j,
        "name_i": entity_names[i],
        "name_j": entity_names[j],
        "point_i": pa,
        "point_j": pb,
        "distance": float(d),
        "rank": int(ridx + 1),
    }


def print_distance_verification(pcd_data, distance_data, frame, topk=3, pred_matrix=None):
    if distance_data is None:
        print(f"[distance-check] frame={frame}: no object_distances found in dataset")
        return None

    names = list(distance_data["entity_names"])
    if not names:
        print(f"[distance-check] frame={frame}: empty entity_names in object_distances")
        return None

    missing = [n for n in names if n not in pcd_data]
    if missing:
        print(f"[distance-check] frame={frame}: missing entities in object_pcds: {missing}")
        return None

    values = np.asarray(distance_data["values"], dtype=np.float32)
    if values.ndim != 3 or values.shape[1] != len(names) or values.shape[2] != len(names):
        print(
            f"[distance-check] frame={frame}: invalid object_distances/values shape {values.shape}, "
            f"expected (T,{len(names)},{len(names)})"
        )
        return None
    if frame >= values.shape[0]:
        print(f"[distance-check] frame={frame}: out of distance range (max={values.shape[0] - 1})")
        return None

    if pred_matrix is None:
        pred_matrix, _ = compute_pairwise_distance_matrix(pcd_data, names, frame, return_pairs=False)

    ref = values[frame]
    diff = np.abs(pred_matrix - ref)
    mean_abs = float(np.mean(diff))
    max_abs = float(np.max(diff))
    max_flat = int(np.argmax(diff))
    i_max, j_max = np.unravel_index(max_flat, diff.shape)
    print(
        f"[distance-check] frame={frame}: mean_abs={mean_abs:.6e}, max_abs={max_abs:.6e}, "
        f"worst_pair=({names[i_max]}, {names[j_max]}), "
        f"saved={float(ref[i_max, j_max]):.6e}, recompute={float(pred_matrix[i_max, j_max]):.6e}"
    )

    if topk > 0 and len(names) > 1:
        triu = np.triu_indices(len(names), k=1)
        pair_vals = ref[triu]
        order = np.argsort(pair_vals)
        k = min(int(topk), len(order))
        print(f"[distance-check] frame={frame}: top-{k} nearest pairs from saved distances:")
        for rank in range(k):
            idx = order[rank]
            i, j = int(triu[0][idx]), int(triu[1][idx])
            print(
                f"  {rank + 1}. {names[i]} <-> {names[j]} : "
                f"saved={float(ref[i, j]):.6e}, recompute={float(pred_matrix[i, j]):.6e}, "
                f"abs_err={float(diff[i, j]):.6e}"
            )

    return pred_matrix


def print_distance_verification_for_pair(
    pcd_data,
    distance_data,
    frame,
    entity_a,
    entity_b,
    recompute_distance=None,
):
    if distance_data is None:
        print(f"[distance-check] frame={frame}: no object_distances found in dataset")
        return None

    names = list(distance_data["entity_names"])
    if not names:
        print(f"[distance-check] frame={frame}: empty entity_names in object_distances")
        return None

    values = np.asarray(distance_data["values"], dtype=np.float32)
    if values.ndim != 3 or values.shape[1] != len(names) or values.shape[2] != len(names):
        print(
            f"[distance-check] frame={frame}: invalid object_distances/values shape {values.shape}, "
            f"expected (T,{len(names)},{len(names)})"
        )
        return None
    if frame >= values.shape[0]:
        print(f"[distance-check] frame={frame}: out of distance range (max={values.shape[0] - 1})")
        return None

    if entity_a not in names or entity_b not in names:
        print(
            f"[distance-check] frame={frame}: fixed pair not found in object_distances/entity_names: "
            f"{entity_a}, {entity_b}"
        )
        return None

    if entity_a not in pcd_data or entity_b not in pcd_data:
        print(
            f"[distance-check] frame={frame}: fixed pair not found in object_pcds: "
            f"{entity_a}, {entity_b}"
        )
        return None

    i = int(names.index(entity_a))
    j = int(names.index(entity_b))
    saved = float(values[frame, i, j])

    if recompute_distance is None:
        seg = _nearest_points_between_entities(pcd_data, entity_a, entity_b, frame)
        recompute = float(seg["distance"])
    else:
        recompute = float(recompute_distance)

    abs_err = abs(recompute - saved)
    print(
        f"[distance-check] frame={frame}: fixed_pair=({entity_a}, {entity_b}), "
        f"saved={saved:.6e}, recompute={recompute:.6e}, abs_err={abs_err:.6e}"
    )
    return recompute


def generate_distinct_colors(object_names):
    """Generate deterministic distinct colors for each object name."""
    colors = {}
    golden = 0.618033988749895
    hue = 0.11
    for i, name in enumerate(sorted(object_names)):
        hue = (hue + golden) % 1.0
        sat = 0.75
        val = 0.95 if (i % 2 == 0) else 0.85
        rgb = colorsys.hsv_to_rgb(hue, sat, val)
        colors[name] = [rgb[0], rgb[1], rgb[2]]
    return colors


def visualize(
    hdf5_path,
    episode=None,
    start_frame=0,
    point_size=3.0,
    verify_distance=False,
    distance_topk=3,
    draw_nearest_segment=False,
    segment_rank=1,
    segment_marker_radius=0.008,
    segment_cross_instance_only=True,
    segment_object_a=None,
    segment_object_b=None,
    overlap=False,
    overlap_source='ref',
    overlap_unit_sphere_norm=False,
):
    """Visualize point cloud trajectory with interactive controls."""
    pcd_data, current_frame, num_frames, distance_data = load_pcd_data(hdf5_path, episode, None)

    if pcd_data is None:
        return

    current_frame = int(max(0, min(start_frame, num_frames - 1)))

    object_names = list(pcd_data.keys())
    object_colors = generate_distinct_colors(object_names)
    overlap_colors = {
        name: (0.35 + 0.65 * np.asarray(color, dtype=np.float32)).clip(0.0, 1.0).tolist()
        for name, color in object_colors.items()
    }
    pose_data = None
    if overlap:
        prefer_aligned_pose = str(overlap_source) != 'npz'
        pose_data = load_pose_data(
            hdf5_path,
            episode,
            prefer_aligned=prefer_aligned_pose,
            prefer_npz=(str(overlap_source) == 'npz'),
        )
    overlap_local_templates = {}
    overlap_sources = {}
    if overlap:
        overlap_local_templates, overlap_sources = build_local_templates(
            pcd_data,
            pose_data,
            hdf5_path,
            ref_frame=0,
            template_source=overlap_source,
            npz_unit_sphere_norm=bool(overlap_unit_sphere_norm),
        )
    fixed_pair_mode = bool(segment_object_a and segment_object_b)

    # Names used for distance/segment checks: prefer saved distance entity order.
    segment_entity_names = []
    if distance_data is not None and len(distance_data.get("entity_names", [])) > 0:
        segment_entity_names = [n for n in distance_data["entity_names"] if n in pcd_data]
    else:
        segment_entity_names = [n for n in sorted(object_names) if n != "robot_gripper"]

    print(f"Trajectory: {num_frames} frames")
    print(f"Objects: {object_names}")
    print("SPACE: next | BACKSPACE: prev | ESC: quit")
    if overlap:
        if str(overlap_source) == 'ref':
            print(
                "Overlap mode: recover local templates from frame 0 of the stored HDF5 point clouds, "
                "then transform them with dataset poses."
            )
        else:
            print(
                "Overlap mode: read local templates from latent NPZ files, center them, "
                + ("apply unit-sphere normalization, " if overlap_unit_sphere_norm else "")
                + "then transform them with object_poses_npz if present, otherwise object_poses."
            )
        if pose_data is None:
            print("Overlap mode requested, but no compatible pose group was found in this episode.")
        missing = [n for n in object_names if n not in overlap_local_templates]
        if missing:
            print(f"Overlap mode: missing local-template entries for {missing}")
        if overlap_sources:
            npz_count = sum(1 for v in overlap_sources.values() if v.startswith('npz-'))
            ref_fb_count = sum(1 for v in overlap_sources.values() if v.startswith('ref-fallback'))
            print(f"Overlap templates from npz: {npz_count}")
            if ref_fb_count:
                print(f"Overlap templates using reference fallback: {ref_fb_count}")
    if verify_distance:
        if distance_data is None:
            print("Distance verification: object_distances not found in this episode.")
        else:
            n = len(distance_data["entity_names"])
            tshape = tuple(distance_data["values"].shape)
            print(f"Distance verification enabled: values_shape={tshape}, entities={n}")
    if draw_nearest_segment:
        mode = "cross-instance only" if bool(segment_cross_instance_only) else "all pairs"
        if segment_object_a and segment_object_b:
            print(
                "Nearest segment drawing enabled for fixed pair: "
                f"{segment_object_a} <-> {segment_object_b} (red line)."
            )
        else:
            print(f"Nearest segment drawing enabled (rank={int(segment_rank)} pair, {mode}, red line).")
    print(f"=== Frame {current_frame}/{num_frames-1} ===")
    sys.stdout.flush()

    pcd_geos = []
    overlap_geos = []
    for obj_name in object_names:
        points = pcd_data[obj_name][current_frame]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        color = object_colors.get(obj_name, [0.7, 0.7, 0.7])
        colors = np.tile(color, (len(points), 1))
        pcd.colors = o3d.utility.Vector3dVector(colors)
        pcd_geos.append(pcd)

        if overlap:
            pcd_overlap = o3d.geometry.PointCloud()
            overlap_points = np.zeros((0, 3), dtype=np.float32)
            if obj_name in overlap_local_templates and pose_data is not None and obj_name in pose_data and current_frame < pose_data[obj_name].shape[0]:
                pose = pose_data[obj_name][current_frame]
                overlap_points = transform_points_wxyz(overlap_local_templates[obj_name], pose[:3], pose[3:7])
            pcd_overlap.points = o3d.utility.Vector3dVector(overlap_points)
            overlap_color = overlap_colors.get(obj_name, [0.9, 0.9, 0.9])
            overlap_colors_arr = np.tile(overlap_color, (len(overlap_points), 1))
            pcd_overlap.colors = o3d.utility.Vector3dVector(overlap_colors_arr)
            overlap_geos.append(pcd_overlap)

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Point Cloud", width=1280, height=720)

    for p in pcd_geos:
        vis.add_geometry(p)
    for p in overlap_geos:
        vis.add_geometry(p)
    vis.add_geometry(coord_frame)

    nearest_line = None
    nearest_marker_a = None
    nearest_marker_b = None
    marker_a_center = np.zeros((3,), dtype=np.float32)
    marker_b_center = np.zeros((3,), dtype=np.float32)
    if draw_nearest_segment and len(segment_entity_names) >= 2:
        nearest_line = o3d.geometry.LineSet()
        nearest_line.points = o3d.utility.Vector3dVector(np.zeros((2, 3), dtype=np.float32))
        nearest_line.lines = o3d.utility.Vector2iVector(np.array([[0, 1]], dtype=np.int32))
        nearest_line.colors = o3d.utility.Vector3dVector(np.array([[1.0, 0.1, 0.1]], dtype=np.float32))
        vis.add_geometry(nearest_line)

        rad = max(float(segment_marker_radius), 1e-4)
        nearest_marker_a = o3d.geometry.TriangleMesh.create_sphere(radius=rad)
        nearest_marker_b = o3d.geometry.TriangleMesh.create_sphere(radius=rad)
        nearest_marker_a.paint_uniform_color([1.0, 0.15, 0.15])
        nearest_marker_b.paint_uniform_color([1.0, 0.9, 0.1])
        vis.add_geometry(nearest_marker_a)
        vis.add_geometry(nearest_marker_b)

    ctrl = vis.get_view_control()
    ctrl.set_lookat([0, 0, 0])
    ctrl.set_front([0, -1, 1])
    ctrl.set_up([0, 0, 1])
    ctrl.set_zoom(0.8)
    vis.get_render_option().point_size = float(point_size)

    def _update_nearest_line(seg_info):
        nonlocal marker_a_center, marker_b_center
        if nearest_line is None:
            return

        if seg_info is None or seg_info["point_i"] is None or seg_info["point_j"] is None:
            p0 = np.zeros((3,), dtype=np.float32)
            p1 = np.zeros((3,), dtype=np.float32)
        else:
            p0 = np.asarray(seg_info["point_i"], dtype=np.float32)
            p1 = np.asarray(seg_info["point_j"], dtype=np.float32)

        pts = np.vstack([p0, p1]).astype(np.float32)
        nearest_line.points = o3d.utility.Vector3dVector(pts)
        nearest_line.lines = o3d.utility.Vector2iVector(np.array([[0, 1]], dtype=np.int32))
        nearest_line.colors = o3d.utility.Vector3dVector(np.array([[1.0, 0.1, 0.1]], dtype=np.float32))
        vis.update_geometry(nearest_line)

        if nearest_marker_a is not None and nearest_marker_b is not None:
            nearest_marker_a.translate((p0 - marker_a_center).astype(np.float64), relative=True)
            nearest_marker_b.translate((p1 - marker_b_center).astype(np.float64), relative=True)
            marker_a_center = p0
            marker_b_center = p1
            vis.update_geometry(nearest_marker_a)
            vis.update_geometry(nearest_marker_b)

    def update_frame():
        nonlocal current_frame
        for i, obj_name in enumerate(object_names):
            points = pcd_data[obj_name][current_frame]
            pcd_geos[i].points = o3d.utility.Vector3dVector(points)
            vis.update_geometry(pcd_geos[i])
            if overlap and i < len(overlap_geos):
                overlap_points = np.zeros((0, 3), dtype=np.float32)
                if obj_name in overlap_local_templates and pose_data is not None and obj_name in pose_data and current_frame < pose_data[obj_name].shape[0]:
                    pose = pose_data[obj_name][current_frame]
                    overlap_points = transform_points_wxyz(overlap_local_templates[obj_name], pose[:3], pose[3:7])
                overlap_geos[i].points = o3d.utility.Vector3dVector(overlap_points)
                overlap_color = overlap_colors.get(obj_name, [0.9, 0.9, 0.9])
                overlap_colors_arr = np.tile(overlap_color, (len(overlap_points), 1))
                overlap_geos[i].colors = o3d.utility.Vector3dVector(overlap_colors_arr)
                vis.update_geometry(overlap_geos[i])

        pred_matrix = None
        pair_points = None
        seg_info = None

        if (verify_distance or draw_nearest_segment) and (not fixed_pair_mode):
            if len(segment_entity_names) >= 2:
                pred_matrix, pair_points = compute_pairwise_distance_matrix(
                    pcd_data=pcd_data,
                    entity_names=segment_entity_names,
                    frame=current_frame,
                    return_pairs=draw_nearest_segment,
                )

        print(f"=== Frame {current_frame}/{num_frames-1} ===")

        if verify_distance:
            if fixed_pair_mode:
                print_distance_verification_for_pair(
                    pcd_data=pcd_data,
                    distance_data=distance_data,
                    frame=current_frame,
                    entity_a=segment_object_a,
                    entity_b=segment_object_b,
                    recompute_distance=(None if seg_info is None else seg_info["distance"]),
                )
            else:
                print_distance_verification(
                    pcd_data=pcd_data,
                    distance_data=distance_data,
                    frame=current_frame,
                    topk=distance_topk,
                    pred_matrix=pred_matrix,
                )

        if draw_nearest_segment:
            if len(segment_entity_names) < 2:
                print("[nearest-segment] not enough entities to draw segment")
            else:
                if segment_object_a and segment_object_b:
                    if segment_object_a not in pcd_data or segment_object_b not in pcd_data:
                        print(
                            "[nearest-segment] fixed pair not found in object_pcds: "
                            f"{segment_object_a}, {segment_object_b}"
                        )
                        seg_info = None
                    else:
                        seg_info = _nearest_points_between_entities(
                            pcd_data=pcd_data,
                            entity_a=segment_object_a,
                            entity_b=segment_object_b,
                            frame=current_frame,
                        )
                else:
                    seg_info = _ranked_pair_from_matrix(
                        segment_entity_names,
                        pred_matrix,
                        pair_points,
                        rank=segment_rank,
                        cross_instance_only=bool(segment_cross_instance_only),
                    )
                if seg_info is not None:
                    print(
                        f"[nearest-segment] frame={current_frame}: "
                        f"rank={seg_info['rank']} "
                        f"{seg_info['name_i']} <-> {seg_info['name_j']} "
                        f"distance={seg_info['distance']:.6e}, "
                        f"p_i={np.asarray(seg_info['point_i']).round(5).tolist() if seg_info['point_i'] is not None else None}, "
                        f"p_j={np.asarray(seg_info['point_j']).round(5).tolist() if seg_info['point_j'] is not None else None}"
                    )
                else:
                    print(
                        f"[nearest-segment] frame={current_frame}: no valid pair under "
                        f"cross_instance_only={bool(segment_cross_instance_only)}"
                    )
            _update_nearest_line(seg_info)

        sys.stdout.flush()

    def next_cb(vis_obj):
        nonlocal current_frame
        current_frame = (current_frame + 1) % num_frames
        update_frame()
        vis_obj.poll_events()
        vis_obj.update_renderer()

    def prev_cb(vis_obj):
        nonlocal current_frame
        current_frame = (current_frame - 1 + num_frames) % num_frames
        update_frame()
        vis_obj.poll_events()
        vis_obj.update_renderer()

    def quit_cb(vis_obj):
        vis_obj.close()

    vis.register_key_callback(32, next_cb)  # SPACE
    vis.register_key_callback(8, prev_cb)   # BACKSPACE
    vis.register_key_callback(27, quit_cb)  # ESC

    update_frame()
    vis.run()
    vis.destroy_window()


def main():
    parser = argparse.ArgumentParser(description="Visualize point cloud data")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--episode", type=str, default=None)
    parser.add_argument("--frame", type=int, default=None)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--point-size", type=float, default=3.0)
    parser.add_argument(
        "--verify-distance",
        action="store_true",
        help="Recompute per-frame pairwise distances from point clouds and compare with object_distances/values",
    )
    parser.add_argument(
        "--distance-topk",
        type=int,
        default=3,
        help="When --verify-distance is enabled, print top-k nearest pairs from saved distance matrix",
    )
    parser.add_argument(
        "--draw-nearest-segment",
        action="store_true",
        help="Draw shortest segment between a selected near-object pair",
    )
    parser.add_argument(
        "--segment-rank",
        type=int,
        default=1,
        help="Draw rank-k nearest pair (1=closest, 2=second closest, ...)",
    )
    parser.add_argument(
        "--segment-marker-radius",
        type=float,
        default=0.008,
        help="Radius (meters) of endpoint sphere markers for nearest segment",
    )
    parser.add_argument(
        "--segment-cross-instance-only",
        dest="segment_cross_instance_only",
        action="store_true",
        default=True,
        help="Only search nearest segment across different object instances (default: enabled)",
    )
    parser.add_argument(
        "--segment-allow-same-instance",
        dest="segment_cross_instance_only",
        action="store_false",
        help="Allow nearest segment to be picked from parts of the same instance",
    )
    parser.add_argument(
        "--segment-object-a",
        type=str,
        default=None,
        help="Fixed object A name for nearest-point segment (must exist in object_pcds)",
    )
    parser.add_argument(
        "--segment-object-b",
        type=str,
        default=None,
        help="Fixed object B name for nearest-point segment (must exist in object_pcds)",
    )
    parser.add_argument(
        "--overlap",
        action="store_true",
        help="Overlay pose-reconstructed point clouds on top of stored global point clouds.",
    )
    parser.add_argument(
        "--overlap-source",
        type=str,
        default="npz",
        choices=["ref", "npz"],
        help="Source of local template for --overlap: ref=frame-0 HDF5 local template, npz=latent NPZ surface points.",
    )
    parser.add_argument(
        "--overlap-unit-sphere-norm",
        action="store_true",
        help="When --overlap-source=npz, also apply the SDF training unit-sphere normalization to NPZ surface points.",
    )

    args = parser.parse_args()
    start = args.frame if args.frame is not None else args.start_frame

    visualize(
        args.dataset,
        episode=args.episode,
        start_frame=start,
        point_size=args.point_size,
        verify_distance=bool(args.verify_distance),
        distance_topk=int(args.distance_topk),
        draw_nearest_segment=bool(args.draw_nearest_segment),
        segment_rank=int(args.segment_rank),
        segment_marker_radius=float(args.segment_marker_radius),
        segment_cross_instance_only=bool(args.segment_cross_instance_only),
        segment_object_a=args.segment_object_a,
        segment_object_b=args.segment_object_b,
        overlap=bool(args.overlap),
        overlap_source=str(args.overlap_source),
        overlap_unit_sphere_norm=bool(args.overlap_unit_sphere_norm),
    )


if __name__ == "__main__":
    main()
