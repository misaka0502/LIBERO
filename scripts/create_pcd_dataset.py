"""
Generate object point cloud trajectories from LIBERO datasets.

Key behaviors:
1. Parse each object category XML dynamically (no hard-coded part names).
2. Treat each body-part with mesh geoms as an independent object entry.
3. Support mesh assets in obj / msh / stl (via trimesh).
4. Extract poses from MuJoCo states and transform each part point cloud per timestep.
5. Build Franka gripper part point clouds (hand/left_finger/right_finger) from MuJoCo robot model + states per timestep.
6. Save object and gripper point clouds into HDF5 under data/<demo>/object_pcds.
7. Save per-step poses under data/<demo>/object_poses and data/<demo>/gripper_poses.
8. Optionally save per-step pairwise entity minimum distances under data/<demo>/object_distances.
9. Optional center-by-surface mode: recenter each object by local surface centroid.
"""

import argparse
import glob
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET
from typing import List, Set, Tuple

import h5py
import numpy as np
import trimesh

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover - optional speedup
    cKDTree = None


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSETS_BASE_PATH = REPO_ROOT / "libero" / "libero" / "assets"
BDDL_BASE_PATH = REPO_ROOT / "libero" / "libero" / "bddl_files"
BDDL_NEW_BASE_PATH = REPO_ROOT / "libero" / "libero" / "bddl_files_new"
MESH_EXTENSIONS = {".obj", ".msh", ".stl"}
MJ_GEOM_TYPE_TO_PRIMITIVE = {2: "sphere", 3: "capsule", 4: "ellipsoid", 5: "cylinder", 6: "box"}
GRIPPER_ENTITY_NAME = "robot_gripper"
CATEGORY_XML_ALIASES = {
    "white_cabinet": "wooden_cabinet",
}

TOKEN_BLACKLIST = {
    "textured",
    "model",
    "mesh",
    "coll",
    "vis",
    "object",
    "hope",
    "site",
}


def _basename_npz(path):
    if not path:
        return ""
    return os.path.basename(str(path).replace("\\", "/"))


def _latent_center_from_meta(meta):
    if not isinstance(meta, dict):
        return np.zeros((3,), dtype=np.float32)
    center = meta.get("center")
    if isinstance(center, (list, tuple, np.ndarray)) and len(center) == 3:
        out = np.asarray(center, dtype=np.float32).reshape(3)
        if np.all(np.isfinite(out)):
            return out
    bounds = meta.get("bounds")
    if isinstance(bounds, (list, tuple)) and len(bounds) == 6:
        xmin, xmax, ymin, ymax, zmin, zmax = [float(x) for x in bounds]
        return np.asarray(
            [0.5 * (xmin + xmax), 0.5 * (ymin + ymax), 0.5 * (zmin + zmax)],
            dtype=np.float32,
        )
    return np.zeros((3,), dtype=np.float32)


def _resolve_npz_path(path):
    if not path:
        return None
    p = Path(str(path))
    if p.is_file():
        return p
    candidates = [
        PROJECT_ROOT / "SDF" / str(path),
        PROJECT_ROOT / str(path),
        PROJECT_ROOT / "SDF" / "data" / p.name,
    ]
    for cand in candidates:
        if cand.is_file():
            return cand
    return None


def _surface_center_from_npz_path(npz_path, cache):
    key = str(npz_path or "")
    if key in cache:
        return cache[key]
    resolved = _resolve_npz_path(npz_path)
    if resolved is None:
        cache[key] = None
        return None
    try:
        data = np.load(str(resolved))
        sp = data.get("surface_points", None)
        if isinstance(sp, np.ndarray) and sp.ndim == 2 and sp.shape[1] == 3 and sp.shape[0] > 0:
            center = sp.mean(axis=0).astype(np.float32, copy=False)
            cache[key] = center
            return center
    except Exception:
        pass
    cache[key] = None
    return None


def _centered_surface_points_from_npz_path(npz_path, cache, target_num=None):
    key = (str(npz_path or ""), None if target_num is None else int(target_num))
    if key in cache:
        return cache[key]
    resolved = _resolve_npz_path(npz_path)
    if resolved is None:
        cache[key] = None
        return None
    try:
        data = np.load(str(resolved))
        sp = np.asarray(data.get("surface_points", None), dtype=np.float32)
    except Exception:
        sp = None
    if sp is None or sp.ndim != 2 or sp.shape[1] != 3 or sp.shape[0] == 0:
        cache[key] = None
        return None
    center = sp.mean(axis=0, dtype=np.float32)
    pts = (sp - center.reshape(1, 3)).astype(np.float32, copy=False)
    if target_num is not None and int(target_num) > 0:
        pts = normalize_point_count(pts, int(target_num)).astype(np.float32, copy=False)
    # Keep the latent-local origin at zero even after subsampling/tiling.
    pts = (pts - pts.mean(axis=0, dtype=np.float32).reshape(1, 3)).astype(np.float32, copy=False)
    cache[key] = pts
    return pts


def _entry_asset_stem(entry):
    asset_path = getattr(entry, "asset_path", None)
    if not asset_path:
        return ""
    return canonical_name(Path(str(asset_path)).stem)


def _entry_tokens(entry):
    return set(getattr(entry, "tokens", ())).union(set(getattr(entry, "part_tokens", ())))


def _is_combined_latent_entry(entry):
    asset_path = str(getattr(entry, "asset_path", "") or "").replace("\\", "/").lower()
    if "/libero-merge/" in asset_path or "/libero_merge/" in asset_path:
        return True
    toks = _entry_tokens(entry)
    if "combined" in toks:
        return True
    for name in getattr(entry, "names", ()):
        if "combined" in canonical_name(name):
            return True
    stem = _entry_asset_stem(entry)
    if stem and "combined" in stem:
        return True
    return False


def load_latent_object_meta(latent_ckpt):
    if not latent_ckpt:
        return None

    import torch

    try:
        ckpt = torch.load(str(latent_ckpt), map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(str(latent_ckpt), map_location="cpu")

    object_meta = ckpt.get("object_meta")
    if not isinstance(object_meta, list) or not object_meta:
        print(f"  Warning: latent ckpt has no object_meta, skip latent mapping export: {latent_ckpt}")
        return None
    return object_meta


def _spec_surface_center(spec):
    if isinstance(spec, dict):
        center = spec.get("surface_center", None)
        if isinstance(center, np.ndarray) and center.shape == (3,):
            return center.astype(np.float32, copy=False)
        if isinstance(center, (list, tuple)) and len(center) == 3:
            return np.asarray(center, dtype=np.float32).reshape(3)
        pts = np.asarray(spec.get("points", None), dtype=np.float32)
        if pts.ndim == 2 and pts.shape[1] == 3 and pts.shape[0] > 0:
            return pts.mean(axis=0).astype(np.float32, copy=False)
    return np.zeros((3,), dtype=np.float32)


def _copy_source_item(item):
    if not isinstance(item, dict):
        return {}
    out = {}
    for key, value in item.items():
        if isinstance(value, np.ndarray):
            out[key] = np.asarray(value, dtype=np.float32).copy()
        elif key in {"mesh_path", "mesh_name", "kind", "geom_type"}:
            out[key] = str(value)
        else:
            out[key] = value
    return out


def _make_target_center_lookup(object_meta):
    meta_by_oid = {
        int(item.get("object_id", i)): item
        for i, item in enumerate(object_meta or [])
        if isinstance(item, dict)
    }
    center_cache = {}

    def _target_center_for_oid(oid):
        meta = meta_by_oid.get(int(oid), {})
        asset_path = str(meta.get("asset_path", "") or "").replace("\\", "/").lower()
        # Some repaired/rotated Libero-merge assets (e.g. microwave) need to
        # follow the checkpoint metadata center, because standalone SDF eval and
        # latent canonical visualization already rely on that convention.
        if asset_path.endswith("/microwave_combined_p_scaled.obj") or asset_path.endswith("/microwave_door_combined_p_scaled.obj"):
            center = _latent_center_from_meta(meta)
            if center is not None:
                return center
        surface_center = _surface_center_from_npz_path(meta.get("npz_path", ""), center_cache)
        if surface_center is not None:
            return surface_center
        return _latent_center_from_meta(meta)

    return meta_by_oid, _target_center_for_oid


def build_precomputed_latent_mapping(
    entity_specs,
    gripper_spec,
    latent_ckpt,
    object_meta=None,
):
    object_meta = object_meta if object_meta is not None else load_latent_object_meta(latent_ckpt)
    if not object_meta:
        return {}

    wm_root = PROJECT_ROOT / "WM"
    if str(wm_root) not in sys.path:
        sys.path.append(str(wm_root))

    from train import LatentMatcher, build_latent_entries, expand_aliases, match_entity_to_object_id

    matcher = LatentMatcher(build_latent_entries(object_meta), min_score=8.0)
    meta_by_oid, target_center_for_oid = _make_target_center_lookup(object_meta)
    merge_token_blacklist = {"combined", "merge", "libero", "p", "scaled"}
    generic_root_tokens = {"base", "body", "main", "core", "object"}

    def _assign_mapping(entity_name, spec, object_id, match_kind):
        oid = int(object_id)
        meta = meta_by_oid.get(oid, {})
        target_center = target_center_for_oid(oid)
        source_center = _spec_surface_center(spec)
        offset_local = (target_center - source_center.reshape(3)).astype(np.float32, copy=False)
        mapping[entity_name] = {
            "object_id": oid,
            "target_npz_name": _basename_npz(meta.get("npz_path", "")),
            "pose_source": entity_name,
            "pose_offset_local": offset_local,
            "body_name": str(spec.get("body_name", "")),
            "instance": str(spec.get("instance", "")),
            "category": str(spec.get("category", "")),
            "part": str(spec.get("part", "")),
            "match_kind": str(match_kind),
        }

    def _combined_candidates_for_category(category_name):
        cat_norm = canonical_name(category_name)
        if not cat_norm:
            return []
        cat_aliases = expand_aliases(cat_norm)
        cat_aliases.add(cat_norm)
        category_tokens = set()
        for alias in cat_aliases:
            category_tokens.update(name_tokens(alias))

        out = []
        for entry in matcher.entries:
            if not _is_combined_latent_entry(entry):
                continue
            etoks = _entry_tokens(entry)
            entry_names = {canonical_name(name) for name in getattr(entry, "names", ())}
            stem = _entry_asset_stem(entry)
            if not (
                cat_aliases.intersection(entry_names)
                or category_tokens.intersection(etoks)
                or any(stem == alias or stem.startswith(f"{alias}_") for alias in cat_aliases)
            ):
                continue
            semantic_tokens = set(name_tokens(stem)).union(etoks)
            semantic_tokens.difference_update(category_tokens)
            semantic_tokens.difference_update(merge_token_blacklist)
            semantic_tokens.difference_update(TOKEN_BLACKLIST)
            is_full_category_merge = any(
                stem == alias or stem.startswith(f"{alias}_combined") or stem.startswith(f"{alias}_p_scaled")
                for alias in cat_aliases
            )
            out.append(
                {
                    "object_id": int(entry.object_id),
                    "semantic_tokens": semantic_tokens,
                    "full_category_merge": bool(is_full_category_merge),
                }
            )
        return out

    def _best_combined_oid_for_group(category_name, specs):
        candidates = _combined_candidates_for_category(category_name)
        if not candidates:
            return None

        cat_norm = canonical_name(category_name)
        category_tokens = set()
        for alias in expand_aliases(cat_norm) | {cat_norm}:
            category_tokens.update(name_tokens(alias))

        query_tokens = set()
        for spec in specs:
            part_text = str(spec.get("part", ""))
            body_text = str(spec.get("body_name", ""))
            query_tokens.update(name_tokens(part_text))
            query_tokens.update(name_tokens(body_text))
            raw_norm = canonical_name(part_text) + "_" + canonical_name(body_text)
            for semantic_hint in ("door", "handle", "button", "burner", "drawer", "knob"):
                if semantic_hint in raw_norm:
                    query_tokens.add(semantic_hint)
        query_tokens = {tok for tok in query_tokens if tok and not str(tok).isdigit()}
        query_tokens.difference_update(category_tokens)
        query_tokens.difference_update(merge_token_blacklist)
        query_tokens.difference_update(TOKEN_BLACKLIST)

        best_oid = None
        best_score = -1e9
        for cand in candidates:
            semantic_tokens = set(cand["semantic_tokens"])
            score = -1e9
            overlap = query_tokens.intersection(semantic_tokens)
            if overlap:
                score = 25.0 * float(len(overlap))
                if semantic_tokens and overlap == semantic_tokens:
                    score += 4.0
                if semantic_tokens and semantic_tokens.issubset(query_tokens.union(generic_root_tokens)):
                    score += 2.0
            elif cand["full_category_merge"] and (not query_tokens or query_tokens.issubset(generic_root_tokens)):
                score = 12.0
            if score > best_score:
                best_score = score
                best_oid = int(cand["object_id"]) if score > -1e8 else None
        if best_oid is None or best_score < 10.0:
            return None
        return best_oid

    mapping = {}

    for entity_name, spec in entity_specs.items():
        instance_name = str(spec.get("instance", ""))
        part_name = str(spec.get("part", ""))
        category_name = str(spec.get("category", ""))
        oid, _score = match_entity_to_object_id(
            matcher=matcher,
            entity_name=entity_name,
            instance_name=instance_name,
            category_name=category_name,
            part_name=part_name,
        )
        if oid is not None:
            _assign_mapping(entity_name, spec, oid, match_kind="direct")

    grouped_specs = {}
    for entity_name, spec in entity_specs.items():
        group_key = (
            str(spec.get("instance", "")),
            str(spec.get("category", "")),
            str(spec.get("body_name", "")),
        )
        grouped_specs.setdefault(group_key, []).append((entity_name, spec))

    remapped_combined = 0
    for (_instance_name, category_name, _body_name), members in grouped_specs.items():
        member_specs = [spec for _, spec in members]
        best_combined_oid = _best_combined_oid_for_group(category_name, member_specs)
        if best_combined_oid is None:
            continue
        for entity_name, spec in members:
            prev = mapping.get(entity_name, {})
            prev_oid = int(prev.get("object_id", -1)) if prev else -1
            if prev_oid != int(best_combined_oid):
                remapped_combined += 1
            _assign_mapping(entity_name, spec, best_combined_oid, match_kind="combined_same_body")

    if remapped_combined > 0:
        print(f"  Combined latent remaps applied: {remapped_combined}")

    if gripper_spec is not None:
        for gspec in list(gripper_spec.get("entities", [])):
            entity_name = str(gspec.get("entity_name", ""))
            if not entity_name:
                continue
            part_name = str(gspec.get("part", ""))
            oid, _score = match_entity_to_object_id(
                matcher=matcher,
                entity_name=entity_name,
                instance_name="robot_gripper",
                category_name="robot_gripper",
                part_name=part_name,
            )
            if oid is None:
                continue
            spec = {
                "instance": "robot_gripper",
                "category": "robot_gripper",
                "part": part_name,
                "body_name": str(gspec.get("body_name", "")),
                "surface_center": _spec_surface_center(gspec),
            }
            _assign_mapping(entity_name, spec, oid, match_kind="direct_gripper")

    return mapping


def apply_target_surface_centers(entity_specs, gripper_spec, latent_mapping, object_meta):
    if not object_meta:
        return

    _meta_by_oid, target_center_for_oid = _make_target_center_lookup(object_meta)
    gripper_by_name = {}
    if gripper_spec is not None:
        gripper_by_name = {
            str(gspec.get("entity_name", "")): gspec for gspec in list(gripper_spec.get("entities", []))
        }

    aligned_count = 0
    fallback_count = 0
    for entity_name, spec in entity_specs.items():
        info = latent_mapping.get(entity_name, None)
        pts = np.asarray(spec.get("points", None), dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] == 0:
            spec["surface_center"] = np.zeros((3,), dtype=np.float32)
            continue
        target_center = None
        if info is not None:
            target_center = target_center_for_oid(info["object_id"])
        if target_center is None:
            target_center = pts.mean(axis=0).astype(np.float32, copy=False)
            fallback_count += 1
        else:
            aligned_count += 1
        spec["surface_center"] = np.asarray(target_center, dtype=np.float32).reshape(3)
        spec["points"] = (pts - spec["surface_center"].reshape(1, 3)).astype(np.float32, copy=False)

    for entity_name, gspec in gripper_by_name.items():
        pts = np.asarray(gspec.get("points", None), dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] == 0:
            gspec["surface_center"] = np.zeros((3,), dtype=np.float32)
            continue
        # Use body-local centroid (not target_center from NPZ) so that the stored
        # PCD and pose_pos_aligned (= body_pos when trans_local=0) are consistent.
        center = pts.mean(axis=0).astype(np.float32, copy=False)
        gspec["surface_center"] = center
        gspec["points"] = (pts - center.reshape(1, 3)).astype(np.float32, copy=False)

    if aligned_count > 0 or fallback_count > 0:
        print(
            f"  Applied target-centroid recentering: aligned={aligned_count}, fallback={fallback_count}"
        )

def _combined_representative_name(members, entity_specs):
    deprioritize = {"handle", "button", "knob", "plate", "finger"}

    def _score(name):
        spec = entity_specs.get(name, {})
        part = canonical_name(spec.get("part", ""))
        toks = set(name_tokens(part)) | set(name_tokens(name))
        penalty = sum(1 for tok in toks if tok in deprioritize)
        return (penalty, len(name), name)

    return sorted(members, key=_score)[0]


def collapse_combined_entity_specs(entity_specs, latent_mapping, num_points_per_entity):
    if not entity_specs or not latent_mapping:
        return entity_specs, latent_mapping

    grouped = {}
    for entity_name in entity_specs.keys():
        info = latent_mapping.get(entity_name, None)
        if info is None:
            continue
        match_kind = str(info.get("match_kind", ""))
        if not match_kind.startswith("combined"):
            continue
        group_key = (
            str(info.get("instance", "")),
            int(info.get("object_id", -1)),
        )
        grouped.setdefault(group_key, []).append(entity_name)

    if not grouped:
        return entity_specs, latent_mapping

    collapsed_specs = {}
    collapsed_mapping = {}
    consumed = set()

    for group_key, members in grouped.items():
        if len(members) <= 1:
            name = members[0]
            collapsed_specs[name] = entity_specs[name]
            collapsed_mapping[name] = latent_mapping[name]
            consumed.add(name)
            continue

        rep_name = _combined_representative_name(members, entity_specs)
        merged_points = []
        for name in members:
            pts = np.asarray(entity_specs[name].get("points", None), dtype=np.float32)
            if pts.ndim == 2 and pts.shape[1] == 3 and pts.shape[0] > 0:
                merged_points.append(pts)
            consumed.add(name)
        if not merged_points:
            collapsed_specs[rep_name] = dict(entity_specs[rep_name])
            collapsed_mapping[rep_name] = dict(latent_mapping[rep_name])
            continue

        merged = np.vstack(merged_points)
        merged = normalize_point_count(merged, num_points_per_entity).astype(np.float32, copy=False)
        merged_raw_points = []
        for name in members:
            raw_pts = np.asarray(entity_specs[name].get("raw_points", None), dtype=np.float32)
            if raw_pts.ndim == 2 and raw_pts.shape[1] == 3 and raw_pts.shape[0] > 0:
                merged_raw_points.append(raw_pts)
        merged_raw = None
        if merged_raw_points:
            merged_raw = normalize_point_count(np.vstack(merged_raw_points), num_points_per_entity).astype(np.float32, copy=False)
        rep_spec = dict(entity_specs[rep_name])
        rep_spec["points"] = merged
        if merged_raw is not None:
            rep_spec["raw_points"] = merged_raw
        merged_source_items = []
        for name in members:
            for item in entity_specs[name].get("source_items", []) or []:
                merged_source_items.append(_copy_source_item(item))
        if merged_source_items:
            rep_spec["source_items"] = merged_source_items
        rep_spec["surface_center"] = np.asarray(rep_spec.get("surface_center", np.zeros((3,), dtype=np.float32)), dtype=np.float32)
        collapsed_specs[rep_name] = rep_spec

        rep_map = dict(latent_mapping[rep_name])
        rep_map["pose_source"] = rep_name
        collapsed_mapping[rep_name] = rep_map

    for entity_name, spec in entity_specs.items():
        if entity_name in consumed:
            continue
        collapsed_specs[entity_name] = spec
        if entity_name in latent_mapping:
            collapsed_mapping[entity_name] = latent_mapping[entity_name]

    # Preserve mappings for entities that are not part of entity_specs, such as
    # gripper parts carried separately in gripper_spec.
    for entity_name, info in latent_mapping.items():
        if entity_name not in collapsed_mapping and entity_name not in entity_specs:
            collapsed_mapping[entity_name] = info

    if len(collapsed_specs) != len(entity_specs):
        print(f"  Collapsed combined entities: {len(entity_specs)} -> {len(collapsed_specs)}")
    return collapsed_specs, collapsed_mapping


def parse_scale(scale_str):
    if not scale_str:
        return np.array([1.0, 1.0, 1.0], dtype=np.float64)
    vals = [float(x) for x in scale_str.split()]
    if len(vals) == 1:
        vals = [vals[0], vals[0], vals[0]]
    return np.array(vals[:3], dtype=np.float64)


def parse_vec(vec_str, default):
    if not vec_str:
        return np.array(default, dtype=np.float64)
    vals = [float(x) for x in vec_str.split()]
    return np.array(vals, dtype=np.float64)


def sanitize_name(name):
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name.strip().lower())


def canonical_name(text):
    if text is None:
        return ""
    return re.sub(r"_+", "_", re.sub(r"[^a-zA-Z0-9]+", "_", str(text).strip().lower())).strip("_")


def name_tokens(text):
    norm = canonical_name(text)
    if not norm:
        return []
    return [t for t in norm.split("_") if t]


def _strip_mesh_suffix(name):
    out = canonical_name(name)
    for suf in ("_visual", "_vis", "_collision", "_coll", "_mesh"):
        if out.endswith(suf):
            out = out[: -len(suf)]
    return out.strip("_")


def infer_mesh_subpart_name(mesh_name, base_part, obj_category):
    mesh_norm = _strip_mesh_suffix(mesh_name)
    if not mesh_norm:
        return canonical_name(base_part)

    mesh_toks = name_tokens(mesh_norm)
    base_norm = canonical_name(base_part)
    base_toks = name_tokens(base_norm)
    cat_toks = name_tokens(obj_category)

    # Remove leading category prefix, e.g. wooden_cabinet_top_handle -> top_handle.
    if cat_toks and len(mesh_toks) >= len(cat_toks) and mesh_toks[: len(cat_toks)] == cat_toks:
        mesh_toks = mesh_toks[len(cat_toks) :]
    if not mesh_toks:
        return base_norm

    if not base_toks:
        return "_".join(mesh_toks)
    if mesh_toks == base_toks:
        return base_norm
    if len(mesh_toks) >= len(base_toks) and mesh_toks[: len(base_toks)] == base_toks:
        return "_".join(mesh_toks)

    shared_idx = [i for i, t in enumerate(mesh_toks) if t in base_toks]
    if shared_idx:
        first_shared = min(shared_idx)
        suffix_extras = [mesh_toks[i] for i in range(first_shared + 1, len(mesh_toks)) if mesh_toks[i] not in base_toks]
        if suffix_extras:
            return f"{base_norm}_{'_'.join(suffix_extras)}"
    return base_norm


def split_parts_by_mesh_subparts(parts, obj_category):
    """
    Split object XML parts to mesh-granularity entities.

    Behavior:
    - each mesh geom item becomes an independent part entry (no mesh merge)
    - repeated inferred names are suffixed as _2, _3, ...
    - non-mesh primitive geoms stay grouped by base part
    """
    out = {}
    for base_part, mesh_items in parts.items():
        base_norm = canonical_name(base_part) or "object"
        name_counts = {}
        primitive_items = []

        for item in mesh_items:
            if item.get("kind") == "mesh":
                mesh_name = item.get("mesh_name", "")
                sub_name = infer_mesh_subpart_name(mesh_name, base_part=base_part, obj_category=obj_category)
                sub_norm = canonical_name(sub_name) or base_norm

                occ = int(name_counts.get(sub_norm, 0)) + 1
                name_counts[sub_norm] = occ
                target = sub_norm if occ == 1 else f"{sub_norm}_{occ}"
                out[target] = [item]
            else:
                primitive_items.append(item)

        if primitive_items:
            target = base_norm
            if target in out:
                k = 1
                while f"{base_norm}_primitive_{k}" in out:
                    k += 1
                target = f"{base_norm}_primitive_{k}"
            out[target] = primitive_items

    return out


def mesh_name_with_category_alias(mesh_name, src_category, dst_category):
    mesh_norm = sanitize_name(mesh_name)
    src_norm = sanitize_name(src_category)
    dst_norm = sanitize_name(dst_category)
    if not mesh_norm or not src_norm or not dst_norm or src_norm == dst_norm:
        return mesh_norm
    prefix = f"{src_norm}_"
    if mesh_norm.startswith(prefix):
        return f"{dst_norm}_{mesh_norm[len(prefix):]}"
    return mesh_norm


def mesh_name_matches_allowed(mesh_name, allowed_names):
    mesh_norm = sanitize_name(mesh_name)
    if not allowed_names:
        return True
    if mesh_norm in allowed_names:
        return True
    for allowed in allowed_names:
        if mesh_norm.endswith(f"_{allowed}"):
            return True
    return False


def find_object_xml(object_type):
    obj_norm = canonical_name(object_type)
    query_order = []
    alias = CATEGORY_XML_ALIASES.get(obj_norm, "")
    if alias:
        query_order.append(alias)
    if obj_norm:
        query_order.append(obj_norm)

    priority = [
        "stable_scanned_objects",
        "stable_hope_objects",
        "turbosquid_objects",
        "articulated_objects",
    ]

    def candidate_key(path):
        path_str = str(path)
        pri = len(priority)
        for i, marker in enumerate(priority):
            if marker in path_str:
                pri = i
                break
        return (pri, len(path_str))

    for q in query_order:
        candidates = list(ASSETS_BASE_PATH.rglob(f"{q}.xml"))
        if candidates:
            return sorted(candidates, key=candidate_key)[0]
    return None


def resolve_mesh_file(xml_path, mesh_file, meshdir):
    xml_dir = xml_path.parent
    mesh_roots = [xml_dir]
    if meshdir:
        mesh_roots.insert(0, (xml_dir / meshdir).resolve())

    # Handle asset paths that are relative to assets root.
    mesh_roots.append(ASSETS_BASE_PATH)

    mesh_path = Path(mesh_file)
    preferred_relatives = [mesh_path]
    # Keep the XML-declared mesh first. If that format cannot be sampled directly
    # (for example .msh), the caller will fall back to MuJoCo model meshes, which
    # match the simulator exactly. Alternate OBJ/COL files are only secondary
    # fallbacks and should not silently override the declared visual mesh.
    if mesh_path.suffix.lower() == ".msh":
        preferred_relatives.append(mesh_path.with_suffix(".obj"))
        preferred_relatives.append(Path(mesh_path.name).with_suffix(".obj"))
        preferred_relatives.append(Path(mesh_path.stem.replace("_vis", "")).with_suffix(".obj"))
        preferred_relatives.append(Path(mesh_path.stem.replace("_vis", "") + "_col.obj"))
        if len(mesh_path.parts) >= 2 and mesh_path.parts[-2] == "visual":
            parent_dir = Path(*mesh_path.parts[:-2])
            base_name = mesh_path.stem.replace("_vis", "")
            preferred_relatives.append(parent_dir / f"{base_name}.obj")
            preferred_relatives.append(parent_dir / f"{base_name}_col.obj")
            parent_leaf = parent_dir.name
            preferred_relatives.append(parent_dir / f"{parent_leaf}.obj")
            preferred_relatives.append(parent_dir / f"{parent_leaf}_col.obj")

    for root in mesh_roots:
        for rel in preferred_relatives:
            candidate = (root / rel).resolve()
            if candidate.exists() and candidate.suffix.lower() in MESH_EXTENSIONS:
                return candidate

    return None


def part_name_aliases(part_name):
    aliases = {part_name.lower()}
    cleaned = part_name.lower()
    for prefix in ["cabinet_", "drawer_", "door_", "stove_", "knob_"]:
        if cleaned.startswith(prefix):
            aliases.add(cleaned[len(prefix) :])
    tokens = [t for t in cleaned.split("_") if t]
    aliases.update(tokens)
    if len(tokens) >= 2:
        aliases.add("_".join(tokens[-2:]))
    # Remove overly generic tokens that cause wrong cross-part matches.
    generic = {"cabinet", "drawer", "door", "stove", "knob", "object"}
    return {a for a in aliases if a and a not in generic}


def apply_scale_to_mesh(mesh, scale):
    if np.allclose(scale, np.array([1.0, 1.0, 1.0])):
        return
    if isinstance(mesh, trimesh.Scene):
        for geom in mesh.geometry.values():
            if isinstance(geom, trimesh.Trimesh):
                geom.apply_scale(scale)
    else:
        mesh.apply_scale(scale)


def load_mesh_with_cache(mesh_path, scale, mesh_cache):
    key = (str(mesh_path), tuple(float(x) for x in scale))
    if key in mesh_cache:
        return mesh_cache[key]

    mesh = trimesh.load(str(mesh_path), process=False)
    apply_scale_to_mesh(mesh, scale)
    mesh_cache[key] = mesh
    return mesh


def parse_object_parts_from_xml(xml_path):
    """
    Returns a dict:
    {
        part_name: [
            {"mesh_path": Path, "scale": np.array([sx,sy,sz])},
            ...
        ],
        ...
    }
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    compiler = root.find("compiler")
    meshdir = compiler.get("meshdir") if compiler is not None else ""

    mesh_assets = {}
    for mesh_el in root.iter("mesh"):
        mesh_name = mesh_el.get("name")
        mesh_file = mesh_el.get("file")
        if not mesh_name or not mesh_file:
            continue
        resolved = resolve_mesh_file(xml_path, mesh_file, meshdir)
        if resolved is None:
            continue
        mesh_assets[mesh_name] = {
            "mesh_path": resolved,
            "scale": parse_scale(mesh_el.get("scale", "1 1 1")),
        }

    parts = {}

    primitive_types = {"box", "sphere", "cylinder", "capsule", "ellipsoid"}

    def walk_body(body_el, current_named_body=""):
        body_name = body_el.get("name")
        active_body = body_name if body_name else current_named_body

        for geom in body_el.findall("geom"):
            if not active_body:
                active_body = "object"
            gtype = geom.get("type", "")
            mesh_name = geom.get("mesh")

            # Some LIBERO XMLs rely on default classes, so mesh geoms may omit type="mesh".
            # If a geom references a mesh asset, treat it as a mesh geom.
            if gtype == "mesh" or (not gtype and mesh_name):
                if mesh_name not in mesh_assets:
                    continue
                geom_pos = parse_vec(geom.get("pos", ""), [0.0, 0.0, 0.0])
                geom_quat = parse_vec(geom.get("quat", ""), [1.0, 0.0, 0.0, 0.0])  # wxyz
                parts.setdefault(active_body, []).append(
                    {
                        "kind": "mesh",
                        "mesh_name": mesh_name,
                        **mesh_assets[mesh_name],
                        "pos": geom_pos,
                        "quat": geom_quat,
                    }
                )
                continue

            # Include visual primitive geoms (group=1) to preserve object shape,
            # e.g. flat_stove base is rendered as box geoms instead of mesh.
            if gtype in primitive_types and geom.get("group", "") == "1":
                size = parse_vec(geom.get("size", ""), [0.0, 0.0, 0.0])
                pos = parse_vec(geom.get("pos", ""), [0.0, 0.0, 0.0])
                quat = parse_vec(geom.get("quat", ""), [1.0, 0.0, 0.0, 0.0])  # wxyz
                parts.setdefault(active_body, []).append(
                    {
                        "kind": "primitive",
                        "geom_type": gtype,
                        "size": size,
                        "pos": pos,
                        "quat": quat,
                    }
                )

        for child_body in body_el.findall("body"):
            walk_body(child_body, active_body)

    for worldbody in root.findall("worldbody"):
        for body in worldbody.findall("body"):
            walk_body(body, "")

    return parts


def sample_points_from_mesh(mesh, num_points):
    if mesh is None or num_points <= 0:
        return None

    try:
        if isinstance(mesh, trimesh.Scene):
            geoms = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh) and len(g.faces) > 0]
            if not geoms:
                return None
            per_geom = max(1, num_points // len(geoms))
            sampled = []
            for geom in geoms:
                pts, _ = trimesh.sample.sample_surface(geom, per_geom)
                sampled.append(pts)
            points = np.vstack(sampled)
        else:
            if len(mesh.faces) == 0:
                return None
            points, _ = trimesh.sample.sample_surface(mesh, num_points)

        if points.shape[0] > num_points:
            idx = np.random.choice(points.shape[0], num_points, replace=False)
            points = points[idx]
        elif points.shape[0] < num_points:
            pad = np.zeros((num_points - points.shape[0], 3), dtype=np.float32)
            points = np.vstack([points, pad])

        return points.astype(np.float32)
    except Exception as e:
        print(f"Error sampling points: {e}")
        return None


def sample_points_from_primitive(geom_type, size, num_points):
    if num_points <= 0:
        return None

    try:
        if geom_type == "box":
            # MuJoCo size is half-extent.
            ext = 2.0 * np.array(size[:3], dtype=np.float64)
            mesh = trimesh.creation.box(extents=ext)
        elif geom_type == "sphere":
            mesh = trimesh.creation.icosphere(subdivisions=2, radius=float(size[0]))
        elif geom_type == "cylinder":
            # MuJoCo cylinder size: [radius, half_height]
            mesh = trimesh.creation.cylinder(radius=float(size[0]), height=2.0 * float(size[1]), sections=32)
        elif geom_type == "capsule":
            # MuJoCo capsule size: [radius, half_height]
            mesh = trimesh.creation.capsule(radius=float(size[0]), height=2.0 * float(size[1]), count=[16, 16])
        elif geom_type == "ellipsoid":
            # MuJoCo ellipsoid size: half-axes.
            base = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
            base.apply_scale(np.array(size[:3], dtype=np.float64))
            mesh = base
        else:
            return None

        pts, _ = trimesh.sample.sample_surface(mesh, num_points)
        return pts.astype(np.float32)
    except Exception:
        return None


def transform_point_cloud(points, position, quaternion_wxyz):
    if points is None:
        return None

    q = np.array(quaternion_wxyz, dtype=np.float64)
    q_norm = np.linalg.norm(q)
    if q_norm < 1e-12:
        q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    else:
        q = q / q_norm

    w, x, y, z = q
    rot = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])

    transformed = (rot @ points.T).T + np.array(position, dtype=np.float64)
    return transformed.astype(np.float32)


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


def quat_to_rotmat_wxyz(quat_wxyz):
    w, x, y, z = normalize_quaternion_wxyz(quat_wxyz)
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def align_points_to_latent_local_frame(points_raw_local, trans_local, rot_wxyz):
    if points_raw_local is None:
        return None
    pts = np.asarray(points_raw_local, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        return np.asarray(points_raw_local, dtype=np.float32)
    rot = quat_to_rotmat_wxyz(rot_wxyz)
    trans = np.asarray(trans_local, dtype=np.float64).reshape(1, 3)
    aligned = (pts - trans) @ rot
    return aligned.astype(np.float32, copy=False)


def gripper_pose_quat_offset_wxyz(part_name, body_name):
    part_l = str(part_name).strip().lower()
    body_l = str(body_name).strip().lower()
    # Force right finger orientation to be opposite so shared finger latent
    # can be disambiguated by pose rotation.
    if part_l == "right_finger" or "rightfinger" in body_l:
        return np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)  # 180 deg around +Z
    return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)



def _axis_aligned_rotation_candidates():
    mats = []
    eye = np.eye(3, dtype=np.float64)
    import itertools
    for perm in itertools.permutations(range(3)):
        base = eye[list(perm)]
        for signs in itertools.product([-1.0, 1.0], repeat=3):
            mat = (np.asarray(signs, dtype=np.float64).reshape(3, 1) * base).astype(np.float64)
            if np.linalg.det(mat) > 0.5:
                mats.append(mat)
    return mats


_AXIS_ROT_CANDIDATES = _axis_aligned_rotation_candidates()


def _nearest_neighbor_indices(points_query, points_ref):
    q = np.asarray(points_query, dtype=np.float64)
    r = np.asarray(points_ref, dtype=np.float64)
    if q.ndim != 2 or r.ndim != 2 or q.shape[0] == 0 or r.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float64)
    if cKDTree is not None:
        tree = cKDTree(r)
        try:
            dists, idx = tree.query(q, k=1, workers=-1)
        except TypeError:
            dists, idx = tree.query(q, k=1)
        return np.asarray(idx, dtype=np.int64).reshape(-1), np.asarray(dists, dtype=np.float64).reshape(-1)
    diff = q[:, None, :] - r[None, :, :]
    d2 = np.einsum('ijk,ijk->ij', diff, diff)
    idx = np.argmin(d2, axis=1)
    dists = np.sqrt(np.take_along_axis(d2, idx[:, None], axis=1).reshape(-1))
    return idx.astype(np.int64, copy=False), dists.astype(np.float64, copy=False)


def _mean_bidirectional_nn_distance(points_a, points_b):
    _, d_ab = _nearest_neighbor_indices(points_a, points_b)
    _, d_ba = _nearest_neighbor_indices(points_b, points_a)
    if d_ab.size == 0 and d_ba.size == 0:
        return np.inf
    vals = []
    if d_ab.size > 0:
        vals.append(float(d_ab.mean()))
    if d_ba.size > 0:
        vals.append(float(d_ba.mean()))
    return float(np.mean(vals)) if vals else np.inf


def _solve_row_orthogonal_procrustes(source_points, target_points):
    src = np.asarray(source_points, dtype=np.float64)
    tgt = np.asarray(target_points, dtype=np.float64)
    if src.ndim != 2 or tgt.ndim != 2 or src.shape != tgt.shape or src.shape[0] == 0:
        return np.eye(3, dtype=np.float64)
    src_c = src - src.mean(axis=0, keepdims=True)
    tgt_c = tgt - tgt.mean(axis=0, keepdims=True)
    mat = src_c.T @ tgt_c
    U, _, Vt = np.linalg.svd(mat, full_matrices=False)
    rot_row = U @ Vt
    if np.linalg.det(rot_row) < 0.0:
        U[:, -1] *= -1.0
        rot_row = U @ Vt
    return rot_row.astype(np.float64, copy=False)


def estimate_latent_pose_rotation(points_body_local, template_points_local, init_rot_wxyz=None, max_iters=12):
    src = np.asarray(points_body_local, dtype=np.float64)
    tgt = np.asarray(template_points_local, dtype=np.float64)
    if src.ndim != 2 or tgt.ndim != 2 or src.shape[0] == 0 or tgt.shape[0] == 0:
        rot = quat_to_rotmat_wxyz(init_rot_wxyz if init_rot_wxyz is not None else [1.0, 0.0, 0.0, 0.0])
        return rotmat_to_quat_wxyz(rot), np.inf

    src_centered = src - src.mean(axis=0, keepdims=True)
    tgt_centered = tgt - tgt.mean(axis=0, keepdims=True)

    candidate_rots = []
    if init_rot_wxyz is not None:
        candidate_rots.append(quat_to_rotmat_wxyz(init_rot_wxyz))
    candidate_rots.extend(_AXIS_ROT_CANDIDATES)

    best_rot = np.eye(3, dtype=np.float64)
    best_err = np.inf
    for rot in candidate_rots:
        transformed = tgt_centered @ rot.T
        err = _mean_bidirectional_nn_distance(transformed, src_centered)
        if err < best_err:
            best_err = err
            best_rot = np.asarray(rot, dtype=np.float64)

    cur_rot = best_rot
    for _ in range(max_iters):
        transformed = tgt_centered @ cur_rot.T
        idx, _ = _nearest_neighbor_indices(transformed, src_centered)
        if idx.size == 0:
            break
        matched = src_centered[idx]
        new_rot_row = _solve_row_orthogonal_procrustes(tgt_centered, matched)
        new_rot = new_rot_row.T
        if np.linalg.norm(new_rot - cur_rot) < 1e-7:
            cur_rot = new_rot
            break
        cur_rot = new_rot

    final_err = _mean_bidirectional_nn_distance(tgt_centered @ cur_rot.T, src_centered)
    return rotmat_to_quat_wxyz(cur_rot), final_err


def normalize_point_count(points, target_num):
    if points is None or target_num <= 0:
        return np.zeros((max(target_num, 0), 3), dtype=np.float32)

    pts = np.asarray(points, dtype=np.float32)
    if pts.shape[0] == target_num:
        return pts

    if pts.shape[0] > target_num:
        idx = np.linspace(0, pts.shape[0] - 1, num=target_num, dtype=np.int64)
        return pts[idx]

    repeat = int(np.ceil(target_num / max(1, pts.shape[0])))
    tiled = np.tile(pts, (repeat, 1))
    return tiled[:target_num]


def distribute_counts(total, bins):
    if bins <= 0:
        return []
    base = total // bins
    rem = total % bins
    return [base + (1 if i < rem else 0) for i in range(bins)]

def rotmat_to_quat_wxyz(rot):
    r = np.asarray(rot, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(r))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (r[2, 1] - r[1, 2]) / s
        y = (r[0, 2] - r[2, 0]) / s
        z = (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        w = (r[2, 1] - r[1, 2]) / s
        x = 0.25 * s
        y = (r[0, 1] + r[1, 0]) / s
        z = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        w = (r[0, 2] - r[2, 0]) / s
        x = (r[0, 1] + r[1, 0]) / s
        y = 0.25 * s
        z = (r[1, 2] + r[2, 1]) / s
    else:
        s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        w = (r[1, 0] - r[0, 1]) / s
        x = (r[0, 2] + r[2, 0]) / s
        y = (r[1, 2] + r[2, 1]) / s
        z = 0.25 * s
    return normalize_quaternion_wxyz(np.asarray([w, x, y, z], dtype=np.float64))


def build_latent_pose_alignment(entity_specs, gripper_spec, latent_mapping, object_meta, num_points_per_entity):
    if not latent_mapping or not object_meta:
        return {}

    meta_by_oid, target_center_for_oid = _make_target_center_lookup(object_meta)
    template_cache = {}
    out = {}

    def _raw_center(spec):
        raw_points = np.asarray(spec.get("raw_points", None), dtype=np.float32)
        if raw_points.ndim == 2 and raw_points.shape[1] == 3 and raw_points.shape[0] > 0:
            return raw_points.mean(axis=0).astype(np.float32, copy=False)
        return np.asarray(spec.get("surface_center", np.zeros((3,), dtype=np.float32)), dtype=np.float32).reshape(3)

    def _template_for_oid(oid):
        meta = meta_by_oid.get(int(oid), {})
        return _centered_surface_points_from_npz_path(
            meta.get("npz_path", ""),
            template_cache,
            target_num=num_points_per_entity,
        )

    def _fit_error_for_entry(raw_points, template_points, trans_local, rot_wxyz):
        if raw_points.ndim != 2 or raw_points.shape[1] != 3 or raw_points.shape[0] == 0:
            return 0.0
        if template_points is None:
            return 0.0
        aligned_raw = align_points_to_latent_local_frame(raw_points, trans_local, rot_wxyz)
        return float(_mean_bidirectional_nn_distance(aligned_raw, template_points))

    def _analytic_alignment(spec, info, init_rot_wxyz):
        oid = int(info.get("object_id", -1))
        match_kind = str(info.get("match_kind", ""))
        meta = meta_by_oid.get(oid, {})
        target_center = target_center_for_oid(oid)
        if target_center is None:
            return None
        source_items = list(spec.get("source_items", []) or [])
        asset_path = str(meta.get("asset_path", "") or "").replace("\\", "/").lower()
        raw_center = _raw_center(spec)

        if match_kind == "direct_gripper":
            trans_local = raw_center.astype(np.float32, copy=False)
            rot_wxyz = np.asarray(init_rot_wxyz, dtype=np.float32).reshape(4)
            return trans_local, rot_wxyz, "analytic-gripper-raw"

        if len(source_items) == 1:
            item = source_items[0]
            item_pos = np.asarray(item.get("pos", [0.0, 0.0, 0.0]), dtype=np.float32).reshape(3)
            item_quat = np.asarray(item.get("quat", [1.0, 0.0, 0.0, 0.0]), dtype=np.float32).reshape(4)
            item_kind = str(item.get("kind", ""))
            mesh_path = str(item.get("mesh_path", "") or "").lower()
            if item_kind == "mesh" and not mesh_path:
                trans_local = raw_center.astype(np.float32, copy=False)
                rot_wxyz = np.asarray(init_rot_wxyz, dtype=np.float32).reshape(4)
                return trans_local, rot_wxyz, "analytic-gripper-raw"

            if item_kind == "geom" or mesh_path.endswith('.msh'):
                trans_local = transform_point_cloud(target_center.reshape(1, 3), item_pos, item_quat)[0].astype(np.float32, copy=False)
                rot_wxyz = quat_mul_wxyz(item_quat, init_rot_wxyz).astype(np.float32)
                return trans_local, rot_wxyz, "analytic-single-item-raw-center"
            trans_local = transform_point_cloud(target_center.reshape(1, 3), item_pos, item_quat)[0].astype(np.float32, copy=False)
            rot_wxyz = quat_mul_wxyz(item_quat, init_rot_wxyz).astype(np.float32)
            return trans_local, rot_wxyz, "analytic-single-item"

        _is_libero_merge = match_kind.startswith("combined") and (
            "/libero-merge/" in asset_path or "/libero_merge/" in asset_path
        )
        if not _is_libero_merge and source_items and all(
            str(item.get("mesh_path", "") or "").lower().endswith(".msh")
            or str(item.get("kind", "")) == "geom"
            for item in source_items
        ):
            primary = next(
                (
                    item for item in source_items
                    if str(item.get("kind", "")) == "mesh"
                    and "handle" not in str(item.get("mesh_name", "") or item.get("mesh_path", "") or "").lower()
                ),
                source_items[0],
            )
            p_pos = np.asarray(primary.get("pos", [0.0, 0.0, 0.0]), dtype=np.float32).reshape(3)
            p_quat = np.asarray(primary.get("quat", [1.0, 0.0, 0.0, 0.0]), dtype=np.float32).reshape(4)
            trans_local = transform_point_cloud(target_center.reshape(1, 3), p_pos, p_quat)[0].astype(np.float32, copy=False)
            rot_wxyz = quat_mul_wxyz(p_quat, init_rot_wxyz).astype(np.float32)
            return trans_local, rot_wxyz, "analytic-multi-msh-primary"

        if match_kind.startswith("combined") and ("/libero-merge/" in asset_path or "/libero_merge/" in asset_path):
            # Combined Libero-merge NPZs usually need a body-local centroid check
            # because some merged assets were exported in a parent/scene-root frame.
            # Microwave body/door are an exception here: their latent canonical frame
            # is already aligned to the original asset local frame, so keep the
            # rotation unchanged and use the NPZ/SDF target center directly.
            raw_center_arr = np.asarray(raw_center, dtype=np.float32).reshape(3)
            target_center_arr = target_center.reshape(3).astype(np.float32, copy=False)
            asset_name = os.path.basename(asset_path).lower()
            if asset_name == "microwave_combined_p_scaled.obj":
                trans_local = target_center_arr
                rot_wxyz = np.asarray(init_rot_wxyz, dtype=np.float32).reshape(4)
                return trans_local, rot_wxyz, "analytic-libero-merge-target-center"
            if asset_name == "microwave_door_combined_p_scaled.obj" and source_items:
                primary = next(
                    (
                        item for item in source_items
                        if "microdoor" in str(item.get("mesh_name", "") or item.get("mesh_path", "") or "").lower()
                    ),
                    next(
                        (
                            item for item in source_items
                            if str(item.get("kind", "")) == "mesh"
                            and "handle" not in str(item.get("mesh_name", "") or item.get("mesh_path", "") or "").lower()
                        ),
                        source_items[0],
                    ),
                )
                p_pos = np.asarray(primary.get("pos", [0.0, 0.0, 0.0]), dtype=np.float32).reshape(3)
                p_quat = np.asarray(primary.get("quat", [1.0, 0.0, 0.0, 0.0]), dtype=np.float32).reshape(4)
                trans_local = transform_point_cloud(target_center.reshape(1, 3), p_pos, p_quat)[0].astype(np.float32, copy=False)
                rot_wxyz = quat_mul_wxyz(p_quat, init_rot_wxyz).astype(np.float32)
                return trans_local, rot_wxyz, "analytic-libero-merge-door-primary"
            if float(np.linalg.norm(target_center_arr - raw_center_arr)) > 0.10:
                trans_local = raw_center_arr
            else:
                trans_local = target_center_arr
            rot_wxyz = np.asarray(init_rot_wxyz, dtype=np.float32).reshape(4)
            return trans_local, rot_wxyz, "analytic-libero-merge-raw-center"

        return None

    def _build_entry(spec, info, init_rot_wxyz):
        oid = int(info.get("object_id", -1))
        raw_points = np.asarray(spec.get("raw_points", None), dtype=np.float32)
        trans_local = _raw_center(spec)
        rot_wxyz = np.asarray(init_rot_wxyz, dtype=np.float32).reshape(4)
        template_points = _template_for_oid(oid)
        mode = "raw-center"

        analytic = _analytic_alignment(spec, info, init_rot_wxyz)
        if analytic is not None:
            trans_local, rot_wxyz, mode = analytic
            if template_points is not None and raw_points.ndim == 2 and raw_points.shape[1] == 3 and raw_points.shape[0] > 0 and mode in ("analytic-single-item-raw-center", "analytic-multi-msh-primary", "analytic-gripper-raw"):
                rot_wxyz_refined, _ = estimate_latent_pose_rotation(
                    raw_points,
                    template_points,
                    init_rot_wxyz=rot_wxyz,
                )
                rot_wxyz = np.asarray(rot_wxyz_refined, dtype=np.float32).reshape(4)
                mode = mode + "+estimated-rotation"
        elif template_points is not None and raw_points.ndim == 2 and raw_points.shape[1] == 3 and raw_points.shape[0] > 0:
            rot_wxyz, _ = estimate_latent_pose_rotation(
                raw_points,
                template_points,
                init_rot_wxyz=init_rot_wxyz,
            )
            rot_wxyz = np.asarray(rot_wxyz, dtype=np.float32).reshape(4)
            mode = "estimated-rotation-fallback"

        fit_error = _fit_error_for_entry(raw_points, template_points, trans_local, rot_wxyz)
        entry = {
            "rot_wxyz": np.asarray(rot_wxyz, dtype=np.float32).reshape(4),
            "trans_local": np.asarray(trans_local, dtype=np.float32).reshape(3),
            "fit_error": float(fit_error),
            "alignment_mode": mode,
        }
        if template_points is not None:
            entry["template_points_local"] = np.asarray(template_points, dtype=np.float32)
        return entry

    for entity_name, spec in entity_specs.items():
        info = latent_mapping.get(entity_name, None)
        if info is None:
            continue
        out[entity_name] = _build_entry(spec, info, [1.0, 0.0, 0.0, 0.0])

    if gripper_spec is not None:
        for gspec in list(gripper_spec.get("entities", [])):
            entity_name = str(gspec.get("entity_name", ""))
            info = latent_mapping.get(entity_name, None)
            if not entity_name or info is None:
                continue
            body_name = str(gspec.get("body_name", ""))
            part_name = str(gspec.get("part", ""))
            out[entity_name] = _build_entry(
                gspec,
                info,
                gripper_pose_quat_offset_wxyz(part_name, body_name),
            )
    return out




def _min_distance_points_to_tree(points, tree):
    # scipy API changed across versions (workers argument).
    try:
        dists, _ = tree.query(points, k=1, workers=-1)
    except TypeError:
        dists, _ = tree.query(points, k=1)
    dists = np.asarray(dists, dtype=np.float64).reshape(-1)
    if dists.size == 0:
        return np.nan
    return float(np.min(dists))


def pairwise_min_distance_points(points_a, points_b):
    a = np.asarray(points_a, dtype=np.float32)
    b = np.asarray(points_b, dtype=np.float32)
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] == 0 or b.shape[0] == 0:
        return np.nan

    if cKDTree is not None:
        tree_b = cKDTree(b)
        return _min_distance_points_to_tree(a, tree_b)

    # Fallback without scipy.
    diff = a[:, None, :] - b[None, :, :]
    d2 = np.einsum("ijk,ijk->ij", diff, diff)
    return float(np.sqrt(np.min(d2)))


def compute_pairwise_distance_traj(object_pcds, entity_names):
    if not entity_names:
        return np.zeros((0, 0, 0), dtype=np.float32)

    first_name = entity_names[0]
    num_steps = int(object_pcds[first_name].shape[0])
    n = len(entity_names)
    distances = np.zeros((num_steps, n, n), dtype=np.float32)

    for t in range(num_steps):
        frame_points = [np.asarray(object_pcds[name][t], dtype=np.float32) for name in entity_names]
        trees = None
        if cKDTree is not None:
            trees = [cKDTree(pts) for pts in frame_points]

        for i in range(n):
            distances[t, i, i] = 0.0
            for j in range(i + 1, n):
                if trees is not None:
                    d = _min_distance_points_to_tree(frame_points[i], trees[j])
                else:
                    d = pairwise_min_distance_points(frame_points[i], frame_points[j])
                d = 0.0 if not np.isfinite(d) else float(d)
                distances[t, i, j] = d
                distances[t, j, i] = d
    return distances


def infer_gripper_width_from_arrays(obs_gripper_states, robot_states, t):
    if obs_gripper_states is not None and t < len(obs_gripper_states):
        vals = np.asarray(obs_gripper_states[t], dtype=np.float64).reshape(-1)
        if vals.shape[0] >= 2:
            return float(abs(vals[0] - vals[1]))

    if robot_states is not None and t < len(robot_states):
        vals = np.asarray(robot_states[t], dtype=np.float64).reshape(-1)
        if vals.shape[0] >= 2:
            return float(abs(vals[0] - vals[1]))

    return None


def inspect_gripper_signal_layout(demo_group):
    cmd_binary = None
    cmd_values = None
    has_robot_gripper_state = False
    has_obs_gripper_state = False

    if "actions" in demo_group:
        actions = np.asarray(demo_group["actions"])
        if actions.ndim == 2 and actions.shape[1] > 0:
            cmd_values = np.unique(np.round(actions[:, -1].astype(np.float64), 6))
            cmd_binary = bool(cmd_values.size <= 2 and np.all(np.isin(cmd_values, np.array([-1.0, 1.0]))))

    if "robot_states" in demo_group:
        rs = np.asarray(demo_group["robot_states"])
        has_robot_gripper_state = (rs.ndim == 2 and rs.shape[1] >= 2)

    if "obs" in demo_group and "gripper_states" in demo_group["obs"]:
        gs = np.asarray(demo_group["obs/gripper_states"])
        has_obs_gripper_state = (gs.ndim >= 2 and gs.shape[1] >= 2)

    return {
        "command_binary": cmd_binary,
        "command_values": cmd_values,
        "has_robot_gripper_state": has_robot_gripper_state,
        "has_obs_gripper_state": has_obs_gripper_state,
    }



def _decode_attr_text(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8", errors="ignore")
    return str(value)


def _tokenize_bddl_text(text):
    tokens = []
    for line in str(text).splitlines():
        line = line.split(";", 1)[0]
        if not line.strip():
            continue
        tokens.extend(line.replace("(", " ( ").replace(")", " ) ").split())
    return tokens


def _parse_bddl_tokens(tokens):
    def parse_at(index):
        if index >= len(tokens):
            raise RuntimeError("Unexpected end of BDDL while parsing.")
        token = tokens[index]
        if token != "(":
            if token == ")":
                raise RuntimeError("Unexpected ')' in BDDL.")
            return token, index + 1
        expr = []
        index += 1
        while index < len(tokens) and tokens[index] != ")":
            child, index = parse_at(index)
            expr.append(child)
        if index >= len(tokens):
            raise RuntimeError("Unclosed '(' in BDDL.")
        return expr, index + 1

    expr, next_index = parse_at(0)
    if next_index != len(tokens):
        raise RuntimeError("BDDL has trailing tokens after top-level expression.")
    return expr


def _bddl_section(root, section_name):
    for child in root:
        if isinstance(child, list) and child and child[0] == section_name:
            return child
    return None


def _typed_section_pairs(section, default_category):
    pairs = []
    pending = []
    idx = 1
    while idx < len(section):
        token = section[idx]
        if token == "-":
            idx += 1
            if idx >= len(section):
                raise RuntimeError("Malformed typed BDDL section: '-' without category.")
            category = str(section[idx])
            pairs.extend((str(name), category) for name in pending)
            pending = []
        else:
            pending.append(str(token))
        idx += 1
    pairs.extend((str(name), str(default_category)) for name in pending)
    return pairs


def extract_object_names_from_bddl(bddl_file_path):
    if not os.path.exists(bddl_file_path):
        return []

    try:
        text = Path(bddl_file_path).read_text(encoding="utf-8")
        root = _parse_bddl_tokens(_tokenize_bddl_text(text))
        typed_sections = [
            section
            for section in (_bddl_section(root, ":fixtures"), _bddl_section(root, ":objects"))
            if section is not None
        ]
        if not typed_sections:
            return []
        objects = []
        for section in typed_sections:
            default_category = "fixture" if section[0] == ":fixtures" else "object"
            objects.extend(
                (inst, obj_type)
                for inst, obj_type in _typed_section_pairs(section, default_category)
                if "_" in inst and inst.split("_")[-1].isdigit()
            )
        return objects
    except Exception as exc:
        print(f"  Warning: typed BDDL parse failed for {bddl_file_path}, falling back to line parser: {exc}")
        objects = []
        with open(bddl_file_path, "r") as f:
            for raw_line in f:
                line = raw_line.strip()
                if " - " not in line:
                    continue
                left, right = line.split(" - ", 1)
                obj_type = right.strip()
                instances = left.strip().split()
                for inst in instances:
                    if "_" not in inst:
                        continue
                    if inst.split("_")[-1].isdigit():
                        objects.append((inst, obj_type))
        return objects


def find_bddl_file(bddl_file_name):
    if not bddl_file_name:
        return None

    candidate = Path(str(bddl_file_name)).expanduser()
    if candidate.is_file():
        return str(candidate.resolve())

    parts = str(bddl_file_name).split("/")
    subdir = parts[-2] if len(parts) >= 2 else ""
    basename = parts[-1]

    prefer_new_tree = "bddl_files_new" in str(bddl_file_name).replace("\\", "/")
    base_paths = (BDDL_NEW_BASE_PATH, BDDL_BASE_PATH) if prefer_new_tree else (BDDL_BASE_PATH, BDDL_NEW_BASE_PATH)

    for base_path in base_paths:
        exact = (base_path / subdir / basename) if subdir else (base_path / basename)
        if exact.exists():
            return str(exact)

    search_dirs = []
    for base_path in base_paths:
        search_dir = (base_path / subdir) if subdir else base_path
        if search_dir.exists():
            search_dirs.append(search_dir)
        else:
            search_dirs.append(base_path)

    basename_no_ext = Path(basename).stem
    for search_dir in search_dirs:
        for p in search_dir.rglob("*.bddl"):
            if Path(p).stem == basename_no_ext:
                return str(p)

    # fallback loose match
    tokens = set(basename_no_ext.split("_"))
    best = None
    best_score = -1
    for search_dir in search_dirs:
        for p in search_dir.rglob("*.bddl"):
            cand_tokens = set(Path(p).stem.split("_"))
            score = len(tokens.intersection(cand_tokens))
            if score > best_score:
                best_score = score
                best = p
    return str(best) if best is not None else None


def _materialize_bddl_content_from_hdf5(hdf5_path, content, preferred_name="embedded.bddl"):
    text = _decode_attr_text(content).strip()
    if not text:
        return None
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    stem = Path(str(preferred_name)).stem or "embedded"
    out_dir = Path("/tmp/sdf_wm_bddl_from_hdf5")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{Path(hdf5_path).stem}_{stem}_{digest}.bddl"
    if not out_path.exists() or out_path.read_text(encoding="utf-8", errors="ignore") != text:
        out_path.write_text(text + "\n", encoding="utf-8")
    return str(out_path)


def get_object_types_and_bddl(hdf5_path):
    with h5py.File(hdf5_path, "r") as f:
        env_args = json.loads(_decode_attr_text(f["data"].attrs["env_args"]))
        bddl_file_name = env_args.get("env_kwargs", {}).get("bddl_file_name", "")
        if not bddl_file_name:
            bddl_file_name = _decode_attr_text(f["data"].attrs.get("bddl_file_name", ""))
        bddl_file_content = f["data"].attrs.get("bddl_file_content", None)

    bddl_path = find_bddl_file(bddl_file_name)
    if not bddl_path and bddl_file_content is not None:
        bddl_path = _materialize_bddl_content_from_hdf5(hdf5_path, bddl_file_content, bddl_file_name)
    if not bddl_path:
        return [], None

    return extract_object_names_from_bddl(bddl_path), bddl_path


class LiberoPoseExtractor:
    def __init__(self):
        self.env = None
        self.model = None
        self.data = None
        self.body_name_to_idx = {}
        self.joint_name_to_idx = {}
        self.gripper_joint_names = []

    def initialize(self, bddl_path):
        try:
            import libero.libero.envs.bddl_utils as BDDLUtils
            from libero.libero.envs import TASK_MAPPING

            problem_info = BDDLUtils.get_problem_info(bddl_path)
            self.env = TASK_MAPPING[problem_info["problem_name"]](
                bddl_path,
                robots=["Panda"],
                has_renderer=False,
                has_offscreen_renderer=False,
                use_camera_obs=False,
            )
            self.env.reset()

            self.model = self.env.sim.model
            self.data = self.env.sim.data

            for i in range(self.model.nbody):
                body_name_addr = self.model.name_bodyadr[i]
                if body_name_addr > 0:
                    name = self.model.names[body_name_addr:].split(b"\x00")[0].decode("utf-8")
                    self.body_name_to_idx[name] = i

            finger_joint_candidates = []
            for i in range(self.model.njnt):
                joint_name_addr = self.model.name_jntadr[i]
                if joint_name_addr <= 0:
                    continue
                name = self.model.names[joint_name_addr:].split(b"\x00")[0].decode("utf-8")
                self.joint_name_to_idx[name] = i
                lname = name.lower()
                if "gripper" in lname and "finger_joint" in lname:
                    finger_joint_candidates.append((i, name))

            if len(finger_joint_candidates) < 2:
                for jname, jidx in self.joint_name_to_idx.items():
                    lname = jname.lower()
                    if "finger" in lname and "joint" in lname:
                        finger_joint_candidates.append((jidx, jname))

            finger_joint_candidates.sort(key=lambda x: x[0])
            self.gripper_joint_names = [name for _, name in finger_joint_candidates[:2]]

            print(
                f"    Initialized pose extractor with {len(self.body_name_to_idx)} bodies, "
                f"{len(self.joint_name_to_idx)} joints"
            )
            return True
        except Exception as e:
            print(f"    Warning: Could not initialize pose extractor: {e}")
            return False

    def extract_poses(self, state):
        if self.env is None:
            return {}
        try:
            self.env.sim.set_state_from_flattened(state)
            self.env.sim.forward()

            poses = {}
            for body_name, body_idx in self.body_name_to_idx.items():
                pos = self.env.sim.data.body_xpos[body_idx].copy()
                quat = self.env.sim.data.body_xquat[body_idx].copy()  # MuJoCo: (w,x,y,z)
                poses[body_name] = (pos, quat)
            return poses
        except Exception:
            return {}

    def get_gripper_width(self):
        if self.env is None or len(self.gripper_joint_names) < 2:
            return None
        vals = []
        for joint_name in self.gripper_joint_names[:2]:
            try:
                vals.append(float(self.env.sim.data.get_joint_qpos(joint_name)))
            except Exception:
                continue

        if len(vals) < 2:
            return None
        return float(abs(vals[0] - vals[1]))

    def close(self):
        if self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass



def select_best_body_name(candidates, obj_instance, part_name):
    if not candidates:
        return None

    obj_l = obj_instance.lower()
    part_l = part_name.lower()

    scored = []
    for cand in candidates:
        cand_l = cand.lower()
        score = 0
        if cand_l.startswith(obj_l):
            score += 40
        if cand_l == f"{obj_l}_{part_l}":
            score += 100
        if cand_l.endswith(f"_{part_l}"):
            score += 80
        if f"_{part_l}_" in cand_l:
            score += 60
        if part_l in cand_l:
            score += 20
        # shorter candidate is usually less noisy
        score -= len(cand_l) * 0.01
        scored.append((score, cand))

    scored.sort(reverse=True)
    return scored[0][1]


def _mesh_from_mujoco_model(model, mesh_id):
    """Build a trimesh mesh from MuJoCo model mesh buffers."""
    if mesh_id < 0:
        return None

    try:
        v_start = int(model.mesh_vertadr[mesh_id])
        v_count = int(model.mesh_vertnum[mesh_id])
        f_start = int(model.mesh_faceadr[mesh_id])
        f_count = int(model.mesh_facenum[mesh_id])
    except Exception:
        return None

    if v_count <= 0 or f_count <= 0:
        return None

    verts = np.array(model.mesh_vert[v_start : v_start + v_count], dtype=np.float64)
    faces_raw = np.array(model.mesh_face[f_start : f_start + f_count], dtype=np.int64)
    if faces_raw.ndim == 1:
        if faces_raw.size % 3 != 0:
            return None
        faces_raw = faces_raw.reshape(-1, 3)
    faces = faces_raw.reshape(-1, 3)

    # Different MuJoCo/asset pipelines may store face indices as:
    # 1) local indices in [0, v_count)
    # 2) global indices into model.mesh_vert (need minus v_start)
    # Try local first, then global->local conversion.
    local_faces = faces.copy()
    if np.all(local_faces >= 0) and np.all(local_faces < v_count):
        return trimesh.Trimesh(vertices=verts, faces=local_faces, process=False)

    shifted_faces = faces - v_start
    if np.all(shifted_faces >= 0) and np.all(shifted_faces < v_count):
        return trimesh.Trimesh(vertices=verts, faces=shifted_faces, process=False)

    return None


def _mujoco_mesh_name_map(model):
    out = {}
    if model is None or not hasattr(model, "nmesh") or not hasattr(model, "name_meshadr"):
        return out
    names_blob = model.names if hasattr(model, "names") else b""
    for mesh_id in range(int(model.nmesh)):
        try:
            adr = int(model.name_meshadr[mesh_id])
        except Exception:
            continue
        if adr < 0:
            continue
        try:
            name = names_blob[adr:].split(b"\x00")[0].decode("utf-8")
        except Exception:
            name = ""
        out[int(mesh_id)] = sanitize_name(name)
    return out


def _infer_part_name(obj_instance, body_name):
    obj_l = obj_instance.lower()
    body_l = body_name.lower()
    if body_l == obj_l:
        return "object"
    if body_l.startswith(obj_l + "_"):
        return body_name[len(obj_instance) + 1 :]
    return body_name


def _name_tokens(name):
    toks = [t for t in re.split(r"[^a-zA-Z0-9]+", name.lower()) if t]
    out = []
    for t in toks:
        if t.isdigit():
            continue
        out.append(t)
        if t.endswith("s") and len(t) > 3:
            out.append(t[:-1])
    return set(out)


def _body_match_score(body_name, obj_instance, obj_category):
    body_l = body_name.lower()
    inst_l = obj_instance.lower()
    cat_l = obj_category.lower()
    score = 0.0
    if inst_l in body_l:
        score += 5.0
    if cat_l in body_l:
        score += 4.0
    body_t = _name_tokens(body_l)
    inst_t = _name_tokens(inst_l)
    cat_t = _name_tokens(cat_l)
    score += 1.5 * len(body_t.intersection(inst_t))
    score += 1.2 * len(body_t.intersection(cat_t))
    return score


def _alias_match_score(body_name, aliases):
    b = body_name.lower()
    score = 0.0
    for a in aliases:
        if b == a:
            score += 20.0
        if b.endswith("_" + a):
            score += 12.0
        if f"_{a}_" in b:
            score += 8.0
        if a in b:
            score += 2.0
    return score


def infer_gripper_part_label(body_name):
    lname = canonical_name(body_name)
    toks = set(name_tokens(lname))
    if "leftfinger" in lname or ("left" in toks and "finger" in toks):
        return "left_finger"
    if "rightfinger" in lname or ("right" in toks and "finger" in toks):
        return "right_finger"
    if "finger" in lname or "finger" in toks:
        return "finger"
    if "right_hand" in lname or "right_gripper" in lname:
        return "hand"
    if "hand" in toks:
        return "hand"
    if "gripper" in toks:
        return "hand"
    return lname or "part"


def build_gripper_entities(parts):
    out = []
    used = set()
    for part in parts:
        body_name = str(part["body_name"])
        base = f"{GRIPPER_ENTITY_NAME}__{infer_gripper_part_label(body_name)}"
        entity_name = base
        suffix = 2
        while entity_name in used:
            entity_name = f"{base}_{suffix}"
            suffix += 1
        used.add(entity_name)
        pts = np.asarray(part["points"], dtype=np.float32)
        center_local = np.asarray(pts.mean(axis=0), dtype=np.float32) if pts.size > 0 else np.zeros((3,), dtype=np.float32)
        out.append(
            {
                "entity_name": entity_name,
                "part": infer_gripper_part_label(body_name),
                "body_name": body_name,
                "points": pts,
                "raw_points": pts.copy(),
                "surface_center": center_local,
                "source_items": [_copy_source_item(item) for item in part.get("source_items", []) or []],
            }
        )
    return out


def select_primary_gripper_entities(entities):
    if len(entities) <= 3:
        return list(entities)

    selected = []
    used_names = set()

    def _take_first(part_name):
        for e in entities:
            if e.get("part") != part_name:
                continue
            name = str(e.get("entity_name", ""))
            if name in used_names:
                continue
            used_names.add(name)
            selected.append(e)
            return

    _take_first("hand")
    _take_first("left_finger")
    _take_first("right_finger")

    if len(selected) < 3:
        for e in entities:
            if e.get("part") != "finger":
                continue
            name = str(e.get("entity_name", ""))
            if name in used_names:
                continue
            used_names.add(name)
            selected.append(e)
            if len(selected) >= 3:
                break

    if len(selected) < 3:
        for e in entities:
            name = str(e.get("entity_name", ""))
            if name in used_names:
                continue
            used_names.add(name)
            selected.append(e)
            if len(selected) >= 3:
                break

    return selected[:3]



def build_gripper_spec(pose_extractor, num_points_per_entity):
    model = pose_extractor.model
    if model is None:
        return None

    body_names = list(pose_extractor.body_name_to_idx.keys())
    mesh_name_map = _mujoco_mesh_name_map(model)

    def body_priority(name):
        lname = name.lower()
        if "right_hand" in lname or "right_gripper" in lname:
            return 0
        if "leftfinger" in lname or "rightfinger" in lname:
            return 1
        if "finger" in lname:
            return 2
        if "gripper" in lname:
            return 3
        return 10

    part_specs = []

    def _find_body(tokens):
        for body_name in body_names:
            lname = body_name.lower()
            if any(tok in lname for tok in tokens):
                return body_name
        return None

    def _collect_collision_mesh_points(body_name, allowed_mesh_names):
        body_idx = pose_extractor.body_name_to_idx.get(body_name, None)
        if body_idx is None:
            return None

        body_geom_ids = np.where(np.asarray(model.geom_bodyid) == body_idx)[0].tolist()
        if not body_geom_ids:
            return None

        mesh_gids = []
        for gid in body_geom_ids:
            gtype = int(model.geom_type[gid]) if hasattr(model, "geom_type") else -1
            if gtype != 7 or not hasattr(model, "geom_dataid"):
                continue
            mesh_id = int(model.geom_dataid[gid])
            if mesh_id < 0:
                continue
            if hasattr(model, "geom_group") and int(model.geom_group[gid]) != 0:
                continue
            mesh_name = str(mesh_name_map.get(mesh_id, ""))
            if allowed_mesh_names and mesh_name not in allowed_mesh_names:
                continue
            mesh_gids.append(gid)

        # Fallback for naming differences: keep collision meshes and drop *_vis.
        if not mesh_gids and allowed_mesh_names:
            for gid in body_geom_ids:
                gtype = int(model.geom_type[gid]) if hasattr(model, "geom_type") else -1
                if gtype != 7 or not hasattr(model, "geom_dataid"):
                    continue
                mesh_id = int(model.geom_dataid[gid])
                if mesh_id < 0:
                    continue
                if hasattr(model, "geom_group") and int(model.geom_group[gid]) != 0:
                    continue
                mesh_name = str(mesh_name_map.get(mesh_id, ""))
                if "vis" in mesh_name:
                    continue
                mesh_gids.append(gid)

        if not mesh_gids:
            return None

        per_geom_points = max(32, num_points_per_entity // max(1, len(mesh_gids)))
        geom_points = []
        source_items = []
        for gid in mesh_gids:
            mesh_id = int(model.geom_dataid[gid])
            mesh = _mesh_from_mujoco_model(model, mesh_id)
            if mesh is None:
                continue
            points = sample_points_from_mesh(mesh, per_geom_points)
            if points is None:
                continue
            geom_pos = np.asarray(model.geom_pos[gid], dtype=np.float64)
            geom_quat = np.asarray(model.geom_quat[gid], dtype=np.float64)  # wxyz
            geom_points.append(transform_point_cloud(points, geom_pos, geom_quat))
            source_items.append({
                "kind": "mesh",
                "mesh_name": str(mesh_name_map.get(mesh_id, "")),
                "pos": geom_pos.astype(np.float32),
                "quat": geom_quat.astype(np.float32),
            })

        if not geom_points:
            return None
        return {"points": np.vstack(geom_points), "source_items": source_items}

    # Prefer strict SDF-consistent gripper parts:
    # 1 hand body + 2 finger bodies, sampled from collision meshes hand.stl / finger.stl.
    hand_body = _find_body(["right_gripper", "right_hand"])
    left_finger_body = _find_body(["leftfinger"])
    right_finger_body = _find_body(["rightfinger"])
    strict_targets = [
        (hand_body, {"hand"}),
        (left_finger_body, {"finger"}),
        (right_finger_body, {"finger"}),
    ]
    strict_part_specs = []
    for body_name, allowed_mesh_names in strict_targets:
        if body_name is None:
            continue
        part_info = _collect_collision_mesh_points(body_name, allowed_mesh_names=allowed_mesh_names)
        if part_info is None:
            continue
        strict_part_specs.append({"body_name": body_name, **part_info})

    if len(strict_part_specs) == 3:
        part_specs = strict_part_specs
        print(
            "  Gripper sampling mode: sdf-consistent collision meshes "
            f"(hand+2finger): {[p['body_name'] for p in part_specs]}"
        )
    else:
        if len(strict_part_specs) > 0:
            print(
                "  Warning: strict gripper match incomplete "
                f"({len(strict_part_specs)}/3), fallback to legacy body-based sampling."
            )
        else:
            print(
                "  Warning: strict gripper match failed (0/3), fallback to collision-only body sampling."
            )
        candidate_body_names = []
        for body_name in body_names:
            lname = body_name.lower()
            if "tip" in lname:
                continue
            if ("gripper" in lname) or ("finger" in lname) or lname.endswith("_right_hand"):
                candidate_body_names.append(body_name)

        candidate_body_names = sorted(set(candidate_body_names), key=body_priority)

        for body_name in candidate_body_names:
            body_idx = pose_extractor.body_name_to_idx.get(body_name, None)
            if body_idx is None:
                continue

            body_geom_ids = np.where(np.asarray(model.geom_bodyid) == body_idx)[0].tolist()
            if not body_geom_ids:
                continue

            collision_geom_ids = [gid for gid in body_geom_ids if int(model.geom_group[gid]) == 0]
            active_geom_ids = collision_geom_ids if collision_geom_ids else body_geom_ids

            per_geom_points = max(32, num_points_per_entity // max(1, len(active_geom_ids)))
            geom_points = []

            for gid in active_geom_ids:
                gtype = int(model.geom_type[gid]) if hasattr(model, "geom_type") else -1
                points = None

                if gtype == 7 and hasattr(model, "geom_dataid"):
                    mesh_id = int(model.geom_dataid[gid])
                    if mesh_id >= 0:
                        mesh_name = str(mesh_name_map.get(mesh_id, "")).lower()
                        if "vis" in mesh_name:
                            continue
                        mesh = _mesh_from_mujoco_model(model, mesh_id)
                        if mesh is not None:
                            points = sample_points_from_mesh(mesh, per_geom_points)
                elif gtype in MJ_GEOM_TYPE_TO_PRIMITIVE:
                    size = np.asarray(model.geom_size[gid], dtype=np.float64)
                    points = sample_points_from_primitive(
                        MJ_GEOM_TYPE_TO_PRIMITIVE[gtype], size, per_geom_points
                    )

                if points is None:
                    continue

                geom_pos = np.asarray(model.geom_pos[gid], dtype=np.float64)
                geom_quat = np.asarray(model.geom_quat[gid], dtype=np.float64)  # wxyz
                points_body = transform_point_cloud(points, geom_pos, geom_quat)
                geom_points.append(points_body)

            if not geom_points:
                continue

            merged = np.vstack(geom_points)
            source_items = []
            for gid in active_geom_ids:
                source_items.append({
                    "kind": "geom",
                    "geom_type": str(MJ_GEOM_TYPE_TO_PRIMITIVE.get(int(model.geom_type[gid]), "mesh")) if int(model.geom_type[gid]) in MJ_GEOM_TYPE_TO_PRIMITIVE else "mesh",
                    "pos": np.asarray(model.geom_pos[gid], dtype=np.float32),
                    "quat": np.asarray(model.geom_quat[gid], dtype=np.float32),
                })
            part_specs.append({"body_name": body_name, "points": merged, "source_items": source_items})

    if not part_specs:
        print("  Warning: Could not build gripper spec from robot model")
        return None

    part_specs.sort(key=lambda x: body_priority(x["body_name"]))

    parts = []
    for part_info in part_specs:
        body_name = str(part_info["body_name"])
        points = np.asarray(part_info["points"], dtype=np.float32)
        parts.append(
            {
                "body_name": body_name,
                "points": normalize_point_count(points, num_points_per_entity).astype(np.float32),
                "source_items": [_copy_source_item(item) for item in part_info.get("source_items", []) or []],
            }
        )

    if not parts:
        print("  Warning: Gripper spec exists but point allocation failed")
        return None

    part_body_names = [p["body_name"] for p in parts]
    root_body_name = None
    for cand in ["robot0_right_hand", "gripper0_right_gripper", part_body_names[0]]:
        if cand in part_body_names:
            root_body_name = cand
            break

    entities = select_primary_gripper_entities(build_gripper_entities(parts))
    entity_names = [e["entity_name"] for e in entities]
    if len(entities) < 3:
        print(
            f"  Warning: expected 3 gripper entities (hand+2finger), got {len(entities)}; "
            "missing parts will be unavailable."
        )
    print(
        f"  Built gripper entities ({GRIPPER_ENTITY_NAME}) with "
        f"{len(entities)} parts: {entity_names}"
    )
    return {
        "root_body_name": root_body_name,
        "parts": parts,
        "entities": entities,
    }


def build_object_specs(object_types, pose_extractor, num_points_per_entity, split_mesh_subparts=True):
    """
    Build entity specs where each mesh-bearing body-part is treated as one object entity.

    Returns dict:
      {
        entity_name: {
          'instance': str,
          'category': str,
          'part': str,
          'body_name': str,
          'points': (N,3)
        }
      }
    """
    entity_specs = {}
    model = pose_extractor.model
    mesh_cache = {}
    body_names = list(pose_extractor.body_name_to_idx.keys())
    mesh_name_map = _mujoco_mesh_name_map(model)

    def mesh_geom_ids_for_body_idx(body_idx, allowed_mesh_names=None):
        geom_ids = np.where(np.array(model.geom_bodyid) == body_idx)[0]
        # Prefer actual mesh geoms; fall back to dataid-based check for compatibility.
        mesh_like = []
        allowed_set = None
        if allowed_mesh_names:
            allowed_set = {sanitize_name(x) for x in allowed_mesh_names if x}
        for gid in geom_ids:
            gtype = int(model.geom_type[gid]) if hasattr(model, "geom_type") else -1
            dataid = int(model.geom_dataid[gid]) if hasattr(model, "geom_dataid") else -1
            if not ((gtype == 7 and dataid >= 0) or (dataid >= 0)):
                continue
            if allowed_set is not None:
                mesh_name = mesh_name_map.get(int(dataid), "")
                if not mesh_name_matches_allowed(mesh_name, allowed_set):
                    continue
            mesh_like.append(gid)
        return mesh_like

    for obj_instance, obj_category in object_types:
        xml_path = find_object_xml(obj_category)
        if xml_path is None:
            print(f"  Warning: XML not found for object category '{obj_category}'")
            continue
        xml_category = canonical_name(Path(xml_path).stem)
        obj_category_norm = canonical_name(obj_category)
        if xml_category and obj_category_norm and xml_category != obj_category_norm:
            print(f"  XML alias: {obj_category_norm} -> {xml_category}")

        try:
            parts = parse_object_parts_from_xml(xml_path)
        except Exception as e:
            print(f"  Warning: Failed to parse XML for {obj_category}: {e}")
            continue

        if split_mesh_subparts:
            parts = split_parts_by_mesh_subparts(parts, obj_category=xml_category or obj_category_norm)

        if not parts:
            print(f"  Warning: No mesh parts found in XML for {obj_category}")
            continue

        scored_candidates = [
            (b, _body_match_score(b, obj_instance, obj_category)) for b in body_names
        ]
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        instance_candidates = [b for b, s in scored_candidates if s > 0]
        if not instance_candidates:
            # Last fallback: allow all body names and rely on mesh-existence filtering below.
            instance_candidates = body_names[:]

        used_body_names = set()
        for part_name, mesh_items in parts.items():
            desired_mesh_names = {
                str(mi.get("mesh_name", ""))
                for mi in mesh_items
                if isinstance(mi, dict) and mi.get("kind") == "mesh" and mi.get("mesh_name")
            }
            if desired_mesh_names and xml_category and obj_category_norm and xml_category != obj_category_norm:
                desired_mesh_names = desired_mesh_names.union(
                    {
                        mesh_name_with_category_alias(mn, src_category=xml_category, dst_category=obj_category_norm)
                        for mn in desired_mesh_names
                    }
                )
            aliases = part_name_aliases(part_name)
            part_candidate_names = [
                b
                for b in instance_candidates
                if any(alias in b.lower() for alias in aliases)
            ]
            body_pool = part_candidate_names if part_candidate_names else instance_candidates
            if not body_pool:
                continue

            ranked = []
            for cand in body_pool:
                score = _body_match_score(cand, obj_instance, obj_category)
                score += _alias_match_score(cand, aliases)
                cand_idx = pose_extractor.body_name_to_idx.get(cand, None)
                mesh_count = (
                    len(
                        mesh_geom_ids_for_body_idx(
                            cand_idx,
                            allowed_mesh_names=desired_mesh_names if desired_mesh_names else None,
                        )
                    )
                    if cand_idx is not None
                    else 0
                )
                # Prefer assigning distinct body for each part.
                reuse_penalty = -50.0 if cand in used_body_names else 0.0
                ranked.append((score + reuse_penalty, mesh_count, cand))
            ranked.sort(key=lambda x: (x[0], x[1]), reverse=True)

            # Choose the best candidate that actually has mesh geoms first.
            body_name = None
            for _, mesh_count, cand in ranked:
                if mesh_count > 0:
                    body_name = cand
                    break
            if body_name is None:
                body_name = ranked[0][2]
            used_body_names.add(body_name)

            part_points_list = []
            per_mesh_points = max(1, num_points_per_entity // len(mesh_items))
            for mesh_item in mesh_items:
                try:
                    if mesh_item.get("kind") == "mesh":
                        mesh = load_mesh_with_cache(mesh_item["mesh_path"], mesh_item["scale"], mesh_cache)
                        points = sample_points_from_mesh(mesh, per_mesh_points)
                        if points is not None:
                            points = transform_point_cloud(points, mesh_item["pos"], mesh_item["quat"])
                    else:
                        points = sample_points_from_primitive(
                            mesh_item["geom_type"], mesh_item["size"], per_mesh_points
                        )
                        if points is not None:
                            points = transform_point_cloud(points, mesh_item["pos"], mesh_item["quat"])

                    if points is not None:
                        part_points_list.append(points)
                except Exception:
                    continue

            # Fallback: if file-based loading fails for this part, sample from MuJoCo-loaded mesh.
            if not part_points_list and model is not None:
                body_idx = pose_extractor.body_name_to_idx.get(body_name, None)
                mesh_geom_ids = []
                if body_idx is not None:
                    mesh_geom_ids = mesh_geom_ids_for_body_idx(
                        body_idx,
                        allowed_mesh_names=desired_mesh_names if desired_mesh_names else None,
                    )

                # If selected body has no mesh, pick another instance candidate that has mesh.
                if not mesh_geom_ids:
                    for cand_body_name in instance_candidates:
                        cand_idx = pose_extractor.body_name_to_idx.get(cand_body_name, None)
                        if cand_idx is None:
                            continue
                        cand_mesh_ids = mesh_geom_ids_for_body_idx(
                            cand_idx,
                            allowed_mesh_names=desired_mesh_names if desired_mesh_names else None,
                        )
                        if cand_mesh_ids:
                            body_name = cand_body_name
                            body_idx = cand_idx
                            mesh_geom_ids = cand_mesh_ids
                            break

                if body_idx is not None and mesh_geom_ids:
                    per_geom_points = max(1, num_points_per_entity // max(1, len(mesh_geom_ids)))
                    for gid in mesh_geom_ids:
                        mesh_id = int(model.geom_dataid[gid])
                        mesh = _mesh_from_mujoco_model(model, mesh_id)
                        if mesh is None:
                            if obj_category == "cookies":
                                try:
                                    v_count = int(model.mesh_vertnum[mesh_id])
                                    f_count = int(model.mesh_facenum[mesh_id])
                                    print(
                                        f"    Debug cookies mesh parse failed: body={body_name}, "
                                        f"mesh_id={mesh_id}, vertnum={v_count}, facenum={f_count}"
                                    )
                                except Exception:
                                    pass
                            continue
                        points = sample_points_from_mesh(mesh, per_geom_points)
                        if points is None:
                            continue
                        geom_pos = np.array(model.geom_pos[gid], dtype=np.float64)
                        geom_quat_wxyz = np.array(model.geom_quat[gid], dtype=np.float64)
                        points = transform_point_cloud(points, geom_pos, geom_quat_wxyz)
                        part_points_list.append(points)

            if not part_points_list:
                continue

            merged = np.vstack(part_points_list)
            if merged.shape[0] > num_points_per_entity:
                idx = np.random.choice(merged.shape[0], num_points_per_entity, replace=False)
                merged = merged[idx]
            elif merged.shape[0] < num_points_per_entity:
                pad = np.zeros((num_points_per_entity - merged.shape[0], 3), dtype=np.float32)
                merged = np.vstack([merged, pad])

            part_key = sanitize_name(part_name)
            entity_name = f"{obj_instance}__{part_key}"
            if len(parts) == 1 or part_name == "object" or part_key == "object":
                entity_name = obj_instance

            # avoid collision in rare cases
            suffix = 1
            base_entity_name = entity_name
            while entity_name in entity_specs:
                suffix += 1
                entity_name = f"{base_entity_name}_{suffix}"

            entity_specs[entity_name] = {
                "instance": obj_instance,
                "category": obj_category,
                "part": part_name,
                "body_name": body_name,
                "points": merged.astype(np.float32),
                "raw_points": merged.astype(np.float32),
                "source_items": [_copy_source_item(item) for item in mesh_items],
            }
        built_count = sum(1 for v in entity_specs.values() if v["instance"] == obj_instance)
        print(f"  Built entities for {obj_instance} ({obj_category}): {built_count}")
        if built_count > 0:
            mapping = {
                v["part"]: v["body_name"]
                for v in entity_specs.values()
                if v["instance"] == obj_instance
            }
            print(f"    Part-body mapping: {mapping}")
        if built_count == 0:
            debug_candidates = []
            for cand in instance_candidates[:10]:
                cand_idx = pose_extractor.body_name_to_idx.get(cand, None)
                mesh_count = len(mesh_geom_ids_for_body_idx(cand_idx)) if cand_idx is not None else 0
                debug_candidates.append((cand, mesh_count))
            print(f"    Debug candidates (top {len(debug_candidates)}): {debug_candidates}")

    return entity_specs


def process_hdf5_file(
    hdf5_path,
    output_path,
    num_points_per_obj,
    overwrite=False,
    split_mesh_subparts=False,
    add_distance=False,
    center_by_surface=False,
    latent_ckpt=None,
    ignore_image=False,
):
    print(f"\nProcessing: {hdf5_path}")
    if split_mesh_subparts:
        print("  Mode: split mesh subparts enabled")
    if add_distance:
        print("  Mode: add pairwise object distances enabled")
    if center_by_surface:
        print("  Mode: center-by-surface enabled")
    if ignore_image:
        print("  Mode: ignore image observations enabled")

    object_types, bddl_path = get_object_types_and_bddl(hdf5_path)
    if not object_types:
        print("  Warning: Could not extract object types from BDDL, skipping")
        return False
    print(f"  Found objects: {object_types}")

    pose_extractor = LiberoPoseExtractor()
    if not bddl_path:
        print("  Warning: Could not resolve BDDL path")
        return False

    if not pose_extractor.initialize(bddl_path):
        print("  Warning: Failed to initialize pose extractor")
        return False

    try:
        entity_specs = build_object_specs(
            object_types,
            pose_extractor,
            num_points_per_obj,
            split_mesh_subparts=split_mesh_subparts,
        )
        if not entity_specs:
            print("  Warning: No valid object entities to process, skipping")
            return False

        object_meta = load_latent_object_meta(latent_ckpt)

        gripper_spec = build_gripper_spec(pose_extractor, num_points_per_obj)
        if gripper_spec is not None:
            gripper_spec["entities"] = list(gripper_spec.get("entities", []))

        if center_by_surface:
            preliminary_mapping = build_precomputed_latent_mapping(
                entity_specs=entity_specs,
                gripper_spec=gripper_spec,
                latent_ckpt=latent_ckpt,
                object_meta=object_meta,
            )
            if object_meta and preliminary_mapping:
                apply_target_surface_centers(
                    entity_specs=entity_specs,
                    gripper_spec=gripper_spec,
                    latent_mapping=preliminary_mapping,
                    object_meta=object_meta,
                )
            else:
                for spec in entity_specs.values():
                    pts = np.asarray(spec["points"], dtype=np.float32)
                    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] == 0:
                        spec["surface_center"] = np.zeros((3,), dtype=np.float32)
                        continue
                    center = pts.mean(axis=0).astype(np.float32, copy=False)
                    spec["surface_center"] = center
                    spec["points"] = (pts - center.reshape(1, 3)).astype(np.float32, copy=False)

                if gripper_spec is not None:
                    gripper_entities = list(gripper_spec.get("entities", []))
                    for gspec in gripper_entities:
                        pts = np.asarray(gspec.get("points"), dtype=np.float32)
                        if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] == 0:
                            gspec["surface_center"] = np.zeros((3,), dtype=np.float32)
                            continue
                        center = pts.mean(axis=0).astype(np.float32, copy=False)
                        gspec["surface_center"] = center
                        gspec["points"] = (pts - center.reshape(1, 3)).astype(np.float32, copy=False)
                    gripper_spec["entities"] = gripper_entities
        else:
            for spec in entity_specs.values():
                spec["surface_center"] = np.zeros((3,), dtype=np.float32)
            if gripper_spec is not None:
                gripper_entities = list(gripper_spec.get("entities", []))
                for gspec in gripper_entities:
                    gspec["surface_center"] = np.zeros((3,), dtype=np.float32)
                gripper_spec["entities"] = gripper_entities

        latent_mapping = build_precomputed_latent_mapping(
            entity_specs=entity_specs,
            gripper_spec=gripper_spec,
            latent_ckpt=latent_ckpt,
            object_meta=object_meta,
        )
        latent_pose_alignment = {}
        if latent_mapping:
            entity_specs, latent_mapping = collapse_combined_entity_specs(
                entity_specs=entity_specs,
                latent_mapping=latent_mapping,
                num_points_per_entity=num_points_per_obj,
            )
            latent_pose_alignment = build_latent_pose_alignment(
                entity_specs=entity_specs,
                gripper_spec=gripper_spec,
                latent_mapping=latent_mapping,
                object_meta=object_meta,
                num_points_per_entity=num_points_per_obj,
            )
            for entity_name, align_info in latent_pose_alignment.items():
                if entity_name not in latent_mapping:
                    continue
                latent_mapping[entity_name]["pose_offset_local"] = np.asarray(
                    align_info.get("trans_local", np.zeros((3,), dtype=np.float32)),
                    dtype=np.float32,
                ).reshape(3)
                latent_mapping[entity_name]["pose_rot_offset_wxyz"] = np.asarray(
                    align_info.get("rot_wxyz", np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)),
                    dtype=np.float32,
                ).reshape(4)
                latent_mapping[entity_name]["pose_align_error"] = float(align_info.get("fit_error", 0.0))
                latent_mapping[entity_name]["alignment_mode"] = str(align_info.get("alignment_mode", ""))
            print(f"  Latent mapping entries: {len(latent_mapping)}")
            if latent_pose_alignment:
                errs = np.asarray([float(v.get("fit_error", 0.0)) for v in latent_pose_alignment.values()], dtype=np.float32)
                print(
                    f"  Latent-frame alignment: count={len(latent_pose_alignment)}, "
                    f"mean_err={float(errs.mean()):.6f}, max_err={float(errs.max()):.6f}"
                )

        with h5py.File(hdf5_path, "r") as src_f:
            first_ep = list(src_f["data"].keys())[0]
            first_demo_group = src_f[f"data/{first_ep}"]

            if "object_pcds" in first_demo_group and not overwrite:
                print("  Point cloud data already exists, skipping (use --overwrite to replace)")
                return False

            signal_info = inspect_gripper_signal_layout(first_demo_group)
            if signal_info["command_values"] is not None:
                cmd_type = "binary open/close" if signal_info["command_binary"] else "continuous"
                print(
                    f"  Gripper action channel: {cmd_type}, "
                    f"unique={signal_info['command_values'].tolist()}"
                )
            print(
                "  Gripper width source availability: "
                f"robot_states[:2]={signal_info['has_robot_gripper_state']}, "
                f"obs/gripper_states={signal_info['has_obs_gripper_state']}, "
                f"sim_joints={len(pose_extractor.gripper_joint_names) >= 2}"
            )

            demo_keys = sorted(list(src_f["data"].keys()), key=lambda x: int(x.split("_")[1]))
            print(f"  Number of trajectories: {len(demo_keys)}")

        if output_path is None:
            output_path = hdf5_path.replace(".hdf5", "_pcd.hdf5")

        import shutil

        output_parent = os.path.dirname(output_path)
        if output_parent:
            os.makedirs(output_parent, exist_ok=True)
        shutil.copy2(hdf5_path, output_path)

        with h5py.File(output_path, "a") as f:
            if ignore_image:
                removed_image_count = strip_image_observations_inplace(f)
                print(f"  Removed image observation datasets: {removed_image_count}")
            if latent_mapping:
                meta_group = f.require_group("meta")
                mapping_group = meta_group.require_group("latent_mapping")
                for key in list(mapping_group.keys()):
                    del mapping_group[key]
                str_dtype = h5py.string_dtype(encoding="utf-8")
                ordered_entity_names = sorted(latent_mapping.keys())
                mapping_group.create_dataset("entity_names", data=np.asarray(ordered_entity_names, dtype=str_dtype))
                mapping_group.create_dataset(
                    "pose_source_names",
                    data=np.asarray([latent_mapping[n]["pose_source"] for n in ordered_entity_names], dtype=str_dtype),
                )
                mapping_group.create_dataset(
                    "target_npz_names",
                    data=np.asarray([latent_mapping[n]["target_npz_name"] for n in ordered_entity_names], dtype=str_dtype),
                )
                mapping_group.create_dataset(
                    "target_object_ids",
                    data=np.asarray([latent_mapping[n]["object_id"] for n in ordered_entity_names], dtype=np.int32),
                )
                mapping_group.create_dataset(
                    "pose_offset_local",
                    data=np.stack([latent_mapping[n]["pose_offset_local"] for n in ordered_entity_names], axis=0).astype(np.float32),
                )
                mapping_group.create_dataset(
                    "pose_rot_offset_wxyz",
                    data=np.stack([
                        np.asarray(latent_mapping[n].get("pose_rot_offset_wxyz", np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)), dtype=np.float32).reshape(4)
                        for n in ordered_entity_names
                    ], axis=0).astype(np.float32),
                )
                mapping_group.create_dataset(
                    "pose_align_error",
                    data=np.asarray([float(latent_mapping[n].get("pose_align_error", 0.0)) for n in ordered_entity_names], dtype=np.float32),
                )
                mapping_group.create_dataset(
                    "alignment_mode",
                    data=np.asarray([
                        str(latent_mapping[n].get("alignment_mode", "")) for n in ordered_entity_names
                    ], dtype=str_dtype),
                )
                for field_name, map_key in [
                    ("body_names", "body_name"),
                    ("instances", "instance"),
                    ("categories", "category"),
                    ("parts", "part"),
                    ("match_kind", "match_kind"),
                ]:
                    mapping_group.create_dataset(
                        field_name,
                        data=np.asarray([latent_mapping[n][map_key] for n in ordered_entity_names], dtype=str_dtype),
                    )
                mapping_group.attrs["latent_ckpt"] = str(latent_ckpt)
                mapping_group.attrs["mapping_version"] = 4

            for ep in demo_keys:
                demo_group = f[f"data/{ep}"]
                if "states" not in demo_group:
                    print(f"  Warning: {ep} has no states, skip")
                    continue

                states = np.asarray(demo_group["states"])
                if states.ndim == 1:
                    states = states.reshape(1, -1)
                num_steps = int(states.shape[0])
                if num_steps <= 0:
                    print(f"  Warning: {ep} has zero steps, skip")
                    continue

                robot_states = np.asarray(demo_group["robot_states"]) if "robot_states" in demo_group else None
                obs_group = demo_group["obs"] if "obs" in demo_group else None
                obs_gripper_states = None
                if obs_group is not None and "gripper_states" in obs_group:
                    obs_gripper_states = np.asarray(obs_group["gripper_states"])

                object_pcds = {
                    entity_name: np.zeros((num_steps, num_points_per_obj, 3), dtype=np.float32)
                    for entity_name in entity_specs
                }
                object_poses = {entity_name: [] for entity_name in entity_specs}
                object_poses_aligned = {entity_name: [] for entity_name in entity_specs}

                gripper_entities = []
                has_gripper_entity = False
                gripper_root_name = None
                if gripper_spec is not None:
                    gripper_entities = list(gripper_spec.get("entities", []))
                    has_gripper_entity = len(gripper_entities) > 0
                    gripper_root_name = gripper_spec.get("root_body_name", None)
                    for gspec in gripper_entities:
                        gname = str(gspec["entity_name"])
                        object_pcds[gname] = np.zeros((num_steps, num_points_per_obj, 3), dtype=np.float32)
                        object_poses[gname] = []
                        object_poses_aligned[gname] = []

                gripper_pose_list = []
                gripper_width_traj = np.zeros((num_steps, 1), dtype=np.float32)

                for t in range(num_steps):
                    poses = pose_extractor.extract_poses(states[t])

                    for entity_name, spec in entity_specs.items():
                        body_name = spec["body_name"]
                        if body_name in poses:
                            pos, quat = poses[body_name]
                        else:
                            pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                            quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

                        quat_raw = normalize_quaternion_wxyz(quat).astype(np.float32)
                        pose_pos = pos
                        if center_by_surface:
                            center_local = np.asarray(spec.get("surface_center", np.zeros((3,), dtype=np.float32)), dtype=np.float32)
                            center_world = transform_point_cloud(center_local.reshape(1, 3), pos, quat_raw)[0]
                            pose_pos = center_world.astype(np.float32, copy=False)
                        raw_points = np.asarray(spec.get("raw_points", spec["points"]), dtype=np.float32)

                        align_info = latent_pose_alignment.get(entity_name, None)
                        if align_info is not None:
                            align_t = np.asarray(align_info.get("trans_local", np.zeros((3,), dtype=np.float32)), dtype=np.float32).reshape(3)
                            align_q = np.asarray(align_info.get("rot_wxyz", np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)), dtype=np.float32).reshape(4)
                        else:
                            align_t = np.asarray(spec.get("surface_center", np.zeros((3,), dtype=np.float32)), dtype=np.float32).reshape(3) if center_by_surface else np.zeros((3,), dtype=np.float32)
                            align_q = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
                        pose_pos_aligned = transform_point_cloud(align_t.reshape(1, 3), pos, quat_raw)[0].astype(np.float32, copy=False)
                        pose_quat_aligned = quat_mul_wxyz(quat_raw, align_q).astype(np.float32)
                        object_poses_aligned[entity_name].append((pose_pos_aligned, pose_quat_aligned))

                        transformed = transform_point_cloud(raw_points, pos, quat_raw)
                        object_pcds[entity_name][t] = transformed
                        object_poses[entity_name].append((pose_pos, quat_raw))

                    sim_width = pose_extractor.get_gripper_width()
                    if sim_width is None:
                        sim_width = infer_gripper_width_from_arrays(obs_gripper_states, robot_states, t)
                    if sim_width is None:
                        sim_width = 0.0
                    gripper_width_traj[t, 0] = float(sim_width)

                    if has_gripper_entity:
                        for gspec in gripper_entities:
                            entity_name = str(gspec["entity_name"])
                            body_name = str(gspec["body_name"])
                            if body_name in poses:
                                pos, quat = poses[body_name]
                            else:
                                pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                                quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
                            quat_raw = normalize_quaternion_wxyz(quat).astype(np.float32)
                            q_off = gripper_pose_quat_offset_wxyz(gspec.get("part", ""), body_name).astype(np.float32)
                            quat_aligned = quat_mul_wxyz(quat_raw, q_off).astype(np.float32)
                            pose_pos = pos
                            if center_by_surface:
                                center_local = np.asarray(
                                    gspec.get("surface_center", np.zeros((3,), dtype=np.float32)),
                                    dtype=np.float32,
                                )
                                center_world = transform_point_cloud(center_local.reshape(1, 3), pos, quat_raw)[0]
                                pose_pos = center_world.astype(np.float32, copy=False)
                            raw_points = np.asarray(gspec.get("raw_points", gspec["points"]), dtype=np.float32)

                            align_info = latent_pose_alignment.get(entity_name, None)
                            if align_info is not None:
                                align_t = np.asarray(align_info.get("trans_local", np.zeros((3,), dtype=np.float32)), dtype=np.float32).reshape(3)
                                align_q = np.asarray(align_info.get("rot_wxyz", np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)), dtype=np.float32).reshape(4)
                            else:
                                align_t = np.asarray(gspec.get("surface_center", np.zeros((3,), dtype=np.float32)), dtype=np.float32).reshape(3) if center_by_surface else np.zeros((3,), dtype=np.float32)
                                align_q = q_off
                            pose_pos_aligned = transform_point_cloud(align_t.reshape(1, 3), pos, quat_raw)[0].astype(np.float32, copy=False)
                            pose_quat_aligned = quat_mul_wxyz(quat_raw, align_q).astype(np.float32)
                            object_poses_aligned[entity_name].append((pose_pos_aligned, pose_quat_aligned))

                            transformed = transform_point_cloud(raw_points, pos, quat_raw)
                            object_pcds[entity_name][t] = transformed
                            object_poses[entity_name].append((pose_pos, quat_aligned))

                        if gripper_root_name in poses:
                            root_pos, root_quat = poses[gripper_root_name]
                        else:
                            root_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                            root_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
                        gripper_pose_list.append((root_pos, np.asarray(root_quat, dtype=np.float32)))

                pcd_group = demo_group.require_group("object_pcds")
                for key in list(pcd_group.keys()):
                    del pcd_group[key]
                for entity_name, pcd_traj in object_pcds.items():
                    pcd_group.create_dataset(entity_name, data=np.asarray(pcd_traj, dtype=np.float32), compression="gzip")

                poses_group = demo_group.require_group("object_poses")
                for key in list(poses_group.keys()):
                    del poses_group[key]
                for entity_name, pose_list in object_poses.items():
                    pose_arr = np.asarray([np.concatenate([p, q], axis=0) for p, q in pose_list], dtype=np.float32)
                    poses_group.create_dataset(entity_name, data=pose_arr, compression="gzip")

                aligned_group = demo_group.require_group("object_poses_aligned")
                for key in list(aligned_group.keys()):
                    del aligned_group[key]
                for entity_name, pose_list in object_poses_aligned.items():
                    pose_arr = np.asarray([np.concatenate([p, q], axis=0) for p, q in pose_list], dtype=np.float32)
                    aligned_group.create_dataset(entity_name, data=pose_arr, compression="gzip")

                gripper_pose_group = demo_group.require_group("gripper_poses")
                for key in list(gripper_pose_group.keys()):
                    del gripper_pose_group[key]
                if gripper_pose_list:
                    gripper_pose_arr = np.asarray([np.concatenate([p, q], axis=0) for p, q in gripper_pose_list], dtype=np.float32)
                    gripper_pose_group.create_dataset("robot_gripper", data=gripper_pose_arr, compression="gzip")
                if "gripper_width" in demo_group:
                    del demo_group["gripper_width"]
                demo_group.create_dataset("gripper_width", data=gripper_width_traj, compression="gzip")

                if add_distance:
                    distance_entity_names = sorted(object_pcds.keys())
                    dist_values = compute_pairwise_distance_traj(object_pcds, distance_entity_names)
                    dist_group = demo_group.require_group("object_distances")
                    for key in list(dist_group.keys()):
                        del dist_group[key]
                    str_dtype = h5py.string_dtype(encoding="utf-8")
                    dist_group.create_dataset(
                        "entity_names",
                        data=np.asarray(distance_entity_names, dtype=str_dtype),
                    )
                    dist_group.create_dataset(
                        "values",
                        data=np.asarray(dist_values, dtype=np.float32),
                        compression="gzip",
                    )

                total_entities = len(object_pcds)
                print(f"    {ep}: {num_steps} steps, {total_entities} entities")

        print(f"  Saved to: {output_path}")
        return True
    finally:
        pose_extractor.close()


def _parse_exclude_data_arg(exclude_data: object) -> List[str]:
    if exclude_data is None:
        return []
    if isinstance(exclude_data, (list, tuple, set)):
        out = []
        for item in exclude_data:
            out.extend(_parse_exclude_data_arg(item))
        return out
    text = str(exclude_data).strip()
    if not text:
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


IMAGE_OBS_TOKENS = ("image", "rgb", "depth")


def _is_image_obs_key(name: str) -> bool:
    norm = canonical_name(name)
    return any(tok in norm for tok in IMAGE_OBS_TOKENS)


def strip_image_observations_inplace(h5_file: h5py.File) -> int:
    data_group = h5_file.get("data")
    if not isinstance(data_group, h5py.Group):
        return 0
    removed = 0
    for demo_key in list(data_group.keys()):
        demo_group = data_group.get(demo_key)
        if not isinstance(demo_group, h5py.Group):
            continue
        for obs_key in ("obs", "next_obs"):
            obs_group = demo_group.get(obs_key)
            if not isinstance(obs_group, h5py.Group):
                continue
            to_delete = [key for key in list(obs_group.keys()) if _is_image_obs_key(key)]
            for key in to_delete:
                del obs_group[key]
                removed += 1
    return removed


def discover_excluded_hdf5_files(exclude_data: object) -> Tuple[Set[str], List[str]]:
    patterns = _parse_exclude_data_arg(exclude_data)
    excluded: Set[str] = set()
    unresolved: List[str] = []
    for pat in patterns:
        path = os.path.expanduser(pat)
        matched: List[str] = []
        if os.path.isfile(path):
            if path.endswith('.hdf5'):
                matched = [path]
        elif os.path.isdir(path):
            matched = sorted(glob.glob(os.path.join(path, '**', '*.hdf5'), recursive=True))
        else:
            matched = sorted(glob.glob(path, recursive=True))
            matched = [f for f in matched if os.path.isfile(f) and f.endswith('.hdf5')]

        if not matched:
            unresolved.append(pat)
            continue
        for f in matched:
            excluded.add(os.path.realpath(f))
    return excluded, unresolved


def process_dataset(
    dataset_path,
    output_dir,
    num_points_per_obj,
    overwrite=False,
    split_mesh_subparts=False,
    add_distance=False,
    center_by_surface=False,
    exclude_data=None,
    skip_existing_output=False,
    latent_ckpt=None,
    ignore_image=False,
):
    excluded_files, unresolved_excludes = discover_excluded_hdf5_files(exclude_data)
    if unresolved_excludes:
        print(f"[WARN] --exclude-data has no matched hdf5: {unresolved_excludes}")
    if excluded_files:
        print(f"Resolved excluded hdf5 files: {len(excluded_files)}")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if os.path.isfile(dataset_path):
        if not dataset_path.endswith(".hdf5"):
            print(f"Error: {dataset_path} is not an HDF5 file")
            return
        if os.path.realpath(dataset_path) in excluded_files:
            print(f"Skip excluded file: {dataset_path}")
            return

        basename = os.path.basename(dataset_path)
        basename_pcd = basename.replace(".hdf5", "_pcd.hdf5")
        output_path = os.path.join(output_dir, basename_pcd) if output_dir else dataset_path.replace(".hdf5", "_pcd.hdf5")
        if skip_existing_output and os.path.exists(output_path):
            print(f"Skip existing output: {output_path}")
            return
        process_hdf5_file(
            dataset_path,
            output_path,
            num_points_per_obj,
            overwrite=overwrite,
            split_mesh_subparts=split_mesh_subparts,
            add_distance=add_distance,
            center_by_surface=center_by_surface,
            latent_ckpt=latent_ckpt,
            ignore_image=ignore_image,
        )
        return

    if os.path.isdir(dataset_path):
        hdf5_files = sorted(glob.glob(os.path.join(dataset_path, "**", "*.hdf5"), recursive=True))
        if excluded_files:
            total_before = len(hdf5_files)
            hdf5_files = [f for f in hdf5_files if os.path.realpath(f) not in excluded_files]
            print(f"Excluded files by --exclude-data: {total_before - len(hdf5_files)}")
        print(f"Found {len(hdf5_files)} HDF5 files in {dataset_path}")
        for hdf5_file in hdf5_files:
            rel_path = os.path.relpath(hdf5_file, start=dataset_path)
            rel_dir = os.path.dirname(rel_path)
            basename = os.path.basename(rel_path)
            basename_pcd = basename.replace(".hdf5", "_pcd.hdf5")

            if output_dir:
                if rel_dir and rel_dir != ".":
                    output_path = os.path.join(output_dir, rel_dir, basename_pcd)
                else:
                    output_path = os.path.join(output_dir, basename_pcd)
            else:
                output_path = hdf5_file.replace(".hdf5", "_pcd.hdf5")

            if skip_existing_output and os.path.exists(output_path):
                print(f"Skip existing output: {output_path}")
                continue

            try:
                process_hdf5_file(
                    hdf5_file,
                    output_path,
                    num_points_per_obj,
                    overwrite=overwrite,
                    split_mesh_subparts=split_mesh_subparts,
                    add_distance=add_distance,
                    center_by_surface=center_by_surface,
                    latent_ckpt=latent_ckpt,
                    ignore_image=ignore_image,
                )
            except Exception as e:
                print(f"Error processing {hdf5_file}: {e}")
        return

    print(f"Error: {dataset_path} does not exist")


def main():
    parser = argparse.ArgumentParser(description="Generate point cloud trajectories from LIBERO dataset")
    parser.add_argument("--dataset", type=str, required=True, help="Path to HDF5 file or directory")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/franka-client/datasets/libero_pcd_center_aligned",
        help="Output directory (default: use this fixed path)",
    )
    parser.add_argument(
        "--num_points_per_obj",
        type=int,
        default=1024,
        help="Number of points per output entity",
    )
    parser.add_argument(
        "--num_points",
        type=int,
        default=None,
        help="Alias of --num_points_per_obj",
    )
    parser.add_argument(
        "--split-mesh-subparts",
        dest="split_mesh_subparts",
        action="store_true",
        default=True,
        help=(
            "Split mesh geoms to per-mesh entities (default: enabled)."
        ),
    )
    parser.add_argument(
        "--merge-mesh-subparts",
        dest="split_mesh_subparts",
        action="store_false",
        help=(
            "Legacy mode: merge multiple mesh geoms inside one XML part into one entity."
        ),
    )
    parser.add_argument(
        "--add-distance",
        action="store_true",
        help=(
            "Compute and save per-step pairwise minimum distances between entity point clouds "
            "into data/<demo>/object_distances."
        ),
    )
    parser.add_argument(
        "--center-by-surface",
        action="store_true",
        help=(
            "Recenter each object by subtracting its local sampled surface centroid. "
            "Object poses are shifted to centroid poses accordingly."
        ),
    )
    parser.add_argument(
        "--latent-ckpt",
        type=str,
        default=None,
        help=(
            "Optional SDF checkpoint. If provided, export a stable latent mapping table into the dataset "
            "so train/eval can reuse it without rematching."
        ),
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing object_pcds")
    parser.add_argument(
        "--skip-existing-output",
        action="store_true",
        help="Skip processing when target output *_pcd.hdf5 already exists.",
    )
    parser.add_argument(
        "--ignore-image",
        action="store_true",
        help=(
            "Remove image-like observation datasets (e.g. *image*, *rgb*, *depth*) "
            "from the copied output HDF5 while keeping non-image data."
        ),
    )
    parser.add_argument(
        "--exclude-data",
        type=str,
        default=None,
        help=(
            "Optional hdf5 file/dir/glob to exclude from dataset discovery. "
            "Supports comma-separated patterns."
        ),
    )
    args = parser.parse_args()

    if args.num_points is not None:
        args.num_points_per_obj = args.num_points

    if args.overwrite and args.skip_existing_output:
        parser.error("--overwrite and --skip-existing-output cannot be enabled together")

    process_dataset(
        args.dataset,
        args.output_dir,
        args.num_points_per_obj,
        overwrite=args.overwrite,
        split_mesh_subparts=bool(args.split_mesh_subparts),
        add_distance=bool(args.add_distance),
        center_by_surface=bool(args.center_by_surface),
        exclude_data=args.exclude_data,
        skip_existing_output=bool(args.skip_existing_output),
        latent_ckpt=args.latent_ckpt,
        ignore_image=bool(args.ignore_image),
    )


if __name__ == "__main__":
    main()
