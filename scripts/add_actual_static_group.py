from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np


GROUP_NAME = "object_context_mask_info"
SCHEMA_VERSION = 2


TASK_CRITICAL_ALIASES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("wooden_cabinet", ("wooden_cabinet", "cabinet", "drawer")),
    ("cabinet", ("wooden_cabinet", "cabinet", "drawer")),
    ("flat_stove", ("flat_stove", "stove", "burner")),
    ("stove", ("flat_stove", "stove", "burner")),
    ("microwave", ("microwave", "microwave_oven")),
    ("plate", ("plate",)),
    ("bowl", ("bowl",)),
    ("ramekin", ("ramekin",)),
    ("cookie", ("cookie", "cookies", "cookie_box")),
    ("mug", ("mug",)),
    ("basket", ("basket",)),
    ("rack", ("rack",)),
    ("wine_bottle", ("wine_bottle", "bottle", "wine")),
    ("book", ("book",)),
    ("caddy", ("caddy",)),
    ("moka_pot", ("moka_pot", "moka")),
    ("cream_cheese", ("cream_cheese", "cheese")),
    ("alphabet_soup", ("alphabet_soup", "soup")),
    ("bbq_sauce", ("bbq_sauce", "barbecue_sauce")),
    ("butter", ("butter",)),
    ("chocolate_pudding", ("chocolate_pudding", "pudding")),
    ("ketchup", ("ketchup",)),
    ("milk", ("milk",)),
    ("orange_juice", ("orange_juice", "juice")),
    ("salad_dressing", ("salad_dressing", "dressing")),
    ("tomato_sauce", ("tomato_sauce",)),
)

GENERIC_ENTITY_TOKENS = {
    "1",
    "2",
    "3",
    "4",
    "5",
    "a",
    "an",
    "base",
    "body",
    "bottom",
    "button",
    "cabinet",
    "door",
    "drawer",
    "flat",
    "geom",
    "handle",
    "left",
    "middle",
    "object",
    "part",
    "right",
    "scene",
    "table",
    "top",
}


def discover_hdf5_files(dataset_root: str) -> List[str]:
    path = os.path.expanduser(str(dataset_root))
    if os.path.isfile(path):
        return [path] if path.endswith(".hdf5") else []
    if os.path.isdir(path):
        return [f for f in sorted(glob.glob(os.path.join(path, "**", "*.hdf5"), recursive=True)) if os.path.isfile(f)]
    return []


def canonical_name(text: str) -> str:
    s = str(text).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def parse_entity_name(entity_name: str) -> Tuple[str, str]:
    if "__" not in str(entity_name):
        return str(entity_name), ""
    instance_name, part_name = str(entity_name).split("__", 1)
    return instance_name, part_name


def decode_h5_text(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def get_preferred_object_pose_group(
    demo_group: h5py.Group,
    pose_group_name: str = "auto",
) -> Optional[h5py.Group]:
    if str(pose_group_name) != "auto":
        group = demo_group.get(str(pose_group_name))
        return group if isinstance(group, h5py.Group) else None

    poses_npz = demo_group.get("object_poses_npz")
    if isinstance(poses_npz, h5py.Group) and len(poses_npz.keys()) > 0:
        return poses_npz
    aligned = demo_group.get("object_poses_aligned")
    if isinstance(aligned, h5py.Group) and len(aligned.keys()) > 0:
        return aligned
    poses = demo_group.get("object_poses")
    if isinstance(poses, h5py.Group) and len(poses.keys()) > 0:
        return poses
    return None


def quat_relative_angle_deg(q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
    qa = np.asarray(q0, dtype=np.float64)
    qb = np.asarray(q1, dtype=np.float64)
    qa = qa / np.maximum(np.linalg.norm(qa, axis=-1, keepdims=True), 1e-12)
    qb = qb / np.maximum(np.linalg.norm(qb, axis=-1, keepdims=True), 1e-12)
    dot = np.abs(np.sum(qa * qb, axis=-1))
    dot = np.clip(dot, -1.0, 1.0)
    return np.degrees(2.0 * np.arccos(dot)).astype(np.float32, copy=False)


def is_actual_static_pose_series(
    poses: np.ndarray,
    *,
    position_threshold_m: float,
    rotation_threshold_deg: float,
) -> Tuple[bool, float, float]:
    arr = np.asarray(poses, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 7 or arr.shape[0] <= 0:
        return True, 0.0, 0.0
    pos = arr[:, 0:3]
    quat = arr[:, 3:7]
    trans = np.linalg.norm(pos - pos[0:1], axis=-1)
    rot = quat_relative_angle_deg(quat[0:1], quat)
    max_trans = float(np.nanmax(trans)) if trans.size else 0.0
    max_rot = float(np.nanmax(rot)) if rot.size else 0.0
    is_static = bool(max_trans <= float(position_threshold_m) and max_rot <= float(rotation_threshold_deg))
    return is_static, max_trans, max_rot


def _part_tokens(part_name: str) -> set[str]:
    return {tok for tok in canonical_name(part_name).split("_") if tok}


def infer_mask_group_name(entity_name: str) -> str:
    """Return a conservative structural group for context masking.

    We group multi-part fixtures by instance so training never masks a support
    piece while leaving dependent parts visible. Ordinary standalone objects
    remain single-entity groups.
    """
    entity_name = str(entity_name)
    inst, part = parse_entity_name(entity_name)
    inst_c = canonical_name(inst)
    entity_c = canonical_name(entity_name)
    part_toks = _part_tokens(part)

    if entity_c.startswith("robot_gripper"):
        return "robot_gripper"
    if entity_c == "scene_table":
        return "scene_table"
    if "__" in entity_name:
        if any(tok in inst_c for tok in ("cabinet", "microwave", "stove")):
            return f"assembly:{inst_c}"
        if part_toks & {
            "base",
            "top",
            "middle",
            "bottom",
            "drawer",
            "door",
            "handle",
            "button",
            "burner",
            "lid",
            "knob",
            "panel",
        }:
            return f"assembly:{inst_c}"
    return f"entity:{entity_c}"


def _task_tokens(task_name: str) -> set[str]:
    return {tok for tok in canonical_name(task_name).split("_") if tok}


def _contains_alias(task_c: str, aliases: Sequence[str]) -> bool:
    padded = f"_{task_c}_"
    return any(f"_{canonical_name(alias)}_" in padded for alias in aliases)


def infer_context_critical(entity_name: str, task_name: str) -> bool:
    """Return whether an entity is task-mentioned support/target context.

    This is intentionally conservative: for random static-context masking it is
    better to keep a few irrelevant objects than to remove a target/support
    object that explains the interaction.
    """
    entity_c = canonical_name(entity_name)
    task_c = canonical_name(task_name)
    task_toks = _task_tokens(task_name)
    inst, part = parse_entity_name(entity_name)
    inst_c = canonical_name(inst)
    part_c = canonical_name(part)

    if entity_c.startswith("robot_gripper") or entity_c == "scene_table":
        return True

    for entity_key, aliases in TASK_CRITICAL_ALIASES:
        if entity_key in entity_c and _contains_alias(task_c, aliases):
            return True
        if entity_key in inst_c and _contains_alias(task_c, aliases):
            return True

    entity_toks = {tok for tok in re.split(r"_+", entity_c) if tok and tok not in GENERIC_ENTITY_TOKENS}
    inst_toks = {tok for tok in re.split(r"_+", inst_c) if tok and tok not in GENERIC_ENTITY_TOKENS}
    part_toks = {tok for tok in re.split(r"_+", part_c) if tok and tok not in GENERIC_ENTITY_TOKENS}
    if (entity_toks | inst_toks | part_toks) & task_toks:
        return True

    return False


def _group_arrays(
    group_ids: Sequence[int],
    group_names_by_id: Sequence[str],
    is_static_vals: Sequence[bool],
    is_context_critical_vals: Sequence[bool],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    group_count = len(group_names_by_id)
    all_static = np.ones((group_count,), dtype=np.bool_)
    any_critical = np.zeros((group_count,), dtype=np.bool_)
    for idx, is_static, is_critical in zip(group_ids, is_static_vals, is_context_critical_vals):
        group_idx = int(idx)
        all_static[group_idx] = bool(all_static[group_idx] and bool(is_static))
        any_critical[group_idx] = bool(any_critical[group_idx] or bool(is_critical))

    eligible = all_static & ~any_critical
    for idx, group_name in enumerate(group_names_by_id):
        if str(group_name) in {"robot_gripper", "scene_table"}:
            eligible[idx] = False
    return all_static, any_critical, eligible


def write_context_mask_info(
    demo_group: h5py.Group,
    pose_group: h5py.Group,
    *,
    task_name: str,
    position_threshold_m: float,
    rotation_threshold_deg: float,
    dry_run: bool,
    overwrite: bool,
) -> Dict[str, object]:
    entity_names = sorted(str(name) for name in pose_group.keys())
    is_static_vals: List[bool] = []
    is_context_critical_vals: List[bool] = []
    max_trans_vals: List[float] = []
    max_rot_vals: List[float] = []
    group_names: List[str] = []
    group_name_to_id: Dict[str, int] = {}
    group_ids: List[int] = []

    for entity_name in entity_names:
        poses = np.asarray(pose_group[entity_name], dtype=np.float32)
        is_static, max_trans, max_rot = is_actual_static_pose_series(
            poses,
            position_threshold_m=position_threshold_m,
            rotation_threshold_deg=rotation_threshold_deg,
        )
        group_name = infer_mask_group_name(entity_name)
        if group_name not in group_name_to_id:
            group_name_to_id[group_name] = len(group_name_to_id)
        is_context_critical = infer_context_critical(entity_name, task_name)
        is_static_vals.append(bool(is_static))
        is_context_critical_vals.append(bool(is_context_critical))
        max_trans_vals.append(float(max_trans))
        max_rot_vals.append(float(max_rot))
        group_names.append(group_name)
        group_ids.append(int(group_name_to_id[group_name]))

    group_names_by_id = [""] * len(group_name_to_id)
    for name, idx in group_name_to_id.items():
        group_names_by_id[int(idx)] = name
    group_all_static, group_context_critical, group_mask_eligible = _group_arrays(
        group_ids,
        group_names_by_id,
        is_static_vals,
        is_context_critical_vals,
    )

    if not dry_run:
        if GROUP_NAME in demo_group:
            if not overwrite:
                return {
                    "entities": len(entity_names),
                    "actual_static": int(np.sum(np.asarray(is_static_vals, dtype=np.bool_))),
                    "actual_moving": int(len(entity_names) - np.sum(np.asarray(is_static_vals, dtype=np.bool_))),
                    "context_critical": int(np.sum(np.asarray(is_context_critical_vals, dtype=np.bool_))),
                    "groups": len(group_name_to_id),
                    "eligible_groups": int(np.sum(group_mask_eligible)),
                    "skipped_existing": 1,
                }
            del demo_group[GROUP_NAME]
        out = demo_group.create_group(GROUP_NAME)
        str_dtype = h5py.string_dtype(encoding="utf-8")

        out.create_dataset("entity_names", data=np.asarray(entity_names, dtype=str_dtype), dtype=str_dtype)
        out.create_dataset("is_actual_static", data=np.asarray(is_static_vals, dtype=np.bool_))
        out.create_dataset("is_context_critical", data=np.asarray(is_context_critical_vals, dtype=np.bool_))
        out.create_dataset("mask_group_ids", data=np.asarray(group_ids, dtype=np.int32))
        out.create_dataset("mask_group_names", data=np.asarray(group_names_by_id, dtype=str_dtype), dtype=str_dtype)
        out.create_dataset("entity_mask_group_names", data=np.asarray(group_names, dtype=str_dtype), dtype=str_dtype)
        out.create_dataset("group_all_actual_static", data=group_all_static)
        out.create_dataset("group_is_context_critical", data=group_context_critical)
        out.create_dataset("group_mask_eligible", data=group_mask_eligible)
        out.create_dataset("max_translation_m", data=np.asarray(max_trans_vals, dtype=np.float32))
        out.create_dataset("max_rotation_deg", data=np.asarray(max_rot_vals, dtype=np.float32))
        out.attrs["schema_version"] = SCHEMA_VERSION
        out.attrs["pose_group_name"] = str(pose_group.name).split("/")[-1]
        out.attrs["position_threshold_m"] = float(position_threshold_m)
        out.attrs["rotation_threshold_deg"] = float(rotation_threshold_deg)
        out.attrs["task_name"] = str(task_name)
        out.attrs["description"] = (
            "Auxiliary context-mask metadata. is_actual_static is trajectory-level; "
            "is_context_critical is task-semantics-level and conservative; mask_group_ids encode "
            "structural groups; group_mask_eligible marks groups safe for runtime random masking."
        )

    return {
        "entities": len(entity_names),
        "actual_static": int(np.sum(np.asarray(is_static_vals, dtype=np.bool_))),
        "actual_moving": int(len(entity_names) - np.sum(np.asarray(is_static_vals, dtype=np.bool_))),
        "context_critical": int(np.sum(np.asarray(is_context_critical_vals, dtype=np.bool_))),
        "groups": len(group_name_to_id),
        "eligible_groups": int(np.sum(group_mask_eligible)),
        "skipped_existing": 0,
    }


def task_name_from_path(file_path: str, dataset_root: str) -> str:
    fp = Path(file_path).resolve()
    root = Path(dataset_root).resolve()
    if fp == root or root.is_file():
        return fp.stem
    try:
        rel = fp.relative_to(root)
        return str(rel.with_suffix(""))
    except Exception:
        return fp.stem


def process_file(
    file_path: str,
    *,
    dataset_root: str,
    pose_group_name: str,
    position_threshold_m: float,
    rotation_threshold_deg: float,
    dry_run: bool,
    overwrite: bool,
    manifest_rows: List[Dict[str, object]],
) -> Dict[str, int]:
    demos_total = 0
    demos_skipped = 0
    entities_total = 0
    actual_static_total = 0
    actual_moving_total = 0
    context_critical_total = 0
    groups_total = 0
    eligible_groups_total = 0
    skipped_existing_total = 0
    task_name = task_name_from_path(file_path, dataset_root=dataset_root)

    with h5py.File(file_path, "r" if dry_run else "r+") as f:
        data_group = f.get("data")
        if not isinstance(data_group, h5py.Group):
            return {
                "demos": 0,
                "skipped": 0,
                "entities": 0,
                "actual_static": 0,
                "actual_moving": 0,
                "context_critical": 0,
                "groups": 0,
                "eligible_groups": 0,
                "skipped_existing": 0,
            }

        for demo_key in sorted(data_group.keys()):
            demo_group = data_group[demo_key]
            if not isinstance(demo_group, h5py.Group):
                demos_skipped += 1
                continue
            pose_group = get_preferred_object_pose_group(demo_group, pose_group_name=pose_group_name)
            if not isinstance(pose_group, h5py.Group):
                demos_skipped += 1
                continue

            stats = write_context_mask_info(
                demo_group,
                pose_group,
                task_name=task_name,
                position_threshold_m=position_threshold_m,
                rotation_threshold_deg=rotation_threshold_deg,
                dry_run=dry_run,
                overwrite=overwrite,
            )
            demos_total += 1
            entities_total += int(stats["entities"])
            actual_static_total += int(stats["actual_static"])
            actual_moving_total += int(stats["actual_moving"])
            context_critical_total += int(stats["context_critical"])
            groups_total += int(stats["groups"])
            eligible_groups_total += int(stats["eligible_groups"])
            skipped_existing_total += int(stats["skipped_existing"])

            row_entities: List[str] = []
            row_static: List[bool] = []
            row_critical: List[bool] = []
            row_groups: List[str] = []
            row_max_trans: List[float] = []
            row_max_rot: List[float] = []
            for entity_name in sorted(str(name) for name in pose_group.keys()):
                poses = np.asarray(pose_group[entity_name], dtype=np.float32)
                is_static, max_trans, max_rot = is_actual_static_pose_series(
                    poses,
                    position_threshold_m=position_threshold_m,
                    rotation_threshold_deg=rotation_threshold_deg,
                )
                row_entities.append(str(entity_name))
                row_static.append(bool(is_static))
                row_critical.append(bool(infer_context_critical(entity_name, task_name)))
                row_groups.append(infer_mask_group_name(entity_name))
                row_max_trans.append(float(max_trans))
                row_max_rot.append(float(max_rot))

            group_all_static_map: Dict[str, bool] = {}
            group_critical_map: Dict[str, bool] = {}
            for group_name in sorted(set(row_groups)):
                member_idx = [idx for idx, name in enumerate(row_groups) if name == group_name]
                group_all_static_map[group_name] = all(row_static[idx] for idx in member_idx)
                group_critical_map[group_name] = any(row_critical[idx] for idx in member_idx)

            for entity_name, is_static, is_critical, group_name, max_trans, max_rot in zip(
                row_entities,
                row_static,
                row_critical,
                row_groups,
                row_max_trans,
                row_max_rot,
            ):
                group_all_static = bool(group_all_static_map[group_name])
                group_is_critical = bool(group_critical_map[group_name])
                group_mask_eligible = bool(
                    group_all_static
                    and not group_is_critical
                    and group_name not in {"robot_gripper", "scene_table"}
                )
                manifest_rows.append(
                    {
                        "task": task_name,
                        "demo": str(demo_key),
                        "entity_name": str(entity_name),
                        "is_actual_static": int(bool(is_static)),
                        "is_context_critical": int(bool(is_critical)),
                        "mask_group_name": infer_mask_group_name(entity_name),
                        "group_all_actual_static": int(group_all_static),
                        "group_is_context_critical": int(group_is_critical),
                        "group_mask_eligible": int(group_mask_eligible),
                        "max_translation_m": f"{float(max_trans):.8f}",
                        "max_rotation_deg": f"{float(max_rot):.6f}",
                    }
                )

    return {
        "demos": int(demos_total),
        "skipped": int(demos_skipped),
        "entities": int(entities_total),
        "actual_static": int(actual_static_total),
        "actual_moving": int(actual_moving_total),
        "context_critical": int(context_critical_total),
        "groups": int(groups_total),
        "eligible_groups": int(eligible_groups_total),
        "skipped_existing": int(skipped_existing_total),
    }


def default_export_csv_path(dataset_path: str) -> str:
    path = Path(dataset_path).expanduser()
    if path.is_dir():
        return str(path / "actual_static_group_manifest.csv")
    return str(path.with_name(path.stem + "_actual_static_group_manifest.csv"))


def export_manifest_csv(rows: Sequence[Dict[str, object]], csv_path: str) -> None:
    out_path = Path(csv_path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "task",
                "demo",
                "entity_name",
                "is_actual_static",
                "is_context_critical",
                "mask_group_name",
                "group_all_actual_static",
                "group_is_context_critical",
                "group_mask_eligible",
                "max_translation_m",
                "max_rotation_deg",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Add trajectory-level actual-static labels and structural mask groups for runtime "
            "random context-object masking."
        )
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to a single .hdf5 file or a dataset directory containing .hdf5 files.",
    )
    parser.add_argument(
        "--pose-group",
        default="auto",
        choices=["auto", "object_poses_npz", "object_poses_aligned", "object_poses"],
        help="Which pose group to use when enumerating entities and detecting actual static entities.",
    )
    parser.add_argument(
        "--position-threshold-m",
        type=float,
        default=0.01,
        help="Max translation from frame 0 below which an entity is considered actual-static.",
    )
    parser.add_argument(
        "--rotation-threshold-deg",
        type=float,
        default=5.0,
        help="Max rotation from frame 0 below which an entity is considered actual-static.",
    )
    parser.add_argument(
        "--export-csv",
        default="",
        help="Optional path to export a CSV manifest. Defaults next to dataset.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report statistics and export CSV without writing back to HDF5.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=f"Replace an existing {GROUP_NAME} group. Without this, existing groups are left untouched.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = discover_hdf5_files(args.dataset)
    if not files:
        raise FileNotFoundError(f"No .hdf5 files found under: {args.dataset}")

    export_csv_path = str(args.export_csv).strip() or default_export_csv_path(args.dataset)
    manifest_rows: List[Dict[str, object]] = []

    totals = {
        "demos": 0,
        "skipped": 0,
        "entities": 0,
        "actual_static": 0,
        "actual_moving": 0,
        "context_critical": 0,
        "groups": 0,
        "eligible_groups": 0,
        "skipped_existing": 0,
    }
    print(
        f"[INFO] Processing {len(files)} files | pose_group={args.pose_group} | "
        f"position_threshold_m={float(args.position_threshold_m):.4f} | "
        f"rotation_threshold_deg={float(args.rotation_threshold_deg):.2f} | "
        f"dry_run={bool(args.dry_run)} | overwrite={bool(args.overwrite)} | export_csv={export_csv_path}"
    )

    for idx, file_path in enumerate(files, start=1):
        stats = process_file(
            file_path,
            dataset_root=str(args.dataset),
            pose_group_name=str(args.pose_group),
            position_threshold_m=float(args.position_threshold_m),
            rotation_threshold_deg=float(args.rotation_threshold_deg),
            dry_run=bool(args.dry_run),
            overwrite=bool(args.overwrite),
            manifest_rows=manifest_rows,
        )
        for key in totals:
            totals[key] += int(stats[key])
        print(
            f"[{idx}/{len(files)}] {file_path} | demos={stats['demos']} | "
            f"entities={stats['entities']} | actual_static={stats['actual_static']} | "
            f"actual_moving={stats['actual_moving']} | context_critical={stats['context_critical']} | "
            f"groups={stats['groups']} | eligible_groups={stats['eligible_groups']} | "
            f"skipped_existing={stats['skipped_existing']} | skipped={stats['skipped']}"
        )

    export_manifest_csv(manifest_rows, export_csv_path)
    print(
        "[DONE] "
        f"files={len(files)} demos={totals['demos']} entities={totals['entities']} "
        f"actual_static={totals['actual_static']} actual_moving={totals['actual_moving']} "
        f"context_critical={totals['context_critical']} groups={totals['groups']} "
        f"eligible_groups={totals['eligible_groups']} skipped_existing={totals['skipped_existing']} "
        f"skipped={totals['skipped']} csv={export_csv_path}"
    )


if __name__ == "__main__":
    main()
