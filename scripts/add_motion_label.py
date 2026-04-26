from __future__ import annotations

import argparse
import csv
import glob
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import h5py
import numpy as np


MOTION_TYPE_TO_ID = {
    "static": 0,
    "movable": 1,
    "articulated": 2,
}
MOTION_LABEL_NAMES = [name for name, _idx in sorted(MOTION_TYPE_TO_ID.items(), key=lambda kv: kv[1])]

ARTICULATED_KEYWORDS = {
    "drawer",
    "door",
    "button",
    "knob",
    "lid",
    "hinge",
}
ARTICULATED_PART_KEYWORDS = {
    "top",
    "middle",
    "bottom",
    "handle",
}
STATIC_KEYWORDS = {
    "table",
    "floor",
    "ground",
    "burner",
    "base",
}
FIXTURE_BODY_KEYWORDS = {
    "cabinet",
    "microwave",
    "stove",
}


def discover_hdf5_files(dataset_root: str) -> List[str]:
    path = os.path.expanduser(str(dataset_root))
    if os.path.isfile(path):
        return [path] if path.endswith(".hdf5") else []
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "**", "*.hdf5"), recursive=True))
        return [f for f in files if os.path.isfile(f)]
    return []


def canonical_name(text: str) -> str:
    s = str(text).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def decode_h5_text(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def tokenize_name(text: str) -> Set[str]:
    return {tok for tok in canonical_name(text).split("_") if tok}


def parse_entity_name(entity_name: str) -> Tuple[str, str]:
    if "__" not in str(entity_name):
        return str(entity_name), ""
    instance_name, part_name = str(entity_name).split("__", 1)
    return instance_name, part_name


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


def read_target_npz_map(h5_file: h5py.File) -> Dict[str, str]:
    out: Dict[str, str] = {}
    meta = h5_file.get("meta")
    if not isinstance(meta, h5py.Group):
        return out
    lm = meta.get("latent_mapping")
    if not isinstance(lm, h5py.Group):
        return out
    if "entity_names" not in lm or "target_npz_names" not in lm:
        return out
    entity_names = [decode_h5_text(x) for x in np.asarray(lm["entity_names"])]
    target_npz_names = [decode_h5_text(x) for x in np.asarray(lm["target_npz_names"])]
    for entity_name, target_npz_name in zip(entity_names, target_npz_names):
        out[str(entity_name)] = str(target_npz_name)
        out[canonical_name(str(entity_name))] = str(target_npz_name)
    return out


def infer_motion_type(entity_name: str, target_npz_name: str = "") -> str:
    entity_name = str(entity_name)
    target_npz_name = str(target_npz_name)
    entity_canon = canonical_name(entity_name)
    instance_name, part_name = parse_entity_name(entity_name)
    instance_tokens = tokenize_name(instance_name)
    part_tokens = tokenize_name(part_name)
    target_tokens = tokenize_name(Path(target_npz_name).name.replace("--sdfae_labels.npz", ""))
    all_tokens = set(instance_tokens) | set(part_tokens) | set(target_tokens)
    assembly_like = (
        ("__" in entity_name)
        or ("articulated_objects" in target_npz_name.lower())
        or ("combined" in target_npz_name.lower())
    )

    if entity_canon.startswith("robot_gripper"):
        return "movable"

    if entity_canon == "scene_table":
        return "static"

    if all_tokens & ARTICULATED_KEYWORDS:
        return "articulated"

    if "cabinet" in all_tokens and (part_tokens & {"top", "middle", "bottom"}):
        return "articulated"

    if assembly_like and ("handle" in part_tokens or ("handle" in all_tokens and "__" in entity_name)):
        return "articulated"

    if part_tokens & {"base", "burner"}:
        return "static"

    if {"table"} & all_tokens:
        return "static"
    if {"floor", "ground"} & all_tokens:
        return "static"
    if "burner" in all_tokens:
        return "static"

    if ("__" not in entity_name) and (FIXTURE_BODY_KEYWORDS & all_tokens):
        return "static"

    return "movable"


def write_motion_labels(
    demo_group: h5py.Group,
    pose_group: h5py.Group,
    target_npz_map: Dict[str, str],
) -> Dict[str, int]:
    label_group_name = "object_motion_labels"
    if label_group_name in demo_group:
        del demo_group[label_group_name]
    label_group = demo_group.create_group(label_group_name)

    entity_names = sorted(str(name) for name in pose_group.keys())

    counts = {name: 0 for name in MOTION_LABEL_NAMES}
    for entity_name in entity_names:
        target_npz_name = str(
            target_npz_map.get(entity_name, target_npz_map.get(canonical_name(entity_name), ""))
        )
        motion_type = infer_motion_type(entity_name, target_npz_name=target_npz_name)
        label_group.create_dataset(
            entity_name,
            data=np.uint8(MOTION_TYPE_TO_ID[motion_type]),
            dtype=np.uint8,
        )
        counts[motion_type] += 1

    label_group.attrs["pose_group_name"] = str(pose_group.name).split("/")[-1]
    label_group.attrs["label_schema_version"] = 2
    label_group.attrs["label_names"] = np.asarray(MOTION_LABEL_NAMES, dtype=h5py.string_dtype(encoding="utf-8"))

    return {
        "entities": int(len(entity_names)),
        "static": int(counts["static"]),
        "movable": int(counts["movable"]),
        "articulated": int(counts["articulated"]),
    }


def collect_motion_labels_for_demo(
    pose_group: h5py.Group,
    target_npz_map: Dict[str, str],
) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for entity_name in sorted(str(name) for name in pose_group.keys()):
        target_npz_name = str(
            target_npz_map.get(entity_name, target_npz_map.get(canonical_name(entity_name), ""))
        )
        motion_type = infer_motion_type(entity_name, target_npz_name=target_npz_name)
        out[entity_name] = int(MOTION_TYPE_TO_ID[motion_type])
    return out


def task_name_from_path(file_path: str, dataset_root: str) -> str:
    fp = Path(file_path).resolve()
    root = Path(dataset_root).resolve()
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
    dry_run: bool,
    manifest: Dict[str, Dict[str, object]],
) -> Dict[str, float]:
    demos_total = 0
    entities_total = 0
    static_total = 0
    movable_total = 0
    articulated_total = 0
    demos_skipped = 0
    task_name = task_name_from_path(file_path, dataset_root=dataset_root)

    with h5py.File(file_path, "r" if dry_run else "r+") as f:
        data_group = f.get("data")
        if not isinstance(data_group, h5py.Group):
            return {
                "demos_total": 0.0,
                "entities_total": 0.0,
                "static_total": 0.0,
                "movable_total": 0.0,
                "articulated_total": 0.0,
                "demos_skipped": 0.0,
            }

        target_npz_map = read_target_npz_map(f)

        for demo_key in sorted(data_group.keys()):
            demo_group = data_group[demo_key]
            if not isinstance(demo_group, h5py.Group):
                demos_skipped += 1
                continue
            pose_group = get_preferred_object_pose_group(demo_group, pose_group_name=pose_group_name)
            if not isinstance(pose_group, h5py.Group):
                demos_skipped += 1
                continue

            stats = (
                write_motion_labels(demo_group, pose_group, target_npz_map=target_npz_map)
                if not dry_run
                else _dry_run_collect(pose_group, target_npz_map=target_npz_map)
            )
            demos_total += 1
            entities_total += int(stats["entities"])
            static_total += int(stats["static"])
            movable_total += int(stats["movable"])
            articulated_total += int(stats["articulated"])

            labels = collect_motion_labels_for_demo(pose_group, target_npz_map=target_npz_map)
            for entity_name, label_id in labels.items():
                motion_type = MOTION_LABEL_NAMES[int(label_id)]
                entry = manifest.setdefault(
                    entity_name,
                    {
                        "motion_type": motion_type,
                        "tasks": set(),
                        "target_npz_names": set(),
                    },
                )
                prev = str(entry["motion_type"])
                if prev != motion_type:
                    raise ValueError(
                        f"Inconsistent motion type for entity {entity_name}: {prev} vs {motion_type}"
                    )
                entry["tasks"].add(task_name)
                target_npz_name = str(
                    target_npz_map.get(entity_name, target_npz_map.get(canonical_name(entity_name), ""))
                )
                if target_npz_name:
                    entry["target_npz_names"].add(target_npz_name)

    return {
        "demos_total": float(demos_total),
        "entities_total": float(entities_total),
        "static_total": float(static_total),
        "movable_total": float(movable_total),
        "articulated_total": float(articulated_total),
        "demos_skipped": float(demos_skipped),
    }


def _dry_run_collect(
    pose_group: h5py.Group,
    *,
    target_npz_map: Dict[str, str],
) -> Dict[str, int]:
    counts = {name: 0 for name in MOTION_LABEL_NAMES}
    entity_names = sorted(str(name) for name in pose_group.keys())
    for entity_name in entity_names:
        target_npz_name = str(
            target_npz_map.get(entity_name, target_npz_map.get(canonical_name(entity_name), ""))
        )
        motion_type = infer_motion_type(entity_name, target_npz_name=target_npz_name)
        counts[motion_type] += 1
    return {
        "entities": int(len(entity_names)),
        "static": int(counts["static"]),
        "movable": int(counts["movable"]),
        "articulated": int(counts["articulated"]),
    }


def export_manifest_csv(manifest: Dict[str, Dict[str, object]], csv_path: str) -> None:
    out_path = Path(csv_path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "entity_name",
                "motion_type",
                "tasks",
                "target_npz_names",
            ],
        )
        writer.writeheader()
        for entity_name in sorted(manifest.keys()):
            entry = manifest[entity_name]
            writer.writerow(
                {
                    "entity_name": entity_name,
                    "motion_type": str(entry["motion_type"]),
                    "tasks": ";".join(sorted(str(x) for x in entry["tasks"])),
                    "target_npz_names": ";".join(sorted(str(x) for x in entry["target_npz_names"])),
                }
            )


def default_export_csv_path(dataset_path: str) -> str:
    path = Path(dataset_path).expanduser()
    if path.is_dir():
        return str(path / "object_motion_manifest.csv")
    return str(path.with_name(path.stem + "_object_motion_manifest.csv"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add stable motion-type labels (static/movable/articulated) to LIBERO HDF5 files.")
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to a single .hdf5 file or a dataset directory containing .hdf5 files.",
    )
    parser.add_argument(
        "--pose-group",
        default="auto",
        choices=["auto", "object_poses_npz", "object_poses_aligned", "object_poses"],
        help="Which pose group to use when enumerating entities.",
    )
    parser.add_argument(
        "--export-csv",
        default="",
        help="Optional path to export a CSV manifest of entity_name/tasks/motion_type. Defaults next to dataset.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report statistics without writing labels back to the dataset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = discover_hdf5_files(args.dataset)
    if not files:
        raise FileNotFoundError(f"No .hdf5 files found under: {args.dataset}")

    export_csv_path = str(args.export_csv).strip() or default_export_csv_path(args.dataset)

    demos_total = 0
    entities_total = 0
    static_total = 0
    movable_total = 0
    articulated_total = 0
    skipped_total = 0
    manifest: Dict[str, Dict[str, object]] = {}

    print(
        f"[INFO] Processing {len(files)} files | pose_group={args.pose_group} | "
        f"dry_run={bool(args.dry_run)} | export_csv={export_csv_path}"
    )

    for idx, file_path in enumerate(files, start=1):
        stats = process_file(
            file_path,
            dataset_root=str(args.dataset),
            pose_group_name=str(args.pose_group),
            dry_run=bool(args.dry_run),
            manifest=manifest,
        )
        demos_total += int(stats["demos_total"])
        entities_total += int(stats["entities_total"])
        static_total += int(stats["static_total"])
        movable_total += int(stats["movable_total"])
        articulated_total += int(stats["articulated_total"])
        skipped_total += int(stats["demos_skipped"])
        print(
            f"[{idx}/{len(files)}] {file_path} | demos={int(stats['demos_total'])} | "
            f"entities={int(stats['entities_total'])} | static={int(stats['static_total'])} | "
            f"movable={int(stats['movable_total'])} | articulated={int(stats['articulated_total'])} | "
            f"skipped={int(stats['demos_skipped'])}"
        )

    export_manifest_csv(manifest, export_csv_path)
    print(
        "[DONE] "
        f"files={len(files)} demos={demos_total} entities={entities_total} "
        f"static={static_total} movable={movable_total} articulated={articulated_total} "
        f"skipped_demos={skipped_total} csv={export_csv_path}"
    )


if __name__ == "__main__":
    main()
