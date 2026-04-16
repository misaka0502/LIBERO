#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable

import h5py


IMAGE_KEY_HINTS = ("image", "rgb", "depth")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy a LIBERO dataset directory to a new location and remove image "
            "observations from the copied HDF5 files. The source dataset is left unchanged."
        )
    )
    parser.add_argument(
        "--src",
        required=True,
        help="Source dataset root, for example /home/franka-client/datasets/libero_pcd_center_aligned_new",
    )
    parser.add_argument(
        "--dst",
        required=True,
        help="Destination dataset root to create.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the destination directory if it already exists.",
    )
    return parser.parse_args()


def is_image_key(name: str) -> bool:
    lower_name = name.lower()
    return any(hint in lower_name for hint in IMAGE_KEY_HINTS)


def should_skip_h5_child(parent_path: str, child_name: str) -> bool:
    if parent_path.endswith("/obs") or parent_path.endswith("/next_obs"):
        return is_image_key(child_name)
    return False


def copy_attrs(src_obj: h5py.HLObject, dst_obj: h5py.HLObject) -> None:
    for key, value in src_obj.attrs.items():
        dst_obj.attrs[key] = value


def create_dataset_like(src_dataset: h5py.Dataset, dst_group: h5py.Group, name: str) -> None:
    dataset_kwargs = {}
    if src_dataset.chunks is not None:
        dataset_kwargs["chunks"] = src_dataset.chunks
    if src_dataset.compression is not None:
        dataset_kwargs["compression"] = src_dataset.compression
    if src_dataset.compression_opts is not None:
        dataset_kwargs["compression_opts"] = src_dataset.compression_opts
    if src_dataset.shuffle:
        dataset_kwargs["shuffle"] = src_dataset.shuffle
    if src_dataset.fletcher32:
        dataset_kwargs["fletcher32"] = src_dataset.fletcher32
    if src_dataset.scaleoffset is not None:
        dataset_kwargs["scaleoffset"] = src_dataset.scaleoffset
    if src_dataset.maxshape is not None:
        dataset_kwargs["maxshape"] = src_dataset.maxshape

    dst_dataset = dst_group.create_dataset(
        name,
        data=src_dataset[()],
        dtype=src_dataset.dtype,
        **dataset_kwargs,
    )
    copy_attrs(src_dataset, dst_dataset)


def copy_h5_group(src_group: h5py.Group, dst_group: h5py.Group, group_path: str = "") -> int:
    skipped_count = 0
    copy_attrs(src_group, dst_group)
    for child_name, child_obj in src_group.items():
        if should_skip_h5_child(group_path, child_name):
            skipped_count += 1
            continue
        child_path = f"{group_path}/{child_name}" if group_path else f"/{child_name}"
        if isinstance(child_obj, h5py.Group):
            new_group = dst_group.create_group(child_name)
            skipped_count += copy_h5_group(child_obj, new_group, child_path)
        elif isinstance(child_obj, h5py.Dataset):
            create_dataset_like(child_obj, dst_group, child_name)
        else:
            raise TypeError(f"Unsupported HDF5 object at {child_path}: {type(child_obj)!r}")
    return skipped_count


def rewrite_hdf5_without_images(src_path: Path, dst_path: Path) -> int:
    with h5py.File(src_path, "r") as src_h5, h5py.File(dst_path, "w") as dst_h5:
        return copy_h5_group(src_h5, dst_h5, "")


def iter_relative_paths(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        yield path.relative_to(root)


def main() -> int:
    args = parse_args()
    src_root = Path(args.src).expanduser().resolve()
    dst_root = Path(args.dst).expanduser().resolve()

    if not src_root.exists():
        raise FileNotFoundError(f"Source dataset root does not exist: {src_root}")
    if not src_root.is_dir():
        raise NotADirectoryError(f"Source dataset root is not a directory: {src_root}")

    if dst_root.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Destination already exists: {dst_root}. Use --overwrite to replace it."
            )
        shutil.rmtree(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)

    relative_paths = list(iter_relative_paths(src_root))
    hdf5_relative_paths = [path for path in relative_paths if path.suffix == ".hdf5"]
    total_removed = 0

    print(f"[copy] {src_root} -> {dst_root}")
    for rel_path in relative_paths:
        src_path = src_root / rel_path
        dst_path = dst_root / rel_path
        if src_path.is_dir():
            dst_path.mkdir(parents=True, exist_ok=True)
            continue
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        if src_path.suffix == ".hdf5":
            removed = rewrite_hdf5_without_images(src_path, dst_path)
            total_removed += removed
            hdf5_index = hdf5_relative_paths.index(rel_path) + 1
            print(
                f"[rewrite] ({hdf5_index}/{len(hdf5_relative_paths)}) {dst_path} "
                f"removed_keys={removed}"
            )
        else:
            shutil.copy2(src_path, dst_path)

    print(
        f"[done] copied dataset to {dst_root}, rewrote {len(hdf5_relative_paths)} hdf5 files, and removed "
        f"{total_removed} image datasets"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
