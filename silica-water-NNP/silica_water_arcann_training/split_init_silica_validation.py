#!/usr/bin/env python3
"""
Split init_silica_water into training and validation DeepMD systems.

Validation selection, using zero-based frame indices:
  - trajectory 1, last 60 frames: 341..400
  - trajectory 2, last 60 frames: 742..801
  - trajectory 3, first 60 frames: 802..861

The script preserves the original full dataset in:
  data/init_silica_water_full_before_validation_split

and writes:
  data/init_silica_water       -> training frames
  data/test_init_silica_water  -> validation frames
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any, Tuple

import numpy as np
from numpy.typing import NDArray


IndexArray = NDArray[np.int_]
DeepmdArray = NDArray[Any]


SYSTEM_NAME: str = "init_silica_water"
VALIDATION_SYSTEM_NAME: str = "test_init_silica_water"
BACKUP_SYSTEM_NAME: str = "init_silica_water_full_before_validation_split"

TRAJ_COUNT: int = 3
FRAMES_PER_TRAJ: int = 401
VALID_FRAMES_PER_TRAJ: int = 60
EXPECTED_FRAMES: int = TRAJ_COUNT * FRAMES_PER_TRAJ

ARRAY_NAMES: Tuple[str, ...] = ("box", "coord", "energy", "force")


def validation_indices() -> IndexArray:
    traj1 = np.arange(FRAMES_PER_TRAJ - VALID_FRAMES_PER_TRAJ, FRAMES_PER_TRAJ)
    traj2 = np.arange(
        2 * FRAMES_PER_TRAJ - VALID_FRAMES_PER_TRAJ,
        2 * FRAMES_PER_TRAJ,
    )
    traj3 = np.arange(2 * FRAMES_PER_TRAJ, 2 * FRAMES_PER_TRAJ + VALID_FRAMES_PER_TRAJ)
    return np.concatenate([traj1, traj2, traj3])


def training_indices(valid_idx: IndexArray) -> IndexArray:
    mask = np.ones(EXPECTED_FRAMES, dtype=bool)
    mask[valid_idx] = False
    return np.arange(EXPECTED_FRAMES)[mask]


def validate_source(source: Path) -> None:
    if not source.is_dir():
        raise SystemExit(f"Missing source dataset: {source}")
    if not (source / "type.raw").is_file():
        raise SystemExit(f"Missing type.raw in {source}")

    for name in ARRAY_NAMES:
        path = source / "set.000" / f"{name}.npy"
        if not path.is_file():
            raise SystemExit(f"Missing array file: {path}")
        frames = np.load(path, mmap_mode="r").shape[0]
        if frames != EXPECTED_FRAMES:
            raise SystemExit(
                f"{path} has {frames} frames, expected {EXPECTED_FRAMES}. "
                "Use the unsplit full dataset as input."
            )


def copy_static_files(source: Path, dest: Path) -> None:
    shutil.copy2(source / "type.raw", dest / "type.raw")


def write_split(source: Path, dest: Path, indices: IndexArray) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    (dest / "set.000").mkdir(parents=True)
    copy_static_files(source, dest)

    for name in ARRAY_NAMES:
        arr: DeepmdArray = np.load(source / "set.000" / f"{name}.npy")
        split_arr: DeepmdArray = arr[indices]
        np.save(dest / "set.000" / f"{name}.npy", split_arr)

        raw_path = source / f"{name}.raw"
        if raw_path.is_file():
            np.savetxt(dest / f"{name}.raw", split_arr.reshape(split_arr.shape[0], -1))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create train/validation DeepMD systems from data/init_silica_water."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="ArcaNN training root containing the data directory.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing split outputs if they are present.",
    )
    args: argparse.Namespace = parser.parse_args()

    data_dir = args.root / "data"
    source = data_dir / SYSTEM_NAME
    backup = data_dir / BACKUP_SYSTEM_NAME
    train_dest = data_dir / SYSTEM_NAME
    valid_dest = data_dir / VALIDATION_SYSTEM_NAME

    if backup.exists():
        source_for_split = backup
    else:
        source_for_split = source

    validate_source(source_for_split)

    if valid_dest.exists() and not args.force:
        raise SystemExit(f"{valid_dest} already exists. Re-run with --force to overwrite.")
    if backup.exists() and source == source_for_split and not args.force:
        raise SystemExit(f"{backup} already exists. Re-run with --force to overwrite.")

    if not backup.exists():
        shutil.copytree(source, backup)
        source_for_split = backup

    valid_idx = validation_indices()
    train_idx = training_indices(valid_idx)

    write_split(source_for_split, train_dest, train_idx)
    write_split(source_for_split, valid_dest, valid_idx)

    print(f"Source frames:     {EXPECTED_FRAMES}")
    print(f"Training frames:   {len(train_idx)} -> {train_dest}")
    print(f"Validation frames: {len(valid_idx)} -> {valid_dest}")
    print(f"Backup preserved:  {backup}")
    print()
    print("Validation frame ranges, one-based:")
    print("  trajectory 1: 342-401")
    print("  trajectory 2: 743-802")
    print("  trajectory 3: 803-862")


if __name__ == "__main__":
    main()
