#!/usr/bin/env python3
"""
Check candidate structures for suspicious distances before CP2K labeling.

The default workflow learns conservative distance ranges from a known-good
DeepMD dataset, then scans ArcaNN candidates written as extended XYZ or
LAMMPS data files. Run from the main ArcaNN training directory, for example:

    python check_candidate_distances.py \
        --reference-data old_data/init_silica_water \
        --candidate-root 002-exploration/silica_water \
        --report-prefix 002-exploration/candidate_distance_report

The candidate root may be a labeling/exploration directory tree, a single
.xyz file, or a single .lmp file. Reports are written as:
  - <report-prefix>.csv
  - <report-prefix>.json
  - <report-prefix>_thresholds.json

For this silica-water system the main checks are:
  - each H atom should have a nearest O within the learned O-H range
  - each Si atom should have a nearest O within the learned Si-O range
  - no element pair should be closer than the learned clean-data contact limit
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, TypedDict

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int_]
SymbolArray = NDArray[np.object_]
Finding = Dict[str, object]
ReportRow = Dict[str, object]
ElementPair = Tuple[str, str]


class NearestLimits(TypedDict):
    low: float
    high: float
    reference_low_percentile: float
    reference_high_percentile: float
    samples: int


class ContactLimits(TypedDict):
    low: float
    reference_low_percentile: float
    samples: int


class Thresholds(TypedDict):
    nearest: Dict[str, NearestLimits]
    contact_min: Dict[str, ContactLimits]


DEFAULT_TYPE_MAP: Tuple[str, ...] = ("Si", "O", "H")
ARRAY_NAMES: Tuple[str, ...] = ("box", "coord")
NEAREST_CHECKS: Tuple[ElementPair, ...] = (("H", "O"), ("Si", "O"))
CONTACT_PAIRS: Tuple[ElementPair, ...] = (
    ("H", "H"),
    ("O", "H"),
    ("O", "O"),
    ("Si", "H"),
    ("Si", "O"),
    ("Si", "Si"),
)


@dataclass
class Frame:
    name: str
    symbols: SymbolArray
    coords: FloatArray
    cell: Optional[FloatArray]


def parse_type_map(raw: str) -> Tuple[str, ...]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise SystemExit("--type-map must contain at least one element")
    return tuple(values)


def load_type_symbols(type_raw: Path, type_map: Sequence[str]) -> SymbolArray:
    type_ids = np.loadtxt(type_raw, dtype=int)
    if type_ids.ndim == 0:
        type_ids = np.array([int(type_ids)])

    if type_ids.min() >= 1 and type_ids.max() <= len(type_map):
        type_ids = type_ids - 1

    if type_ids.min() < 0 or type_ids.max() >= len(type_map):
        raise SystemExit(
            f"{type_raw} contains type ids outside type map {list(type_map)}"
        )

    return np.array([type_map[idx] for idx in type_ids], dtype=object)


def load_deepmd_frames(system_path: Path, type_map: Sequence[str]) -> List[Frame]:
    set_path = system_path / "set.000"
    coord_path = set_path / "coord.npy"
    box_path = set_path / "box.npy"
    type_path = system_path / "type.raw"

    for path in (coord_path, box_path, type_path):
        if not path.is_file():
            raise SystemExit(f"Missing required DeepMD file: {path}")

    symbols = load_type_symbols(type_path, type_map)
    coords_raw = np.load(coord_path)
    boxes_raw = np.load(box_path)

    natoms = len(symbols)
    coords = coords_raw.reshape(coords_raw.shape[0], natoms, 3)
    boxes = boxes_raw.reshape(boxes_raw.shape[0], 3, 3)

    frames: List[Frame] = []
    for idx in range(coords.shape[0]):
        frames.append(
            Frame(
                name=f"{system_path.name}:frame_{idx:06d}",
                symbols=symbols,
                coords=coords[idx],
                cell=boxes[idx],
            )
        )
    return frames


def parse_lattice(comment: str) -> Optional[FloatArray]:
    match = re.search(r'Lattice="([^"]+)"', comment)
    if not match:
        return None
    values = [float(item) for item in match.group(1).split()]
    if len(values) != 9:
        raise ValueError(f"Expected 9 Lattice values, found {len(values)}")
    return np.array(values, dtype=float).reshape(3, 3)


def read_xyz_frames(path: Path) -> Iterator[Frame]:
    with path.open() as handle:
        frame_idx = 0
        while True:
            first = handle.readline()
            if not first:
                break
            if not first.strip():
                continue

            natoms = int(first.strip())
            comment = handle.readline().strip()
            cell = parse_lattice(comment)
            symbols: List[str] = []
            coords: List[List[float]] = []

            for _ in range(natoms):
                parts = handle.readline().split()
                if len(parts) < 4:
                    raise ValueError(f"Malformed XYZ atom line in {path}")
                symbols.append(parts[0])
                coords.append([float(parts[1]), float(parts[2]), float(parts[3])])

            yield Frame(
                name=f"{path}:frame_{frame_idx:06d}",
                symbols=np.array(symbols, dtype=object),
                coords=np.array(coords, dtype=float),
                cell=cell,
            )
            frame_idx += 1


def read_lammps_data_frame(path: Path, type_map: Sequence[str]) -> Frame:
    lines = path.read_text().splitlines()
    xlo = xhi = ylo = yhi = zlo = zhi = None

    for line in lines:
        parts = line.split()
        if len(parts) >= 4 and parts[-2:] == ["xlo", "xhi"]:
            xlo, xhi = map(float, parts[:2])
        elif len(parts) >= 4 and parts[-2:] == ["ylo", "yhi"]:
            ylo, yhi = map(float, parts[:2])
        elif len(parts) >= 4 and parts[-2:] == ["zlo", "zhi"]:
            zlo, zhi = map(float, parts[:2])

    if None in (xlo, xhi, ylo, yhi, zlo, zhi):
        raise ValueError(f"Could not parse orthorhombic box from {path}")

    start = None
    for idx, line in enumerate(lines):
        if line.strip().startswith("Atoms"):
            start = idx + 1
            break
    if start is None:
        raise ValueError(f"No Atoms section found in {path}")

    while start < len(lines) and not lines[start].strip():
        start += 1

    symbols: List[str] = []
    coords: List[List[float]] = []
    for line in lines[start:]:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped[0].isalpha():
            break

        parts = stripped.split()
        if len(parts) < 5:
            continue

        atom_type = int(parts[1])
        if atom_type < 1 or atom_type > len(type_map):
            raise ValueError(f"Atom type {atom_type} outside type map in {path}")

        symbols.append(type_map[atom_type - 1])
        coords.append([float(parts[2]), float(parts[3]), float(parts[4])])

    cell = np.array(
        [
            [xhi - xlo, 0.0, 0.0],
            [0.0, yhi - ylo, 0.0],
            [0.0, 0.0, zhi - zlo],
        ],
        dtype=float,
    )
    return Frame(
        name=str(path),
        symbols=np.array(symbols, dtype=object),
        coords=np.array(coords, dtype=float),
        cell=cell,
    )


def candidate_frames(candidate_root: Path, type_map: Sequence[str]) -> Iterator[Frame]:
    if candidate_root.is_file() and candidate_root.suffix == ".xyz":
        yield from read_xyz_frames(candidate_root)
        return
    if candidate_root.is_file() and candidate_root.suffix == ".lmp":
        yield read_lammps_data_frame(candidate_root, type_map)
        return

    xyz_files = sorted(candidate_root.rglob("labeling_*.xyz"))
    if not xyz_files:
        xyz_files = sorted(candidate_root.rglob("*.xyz"))
    for xyz_file in xyz_files:
        yield from read_xyz_frames(xyz_file)

    lmp_files = sorted(candidate_root.rglob("*.lmp"))
    for lmp_file in lmp_files:
        yield read_lammps_data_frame(lmp_file, type_map)

    if not xyz_files and not lmp_files:
        raise SystemExit(f"No XYZ or LAMMPS candidate files found under {candidate_root}")


def minimum_image(delta: FloatArray, cell: Optional[FloatArray]) -> FloatArray:
    if cell is None:
        return delta
    inv_cell = np.linalg.inv(cell)
    frac = delta @ inv_cell
    frac -= np.round(frac)
    return frac @ cell


def pair_distance_matrix(
    frame: Frame, element_a: str, element_b: str
) -> Tuple[FloatArray, IntArray, IntArray]:
    idx_a = np.flatnonzero(frame.symbols == element_a)
    idx_b = np.flatnonzero(frame.symbols == element_b)
    if len(idx_a) == 0 or len(idx_b) == 0:
        raise ValueError(f"Missing atoms for pair {element_a}-{element_b}")

    delta = frame.coords[idx_a, None, :] - frame.coords[idx_b][None, :, :]
    delta = minimum_image(delta, frame.cell)
    dist = np.linalg.norm(delta, axis=2)

    if element_a == element_b:
        np.fill_diagonal(dist, np.inf)
    return dist, idx_a, idx_b


def finite_min(dist: FloatArray) -> float:
    finite = dist[np.isfinite(dist)]
    if finite.size == 0:
        return math.inf
    return float(finite.min())


def percentile_range(values: Sequence[float], low_pct: float, high_pct: float) -> Tuple[float, float]:
    arr = np.array(values, dtype=float)
    return float(np.percentile(arr, low_pct)), float(np.percentile(arr, high_pct))


def learn_thresholds(
    frames: Sequence[Frame],
    low_pct: float,
    high_pct: float,
    nearest_margin: float,
    contact_margin: float,
) -> Thresholds:
    nearest_values: Dict[str, List[float]] = {
        f"{a}-{b}": [] for a, b in NEAREST_CHECKS
    }
    contact_values: Dict[str, List[float]] = {
        f"{a}-{b}": [] for a, b in CONTACT_PAIRS
    }

    for frame in frames:
        for element_a, element_b in NEAREST_CHECKS:
            key = f"{element_a}-{element_b}"
            dist, _, _ = pair_distance_matrix(frame, element_a, element_b)
            nearest_values[key].extend(np.min(dist, axis=1).tolist())

        for element_a, element_b in CONTACT_PAIRS:
            key = f"{element_a}-{element_b}"
            dist, _, _ = pair_distance_matrix(frame, element_a, element_b)
            contact_values[key].append(finite_min(dist))

    thresholds: Thresholds = {"nearest": {}, "contact_min": {}}

    for key, values in nearest_values.items():
        low, high = percentile_range(values, low_pct, high_pct)
        thresholds["nearest"][key] = {
            "low": max(0.0, low - nearest_margin),
            "high": high + nearest_margin,
            "reference_low_percentile": low,
            "reference_high_percentile": high,
            "samples": len(values),
        }

    for key, values in contact_values.items():
        low, _ = percentile_range(values, low_pct, high_pct)
        thresholds["contact_min"][key] = {
            "low": max(0.0, low - contact_margin),
            "reference_low_percentile": low,
            "samples": len(values),
        }

    return thresholds


def check_frame(frame: Frame, thresholds: Thresholds) -> List[Finding]:
    findings: List[Finding] = []

    for element_a, element_b in NEAREST_CHECKS:
        key = f"{element_a}-{element_b}"
        limits = thresholds["nearest"][key]
        dist, idx_a, _ = pair_distance_matrix(frame, element_a, element_b)
        nearest = np.min(dist, axis=1)
        too_low = np.flatnonzero(nearest < limits["low"])
        too_high = np.flatnonzero(nearest > limits["high"])

        for label, offenders in (("too_short", too_low), ("too_long", too_high)):
            if offenders.size == 0:
                continue
            worst_local = offenders[np.argmin(nearest[offenders]) if label == "too_short" else np.argmax(nearest[offenders])]
            findings.append(
                {
                    "check": f"nearest_{key}",
                    "status": label,
                    "count": int(offenders.size),
                    "limit_low": limits["low"],
                    "limit_high": limits["high"],
                    "worst_atom_index_1based": int(idx_a[worst_local] + 1),
                    "worst_distance": float(nearest[worst_local]),
                }
            )

    for element_a, element_b in CONTACT_PAIRS:
        key = f"{element_a}-{element_b}"
        limits = thresholds["contact_min"][key]
        dist, idx_a, idx_b = pair_distance_matrix(frame, element_a, element_b)
        min_dist = finite_min(dist)
        if min_dist < limits["low"]:
            pos = np.argwhere(dist == min_dist)[0]
            findings.append(
                {
                    "check": f"min_contact_{key}",
                    "status": "too_short",
                    "count": 1,
                    "limit_low": limits["low"],
                    "limit_high": "",
                    "worst_atom_index_1based": int(idx_a[pos[0]] + 1),
                    "partner_atom_index_1based": int(idx_b[pos[1]] + 1),
                    "worst_distance": min_dist,
                }
            )

    return findings


def write_reports(
    report_prefix: Path,
    rows: Sequence[ReportRow],
    thresholds: Thresholds,
) -> None:
    json_path = report_prefix.with_suffix(".json")
    csv_path = report_prefix.with_suffix(".csv")
    thresholds_path = report_prefix.with_name(report_prefix.name + "_thresholds.json")

    json_path.write_text(json.dumps(list(rows), indent=2))
    thresholds_path.write_text(json.dumps(thresholds, indent=2))

    fieldnames = [
        "candidate",
        "pass",
        "check",
        "status",
        "count",
        "worst_distance",
        "limit_low",
        "limit_high",
        "worst_atom_index_1based",
        "partner_atom_index_1based",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check labeling candidates for suspicious Si-O/O-H/contact distances."
    )
    parser.add_argument(
        "--reference-data",
        type=Path,
        default=None,
        help=(
            "Known-good DeepMD system used to learn distance thresholds. "
            "Defaults to old_data/init_silica_water if present, otherwise "
            "data/init_silica_water."
        ),
    )
    parser.add_argument(
        "--candidate-root",
        type=Path,
        required=True,
        help="Labeling directory, candidate XYZ file, or tree containing labeling_*.xyz.",
    )
    parser.add_argument(
        "--type-map",
        default="Si,O,H",
        help="Comma-separated type map for DeepMD type.raw files.",
    )
    parser.add_argument("--low-percentile", type=float, default=0.1)
    parser.add_argument("--high-percentile", type=float, default=99.9)
    parser.add_argument("--nearest-margin", type=float, default=0.15)
    parser.add_argument("--contact-margin", type=float, default=0.20)
    parser.add_argument(
        "--report-prefix",
        type=Path,
        default=Path("candidate_distance_report"),
        help="Output prefix for .csv, .json, and _thresholds.json reports.",
    )
    args: argparse.Namespace = parser.parse_args()

    if args.reference_data is None:
        old_initial = Path("old_data/init_silica_water")
        args.reference_data = (
            old_initial if old_initial.is_dir() else Path("data/init_silica_water")
        )

    type_map = parse_type_map(args.type_map)
    reference_frames = load_deepmd_frames(args.reference_data, type_map)
    thresholds = learn_thresholds(
        reference_frames,
        low_pct=args.low_percentile,
        high_pct=args.high_percentile,
        nearest_margin=args.nearest_margin,
        contact_margin=args.contact_margin,
    )

    rows: List[ReportRow] = []
    total = 0
    failed = 0
    for frame in candidate_frames(args.candidate_root, type_map):
        total += 1
        findings = check_frame(frame, thresholds)
        if findings:
            failed += 1
            for finding in findings:
                rows.append({"candidate": frame.name, "pass": False, **finding})
        else:
            rows.append({"candidate": frame.name, "pass": True})

    write_reports(args.report_prefix, rows, thresholds)

    print(f"Reference frames: {len(reference_frames)} from {args.reference_data}")
    print(f"Candidates checked: {total}")
    print(f"Candidates with findings: {failed}")
    print(f"CSV report: {args.report_prefix.with_suffix('.csv')}")
    print(f"JSON report: {args.report_prefix.with_suffix('.json')}")
    print(
        "Thresholds: "
        f"{args.report_prefix.with_name(args.report_prefix.name + '_thresholds.json')}"
    )


if __name__ == "__main__":
    main()
