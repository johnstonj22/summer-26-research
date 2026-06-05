#!/usr/bin/env python3
"""Plot DeepMD model-deviation force curves for an ArcaNN exploration step.

Run from an exploration directory, for example:

    python ../plot_model_devi.py

Or pass the exploration directory explicitly:

    python plot_model_devi.py 001-exploration
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def as_scalar(value):
    """Return the first value when ArcaNN stores scalar settings as lists."""
    if value is None:
        return None
    if isinstance(value, list):
        if not value:
            raise ValueError("Expected a non-empty list for a sigma value.")
        return value[0]
    return value


def load_sigma_values(exploration_dir: Path) -> dict[str, float]:
    input_path = exploration_dir / "used_input.json"
    with input_path.open() as handle:
        settings = json.load(handle)

    missing = []
    sigma_values = {}
    for key in ("sigma_low", "sigma_high", "sigma_high_limit"):
        value = as_scalar(settings.get(key))
        if value is None:
            missing.append(key)
        else:
            sigma_values[key] = float(value)

    if missing:
        missing_values = ", ".join(missing)
        raise ValueError(f"{missing_values} are not set in {input_path}")

    return sigma_values


def load_model_devi(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, comments="#")
    data = np.atleast_2d(data)
    step = data[:, 0]
    max_dev_f = data[:, 4]
    return step, max_dev_f


def sorted_numeric_dirs(path: Path) -> list[Path]:
    dirs = [item for item in path.iterdir() if item.is_dir()]
    return sorted(dirs, key=lambda item: int(item.name) if item.name.isdigit() else item.name)


def plot_nnp(nnp_dir: Path, sigma_values: dict[str, float], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))

    traj_dirs = sorted_numeric_dirs(nnp_dir)
    for traj_dir in traj_dirs:
        devi_files = sorted(traj_dir.glob("model_devi_*.out"))
        if not devi_files:
            continue

        step, max_dev_f = load_model_devi(devi_files[0])
        ax.plot(step, max_dev_f, linewidth=1.2, label=traj_dir.name)

    threshold_styles = {
        "sigma_low": ("tab:green", "sigma_low"),
        "sigma_high": ("tab:orange", "sigma_high"),
        "sigma_high_limit": ("tab:red", "sigma_high_limit"),
    }
    for key, (color, label) in threshold_styles.items():
        ax.axhline(
            sigma_values[key],
            color=color,
            linestyle=":",
            linewidth=2.0,
            label=f"{label} = {sigma_values[key]:g}",
        )

    ax.set_title(f"NNP {nnp_dir.name} Model Deviation")
    ax.set_xlabel("Step")
    ax.set_ylabel("max_dev_f")
    ax.set_ylim(bottom=0, top=2.0 * sigma_values["sigma_high_limit"])
    ax.grid(True, alpha=0.35)
    ax.legend(title="Trajectory", fontsize=8)

    fig.tight_layout()
    output_path = output_dir / f"model_devi_nnp{nnp_dir.name}.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Wrote {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot max_dev_f vs step for each NNP in an ArcaNN exploration directory."
    )
    parser.add_argument(
        "exploration_dir",
        nargs="?",
        default=".",
        help="Exploration directory containing used_input.json and the silica_water folder.",
    )
    parser.add_argument(
        "--system",
        default="silica_water",
        help="System folder inside the exploration directory. Default: silica_water.",
    )
    args = parser.parse_args()

    exploration_dir = Path(args.exploration_dir).resolve()
    system_dir = exploration_dir / args.system

    if not (exploration_dir / "used_input.json").is_file():
        raise FileNotFoundError(f"Could not find {exploration_dir / 'used_input.json'}")
    if not system_dir.is_dir():
        raise FileNotFoundError(f"Could not find system directory {system_dir}")

    sigma_values = load_sigma_values(exploration_dir)
    nnp_dirs = [path for path in sorted_numeric_dirs(system_dir) if any(path.glob("*/model_devi_*.out"))]

    if not nnp_dirs:
        raise FileNotFoundError(f"No model_devi_*.out files found under {system_dir}")

    for nnp_dir in nnp_dirs:
        plot_nnp(nnp_dir, sigma_values, exploration_dir)


if __name__ == "__main__":
    main()
