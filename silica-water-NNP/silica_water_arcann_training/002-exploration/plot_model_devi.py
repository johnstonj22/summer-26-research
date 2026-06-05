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
import math
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


def numeric_name_key(item):
    name = item.name if isinstance(item, Path) else str(item)
    if name.isdigit():
        return (0, int(name))
    return (1, name)


def sorted_numeric_dirs(path: Path) -> list[Path]:
    dirs = [item for item in path.iterdir() if item.is_dir()]
    return sorted(dirs, key=numeric_name_key)


def collect_trajectory_data(nnp_dirs: list[Path]) -> dict[str, list[tuple[str, Path]]]:
    trajectory_data: dict[str, list[tuple[str, Path]]] = {}

    for nnp_dir in nnp_dirs:
        for traj_dir in sorted_numeric_dirs(nnp_dir):
            devi_files = sorted(traj_dir.glob("model_devi_*.out"))
            if not devi_files:
                continue

            trajectory_data.setdefault(traj_dir.name, []).append((nnp_dir.name, devi_files[0]))

    for trajectory_name in trajectory_data:
        trajectory_data[trajectory_name].sort(key=lambda item: numeric_name_key(item[0]))

    return trajectory_data


def plot_trajectories(
    trajectory_data: dict[str, list[tuple[str, Path]]],
    sigma_values: dict[str, float],
    output_path: Path,
    columns: int,
    legend_location: str,
) -> None:
    trajectory_names = sorted(trajectory_data, key=numeric_name_key)
    n_trajectories = len(trajectory_names)
    ncols = min(max(columns, 1), n_trajectories)
    nrows = math.ceil(n_trajectories / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6.0 * ncols, 3.8 * nrows),
        sharey=True,
        squeeze=False,
    )
    flat_axes = axes.ravel()

    threshold_styles = {
        "sigma_low": ("tab:green", "sigma_low"),
        "sigma_high": ("tab:orange", "sigma_high"),
        "sigma_high_limit": ("tab:red", "sigma_high_limit"),
    }

    nnp_names = sorted(
        {nnp_name for runs in trajectory_data.values() for nnp_name, _ in runs},
        key=numeric_name_key,
    )
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    nnp_colors = {name: colors[index % len(colors)] for index, name in enumerate(nnp_names)}
    nnp_handles = {}
    threshold_handles = {}

    for ax, trajectory_name in zip(flat_axes, trajectory_names):
        for nnp_name, devi_file in trajectory_data[trajectory_name]:
            step, max_dev_f = load_model_devi(devi_file)
            label = f"NNP {nnp_name}"
            (line,) = ax.plot(
                step,
                max_dev_f,
                linewidth=1.2,
                color=nnp_colors[nnp_name],
                label=label,
            )
            nnp_handles.setdefault(label, line)

        for key, (color, label) in threshold_styles.items():
            threshold_label = f"{label} = {sigma_values[key]:g}"
            threshold_line = ax.axhline(
                sigma_values[key],
                color=color,
                linestyle=":",
                linewidth=1.6,
                label=threshold_label,
            )
            threshold_handles.setdefault(threshold_label, threshold_line)

        ax.set_title(f"Trajectory {trajectory_name}")
        ax.set_xlabel("Step")
        ax.set_ylabel("max_dev_f")
        ax.set_ylim(bottom=0, top=2.0 * sigma_values["sigma_high_limit"])
        ax.grid(True, alpha=0.35)
        if legend_location == "subplot":
            ax.legend(loc="upper right", title="Curves", fontsize=7)

    for ax in flat_axes[n_trajectories:]:
        ax.set_visible(False)

    fig.suptitle("Model Deviation by Trajectory")
    if legend_location == "figure":
        fig.legend(
            nnp_handles.values(),
            nnp_handles.keys(),
            loc="upper right",
            bbox_to_anchor=(0.985, 0.93),
            title="NNP",
            fontsize=8,
        )
        fig.legend(
            threshold_handles.values(),
            threshold_handles.keys(),
            loc="upper right",
            bbox_to_anchor=(0.985, 0.72),
            title="Thresholds",
            fontsize=8,
        )
        fig.tight_layout(rect=(0.03, 0.03, 0.84, 0.96))
    else:
        fig.tight_layout(rect=(0.03, 0.03, 0.98, 0.96))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot max_dev_f vs step by trajectory in an ArcaNN exploration directory."
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
    parser.add_argument(
        "--columns",
        type=int,
        default=2,
        help="Number of subplot columns in the output figure. Default: 2.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output PNG path. Default: model_devi_by_trajectory.png in the exploration directory.",
    )
    parser.add_argument(
        "--legend",
        choices=("figure", "subplot"),
        default="subplot",
        help="Legend placement. Default: subplot.",
    )
    args = parser.parse_args()

    exploration_dir = Path(args.exploration_dir).resolve()
    system_dir = exploration_dir / args.system
    if args.output is None:
        output_path = exploration_dir / "model_devi_by_trajectory.png"
    else:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = exploration_dir / output_path

    if not (exploration_dir / "used_input.json").is_file():
        raise FileNotFoundError(f"Could not find {exploration_dir / 'used_input.json'}")
    if not system_dir.is_dir():
        raise FileNotFoundError(f"Could not find system directory {system_dir}")

    sigma_values = load_sigma_values(exploration_dir)
    nnp_dirs = [path for path in sorted_numeric_dirs(system_dir) if any(path.glob("*/model_devi_*.out"))]

    if not nnp_dirs:
        raise FileNotFoundError(f"No model_devi_*.out files found under {system_dir}")

    trajectory_data = collect_trajectory_data(nnp_dirs)
    if not trajectory_data:
        raise FileNotFoundError(f"No trajectory model_devi_*.out files found under {system_dir}")

    plot_trajectories(trajectory_data, sigma_values, output_path, args.columns, args.legend)


if __name__ == "__main__":
    main()
