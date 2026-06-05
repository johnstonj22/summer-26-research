from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
LcurveData = Dict[str, FloatArray]
Metric = Tuple[str, str]

plt.style.use("default")

training_dir: Path = Path(__file__).resolve().parent


def read_lcurve(path: Path) -> LcurveData:
    columns: Optional[List[str]] = None

    with path.open() as handle:
        for line in handle:
            fields = line.strip().lstrip("#").split()
            if fields and fields[0] == "step":
                columns = fields
                break

    data = np.loadtxt(path, comments="#")
    data = np.atleast_2d(data)

    if columns is None:
        columns = ["step", "rmse_trn", "rmse_e_trn", "rmse_f_trn", "lr"]

    return {
        name: data[:, index]
        for index, name in enumerate(columns)
        if index < data.shape[1]
    }


def has_values(values: Optional[FloatArray]) -> bool:
    return values is not None and np.any(np.isfinite(values))


metrics: Sequence[Metric] = [
    ("rmse", "Total RMSE"),
    ("rmse_e", "Energy RMSE"),
    ("rmse_f", "Force RMSE"),
]

train_color: str = "tab:blue"
validation_color: str = "tab:orange"
learning_rate_color: str = "tab:green"


def plot_model(i: int, lcurve: LcurveData) -> Path:
    fig, axs = plt.subplots(4, 1, figsize=(8, 13), sharex=True)
    step = lcurve["step"]

    for ax_index, (metric, ylabel) in enumerate(metrics):
        train_values = lcurve.get(f"{metric}_trn")
        val_values = lcurve.get(f"{metric}_val")

        if has_values(train_values):
            axs[ax_index].plot(
                step,
                train_values,
                color=train_color,
                label="Training",
            )

        if has_values(val_values):
            axs[ax_index].plot(
                step,
                val_values,
                color=validation_color,
                label="Validation",
            )

        axs[ax_index].set_ylabel(ylabel)
        axs[ax_index].legend()

    lr = lcurve.get("lr")
    if has_values(lr):
        axs[3].plot(step, lr, color=learning_rate_color, label="Learning rate")
        axs[3].legend()

    axs[3].set_ylabel("Learning Rate")
    axs[3].set_xlabel("Training Steps")

    for ax in axs:
        ax.grid(True)
        ax.set_yscale("log")

    fig.suptitle(f"NNP {i} Training and Validation Curves")
    fig.tight_layout()

    output_path = training_dir / f"loss_plots_nnp_{i}.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


for i in range(1, 4):
    lcurve_path = training_dir / str(i) / "lcurve.out"
    if not lcurve_path.exists():
        print(f"Skipping NNP {i}: {lcurve_path} does not exist")
        continue

    output_path = plot_model(i, read_lcurve(lcurve_path))
    print(f"Saved {output_path}")
