import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from load_flucs_spectra import load_flucs_spectra

SCAN_DIR = "/cephfs/store/astro-hl278/dnh26/hbi_flucs/runs/2_production_chi_scan_512"
TIME_MIN = 2.0  # drop startup transient; set to None to keep all times

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_OUTPATH = os.path.join(SCRIPT_DIR, "total_energy_vs_time.png")

directions = ["x", "y", "z"]
fields = ["kinetic_energy", "magnetic_fluctuation_energy", "theta_variance"]

CHI_PATTERN = re.compile(r"chi1over(\d+)")


def parse_chi(folder_name):
    match = CHI_PATTERN.search(folder_name)
    if match is None:
        raise ValueError(f"could not parse chi from folder name {folder_name!r}")
    return 1.0 / float(match.group(1))


def total_energy_time_series(run_dir):
    """Returns (time, {field_direction: int(E(k,t) dk)}) for one run.

    Total spectral energy at each timestep, integrated over k (k=0 excluded,
    same rationale as elsewhere). No time-smoothing.
    """
    data = load_flucs_spectra(run_dir)
    time = data["time"]
    mask = time >= TIME_MIN if TIME_MIN is not None else np.ones_like(time, dtype=bool)
    time = time[mask]

    result = {}
    for field in fields:
        for direction in directions:
            k = data[f"k{direction}"]
            kpos = k > 0
            kk = k[kpos]
            ee = data[f"{field}_{direction}"][mask][:, kpos]
            result[f"{field}_{direction}"] = np.trapz(ee, kk, axis=1)
    return time, result


run_dirs = sorted(
    (
        name
        for name in os.listdir(SCAN_DIR)
        if os.path.isdir(os.path.join(SCAN_DIR, name)) and CHI_PATTERN.search(name)
    ),
    key=parse_chi,
)

chis = [parse_chi(name) for name in run_dirs]
colors = plt.cm.viridis(np.linspace(0, 1, len(run_dirs)))

per_run_data = []  # list of (chi, color, time, energy_dict)
for name, chi, color in zip(run_dirs, chis, colors):
    run_dir = os.path.join(SCAN_DIR, name)
    print(f"processing {name} (chi={chi:.6g}) ...")
    time, energy_dict = total_energy_time_series(run_dir)
    per_run_data.append((chi, color, time, energy_dict))

fig, axes = plt.subplots(3, 3, figsize=(14, 11), sharex=False, sharey=True)

for row, field in enumerate(fields):
    for col, direction in enumerate(directions):
        ax = axes[row, col]
        key = f"{field}_{direction}"
        for chi, color, time, energy_dict in per_run_data:
            ax.plot(time, energy_dict[key], color=color, linewidth=1, label=f"$\\chi=1/{1/chi:.0f}$")

        ax.set_yscale("log")
        if row == 0:
            ax.set_title(f"{direction}-direction")
        if col == 0:
            ax.set_ylabel(f"total energy ({field})")
        if row == len(fields) - 1:
            ax.set_xlabel("time")

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=len(run_dirs), fontsize=8, bbox_to_anchor=(0.5, -0.02))

fig.suptitle("Time evolution of total spectral energy $\\int E(k)\\,dk$")
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(PLOT_OUTPATH, dpi=150, bbox_inches="tight")
print(f"saved {PLOT_OUTPATH}")
