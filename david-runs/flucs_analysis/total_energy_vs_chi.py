"""
Total spectral energy, averaged over the last N_LAST timesteps, vs chi.

Reuses total_energy_vs_time.py's per-timestep total_energy_time_series()
(no spectral smoothing -- raw energy at each saved snapshot), then simply
averages the trailing N_LAST values of that already-computed time series
for each chi, rather than re-averaging the raw spectra first.
"""
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from load_flucs_spectra import load_flucs_spectra

SCAN_DIR = "/cephfs/store/astro-hl278/dnh26/hbi_flucs/runs/2_production_chi_scan_512"
TIME_MIN = 2.0  # drop startup transient; set to None to keep all times
N_LAST = 50      # trailing timesteps of the time series to average per chi

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_OUTPATH = os.path.join(SCRIPT_DIR, "total_energy_vs_chi.png")
TABLE_OUTPATH = os.path.join(SCRIPT_DIR, "total_energy_vs_chi_table.txt")

directions = ["x", "y", "z"]
fields = ["kinetic_energy", "magnetic_fluctuation_energy", "theta_variance"]

N_LOW = 3  # number of smallest-chi points fit as one power law; the rest fit as a second

CHI_PATTERN = re.compile(r"chi1over(\d+)")


def parse_chi(folder_name):
    match = CHI_PATTERN.search(folder_name)
    if match is None:
        raise ValueError(f"could not parse chi from folder name {folder_name!r}")
    return 1.0 / float(match.group(1))


def fit_power_law(chi, y):
    """Least-squares fit of y = A * chi**b (both A and b free), via linear
    regression in log-log space."""
    exponent, log_amplitude = np.polyfit(np.log(chi), np.log(y), 1)
    return np.exp(log_amplitude), exponent


def total_energy_time_series(run_dir):
    """Returns (time, {field_direction: int(E(k,t) dk)}) for one run."""
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

chi_values = []
avg_energy_by_key = {f"{field}_{direction}": [] for field in fields for direction in directions}

for name in run_dirs:
    run_dir = os.path.join(SCAN_DIR, name)
    chi = parse_chi(name)
    print(f"processing {name} (chi={chi:.6g}) ...")
    time, energy_dict = total_energy_time_series(run_dir)
    if len(time) < N_LAST:
        raise ValueError(f"{run_dir}: only {len(time)} timesteps available, requested N_LAST={N_LAST}")
    chi_values.append(chi)
    for key, series in energy_dict.items():
        avg_energy_by_key[key].append(np.mean(series[-N_LAST:]))

chi_values = np.array(chi_values)
order = np.argsort(chi_values)

if len(chi_values) < 2 * N_LOW:
    raise ValueError(
        f"found {len(chi_values)} chi values, need at least {2 * N_LOW} "
        f"for a broken power law with {N_LOW} points per segment"
    )

fig, axes = plt.subplots(3, 3, figsize=(12, 10), sharex=True)

for row, field in enumerate(fields):
    for col, direction in enumerate(directions):
        ax = axes[row, col]
        key = f"{field}_{direction}"
        values = np.array(avg_energy_by_key[key])
        chi_sorted = chi_values[order]
        values_sorted = values[order]
        ax.scatter(chi_sorted, values_sorted, marker="o", linewidth=1, label="data")

        segments = [chi_sorted[:N_LOW], chi_sorted[N_LOW:]]
        value_segments = [values_sorted[:N_LOW], values_sorted[N_LOW:]]
        for chi_segment, values_segment in zip(segments, value_segments):
            amplitude, exponent = fit_power_law(chi_segment, values_segment)
            chi_smooth = np.geomspace(chi_segment.min(), chi_segment.max(), 50)
            ax.plot(
                chi_smooth,
                amplitude * chi_smooth ** exponent,
                "--",
                linewidth=1,
                label=f"$\\chi^{{{exponent:.3g}}}$ fit",
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(fontsize=6)
        if row == 0:
            ax.set_title(f"{direction}-direction")
        if col == 0:
            ax.set_ylabel(f"total energy ({field})")
        if row == len(fields) - 1:
            ax.set_xlabel(r"$\chi$")

fig.suptitle(f"Total spectral energy vs $\\chi$ (averaged over last {N_LAST} timesteps)")
fig.tight_layout()
fig.savefig(PLOT_OUTPATH, dpi=150, bbox_inches="tight")
print(f"saved {PLOT_OUTPATH}")

with open(TABLE_OUTPATH, "w") as fh:
    fh.write("chi\t" + "\t".join(avg_energy_by_key.keys()) + "\n")
    for i in order:
        row_vals = "\t".join(f"{avg_energy_by_key[key][i]:.6g}" for key in avg_energy_by_key)
        fh.write(f"{chi_values[i]:.6g}\t{row_vals}\n")
print(f"saved {TABLE_OUTPATH}")
