import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from load_flucs_spectra import load_flucs_spectra

SCAN_DIR = "/cephfs/store/astro-hl278/dnh26/hbi_flucs/runs/2_production_chi_scan_512"
N_LAST = 50  # trailing timesteps to average per run

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_OUTPATH = os.path.join(SCRIPT_DIR, "kmax_vs_chi.png")
TABLE_OUTPATH = os.path.join(SCRIPT_DIR, "kmax_vs_chi_table.txt")

directions = ["x", "y", "z"]
fields = ["kinetic_energy", "magnetic_fluctuation_energy", "theta_variance"]

# Broken power law: (exponent for the 3 smallest chi, exponent for the 3 largest chi)
DIRECTION_EXPONENTS = {
    "x": (-1 / 16, -1 / 5),
    "y": (-3 / 8, -2 / 5),
    "z": (-1 / 16, -1 / 5),
}
N_LOW = 3  # number of smallest-chi points fit with the first exponent

CHI_PATTERN = re.compile(r"chi1over(\d+)")


def fit_power_law_fixed_exponent(chi, y, exponent):
    """Least-squares prefactor A for y = A * chi**exponent (fixed exponent)."""
    log_amplitude = np.mean(np.log(y) - exponent * np.log(chi))
    return np.exp(log_amplitude)


def parse_chi(folder_name):
    match = CHI_PATTERN.search(folder_name)
    if match is None:
        raise ValueError(f"could not parse chi from folder name {folder_name!r}")
    return 1.0 / float(match.group(1))


def compute_kmax_table(run_dir, n_last=N_LAST):
    data = load_flucs_spectra(run_dir)
    time = data["time"]
    if len(time) < n_last:
        raise ValueError(f"{run_dir}: only {len(time)} timesteps available, requested n_last={n_last}")

    table = {}
    for field in fields:
        for direction in directions:
            k = data[f"k{direction}"]
            window = data[f"{field}_{direction}"][-n_last:]
            avg_spectrum = window.mean(axis=0)
            compensated = avg_spectrum * k
            kpos = k > 0
            kk, cc = k[kpos], compensated[kpos]
            table[f"{field}_{direction}"] = kk[np.argmax(cc)]
    return table


run_dirs = sorted(
    (
        name
        for name in os.listdir(SCAN_DIR)
        if os.path.isdir(os.path.join(SCAN_DIR, name)) and CHI_PATTERN.search(name)
    ),
    key=parse_chi,
)

chi_values = []
kmax_by_key = {f"{field}_{direction}": [] for field in fields for direction in directions}

for name in run_dirs:
    run_dir = os.path.join(SCAN_DIR, name)
    chi = parse_chi(name)
    print(f"processing {name} (chi={chi:.6g}) ...")
    table = compute_kmax_table(run_dir)
    chi_values.append(chi)
    for key, value in table.items():
        kmax_by_key[key].append(value)

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
        values = np.array(kmax_by_key[key])
        chi_sorted = chi_values[order]
        values_sorted = values[order]
        ax.plot(chi_sorted, values_sorted, marker="o", linewidth=1, label="data")

        low_exponent, high_exponent = DIRECTION_EXPONENTS[direction]
        segments = [
            (chi_sorted[:N_LOW], values_sorted[:N_LOW], low_exponent),
            (chi_sorted[N_LOW:], values_sorted[N_LOW:], high_exponent),
        ]
        for chi_segment, values_segment, exponent in segments:
            amplitude = fit_power_law_fixed_exponent(chi_segment, values_segment, exponent)
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
        if row == 0:
            ax.set_title(f"{direction}-direction")
        if col == 0:
            ax.set_ylabel(f"$k_{{peak}}$ ({field})")
        if row == len(fields) - 1:
            ax.set_xlabel(r"$\chi$")
        ax.legend(fontsize=6)

fig.suptitle(f"Peak wavenumber vs $\\chi$ (spectra averaged over last {N_LAST} timesteps)")
fig.tight_layout()
fig.savefig(PLOT_OUTPATH, dpi=150, bbox_inches="tight")
print(f"saved {PLOT_OUTPATH}")

with open(TABLE_OUTPATH, "w") as fh:
    fh.write("chi\t" + "\t".join(kmax_by_key.keys()) + "\n")
    for i in order:
        row_vals = "\t".join(f"{kmax_by_key[key][i]:.6g}" for key in kmax_by_key)
        fh.write(f"{chi_values[i]:.6g}\t{row_vals}\n")
print(f"saved {TABLE_OUTPATH}")
