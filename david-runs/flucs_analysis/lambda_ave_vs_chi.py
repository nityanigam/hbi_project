"""
Energy-weighted mean length scale, integrated directly in lambda = 2*pi/k
space, for each of the 9 (field, direction) cases across the chi scan.

This is NOT lambda_ave = 2*pi / k_ave (see kave_vs_chi.py) -- that would be
the wavelength of the energy-weighted mean wavenumber, a different quantity
from the energy-weighted mean wavelength. Here we change the integration
variable from k to lambda and integrate there instead.

To do that correctly, E(k) must be converted into an energy density in
lambda-space via the Jacobian of the transform: since energy in a bin is a
physical, coordinate-independent quantity, E(k) dk = F(lambda) d(lambda)
must hold, i.e. F(lambda) = E(k(lambda)) * |dk/dlambda|. With
lambda = 2*pi/k (so k = 2*pi/lambda), |dk/dlambda| = 2*pi/lambda**2 =
k**2 / (2*pi). Skipping this factor (just relabeling E(k) as a function of
lambda and integrating over lambda with no Jacobian) would silently change
how much weight each k-bin carries and would not even preserve total
energy between the k-space and lambda-space integrals.

lambda_ave = int(lambda * F(lambda) dlambda) / int(F(lambda) dlambda)

is then computed via the trapezoidal rule using lambda (not k) as the
integration axis, matching the native (non-uniform) k grid remapped to its
corresponding lambda grid and re-sorted ascending.

As in kave_vs_chi.py, each run's spectrum is first averaged over its last
N_LAST timesteps, and lambda_ave is computed from that time-averaged
spectrum; the k=0 (DC / mean-field) mode is excluded (lambda -> infinity
there, and it represents the background state, not a fluctuation).
"""
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
PLOT_OUTPATH = os.path.join(SCRIPT_DIR, "lambda_ave_vs_chi.png")
TABLE_OUTPATH = os.path.join(SCRIPT_DIR, "lambda_ave_vs_chi_table.txt")

directions = ["x", "y", "z"]
fields = ["kinetic_energy", "magnetic_fluctuation_energy", "theta_variance"]

# Reciprocals of the k_max/k_ave broken power laws (see kmax_vs_chi.py /
# kave_vs_chi.py): if k ~ A * chi**b, then a length scale ~ 1/k scales as
# chi**(-b), i.e. the exponent sign is flipped.
# (exponent for the 3 smallest chi, exponent for the 3 largest chi)
DIRECTION_EXPONENTS = {
    "x": (1 / 16, 1 / 5),
    "y": (3 / 8, 2 / 5),
    "z": (1 / 16, 1 / 5),
}
N_LOW = 3  # number of smallest-chi points fit with the first exponent

CHI_PATTERN = re.compile(r"chi1over(\d+)")


def parse_chi(folder_name):
    match = CHI_PATTERN.search(folder_name)
    if match is None:
        raise ValueError(f"could not parse chi from folder name {folder_name!r}")
    return 1.0 / float(match.group(1))


def fit_power_law_fixed_exponent(chi, y, exponent):
    """Least-squares prefactor A for y = A * chi**exponent (fixed exponent)."""
    log_amplitude = np.mean(np.log(y) - exponent * np.log(chi))
    return np.exp(log_amplitude)


def compute_lambda_ave_table(run_dir, n_last=N_LAST):
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

            kpos = k > 0
            kk, ee = k[kpos], avg_spectrum[kpos]

            lam = 2.0 * np.pi / kk
            jacobian = kk ** 2 / (2.0 * np.pi)  # |dk/dlambda|
            density_lambda = ee * jacobian  # F(lambda) = E(k) * |dk/dlambda|

            order = np.argsort(lam)
            lam_sorted = lam[order]
            density_sorted = density_lambda[order]

            numerator = np.trapz(lam_sorted * density_sorted, lam_sorted)
            denominator = np.trapz(density_sorted, lam_sorted)
            table[f"{field}_{direction}"] = numerator / denominator
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
lambda_by_key = {f"{field}_{direction}": [] for field in fields for direction in directions}

for name in run_dirs:
    run_dir = os.path.join(SCAN_DIR, name)
    chi = parse_chi(name)
    print(f"processing {name} (chi={chi:.6g}) ...")
    table = compute_lambda_ave_table(run_dir)
    chi_values.append(chi)
    for key, value in table.items():
        lambda_by_key[key].append(value)

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
        values = np.array(lambda_by_key[key])
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
            ax.set_ylabel(f"$\\lambda_{{ave}}$ ({field})")
        ax.legend(fontsize=6)
        if row == len(fields) - 1:
            ax.set_xlabel(r"$\chi$")

fig.suptitle(
    f"Energy-weighted mean length scale vs $\\chi$ "
    f"(spectra averaged over last {N_LAST} timesteps)"
)
fig.tight_layout()
fig.savefig(PLOT_OUTPATH, dpi=150, bbox_inches="tight")
print(f"saved {PLOT_OUTPATH}")

with open(TABLE_OUTPATH, "w") as fh:
    fh.write("chi\t" + "\t".join(lambda_by_key.keys()) + "\n")
    for i in order:
        row_vals = "\t".join(f"{lambda_by_key[key][i]:.6g}" for key in lambda_by_key)
        fh.write(f"{chi_values[i]:.6g}\t{row_vals}\n")
print(f"saved {TABLE_OUTPATH}")
