import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from load_flucs_spectra import load_flucs_spectra

run_dir = "/cephfs/store/astro-hl278/dnh26/hbi_flucs/runs/2_production_chi_scan_512/1_chi1over200_full_y_nu1e6"
data = load_flucs_spectra(run_dir)

time = data["time"]
mask = time >= 2

directions = ["x", "y", "z"]
fields = ["kinetic_energy", "magnetic_fluctuation_energy", "theta_variance"]

fig, axes = plt.subplots(3, 3, figsize=(12, 10), sharex="col")

for row, field in enumerate(fields):
    for col, direction in enumerate(directions):
        ax = axes[row, col]
        k = data[f"k{direction}"]
        spectrum = data[f"{field}_{direction}"][-1]
        ax.loglog(k, spectrum, linewidth=0.75)
        if row == 0:
            ax.set_title(f"{direction}-direction")
        if col == 0:
            ax.set_ylabel(field)
        if row == len(fields) - 1:
            ax.set_xlabel(f"k{direction}")

fig.suptitle(f"Directional spectra at t = {time[-1]:.3g}")
fig.tight_layout()
fig.savefig("spectra_grid_final.png", dpi=150, bbox_inches="tight")
print("saved spectra_grid_final.png")