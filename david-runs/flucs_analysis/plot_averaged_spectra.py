import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from load_flucs_spectra import load_flucs_spectra

RUN_DIR = "/cephfs/store/astro-hl278/dnh26/hbi_flucs/runs/2_production_chi_scan_512/1_chi1over200_full_y_nu1e6"
N_LAST = 50  # number of trailing timesteps to average over
OUTPATH = "spectra_grid_averaged.png"

directions = ["x", "y", "z"]
fields = ["kinetic_energy", "magnetic_fluctuation_energy", "theta_variance"]

data = load_flucs_spectra(RUN_DIR)
time = data["time"]
if len(time) < N_LAST:
    raise ValueError(f"only {len(time)} timesteps available, requested N_LAST={N_LAST}")

fig, axes = plt.subplots(3, 3, figsize=(12, 10), sharex="col")

print("running")

kmax_table = {}
for row, field in enumerate(fields):
    for col, direction in enumerate(directions):
        ax = axes[row, col]
        k = data[f"k{direction}"]
        window = data[f"{field}_{direction}"][-N_LAST:]
        avg_spectrum = window.mean(axis=0)
        compensated = avg_spectrum * k

        kpos = k > 0
        kk, cc = k[kpos], compensated[kpos]
        peak_idx = np.argmax(cc)
        k_at_peak = kk[peak_idx]
        kmax_table[f"{field}_{direction}"] = k_at_peak

        ax.loglog(kk, cc, linewidth=0.9)
        ax.axvline(k_at_peak, color="tab:red", linestyle="--", linewidth=0.8)
        ax.annotate(
            f"$k_{{peak}}$={k_at_peak:.3g}",
            xy=(k_at_peak, cc[peak_idx]),
            xytext=(0.05, 0.9),
            textcoords="axes fraction",
            fontsize=7,
            color="tab:red",
        )

        if row == 0:
            ax.set_title(f"{direction}-direction")
        if col == 0:
            ax.set_ylabel(f"$k\\,E$ ({field})")
        if row == len(fields) - 1:
            ax.set_xlabel(f"k{direction}")

fig.suptitle(
    f"Spectra averaged over last {N_LAST} timesteps "
    f"(t = {time[-N_LAST]:.3g} to {time[-1]:.3g})"
)
fig.tight_layout()
fig.savefig(OUTPATH, dpi=150, bbox_inches="tight")
print(f"saved {OUTPATH}")

print("\npeak k (argmax of k*E):")
for name, k_at_peak in kmax_table.items():
    print(f"  {name:35s} k_peak = {k_at_peak:.6g}")

KMAX_TXT_PATH = "kmax_table.txt"
with open(KMAX_TXT_PATH, "w") as fh:
    fh.write(
        f"peak k (argmax of k*E), averaged over last {N_LAST} timesteps "
        f"(t = {time[-N_LAST]:.6g} to {time[-1]:.6g})\n\n"
    )
    for name, k_at_peak in kmax_table.items():
        fh.write(f"{name:35s} k_peak = {k_at_peak:.6g}\n")
print(f"saved {KMAX_TXT_PATH}")