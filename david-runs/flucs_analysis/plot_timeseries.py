import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from load_flucs_spectra import load_flucs_spectra

run_dir = "/cephfs/store/astro-hl278/dnh26/hbi_flucs/runs/13_gradient_complete_beta_scan_3d_512x1024x512/01_beta000100"
data = load_flucs_spectra(run_dir, diagnostic="compressible_spectral_resolution")

time = data["time"]
mask = time >= 2
fields = [k for k in data if k.endswith("_outer_fraction")]

fig, ax = plt.subplots()
for name in fields:
    ax.plot(time[mask], data[name][mask], label=name, linewidth=0.75)
ax.set_xlabel("time")
ax.set_ylabel("outer fraction")
ax.legend(fontsize=7)
fig.savefig("resolution_diagnostics.png", dpi=150, bbox_inches="tight")
print("saved resolution_diagnostics.png")