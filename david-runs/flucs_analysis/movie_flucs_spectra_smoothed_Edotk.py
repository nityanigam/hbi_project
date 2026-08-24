"""
Time evolution of FLUCS directional spectra, rendered as movies with a
short boxcar average in time applied before each frame is drawn. Each
frame plots k*E(k) vs k (compensated spectrum) rather than E(k) vs k.

Unlike the snoopy-style spectrum_X.dat/spectrum_Y.dat files (see
movie_spectra_smoothed_Edotk.py), FLUCS's boussinesq_mhd_directional_spectra
diagnostic does not split each field into vector components independently
of the binning direction -- kinetic_energy_x is already the total kinetic
energy summed over components, binned against kx (and similarly for y, z,
and for magnetic_fluctuation_energy / theta_variance). So there is no
"parallel field vs horizontal direction" cross-combination available here;
instead this script renders one movie per direction (x, y, z), each
overlaying that direction's kinetic, magnetic, and theta spectra.

As with the snoopy directional spectra, each directional bin here sums
over a thin slice of k-space, so individual snapshots are noisy -- the
same boxcar-in-time smoothing (SMOOTH_WINDOW, applied at the native
output cadence, before subsampling every STRIDE-th frame) is used here.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from scipy.ndimage import uniform_filter1d

import imageio_ffmpeg
matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

from load_flucs_spectra import load_flucs_spectra

RUN_DIR = "/cephfs/store/astro-hl278/dnh26/hbi_flucs/runs/2_production_chi_scan_512/1_chi1over200_full_y_nu1e6"
OUTDIR = "."          # movies write here, not into the (read-only) run dir
STRIDE = 15            # render every 15th (smoothed) timestep as a frame
SMOOTH_WINDOW = 10     # boxcar width in raw snapshots, applied before striding
FPS = 6
TIME_MIN = 2.0         # drop startup transient; set to None to keep all times

matplotlib.rcParams.update({
    "figure.dpi": 130,
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
})


def smooth_time(field, window):
    """Centered boxcar average along the time (snapshot) axis.

    Structurally-empty bins stay exactly zero (they're zero at every
    snapshot, so averaging zeros gives zero) -- smoothing never bleeds
    energy into bins the grid doesn't populate.
    """
    if window <= 1:
        return field
    return uniform_filter1d(field, size=window, axis=0, mode="nearest")


def axis_limits(k, field, idx):
    kpos = k > 0
    kk = k[kpos]
    y = field[np.ix_(idx, kpos)]
    valid = y > 0
    xmin, xmax = kk.min(), kk.max()
    ymin = y[valid].min() if valid.any() else 1e-10
    ymax = y[valid].max() if valid.any() else 1.0
    return (xmin, xmax), (ymin * 0.5, ymax * 2.0)


def make_movie(k, t, fields, direction, outpath, smooth_window=SMOOTH_WINDOW):
    """fields: dict of {label: (nt, nk) array, color}."""
    kpos = k > 0
    kk = k[kpos]
    idx = np.arange(0, len(t), STRIDE)

    smoothed = {
        label: smooth_time(field, smooth_window) * k
        for label, (field, color) in fields.items()
    }

    fig, axes = plt.subplots(1, len(fields), figsize=(6.2 * len(fields), 5.5))
    if len(fields) == 1:
        axes = [axes]

    lines = {}
    for ax, (label, (field, color)) in zip(axes, fields.items()):
        (line,) = ax.loglog([], [], color=color)
        lines[label] = line
        xlim, ylim = axis_limits(k, smoothed[label], idx)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel(f"Wavenumber $k_{direction}$")
        ax.set_ylabel(f"$k\\,E_{{{label}}}(k_{direction})$")
        ax.set_title(label)
        ax.grid(True, which="both", ls="--", lw=0.5)

    suptitle_obj = fig.suptitle("", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    writer = FFMpegWriter(fps=FPS, metadata={"artist": "movie_flucs_spectra_smoothed_Edotk.py"})
    with writer.saving(fig, outpath, dpi=130):
        for i in idx:
            for label, field in smoothed.items():
                y = field[i, kpos]
                valid = y > 0
                lines[label].set_data(kk[valid], y[valid])
            suptitle_obj.set_text(
                f"{direction}-direction spectra\n"
                f"$t = {t[i]:.3f}$  (smoothed over {smooth_window} snapshots)"
            )
            writer.grab_frame()

    plt.close(fig)
    print(f"Saved {outpath}")


data = load_flucs_spectra(RUN_DIR)
time = data["time"]
time_mask = time >= TIME_MIN if TIME_MIN is not None else np.ones_like(time, dtype=bool)
time = time[time_mask]

for direction in ["x", "y", "z"]:
    k = data[f"k{direction}"]
    fields = {
        "kinetic": (data[f"kinetic_energy_{direction}"][time_mask], "tab:blue"),
        "magnetic": (data[f"magnetic_fluctuation_energy_{direction}"][time_mask], "tab:red"),
        "theta": (data[f"theta_variance_{direction}"][time_mask], "tab:green"),
    }
    make_movie(
        k, time, fields, direction,
        f"{OUTDIR}/movie_flucs_{direction}dir_spectra_smoothed_Edotk.mp4",
    )

print("Done.")
