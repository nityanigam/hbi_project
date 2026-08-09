"""
Time evolution of directional velocity & magnetic spectra, drawn as
loglog line plots colored by time (viridis), one line per sampled
snapshot -- only every 100th timestep is plotted.

spectrum_X.dat / spectrum_Y.dat layout (snoopy-style):
  line0 : k bins (nbin)
  line1 : mode counts (nbin)
  lines2+: t  s0 ... s_{nbin-1}   (nspec=6 rows per snapshot, same t)
    row0: vx*vx   row1: vy*vy   row2: vz*vz
    row3: bx*bx   row4: by*by   row5: bz*bz

y = parallel to B0 / gravity  -> "parallel fields" = vy, by
x = horizontal direction (z has no separate output in this run) ->
    "horizontal fields" = vx + vz, bx + bz
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.cm import viridis
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

DATADIR = "/cephfs/store/astro-hl278/dnh26/hbi_snoopy/runs/03_alpha30_vertical_2d_to_3d/hbi3d_R1p2_vertical_byfield_chi1over1600_3dinit0p1K2d_B0_0p01_n3_box0p25_aspect2_start0/"
STRIDE = 100  # plot every 100th timestep

matplotlib.rcParams.update({
    "figure.dpi": 130,
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
})


def load(path):
    with open(path) as fh:
        lines = [l for l in fh.readlines() if l.strip()]
    k = np.array(lines[0].split(), dtype=float)
    nbin = len(k)
    rows = np.array([l.split() for l in lines[2:]], dtype=float)
    times_all = rows[:, 0]
    specs_all = rows[:, 1:]
    t0 = times_all[0]
    nspec = int(np.sum(times_all == t0))
    nsnap, rem = divmod(len(rows), nspec)
    if rem:
        rows = rows[: nsnap * nspec]
        times_all = rows[:, 0]
        specs_all = rows[:, 1:]
    times = times_all[::nspec]
    data = specs_all.reshape(nsnap, nspec, nbin)  # (nsnap, 6, nbin)
    return k, times, data


kx, tx, datax = load(f"{DATADIR}/spectrum_X.dat")
ky, ty, datay = load(f"{DATADIR}/spectrum_Y.dat")


def plot_evolution(ax, k, t, field, title, ylabel, cmap=viridis):
    kpos = k > 0
    kk = k[kpos]
    norm = Normalize(vmin=t.min(), vmax=t.max())

    idx = np.arange(0, len(t), STRIDE)
    for i in idx:
        y = field[i, kpos]
        valid = y > 0
        # drop structurally-empty bins (some directional grids only
        # populate every other shell) instead of leaving NaN gaps --
        # a plain line plot never draws a segment between two points
        # that straddle a NaN, so an alternating-zero pattern would
        # otherwise render as no line at all.
        ax.loglog(kk[valid], y[valid], color=cmap(norm(t[i])), alpha=0.8)

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = ax.figure.colorbar(sm, ax=ax)
    cbar.set_label("Time $t$")

    ax.set_xlabel("Wavenumber $k$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", ls="--", lw=0.5)


def make_fig(k, t, v_field, b_field, v_title, b_title, suptitle, outpath):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    plot_evolution(axes[0], k, t, v_field, v_title, "$E_v(k)$")
    plot_evolution(axes[1], k, t, b_field, b_title, "$E_b(k)$")
    fig.suptitle(suptitle, fontweight="bold")
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    print(f"Saved {outpath}")
    plt.close(fig)


# 1) parallel fields (vy, by) vs k_y (parallel direction)
make_fig(ky, ty, datay[:, 1, :], datay[:, 4, :],
          "$E_{v_y}(k_y)$", "$E_{b_y}(k_y)$",
          "Parallel fields ($v_y$, $b_y$) — spectrum vs $k_y$ (parallel direction)",
          f"lines_fig1_parallel_field_parallel_dir.png")

# 2) parallel fields (vy, by) vs k_x (horizontal direction)
make_fig(kx, tx, datax[:, 1, :], datax[:, 4, :],
          "$E_{v_y}(k_x)$", "$E_{b_y}(k_x)$",
          "Parallel fields ($v_y$, $b_y$) — spectrum vs $k_x$ (horizontal direction)",
          f"lines_fig2_parallel_field_horizontal_dir.png")

# 3) horizontal fields (vx+vz, bx+bz) vs k_y (parallel direction) -- "opposite" of 1
v_horiz_y = datay[:, 0, :] + datay[:, 2, :]
b_horiz_y = datay[:, 3, :] + datay[:, 5, :]
make_fig(ky, ty, v_horiz_y, b_horiz_y,
          "$E_{v_x+v_z}(k_y)$", "$E_{b_x+b_z}(k_y)$",
          "Horizontal fields ($v_x{+}v_z$, $b_x{+}b_z$) — spectrum vs $k_y$ (parallel direction)",
          f"lines_fig3_horizontal_field_parallel_dir.png")

# 4) horizontal fields (vx+vz, bx+bz) vs k_x (horizontal direction) -- "opposite" of 2
v_horiz_x = datax[:, 0, :] + datax[:, 2, :]
b_horiz_x = datax[:, 3, :] + datax[:, 5, :]
make_fig(kx, tx, v_horiz_x, b_horiz_x,
          "$E_{v_x+v_z}(k_x)$", "$E_{b_x+b_z}(k_x)$",
          "Horizontal fields ($v_x{+}v_z$, $b_x{+}b_z$) — spectrum vs $k_x$ (horizontal direction)",
          f"lines_fig4_horizontal_field_horizontal_dir.png")

print("Done.")
