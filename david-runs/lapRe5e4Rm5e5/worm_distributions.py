"""
worm_distributions.py

Make movies of the time evolution of the distribution (histogram) of each
worm quantity -- length, width, volume, and cross-sectional area -- one
movie per quantity. Each frame of a movie is a histogram built from every
worm detected in the corresponding VTK file; frames are stepped through in
the same time order used by worm_tracking.py, with the underlying VTK file
name / frame index shown as the title.

Reuses the exact same detection/measurement pipeline as worm_tracking.py
(WormParams, analyze_file, frame_index) so the per-worm quantities are
identical to the ones summarized in worm_time_series.png.

Usage (from the command line):
    python worm_distributions.py "v*.vtk" --outdir results

or from another script / notebook:
    from worm_distributions import collect_per_worm, make_all_movies
    frames, data = collect_per_worm(sorted(glob.glob("v*.vtk")))
    make_all_movies(frames, data, outdir="results")
"""

from __future__ import annotations

import argparse
import glob
import shutil
from pathlib import Path

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from worm_tracking import WormParams, analyze_file, frame_index


# If there's no ffmpeg on PATH (common on clusters with no sudo / no module
# system), fall back to the statically-linked ffmpeg binary shipped by the
# 'imageio-ffmpeg' pip package (installable with `pip install --user
# imageio-ffmpeg`, no root required). This only changes *which* ffmpeg
# binary matplotlib uses -- the output is still a real mp4 either way.
if shutil.which("ffmpeg") is None:
    try:
        import imageio_ffmpeg
        matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass  # no system ffmpeg and no imageio-ffmpeg; _save_animation
              # below will raise a clear, actionable error


# Quantities to animate -- matches every per-worm column whose mean feeds
# worm_time_series.png in worm_tracking.py, excluding n_worms (a count,
# not a per-worm quantity with its own distribution).
QUANTITIES = [
    ("length", "Length"),
    ("width", "Width"),
    ("volume", "Volume"),
    ("area_xz", "x-z cross-sectional area"),
]


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def collect_per_worm(
    paths: list[str | Path],
    params: WormParams | None = None,
    verbose: bool = True,
) -> tuple[list[int], dict[int, "pd.DataFrame"]]:
    """Run the detection pipeline on every file.

    Returns
    -------
    frames : time-ordered list of frame indices
    data   : {frame_index: per-worm DataFrame} for that file
    """
    params = params or WormParams()
    paths = sorted(paths, key=frame_index)

    frames, data = [], {}
    for path in paths:
        frame = frame_index(path)
        worms, _ = analyze_file(path, params)
        frames.append(frame)
        data[frame] = worms
        if verbose:
            print(f"{Path(path).name}: {len(worms)} worms")
    return frames, data


# ---------------------------------------------------------------------------
# Binning
# ---------------------------------------------------------------------------

def compute_bin_edges(data: dict, n_bins: int = 25) -> dict[str, np.ndarray]:
    """One fixed set of bin edges per quantity, spanning its pooled range
    across every frame, so the x-axis stays put across the whole movie."""
    edges = {}
    for col, _ in QUANTITIES:
        pieces = [df[col].to_numpy() for df in data.values() if len(df)]
        if pieces:
            all_vals = np.concatenate(pieces)
            lo, hi = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
        else:
            lo, hi = 0.0, 1.0
        if lo == hi:
            hi = lo + 1.0
        edges[col] = np.linspace(lo, hi, n_bins + 1)
    return edges


# ---------------------------------------------------------------------------
# Movie making -- one movie per quantity
# ---------------------------------------------------------------------------

def _save_animation(ani: animation.FuncAnimation, outfile: Path, fps: int) -> Path:
    """Save as mp4 via ffmpeg. Raises a clear error (instead of silently
    switching to GIF) if ffmpeg can't be found, so the mp4-vs-gif choice
    is never made for you."""
    if not animation.FFMpegWriter.isAvailable():
        raise RuntimeError(
            "ffmpeg was not found, so an mp4 can't be written. No root/sudo "
            "is needed for any of these:\n"
            "  1) pip install --user imageio-ffmpeg\n"
            "     (this script auto-detects it -- just rerun afterwards)\n"
            "  2) if you use conda/mamba: conda install -c conda-forge ffmpeg\n"
            "  3) check for a cluster environment module: module avail ffmpeg "
            "&& module load ffmpeg\n"
            "  4) if ffmpeg is installed somewhere non-standard, point "
            "matplotlib at it directly near the top of this script:\n"
            "       matplotlib.rcParams['animation.ffmpeg_path'] = "
            "'/full/path/to/ffmpeg'"
        )
    ani.save(outfile, writer=animation.FFMpegWriter(fps=fps))
    return outfile


def make_movie_for_quantity(
    frames: list[int],
    data: dict,
    col: str,
    label: str,
    edges: dict[str, np.ndarray],
    outfile: str | Path,
    fps: int = 6,
) -> Path:
    """Animate the histogram of a single quantity over time."""
    frames_sorted = sorted(frames)
    bin_edges = edges[col]
    widths = np.diff(bin_edges)

    # shared y-limit so bars don't jump around vertically frame to frame
    ymax = 1
    for df in data.values():
        if len(df):
            counts, _ = np.histogram(df[col], bins=bin_edges)
            ymax = max(ymax, int(counts.max()))
    ymax = int(np.ceil(ymax * 1.15))

    fig, ax = plt.subplots(figsize=(7, 5))

    def init():
        ax.set_xlim(bin_edges[0], bin_edges[-1])
        ax.set_ylim(0, ymax)
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        ax.grid(alpha=0.3)
        return []

    def update(frame):
        for patch in list(ax.patches):
            patch.remove()
        for line in list(ax.lines):
            line.remove()
        df = data[frame]
        n = len(df)
        if n:
            counts, _ = np.histogram(df[col], bins=bin_edges)
            ax.bar(bin_edges[:-1], counts, width=widths, align="edge",
                   color="tab:blue", edgecolor="black", alpha=0.85)
            mean_val = df[col].mean()
            ax.axvline(mean_val, color="tab:red", ls="--", lw=1.5,
                       label=f"mean = {mean_val:.3g}")
            ax.legend(loc="upper right", fontsize=8)
        ax.set_title(f"{label} distribution -- frame {frame} (n = {n} worms)")
        return list(ax.patches) + list(ax.lines)

    ani = animation.FuncAnimation(
        fig, update, frames=frames_sorted, init_func=init,
        blit=False, repeat=False,
    )
    outfile = _save_animation(ani, Path(outfile), fps)
    plt.close(fig)
    return outfile


def make_all_movies(
    frames: list[int],
    data: dict,
    outdir: str | Path = "worm_results",
    n_bins: int = 25,
    fps: int = 6,
) -> dict[str, Path]:
    """Make one distribution movie per quantity and return their paths."""
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True, parents=True)
    edges = compute_bin_edges(data, n_bins=n_bins)

    made = {}
    for col, label in QUANTITIES:
        outfile = outdir / f"worm_distribution_{col}.mp4"
        path = make_movie_for_quantity(frames, data, col, label, edges,
                                       outfile, fps=fps)
        made[col] = path
        print(f"  wrote {path}")
    return made


# ---------------------------------------------------------------------------
# Command-line entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Movies of worm quantity distributions over time")
    ap.add_argument("pattern", help='glob pattern, e.g. "v*.vtk"')
    ap.add_argument("--outdir", default="worm_results")
    ap.add_argument("--percentile", type=float, default=98.0)
    ap.add_argument("--min-aspect", type=float, default=8.0)
    ap.add_argument("--bins", type=int, default=25,
                    help="number of histogram bins per quantity")
    ap.add_argument("--fps", type=int, default=6,
                    help="frames per second for the output movies")
    args = ap.parse_args()

    with open("vtk_dir.txt") as f:
        vtk_dir = f.read().strip()

    paths = sorted(glob.glob(str(Path(vtk_dir) / args.pattern)), key=frame_index)
    if not paths:
        raise SystemExit(f"No files match {args.pattern!r}")
    print(paths)
    params = WormParams(threshold_percentile=args.percentile,
                        min_aspect=args.min_aspect)

    frames, data = collect_per_worm(paths, params)

    print(f"\nBuilding distribution movies for {len(frames)} frames...")
    made = make_all_movies(frames, data, outdir=args.outdir,
                           n_bins=args.bins, fps=args.fps)

    print(f"\nDone -> {args.outdir}/")
    for col, path in made.items():
        print(f"  {col}: {path}")


if __name__ == "__main__":
    main()