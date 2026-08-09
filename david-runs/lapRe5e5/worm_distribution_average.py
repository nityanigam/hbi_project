"""
worm_distribution_average.py

Static histograms of each worm quantity -- length, width, volume, and
cross-sectional area -- averaged over the last N VTK files in a time
series. One PNG image is produced per quantity.

"Averaged over the last N files" means: build a histogram (using the same
bin edges) for each of the last N files, then average the bin heights
across those N histograms. Each frame contributes equally regardless of
how many worms it contains, and the frame-to-frame spread is shown as
error bars (+/- 1 std across the N per-frame histograms).

Usage (from the command line):
    python worm_distribution_average.py "v*.vtk" --outdir results --last-n 10

or from another script / notebook:
    from worm_distribution_average import collect_last_n, plot_averaged_histograms
    frames, data = collect_last_n(sorted(glob.glob("v*.vtk")), n=10)
    plot_averaged_histograms(frames, data, outdir="results")
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from worm_tracking import WormParams, analyze_file, frame_index


# Quantities to histogram -- matches every per-worm column whose mean feeds
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

def collect_last_n(
    paths: list[str | Path],
    n: int,
    params: WormParams | None = None,
    verbose: bool = True,
) -> tuple[list[int], dict[int, "pd.DataFrame"]]:
    """Run the detection pipeline on the last n files of a time-ordered
    series (fewer if the series is shorter than n).

    Returns
    -------
    frames : frame indices of the files that were used, time-ordered
    data   : {frame_index: per-worm DataFrame} for that file
    """
    if n < 1:
        raise ValueError("n must be >= 1")

    params = params or WormParams()
    paths = sorted(paths, key=frame_index)

    if len(paths) < n:
        print(f"Warning: only {len(paths)} files available, using all of "
              f"them instead of the requested last {n}.")
    last_paths = paths[-n:]

    frames, data = [], {}
    for path in last_paths:
        frame = frame_index(path)
        worms, _ = analyze_file(path, params)
        frames.append(frame)
        data[frame] = worms
        if verbose:
            print(f"{Path(path).name}: {len(worms)} worms")
    return frames, data


# ---------------------------------------------------------------------------
# Binning + averaging
# ---------------------------------------------------------------------------

def compute_bin_edges(data: dict, n_bins: int = 25) -> dict[str, np.ndarray]:
    """One fixed set of bin edges per quantity, spanning the pooled range
    of that quantity across the frames being averaged."""
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


def average_histogram(
    data: dict, col: str, bin_edges: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int]:
    """Histogram each frame with the same bin edges, then average the bin
    heights across frames (every frame weighted equally, regardless of
    how many worms it contributed).

    Returns
    -------
    mean_counts : mean bin height across frames
    std_counts  : standard deviation of bin height across frames
    n_used      : number of frames that actually had worms to histogram
    """
    per_frame_counts = []
    for df in data.values():
        if len(df):
            counts, _ = np.histogram(df[col], bins=bin_edges)
            per_frame_counts.append(counts)
    if not per_frame_counts:
        n_bins = len(bin_edges) - 1
        return np.zeros(n_bins), np.zeros(n_bins), 0
    stacked = np.stack(per_frame_counts)
    return stacked.mean(axis=0), stacked.std(axis=0), len(per_frame_counts)


# ---------------------------------------------------------------------------
# Plotting -- one static image per quantity
# ---------------------------------------------------------------------------

def plot_averaged_histograms(
    frames: list[int],
    data: dict,
    outdir: str | Path = "worm_results",
    n_bins: int = 25,
) -> dict[str, Path]:
    """Make one PNG per quantity: bin heights averaged across the given
    frames, with +/- 1 std error bars showing frame-to-frame spread."""
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True, parents=True)
    edges = compute_bin_edges(data, n_bins=n_bins)

    frame_lo, frame_hi = min(frames), max(frames)
    n_frames = len(frames)

    made = {}
    for col, label in QUANTITIES:
        bin_edges = edges[col]
        centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        widths = np.diff(bin_edges)

        mean_counts, std_counts, n_used = average_histogram(data, col, bin_edges)

        pooled = (np.concatenate([df[col].to_numpy()
                                  for df in data.values() if len(df)])
                 if n_used else np.array([]))
        overall_mean = pooled.mean() if pooled.size else np.nan

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.bar(centers, mean_counts, width=widths * 0.9, align="center",
              yerr=std_counts, capsize=3, color="tab:blue",
              edgecolor="black", alpha=0.85, ecolor="black")
        if pooled.size:
            ax.axvline(overall_mean, color="tab:red", ls="--", lw=1.5,
                       label=f"mean = {overall_mean:.3g}")
            ax.legend(loc="upper right", fontsize=8)
        ax.set_xlabel(label)
        ax.set_ylabel(f"Mean count per frame (n = {n_frames} frames)")
        ax.set_title(f"{label} distribution averaged over frames "
                    f"{frame_lo}-{frame_hi}")
        ax.grid(alpha=0.3)
        fig.tight_layout()

        outfile = outdir / f"worm_distribution_avg_{col}.png"
        fig.savefig(outfile, dpi=150)
        plt.close(fig)
        made[col] = outfile
        print(f"  wrote {outfile}")

    return made


# ---------------------------------------------------------------------------
# Command-line entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Static histograms of worm quantities averaged over "
                    "the last N VTK files")
    ap.add_argument("pattern", help='glob pattern, e.g. "v*.vtk"')
    ap.add_argument("--outdir", default="worm_results")
    ap.add_argument("--percentile", type=float, default=98.0)
    ap.add_argument("--min-aspect", type=float, default=8.0)
    ap.add_argument("--last-n", type=int, default=10,
                    help="number of most recent VTK files to average over")
    ap.add_argument("--bins", type=int, default=25,
                    help="number of histogram bins per quantity")
    args = ap.parse_args()

    with open("vtk_dir.txt") as f:
        vtk_dir = f.read().strip()

    paths = sorted(glob.glob(str(Path(vtk_dir) / args.pattern)), key=frame_index)
    if not paths:
        raise SystemExit(f"No files match {args.pattern!r}")

    params = WormParams(threshold_percentile=args.percentile,
                        min_aspect=args.min_aspect)

    frames, data = collect_last_n(paths, args.last_n, params)

    print(f"\nAveraging histograms over {len(frames)} frames "
         f"(frames {min(frames)}-{max(frames)})...")
    made = plot_averaged_histograms(frames, data, outdir=args.outdir,
                                    n_bins=args.bins)

    print(f"\nDone -> {args.outdir}/")
    for col, path in made.items():
        print(f"  {col}: {path}")


if __name__ == "__main__":
    main()