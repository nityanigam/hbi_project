"""
Usage (from the command line):
    python worm_tracking.py "v*.vtk" --outdir results

or from another script / notebook:
    from worm_tracking import WormParams, analyze_series
    per_frame, per_worm, events = analyze_series(sorted(glob.glob("v*.vtk")))
"""

from __future__ import annotations

import argparse
import glob
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyvista as pv
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, distance_transform_edt
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects, skeletonize


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

@dataclass
class WormParams:
    """Tunable detection parameters (all lengths in voxels unless noted)."""
    smooth_sigma: float = 1.0          # Gaussian smoothing of |B|
    threshold_percentile: float = 98.0 # |B| percentile used as threshold
    min_voxels: int = 100              # discard components smaller than this
    min_skeleton_length: int = 30      # discard short skeletons (voxels)
    min_aspect: float = 8.0            # length / width required to be a worm
    min_track_overlap: int = 20        # voxel overlap needed to link frames


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_b_magnitude(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a VTK file and return (|B| as an (nx, ny, nz) array, spacing).

    VTK point data is stored with x varying fastest, so a Fortran-order
    reshape onto (nx, ny, nz) gives an array indexed as B[ix, iy, iz].
    """
    mesh = pv.read(str(path))
    nx, ny, nz = mesh.dimensions

    def grab(name: str) -> np.ndarray:
        return np.asarray(mesh[name]).reshape((nx, ny, nz), order="F")

    bx, by, bz = grab("bx"), grab("by"), grab("bz")
    B = np.sqrt(bx ** 2 + by ** 2 + bz ** 2)

    spacing = np.asarray(getattr(mesh, "spacing", (1.0, 1.0, 1.0)), dtype=float)
    return B, spacing


def frame_index(path: str | Path) -> int:
    """Extract the time index from a filename such as 'v0089.vtk'."""
    m = re.search(r"(\d+)", Path(path).stem)
    return int(m.group(1)) if m else -1


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

def segment(B: np.ndarray, params: WormParams) -> np.ndarray:
    """Threshold |B| and return a labelled array of connected components."""
    Bs = gaussian_filter(B, sigma=params.smooth_sigma)
    mask = Bs > np.percentile(Bs, params.threshold_percentile)
    mask = remove_small_objects(mask, min_size=params.min_voxels)
    return label(mask, connectivity=3)


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

def measure_worms(
    labels_arr: np.ndarray,
    spacing: np.ndarray,
    params: WormParams,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Measure every component and keep the worm-like ones.

    Returns
    -------
    worms : DataFrame with one row per accepted worm (physical units).
    worm_labels : labelled array containing ONLY the accepted worms,
                  relabelled 1..n_worms (used for frame-to-frame tracking).
    """
    dx, dy, dz = spacing
    voxel_volume = dx * dy * dz
    mean_spacing = spacing.mean()

    # Skeleton and distance transform of the full mask.  Components are
    # disjoint, so computing these once globally is safe and much faster
    # than per component.
    mask = labels_arr > 0
    skel = skeletonize(mask)
    edt = distance_transform_edt(mask, sampling=spacing)  # physical units

    records = []
    worm_labels = np.zeros_like(labels_arr)
    next_id = 0

    for rp in regionprops(labels_arr):
        sl = rp.slice
        comp = labels_arr[sl] == rp.label

        # --- length: skeleton voxel count (handles curved worms) ---------
        comp_skel = skel[sl] & comp
        n_skel = int(comp_skel.sum())
        if n_skel < params.min_skeleton_length:
            continue
        length_skel = n_skel * mean_spacing

        # extent along y (the elongation axis) as a second length measure
        x0, y0, z0, x1, y1, z1 = rp.bbox
        length_y = (y1 - y0) * dy

        # --- width: 2 x mean skeleton-to-surface distance ----------------
        radii = edt[sl][comp_skel]
        width = 2.0 * radii.mean()

        aspect = length_skel / width
        if aspect < params.min_aspect:
            continue

        # --- cross-sectional area in the x-z plane -----------------------
        # voxels per occupied y slice, averaged, converted to area
        per_slice = comp.sum(axis=(0, 2))          # comp is (x, y, z)
        per_slice = per_slice[per_slice > 0]
        area_xz = per_slice.mean() * dx * dz

        volume = rp.area * voxel_volume
        cx, cy, cz = (np.asarray(rp.centroid) * spacing)

        next_id += 1
        worm_labels[sl][comp] = next_id
        records.append(
            dict(
                worm=next_id,
                length=length_skel,
                length_y=length_y,
                width=width,
                aspect=aspect,
                area_xz=area_xz,
                volume=volume,
                voxels=int(rp.area),
                cx=cx, cy=cy, cz=cz,
            )
        )

    return pd.DataFrame(records), worm_labels


# ---------------------------------------------------------------------------
# Frame-to-frame tracking: merges, splits, births, deaths
# ---------------------------------------------------------------------------

def overlap_pairs(
    prev_labels: np.ndarray,
    curr_labels: np.ndarray,
    min_overlap: int,
) -> np.ndarray:
    """Return an (n, 3) array of (prev_worm, curr_worm, overlap_voxels)."""
    both = (prev_labels > 0) & (curr_labels > 0)
    if not both.any():
        return np.empty((0, 3), dtype=int)
    pairs, counts = np.unique(
        np.stack([prev_labels[both], curr_labels[both]]),
        axis=1,
        return_counts=True,
    )
    keep = counts >= min_overlap
    return np.column_stack([pairs[0][keep], pairs[1][keep], counts[keep]])


def track_events(
    prev_labels: np.ndarray,
    curr_labels: np.ndarray,
    frame_prev: int,
    frame_curr: int,
    params: WormParams,
) -> pd.DataFrame:
    """Classify what happened to each worm between two consecutive frames.

    Events:
        merge     -- a current worm overlaps >= 2 previous worms
        split     -- a previous worm overlaps >= 2 current worms
        appear    -- a current worm overlaps no previous worm
        disappear -- a previous worm overlaps no current worm
    """
    links = overlap_pairs(prev_labels, curr_labels, params.min_track_overlap)

    prev_ids = set(np.unique(prev_labels)) - {0}
    curr_ids = set(np.unique(curr_labels)) - {0}

    parents: dict[int, list[int]] = {c: [] for c in curr_ids}
    children: dict[int, list[int]] = {p: [] for p in prev_ids}
    for p, c, _ in links:
        parents[c].append(p)
        children[p].append(c)

    events = []
    for c, ps in parents.items():
        if len(ps) >= 2:
            events.append(dict(frame=frame_curr, event="merge",
                               worms_before=sorted(ps), worms_after=[c]))
        elif len(ps) == 0:
            events.append(dict(frame=frame_curr, event="appear",
                               worms_before=[], worms_after=[c]))
    for p, cs in children.items():
        if len(cs) >= 2:
            events.append(dict(frame=frame_curr, event="split",
                               worms_before=[p], worms_after=sorted(cs)))
        elif len(cs) == 0:
            events.append(dict(frame=frame_curr, event="disappear",
                               worms_before=[p], worms_after=[]))
    return pd.DataFrame(events)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def analyze_file(path: str | Path, params: WormParams):
    """Detect and measure worms in a single VTK file."""
    B, spacing = load_b_magnitude(path)
    labels_arr = segment(B, params)
    worms, worm_labels = measure_worms(labels_arr, spacing, params)
    return worms, worm_labels


def analyze_series(
    paths: list[str | Path],
    params: WormParams | None = None,
    verbose: bool = True,
):
    """Run the full pipeline over a time-ordered list of VTK files.

    Returns
    -------
    per_frame : DataFrame -- one row per file (counts and averages)
    per_worm  : DataFrame -- one row per worm per file
    events    : DataFrame -- merge / split / appear / disappear events
    """
    params = params or WormParams()
    paths = sorted(paths, key=frame_index)

    frame_rows, worm_rows, event_frames = [], [], []
    prev_labels, prev_frame = None, None

    for path in paths:
        frame = frame_index(path)
        worms, worm_labels = analyze_file(path, params)

        if verbose:
            print(f"{Path(path).name}: {len(worms)} worms")

        worms.insert(0, "frame", frame)
        worm_rows.append(worms)

        frame_rows.append(
            dict(
                frame=frame,
                n_worms=len(worms),
                mean_length=worms["length"].mean() if len(worms) else np.nan,
                var_length=worms["length"].var() if len(worms) else np.nan,
                mean_width=worms["width"].mean() if len(worms) else np.nan,
                var_width=worms["width"].var() if len(worms) else np.nan,
                mean_volume=worms["volume"].mean() if len(worms) else np.nan,
                var_volume=worms["volume"].var() if len(worms) else np.nan,
                mean_area_xz=worms["area_xz"].mean() if len(worms) else np.nan,
                var_area_xz=worms["area_xz"].var() if len(worms) else np.nan,
                total_volume=worms["volume"].sum() if len(worms) else 0.0,
            )
        )

        if prev_labels is not None:
            ev = track_events(prev_labels, worm_labels,
                              prev_frame, frame, params)
            if len(ev):
                event_frames.append(ev)

        prev_labels, prev_frame = worm_labels, frame

    per_frame = pd.DataFrame(frame_rows)
    per_worm = (pd.concat(worm_rows, ignore_index=True)
                if worm_rows else pd.DataFrame())
    events = (pd.concat(event_frames, ignore_index=True)
              if event_frames else pd.DataFrame(
                  columns=["frame", "event", "worms_before", "worms_after"]))
    return per_frame, per_worm, events


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_time_series(per_frame: pd.DataFrame,
                     events: pd.DataFrame,
                     outfile: str | Path = "worm_time_series.png"):
    """Plot worm statistics against frame index, marking merges and splits."""
    fig, axes = plt.subplots(5, 1, figsize=(9, 13), sharex=True)
    quantities = [
        ("n_worms", "Number of worms"),
        ("mean_length", "Mean length"),
        ("var_length", "Variance of length"),
        ("mean_width", "Mean width"),
        ("var_width", "Variance of width"),
        ("mean_volume", "Mean volume"),
        ("var_volume", "Variance of volume"),
        ("mean_area_xz", "Mean x-z cross-sectional area"),
        ("var_area_xz", "Variance of x-z cross-sectional area")
    ]
    for ax, (col, title) in zip(axes, quantities):
        ax.plot(per_frame["frame"], per_frame[col], "o-", ms=4)
        ax.set_ylabel(title)
        ax.grid(alpha=0.3)

    # mark merge / split events on the worm-count panel
    for kind, color in [("merge", "tab:red"), ("split", "tab:green")]:
        frames = events.loc[events["event"] == kind, "frame"].unique() \
            if len(events) else []
        for k, f in enumerate(frames):
            axes[0].axvline(f, color=color, ls="--", alpha=0.5,
                            label=kind if k == 0 else None)
    if len(events):
        axes[0].legend(loc="best", fontsize=8)

    axes[-1].set_xlabel("Frame")
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    return Path(outfile)


# ---------------------------------------------------------------------------
# Command-line entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Worm detection and tracking")
    ap.add_argument("pattern", help='glob pattern, e.g. "v*.vtk"')
    ap.add_argument("--outdir", default="worm_results")
    ap.add_argument("--percentile", type=float, default=98.0)
    ap.add_argument("--min-aspect", type=float, default=8.0)
    args = ap.parse_args()

    with open("vtk_dir.txt") as f:
        vtk_dir = f.read().strip()

    paths = sorted(glob.glob(str(Path(vtk_dir) / args.pattern)), key=frame_index)
    if not paths:
        raise SystemExit(f"No files match {args.pattern!r}")

    params = WormParams(threshold_percentile=args.percentile,
                        min_aspect=args.min_aspect)
    per_frame, per_worm, events = analyze_series(paths, params)

    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)
    per_frame.to_csv(outdir / "per_frame.csv", index=False)
    per_worm.to_csv(outdir / "per_worm.csv", index=False)
    events.to_csv(outdir / "events.csv", index=False)
    plot_time_series(per_frame, events, outdir / "worm_time_series.png")

    print(f"\nAnalyzed {len(paths)} files -> {outdir}/")
    if len(events):
        n_merge = (events["event"] == "merge").sum()
        n_split = (events["event"] == "split").sum()
        print(f"Detected {n_merge} merges and {n_split} splits:")
        for _, e in events.iterrows():
            if e["event"] in ("merge", "split"):
                print(f"  frame {e['frame']:4d}: {e['event']:5s} "
                      f"{e['worms_before']} -> {e['worms_after']}")


if __name__ == "__main__":
    main()
