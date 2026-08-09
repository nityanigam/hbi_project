"""
Visual sanity-check for worm detection.

For a single VTK frame this renders a 3-D isosurface of |B| at a chosen
fraction (default 60%) of the maximum |B| in that frame, and overplots the
centres of the worms that ``worm_tracking.py`` identifies in the same frame.

Everything is drawn in the SAME coordinate frame the detector uses, i.e.
``voxel_index * spacing`` (the VTK ``origin`` offset is deliberately dropped,
because ``worm_tracking.measure_worms`` computes centroids as
``rp.centroid * spacing`` without the origin).  That guarantees the surface
and the worm centres are directly comparable.

Usage:
    python check_worms_3d.py v0010.vtk
    python check_worms_3d.py v0010.vtk --frac 0.6 --interactive
    python check_worms_3d.py v0010.vtk --out worm_check_0010.png

The detection parameters (percentile, min-aspect, ...) can be overridden with
the same flags exposed by worm_tracking; defaults match that script.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyvista as pv

from worm_tracking import (
    WormParams,
    analyze_file,
    frame_index,
    load_b_magnitude,
)


def build_isosurface(B: np.ndarray, spacing: np.ndarray, frac: float):
    """Return (contour PolyData, iso_value) for |B| = frac * max(|B|).

    The scalar field is put back onto a pyvista ImageData with origin 0 so the
    surface lives in ``voxel_index * spacing`` coordinates, matching the worm
    centroids from worm_tracking.
    """
    nx, ny, nz = B.shape
    grid = pv.ImageData(dimensions=(nx, ny, nz),
                        spacing=tuple(spacing),
                        origin=(0.0, 0.0, 0.0))
    # B is indexed [ix, iy, iz]; VTK point data wants x varying fastest -> 'F'.
    grid.point_data["B"] = B.ravel(order="F")

    iso = frac * float(B.max())
    contour = grid.contour(isosurfaces=[iso], scalars="B")
    return contour, iso


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("vtk", help="path to a single VTK frame, e.g. v0010.vtk")
    ap.add_argument("--frac", type=float, default=0.6,
                    help="isosurface level as a fraction of max|B| (default 0.6)")
    ap.add_argument("--out", default=None,
                    help="output image path (default worm_check_<frame>.png)")
    ap.add_argument("--interactive", action="store_true",
                    help="open an interactive window instead of saving a PNG")
    ap.add_argument("--percentile", type=float, default=98.0,
                    help="|B| percentile threshold for detection (default 98)")
    ap.add_argument("--min-aspect", type=float, default=8.0,
                    help="minimum length/width to count as a worm (default 8)")
    args = ap.parse_args()

    with open("vtk_dir.txt") as f:
        path = f.read().strip()

    full_path = str(path+str(args.vtk))
    vtk_path = Path(full_path)
    frame = frame_index(Path(args.vtk))

    params = WormParams(threshold_percentile=args.percentile,
                        min_aspect=args.min_aspect)

    print(f"Loading {vtk_path.name} ...")
    B, spacing = load_b_magnitude(vtk_path)

    print("Detecting worms (this runs the full worm_tracking pipeline) ...")
    worms, _ = analyze_file(vtk_path, params)
    print(f"  {len(worms)} worms identified")

    contour, iso = build_isosurface(B, spacing, args.frac)
    print(f"  isosurface at |B| = {iso:.4g} "
          f"({args.frac:.0%} of max = {B.max():.4g}), "
          f"{contour.n_points} surface points")

    # --- render -----------------------------------------------------------
    pl = pv.Plotter(off_screen=not args.interactive, window_size=(1200, 1000))
    pl.set_background("white")
    pl.add_mesh(contour, color="lightsteelblue", opacity=0.35,
                smooth_shading=True, label="|B| iso")

    if len(worms):
        centres = worms[["cx", "cy", "cz"]].to_numpy()
        pts = pv.PolyData(centres)
        # scale spheres to the grid so they are visible but not huge
        extent = float(np.ptp(np.asarray(pl.bounds).reshape(3, 2), axis=1).max())
        radius = 0.02 * extent
        glyphs = pts.glyph(scale=False, geom=pv.Sphere(radius=radius))
        pl.add_mesh(glyphs, color="red", label="worm centre")
        # number each worm centre
        pl.add_point_labels(centres, [str(int(w)) for w in worms["worm"]],
                            font_size=14, text_color="black",
                            point_size=1, shape=None, always_visible=True)

    pl.add_axes()
    pl.add_text(f"{vtk_path.name}  frame {frame}\n"
                f"|B| iso = {args.frac:.0%} of max\n"
                f"{len(worms)} worms",
                font_size=11, color="black")
    pl.add_legend(bcolor="white", size=(0.16, 0.10), loc="lower right")
    pl.camera_position = "iso"

    if args.interactive:
        pl.show()
    else:
        out = args.out or f"worm_check_{frame:04d}.png"
        pl.screenshot(out)
        pl.close()
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
