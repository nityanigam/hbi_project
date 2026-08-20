#!/usr/bin/env python3
"""Load every saved FLUCS directional spectrum into NumPy arrays.

Example
-------
from load_flucs_spectra import load_flucs_spectra

spectra = load_flucs_spectra("path/to/run")
time = spectra["time"]
kx = spectra["kx"]
kinetic_x = spectra["kinetic_energy_x"]

The time-dependent arrays have time as their first axis.  Rows are not
deduplicated: ``segment`` records the numbered FLUCS output group from which
each row came, so restart boundaries remain available to the caller.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from netCDF4 import Dataset, Group, Variable

DEFAULT_DIAGNOSTIC = "boussinesq_mhd_directional_spectra"


def _plain_array(variable: Variable) -> np.ndarray:
    """Read a NetCDF variable as an unmasked NumPy array."""
    values = variable[:]
    if np.ma.isMaskedArray(values) and np.any(np.ma.getmaskarray(values)):
        raise ValueError(f"{variable.group().path}/{variable.name} has missing data")
    return np.asarray(values)


def _numbered_groups(dataset: Dataset) -> list[tuple[int, Group]]:
    groups: list[tuple[int, Group]] = []
    for name, group in dataset.groups.items():
        try:
            number = int(name)
        except ValueError as error:
            raise ValueError(f"FLUCS output group {name!r} is not numbered") from error
        groups.append((number, group))
    return sorted(groups)


def _diagnostic_group(group: Group, name: str | None) -> Group:
    if name is not None:
        try:
            return group.groups[name]
        except KeyError as error:
            raise KeyError(f"{group.path} has no diagnostic group {name!r}") from error

    candidates = [
        subgroup
        for subgroup in group.groups.values()
        if any("time" in variable.dimensions for variable in subgroup.variables.values())
    ]
    if len(candidates) != 1:
        names = [candidate.path for candidate in candidates]
        raise ValueError(
            f"expected one spectral diagnostic below {group.path}, found {names}"
        )
    return candidates[0]


def load_flucs_spectra(
    path: str | Path,
    diagnostic: str | None = DEFAULT_DIAGNOSTIC,
) -> dict[str, np.ndarray]:
    """Return all spectra in a FLUCS ``output.1d.nc`` as NumPy arrays.

    ``path`` may name either the NetCDF file or its run directory.  Returned
    keys are ``time``, ``dt``, ``segment``, ``row_in_segment``, all coordinate
    variables (normally ``kx``, ``ky``, and ``kz``), and every spectrum in the
    selected diagnostic group.  Every spectrum has shape ``(nt, nk)``.

    Set ``diagnostic=None`` to auto-detect the sole time-dependent diagnostic
    subgroup.  This is useful for other FLUCS systems whose diagnostic group
    has a different name.
    """
    source = Path(path).expanduser()
    if source.is_dir():
        source = source / "output.1d.nc"
    if not source.is_file():
        raise FileNotFoundError(source)

    time_parts: list[np.ndarray] = []
    dt_parts: list[np.ndarray] = []
    segment_parts: list[np.ndarray] = []
    row_parts: list[np.ndarray] = []
    coordinates: dict[str, np.ndarray] = {}
    spectrum_parts: dict[str, list[np.ndarray]] = {}
    spectrum_dimensions: dict[str, tuple[str, ...]] = {}

    with Dataset(source, "r") as dataset:
        groups = _numbered_groups(dataset)
        if not groups:
            raise ValueError(f"{source} contains no FLUCS output groups")

        for segment, group in groups:
            time = _plain_array(group.variables["time"])
            dt = _plain_array(group.variables["dt"])
            if time.ndim != 1 or dt.shape != time.shape:
                raise ValueError(f"invalid time or dt shape in {group.path}")
            if time.size == 0:
                continue
            if np.any(np.diff(time) <= 0.0):
                raise ValueError(f"times are not strictly increasing in {group.path}")

            spectra = _diagnostic_group(group, diagnostic)
            group_coordinates: dict[str, np.ndarray] = {}
            group_spectra: dict[str, np.ndarray] = {}
            group_dimensions: dict[str, tuple[str, ...]] = {}

            for name, variable in spectra.variables.items():
                values = _plain_array(variable)
                if "time" not in variable.dimensions:
                    group_coordinates[name] = values
                    continue
                time_axis = variable.dimensions.index("time")
                values = np.moveaxis(values, time_axis, 0)
                if values.shape[0] != time.size:
                    raise ValueError(f"time length mismatch for {variable.group().path}/{name}")
                group_spectra[name] = values
                group_dimensions[name] = tuple(
                    dimension
                    for dimension in variable.dimensions
                    if dimension != "time"
                )

            if not group_spectra:
                raise ValueError(f"no spectra found in {spectra.path}")
            if not spectrum_parts:
                coordinates = group_coordinates
                spectrum_parts = {name: [] for name in group_spectra}
                spectrum_dimensions = group_dimensions
            else:
                if group_spectra.keys() != spectrum_parts.keys():
                    raise ValueError(f"spectrum names changed in {spectra.path}")
                if group_coordinates.keys() != coordinates.keys():
                    raise ValueError(f"coordinate names changed in {spectra.path}")
                for name, values in group_coordinates.items():
                    if not np.array_equal(values, coordinates[name], equal_nan=True):
                        raise ValueError(f"coordinate {name!r} changed in {spectra.path}")
                if group_dimensions != spectrum_dimensions:
                    raise ValueError(f"spectrum dimensions changed in {spectra.path}")

            time_parts.append(time)
            dt_parts.append(dt)
            segment_parts.append(np.full(time.size, segment, dtype=np.int64))
            row_parts.append(np.arange(time.size, dtype=np.int64))
            for name, values in group_spectra.items():
                spectrum_parts[name].append(values)

    if not time_parts:
        raise ValueError(f"{source} contains no saved spectrum rows")

    result = {
        "time": np.concatenate(time_parts),
        "dt": np.concatenate(dt_parts),
        "segment": np.concatenate(segment_parts),
        "row_in_segment": np.concatenate(row_parts),
        **coordinates,
    }
    result.update(
        {name: np.concatenate(parts, axis=0) for name, parts in spectrum_parts.items()}
    )
    if np.any(np.diff(result["time"]) < 0.0):
        raise ValueError(f"time moves backwards between groups in {source}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="output.1d.nc or its run directory")
    parser.add_argument(
        "--diagnostic",
        default=DEFAULT_DIAGNOSTIC,
        help="diagnostic subgroup name; use 'auto' to auto-detect it",
    )
    parser.add_argument("--npz", type=Path, help="optionally save the arrays here")
    arguments = parser.parse_args()

    diagnostic = None if arguments.diagnostic == "auto" else arguments.diagnostic
    arrays = load_flucs_spectra(arguments.path, diagnostic=diagnostic)
    for name, values in arrays.items():
        print(f"{name:40s} shape={values.shape!s:16s} dtype={values.dtype}")
    if arguments.npz is not None:
        np.savez(arguments.npz, **arrays)
        print(f"saved {arguments.npz}")


if __name__ == "__main__":
    main()
