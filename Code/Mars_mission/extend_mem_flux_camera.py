#!/usr/bin/env python3
"""
Extend a MEM 3 speed-binned flux run beyond its limiting mass and fold the
result through an empirical METEORCAM mass-speed detection model.

The script is designed for a MEM run performed at 10 g, but the reference mass
is configurable. It:

1. By default, reads every HiDensity/LoDensity flux_N.txt pair and the
   matching state-vector row in input.txt.
2. Treats the MEM trajectory as a fictitious sampling observer near 100 km
   altitude. For every angular-speed cell it removes only that observer's
   Mars-relative orbital velocity, recovers the meteoroid velocity in the
   Mars-centred frame at the same sampling position, projects the result onto
   the requested local surface, and re-bins by Mars-relative speed.
3. Uses the Grün cumulative mass-scaling equation used by MEM 3:

       F(>m, v) = F_MEM(>m_ref, v) * g(m)/g(m_ref)

4. Converts cumulative fluxes into finite mass bins:

       F([m1,m2), v) = F_MEM(>m_ref, v) * [g(m1)-g(m2)]/g(m_ref)

5. Loads the detection-likelihood pickle written by the Mars synthetic-speed
   script, or fits a compatible NumPy logistic model from its CSV output.
6. Integrates detection probability within each mass bin.
7. Saves incident and camera-detectable mass-speed flux grids, plots, CSV and
   JSON/text summaries.

The older cube_avg.txt pathway is retained with
``--environment-source cube-average``.

Important MEM interpretation:
- Cube face fluxes are not the same as total cross-sectional flux.
- For the METEORCAM surface-flux calculation, the default uses only the MEM
  ``+z zenith`` face via ``--flux-mode zenith``.
- Use ``--flux-mode selected-direction --selected-direction "..."`` to choose
  any other individual MEM direction.
- The cross-sectional-normalized and six-face modes remain available for
  comparison with other flux definitions.

Only NumPy and Matplotlib are required; SciPy and pandas are not used.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np


DIRECTIONS = [
    "+x ram", "-x wake", "+y port", "-y starboard",
    "+z zenith", "-z nadir", "Earth", "Sun", "anti-Sun",
    "rot (x)", "rot (y)", "rot (z)",
]

DEFAULT_FACE_DIRECTIONS = [
    "+x ram", "-x wake", "+y port", "-y starboard",
    "+z zenith", "-z nadir",
]

DEFAULT_MASS_EDGES_G = [10, 50, 100, 250, 500, 1000, 2000, 5000, 10000]


@dataclass
class MemCubeData:
    path: Path
    speeds_kms: np.ndarray
    direction_flux: dict[str, np.ndarray]
    total_cross_sectional_flux_m2_yr: float


# -----------------------------------------------------------------------------
# Grün cumulative mass scaling used by MEM 3
# -----------------------------------------------------------------------------

def grun_cumulative_flux(mass_g: np.ndarray | float) -> np.ndarray:
    """
    Return the Grün cumulative flux function g(m) for mass in grams.

    The absolute value is less important here than the ratio g(m)/g(m_ref),
    which is the mass-rescaling prescribed by the MEM 3 user guide.
    """
    m = np.asarray(mass_g, dtype=float)
    valid = np.isfinite(m) & (m > 0.0)

    c4 = 2.2e3
    c5 = 15.0
    c6 = 1.3e-9
    c7 = 1.0e-11
    c8 = 1.0e27
    c9 = 1.3e-16
    c10 = 1.0e6

    gamma4 = 0.306
    gamma5 = -4.38
    gamma6 = 2.0
    gamma7 = 4.0
    gamma8 = -0.36
    gamma9 = 2.0
    gamma10 = -0.85

    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        value = (
            (c4*m**gamma4 + c5)**gamma5
            + c6*(m + c7*m**gamma6 + c8*m**gamma7)**gamma8
            + c9*(m + c10*m**gamma9)**gamma10
        )
    return np.asarray(np.where(valid, value, np.nan), dtype=float)


def grun_scale_ratio(mass_g: np.ndarray | float, reference_mass_g: float) -> np.ndarray:
    reference_value = float(grun_cumulative_flux(float(reference_mass_g)))
    if not np.isfinite(reference_value) or reference_value <= 0.0:
        raise ValueError("The Grün function is invalid at the reference mass.")
    return np.asarray(grun_cumulative_flux(mass_g)/reference_value, dtype=float)


# -----------------------------------------------------------------------------
# MEM cube-file reading and combination
# -----------------------------------------------------------------------------

def read_mem_cube_file(path: str | Path) -> MemCubeData:
    path = Path(path)
    rows: list[list[float]] = []
    total_cross_sectional = np.nan

    cross_pattern = re.compile(
        r"total\s+cross-sectional\s+flux\s+([0-9.+\-Ee]+)",
        re.IGNORECASE,
    )

    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            match = cross_pattern.search(line)
            if match:
                try:
                    total_cross_sectional = float(match.group(1))
                except ValueError:
                    pass

            parts = line.split()
            if len(parts) != 13:
                continue
            try:
                rows.append([float(value) for value in parts])
            except ValueError:
                continue

    if not rows:
        raise ValueError(f"No 13-column speed-bin rows were found in {path}.")

    array = np.asarray(rows, dtype=float)
    if array.ndim != 2 or array.shape[1] != 13:
        raise ValueError(f"Could not parse MEM cube data from {path}.")

    speeds = array[:, 0]
    direction_flux = {
        direction: array[:, index + 1]
        for index, direction in enumerate(DIRECTIONS)
    }

    return MemCubeData(
        path=path,
        speeds_kms=speeds,
        direction_flux=direction_flux,
        total_cross_sectional_flux_m2_yr=float(total_cross_sectional),
    )


def discover_cube_files(mem_directory: str | Path) -> list[Path]:
    root = Path(mem_directory)
    if not root.exists():
        raise FileNotFoundError(f"MEM directory does not exist: {root}")

    candidates = sorted(root.rglob("cube_avg.txt"))
    if not candidates:
        raise FileNotFoundError(
            f"No cube_avg.txt files were found recursively under {root}."
        )

    # Prefer the standard HiDensity and LoDensity pair when present.
    preferred = [
        path for path in candidates
        if path.parent.name.lower() in {"hidensity", "lodensity"}
    ]
    return preferred if preferred else candidates


def combine_mem_cube_files(paths: Iterable[str | Path]) -> tuple[np.ndarray, dict[str, np.ndarray], float, list[MemCubeData]]:
    cubes = [read_mem_cube_file(path) for path in paths]
    if not cubes:
        raise ValueError("At least one MEM cube file is required.")

    speeds = cubes[0].speeds_kms.copy()
    combined = {direction: np.zeros_like(speeds) for direction in DIRECTIONS}
    total_cross_sectional = 0.0
    finite_cross_count = 0

    for cube in cubes:
        if cube.speeds_kms.shape != speeds.shape or not np.allclose(
            cube.speeds_kms, speeds, rtol=0.0, atol=1.0e-9
        ):
            raise ValueError(
                f"Speed bins in {cube.path} do not match the first cube file."
            )
        for direction in DIRECTIONS:
            combined[direction] += cube.direction_flux[direction]

        if np.isfinite(cube.total_cross_sectional_flux_m2_yr):
            total_cross_sectional += cube.total_cross_sectional_flux_m2_yr
            finite_cross_count += 1

    if finite_cross_count == 0:
        total_cross_sectional = np.nan

    return speeds, combined, float(total_cross_sectional), cubes



@dataclass(frozen=True)
class TrajectoryState:
    index: int
    julian_date: float
    position_km: np.ndarray
    velocity_kms: np.ndarray


@dataclass
class MemFluxGrid:
    path: Path
    elevation_low_deg: np.ndarray
    azimuth_low_deg: np.ndarray
    speed_labels_raw_kms: np.ndarray
    speed_midpoints_kms: np.ndarray
    flux_m2_yr: np.ndarray
    elevation_step_deg: float
    azimuth_step_deg: float
    speed_label_shift_kms: float


@dataclass
class DetailedFluxResult:
    mars_speed_midpoints_kms: np.ndarray
    transformed_speed_flux_m2_yr: np.ndarray
    spacecraft_speed_midpoints_kms: np.ndarray
    spacecraft_surface_flux_m2_yr: np.ndarray
    state_rows: list[dict[str, Any]]
    flux_files: list[str]
    trajectory_file: str
    output_axes: str
    speed_label_shift_kms: float
    state_weighting: str
    sampling_altitude_mean_km: float
    sampling_altitude_min_km: float
    sampling_altitude_max_km: float
    flux_frame_scaling: str
    selected_surface: str


def read_trajectory_file(path: str | Path) -> list[TrajectoryState]:
    """Read MEM's copied trajectory file, ignoring comment/header lines."""
    path = Path(path)
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 7:
                continue
            try:
                rows.append([float(value) for value in parts[:7]])
            except ValueError:
                continue

    if not rows:
        raise ValueError(f"No seven-column state vectors were found in {path}.")

    return [
        TrajectoryState(
            index=index,
            julian_date=float(row[0]),
            position_km=np.asarray(row[1:4], dtype=float),
            velocity_kms=np.asarray(row[4:7], dtype=float),
        )
        for index, row in enumerate(rows, start=1)
    ]


def discover_trajectory_file(
    mem_directory: str | Path,
    explicit: str | Path | None = None,
) -> Path:
    root = Path(mem_directory)
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(f"Trajectory input file does not exist: {path}")
        return path.resolve()

    exact = root / "input.txt"
    if exact.exists():
        return exact.resolve()

    matches = sorted(root.rglob("input.txt"))
    if not matches:
        raise FileNotFoundError(
            f"No input.txt trajectory copy was found under {root}. "
            "Supply --trajectory-file explicitly."
        )
    return matches[0].resolve()


def _indexed_files(directory: Path, stem: str) -> dict[int, Path]:
    pattern = re.compile(rf"^{re.escape(stem)}_(\d+)\.txt$", re.IGNORECASE)
    indexed: dict[int, Path] = {}
    if not directory.exists():
        return indexed
    for path in directory.iterdir():
        if path.is_file():
            match = pattern.match(path.name)
            if match:
                indexed[int(match.group(1))] = path.resolve()
    return indexed


def discover_flux_pairs(mem_directory: str | Path) -> list[tuple[int, Path, Path]]:
    """Find matching HiDensity/LoDensity flux_N.txt products."""
    root = Path(mem_directory)
    high_dirs = [
        path for path in root.rglob("*")
        if path.is_dir() and path.name.lower() == "hidensity"
    ]
    low_dirs = [
        path for path in root.rglob("*")
        if path.is_dir() and path.name.lower() == "lodensity"
    ]
    if not high_dirs or not low_dirs:
        raise FileNotFoundError(
            f"Could not find both HiDensity and LoDensity folders under {root}."
        )

    high_dir = sorted(high_dirs, key=lambda path: len(path.parts))[0]
    low_dir = sorted(low_dirs, key=lambda path: len(path.parts))[0]
    high = _indexed_files(high_dir, "flux")
    low = _indexed_files(low_dir, "flux")

    common = sorted(set(high) & set(low))
    if not common:
        raise FileNotFoundError(
            f"No matching flux_N.txt pairs were found in {high_dir} and {low_dir}."
        )

    missing_high = sorted(set(low) - set(high))
    missing_low = sorted(set(high) - set(low))
    if missing_high or missing_low:
        raise ValueError(
            "HiDensity and LoDensity flux files do not form complete pairs. "
            f"Missing HiDensity indices: {missing_high[:10]}; "
            f"missing LoDensity indices: {missing_low[:10]}."
        )

    return [(index, high[index], low[index]) for index in common]


def _infer_regular_step(values: np.ndarray, name: str) -> float:
    unique = np.unique(np.asarray(values, dtype=float))
    differences = np.diff(np.sort(unique))
    differences = differences[differences > 1.0e-10]
    if differences.size == 0:
        raise ValueError(f"Could not infer {name} resolution.")
    return float(np.median(differences))


def _correct_speed_labels(
    labels: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, float]:
    """
    Correct legacy MEM headers that printed velocity-bin midpoints rounded down.

    With 1 km/s bins, an old header may contain 0,1,...,59 although the true
    bin midpoints are 0.5,1.5,...,59.5 km/s.
    """
    labels = np.asarray(labels, dtype=float)
    if labels.size < 1:
        raise ValueError("The flux file contains no speed labels.")

    width = float(np.median(np.diff(labels))) if labels.size > 1 else 1.0
    if width <= 0.0:
        raise ValueError("Speed labels are not strictly increasing.")

    if mode == "none":
        shift = 0.0
    elif mode == "add-half-bin":
        shift = 0.5 * width
    elif mode == "auto":
        regular = np.allclose(
            labels,
            labels[0] + np.arange(labels.size) * width,
            rtol=0.0,
            atol=1.0e-8,
        )
        shift = 0.5 * width if regular and abs(float(labels[0])) < 1.0e-8 else 0.0
    else:
        raise ValueError(f"Unknown speed-label mode: {mode}")

    return labels + shift, float(shift)


def read_mem_flux_file(
    path: str | Path,
    speed_label_mode: str = "auto",
) -> MemFluxGrid:
    """Read one MEM flux_N.txt angular-speed grid."""
    path = Path(path)
    speed_labels: np.ndarray | None = None
    rows: list[list[float]] = []

    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            cleaned = line.lstrip("#").strip()
            tokens = cleaned.split()
            upper = [token.upper() for token in tokens]
            if "PHI1" in upper and "THETA1" in upper:
                theta_index = upper.index("THETA1")
                try:
                    speed_labels = np.asarray(
                        [float(value) for value in tokens[theta_index + 1:]],
                        dtype=float,
                    )
                except ValueError as exc:
                    raise ValueError(
                        f"Could not parse speed labels in {path}: {line.strip()}"
                    ) from exc
                continue

            if speed_labels is None:
                continue
            parts = line.split()
            if len(parts) != speed_labels.size + 2:
                continue
            try:
                rows.append([float(value) for value in parts])
            except ValueError:
                continue

    if speed_labels is None:
        raise ValueError(f"No PHI1 THETA1 speed-header line was found in {path}.")
    if not rows:
        raise ValueError(f"No angular flux rows were found in {path}.")

    array = np.asarray(rows, dtype=float)
    elevations = array[:, 0]
    azimuths = array[:, 1]
    flux = array[:, 2:]
    corrected_speed, shift = _correct_speed_labels(speed_labels, speed_label_mode)

    return MemFluxGrid(
        path=path.resolve(),
        elevation_low_deg=elevations,
        azimuth_low_deg=azimuths,
        speed_labels_raw_kms=speed_labels,
        speed_midpoints_kms=corrected_speed,
        flux_m2_yr=flux,
        elevation_step_deg=_infer_regular_step(elevations, "elevation"),
        azimuth_step_deg=_infer_regular_step(azimuths, "azimuth"),
        speed_label_shift_kms=shift,
    )


def _assert_matching_flux_grids(first: MemFluxGrid, second: MemFluxGrid) -> None:
    checks = (
        np.array_equal(first.elevation_low_deg, second.elevation_low_deg),
        np.array_equal(first.azimuth_low_deg, second.azimuth_low_deg),
        np.allclose(first.speed_midpoints_kms, second.speed_midpoints_kms),
        first.flux_m2_yr.shape == second.flux_m2_yr.shape,
    )
    if not all(checks):
        raise ValueError(
            f"Flux grids do not match: {first.path} and {second.path}."
        )


def angular_bin_radiants(grid: MemFluxGrid) -> np.ndarray:
    """
    Return representative radiant unit vectors in the selected MEM output frame.

    The elevation representative is the equal-solid-angle midpoint of each
    rectangular angular cell.
    """
    phi1 = np.radians(grid.elevation_low_deg)
    phi2 = np.radians(
        np.minimum(grid.elevation_low_deg + grid.elevation_step_deg, 90.0)
    )
    phi_mid = np.arcsin(
        np.clip(0.5 * (np.sin(phi1) + np.sin(phi2)), -1.0, 1.0)
    )
    theta_mid = np.radians(
        np.mod(grid.azimuth_low_deg + 0.5 * grid.azimuth_step_deg, 360.0)
    )
    cos_phi = np.cos(phi_mid)
    return np.column_stack(
        (
            cos_phi * np.cos(theta_mid),
            cos_phi * np.sin(theta_mid),
            np.sin(phi_mid),
        )
    )


def body_fixed_basis(
    position_km: np.ndarray,
    velocity_kms: np.ndarray,
) -> np.ndarray:
    """
    MEM planet-origin body-fixed basis in trajectory coordinates.

    +x is velocity/ram, +y is angular momentum, and +z = +x cross +y.
    """
    r = np.asarray(position_km, dtype=float)
    v = np.asarray(velocity_kms, dtype=float)
    v_norm = float(np.linalg.norm(v))
    if v_norm <= 0.0:
        raise ValueError("A body-fixed MEM frame requires non-zero velocity.")
    x_hat = v / v_norm
    y_vector = np.cross(r, x_hat)
    y_norm = float(np.linalg.norm(y_vector))
    if y_norm <= 0.0:
        raise ValueError("Could not construct body-fixed frame from parallel r and v.")
    y_hat = y_vector / y_norm
    z_hat = np.cross(x_hat, y_hat)
    z_hat /= np.linalg.norm(z_hat)
    return np.column_stack((x_hat, y_hat, z_hat))


def _rotate_about_x(vectors: np.ndarray, angle_deg: float) -> np.ndarray:
    angle = math.radians(float(angle_deg))
    c, s = math.cos(angle), math.sin(angle)
    rotation = np.array(
        [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]],
        dtype=float,
    )
    return np.asarray(vectors, dtype=float) @ rotation.T


def convert_inertial_axes(
    vectors: np.ndarray,
    output_axes: str,
    trajectory_axes: str,
) -> np.ndarray:
    if output_axes == trajectory_axes:
        return np.asarray(vectors, dtype=float)
    obliquity_deg = 23.439291111
    if output_axes == "ecliptic" and trajectory_axes == "equatorial":
        return _rotate_about_x(vectors, obliquity_deg)
    if output_axes == "equatorial" and trajectory_axes == "ecliptic":
        return _rotate_about_x(vectors, -obliquity_deg)
    raise ValueError(
        f"Unsupported inertial-axis conversion: {output_axes} -> {trajectory_axes}"
    )


def detect_flux_output_axes(mem_directory: str | Path) -> str | None:
    """Try to read the selected output axes from MEM's options.txt."""
    root = Path(mem_directory)
    candidates = [root / "options.txt"] + sorted(root.rglob("options.txt"))
    seen: set[Path] = set()
    for path in candidates:
        if path in seen or not path.exists():
            continue
        seen.add(path)
        content = path.read_text(encoding="utf-8", errors="replace").lower()
        relevant_lines = [
            line for line in content.splitlines()
            if "output" in line and ("axes" in line or "frame" in line)
        ]
        relevant = "\n".join(relevant_lines) if relevant_lines else content
        if "body-fixed" in relevant or "body fixed" in relevant:
            return "body-fixed"
        if "equatorial" in relevant:
            return "equatorial"
        if "ecliptic" in relevant:
            return "ecliptic"
    return None


def state_weights(
    states: list[TrajectoryState],
    mode: str,
) -> dict[int, float]:
    if not states:
        raise ValueError("No trajectory states were supplied.")
    if mode == "equal" or len(states) == 1:
        value = 1.0 / len(states)
        return {state.index: value for state in states}
    if mode != "time":
        raise ValueError(f"Unknown state-weighting mode: {mode}")

    ordered = sorted(states, key=lambda state: state.julian_date)
    times = np.asarray([state.julian_date for state in ordered], dtype=float)
    dt = np.diff(times)
    if np.any(dt <= 0.0):
        raise ValueError("Time weighting requires strictly increasing Julian dates.")
    closure = float(np.median(dt))
    previous = np.concatenate(([closure], dt))
    following = np.concatenate((dt, [closure]))
    widths = 0.5 * (previous + following)
    widths /= np.sum(widths)
    return {
        state.index: float(weight)
        for state, weight in zip(ordered, widths)
    }


def detailed_surface_factor(
    radiant_vectors: np.ndarray,
    basis: np.ndarray,
    mode: str,
    selected_direction: str,
) -> np.ndarray:
    """Projected-area factor for the requested plane/cube interpretation."""
    components = np.asarray(radiant_vectors, dtype=float) @ np.asarray(basis, dtype=float)
    positive = np.clip(components, 0.0, None)
    negative = np.clip(-components, 0.0, None)

    face_map = {
        "+x ram": positive[:, 0],
        "-x wake": negative[:, 0],
        "+y port": positive[:, 1],
        "-y starboard": negative[:, 1],
        "+z zenith": positive[:, 2],
        "-z nadir": negative[:, 2],
    }

    if mode == "zenith":
        return face_map["+z zenith"]
    if mode == "selected-direction":
        if selected_direction not in face_map:
            raise ValueError(
                "Detailed flux_N mode supports only the six body-fixed cube faces: "
                + ", ".join(face_map)
            )
        return face_map[selected_direction]
    if mode == "six-face-sum":
        return np.sum(np.column_stack(list(face_map.values())), axis=1)
    if mode == "six-face-mean":
        return np.mean(np.column_stack(list(face_map.values())), axis=1)
    if mode == "cross-sectional-normalized":
        return np.ones(len(radiant_vectors), dtype=float)
    raise ValueError(f"Unsupported flux mode: {mode}")


def transform_flux_files_to_mars_frame(
    mem_directory: str | Path,
    trajectory_file: str | Path | None,
    output_axes: str,
    trajectory_axes: str,
    speed_label_mode: str,
    state_weighting_mode: str,
    flux_mode: str,
    selected_direction: str,
    mars_speed_bin_width_kms: float,
    mars_radius_km: float,
    expected_sampling_altitude_km: float,
    sampling_altitude_tolerance_km: float,
    skip_sampling_altitude_check: bool,
    flux_frame_scaling: str,
) -> DetailedFluxResult:
    """
    Remove the velocity of the fictitious MEM sampling observer.

    MEM reports speed and flux relative to each input state vector. Here the
    input trajectory is assumed to be a fictitious Mars orbiter sampling the
    meteoroid environment near the atmospheric reference altitude (normally
    100 km). For each angular-speed cell:

        v_mars = v_rel_to_sampler + v_sampler

    The recovered ``v_mars`` is already the meteoroid speed at that same
    100-km sampling position. The real camera altitude is deliberately not
    used in this transformation.

    ``flux_frame_scaling='number-density'`` converts the directional flux to
    the stationary Mars frame using F = n v before projecting it onto the
    selected local plane. ``preserve-mem-flux`` changes only the speed/radiant
    assignment and keeps the original MEM cell flux amplitude as a diagnostic
    approximation.
    """
    root = Path(mem_directory).resolve()
    trajectory_path = discover_trajectory_file(root, trajectory_file)
    states = read_trajectory_file(trajectory_path)
    states_by_index = {state.index: state for state in states}
    pairs = discover_flux_pairs(root)

    missing_states = [
        index for index, _, _ in pairs if index not in states_by_index
    ]
    if missing_states:
        raise ValueError(
            f"Flux files have no matching trajectory rows for {missing_states[:10]}."
        )

    selected_states = [states_by_index[index] for index, _, _ in pairs]
    weights = state_weights(selected_states, state_weighting_mode)

    resolved_axes = output_axes
    if resolved_axes == "auto":
        detected_axes = detect_flux_output_axes(root)
        resolved_axes = detected_axes or "body-fixed"
        if detected_axes is None:
            print(
                "Could not determine output axes from options.txt; assuming body-fixed."
            )

    if mars_speed_bin_width_kms <= 0.0:
        raise ValueError("--mars-speed-bin-width-kms must be positive.")
    if flux_frame_scaling not in {"number-density", "preserve-mem-flux"}:
        raise ValueError(
            "--flux-frame-scaling must be 'number-density' or 'preserve-mem-flux'."
        )

    sampling_altitudes = np.asarray(
        [np.linalg.norm(state.position_km) - float(mars_radius_km)
         for state in selected_states],
        dtype=float,
    )
    if np.any(~np.isfinite(sampling_altitudes)):
        raise ValueError("Could not determine the MEM sampling altitude from input.txt.")

    altitude_min = float(np.min(sampling_altitudes))
    altitude_max = float(np.max(sampling_altitudes))
    altitude_mean = float(np.mean(sampling_altitudes))
    max_altitude_error = float(
        np.max(np.abs(sampling_altitudes - float(expected_sampling_altitude_km)))
    )
    if (
        not skip_sampling_altitude_check
        and max_altitude_error > float(sampling_altitude_tolerance_km)
    ):
        raise ValueError(
            "The selected MEM input trajectory is not the expected atmospheric "
            f"sampling orbit. Expected {expected_sampling_altitude_km:.3f} ± "
            f"{sampling_altitude_tolerance_km:.3f} km, but the matched flux_N "
            f"states span {altitude_min:.3f} to {altitude_max:.3f} km "
            f"(mean {altitude_mean:.3f} km). The 5720-km camera trajectory must "
            "not be used for this frame correction; select the input.txt used for "
            "the fictitious ~100-km MEM sampling run."
        )

    transformed_histograms: list[tuple[float, np.ndarray]] = []
    raw_histograms: list[tuple[float, np.ndarray]] = []
    state_rows: list[dict[str, Any]] = []
    all_flux_files: list[str] = []
    common_raw_speeds: np.ndarray | None = None
    common_shift: float | None = None
    selected_surface = "+z zenith" if flux_mode == "zenith" else selected_direction

    for state_index, high_path, low_path in pairs:
        state = states_by_index[state_index]
        high = read_mem_flux_file(high_path, speed_label_mode)
        low = read_mem_flux_file(low_path, speed_label_mode)
        _assert_matching_flux_grids(high, low)
        all_flux_files.extend([str(high.path), str(low.path)])

        if common_raw_speeds is None:
            common_raw_speeds = high.speed_midpoints_kms.copy()
            common_shift = float(high.speed_label_shift_kms)
        elif not np.allclose(common_raw_speeds, high.speed_midpoints_kms):
            raise ValueError("Speed grids differ between flux_N files.")

        # MEM values are integrated fluxes per angular and speed cell.
        cell_flux = np.asarray(high.flux_m2_yr + low.flux_m2_yr, dtype=float)
        radiants_output = angular_bin_radiants(high)
        basis = body_fixed_basis(state.position_km, state.velocity_kms)

        if resolved_axes == "body-fixed":
            radiants_trajectory = radiants_output @ basis.T
        else:
            radiants_trajectory = convert_inertial_axes(
                radiants_output,
                resolved_axes,
                trajectory_axes,
            )

        r_sc = float(np.linalg.norm(state.position_km))
        sampling_altitude_km = r_sc - float(mars_radius_km)
        v_sampler = np.asarray(state.velocity_kms, dtype=float)
        speed_rel = np.asarray(high.speed_midpoints_kms, dtype=float)

        # Diagnostic: what MEM reports on the chosen face before removing the
        # fictitious observer velocity.
        raw_factor = detailed_surface_factor(
            radiants_trajectory,
            basis,
            flux_mode,
            selected_direction,
        )
        raw_by_speed = np.sum(cell_flux * raw_factor[:, None], axis=0)
        raw_histograms.append((weights[state_index], raw_by_speed))

        # The angular coordinates specify radiants. Physical velocity is opposite
        # to the radiant. Add the fictitious observer velocity to recover the
        # Mars-centred meteoroid velocity at this same ~100-km position.
        v_rel_vectors = -radiants_trajectory[:, None, :] * speed_rel[None, :, None]
        v_mars_vectors = v_rel_vectors + v_sampler[None, None, :]
        speed_mars = np.linalg.norm(v_mars_vectors, axis=2)

        radiant_mars = np.divide(
            -v_mars_vectors,
            speed_mars[:, :, None],
            out=np.zeros_like(v_mars_vectors),
            where=speed_mars[:, :, None] > 0.0,
        )
        stationary_surface_factor = detailed_surface_factor(
            radiant_mars.reshape(-1, 3),
            basis,
            flux_mode,
            selected_direction,
        ).reshape(speed_mars.shape)

        if flux_frame_scaling == "number-density":
            # MEM directional flux is relative to the moving fictitious observer.
            # Preserve the inferred directional number density n = F/v, then form
            # the stationary-frame flux n*v_mars.
            frame_ratio = np.divide(
                speed_mars,
                speed_rel[None, :],
                out=np.zeros_like(speed_mars),
                where=speed_rel[None, :] > 0.0,
            )
        else:
            # Diagnostic approximation: only move cells to the Mars-relative
            # speed/radiant while preserving their MEM amplitudes.
            frame_ratio = np.ones_like(speed_mars)

        transformed_cell_flux = (
            cell_flux * frame_ratio * stationary_surface_factor
        )

        # No propagation from the real camera orbit is performed. The recovered
        # speed is already the Mars-relative meteoroid speed at the MEM sampling
        # position near 100 km.
        bin_index = np.floor(
            speed_mars / float(mars_speed_bin_width_kms)
        ).astype(int)
        valid = (
            np.isfinite(speed_mars)
            & np.isfinite(transformed_cell_flux)
            & (transformed_cell_flux > 0.0)
            & (bin_index >= 0)
        )
        histogram = (
            np.bincount(
                bin_index[valid].ravel(),
                weights=transformed_cell_flux[valid].ravel(),
            )
            if np.any(valid)
            else np.zeros(1, dtype=float)
        )
        transformed_histograms.append((weights[state_index], histogram))

        transformed_total = float(np.sum(transformed_cell_flux))
        weighted_speed = (
            float(np.sum(speed_mars * transformed_cell_flux) / transformed_total)
            if transformed_total > 0.0
            else np.nan
        )
        state_rows.append(
            {
                "state_index": state_index,
                "julian_date": state.julian_date,
                "sampling_radius_km": r_sc,
                "sampling_altitude_km": sampling_altitude_km,
                "fictitious_sampler_speed_kms": float(np.linalg.norm(v_sampler)),
                "raw_mem_surface_flux_m2_yr": float(np.sum(raw_by_speed)),
                "stationary_mars_surface_flux_m2_yr": transformed_total,
                "stationary_flux_weighted_speed_kms": weighted_speed,
                "state_weight": float(weights[state_index]),
            }
        )

    max_bins = max(len(histogram) for _, histogram in transformed_histograms)
    transformed_average = np.zeros(max_bins, dtype=float)
    for weight, histogram in transformed_histograms:
        transformed_average[:len(histogram)] += float(weight) * histogram

    if common_raw_speeds is None:
        raise ValueError("No detailed flux files were processed.")
    raw_average = np.zeros_like(common_raw_speeds, dtype=float)
    for weight, histogram in raw_histograms:
        raw_average += float(weight) * histogram

    mars_midpoints = (
        np.arange(max_bins, dtype=float) + 0.5
    ) * float(mars_speed_bin_width_kms)

    return DetailedFluxResult(
        mars_speed_midpoints_kms=mars_midpoints,
        transformed_speed_flux_m2_yr=transformed_average,
        spacecraft_speed_midpoints_kms=common_raw_speeds,
        spacecraft_surface_flux_m2_yr=raw_average,
        state_rows=state_rows,
        flux_files=all_flux_files,
        trajectory_file=str(trajectory_path),
        output_axes=resolved_axes,
        speed_label_shift_kms=float(common_shift or 0.0),
        state_weighting=state_weighting_mode,
        sampling_altitude_mean_km=altitude_mean,
        sampling_altitude_min_km=altitude_min,
        sampling_altitude_max_km=altitude_max,
        flux_frame_scaling=flux_frame_scaling,
        selected_surface=selected_surface,
    )


def save_state_transform_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    fields = [
        "state_index",
        "julian_date",
        "sampling_radius_km",
        "sampling_altitude_km",
        "fictitious_sampler_speed_kms",
        "raw_mem_surface_flux_m2_yr",
        "stationary_mars_surface_flux_m2_yr",
        "stationary_flux_weighted_speed_kms",
        "state_weight",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_directions(text: str) -> list[str]:
    requested = [part.strip() for part in text.split(",") if part.strip()]
    if not requested:
        raise ValueError("At least one MEM direction must be selected.")

    canonical = {name.lower(): name for name in DIRECTIONS}
    directions = []
    for item in requested:
        key = item.lower()
        if key not in canonical:
            raise ValueError(
                f"Unknown direction {item!r}. Valid values are: {', '.join(DIRECTIONS)}"
            )
        directions.append(canonical[key])
    return directions


def reference_speed_flux(
    direction_flux: dict[str, np.ndarray],
    directions: list[str],
    total_cross_sectional_flux: float,
    mode: str,
    selected_direction: str = "+z zenith",
) -> tuple[np.ndarray, dict[str, np.ndarray | float | str]]:
    matrix = np.vstack([np.asarray(direction_flux[name], dtype=float) for name in directions])
    face_sum = np.sum(matrix, axis=0)
    face_mean = np.mean(matrix, axis=0)

    canonical = {name.lower(): name for name in DIRECTIONS}
    selected_key = str(selected_direction).strip().lower()
    if selected_key not in canonical:
        raise ValueError(
            f"Unknown selected direction {selected_direction!r}. "
            f"Valid values are: {', '.join(DIRECTIONS)}"
        )
    selected_direction = canonical[selected_key]
    individual_direction_flux = np.asarray(
        direction_flux[selected_direction], dtype=float
    )

    diagnostics: dict[str, np.ndarray | float | str] = {
        "six_face_or_selected_sum_speed_flux": face_sum,
        "six_face_or_selected_mean_speed_flux": face_mean,
        "selected_direction_speed_flux": individual_direction_flux,
        "selected_direction_name": selected_direction,
        "selected_direction_sum_total": float(np.sum(face_sum)),
        "selected_direction_mean_total": float(np.sum(face_mean)),
        "individual_direction_total": float(np.sum(individual_direction_flux)),
        "total_cross_sectional_flux": float(total_cross_sectional_flux),
    }

    if mode == "zenith":
        selected_direction = "+z zenith"
        selected = np.asarray(direction_flux[selected_direction], dtype=float)
        diagnostics["selected_direction_name"] = selected_direction
        diagnostics["selected_direction_speed_flux"] = selected
        diagnostics["individual_direction_total"] = float(np.sum(selected))
    elif mode == "selected-direction":
        selected = individual_direction_flux
    elif mode == "six-face-sum":
        selected = face_sum
    elif mode == "six-face-mean":
        selected = face_mean
    elif mode == "cross-sectional-normalized":
        if not np.isfinite(total_cross_sectional_flux) or total_cross_sectional_flux <= 0.0:
            raise ValueError(
                "The cube headers contain no valid total cross-sectional flux, so "
                "--flux-mode cross-sectional-normalized cannot be used."
            )
        shape_total = float(np.sum(face_sum))
        if shape_total <= 0.0:
            raise ValueError("The selected direction flux has zero total flux.")
        selected = face_sum/shape_total*float(total_cross_sectional_flux)
    else:
        raise ValueError(f"Unsupported flux mode: {mode}")

    diagnostics["reference_speed_flux"] = selected
    diagnostics["reference_speed_flux_total"] = float(np.sum(selected))
    return selected, diagnostics


# -----------------------------------------------------------------------------
# Camera likelihood model loading/fitting/prediction
# -----------------------------------------------------------------------------

def _sigmoid(values: np.ndarray | float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return 1.0/(1.0 + np.exp(-np.clip(values, -60.0, 60.0)))


def _fit_regularized_camera_logistic(
    mass_kg: np.ndarray,
    speed_kms: np.ndarray,
    detected: np.ndarray,
    l2_penalty: float = 1.0e-3,
    max_iterations: int = 100,
    tolerance: float = 1.0e-9,
) -> dict[str, Any]:
    mass = np.asarray(mass_kg, dtype=float)
    speed = np.asarray(speed_kms, dtype=float)
    y = np.asarray(detected, dtype=float)

    valid = (
        np.isfinite(mass) & (mass > 0.0)
        & np.isfinite(speed) & (speed > 0.0)
        & np.isfinite(y)
    )
    mass = mass[valid]
    speed = speed[valid]
    y = y[valid]

    if y.size < 6 or np.unique(y).size < 2:
        raise ValueError(
            "The detection CSV needs at least six valid rows and both detected "
            "and non-detected cases."
        )

    raw = np.column_stack([np.log10(mass), np.log10(speed)])
    center = np.mean(raw, axis=0)
    scale = np.std(raw, axis=0)
    scale = np.where(scale > np.finfo(float).eps, scale, 1.0)
    design = np.column_stack([np.ones(len(raw)), (raw - center)/scale])

    mean_detection = float(np.clip(np.mean(y), 1.0e-6, 1.0 - 1.0e-6))
    coefficients = np.zeros(3, dtype=float)
    coefficients[0] = math.log(mean_detection/(1.0 - mean_detection))
    penalty = np.diag([0.0, float(l2_penalty), float(l2_penalty)])

    converged = False
    iterations = 0
    for iterations in range(1, int(max_iterations) + 1):
        probability = _sigmoid(design @ coefficients)
        weights = np.clip(probability*(1.0 - probability), 1.0e-7, None)
        gradient = design.T @ (y - probability) - penalty @ coefficients
        information = design.T @ (weights[:, None]*design) + penalty
        try:
            step = np.linalg.solve(information, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(information, gradient, rcond=None)[0]
        coefficients += step
        if np.max(np.abs(step)) < float(tolerance):
            converged = True
            break

    fitted = _sigmoid(design @ coefficients)
    eps = 1.0e-12
    log_loss = float(-np.mean(
        y*np.log(np.clip(fitted, eps, 1.0))
        + (1.0 - y)*np.log(np.clip(1.0 - fitted, eps, 1.0))
    ))

    return {
        "format_version": 2,
        "model_type": "camera_detection_logistic_from_csv",
        "camera_logistic_model": {
            "coefficients": coefficients.tolist(),
            "feature_center": center.tolist(),
            "feature_scale": scale.tolist(),
            "l2_penalty": float(l2_penalty),
            "iterations": int(iterations),
            "converged": bool(converged),
            "training_log_loss": log_loss,
        },
        "sample_count": int(len(y)),
        "detected_sample_count": int(np.count_nonzero(y)),
        "mass_range_kg": [float(np.min(mass)), float(np.max(mass))],
        "speed_range_kms": [float(np.min(speed)), float(np.max(speed))],
    }


def fit_camera_model_from_csv(path: str | Path, minimum_frames: int = 10) -> dict[str, Any]:
    mass: list[float] = []
    speed: list[float] = []
    detected: list[float] = []

    with Path(path).open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        fields = set(reader.fieldnames or [])
        required = {"initial_mass_kg", "mars_velocity_kms"}
        if not required.issubset(fields):
            raise ValueError(
                f"Camera CSV {path} is missing columns: {sorted(required - fields)}"
            )

        for row in reader:
            try:
                m = float(row["initial_mass_kg"])
                v = float(row["mars_velocity_kms"])
            except (TypeError, ValueError):
                continue

            label: bool | None = None
            if "is_camera_detected" in row and str(row["is_camera_detected"]).strip() != "":
                label = str(row["is_camera_detected"]).strip().lower() in {"1", "true", "yes"}
            elif "detected_frame_count" in row:
                try:
                    label = float(row["detected_frame_count"]) >= int(minimum_frames)
                except (TypeError, ValueError):
                    label = None
            elif "is_physically_detected" in row and str(row["is_physically_detected"]).strip() != "":
                label = str(row["is_physically_detected"]).strip().lower() in {"1", "true", "yes"}

            if label is None:
                continue
            mass.append(m)
            speed.append(v)
            detected.append(float(label))

    model = _fit_regularized_camera_logistic(
        np.asarray(mass), np.asarray(speed), np.asarray(detected)
    )
    model["minimum_detected_frames"] = int(minimum_frames)
    model["camera_detection_definition"] = (
        f"at least {int(minimum_frames)} sampled frames above the camera limiting magnitude"
    )
    model["source_csv"] = str(Path(path).resolve())
    return model


def unwrap_camera_model(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("The camera-model pickle does not contain a dictionary.")

    for key in (
        "synthetic_detection_likelihood_model",
        "detection_likelihood_model",
        "camera_detection_likelihood_model",
    ):
        value = payload.get(key)
        if isinstance(value, dict):
            return value

    if "camera_logistic_model" in payload or (
        "intercept" in payload and "b_log10_mass" in payload and "b_log10_speed" in payload
    ):
        return payload

    raise ValueError("Could not find a compatible camera likelihood model in the pickle.")


def load_camera_model(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as fh:
        return unwrap_camera_model(pickle.load(fh))


def discover_camera_model(search_directory: str | Path) -> tuple[str, Path] | None:
    root = Path(search_directory)
    preferred_pickles = [
        "Mars_detection_synthetic_likelihood_model.pkl",
        "Mars_detection_likelihood_model.pkl",
    ]
    for name in preferred_pickles:
        matches = sorted(root.rglob(name))
        if matches:
            return "pickle", matches[0]

    preferred_csvs = [
        "Mars_detection_synthetic_with_likelihood.csv",
        "Mars_detection_synthetic_speed_sweep.csv",
        "Mars_detection_velocity_mass_with_likelihood.csv",
        "Mars_detection_velocity_mass_summary.csv",
    ]
    for name in preferred_csvs:
        matches = sorted(root.rglob(name))
        if matches:
            return "csv", matches[0]
    return None


def manual_proxy_model(
    exponent_x: float,
    reference_mass_kg: float,
    reference_speed_kms: float,
    log_width_dex: float,
    minimum_frames: int,
) -> dict[str, Any]:
    if reference_mass_kg <= 0.0 or reference_speed_kms <= 0.0:
        raise ValueError("Manual proxy reference mass and speed must be positive.")
    threshold = reference_mass_kg*reference_speed_kms**float(exponent_x)
    return {
        "format_version": 1,
        "model_type": "manual_mass_speed_proxy",
        "proxy_exponent_x": float(exponent_x),
        "proxy_threshold_50": float(threshold),
        "proxy_log_width_dex": float(log_width_dex),
        "minimum_detected_frames": int(minimum_frames),
        "camera_detection_definition": (
            f"manual 50% boundary at m={reference_mass_kg:g} kg, "
            f"v={reference_speed_kms:g} km/s; minimum {minimum_frames} frames"
        ),
    }


def camera_proxy_parameters(model: dict[str, Any]) -> dict[str, float]:
    logistic = model.get("camera_logistic_model")
    if isinstance(logistic, dict):
        coefficients = np.asarray(logistic["coefficients"], dtype=float)
        center = np.asarray(logistic["feature_center"], dtype=float)
        scale = np.asarray(logistic["feature_scale"], dtype=float)
        b_mass = coefficients[1]/scale[0]
        b_speed = coefficients[2]/scale[1]
        intercept = coefficients[0] - coefficients[1]*center[0]/scale[0] - coefficients[2]*center[1]/scale[1]
        if np.isfinite(b_mass) and abs(b_mass) > np.finfo(float).eps:
            gamma = b_speed/b_mass
            threshold = 10.0**(-intercept/b_mass)
            return {
                "speed_exponent_x": float(gamma),
                "proxy_threshold_50": float(threshold),
            }

    if all(key in model for key in ("intercept", "b_log10_mass", "b_log10_speed")):
        b_mass = float(model["b_log10_mass"])
        b_speed = float(model["b_log10_speed"])
        lm = float(model.get("representative_limiting_magnitude", 4.0))
        if abs(b_mass) > np.finfo(float).eps:
            gamma = b_speed/b_mass
            threshold = 10.0**((lm - float(model["intercept"]))/b_mass)
            return {
                "speed_exponent_x": float(gamma),
                "proxy_threshold_50": float(threshold),
            }

    if model.get("model_type") == "manual_mass_speed_proxy":
        return {
            "speed_exponent_x": float(model["proxy_exponent_x"]),
            "proxy_threshold_50": float(model["proxy_threshold_50"]),
        }

    return {"speed_exponent_x": np.nan, "proxy_threshold_50": np.nan}


def predict_camera_probability(
    mass_kg: np.ndarray | float,
    speed_kms: np.ndarray | float,
    model: dict[str, Any],
    extrapolation: str = "clip",
) -> np.ndarray:
    mass = np.asarray(mass_kg, dtype=float)
    speed = np.asarray(speed_kms, dtype=float)
    mass, speed = np.broadcast_arrays(mass, speed)

    valid = (
        np.isfinite(mass) & (mass > 0.0)
        & np.isfinite(speed) & (speed > 0.0)
    )

    eval_mass = mass.copy()
    eval_speed = speed.copy()
    outside = np.zeros(mass.shape, dtype=bool)

    mass_range = model.get("mass_range_kg")
    speed_range = model.get("speed_range_kms")
    if isinstance(mass_range, (list, tuple)) and len(mass_range) == 2:
        low, high = map(float, mass_range)
        outside |= (mass < low) | (mass > high)
        if extrapolation == "clip":
            eval_mass = np.clip(eval_mass, low, high)
    if isinstance(speed_range, (list, tuple)) and len(speed_range) == 2:
        low, high = map(float, speed_range)
        outside |= (speed < low) | (speed > high)
        if extrapolation == "clip":
            eval_speed = np.clip(eval_speed, low, high)

    logistic = model.get("camera_logistic_model")
    if isinstance(logistic, dict):
        coefficients = np.asarray(logistic["coefficients"], dtype=float)
        center = np.asarray(logistic["feature_center"], dtype=float)
        scale = np.asarray(logistic["feature_scale"], dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            standardized_mass = (np.log10(eval_mass) - center[0])/scale[0]
            standardized_speed = (np.log10(eval_speed) - center[1])/scale[1]
            linear = (
                coefficients[0]
                + coefficients[1]*standardized_mass
                + coefficients[2]*standardized_speed
            )
        probability = _sigmoid(linear)
    elif all(key in model for key in ("intercept", "b_log10_mass", "b_log10_speed")):
        lm = float(model.get("representative_limiting_magnitude", 4.0))
        sigma = max(float(model.get("residual_sigma_mag", 0.5)), np.finfo(float).eps)
        with np.errstate(divide="ignore", invalid="ignore"):
            predicted_mag = (
                float(model["intercept"])
                + float(model["b_log10_mass"])*np.log10(eval_mass)
                + float(model["b_log10_speed"])*np.log10(eval_speed)
            )
        z = (lm - predicted_mag)/sigma
        probability = 0.5*(1.0 + np.vectorize(math.erf)(z/np.sqrt(2.0)))
    elif model.get("model_type") == "manual_mass_speed_proxy":
        exponent = float(model["proxy_exponent_x"])
        threshold = float(model["proxy_threshold_50"])
        width = max(float(model.get("proxy_log_width_dex", 0.2)), 1.0e-6)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_proxy_offset = np.log10(eval_mass*eval_speed**exponent/threshold)
        probability = _sigmoid(log_proxy_offset/width)
    else:
        raise ValueError("Unsupported camera likelihood model structure.")

    probability = np.asarray(probability, dtype=float)
    if extrapolation == "mask":
        valid &= ~outside
    probability = np.where(valid, probability, np.nan)
    return np.asarray(np.clip(probability, 0.0, 1.0), dtype=float)


# -----------------------------------------------------------------------------
# Flux-grid integration
# -----------------------------------------------------------------------------

def parse_mass_edges(text: str) -> np.ndarray:
    values = [float(token) for token in re.split(r"[,;\s]+", text.strip()) if token]
    edges = np.unique(np.asarray(values, dtype=float))
    if len(edges) < 2 or np.any(~np.isfinite(edges)) or np.any(edges <= 0.0):
        raise ValueError("Mass edges must contain at least two positive finite values.")
    edges.sort()
    return edges


def integrate_mass_speed_flux(
    reference_speed_flux_m2_yr: np.ndarray,
    speeds_kms: np.ndarray,
    mass_edges_g: np.ndarray,
    reference_mass_g: float,
    camera_model: dict[str, Any],
    substeps_per_mass_bin: int = 80,
    extrapolation: str = "clip",
) -> dict[str, np.ndarray]:
    speed = np.asarray(speeds_kms, dtype=float)
    reference_flux = np.asarray(reference_speed_flux_m2_yr, dtype=float)
    edges_g = np.asarray(mass_edges_g, dtype=float)

    n_mass = len(edges_g) - 1
    n_speed = len(speed)
    incident = np.zeros((n_mass, n_speed), dtype=float)
    detectable = np.zeros((n_mass, n_speed), dtype=float)
    mean_probability = np.full((n_mass, n_speed), np.nan, dtype=float)
    representative_mass_g = np.sqrt(edges_g[:-1]*edges_g[1:])

    g_reference = float(grun_cumulative_flux(reference_mass_g))
    if not np.isfinite(g_reference) or g_reference <= 0.0:
        raise ValueError("Invalid Grün value at the MEM reference mass.")

    for mass_index, (low_g, high_g) in enumerate(zip(edges_g[:-1], edges_g[1:])):
        sub_edges_g = np.geomspace(low_g, high_g, int(substeps_per_mass_bin) + 1)
        sub_mass_g = np.sqrt(sub_edges_g[:-1]*sub_edges_g[1:])
        sub_fraction = (
            grun_cumulative_flux(sub_edges_g[:-1])
            - grun_cumulative_flux(sub_edges_g[1:])
        )/g_reference
        sub_fraction = np.clip(np.asarray(sub_fraction, dtype=float), 0.0, None)

        probability = predict_camera_probability(
            mass_kg=sub_mass_g[:, None]/1000.0,
            speed_kms=speed[None, :],
            model=camera_model,
            extrapolation=extrapolation,
        )

        # For mask mode, unavailable probability does not contribute to detected
        # flux. The excluded incident flux is reported separately in the summary.
        finite_probability = np.where(np.isfinite(probability), probability, 0.0)
        incident_fraction = float(np.sum(sub_fraction))
        detected_fraction_by_speed = np.sum(
            sub_fraction[:, None]*finite_probability,
            axis=0,
        )

        incident[mass_index, :] = reference_flux*incident_fraction
        detectable[mass_index, :] = reference_flux*detected_fraction_by_speed
        if incident_fraction > 0.0:
            mean_probability[mass_index, :] = detected_fraction_by_speed/incident_fraction

    return {
        "incident_flux": incident,
        "detectable_flux": detectable,
        "mean_detection_probability": mean_probability,
        "representative_mass_g": representative_mass_g,
    }


def midpoint_edges(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 1:
        return np.asarray([values[0] - 0.5, values[0] + 0.5])
    mid = 0.5*(values[:-1] + values[1:])
    first = values[0] - (mid[0] - values[0])
    last = values[-1] + (values[-1] - mid[-1])
    return np.concatenate(([first], mid, [last]))


# -----------------------------------------------------------------------------
# Output functions
# -----------------------------------------------------------------------------

def save_reference_speed_plot(
    path: Path,
    speeds: np.ndarray,
    selected_flux: np.ndarray,
    diagnostics: dict[str, Any],
    reference_mass_g: float,
    flux_mode: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    positive = selected_flux > 0.0
    ax.plot(
        speeds[positive],
        selected_flux[positive],
        marker="o",
        linewidth=2.0,
        markersize=4,
        label=(
            f"Flux used ({flux_mode}), total={np.sum(selected_flux):.6e} #/m²/yr"
        ),
    )

    if (
        "comparison_speed_kms" in diagnostics
        and "comparison_flux_m2_yr" in diagnostics
    ):
        comparison_speed = np.asarray(
            diagnostics["comparison_speed_kms"], dtype=float
        )
        comparison_flux = np.asarray(
            diagnostics["comparison_flux_m2_yr"], dtype=float
        )
        mask = comparison_flux > 0.0
        ax.plot(
            comparison_speed[mask],
            comparison_flux[mask],
            marker=".",
            linestyle="--",
            label=(
                "Original spacecraft-relative surface flux, "
                f"total={np.sum(comparison_flux):.6e} #/m²/yr"
            ),
        )
        ax.set_title(
            f"MEM flux transformed to Mars-relative speed for m ≥ {reference_mass_g:g} g"
        )
    else:
        raw = np.asarray(
            diagnostics.get(
                "six_face_or_selected_sum_speed_flux",
                selected_flux,
            ),
            dtype=float,
        )
        raw_positive = raw > 0.0
        if flux_mode not in {"six-face-sum", "zenith", "selected-direction"}:
            ax.plot(
                speeds[raw_positive],
                raw[raw_positive],
                marker=".",
                linestyle="--",
                label=(
                    "Selected cube-face sum, "
                    f"total={np.sum(raw):.6e} #/m²/yr"
                ),
            )
        if flux_mode in {"zenith", "selected-direction"}:
            direction_name = str(
                diagnostics.get("selected_direction_name", "+z zenith")
            )
            ax.set_title(
                f"MEM {direction_name} flux for m ≥ {reference_mass_g:g} g"
            )
        else:
            ax.set_title(
                f"MEM reference speed distribution for m ≥ {reference_mass_g:g} g"
            )

    ax.set_yscale("log")
    ax.set_xlabel(
        str(diagnostics.get("speed_axis_label", "Speed [km/s]"))
    )
    ax.set_ylabel("Cumulative flux per speed bin [#/m²/yr]")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_grun_plot(path: Path, mass_edges_g: np.ndarray, reference_mass_g: float) -> None:
    minimum = min(float(reference_mass_g), float(np.min(mass_edges_g)))
    maximum = max(float(reference_mass_g), float(np.max(mass_edges_g)))
    mass = np.geomspace(minimum, maximum, 600)
    ratio = grun_scale_ratio(mass, reference_mass_g)

    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.plot(mass, ratio, linewidth=2.0)
    ax.scatter(mass_edges_g, grun_scale_ratio(mass_edges_g, reference_mass_g), s=38)
    for value in mass_edges_g:
        ax.annotate(
            f"{value:g} g",
            (value, float(grun_scale_ratio(value, reference_mass_g))),
            xytext=(4, 5), textcoords="offset points", fontsize=8,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Limiting mass [g]")
    ax.set_ylabel(f"Cumulative flux ratio g(m)/g({reference_mass_g:g} g)")
    ax.set_title("Grün cumulative mass scaling used by MEM 3")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _positive_lognorm(data: np.ndarray) -> LogNorm | None:
    positive = np.asarray(data, dtype=float)
    positive = positive[np.isfinite(positive) & (positive > 0.0)]
    if positive.size == 0:
        return None
    return LogNorm(vmin=float(np.min(positive)), vmax=float(np.max(positive)))


def save_flux_heatmap(
    path: Path,
    speeds: np.ndarray,
    mass_edges_g: np.ndarray,
    data: np.ndarray,
    title: str,
    colorbar_label: str,
    probability: np.ndarray | None = None,
    visible_probability_threshold: float | None = None,
) -> None:
    speed_edges = midpoint_edges(speeds)
    mass_edges_kg = mass_edges_g/1000.0
    plotted = np.asarray(data, dtype=float).copy()

    if probability is not None and visible_probability_threshold is not None:
        plotted[np.asarray(probability) < float(visible_probability_threshold)] = np.nan

    plotted = np.ma.masked_invalid(plotted)
    plotted = np.ma.masked_less_equal(plotted, 0.0)
    norm = _positive_lognorm(plotted.filled(np.nan))

    fig, ax = plt.subplots(figsize=(11, 7.5))
    mesh = ax.pcolormesh(
        speed_edges,
        mass_edges_kg,
        plotted,
        shading="flat",
        norm=norm,
    )
    colorbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    colorbar.set_label(colorbar_label)

    ax.set_yscale("log")
    ax.set_xlabel("Mars meteoroid speed [km/s]")
    ax.set_ylabel("Initial meteoroid mass [kg]")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_probability_heatmap(
    path: Path,
    speeds: np.ndarray,
    mass_edges_g: np.ndarray,
    probability: np.ndarray,
    threshold: float,
    model: dict[str, Any],
) -> None:
    speed_edges = midpoint_edges(speeds)
    mass_edges_kg = mass_edges_g/1000.0
    parameters = camera_proxy_parameters(model)
    gamma = parameters["speed_exponent_x"]

    fig, ax = plt.subplots(figsize=(11, 7.5))
    mesh = ax.pcolormesh(
        speed_edges,
        mass_edges_kg,
        np.asarray(probability, dtype=float),
        shading="flat",
        vmin=0.0,
        vmax=1.0,
    )
    colorbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    colorbar.set_label("Mean probability of camera detection")

    representative_mass_kg = np.sqrt(mass_edges_kg[:-1]*mass_edges_kg[1:])
    speed_mesh, mass_mesh = np.meshgrid(speeds, representative_mass_kg)
    try:
        ax.contour(
            speed_mesh,
            mass_mesh,
            probability,
            levels=[float(threshold)],
            linewidths=2.0,
        )
    except ValueError:
        pass

    minimum_frames = int(model.get("minimum_detected_frames", 10))
    ax.set_yscale("log")
    ax.set_xlabel("Mars meteoroid speed [km/s]")
    ax.set_ylabel("Initial meteoroid mass [kg]")
    ax.set_title(
        "METEORCAM mass-speed detection probability\n"
        f"visible threshold P ≥ {threshold:.2f}; minimum={minimum_frames} frames; "
        f"proxy approximately m v^{gamma:.2f}"
    )
    ax.grid(True, which="both", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_grid_csv(
    path: Path,
    speeds: np.ndarray,
    mass_edges_g: np.ndarray,
    reference_speed_flux: np.ndarray,
    incident_flux: np.ndarray,
    detectable_flux: np.ndarray,
    probability: np.ndarray,
    visible_threshold: float,
) -> None:
    columns = [
        "mass_bin_low_g",
        "mass_bin_high_g",
        "representative_mass_g",
        "speed_bin_mid_kms",
        "reference_flux_above_mem_mass_m2_yr",
        "incident_flux_in_mass_speed_bin_m2_yr",
        "mean_camera_detection_probability",
        "probability_weighted_detectable_flux_m2_yr",
        "visible_probability_threshold",
        "inside_visible_region",
        "binary_visible_region_flux_m2_yr",
    ]

    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for mass_index, (low_g, high_g) in enumerate(zip(mass_edges_g[:-1], mass_edges_g[1:])):
            representative = math.sqrt(low_g*high_g)
            for speed_index, speed in enumerate(speeds):
                p = float(probability[mass_index, speed_index])
                visible = bool(np.isfinite(p) and p >= visible_threshold)
                incident = float(incident_flux[mass_index, speed_index])
                writer.writerow({
                    "mass_bin_low_g": low_g,
                    "mass_bin_high_g": high_g,
                    "representative_mass_g": representative,
                    "speed_bin_mid_kms": speed,
                    "reference_flux_above_mem_mass_m2_yr": reference_speed_flux[speed_index],
                    "incident_flux_in_mass_speed_bin_m2_yr": incident,
                    "mean_camera_detection_probability": p if np.isfinite(p) else "",
                    "probability_weighted_detectable_flux_m2_yr": detectable_flux[mass_index, speed_index],
                    "visible_probability_threshold": visible_threshold,
                    "inside_visible_region": int(visible),
                    "binary_visible_region_flux_m2_yr": incident if visible else 0.0,
                })


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, dict):
        return {str(key): json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def save_text_summary(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "MEM mass extension and METEORCAM detectable-flux summary",
        "="*62,
        f"Environment source: {summary.get('environment_source', 'cube-average')}",
        f"MEM cube files: {len(summary['mem_cube_files'])}",
        f"MEM detailed flux files: {len(summary.get('mem_flux_files', []))}",
        f"Trajectory file: {summary.get('trajectory_file')}",
        f"Resolved flux output axes: {summary.get('flux_output_axes')}",
        f"Legacy speed-label shift: {summary.get('speed_label_shift_kms', 0.0):.6g} km/s",
        f"State weighting: {summary.get('state_weighting')}",
        f"Mars speed location: {summary.get('mars_speed_location')}",
        f"Reference limiting mass: {summary['reference_mass_g']:.6g} g",
        f"Mass interval integrated: {summary['mass_range_g'][0]:.6g} to {summary['mass_range_g'][1]:.6g} g",
        f"Flux mode: {summary['flux_mode']}",
        f"Selected individual direction: {summary.get('selected_individual_direction', '+z zenith')}",
        f"Directions used only by summed/normalized modes: {', '.join(summary['selected_directions'])}",
        "",
        f"MEM header total cross-sectional flux >= reference mass: {summary['mem_total_cross_sectional_flux_above_reference_m2_yr']:.8e} #/m^2/yr",
        f"Selected individual-direction flux >= reference mass: {summary['individual_direction_flux_above_reference_m2_yr']:.8e} #/m^2/yr",
        f"Selected summed-direction flux >= reference mass: {summary['selected_direction_sum_flux_above_reference_m2_yr']:.8e} #/m^2/yr",
        f"Reference flux used in integration: {summary['reference_flux_used_above_reference_m2_yr']:.8e} #/m^2/yr",
        f"Incident flux inside requested mass range: {summary['incident_flux_in_mass_range_m2_yr']:.8e} #/m^2/yr",
        f"Probability-weighted camera-detectable flux: {summary['camera_detectable_flux_m2_yr']:.8e} #/m^2/yr",
        f"Binary visible-region flux (P >= {summary['visible_probability_threshold']:.3f}): {summary['binary_visible_region_flux_m2_yr']:.8e} #/m^2/yr",
        f"Mean probability-weighted detection efficiency: {summary['mean_detection_efficiency']:.6f}",
        "",
        f"Camera minimum detected frames: {summary['minimum_detected_frames']}",
        f"Equivalent fitted sensitivity proxy: m * v^{summary['camera_proxy_speed_exponent_x']:.6f}",
        f"50% proxy threshold: {summary['camera_proxy_threshold_50']:.8e} kg*(km/s)^x",
    ]

    if summary.get("effective_area_m2") is not None:
        lines.extend([
            "",
            f"Effective collecting area: {summary['effective_area_m2']:.8e} m^2",
            f"Expected incident events per year: {summary['expected_incident_events_per_year']:.8g}",
            f"Expected camera-detectable events per year: {summary['expected_detectable_events_per_year']:.8g}",
            f"Expected camera-detectable events per year in the dark side: {summary['expected_detectable_events_per_year']/2:.8g}",
            f"Expected camera-detectable events per year in the dark side for a Straylight rejection factor of 0.5 : {summary['expected_detectable_events_per_year']/4:.8g}",
        ])

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extend a MEM speed distribution from its reference limiting mass "
            "to larger mass bins using Grün scaling, then apply the fitted "
            "METEORCAM camera-detection surface."
        )
    )
    parser.add_argument(
        "--mem-directory",
        default=r"C:\Users\maxiv\Documents\UWO\Papers\0.5)METEORCAM-Strawman\METEORCAM\MEM-10gFlux-100km\Flux_0-30-60-90deg",
        help=(
            "MEM output directory containing input.txt and HiDensity/LoDensity "
            "flux_N.txt files. The cube_avg.txt pathway remains available."
        ),
    )
    parser.add_argument(
        "--environment-source",
        choices=["detailed-flux", "cube-average"],
        default="detailed-flux",
        help=(
            "Default detailed-flux transforms every flux_N angular-speed cell "
            "using its matching state vector. cube-average retains the old method."
        ),
    )
    parser.add_argument(
        "--trajectory-file",
        default=None,
        help="Explicit MEM trajectory file. Default: <mem-directory>/input.txt.",
    )
    parser.add_argument(
        "--flux-output-axes",
        choices=["auto", "body-fixed", "equatorial", "ecliptic"],
        default="auto",
        help=(
            "Axes used by flux_N angles. auto reads options.txt and otherwise "
            "assumes body-fixed."
        ),
    )
    parser.add_argument(
        "--trajectory-axes",
        choices=["equatorial", "ecliptic"],
        default="equatorial",
        help="Inertial axes of input.txt state vectors. Default: equatorial.",
    )
    parser.add_argument(
        "--speed-label-mode",
        choices=["auto", "none", "add-half-bin"],
        default="auto",
        help=(
            "Correct legacy integer speed labels. auto maps 0,1,... to "
            "0.5,1.5,... when appropriate."
        ),
    )
    parser.add_argument(
        "--state-weighting",
        choices=["equal", "time"],
        default="equal",
        help="Average instantaneous flux_N environments equally or by time spacing.",
    )
    parser.add_argument(
        "--mars-speed-bin-width-kms",
        type=float,
        default=1.0,
        help="Output Mars-relative speed-bin width [km/s]. Default: 1.",
    )
    parser.add_argument("--mars-radius-km", type=float, default=3389.5)
    parser.add_argument(
        "--expected-mem-altitude-km",
        type=float,
        default=100.0,
        help=(
            "Expected altitude of the fictitious MEM sampling trajectory [km]. "
            "Default: 100. This is not the real camera altitude."
        ),
    )
    parser.add_argument(
        "--sampling-altitude-tolerance-km",
        type=float,
        default=25.0,
        help=(
            "Maximum allowed difference from --expected-mem-altitude-km before "
            "the script stops. Default: 25 km."
        ),
    )
    parser.add_argument(
        "--skip-sampling-altitude-check",
        action="store_true",
        help="Process the trajectory even when it is not near the expected altitude.",
    )
    parser.add_argument(
        "--flux-frame-scaling",
        choices=["number-density", "preserve-mem-flux"],
        default="number-density",
        help=(
            "number-density removes the fictitious observer motion from both "
            "speed and directional flux using F=n*v. preserve-mem-flux changes "
            "only speed/radiant bin assignment as a diagnostic."
        ),
    )
    parser.add_argument(
        "--cube-file",
        action="append",
        default=None,
        help="Explicit cube_avg.txt path for --environment-source cube-average.",
    )
    parser.add_argument(
        "--reference-mass-g",
        type=float,
        default=10.0,
        help="Limiting mass used for the MEM run [g]. Default: 10.",
    )
    parser.add_argument(
        "--mass-edges-g",
        default=",".join(str(value) for value in DEFAULT_MASS_EDGES_G),
        help=(
            "Mass-bin edges in grams. Default: "
            + ",".join(str(value) for value in DEFAULT_MASS_EDGES_G)
        ),
    )
    parser.add_argument(
        "--directions",
        default=",".join(DEFAULT_FACE_DIRECTIONS),
        help="Comma-separated cube directions used to estimate the speed distribution.",
    )
    parser.add_argument(
        "--flux-mode",
        choices=[
            "zenith",
            "selected-direction",
            "cross-sectional-normalized",
            "six-face-sum",
            "six-face-mean",
        ],
        default="zenith",
        help=(
            "Flux used for the mass-speed integration. In detailed-flux mode, "
            "zenith projects every transformed angular cell onto the body-fixed "
            "+z surface. Use selected-direction for another cube face. The "
            "cross-sectional and six-face modes remain available."
        ),
    )
    parser.add_argument(
        "--selected-direction",
        choices=DIRECTIONS,
        default="+z zenith",
        help=(
            "Surface used when --flux-mode selected-direction. Detailed-flux "
            "mode supports the six ram/wake/port/starboard/zenith/nadir faces; "
            "cube-average mode also supports the remaining MEM columns. "
            "Default: +z zenith."
        ),
    )

    parser.add_argument(
        "--camera-model",
        default=r"C:\Users\maxiv\Documents\UWO\Papers\0.5)METEORCAM-Strawman\METEORCAM\All\10FPS\Mars_detection_likelihood_model.pkl",
        help="Mars_detection_*_likelihood_model.pkl from the fireball fitting script.",
    )
    parser.add_argument(
        "--camera-csv",
        default=r"C:\Users\maxiv\Documents\UWO\Papers\0.5)METEORCAM-Strawman\METEORCAM\All\10FPS\Mars_detection_velocity_mass_summary.csv",
        help="Likelihood/synthetic CSV used to fit a NumPy logistic camera model.",
    )
    parser.add_argument(
        "--camera-search-directory",
        default=None,#r"C:\Users\maxiv\Documents\UWO\Papers\0.5)METEORCAM-Strawman\All",
        help="Directory recursively searched for a camera model/CSV when none is supplied.",
    )
    parser.add_argument(
        "--minimum-detected-frames",
        type=int,
        default=10,
        help="Frame requirement used when fitting labels from a CSV. Default: 10.",
    )

    # Manual fallback for testing a known m*v^x sensitivity boundary.
    parser.add_argument("--proxy-exponent-x", type=float, default=None)
    parser.add_argument("--proxy-reference-mass-kg", type=float, default=None)
    parser.add_argument("--proxy-reference-speed-kms", type=float, default=None)
    parser.add_argument("--proxy-log-width-dex", type=float, default=0.20)

    parser.add_argument(
        "--visible-probability-threshold",
        type=float,
        default=0.99,
        help="Only cells at or above this probability are colored in the visible-flux plot.",
    )
    parser.add_argument(
        "--substeps-per-mass-bin",
        type=int,
        default=100,
        help="Logarithmic integration substeps per finite mass bin. Default: 100.",
    )
    parser.add_argument(
        "--model-extrapolation",
        choices=["allow", "clip", "mask"],
        default="allow",
        help="Treatment outside the fitted camera-model mass/speed range. Default: allow.",
    )
    parser.add_argument(
        "--effective-area-km2",
        type=float,
        default=31.1e6,
        help=(
            "Optional effective collecting area. When supplied, the summary also "
            "reports expected events/year. Only use an area consistent with the "
            "selected MEM flux interpretation."
        ),
    )
    parser.add_argument(
        "--output-directory",
        default=r"C:\Users\maxiv\Documents\UWO\Papers\0.5)METEORCAM-Strawman\METEORCAM\FPS-10corrV",
        help="Output directory. Default: <mem-directory>/MEM_camera_flux_extension.",
    )
    parser.add_argument(
        "--output-prefix",
        default="Mars_MEM_camera",
        help="Filename prefix for outputs.",
    )
    return parser.parse_args()


def resolve_camera_model(args: argparse.Namespace, mem_directory: Path) -> tuple[dict[str, Any], str]:
    if args.camera_model:
        path = Path(args.camera_model)
        return load_camera_model(path), str(path.resolve())

    if args.camera_csv:
        path = Path(args.camera_csv)
        return fit_camera_model_from_csv(path, args.minimum_detected_frames), str(path.resolve())

    if args.proxy_exponent_x is not None:
        if args.proxy_reference_mass_kg is None or args.proxy_reference_speed_kms is None:
            raise ValueError(
                "Manual proxy mode requires --proxy-reference-mass-kg and "
                "--proxy-reference-speed-kms."
            )
        return manual_proxy_model(
            exponent_x=args.proxy_exponent_x,
            reference_mass_kg=args.proxy_reference_mass_kg,
            reference_speed_kms=args.proxy_reference_speed_kms,
            log_width_dex=args.proxy_log_width_dex,
            minimum_frames=args.minimum_detected_frames,
        ), "manual command-line proxy"

    search_root = Path(args.camera_search_directory) if args.camera_search_directory else mem_directory
    discovered = discover_camera_model(search_root)
    if discovered is None:
        raise FileNotFoundError(
            "No camera likelihood model was found. Supply --camera-model, "
            "--camera-csv, or the manual --proxy-* arguments."
        )
    kind, path = discovered
    if kind == "pickle":
        return load_camera_model(path), str(path.resolve())
    return fit_camera_model_from_csv(path, args.minimum_detected_frames), str(path.resolve())


def main() -> None:
    args = parse_args()
    mem_directory = Path(args.mem_directory).resolve()
    output_directory = (
        Path(args.output_directory).resolve()
        if args.output_directory
        else mem_directory/"MEM_camera_flux_extension"
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    detailed_result: DetailedFluxResult | None = None
    cubes: list[MemCubeData] = []
    total_cross_flux = np.nan
    directions = parse_directions(args.directions)

    if args.environment_source == "detailed-flux":
        detailed_result = transform_flux_files_to_mars_frame(
            mem_directory=mem_directory,
            trajectory_file=args.trajectory_file,
            output_axes=args.flux_output_axes,
            trajectory_axes=args.trajectory_axes,
            speed_label_mode=args.speed_label_mode,
            state_weighting_mode=args.state_weighting,
            flux_mode=args.flux_mode,
            selected_direction=args.selected_direction,
            mars_speed_bin_width_kms=float(args.mars_speed_bin_width_kms),
            mars_radius_km=float(args.mars_radius_km),
            expected_sampling_altitude_km=float(args.expected_mem_altitude_km),
            sampling_altitude_tolerance_km=float(args.sampling_altitude_tolerance_km),
            skip_sampling_altitude_check=bool(args.skip_sampling_altitude_check),
            flux_frame_scaling=args.flux_frame_scaling,
        )
        speeds = detailed_result.mars_speed_midpoints_kms
        reference_flux = detailed_result.transformed_speed_flux_m2_yr
        diagnostics = {
            "reference_speed_flux": reference_flux,
            "reference_speed_flux_total": float(np.sum(reference_flux)),
            "comparison_speed_kms": detailed_result.spacecraft_speed_midpoints_kms,
            "comparison_flux_m2_yr": detailed_result.spacecraft_surface_flux_m2_yr,
            "selected_direction_name": detailed_result.selected_surface,
            "individual_direction_total": float(np.sum(reference_flux)),
            "selected_direction_sum_total": float(np.sum(reference_flux)),
            "selected_direction_mean_total": float(np.sum(reference_flux)),
            "speed_axis_label": (
                "Mars-relative meteoroid speed at MEM sampling altitude [km/s]"
            ),
        }
    else:
        cube_paths = (
            [Path(path).resolve() for path in args.cube_file]
            if args.cube_file
            else discover_cube_files(mem_directory)
        )
        speeds, combined_direction_flux, total_cross_flux, cubes = (
            combine_mem_cube_files(cube_paths)
        )
        reference_flux, diagnostics = reference_speed_flux(
            combined_direction_flux,
            directions,
            total_cross_flux,
            args.flux_mode,
            selected_direction=args.selected_direction,
        )

    mass_edges_g = parse_mass_edges(args.mass_edges_g)
    if mass_edges_g[0] < float(args.reference_mass_g) - 1.0e-12:
        raise ValueError(
            "The first requested mass edge is below the MEM reference mass. "
            "This tool is intended to extend MEM to larger masses."
        )

    camera_model, camera_source = resolve_camera_model(args, mem_directory)
    minimum_frames = int(camera_model.get("minimum_detected_frames", args.minimum_detected_frames))

    grid = integrate_mass_speed_flux(
        reference_speed_flux_m2_yr=reference_flux,
        speeds_kms=speeds,
        mass_edges_g=mass_edges_g,
        reference_mass_g=float(args.reference_mass_g),
        camera_model=camera_model,
        substeps_per_mass_bin=int(args.substeps_per_mass_bin),
        extrapolation=args.model_extrapolation,
    )

    incident = grid["incident_flux"]
    detectable = grid["detectable_flux"]
    probability = grid["mean_detection_probability"]
    visible = np.isfinite(probability) & (probability >= float(args.visible_probability_threshold))

    incident_total = float(np.sum(incident))
    detectable_total = float(np.sum(detectable))
    binary_visible_total = float(np.sum(np.where(visible, incident, 0.0)))
    detection_efficiency = detectable_total/incident_total if incident_total > 0.0 else np.nan

    proxy = camera_proxy_parameters(camera_model)
    effective_area_m2 = (
        float(args.effective_area_km2)*1.0e6
        if args.effective_area_km2 is not None
        else None
    )

    prefix = args.output_prefix
    outputs = {
        "reference_speed_plot": output_directory/f"{prefix}_reference_speed_flux.png",
        "grun_plot": output_directory/f"{prefix}_grun_mass_scaling.png",
        "probability_plot": output_directory/f"{prefix}_camera_detection_probability.png",
        "incident_plot": output_directory/f"{prefix}_incident_flux_mass_speed.png",
        "visible_flux_plot": output_directory/f"{prefix}_visible_detectable_flux_mass_speed.png",
        "grid_csv": output_directory/f"{prefix}_mass_speed_flux.csv",
        "summary_json": output_directory/f"{prefix}_summary.json",
        "summary_txt": output_directory/f"{prefix}_summary.txt",
        "camera_model_pickle": output_directory/f"{prefix}_camera_model_used.pkl",
        "state_transform_csv": output_directory/f"{prefix}_state_transform_summary.csv",
    }

    save_reference_speed_plot(
        outputs["reference_speed_plot"], speeds, reference_flux, diagnostics,
        float(args.reference_mass_g), args.flux_mode,
    )
    save_grun_plot(outputs["grun_plot"], mass_edges_g, float(args.reference_mass_g))
    save_probability_heatmap(
        outputs["probability_plot"], speeds, mass_edges_g, probability,
        float(args.visible_probability_threshold), camera_model,
    )
    save_flux_heatmap(
        outputs["incident_plot"], speeds, mass_edges_g, incident,
        title=(
            f"Incident MEM flux by mass and speed\n"
            f"{mass_edges_g[0]:g}–{mass_edges_g[-1]:g} g; Grün-scaled from {args.reference_mass_g:g} g"
        ),
        colorbar_label="Incident differential flux [#/m²/yr per mass-speed bin]",
    )
    save_flux_heatmap(
        outputs["visible_flux_plot"], speeds, mass_edges_g, detectable,
        title=(
            "Probability-weighted METEORCAM-detectable flux\n"
            f"only cells with P(detection) ≥ {args.visible_probability_threshold:.2f}; "
            f"minimum={minimum_frames} frames"
        ),
        colorbar_label="Camera-detectable flux [#/m²/yr per mass-speed bin]",
        probability=probability,
        visible_probability_threshold=float(args.visible_probability_threshold),
    )
    save_grid_csv(
        outputs["grid_csv"], speeds, mass_edges_g, reference_flux,
        incident, detectable, probability,
        float(args.visible_probability_threshold),
    )
    if detailed_result is not None:
        save_state_transform_csv(
            outputs["state_transform_csv"],
            detailed_result.state_rows,
        )

    with outputs["camera_model_pickle"].open("wb") as fh:
        pickle.dump(camera_model, fh, protocol=pickle.HIGHEST_PROTOCOL)

    summary: dict[str, Any] = {
        "format_version": 1,
        "mem_directory": str(mem_directory),
        "environment_source": args.environment_source,
        "mem_cube_files": [str(cube.path) for cube in cubes],
        "mem_flux_files": (
            detailed_result.flux_files if detailed_result is not None else []
        ),
        "trajectory_file": (
            detailed_result.trajectory_file if detailed_result is not None else None
        ),
        "flux_output_axes": (
            detailed_result.output_axes if detailed_result is not None else None
        ),
        "speed_label_shift_kms": (
            detailed_result.speed_label_shift_kms
            if detailed_result is not None
            else 0.0
        ),
        "state_weighting": (
            detailed_result.state_weighting if detailed_result is not None else None
        ),
        "mem_sampling_altitude_mean_km": (
            detailed_result.sampling_altitude_mean_km
            if detailed_result is not None else None
        ),
        "mem_sampling_altitude_min_km": (
            detailed_result.sampling_altitude_min_km
            if detailed_result is not None else None
        ),
        "mem_sampling_altitude_max_km": (
            detailed_result.sampling_altitude_max_km
            if detailed_result is not None else None
        ),
        "flux_frame_scaling": (
            detailed_result.flux_frame_scaling
            if detailed_result is not None else None
        ),
        "reference_mass_g": float(args.reference_mass_g),
        "mass_edges_g": mass_edges_g,
        "mass_range_g": [float(mass_edges_g[0]), float(mass_edges_g[-1])],
        "selected_directions": directions,
        "selected_individual_direction": str(
            diagnostics.get("selected_direction_name", args.selected_direction)
        ),
        "flux_mode": args.flux_mode,
        "mem_total_cross_sectional_flux_above_reference_m2_yr": float(total_cross_flux),
        "individual_direction_flux_above_reference_m2_yr": float(
            diagnostics["individual_direction_total"]
        ),
        "selected_direction_sum_flux_above_reference_m2_yr": float(diagnostics["selected_direction_sum_total"]),
        "selected_direction_mean_flux_above_reference_m2_yr": float(diagnostics["selected_direction_mean_total"]),
        "reference_flux_used_above_reference_m2_yr": float(np.sum(reference_flux)),
        "incident_flux_in_mass_range_m2_yr": incident_total,
        "camera_detectable_flux_m2_yr": detectable_total,
        "binary_visible_region_flux_m2_yr": binary_visible_total,
        "visible_probability_threshold": float(args.visible_probability_threshold),
        "mean_detection_efficiency": float(detection_efficiency),
        "minimum_detected_frames": minimum_frames,
        "camera_model_source": camera_source,
        "camera_model_type": camera_model.get("model_type", "unknown"),
        "camera_model_extrapolation": args.model_extrapolation,
        "camera_proxy_speed_exponent_x": proxy["speed_exponent_x"],
        "camera_proxy_threshold_50": proxy["proxy_threshold_50"],
        "grun_reference_value": float(grun_cumulative_flux(args.reference_mass_g)),
        "grun_ratio_at_maximum_mass": float(grun_scale_ratio(mass_edges_g[-1], args.reference_mass_g)),
        "effective_area_m2": effective_area_m2,
        "expected_incident_events_per_year": (
            incident_total*effective_area_m2 if effective_area_m2 is not None else None
        ),
        "expected_detectable_events_per_year": (
            detectable_total*effective_area_m2 if effective_area_m2 is not None else None
        ),
        "outputs": {key: str(value) for key, value in outputs.items()},
        "interpretation_note": (
            "Detailed-flux mode removes the orbital velocity of the fictitious MEM "
            "sampling observer at approximately 100 km and recovers the "
            "Mars-relative meteoroid speed at the same position before the "
            "camera model is applied. The real 5720-km camera altitude is not "
            "used in this velocity transformation. Probability-weighted detectable flux "
            "integrates P(camera detection) within each mass bin."
        ),
    }

    with outputs["summary_json"].open("w", encoding="utf-8") as fh:
        json.dump(json_safe(summary), fh, indent=2)
    save_text_summary(outputs["summary_txt"], summary)

    print("MEM mass extension completed.")
    print(f"Environment source: {args.environment_source}")
    if detailed_result is not None:
        print(f"Instantaneous states transformed: {len(detailed_result.state_rows)}")
        print(f"Flux files combined: {len(detailed_result.flux_files)}")
        print(f"Resolved MEM output axes: {detailed_result.output_axes}")
        print(
            "MEM sampling altitude: "
            f"{detailed_result.sampling_altitude_mean_km:.3f} km mean "
            f"({detailed_result.sampling_altitude_min_km:.3f}-"
            f"{detailed_result.sampling_altitude_max_km:.3f} km)"
        )
        print(f"Flux frame scaling: {detailed_result.flux_frame_scaling}")
        print(
            "Speed-label shift applied: "
            f"{detailed_result.speed_label_shift_kms:.3f} km/s"
        )
    else:
        print(f"Cube files combined: {len(cubes)}")
    print(f"Reference flux used: {np.sum(reference_flux):.8e} #/m^2/yr")
    print(f"Incident flux {mass_edges_g[0]:g}-{mass_edges_g[-1]:g} g: {incident_total:.8e} #/m^2/yr")
    print(f"Camera-detectable flux: {detectable_total:.8e} #/m^2/yr")
    print(f"Mean detection efficiency: {detection_efficiency:.6f}")
    print(f"Outputs saved to: {output_directory}")


if __name__ == "__main__":
    main()
