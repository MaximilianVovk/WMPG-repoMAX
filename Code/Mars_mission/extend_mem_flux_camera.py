#!/usr/bin/env python3
"""
Extend a MEM 3 speed-binned flux run beyond its limiting mass and fold the
result through an empirical METEORCAM mass-speed detection model.

The script is designed for a MEM run performed at 10 g, but the reference mass
is configurable. It:

1. Reads and combines HiDensity/LoDensity cube_avg.txt files.
2. Uses the Grün cumulative mass-scaling equation used by MEM 3:

       F(>m, v) = F_MEM(>m_ref, v) * g(m)/g(m_ref)

3. Converts cumulative fluxes into finite mass bins:

       F([m1,m2), v) = F_MEM(>m_ref, v) * [g(m1)-g(m2)]/g(m_ref)

4. Loads the detection-likelihood pickle written by the Mars synthetic-speed
   script, or fits a compatible NumPy logistic model from its CSV output.
5. Integrates detection probability within each mass bin.
6. Saves incident and camera-detectable mass-speed flux grids, plots, CSV and
   JSON/text summaries.

Important MEM interpretation:
- Cube face fluxes are not the same as total cross-sectional flux.
- The default flux mode uses the six cube faces only to estimate the speed-
  distribution shape, and normalizes that shape to the total cross-sectional
  flux reported in the cube-file headers.
- Use --flux-mode six-face-sum to reproduce the user's earlier plotting method.

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

# DEFAULT_MASS_EDGES_G = [10, 50, 100, 250, 500, 1000, 2000, 5000, 10000]
DEFAULT_MASS_EDGES_G = [50, 100, 250, 500, 1000, 2000, 5000, 10000]


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
) -> tuple[np.ndarray, dict[str, np.ndarray | float]]:
    matrix = np.vstack([np.asarray(direction_flux[name], dtype=float) for name in directions])
    face_sum = np.sum(matrix, axis=0)
    face_mean = np.mean(matrix, axis=0)

    diagnostics: dict[str, np.ndarray | float] = {
        "six_face_or_selected_sum_speed_flux": face_sum,
        "six_face_or_selected_mean_speed_flux": face_mean,
        "selected_direction_sum_total": float(np.sum(face_sum)),
        "selected_direction_mean_total": float(np.sum(face_mean)),
        "total_cross_sectional_flux": float(total_cross_sectional_flux),
    }

    if mode == "six-face-sum":
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
        speeds[positive], selected_flux[positive], marker="o", linewidth=2.0,
        markersize=4,
        label=(
            f"Flux used ({flux_mode}), total={np.sum(selected_flux):.6e} #/m²/yr"
        ),
    )

    raw = np.asarray(diagnostics["six_face_or_selected_sum_speed_flux"], dtype=float)
    raw_positive = raw > 0.0
    if flux_mode != "six-face-sum":
        ax.plot(
            speeds[raw_positive], raw[raw_positive], marker=".", linestyle="--",
            label=f"Selected cube-face sum, total={np.sum(raw):.6e} #/m²/yr",
        )

    ax.set_yscale("log")
    ax.set_xlabel("Speed [km/s]")
    ax.set_ylabel("Cumulative flux per speed bin [#/m²/yr]")
    ax.set_title(f"MEM reference speed distribution for m ≥ {reference_mass_g:g} g")
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
        f"MEM cube files: {len(summary['mem_cube_files'])}",
        f"Reference limiting mass: {summary['reference_mass_g']:.6g} g",
        f"Mass interval integrated: {summary['mass_range_g'][0]:.6g} to {summary['mass_range_g'][1]:.6g} g",
        f"Flux mode: {summary['flux_mode']}",
        f"Selected directions: {', '.join(summary['selected_directions'])}",
        "",
        f"MEM header total cross-sectional flux >= reference mass: {summary['mem_total_cross_sectional_flux_above_reference_m2_yr']:.8e} #/m^2/yr",
        f"Selected six-face/direction sum >= reference mass: {summary['selected_direction_sum_flux_above_reference_m2_yr']:.8e} #/m^2/yr",
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
            f"Expected incident events per year: {summary['expected_incident_events_per_year']:.8e}",
            f"Expected camera-detectable events per year: {summary['expected_detectable_events_per_year']:.8e}",
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
        default=r"C:\Users\maxiv\Documents\UWO\Papers\0.5)METEORCAM-Strawman\Test-Flux-surface\test-surf-mars - allorbit",
        help="MEM output directory containing HiDensity/LoDensity cube_avg.txt files.",
    )
    parser.add_argument(
        "--cube-file",
        action="append",
        default=None,
        help="Explicit cube_avg.txt path. Repeat for multiple files; overrides discovery.",
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
        choices=["cross-sectional-normalized", "six-face-sum", "six-face-mean"],
        default="cross-sectional-normalized",
        help=(
            "Flux used for the mass-speed integration. The default normalizes the "
            "selected face speed distribution to the total cross-sectional flux "
            "reported in the MEM headers."
        ),
    )

    parser.add_argument(
        "--camera-model",
        default=r"C:\Users\maxiv\Documents\UWO\Papers\0.5)METEORCAM-Strawman\All\Mars_detection_likelihood_model.pkl",
        help="Mars_detection_*_likelihood_model.pkl from the fireball fitting script.",
    )
    parser.add_argument(
        "--camera-csv",
        default=r"C:\Users\maxiv\Documents\UWO\Papers\0.5)METEORCAM-Strawman\All\Mars_detection_velocity_mass_summary.csv",
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
        default=r"C:\Users\maxiv\Documents\UWO\Papers\0.5)METEORCAM-Strawman",
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

    cube_paths = [Path(path).resolve() for path in args.cube_file] if args.cube_file else discover_cube_files(mem_directory)
    speeds, combined_direction_flux, total_cross_flux, cubes = combine_mem_cube_files(cube_paths)
    directions = parse_directions(args.directions)
    reference_flux, diagnostics = reference_speed_flux(
        combined_direction_flux,
        directions,
        total_cross_flux,
        args.flux_mode,
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

    with outputs["camera_model_pickle"].open("wb") as fh:
        pickle.dump(camera_model, fh, protocol=pickle.HIGHEST_PROTOCOL)

    summary: dict[str, Any] = {
        "format_version": 1,
        "mem_directory": str(mem_directory),
        "mem_cube_files": [str(cube.path) for cube in cubes],
        "reference_mass_g": float(args.reference_mass_g),
        "mass_edges_g": mass_edges_g,
        "mass_range_g": [float(mass_edges_g[0]), float(mass_edges_g[-1])],
        "selected_directions": directions,
        "flux_mode": args.flux_mode,
        "mem_total_cross_sectional_flux_above_reference_m2_yr": float(total_cross_flux),
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
            "The probability-weighted detectable flux integrates P(camera detection) "
            "within each mass bin. The binary visible-region flux instead includes the "
            "entire incident cell only when its mean probability is above the selected threshold."
        ),
    }

    with outputs["summary_json"].open("w", encoding="utf-8") as fh:
        json.dump(json_safe(summary), fh, indent=2)
    save_text_summary(outputs["summary_txt"], summary)

    print("MEM mass extension completed.")
    print(f"Cube files combined: {len(cubes)}")
    print(f"Reference flux used: {np.sum(reference_flux):.8e} #/m^2/yr")
    print(f"Incident flux {mass_edges_g[0]:g}-{mass_edges_g[-1]:g} g: {incident_total:.8e} #/m^2/yr")
    print(f"Camera-detectable flux: {detectable_total:.8e} #/m^2/yr")
    print(f"Mean detection efficiency: {detection_efficiency:.6f}")
    print(f"Outputs saved to: {output_directory}")


if __name__ == "__main__":
    main()
