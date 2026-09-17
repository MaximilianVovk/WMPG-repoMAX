#!/usr/bin/env python3
"""
Standalone single-event MetSim -> custom-planet/Mars forward model.

Purpose
-------
This is a complete simplification of the old
Mars_detection1_from_satellite_with_synthetic.py workflow.

Only ONE argument is normally typed:
    --json PATH

A physical rerun also requires the matching trajectory pickle, but it is
normally auto-discovered beside the JSON.

The input JSON may be either:
1) a fitted MetSim constants JSON, or
2) a result JSON previously written by this program.

For a fitted MetSim JSON, the program:
    * reads planet_parameters.txt and the height-density atmosphere CSV it names;
    * fits the WMPL/MetSim atmosphere polynomial directly from that table;
    * loads the matching trajectory pickle to recompute target-planet speed and zenith angle;
    * chooses the new simulation start height by matching the atmospheric
      density at the original JSON h_init, unless --start-height-km is supplied;
    * remaps erosion_height_start, erosion_height_change, and every
      fragmentation entry using one of:
          dynamic_pressure (default)
          energy
          density
    * runs the new-atmosphere simulation;
    * optionally determines whether finite-FPS luminosity integration gives a
      better RMSD against a supplied trajectory/observation pickle;
    * saves the complete useful simulation arrays and metadata to JSON;
    * writes the original Mars_detection-style absolute-magnitude/height figure;
    * optionally computes apparent magnitude and sampled visible frames for a
      supplied observer position and limiting magnitude, and writes a second
      detection figure only when at least one frame is above the limit.

The final result JSON contains enough trajectory/light-curve information to be
used as --json on a later invocation. This allows changes to limiting magnitude,
observer position, FPS, or plot format WITHOUT rerunning MetSim.

Notes
-----
* Dynamic pressure is the default and fastest physically motivated trigger.
* Density mapping is also fast.
* Energy mapping evaluates a cumulative received-energy profile using the
  simulated changing speed and main-body mass. Fragmentation-heavy or very
  finely sampled events can therefore take noticeably longer to prepare.
* The atmosphere CSV is expected to contain height and density as its two main
  numeric columns. Named columns containing "alt"/"height" and "rho"/"dens"
  are preferred. Extra columns are ignored.
* Mars_Vel.py is used when available for Mars orbit-intercept speeds; an internal fallback is retained.

Examples
--------
Basic Mars run:
    python Mars_detection1_from_satellite_with_synthetic.py ^
        --json 20230811_082648_sim_fit_latest.json

Use a different trigger:
    python Mars_detection1_from_satellite_with_synthetic.py ^
        --json event_sim_fit_latest.json ^
        --trigger energy

Validate finite-FPS integration against observations:
    python Mars_detection1_from_satellite_with_synthetic.py ^
        --json event_sim_fit_latest.json ^
        --pickle event_trajectory.pickle ^
        --integration-mode auto

Satellite apparent-magnitude/frame test:
    python Mars_detection1_from_satellite_with_synthetic.py ^
        --json event_sim_fit_latest.json ^
        --limiting-mag 4 ^
        --observer-altitude-km 5720 ^
        --camera-fps 15

Reprocess a saved result without rerunning MetSim:
    python Mars_detection1_from_satellite_with_synthetic.py ^
        --json event_planet_run.json ^
        --limiting-mag 3 ^
        --observer-altitude-km 5720 ^
        --camera-fps 15
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import pickle
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np


PROGRAM_SCHEMA = "planet_metsim_single_run_v2"
LEGACY_PROGRAM_SCHEMAS = {"planet_metsim_single_run_v1"}
CACHE_VERSION = 3

DEFAULT_PLANET_PARAMETER_FILENAME = "planet_parameters.txt"
DEFAULT_MARS_PARAMETER_FILENAME = "Mars_planet_parameters.txt"

MU_SUN_M3_S2 = 1.32712440018e20
AU_M = 1.495978707e11


# =============================================================================
# Small utilities
# =============================================================================

def finite_float(value: Any, default: float = np.nan) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)
    return value if np.isfinite(value) else float(default)


def is_finite_number(value: Any) -> bool:
    return bool(np.isfinite(finite_float(value)))


def json_safe(value: Any, _seen: set[int] | None = None, _depth: int = 0) -> Any:
    """Recursively convert values to JSON-safe objects without following cycles.

    WMPL objects (especially fragmentation entries/results) can contain back
    references.  The previous generic ``__dict__`` recursion could therefore
    loop forever for some meteors.  This serializer keeps cycle detection as a
    last line of defence; important WMPL objects are flattened explicitly
    before they reach this function.
    """
    if _seen is None:
        _seen = set()

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, np.generic):
        return json_safe(value.item(), _seen, _depth)

    if _depth > 20:
        return None

    # Containers/objects may be self-referential.
    track_identity = isinstance(value, (np.ndarray, list, tuple, dict)) or hasattr(value, "__dict__")
    obj_id = id(value)
    if track_identity:
        if obj_id in _seen:
            return None
        _seen.add(obj_id)

    try:
        if isinstance(value, np.ndarray):
            return [json_safe(v, _seen, _depth + 1) for v in value.tolist()]
        if isinstance(value, (list, tuple)):
            return [json_safe(v, _seen, _depth + 1) for v in value]
        if isinstance(value, dict):
            return {
                str(k): json_safe(v, _seen, _depth + 1)
                for k, v in value.items()
            }
        if hasattr(value, "__dict__"):
            return {
                str(k): json_safe(v, _seen, _depth + 1)
                for k, v in vars(value).items()
                if not str(k).startswith("_")
            }
        try:
            return json_safe(float(value), _seen, _depth + 1)
        except Exception:
            return str(value)
    finally:
        if track_identity:
            _seen.discard(obj_id)


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return data


def save_json(path: str | Path, data: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(json_safe(data), fh, indent=2)
    return path


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def output_paths_for_format(base: Path, plot_format: str) -> list[Path]:
    fmt = str(plot_format).lower()
    if fmt == "both":
        return [base.with_suffix(".png"), base.with_suffix(".pdf")]
    return [base.with_suffix("." + fmt)]


def save_figure(fig: Any, base: Path, plot_format: str) -> list[str]:
    paths = output_paths_for_format(base, plot_format)
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return [str(path) for path in paths]


def monotonic_unique_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return finite x/y sorted by x with duplicate x values averaged."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = min(len(x), len(y))
    x = x[:n]
    y = y[:n]
    good = np.isfinite(x) & np.isfinite(y)
    x = x[good]
    y = y[good]
    if x.size == 0:
        return x, y

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    unique_x, inverse = np.unique(x, return_inverse=True)
    if len(unique_x) == len(x):
        return x, y

    sums = np.zeros(len(unique_x), dtype=float)
    counts = np.zeros(len(unique_x), dtype=float)
    np.add.at(sums, inverse, y)
    np.add.at(counts, inverse, 1.0)
    return unique_x, sums / np.maximum(counts, 1.0)


def interpolate_profile_at_height(
    height_m: np.ndarray,
    values: np.ndarray,
    target_height_m: float,
) -> float:
    h, v = monotonic_unique_xy(height_m, values)
    if h.size == 0:
        return float("nan")
    target = float(np.clip(target_height_m, h[0], h[-1]))
    return float(np.interp(target, h, v))


# =============================================================================
# Atmosphere CSV
# =============================================================================

@dataclass
class AtmosphereTable:
    path: Path
    height_m: np.ndarray
    density_kg_m3: np.ndarray
    dens_co: np.ndarray
    height_column: str
    density_column: str
    fit_degree: int
    fit_rms_log10_density: float

    @property
    def min_height_m(self) -> float:
        return float(np.nanmin(self.height_m))

    @property
    def max_height_m(self) -> float:
        return float(np.nanmax(self.height_m))


def _column_numeric_fraction(rows: list[dict[str, str]], name: str) -> float:
    total = 0
    valid = 0
    for row in rows:
        total += 1
        try:
            val = float(row.get(name, ""))
            if np.isfinite(val):
                valid += 1
        except Exception:
            pass
    return valid / max(total, 1)


def read_atmosphere_csv(path: str | Path, polynomial_degree: int = 6) -> AtmosphereTable:
    """
    Read only the height and density columns and fit the MetSim density polynomial.

    WMPL/MetSim dens_co uses:
        log10(rho) = sum_i dens_co[i] * (height_m / 1e6)**i
    """
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Atmosphere CSV not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            raise ValueError(f"Atmosphere CSV has no header: {path}")
        rows = list(reader)
        names = [str(name).strip() for name in reader.fieldnames]

    lowered = {name: name.lower() for name in names}
    height_candidates = [
        name for name in names
        if ("alt" in lowered[name] or "height" in lowered[name])
        and _column_numeric_fraction(rows, name) > 0.5
    ]
    density_candidates = [
        name for name in names
        if ("rho" in lowered[name] or "dens" in lowered[name])
        and _column_numeric_fraction(rows, name) > 0.5
    ]

    if height_candidates and density_candidates:
        height_col = height_candidates[0]
        density_col = density_candidates[0]
    else:
        numeric_cols = [
            name for name in names
            if _column_numeric_fraction(rows, name) > 0.5
        ]
        if len(numeric_cols) < 2:
            raise ValueError(
                "Could not identify two numeric height/density columns in "
                f"{path.name}. Columns: {names}"
            )
        height_col, density_col = numeric_cols[:2]

    height_vals = []
    density_vals = []
    for row in rows:
        try:
            h = float(row[height_col])
            rho = float(row[density_col])
        except Exception:
            continue
        if np.isfinite(h) and np.isfinite(rho) and rho > 0.0:
            height_vals.append(h)
            density_vals.append(rho)

    if len(height_vals) < 4:
        raise ValueError(f"Too few valid atmosphere rows in {path}")

    height = np.asarray(height_vals, dtype=float)
    density = np.asarray(density_vals, dtype=float)

    # Infer height units. Named km columns or small maximum values are treated as km.
    hname = height_col.lower()
    if "km" in hname or np.nanmax(np.abs(height)) < 2000.0:
        height_m = height * 1000.0
    else:
        height_m = height

    order = np.argsort(height_m)
    height_m = height_m[order]
    density = density[order]

    # Remove duplicate heights by averaging density in log space.
    unique_h = np.unique(height_m)
    if len(unique_h) != len(height_m):
        out_rho = []
        for h in unique_h:
            vals = density[height_m == h]
            out_rho.append(10.0 ** np.mean(np.log10(vals)))
        height_m = unique_h
        density = np.asarray(out_rho, dtype=float)

    degree = min(int(polynomial_degree), len(height_m) - 1)
    x = height_m / 1.0e6
    y = np.log10(density)
    dens_co = np.polynomial.polynomial.polyfit(x, y, degree)
    y_fit = np.polynomial.polynomial.polyval(x, dens_co)
    rms = float(np.sqrt(np.mean((y - y_fit) ** 2)))

    return AtmosphereTable(
        path=path,
        height_m=height_m,
        density_kg_m3=density,
        dens_co=np.asarray(dens_co, dtype=float),
        height_column=height_col,
        density_column=density_col,
        fit_degree=degree,
        fit_rms_log10_density=rms,
    )


def density_from_dens_co(height_m: np.ndarray | float, dens_co: Iterable[float]) -> np.ndarray:
    h = np.asarray(height_m, dtype=float)
    co = np.asarray(dens_co, dtype=float)
    log_rho = np.polynomial.polynomial.polyval(h / 1.0e6, co)
    with np.errstate(over="ignore", invalid="ignore"):
        return np.power(10.0, log_rho)


def density_from_table(table: AtmosphereTable, height_m: float) -> float:
    h = float(np.clip(height_m, table.min_height_m, table.max_height_m))
    log_rho = np.interp(
        h,
        table.height_m,
        np.log10(table.density_kg_m3),
    )
    return float(10.0 ** log_rho)


def height_for_density(
    table: AtmosphereTable,
    target_density: float,
) -> tuple[float, str]:
    """Return target-atmosphere height matching density, with boundary diagnostics."""
    target_density = float(target_density)
    if not np.isfinite(target_density) or target_density <= 0.0:
        raise ValueError(f"Invalid target density: {target_density}")

    log_rho = np.log10(table.density_kg_m3)
    heights = table.height_m

    # Sort by density because density normally decreases with height.
    order = np.argsort(log_rho)
    x = log_rho[order]
    y = heights[order]
    target = math.log10(target_density)

    if target < x[0]:
        return float(y[0]), "clamped_to_lowest_density_boundary"
    if target > x[-1]:
        return float(y[-1]), "clamped_to_highest_density_boundary"
    return float(np.interp(target, x, y)), "interpolated_density_match"



def _parse_parameter_value(raw: str) -> Any:
    value = str(raw).strip()
    lower = value.lower()
    if lower in {"auto", "none", "null", ""}:
        return None
    if lower in {"true", "yes", "on"}:
        return True
    if lower in {"false", "no", "off"}:
        return False
    try:
        return float(value)
    except ValueError:
        return value


def read_planet_parameters(path: str | Path) -> dict[str, Any]:
    """
    Read a simple editable key=value planet parameter file.

    Lines beginning with # are comments. Unknown keys are preserved in the
    output JSON but otherwise ignored by the simulation.
    """
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Planet parameter file not found: {path}")

    params: dict[str, Any] = {}
    with path.open("r", encoding="utf-8-sig") as fh:
        for lineno, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            # Permit inline comments after whitespace + #.
            if " #" in stripped:
                stripped = stripped.split(" #", 1)[0].rstrip()
            if "=" not in stripped:
                raise ValueError(
                    f"{path.name}:{lineno}: expected key = value, got {line.rstrip()!r}"
                )
            key, value = stripped.split("=", 1)
            key = key.strip()
            if not key:
                raise ValueError(f"{path.name}:{lineno}: empty parameter name.")
            params[key] = _parse_parameter_value(value)

    required = (
        "planet_name",
        "radius_km",
        "mu_m3_s2",
        "surface_gravity_m_s2",
        "orbit_radius_au",
        "atmosphere_csv",
    )
    missing = [key for key in required if params.get(key) is None]
    if missing:
        raise ValueError(
            f"Planet parameter file {path.name} is missing required keys: {missing}"
        )

    params["_parameter_file"] = str(path)
    return params


def resolve_planet_parameter_file(
    input_json: Path,
    explicit: str | None,
) -> Path:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Planet parameter file not found: {path}")
        return path

    search_dirs = [input_json.parent]
    if input_json.parent.name.lower() == "fit_plots":
        search_dirs.append(input_json.parent.parent)
    search_dirs.extend([Path(__file__).resolve().parent, Path.cwd()])

    names = (
        DEFAULT_PLANET_PARAMETER_FILENAME,
        DEFAULT_MARS_PARAMETER_FILENAME,
    )
    seen = set()
    for directory in search_dirs:
        for name in names:
            candidate = (directory / name).resolve()
            if candidate in seen:
                continue
            seen.add(candidate)
            if candidate.is_file():
                return candidate

    raise FileNotFoundError(
        "No planet parameter file was found. Put planet_parameters.txt beside "
        "the event JSON (recommended), put Mars_planet_parameters.txt beside "
        "the script, or pass --planet-params PATH."
    )


def resolve_atmosphere_from_parameters(
    params: dict[str, Any],
    parameter_file: Path,
    input_json: Path,
) -> Path:
    raw = str(params["atmosphere_csv"])
    candidate = Path(raw).expanduser()
    if candidate.is_absolute() and candidate.is_file():
        return candidate.resolve()

    search_dirs = [
        parameter_file.parent,
        input_json.parent,
    ]
    if input_json.parent.name.lower() == "fit_plots":
        search_dirs.append(input_json.parent.parent)
    search_dirs.extend([Path(__file__).resolve().parent, Path.cwd()])

    for directory in search_dirs:
        path = (directory / candidate).resolve()
        if path.is_file():
            return path

    raise FileNotFoundError(
        f"Atmosphere CSV {raw!r} from {parameter_file.name} was not found "
        "relative to the parameter file, event JSON, script, or current directory."
    )


def _event_prefix_from_json(path: Path) -> str:
    stem = path.stem
    suffixes = (
        "_sim_fit_latest",
        "_sim_fit_dynesty_BestGuess",
        "_sim_fit",
        "_planet_run",
    )
    lower = stem.lower()
    for suffix in suffixes:
        if lower.endswith(suffix.lower()):
            return stem[: -len(suffix)]
    return stem


def discover_trajectory_pickle(
    input_json: Path,
    explicit: str | None = None,
) -> Path:
    """
    Locate the trajectory pickle required for a physical re-simulation.

    Priority:
      1. explicit --pickle;
      2. EVENT_trajectory.pickle beside the JSON;
      3. EVENT_trajectory.pickle one level above a fit_plots directory;
      4. one unambiguous *EVENT*trajectory*.pickle in those directories;
      5. one unambiguous *.pickle in the JSON directory.
    """
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Trajectory pickle not found: {path}")
        return path

    event = _event_prefix_from_json(input_json)
    search_dirs = [input_json.parent]
    if input_json.parent.name.lower() == "fit_plots":
        search_dirs.append(input_json.parent.parent)

    exact_names = (
        f"{event}_trajectory.pickle",
        f"{event}_trajectory.pkl",
        f"{event}.pickle",
        f"{event}.pkl",
    )
    for directory in search_dirs:
        lower_map = {
            p.name.lower(): p
            for p in directory.iterdir()
            if p.is_file()
        }
        for name in exact_names:
            match = lower_map.get(name.lower())
            if match is not None:
                return match.resolve()

    fuzzy: list[Path] = []
    for directory in search_dirs:
        for pattern in ("*.pickle", "*.pkl"):
            for path in directory.glob(pattern):
                low = path.name.lower()
                if event.lower() in low and "trajectory" in low:
                    fuzzy.append(path.resolve())
    fuzzy = sorted(set(fuzzy))
    if len(fuzzy) == 1:
        return fuzzy[0]
    if len(fuzzy) > 1:
        raise RuntimeError(
            "Multiple trajectory pickles match the event. Pass --pickle explicitly:\n  "
            + "\n  ".join(str(p) for p in fuzzy)
        )

    direct_pickles = []
    for pattern in ("*.pickle", "*.pkl"):
        direct_pickles.extend(input_json.parent.glob(pattern))
    direct_pickles = sorted(set(p.resolve() for p in direct_pickles))
    if len(direct_pickles) == 1:
        warnings.warn(
            f"No exact {event}_trajectory.pickle match was found; using the only "
            f"pickle beside the JSON: {direct_pickles[0].name}"
        )
        return direct_pickles[0]

    raise FileNotFoundError(
        f"A trajectory pickle is required for a physical run, but no matching "
        f"pickle was found for event {event!r} beside {input_json.name}. "
        "Expected e.g. EVENT_trajectory.pickle. Use --pickle PATH to override."
    )


def planet_float(
    params: dict[str, Any],
    key: str,
    default: float | None = None,
) -> float:
    value = params.get(key, default)
    if value is None:
        if default is None:
            raise ValueError(f"Planet parameter {key!r} must be numeric.")
        return float(default)
    out = finite_float(value)
    if not np.isfinite(out):
        raise ValueError(f"Planet parameter {key!r} must be numeric, got {value!r}.")
    return float(out)


def planet_string(
    params: dict[str, Any],
    key: str,
    default: str,
) -> str:
    value = params.get(key)
    return default if value is None else str(value).strip()


# =============================================================================
# Lazy MetSim/WMPL imports
# =============================================================================

_METSIM_DEPS: dict[str, Any] | None = None


def metsim_dependencies() -> dict[str, Any]:
    """Load the MetSim API used by this script.

    The current WMPL branch exposes the camelCase API from
    ``wmpl.Dynesty.DynestyMetSim``.  Compatibility aliases are deliberately
    provided because the original Mars_detection script used the older
    snake_case names.
    """
    global _METSIM_DEPS
    if _METSIM_DEPS is not None:
        return _METSIM_DEPS

    try:
        from wmpl.Dynesty.DynestyMetSim import (
            Constants,
            SimulationResults,
            loadConstants,
            loadPickle,
            ObservationData,
            runSimulation,
            integrateLuminosity,
            zenithAngleAtSimulationBegin,
        )
        try:
            from wmpl.Dynesty.DynestyMetSim import prepareSimulationPhotometry
        except Exception:
            prepareSimulationPhotometry = None
        api_name = "wmpl.Dynesty.DynestyMetSim"
    except Exception as camel_exc:
        # Keep the script usable with the older local research checkout too.
        try:
            from DynNestSapl_metsim import (
                Constants,
                SimulationResults,
                loadConstants,
                loadPickle,
                observation_data as ObservationData,
                runSimulation,
                luminosity_integration as integrateLuminosity,
                zenithAngleAtSimulationBegin,
            )
            prepareSimulationPhotometry = None
            api_name = "DynNestSapl_metsim"
        except Exception as old_exc:
            raise RuntimeError(
                "The full MetSim rerun requires WMPL/DynestyMetSim in the active "
                "Python environment. The current camelCase import failed with "
                f"{camel_exc!r}; the legacy import failed with {old_exc!r}."
            ) from old_exc

    from wmpl.MetSim.GUI import FragmentationEntry

    _METSIM_DEPS = {
        "api_name": api_name,
        "Constants": Constants,
        "SimulationResults": SimulationResults,
        "loadConstants": loadConstants,
        "loadPickle": loadPickle,
        "ObservationData": ObservationData,
        # aliases used by old code blocks
        "observation_data": ObservationData,
        "runSimulation": runSimulation,
        "integrateLuminosity": integrateLuminosity,
        "luminosity_integration": integrateLuminosity,
        "prepareSimulationPhotometry": prepareSimulationPhotometry,
        "zenithAngleAtSimulationBegin": zenithAngleAtSimulationBegin,
        "FragmentationEntry": FragmentationEntry,
    }
    return _METSIM_DEPS


# =============================================================================
# Constants and fragmentation reconstruction
# =============================================================================

def _as_numeric_array_if_possible(value: Any) -> Any:
    if isinstance(value, list):
        try:
            return np.asarray(value, dtype=float)
        except Exception:
            return value
    return value


def fragmentation_entry_from_mapping(entry_data: dict[str, Any]) -> Any:
    FragmentationEntry = metsim_dependencies()["FragmentationEntry"]

    frag_type = str(entry_data.get("frag_type", "")).strip().upper()
    if frag_type not in {"EF", "D", "M", "A", "F"}:
        raise ValueError(f"Unsupported fragmentation type {frag_type!r}")

    height = finite_float(entry_data.get("height"))
    if not np.isfinite(height):
        raise ValueError(f"Fragmentation entry has no finite height: {entry_data}")

    erosion_coeff = entry_data.get("erosion_coeff")
    if frag_type == "D" and erosion_coeff is None:
        erosion_coeff = 0.0

    entry = FragmentationEntry(
        frag_type,
        float(height),
        entry_data.get("number"),
        entry_data.get("mass_percent"),
        entry_data.get("sigma"),
        entry_data.get("gamma"),
        erosion_coeff,
        entry_data.get("grain_mass_min"),
        entry_data.get("grain_mass_max"),
        entry_data.get("mass_index"),
    )

    for field in ("id", "upward_only", "dyn_pressure"):
        if field in entry_data:
            try:
                setattr(entry, field, entry_data[field])
            except Exception:
                pass
    if hasattr(entry, "done"):
        entry.done = False
    return entry


def reset_fragmentation_entries(const: Any) -> list[Any]:
    rebuilt = []
    for item in list(getattr(const, "fragmentation_entries", []) or []):
        mapping = item if isinstance(item, dict) else vars(item)
        rebuilt.append(fragmentation_entry_from_mapping(mapping))
    rebuilt.sort(key=lambda e: float(e.height), reverse=True)
    const.fragmentation_entries = rebuilt
    const.fragmentation_on = bool(rebuilt)
    return rebuilt


def clone_fragment_at_height(fragment: Any, height_m: float) -> Any:
    mapping = dict(vars(fragment)) if not isinstance(fragment, dict) else dict(fragment)
    mapping["height"] = float(height_m)
    mapping["done"] = False
    return fragmentation_entry_from_mapping(mapping)


def load_best_fit_constants(json_path: str | Path) -> Any:
    deps = metsim_dependencies()
    Constants = deps["Constants"]
    loadConstants = deps["loadConstants"]

    json_path = str(json_path)
    data = load_json(json_path)
    const_dict = data.get("const", data)
    if not isinstance(const_dict, dict):
        raise ValueError(f"Could not find a constants dictionary in {json_path}")

    try:
        const, _ = loadConstants(json_path)
    except Exception:
        const = Constants()
        for key, value in const_dict.items():
            if key == "fragmentation_entries":
                continue
            setattr(const, key, _as_numeric_array_if_possible(value))
        const.fragmentation_entries = [
            fragmentation_entry_from_mapping(item)
            for item in (const_dict.get("fragmentation_entries", []) or [])
        ]

    if hasattr(const, "dens_co"):
        const.dens_co = np.asarray(const.dens_co, dtype=float)

    # Explicit JSON values remain the source of truth.
    for key, value in const_dict.items():
        if key == "fragmentation_entries":
            continue
        try:
            setattr(const, key, _as_numeric_array_if_possible(value))
        except Exception:
            pass

    const.fragmentation_entries = [
        fragmentation_entry_from_mapping(item)
        for item in (const_dict.get("fragmentation_entries", []) or [])
    ]
    reset_fragmentation_entries(const)
    return const


# =============================================================================
# Simulation helpers
# =============================================================================

def run_model_raw(const: Any) -> Any:
    deps = metsim_dependencies()
    runSimulation = deps["runSimulation"]
    SimulationResults = deps["SimulationResults"]

    run_const = copy.deepcopy(const)
    reset_fragmentation_entries(run_const)
    frag_main, results_list, wake_results = runSimulation(run_const, compute_wake=False)
    return SimulationResults(run_const, frag_main, results_list, wake_results)


def run_with_status(const: Any, label: str) -> Any:
    print(f"\n[{label}] running MetSim...")
    result = run_model_raw(const)
    print(f"[{label}] done: {len(np.asarray(result.time_arr))} simulation samples.")
    return result


def sim_profile(result: Any, abs_magnitude: np.ndarray | None = None) -> dict[str, np.ndarray]:
    if abs_magnitude is None:
        abs_magnitude = np.asarray(getattr(result, "abs_magnitude", []), dtype=float)
    else:
        abs_magnitude = np.asarray(abs_magnitude, dtype=float)

    time_s = np.asarray(getattr(result, "time_arr", []), dtype=float)
    height_m = np.asarray(getattr(result, "leading_frag_height_arr", []), dtype=float)
    length_m = np.asarray(
        getattr(result, "leading_frag_length_arr", np.full(len(time_s), np.nan)),
        dtype=float,
    )
    speed_mps = np.asarray(getattr(result, "leading_frag_vel_arr", []), dtype=float)
    luminosity_w = np.asarray(getattr(result, "luminosity_arr", []), dtype=float)
    dyn_pa = np.asarray(
        getattr(result, "leading_frag_dyn_press_arr", np.full(len(time_s), np.nan)),
        dtype=float,
    )
    main_mass = np.asarray(
        getattr(result, "main_mass_arr", np.full(len(time_s), np.nan)),
        dtype=float,
    )
    active_mass = np.asarray(
        getattr(result, "mass_total_active_arr", np.full(len(time_s), np.nan)),
        dtype=float,
    )

    arrays = [
        time_s, height_m, length_m, speed_mps, luminosity_w,
        abs_magnitude, dyn_pa, main_mass, active_mass,
    ]
    nonempty_lengths = [len(a) for a in arrays if len(a) > 0]
    n = min(nonempty_lengths) if nonempty_lengths else 0

    def cut(a: np.ndarray) -> np.ndarray:
        if len(a) == 0:
            return np.full(n, np.nan)
        return a[:n]

    return {
        "time_s": cut(time_s),
        "height_m": cut(height_m),
        "length_m": cut(length_m),
        "speed_mps": cut(speed_mps),
        "luminosity_w": cut(luminosity_w),
        "abs_magnitude": cut(abs_magnitude),
        "dynamic_pressure_pa": cut(dyn_pa),
        "main_mass_kg": cut(main_mass),
        "active_mass_kg": cut(active_mass),
    }


def dynamic_pressure_profile(result: Any) -> tuple[np.ndarray, np.ndarray]:
    p = sim_profile(result)
    h = p["height_m"]
    q = p["dynamic_pressure_pa"]

    finite_q = np.isfinite(q) & (q > 0.0)
    if np.count_nonzero(finite_q) >= 2:
        return h, q

    # Robust fallback from rho * v^2.
    rho = density_from_dens_co(h, np.asarray(result.const.dens_co, dtype=float))
    q = rho * np.square(p["speed_mps"])
    return h, q


def value_at_trigger_height(
    height_m: np.ndarray,
    values: np.ndarray,
    trigger_height_m: float,
) -> float:
    return interpolate_profile_at_height(height_m, values, trigger_height_m)


def highest_altitude_crossing(
    height_m: np.ndarray,
    values: np.ndarray,
    target_value: float,
    log_values: bool = False,
) -> tuple[float, float, str]:
    """
    Find the highest-altitude crossing of target_value.
    Falls back to the nearest profile point if no crossing exists.
    """
    h = np.asarray(height_m, dtype=float)
    v = np.asarray(values, dtype=float)
    n = min(len(h), len(v))
    h = h[:n]
    v = v[:n]

    good = np.isfinite(h) & np.isfinite(v)
    if log_values:
        good &= (v > 0.0) & (target_value > 0.0)
    h = h[good]
    v = v[good]
    if h.size == 0:
        raise ValueError("No finite profile values are available for trigger mapping.")

    vv = np.log(v) if log_values else v
    target = math.log(target_value) if log_values else float(target_value)

    candidates: list[tuple[float, float]] = []
    for i in range(len(h) - 1):
        a = vv[i] - target
        b = vv[i + 1] - target
        if a == 0.0:
            candidates.append((float(h[i]), float(v[i])))
            continue
        if a * b <= 0.0 and vv[i + 1] != vv[i]:
            frac = (target - vv[i]) / (vv[i + 1] - vv[i])
            hc = float(h[i] + frac * (h[i + 1] - h[i]))
            vc = float(v[i] + frac * (v[i + 1] - v[i]))
            candidates.append((hc, vc))

    if candidates:
        hc, vc = max(candidates, key=lambda item: item[0])
        return hc, vc, "profile_crossing"

    metric = np.abs(vv - target)
    idx = int(np.nanargmin(metric))
    return float(h[idx]), float(v[idx]), "nearest_profile_value"


def cumulative_received_energy_per_mass(result: Any) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate cumulative received aerodynamic energy per unit meteoroid mass [J/kg].

    This follows the same physical quantity used by the erosion-energy method in
    Mars_meteors.py, but evaluates it along every available simulation segment so
    that later erosion/fragmentation entries can also be mapped.

    dE/A = 0.5 * Lambda * v^2 * rho_atm * ds
    A/m  = shape_factor * rho_bulk^(-2/3) * m^(-1/3)

    The changing simulated speed and main-body mass are used where available.
    """
    p = sim_profile(result)
    h = p["height_m"]
    v = p["speed_mps"]
    length = p["length_m"]
    time = p["time_s"]
    mass = p["main_mass_kg"]

    n = min(len(h), len(v), len(time))
    if n < 2:
        raise ValueError("Too few simulation points for cumulative energy mapping.")

    h = h[:n]
    v = v[:n]
    time = time[:n]
    length = length[:n] if len(length) >= n else np.full(n, np.nan)
    mass = mass[:n] if len(mass) >= n else np.full(n, np.nan)

    rho_atm = density_from_dens_co(h, np.asarray(result.const.dens_co, dtype=float))

    bulk_rho = max(finite_float(getattr(result.const, "rho", np.nan), 1000.0), 1e-12)
    shape_factor = finite_float(getattr(result.const, "shape_factor", np.nan), 1.21)
    heat_transfer = 1.0

    fallback_mass = max(finite_float(getattr(result.const, "m_init", np.nan), 1e-12), 1e-12)
    mass_eff = np.where(np.isfinite(mass) & (mass > 0.0), mass, fallback_mass)

    if np.count_nonzero(np.isfinite(length)) >= 2:
        ds = np.abs(np.diff(length))
    else:
        dt = np.abs(np.diff(time))
        ds = 0.5 * (np.abs(v[:-1]) + np.abs(v[1:])) * dt

    v_mid = 0.5 * (np.abs(v[:-1]) + np.abs(v[1:]))
    rho_mid = np.sqrt(
        np.maximum(rho_atm[:-1], np.finfo(float).tiny)
        * np.maximum(rho_atm[1:], np.finfo(float).tiny)
    )
    mass_mid = np.sqrt(
        np.maximum(mass_eff[:-1], np.finfo(float).tiny)
        * np.maximum(mass_eff[1:], np.finfo(float).tiny)
    )

    dE_area = 0.5 * heat_transfer * np.square(v_mid) * rho_mid * ds
    area_over_mass = (
        shape_factor
        * bulk_rho ** (-2.0 / 3.0)
        * np.power(mass_mid, -1.0 / 3.0)
    )
    dE_mass = dE_area * area_over_mass
    dE_mass = np.where(np.isfinite(dE_mass) & (dE_mass >= 0.0), dE_mass, 0.0)

    cumulative = np.concatenate([[0.0], np.cumsum(dE_mass)])
    return h, cumulative


# =============================================================================
# Trigger mapping
# =============================================================================

def collect_trigger_specs(source_const: Any) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []

    for field in ("erosion_height_start", "erosion_height_change"):
        value = finite_float(getattr(source_const, field, np.nan))
        if np.isfinite(value) and value > 0.0:
            specs.append({
                "kind": "global_erosion",
                "field": field,
                "source_height_m": value,
            })

    for index, fragment in enumerate(list(getattr(source_const, "fragmentation_entries", []) or [])):
        h = finite_float(getattr(fragment, "height", np.nan))
        if np.isfinite(h) and h > 0.0:
            specs.append({
                "kind": "fragmentation_entry",
                "index": index,
                "frag_type": str(getattr(fragment, "frag_type", "")),
                "source_height_m": h,
            })

    return specs


def apply_trigger_heights(
    target_const: Any,
    source_const: Any,
    mappings: list[dict[str, Any]],
) -> Any:
    out = copy.deepcopy(target_const)

    for item in mappings:
        if item["kind"] == "global_erosion":
            setattr(out, item["field"], float(item["target_height_m"]))

    source_fragments = list(getattr(source_const, "fragmentation_entries", []) or [])
    mapped_by_index = {
        int(item["index"]): item
        for item in mappings
        if item["kind"] == "fragmentation_entry"
    }
    new_fragments = []
    for index, fragment in enumerate(source_fragments):
        if index in mapped_by_index:
            h = float(mapped_by_index[index]["target_height_m"])
        else:
            h = float(getattr(fragment, "height"))
        new_fragments.append(clone_fragment_at_height(fragment, h))

    new_fragments.sort(key=lambda e: float(e.height), reverse=True)
    out.fragmentation_entries = new_fragments
    out.fragmentation_on = bool(new_fragments)
    return out


def map_triggers_by_density(
    source_const: Any,
    target_base_const: Any,
    atmosphere: AtmosphereTable,
) -> tuple[Any, list[dict[str, Any]]]:
    mappings = []
    source_dens_co = np.asarray(source_const.dens_co, dtype=float)

    for spec in collect_trigger_specs(source_const):
        source_h = float(spec["source_height_m"])
        target_rho = float(density_from_dens_co(source_h, source_dens_co))
        target_h, status = height_for_density(atmosphere, target_rho)
        record = dict(spec)
        record.update({
            "method": "density",
            "source_value": target_rho,
            "source_value_units": "kg/m^3",
            "target_height_m": target_h,
            "matched_target_value": density_from_table(atmosphere, target_h),
            "mapping_status": status,
        })
        mappings.append(record)

    return apply_trigger_heights(target_base_const, source_const, mappings), mappings


def map_triggers_by_dynamic_pressure(
    source_const: Any,
    target_reference_const: Any,
    source_result: Any,
    target_reference_result: Any,
) -> tuple[Any, list[dict[str, Any]]]:
    source_h_prof, source_q = dynamic_pressure_profile(source_result)
    target_h_prof, target_q = dynamic_pressure_profile(target_reference_result)

    mappings = []
    for spec in collect_trigger_specs(source_const):
        source_h = float(spec["source_height_m"])

        # Prefer a saved dynamic pressure on the original fragmentation entry.
        target_value = np.nan
        if spec["kind"] == "fragmentation_entry":
            frag = list(getattr(source_const, "fragmentation_entries", []) or [])[int(spec["index"])]
            saved = finite_float(getattr(frag, "dyn_pressure", np.nan))
            if np.isfinite(saved) and saved > 0.0:
                target_value = saved

        if not np.isfinite(target_value):
            target_value = value_at_trigger_height(source_h_prof, source_q, source_h)

        target_h, matched, status = highest_altitude_crossing(
            target_h_prof,
            target_q,
            target_value,
            log_values=True,
        )

        record = dict(spec)
        record.update({
            "method": "dynamic_pressure",
            "source_value": float(target_value),
            "source_value_units": "Pa",
            "target_height_m": target_h,
            "matched_target_value": matched,
            "mapping_status": status,
        })
        mappings.append(record)

    return apply_trigger_heights(
        target_reference_const, source_const, mappings
    ), mappings


def map_triggers_by_energy(
    source_const: Any,
    target_reference_const: Any,
    source_result: Any,
    target_reference_result: Any,
) -> tuple[Any, list[dict[str, Any]]]:
    print(
        "\nENERGY TRIGGER: building cumulative received-energy profiles. "
        "This mode can take noticeably longer for long or fragmentation-rich runs."
    )
    source_h, source_energy = cumulative_received_energy_per_mass(source_result)
    target_h, target_energy = cumulative_received_energy_per_mass(target_reference_result)

    mappings = []
    for spec in collect_trigger_specs(source_const):
        source_trigger_h = float(spec["source_height_m"])
        target_value = value_at_trigger_height(
            source_h, source_energy, source_trigger_h
        )
        target_h_match, matched, status = highest_altitude_crossing(
            target_h,
            target_energy,
            target_value,
            log_values=False,
        )

        record = dict(spec)
        record.update({
            "method": "energy",
            "source_value": float(target_value),
            "source_value_units": "J/kg cumulative received energy",
            "target_height_m": target_h_match,
            "matched_target_value": matched,
            "mapping_status": status,
        })
        mappings.append(record)

    return apply_trigger_heights(
        target_reference_const, source_const, mappings
    ), mappings



# =============================================================================
# Trajectory orbit -> target-planet speed and zenith angle
# =============================================================================

def load_trajectory_pickle(pickle_path: str | Path) -> Any:
    loadPickle = metsim_dependencies()["loadPickle"]
    path = Path(pickle_path).expanduser().resolve()
    return loadPickle(str(path.parent), path.name)


def _angle_radians_to_degrees(value: float) -> float:
    value = float(value)
    return math.degrees(value)


def trajectory_orbit_elements(trajectory: Any) -> dict[str, float]:
    orbit = getattr(trajectory, "orbit", None)
    if orbit is None:
        raise ValueError("Trajectory pickle has no orbit object.")

    a_au = finite_float(getattr(orbit, "a", np.nan))
    e = finite_float(getattr(orbit, "e", np.nan))
    i_rad = finite_float(
        getattr(orbit, "i", getattr(orbit, "incl", np.nan))
    )
    peri_rad = finite_float(getattr(orbit, "peri", np.nan))
    node_rad = finite_float(getattr(orbit, "node", np.nan))

    vals = (a_au, e, i_rad, peri_rad, node_rad)
    if not all(np.isfinite(v) for v in vals):
        raise ValueError(
            "Trajectory orbit is missing one or more of a, e, i/incl, peri, node."
        )

    return {
        "a_au": float(a_au),
        "e": float(e),
        "i_deg": _angle_radians_to_degrees(i_rad),
        "peri_deg": _angle_radians_to_degrees(peri_rad),
        "node_deg": _angle_radians_to_degrees(node_rad),
        "earth_zc_rad": finite_float(getattr(orbit, "zc", np.nan)),
        "earth_rbeg_ele_m": finite_float(getattr(trajectory, "rbeg_ele", np.nan)),
    }


def keplerian_to_state_vectors(
    a_m: float,
    e: float,
    i_rad: float,
    peri_rad: float,
    node_rad: float,
    nu_rad: float,
    mu_sun: float = MU_SUN_M3_S2,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert Keplerian elements to heliocentric ecliptic state vectors."""
    p = a_m * (1.0 - e**2)
    if p <= 0.0:
        raise ValueError("Orbit has non-positive semi-latus rectum.")

    r = p / (1.0 + e * math.cos(nu_rad))
    r_pf = np.array([r * math.cos(nu_rad), r * math.sin(nu_rad), 0.0])
    v_pf = np.array([
        -math.sqrt(mu_sun / p) * math.sin(nu_rad),
        math.sqrt(mu_sun / p) * (e + math.cos(nu_rad)),
        0.0,
    ])

    cp = math.cos(peri_rad)
    sp = math.sin(peri_rad)
    cn = math.cos(node_rad)
    sn = math.sin(node_rad)
    ci = math.cos(i_rad)
    si = math.sin(i_rad)

    rotation = np.array([
        [cn*cp - sn*sp*ci, -cn*sp - sn*cp*ci, sn*si],
        [sn*cp + cn*sp*ci, -sn*sp + cn*cp*ci, -cn*si],
        [sp*si, cp*si, ci],
    ])
    return rotation.dot(r_pf), rotation.dot(v_pf)


def circular_planet_velocity_vector(
    r_vec: np.ndarray,
    mu_sun: float = MU_SUN_M3_S2,
) -> np.ndarray:
    """
    Circular prograde planet velocity at the supplied heliocentric position.

    The planet-parameter file intentionally uses a mean orbit_radius_au, matching
    the approximation in Mars_Vel.py.
    """
    r_vec = np.asarray(r_vec, dtype=float)
    r_mag = float(np.linalg.norm(r_vec))
    if r_mag <= 0.0:
        raise ValueError("Zero heliocentric position vector.")

    v_mag = math.sqrt(mu_sun / r_mag)
    r_xy = np.array([r_vec[0], r_vec[1], 0.0], dtype=float)
    xy_norm = float(np.linalg.norm(r_xy))
    if xy_norm <= 0.0:
        return np.array([0.0, v_mag, 0.0])

    z_axis = np.array([0.0, 0.0, 1.0])
    direction = np.cross(z_axis, r_xy / xy_norm)
    return direction * v_mag


def atmospheric_entry_speed_from_vinf(
    v_inf_mps: float,
    mu_planet_m3_s2: float,
    radius_m: float,
    altitude_m: float,
) -> float:
    r = float(radius_m) + float(altitude_m)
    if r <= 0.0:
        raise ValueError("Planet radius + entry altitude must be positive.")
    return math.sqrt(float(v_inf_mps)**2 + 2.0*float(mu_planet_m3_s2)/r)


def _orbit_radial_bounds_au(a_au: float, e: float) -> tuple[float, float]:
    return a_au * (1.0 - e), a_au * (1.0 + e)


def generic_planet_intercept(
    orbit_elements: dict[str, float],
    planet_params: dict[str, Any],
    start_height_m: float,
) -> dict[str, Any]:
    """
    Generic version of the supplied Mars_Vel method.

    The target planet is assumed to follow a circular, prograde ecliptic orbit at
    orbit_radius_au. Both radial crossings (outbound/inbound) are retained.
    """
    a_au = float(orbit_elements["a_au"])
    e = float(orbit_elements["e"])
    i_rad = math.radians(float(orbit_elements["i_deg"]))
    peri_rad = math.radians(float(orbit_elements["peri_deg"]))
    node_rad = math.radians(float(orbit_elements["node_deg"]))

    orbit_radius_au = planet_float(planet_params, "orbit_radius_au")
    q_au, Q_au = _orbit_radial_bounds_au(a_au, e)
    tolerance = planet_float(
        planet_params, "orbit_crossing_tolerance_au", 1.0e-6
    )
    if orbit_radius_au < q_au - tolerance or orbit_radius_au > Q_au + tolerance:
        raise ValueError(
            f"The meteoroid orbit does not reach {planet_params['planet_name']} "
            f"({orbit_radius_au:.6f} AU): q={q_au:.6f} AU, Q={Q_au:.6f} AU. "
            "Supply a numeric entry_speed_kms in the planet parameter file only "
            "if you intentionally want to override the orbital-intersection check."
        )

    a_m = a_au * AU_M
    target_r = orbit_radius_au * AU_M
    p_m = a_m * (1.0 - e**2)

    if abs(e) < 1.0e-12:
        if not math.isclose(a_au, orbit_radius_au, rel_tol=0.0, abs_tol=max(tolerance, 1e-8)):
            raise ValueError("Circular meteoroid orbit does not intersect target orbit.")
        nu_values = [-peri_rad]
        labels = ["circular_orbit_reference"]
    else:
        cos_nu = (p_m / target_r - 1.0) / e
        if abs(cos_nu) > 1.0 + 1.0e-10:
            raise ValueError(
                f"No real heliocentric radial crossing exists at {orbit_radius_au:.6f} AU."
            )
        nu_abs = math.acos(max(-1.0, min(1.0, cos_nu)))
        nu_values = [nu_abs, -nu_abs]
        labels = ["outbound", "inbound"]

    radius_m = planet_float(planet_params, "radius_km") * 1000.0
    mu_planet = planet_float(planet_params, "mu_m3_s2")

    scenarios = []
    for label, nu_rad in zip(labels, nu_values):
        r_obj, v_obj = keplerian_to_state_vectors(
            a_m, e, i_rad, peri_rad, node_rad, nu_rad, MU_SUN_M3_S2
        )
        v_planet = circular_planet_velocity_vector(r_obj, MU_SUN_M3_S2)
        v_inf_vec = v_obj - v_planet
        v_inf = float(np.linalg.norm(v_inf_vec))
        v_entry = atmospheric_entry_speed_from_vinf(
            v_inf, mu_planet, radius_m, start_height_m
        )
        scenarios.append({
            "label": label,
            "true_anomaly_deg": math.degrees(nu_rad),
            "object_heliocentric_speed_kms": float(np.linalg.norm(v_obj))/1000.0,
            "planet_heliocentric_speed_kms": float(np.linalg.norm(v_planet))/1000.0,
            "v_inf_kms": v_inf/1000.0,
            "entry_speed_at_simulation_start_kms": v_entry/1000.0,
        })

    return {
        "method": "internal_generic_3d_intercept",
        "scenarios": scenarios,
    }


def _select_intercept_speed(
    speeds_kms: np.ndarray,
    labels: list[str],
    selection: str,
) -> tuple[float, str]:
    speeds = np.asarray(speeds_kms, dtype=float)
    good = np.isfinite(speeds)
    speeds = speeds[good]
    good_labels = [lab for lab, ok in zip(labels, good) if ok]
    if len(speeds) == 0:
        raise ValueError("No finite target-planet entry speed was calculated.")

    selection = str(selection).strip().lower()
    if selection == "mean":
        return float(np.mean(speeds)), "mean_of_valid_intersections"
    if selection == "min":
        return float(np.min(speeds)), "minimum_valid_intersection"
    if selection == "max":
        return float(np.max(speeds)), "maximum_valid_intersection"
    if selection in {"outbound", "inbound"}:
        for speed, label in zip(speeds, good_labels):
            if str(label).lower() == selection:
                return float(speed), selection
        raise ValueError(
            f"Requested entry_speed_selection={selection!r}, but available "
            f"intersections are {good_labels}."
        )
    raise ValueError(
        "entry_speed_selection must be one of mean, min, max, outbound, inbound."
    )


def target_planet_kinematics(
    trajectory: Any,
    planet_params: dict[str, Any],
    start_height_m: float,
) -> dict[str, Any]:
    """
    Calculate target-planet entry speed and MetSim zenith angle from the pickle.

    For Mars, Mars_Vel.calculate_3d_intercept_speeds is preferred when importable.
    A generic implementation of the same circular-planet approximation is the
    fallback and is also used for other planetary bodies.
    """
    elements = trajectory_orbit_elements(trajectory)
    planet_name = planet_string(planet_params, "planet_name", "Planet")
    selection = planet_string(
        planet_params, "entry_speed_selection", "mean"
    ).lower()

    speed_override = planet_params.get("entry_speed_kms")
    zenith_override = planet_params.get("zenith_angle_deg")

    intercept: dict[str, Any]
    if speed_override is not None:
        selected_speed = planet_float(planet_params, "entry_speed_kms")
        intercept = {
            "method": "planet_parameter_override",
            "scenarios": [{
                "label": "override",
                "entry_speed_at_simulation_start_kms": selected_speed,
            }],
            "selected_speed_kms": selected_speed,
            "selection": "numeric_entry_speed_kms_parameter",
        }
    else:
        intercept = None
        if planet_name.strip().lower() == "mars":
            try:
                from Mars_Vel import calculate_3d_intercept_speeds
            except Exception:
                calculate_3d_intercept_speeds = None

            if calculate_3d_intercept_speeds is not None:
                try:
                    # Check q/Q before calling the historical Mars routine, because
                    # some old versions clamp a non-intersecting orbit instead of
                    # rejecting it.
                    q_au, Q_au = _orbit_radial_bounds_au(
                        elements["a_au"], elements["e"]
                    )
                    mars_r_au = planet_float(planet_params, "orbit_radius_au")
                    if not (q_au <= mars_r_au <= Q_au):
                        raise ValueError(
                            f"orbit does not cross Mars: q={q_au:.6f}, "
                            f"Q={Q_au:.6f}, Mars={mars_r_au:.6f} AU"
                        )

                    result = calculate_3d_intercept_speeds(
                        elements["a_au"],
                        elements["e"],
                        elements["i_deg"],
                        elements["peri_deg"],
                        elements["node_deg"],
                    )
                    if result is None or len(result) < 3:
                        raise ValueError("Mars_Vel returned no Mars intercept.")

                    v_inf_arr = np.asarray(result[1], dtype=float)
                    radius_m = planet_float(planet_params, "radius_km") * 1000.0
                    mu_planet = planet_float(planet_params, "mu_m3_s2")
                    start_speeds = np.asarray([
                        atmospheric_entry_speed_from_vinf(
                            v_inf*1000.0, mu_planet, radius_m, start_height_m
                        )/1000.0
                        for v_inf in v_inf_arr
                    ])
                    labels = ["outbound", "inbound"][:len(start_speeds)]
                    scenarios = []
                    for k, speed in enumerate(start_speeds):
                        scenarios.append({
                            "label": labels[k] if k < len(labels) else f"solution_{k}",
                            "v_inf_kms": float(v_inf_arr[k]),
                            "entry_speed_at_simulation_start_kms": float(speed),
                        })
                    intercept = {
                        "method": "Mars_Vel.calculate_3d_intercept_speeds",
                        "scenarios": scenarios,
                    }
                except Exception as exc:
                    warnings.warn(
                        f"Mars_Vel intercept calculation failed ({exc}); "
                        "using the internal generic calculation."
                    )
                    intercept = None

        if intercept is None:
            intercept = generic_planet_intercept(
                elements, planet_params, start_height_m
            )

        scenario_speeds = np.asarray([
            finite_float(item.get("entry_speed_at_simulation_start_kms"))
            for item in intercept["scenarios"]
        ])
        labels = [str(item.get("label", "")) for item in intercept["scenarios"]]
        selected_speed, selected_rule = _select_intercept_speed(
            scenario_speeds, labels, selection
        )
        intercept["selected_speed_kms"] = selected_speed
        intercept["selection"] = selected_rule

    # Zenith angle: preserve the existing WMPL transformation used by the old
    # Mars script, but recompute it at the target radius and target start height.
    if zenith_override is not None:
        zenith_rad = math.radians(planet_float(planet_params, "zenith_angle_deg"))
        zenith_method = "numeric_zenith_angle_deg_parameter"
    else:
        zc = float(elements["earth_zc_rad"])
        rbeg_ele = float(elements["earth_rbeg_ele_m"])
        if not (np.isfinite(zc) and np.isfinite(rbeg_ele)):
            raise ValueError(
                "Trajectory pickle lacks orbit.zc or rbeg_ele; set zenith_angle_deg "
                "in planet_parameters.txt to override."
            )
        zenith_func = metsim_dependencies()["zenithAngleAtSimulationBegin"]
        zenith_rad = float(
            zenith_func(
                float(start_height_m),
                rbeg_ele,
                zc,
                planet_float(planet_params, "radius_km")*1000.0,
            )
        )
        zenith_method = "zenithAngleAtSimulationBegin_from_trajectory_pickle"

    return {
        "planet_name": planet_name,
        "orbit_elements": elements,
        "intercept": intercept,
        "selected_entry_speed_kms": float(intercept["selected_speed_kms"]),
        "zenith_angle_rad": zenith_rad,
        "zenith_angle_deg": math.degrees(zenith_rad),
        "zenith_method": zenith_method,
    }

# =============================================================================
# New-planet constants
# =============================================================================


def build_target_base_constants(
    source_const: Any,
    atmosphere: AtmosphereTable,
    planet_params: dict[str, Any],
    trajectory: Any,
    args: argparse.Namespace,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """
    Build target-planet constants.

    Start altitude is either explicitly requested or found by matching the
    original JSON's atmospheric density at h_init. Speed and zenith angle are
    then recomputed from the required trajectory pickle unless numeric overrides
    are present in the planet parameter file.
    """
    out = copy.deepcopy(source_const)

    source_start_h = finite_float(getattr(source_const, "h_init", np.nan))
    if not np.isfinite(source_start_h):
        raise ValueError("Input JSON has no finite h_init.")

    source_start_rho = float(
        density_from_dens_co(
            source_start_h,
            np.asarray(source_const.dens_co, dtype=float),
        )
    )

    if args.start_height_km is not None:
        requested_h = float(args.start_height_km) * 1000.0
        target_h = float(np.clip(
            requested_h, atmosphere.min_height_m, atmosphere.max_height_m
        ))
        if not np.isclose(target_h, requested_h):
            start_status = "user_height_clamped_to_atmosphere_range"
            print(
                f"Requested start height {requested_h/1000:.3f} km is outside "
                f"the atmosphere table; using {target_h/1000:.3f} km."
            )
        else:
            start_status = "user_height"
    else:
        target_h, start_status = height_for_density(
            atmosphere, source_start_rho
        )

    out.h_init = float(target_h)
    out.dens_co = np.asarray(atmosphere.dens_co, dtype=float)
    out.G0 = planet_float(
        planet_params, "surface_gravity_m_s2"
    )
    out.r_earth = planet_float(
        planet_params, "radius_km"
    ) * 1000.0

    p0m = planet_params.get("meteor_zero_magnitude_power_W")
    if p0m is not None:
        out.P_0m = planet_float(
            planet_params, "meteor_zero_magnitude_power_W"
        )

    kinematics = target_planet_kinematics(
        trajectory=trajectory,
        planet_params=planet_params,
        start_height_m=float(out.h_init),
    )
    out.v_init = (
        float(kinematics["selected_entry_speed_kms"]) * 1000.0
    )
    out.zenith_angle = float(kinematics["zenith_angle_rad"])

    h_kill = planet_params.get("h_kill_km")
    if h_kill is None:
        out.h_kill = float(atmosphere.min_height_m)
    else:
        out.h_kill = planet_float(
            planet_params, "h_kill_km"
        ) * 1000.0
    if out.h_kill >= out.h_init:
        out.h_kill = max(
            atmosphere.min_height_m, out.h_init - 1000.0
        )

    v_kill = planet_params.get("v_kill_kms")
    if v_kill is not None:
        out.v_kill = planet_float(
            planet_params, "v_kill_kms"
        ) * 1000.0
    else:
        old_vkill = finite_float(getattr(out, "v_kill", np.nan))
        suggested = max(float(out.v_init) - 10_000.0, 2500.0)
        if np.isfinite(old_vkill) and old_vkill > 0.0:
            out.v_kill = min(old_vkill, suggested)
        else:
            out.v_kill = suggested

    dt_override = planet_params.get("simulation_dt_s")
    if dt_override is not None:
        out.dt = planet_float(planet_params, "simulation_dt_s")

    for field in (
        "erosion_beg_vel",
        "erosion_beg_mass",
        "erosion_beg_dyn_press",
        "mass_at_erosion_change",
        "energy_per_cs_before_erosion",
        "energy_per_mass_before_erosion",
        "main_mass_exhaustion_ht",
        "main_bottom_ht",
    ):
        if hasattr(out, field):
            try:
                setattr(out, field, None)
            except Exception:
                pass

    reset_fragmentation_entries(out)

    metadata = {
        "source_h_init_km": source_start_h / 1000.0,
        "source_initial_density_kg_m3": source_start_rho,
        "target_h_init_km": float(out.h_init) / 1000.0,
        "target_initial_density_kg_m3": float(
            density_from_dens_co(out.h_init, out.dens_co)
        ),
        "selection": start_status,
        "user_start_height_km": args.start_height_km,
    }
    return out, metadata, kinematics


# =============================================================================
# Optional observation data and finite-FPS integration
# =============================================================================

def load_observations(pickle_path: str | Path, fitted_p0m: float) -> Any:
    """Load the standard WMPL ObservationData object.

    This object is retained for backwards compatibility and ancillary fit
    metadata, but plotting and automatic FPS selection no longer depend on its
    camera-name heuristics.  All camera light curves are extracted directly
    from trajectory.observations below.
    """
    ObservationData = metsim_dependencies()["ObservationData"]
    return ObservationData(
        str(pickle_path),
        use_all_cameras=True,
        lag_noise_prior=40.0,
        lum_noise_prior=2.5,
        fps_prior=np.nan,
        P_0m_prior=float(fitted_p0m),
        pick_position=0.0,
        prior_file_path="",
    )


def estimate_camera_fps(time_s: np.ndarray) -> float:
    """Robustly estimate a camera FPS from its actual trajectory time stamps."""
    t = np.asarray(time_s, dtype=float)
    t = np.sort(t[np.isfinite(t)])
    if t.size < 2:
        return float("nan")

    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0.0)]
    if dt.size == 0:
        return float("nan")

    # Missing frames create integer multiples of the true exposure interval.
    # The median is therefore much safer than the mean.
    dt_med = float(np.nanmedian(dt))
    if not np.isfinite(dt_med) or dt_med <= 0.0:
        return float("nan")
    return 1.0 / dt_med


def extract_all_camera_lightcurves_from_trajectory(
    trajectory: Any,
    plot_faint_limit_mag: float = 8.0,
) -> list[dict[str, Any]]:
    """Extract every usable light curve directly from ``trajectory.observations``.

    No station-name whitelist is used.  A camera is accepted whether it is
    called 01G, 02G, a European Fireball Network station, or anything else.

    A camera is considered useful for the Earth-fit plot when it contains at
    least one finite measured absolute magnitude brighter than the same +8 mag
    faint limit used by the historical Mars_detection figure.  Cameras which
    contain only placeholder/faint values are still recorded in the JSON with
    ``used_for_plot=False`` so it is obvious that they were inspected rather
    than silently omitted.

    The ignore_list is NOT applied to the displayed points, matching the old
    plotting function.  It IS applied to the RMSD/FPS test so points rejected
    from the fit do not decide the integration mode.
    """
    cameras: list[dict[str, Any]] = []
    observations = list(getattr(trajectory, "observations", []) or [])

    for index, obs in enumerate(observations):
        station = str(getattr(obs, "station_id", f"camera_{index}")).strip()

        mags_raw = getattr(obs, "absolute_magnitudes", None)
        heights_raw = getattr(obs, "model_ht", None)
        times_raw = getattr(obs, "time_data", None)

        record: dict[str, Any] = {
            "station": station,
            "observation_index": index,
            "used_for_plot": False,
            "used_for_auto_integration": False,
            "reason": None,
            "fps_estimate": None,
            "absolute_magnitude": [],
            "height_km": [],
            "time_s": [],
            "fit_absolute_magnitude": [],
            "fit_height_km": [],
            "fit_time_s": [],
        }

        if mags_raw is None:
            record["reason"] = "absolute_magnitudes_is_None"
            cameras.append(record)
            continue
        if heights_raw is None or times_raw is None:
            record["reason"] = "missing_height_or_time_array"
            cameras.append(record)
            continue

        mags = np.asarray(mags_raw, dtype=float)
        heights_m = np.asarray(heights_raw, dtype=float)
        times = np.asarray(times_raw, dtype=float)
        n = min(len(mags), len(heights_m), len(times))
        mags = mags[:n]
        heights_m = heights_m[:n]
        times = times[:n]

        if n == 0:
            record["reason"] = "empty_lightcurve_arrays"
            cameras.append(record)
            continue

        ignore_raw = getattr(obs, "ignore_list", None)
        if ignore_raw is None:
            ignore = np.zeros(n, dtype=int)
        else:
            ignore = np.asarray(ignore_raw)[:n]
            if len(ignore) < n:
                ignore = np.pad(
                    ignore,
                    (0, n - len(ignore)),
                    mode="constant",
                    constant_values=0,
                )

        finite = (
            np.isfinite(mags)
            & np.isfinite(heights_m)
            & np.isfinite(times)
        )

        # Preserve the historical plot's +8 mag faint cut.  This also prevents
        # non-photometric placeholder tracks (often ~15--20 mag) from appearing
        # as if they were real light-curve constraints.
        plot_mask = finite & (mags < float(plot_faint_limit_mag))

        fps_est = estimate_camera_fps(times[finite])
        if np.isfinite(fps_est) and fps_est > 0:
            record["fps_estimate"] = float(fps_est)

        if not np.any(plot_mask):
            finite_mags = mags[finite]
            if finite_mags.size:
                record["finite_magnitude_range"] = [
                    float(np.nanmin(finite_mags)),
                    float(np.nanmax(finite_mags)),
                ]
            record["reason"] = (
                f"no_finite_points_brighter_than_{plot_faint_limit_mag:g}_mag"
            )
            cameras.append(record)
            continue

        record["used_for_plot"] = True
        record["absolute_magnitude"] = mags[plot_mask]
        record["height_km"] = heights_m[plot_mask] / 1000.0
        record["time_s"] = times[plot_mask]

        # For deciding how the source fit treated finite exposure time, use only
        # points which were not explicitly ignored in the trajectory solution.
        fit_mask = plot_mask & (ignore == 0)
        if np.count_nonzero(fit_mask) >= 2:
            record["used_for_auto_integration"] = True
            record["fit_absolute_magnitude"] = mags[fit_mask]
            record["fit_height_km"] = heights_m[fit_mask] / 1000.0
            record["fit_time_s"] = times[fit_mask]
            record["reason"] = "usable_lightcurve"
        else:
            record["reason"] = "plotted_but_too_few_nonignored_points_for_RMSD"

        cameras.append(record)

    return cameras


def integrated_lightcurve_arrays(
    result: Any,
    fps: float,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Return simulation luminosity/magnitude after finite-exposure integration."""
    luminosity = np.asarray(result.luminosity_arr, dtype=float)
    abs_mag = np.asarray(result.abs_magnitude, dtype=float)
    dt = finite_float(getattr(result.const, "dt", np.nan))

    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"Invalid integration FPS: {fps}")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("Simulation has no valid dt for FPS integration.")

    if (1.0 / float(fps)) <= dt:
        return luminosity.copy(), abs_mag.copy(), False

    integrate_luminosity = metsim_dependencies()["integrateLuminosity"]
    lum_i, mag_i = integrate_luminosity(
        np.asarray(result.time_arr, dtype=float),
        np.asarray(result.time_arr, dtype=float),
        luminosity,
        dt,
        float(fps),
        float(result.const.P_0m),
    )
    return np.asarray(lum_i, dtype=float), np.asarray(mag_i, dtype=float), True


def _sim_values_at_camera_heights(
    sim_result: Any,
    values: np.ndarray,
    camera_height_km: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate a simulation array and simulation time at camera heights."""
    sim_h = np.asarray(sim_result.leading_frag_height_arr, dtype=float)
    sim_t = np.asarray(sim_result.time_arr, dtype=float)
    values = np.asarray(values, dtype=float)
    camera_h_m = np.asarray(camera_height_km, dtype=float) * 1000.0

    n = min(len(sim_h), len(sim_t), len(values))
    sim_h = sim_h[:n]
    sim_t = sim_t[:n]
    values = values[:n]

    good = np.isfinite(sim_h) & np.isfinite(sim_t) & np.isfinite(values)
    sim_h = sim_h[good]
    sim_t = sim_t[good]
    values = values[good]
    if sim_h.size < 2:
        return (
            np.full(camera_h_m.shape, np.nan, dtype=float),
            np.full(camera_h_m.shape, np.nan, dtype=float),
        )

    order = np.argsort(sim_h)
    sim_h = sim_h[order]
    sim_t = sim_t[order]
    values = values[order]

    # Average exact duplicate heights instead of allowing np.interp to choose
    # one arbitrarily.
    unique_h, inv = np.unique(sim_h, return_inverse=True)
    if len(unique_h) != len(sim_h):
        time_sum = np.zeros(len(unique_h), dtype=float)
        value_sum = np.zeros(len(unique_h), dtype=float)
        counts = np.zeros(len(unique_h), dtype=float)
        np.add.at(time_sum, inv, sim_t)
        np.add.at(value_sum, inv, values)
        np.add.at(counts, inv, 1.0)
        sim_h = unique_h
        sim_t = time_sum / counts
        values = value_sum / counts

    within = (
        np.isfinite(camera_h_m)
        & (camera_h_m >= sim_h[0])
        & (camera_h_m <= sim_h[-1])
    )
    pred = np.full(camera_h_m.shape, np.nan, dtype=float)
    pred_time = np.full(camera_h_m.shape, np.nan, dtype=float)
    pred[within] = np.interp(camera_h_m[within], sim_h, values)
    pred_time[within] = np.interp(camera_h_m[within], sim_h, sim_t)
    return pred, pred_time


def _camera_rmsd_raw_and_integrated(
    sim_result: Any,
    camera: dict[str, Any],
    fps_override: float | None = None,
) -> dict[str, Any]:
    """Evaluate raw and finite-FPS source fits for one actual camera.

    Integration is evaluated at the simulation times corresponding to each
    camera measurement height.  This is more faithful than integrating the whole
    simulation on one global FPS grid and then interpolating the smeared curve.
    """
    obs_mag = np.asarray(
        camera.get("fit_absolute_magnitude", []), dtype=float
    )
    obs_h_km = np.asarray(
        camera.get("fit_height_km", []), dtype=float
    )
    n = min(len(obs_mag), len(obs_h_km))
    obs_mag = obs_mag[:n]
    obs_h_km = obs_h_km[:n]

    out = {
        "station": camera.get("station"),
        "n_points": int(n),
        "fps": None,
        "rmsd_nointegfps_mag": None,
        "rmsd_integfps_mag": None,
        "integration_applied": False,
    }
    if n < 2:
        return out

    raw_mag = np.asarray(sim_result.abs_magnitude, dtype=float)
    raw_pred, sim_times_at_obs = _sim_values_at_camera_heights(
        sim_result, raw_mag, obs_h_km
    )
    valid_raw = np.isfinite(obs_mag) & np.isfinite(raw_pred)
    if np.any(valid_raw):
        out["rmsd_nointegfps_mag"] = float(
            np.sqrt(np.mean((obs_mag[valid_raw] - raw_pred[valid_raw]) ** 2))
        )

    fps = finite_float(fps_override) if fps_override is not None else finite_float(
        camera.get("fps_estimate")
    )
    if not np.isfinite(fps) or fps <= 0.0:
        return out
    out["fps"] = float(fps)

    dt = finite_float(getattr(sim_result.const, "dt", np.nan))
    if not np.isfinite(dt) or dt <= 0.0 or (1.0 / fps) <= dt:
        # Exposure integration is unresolved at this simulation timestep.
        if out["rmsd_nointegfps_mag"] is not None:
            out["rmsd_integfps_mag"] = out["rmsd_nointegfps_mag"]
        return out

    good_times = np.isfinite(sim_times_at_obs) & np.isfinite(obs_mag)
    if np.count_nonzero(good_times) < 2:
        return out

    integrate_luminosity = metsim_dependencies()["integrateLuminosity"]
    try:
        _, integrated_mag = integrate_luminosity(
            np.asarray(sim_result.time_arr, dtype=float),
            sim_times_at_obs[good_times],
            np.asarray(sim_result.luminosity_arr, dtype=float),
            dt,
            fps,
            float(sim_result.const.P_0m),
        )
        integrated_mag = np.asarray(integrated_mag, dtype=float)
        finite_i = np.isfinite(integrated_mag) & np.isfinite(obs_mag[good_times])
        if np.any(finite_i):
            out["rmsd_integfps_mag"] = float(
                np.sqrt(
                    np.mean(
                        (
                            obs_mag[good_times][finite_i]
                            - integrated_mag[finite_i]
                        ) ** 2
                    )
                )
            )
            out["integration_applied"] = True
    except Exception as exc:
        out["integration_error"] = str(exc)

    return out


def rmsd_magnitude_at_observed_heights(
    sim_result: Any,
    sim_abs_mag: np.ndarray,
    obs_data: Any,
) -> float:
    """Legacy aggregate RMSD fallback when raw camera records are unavailable."""
    sim_h = np.asarray(sim_result.leading_frag_height_arr, dtype=float)
    sim_m = np.asarray(sim_abs_mag, dtype=float)
    obs_h = np.asarray(getattr(obs_data, "height_lum", []), dtype=float)
    obs_m = np.asarray(getattr(obs_data, "absolute_magnitudes", []), dtype=float)

    n_obs = min(len(obs_h), len(obs_m))
    obs_h = obs_h[:n_obs]
    obs_m = obs_m[:n_obs]
    good_obs = np.isfinite(obs_h) & np.isfinite(obs_m)
    obs_h = obs_h[good_obs]
    obs_m = obs_m[good_obs]
    if obs_h.size == 0:
        return float("nan")

    h_sorted, m_sorted = monotonic_unique_xy(sim_h, sim_m)
    good = np.isfinite(h_sorted) & np.isfinite(m_sorted)
    h_sorted = h_sorted[good]
    m_sorted = m_sorted[good]
    if h_sorted.size < 2:
        return float("nan")

    within = (obs_h >= h_sorted[0]) & (obs_h <= h_sorted[-1])
    if not np.any(within):
        return float("nan")

    pred = np.interp(obs_h[within], h_sorted, m_sorted)
    residual = obs_m[within] - pred
    return float(np.sqrt(np.mean(np.square(residual))))


def choose_integration_mode(
    requested_mode: str,
    source_result: Any | None,
    obs_data: Any | None,
    user_fps: float | None,
    camera_lightcurves: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Choose raw vs finite-FPS photometry using every usable camera.

    In auto mode, each camera is evaluated independently at its own measured FPS
    (estimated from its trajectory time stamps).  Camera RMSDs are then combined
    with EQUAL CAMERA WEIGHT, so a camera with many more frames cannot dominate
    the decision simply because it contributed more points.
    """
    requested = str(requested_mode).lower()
    result: dict[str, Any] = {
        "requested_mode": requested,
        "selected_mode": "nointegfps",
        "fps": None,
        "rmsd_nointegfps_mag": None,
        "rmsd_integfps_mag": None,
        "integration_applied": False,
        "selection_reason": None,
        "per_camera": [],
        "camera_fps_values": [],
    }

    cameras = [
        cam for cam in (camera_lightcurves or [])
        if cam.get("used_for_auto_integration")
    ]

    # Representative FPS for applying the chosen convention to the target-planet
    # light curve.  The comparison itself still uses each camera's own FPS.
    camera_fps = [
        finite_float(cam.get("fps_estimate"))
        for cam in cameras
        if np.isfinite(finite_float(cam.get("fps_estimate")))
        and finite_float(cam.get("fps_estimate")) > 0
    ]

    obs_fps = (
        finite_float(getattr(obs_data, "fps_lum", np.nan))
        if obs_data is not None else np.nan
    )
    if user_fps is not None and np.isfinite(finite_float(user_fps)) and finite_float(user_fps) > 0:
        representative_fps = float(user_fps)
    elif camera_fps:
        representative_fps = float(np.nanmedian(camera_fps))
    elif np.isfinite(obs_fps) and obs_fps > 0:
        representative_fps = float(obs_fps)
    else:
        representative_fps = np.nan

    if np.isfinite(representative_fps):
        result["fps"] = representative_fps
    result["camera_fps_values"] = [float(v) for v in camera_fps]

    if requested == "nointegfps":
        result["selection_reason"] = "explicit_user_choice"
        return result

    if requested == "integfps":
        if not np.isfinite(representative_fps) or representative_fps <= 0:
            raise ValueError(
                "--integration-mode integfps requires either --integration-fps "
                "or at least one camera with a measurable cadence."
            )
        result["selected_mode"] = "integfps"
        result["selection_reason"] = "explicit_user_choice"
        return result

    # AUTO: preferred path -- evaluate every usable raw trajectory camera.
    if source_result is not None and cameras:
        per_camera = []
        for cam in cameras:
            diagnostic = _camera_rmsd_raw_and_integrated(
                source_result,
                cam,
                fps_override=(
                    float(user_fps)
                    if user_fps is not None
                    and np.isfinite(finite_float(user_fps))
                    and finite_float(user_fps) > 0
                    else None
                ),
            )
            per_camera.append(diagnostic)

        result["per_camera"] = per_camera

        raw_vals = np.asarray([
            finite_float(item.get("rmsd_nointegfps_mag"))
            for item in per_camera
            if np.isfinite(finite_float(item.get("rmsd_nointegfps_mag")))
        ], dtype=float)
        int_vals = np.asarray([
            finite_float(item.get("rmsd_integfps_mag"))
            for item in per_camera
            if np.isfinite(finite_float(item.get("rmsd_integfps_mag")))
        ], dtype=float)

        # Equal-camera RMSD: sqrt(mean(camera_RMSD^2)).
        rms_raw = (
            float(np.sqrt(np.mean(raw_vals**2)))
            if raw_vals.size else np.nan
        )
        rms_int = (
            float(np.sqrt(np.mean(int_vals**2)))
            if int_vals.size else np.nan
        )
        result["rmsd_nointegfps_mag"] = (
            rms_raw if np.isfinite(rms_raw) else None
        )
        result["rmsd_integfps_mag"] = (
            rms_int if np.isfinite(rms_int) else None
        )
        result["integration_applied"] = bool(
            any(item.get("integration_applied") for item in per_camera)
        )

        if np.isfinite(rms_int) and (
            not np.isfinite(rms_raw) or rms_int < rms_raw
        ):
            result["selected_mode"] = "integfps"
            result["selection_reason"] = (
                "lower_equal_camera_weight_RMSD_using_each_camera_own_FPS"
            )
        elif np.isfinite(rms_raw):
            result["selected_mode"] = "nointegfps"
            result["selection_reason"] = (
                "lower_or_equal_equal_camera_weight_RMSD_without_integration"
            )
        else:
            result["selection_reason"] = (
                "auto_camera_RMSD_unavailable_falling_back"
            )
            # continue to legacy fallback below only if no valid camera metric
            if np.isfinite(rms_int):
                result["selected_mode"] = "integfps"
                return result

        if np.isfinite(rms_raw) or np.isfinite(rms_int):
            return result

    # Legacy fallback if the trajectory object did not expose usable individual
    # camera light curves.
    if source_result is None or obs_data is None:
        result["selection_reason"] = (
            "auto_without_usable_camera_photometry_defaults_to_nointegfps"
        )
        return result

    raw_mag = np.asarray(source_result.abs_magnitude, dtype=float)
    rms_raw = rmsd_magnitude_at_observed_heights(
        source_result, raw_mag, obs_data
    )
    result["rmsd_nointegfps_mag"] = (
        float(rms_raw) if np.isfinite(rms_raw) else None
    )

    if not np.isfinite(representative_fps) or representative_fps <= 0.0:
        result["selection_reason"] = "auto_no_valid_observation_fps"
        return result

    try:
        _, integrated_mag, was_applied = integrated_lightcurve_arrays(
            source_result, representative_fps
        )
        rms_int = rmsd_magnitude_at_observed_heights(
            source_result, integrated_mag, obs_data
        )
        result["rmsd_integfps_mag"] = (
            float(rms_int) if np.isfinite(rms_int) else None
        )
        result["integration_applied"] = bool(was_applied)

        if np.isfinite(rms_int) and (
            not np.isfinite(rms_raw) or rms_int < rms_raw
        ):
            result["selected_mode"] = "integfps"
            result["selection_reason"] = (
                "legacy_lower_RMSD_against_combined_observed_lightcurve"
            )
        else:
            result["selected_mode"] = "nointegfps"
            result["selection_reason"] = (
                "legacy_lower_or_equal_RMSD_without_integration"
            )
    except Exception as exc:
        result["selection_reason"] = f"integration_test_failed: {exc}"

    return result


def selected_lightcurve_arrays(
    result: Any,
    integration_info: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, bool]:
    if integration_info.get("selected_mode") != "integfps":
        return (
            np.asarray(result.luminosity_arr, dtype=float).copy(),
            np.asarray(result.abs_magnitude, dtype=float).copy(),
            False,
        )

    fps = finite_float(integration_info.get("fps"))
    if not np.isfinite(fps):
        raise ValueError("Selected integfps mode but no finite FPS is available.")
    return integrated_lightcurve_arrays(result, fps)


def observation_payload(
    obs_data: Any | None,
    camera_lightcurves: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Serialize observations, preferring direct all-camera trajectory extraction."""
    cameras = list(camera_lightcurves or [])

    if obs_data is None and not cameras:
        return None

    payload: dict[str, Any] = {
        "cameras": cameras,
        "all_station_ids": [str(cam.get("station")) for cam in cameras],
        "plotted_station_ids": [
            str(cam.get("station"))
            for cam in cameras
            if cam.get("used_for_plot")
        ],
        "auto_integration_station_ids": [
            str(cam.get("station"))
            for cam in cameras
            if cam.get("used_for_auto_integration")
        ],
    }

    # Flatten the directly extracted plot points as a backwards-compatible view.
    if cameras:
        flat_mag = []
        flat_h = []
        flat_t = []
        flat_station = []
        for cam in cameras:
            if not cam.get("used_for_plot"):
                continue
            mags = np.asarray(cam.get("absolute_magnitude", []), dtype=float)
            heights = np.asarray(cam.get("height_km", []), dtype=float)
            times = np.asarray(cam.get("time_s", []), dtype=float)
            n = min(len(mags), len(heights), len(times))
            if n == 0:
                continue
            flat_mag.extend(mags[:n].tolist())
            flat_h.extend(heights[:n].tolist())
            flat_t.extend(times[:n].tolist())
            flat_station.extend([str(cam.get("station"))] * n)

        payload.update({
            "height_km": flat_h,
            "absolute_magnitude": flat_mag,
            "time_s": flat_t,
            "stations": flat_station,
        })
    elif obs_data is not None:
        payload.update({
            "height_km": (
                np.asarray(getattr(obs_data, "height_lum", []), dtype=float)
                / 1000.0
            ),
            "absolute_magnitude": np.asarray(
                getattr(obs_data, "absolute_magnitudes", []), dtype=float
            ),
            "time_s": np.asarray(
                getattr(obs_data, "time_lum", []), dtype=float
            ),
        })
        stations = getattr(obs_data, "stations_lum", None)
        if stations is not None:
            payload["stations"] = [
                str(v) for v in np.asarray(stations).tolist()
            ]

    if obs_data is not None:
        payload["fps_lum_legacy"] = finite_float(
            getattr(obs_data, "fps_lum", np.nan)
        )
        payload["P_0m_W"] = finite_float(
            getattr(obs_data, "P_0m", np.nan)
        )

    return json_safe(payload)


# =============================================================================
# Output-run serialization
# =============================================================================

def fragmentation_entry_summary(entry: Any) -> dict[str, Any]:
    """Flatten only the physical/provenance fields needed to reproduce a release."""
    fields = (
        "frag_type", "height", "number", "mass_percent", "sigma", "gamma",
        "erosion_coeff", "grain_mass_min", "grain_mass_max", "mass_index",
        "id", "upward_only", "dyn_pressure", "done",
    )
    if isinstance(entry, dict):
        source = entry
        getter = source.get
    else:
        getter = lambda key, default=None: getattr(entry, key, default)

    out: dict[str, Any] = {}
    for key in fields:
        value = getter(key, None)
        if value is not None:
            out[key] = json_safe(value)
    return out


def constants_summary(const: Any) -> dict[str, Any]:
    """Serialize only MetSim inputs/results needed for reuse, never the object graph."""
    keep = (
        "dt", "h_init", "h_kill", "v_init", "v_kill", "m_init", "rho",
        "shape_factor", "sigma", "zenith_angle", "gamma", "rho_grain",
        "lum_eff_type", "lum_eff", "P_0m", "G0", "r_earth",
        "erosion_on", "erosion_height_start", "erosion_height_change",
        "erosion_coeff", "erosion_coeff_change", "erosion_rho_change",
        "erosion_sigma_change", "erosion_mass_index",
        "erosion_mass_min", "erosion_mass_max", "fragmentation_on",
    )
    out: dict[str, Any] = {}
    for key in keep:
        if hasattr(const, key):
            out[key] = json_safe(getattr(const, key))

    out["fragmentation_entries"] = [
        fragmentation_entry_summary(entry)
        for entry in list(getattr(const, "fragmentation_entries", []) or [])
    ]
    out["dens_co"] = json_safe(
        np.asarray(getattr(const, "dens_co", []), dtype=float)
    )
    return out


def serializable_run(
    result: Any,
    selected_luminosity: np.ndarray,
    selected_abs_mag: np.ndarray,
) -> dict[str, Any]:
    raw = sim_profile(result)
    selected = sim_profile(result, abs_magnitude=selected_abs_mag)
    return {
        "constants": constants_summary(result.const),
        "arrays": {
            "time_s": selected["time_s"],
            "height_km": selected["height_m"] / 1000.0,
            "length_km": selected["length_m"] / 1000.0,
            "speed_kms": selected["speed_mps"] / 1000.0,
            "luminosity_W_raw": raw["luminosity_w"],
            "luminosity_W_selected": np.asarray(
                selected_luminosity, dtype=float
            )[:len(selected["time_s"])],
            "absolute_magnitude_raw": raw["abs_magnitude"],
            "absolute_magnitude_selected": selected["abs_magnitude"],
            "dynamic_pressure_Pa": selected["dynamic_pressure_pa"],
            "main_mass_kg": selected["main_mass_kg"],
            "active_mass_kg": selected["active_mass_kg"],
        },
        "summary": summarize_saved_arrays({
            "height_km": selected["height_m"] / 1000.0,
            "speed_kms": selected["speed_mps"] / 1000.0,
            "absolute_magnitude_selected": selected["abs_magnitude"],
            "time_s": selected["time_s"],
        }),
    }


def summarize_saved_arrays(arrays: dict[str, Any]) -> dict[str, Any]:
    h = np.asarray(arrays.get("height_km", []), dtype=float)
    speed = np.asarray(arrays.get("speed_kms", []), dtype=float)
    mag = np.asarray(arrays.get("absolute_magnitude_selected", []), dtype=float)
    time = np.asarray(arrays.get("time_s", []), dtype=float)

    summary: dict[str, Any] = {}
    if np.any(np.isfinite(mag)):
        idx = int(np.nanargmin(mag))
        summary["peak_absolute_magnitude"] = float(mag[idx])
        if idx < len(h) and np.isfinite(h[idx]):
            summary["peak_height_km"] = float(h[idx])
        if idx < len(speed) and np.isfinite(speed[idx]):
            summary["speed_at_peak_kms"] = float(speed[idx])
    if np.any(np.isfinite(h)):
        summary["begin_height_km"] = float(h[np.flatnonzero(np.isfinite(h))[0]])
        summary["end_height_km"] = float(h[np.flatnonzero(np.isfinite(h))[-1]])
    if np.any(np.isfinite(speed)):
        summary["initial_speed_kms"] = float(speed[np.flatnonzero(np.isfinite(speed))[0]])
        summary["final_speed_kms"] = float(speed[np.flatnonzero(np.isfinite(speed))[-1]])
    if np.count_nonzero(np.isfinite(time)) >= 2:
        valid = time[np.isfinite(time)]
        summary["simulated_duration_s"] = float(valid[-1] - valid[0])
    return summary


# =============================================================================
# Observer / limiting magnitude post-processing
# =============================================================================

def build_local_3d_track(
    run_arrays: dict[str, Any],
    planet_radius_km: float,
    zenith_angle_rad: float,
) -> np.ndarray:
    """
    Construct a local Mars-centred 3D track without requiring a trajectory pickle.

    The atmospheric entry point is placed on +X. The simulated height is kept
    exactly, while along-track horizontal motion is derived from simulated path
    length and the JSON zenith angle.
    """
    h = np.asarray(run_arrays["height_km"], dtype=float)
    length = np.asarray(run_arrays.get("length_km", []), dtype=float)

    n = len(h)
    if len(length) != n or np.count_nonzero(np.isfinite(length)) < 2:
        speed = np.asarray(run_arrays.get("speed_kms", []), dtype=float)
        time = np.asarray(run_arrays.get("time_s", []), dtype=float)
        if len(speed) == n and len(time) == n and n >= 2:
            dt = np.diff(time)
            ds = 0.5 * (speed[:-1] + speed[1:]) * dt
            length = np.concatenate([[0.0], np.cumsum(np.nan_to_num(ds))])
        else:
            length = np.arange(n, dtype=float)

    length0 = float(length[np.flatnonzero(np.isfinite(length))[0]]) if np.any(np.isfinite(length)) else 0.0
    along = np.nan_to_num(length - length0)
    horizontal = along * math.sin(float(zenith_angle_rad))

    # Move along a local great-circle while preserving the simulated radial
    # altitude exactly. This avoids artificially increasing altitude when a
    # long horizontal trail is represented in Cartesian coordinates.
    radius = float(planet_radius_km) + h
    theta = horizontal / np.maximum(radius, 1.0e-12)
    positions = np.column_stack([
        radius * np.cos(theta),
        radius * np.sin(theta),
        np.zeros(n, dtype=float),
    ])
    return positions


def apparent_magnitude(abs_mag: np.ndarray, range_km: np.ndarray) -> np.ndarray:
    abs_mag = np.asarray(abs_mag, dtype=float)
    range_km = np.asarray(range_km, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return abs_mag + 5.0 * np.log10(range_km / 100.0)


def sample_camera_frames(
    time_s: np.ndarray,
    values: dict[str, np.ndarray],
    fps: float,
    seed: int,
) -> tuple[np.ndarray, dict[str, np.ndarray], float]:
    time_s = np.asarray(time_s, dtype=float)
    good = np.isfinite(time_s)
    time_s = time_s[good]
    if len(time_s) < 2:
        return np.asarray([], dtype=float), {
            key: np.asarray([], dtype=float) for key in values
        }, 0.0

    order = np.argsort(time_s)
    time_sorted = time_s[order]
    frame_dt = 1.0 / float(fps)
    rng = np.random.default_rng(int(seed))
    phase = float(rng.uniform(0.0, frame_dt))
    first = float(time_sorted[0] + phase)
    if first > time_sorted[-1]:
        return np.asarray([], dtype=float), {
            key: np.asarray([], dtype=float) for key in values
        }, phase

    sample_times = np.arange(first, time_sorted[-1] + 0.5 * frame_dt, frame_dt)

    sampled: dict[str, np.ndarray] = {}
    original_indices = np.flatnonzero(good)[order]
    for key, arr in values.items():
        arr = np.asarray(arr, dtype=float)
        arr = arr[original_indices]
        sampled[key] = np.interp(sample_times, time_sorted, arr)

    return sample_times, sampled, phase


def compute_detection(
    saved_run: dict[str, Any],
    observer_altitude_km: float,
    limiting_mag: float,
    camera_fps: float,
    minimum_frames: int,
    sampling_seed: int,
    planet_radius_km: float,
    planet_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    arrays = saved_run["arrays"]
    abs_mag = np.asarray(
        arrays["absolute_magnitude_selected"], dtype=float
    )
    h = np.asarray(arrays["height_km"], dtype=float)
    time_s = np.asarray(arrays["time_s"], dtype=float)
    speed = np.asarray(
        arrays.get("speed_kms", []), dtype=float
    )

    zenith = finite_float(
        saved_run.get("constants", {}).get("zenith_angle"), 0.0
    )
    positions = build_local_3d_track(
        arrays, planet_radius_km, zenith
    )
    # The observer input is an ALTITUDE above the target body's surface.
    # For the local nadir geometry used here, place the observer on +X directly
    # above the nominal entry point.  This makes 0 km a surface observer and,
    # for example, 5720 km a spacecraft 5720 km above Mars.
    observer_altitude_km = float(observer_altitude_km)
    if not np.isfinite(observer_altitude_km):
        raise ValueError("Observer altitude must be finite.")
    if observer_altitude_km < 0.0:
        raise ValueError("Observer altitude cannot be negative.")

    observer = np.asarray(
        [float(planet_radius_km) + observer_altitude_km, 0.0, 0.0],
        dtype=float,
    )

    ranges = np.linalg.norm(
        positions - observer[None, :], axis=1
    )
    geometric_app_mag = apparent_magnitude(abs_mag, ranges)

    params = planet_params or {}
    observer_radius = float(np.linalg.norm(observer))
    # observer_altitude_km is already the user-facing altitude above the surface.
    ground_limit_km = finite_float(
        params.get("ground_observer_max_altitude_km"), 20.0
    )
    is_ground = bool(
        observer_radius > 0.0
        and observer_altitude_km <= ground_limit_km
    )

    elevation_deg = np.full(len(positions), np.nan, dtype=float)
    magnitude_penalty = np.zeros(len(positions), dtype=float)
    ground_flux_factor = np.ones(len(positions), dtype=float)
    above_horizon = np.ones(len(positions), dtype=bool)

    if is_ground:
        radial_hat = observer / observer_radius
        los = positions - observer[None, :]
        los_norm = np.linalg.norm(los, axis=1)
        good = los_norm > 0.0
        los_hat = np.zeros_like(los)
        los_hat[good] = los[good] / los_norm[good, None]
        sin_elevation = np.sum(
            los_hat * radial_hat[None, :], axis=1
        )
        sin_elevation = np.clip(
            sin_elevation, -1.0, 1.0
        )
        elevation_deg = np.degrees(
            np.arcsin(sin_elevation)
        )
        above_horizon = elevation_deg > 0.0

        # These two quantities default to 1.0 in the planet file, hence no
        # ground-atmosphere/seeing penalty unless the user intentionally changes
        # them.
        transmission_zenith = float(np.clip(
            finite_float(
                params.get(
                    "ground_atmospheric_transmission_zenith"
                ),
                1.0,
            ),
            np.finfo(float).tiny,
            1.0,
        ))
        seeing_factor = float(np.clip(
            finite_float(
                params.get(
                    "ground_seeing_sensitivity_factor"
                ),
                1.0,
            ),
            np.finfo(float).tiny,
            1.0,
        ))
        max_airmass = max(
            finite_float(
                params.get("ground_max_airmass"), 20.0
            ),
            1.0,
        )

        positive_sin = np.maximum(
            np.sin(np.radians(np.maximum(elevation_deg, 0.0))),
            1.0/max_airmass,
        )
        airmass = np.minimum(
            1.0/positive_sin, max_airmass
        )
        ground_flux_factor = (
            seeing_factor
            * np.power(transmission_zenith, airmass)
        )
        ground_flux_factor = np.clip(
            ground_flux_factor,
            np.finfo(float).tiny,
            1.0,
        )
        magnitude_penalty = (
            -2.5*np.log10(ground_flux_factor)
        )

    app_mag = geometric_app_mag + magnitude_penalty
    app_mag = np.where(
        above_horizon, app_mag, np.inf
    )

    sample_times, sampled, phase = sample_camera_frames(
        time_s,
        {
            "apparent_magnitude": app_mag,
            "geometric_apparent_magnitude": geometric_app_mag,
            "absolute_magnitude": abs_mag,
            "height_km": h,
            "range_km": ranges,
            "speed_kms": (
                speed if len(speed) == len(time_s)
                else np.full(len(time_s), np.nan)
            ),
            "elevation_deg": elevation_deg,
            "ground_flux_factor": ground_flux_factor,
        },
        fps=float(camera_fps),
        seed=int(sampling_seed),
    )

    visible = (
        sampled["apparent_magnitude"]
        <= float(limiting_mag)
    )
    n_visible = int(np.count_nonzero(visible))

    finite_app = np.isfinite(app_mag)
    if np.any(finite_app):
        candidate = np.where(
            finite_app, app_mag, np.inf
        )
        peak_idx = int(np.argmin(candidate))
        peak = {
            "apparent_magnitude": float(app_mag[peak_idx]),
            "geometric_apparent_magnitude": float(
                geometric_app_mag[peak_idx]
            ),
            "absolute_magnitude": float(abs_mag[peak_idx]),
            "height_km": float(h[peak_idx]),
            "range_km": float(ranges[peak_idx]),
            "elevation_deg": (
                float(elevation_deg[peak_idx])
                if np.isfinite(elevation_deg[peak_idx])
                else None
            ),
            "ground_atmosphere_magnitude_penalty": float(
                magnitude_penalty[peak_idx]
            ),
        }
    else:
        peak = {}

    # Quantify how far the event is from the supplied limiting magnitude.
    # Positive delta_mag_to_limit means the event is fainter than the limit by
    # that many magnitudes. Negative means it is brighter than the limit.
    finite_cont = app_mag[np.isfinite(app_mag)]
    peak_cont_mag = float(np.nanmin(finite_cont)) if finite_cont.size else float("nan")
    finite_sample = sampled["apparent_magnitude"][
        np.isfinite(sampled["apparent_magnitude"])
    ]
    peak_sample_mag = (
        float(np.nanmin(finite_sample)) if finite_sample.size else float("nan")
    )
    delta_cont = (
        peak_cont_mag - float(limiting_mag)
        if np.isfinite(peak_cont_mag) else float("nan")
    )
    delta_sample = (
        peak_sample_mag - float(limiting_mag)
        if np.isfinite(peak_sample_mag) else float("nan")
    )

    if np.isfinite(delta_sample):
        if delta_sample > 0.0:
            detectability_text = (
                f"Not detectable at sampled frames: closest frame is "
                f"{delta_sample:.2f} mag fainter than LM; "
                f"LM would need to be about {peak_sample_mag:.2f} or fainter."
            )
        else:
            detectability_text = (
                f"Detectable: brightest sampled frame is "
                f"{-delta_sample:.2f} mag brighter than LM."
            )
    elif np.isfinite(delta_cont):
        if delta_cont > 0.0:
            detectability_text = (
                f"Not detectable: model peak is {delta_cont:.2f} mag fainter "
                f"than LM; LM would need to be about {peak_cont_mag:.2f} or fainter."
            )
        else:
            detectability_text = (
                f"Continuous model peak is {-delta_cont:.2f} mag brighter than LM."
            )
    else:
        detectability_text = "No finite apparent-magnitude samples were available."

    return {
        "observer_position_planet_centered_km": observer,
        # Legacy alias retained so old result JSONs/scripts remain readable.
        "observer_position_mars_centered_km": observer,
        "observer_altitude_km": observer_altitude_km,
        "observer_radius_from_planet_center_km": float(planet_radius_km) + observer_altitude_km,
        "observer_classification": (
            "ground" if is_ground else "space"
        ),
        "geometry_note": (
            "Entry point is placed on +X; simulated height is exact and "
            "horizontal motion is reconstructed from path length and the "
            "target-planet zenith angle."
        ),
        "ground_atmosphere": {
            "applied": is_ground,
            "transmission_zenith": finite_float(
                params.get(
                    "ground_atmospheric_transmission_zenith"
                ),
                1.0,
            ),
            "seeing_sensitivity_factor": finite_float(
                params.get(
                    "ground_seeing_sensitivity_factor"
                ),
                1.0,
            ),
            "ground_observer_max_altitude_km": ground_limit_km,
            "note": (
                "Both factors default to 1.0, which gives zero magnitude "
                "penalty. Transmission is applied with a simple secant-like "
                "airmass capped by ground_max_airmass; seeing factor is a "
                "multiplicative sensitivity factor."
            ),
        },
        "limiting_apparent_magnitude": float(limiting_mag),
        "camera_fps": float(camera_fps),
        "minimum_frames": int(minimum_frames),
        "sampling_seed": int(sampling_seed),
        "sampling_phase_s": float(phase),
        "continuous": {
            "time_s": time_s,
            "height_km": h,
            "absolute_magnitude": abs_mag,
            "range_km": ranges,
            "geometric_apparent_magnitude": geometric_app_mag,
            "apparent_magnitude": app_mag,
            "elevation_deg": elevation_deg,
            "ground_flux_factor": ground_flux_factor,
            "ground_atmosphere_magnitude_penalty": magnitude_penalty,
            "position_planet_centered_km": positions,
        },
        "sampled_frames": {
            "time_s": sample_times,
            "height_km": sampled["height_km"],
            "speed_kms": sampled["speed_kms"],
            "range_km": sampled["range_km"],
            "absolute_magnitude": sampled["absolute_magnitude"],
            "geometric_apparent_magnitude": sampled[
                "geometric_apparent_magnitude"
            ],
            "apparent_magnitude": sampled["apparent_magnitude"],
            "elevation_deg": sampled["elevation_deg"],
            "ground_flux_factor": sampled["ground_flux_factor"],
            "above_limiting_magnitude": visible,
            "visible_frame_count": n_visible,
        },
        "peak": peak,
        "detectability": {
            "continuous_peak_apparent_magnitude": (
                peak_cont_mag if np.isfinite(peak_cont_mag) else None
            ),
            "brightest_sampled_frame_apparent_magnitude": (
                peak_sample_mag if np.isfinite(peak_sample_mag) else None
            ),
            "delta_mag_to_limit_continuous": (
                delta_cont if np.isfinite(delta_cont) else None
            ),
            "delta_mag_to_limit_sampled": (
                delta_sample if np.isfinite(delta_sample) else None
            ),
            "required_limiting_magnitude_for_brightest_sampled_frame": (
                peak_sample_mag if np.isfinite(peak_sample_mag) else None
            ),
            "message": detectability_text,
        },
        "has_visible_frames": bool(n_visible > 0),
        "meets_minimum_frames": bool(
            n_visible >= int(minimum_frames)
        ),
    }


# =============================================================================
# Plotting
# =============================================================================

def _plot_arrays_from_run(run: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    arrays = run.get("arrays", {})
    mag = np.asarray(arrays.get("absolute_magnitude_selected", []), dtype=float)
    height = np.asarray(arrays.get("height_km", []), dtype=float)
    n = min(len(mag), len(height))
    mag = mag[:n]
    height = height[:n]
    good = np.isfinite(mag) & np.isfinite(height)
    return mag[good], height[good]


def plot_lightcurve_earth_vs_planet(
    result_payload: dict[str, Any],
    output_dir: Path,
    base_name: str,
    plot_format: str,
) -> list[str]:
    """Reproduce the old Mars_detection light-curve plot with ground stations.

    This keeps the original single-panel absolute-magnitude/height style and
    explicitly shows the observed ground-camera curves (e.g. 01G, 02G, ... )
    whenever they are available in the loaded observation data or cached JSON.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    planet_name = str(result_payload.get("configuration", {}).get("planet_name", "Planet"))

    fig, ax = plt.subplots(figsize=(6, 6))
    station_colors: dict[str, Any] = {}
    cmap = plt.get_cmap("tab10")

    obs = result_payload.get("observations") or {}
    cameras = list(obs.get("cameras") or [])

    # Direct camera records are the source of truth.  There is deliberately no
    # whitelist such as 01G/02G/01T/02T: if a trajectory observation contains a
    # real light curve, its actual station ID is plotted.
    plotted_heights: list[float] = []
    if cameras:
        for cam in cameras:
            if not cam.get("used_for_plot"):
                continue
            station = str(cam.get("station", "camera"))
            mags = np.asarray(cam.get("absolute_magnitude", []), dtype=float)
            heights = np.asarray(cam.get("height_km", []), dtype=float)
            n_cam = min(len(mags), len(heights))
            mags = mags[:n_cam]
            heights = heights[:n_cam]
            good = np.isfinite(mags) & np.isfinite(heights)
            if not np.any(good):
                continue

            if station not in station_colors:
                station_colors[station] = cmap(len(station_colors) % 10)
            ax.plot(
                mags[good],
                heights[good],
                "x--",
                color=station_colors[station],
                label=station,
                linewidth=1.0,
                markersize=5.5,
                markeredgewidth=1.1,
                alpha=0.95,
                zorder=5,
            )
            plotted_heights.extend(heights[good].tolist())

        have_obs = bool(plotted_heights)
        if have_obs:
            y_min, y_max = ax.get_ylim()
        else:
            y_min = y_max = np.nan

    else:
        # Backwards-compatible fallback for an old saved result JSON.
        obs_mag = np.asarray(obs.get("absolute_magnitude", []), dtype=float)
        obs_h = np.asarray(obs.get("height_km", []), dtype=float)
        stations = obs.get("stations")
        n_obs = min(len(obs_mag), len(obs_h))
        obs_mag = obs_mag[:n_obs]
        obs_h = obs_h[:n_obs]

        if stations is not None and len(stations) >= n_obs and n_obs > 0:
            stations_arr = np.asarray(stations[:n_obs], dtype=str)
            for station in np.unique(stations_arr):
                mask = (
                    (stations_arr == station)
                    & np.isfinite(obs_mag)
                    & np.isfinite(obs_h)
                )
                if not np.any(mask):
                    continue
                if station not in station_colors:
                    station_colors[station] = cmap(len(station_colors) % 10)
                ax.plot(
                    obs_mag[mask],
                    obs_h[mask],
                    "x--",
                    color=station_colors[station],
                    label=station,
                    linewidth=1.0,
                    markersize=5.5,
                    markeredgewidth=1.1,
                    alpha=0.95,
                    zorder=5,
                )
        elif n_obs > 0:
            good = np.isfinite(obs_mag) & np.isfinite(obs_h)
            if np.any(good):
                ax.plot(
                    obs_mag[good],
                    obs_h[good],
                    "x--",
                    color=cmap(0),
                    label="Observed data",
                    linewidth=1.0,
                    markersize=5.5,
                    markeredgewidth=1.1,
                    alpha=0.95,
                    zorder=5,
                )

        have_obs = bool(n_obs > 0 and np.any(np.isfinite(obs_h)))
        if have_obs:
            y_min, y_max = ax.get_ylim()
        else:
            y_min = y_max = np.nan

    # Optional synthetic overlays retained for compatibility with the previous
    # synthetic-speed workflow.
    synthetic_curve_records = list(result_payload.get("synthetic_curve_records") or [])
    synthetic_speed_values = [
        finite_float(record.get("synthetic_speed_kms"))
        for record in synthetic_curve_records
        if isinstance(record, dict) and record.get("lightcurve_overlay")
    ]
    synthetic_speed_values = [v for v in synthetic_speed_values if np.isfinite(v)]
    synth_norm = None
    synth_cmap = None
    first_detected_label = False
    first_nondetected_label = False
    if synthetic_speed_values:
        synth_cmap = plt.get_cmap("viridis")
        vmin = float(np.nanmin(synthetic_speed_values))
        vmax = float(np.nanmax(synthetic_speed_values))
        if np.isfinite(vmin) and np.isfinite(vmax):
            if np.isclose(vmin, vmax):
                synth_norm = plt.Normalize(vmin=vmin - 1.0, vmax=vmax + 1.0)
            else:
                synth_norm = plt.Normalize(vmin=vmin, vmax=vmax)

        for record in synthetic_curve_records:
            if not isinstance(record, dict):
                continue
            overlay = record.get("lightcurve_overlay") or {}
            abs_vals = np.asarray(overlay.get("abs_magnitude", []), dtype=float)
            ht_vals = np.asarray(overlay.get("height_km", []), dtype=float)
            n_curve = min(len(abs_vals), len(ht_vals))
            abs_vals = abs_vals[:n_curve]
            ht_vals = ht_vals[:n_curve]
            good = np.isfinite(abs_vals) & np.isfinite(ht_vals)
            abs_vals = abs_vals[good]
            ht_vals = ht_vals[good]
            if abs_vals.size == 0:
                continue

            speed_value = finite_float(record.get("synthetic_speed_kms"))
            if synth_norm is None or not np.isfinite(speed_value):
                color = "0.55"
            else:
                color = synth_cmap(synth_norm(speed_value))

            detected = bool(record.get("is_camera_detected", record.get("is_physically_detected", False)))
            minimum_frames = int(record.get("minimum_detected_frames", 1) or 1)
            linestyle = "-" if detected else "--"
            alpha = 0.70 if detected else 0.35
            linewidth = 0.90 if detected else 0.75
            label = None
            if detected and not first_detected_label:
                label = f"Synthetic camera detections (>= {minimum_frames} frames)"
                first_detected_label = True
            elif (not detected) and not first_nondetected_label:
                label = f"Synthetic non-detections (< {minimum_frames} frames)"
                first_nondetected_label = True

            ax.plot(
                abs_vals,
                ht_vals,
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                alpha=alpha,
                zorder=1,
                label=label,
            )

    source = result_payload.get("source_run")
    if source is not None:
        source_mag, source_h = _plot_arrays_from_run(source)
        if source_mag.size:
            ax.plot(
                source_mag,
                source_h,
                color="k",
                label="Best Fit Simulation",
                linewidth=1.5,
                zorder=3,
            )

    target = result_payload.get("target_run")
    if target is not None:
        target_mag, target_h = _plot_arrays_from_run(target)
        if target_mag.size:
            ax.plot(
                target_mag,
                target_h,
                color="red",
                label=f"{planet_name} Meteor",
                linewidth=1.5,
                zorder=4,
            )

    ax.set_ylabel("Height [km]", fontsize=15)
    ax.set_xlabel("Abs.Mag [-]", fontsize=15)
    ax.grid()

    x_min = ax.get_xlim()[0]
    x_max = 8.0
    ax.set_xlim(x_max, x_min)

    if have_obs:
        current_ylim = ax.get_ylim()
        new_ax_min = np.min([y_min, current_ylim[0]])
        new_ax_max = np.min([y_max, current_ylim[1]])

        source_const = (source or {}).get("constants", {})
        erosion_h_m = finite_float(source_const.get("erosion_height_start"))
        if np.isfinite(erosion_h_m):
            new_ax_max = np.max([erosion_h_m / 1000.0 + 2.0, new_ax_max])
        ax.set_ylim(new_ax_min, new_ax_max)
    else:
        all_h = []
        for run in (source, target):
            if run is None:
                continue
            mag, h = _plot_arrays_from_run(run)
            mask = mag <= x_max
            if np.any(mask):
                all_h.extend(h[mask].tolist())
        if all_h:
            lo = float(np.nanmin(all_h))
            hi = float(np.nanmax(all_h))
            pad = max(1.0, 0.03 * max(hi - lo, 1.0))
            ax.set_ylim(lo - pad, hi + pad)

    if synthetic_speed_values and synth_norm is not None and synth_cmap is not None:
        sm = plt.cm.ScalarMappable(norm=synth_norm, cmap=synth_cmap)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label("Synthetic Mars speed [km/s]")

    # Keep the legend readable even when several ground stations are present.
    n_legend = len(ax.get_legend_handles_labels()[1])
    legend_fontsize = 10 if n_legend > 6 else 12
    ax.legend(fontsize=legend_fontsize, loc="upper left")
    plt.tight_layout()

    suffix = "Mars" if planet_name.strip().lower() == "mars" else planet_name.replace(" ", "_")
    return save_figure(
        fig,
        output_dir / f"{base_name}_Lightcurve_Earth_vs_{suffix}",
        plot_format,
    )


# Backward-compatible function name used by the rest of this script.
def plot_absolute_magnitude_and_speed(
    result_payload: dict[str, Any],
    output_dir: Path,
    base_name: str,
    plot_format: str,
) -> list[str]:
    return plot_lightcurve_earth_vs_planet(
        result_payload, output_dir, base_name, plot_format
    )


def plot_detection_frames(
    detection: dict[str, Any],
    output_dir: Path,
    base_name: str,
    plot_format: str,
) -> list[str]:
    sampled = detection["sampled_frames"]
    continuous = detection["continuous"]

    sample_t = np.asarray(sampled["time_s"], dtype=float)
    sample_mag = np.asarray(sampled["apparent_magnitude"], dtype=float)
    sample_h = np.asarray(sampled["height_km"], dtype=float)
    visible = np.asarray(sampled["above_limiting_magnitude"], dtype=bool)
    lm = float(detection["limiting_apparent_magnitude"])

    fig, (ax_t, ax_h) = plt.subplots(1, 2, figsize=(11, 6))

    cont_t = np.asarray(continuous["time_s"], dtype=float)
    cont_mag = np.asarray(continuous["apparent_magnitude"], dtype=float)
    cont_h = np.asarray(continuous["height_km"], dtype=float)

    ax_t.plot(cont_t, cont_mag, linewidth=1.2, label="Continuous apparent magnitude")
    if len(sample_t):
        ax_t.scatter(sample_t[~visible], sample_mag[~visible], facecolors="none", s=28, label="Sampled below LM")
        ax_t.scatter(sample_t[visible], sample_mag[visible], s=32, label="Frames above LM")
    ax_t.axhline(lm, linestyle="--", linewidth=1.0, label=f"LM = {lm:g}")
    ax_t.invert_yaxis()
    ax_t.set_xlabel("Time [s]")
    ax_t.set_ylabel("Apparent magnitude")
    ax_t.grid(alpha=0.3)
    ax_t.legend(fontsize=8)

    ax_h.plot(cont_mag, cont_h, linewidth=1.2)
    if len(sample_t):
        ax_h.scatter(sample_mag[~visible], sample_h[~visible], facecolors="none", s=28)
        ax_h.scatter(sample_mag[visible], sample_h[visible], s=32)
    ax_h.axvline(lm, linestyle="--", linewidth=1.0)
    ax_h.invert_xaxis()
    ax_h.set_xlabel("Apparent magnitude")
    ax_h.set_ylabel("Height [km]")
    ax_h.grid(alpha=0.3)

    n_visible = int(sampled["visible_frame_count"])
    det_info = detection.get("detectability", {}) or {}
    delta_sample = finite_float(det_info.get("delta_mag_to_limit_sampled"))
    peak_sample = finite_float(
        det_info.get("brightest_sampled_frame_apparent_magnitude")
    )

    if np.isfinite(delta_sample):
        if delta_sample > 0.0:
            status = (
                f"NOT DETECTED: brightest sampled frame m={peak_sample:.2f}, "
                f"{delta_sample:.2f} mag fainter than LM={lm:g}"
            )
        else:
            status = (
                f"DETECTED: brightest sampled frame m={peak_sample:.2f}, "
                f"{-delta_sample:.2f} mag brighter than LM={lm:g}"
            )
    else:
        status = f"LM={lm:g}; no finite sampled apparent magnitude"

    observer_altitude = finite_float(detection.get("observer_altitude_km"), 0.0)
    fig.suptitle(
        f"{base_name}: observer altitude {observer_altitude:g} km | {status} | "
        f"{n_visible} frames above LM at {detection['camera_fps']:g} FPS",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    return save_figure(
        fig,
        output_dir / f"{base_name}_observer_visible_frames",
        plot_format,
    )


# =============================================================================
# Cache
# =============================================================================

def cache_signature(
    input_json: Path,
    atmosphere_csv: Path,
    parameter_file: Path,
    trajectory_pickle: Path,
    trigger: str,
    start_height_km: float | None,
) -> str:
    payload = {
        "version": CACHE_VERSION,
        "input_json_sha256": file_sha256(input_json),
        "atmosphere_sha256": file_sha256(atmosphere_csv),
        "planet_parameters_sha256": file_sha256(parameter_file),
        "trajectory_pickle_sha256": file_sha256(trajectory_pickle),
        "trigger": trigger,
        "start_height_km": start_height_km,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def save_cache(path: Path, signature: str, state: dict[str, Any]) -> None:
    try:
        with path.open("wb") as fh:
            pickle.dump(
                {
                    "cache_version": CACHE_VERSION,
                    "signature": signature,
                    "state": state,
                },
                fh,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        print(f"Saved intermediate cache: {path}")
    except Exception as exc:
        warnings.warn(f"Could not save intermediate cache {path}: {exc}")


def load_cache(path: Path, signature: str) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        with path.open("rb") as fh:
            data = pickle.load(fh)
        if (
            data.get("cache_version") == CACHE_VERSION
            and data.get("signature") == signature
            and isinstance(data.get("state"), dict)
        ):
            print(f"Loaded matching intermediate cache: {path}")
            return data["state"]
    except Exception as exc:
        warnings.warn(f"Could not load cache {path}: {exc}")
    return {}


# =============================================================================
# Previous-result postprocessing
# =============================================================================

def is_saved_result(data: dict[str, Any]) -> bool:
    return data.get("schema") == PROGRAM_SCHEMA and "target_run" in data


def postprocess_saved_result(
    data: dict[str, Any],
    args: argparse.Namespace,
    input_json: Path,
) -> tuple[dict[str, Any], Path]:
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else input_json.parent
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = args.base_name or input_json.stem.replace("_planet_run", "")
    result = copy.deepcopy(data)
    result["postprocessed_utc"] = datetime.now(timezone.utc).isoformat()

    # Reuse previous observer geometry/settings when omitted.
    old_detection = result.get("detection") or {}
    if args.observer_altitude_km is None:
        observer_altitude_km = finite_float(
            old_detection.get("observer_altitude_km"), 0.0
        )
    else:
        observer_altitude_km = float(args.observer_altitude_km)

    limiting_mag = args.limiting_mag
    if limiting_mag is None:
        limiting_mag = old_detection.get("limiting_apparent_magnitude")

    outputs: dict[str, Any] = {}
    outputs["main_plots"] = plot_absolute_magnitude_and_speed(
        result, output_dir, base_name, args.plot_format
    )

    if limiting_mag is not None:
        planet_params = result.get("planet_parameters", {}) or {}
        planet_radius = finite_float(
            planet_params.get(
                "radius_km",
                result.get("configuration", {}).get(
                    "planet_radius_km", 3389.5
                ),
            ),
            3389.5,
        )
        detection = compute_detection(
            result["target_run"],
            observer_altitude_km=observer_altitude_km,
            limiting_mag=float(limiting_mag),
            camera_fps=float(args.camera_fps),
            minimum_frames=int(args.minimum_frames),
            sampling_seed=int(args.sampling_seed),
            planet_radius_km=planet_radius,
            planet_params=planet_params,
        )
        result["detection"] = detection
        # Always write the limiting-magnitude diagnostic when LM is supplied,
        # including non-detections. This is important because the plot shows how
        # many magnitudes the event is from being detectable.
        outputs["detection_plots"] = plot_detection_frames(
            detection, output_dir, base_name, args.plot_format
        )
        print(detection.get("detectability", {}).get("message", ""))

    result["outputs"] = outputs
    result_path = output_dir / f"{base_name}_planet_run.json"
    save_json(result_path, result)
    return result, result_path


# =============================================================================
# Full physical run
# =============================================================================

def full_physical_run(
    args: argparse.Namespace,
    input_json: Path,
) -> tuple[dict[str, Any], Path]:
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else input_json.parent
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = args.base_name or input_json.stem

    # Planet setup is file-driven so changing planetary body does not require
    # editing a long command line.
    parameter_file = resolve_planet_parameter_file(
        input_json, args.planet_params
    )
    planet_params = read_planet_parameters(parameter_file)

    atmosphere_path = resolve_atmosphere_from_parameters(
        planet_params, parameter_file, input_json
    )
    polynomial_degree = int(
        planet_float(
            planet_params, "atmosphere_polynomial_degree", 6
        )
    )
    atmosphere = read_atmosphere_csv(
        atmosphere_path,
        polynomial_degree=polynomial_degree,
    )

    # The trajectory pickle is required because it supplies the heliocentric
    # orbit and the geometry needed to recompute speed and zenith angle.
    trajectory_path = discover_trajectory_pickle(
        input_json, args.pickle
    )
    trajectory = load_trajectory_pickle(trajectory_path)

    # Extract ALL photometric cameras directly from the trajectory.  This is
    # intentionally independent of the historical camera-name recognition in
    # ObservationData.
    camera_lightcurves = extract_all_camera_lightcurves_from_trajectory(
        trajectory,
        plot_faint_limit_mag=8.0,
    )
    print("\nTrajectory camera light curves:")
    for cam in camera_lightcurves:
        fps_text = (
            f"{cam['fps_estimate']:.3f} FPS"
            if cam.get("fps_estimate") is not None else "FPS unknown"
        )
        status = "PLOT + RMSD" if cam.get("used_for_auto_integration") else (
            "PLOT" if cam.get("used_for_plot") else "SKIP"
        )
        print(
            f"  {cam.get('station')}: {status}, {fps_text}"
            + (f" ({cam.get('reason')})" if cam.get("reason") else "")
        )

    trigger = (
        str(args.trigger).lower()
        if args.trigger is not None
        else planet_string(
            planet_params, "default_trigger", "dynamic_pressure"
        ).lower()
    )
    if trigger not in {
        "dynamic_pressure", "energy", "density"
    }:
        raise ValueError(
            f"Unsupported trigger {trigger!r}. Use dynamic_pressure, "
            "energy, or density."
        )

    print(
        f"Planet parameters: {parameter_file}\n"
        f"  planet: {planet_params['planet_name']}\n"
        f"  radius: {planet_float(planet_params, 'radius_km'):.3f} km\n"
        f"  orbit radius: {planet_float(planet_params, 'orbit_radius_au'):.6f} AU\n"
        f"Trajectory pickle: {trajectory_path}\n"
        f"Atmosphere: {atmosphere.path}\n"
        f"  height column: {atmosphere.height_column}\n"
        f"  density column: {atmosphere.density_column}\n"
        f"  range: {atmosphere.min_height_m/1000:.1f}--"
        f"{atmosphere.max_height_m/1000:.1f} km\n"
        f"  polynomial degree: {atmosphere.fit_degree}\n"
        f"  RMS log10(rho) fit error: "
        f"{atmosphere.fit_rms_log10_density:.4g}\n"
        f"Trigger: {trigger}"
    )

    source_const = load_best_fit_constants(input_json)

    target_base_const, start_meta, kinematics = (
        build_target_base_constants(
            source_const=source_const,
            atmosphere=atmosphere,
            planet_params=planet_params,
            trajectory=trajectory,
            args=args,
        )
    )

    print(
        f"\nTarget-planet kinematics from trajectory pickle:\n"
        f"  entry speed: "
        f"{kinematics['selected_entry_speed_kms']:.3f} km/s\n"
        f"  zenith angle: {kinematics['zenith_angle_deg']:.3f} deg\n"
        f"  speed method: "
        f"{kinematics['intercept']['method']}\n"
        f"  zenith method: {kinematics['zenith_method']}"
    )
    for scenario in kinematics["intercept"].get("scenarios", []):
        print(
            "    "
            f"{scenario.get('label')}: "
            f"Vinf={finite_float(scenario.get('v_inf_kms')):.3f} km/s, "
            f"Ventry={finite_float(scenario.get('entry_speed_at_simulation_start_kms')):.3f} km/s"
        )

    density_const, density_mapping = map_triggers_by_density(
        source_const, target_base_const, atmosphere
    )

    cache_path = output_dir / (
        f"{base_name}_planet_run_cache.pkl"
    )
    signature = cache_signature(
        input_json=input_json,
        atmosphere_csv=atmosphere_path,
        parameter_file=parameter_file,
        trajectory_pickle=trajectory_path,
        trigger=trigger,
        start_height_km=args.start_height_km,
    )
    state = (
        {} if args.force_rerun
        else load_cache(cache_path, signature)
    )

    source_result = state.get("source_result")
    reference_result = state.get("reference_result")
    final_result = state.get("final_result")
    trigger_mapping = state.get("trigger_mapping")

    # Source result is always useful now: it is needed for p_dyn/energy mapping
    # and for the automatic real-light-curve FPS comparison.
    if source_result is None:
        source_result = run_with_status(
            source_const, "original atmosphere"
        )
        state["source_result"] = source_result
        save_cache(cache_path, signature, state)

    if trigger == "density":
        final_const = density_const
        trigger_mapping = density_mapping
        if final_result is None:
            final_result = run_with_status(
                final_const,
                "new atmosphere / density trigger",
            )
            state["final_result"] = final_result
            state["trigger_mapping"] = trigger_mapping
            save_cache(cache_path, signature, state)

    else:
        if reference_result is None:
            reference_result = run_with_status(
                density_const,
                "new atmosphere / density-reference trigger",
            )
            state["reference_result"] = reference_result
            save_cache(cache_path, signature, state)

        if trigger_mapping is None or final_result is None:
            if trigger == "dynamic_pressure":
                final_const, trigger_mapping = (
                    map_triggers_by_dynamic_pressure(
                        source_const,
                        density_const,
                        source_result,
                        reference_result,
                    )
                )
            elif trigger == "energy":
                final_const, trigger_mapping = (
                    map_triggers_by_energy(
                        source_const,
                        density_const,
                        source_result,
                        reference_result,
                    )
                )
            else:
                raise ValueError(
                    f"Unsupported trigger: {trigger}"
                )

            print("\nMapped trigger heights:")
            for item in trigger_mapping:
                print(
                    f"  "
                    f"{item.get('field', item.get('frag_type', 'trigger'))}: "
                    f"{item['source_height_m']/1000:.3f} km -> "
                    f"{item['target_height_m']/1000:.3f} km "
                    f"({item['method']})"
                )

            final_result = run_with_status(
                final_const,
                f"new atmosphere / {trigger} trigger",
            )
            state["final_result"] = final_result
            state["trigger_mapping"] = trigger_mapping
            save_cache(cache_path, signature, state)

    # The same required pickle is also used for real photometry/FPS validation
    # when observation_data can extract those arrays.
    obs_data = None
    try:
        obs_data = load_observations(
            trajectory_path,
            fitted_p0m=finite_float(
                getattr(source_const, "P_0m", np.nan),
                840.0,
            ),
        )
    except Exception as exc:
        warnings.warn(
            "The trajectory pickle was loaded successfully for orbit/geometry, "
            f"but observation_data could not extract photometry ({exc}). "
            "Orbit-derived speed/zenith remain valid; RMSD auto-selection will "
            "fall back to the non-integrated light curve."
        )

    integration_info = choose_integration_mode(
        requested_mode=args.integration_mode,
        source_result=source_result,
        obs_data=obs_data,
        user_fps=args.integration_fps,
        camera_lightcurves=camera_lightcurves,
    )

    print("\nLight-curve integration diagnostic:")
    print(
        f"  selected = {integration_info.get('selected_mode')}; "
        f"representative FPS = {integration_info.get('fps')}; "
        f"RMSD raw = {integration_info.get('rmsd_nointegfps_mag')}; "
        f"RMSD integrated = {integration_info.get('rmsd_integfps_mag')}"
    )
    for item in integration_info.get("per_camera", []):
        print(
            f"    {item.get('station')}: FPS={item.get('fps')}, "
            f"raw={item.get('rmsd_nointegfps_mag')}, "
            f"integrated={item.get('rmsd_integfps_mag')}"
        )

    target_lum, target_mag, target_integration_applied = (
        selected_lightcurve_arrays(
            final_result, integration_info
        )
    )
    integration_info["target_integration_applied"] = bool(
        target_integration_applied
    )

    try:
        source_lum, source_mag, source_integration_applied = (
            selected_lightcurve_arrays(
                source_result, integration_info
            )
        )
    except Exception:
        source_lum = np.asarray(
            source_result.luminosity_arr, dtype=float
        )
        source_mag = np.asarray(
            source_result.abs_magnitude, dtype=float
        )
        source_integration_applied = False
    integration_info["source_integration_applied"] = bool(
        source_integration_applied
    )
    source_serialized = serializable_run(
        source_result, source_lum, source_mag
    )

    radius_km = planet_float(
        planet_params, "radius_km"
    )
    gravity = planet_float(
        planet_params, "surface_gravity_m_s2"
    )

    result: dict[str, Any] = {
        "schema": PROGRAM_SCHEMA,
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "input": {
            "metsim_json": str(input_json),
            "metsim_json_sha256": file_sha256(
                input_json
            ),
            "trajectory_pickle": str(trajectory_path),
            "trajectory_pickle_sha256": file_sha256(
                trajectory_path
            ),
            "planet_parameter_file": str(
                parameter_file
            ),
            "planet_parameter_file_sha256": file_sha256(
                parameter_file
            ),
            "atmosphere_csv": str(atmosphere_path),
            "atmosphere_csv_sha256": file_sha256(
                atmosphere_path
            ),
        },
        "planet_parameters": {
            key: value
            for key, value in planet_params.items()
            if not key.startswith("_")
        },
        "configuration": {
            "trigger": trigger,
            "planet_name": str(
                planet_params["planet_name"]
            ),
            "planet_radius_km": radius_km,
            "planet_g0_m_s2": gravity,
            "plot_format": args.plot_format,
        },
        "trajectory_kinematics": kinematics,
        "atmosphere": {
            "height_column": atmosphere.height_column,
            "density_column": atmosphere.density_column,
            "height_km": atmosphere.height_m / 1000.0,
            "density_kg_m3": atmosphere.density_kg_m3,
            "dens_co": atmosphere.dens_co,
            "fit_degree": atmosphere.fit_degree,
            "fit_rms_log10_density": (
                atmosphere.fit_rms_log10_density
            ),
        },
        "start_height_mapping": start_meta,
        "trigger_mapping": trigger_mapping or [],
        "integration": integration_info,
        "observations": observation_payload(
            obs_data,
            camera_lightcurves=camera_lightcurves,
        ),
        "source_run": source_serialized,
        "target_run": serializable_run(
            final_result, target_lum, target_mag
        ),
        "cache": {
            "path": str(cache_path),
            "signature": signature,
            "force_rerun": bool(args.force_rerun),
        },
        "detection": None,
        "outputs": {},
    }

    result["outputs"]["main_plots"] = (
        plot_absolute_magnitude_and_speed(
            result,
            output_dir,
            base_name,
            args.plot_format,
        )
    )

    if args.limiting_mag is not None:
        detection = compute_detection(
            result["target_run"],
            observer_altitude_km=(
                0.0 if args.observer_altitude_km is None
                else float(args.observer_altitude_km)
            ),
            limiting_mag=float(args.limiting_mag),
            camera_fps=float(args.camera_fps),
            minimum_frames=int(args.minimum_frames),
            sampling_seed=int(args.sampling_seed),
            planet_radius_km=radius_km,
            planet_params=planet_params,
        )
        result["detection"] = detection
        # Always produce the LM diagnostic, even for a non-detection, so the
        # user sees how far the model is from the camera threshold.
        result["outputs"]["detection_plots"] = (
            plot_detection_frames(
                detection,
                output_dir,
                base_name,
                args.plot_format,
            )
        )
        print(detection.get("detectability", {}).get("message", ""))

    result_path = output_dir / (
        f"{base_name}_planet_run.json"
    )
    save_json(result_path, result)
    return result, result_path


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one fitted MetSim event in a target-planet atmosphere. "
            "Only --json is normally required: the script finds the matching "
            "trajectory pickle, planet_parameters.txt, and atmosphere CSV."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--json",
        # required=True,
        default=r"C:\Users\maxiv\Documents\UWO\Papers\5)Mars meteors\Fireball-test\20190628_063255\20190628_063255_sim_fit_dynesty_BestGuess.json",
        # default=r"C:\Users\maxiv\Documents\UWO\Papers\5)Mars meteors\Fireball-test\EN040326_201155\EN040326_201155_sim_fit.json",
        help=(
            "Fitted MetSim JSON, or a *_planet_run.json previously written by "
            "this program for fast observer/limiting-magnitude reprocessing."
        ),
    )
    parser.add_argument(
        "--pickle",
        default=None,
        help=(
            "Override the automatically discovered trajectory pickle. A physical "
            "rerun requires this pickle for orbit-derived speed and zenith angle."
        ),
    )
    parser.add_argument(
        "--planet-params",
        default=None,
        help=(
            "Override planet_parameters.txt. By default it is searched beside "
            "the event JSON, then beside the script."
        ),
    )

    parser.add_argument(
        "--trigger",
        choices=("dynamic_pressure", "energy", "density"),
        default=None,
        help=(
            "Override default_trigger from the planet parameter file. Energy "
            "mapping can take longer for fragmentation-rich events."
        ),
    )
    parser.add_argument(
        "--start-height-km",
        type=float,
        default=None,
        help=(
            "Optional target simulation start height. Otherwise the density at "
            "the source JSON h_init is matched in the target atmosphere."
        ),
    )

    parser.add_argument(
        "--integration-mode",
        choices=("auto", "integfps", "nointegfps"),
        default="auto",
        help=(
            "Auto compares raw vs finite-FPS light curves against the real "
            "photometry extracted from the required trajectory pickle."
        ),
    )
    parser.add_argument(
        "--integration-fps",
        type=float,
        default=None,
        help=(
            "FPS used for integfps if the observation pickle does not provide "
            "a valid fps_lum."
        ),
    )

    parser.add_argument(
        "--limiting-mag",
        type=float,
        default=4,
        help="Optional apparent limiting magnitude at the observer.",
    )
    parser.add_argument(
        "--observer-altitude-km",
        "--observer-position-km",
        dest="observer_altitude_km",
        type=float,
        default=None,
        help=(
            "Observer altitude above the planet surface [km]. Ground is 0 km "
            "by default. The code internally places the observer at "
            "[planet_radius + altitude, 0, 0]. The old --observer-position-km "
            "name is kept as a one-number alias for compatibility."
        ),
    )
    parser.add_argument(
        "--camera-fps",
        type=float,
        default=10.0,
        help="Camera cadence for sampled apparent-magnitude frames.",
    )
    parser.add_argument(
        "--minimum-frames",
        type=int,
        default=10,
        help="Above-LM frames required for the saved detection flag.",
    )
    parser.add_argument(
        "--sampling-seed",
        type=int,
        default=0,
        help="Seed controlling the sub-frame camera phase.",
    )

    parser.add_argument(
        "--output-dir",
        default=r"C:\Users\maxiv\Documents\UWO\Papers\5)Mars meteors\Fireball-test",
        help="Output directory; default is the input JSON directory.",
    )
    parser.add_argument(
        "--base-name",
        default=None,
        help="Output basename; default is the input JSON stem.",
    )
    parser.add_argument(
        "--plot-format",
        choices=("png", "pdf", "both"),
        default="png",
        help="Figure output format.",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Ignore any matching intermediate simulation cache.",
    )
    return parser


def print_final_summary(result: dict[str, Any], result_path: Path) -> None:
    summary = result.get("target_run", {}).get("summary", {})
    print("\n" + "=" * 72)
    print("FINAL RUN SUMMARY")
    print("=" * 72)
    print(f"Saved result JSON: {result_path}")
    if summary:
        print(
            f"Peak absolute magnitude: "
            f"{finite_float(summary.get('peak_absolute_magnitude')):.3f}"
        )
        print(
            f"Peak height: {finite_float(summary.get('peak_height_km')):.3f} km"
        )
        print(
            f"Initial speed: {finite_float(summary.get('initial_speed_kms')):.3f} km/s"
        )

    integration = result.get("integration", {})
    print(f"Light-curve integration mode: {integration.get('selected_mode')}")
    if integration.get("rmsd_nointegfps_mag") is not None:
        print(
            "RMSD raw / integrated [mag]: "
            f"{integration.get('rmsd_nointegfps_mag')} / "
            f"{integration.get('rmsd_integfps_mag')}"
        )

    detection = result.get("detection")
    if detection:
        frames = detection.get("sampled_frames", {}).get("visible_frame_count", 0)
        print(
            f"Observer visible frames: {frames} "
            f"(LM={detection.get('limiting_apparent_magnitude')}, "
            f"FPS={detection.get('camera_fps')})"
        )
    print("=" * 72)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.camera_fps <= 0:
        parser.error("--camera-fps must be > 0.")
    if args.minimum_frames < 1:
        parser.error("--minimum-frames must be >= 1.")
    input_json = Path(args.json).expanduser().resolve()
    if not input_json.is_file():
        parser.error(f"Input JSON not found: {input_json}")

    data = load_json(input_json)
    if is_saved_result(data):
        result, result_path = postprocess_saved_result(data, args, input_json)
    else:
        result, result_path = full_physical_run(args, input_json)

    print_final_summary(result, result_path)


if __name__ == "__main__":
    main()
