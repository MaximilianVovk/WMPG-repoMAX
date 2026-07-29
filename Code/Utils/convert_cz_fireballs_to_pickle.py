#!/usr/bin/env python3
"""
Convert EN fireball network .res/.inp/.lc/.rlc files into one pickle
containing trajectory-like dictionaries or SimpleNamespace objects.

The output is intentionally shaped like the WMPL trajectory JSON/pickle style:
- one trajectory-like object per fireball
- top-level metadata: jdt_ref, file_name, traj_id, rbeg_*, rend_*, orbit, etc.
- observations: one item per station from the .inp file, with arrays such as
  time_data, length, state_vect_dist, model_ht, meas_ht, velocities,
  lag, absolute_magnitudes, and embedded lc/rlc photometry.

Typical use:
    python convert_en_fireballs_to_pickle.py /path/to/fireballs \
        --template-json 20230811_082648_trajectory.json \
        --output en_fireballs_trajectory_like.pkl \
        --object-style dict

The input can be:
- a folder containing many event folders
- a folder containing many .zip files, one per event
- a single .zip file
- a single folder containing EN*.res, EN*.inp, EN*.lc, EN*.rlc
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import pickle
import re
import sys
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover
    raise SystemExit("This script needs numpy: pip install numpy") from exc


FLOAT_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[Ee][-+]?\d+)?")


def floats_in(line: str) -> List[float]:
    return [float(x) for x in FLOAT_RE.findall(line)]


def first_match(pattern: str, text: str, flags: int = re.I | re.M) -> Optional[re.Match]:
    return re.search(pattern, text, flags)


def deg2rad(x: Optional[float]) -> Optional[float]:
    return None if x is None else math.radians(x)


def km2m(x: Optional[float]) -> Optional[float]:
    return None if x is None else 1000.0 * x


def safe_decode(data: bytes) -> str:
    for enc in ("utf-8", "latin-1", "cp1250"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            pass
    return data.decode("latin-1", errors="replace")


def clean_code_from_stem(stem: str) -> str:
    # Normalize filenames like EN040326_201155.res -> en040326_201155
    stem = Path(stem).stem
    return stem.lower()


def camera_base_id(camera_id: Any) -> str:
    """Return the station/base part of a camera ID.

    EN .res camera names can be full camera IDs like 102-72, while .inp/.lc/.rlc
    blocks often only carry the base station ID like 102. This helper keeps the
    full camera ID for display, but lets photometry still match by base station.
    """
    camera_id = str(camera_id).strip()
    return camera_id.split("-", 1)[0]


def make_prefixed_camera_id(camera_id: Any, prefix: str = "cz_") -> str:
    """Add the CZ prefix without removing hyphenated camera suffixes."""
    camera_id = str(camera_id).strip()
    return camera_id if camera_id.startswith(prefix) else f"{prefix}{camera_id}"


def remove_station_prefix(station_id: Any, prefix: str = "cz_") -> str:
    """Remove only the display prefix; preserve camera suffixes like 102-72."""
    station_id = str(station_id).strip()
    return station_id[len(prefix):] if station_id.startswith(prefix) else station_id


# Only these fields are copied from an optional template JSON/pickle.  Everything
# trajectory-specific (orbit, state vector, radiant, observations, LLA points,
# uncertainties, etc.) must be recomputed or parsed from the EN files.  Copying
# those arrays from a previous meteor is exactly what causes stale zc/zg and
# other old-solution values to leak into the converted pickle.
CONSTANT_TEMPLATE_KEYS = {
    "meastype",
    "max_toffset",
    "verbose",
    "v_init_part",
    "v_init_ht",
    "fixed_time_offsets",
    "estimate_timing_vel",
    "fixed_times",
    "monte_carlo",
    "mc_runs",
    "mc_runs_max",
    "mc_pick_multiplier",
    "mc_noise_std",
    "enable_OSM_plot",
    "geometric_uncert",
    "filter_picks",
    "calc_orbit",
    "show_plots",
    "show_jacchia",
    "save_results",
    "gravity_correction",
    "gravity_factor",
    "plot_all_spatial_residuals",
    "plot_file_type",
    "reject_n_sigma_outliers",
    "mc_cores",
    "phase_1_only",
}


# ---------------------------- .res parser ----------------------------

def parse_res(text: str) -> Dict[str, Any]:
    """Parse the EN*.res summary file into a structured dictionary.

    Units in this parsed dictionary are kept as written in the .res file:
    degrees, km, km/s, kg, J, MPa. Unit conversion happens later when
    creating WMPL-like fields.
    """
    out: Dict[str, Any] = {}

    m = first_match(r"METEOR\s+CODE:\s*(\S+)", text)
    if m:
        out["meteor_code"] = m.group(1)

    m = first_match(r"DATE:\s*(\d{4}-\d{2}-\d{2})", text)
    if m:
        out["date"] = m.group(1)

    m = first_match(r"TIME:\s*([^\n]+?)\s+UT", text)
    if m:
        out["time_ut"] = m.group(1).strip()

    m = first_match(r"JULIAN\s+DATE:\s*([\d.]+)", text)
    if m:
        out["julian_date"] = float(m.group(1))

    m = first_match(r"SOLAR\s+LONGITUDE.*?:\s*([\d.+\-Ee]+)", text)
    if m:
        out["solar_longitude_deg"] = float(m.group(1))

    m = first_match(r"SIDEREAL\s+TIME\s+AT\s+GREENWICH:\s*([\d.+\-Ee]+)", text)
    if m:
        out["sidereal_time_greenwich_deg"] = float(m.group(1))

    # Trajectory points: lon east, lat north, height km
    for label, key in [("BEGINNING", "begin"), ("END", "end"), ("AVERAGE", "average")]:
        m = first_match(rf"^\s*{label}\s+([-+\d.Ee]+)\s+([-+\d.Ee]+)\s+([-+\d.Ee]+)", text)
        if m:
            out[key] = {
                "lon_deg_east": float(m.group(1)),
                "lat_deg_north": float(m.group(2)),
                "height_km": float(m.group(3)),
            }

    m = first_match(r"LINEAR\s+LENGTH\s+OF\s+THE\s+TRAJECTORY:\s*([-+\d.Ee]+)\s*KM", text)
    if m:
        out["linear_length_km"] = float(m.group(1))

    m = first_match(
        r"DURATION:\s*([-+\d.Ee]+)\s*S\s*\(OVER\s+THE\s+LENGTH\s+OF\s+([-+\d.Ee]+)\s*KM\s+BETWEEN\s+HEIGHTS\s+([-+\d.Ee]+)\s+AND\s+([-+\d.Ee]+)\s+KM",
        text,
    )
    if m:
        out["duration"] = {
            "seconds": float(m.group(1)),
            "length_km": float(m.group(2)),
            "height_start_km": float(m.group(3)),
            "height_end_km": float(m.group(4)),
        }

    m = first_match(r"AVERAGE\s+([-+\d.Ee]+)\s+([-+\d.Ee]+)\s*\n\s*\n\s*\n\s*RADIANT\s+RIGHT", text)
    # This would be the average azimuth/ZD if needed, but keep the named block below clearer.

    # Radiant azimuth and zenith distance block.  EN reports this as
    # "RADIANT AZIMUTH (S=0)  ZENITH DISTANCE".  The zenith distance at
    # BEGINNING is the apparent zenith angle zc needed by the WMPL trajectory.
    m = first_match(
        r"RADIANT\s+AZIMUTH.*?ZENITH\s+DISTANCE\s*\n\s*"
        r"BEGINNING\s+([-+\d.Ee]+)\s+([-+\d.Ee]+).*?\n"
        r"\s*END\s+([-+\d.Ee]+)\s+([-+\d.Ee]+).*?\n"
        r"(?:\s*[-+\d.Ee]+\s+[-+\d.Ee]+\s*\n)?"
        r"\s*AVERAGE\s+([-+\d.Ee]+)\s+([-+\d.Ee]+)",
        text,
        re.I | re.S,
    )
    if m:
        out["radiant_azimuth_zd"] = {
            "begin": {
                "azimuth_s0_deg": float(m.group(1)),
                "zenith_distance_deg": float(m.group(2)),
            },
            "end": {
                "azimuth_s0_deg": float(m.group(3)),
                "zenith_distance_deg": float(m.group(4)),
            },
            "average": {
                "azimuth_s0_deg": float(m.group(5)),
                "zenith_distance_deg": float(m.group(6)),
            },
        }

    m = first_match(
        r"APPARENT\s+([-+\d.Ee]+)\s+([-+\d.Ee]+)\s+([-+\d.Ee]+)\s*\n\(at beginning\)",
        text,
    )
    if m:
        out["radiant_apparent"] = {
            "ra_deg": float(m.group(1)),
            "dec_deg": float(m.group(2)),
            "velocity_km_s": float(m.group(3)),
        }

    m = first_match(r"J2000\.0\s+([-+\d.Ee]+)\s+([-+\d.Ee]+)", text)
    if m:
        out["radiant_j2000"] = {"ra_deg": float(m.group(1)), "dec_deg": float(m.group(2))}

    m = first_match(r"GEOCENTRIC\s+([-+\d.Ee]+)\s+([-+\d.Ee]+)\s+([-+\d.Ee]+)", text)
    if m:
        out["radiant_geocentric"] = {
            "ra_deg": float(m.group(1)),
            "dec_deg": float(m.group(2)),
            "velocity_km_s": float(m.group(3)),
        }

    m = first_match(r"HELIOCENTRIC\s+([-+\d.Ee]+)\s+([-+\d.Ee]+)\s+([-+\d.Ee]+)", text)
    if m:
        out["radiant_heliocentric"] = {
            "ecl_lon_deg": float(m.group(1)),
            "ecl_lat_deg": float(m.group(2)),
            "velocity_km_s": float(m.group(3)),
        }

    m = first_match(r"VELOCITY\s+AT\s+THE\s+AVERAGE\s+TRAJECTORY\s+POINT:\s*([-+\d.Ee]+)\s*KM/S", text)
    if m:
        out["velocity_avg_km_s"] = float(m.group(1))

    m = first_match(r"TERMINAL\s+VELOCITY.*?:\s*([-+\d.Ee]+)\s*KM/S", text)
    if m:
        out["terminal_velocity_km_s"] = float(m.group(1))

    # Orbital elements table: A E Q PER Q APH INCL OMEGA ASC NODE PI
    m = first_match(
        r"ORBITAL\s+ELEMENTS.*?\n\s*A\s+E\s+Q\s+PER\s+Q\s+APH\s+INCL\s+OMEGA\s+ASC\s+NODE\s+PI\s*\n\s*([^\n]+)\n\s*([^\n]+)",
        text,
        re.I | re.S,
    )
    if m:
        vals = floats_in(m.group(1))
        sigs = floats_in(m.group(2))
        names = ["a_au", "e", "q_per_au", "q_aph_au", "incl_deg", "omega_deg", "asc_node_deg", "pi_deg"]
        if len(vals) >= len(names):
            out["orbital_elements"] = dict(zip(names, vals[: len(names)]))
            if len(sigs) >= len(names):
                out["orbital_element_uncertainties"] = dict(zip(names, sigs[: len(names)]))

    m = first_match(
        r"TISSERAND\s+PERIOD\s+PERIHELION\s+PASSAGE\s*\n\s*([-+\d.Ee]+)\s+([-+\d.Ee]+)\s+([^\n]+)",
        text,
    )
    if m:
        out["tisserand"] = float(m.group(1))
        out["period_years"] = float(m.group(2))
        out["perihelion_passage"] = m.group(3).strip()

    m = first_match(r"SHOWER\s+MEMBERSHIP:\s*([^\n]+)", text)
    if m:
        out["shower_membership"] = m.group(1).strip()

    m = first_match(r"MAXIMUM\s+MAGNITUDE:\s*([-+\d.Ee]+)\s*\(AT\s+HEIGHT\s+([-+\d.Ee]+)\s*KM\)", text)
    if m:
        out["maximum_magnitude"] = float(m.group(1))
        out["maximum_magnitude_height_km"] = float(m.group(2))

    for label, key in [
        ("RADIATED ENERGY", "radiated_energy_j"),
        ("INITIAL PHOTOMETRIC MASS", "initial_photometric_mass_kg"),
        ("INITIAL DYNAMIC MASS", "initial_dynamic_mass_kg"),
        ("TERMINAL DYNAMIC MASS", "terminal_dynamic_mass_kg"),
    ]:
        m = first_match(rf"{label}:\s*([-+\d.Ee]+)", text)
        if m:
            out[key] = float(m.group(1))

    m = first_match(r"APPARENT\s+ABLATION\s+COEFFICIENT:\s*([-+\d.Ee]+)", text)
    if m:
        out["apparent_ablation_coeff_s2_km2"] = float(m.group(1))

    m = first_match(
        r"PE\s+COEFFICIENT\s*=\s*([-+\d.Ee]+)\s+AL\s+COEFFICIENT\s*=\s*([-+\d.Ee]+)\s+KB\s+COEFFICIENT\s*=\s*([-+\d.Ee]+)",
        text,
    )
    if m:
        out["PE"] = float(m.group(1))
        out["AL"] = float(m.group(2))
        out["KB"] = float(m.group(3))

    m = first_match(r"CLASSIFICATION:\s*(\S+)", text)
    if m:
        out["classification"] = m.group(1)

    m = first_match(r"MAXIMUM\s+DYNAMIC\s+PRESSURE\s*=\s*([-+\d.Ee]+)\s*MPA\s+AT\s+HEIGHT\s+([-+\d.Ee]+)\s*KM", text)
    if m:
        out["maximum_dynamic_pressure_mpa"] = float(m.group(1))
        out["maximum_dynamic_pressure_height_km"] = float(m.group(2))

    m = first_match(
        r"MAXIMUM\s+BRIGHTNESS\s+POINT\s*\n\s*LONGITUDE\s+LATITUDE\s+HEIGHT\s+TIME\s+LENGTH\s+VELOCITY\s*\n\s*([^\n]+)",
        text,
    )
    if m:
        vals = floats_in(m.group(1))
        if len(vals) >= 6:
            out["maximum_brightness_point"] = {
                "lon_deg_east": vals[0],
                "lat_deg_north": vals[1],
                "height_km": vals[2],
                "time_s": vals[3],
                "length_km": vals[4],
                "velocity_km_s": vals[5],
            }

    m = first_match(r"NUMBER\s+OF\s+CAMERAS:\s*(\d+)", text)
    if m:
        out["number_of_cameras"] = int(m.group(1))

    m = first_match(r"\(([^()]+)\)\s*\nCLOSEST\s+DISTANCE", text)
    if m:
        out["camera_list"] = [x.strip() for x in m.group(1).split(",")]

    m = first_match(r"CLOSEST\s+DISTANCE:\s*([-+\d.Ee]+)\s*KM\s+TO\s+STATION\s+([^\n]+)", text)
    if m:
        out["closest_distance_km"] = float(m.group(1))
        out["closest_station"] = m.group(2).strip()

    m = first_match(r"MAXIMAL\s+CONVERGENCE\s+ANGLE:\s*([-+\d.Ee]+)\s*DEGREES", text)
    if m:
        out["maximal_convergence_angle_deg"] = float(m.group(1))

    return out


# ---------------------------- station block parsers ----------------------------

def is_station_header_line(line: str, ext: str) -> bool:
    s = line.strip()
    if not s:
        return False
    toks = s.split()
    if ext == ".inp":
        # Example: 50403260     EN  40326 105
        # Some files may already use a hyphenated camera ID like 102-72.
        return len(toks) >= 4 and toks[1].upper() == "EN" and re.fullmatch(r"\d+(?:-\d+)?", toks[-1]) is not None
    # .lc/.rlc station block usually starts with a station/camera number, e.g. 105 or 102-72.
    return len(toks) == 1 and re.fullmatch(r"\d+(?:-\d+)?", toks[0]) is not None


def parse_inp(text: str, camera_list: Optional[List[str]] = None, station_prefix: str = "cz_") -> Dict[str, Dict[str, Any]]:
    """Parse EN*.inp blocks.

    The data rows are interpreted as: time_s, length_km, height_km.
    Blocks end with a -1000 sentinel.
    """
    lines = text.splitlines()
    i = 0
    stations: Dict[str, Dict[str, Any]] = {}

    # The .res file can list full camera IDs, e.g.
    #   105-30, 102-30, 104-30, 111-30, 204-33, 102-72, 102-71
    # while the .inp headers often only contain the base station, e.g. 102.
    # Assign the next matching full camera ID, preserving the order of the .res
    # list and skipping cameras that do not appear in the .inp file.
    camera_queue_by_base: Dict[str, List[str]] = defaultdict(list)
    for cam in camera_list or []:
        cam = str(cam).strip()
        if cam:
            camera_queue_by_base[camera_base_id(cam)].append(cam)

    while i < len(lines):
        line = lines[i]
        if not is_station_header_line(line, ".inp"):
            i += 1
            continue

        toks = line.split()
        raw_station_id = toks[-1]
        block_id = toks[0]

        # Use the full camera ID from the .res camera_list where possible.
        # This avoids collapsing 102-30, 102-72, and 102-71 into the same cz_102 ID.
        matching_cameras = camera_queue_by_base.get(camera_base_id(raw_station_id), [])
        if matching_cameras:
            camera_id = matching_cameras.pop(0)
        else:
            camera_id = raw_station_id

        station_id = make_prefixed_camera_id(camera_id, station_prefix)

        meta = {
            "raw_block_id": block_id,
            "network": toks[1] if len(toks) > 1 else None,
            "date_code": toks[2] if len(toks) > 2 else None,
            "camera_id": camera_id,
            "raw_station_id": raw_station_id,
        }
        i += 1

        header_lines: List[str] = []
        # The next 4 lines are metadata in the examples.
        for _ in range(4):
            if i < len(lines):
                header_lines.append(lines[i].rstrip())
                i += 1
        meta["header_lines"] = header_lines
        if len(header_lines) >= 2:
            vals = floats_in(header_lines[1])
            if len(vals) >= 3:
                meta["trajectory_reference_xyz_km"] = vals[:3]
        if len(header_lines) >= 3:
            vals = floats_in(header_lines[2])
            if len(vals) >= 4:
                meta["trajectory_direction_cosines"] = vals[:4]

        rows: List[List[float]] = []
        while i < len(lines):
            s = lines[i].strip()
            vals = floats_in(s)
            i += 1
            if not vals:
                continue
            if vals[0] <= -999:
                break
            if len(vals) >= 3:
                rows.append(vals[:3])

        if rows:
            # In these EN .inp files, the first and/or last rows can be zero-time
            # bookend/reference points rather than measured frames. They make the
            # time array non-monotonic, so remove the obvious bookends but store
            # them in metadata. The original raw text can also be retained with
            # --include-raw.
            cleaned_rows = list(rows)
            dropped_bookends = []
            if len(cleaned_rows) >= 2 and abs(cleaned_rows[0][0]) < 1e-12:
                if abs(cleaned_rows[0][1] - cleaned_rows[1][1]) < 1e-6 and abs(cleaned_rows[0][2] - cleaned_rows[1][2]) < 1e-6:
                    dropped_bookends.append({"position": "first", "row": cleaned_rows.pop(0)})
            if len(cleaned_rows) >= 2 and abs(cleaned_rows[-1][0]) < 1e-12 and cleaned_rows[-2][0] > 0:
                dropped_bookends.append({"position": "last", "row": cleaned_rows.pop(-1)})
            if dropped_bookends:
                meta["dropped_zero_time_bookends"] = dropped_bookends
            cleaned_rows = sorted(cleaned_rows, key=lambda r: r[0])
            arr = np.array(cleaned_rows, dtype=float)
            # Preserve full camera IDs as the observation keys. If a file repeats
            # an identical camera ID, still keep the duplicate as a separate record.
            key = camera_id
            if key in stations:
                key = f"{camera_id}__{block_id}"
            n_dup = 2
            while key in stations:
                key = f"{camera_id}__{block_id}__{n_dup}"
                n_dup += 1
            stations[key] = {
                "station_id": station_id,
                "raw_station_id": raw_station_id,
                "camera_id": camera_id,
                "record_id": key,
                "meta": meta,
                "time_s": arr[:, 0].tolist(),
                "length_km": arr[:, 1].tolist(),
                "height_km": arr[:, 2].tolist(),
            }

    return stations


def parse_lc_like(text: str, *, has_sd: bool, ext: str, station_prefix: str = "cz_") -> Dict[str, Dict[str, Any]]:
    """Parse EN*.lc or EN*.rlc station photometry blocks.

    .lc rows:  time_s, abs_mag, sd_mag, length_km, height_km
    .rlc rows: time_s, abs_mag, length_km, height_km
    Some .rlc blocks include a one-number camera/channel line after the station id;
    it is stored in metadata and skipped.
    """
    lines = text.splitlines()
    i = 0
    stations: Dict[str, Dict[str, Any]] = {}
    while i < len(lines):
        line = lines[i]
        if not is_station_header_line(line, ext):
            i += 1
            continue
        raw_station_id = line.strip().split()[0]
        camera_id = raw_station_id
        station_id = make_prefixed_camera_id(camera_id, station_prefix)
        i += 1

        meta: Dict[str, Any] = {}
        rows: List[List[float]] = []
        while i < len(lines):
            s = lines[i].strip()
            vals = floats_in(s)
            i += 1
            if not vals:
                continue
            if vals[0] <= -999:
                break
            # .rlc often has a one-number channel/count line before data.
            if not has_sd and len(vals) == 1 and not rows:
                meta["first_single_value_line"] = vals[0]
                continue
            need = 5 if has_sd else 4
            if len(vals) >= need:
                rows.append(vals[:need])

        if rows:
            arr = np.array(rows, dtype=float)
            if has_sd:
                stations[raw_station_id] = {
                    "station_id": station_id,
                    "raw_station_id": raw_station_id,
                    "camera_id": camera_id,
                    "meta": meta,
                    "time_s": arr[:, 0].tolist(),
                    "absolute_magnitude": arr[:, 1].tolist(),
                    "magnitude_sd": arr[:, 2].tolist(),
                    "length_km": arr[:, 3].tolist(),
                    "height_km": arr[:, 4].tolist(),
                }
            else:
                stations[raw_station_id] = {
                    "station_id": station_id,
                    "raw_station_id": raw_station_id,
                    "camera_id": camera_id,
                    "meta": meta,
                    "time_s": arr[:, 0].tolist(),
                    "absolute_magnitude": arr[:, 1].tolist(),
                    "length_km": arr[:, 2].tolist(),
                    "height_km": arr[:, 3].tolist(),
                }

    return stations


# ---------------------------- builder ----------------------------

def force_numeric_length(values: Any, n: int, fill_value: float = np.nan) -> List[float]:
    """Return a numeric list of exactly n values.

    Values which are missing, None, NaN, or infinite are replaced by fill_value.
    This avoids object arrays and prevents downstream indexing errors when
    combined arrays are sorted using the time array indices.
    """
    out: List[float] = []
    if values is None:
        values = []
    try:
        seq = list(values)
    except TypeError:
        seq = []

    for i in range(n):
        v = seq[i] if i < len(seq) else fill_value
        try:
            v = float(v)
        except (TypeError, ValueError):
            v = float(fill_value)

        # For magnitude arrays, fill_value may be np.nan for uncertainties.
        if not np.isfinite(v) and np.isfinite(fill_value):
            v = float(fill_value)
        out.append(v)

    return out

def interp_to_times(src_t: List[float], src_y: List[float], dst_t: List[float], fill_value: float = np.nan) -> List[float]:
    """Interpolate source values to destination times and always return len(dst_t).

    This is important for the Dynesty pipeline: every observation array
    (time_data, height, absolute_magnitudes, etc.) must have the same length.
    If photometry is missing or the destination time lies outside the photometry
    coverage, fill with fill_value instead of returning a shorter array.
    """
    if not dst_t:
        return []
    if not src_t or not src_y:
        return [float(fill_value) for _ in dst_t]
    t = np.asarray(src_t, dtype=float)
    y = np.asarray(src_y, dtype=float)
    mask = np.isfinite(t) & np.isfinite(y)
    t = t[mask]
    y = y[mask]
    if len(t) == 0:
        return [float(fill_value) for _ in dst_t]
    order = np.argsort(t)
    t = t[order]
    y = y[order]
    # Collapse duplicate times by keeping first occurrence.
    unique_t, unique_idx = np.unique(t, return_index=True)
    unique_y = y[unique_idx]
    d = np.asarray(dst_t, dtype=float)
    if len(unique_t) == 1:
        return [float(unique_y[0]) for _ in dst_t]
    out = np.interp(d, unique_t, unique_y, left=np.nan, right=np.nan)
    return [float(fill_value) if not np.isfinite(v) else float(v) for v in out]



def interp_to_axis(src_x: List[float], src_y: List[float], dst_x: List[float], fill_value: float = np.nan) -> List[float]:
    """Interpolate source values to destination values on an arbitrary axis.

    This is used for EN photometry because .lc/.rlc times are not always in
    the same time reference as .inp. The common columns are usually trajectory
    length and height, so mapping magnitudes by length/height is safer than
    mapping by time alone.
    """
    if not dst_x:
        return []
    if not src_x or not src_y:
        return [float(fill_value) for _ in dst_x]

    x = np.asarray(src_x, dtype=float)
    y = np.asarray(src_y, dtype=float)
    d = np.asarray(dst_x, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) == 0:
        return [float(fill_value) for _ in dst_x]

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    # Collapse duplicate axis values by keeping the first occurrence.
    unique_x, unique_idx = np.unique(x, return_index=True)
    unique_y = y[unique_idx]

    if len(unique_x) == 1:
        return [float(unique_y[0]) if np.isfinite(unique_y[0]) else float(fill_value) for _ in dst_x]

    out = np.interp(d, unique_x, unique_y, left=np.nan, right=np.nan)
    return [float(fill_value) if not np.isfinite(v) else float(v) for v in out]


def photometry_overlap_count(src_x: List[float], dst_x: List[float]) -> int:
    """Count destination points covered by the finite source axis range."""
    if not src_x or not dst_x:
        return 0
    sx = np.asarray(src_x, dtype=float)
    dx = np.asarray(dst_x, dtype=float)
    sx = sx[np.isfinite(sx)]
    dx = dx[np.isfinite(dx)]
    if len(sx) == 0 or len(dx) == 0:
        return 0
    lo, hi = np.nanmin(sx), np.nanmax(sx)
    return int(np.sum((dx >= lo) & (dx <= hi)))


def choose_photometry_axis(phot: Optional[Dict[str, Any]], time_s: List[float], length_km: List[float], height_km: List[float]) -> Tuple[str, List[float], List[float]]:
    """Choose the best axis for mapping EN photometry to the .inp trajectory.

    EN .lc/.rlc files sometimes have a time zero that is offset from .inp.
    For example, photometry time can start near +0.58 s while the matching
    .inp point is near -0.41 s. In that case interpolation by time produces
    +20 mag placeholders even though valid magnitudes exist. Length/height are
    already on the solved trajectory and should be preferred when available.
    """
    if not phot:
        return "none", [], []

    candidates = []
    if phot.get("length_km") and length_km:
        candidates.append(("length", phot.get("length_km", []), length_km))
    if phot.get("height_km") and height_km:
        candidates.append(("height", phot.get("height_km", []), height_km))
    if phot.get("time_s") and time_s:
        candidates.append(("time", phot.get("time_s", []), time_s))

    if not candidates:
        return "none", [], []

    # Prefer the axis that covers the most destination points. If tied, prefer
    # length, then height, then time, because length/height are safer for EN.
    priority = {"length": 0, "height": 1, "time": 2}
    candidates.sort(key=lambda c: (-photometry_overlap_count(c[1], c[2]), priority[c[0]]))
    return candidates[0]


def map_one_photometry_to_observation(
    phot: Optional[Dict[str, Any]],
    value_key: str,
    time_s: List[float],
    length_km: List[float],
    height_km: List[float],
    fill_value: float = np.nan,
) -> Tuple[List[float], str]:
    """Map one photometry series to the observation grid using length/height/time."""
    if not phot or not phot.get(value_key):
        return [float(fill_value) for _ in time_s], "none"

    axis_name, src_axis, dst_axis = choose_photometry_axis(phot, time_s, length_km, height_km)
    if axis_name == "none":
        return [float(fill_value) for _ in time_s], "none"

    return interp_to_axis(src_axis, phot.get(value_key, []), dst_axis, fill_value=fill_value), axis_name


def combine_lc_rlc_photometry(
    lc: Optional[Dict[str, Any]],
    rlc: Optional[Dict[str, Any]],
    time_s: List[float],
    length_km: List[float],
    height_km: List[float],
    missing_mag: float = 20.0,
) -> Tuple[List[float], List[float], Dict[str, Any]]:
    """Create full-length absolute magnitude and uncertainty arrays.

    LC is preferred where it maps to finite values. RLC is used only to fill
    gaps. Values still missing after both are set to missing_mag. This keeps all
    arrays the same length, but avoids replacing real photometry with +20 mag
    just because .lc/.rlc time references differ from .inp.
    """
    n = len(time_s)

    lc_mag, lc_axis = map_one_photometry_to_observation(lc, "absolute_magnitude", time_s, length_km, height_km, fill_value=np.nan)
    rlc_mag, rlc_axis = map_one_photometry_to_observation(rlc, "absolute_magnitude", time_s, length_km, height_km, fill_value=np.nan)
    lc_sd, lc_sd_axis = map_one_photometry_to_observation(lc, "magnitude_sd", time_s, length_km, height_km, fill_value=np.nan)

    lc_mag_arr = np.asarray(force_numeric_length(lc_mag, n, fill_value=np.nan), dtype=float)
    rlc_mag_arr = np.asarray(force_numeric_length(rlc_mag, n, fill_value=np.nan), dtype=float)
    lc_sd_arr = np.asarray(force_numeric_length(lc_sd, n, fill_value=np.nan), dtype=float)

    out_mag = np.where(np.isfinite(lc_mag_arr), lc_mag_arr, rlc_mag_arr)
    out_mag = np.where(np.isfinite(out_mag), out_mag, float(missing_mag))

    # Keep LC uncertainty where LC was used. RLC has no magnitude_sd in this format.
    out_sd = np.where(np.isfinite(lc_mag_arr), lc_sd_arr, np.nan)

    meta = {
        "lc_mapping_axis": lc_axis,
        "rlc_mapping_axis": rlc_axis,
        "lc_valid_points_on_inp_grid": int(np.sum(np.isfinite(lc_mag_arr))),
        "rlc_valid_points_on_inp_grid": int(np.sum(np.isfinite(rlc_mag_arr))),
        "filled_missing_mag_count": int(np.sum(out_mag == float(missing_mag))),
    }

    return out_mag.tolist(), out_sd.tolist(), meta

def derive_velocity_and_lag(time_s: List[float], length_km: List[float]) -> Tuple[List[float], List[float], Optional[float], Optional[float]]:
    """Return velocities_m_s, lag_m, v_init_m_s, v_avg_m_s."""
    if len(time_s) < 2 or len(length_km) < 2:
        return [], [], None, None

    t = np.asarray(time_s, dtype=float)
    length_m = 1000.0 * np.asarray(length_km, dtype=float)
    mask = np.isfinite(t) & np.isfinite(length_m)
    t = t[mask]
    length_m = length_m[mask]
    if len(t) < 2:
        return [], [], None, None

    order = np.argsort(t)
    t_sorted = t[order]
    l_sorted = length_m[order]

    # Remove exact duplicate times for derivative/interpolation.
    unique_t, unique_idx = np.unique(t_sorted, return_index=True)
    unique_l = l_sorted[unique_idx]
    if len(unique_t) < 2:
        return [], [], None, None

    dt = np.diff(unique_t)
    dl = np.diff(unique_l)
    seg_v = dl / dt
    finite_seg_v = seg_v[np.isfinite(seg_v) & (np.abs(seg_v) < 1_000_000)]
    if len(finite_seg_v) == 0:
        v_init = None
        v_avg = None
    else:
        # Use early segments but avoid single bad segment dominating.
        early = finite_seg_v[: min(5, len(finite_seg_v))]
        v_init = float(np.nanmedian(early))
        v_avg = float(np.nanmedian(finite_seg_v))

    if len(unique_t) >= 3:
        vel_unique = np.gradient(unique_l, unique_t).astype(float)
    else:
        vel_unique = np.array([finite_seg_v[0], finite_seg_v[0]], dtype=float)

    # Interpolate derivative and lag back to original order.
    vel_orig = np.interp(t, unique_t, vel_unique, left=np.nan, right=np.nan)
    if v_init is not None:
        line = unique_l[0] + v_init * (unique_t - unique_t[0])
        lag_unique = unique_l - line
        lag_orig = np.interp(t, unique_t, lag_unique, left=np.nan, right=np.nan)
    else:
        lag_orig = np.full_like(t, np.nan, dtype=float)

    # Restore to input ordering.
    result_vel = np.full(len(time_s), np.nan, dtype=float)
    result_lag = np.full(len(time_s), np.nan, dtype=float)
    input_indices = np.where(mask)[0]
    result_vel[input_indices] = vel_orig[np.argsort(order)] if len(vel_orig) == len(order) else vel_orig
    result_lag[input_indices] = lag_orig[np.argsort(order)] if len(lag_orig) == len(order) else lag_orig

    velocities = [None if not np.isfinite(x) else float(x) for x in result_vel]
    lag = [None if not np.isfinite(x) else float(x) for x in result_lag]
    return velocities, lag, v_init, v_avg


def empty_template() -> Dict[str, Any]:
    return {
        "jdt_ref": None,
        "meastype": None,
        "output_dir": None,
        "max_toffset": None,
        "verbose": False,
        "v_init_part": None,
        "v_init_ht": None,
        "fixed_time_offsets": {},
        "estimate_timing_vel": True,
        "fixed_times": True,
        "monte_carlo": False,
        "mc_runs": 0,
        "mc_pick_multiplier": None,
        "mc_noise_std": None,
        "geometric_uncert": False,
        "filter_picks": False,
        "calc_orbit": True,
        "show_plots": False,
        "show_jacchia": False,
        "save_results": True,
        "gravity_correction": None,
        "plot_all_spatial_residuals": False,
        "plot_file_type": None,
        "traj_id": None,
        "reject_n_sigma_outliers": None,
        "mc_cores": None,
        "file_name": None,
        "meas_count": 0,
        "observations": [],
        "los_mini_status": None,
        "t_ref_station": 0,
        "time_diffs_final": [],
        "intersection_list": [],
        "rbeg_lat": None,
        "rbeg_lon": None,
        "rbeg_ele": None,
        "rbeg_jd": None,
        "rend_lat": None,
        "rend_lon": None,
        "rend_ele": None,
        "rend_jd": None,
        "state_vect": None,
        "incident_angles": [],
        "state_vect_mini": None,
        "radiant_eci_mini": None,
        "radiant_eq_mini": None,
        "v_init": None,
        "v_avg": None,
        "timing_minimization_successful": None,
        "velocity_fit": None,
        "jacchia_fit": None,
        "timing_res": None,
        "timing_stddev": None,
        "state_vect_avg": None,
        "jd_avg": None,
        "orbit": {},
        "uncertainties": None,
        "uncertanties": None,
        "orbit_cov": None,
        "state_vect_cov": None,
        "avg_radiant": None,
        "radiant_eq": None,
        "best_conv_inter": {},
        "stations_time_dict": {},
        "v_init_stddev": None,
        "time_diffs": [],
        "gravity_factor": None,
        "v0z": None,
    }


def load_template(path: Optional[Path]) -> Dict[str, Any]:
    """Load only solver/config constants from a template JSON.

    Do NOT copy trajectory solution data from the template.  The template JSON is
    from a different meteor, so fields such as orbit.zc, state_vect, radiant,
    observations, LLA arrays, uncertainties, etc. would be stale and physically
    wrong for the EN event.
    """
    out = empty_template()
    if path is None:
        return out

    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    for key in CONSTANT_TEMPLATE_KEYS:
        if key in obj:
            out[key] = copy.deepcopy(obj[key])

    # Good defaults for the converted EN files.
    out["meastype"] = 1 if out.get("meastype") is None else out["meastype"]
    out["calc_orbit"] = True if out.get("calc_orbit") is None else out["calc_orbit"]
    out["gravity_factor"] = 1.0 if out.get("gravity_factor") is None else out["gravity_factor"]
    return out


def get_en_zc_rad(res: Dict[str, Any]) -> Optional[float]:
    """Return apparent zenith angle zc from EN .res, in radians.

    WMPL normally computes this in calcLLA/calcECIEqAltAz from the apparent
    radiant altitude at the beginning point.  EN already reports the same
    quantity as RADIANT ... ZENITH DISTANCE / BEGINNING, so use that directly.
    """
    zd = res.get("radiant_azimuth_zd", {}).get("begin", {}).get("zenith_distance_deg")
    return deg2rad(zd)


def get_en_apparent_elevation_rad(res: Dict[str, Any]) -> Optional[float]:
    zc = get_en_zc_rad(res)
    return None if zc is None else (math.pi / 2.0 - zc)


def local_zenith_distance_from_ra_dec(
    ra_rad: Optional[float],
    dec_rad: Optional[float],
    lat_rad: Optional[float],
    lon_rad: Optional[float],
    gst_rad: Optional[float],
) -> Optional[float]:
    """Compute local zenith distance from RA/Dec, latitude, longitude and GST.

    This is a lightweight replacement for the part of WMPL calcLLA that would
    normally start from the radiant vector.  EN .res already gives GST, the
    beginning latitude/longitude, and the geocentric RA/Dec.
    """
    if None in (ra_rad, dec_rad, lat_rad, lon_rad, gst_rad):
        return None
    lst = (gst_rad + lon_rad) % (2.0 * math.pi)
    hour_angle = (lst - ra_rad + math.pi) % (2.0 * math.pi) - math.pi
    sin_alt = (
        math.sin(lat_rad) * math.sin(dec_rad)
        + math.cos(lat_rad) * math.cos(dec_rad) * math.cos(hour_angle)
    )
    alt = math.asin(max(-1.0, min(1.0, sin_alt)))
    return math.pi / 2.0 - alt


def get_en_zg_rad(res: Dict[str, Any]) -> Optional[float]:
    """Estimate geocentric zenith angle zg from EN geocentric RA/Dec.

    Unlike zc, EN does not print a dedicated geocentric zenith distance in the
    shown .res summary, but it does print geocentric RA/Dec, GST, and trajectory
    beginning LLA.  Compute zg from those instead of copying the template value.
    """
    begin = res.get("begin", {})
    geo = res.get("radiant_geocentric", {})
    return local_zenith_distance_from_ra_dec(
        deg2rad(geo.get("ra_deg")),
        deg2rad(geo.get("dec_deg")),
        deg2rad(begin.get("lat_deg_north")),
        deg2rad(begin.get("lon_deg_east")),
        deg2rad(res.get("sidereal_time_greenwich_deg")),
    )


def get_en_azimuth_apparent_rad(res: Dict[str, Any]) -> Optional[float]:
    """Return EN apparent radiant azimuth in radians when available.

    The EN summary says azimuth is S=0.  To avoid silently mixing azimuth
    conventions, we keep the original S=0 value in orbit.azimuth_apparent_en_s0.
    For orbit.azimuth_apparent we also store the S=0 value in radians only as a
    record of the EN summary; downstream code here mainly needs zc/elevation.
    """
    az = res.get("radiant_azimuth_zd", {}).get("begin", {}).get("azimuth_s0_deg")
    return deg2rad(az)


def get_en_v0z(res: Dict[str, Any]) -> Optional[float]:
    """Compute vertical velocity component using the same sign convention as WMPL.

    WMPL calcLLA/calcECIEqAltAz sets: v0z = -v_init*cos(zc).
    """
    zc = get_en_zc_rad(res)
    v_init = km2m(res.get("radiant_apparent", {}).get("velocity_km_s"))
    if zc is None or v_init is None:
        return None
    return -v_init * math.cos(zc)


def clean_event_specific_traj_fields(traj: Dict[str, Any]) -> None:
    """Remove any old-solution fields that should not be inherited from a template."""
    for key in [
        "state_vect", "state_vect_mini", "radiant_eci_mini", "radiant_eq_mini",
        "state_vect_avg", "avg_radiant", "orbit_cov", "state_vect_cov",
        "uncertainties", "uncertanties", "best_conv_inter", "intersection_list",
        "stations_time_dict", "stations_time_dict_copy", "time_diffs",
        "time_diffs_final", "fixed_time_offsets_copy", "velocity_fit",
        "jacchia_fit", "timing_res", "timing_stddev",
        "timing_minimization_successful",
    ]:
        if key in traj:
            # Keep expected container types for a few fields.
            if key in {"intersection_list"}:
                traj[key] = []
            elif key in {"stations_time_dict", "stations_time_dict_copy", "best_conv_inter"}:
                traj[key] = {}
            elif key in {"time_diffs", "time_diffs_final"}:
                traj[key] = []
            else:
                traj[key] = None


def make_orbit_like(res: Dict[str, Any]) -> Dict[str, Any]:
    apparent = res.get("radiant_apparent", {})
    geo = res.get("radiant_geocentric", {})
    helio = res.get("radiant_heliocentric", {})
    elems = res.get("orbital_elements", {})
    avg = res.get("average", {})
    begin = res.get("begin", avg)

    zc = get_en_zc_rad(res)
    zg = get_en_zg_rad(res)
    elevation_apparent = get_en_apparent_elevation_rad(res)
    azimuth_apparent_en_s0 = get_en_azimuth_apparent_rad(res)

    orbit = {
        "ra": deg2rad(apparent.get("ra_deg")),
        "dec": deg2rad(apparent.get("dec_deg")),
        "azimuth_apparent": azimuth_apparent_en_s0,
        "azimuth_apparent_en_s0": azimuth_apparent_en_s0,
        "elevation_apparent": elevation_apparent,
        "v_avg": km2m(res.get("velocity_avg_km_s")),
        "v_init": km2m(apparent.get("velocity_km_s")),
        "v_init_stddev": None,
        "jd_ref": res.get("julian_date"),
        "jd_dyn": res.get("julian_date"),
        "lst_ref": deg2rad(res.get("sidereal_time_greenwich_deg")),
        "lon_ref": deg2rad(begin.get("lon_deg_east")),
        "lat_ref": deg2rad(begin.get("lat_deg_north")),
        "ht_ref": km2m(begin.get("height_km")),
        "zc": zc,
        # zc comes directly from EN's apparent zenith distance.  zg is computed
        # from EN geocentric RA/Dec + GST + beginning latitude/longitude.
        "zg": zg,
        "v_inf": km2m(apparent.get("velocity_km_s")),
        "v_g": km2m(geo.get("velocity_km_s")),
        "ra_g": deg2rad(geo.get("ra_deg")),
        "dec_g": deg2rad(geo.get("dec_deg")),
        "L_h": deg2rad(helio.get("ecl_lon_deg")),
        "B_h": deg2rad(helio.get("ecl_lat_deg")),
        "v_h": km2m(helio.get("velocity_km_s")),
        "la_sun": deg2rad(res.get("solar_longitude_deg")),
        "a": elems.get("a_au"),
        "e": elems.get("e"),
        "i": deg2rad(elems.get("incl_deg")),
        "peri": deg2rad(elems.get("omega_deg")),
        "node": deg2rad(elems.get("asc_node_deg")),
        "pi": deg2rad(elems.get("pi_deg")),
        "q": elems.get("q_per_au"),
        "Q": elems.get("q_aph_au"),
        "Tj": res.get("tisserand"),
        "T": res.get("period_years"),
        "terminal_velocity": km2m(res.get("terminal_velocity_km_s")),
    }
    return orbit




def relabel_photometry_for_observation(phot: Optional[Dict[str, Any]], full_station_id: str) -> Optional[Dict[str, Any]]:
    """Return a copy of a .lc/.rlc photometry block labelled with the full camera ID.

    The source .lc/.rlc files often only carry the base station number (e.g. 102),
    while the .res camera list distinguishes multiple cameras at that station
    (e.g. 102-30, 102-72, 102-71).  Downstream JSON/pickle inspection was still
    showing nested photometry station_id='cz_102'.  This function makes the
    nested photometry block use the same full station_id as the parent observation.
    """
    if phot is None:
        return None

    out = copy.deepcopy(phot)
    old_station_id = out.get("station_id")
    old_camera_id = out.get("camera_id")

    full_camera_id = str(full_station_id)
    if full_camera_id.startswith("cz_"):
        full_camera_id = full_camera_id[3:]

    out["station_id"] = full_station_id
    out["camera_id"] = full_camera_id
    out["base_station_id"] = camera_base_id(full_camera_id)
    out["source_station_id"] = old_station_id
    out["source_camera_id"] = old_camera_id
    return out


def assign_full_camera_ids_to_observations(
    observations: List[Dict[str, Any]],
    camera_list: Optional[List[str]],
    station_prefix: str = "cz_",
) -> List[Dict[str, Any]]:
    """Patch observation station IDs using the full camera IDs from the .res file.

    Some EN files list full cameras in the .res file, e.g.
    ``105-30, 102-30, 104-30, 111-30, 204-33, 102-72, 102-71``,
    while .inp/.lc/.rlc blocks often only contain the base station, e.g. ``102``.

    This function is intentionally a final pass over the already-built observations,
    so it fixes old/partial parsing too. It assigns repeated base IDs by occurrence:
    the three observations with base ``102`` become ``102-30``, ``102-72``,
    and ``102-71`` instead of all becoming ``102``.
    """
    if not camera_list:
        return observations

    queues: Dict[str, List[str]] = defaultdict(list)
    for cam in camera_list:
        cam = str(cam).strip()
        if cam:
            queues[camera_base_id(cam)].append(cam)

    used_full_ids: Dict[str, int] = defaultdict(int)

    for obs in observations:
        old_station_id = obs.get("station_id")
        old_camera = remove_station_prefix(old_station_id, station_prefix)

        src = obs.get("source_inp") if isinstance(obs.get("source_inp"), dict) else {}
        raw_station = src.get("raw_station_id") or obs.get("raw_station_id") or old_camera
        raw_station = remove_station_prefix(raw_station, station_prefix)
        base = camera_base_id(raw_station)

        # If the observation already has a full hyphenated camera ID, keep it and
        # remove that ID from the queue so later duplicate-base observations do
        # not reuse it. Otherwise assign the next full camera ID for this base.
        if "-" in old_camera:
            full_camera = old_camera
            if full_camera in queues.get(base, []):
                queues[base].remove(full_camera)
        elif queues.get(base):
            full_camera = queues[base].pop(0)
        else:
            full_camera = old_camera

        full_station_id = make_prefixed_camera_id(full_camera, station_prefix)
        used_full_ids[full_station_id] += 1
        record_id = full_camera if used_full_ids[full_station_id] == 1 else f"{full_camera}__{used_full_ids[full_station_id]}"

        obs["station_id"] = full_station_id
        obs["camera_id"] = full_camera
        obs["base_station_id"] = camera_base_id(full_camera)
        obs["raw_station_id"] = raw_station
        obs["record_id"] = record_id

        if isinstance(src, dict):
            src["station_id"] = full_station_id
            src["camera_id"] = full_camera
            src["base_station_id"] = camera_base_id(full_camera)
            src["raw_station_id"] = raw_station
            src["record_id"] = record_id
            meta = src.get("meta")
            if isinstance(meta, dict):
                meta["camera_id"] = full_camera
                meta["base_station_id"] = camera_base_id(full_camera)
                meta["raw_station_id"] = raw_station

        # Keep nested photometry labels synchronized with the parent observation.
        for phot_key in ("photometry_lc", "photometry_rlc"):
            phot = obs.get(phot_key)
            if isinstance(phot, dict):
                phot.setdefault("source_station_id", phot.get("station_id"))
                phot.setdefault("source_camera_id", phot.get("camera_id"))
                phot["station_id"] = full_station_id
                phot["camera_id"] = full_camera
                phot["base_station_id"] = camera_base_id(full_camera)

    return observations


def build_observation(
    station_id: str,
    inp: Dict[str, Any],
    lc: Optional[Dict[str, Any]],
    rlc: Optional[Dict[str, Any]],
    jdt_ref: Optional[float],
    obs_id: int,
) -> Dict[str, Any]:
    time_s = list(inp.get("time_s", []))
    length_km = list(inp.get("length_km", []))
    height_km = list(inp.get("height_km", []))
    velocities, lag, v_init, v_avg = derive_velocity_and_lag(time_s, length_km)

    # Photometry is not guaranteed to share the same time zero as the .inp
    # trajectory points. Map photometry using trajectory length/height first,
    # and use time only as a fallback. Always return len(time_s) so the Dynesty
    # arrays can be sorted together.
    n_time = len(time_s)
    missing_mag = 20.0
    abs_mag, mag_sd, photometry_mapping = combine_lc_rlc_photometry(
        lc=lc,
        rlc=rlc,
        time_s=time_s,
        length_km=length_km,
        height_km=height_km,
        missing_mag=missing_mag,
    )

    # Last safety check: no shorter/longer photometry arrays are allowed.
    abs_mag = force_numeric_length(abs_mag, n_time, fill_value=missing_mag)
    mag_sd = force_numeric_length(mag_sd, n_time, fill_value=np.nan)

    jd_data = None
    if jdt_ref is not None:
        jd_data = [jdt_ref + t / 86400.0 for t in time_s]

    length_m = [1000.0 * x for x in length_km]
    height_m = [1000.0 * x for x in height_km]

    lc_for_obs = relabel_photometry_for_observation(lc, station_id)
    rlc_for_obs = relabel_photometry_for_observation(rlc, station_id)

    obs = {
        "station_id": station_id,
        "obs_id": obs_id,
        "jdt_ref": jdt_ref,
        "time_data": time_s,
        "JD_data": jd_data,
        "kmeas": len(time_s),
        "ignore_station": False,
        "ignore_list": [0] * len(time_s),
        "length": length_m,
        "state_vect_dist": length_m,
        "meas_ht": height_m,
        "model_ht": height_m,
        "velocities": velocities,
        "velocities_prev_point": velocities,
        "v_init": v_init,
        "v_avg": v_avg,
        "lag": lag,
        "absolute_magnitudes": abs_mag,
        "magnitudes": abs_mag,
        "magnitude_sd": mag_sd,
        "photometry_lc": lc_for_obs,
        "photometry_rlc": rlc_for_obs,
        "photometry_mapping": photometry_mapping,
        "source_inp": inp,
        # Fields present in WMPL trajectories but unavailable from t,l,h files.
        "lat": None,
        "lon": None,
        "ele": None,
        "ra_data": [],
        "dec_data": [],
        "azim_data": [],
        "elev_data": [],
        "model_lat": [],
        "model_lon": [],
        "meas_lat": [],
        "meas_lon": [],
        "meas_range": [],
        "model_range": [],
        "comment": "Converted from EN .inp/.lc/.rlc; angular line-of-sight data are not present in these source files.",
    }
    return obs


def build_trajectory_like(
    files: Dict[str, str],
    template: Dict[str, Any],
    source_name: str,
    include_raw: bool = False,
) -> Dict[str, Any]:
    res = parse_res(files[".res"]) if ".res" in files else {}
    camera_list = res.get("camera_list", [])
    inp_by_station = parse_inp(files[".inp"], camera_list=camera_list) if ".inp" in files else {}
    lc_by_station = parse_lc_like(files[".lc"], has_sd=True, ext=".lc") if ".lc" in files else {}
    rlc_by_station = parse_lc_like(files[".rlc"], has_sd=False, ext=".rlc") if ".rlc" in files else {}

    code = res.get("meteor_code") or Path(source_name).stem
    jdt_ref = res.get("julian_date")

    traj = copy.deepcopy(template)
    clean_event_specific_traj_fields(traj)
    traj["jdt_ref"] = jdt_ref
    traj["file_name"] = code
    traj["traj_id"] = code
    traj["source_name"] = source_name
    traj["source_format"] = "EN .res/.inp/.lc/.rlc"
    traj["converted_utc"] = datetime.now(timezone.utc).isoformat()
    traj["metadata_res"] = res

    begin = res.get("begin", {})
    end = res.get("end", {})
    avg = res.get("average", {})
    duration = res.get("duration", {})

    traj["rbeg_lat"] = deg2rad(begin.get("lat_deg_north"))
    traj["rbeg_lon"] = deg2rad(begin.get("lon_deg_east"))
    traj["rbeg_ele"] = km2m(begin.get("height_km"))
    traj["rbeg_ele_wgs84"] = traj["rbeg_ele"]
    traj["rbeg_jd"] = jdt_ref
    traj["rend_lat"] = deg2rad(end.get("lat_deg_north"))
    traj["rend_lon"] = deg2rad(end.get("lon_deg_east"))
    traj["rend_ele"] = km2m(end.get("height_km"))
    traj["rend_ele_wgs84"] = traj["rend_ele"]
    if jdt_ref is not None and duration.get("seconds") is not None:
        traj["rend_jd"] = jdt_ref + duration["seconds"] / 86400.0
    else:
        traj["rend_jd"] = jdt_ref

    # Lowest point: use END from .res for normal fireballs; if later you add
    # grazers, this can be replaced by the minimum of model_ht.
    traj["htmin_lat"] = traj["rend_lat"]
    traj["htmin_lon"] = traj["rend_lon"]
    traj["htmin_ele"] = traj["rend_ele"]
    traj["htmin_ele_wgs84"] = traj["rend_ele"]
    traj["htmin_jd"] = traj["rend_jd"]

    traj["orbit"] = make_orbit_like(res)
    traj["v_init"] = km2m(res.get("radiant_apparent", {}).get("velocity_km_s"))
    traj["v_avg"] = km2m(res.get("velocity_avg_km_s"))
    traj["jd_avg"] = jdt_ref
    traj["radiant_eq"] = [
        deg2rad(res.get("radiant_apparent", {}).get("ra_deg")),
        deg2rad(res.get("radiant_apparent", {}).get("dec_deg")),
    ]
    traj["best_conv_inter"] = {
        "maximal_convergence_angle_deg": res.get("maximal_convergence_angle_deg"),
        "closest_distance_km": res.get("closest_distance_km"),
        "closest_station": res.get("closest_station"),
    }

    # Build observations from each .inp record. Duplicate station IDs are kept as
    # separate observations, but they can share the same .lc/.rlc station photometry.
    # Preserve .inp order. This matters for assigning repeated base stations to
    # full camera IDs from camera_list, e.g. 102-30, 102-72, 102-71.
    inp_record_ids = list(inp_by_station.keys())
    raw_inp_station_ids = {camera_base_id(inp_by_station[k].get("raw_station_id", str(inp_by_station[k].get("station_id", "")).replace("cz_", ""))) for k in inp_by_station}
    phot_only_station_ids = sorted((set(lc_by_station) | set(rlc_by_station)) - raw_inp_station_ids, key=lambda x: str(x))
    observation_keys = inp_record_ids + phot_only_station_ids

    observations: List[Dict[str, Any]] = []
    for obs_id, key in enumerate(observation_keys, start=1):
        inp = inp_by_station.get(key)
        if inp is None:
            raw_station_id = str(key)
            station_id = make_prefixed_camera_id(raw_station_id)
            # If there is only photometry and no trajectory t/l/h, build a minimal observation on that time grid.
            phot = lc_by_station.get(raw_station_id) or rlc_by_station.get(raw_station_id)
            inp = {
                "station_id": station_id,
                "raw_station_id": raw_station_id,
                "record_id": key,
                "meta": {"note": "No .inp block for this station; built from photometry only."},
                "time_s": phot.get("time_s", []) if phot else [],
                "length_km": phot.get("length_km", []) if phot else [],
                "height_km": phot.get("height_km", []) if phot else [],
            }
        station_id = inp.get("station_id", key)
        raw_station_id = camera_base_id(inp.get("raw_station_id", str(station_id).replace("cz_", "").split("__")[0]))
        obs = build_observation(
            station_id=station_id,
            inp=inp,
            lc=lc_by_station.get(raw_station_id),
            rlc=rlc_by_station.get(raw_station_id),
            jdt_ref=jdt_ref,
            obs_id=obs_id,
        )
        obs["record_id"] = inp.get("record_id", key)
        observations.append(obs)

    # Final robust pass: preserve hyphenated camera names from metadata_res["camera_list"].
    # This prevents all repeated 102 cameras from collapsing to the same cz_102 ID.
    observations = assign_full_camera_ids_to_observations(observations, camera_list)

    traj["observations"] = observations
    traj["meas_count"] = int(sum(len(o.get("time_data", [])) for o in observations))
    traj["stations_time_dict"] = {
        o["station_id"]: {"time_data": o.get("time_data", []), "JD_data": o.get("JD_data", [])}
        for o in observations
    }
    if observations:
        # Use first station as reference by convention.
        traj["t_ref_station"] = 0
        traj["time_diffs_final"] = [0.0 for _ in observations]
        traj["time_diffs"] = [0.0 for _ in observations]

    # These are added by WMPL.loadPickle() for older real trajectory pickles.
    # Set them here explicitly so they are never None.
    traj["gravity_factor"] = 1.0
    # WMPL computes this from zc in calcLLA/calcECIEqAltAz; here zc is parsed
    # from the EN .res beginning zenith distance, so compute v0z immediately.
    traj["v0z"] = get_en_v0z(res)
    if traj["v0z"] is None:
        traj["v0z"] = 0.0
    traj["fixed_time_offsets_copy"] = copy.deepcopy(traj.get("fixed_time_offsets", {}))
    traj["stations_time_dict_copy"] = copy.deepcopy(traj.get("stations_time_dict", {}))

    if include_raw:
        traj["raw_files"] = files

    return traj


# ---------------------------- input discovery ----------------------------

def read_event_files_from_zip(zip_path: Path) -> List[Tuple[str, Dict[str, str]]]:
    groups: Dict[str, Dict[str, str]] = defaultdict(dict)
    with zipfile.ZipFile(zip_path, "r") as z:
        for name in z.namelist():
            ext = Path(name).suffix.lower()
            if ext not in {".res", ".inp", ".lc", ".rlc"}:
                continue
            stem = clean_code_from_stem(Path(name).name)
            groups[stem][ext] = safe_decode(z.read(name))
    return [(f"{zip_path.name}:{stem}", files) for stem, files in groups.items()]


def read_event_files_from_directory(root: Path) -> List[Tuple[str, Dict[str, str]]]:
    events: List[Tuple[str, Dict[str, str]]] = []

    # First process any zip files recursively.
    for zip_path in sorted(root.rglob("*.zip")):
        events.extend(read_event_files_from_zip(zip_path))

    # Then process loose files recursively, grouped by stem. This also handles one event folder.
    loose_groups: Dict[Tuple[Path, str], Dict[str, str]] = defaultdict(dict)
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        ext = path.suffix.lower()
        if ext not in {".res", ".inp", ".lc", ".rlc"}:
            continue
        stem = clean_code_from_stem(path.name)
        loose_groups[(path.parent, stem)][ext] = safe_decode(path.read_bytes())

    for (parent, stem), files in loose_groups.items():
        events.append((str(parent / stem), files))

    return events


def collect_events(input_path: Path) -> List[Tuple[str, Dict[str, str]]]:
    if input_path.is_file() and input_path.suffix.lower() == ".zip":
        return read_event_files_from_zip(input_path)
    if input_path.is_dir():
        return read_event_files_from_directory(input_path)
    if input_path.is_file() and input_path.suffix.lower() in {".res", ".inp", ".lc", ".rlc"}:
        stem = clean_code_from_stem(input_path.name)
        files = {input_path.suffix.lower(): safe_decode(input_path.read_bytes())}
        parent = input_path.parent
        for ext in [".res", ".inp", ".lc", ".rlc"]:
            candidate = parent / f"{stem}{ext}"
            if candidate.exists():
                files[ext] = safe_decode(candidate.read_bytes())
        return [(str(parent / stem), files)]
    raise FileNotFoundError(f"Input path not found or unsupported: {input_path}")


# ---------------------------- object conversion/output ----------------------------

def list_to_numpy(obj: Any) -> Any:
    """Optionally convert numeric lists to numpy arrays while leaving nested records readable."""
    if isinstance(obj, dict):
        return {k: list_to_numpy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        if not obj:
            return obj
        if all((x is None) or isinstance(x, (int, float, bool)) for x in obj):
            if any(x is None for x in obj):
                return np.array(obj, dtype=object)
            return np.array(obj)
        return [list_to_numpy(x) for x in obj]
    return obj


def to_namespace(obj: Any) -> Any:
    """Recursively convert dictionaries to SimpleNamespace for attribute access."""
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [to_namespace(x) for x in obj]
    return obj


def load_pickle_py2_compatible(path: Path) -> Any:
    """Load pickle in the same way as WMPL's loadPickle function."""
    with open(path, "rb") as f:
        if sys.version_info[0] < 3:
            return pickle.load(f)
        return pickle.load(f, encoding="latin1")


def find_wmpl_orbit_class():
    """Try known WMPL Orbit import paths."""
    errors = []
    for mod_name, cls_name in [
        ("wmpl.Trajectory.Orbit", "Orbit"),
        ("wmpl.Trajectory.Trajectory", "Orbit"),
    ]:
        try:
            mod = __import__(mod_name, fromlist=[cls_name])
            return getattr(mod, cls_name)
        except Exception as exc:
            errors.append(f"{mod_name}.{cls_name}: {exc}")
    raise ImportError("Could not import a WMPL Orbit class. Use --template-pickle from a normal WMPL trajectory pickle. Tried: " + "; ".join(errors))


def make_wmpl_orbit_object(orbit_dict: Dict[str, Any], template_orbit: Any = None) -> Any:
    """Create an orbit object with fixMissingParameters(), preferably from the template pickle."""
    if template_orbit is not None:
        orbit = copy.deepcopy(template_orbit)
        if hasattr(orbit, "__dict__"):
            orbit.__dict__.clear()
    else:
        Orbit = find_wmpl_orbit_class()
        orbit = Orbit()

    for key, value in orbit_dict.items():
        setattr(orbit, key, value)

    if hasattr(orbit, "fixMissingParameters"):
        try:
            orbit.fixMissingParameters()
        except Exception as exc:
            print(f"Warning: orbit.fixMissingParameters() failed during conversion: {exc}")
    else:
        raise TypeError("The orbit object has no fixMissingParameters() method; loadPickle() will fail.")

    return orbit


def make_wmpl_observed_points_object(obs_dict: Dict[str, Any], template_obs: Any = None) -> Any:
    """Create a WMPL ObservedPoints object without rerunning the solver.

    The EN files contain already-solved time/length/height/photometry, not the original
    angular line-of-sight measurements needed by ObservedPoints.__init__. Therefore we
    allocate an ObservedPoints instance with object.__new__ and fill its __dict__ directly.
    This preserves the class/type expected by older WMPL-style downstream code while
    avoiding fake RA/Dec reconstruction.
    """
    data = copy.deepcopy(obs_dict)

    # Prefer cloning a real observation from a normal trajectory pickle.
    if template_obs is not None:
        obs = copy.deepcopy(template_obs)
        if hasattr(obs, "__dict__"):
            obs.__dict__.clear()
    else:
        try:
            from wmpl.Trajectory.Trajectory import ObservedPoints
            obs = object.__new__(ObservedPoints)
        except Exception:
            # Last resort: this still gives attribute access, but the class will not be WMPL.
            obs = SimpleNamespace()

    # Set all parsed/calculated values.
    for key, value in data.items():
        setattr(obs, key, value)

    # Compatibility defaults normally present on ObservedPoints after the solver.
    n = len(getattr(obs, "time_data", []))
    defaults = {
        "meas1": np.array([]),
        "meas2": np.array([]),
        "ignore_station": False,
        "ignore_list": np.zeros(n, dtype=np.uint8),
        "kmeas": n,
        "azim_data": np.array([]),
        "elev_data": np.array([]),
        "ra_data": np.array([]),
        "dec_data": np.array([]),
        "ra_data_los": np.array([]),
        "dec_data_los": np.array([]),
        "magnitudes": np.array([]),
        "fov_beg": None,
        "fov_end": None,
        "incident_angle": None,
        "weight": 1.0,
        "h_residuals": None,
        "h_res_rms": None,
        "v_residuals": None,
        "v_res_rms": None,
        "lag_line": None,
        "v_init_stddev": None,
        "jacchia_fit": None,
        "model_ra": np.array([]),
        "model_dec": np.array([]),
        "model_azim": np.array([]),
        "model_elev": np.array([]),
        "model_fit1": np.array([]),
        "model_fit2": np.array([]),
        "meas_eci": np.empty((0, 3)),
        "meas_eci_los": np.empty((0, 3)),
        "model_eci": np.empty((0, 3)),
        "meas_lat": np.array([]),
        "meas_lon": np.array([]),
        "meas_range": np.array([]),
        "model_lat": np.array([]),
        "model_lon": np.array([]),
        "model_range": np.array([]),
        "excluded_time": None,
        "excluded_indx_range": [],
        "ang_res": None,
        "ang_res_std": None,
    }
    for key, value in defaults.items():
        if not hasattr(obs, key) or getattr(obs, key) is None:
            setattr(obs, key, value)

    # Ensure array-like fields are numpy arrays, matching WMPL pickles.
    array_fields = [
        "time_data", "JD_data", "ignore_list", "length", "state_vect_dist",
        "meas_ht", "model_ht", "velocities", "velocities_prev_point", "lag",
        "absolute_magnitudes", "magnitudes", "magnitude_sd"
    ]
    for key in array_fields:
        if hasattr(obs, key) and isinstance(getattr(obs, key), list):
            setattr(obs, key, np.array(getattr(obs, key)))

    # The EN files give trajectory solution coordinates but not station coordinates.
    # Keep these attributes present to avoid AttributeError in downstream code.
    for key in ["lat", "lon", "ele"]:
        if not hasattr(obs, key):
            setattr(obs, key, None)

    return obs


def make_wmpl_trajectory_object(traj_dict: Dict[str, Any], template_traj: Any = None, *, numpy_arrays: bool = True) -> Any:
    """Return a real WMPL Trajectory object as the pickle root.

    This is the important part for your Dynesty code. The object written to disk is
    not {"events": [...]}; it is directly a Trajectory instance, so loadPickle()
    returns something with traj.__dict__, traj.orbit and traj.observations.
    """
    data = copy.deepcopy(traj_dict)

    # Convert numeric lists to numpy arrays before objectifying observations.
    if numpy_arrays:
        data = list_to_numpy(data)

    template_orbit = getattr(template_traj, "orbit", None) if template_traj is not None else None
    data["orbit"] = make_wmpl_orbit_object(data.get("orbit", {}), template_orbit=template_orbit)

    # Observations in a real WMPL pickle are ObservedPoints objects.
    template_obs_list = getattr(template_traj, "observations", []) if template_traj is not None else []
    template_obs = template_obs_list[0] if len(template_obs_list) else None
    data["observations"] = [
        make_wmpl_observed_points_object(obs, template_obs=template_obs) if isinstance(obs, dict) else obs
        for obs in data.get("observations", [])
    ]

    # These two must be real numbers, not None, because loadPickle only adds them if missing.
    data["gravity_factor"] = 1.0 if data.get("gravity_factor") is None else data.get("gravity_factor")
    data["v0z"] = 0.0 if data.get("v0z") is None else data.get("v0z")

    if template_traj is not None:
        traj = copy.deepcopy(template_traj)
        traj.__dict__.clear()
        traj.__dict__.update(data)
        return traj

    # Fallback: create a new real WMPL Trajectory instance. This requires running
    # the converter in the same Python environment where WMPL works.
    from wmpl.Trajectory.Trajectory import Trajectory
    # chek if data.get("jdt_ref") if unsupported type for timedelta days component: NoneType
    traj = Trajectory(data.get("jdt_ref"), output_dir=data.get("output_dir"), meastype=data.get("meastype", 1))
    traj.__dict__.clear()
    traj.__dict__.update(data)
    return traj


def json_safe(obj: Any) -> Any:
    """Convert numpy/object instances into JSON-safe objects for sidecar only."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, SimpleNamespace):
        return json_safe(vars(obj))
    if hasattr(obj, "__dict__") and obj.__class__.__module__.startswith("wmpl"):
        return json_safe(vars(obj))
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return str(obj)


def write_pipeline_pickles(trajectories: List[Dict[str, Any]], args: argparse.Namespace, template_traj: Any) -> List[Path]:
    """Write one pipeline-compatible pickle per event and return paths."""
    written: List[Path] = []

    # Multiple events must be a directory in pipeline mode.
    if len(trajectories) > 1:
        out_dir = args.output
        if out_dir is None:
            out_dir = args.input if args.input.is_dir() else args.input.parent
        if out_dir.suffix.lower() in {".pkl", ".pickle"}:
            raise SystemExit("Pipeline mode found multiple events, so --output must be a directory, not one .pickle file.")
        out_dir.mkdir(parents=True, exist_ok=True)
        for traj_dict in trajectories:
            traj_obj = make_wmpl_trajectory_object(traj_dict, template_traj, numpy_arrays=args.numpy_arrays)
            code = str(traj_dict.get("file_name") or traj_dict.get("traj_id") or "event")
            out_path = out_dir / f"{code}_trajectory.pickle"
            with open(out_path, "wb") as f:
                pickle.dump(traj_obj, f, protocol=pickle.HIGHEST_PROTOCOL)
            written.append(out_path)
        return written

    # One event: output is a pickle path by default.
    traj_dict = trajectories[0]
    if args.output is None:
        code = str(traj_dict.get("file_name") or traj_dict.get("traj_id") or args.input.stem)
        default_dir = args.input if args.input.is_dir() else args.input.parent
        args.output = default_dir / f"{code}_trajectory.pickle"
    elif args.output.suffix.lower() not in {".pkl", ".pickle"}:
        code = str(traj_dict.get("file_name") or traj_dict.get("traj_id") or args.input.stem)
        args.output.mkdir(parents=True, exist_ok=True)
        args.output = args.output / f"{code}_trajectory.pickle"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    traj_obj = make_wmpl_trajectory_object(traj_dict, template_traj, numpy_arrays=args.numpy_arrays)
    with open(args.output, "wb") as f:
        pickle.dump(traj_obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    written.append(args.output)
    return written


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default=Path(r"C:\Users\maxiv\Documents\UWO\Papers\4)Mars meteors\Fireball\EN181125_040337"), type=Path, help="Folder, zip, or one EN*.res/.inp/.lc/.rlc file")
    parser.add_argument("--template-json", default=Path(r"C:\Users\maxiv\WMPG-repoMAX\Code\Utils\20230811_082648_trajectory.json"), type=Path, help="JSON made from your original WMPL trajectory pickle; used as key/default template")
    parser.add_argument("--template-pickle", type=Path, default=None, help="Recommended: one normal WMPL trajectory pickle. The converter clones its Trajectory/Orbit classes, then replaces the data.")
    parser.add_argument("--output", "-o", type=Path, help="Pipeline mode: output pickle path for one event, or output directory for many events")
    parser.add_argument("--output-mode", choices=["pipeline", "collection"], default="pipeline", help="pipeline writes real Trajectory root pickle(s); collection writes the old {'events': [...]} wrapper")
    parser.add_argument("--object-style", choices=["dict", "namespace"], default="dict", help="Only used in collection mode")
    parser.add_argument("--numpy-arrays", action="store_true", default=True, help="Convert numeric lists to numpy arrays before pickling")
    parser.add_argument("--no-numpy-arrays", dest="numpy_arrays", action="store_false", help="Keep numeric arrays as Python lists")
    parser.add_argument("--include-raw", action="store_true", help="Include raw source file text inside the pickle")
    parser.add_argument("--json-copy", action="store_true", help="Optional JSON sidecar output for quick inspection")
    args = parser.parse_args(argv)

    template = load_template(args.template_json)
    template_traj = None
    if args.template_pickle is not None:
        template_traj = load_pickle_py2_compatible(args.template_pickle)
        print(f"Loaded WMPL template pickle: {args.template_pickle}")

    event_groups = collect_events(args.input)
    if not event_groups:
        raise SystemExit(f"No EN .res/.inp/.lc/.rlc files found under {args.input}")

    trajectories: List[Dict[str, Any]] = []
    skipped: List[Tuple[str, str]] = []
    for source_name, files in event_groups:
        missing_core = [ext for ext in [".res", ".inp"] if ext not in files]
        if missing_core:
            skipped.append((source_name, f"missing {', '.join(missing_core)}"))
        try:
            trajectories.append(build_trajectory_like(files, template, source_name, include_raw=args.include_raw))
        except Exception as exc:
            skipped.append((source_name, repr(exc)))

    if not trajectories:
        raise SystemExit("No trajectories could be built. " + repr(skipped))

    # Show the station IDs that will be written. This is useful for catching
    # collapsed IDs such as cz_102 before the pickle is used downstream.
    for traj_dict in trajectories:
        station_ids_preview = [str(o.get("station_id")) for o in traj_dict.get("observations", [])]
        zc_preview = traj_dict.get("orbit", {}).get("zc")
        zg_preview = traj_dict.get("orbit", {}).get("zg")
        v0z_preview = traj_dict.get("v0z")
        print(f"Station IDs for {traj_dict.get('file_name', traj_dict.get('traj_id'))}: {station_ids_preview}")
        print(f"Parsed EN zc={zc_preview} rad ({np.degrees(zc_preview) if zc_preview is not None else None} deg), zg={zg_preview} rad ({np.degrees(zg_preview) if zg_preview is not None else None} deg), v0z={v0z_preview} m/s")
        for o in traj_dict.get("observations", []):
            n_time = len(o.get("time_data", []))
            n_mag = len(o.get("absolute_magnitudes", []))
            if n_time != n_mag:
                print(f"WARNING length mismatch before writing: {o.get('station_id')} time_data={n_time}, absolute_magnitudes={n_mag}")
            mapping = o.get("photometry_mapping", {})
            if mapping:
                print(f"  {o.get('station_id')} photometry mapping: LC={mapping.get('lc_mapping_axis')} ({mapping.get('lc_valid_points_on_inp_grid')} pts), RLC={mapping.get('rlc_mapping_axis')} ({mapping.get('rlc_valid_points_on_inp_grid')} pts), +20 placeholders={mapping.get('filled_missing_mag_count')}")

    if args.output_mode == "pipeline":
        written = write_pipeline_pickles(trajectories, args, template_traj)
        for out_path in written:
            print(f"Wrote pipeline-compatible WMPL Trajectory root pickle: {out_path}")

        if args.json_copy:
            # One JSON next to each pickle for inspection.
            for traj_dict, out_path in zip(trajectories, written):
                json_path = out_path.with_suffix(".json")
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(json_safe(traj_dict), f, indent=2)
                print(f"Wrote JSON sidecar: {json_path}")

    else:
        # Old collection mode, kept only in case you want a single database pickle.
        if args.output is None:
            args.output = args.input / f"{args.input.stem}_trajectory_collection.pickle" if args.input.is_dir() else args.input.with_suffix(".trajectory_collection.pickle")
        payload: Dict[str, Any] = {
            "format": "trajectory_like_en_fireball_collection_v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "n_events": len(trajectories),
            "events": trajectories,
            "by_code": {str(t.get("file_name")): t for t in trajectories},
            "skipped": skipped,
        }
        payload_for_pickle: Any = list_to_numpy(payload) if args.numpy_arrays else payload
        if args.object_style == "namespace":
            payload_for_pickle = to_namespace(payload_for_pickle)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "wb") as f:
            pickle.dump(payload_for_pickle, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Wrote collection wrapper pickle: {args.output}")

    if skipped:
        print("Warnings/skipped/partial events:")
        for name, reason in skipped:
            print(f"  - {name}: {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
