#!/usr/bin/env python3
"""
Convert ESABASE/LISKIN orbit CSV files into the seven-column MEM 3 orbit format:

    Julian_Date  X_km  Y_km  Z_km  VX_km_s  VY_km_s  VZ_km_s

Two input styles are supported automatically:

1) Cartesian state-vector CSV
   Required columns:
       orbital_point, datetime,
       x_km, y_km, z_km, vx_km_s, vy_km_s, vz_km_s

2) Classical orbital-elements CSV (e.g. LISDM_orbits.csv)
   Required columns:
       orbital_point, semi_major_axis_km, eccentricity,
       inclination_deg, raan_deg,
       argument_of_periapsis_deg, true_anomaly_deg

   The datetime for each orbital point is read from a companion CSV containing
   at least:
       orbital_point, datetime

   By default the script searches the same directory for
   LISKIN_orbital_points*.csv.  A specific file can be supplied with
   --points_file.

Examples:
    python orbit_IMEM2MEM_updated.py --csv_file LISKIN_states.csv

    python orbit_IMEM2MEM_updated.py --csv_file LISDM_orbits.csv

    python orbit_IMEM2MEM_updated.py --csv_file LISDM_orbits.csv \
        --points_file LISKIN_orbital_points.csv --output orbit.txt
"""

from __future__ import annotations

import argparse
import csv
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


# Earth gravitational parameter in km^3/s^2.
# This is used only when converting classical orbital elements to Cartesian
# state vectors.  It can be overridden from the command line with --mu.
MU_EARTH_KM3_S2 = 398600.4418


CARTESIAN_REQUIRED = {
    "orbital_point",
    "datetime",
    "x_km",
    "y_km",
    "z_km",
    "vx_km_s",
    "vy_km_s",
    "vz_km_s",
}

ELEMENT_REQUIRED = {
    "orbital_point",
    "semi_major_axis_km",
    "eccentricity",
    "inclination_deg",
    "raan_deg",
    "argument_of_periapsis_deg",
    "true_anomaly_deg",
}

TIME_REQUIRED = {"orbital_point", "datetime"}


def datetime_to_julian_date(dt: datetime) -> float:
    """Convert a datetime to Julian Date, treating naive input as UTC."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)

    year = dt.year
    month = dt.month
    day_fraction = (
        dt.day
        + dt.hour / 24.0
        + dt.minute / 1440.0
        + (dt.second + dt.microsecond / 1_000_000.0) / 86400.0
    )

    if month <= 2:
        year -= 1
        month += 12

    a = math.floor(year / 100)
    b = 2 - a + math.floor(a / 4)

    return (
        math.floor(365.25 * (year + 4716))
        + math.floor(30.6001 * (month + 1))
        + day_fraction
        + b
        - 1524.5
    )


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read a CSV file and strip whitespace from column names."""
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")

        # Normalize only surrounding whitespace; keep names otherwise intact.
        reader.fieldnames = [name.strip() for name in reader.fieldnames]
        rows = []
        for row in reader:
            clean = {
                (key.strip() if key is not None else key):
                (value.strip() if isinstance(value, str) else value)
                for key, value in row.items()
            }
            rows.append(clean)

    if not rows:
        raise ValueError(f"CSV contains no data rows: {path}")

    return rows


def columns_of(rows: list[dict[str, str]]) -> set[str]:
    return {key for key in rows[0].keys() if key is not None}


def sort_and_validate_points(rows: list[dict[str, str]], label: str) -> list[dict[str, str]]:
    """Sort by orbital_point and ensure point IDs are valid integers."""
    try:
        rows.sort(key=lambda row: int(row["orbital_point"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{label}: invalid or missing orbital_point column") from exc

    points = [int(row["orbital_point"]) for row in rows]
    if len(points) != len(set(points)):
        duplicates = sorted({p for p in points if points.count(p) > 1})
        raise ValueError(
            f"{label}: duplicate orbital_point values found: {duplicates[:10]}. "
            "If the file contains multiple orbital arcs, select one arc first."
        )

    return rows


def parse_datetime(text: str) -> datetime:
    """Parse the ISO datetime strings produced in the LISKIN orbital-points CSV."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def orbital_elements_to_state(
    semi_major_axis_km: float,
    eccentricity: float,
    inclination_deg: float,
    raan_deg: float,
    argument_of_periapsis_deg: float,
    true_anomaly_deg: float,
    mu_km3_s2: float = MU_EARTH_KM3_S2,
) -> tuple[float, float, float, float, float, float]:
    """
    Convert classical Keplerian elements to an inertial Cartesian state vector.

    The standard perifocal -> inertial rotation R3(RAAN) R1(i) R3(arg_periapsis)
    is used.  Distances are km and velocities are km/s.
    """
    a = semi_major_axis_km
    e = eccentricity

    if a <= 0.0:
        raise ValueError(f"semi_major_axis_km must be > 0, got {a}")
    if not (0.0 <= e < 1.0):
        raise ValueError(
            f"This converter currently expects an elliptical orbit with 0 <= e < 1; got {e}"
        )
    if mu_km3_s2 <= 0.0:
        raise ValueError(f"mu must be > 0, got {mu_km3_s2}")

    inc = math.radians(inclination_deg)
    raan = math.radians(raan_deg)
    argp = math.radians(argument_of_periapsis_deg)
    nu = math.radians(true_anomaly_deg)

    p = a * (1.0 - e * e)
    denominator = 1.0 + e * math.cos(nu)
    if abs(denominator) < 1e-15:
        raise ValueError("Invalid orbital elements: radius denominator is zero")

    rmag = p / denominator
    speed_factor = math.sqrt(mu_km3_s2 / p)

    # Position and velocity in the perifocal (PQW) frame.
    rx_p = rmag * math.cos(nu)
    ry_p = rmag * math.sin(nu)
    vx_p = -speed_factor * math.sin(nu)
    vy_p = speed_factor * (e + math.cos(nu))

    cO = math.cos(raan)
    sO = math.sin(raan)
    ci = math.cos(inc)
    si = math.sin(inc)
    cw = math.cos(argp)
    sw = math.sin(argp)

    # First two columns of R3(Omega) R1(i) R3(omega).
    r11 = cO * cw - sO * sw * ci
    r12 = -cO * sw - sO * cw * ci
    r21 = sO * cw + cO * sw * ci
    r22 = -sO * sw + cO * cw * ci
    r31 = sw * si
    r32 = cw * si

    x = r11 * rx_p + r12 * ry_p
    y = r21 * rx_p + r22 * ry_p
    z = r31 * rx_p + r32 * ry_p

    vx = r11 * vx_p + r12 * vy_p
    vy = r21 * vx_p + r22 * vy_p
    vz = r31 * vx_p + r32 * vy_p

    return x, y, z, vx, vy, vz


def csv_has_columns(path: Path, required: set[str]) -> bool:
    """Check only the header of a CSV file."""
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
    except (OSError, UnicodeError):
        return False

    if not header:
        return False
    return required.issubset({name.strip() for name in header})


def find_companion_points_file(source_csv: Path) -> Path:
    """
    Find a companion CSV in the source directory that contains orbital_point
    and datetime.  LISKIN_orbital_points*.csv files are preferred.
    """
    directory = source_csv.parent

    preferred = sorted(
        path for path in directory.glob("LISKIN_orbital_points*.csv")
        if path.resolve() != source_csv.resolve() and csv_has_columns(path, TIME_REQUIRED)
    )

    if len(preferred) == 1:
        return preferred[0]
    if len(preferred) > 1:
        exact = [p for p in preferred if p.name == "LISKIN_orbital_points.csv"]
        if len(exact) == 1:
            return exact[0]
        raise ValueError(
            "More than one LISKIN orbital-points CSV could provide the times: "
            + ", ".join(p.name for p in preferred)
            + ". Use --points_file to choose one explicitly."
        )

    # Fallback: inspect any other CSV in the same folder.
    candidates = sorted(
        path for path in directory.glob("*.csv")
        if path.resolve() != source_csv.resolve() and csv_has_columns(path, TIME_REQUIRED)
    )

    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            "The orbital-elements CSV has no datetime column and no companion "
            "CSV containing orbital_point + datetime was found in the same folder. "
            "Provide it with --points_file."
        )

    raise ValueError(
        "Several CSV files could provide the orbital-point times: "
        + ", ".join(p.name for p in candidates)
        + ". Use --points_file to choose one explicitly."
    )


def make_time_lookup(points_file: Path) -> dict[int, datetime]:
    """Read orbital_point -> datetime from the companion orbital-points CSV."""
    rows = read_csv_rows(points_file)
    cols = columns_of(rows)
    missing = TIME_REQUIRED.difference(cols)
    if missing:
        raise ValueError(
            f"Companion points file {points_file.name} is missing columns: {sorted(missing)}"
        )

    lookup: dict[int, datetime] = {}
    for row in rows:
        point = int(row["orbital_point"])
        if point in lookup:
            raise ValueError(
                f"Companion points file {points_file.name} contains orbital_point "
                f"{point} more than once. If it contains multiple arcs, provide a "
                "single-arc orbital-points file."
            )
        lookup[point] = parse_datetime(row["datetime"])

    return lookup


def format_mem_lines(
    states: Iterable[tuple[datetime, float, float, float, float, float, float]],
    source_description: str,
) -> list[str]:
    states = list(states)
    header = [
        f"# Input file for MEM 3 generated from {source_description}",
        f"# contains {len(states)} lines of orbital state vectors",
        "# Earth-centered inertial equatorial Cartesian coordinates",
        "#",
        "# do not remove this 6-line header",
        "#",
    ]

    lines = header.copy()
    for dt, x, y, z, vx, vy, vz in states:
        jd = datetime_to_julian_date(dt)
        lines.append(
            f"{jd:.8f} "
            f"{x:.2f} {y:.2f} {z:.2f} "
            f"{vx:.6f} {vy:.6f} {vz:.6f}"
        )
    return lines


def convert_cartesian(rows: list[dict[str, str]]) -> list[tuple[datetime, float, float, float, float, float, float]]:
    rows = sort_and_validate_points(rows, "Cartesian CSV")
    states = []

    for row in rows:
        dt = parse_datetime(row["datetime"])
        states.append(
            (
                dt,
                float(row["x_km"]),
                float(row["y_km"]),
                float(row["z_km"]),
                float(row["vx_km_s"]),
                float(row["vy_km_s"]),
                float(row["vz_km_s"]),
            )
        )

    return states


def convert_elements(
    rows: list[dict[str, str]],
    points_file: Path,
    mu_km3_s2: float,
) -> list[tuple[datetime, float, float, float, float, float, float]]:
    rows = sort_and_validate_points(rows, "Orbital-elements CSV")
    times = make_time_lookup(points_file)
    states = []

    missing_times = [int(row["orbital_point"]) for row in rows if int(row["orbital_point"]) not in times]
    if missing_times:
        raise ValueError(
            f"No datetime was found for {len(missing_times)} orbital point(s) in "
            f"{points_file.name}. First missing points: {missing_times[:10]}"
        )

    for row in rows:
        point = int(row["orbital_point"])
        x, y, z, vx, vy, vz = orbital_elements_to_state(
            semi_major_axis_km=float(row["semi_major_axis_km"]),
            eccentricity=float(row["eccentricity"]),
            inclination_deg=float(row["inclination_deg"]),
            raan_deg=float(row["raan_deg"]),
            argument_of_periapsis_deg=float(row["argument_of_periapsis_deg"]),
            true_anomaly_deg=float(row["true_anomaly_deg"]),
            mu_km3_s2=mu_km3_s2,
        )
        states.append((times[point], x, y, z, vx, vy, vz))

    return states


def convert(
    source_csv: Path,
    output_file: Path,
    points_file: Path | None = None,
    mu_km3_s2: float = MU_EARTH_KM3_S2,
) -> tuple[int, str, Path | None]:
    """Convert either supported input style and write a MEM-format text file."""
    rows = read_csv_rows(source_csv)
    cols = columns_of(rows)

    used_points_file: Path | None = None

    if CARTESIAN_REQUIRED.issubset(cols):
        mode = "Cartesian state-vector CSV"
        states = convert_cartesian(rows)
        description = "ESABASE2/LISKIN Cartesian states"

    elif ELEMENT_REQUIRED.issubset(cols):
        mode = "classical orbital-elements CSV"
        used_points_file = points_file or find_companion_points_file(source_csv)
        states = convert_elements(rows, used_points_file, mu_km3_s2)
        description = "ESABASE2/LISKIN orbital elements"

    else:
        cart_missing = sorted(CARTESIAN_REQUIRED.difference(cols))
        elem_missing = sorted(ELEMENT_REQUIRED.difference(cols))
        raise ValueError(
            f"Unrecognized CSV layout in {source_csv.name}.\n"
            f"Missing for Cartesian mode: {cart_missing}\n"
            f"Missing for orbital-elements mode: {elem_missing}"
        )

    lines = format_mem_lines(states, description)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(lines) + "\n", encoding="ascii")

    return len(states), mode, used_points_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert LISKIN Cartesian states or orbital elements to MEM 3 orbit input."
    )
    parser.add_argument(
        "--csv_file",
        type=Path,
        default=Path(
            r"C:\Users\maxiv\Documents\UWO\Papers\3)Sporadics\ISS-risk\debris-result-ISSspor10-3-10g_extracted\LISDM_orbits.csv"
        ),
        help=(
            "Input CSV. May contain Cartesian state vectors or the classical "
            "orbital elements from LISDM_orbits.csv."
        ),
    )
    parser.add_argument(
        "--points_file",
        type=Path,
        default=None,
        help=(
            "Companion CSV containing orbital_point and datetime. Needed only "
            "for orbital-elements input. If omitted, the script searches the "
            "same folder for LISKIN_orbital_points*.csv."
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output MEM TXT file. Default: <input stem>_MEM.txt",
    )
    parser.add_argument(
        "--mu",
        type=float,
        default=MU_EARTH_KM3_S2,
        help=(
            "Central-body gravitational parameter in km^3/s^2 used for orbital "
            f"elements -> Cartesian conversion (default Earth: {MU_EARTH_KM3_S2})."
        ),
    )

    args = parser.parse_args()

    output = args.output or args.csv_file.with_name(args.csv_file.stem + "_MEM.txt")

    count, mode, used_points_file = convert(
        source_csv=args.csv_file,
        output_file=output,
        points_file=args.points_file,
        mu_km3_s2=args.mu,
    )

    print(f"Input mode: {mode}")
    if used_points_file is not None:
        print(f"Datetime source: {used_points_file}")
    print(f"Wrote {count} orbital state vectors to: {output}")


if __name__ == "__main__":
    main()
