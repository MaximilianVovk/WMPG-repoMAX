#!/usr/bin/env python3
"""
Convert an ESABASE2 LISKIN_orbital_points.csv file into the seven-column
MEM orbit-input format:

    Julian_Date  X_km  Y_km  Z_km  VX_km_s  VY_km_s  VZ_km_s

The output retains the required six-line comment header.

Example:
    python convert_liskin_csv_to_mem.py LISKIN_orbital_points.csv
    python convert_liskin_csv_to_mem.py LISKIN_orbital_points.csv --output orbit.txt
"""

from __future__ import annotations

import argparse
import csv
import math
from datetime import datetime, timezone
from pathlib import Path


def datetime_to_julian_date(dt: datetime) -> float:
    """Convert a UTC datetime to Julian Date."""
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


def read_liskin_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise ValueError("The CSV contains no orbital points.")

    required = {
        "orbital_point",
        "datetime",
        "x_km",
        "y_km",
        "z_km",
        "vx_km_s",
        "vy_km_s",
        "vz_km_s",
    }
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    rows.sort(key=lambda row: int(row["orbital_point"]))

    actual = [int(row["orbital_point"]) for row in rows]
    expected = list(range(1, len(rows) + 1))
    if actual != expected:
        raise ValueError(
            "Orbital points must be consecutive from 1 to N. "
            f"Found first/last values {actual[0]} and {actual[-1]}."
        )

    return rows


def convert(source_csv: Path, output_file: Path) -> int:
    rows = read_liskin_rows(source_csv)

    header = [
        "# Input file for MEM 3 generated from ESABASE2 LISKIN",
        f"# contains {len(rows)} lines of orbital state vectors",
        "# coordinates are those in the LISKIN inertial equatorial frame",
        "#",
        "# do not remove this 6-line header",
        "#",
    ]

    lines = header.copy()

    for row in rows:
        dt = datetime.fromisoformat(row["datetime"].replace("Z", "+00:00"))
        jd = datetime_to_julian_date(dt)

        values = [
            float(row["x_km"]),
            float(row["y_km"]),
            float(row["z_km"]),
            float(row["vx_km_s"]),
            float(row["vy_km_s"]),
            float(row["vz_km_s"]),
        ]

        lines.append(
            f"{jd:.8f} "
            f"{values[0]:.2f} {values[1]:.2f} {values[2]:.2f} "
            f"{values[3]:.6f} {values[4]:.6f} {values[5]:.6f}"
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(lines) + "\n", encoding="ascii")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "csv_file",
        type=Path,
        help="LISKIN_orbital_points.csv produced by the ESABASE reader",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output TXT path; defaults to <CSV stem>_MEM.txt",
    )
    args = parser.parse_args()

    output = args.output or args.csv_file.with_name(
        args.csv_file.stem + "_MEM.txt"
    )

    count = convert(args.csv_file, output)
    print(f"Wrote {count} orbital state vectors to: {output}")


if __name__ == "__main__":
    main()
