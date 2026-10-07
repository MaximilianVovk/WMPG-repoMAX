#!/usr/bin/env python3
"""
Extract tabular data from ESABASE2 ASCII listing files:

    *.FAI_D   failures versus ballistic limit
    *.CRT_D   hits/craters versus crater diameter
    *.LISKIN  kinematic/orientation listing
    *.LISORB  orbital/ephemeris listing
    *.LISDM   debris/meteoroid flux and damage listing

These files are plain text. They are different from the binary Open
CASCADE/OCAF *.output document and should not be passed to the binary
.output reader.

Only the Python standard library is required.

Examples
--------
# Discover and process every related listing set in a directory:
python read_esabase_listing_files.py /path/to/results

# Start from one LISDM/LISKIN/etc. file and discover its siblings:
python read_esabase_listing_files.py result_LISDM.LISDM

# The previous explicit-file interface remains supported:
python read_esabase_listing_files.py \
    --fai-d result_FAI_D.FAI_D \
    --crt-d result_CRT_D.CRT_D \
    --liskin result_LISKIN.LISKIN \
    --lisorb result_LISORB.LISORB \
    --lisdm result_LISDM.LISDM \
    --out-dir extracted
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable


NUMBER_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?"


def as_float(value: str) -> float:
    return float(value.replace("D", "E").replace("d", "e"))


def numbers(text: str) -> list[float]:
    return [as_float(x) for x in re.findall(NUMBER_RE, text)]


def read_text(path: Path) -> str:
    # These samples are ASCII, but latin-1 safely preserves legacy listings.
    return path.read_text(encoding="latin-1")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_threshold_matrix(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Parse FAI_D or CRT_D.

    Returns:
      wide_rows: one row per spacecraft element
      long_rows: one row per element and threshold
    """
    lines = read_text(path).splitlines()
    header_index = next(
        i for i, line in enumerate(lines) if line.strip().startswith("IELMT")
    )
    header = lines[header_index].split()
    thresholds = [as_float(x) for x in header[3:]]

    wide_rows: list[dict[str, Any]] = []
    long_rows: list[dict[str, Any]] = []

    for line in lines[header_index + 1 :]:
        parts = line.split()
        if len(parts) < 2 + len(thresholds) or not parts[0].isdigit():
            continue

        element_id = int(parts[0])
        area_m2 = as_float(parts[1])
        values = [as_float(x) for x in parts[2 : 2 + len(thresholds)]]

        wide: dict[str, Any] = {
            "element_id": element_id,
            "element_area_m2": area_m2,
        }
        for threshold, value in zip(thresholds, values):
            wide[f"value_at_{threshold:.8g}_cm"] = value
            long_rows.append(
                {
                    "element_id": element_id,
                    "element_area_m2": area_m2,
                    "threshold_cm": threshold,
                    "value": value,
                }
            )
        wide_rows.append(wide)

    return wide_rows, long_rows


def parse_liskin(path: Path) -> list[dict[str, Any]]:
    text = read_text(path)
    markers = list(
        re.finditer(
            r"(?m)^1\s+Orbital arc:\s*(\d+)\s+Orbital point:\s*(\d+)\s*$",
            text,
        )
    )

    rows: list[dict[str, Any]] = []

    for i, marker in enumerate(markers):
        end = markers[i + 1].start() if i + 1 < len(markers) else len(text)
        block = text[marker.start() : end]

        row: dict[str, Any] = {
            "orbital_arc": int(marker.group(1)),
            "orbital_point": int(marker.group(2)),
        }

        date_match = re.search(
            r"Date:\s*(\d{4})/(\d{2})/(\d{2})\s+"
            r"(\d{2})/(\d{2})/(\d{2})\s+"
            r"Elapsed time \(min\):\s*(" + NUMBER_RE + r")",
            block,
        )
        if date_match:
            year, month, day, hour, minute, second, elapsed = date_match.groups()
            row.update(
                {
                    "year": int(year),
                    "month": int(month),
                    "day": int(day),
                    "hour": int(hour),
                    "minute": int(minute),
                    "second": int(second),
                    "datetime": (
                        f"{int(year):04d}-{int(month):02d}-{int(day):02d}T"
                        f"{int(hour):02d}:{int(minute):02d}:{int(second):02d}"
                    ),
                    "elapsed_min": as_float(elapsed),
                }
            )

        classical_match = re.search(
            r"Classical orbital elements in inertial equatorial frame\s*\r?\n"
            r"\s*([^\r\n]+)",
            block,
        )
        if classical_match:
            values = numbers(classical_match.group(1))
            names = [
                "semi_major_axis_km",
                "eccentricity",
                "inclination_deg",
                "raan_deg",
                "argument_of_periapsis_deg",
                "true_anomaly_deg",
            ]
            row.update(dict(zip(names, values)))

        state_match = re.search(
            r"Spacecraft state \(km,km/s\) in inertial equatorial frame\s*\r?\n"
            r"\s*([^\r\n]+)",
            block,
        )
        if state_match:
            names = [
                "x_km",
                "y_km",
                "z_km",
                "vx_km_s",
                "vy_km_s",
                "vz_km_s",
            ]
            row.update(dict(zip(names, numbers(state_match.group(1)))))

        central_match = re.search(
            r"Central Body \(in system frame\)\s*\r?\n\s*([^\r\n]+)",
            block,
        )
        if central_match:
            row.update(
                dict(
                    zip(
                        ["central_body_x", "central_body_y", "central_body_z"],
                        numbers(central_match.group(1)),
                    )
                )
            )

        earth_sun_match = re.search(
            r"Earth \(in system frame\)\s+Sun \(in system frame\)\s*\r?\n"
            r"\s*([^\r\n]+)",
            block,
        )
        if earth_sun_match:
            row.update(
                dict(
                    zip(
                        [
                            "earth_x",
                            "earth_y",
                            "earth_z",
                            "sun_x",
                            "sun_y",
                            "sun_z",
                        ],
                        numbers(earth_sun_match.group(1)),
                    )
                )
            )

        for label, prefix in (("First", "first"), ("Second", "second")):
            pointing_match = re.search(
                rf"{label} pointing.*?\r?\n\s*\r?\n"
                r"\s*pointing vector\s+pointing direction\s*\r?\n"
                r"\s*BODYCENTR\s*:\s*([^\r\n]+)\r?\n"
                r"\s*SYSTEM\s*:\s*([^\r\n]+)",
                block,
                re.DOTALL,
            )
            if not pointing_match:
                continue

            body_values = numbers(pointing_match.group(1))
            system_values = numbers(pointing_match.group(2))

            body_names = [
                f"{prefix}_body_vector_x",
                f"{prefix}_body_vector_y",
                f"{prefix}_body_vector_z",
                f"{prefix}_body_direction_x",
                f"{prefix}_body_direction_y",
                f"{prefix}_body_direction_z",
            ]
            system_names = [
                f"{prefix}_system_vector_x",
                f"{prefix}_system_vector_y",
                f"{prefix}_system_vector_z",
                f"{prefix}_system_direction_x",
                f"{prefix}_system_direction_y",
                f"{prefix}_system_direction_z",
            ]
            row.update(dict(zip(body_names, body_values)))
            row.update(dict(zip(system_names, system_values)))

        rows.append(row)

    return rows


def parse_lisorb(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    text = read_text(path)

    metadata_patterns = {
        "perigee_height_km": r"HEIGHT OF PERIGEE \(KM\)\s*=\s*(" + NUMBER_RE + r")",
        "apogee_height_km": r"HEIGHT OF APOGEE \(KM\)\s*=\s*(" + NUMBER_RE + r")",
        "semi_latus_rectum_km": r"SEMI LATUS RECTUM \(KM\)\s*=\s*(" + NUMBER_RE + r")",
        "semi_major_axis_km": r"SEMI MAJOR AXIS \(KM\)\s*=\s*(" + NUMBER_RE + r")",
        "eccentricity": r"ECCENTRICITY\s*=\s*(" + NUMBER_RE + r")",
        "inclination_deg": r"INCLINATION \(DEG\)\s*=\s*(" + NUMBER_RE + r")",
        "ascending_node_deg": r"ASCENDING NODE \(DEG\)\s*=\s*(" + NUMBER_RE + r")",
        "argument_of_perigee_deg": r"ARG\. OF PERIGEE \(DEG\)\s*=\s*(" + NUMBER_RE + r")",
        "true_anomaly_deg": r"TRUE ANOMALY \(DEG\)\s*=\s*(" + NUMBER_RE + r")",
        "mean_motion_rad_day": r"MEAN MOTION \(RAD/DAY\)\s*=\s*(" + NUMBER_RE + r")",
        "ephemeris_cover_days": r"EPHEMERIDES COVER \(DAYS\)\s*:\s*(" + NUMBER_RE + r")",
        "integration_step_deg": r"INTEGRATION STEP \(DEG\)\s*:\s*(" + NUMBER_RE + r")",
    }

    metadata: dict[str, Any] = {}
    for key, pattern in metadata_patterns.items():
        match = re.search(pattern, text)
        if match:
            metadata[key] = as_float(match.group(1))

    central = re.search(r"CENTRAL BODY\s*:\s*([A-Za-z0-9_-]+)", text)
    if central:
        metadata["central_body"] = central.group(1)

    ephemeris_rows: list[dict[str, Any]] = []
    row_pattern = re.compile(
        r"\*(\d{4})/\s*(\d+)/\s*(\d+)/\s*(\d+)/\s*(\d+)/\s*(\d+)\s*"
        r"\*\s*(" + NUMBER_RE + r")\s+(" + NUMBER_RE + r")\s+(" + NUMBER_RE + r")\s*"
        r"\*\s*(" + NUMBER_RE + r")\s+(" + NUMBER_RE + r")\s+(" + NUMBER_RE + r")\s*"
        r"\*\s*(\d+)\s*\*"
    )

    for match in row_pattern.finditer(text):
        (
            year,
            month,
            day,
            hour,
            minute,
            second,
            longitude,
            latitude,
            height,
            x,
            y,
            z,
            record_number,
        ) = match.groups()

        ephemeris_rows.append(
            {
                "year": int(year),
                "month": int(month),
                "day": int(day),
                "hour": int(hour),
                "minute": int(minute),
                "second": int(second),
                "datetime": (
                    f"{int(year):04d}-{int(month):02d}-{int(day):02d}T"
                    f"{int(hour):02d}:{int(minute):02d}:{int(second):02d}"
                ),
                "longitude_deg": as_float(longitude),
                "latitude_deg": as_float(latitude),
                "height_km": as_float(height),
                "x_km": as_float(x),
                "y_km": as_float(y),
                "z_km": as_float(z),
                "record_number": int(record_number),
            }
        )

    return metadata, ephemeris_rows


ELEMENT_COLUMNS = [
    "ks_factor",
    "impact_angle_deg",
    "impact_velocity_km_s",
    "crater_area",
    "impact_flux_m2_yr",
    "impact_fluence_m2",
    "number_impacts",
    "failure_flux_m2_yr",
    "failure_fluence_m2",
    "total_failures",
]

OBJECT_COLUMNS = [
    "ks_factor",
    "crater_area",
    "impact_flux_m2_yr",
    "impact_fluence_m2",
    "total_impacts",
    "failure_flux_m2_yr",
    "failure_fluence_m2",
    "total_failures",
]


def element_rows_from_block(
    block: str, level: str, orbital_point: int | None = None
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for line in block.splitlines():
        parts = line.split()
        if len(parts) != 11 or not parts[0].isdigit():
            continue
        try:
            values = [as_float(x) for x in parts[1:]]
        except ValueError:
            continue
        row: dict[str, Any] = {
            "level": level,
            "orbital_point": orbital_point,
            "element_id": int(parts[0]),
        }
        row.update(dict(zip(ELEMENT_COLUMNS, values)))
        output.append(row)
    return output


def object_rows_from_block(
    block: str, level: str, orbital_point: int | None = None
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for line in block.splitlines():
        parts = line.split()
        if not parts or parts[0] not in {"BOX", "SPACECRAFT"}:
            continue
        try:
            values = [as_float(x) for x in parts[1:]]
        except ValueError:
            continue
        if len(values) < 8:
            continue

        row: dict[str, Any] = {
            "level": level,
            "orbital_point": orbital_point,
            "object_name": parts[0],
        }
        row.update(dict(zip(OBJECT_COLUMNS, values[:8])))
        if len(values) >= 9:
            row["surface_area_m2"] = values[8]
        if len(values) >= 10:
            row["probability_no_failure"] = values[9]
        output.append(row)
    return output


def parse_distribution(
    segment: str, heading: str, threshold_label: str, value_label: str
) -> list[dict[str, Any]]:
    match = re.search(
        re.escape(heading)
        + r".*?"
        + re.escape(threshold_label)
        + r"\s+([^\r\n]+).*?"
        + r"OBJECT NAME\s+Area[^\r\n]*\r?\n"
        + r"\s*([A-Za-z0-9_-]+)\s+("
        + NUMBER_RE
        + r")\s+([^\r\n]+)",
        segment,
        re.DOTALL,
    )
    if not match:
        return []

    thresholds = numbers(match.group(1))
    object_name = match.group(2)
    area = as_float(match.group(3))
    values = numbers(match.group(4))[: len(thresholds)]

    return [
        {
            "distribution": value_label,
            "object_name": object_name,
            "area_m2": area,
            "threshold_cm": threshold,
            "value": value,
        }
        for threshold, value in zip(thresholds, values)
    ]


def parse_lisdm(
    path: Path,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    text = read_text(path)
    markers = list(
        re.finditer(r"ORBITAL POINT NUMBER\s*:\s*(\d+)", text)
    )

    orbit_rows: list[dict[str, Any]] = []
    element_rows: list[dict[str, Any]] = []
    object_rows: list[dict[str, Any]] = []

    for i, marker in enumerate(markers):
        end = markers[i + 1].start() if i + 1 < len(markers) else len(text)
        block = text[marker.start() : end]
        point = int(marker.group(1))

        if "CLASSICAL ORBITAL ELEMENTS" in block:
            orbit: dict[str, Any] = {"orbital_point": point}

            first = re.search(
                r"A\s*:\s*(" + NUMBER_RE + r")\s*\[km\]\s*"
                r"E\s*:\s*(" + NUMBER_RE + r")\[\]\s*"
                r"Inclination\s*:\s*(" + NUMBER_RE + r")\[deg\]",
                block,
            )
            second = re.search(
                r"RAAN:\s*(" + NUMBER_RE + r")\[deg\]\s*"
                r"ArgOfPer:\s*(" + NUMBER_RE + r")\[deg\]\s*"
                r"True Anomaly:\s*(" + NUMBER_RE + r")\[deg\]\s*"
                r"True Latitude:\s*(" + NUMBER_RE + r")\[deg\]",
                block,
            )

            if first:
                orbit.update(
                    {
                        "semi_major_axis_km": as_float(first.group(1)),
                        "eccentricity": as_float(first.group(2)),
                        "inclination_deg": as_float(first.group(3)),
                    }
                )
            if second:
                orbit.update(
                    {
                        "raan_deg": as_float(second.group(1)),
                        "argument_of_periapsis_deg": as_float(second.group(2)),
                        "true_anomaly_deg": as_float(second.group(3)),
                        "true_latitude_deg": as_float(second.group(4)),
                    }
                )

            orbit_rows.append(orbit)
            element_rows.extend(
                element_rows_from_block(block, "orbital_point", point)
            )

        elif "OBJECT NAME" in block:
            # The second block for point 110 also contains arc/mission summaries.
            point_only = block.split("ORBITAL ARC LEVEL", 1)[0]
            object_rows.extend(
                object_rows_from_block(point_only, "orbital_point", point)
            )

    arc_start = text.find("ORBITAL ARC LEVEL")
    mission_start = text.find("MISSION LEVEL")
    crater_start = text.find("CRATERS VS CRATER DIAMETER")

    if arc_start >= 0:
        arc_end = mission_start if mission_start >= 0 else len(text)
        arc_block = text[arc_start:arc_end]
        element_rows.extend(element_rows_from_block(arc_block, "orbital_arc"))
        object_rows.extend(object_rows_from_block(arc_block, "orbital_arc"))

    if mission_start >= 0:
        mission_end = crater_start if crater_start >= 0 else len(text)
        mission_block = text[mission_start:mission_end]
        element_rows.extend(element_rows_from_block(mission_block, "mission"))
        object_rows.extend(object_rows_from_block(mission_block, "mission"))

    distributions: list[dict[str, Any]] = []
    if crater_start >= 0:
        ending = text[crater_start:]
        distributions.extend(
            parse_distribution(
                ending,
                "CRATERS VS CRATER DIAMETER",
                "Crater Diameters [cm]",
                "number_of_craters",
            )
        )
        distributions.extend(
            parse_distribution(
                ending,
                "FAILURES VS BALLISTIC LIMIT",
                "Ballistic Limits [cm]",
                "number_of_failures",
            )
        )

    return orbit_rows, element_rows, object_rows, distributions



LISTING_SUFFIXES = {
    "FAI_D": "_FAI_D.FAI_D",
    "CRT_D": "_CRT_D.CRT_D",
    "LISKIN": "_LISKIN.LISKIN",
    "LISORB": "_LISORB.LISORB",
    "LISDM": "_LISDM.LISDM",
}


def identify_listing_file(path: Path) -> tuple[str, str] | None:
    """Return ``(case_name, file_type)`` for a recognized listing filename."""
    lower_name = path.name.lower()
    for file_type, suffix in LISTING_SUFFIXES.items():
        if lower_name.endswith(suffix.lower()):
            case_name = path.name[: -len(suffix)]
            if case_name:
                return case_name, file_type
    return None


def discover_listing_groups(
    directory: Path,
    recursive: bool = False,
) -> list[dict[str, Any]]:
    """Group related listing files by directory and common case-name prefix."""
    iterator = directory.rglob("*") if recursive else directory.iterdir()
    groups: dict[tuple[Path, str], dict[str, Path]] = {}

    for path in iterator:
        if not path.is_file():
            continue
        identified = identify_listing_file(path)
        if identified is None:
            continue
        case_name, file_type = identified
        key = (path.parent.resolve(), case_name)
        files = groups.setdefault(key, {})
        if file_type in files:
            raise ValueError(
                f"Duplicate {file_type} files for case {case_name!r} in {path.parent}"
            )
        files[file_type] = path.resolve()

    return [
        {"parent": parent, "case_name": case_name, "files": files}
        for (parent, case_name), files in sorted(
            groups.items(), key=lambda item: (str(item[0][0]), item[0][1].lower())
        )
    ]


def discover_sibling_group(source: Path) -> dict[str, Any]:
    """Given one listing file, find all related files with the same prefix."""
    identified = identify_listing_file(source)
    if identified is None:
        expected = ", ".join(LISTING_SUFFIXES.values())
        raise ValueError(f"Unrecognized listing filename {source.name}; expected {expected}")

    case_name, _ = identified
    files: dict[str, Path] = {}
    for file_type, suffix in LISTING_SUFFIXES.items():
        candidate_name = case_name + suffix
        # First try the exact conventional spelling.
        candidate = source.parent / candidate_name
        if candidate.is_file():
            files[file_type] = candidate.resolve()
            continue

        # Fall back to a case-insensitive directory lookup.
        lower_candidate = candidate_name.lower()
        for sibling in source.parent.iterdir():
            if sibling.is_file() and sibling.name.lower() == lower_candidate:
                files[file_type] = sibling.resolve()
                break

    return {"parent": source.parent.resolve(), "case_name": case_name, "files": files}


def process_listing_group(
    files: dict[str, Path],
    out_dir: Path,
    case_name: str,
) -> dict[str, Any]:
    """Extract all available listing files belonging to one ESABASE2 case."""
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "case_name": case_name,
        "output_directory": str(out_dir.resolve()),
        "source_files": {key: str(value) for key, value in sorted(files.items())},
    }

    if "FAI_D" in files:
        wide, long = parse_threshold_matrix(files["FAI_D"])
        write_csv(out_dir / "FAI_D_wide.csv", wide)
        write_csv(out_dir / "FAI_D_long.csv", long)
        manifest["FAI_D"] = {"elements": len(wide), "values": len(long)}

    if "CRT_D" in files:
        wide, long = parse_threshold_matrix(files["CRT_D"])
        write_csv(out_dir / "CRT_D_wide.csv", wide)
        write_csv(out_dir / "CRT_D_long.csv", long)
        manifest["CRT_D"] = {"elements": len(wide), "values": len(long)}

    if "LISKIN" in files:
        rows = parse_liskin(files["LISKIN"])
        write_csv(out_dir / "LISKIN_orbital_points.csv", rows)
        manifest["LISKIN"] = {"orbital_points": len(rows)}

    if "LISORB" in files:
        metadata, ephemeris = parse_lisorb(files["LISORB"])
        (out_dir / "LISORB_metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        write_csv(out_dir / "LISORB_ephemeris.csv", ephemeris)
        manifest["LISORB"] = {
            "metadata_fields": len(metadata),
            "printed_ephemeris_rows": len(ephemeris),
        }

    if "LISDM" in files:
        orbits, elements, objects, distributions = parse_lisdm(files["LISDM"])
        write_csv(out_dir / "LISDM_orbits.csv", orbits)
        write_csv(out_dir / "LISDM_elements.csv", elements)
        write_csv(out_dir / "LISDM_objects.csv", objects)
        write_csv(out_dir / "LISDM_distributions.csv", distributions)
        manifest["LISDM"] = {
            "orbital_points": len(orbits),
            "element_rows": len(elements),
            "object_rows": len(objects),
            "distribution_rows": len(distributions),
        }

    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest


def group_destination(
    group_parent: Path,
    case_name: str,
    input_directory: Path,
    output_root: Path | None,
) -> Path:
    if output_root is None:
        return group_parent / f"{case_name}_extracted"

    relative_parent = group_parent.resolve().relative_to(input_directory.resolve())
    return output_root / relative_parent / f"{case_name}_extracted"


def explicit_files_from_args(args: argparse.Namespace) -> dict[str, Path]:
    mapping = {
        "FAI_D": args.fai_d,
        "CRT_D": args.crt_d,
        "LISKIN": args.liskin,
        "LISORB": args.lisorb,
        "LISDM": args.lisdm,
    }
    return {key: path.expanduser().resolve() for key, path in mapping.items() if path}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract one ESABASE2 listing set, or scan a directory and group "
            "all related *_FAI_D, *_CRT_D, *_LISKIN, *_LISORB and *_LISDM files "
            "by their common case-name prefix."
        )
    )
    parser.add_argument(
        "--input_path",
        default=r"C:\Users\maxiv\Documents\UWO\Papers\0.5)METEORCAM-Strawman\Strawman\Orbits\IMEM2\test\demo-project\ListingFiles",
        nargs="?",
        type=Path,
        help=(
            "A directory to scan, or one listing file whose related sibling "
            "files should be discovered automatically"
        ),
    )
    parser.add_argument("--fai-d", type=Path)
    parser.add_argument("--crt-d", type=Path)
    parser.add_argument("--liskin", type=Path)
    parser.add_argument("--lisorb", type=Path)
    parser.add_argument("--lisdm", type=Path)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=(
            "For explicit/single-case mode, use this exact destination. For "
            "directory mode, use this as the root containing one "
            "<case>_extracted folder per case."
        ),
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search subdirectories when input_path is a directory",
    )
    args = parser.parse_args()

    explicit_files = explicit_files_from_args(args)
    if args.input_path is not None and explicit_files:
        parser.error("Use either input_path discovery or explicit --fai-d/... arguments, not both")

    # Backward-compatible explicit-file mode.
    if explicit_files:
        destination = args.out_dir or Path("extracted")
        manifest = process_listing_group(explicit_files, destination, "explicit_case")
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        return

    if args.input_path is None:
        parser.error("Provide a directory, one listing file, or explicit listing arguments")

    input_path = args.input_path.expanduser().resolve()

    # Supplying one LISDM/LISKIN/etc. file discovers its related sibling files.
    if input_path.is_file():
        group = discover_sibling_group(input_path)
        destination = args.out_dir or (
            group["parent"] / f"{group['case_name']}_extracted"
        )
        manifest = process_listing_group(
            group["files"], destination, group["case_name"]
        )
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        return

    if not input_path.is_dir():
        raise SystemExit(f"Input path does not exist: {input_path}")

    groups = discover_listing_groups(input_path, recursive=args.recursive)
    if not groups:
        raise SystemExit(f"No recognized ESABASE2 listing files found in {input_path}")

    output_root = args.out_dir.expanduser().resolve() if args.out_dir else None
    if output_root is not None:
        output_root.mkdir(parents=True, exist_ok=True)

    successes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for group in groups:
        destination = group_destination(
            group["parent"], group["case_name"], input_path, output_root
        )
        try:
            manifest = process_listing_group(
                group["files"], destination, group["case_name"]
            )
            successes.append(manifest)
            types = ", ".join(sorted(group["files"]))
            print(f"[OK] {group['case_name']}: {types} -> {destination}")
        except Exception as exc:  # Continue processing the remaining cases.
            failure = {
                "case_name": group["case_name"],
                "source_files": {
                    key: str(value) for key, value in sorted(group["files"].items())
                },
                "error": str(exc),
            }
            failures.append(failure)
            print(f"[ERROR] {group['case_name']}: {exc}")

    batch_manifest = {
        "input_directory": str(input_path),
        "recursive": args.recursive,
        "processed": len(successes),
        "failed": len(failures),
        "results": successes,
        "errors": failures,
    }
    manifest_root = output_root or input_path
    manifest_file = manifest_root / "esabase_listing_batch_manifest.json"
    manifest_file.write_text(
        json.dumps(batch_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(
        f"Finished: {len(successes)} cases processed, {len(failures)} failed. "
        f"Batch manifest: {manifest_file}"
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
