#!/usr/bin/env python3
"""
Extract IMEM2/ESABASE2 TABDATA arrays and embedded text reports from a
binary .output file.

The file is an Open CASCADE/OCAF binary document, not a text table.
This reader targets the standard attributes used in ESABASE2 result files
and requires only Python's standard library.

Usage:
    # One file (backward-compatible)
    python read_esabase_imem_output.py result.output

    # Every .output file in one directory
    python read_esabase_imem_output.py /path/to/results

    # Include subdirectories and place all extraction folders under one root
    python read_esabase_imem_output.py /path/to/results --recursive --out-dir extracted
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import struct
from pathlib import Path
from typing import Iterable

# GUID that immediately precedes the serialized TDataStd_RealArray record
# in the ESABASE2 files examined here.
_REAL_ARRAY_GUID = bytes.fromhex(
    "08 b6 96 2a 8b ec d0 11 be e7 08 00 09 dc 33 33"
)
_DATASET_RE = re.compile(
    r"^IMEM2_(dia|ele|azi|vel|dens)_(op\d{4}|avg)\s*$"
)
_ALLOWED_UTF16_CHARS = set(range(32, 127)) | {9, 10, 13}


def utf16_ascii_runs(
    blob: bytes, start: int = 0, end: int | None = None, min_chars: int = 4
) -> list[tuple[int, int, str]]:
    """Find contiguous printable UTF-16LE strings inside arbitrary binary data."""
    if end is None:
        end = len(blob)

    runs: list[tuple[int, int, str]] = []
    i = start

    while i + 1 < end:
        if blob[i] in _ALLOWED_UTF16_CHARS and blob[i + 1] == 0:
            j = i
            chars: list[str] = []

            while (
                j + 1 < end
                and blob[j] in _ALLOWED_UTF16_CHARS
                and blob[j + 1] == 0
            ):
                chars.append(chr(blob[j]))
                j += 2

            if len(chars) >= min_chars:
                runs.append((i, j, "".join(chars)))

            i = max(j, i + 1)
        else:
            i += 1

    return runs


def exact_runs(
    runs: Iterable[tuple[int, int, str]], text: str
) -> list[tuple[int, int, str]]:
    return [run for run in runs if run[2].strip() == text]


def value_after_token(
    block_runs: list[tuple[int, int, str]], token: str
) -> str | None:
    """
    ESABASE2 generally stores:
        TOKEN, human-readable label, value
    as consecutive UTF-16LE strings.
    """
    ignored = {
        "Content",
        "Units",
        "Names",
        "Data x",
        "Data y",
        "Length of the array",
    }

    for i, run in enumerate(block_runs):
        if run[2].strip() != token:
            continue

        for candidate in block_runs[i + 2 : i + 6]:
            value = candidate[2].strip()
            if value and value not in ignored:
                return value

    return None


def extract_real_array(
    blob: bytes, block_start: int, block_end: int, token: str
) -> list[float]:
    """
    Read a serialized TDataStd_RealArray following DATAX or DATAY.

    Layout used here:
      GUID (16 bytes)
      attribute type ID (uint32; 7 = TDataStd_RealArray)
      record ID (uint32)
      payload size (uint32)
      lower index (int32)
      upper index (int32)
      values (float64, little-endian)
      delta flag (1 byte)
    """
    token_bytes = (token + "\x00").encode("utf-16le")
    token_offset = blob.find(token_bytes, block_start, block_end)

    if token_offset < 0:
        raise ValueError(f"{token} not found in dataset block at {block_start}")

    signature = _REAL_ARRAY_GUID + struct.pack("<I", 7)
    record_offset = blob.find(
        signature, token_offset, min(block_end, token_offset + 512)
    )

    if record_offset < 0:
        raise ValueError(
            f"TDataStd_RealArray record not found after {token} "
            f"at byte {token_offset}"
        )

    header_offset = record_offset + len(_REAL_ARRAY_GUID)
    type_id, record_id, payload_size, lower, upper = struct.unpack_from(
        "<IIIii", blob, header_offset
    )

    if type_id != 7:
        raise ValueError(f"Unexpected attribute type ID {type_id}")

    count = upper - lower + 1
    if count <= 0 or count > 10_000_000:
        raise ValueError(
            f"Invalid array bounds [{lower}, {upper}] at byte {header_offset}"
        )

    values_offset = header_offset + 20
    required = values_offset + count * 8
    if required > len(blob):
        raise ValueError("Real-array data extends beyond end of file")

    return list(struct.unpack_from(f"<{count}d", blob, values_offset))


def split_pair(value: str | None, fallback_a: str, fallback_b: str) -> tuple[str, str]:
    if value and ":" in value:
        return tuple(value.split(":", 1))  # type: ignore[return-value]
    return fallback_a, fallback_b


def extract_tabdata(blob: bytes) -> list[dict]:
    all_runs = utf16_ascii_runs(blob)
    tab_markers = exact_runs(all_runs, "TABDATA")
    chart_markers = exact_runs(all_runs, "CHARTS")

    if not tab_markers or not chart_markers:
        raise ValueError("TABDATA/CHARTS section markers were not found")

    tab_start = tab_markers[-1][1]
    chart_candidates = [r for r in chart_markers if r[0] > tab_start]
    if not chart_candidates:
        raise ValueError("CHARTS marker following TABDATA was not found")
    tab_end = chart_candidates[0][0]

    tab_runs = [r for r in all_runs if tab_start <= r[0] < tab_end]

    starts: list[tuple[int, str]] = []
    for run in tab_runs:
        if not _DATASET_RE.match(run[2]):
            continue

        name = run[2].strip()

        # Dataset IDs are stored twice in succession; retain only the first.
        if starts and starts[-1][1] == name and run[0] - starts[-1][0] < 100:
            continue

        starts.append((run[0], name))

    datasets: list[dict] = []

    for i, (block_start, dataset_name) in enumerate(starts):
        block_end = starts[i + 1][0] if i + 1 < len(starts) else tab_end
        block_runs = [
            run for run in tab_runs if block_start <= run[0] < block_end
        ]

        x_values = extract_real_array(blob, block_start, block_end, "DATAX")
        y_values = extract_real_array(blob, block_start, block_end, "DATAY")

        if len(x_values) != len(y_values):
            raise ValueError(
                f"{dataset_name}: DATAX has {len(x_values)} entries but "
                f"DATAY has {len(y_values)}"
            )

        match = _DATASET_RE.match(dataset_name)
        assert match is not None
        quantity_code, scope_code = match.groups()

        datasets.append(
            {
                "dataset": dataset_name,
                "quantity_code": quantity_code,
                "scope": scope_code,
                "orbital_point": (
                    int(scope_code[2:]) if scope_code.startswith("op") else None
                ),
                "content": value_after_token(block_runs, "CONTENT"),
                "units": value_after_token(block_runs, "UNITS"),
                "names": value_after_token(block_runs, "NAMES"),
                "x": x_values,
                "y": y_values,
            }
        )

    return datasets


def extract_primary_listing(blob: bytes) -> str | None:
    """
    Return the embedded human-readable debris/meteoroid listing.
    """
    candidates = [
        run[2]
        for run in utf16_ascii_runs(blob, min_chars=100)
        if "Debris/Meteoroid Flux and Damage Analysis" in run[2]
    ]

    return max(candidates, key=len) if candidates else None


def write_long_csv(datasets: list[dict], destination: Path) -> None:
    fieldnames = [
        "dataset",
        "quantity_code",
        "scope",
        "orbital_point",
        "content",
        "x_name",
        "x_unit",
        "x_value",
        "y_name",
        "y_unit",
        "y_value",
    ]

    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for dataset in datasets:
            x_name, y_name = split_pair(
                dataset.get("names"), "x", "Flux"
            )
            x_unit, y_unit = split_pair(
                dataset.get("units"), "", "1/m^2/yr"
            )

            for x_value, y_value in zip(dataset["x"], dataset["y"]):
                writer.writerow(
                    {
                        "dataset": dataset["dataset"],
                        "quantity_code": dataset["quantity_code"],
                        "scope": dataset["scope"],
                        "orbital_point": dataset["orbital_point"],
                        "content": dataset["content"],
                        "x_name": x_name,
                        "x_unit": x_unit,
                        "x_value": f"{x_value:.17g}",
                        "y_name": y_name,
                        "y_unit": y_unit,
                        "y_value": f"{y_value:.17g}",
                    }
                )



def process_output_file(source: Path, out_dir: Path | None = None) -> dict:
    """Decode one binary ESABASE2/IMEM ``.output`` file."""
    source = source.resolve()
    blob = source.read_bytes()

    if not blob.startswith(b"BINFILE"):
        raise ValueError(
            f"{source} does not look like an Open CASCADE/OCAF BINFILE document"
        )

    destination = out_dir or source.with_name(source.stem + "_extracted")
    destination.mkdir(parents=True, exist_ok=True)

    datasets = extract_tabdata(blob)

    csv_file = destination / f"{source.stem}_tabdata.csv"
    json_file = destination / f"{source.stem}_datasets.json"
    listing_file = destination / f"{source.stem}_listing.txt"

    write_long_csv(datasets, csv_file)
    json_file.write_text(
        json.dumps(datasets, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    listing = extract_primary_listing(blob)
    if listing is not None:
        listing_file.write_text(listing, encoding="utf-8")

    result = {
        "source": str(source),
        "output_directory": str(destination.resolve()),
        "datasets": len(datasets),
        "rows": sum(len(dataset["x"]) for dataset in datasets),
        "csv": str(csv_file.resolve()),
        "json": str(json_file.resolve()),
        "listing": str(listing_file.resolve()) if listing is not None else None,
    }

    (destination / "manifest.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return result


def discover_output_files(directory: Path, recursive: bool = False) -> list[Path]:
    """Return all ``.output`` files, using a case-insensitive suffix check."""
    iterator = directory.rglob("*") if recursive else directory.iterdir()
    return sorted(
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() == ".output"
    )


def batch_destination(
    source: Path,
    input_directory: Path,
    output_root: Path | None,
) -> Path:
    """Choose a separate extraction folder for one source file."""
    if output_root is None:
        return source.with_name(source.stem + "_extracted")

    relative_parent = source.parent.resolve().relative_to(input_directory.resolve())
    return output_root / relative_parent / f"{source.stem}_extracted"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract one IMEM/ESABASE2 .output file, or scan a directory and "
            "extract every .output file into a separate folder."
        )
    )
    parser.add_argument(
        "--input_path",
        default=r"C:\Users\maxiv\Documents\UWO\Papers\0.5)METEORCAM-Strawman\Strawman\Orbits\IMEM2\test\demo-project",
        type=Path,
        help="A .output file or a directory containing .output files",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=(
            "For one file, use this exact destination. For a directory, use "
            "this as the root containing one <case>_extracted folder per file."
        ),
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search subdirectories when input_path is a directory",
    )
    args = parser.parse_args()

    input_path = args.input_path.expanduser().resolve()

    if input_path.is_file():
        if input_path.suffix.lower() != ".output":
            raise SystemExit(f"Expected a .output file, received: {input_path}")
        result = process_output_file(input_path, args.out_dir)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    if not input_path.is_dir():
        raise SystemExit(f"Input path does not exist: {input_path}")

    sources = discover_output_files(input_path, recursive=args.recursive)
    if not sources:
        raise SystemExit(f"No .output files found in {input_path}")

    output_root = args.out_dir.expanduser().resolve() if args.out_dir else None
    if output_root is not None:
        output_root.mkdir(parents=True, exist_ok=True)

    successes: list[dict] = []
    failures: list[dict] = []

    for source in sources:
        destination = batch_destination(source, input_path, output_root)
        try:
            result = process_output_file(source, destination)
            successes.append(result)
            print(
                f"[OK] {source.name}: {result['datasets']} datasets -> "
                f"{destination}"
            )
        except Exception as exc:  # Continue processing the remaining cases.
            failure = {"source": str(source), "error": str(exc)}
            failures.append(failure)
            print(f"[ERROR] {source}: {exc}")

    batch_manifest = {
        "input_directory": str(input_path),
        "recursive": args.recursive,
        "processed": len(successes),
        "failed": len(failures),
        "results": successes,
        "errors": failures,
    }
    manifest_root = output_root or input_path
    manifest_file = manifest_root / "imem_output_batch_manifest.json"
    manifest_file.write_text(
        json.dumps(batch_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(
        f"Finished: {len(successes)} processed, {len(failures)} failed. "
        f"Batch manifest: {manifest_file}"
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
