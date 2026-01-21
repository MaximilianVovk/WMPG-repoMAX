#!/usr/bin/env python3
"""
Extract metrics from obs_data*.json files and plot height vs trail length.

Usage:
  python extract_obs_data_metrics.py /path/to/input [--outdir /path/to/out]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from statistics import median

import matplotlib.pyplot as plt


NAME_RE = re.compile(r"obs_data_(.+?)\.json$", re.IGNORECASE)


def find_obs_jsons(input_path: Path) -> list[Path]:
    """Return list of *.json files whose filename contains 'obs_data'."""
    if input_path.is_file():
        return [input_path] if (input_path.suffix.lower() == ".json" and "obs_data" in input_path.name.lower()) else []
    # directory
    return sorted([p for p in input_path.rglob("*.json") if "obs_data" in p.name.lower()])


def safe_median(values) -> float | None:
    vals = [float(v) for v in values if v is not None]
    return median(vals) if vals else None


def parse_event_name(json_path: Path) -> str:
    m = NAME_RE.search(json_path.name)
    if m:
        return m.group(1)
    # Fallback: strip extension and try to remove leading obs_data_
    stem = json_path.stem
    return stem[len("obs_data_"):] if stem.lower().startswith("obs_data_") else stem


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def compute_metrics(d: dict) -> dict[str, float | str | None]:
    # Median speed from velocities
    v_med = safe_median(d.get("velocities", []))

    # Total trail length from length array
    length_arr = d.get("length", [])
    trail_len = None
    if isinstance(length_arr, list) and len(length_arr) >= 2:
        try:
            fl = [float(x) for x in length_arr]
            trail_len = max(fl) - min(fl)
        except Exception:
            trail_len = None

    # First height (prefer height_lum; fallback to height_lag)
    h_arr = d.get("height_lum", None)
    if not isinstance(h_arr, list) or len(h_arr) == 0:
        h_arr = d.get("height_lag", None)

    h_first = None
    if isinstance(h_arr, list) and len(h_arr) > 0:
        try:
            h_first = float(h_arr[0])
        except Exception:
            h_first = None

    # F parameter: position of peak brightness (min absolute magnitude) along the trail
    abs_mag = d.get("absolute_magnitudes", [])
    F = None
    if isinstance(abs_mag, list) and len(abs_mag) >= 2:
        try:
            mags = [float(m) for m in abs_mag]
            idx_peak = min(range(len(mags)), key=lambda i: mags[i])  # lower mag => brighter
            n = len(mags)
            F = idx_peak / (n - 1) if n > 1 else 0.0
        except Exception:
            F = None

    return {
        "median_speed_m_per_s": v_med,
        "total_trail_length_m": trail_len,
        "first_height_m": h_first,
        "F_peak_position": F,
    }


def write_csv(rows: list[dict], out_csv: Path) -> None:
    fieldnames = [
        "name",
        "json_path",
        "median_speed_m_per_s",
        "total_trail_length_m",
        "first_height_m",
        "F_peak_position",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def plot_scatter(rows: list[dict], out_png: Path) -> None:
    # filter valid numeric rows for plotting
    plot_rows = []
    for r in rows:
        try:
            x = float(r["first_height_m"])
            y = float(r["total_trail_length_m"])
            F = float(r["F_peak_position"])
            vel = float(r["median_speed_m_per_s"])
            if not (0.0 <= F <= 1.0):
                continue
            plot_rows.append((r["name"], x, y, F, vel))
        except Exception:
            continue

    if not plot_rows:
        print("No rows with valid (first_height_m, total_trail_length_m, F) to plot.")
        return

    names, ys, xs, Fs, vels = zip(*plot_rows)

    fig, ax = plt.subplots(figsize=(10, 7))
    sc_vel = ax.scatter([x / 1000 for x in xs], [y / 1000 for y in ys], c=[v / 1000 for v in vels], cmap="viridis",s=100) #, vmin=11.0, vmax=72.0
    sc = ax.scatter([x / 1000 for x in xs], [y / 1000 for y in ys], c=Fs, cmap="seismic_r", vmin=0.0, vmax=0.6)

    # label points
    for name, y, x, _, _ in plot_rows:
        ax.annotate(name, (x / 1000, y / 1000), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_ylabel("First height (km)")
    ax.set_xlabel("Total trail length (km)")
    # ax.set_title("Height vs Total Trail Length (colored by F peak position)")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("F")
    cbar_vel = fig.colorbar(sc_vel, ax=ax)
    cbar_vel.set_label("Median speed (km/s)")
    plt.grid()

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="/srv/public/mvovk/3rdPaper-Sporadic/JB-base-dynesty", help="Input path (file or directory).")
    ap.add_argument("--outdir", type=str, default="/srv/public/mvovk/3rdPaper-Sporadic/JB-rerun", help="Output directory (default: current working dir).")
    args = ap.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve() if args.outdir else Path.cwd()

    outdir.mkdir(parents=True, exist_ok=True)
    out_csv = outdir / "obs_data_metrics.csv"
    out_png = outdir / "height_vs_trail_length_F.png"

    json_files = find_obs_jsons(input_path)
    print(f"Found {len(json_files)} obs_data JSON files.")

    rows = []
    for jp in json_files:
        try:
            d = load_json(jp)
            name = parse_event_name(jp)
            metrics = compute_metrics(d)
            row = {
                "name": name,
                "json_path": str(jp),
                **metrics,
            }
            rows.append(row)
        except Exception as e:
            print(f"Skipping {jp} due to error: {e}")

    # Save CSV
    write_csv(rows, out_csv)
    print(f"Saved CSV: {out_csv}")

    # Plot
    plot_scatter(rows, out_png)
    print(f"Saved plot: {out_png}")


if __name__ == "__main__":
    main()
