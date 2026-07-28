#!/usr/bin/env python3
"""
Combine decade-wide IMEM2 runs into flux, momentum, and kinetic-energy values for each mass bin.

The script expects folders such as:
    MeteoroidMars-10-4g_10-5g_extracted
    MeteoroidMars-10-5g_10-6g_extracted
    ...

Each folder should contain:
    *_listing.txt
    *_tabdata.csv

Important conventions
---------------------
1. The MISSION LEVEL table in *_listing.txt is treated as the authoritative
   result for each selected mass interval.
2. IMEM surface numbering is mapped as:
       1 = +x ram
       2 = -y
       3 = -x wake
       4 = +y
       5 = +z
       6 = -z
3. Each mass interval is represented by its geometric-mean mass.
4. The code assumes the six BOX faces have equal area, as in the standard
   six-square-metre IMEM box output. Face spectra remain valid independently.
5. The tabdata velocity distribution is averaged across orbital points rather
   than summed. Summing would multiply the environment by the number of points.
6. The velocity histogram is used for energy moments only if its integrated
   flux agrees with the mission-level result. Otherwise the script uses the
   mission-level mean speed and marks kinetic energy as a mean-speed
   approximation.

Outputs
-------
- imem2_mass_spectra_box_average.csv
- imem2_directional_spectra.csv
- imem2_velocity_histograms.csv
- imem2_quality_checks.csv
- Separate PNG plots for mass-bin flux, momentum, energy, directional flux,
  mean speed, and mission impacts.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SURFACE_MAP = {
    1: "+x ram",
    2: "-y",
    3: "-x wake",
    4: "+y",
    5: "+z",
    6: "-z",
}

MASS_RE = re.compile(r"10-(\d+)g_10-(\d+)g", re.IGNORECASE)


def as_float(value: str) -> float:
    """Read standard or Fortran-style scientific notation."""
    return float(value.replace("D", "E").replace("d", "e"))


def parse_mass_bin(folder_name: str) -> dict[str, float]:
    match = MASS_RE.search(folder_name)
    if not match:
        raise ValueError(f"Could not extract mass limits from folder: {folder_name}")

    m_a = 10.0 ** (-int(match.group(1)))
    m_b = 10.0 ** (-int(match.group(2)))
    m_low = min(m_a, m_b)
    m_high = max(m_a, m_b)
    m_mid = math.sqrt(m_low * m_high)
    dex_width = abs(math.log10(m_high) - math.log10(m_low))

    return {
        "mass_low_g": m_low,
        "mass_high_g": m_high,
        "mass_mid_g": m_mid,
        "dex_width": dex_width,
    }


def parse_listing(path: Path) -> tuple[dict[str, float], pd.DataFrame]:
    """
    Parse the MISSION LEVEL BOX row and the six element rows.

    Expected element columns:
    EL, Ks-Fc, Imp.Angle, Imp.Vel, Crat.Area, Imp.Flux, Imp.Flue,
    Nb.Impacts, Fail.Flux, Fail.Flue, Tot.Fail
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    marker = text.find("MISSION LEVEL")
    if marker < 0:
        raise ValueError(f"'MISSION LEVEL' was not found in {path}")

    mission_text = text[marker:]
    box: dict[str, float] | None = None
    rows: list[dict[str, Any]] = []
    in_element_table = False

    for raw_line in mission_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("BOX"):
            tokens = line.split()
            if len(tokens) >= 11:
                numbers = [as_float(token) for token in tokens[1:11]]
                box = {
                    "ks_factor": numbers[0],
                    "crater_area_percent": numbers[1],
                    "impact_flux_per_m2_yr": numbers[2],
                    "impact_fluence_per_m2": numbers[3],
                    "total_impacts": numbers[4],
                    "failure_flux_per_m2_yr": numbers[5],
                    "failure_fluence_per_m2": numbers[6],
                    "total_failures": numbers[7],
                    "surface_area_m2": numbers[8],
                    "probability_no_failure": numbers[9],
                }

        if line.startswith("EL #") or re.match(r"^EL\s+#", line):
            in_element_table = True
            continue

        if in_element_table:
            tokens = line.split()
            if (
                len(tokens) >= 11
                and tokens[0].isdigit()
                and int(tokens[0]) in SURFACE_MAP
            ):
                element = int(tokens[0])
                numbers = [as_float(token) for token in tokens[1:11]]
                rows.append(
                    {
                        "element": element,
                        "direction": SURFACE_MAP[element],
                        "ks_factor": numbers[0],
                        "impact_angle_deg": numbers[1],
                        "mean_velocity_km_s": numbers[2],
                        "crater_area_percent": numbers[3],
                        "impact_flux_per_m2_yr": numbers[4],
                        "impact_fluence_per_m2": numbers[5],
                        "nb_impacts": numbers[6],
                        "failure_flux_per_m2_yr": numbers[7],
                        "failure_fluence_per_m2": numbers[8],
                        "total_failures": numbers[9],
                    }
                )
                if len(rows) == 6:
                    break

    if box is None:
        raise ValueError(f"Could not parse the BOX mission-level row in {path}")
    if len(rows) != 6:
        raise ValueError(
            f"Expected six MISSION LEVEL element rows in {path}, found {len(rows)}"
        )

    return box, pd.DataFrame(rows)


def read_velocity_histogram(path: Path) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Return an orbit-averaged velocity histogram.

    If all orbital points are duplicates, the mean preserves one copy.
    If points vary and are evenly spaced in time, the mean is the orbit average.
    For non-uniform time sampling, replace this mean with a time-weighted mean.
    """
    frame = pd.read_csv(path)
    required = {
        "quantity_code",
        "orbital_point",
        "x_name",
        "x_unit",
        "x_value",
        "y_name",
        "y_unit",
        "y_value",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    velocity = frame.loc[
        frame["quantity_code"].astype(str).str.lower().eq("vel")
    ].copy()
    if velocity.empty:
        raise ValueError(f"No quantity_code='vel' rows were found in {path}")

    velocity["velocity_km_s"] = pd.to_numeric(velocity["x_value"], errors="coerce")
    velocity["flux_per_m2_yr"] = pd.to_numeric(
        velocity["y_value"], errors="coerce"
    )
    velocity = velocity.dropna(subset=["velocity_km_s", "flux_per_m2_yr"])

    point_totals = velocity.groupby("orbital_point")["flux_per_m2_yr"].sum()
    averaged = (
        velocity.groupby("velocity_km_s", as_index=False)["flux_per_m2_yr"]
        .mean()
        .sort_values("velocity_km_s")
    )

    flux = averaged["flux_per_m2_yr"].to_numpy(float)
    speed = averaged["velocity_km_s"].to_numpy(float)
    total = float(flux.sum())

    if total > 0:
        mean_v = float(np.sum(flux * speed) / total)
        mean_v2 = float(np.sum(flux * speed**2) / total)
        rms_v = math.sqrt(mean_v2)
    else:
        mean_v = math.nan
        mean_v2 = math.nan
        rms_v = math.nan

    checks = {
        "n_orbital_points": int(velocity["orbital_point"].nunique()),
        "point_total_mean": float(point_totals.mean()),
        "point_total_std": float(point_totals.std(ddof=0)),
        "velocity_hist_integral": total,
        "hist_flux_weighted_mean_velocity_km_s": mean_v,
        "hist_flux_weighted_rms_velocity_km_s": rms_v,
        "hist_flux_weighted_mean_v2_km2_s2": mean_v2,
    }
    return averaged, checks


def safe_ratio(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b) or b == 0:
        return math.nan
    return a / b


def analyse_folder(folder: Path, histogram_tolerance: float):
    mass = parse_mass_bin(folder.name)

    listing_files = sorted(folder.glob("*_listing.txt"))
    tabdata_files = sorted(folder.glob("*_tabdata.csv"))
    if len(listing_files) != 1:
        raise FileNotFoundError(
            f"Expected exactly one *_listing.txt in {folder}, found {len(listing_files)}"
        )
    if len(tabdata_files) != 1:
        raise FileNotFoundError(
            f"Expected exactly one *_tabdata.csv in {folder}, found {len(tabdata_files)}"
        )

    box, faces = parse_listing(listing_files[0])
    histogram, hist_checks = read_velocity_histogram(tabdata_files[0])

    m_g = mass["mass_mid_g"]
    m_kg = m_g * 1e-3
    dex = mass["dex_width"]

    # Direction-specific spectra use the authoritative face flux and mean velocity.
    faces = faces.copy()
    for key, value in mass.items():
        faces[key] = value
    faces["folder"] = folder.name
    faces["source"] = "IMEM2"

    # Values are reported directly for the complete selected mass bin.
    # The present IMEM2 runs all use one-decade-wide bins, so these values
    # are numerically identical to the former "per dex" quantities.
    faces["flux_per_mass_bin_per_m2_yr"] = faces["impact_flux_per_m2_yr"]
    faces["momentum_per_mass_bin_g_km_s_per_m2_yr"] = (
        faces["impact_flux_per_m2_yr"]
        * m_g
        * faces["mean_velocity_km_s"]
    )
    faces["kinetic_energy_per_mass_bin_J_per_m2_yr"] = (
        faces["impact_flux_per_m2_yr"]
        * 0.5
        * m_kg
        * (faces["mean_velocity_km_s"] * 1000.0) ** 2
    )
    faces["energy_method"] = "face mean-speed approximation"

    # The six-face average reproduces the BOX flux when all faces have equal area.
    number_box_from_faces = float(faces["impact_flux_per_m2_yr"].mean())
    momentum_box_from_faces = float(
        faces["momentum_per_mass_bin_g_km_s_per_m2_yr"].mean()
    )
    energy_box_from_faces = float(
        faces["kinetic_energy_per_mass_bin_J_per_m2_yr"].mean()
    )

    box_flux = float(box["impact_flux_per_m2_yr"])
    hist_flux = float(hist_checks["velocity_hist_integral"])

    # Compare the velocity-histogram integral with both common mission normalisations:
    # (a) area-averaged BOX flux and (b) six-face summed rate for six 1 m² faces.
    hist_to_box_ratio = safe_ratio(hist_flux, box_flux)
    hist_to_face_sum_ratio = safe_ratio(
        hist_flux, float(faces["impact_flux_per_m2_yr"].sum())
    )

    closest_ratio = np.nanmin(
        np.abs(
            np.array([hist_to_box_ratio, hist_to_face_sum_ratio], dtype=float) - 1.0
        )
    )
    histogram_usable = bool(
        np.isfinite(closest_ratio) and closest_ratio <= histogram_tolerance
    )

    if histogram_usable and hist_flux > 0:
        # Normalize the velocity shape to the authoritative BOX area-averaged flux.
        hist_norm = histogram.copy()
        hist_norm["flux_per_m2_yr"] *= box_flux / hist_flux

        f = hist_norm["flux_per_m2_yr"].to_numpy(float)
        v = hist_norm["velocity_km_s"].to_numpy(float)

        momentum_box = float(np.sum(f * m_g * v))
        energy_box = float(
            np.sum(f * 0.5 * m_kg * (v * 1000.0) ** 2)
        )
        energy_method = "renormalized bin-specific velocity histogram"
    else:
        momentum_box = momentum_box_from_faces
        energy_box = energy_box_from_faces
        energy_method = "six-face mean-speed approximation"

    # Infer the modelled duration from face fluence/rate and NB impacts/rate.
    valid_duration = faces.loc[
        faces["impact_flux_per_m2_yr"] > 0,
        ["impact_flux_per_m2_yr", "nb_impacts"],
    ]
    duration_years = float(
        np.median(
            valid_duration["nb_impacts"]
            / valid_duration["impact_flux_per_m2_yr"]
        )
    )

    box_row = {
        **mass,
        "folder": folder.name,
        "source": "IMEM2",
        "box_impact_flux_per_m2_yr": box_flux,
        "box_flux_reconstructed_from_face_mean": number_box_from_faces,
        "flux_per_mass_bin_per_m2_yr": box_flux,
        "momentum_per_mass_bin_g_km_s_per_m2_yr": momentum_box,
        "kinetic_energy_per_mass_bin_J_per_m2_yr": energy_box,
        "energy_method": energy_method,
        "mission_duration_years": duration_years,
        "mission_duration_days": duration_years * 365.25,
        "mission_duration_hours": duration_years * 365.25 * 24.0,
        "box_surface_area_m2": box["surface_area_m2"],
        "box_total_impacts": box["total_impacts"],
    }

    quality = {
        **mass,
        "folder": folder.name,
        "source": "IMEM2",
        **hist_checks,
        "listing_box_flux_per_m2_yr": box_flux,
        "face_flux_sum_per_yr_for_six_1m2_faces": float(
            faces["impact_flux_per_m2_yr"].sum()
        ),
        "face_flux_mean_per_m2_yr": number_box_from_faces,
        "hist_to_box_flux_ratio": hist_to_box_ratio,
        "hist_to_face_sum_ratio": hist_to_face_sum_ratio,
        "velocity_histogram_accepted": histogram_usable,
        "energy_method_used": energy_method,
    }

    histogram = histogram.copy()
    for key, value in mass.items():
        histogram[key] = value
    histogram["folder"] = folder.name
    histogram["source"] = "IMEM2"

    return box_row, faces, histogram, quality


def save_log_plot(
    data: pd.DataFrame,
    y: str,
    ylabel: str,
    title: str,
    output: Path,
):
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.plot(data["mass_mid_g"], data[y], marker="o")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Geometric-mean particle mass (g)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def save_directional_plot(
    data: pd.DataFrame,
    y: str,
    ylabel: str,
    title: str,
    output: Path,
    log_y: bool = True,
):
    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    for direction, group in data.groupby("direction", sort=False):
        group = group.sort_values("mass_mid_g")
        ax.plot(group["mass_mid_g"], group[y], marker="o", label=direction)

    ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel("Geometric-mean particle mass (g)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=r"C:\Users\maxiv\Documents\UWO\Papers\0.5)METEORCAM-Strawman\Strawman\Orbits\IMEM2",
        help="Directory containing the MeteoroidMars-..._extracted folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory. Default: <root>/combined_IMEM2_spectra",
    )
    parser.add_argument(
        "--histogram-tolerance",
        type=float,
        default=0.20,
        help=(
            "Maximum fractional disagreement for accepting a tabdata velocity "
            "histogram as mass-bin-specific. Default: 0.20."
        ),
    )
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else root / "combined_IMEM2_spectra"
    )
    output.mkdir(parents=True, exist_ok=True)

    folders = sorted(
        folder
        for folder in root.iterdir()
        if folder.is_dir() and MASS_RE.search(folder.name)
    )
    if not folders:
        raise FileNotFoundError(
            f"No folders matching the expected decade-bin naming pattern in {root}"
        )

    box_rows = []
    face_frames = []
    histogram_frames = []
    quality_rows = []
    errors = []

    for folder in folders:
        try:
            box_row, faces, histogram, quality = analyse_folder(
                folder, args.histogram_tolerance
            )
            box_rows.append(box_row)
            face_frames.append(faces)
            histogram_frames.append(histogram)
            quality_rows.append(quality)
            print(f"Processed: {folder.name}")
        except Exception as exc:
            errors.append({"folder": folder.name, "error": str(exc)})
            print(f"FAILED: {folder.name}: {exc}")

    if not box_rows:
        raise RuntimeError("No mass-bin folders were processed successfully.")

    box_df = pd.DataFrame(box_rows).sort_values("mass_mid_g")
    faces_df = pd.concat(face_frames, ignore_index=True).sort_values(
        ["element", "mass_mid_g"]
    )
    hist_df = pd.concat(histogram_frames, ignore_index=True).sort_values(
        ["mass_mid_g", "velocity_km_s"]
    )
    quality_df = pd.DataFrame(quality_rows).sort_values("mass_mid_g")

    box_df.to_csv(output / "imem2_mass_spectra_box_average.csv", index=False)
    faces_df.to_csv(output / "imem2_directional_spectra.csv", index=False)
    hist_df.to_csv(output / "imem2_velocity_histograms.csv", index=False)
    quality_df.to_csv(output / "imem2_quality_checks.csv", index=False)
    if errors:
        pd.DataFrame(errors).to_csv(output / "imem2_processing_errors.csv", index=False)

    save_log_plot(
        box_df,
        "flux_per_mass_bin_per_m2_yr",
        r"Impact flux in mass bin (m$^{-2}$ yr$^{-1}$)",
        "IMEM2 impact flux by mass bin",
        output / "01_flux_per_mass_bin_box_average.png",
    )
    save_log_plot(
        box_df,
        "momentum_per_mass_bin_g_km_s_per_m2_yr",
        r"Momentum delivered by mass bin (g km s$^{-1}$ m$^{-2}$ yr$^{-1}$)",
        "IMEM2 momentum delivered by mass bin",
        output / "02_momentum_per_mass_bin_box_average.png",
    )
    save_log_plot(
        box_df,
        "kinetic_energy_per_mass_bin_J_per_m2_yr",
        r"Kinetic energy delivered by mass bin (J m$^{-2}$ yr$^{-1}$)",
        "IMEM2 kinetic energy delivered by mass bin",
        output / "03_kinetic_energy_per_mass_bin_box_average.png",
    )

    save_directional_plot(
        faces_df,
        "flux_per_mass_bin_per_m2_yr",
        r"Impact flux in mass bin (m$^{-2}$ yr$^{-1}$)",
        "Direction-specific IMEM2 impact flux by mass bin",
        output / "04_directional_flux_per_mass_bin.png",
    )
    save_directional_plot(
        faces_df,
        "momentum_per_mass_bin_g_km_s_per_m2_yr",
        r"Momentum delivered by mass bin (g km s$^{-1}$ m$^{-2}$ yr$^{-1}$)",
        "Direction-specific IMEM2 momentum by mass bin",
        output / "05_directional_momentum_per_mass_bin.png",
    )
    save_directional_plot(
        faces_df,
        "kinetic_energy_per_mass_bin_J_per_m2_yr",
        r"Kinetic energy delivered by mass bin (J m$^{-2}$ yr$^{-1}$)",
        "Direction-specific IMEM2 kinetic energy by mass bin",
        output / "06_directional_kinetic_energy_per_mass_bin.png",
    )
    save_directional_plot(
        faces_df,
        "impact_flux_per_m2_yr",
        r"Impact flux (m$^{-2}$ yr$^{-1}$)",
        "Mission-level impact flux by spacecraft face",
        output / "07_directional_flux.png",
    )
    save_directional_plot(
        faces_df,
        "mean_velocity_km_s",
        r"Mean impact velocity (km s$^{-1}$)",
        "Mission-level mean impact velocity by spacecraft face",
        output / "08_directional_mean_velocity.png",
        log_y=False,
    )
    save_directional_plot(
        faces_df,
        "nb_impacts",
        "Number of impacts over the modelled mission interval",
        "Mission-level impacts by spacecraft face and mass bin",
        output / "09_directional_nb_impacts.png",
    )

    print()
    print(f"Results written to: {output}")
    print(
        "Check imem2_quality_checks.csv before using the velocity histograms. "
        "A rejected histogram means its integrated flux did not match the "
        "mission-level mass-bin result."
    )


if __name__ == "__main__":
    main()
