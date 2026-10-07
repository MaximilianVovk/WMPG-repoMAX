#!/usr/bin/env python3
"""
Create sensor-oriented IMEM2 + rough Mars-ring plots.

This version corrects two important interpretation problems:

1. Sensor sensitivity must be compared with PER-IMPACT momentum and kinetic
   energy, not annual momentum/energy delivery after multiplying by flux.

2. The published Phobos/Deimos Figure 5 contains sparse discrete size
   increments. Assigning each point to only one decade-wide mass bin creates
   artificial empty bins. Here, every figure point represents a finite
   logarithmic radius cell, and its contribution is split across overlapping
   mass bins.

Primary outputs
---------------
- Impact flux in each mass bin [m^-2 yr^-1]
- Expected impacts for a chosen detector area and duration
- Mean incident momentum per detected particle [N s = g km/s]
- Mean kinetic energy per detected particle [J]

All of these are kept as four separate series:
- IMEM2 only
- Phobos only
- Deimos only
- Total (IMEM2 + Phobos + Deimos)

Secondary outputs
-----------------
- Annual momentum delivered per unit area
- Annual kinetic energy delivered per unit area

Caveats
-------
- Figure 5 values are approximate hand-digitized values, not author data.
- The figure gives a global steady-state size distribution, not the local
  size distribution at 5720 km. Applying it locally is a rough assumption.
- Total Phobos and Deimos fluxes at 5720 km must be supplied as scenario
  normalisations.
- p = m v is INCIDENT particle momentum. It is not necessarily the momentum
  transferred to a specific film after penetration. A film-transfer curve
  must be applied separately for direct comparison with such sensor models.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PHOBOS_POINTS = [
    (0.5, 3.5e23),
    (1.0, 6.0e22),
    (2.0, 1.2e22),
    (5.0, 3.0e21),
    (10.0, 9.0e21),
    (15.0, 8.0e20),
    (20.0, 2.0e20),
    (25.0, 2.0e20),
    (30.0, 8.0e19),
    (40.0, 4.0e19),
    (60.0, 8.0e18),
    (100.0, 1.0e18),
]

DEIMOS_POINTS = [
    (0.5, 1.1e23),
    (1.0, 1.5e22),
    (2.0, 5.0e21),
    (5.0, 3.0e21),
    (10.0, 6.1e23),
    (15.0, 1.9e23),
    (20.0, 5.0e22),
    (25.0, 1.0e22),
    (30.0, 9.0e21),
    (40.0, 5.0e21),
    (60.0, 2.0e21),
    (100.0, 5.0e20),
]


def radius_um_to_mass_g(radius_um: float, rho_g_cm3: float) -> float:
    radius_cm = radius_um * 1e-4
    return rho_g_cm3 * (4.0 / 3.0) * math.pi * radius_cm**3


def mass_g_to_radius_um(mass_g: float, rho_g_cm3: float) -> float:
    radius_cm = (3.0 * mass_g / (4.0 * math.pi * rho_g_cm3)) ** (1.0 / 3.0)
    return radius_cm * 1e4


def build_size_cells(points, source: str) -> pd.DataFrame:
    """
    Treat each plotted size as one finite logarithmic size increment.

    Interior cell boundaries are geometric means of adjacent plotted radii.
    The distribution is clamped to the published 0.5–100 micrometre range.
    """
    frame = pd.DataFrame(points, columns=["radius_um", "relative_particles"])
    frame = frame.sort_values("radius_um").reset_index(drop=True)

    radii = frame["radius_um"].to_numpy(float)
    lower = np.empty_like(radii)
    upper = np.empty_like(radii)

    lower[0] = radii[0]
    upper[-1] = radii[-1]

    boundaries = np.sqrt(radii[:-1] * radii[1:])
    upper[:-1] = boundaries
    lower[1:] = boundaries

    frame["cell_radius_low_um"] = lower
    frame["cell_radius_high_um"] = upper
    frame["source"] = source
    frame["source_weight"] = (
        frame["relative_particles"] / frame["relative_particles"].sum()
    )
    return frame


def overlap_fraction_log(
    cell_low: float,
    cell_high: float,
    bin_low: float,
    bin_high: float,
) -> float:
    low = max(cell_low, bin_low)
    high = min(cell_high, bin_high)
    if high <= low or cell_high <= cell_low:
        return 0.0
    return math.log(high / low) / math.log(cell_high / cell_low)


def distribute_ring_to_mass_bins(
    imem: pd.DataFrame,
    cells: pd.DataFrame,
    rho_g_cm3: float,
    total_flux_per_m2_yr: float,
    speed_km_s: float,
) -> pd.DataFrame:
    """
    Split every finite size increment across overlapping mass bins.

    Within each overlap segment, use its geometric-mean radius to calculate
    representative mass, incident momentum, and kinetic energy.
    """
    output = []

    for _, mass_bin in imem.iterrows():
        mass_low = float(mass_bin["mass_low_g"])
        mass_high = float(mass_bin["mass_high_g"])
        radius_low = mass_g_to_radius_um(mass_low, rho_g_cm3)
        radius_high = mass_g_to_radius_um(mass_high, rho_g_cm3)

        bin_flux = 0.0
        annual_momentum = 0.0
        annual_energy = 0.0

        for _, cell in cells.iterrows():
            fraction = overlap_fraction_log(
                float(cell["cell_radius_low_um"]),
                float(cell["cell_radius_high_um"]),
                radius_low,
                radius_high,
            )
            if fraction <= 0:
                continue

            overlap_low = max(float(cell["cell_radius_low_um"]), radius_low)
            overlap_high = min(float(cell["cell_radius_high_um"]), radius_high)
            representative_radius = math.sqrt(overlap_low * overlap_high)
            representative_mass_g = radius_um_to_mass_g(
                representative_radius, rho_g_cm3
            )

            flux_piece = (
                total_flux_per_m2_yr
                * float(cell["source_weight"])
                * fraction
            )
            momentum_per_impact = representative_mass_g * speed_km_s
            energy_per_impact = (
                0.5
                * representative_mass_g
                * 1e-3
                * (speed_km_s * 1000.0) ** 2
            )

            bin_flux += flux_piece
            annual_momentum += flux_piece * momentum_per_impact
            annual_energy += flux_piece * energy_per_impact

        if bin_flux > 0:
            mean_momentum = annual_momentum / bin_flux
            mean_energy = annual_energy / bin_flux
        else:
            mean_momentum = math.nan
            mean_energy = math.nan

        output.append(
            {
                "mass_low_g": mass_low,
                "mass_high_g": mass_high,
                "mass_mid_g": float(mass_bin["mass_mid_g"]),
                "flux_per_mass_bin_per_m2_yr": bin_flux,
                "mean_incident_momentum_per_impact_Ns": mean_momentum,
                "mean_kinetic_energy_per_impact_J": mean_energy,
                "annual_momentum_delivery_Ns_per_m2_yr": annual_momentum,
                "annual_kinetic_energy_delivery_J_per_m2_yr": annual_energy,
            }
        )

    return pd.DataFrame(output)


def find_column(frame: pd.DataFrame, candidates: list[str]) -> str:
    for column in candidates:
        if column in frame.columns:
            return column
    raise KeyError(
        "None of the expected columns were found: " + ", ".join(candidates)
    )


def prepare_imem(imem: pd.DataFrame) -> pd.DataFrame:
    flux_column = find_column(
        imem,
        [
            "flux_per_mass_bin_per_m2_yr",
            "number_flux_per_m2_yr_per_dex",
            "box_impact_flux_per_m2_yr",
        ],
    )
    momentum_column = find_column(
        imem,
        [
            "momentum_per_mass_bin_g_km_s_per_m2_yr",
            "momentum_flux_g_km_s_per_m2_yr_per_dex",
        ],
    )
    energy_column = find_column(
        imem,
        [
            "kinetic_energy_per_mass_bin_J_per_m2_yr",
            "energy_flux_J_per_m2_yr_per_dex",
        ],
    )

    result = imem[
        ["mass_low_g", "mass_high_g", "mass_mid_g"]
    ].copy()
    result["imem_flux_per_mass_bin_per_m2_yr"] = imem[flux_column].astype(float)
    result["imem_annual_momentum_delivery_Ns_per_m2_yr"] = imem[
        momentum_column
    ].astype(float)
    result["imem_annual_kinetic_energy_delivery_J_per_m2_yr"] = imem[
        energy_column
    ].astype(float)

    flux = result["imem_flux_per_mass_bin_per_m2_yr"].to_numpy(float)
    momentum = result[
        "imem_annual_momentum_delivery_Ns_per_m2_yr"
    ].to_numpy(float)
    energy = result[
        "imem_annual_kinetic_energy_delivery_J_per_m2_yr"
    ].to_numpy(float)

    result["imem_mean_incident_momentum_per_impact_Ns"] = np.divide(
        momentum,
        flux,
        out=np.full_like(momentum, np.nan),
        where=flux > 0,
    )
    result["imem_mean_kinetic_energy_per_impact_J"] = np.divide(
        energy,
        flux,
        out=np.full_like(energy, np.nan),
        where=flux > 0,
    )
    return result


def positive(values: pd.Series) -> pd.Series:
    return values.where(values > 0, np.nan)


def plot_lines(
    data: pd.DataFrame,
    series: list[tuple[str, str]],
    ylabel: str,
    title: str,
    output: Path,
):
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for column, label in series:
        ax.plot(
            data["mass_mid_g"],
            positive(data[column]),
            marker="o",
            label=label,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Geometric-mean particle mass (g)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--imem-box-csv", 
                        default=r"C:\Users\maxiv\Documents\UWO\Papers\0.5)METEORCAM-Strawman\Strawman\Orbits\IMEM2\combined_IMEM2_spectra\imem2_mass_spectra_box_average.csv", 
                        type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--rho-g-cm3", type=float, default=2.37)
    parser.add_argument("--phobos-total-flux", type=float, default=2.3e5)
    parser.add_argument("--deimos-total-flux", type=float, default=1) # 5.0e4
    parser.add_argument("--phobos-speed-km-s", type=float, default=0.75)
    parser.add_argument("--deimos-speed-km-s", type=float, default=0.75)
    parser.add_argument("--sensor-area-m2", type=float, default=0.01)
    parser.add_argument("--mission-years", type=float, default=1.0)
    args = parser.parse_args()

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else args.imem_box_csv.resolve().parent / "sensor_oriented_ring_scenario"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    imem_raw = pd.read_csv(args.imem_box_csv)
    imem = prepare_imem(imem_raw)

    phobos_cells = build_size_cells(PHOBOS_POINTS, "Phobos")
    deimos_cells = build_size_cells(DEIMOS_POINTS, "Deimos")

    phobos = distribute_ring_to_mass_bins(
        imem,
        phobos_cells,
        args.rho_g_cm3,
        args.phobos_total_flux,
        args.phobos_speed_km_s,
    ).add_prefix("phobos_")
    phobos = phobos.rename(
        columns={
            "phobos_mass_low_g": "mass_low_g",
            "phobos_mass_high_g": "mass_high_g",
            "phobos_mass_mid_g": "mass_mid_g",
        }
    )

    deimos = distribute_ring_to_mass_bins(
        imem,
        deimos_cells,
        args.rho_g_cm3,
        args.deimos_total_flux,
        args.deimos_speed_km_s,
    ).add_prefix("deimos_")
    deimos = deimos.rename(
        columns={
            "deimos_mass_low_g": "mass_low_g",
            "deimos_mass_high_g": "mass_high_g",
            "deimos_mass_mid_g": "mass_mid_g",
        }
    )

    combined = imem.merge(
        phobos, on=["mass_low_g", "mass_high_g", "mass_mid_g"]
    ).merge(
        deimos, on=["mass_low_g", "mass_high_g", "mass_mid_g"]
    )

    combined["ring_flux_per_mass_bin_per_m2_yr"] = (
        combined["phobos_flux_per_mass_bin_per_m2_yr"]
        + combined["deimos_flux_per_mass_bin_per_m2_yr"]
    )

    combined["phobos_total_flux_per_mass_bin_per_m2_yr"] = (
        combined["imem_flux_per_mass_bin_per_m2_yr"]
        + combined["phobos_flux_per_mass_bin_per_m2_yr"]
    )
    combined["deimos_total_flux_per_mass_bin_per_m2_yr"] = (
        combined["imem_flux_per_mass_bin_per_m2_yr"]
        + combined["deimos_flux_per_mass_bin_per_m2_yr"]
    )
    combined["total_flux_with_rings_per_mass_bin_per_m2_yr"] = (
        combined["imem_flux_per_mass_bin_per_m2_yr"]
        + combined["ring_flux_per_mass_bin_per_m2_yr"]
    )

    combined["ring_annual_momentum_delivery_Ns_per_m2_yr"] = (
        combined["phobos_annual_momentum_delivery_Ns_per_m2_yr"]
        + combined["deimos_annual_momentum_delivery_Ns_per_m2_yr"]
    )

    combined["phobos_total_annual_momentum_delivery_Ns_per_m2_yr"] = (
        combined["imem_annual_momentum_delivery_Ns_per_m2_yr"]
        + combined["phobos_annual_momentum_delivery_Ns_per_m2_yr"]
    )
    combined["deimos_total_annual_momentum_delivery_Ns_per_m2_yr"] = (
        combined["imem_annual_momentum_delivery_Ns_per_m2_yr"]
        + combined["deimos_annual_momentum_delivery_Ns_per_m2_yr"]
    )
    combined["total_annual_momentum_delivery_Ns_per_m2_yr"] = (
        combined["imem_annual_momentum_delivery_Ns_per_m2_yr"]
        + combined["ring_annual_momentum_delivery_Ns_per_m2_yr"]
    )

    combined["ring_annual_kinetic_energy_delivery_J_per_m2_yr"] = (
        combined["phobos_annual_kinetic_energy_delivery_J_per_m2_yr"]
        + combined["deimos_annual_kinetic_energy_delivery_J_per_m2_yr"]
    )

    combined["phobos_total_annual_kinetic_energy_delivery_J_per_m2_yr"] = (
        combined["imem_annual_kinetic_energy_delivery_J_per_m2_yr"]
        + combined["phobos_annual_kinetic_energy_delivery_J_per_m2_yr"]
    )
    combined["deimos_total_annual_kinetic_energy_delivery_J_per_m2_yr"] = (
        combined["imem_annual_kinetic_energy_delivery_J_per_m2_yr"]
        + combined["deimos_annual_kinetic_energy_delivery_J_per_m2_yr"]
    )
    combined["total_annual_kinetic_energy_delivery_J_per_m2_yr"] = (
        combined["imem_annual_kinetic_energy_delivery_J_per_m2_yr"]
        + combined["ring_annual_kinetic_energy_delivery_J_per_m2_yr"]
    )

    ring_flux = combined["ring_flux_per_mass_bin_per_m2_yr"].to_numpy(float)
    total_flux = combined[
        "total_flux_with_rings_per_mass_bin_per_m2_yr"
    ].to_numpy(float)

    ring_p_delivery = combined[
        "ring_annual_momentum_delivery_Ns_per_m2_yr"
    ].to_numpy(float)
    total_p_delivery = combined[
        "total_annual_momentum_delivery_Ns_per_m2_yr"
    ].to_numpy(float)

    ring_e_delivery = combined[
        "ring_annual_kinetic_energy_delivery_J_per_m2_yr"
    ].to_numpy(float)
    total_e_delivery = combined[
        "total_annual_kinetic_energy_delivery_J_per_m2_yr"
    ].to_numpy(float)

    combined["ring_mean_incident_momentum_per_impact_Ns"] = np.divide(
        ring_p_delivery,
        ring_flux,
        out=np.full_like(ring_p_delivery, np.nan),
        where=ring_flux > 0,
    )

    combined["phobos_total_mean_incident_momentum_per_impact_Ns"] = np.divide(
        combined["phobos_total_annual_momentum_delivery_Ns_per_m2_yr"].to_numpy(float),
        combined["phobos_total_flux_per_mass_bin_per_m2_yr"].to_numpy(float),
        out=np.full_like(ring_p_delivery, np.nan),
        where=combined["phobos_total_flux_per_mass_bin_per_m2_yr"].to_numpy(float) > 0,
    )
    combined["deimos_total_mean_incident_momentum_per_impact_Ns"] = np.divide(
        combined["deimos_total_annual_momentum_delivery_Ns_per_m2_yr"].to_numpy(float),
        combined["deimos_total_flux_per_mass_bin_per_m2_yr"].to_numpy(float),
        out=np.full_like(ring_p_delivery, np.nan),
        where=combined["deimos_total_flux_per_mass_bin_per_m2_yr"].to_numpy(float) > 0,
    )
    combined["total_mean_incident_momentum_per_impact_Ns"] = np.divide(
        total_p_delivery,
        total_flux,
        out=np.full_like(total_p_delivery, np.nan),
        where=total_flux > 0,
    )

    combined["ring_mean_kinetic_energy_per_impact_J"] = np.divide(
        ring_e_delivery,
        ring_flux,
        out=np.full_like(ring_e_delivery, np.nan),
        where=ring_flux > 0,
    )

    combined["phobos_total_mean_kinetic_energy_per_impact_J"] = np.divide(
        combined["phobos_total_annual_kinetic_energy_delivery_J_per_m2_yr"].to_numpy(float),
        combined["phobos_total_flux_per_mass_bin_per_m2_yr"].to_numpy(float),
        out=np.full_like(ring_e_delivery, np.nan),
        where=combined["phobos_total_flux_per_mass_bin_per_m2_yr"].to_numpy(float) > 0,
    )
    combined["deimos_total_mean_kinetic_energy_per_impact_J"] = np.divide(
        combined["deimos_total_annual_kinetic_energy_delivery_J_per_m2_yr"].to_numpy(float),
        combined["deimos_total_flux_per_mass_bin_per_m2_yr"].to_numpy(float),
        out=np.full_like(ring_e_delivery, np.nan),
        where=combined["deimos_total_flux_per_mass_bin_per_m2_yr"].to_numpy(float) > 0,
    )
    combined["total_mean_kinetic_energy_per_impact_J"] = np.divide(
        total_e_delivery,
        total_flux,
        out=np.full_like(total_e_delivery, np.nan),
        where=total_flux > 0,
    )

    combined["expected_imem_impacts"] = (
        combined["imem_flux_per_mass_bin_per_m2_yr"]
        * args.sensor_area_m2
        * args.mission_years
    )
    combined["expected_ring_impacts"] = (
        combined["ring_flux_per_mass_bin_per_m2_yr"]
        * args.sensor_area_m2
        * args.mission_years
    )

    combined["expected_phobos_impacts"] = (
        combined["phobos_flux_per_mass_bin_per_m2_yr"]
        * args.sensor_area_m2
        * args.mission_years
    )
    combined["expected_deimos_impacts"] = (
        combined["deimos_flux_per_mass_bin_per_m2_yr"]
        * args.sensor_area_m2
        * args.mission_years
    )
    combined["expected_total_impacts_with_rings"] = (
        combined["total_flux_with_rings_per_mass_bin_per_m2_yr"]
        * args.sensor_area_m2
        * args.mission_years
    )

    combined.to_csv(
        output_dir / "sensor_oriented_imem2_with_rough_rings.csv",
        index=False,
    )
    phobos_cells.to_csv(output_dir / "phobos_size_cells.csv", index=False)
    deimos_cells.to_csv(output_dir / "deimos_size_cells.csv", index=False)

    plot_lines(
        combined,
        [
            ("imem_flux_per_mass_bin_per_m2_yr", "IMEM2 only"),
            ("phobos_flux_per_mass_bin_per_m2_yr", "Phobos only"),
            ("deimos_flux_per_mass_bin_per_m2_yr", "Deimos only"),
            ("total_flux_with_rings_per_mass_bin_per_m2_yr", "Total"),
        ],
        r"Impact flux (m$^{-2}$ yr$^{-1}$)",
        "Impact flux",
        output_dir / "01_flux_per_mass_bin.png",
    )

    plot_lines(
        combined,
        [
            ("imem_mean_incident_momentum_per_impact_Ns", "IMEM2 only"),
            ("phobos_mean_incident_momentum_per_impact_Ns", "Phobos only"),
            ("deimos_mean_incident_momentum_per_impact_Ns", "Deimos only"),
            ("total_mean_incident_momentum_per_impact_Ns", "Total"),
        ],
        "Mean incident momentum per impact (N s)",
        "Per-particle incident momentum",
        output_dir / "02_mean_momentum_per_impact.png",
    )

    plot_lines(
        combined,
        [
            ("imem_mean_kinetic_energy_per_impact_J", "IMEM2 only"),
            ("phobos_mean_kinetic_energy_per_impact_J", "Phobos only"),
            ("deimos_mean_kinetic_energy_per_impact_J", "Deimos only"),
            ("total_mean_kinetic_energy_per_impact_J", "Total"),
        ],
        "Mean kinetic energy per impact (J)",
        "Per-particle kinetic energy",
        output_dir / "03_mean_kinetic_energy_per_impact.png",
    )

    plot_lines(
        combined,
        [
            ("expected_imem_impacts", "IMEM2 only"),
            ("expected_phobos_impacts", "Phobos only"),
            ("expected_deimos_impacts", "Deimos only"),
            ("expected_total_impacts_with_rings", "Total"),
        ],
        "Expected impacts",
        (
            f"Expected impacts for {args.sensor_area_m2:g} m² over "
            f"{args.mission_years:g} yr"
        ),
        output_dir / "04_expected_impacts_for_sensor.png",
    )

    plot_lines(
        combined,
        [
            ("imem_annual_momentum_delivery_Ns_per_m2_yr", "IMEM2 only"),
            ("phobos_annual_momentum_delivery_Ns_per_m2_yr", "Phobos only"),
            ("deimos_annual_momentum_delivery_Ns_per_m2_yr", "Deimos only"),
            ("total_annual_momentum_delivery_Ns_per_m2_yr", "Total"),
        ],
        r"Annual incident momentum delivery (N s m$^{-2}$ yr$^{-1}$)",
        "Annual momentum delivery",
        output_dir / "05_annual_momentum_delivery.png",
    )

    plot_lines(
        combined,
        [
            ("imem_annual_kinetic_energy_delivery_J_per_m2_yr", "IMEM2 only"),
            ("phobos_annual_kinetic_energy_delivery_J_per_m2_yr", "Phobos only"),
            ("deimos_annual_kinetic_energy_delivery_J_per_m2_yr", "Deimos only"),
            ("total_annual_kinetic_energy_delivery_J_per_m2_yr", "Total"),
        ],
        r"Annual kinetic-energy delivery (J m$^{-2}$ yr$^{-1}$)",
        "Annual kinetic-energy delivery",
        output_dir / "06_annual_kinetic_energy_delivery.png",
    )

    readme = f"""Sensor-oriented rough ring scenario

Density: {args.rho_g_cm3:g} g/cm^3
Phobos total scenario flux: {args.phobos_total_flux:g} m^-2 yr^-1
Deimos total scenario flux: {args.deimos_total_flux:g} m^-2 yr^-1
Phobos representative speed: {args.phobos_speed_km_s:g} km/s
Deimos representative speed: {args.deimos_speed_km_s:g} km/s
Detector area: {args.sensor_area_m2:g} m^2
Mission duration: {args.mission_years:g} yr

Use plots 01–04 for sensor sizing and sensitivity.
Plots 05–06 are cumulative annual environment-delivery metrics, not
single-impact sensor signals.
All plots show four separate series: IMEM2 only, Phobos only, Deimos only, and Total.

The incident momentum p=m v is not automatically equal to momentum
transferred to a penetrated detector film.
"""
    (output_dir / "README.txt").write_text(readme, encoding="utf-8")

    print(f"Results written to: {output_dir}")


if __name__ == "__main__":
    main()
