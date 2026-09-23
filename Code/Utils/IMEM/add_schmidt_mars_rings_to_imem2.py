#!/usr/bin/env python3
"""
Add Juergen Schmidt's author-provided Phobos/Deimos circum-Martian dust grids
into the IMEM2 mass-spectrum products.

This replaces the earlier *rough* ring treatment that used hand-digitized size
curves plus user-set annual Phobos/Deimos flux normalisations.

What is now taken directly from Schmidt's data
-----------------------------------------------
* 3-D cumulative number density n(>s0) [m^-3] on the cylindrical Mars grid.
* Separate Phobos and Deimos populations.
* Differential source-size slopes -3.4 and -3.7 (file labels pow:3.4/3.7).
* Inertial-frame and solar-fixed-frame density fields.
* Cumulative size thresholds from 0.5 micron upward.

What is still an assumption in this script
------------------------------------------
The files contain number density, not particle velocity vectors. To turn number
 density into impact flux and momentum/energy delivery, the default kinematic
model assumes circum-Martian grains move prograde with the local circular
Keplerian speed in the Mars equatorial plane. This reproduces the expected
sub-km/s relative-speed scale for a ~20 deg inclined spacecraft orbit, but it
is NOT a replacement for author-provided grain velocity distributions.

The script can read either:
1) an unpacked ToMaxVovk directory containing grid.save and *.cumufile files; or
2) ToMaxVovk.tgz directly. Reading the tarball directly is useful on native
   Windows because the archive filenames contain ':' characters, which NTFS
   does not normally permit in filenames.

Primary outputs for each selected power-law slope
-------------------------------------------------
* schmidt_cumulative_orbit_sampling.csv
    Directly sampled cumulative densities and orbit-averaged fluxes at each
    Schmidt size threshold.
* schmidt_cumulative_directional_flux.csv
    Direct face-resolved fluxes at every author-provided cumulative size threshold.
* schmidt_native_size_bins.csv / schmidt_directional_native_bins.csv
    Optional differential products. They are created only when
    --differential-mode is 'conservative' or 'raw'.
* schmidt_binned_to_imem_mass_bins.csv and IMEM2 combined products
    Optional products requiring a differential reconstruction.

Important interpretation notes
------------------------------
* s0 is treated as particle radius, consistent with Schmidt's description of
  grains larger than the stated s0 value.
* The *.cumufile values are treated exactly as J. Schmidt described them: each
  file is a cumulative number-density field n(>s0). The direct cumulative outputs
  do not require any differencing and are therefore the preferred products.
* In the supplied archive the cumulative fields are not strictly monotonic with
  s0 at every location (and, for some cases, not even after orbit averaging).
  Therefore the script does NOT derive differential size bins by default.
* --differential-mode conservative enforces a non-increasing cumulative sequence
  before differencing; --differential-mode raw differences the files directly and
  clips negative bins to zero. Both are diagnostic reconstructions, not new
  author-provided data.
* The open tail above the largest available threshold is retained only in the
  cumulative-threshold products.
* The orbital node/orientation can matter because the grids are not perfectly
  axisymmetric. By default the script averages 36 evenly spaced RAAN values.
  Set --raan-samples 1 and --raan-deg <value> for a specific orbit plane.
* With SolarFixedFrame, RAAN is interpreted relative to the +x subsolar axis.
  With InertialFrame, it is relative to the supplied inertial grid axes.

Dependencies: numpy, pandas, matplotlib, scipy
"""

from __future__ import annotations

import argparse
import math
import re
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.io import readsav


# -----------------------------------------------------------------------------
# Constants / defaults
# -----------------------------------------------------------------------------

MARS_RADIUS_KM = 3389.5
MARS_MU_M3_S2 = 4.282837e13
SECONDS_PER_YEAR = 365.25 * 86400.0

DEFAULT_SCHMIDT_PATH = Path(
    r"C:\Users\maxiv\Documents\UWO\Papers\0.5)METEORCAM-Strawman\Strawman\ToMaxVovk"
)
DEFAULT_IMEM2_PATH = Path(
    r"C:\Users\maxiv\Documents\UWO\Papers\0.5)METEORCAM-Strawman\Strawman\combined_IMEM2_spectra\imem2_mass_spectra_box_average.csv"
)

# Match both original archive names containing ':' and Windows-sanitized names
# where ':' may have been replaced by '_' or '='.
CUMU_RE = re.compile(
    r"^(PHOBOS|DEIMOS)_(InertialFrame|SolarFixedFrame)_"
    r"pow[:=_-]([0-9.+eE-]+)_s0[:=_-]([0-9.+eE-]+)\.cumufile$"
)

FACE_ORDER = ["+x ram", "-y", "-x wake", "+y", "+z", "-z"]
FACE_TO_ELEMENT = {
    "+x ram": 1,
    "-y": 2,
    "-x wake": 3,
    "+y": 4,
    "+z": 5,
    "-z": 6,
}


# -----------------------------------------------------------------------------
# Data-source handling: unpacked directory OR tgz directly
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class DensityFile:
    source: str
    frame: str
    power: float
    s0_m: float
    member_name: str


class SchmidtDataSource:
    """Access Schmidt grid.save / cumufiles from a directory or .tgz archive."""

    def __init__(self, path: Path):
        path = path.expanduser()

        # If a directory was supplied, accept either unpacked data or a tarball
        # sitting inside that directory.
        if path.is_dir():
            if (path / "grid.save").exists():
                self.kind = "directory"
                self.path = path
            else:
                candidates = sorted(path.glob("ToMaxVovk.tgz")) + sorted(path.glob("*.tgz"))
                if not candidates:
                    raise FileNotFoundError(
                        f"{path} does not contain grid.save and no .tgz archive was found."
                    )
                self.kind = "archive"
                self.path = candidates[0]
        elif path.is_file() and path.suffix.lower() in {".tgz", ".gz"}:
            self.kind = "archive"
            self.path = path
        else:
            raise FileNotFoundError(
                f"Schmidt data path does not exist or is not a supported directory/archive: {path}"
            )

        self._tmp: tempfile.TemporaryDirectory[str] | None = None
        self._tar: tarfile.TarFile | None = None
        self._names: list[str] = []

        if self.kind == "archive":
            self._tmp = tempfile.TemporaryDirectory(prefix="schmidt_mars_dust_")
            self._tar = tarfile.open(self.path, "r:gz")
            self._names = [m.name for m in self._tar.getmembers() if m.isfile()]
        else:
            self._names = [p.name for p in self.path.iterdir() if p.is_file()]

    def close(self) -> None:
        if self._tar is not None:
            self._tar.close()
        if self._tmp is not None:
            self._tmp.cleanup()

    def __enter__(self) -> "SchmidtDataSource":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @staticmethod
    def _basename(name: str) -> str:
        return Path(name).name

    def list_density_files(self) -> list[DensityFile]:
        out: list[DensityFile] = []
        for member in self._names:
            base = self._basename(member)
            match = CUMU_RE.match(base)
            if not match:
                continue
            out.append(
                DensityFile(
                    source=match.group(1),
                    frame=match.group(2),
                    power=float(match.group(3)),
                    s0_m=float(match.group(4)),
                    member_name=member,
                )
            )
        return sorted(out, key=lambda x: (x.source, x.frame, x.power, x.s0_m))

    def _find_member(self, basename: str) -> str:
        matches = [name for name in self._names if self._basename(name) == basename]
        if not matches:
            raise FileNotFoundError(f"Could not find {basename} in {self.path}")
        # Ignore AppleDouble metadata entries when both exist.
        matches = [m for m in matches if not self._basename(m).startswith("._")] or matches
        return matches[0]

    def local_path(self, member_name: str) -> Path:
        """Return a local path readable by scipy.io.readsav."""
        if self.kind == "directory":
            # member_name may already be just the basename.
            p = self.path / self._basename(member_name)
            if p.exists():
                return p
            raise FileNotFoundError(p)

        assert self._tar is not None and self._tmp is not None
        member = self._tar.getmember(member_name)
        extracted = self._tar.extractfile(member)
        if extracted is None:
            raise FileNotFoundError(member_name)

        # Sanitize ':' so the temporary name is also portable to Windows.
        safe = self._basename(member_name).replace(":", "_")
        out = Path(self._tmp.name) / safe
        if not out.exists() or out.stat().st_size != member.size:
            with out.open("wb") as f:
                f.write(extracted.read())
        return out

    def grid_path(self) -> Path:
        return self.local_path(self._find_member("grid.save"))


# -----------------------------------------------------------------------------
# Grid reading / interpolation
# -----------------------------------------------------------------------------

@dataclass
class SchmidtGrid:
    rmid_m: np.ndarray
    phimid_rad: np.ndarray
    zmid_m: np.ndarray
    nr: int
    nphi: int
    nz: int


def load_grid(path: Path) -> SchmidtGrid:
    data = readsav(str(path), python_dict=True, verbose=False)
    if "grid" not in data:
        raise KeyError(f"'grid' structure not found in {path}")

    rec = data["grid"][0]
    return SchmidtGrid(
        rmid_m=np.asarray(rec["RMID"], dtype=float),
        phimid_rad=np.asarray(rec["PHIMID"], dtype=float),
        zmid_m=np.asarray(rec["ZMID"], dtype=float),
        nr=int(rec["NR"]),
        nphi=int(rec["NPHI"]),
        nz=int(rec["NZ"]),
    )


def load_number_density(path: Path, grid: SchmidtGrid) -> np.ndarray:
    data = readsav(str(path), python_dict=True, verbose=False)
    if "number_density" not in data:
        raise KeyError(f"'number_density' was not found in {path}")

    arr = np.asarray(data["number_density"], dtype=np.float32)
    expected = (grid.nz, grid.nphi, grid.nr)
    if arr.shape != expected:
        raise ValueError(
            f"Unexpected number-density shape in {path.name}: {arr.shape}; expected {expected}."
        )
    # Densities are physical number densities. Guard against any tiny numerical
    # negatives that may appear in transformed files.
    return np.maximum(arr, 0.0)


def make_periodic_interpolator(grid: SchmidtGrid, density: np.ndarray) -> RegularGridInterpolator:
    """
    Trilinear interpolation in (z, phi, cylindrical-r).

    The saved array order is [z, phi, r]. phi is padded periodically so that
    spacecraft samples around phi=0/2pi interpolate continuously.
    """
    phi = grid.phimid_rad
    phi_ext = np.concatenate(([phi[-1] - 2.0 * np.pi], phi, [phi[0] + 2.0 * np.pi]))
    data_ext = np.concatenate((density[:, -1:, :], density, density[:, :1, :]), axis=1)

    return RegularGridInterpolator(
        (grid.zmid_m, phi_ext, grid.rmid_m),
        data_ext,
        method="linear",
        bounds_error=False,
        fill_value=0.0,
    )


# -----------------------------------------------------------------------------
# Orbit and kinematic model
# -----------------------------------------------------------------------------

@dataclass
class OrbitSamples:
    position_m: np.ndarray
    spacecraft_velocity_m_s: np.ndarray
    relative_velocity_m_s: np.ndarray
    relative_speed_m_s: np.ndarray
    cylindrical_r_m: np.ndarray
    phi_rad: np.ndarray
    z_m: np.ndarray
    face_projected_speed_m_s: dict[str, np.ndarray]
    raan_index: np.ndarray
    raan_deg_values: np.ndarray
    phase_deg: np.ndarray


def generate_orbit_samples(
    altitude_km: float,
    inclination_deg: float,
    raan_deg: float,
    raan_samples: int,
    orbit_samples: int,
    mars_radius_km: float = MARS_RADIUS_KM,
    mars_mu_m3_s2: float = MARS_MU_M3_S2,
) -> OrbitSamples:
    if raan_samples < 1 or orbit_samples < 8:
        raise ValueError("raan_samples must be >=1 and orbit_samples must be >=8")

    radius_m = (mars_radius_km + altitude_km) * 1000.0
    inclination = math.radians(inclination_deg)
    v_sc_mag = math.sqrt(mars_mu_m3_s2 / radius_m)

    if raan_samples == 1:
        raan_values = np.array([float(raan_deg)])
    else:
        # Use the provided RAAN as the start of an evenly spaced orientation
        # ensemble. This is a sensitivity / unknown-orientation average.
        raan_values = raan_deg + np.arange(raan_samples) * (360.0 / raan_samples)

    u = np.linspace(0.0, 2.0 * np.pi, orbit_samples, endpoint=False)
    phase_deg_one = np.degrees(u)

    positions: list[np.ndarray] = []
    velocities: list[np.ndarray] = []
    raan_indices: list[np.ndarray] = []
    phase_values: list[np.ndarray] = []

    ci, si = math.cos(inclination), math.sin(inclination)
    cu, su = np.cos(u), np.sin(u)

    for idx, raan_value in enumerate(raan_values):
        O = math.radians(float(raan_value))
        cO, sO = math.cos(O), math.sin(O)

        # Circular orbit in Mars-centred inertial coordinates.
        x = radius_m * (cO * cu - sO * su * ci)
        y = radius_m * (sO * cu + cO * su * ci)
        z = radius_m * (su * si)

        vx = v_sc_mag * (-cO * su - sO * cu * ci)
        vy = v_sc_mag * (-sO * su + cO * cu * ci)
        vz = v_sc_mag * (cu * si)

        positions.append(np.column_stack((x, y, z)))
        velocities.append(np.column_stack((vx, vy, vz)))
        raan_indices.append(np.full(orbit_samples, idx, dtype=int))
        phase_values.append(phase_deg_one.copy())

    pos = np.vstack(positions)
    v_sc = np.vstack(velocities)
    raan_index = np.concatenate(raan_indices)
    phase_deg = np.concatenate(phase_values)

    r_sph = np.linalg.norm(pos, axis=1)
    r_hat = pos / r_sph[:, None]
    x_hat = v_sc / np.linalg.norm(v_sc, axis=1)[:, None]  # +x = ram
    y_hat = np.cross(r_hat, x_hat)                         # +y cross-track
    y_hat /= np.linalg.norm(y_hat, axis=1)[:, None]
    z_hat = r_hat                                          # +z = zenith

    phi = np.mod(np.arctan2(pos[:, 1], pos[:, 0]), 2.0 * np.pi)
    r_cyl = np.hypot(pos[:, 0], pos[:, 1])

    # KINEMATIC APPROXIMATION: prograde, equatorial circular grain motion.
    e_phi = np.column_stack((-np.sin(phi), np.cos(phi), np.zeros_like(phi)))
    v_dust_mag = np.sqrt(mars_mu_m3_s2 / r_sph)
    v_dust = e_phi * v_dust_mag[:, None]

    v_rel = v_dust - v_sc
    speed = np.linalg.norm(v_rel, axis=1)

    normals = {
        "+x ram": x_hat,
        "-x wake": -x_hat,
        "+y": y_hat,
        "-y": -y_hat,
        "+z": z_hat,
        "-z": -z_hat,
    }
    face_projected: dict[str, np.ndarray] = {}
    for face, normal in normals.items():
        # Outward face normal: particles hit when v_rel points into the face.
        face_projected[face] = np.maximum(0.0, -np.einsum("ij,ij->i", v_rel, normal))

    return OrbitSamples(
        position_m=pos,
        spacecraft_velocity_m_s=v_sc,
        relative_velocity_m_s=v_rel,
        relative_speed_m_s=speed,
        cylindrical_r_m=r_cyl,
        phi_rad=phi,
        z_m=pos[:, 2],
        face_projected_speed_m_s=face_projected,
        raan_index=raan_index,
        raan_deg_values=np.mod(raan_values, 360.0),
        phase_deg=phase_deg,
    )


def sample_density_on_orbit(
    density: np.ndarray,
    grid: SchmidtGrid,
    orbit: OrbitSamples,
) -> np.ndarray:
    interp = make_periodic_interpolator(grid, density)
    points = np.column_stack((orbit.z_m, orbit.phi_rad, orbit.cylindrical_r_m))
    sampled = np.asarray(interp(points), dtype=float)
    return np.maximum(sampled, 0.0)


# -----------------------------------------------------------------------------
# Size / mass conversions and flux calculations
# -----------------------------------------------------------------------------


def radius_m_to_mass_g(radius_m: float, rho_g_cm3: float) -> float:
    radius_cm = radius_m * 100.0
    return rho_g_cm3 * (4.0 / 3.0) * math.pi * radius_cm**3


def positive_for_log(series: pd.Series) -> pd.Series:
    return series.where(series > 0.0, np.nan)


def weighted_mean_speed_km_s(density: np.ndarray, transport_speed_m_s: np.ndarray) -> float:
    weight = density * transport_speed_m_s
    denom = float(np.mean(weight))
    if denom <= 0.0:
        return float("nan")
    return float(np.mean(weight * (transport_speed_m_s / 1000.0)) / denom)


def flux_products(
    density: np.ndarray,
    transport_speed_m_s: np.ndarray,
    relative_speed_m_s: np.ndarray,
    mass_g: float | None,
) -> dict[str, float]:
    """
    Convert sampled number density to orbit-averaged flux products.

    transport_speed_m_s is |v_rel| for the scalar cross-sectional flux or the
    inward normal component for a particular detector face. Particle momentum
    and kinetic energy use the magnitude |v_rel|.
    """
    number_rate = density * transport_speed_m_s             # m^-2 s^-1
    number_flux = float(np.mean(number_rate) * SECONDS_PER_YEAR)

    out = {
        "number_flux_per_m2_yr": number_flux,
        "mean_impact_speed_km_s": weighted_mean_speed_km_s(density, transport_speed_m_s),
    }

    if mass_g is None:
        out["momentum_flux_g_km_s_per_m2_yr"] = float("nan")
        out["energy_flux_J_per_m2_yr"] = float("nan")
        return out

    impact_momentum_g_km_s = mass_g * (relative_speed_m_s / 1000.0)
    impact_energy_j = 0.5 * (mass_g * 1e-3) * relative_speed_m_s**2

    out["momentum_flux_g_km_s_per_m2_yr"] = float(
        np.mean(number_rate * impact_momentum_g_km_s) * SECONDS_PER_YEAR
    )
    out["energy_flux_J_per_m2_yr"] = float(
        np.mean(number_rate * impact_energy_j) * SECONDS_PER_YEAR
    )
    return out


def build_cumulative_table(
    source: str,
    power: float,
    sampled_cumulative: dict[float, np.ndarray],
    orbit: OrbitSamples,
    rho_g_cm3: float,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []

    for s0_m in sorted(sampled_cumulative):
        density = sampled_cumulative[s0_m]
        scalar = flux_products(density, orbit.relative_speed_m_s, orbit.relative_speed_m_s, None)

        row: dict[str, float | str] = {
            "source": source,
            "power_index": -abs(power),
            "s0_m": s0_m,
            "s0_um": s0_m * 1e6,
            "threshold_mass_g": radius_m_to_mass_g(s0_m, rho_g_cm3),
            "mean_number_density_per_m3": float(np.mean(density)),
            "max_number_density_per_m3_on_sampled_orbits": float(np.max(density)),
            "cross_sectional_flux_gt_s0_per_m2_yr": scalar["number_flux_per_m2_yr"],
            "flux_weighted_mean_relative_speed_km_s": scalar["mean_impact_speed_km_s"],
        }

        for face in FACE_ORDER:
            face_result = flux_products(
                density,
                orbit.face_projected_speed_m_s[face],
                orbit.relative_speed_m_s,
                None,
            )
            safe = (
                face.replace("+", "plus_")
                .replace("-", "minus_")
                .replace(" ", "_")
            )
            row[f"{safe}_flux_gt_s0_per_m2_yr"] = face_result["number_flux_per_m2_yr"]

        rows.append(row)

    return pd.DataFrame(rows)


def prepare_cumulative_for_differencing(
    sampled_cumulative: dict[float, np.ndarray],
    mode: str,
) -> tuple[dict[float, np.ndarray], dict[str, float]]:
    """Prepare cumulative arrays for an optional differential reconstruction.

    mode='conservative' applies a cumulative minimum with increasing s0 so that
    n(>s0) cannot increase as the size threshold grows. This preserves the
    smallest-threshold density and only reduces later cumulative curves.

    mode='raw' leaves the author files unchanged; negative finite-bin
    differences are later clipped to zero.
    """
    thresholds = sorted(sampled_cumulative)
    stack = np.column_stack([sampled_cumulative[s] for s in thresholds])
    violations = stack[:, 1:] > stack[:, :-1]
    diagnostics = {
        "cumulative_monotonicity_violations": int(np.count_nonzero(violations)),
        "cumulative_monotonicity_comparisons": int(violations.size),
        "cumulative_monotonicity_violation_fraction": (
            float(np.count_nonzero(violations) / violations.size) if violations.size else 0.0
        ),
    }

    if mode == "conservative":
        stack = np.minimum.accumulate(stack, axis=1)
    elif mode != "raw":
        raise ValueError(f"Unsupported differential mode: {mode}")

    return {s: stack[:, i] for i, s in enumerate(thresholds)}, diagnostics


def build_native_size_bins(
    source: str,
    power: float,
    sampled_cumulative: dict[float, np.ndarray],
    orbit: OrbitSamples,
    rho_g_cm3: float,
    differential_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Build an optional finite-bin reconstruction from cumulative n(>s0)."""
    sampled_cumulative, monotonic_diag = prepare_cumulative_for_differencing(
        sampled_cumulative, differential_mode
    )
    thresholds = sorted(sampled_cumulative)
    rows: list[dict[str, float | str]] = []
    directional_rows: list[dict[str, float | str | int]] = []

    negative_raw_count = 0
    total_raw_count = 0

    for s_low, s_high in zip(thresholds[:-1], thresholds[1:]):
        raw = sampled_cumulative[s_low] - sampled_cumulative[s_high]
        negative_raw_count += int(np.count_nonzero(raw < 0.0))
        total_raw_count += raw.size
        density_bin = np.maximum(raw, 0.0)

        radius_mid_m = math.sqrt(s_low * s_high)
        mass_mid_g = radius_m_to_mass_g(radius_mid_m, rho_g_cm3)
        mass_low_g = radius_m_to_mass_g(s_low, rho_g_cm3)
        mass_high_g = radius_m_to_mass_g(s_high, rho_g_cm3)

        scalar = flux_products(
            density_bin,
            orbit.relative_speed_m_s,
            orbit.relative_speed_m_s,
            mass_mid_g,
        )

        rows.append(
            {
                "source": source,
                "power_index": -abs(power),
                "radius_low_um": s_low * 1e6,
                "radius_high_um": s_high * 1e6,
                "radius_mid_um": radius_mid_m * 1e6,
                "mass_low_g": mass_low_g,
                "mass_high_g": mass_high_g,
                "mass_mid_g": mass_mid_g,
                "mean_number_density_per_m3": float(np.mean(density_bin)),
                "flux_per_mass_bin_per_m2_yr": scalar["number_flux_per_m2_yr"],
                "momentum_per_mass_bin_g_km_s_per_m2_yr": scalar[
                    "momentum_flux_g_km_s_per_m2_yr"
                ],
                "kinetic_energy_per_mass_bin_J_per_m2_yr": scalar[
                    "energy_flux_J_per_m2_yr"
                ],
                "flux_weighted_mean_relative_speed_km_s": scalar[
                    "mean_impact_speed_km_s"
                ],
            }
        )

        for face in FACE_ORDER:
            face_result = flux_products(
                density_bin,
                orbit.face_projected_speed_m_s[face],
                orbit.relative_speed_m_s,
                mass_mid_g,
            )
            directional_rows.append(
                {
                    "source": source,
                    "power_index": -abs(power),
                    "element": FACE_TO_ELEMENT[face],
                    "direction": face,
                    "radius_low_um": s_low * 1e6,
                    "radius_high_um": s_high * 1e6,
                    "mass_low_g": mass_low_g,
                    "mass_high_g": mass_high_g,
                    "mass_mid_g": mass_mid_g,
                    "flux_per_mass_bin_per_m2_yr": face_result["number_flux_per_m2_yr"],
                    "momentum_per_mass_bin_g_km_s_per_m2_yr": face_result[
                        "momentum_flux_g_km_s_per_m2_yr"
                    ],
                    "kinetic_energy_per_mass_bin_J_per_m2_yr": face_result[
                        "energy_flux_J_per_m2_yr"
                    ],
                    "flux_weighted_mean_relative_speed_km_s": face_result[
                        "mean_impact_speed_km_s"
                    ],
                }
            )

    diagnostics = {
        **monotonic_diag,
        "negative_differential_samples_before_clipping": negative_raw_count,
        "total_differential_samples": total_raw_count,
        "negative_fraction_before_clipping": (
            negative_raw_count / total_raw_count if total_raw_count else 0.0
        ),
    }

    return pd.DataFrame(rows), pd.DataFrame(directional_rows), diagnostics


def assign_native_bins_to_imem(native: pd.DataFrame, imem_box: pd.DataFrame) -> pd.DataFrame:
    """
    Project native Schmidt size intervals onto the existing IMEM2 mass bins.

    Because Schmidt provides cumulative density only at discrete size thresholds,
    each finite interval is represented by its geometric-mean particle mass.
    """
    required = {"mass_low_g", "mass_high_g", "mass_mid_g"}
    missing = required.difference(imem_box.columns)
    if missing:
        raise KeyError(f"IMEM box CSV is missing columns: {sorted(missing)}")

    rows = []
    for _, mb in imem_box.iterrows():
        lo = float(mb["mass_low_g"])
        hi = float(mb["mass_high_g"])
        mid = float(mb["mass_mid_g"])
        subset = native[(native["mass_mid_g"] >= lo) & (native["mass_mid_g"] < hi)]

        row = {
            "mass_low_g": lo,
            "mass_high_g": hi,
            "mass_mid_g": mid,
        }
        if "dex_width" in imem_box.columns:
            row["dex_width"] = float(mb["dex_width"])

        for source, prefix in [("PHOBOS", "phobos"), ("DEIMOS", "deimos")]:
            s = subset[subset["source"] == source]
            row[f"{prefix}_flux_per_mass_bin_per_m2_yr"] = float(
                s["flux_per_mass_bin_per_m2_yr"].sum()
            )
            row[f"{prefix}_momentum_per_mass_bin_g_km_s_per_m2_yr"] = float(
                s["momentum_per_mass_bin_g_km_s_per_m2_yr"].sum()
            )
            row[f"{prefix}_kinetic_energy_per_mass_bin_J_per_m2_yr"] = float(
                s["kinetic_energy_per_mass_bin_J_per_m2_yr"].sum()
            )

        row["ring_total_flux_per_mass_bin_per_m2_yr"] = (
            row["phobos_flux_per_mass_bin_per_m2_yr"]
            + row["deimos_flux_per_mass_bin_per_m2_yr"]
        )
        row["ring_total_momentum_per_mass_bin_g_km_s_per_m2_yr"] = (
            row["phobos_momentum_per_mass_bin_g_km_s_per_m2_yr"]
            + row["deimos_momentum_per_mass_bin_g_km_s_per_m2_yr"]
        )
        row["ring_total_kinetic_energy_per_mass_bin_J_per_m2_yr"] = (
            row["phobos_kinetic_energy_per_mass_bin_J_per_m2_yr"]
            + row["deimos_kinetic_energy_per_mass_bin_J_per_m2_yr"]
        )
        rows.append(row)

    return pd.DataFrame(rows)


def assign_directional_native_to_imem(
    directional_native: pd.DataFrame,
    imem_directional: pd.DataFrame,
) -> pd.DataFrame:
    """Project Schmidt native directional intervals into each IMEM mass bin/face."""
    needed = {"mass_low_g", "mass_high_g", "mass_mid_g", "direction"}
    missing = needed.difference(imem_directional.columns)
    if missing:
        raise KeyError(f"IMEM directional CSV is missing columns: {sorted(missing)}")

    mass_bins = (
        imem_directional[["mass_low_g", "mass_high_g", "mass_mid_g"]]
        .drop_duplicates()
        .sort_values("mass_mid_g")
    )

    rows = []
    for _, mb in mass_bins.iterrows():
        lo, hi, mid = map(float, (mb["mass_low_g"], mb["mass_high_g"], mb["mass_mid_g"]))
        native_mass = directional_native[
            (directional_native["mass_mid_g"] >= lo)
            & (directional_native["mass_mid_g"] < hi)
        ]
        for face in FACE_ORDER:
            row: dict[str, float | str | int] = {
                "mass_low_g": lo,
                "mass_high_g": hi,
                "mass_mid_g": mid,
                "direction": face,
                "element": FACE_TO_ELEMENT[face],
            }
            f = native_mass[native_mass["direction"] == face]
            for source, prefix in [("PHOBOS", "phobos"), ("DEIMOS", "deimos")]:
                s = f[f["source"] == source]
                row[f"{prefix}_flux_per_mass_bin_per_m2_yr"] = float(
                    s["flux_per_mass_bin_per_m2_yr"].sum()
                )
                row[f"{prefix}_momentum_per_mass_bin_g_km_s_per_m2_yr"] = float(
                    s["momentum_per_mass_bin_g_km_s_per_m2_yr"].sum()
                )
                row[f"{prefix}_kinetic_energy_per_mass_bin_J_per_m2_yr"] = float(
                    s["kinetic_energy_per_mass_bin_J_per_m2_yr"].sum()
                )
            row["ring_total_flux_per_mass_bin_per_m2_yr"] = (
                row["phobos_flux_per_mass_bin_per_m2_yr"]
                + row["deimos_flux_per_mass_bin_per_m2_yr"]
            )
            row["ring_total_momentum_per_mass_bin_g_km_s_per_m2_yr"] = (
                row["phobos_momentum_per_mass_bin_g_km_s_per_m2_yr"]
                + row["deimos_momentum_per_mass_bin_g_km_s_per_m2_yr"]
            )
            row["ring_total_kinetic_energy_per_mass_bin_J_per_m2_yr"] = (
                row["phobos_kinetic_energy_per_mass_bin_J_per_m2_yr"]
                + row["deimos_kinetic_energy_per_mass_bin_J_per_m2_yr"]
            )
            rows.append(row)

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Combining with IMEM2
# -----------------------------------------------------------------------------


def combine_with_imem(imem_box: pd.DataFrame, ring_bins: pd.DataFrame) -> pd.DataFrame:
    keys = ["mass_low_g", "mass_high_g", "mass_mid_g"]
    if "dex_width" in imem_box.columns and "dex_width" in ring_bins.columns:
        keys.append("dex_width")

    combined = imem_box.merge(ring_bins, on=keys, how="left")
    for col in combined.columns:
        if col.startswith(("phobos_", "deimos_", "ring_total_")):
            combined[col] = combined[col].fillna(0.0)

    required_imem = {
        "flux_per_mass_bin_per_m2_yr",
        "momentum_per_mass_bin_g_km_s_per_m2_yr",
        "kinetic_energy_per_mass_bin_J_per_m2_yr",
    }
    missing = required_imem.difference(combined.columns)
    if missing:
        raise KeyError(f"IMEM box CSV is missing required columns: {sorted(missing)}")

    combined["flux_per_mass_bin_no_rings"] = combined["flux_per_mass_bin_per_m2_yr"]
    combined["flux_per_mass_bin_with_rings"] = (
        combined["flux_per_mass_bin_per_m2_yr"]
        + combined["ring_total_flux_per_mass_bin_per_m2_yr"]
    )
    combined["momentum_per_mass_bin_no_rings"] = combined[
        "momentum_per_mass_bin_g_km_s_per_m2_yr"
    ]
    combined["momentum_per_mass_bin_with_rings"] = (
        combined["momentum_per_mass_bin_g_km_s_per_m2_yr"]
        + combined["ring_total_momentum_per_mass_bin_g_km_s_per_m2_yr"]
    )
    combined["kinetic_energy_per_mass_bin_no_rings"] = combined[
        "kinetic_energy_per_mass_bin_J_per_m2_yr"
    ]
    combined["kinetic_energy_per_mass_bin_with_rings"] = (
        combined["kinetic_energy_per_mass_bin_J_per_m2_yr"]
        + combined["ring_total_kinetic_energy_per_mass_bin_J_per_m2_yr"]
    )
    return combined


def combine_directional_with_imem(
    imem_directional: pd.DataFrame,
    ring_directional_bins: pd.DataFrame,
) -> pd.DataFrame:
    out = imem_directional.merge(
        ring_directional_bins,
        on=["mass_low_g", "mass_high_g", "mass_mid_g", "direction"],
        how="left",
        suffixes=("", "_ring"),
    )

    # If both have element columns, keep the IMEM element and drop the duplicate.
    if "element_ring" in out.columns:
        out = out.drop(columns=["element_ring"])

    for col in out.columns:
        if col.startswith(("phobos_", "deimos_", "ring_total_")):
            out[col] = out[col].fillna(0.0)

    out["flux_per_mass_bin_with_rings"] = (
        out["flux_per_mass_bin_per_m2_yr"]
        + out["ring_total_flux_per_mass_bin_per_m2_yr"]
    )
    out["momentum_per_mass_bin_with_rings"] = (
        out["momentum_per_mass_bin_g_km_s_per_m2_yr"]
        + out["ring_total_momentum_per_mass_bin_g_km_s_per_m2_yr"]
    )
    out["kinetic_energy_per_mass_bin_with_rings"] = (
        out["kinetic_energy_per_mass_bin_J_per_m2_yr"]
        + out["ring_total_kinetic_energy_per_mass_bin_J_per_m2_yr"]
    )
    return out



# -----------------------------------------------------------------------------
# Sensor-oriented products (same quantities/plots as the previous rough script)
# -----------------------------------------------------------------------------


def add_sensor_oriented_products(
    combined: pd.DataFrame,
    sensor_area_m2: float,
    mission_years: float,
) -> pd.DataFrame:
    """Add the old sensor-oriented quantities using the Schmidt ring solution.

    The IMEM2 and Schmidt momentum columns are annual incident-momentum delivery
    per unit area. Because 1 g km/s is numerically identical to 1 N s, dividing
    annual momentum delivery by annual impact flux gives the mean incident
    momentum of one particle in g km/s (or N s).
    """
    out = combined.copy()

    # Short aliases retained to make the generated CSV easy to compare with the
    # previous sensor-oriented rough-ring script.
    out["imem_flux_per_mass_bin_per_m2_yr"] = out["flux_per_mass_bin_per_m2_yr"]
    out["imem_annual_momentum_delivery_g_km_s_per_m2_yr"] = out[
        "momentum_per_mass_bin_g_km_s_per_m2_yr"
    ]
    out["imem_annual_kinetic_energy_delivery_J_per_m2_yr"] = out[
        "kinetic_energy_per_mass_bin_J_per_m2_yr"
    ]

    out["phobos_annual_momentum_delivery_g_km_s_per_m2_yr"] = out[
        "phobos_momentum_per_mass_bin_g_km_s_per_m2_yr"
    ]
    out["deimos_annual_momentum_delivery_g_km_s_per_m2_yr"] = out[
        "deimos_momentum_per_mass_bin_g_km_s_per_m2_yr"
    ]
    out["ring_annual_momentum_delivery_g_km_s_per_m2_yr"] = out[
        "ring_total_momentum_per_mass_bin_g_km_s_per_m2_yr"
    ]

    out["phobos_annual_kinetic_energy_delivery_J_per_m2_yr"] = out[
        "phobos_kinetic_energy_per_mass_bin_J_per_m2_yr"
    ]
    out["deimos_annual_kinetic_energy_delivery_J_per_m2_yr"] = out[
        "deimos_kinetic_energy_per_mass_bin_J_per_m2_yr"
    ]
    out["ring_annual_kinetic_energy_delivery_J_per_m2_yr"] = out[
        "ring_total_kinetic_energy_per_mass_bin_J_per_m2_yr"
    ]

    out["ring_flux_per_mass_bin_per_m2_yr"] = (
        out["phobos_flux_per_mass_bin_per_m2_yr"]
        + out["deimos_flux_per_mass_bin_per_m2_yr"]
    )
    out["total_flux_with_rings_per_mass_bin_per_m2_yr"] = (
        out["imem_flux_per_mass_bin_per_m2_yr"]
        + out["ring_flux_per_mass_bin_per_m2_yr"]
    )

    out["total_annual_momentum_delivery_g_km_s_per_m2_yr"] = (
        out["imem_annual_momentum_delivery_g_km_s_per_m2_yr"]
        + out["ring_annual_momentum_delivery_g_km_s_per_m2_yr"]
    )
    out["total_annual_kinetic_energy_delivery_J_per_m2_yr"] = (
        out["imem_annual_kinetic_energy_delivery_J_per_m2_yr"]
        + out["ring_annual_kinetic_energy_delivery_J_per_m2_yr"]
    )

    def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> np.ndarray:
        num = numerator.to_numpy(float)
        den = denominator.to_numpy(float)
        return np.divide(
            num,
            den,
            out=np.full_like(num, np.nan),
            where=den > 0.0,
        )

    out["imem_mean_incident_momentum_per_impact_g_km_s"] = safe_ratio(
        out["imem_annual_momentum_delivery_g_km_s_per_m2_yr"],
        out["imem_flux_per_mass_bin_per_m2_yr"],
    )
    out["phobos_mean_incident_momentum_per_impact_g_km_s"] = safe_ratio(
        out["phobos_annual_momentum_delivery_g_km_s_per_m2_yr"],
        out["phobos_flux_per_mass_bin_per_m2_yr"],
    )
    out["deimos_mean_incident_momentum_per_impact_g_km_s"] = safe_ratio(
        out["deimos_annual_momentum_delivery_g_km_s_per_m2_yr"],
        out["deimos_flux_per_mass_bin_per_m2_yr"],
    )
    out["total_mean_incident_momentum_per_impact_g_km_s"] = safe_ratio(
        out["total_annual_momentum_delivery_g_km_s_per_m2_yr"],
        out["total_flux_with_rings_per_mass_bin_per_m2_yr"],
    )

    out["imem_mean_kinetic_energy_per_impact_J"] = safe_ratio(
        out["imem_annual_kinetic_energy_delivery_J_per_m2_yr"],
        out["imem_flux_per_mass_bin_per_m2_yr"],
    )
    out["phobos_mean_kinetic_energy_per_impact_J"] = safe_ratio(
        out["phobos_annual_kinetic_energy_delivery_J_per_m2_yr"],
        out["phobos_flux_per_mass_bin_per_m2_yr"],
    )
    out["deimos_mean_kinetic_energy_per_impact_J"] = safe_ratio(
        out["deimos_annual_kinetic_energy_delivery_J_per_m2_yr"],
        out["deimos_flux_per_mass_bin_per_m2_yr"],
    )
    out["total_mean_kinetic_energy_per_impact_J"] = safe_ratio(
        out["total_annual_kinetic_energy_delivery_J_per_m2_yr"],
        out["total_flux_with_rings_per_mass_bin_per_m2_yr"],
    )

    out["expected_imem_impacts"] = (
        out["imem_flux_per_mass_bin_per_m2_yr"] * sensor_area_m2 * mission_years
    )
    out["expected_phobos_impacts"] = (
        out["phobos_flux_per_mass_bin_per_m2_yr"] * sensor_area_m2 * mission_years
    )
    out["expected_deimos_impacts"] = (
        out["deimos_flux_per_mass_bin_per_m2_yr"] * sensor_area_m2 * mission_years
    )
    out["expected_ring_impacts"] = (
        out["ring_flux_per_mass_bin_per_m2_yr"] * sensor_area_m2 * mission_years
    )
    out["expected_total_impacts_with_rings"] = (
        out["total_flux_with_rings_per_mass_bin_per_m2_yr"]
        * sensor_area_m2
        * mission_years
    )

    # Backwards-compatible aliases used by the previous plotting code. Their
    # numerical values are the same because 1 g km/s = 1 N s.
    out["imem_mean_incident_momentum_per_impact_Ns"] = out[
        "imem_mean_incident_momentum_per_impact_g_km_s"
    ]
    out["phobos_mean_incident_momentum_per_impact_Ns"] = out[
        "phobos_mean_incident_momentum_per_impact_g_km_s"
    ]
    out["deimos_mean_incident_momentum_per_impact_Ns"] = out[
        "deimos_mean_incident_momentum_per_impact_g_km_s"
    ]
    out["total_mean_incident_momentum_per_impact_Ns"] = out[
        "total_mean_incident_momentum_per_impact_g_km_s"
    ]

    return out


def save_sensor_oriented_plots(
    combined: pd.DataFrame,
    output_dir: Path,
    sensor_area_m2: float,
    mission_years: float,
    power: float,
) -> None:
    """Reproduce the six principal plots from the previous rough-ring script."""
    suffix = f" (Schmidt q=-{power:.1f})"

    save_overlay(
        combined,
        [
            ("imem_flux_per_mass_bin_per_m2_yr", "IMEM2 only"),
            ("phobos_flux_per_mass_bin_per_m2_yr", "Phobos only"),
            ("deimos_flux_per_mass_bin_per_m2_yr", "Deimos only"),
            ("total_flux_with_rings_per_mass_bin_per_m2_yr", "Total"),
        ],
        r"Impact flux (m$^{-2}$ yr$^{-1}$)",
        "Impact flux" + suffix,
        output_dir / "01_flux_per_mass_bin.png",
    )

    save_overlay(
        combined,
        [
            ("imem_mean_incident_momentum_per_impact_g_km_s", "IMEM2 only"),
            ("phobos_mean_incident_momentum_per_impact_g_km_s", "Phobos only"),
            ("deimos_mean_incident_momentum_per_impact_g_km_s", "Deimos only"),
            ("total_mean_incident_momentum_per_impact_g_km_s", "Total"),
        ],
        r"Mean incident momentum per impact (g km s$^{-1}$)",
        "Per-particle incident momentum" + suffix,
        output_dir / "02_mean_momentum_per_impact.png",
    )

    save_overlay(
        combined,
        [
            ("imem_mean_kinetic_energy_per_impact_J", "IMEM2 only"),
            ("phobos_mean_kinetic_energy_per_impact_J", "Phobos only"),
            ("deimos_mean_kinetic_energy_per_impact_J", "Deimos only"),
            ("total_mean_kinetic_energy_per_impact_J", "Total"),
        ],
        "Mean kinetic energy per impact (J)",
        "Per-particle kinetic energy" + suffix,
        output_dir / "03_mean_kinetic_energy_per_impact.png",
    )

    save_overlay(
        combined,
        [
            ("expected_imem_impacts", "IMEM2 only"),
            ("expected_phobos_impacts", "Phobos only"),
            ("expected_deimos_impacts", "Deimos only"),
            ("expected_total_impacts_with_rings", "Total"),
        ],
        "Expected impacts",
        (
            f"Expected impacts for {sensor_area_m2:g} m$^2$ over "
            f"{mission_years:g} yr" + suffix
        ),
        output_dir / "04_expected_impacts_for_sensor.png",
    )

    save_overlay(
        combined,
        [
            ("imem_annual_momentum_delivery_g_km_s_per_m2_yr", "IMEM2 only"),
            ("phobos_annual_momentum_delivery_g_km_s_per_m2_yr", "Phobos only"),
            ("deimos_annual_momentum_delivery_g_km_s_per_m2_yr", "Deimos only"),
            ("total_annual_momentum_delivery_g_km_s_per_m2_yr", "Total"),
        ],
        r"Annual incident momentum delivery (g km s$^{-1}$ m$^{-2}$ yr$^{-1}$)",
        "Annual momentum delivery" + suffix,
        output_dir / "05_annual_momentum_delivery.png",
    )

    save_overlay(
        combined,
        [
            ("imem_annual_kinetic_energy_delivery_J_per_m2_yr", "IMEM2 only"),
            ("phobos_annual_kinetic_energy_delivery_J_per_m2_yr", "Phobos only"),
            ("deimos_annual_kinetic_energy_delivery_J_per_m2_yr", "Deimos only"),
            ("total_annual_kinetic_energy_delivery_J_per_m2_yr", "Total"),
        ],
        r"Annual kinetic-energy delivery (J m$^{-2}$ yr$^{-1}$)",
        "Annual kinetic-energy delivery" + suffix,
        output_dir / "06_annual_kinetic_energy_delivery.png",
    )

# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------


def save_overlay(
    data: pd.DataFrame,
    series: list[tuple[str, str]],
    ylabel: str,
    title: str,
    output: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for column, label in series:
        if column not in data.columns:
            continue
        ax.plot(data["mass_mid_g"], positive_for_log(data[column]), marker="o", label=label)
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


def save_native_plot(
    native: pd.DataFrame,
    column: str,
    ylabel: str,
    title: str,
    output: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for source, group in native.groupby("source", sort=False):
        g = group.sort_values("mass_mid_g")
        ax.plot(g["mass_mid_g"], positive_for_log(g[column]), marker="o", label=source.title())
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


def save_face_bar(directional: pd.DataFrame, output: Path) -> None:
    summary = (
        directional.groupby(["source", "direction"], as_index=False)["flux_per_mass_bin_per_m2_yr"]
        .sum()
    )
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    x = np.arange(len(FACE_ORDER))
    width = 0.36
    for i, source in enumerate(["PHOBOS", "DEIMOS"]):
        g = summary[summary["source"] == source].set_index("direction")
        values = [float(g.loc[f, "flux_per_mass_bin_per_m2_yr"]) if f in g.index else 0.0 for f in FACE_ORDER]
        ax.bar(x + (i - 0.5) * width, values, width=width, label=source.title())
    ax.set_xticks(x, FACE_ORDER, rotation=20, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel(r"Resolved finite-bin impact flux (m$^{-2}$ yr$^{-1}$)")
    ax.set_title("Schmidt circum-Martian dust: directional flux by spacecraft face")
    ax.grid(True, axis="y", which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def save_directional_combined(data: pd.DataFrame, column: str, ylabel: str, title: str, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    for direction, group in data.groupby("direction", sort=False):
        g = group.sort_values("mass_mid_g")
        ax.plot(g["mass_mid_g"], positive_for_log(g[column]), marker="o", label=direction)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Geometric-mean particle mass (g)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


# -----------------------------------------------------------------------------
# One power-law case
# -----------------------------------------------------------------------------


def run_power_case(
    data_source: SchmidtDataSource,
    grid: SchmidtGrid,
    all_files: list[DensityFile],
    power: float,
    frame: str,
    orbit: OrbitSamples,
    rho_g_cm3: float,
    differential_mode: str,
    output_dir: Path,
    imem_box: pd.DataFrame | None,
    imem_directional: pd.DataFrame | None,
    sensor_area_m2: float,
    mission_years: float,
) -> dict[str, float]:
    output_dir.mkdir(parents=True, exist_ok=True)

    cumulative_frames = []
    native_frames = []
    directional_frames = []
    diagnostic_lines = []

    for source in ["PHOBOS", "DEIMOS"]:
        source_files = [
            f for f in all_files
            if f.source == source and f.frame == frame and abs(f.power - power) < 1e-8
        ]
        if not source_files:
            raise FileNotFoundError(
                f"No {source} {frame} pow:{power} cumufiles were found in {data_source.path}"
            )

        sampled: dict[float, np.ndarray] = {}
        for info in sorted(source_files, key=lambda f: f.s0_m):
            local = data_source.local_path(info.member_name)
            density = load_number_density(local, grid)
            sampled[info.s0_m] = sample_density_on_orbit(density, grid, orbit)
            print(
                f"  sampled {source} {frame} q=-{power:.1f}, "
                f">{info.s0_m*1e6:g} um"
            )

        cumulative = build_cumulative_table(source, power, sampled, orbit, rho_g_cm3)
        cumulative_frames.append(cumulative)

        # Always diagnose whether the supplied cumulative fields are nested.
        _, direct_diag = prepare_cumulative_for_differencing(sampled, "raw")
        diagnostic_lines.append(
            f"{source}: cumulative monotonicity violations = "
            f"{direct_diag['cumulative_monotonicity_violations']} / "
            f"{direct_diag['cumulative_monotonicity_comparisons']} "
            f"({direct_diag['cumulative_monotonicity_violation_fraction']:.3%})"
        )

        if differential_mode != "none":
            native, directional, diagnostics = build_native_size_bins(
                source, power, sampled, orbit, rho_g_cm3, differential_mode
            )
            native_frames.append(native)
            directional_frames.append(directional)
            diagnostic_lines.append(
                f"{source}: negative finite-bin samples before clipping after "
                f"'{differential_mode}' preparation = "
                f"{diagnostics['negative_differential_samples_before_clipping']} / "
                f"{diagnostics['total_differential_samples']} "
                f"({diagnostics['negative_fraction_before_clipping']:.3%})"
            )

    cumulative_all = pd.concat(cumulative_frames, ignore_index=True)
    cumulative_all.to_csv(output_dir / "schmidt_cumulative_orbit_sampling.csv", index=False)

    # A compact direct directional table at every cumulative threshold.
    cumulative_direction_cols = [
        c for c in cumulative_all.columns
        if c in {
            "source", "power_index", "s0_m", "s0_um", "threshold_mass_g",
            "cross_sectional_flux_gt_s0_per_m2_yr",
            "flux_weighted_mean_relative_speed_km_s",
        } or c.endswith("_flux_gt_s0_per_m2_yr")
    ]
    cumulative_all[cumulative_direction_cols].to_csv(
        output_dir / "schmidt_cumulative_directional_flux.csv", index=False
    )

    # Direct author-provided cumulative threshold plot. This is the preferred
    # size-dependent product because no differencing is required.
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for source, group in cumulative_all.groupby("source", sort=False):
        g = group.sort_values("threshold_mass_g")
        ax.plot(
            g["threshold_mass_g"],
            positive_for_log(g["cross_sectional_flux_gt_s0_per_m2_yr"]),
            marker="o",
            label=source.title(),
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Particle-mass threshold corresponding to s0 (g)")
    ax.set_ylabel(r"Direct cumulative flux $F(>s_0)$ (m$^{-2}$ yr$^{-1}$)")
    ax.set_title(f"Schmidt cumulative circum-Martian dust flux, q=-{power:.1f}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "00_schmidt_direct_cumulative_flux.png", dpi=220)
    plt.close(fig)

    native_all = pd.DataFrame()
    directional_all = pd.DataFrame()
    if native_frames:
        native_all = pd.concat(native_frames, ignore_index=True)
        directional_all = pd.concat(directional_frames, ignore_index=True)
        native_all.to_csv(output_dir / "schmidt_native_size_bins.csv", index=False)
        directional_all.to_csv(output_dir / "schmidt_directional_native_bins.csv", index=False)

    # RAAN sensitivity for the smallest available threshold, using cumulative
    # density and the scalar cross-sectional flux definition.
    raan_rows = []
    min_s0 = float(cumulative_all["s0_m"].min())
    for source in ["PHOBOS", "DEIMOS"]:
        info = min(
            [
                f for f in all_files
                if f.source == source and f.frame == frame and abs(f.power - power) < 1e-8
            ],
            key=lambda f: f.s0_m,
        )
        density = load_number_density(data_source.local_path(info.member_name), grid)
        sampled_min = sample_density_on_orbit(density, grid, orbit)
        for idx, raan_value in enumerate(orbit.raan_deg_values):
            mask = orbit.raan_index == idx
            product = flux_products(
                sampled_min[mask],
                orbit.relative_speed_m_s[mask],
                orbit.relative_speed_m_s[mask],
                None,
            )
            raan_rows.append(
                {
                    "source": source,
                    "power_index": -abs(power),
                    "frame": frame,
                    "s0_um": min_s0 * 1e6,
                    "raan_deg": float(raan_value),
                    "mean_number_density_per_m3": float(np.mean(sampled_min[mask])),
                    "cross_sectional_flux_per_m2_yr": product["number_flux_per_m2_yr"],
                    "flux_weighted_mean_relative_speed_km_s": product[
                        "mean_impact_speed_km_s"
                    ],
                }
            )
    pd.DataFrame(raan_rows).to_csv(output_dir / "schmidt_raan_sensitivity_min_size.csv", index=False)

    if differential_mode != "none":
        save_native_plot(
            native_all,
            "flux_per_mass_bin_per_m2_yr",
            r"Impact flux in native Schmidt size bin (m$^{-2}$ yr$^{-1}$)",
            f"Schmidt circum-Martian dust number flux, q=-{power:.1f}",
            output_dir / "01_schmidt_native_number_flux.png",
        )
        save_native_plot(
            native_all,
            "momentum_per_mass_bin_g_km_s_per_m2_yr",
            r"Momentum delivery (g km s$^{-1}$ m$^{-2}$ yr$^{-1}$)",
            f"Schmidt circum-Martian dust momentum, q=-{power:.1f}",
            output_dir / "02_schmidt_native_momentum.png",
        )
        save_native_plot(
            native_all,
            "kinetic_energy_per_mass_bin_J_per_m2_yr",
            r"Kinetic-energy delivery (J m$^{-2}$ yr$^{-1}$)",
            f"Schmidt circum-Martian dust kinetic energy, q=-{power:.1f}",
            output_dir / "03_schmidt_native_energy.png",
        )
        save_face_bar(directional_all, output_dir / "04_schmidt_directional_face_flux.png")

        ring_bins = None
        if imem_box is not None:
            ring_bins = assign_native_bins_to_imem(native_all, imem_box)
            ring_bins.to_csv(output_dir / "schmidt_binned_to_imem_mass_bins.csv", index=False)
            combined = combine_with_imem(imem_box, ring_bins)
            combined = add_sensor_oriented_products(
                combined,
                sensor_area_m2=sensor_area_m2,
                mission_years=mission_years,
            )
            combined.to_csv(output_dir / "imem2_with_schmidt_rings.csv", index=False)
            # Compatibility filename analogous to the previous rough-ring script.
            combined.to_csv(output_dir / "sensor_oriented_imem2_with_schmidt_rings.csv", index=False)

            # Reproduce the six principal sensor-oriented plots from the old code,
            # now using Schmidt's local 3-D density grids and orbit-derived speed.
            save_sensor_oriented_plots(
                combined,
                output_dir=output_dir,
                sensor_area_m2=sensor_area_m2,
                mission_years=mission_years,
                power=power,
            )

            save_overlay(
                combined,
                [
                    ("flux_per_mass_bin_no_rings", "IMEM2"),
                    ("phobos_flux_per_mass_bin_per_m2_yr", "Phobos"),
                    ("deimos_flux_per_mass_bin_per_m2_yr", "Deimos"),
                    ("flux_per_mass_bin_with_rings", "IMEM2 + circum-Martian dust"),
                ],
                r"Impact flux in mass bin (m$^{-2}$ yr$^{-1}$)",
                f"IMEM2 + Schmidt circum-Martian dust, q=-{power:.1f}",
                output_dir / "10_flux_imem2_plus_schmidt.png",
            )
            save_overlay(
                combined,
                [
                    ("momentum_per_mass_bin_no_rings", "IMEM2"),
                    ("phobos_momentum_per_mass_bin_g_km_s_per_m2_yr", "Phobos"),
                    ("deimos_momentum_per_mass_bin_g_km_s_per_m2_yr", "Deimos"),
                    ("momentum_per_mass_bin_with_rings", "IMEM2 + circum-Martian dust"),
                ],
                r"Momentum delivery (g km s$^{-1}$ m$^{-2}$ yr$^{-1}$)",
                f"Momentum: IMEM2 + Schmidt circum-Martian dust, q=-{power:.1f}",
                output_dir / "11_momentum_imem2_plus_schmidt.png",
            )
            save_overlay(
                combined,
                [
                    ("kinetic_energy_per_mass_bin_no_rings", "IMEM2"),
                    ("phobos_kinetic_energy_per_mass_bin_J_per_m2_yr", "Phobos"),
                    ("deimos_kinetic_energy_per_mass_bin_J_per_m2_yr", "Deimos"),
                    ("kinetic_energy_per_mass_bin_with_rings", "IMEM2 + circum-Martian dust"),
                ],
                r"Kinetic-energy delivery (J m$^{-2}$ yr$^{-1}$)",
                f"Energy: IMEM2 + Schmidt circum-Martian dust, q=-{power:.1f}",
                output_dir / "12_energy_imem2_plus_schmidt.png",
            )

        if imem_directional is not None:
            ring_dir_bins = assign_directional_native_to_imem(directional_all, imem_directional)
            ring_dir_bins.to_csv(output_dir / "schmidt_directional_binned_to_imem_mass_bins.csv", index=False)
            combined_dir = combine_directional_with_imem(imem_directional, ring_dir_bins)
            combined_dir.to_csv(output_dir / "imem2_directional_with_schmidt_rings.csv", index=False)

            save_directional_combined(
                combined_dir,
                "flux_per_mass_bin_with_rings",
                r"Directional impact flux (m$^{-2}$ yr$^{-1}$)",
                f"Directional IMEM2 + Schmidt dust, q=-{power:.1f}",
                output_dir / "15_directional_flux_imem2_plus_schmidt.png",
            )
            save_directional_combined(
                combined_dir,
                "momentum_per_mass_bin_with_rings",
                r"Directional momentum delivery (g km s$^{-1}$ m$^{-2}$ yr$^{-1}$)",
                f"Directional momentum: IMEM2 + Schmidt dust, q=-{power:.1f}",
                output_dir / "16_directional_momentum_imem2_plus_schmidt.png",
            )

    # Useful headline values directly from the minimum cumulative threshold.
    headline = {}
    for source in ["PHOBOS", "DEIMOS"]:
        sub = cumulative_all[
            (cumulative_all["source"] == source)
            & (np.isclose(cumulative_all["s0_m"], min_s0))
        ]
        if not sub.empty:
            headline[f"{source.lower()}_flux_gt_{min_s0*1e6:g}um_per_m2_yr"] = float(
                sub.iloc[0]["cross_sectional_flux_gt_s0_per_m2_yr"]
            )
            headline[f"{source.lower()}_mean_density_gt_{min_s0*1e6:g}um_per_m3"] = float(
                sub.iloc[0]["mean_number_density_per_m3"]
            )

    readme = [
        "Schmidt circum-Martian dust integration",
        "========================================",
        f"Data source: {data_source.path}",
        f"Frame: {frame}",
        f"Initial differential size slope: -{power:.1f}",
        f"Grain density used for radius->mass conversion: {rho_g_cm3:g} g/cm^3",
        f"Orbit altitude: {args_global.altitude_km:g} km",
        f"Orbit inclination: {args_global.inclination_deg:g} deg",
        f"RAAN start: {args_global.raan_deg:g} deg",
        f"RAAN samples: {args_global.raan_samples}",
        f"Samples per orbit: {args_global.orbit_samples}",
        f"Differential reconstruction mode: {differential_mode}",
        f"Sensor area for expected-impact plot: {sensor_area_m2:g} m^2",
        f"Mission duration for expected-impact plot: {mission_years:g} yr",
        "",
        "Sensor-oriented legacy-compatible plots:",
        "- 01_flux_per_mass_bin.png",
        "- 02_mean_momentum_per_impact.png",
        "- 03_mean_kinetic_energy_per_impact.png",
        "- 04_expected_impacts_for_sensor.png",
        "- 05_annual_momentum_delivery.png",
        "- 06_annual_kinetic_energy_delivery.png",
        "",
        "Density treatment:",
        "- Schmidt cumufiles are read directly as cumulative number density n(>s0) [m^-3].",
        "- Direct cumulative outputs are preferred and require no differencing.",
        "- The supplied fields are not strictly monotonic with s0 everywhere; see diagnostics.",
        "- Differential files are generated only when explicitly requested.",
        "- The open tail above the largest available s0 remains only in cumulative outputs.",
        "",
        "Velocity caveat:",
        "- The supplied files contain number density, not grain velocity vectors.",
        "- Flux/momentum/energy therefore use a prograde equatorial circular-Keplerian",
        "  velocity approximation for the circum-Martian grains.",
        "- The number-density sampling itself is directly from Schmidt's data.",
        "",
        "Diagnostics:",
        *[f"- {line}" for line in diagnostic_lines],
    ]
    for key, value in headline.items():
        readme.append(f"- {key}: {value:.8g}")
    (output_dir / "README_schmidt_ring_integration.txt").write_text(
        "\n".join(readme), encoding="utf-8"
    )

    return headline


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample Juergen Schmidt's Phobos/Deimos 3-D cumulative density grids "
            "along the LightShip-like Mars orbit and combine them with IMEM2 spectra."
        )
    )
    parser.add_argument(
        "--schmidt-data",
        type=Path,
        default=DEFAULT_SCHMIDT_PATH,
        help=(
            "Unpacked ToMaxVovk directory or ToMaxVovk.tgz. Default: "
            + str(DEFAULT_SCHMIDT_PATH)
        ),
    )
    parser.add_argument(
        "--imem-box-csv",
        type=Path,
        default=DEFAULT_IMEM2_PATH,
        help="Optional imem2_mass_spectra_box_average.csv. Ring-only outputs are still produced without it.",
    )
    parser.add_argument(
        "--imem-directional-csv",
        type=Path,
        default=None,
        help="Optional imem2_directional_spectra.csv for direction-resolved combination.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--frame",
        choices=["InertialFrame", "SolarFixedFrame"],
        default="InertialFrame",
        help="Use the Mars-centred inertial or solar-fixed Schmidt density grid.",
    )
    parser.add_argument(
        "--size-power",
        choices=["3.4", "3.7", "both"],
        default="3.7",
        help="Initial differential size slope magnitude. '3.7' means dN/ds proportional to s^-3.7.",
    )
    parser.add_argument("--rho-g-cm3", type=float, default=2.37)
    parser.add_argument(
        "--sensor-area-m2",
        type=float,
        default=0.01,
        help="Detector sensitive area used for expected-impact plot. Default: 0.01 m^2.",
    )
    parser.add_argument(
        "--mission-years",
        type=float,
        default=1.0,
        help="Mission duration used for expected-impact plot. Default: 1 year.",
    )
    parser.add_argument("--mars-radius-km", type=float, default=MARS_RADIUS_KM)
    parser.add_argument("--mars-mu-m3-s2", type=float, default=MARS_MU_M3_S2)
    parser.add_argument("--altitude-km", type=float, default=5720.0)
    parser.add_argument("--inclination-deg", type=float, default=20.0)
    parser.add_argument(
        "--raan-deg",
        type=float,
        default=0.0,
        help="RAAN for a single-plane run, or starting RAAN for the orientation ensemble.",
    )
    parser.add_argument(
        "--raan-samples",
        type=int,
        default=36,
        help=(
            "Number of evenly spaced RAAN orientations to average. Default 36. "
            "Use 1 with --raan-deg for one specific orbit plane."
        ),
    )
    parser.add_argument(
        "--orbit-samples",
        type=int,
        default=1440,
        help="Uniform samples per circular orbit. Default 1440 (0.25 deg spacing).",
    )
    parser.add_argument(
        "--differential-mode",
        choices=["none", "conservative", "raw"],
        default="conservative",
        help=(
            "How to reconstruct finite size bins from cumulative grids. Default 'conservative' "
            "enforces non-increasing n(>s0) before differencing so the legacy mass-bin sensor plots "
            "can be generated. Use 'none' for direct author-provided cumulative results only. "
            "'raw' differences directly and clips "
            "negative bins; use only as a diagnostic."
        ),
    )
    return parser.parse_args()


# Global is only used when writing the case README, keeping the computation
# functions independent of argparse otherwise.
args_global: argparse.Namespace


def main() -> None:
    global args_global
    args = parse_args()
    args_global = args

    imem_box = pd.read_csv(args.imem_box_csv) if args.imem_box_csv is not None else None

    if args.imem_directional_csv is not None:
        imem_directional = pd.read_csv(args.imem_directional_csv)
    elif args.imem_box_csv is not None:
        auto_directional = args.imem_box_csv.expanduser().resolve().with_name(
            "imem2_directional_spectra.csv"
        )
        imem_directional = pd.read_csv(auto_directional) if auto_directional.exists() else None
    else:
        imem_directional = None

    if args.output_dir is not None:
        root_output = args.output_dir.expanduser().resolve()
    elif args.imem_box_csv is not None:
        root_output = args.imem_box_csv.expanduser().resolve().parent / "with_schmidt_circum_mars_dust"
    else:
        # Path.resolve() on a Windows-style path while running elsewhere can be
        # misleading, so use cwd for the ring-only fallback.
        root_output = Path.cwd() / "schmidt_circum_mars_dust_products"
    root_output.mkdir(parents=True, exist_ok=True)

    powers = [3.4, 3.7] if args.size_power == "both" else [float(args.size_power)]

    if args.differential_mode == "none" and (imem_box is not None or imem_directional is not None):
        print(
            "NOTE: --differential-mode none selected. Direct Schmidt cumulative products will be "
            "generated, but per-mass-bin IMEM2 merging will be skipped. Use 'conservative' only "
            "after accepting the monotonic reconstruction assumption."
        )

    print("Generating spacecraft-orbit samples...")
    orbit = generate_orbit_samples(
        altitude_km=args.altitude_km,
        inclination_deg=args.inclination_deg,
        raan_deg=args.raan_deg,
        raan_samples=args.raan_samples,
        orbit_samples=args.orbit_samples,
        mars_radius_km=args.mars_radius_km,
        mars_mu_m3_s2=args.mars_mu_m3_s2,
    )
    print(
        f"Relative-speed range in kinematic model: "
        f"{orbit.relative_speed_m_s.min()/1000:.4f}--"
        f"{orbit.relative_speed_m_s.max()/1000:.4f} km/s"
    )

    headline_rows = []
    with SchmidtDataSource(args.schmidt_data) as data_source:
        print(f"Reading Schmidt data from: {data_source.path}")
        grid = load_grid(data_source.grid_path())
        files = data_source.list_density_files()
        if not files:
            raise RuntimeError("No Schmidt *.cumufile density grids were discovered.")

        for power in powers:
            print(f"\nProcessing q=-{power:.1f} in {args.frame}...")
            case_dir = root_output / f"{args.frame}_power_{power:.1f}".replace(".", "p")
            headline = run_power_case(
                data_source=data_source,
                grid=grid,
                all_files=files,
                power=power,
                frame=args.frame,
                orbit=orbit,
                rho_g_cm3=args.rho_g_cm3,
                differential_mode=args.differential_mode,
                output_dir=case_dir,
                imem_box=imem_box,
                imem_directional=imem_directional,
                sensor_area_m2=args.sensor_area_m2,
                mission_years=args.mission_years,
            )
            headline_rows.append(
                {
                    "frame": args.frame,
                    "power_index": -power,
                    **headline,
                }
            )

    if headline_rows:
        pd.DataFrame(headline_rows).to_csv(root_output / "schmidt_case_summary.csv", index=False)

    print(f"\nFinished. Outputs written to: {root_output}")


if __name__ == "__main__":
    main()
