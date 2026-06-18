#!/usr/bin/env python3
"""
Run one fitted meteor on Earth and on Mars/custom atmosphere, using the same
pickle-reading pathway as DynNestSapl_metsim.observation_data.

Core idea
---------
1. Build obs_data from the trajectory pickle with observation_data(...). This is
   important because observation_data.load_pickle_data() already knows how to
   read the WMPL trajectory pickle with loadPickle(*os.path.split(file)), choose
   cameras, align time/lag, set fps_lum/P_0m, compute dens_co and z_c, etc.
2. Load the fitted constants from the JSON.
3. Run the Earth best-fit simulation.
4. Create a Mars/custom-atmosphere simulation by mapping Earth h_init and h_e to
   the same atmospheric density in the new atmosphere.
5. Create a second Mars/custom-atmosphere simulation where h_e is moved to the
   height where the new-atmosphere simulation reaches the same dynamic pressure
   as the Earth erosion onset.
6. Plot with essentially the same light-curve plotting block you used before.

Example
-------
python simple_mars_dyn_pressure_observation_loader.py \
    --pickle /path/EN040326_201155_trajectory.pickle \
    --json   /path/EN040326_201155_sim_fit_latest.json \
    --output-dir /path/output

Custom atmosphere coefficients can be supplied as either:
    [c0, c1, c2, ...]
or:
    {"dens_co": [c0, c1, c2, ...]}
where the coefficients are compatible with WMPL atmDensPoly(height_m, dens_co).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# Local Mars atmosphere tools used in your old script.
from Mars_AtmDens import fitAtmPoly_mars

# Make the parent folder importable, as in the original workflow.
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Keep this import because it exposes observation_data, luminosity_integration,
# runSimulation, SimulationResults, Constants, loadConstants, loadPickle, etc. in
# the same environment as the original dynesty/MetSim script.
from DynNestSapl_metsim import *  # noqa: F401,F403

from wmpl.Utils.AtmosphereDensity import atmDensPoly

try:
    from Mars_Vel import calculate_3d_intercept_speeds
except Exception:
    calculate_3d_intercept_speeds = None


MARS_RADIUS_KM = 3389.5
MARS_G0 = 3.75


def _as_numeric_array_if_possible(value: Any) -> Any:
    """Convert JSON numeric lists to numpy arrays; leave other lists as-is."""
    if isinstance(value, list):
        try:
            return np.asarray(value, dtype=float)
        except Exception:
            return value
    return value


def _load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_best_fit_constants(json_path: str | Path) -> Constants:
    """
    Load constants from a fitted JSON.

    First tries WMPL/MetSim loadConstants(), because that is the native route for
    MetSim JSON files. If the JSON is instead one of your saved
    SimulationResults dictionaries with a top-level ``const`` entry, it falls
    back to manually creating a Constants object from that dictionary.
    """
    json_path = str(json_path)

    try:
        const, _ = loadConstants(json_path)
        const.dens_co = np.asarray(const.dens_co, dtype=float)
        return const
    except Exception:
        pass

    data = _load_json(json_path)
    const_dict = data.get("const", data)
    if not isinstance(const_dict, dict):
        raise ValueError(f"Could not find a constants dictionary in {json_path!r}")

    const = Constants()
    for key, value in const_dict.items():
        setattr(const, key, _as_numeric_array_if_possible(value))

    if hasattr(const, "dens_co"):
        const.dens_co = np.asarray(const.dens_co, dtype=float)

    return const


def build_observation_from_pickle(args: argparse.Namespace) -> Any:
    """
    Build obs_data exactly through your existing observation_data class.

    This is the key change from the previous version: do not raw-open the pickle
    here and do not invent a separate adapter. observation_data.load_pickle_data()
    already uses loadPickle(*os.path.split(current_file_name)) and builds the
    camera arrays used by the plotting function.
    """
    obs_data = observation_data(
        args.pickle,
        use_all_cameras=args.use_all_cameras,
        lag_noise_prior=args.lag_noise_prior,
        lum_noise_prior=args.lum_noise_prior,
        fps_prior=args.fps_prior,
        P_0m_prior=args.P_0m_prior,
        pick_position=args.pick_position,
        prior_file_path=args.prior,
    )
    return obs_data


def load_trajectory_list_from_obs(obs_data: Any) -> list[Any]:
    """
    Load the trajectory pickle(s) using the same function as observation_data.

    This is only used for orbit/rbeg_ele/zc information, not for camera extraction.
    """
    file_names = getattr(obs_data, "file_name", [])
    if isinstance(file_names, (str, os.PathLike)):
        file_names = [file_names]

    trajectories = []
    for file_name in file_names:
        try:
            trajectories.append(loadPickle(*os.path.split(str(file_name))))
        except Exception as exc:
            print(f"Could not load trajectory metadata from {file_name!r}: {exc}")
    return trajectories


def patch_constants_from_observation(const: Constants, obs_data: Any) -> Constants:
    """Fill/override basic run settings from obs_data in the same spirit as the old code."""
    out = copy.deepcopy(const)

    # Use the Earth atmosphere and measured geometry created by observation_data.
    if hasattr(obs_data, "dens_co"):
        out.dens_co = np.asarray(obs_data.dens_co, dtype=float)
    if hasattr(obs_data, "zenith_angle"):
        out.zenith_angle = float(obs_data.zenith_angle)
    if hasattr(obs_data, "P_0m"):
        out.P_0m = float(obs_data.P_0m)

    if not hasattr(out, "dt") or out.dt is None:
        v0 = float(getattr(out, "v_init", getattr(obs_data, "v_init", 20_000.0)))
        out.dt = 0.01 if v0 < 30_000 else 0.005

    # Keep the old practical setup for termination and luminosity efficiency.
    out.disruption_on = getattr(out, "disruption_on", False)
    out.lum_eff_type = getattr(out, "lum_eff_type", 5)

    height_lum = np.asarray(obs_data.height_lum, dtype=float)
    height_lag = np.asarray(getattr(obs_data, "height_lag", height_lum), dtype=float)
    out.h_kill = min(float(np.nanmin(height_lum)), float(np.nanmin(height_lag))) - 1000.0
    if out.h_kill < 0:
        out.h_kill = 1.0

    obs_vel = np.asarray(getattr(obs_data, "velocities", [getattr(out, "v_init", 20_000.0)]), dtype=float)
    if np.nanmin(obs_vel) < float(out.v_init) - 10_000.0:
        out.v_kill = float(out.v_init) - 10_000.0
    else:
        out.v_kill = float(np.nanmin(obs_vel)) - 5000.0
    if out.v_kill < 2500.0:
        out.v_kill = 2500.0

    return out


def atmosphere_coefficients(args: argparse.Namespace) -> np.ndarray:
    """Return the new-atmosphere density polynomial, Mars by default."""
    if args.atm_coeff_json:
        data = _load_json(args.atm_coeff_json)
        coeffs = data.get("dens_co", data) if isinstance(data, dict) else data
        return np.asarray(coeffs, dtype=float)

    if args.atm_coeff_npy:
        return np.asarray(np.load(args.atm_coeff_npy), dtype=float)

    return np.asarray(
        fitAtmPoly_mars(args.atm_min_km * 1000.0, args.atm_max_km * 1000.0),
        dtype=float,
    )


def density_at_height(height_m: float, dens_co: Iterable[float]) -> float:
    return float(atmDensPoly(float(height_m), np.asarray(dens_co, dtype=float)))


def height_for_same_density(
    target_density: float,
    dens_co_new: Iterable[float],
    h_min_km: float,
    h_max_km: float,
    step_m: float,
) -> float:
    heights = np.arange(h_min_km * 1000.0, h_max_km * 1000.0 + step_m, step_m)
    densities = np.asarray([density_at_height(h, dens_co_new) for h in heights], dtype=float)
    return float(heights[np.nanargmin(np.abs(densities - target_density))])


def estimate_mars_vinit_from_trajectory(trajectories: list[Any]) -> float | None:
    """Try to reproduce the old optional Mars Vinf estimate from orbit elements."""
    if calculate_3d_intercept_speeds is None or not trajectories:
        return None

    traj = trajectories[0]
    orbit = getattr(traj, "orbit", None)
    if orbit is None:
        return None

    try:
        a_val = float(orbit.a)
        e_val = float(orbit.e)
        inclin_val = float(getattr(orbit, "i", getattr(orbit, "incl", np.nan)))
        peri_val = float(orbit.peri)
        node_val = float(orbit.node)
        _, _, vinf_mars_min_max, *_ = calculate_3d_intercept_speeds(
            a_val, e_val, inclin_val, peri_val, node_val
        )
        return float(np.nanmean(vinf_mars_min_max)) * 1000.0
    except Exception:
        return None


def build_density_mapped_planet_const(
    earth_const: Constants,
    obs_data: Any,
    trajectories: list[Any],
    dens_co_new: np.ndarray,
    args: argparse.Namespace,
) -> Constants:
    """Copy Earth best-fit constants and map heights to same density in the new atmosphere."""
    out = copy.deepcopy(earth_const)

    earth_dens_co = np.asarray(earth_const.dens_co, dtype=float)
    rho_start_earth = density_at_height(float(earth_const.h_init), earth_dens_co)
    rho_erosion_earth = density_at_height(float(earth_const.erosion_height_start), earth_dens_co)

    out.h_init = height_for_same_density(
        rho_start_earth, dens_co_new, args.atm_min_km, args.atm_max_km, args.atm_step_m
    )
    out.erosion_height_start = height_for_same_density(
        rho_erosion_earth, dens_co_new, args.atm_min_km, args.atm_max_km, args.atm_step_m
    )

    if hasattr(earth_const, "erosion_height_change"):
        try:
            rho_change_earth = density_at_height(float(earth_const.erosion_height_change), earth_dens_co)
            out.erosion_height_change = height_for_same_density(
                rho_change_earth, dens_co_new, args.atm_min_km, args.atm_max_km, args.atm_step_m
            )
        except Exception:
            pass

    out.G0 = float(args.planet_g0)
    out.r_earth = float(args.planet_radius_km) * 1000.0
    out.dens_co = np.asarray(dens_co_new, dtype=float)
    out.h_kill = float(args.h_kill_km) * 1000.0

    if args.v_init_kms is not None:
        out.v_init = float(args.v_init_kms) * 1000.0
    else:
        mars_vinit = estimate_mars_vinit_from_trajectory(trajectories)
        if mars_vinit is not None and np.isfinite(mars_vinit):
            out.v_init = mars_vinit
        # Otherwise keep fitted Earth v_init. This is safer than inventing a speed.

    out.v_kill = max(float(out.v_init) - 10_000.0, 2500.0)

    # Same z_c calculation style as the old code: average over trajectory pickles.
    zc_values = []
    for traj in trajectories:
        try:
            zc_values.append(
                zenithAngleAtSimulationBegin(out.h_init, traj.rbeg_ele, traj.orbit.zc, out.r_earth)
            )
        except Exception:
            pass
    if zc_values:
        out.zenith_angle = float(np.nanmean(zc_values))
    elif hasattr(obs_data, "zenith_angle"):
        out.zenith_angle = float(obs_data.zenith_angle)

    return out


def run_model_raw(const: Constants) -> SimulationResults:
    """Run MetSim without luminosity integration; the plotting block does that."""
    frag_main, results_list, wake_results = runSimulation(const, compute_wake=False)
    return SimulationResults(const, frag_main, results_list, wake_results)


def nearest_height_for_dyn_press(sim_result: SimulationResults, target_dyn_press_pa: float) -> float:
    """Return simulated height where dynamic pressure is closest to target_dyn_press_pa."""
    dyn = np.asarray(sim_result.leading_frag_dyn_press_arr, dtype=float)
    heights = np.asarray(sim_result.leading_frag_height_arr, dtype=float)
    n = min(len(dyn), len(heights))
    dyn = dyn[:n]
    heights = heights[:n]
    good = np.isfinite(dyn) & np.isfinite(heights)
    if not np.any(good):
        raise ValueError("No finite dynamic-pressure/height values were produced by the simulation.")
    try:
        idx_good = np.nanargmin(np.abs(dyn[good] - target_dyn_press_pa))
    except Exception:
        idx_good = 0 # raise ValueError("Could not find height for target dynamic pressure.")
    return float(heights[good][idx_good])


def dyn_press_at_height(sim_result: SimulationResults, height_m: float) -> float:
    dyn = np.asarray(sim_result.leading_frag_dyn_press_arr, dtype=float)
    heights = np.asarray(sim_result.leading_frag_height_arr, dtype=float)
    n = min(len(dyn), len(heights))
    dyn = dyn[:n]
    heights = heights[:n]
    good = np.isfinite(dyn) & np.isfinite(heights)
    if not np.any(good):
        raise ValueError("No finite dynamic-pressure/height values were produced by the simulation.")
    idx_good = np.nanargmin(np.abs(heights[good] - height_m))
    return float(dyn[good][idx_good])


def build_dynamic_pressure_trigger_const(
    best_guess_obj_plot: SimulationResults,
    best_guess_obj_plot_mars: SimulationResults,
    best_guess_cost_mars: Constants,
) -> tuple[Constants, float, float | None, float | None]:
    """Move Mars/custom erosion heights to same p_dyn as the Earth simulation."""
    best_guess_cost_mars_dyn_press = copy.deepcopy(best_guess_cost_mars)

    erosion_beg_dyn_press = getattr(
        best_guess_obj_plot.const,
        "erosion_beg_dyn_press",
        dyn_press_at_height(best_guess_obj_plot, best_guess_obj_plot.const.erosion_height_start),
    )

    heightsame_dynpress_mars = nearest_height_for_dyn_press(best_guess_obj_plot_mars, erosion_beg_dyn_press)
    best_guess_cost_mars_dyn_press.erosion_height_start = heightsame_dynpress_mars

    heightsame_dynpress_change_mars = None
    erosion_beg_dyn_press_change = None
    if hasattr(best_guess_obj_plot.const, "erosion_height_change") and hasattr(best_guess_cost_mars_dyn_press, "erosion_height_change"):
        try:
            erosion_beg_dyn_press_change = dyn_press_at_height(
                best_guess_obj_plot,
                float(best_guess_obj_plot.const.erosion_height_change),
            )
            heightsame_dynpress_change_mars = nearest_height_for_dyn_press(
                best_guess_obj_plot_mars,
                erosion_beg_dyn_press_change,
            )
            best_guess_cost_mars_dyn_press.erosion_height_change = heightsame_dynpress_change_mars
        except Exception as exc:
            print(f"Could not set p_dyn erosion_height_change: {exc}")

    return (
        best_guess_cost_mars_dyn_press,
        heightsame_dynpress_mars,
        heightsame_dynpress_change_mars,
        erosion_beg_dyn_press_change,
    )


def maybe_integrate_luminosity(sim: SimulationResults, obs_data: Any) -> None:
    """Exactly the luminosity integration condition used in the old plotting block."""
    if (1.0 / obs_data.fps_lum) > sim.const.dt:
        sim.luminosity_arr, sim.abs_magnitude = luminosity_integration(
            sim.time_arr,
            sim.time_arr,
            sim.luminosity_arr,
            sim.const.dt,
            obs_data.fps_lum,
            obs_data.P_0m,
        )


def plot_lightcurve_earth_vs_mars_dyn_pressure(
    obs_data: Any,
    best_guess_obj_plot: SimulationResults,
    best_guess_obj_plot_mars: SimulationResults,
    best_guess_obj_plot_mars_dyn_press: SimulationResults,
    heightsame_dynpress_mars: float,
    heightsame_dynpress_change_mars: float | None,
    output_dir: str | Path,
    base_name: str,
) -> str:
    """
    Light-curve plot kept close to your original block.

    The only simplification is that the energy and single-body curves are removed;
    this script only compares Earth, Mars/custom same-density mapping, and
    Mars/custom same-dynamic-pressure trigger.
    """
    output_dir = str(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    flag_total_rho = hasattr(best_guess_obj_plot.const, "erosion_height_change")

    fig, ax = plt.subplots(figsize=(6, 6))
    station_colors = {}
    cmap = plt.get_cmap("tab10")

    # ABS MAGNITUDE: detected Earth camera data from obs_data.
    for station in np.unique(obs_data.stations_lum):
        mask = obs_data.stations_lum == station
        if station not in station_colors:
            station_colors[station] = cmap(len(station_colors) % 10)
        ax.plot(
            obs_data.absolute_magnitudes[mask],
            obs_data.height_lum[mask] / 1000,
            "x--",
            color=station_colors[station],
            label=station,
        )

    y_min = ax.get_ylim()[0]
    y_max = ax.get_ylim()[1]

    maybe_integrate_luminosity(best_guess_obj_plot, obs_data)

    ax.plot(
        best_guess_obj_plot.abs_magnitude,
        best_guess_obj_plot.leading_frag_height_arr / 1000,
        color="k",
        label="Best Fit Simulation",
    )
    ax.set_ylabel("Height [km]", fontsize=15)
    ax.set_xlabel("Abs.Mag [-]", fontsize=15)
    # ax.axhline(
    #     y=best_guess_obj_plot.const.erosion_height_start / 1000,
    #     color="gray",
    #     linestyle="--",
    #     label="Erosion Height Start $h_{e}$",
    # )
    # if flag_total_rho:
    #     ax.axhline(
    #         y=best_guess_obj_plot.const.erosion_height_change / 1000,
    #         color="gray",
    #         linestyle="-.",
    #         label="Erosion Height Change $h_{e2}$",
    #     )

    maybe_integrate_luminosity(best_guess_obj_plot_mars, obs_data)
    maybe_integrate_luminosity(best_guess_obj_plot_mars_dyn_press, obs_data)

    # ax.plot(
    #     best_guess_obj_plot_mars.abs_magnitude,
    #     best_guess_obj_plot_mars.leading_frag_height_arr / 1000,
    #     color="tab:purple",
    #     label="Best Fit Simulation (Mars/custom same $\\rho$)",
    # )
    # ax.axhline(
    #     y=best_guess_obj_plot_mars.const.erosion_height_start / 1000,
    #     color="tab:purple",
    #     linestyle="--",
    # )

    ax.plot(
        best_guess_obj_plot_mars_dyn_press.abs_magnitude,
        best_guess_obj_plot_mars_dyn_press.leading_frag_height_arr / 1000,
        color="red",
        label="Mars Meteor",
    )

    # ax.plot(
    #     best_guess_obj_plot_mars_dyn_press.abs_magnitude,
    #     best_guess_obj_plot_mars_dyn_press.leading_frag_height_arr / 1000,
    #     color="tab:brown",
    #     label="Best Fit Simulation (Mars/custom same $p_{dyn}$)",
    # )
    # ax.axhline(y=heightsame_dynpress_mars / 1000, color="tab:brown", linestyle="--")

    # if flag_total_rho:
    #     ax.axhline(
    #         y=best_guess_obj_plot_mars.const.erosion_height_change / 1000,
    #         color="tab:purple",
    #         linestyle="-.",
    #     )
    #     if heightsame_dynpress_change_mars is not None:
    #         ax.axhline(y=heightsame_dynpress_change_mars / 1000, color="tab:brown", linestyle="-.")

    ax.set_xlabel("Abs.Mag [-]", fontsize=15)
    ax.grid()

    x_min = ax.get_xlim()[0]
    x_max = 8
    ax.set_xlim(x_max, x_min)

    new_ax_min = np.min([y_min, ax.get_ylim()[0]])
    new_ax_max = np.min([y_max, ax.get_ylim()[1]])
    new_ax_max = np.max([best_guess_obj_plot.const.erosion_height_start / 1000 + 2, new_ax_max])
    ax.set_ylim(new_ax_min, new_ax_max)

    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()

    output_path = os.path.join(output_dir, base_name + "_Lightcurve_Earth_vs_Mars.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path



def meteor_abs_magnitude_to_apparent(abs_mag, distance_m):
    """Convert meteor absolute magnitude at 100 km to apparent magnitude at distance_m."""
    abs_mag = np.asarray(abs_mag, dtype=float)
    distance_m = np.asarray(distance_m, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        # Meteor absolute magnitude is conventionally referenced to 100 km.
        apparent_mag = abs_mag + 5.0*np.log10(distance_m/100000.0)
    return apparent_mag


def extract_simulation_segment(sim_result: SimulationResults) -> dict:
    """
    Return all finite Mars/custom simulation points.

    Detection is intentionally NOT applied here because apparent magnitude
    depends on the observer range, which is only known after the 3D geometry is built.
    """
    abs_mag = np.asarray(sim_result.abs_magnitude, dtype=float)
    height_m = np.asarray(sim_result.leading_frag_height_arr, dtype=float)
    length_m = np.asarray(getattr(sim_result, "leading_frag_length_arr", np.arange(len(abs_mag), dtype=float)), dtype=float)
    time_s = np.asarray(getattr(sim_result, "time_arr", np.arange(len(abs_mag), dtype=float)), dtype=float)

    n = min(len(abs_mag), len(height_m), len(length_m), len(time_s))
    abs_mag = abs_mag[:n]
    height_m = height_m[:n]
    length_m = length_m[:n]
    time_s = time_s[:n]

    good = np.isfinite(abs_mag) & np.isfinite(height_m) & np.isfinite(length_m) & np.isfinite(time_s)
    if not np.any(good):
        raise ValueError("No finite points found in the Mars/custom simulation.")

    return {
        "abs_mag": abs_mag[good],
        "height_m": height_m[good],
        "length_m": length_m[good],
        "time_s": time_s[good],
    }


# Backward-compatible wrapper. It no longer applies the final detection cut;
# detection is done after apparent magnitudes are computed in geometry.
def extract_detected_segment_from_peak(sim_result: SimulationResults, det_mag_cut: float = 2.5, from_peak_only: bool = True):
    return extract_simulation_segment(sim_result)

def local_basis_from_central_angle(central_angle_deg: float):
    """Local normal/east/north basis at a point on Mars defined in the x-y plane."""
    alpha = np.deg2rad(float(central_angle_deg))
    n = np.array([np.cos(alpha), np.sin(alpha), 0.0], dtype=float)
    east = np.array([-np.sin(alpha), np.cos(alpha), 0.0], dtype=float)
    north = np.array([0.0, 0.0, 1.0], dtype=float)
    return n, east, north


def build_detected_track_geometry(
    detected_segment: dict,
    zenith_angle_rad: float,
    planet_radius_km: float = MARS_RADIUS_KM,
    camera_altitude_km: float = 5720.0,
    central_angle_deg: float = 20.0,
    traj_azimuth_deg: float = 30.0,
    selected_altitude_km: float | None = None,
):
    """
    Build the full 3D meteor geometry and compute apparent magnitude at every point.

    The returned geometry is not yet detection-filtered. Use
    apply_apparent_detection_and_camera_sampling() next.
    """
    R_km = float(planet_radius_km)
    cam = np.array([R_km + float(camera_altitude_km), 0.0, 0.0], dtype=float)

    h_km = np.asarray(detected_segment["height_m"], dtype=float)/1000.0
    abs_mag = np.asarray(detected_segment["abs_mag"], dtype=float)
    length_km = np.asarray(detected_segment["length_m"], dtype=float)/1000.0
    time_s = np.asarray(detected_segment["time_s"], dtype=float)

    npts = min(len(h_km), len(abs_mag), len(length_km), len(time_s))
    h_km = h_km[:npts]
    abs_mag = abs_mag[:npts]
    length_km = length_km[:npts]
    time_s = time_s[:npts]

    good = np.isfinite(h_km) & np.isfinite(abs_mag) & np.isfinite(length_km) & np.isfinite(time_s)
    h_km = h_km[good]
    abs_mag = abs_mag[good]
    length_km = length_km[good]
    time_s = time_s[good]
    if len(abs_mag) == 0:
        raise ValueError("No finite points available to build the Mars geometry.")

    # Anchor the synthetic 3D track at a selected altitude. If none is supplied,
    # use the absolute-magnitude peak as the first reasonable anchor; the final
    # detected peak is recomputed later from apparent magnitude.
    abs_peak_idx = int(np.nanargmin(abs_mag))
    if selected_altitude_km is None:
        selected_idx = abs_peak_idx
        selected_altitude_km = float(h_km[selected_idx])
    else:
        selected_idx = int(np.nanargmin(np.abs(h_km - float(selected_altitude_km))))
        selected_altitude_km = float(h_km[selected_idx])

    n, east, north = local_basis_from_central_angle(central_angle_deg)
    az = np.deg2rad(float(traj_azimuth_deg))
    tang = np.cos(az)*east + np.sin(az)*north
    tang = tang/np.linalg.norm(tang)

    zc = float(zenith_angle_rad)
    dir_down = -np.cos(zc)*n + np.sin(zc)*tang
    dir_down = dir_down/np.linalg.norm(dir_down)

    anchor = (R_km + selected_altitude_km)*n
    positions = anchor + (length_km - length_km[selected_idx])[:, None]*dir_down[None, :]

    range_km = np.linalg.norm(positions - cam[None, :], axis=1)
    app_mag = np.asarray(meteor_abs_magnitude_to_apparent(abs_mag, range_km*1000.0), dtype=float)

    app_peak_idx = int(np.nanargmin(app_mag))
    selected_range_km = float(range_km[selected_idx])
    horizon_central_angle_deg = np.degrees(np.arccos(R_km/(R_km + float(camera_altitude_km))))

    return {
        "camera_km": cam,
        "all_positions_km": positions,
        "track_positions_km": positions,  # before filtering: all finite points
        "anchor_position_km": anchor,
        "selected_position_km": positions[selected_idx],
        "peak_position_km": positions[app_peak_idx],
        "selected_idx": selected_idx,
        "peak_idx": app_peak_idx,
        "selected_altitude_km": selected_altitude_km,
        "peak_altitude_km": float(h_km[app_peak_idx]),
        "selected_abs_mag": float(abs_mag[selected_idx]),
        "peak_abs_mag": float(abs_mag[app_peak_idx]),
        "selected_app_mag": float(app_mag[selected_idx]),
        "peak_app_mag": float(app_mag[app_peak_idx]),
        "selected_range_km": selected_range_km,
        "peak_range_km": float(range_km[app_peak_idx]),
        "central_angle_deg": float(central_angle_deg),
        "traj_azimuth_deg": float(traj_azimuth_deg),
        "horizon_central_angle_deg": float(horizon_central_angle_deg),
        "all_heights_km": h_km,
        "all_abs_mag": abs_mag,
        "all_app_mag": app_mag,
        "all_range_km": range_km,
        "all_time_s": time_s,
        # These are filled by apply_apparent_detection_and_camera_sampling().
        "detected_heights_km": h_km,
        "detected_abs_mag": abs_mag,
        "detected_app_mag": app_mag,
        "detected_range_km": range_km,
        "detected_time_s": time_s,
    }


def _indices_sampled_by_camera_fps(time_s: np.ndarray, camera_fps: float) -> np.ndarray:
    """Return indices closest to camera frame times for the detected interval."""
    time_s = np.asarray(time_s, dtype=float)
    good = np.isfinite(time_s)
    if not np.any(good):
        return np.arange(len(time_s), dtype=int)

    if camera_fps is None or float(camera_fps) <= 0:
        return np.arange(len(time_s), dtype=int)

    fps = float(camera_fps)
    order = np.argsort(time_s)
    t_sorted = time_s[order]

    t0 = float(t_sorted[0])
    t1 = float(t_sorted[-1])
    if t1 <= t0:
        return np.array([order[0]], dtype=int)

    frame_dt = 1.0/fps
    frame_times = np.arange(t0, t1 + 0.5*frame_dt, frame_dt)
    sampled_sorted = []
    for tf in frame_times:
        sampled_sorted.append(int(np.argmin(np.abs(t_sorted - tf))))

    sampled_sorted = np.unique(sampled_sorted)
    return order[sampled_sorted]


def apply_apparent_detection_and_camera_sampling(
    geometry: dict,
    det_mag_cut: float | None = 2,
    limiting_app_mag: float | None = None,
    use_delta_mag_cut: bool = True,
    from_peak_only: bool = True,
    camera_fps: float = 15.0,
    max_dots: int | None = None,
) -> dict:
    """
    Apply detection using APPARENT magnitude, then sample by camera FPS.

    Detection criteria:
      - peak + det_mag_cut in apparent magnitude, enabled by default
      - optional limiting apparent magnitude, m_app <= limiting_app_mag

    If both criteria are active, both must be satisfied.
    """
    out = copy.deepcopy(geometry)

    positions = np.asarray(out["all_positions_km"], dtype=float)
    h_km = np.asarray(out["all_heights_km"], dtype=float)
    abs_mag = np.asarray(out["all_abs_mag"], dtype=float)
    app_mag = np.asarray(out["all_app_mag"], dtype=float)
    range_km = np.asarray(out["all_range_km"], dtype=float)
    time_s = np.asarray(out["all_time_s"], dtype=float)

    n = min(len(positions), len(h_km), len(abs_mag), len(app_mag), len(range_km), len(time_s))
    positions = positions[:n]
    h_km = h_km[:n]
    abs_mag = abs_mag[:n]
    app_mag = app_mag[:n]
    range_km = range_km[:n]
    time_s = time_s[:n]

    good = np.isfinite(h_km) & np.isfinite(abs_mag) & np.isfinite(app_mag) & np.isfinite(range_km) & np.isfinite(time_s)
    if not np.any(good):
        raise ValueError("No finite apparent-magnitude points available for detection.")

    peak_idx = int(np.nanargmin(np.where(good, app_mag, np.nan)))
    peak_app_mag = float(app_mag[peak_idx])

    mask = good.copy()
    active_criteria = []

    if use_delta_mag_cut and det_mag_cut is not None and np.isfinite(float(det_mag_cut)):
        delta_threshold = peak_app_mag + float(det_mag_cut)
        mask &= app_mag <= delta_threshold
        active_criteria.append(f"peak + {float(det_mag_cut):.2f} mag")
    else:
        delta_threshold = np.nan

    if limiting_app_mag is not None and np.isfinite(float(limiting_app_mag)):
        mask &= app_mag <= float(limiting_app_mag)
        active_criteria.append(f"m_app <= {float(limiting_app_mag):.2f}")

    if from_peak_only:
        mask &= np.arange(n) >= peak_idx

    detected_indices = np.flatnonzero(mask)
    if detected_indices.size == 0:
        raise ValueError(
            "No points satisfy the apparent-magnitude detection criteria. "
            f"Peak m_app={peak_app_mag:.2f}, det_mag_cut={det_mag_cut}, "
            f"limiting_app_mag={limiting_app_mag}."
        )

    # Sample detected points at the camera frame rate.
    det_time = time_s[detected_indices]
    rel_sample_idx = _indices_sampled_by_camera_fps(det_time, camera_fps=float(camera_fps))
    sampled_indices = detected_indices[rel_sample_idx]

    # Optional cap for plots if the FPS sampling creates too many dots.
    if max_dots is not None and int(max_dots) > 0 and sampled_indices.size > int(max_dots):
        keep = np.unique(np.round(np.linspace(0, sampled_indices.size - 1, int(max_dots))).astype(int))
        sampled_indices = sampled_indices[keep]

    out["detection_peak_idx"] = peak_idx
    out["peak_idx"] = peak_idx
    out["peak_position_km"] = positions[peak_idx]
    out["peak_altitude_km"] = float(h_km[peak_idx])
    out["peak_abs_mag"] = float(abs_mag[peak_idx])
    out["peak_app_mag"] = peak_app_mag
    out["peak_range_km"] = float(range_km[peak_idx])
    out["det_mag_cut"] = None if det_mag_cut is None else float(det_mag_cut)
    out["delta_app_mag_threshold"] = float(delta_threshold) if np.isfinite(delta_threshold) else None
    out["limiting_app_mag"] = None if limiting_app_mag is None else float(limiting_app_mag)
    out["detection_criteria"] = " and ".join(active_criteria) if active_criteria else "all finite apparent-magnitude points"
    out["camera_fps"] = float(camera_fps)

    out["detected_indices"] = detected_indices
    out["track_positions_km"] = positions[detected_indices]
    out["detected_heights_km"] = h_km[detected_indices]
    out["detected_abs_mag"] = abs_mag[detected_indices]
    out["detected_app_mag"] = app_mag[detected_indices]
    out["detected_range_km"] = range_km[detected_indices]
    out["detected_time_s"] = time_s[detected_indices]

    out["sampled_indices"] = sampled_indices
    out["sampled_positions_km"] = positions[sampled_indices]
    out["sampled_heights_km"] = h_km[sampled_indices]
    out["sampled_abs_mag"] = abs_mag[sampled_indices]
    out["sampled_app_mag"] = app_mag[sampled_indices]
    out["sampled_range_km"] = range_km[sampled_indices]
    out["sampled_time_s"] = time_s[sampled_indices]

    return out

def magnitude_marker_sizes(mag, min_size: float = 10.0, max_size: float = 80.0) -> np.ndarray:
    """Make brighter points larger. Smaller magnitude means brighter."""
    mag = np.asarray(mag, dtype=float)
    if mag.size == 0:
        return np.asarray([], dtype=float)

    brightness = np.nanmax(mag) - mag
    spread = np.nanmax(brightness) - np.nanmin(brightness)

    if not np.isfinite(spread) or spread <= 0:
        return np.full(mag.shape, 0.5*(min_size + max_size), dtype=float)

    brightness_norm = (brightness - np.nanmin(brightness)) / spread
    return min_size + brightness_norm*(max_size - min_size)


def plot_mars_detected_3d_view(
    geometry: dict,
    out_path: str | Path,
    planet_radius_km: float = MARS_RADIUS_KM,
    camera_altitude_km: float = 5720.0,
    zenith_angle_rad: float | None = None,
    det_mag_cut: float = 2,
    title: str | None = None,
):
    """Create a 3D Mars+camera+meteor view using FPS-sampled apparent-magnitude dots."""
    out_path = str(out_path)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    R = float(planet_radius_km)
    u = np.linspace(0, 2*np.pi, 120)
    v = np.linspace(0, np.pi, 80)
    xs = R*np.outer(np.cos(u), np.sin(v))
    ys = R*np.outer(np.sin(u), np.sin(v))
    zs = R*np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(xs, ys, zs, rstride=3, cstride=3, linewidth=0, alpha=0.25, color='sandybrown')
    # make the sphere 

    track = np.asarray(geometry['track_positions_km'], dtype=float)
    dots = np.asarray(geometry.get('sampled_positions_km', track), dtype=float)
    dot_app_mag = np.asarray(geometry.get('sampled_app_mag', geometry.get('detected_app_mag')), dtype=float)
    dot_sizes = magnitude_marker_sizes(dot_app_mag, min_size=12.0, max_size=90.0)

    cam = np.asarray(geometry['camera_km'], dtype=float)
    peak = np.asarray(geometry['peak_position_km'], dtype=float)

    if len(track) > 1:
        ax.plot(track[:, 0], track[:, 1], track[:, 2], color='tab:red', lw=1.5, alpha=0.65, label='Detected interval')
    sc = ax.scatter(
        dots[:, 0], dots[:, 1], dots[:, 2],
        s=dot_sizes,
        c=dot_app_mag,
        cmap='inferno_r',
        alpha=0.95,
        label='Camera-frame samples',
    )
    ax.scatter([cam[0]], [cam[1]], [cam[2]], color='tab:blue', s=80, label='Camera')
    # ax.scatter([peak[0]], [peak[1]], [peak[2]], color='cyan', s=80, marker='*', edgecolors='k', linewidths=0.6, label='Peak apparent')
    ax.plot([cam[0], peak[0]], [cam[1], peak[1]], [cam[2], peak[2]], color='tab:gray', ls='--', lw=1.2, alpha=0.9)

    pad = R + float(camera_altitude_km) + 500.0
    ax.set_xlim(-pad*0.15, pad)
    ax.set_ylim(-pad*0.6, pad*0.6)
    ax.set_zlim(-pad*0.45, pad*0.45)
    ax.set_box_aspect((1.35, 1.0, 0.8))
    ax.set_xlabel('X [km]')
    ax.set_ylabel('Y [km]')
    ax.set_zlabel('Z [km]')
    if title is None:
        title = 'Detected meteor on Mars from orbiter view'
    ax.set_title(title)
    ax.view_init(elev=18, azim=-58)
    ax.legend(loc='upper right', fontsize=9)

    cb = plt.colorbar(sc, ax=ax, fraction=0.036, pad=0.08)
    cb.set_label('Apparent magnitude')

    zc_deg = None if zenith_angle_rad is None else np.degrees(float(zenith_angle_rad))
    label_lines = [
        f'Camera altitude: {camera_altitude_km:.0f} km',
        f'Camera FPS: {geometry.get("camera_fps", np.nan):.2f}',
        f'Detection: {geometry.get("detection_criteria", "apparent magnitude")}',
        f'Sampled dots: {len(geometry.get("sampled_indices", []))}',
        f'Peak altitude: {geometry["peak_altitude_km"]:.2f} km',
        f'Peak M_abs: {geometry["peak_abs_mag"]:.2f}',
        f'Peak m_app: {geometry["peak_app_mag"]:.2f}',
        f'Peak range: {geometry["peak_range_km"]:.1f} km',
    ]
    if zc_deg is not None:
        label_lines.insert(0, f'Zenith angle z_c: {zc_deg:.2f} deg')
    label_lines.append(f'Track azimuth: {geometry["traj_azimuth_deg"]:.1f} deg')
    label_lines.append(f'Central angle from nadir: {geometry["central_angle_deg"]:.1f} deg')

    fig.text(
        0.02, 0.02,
        '\n'.join(label_lines),
        ha='left', va='bottom', fontsize=10,
        bbox=dict(facecolor='white', edgecolor='0.7', alpha=0.9)
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path

def _unit_vector(vec):
    """Normalize a vector safely."""
    vec = np.asarray(vec, dtype=float)
    norm = np.linalg.norm(vec)
    if norm == 0 or not np.isfinite(norm):
        return vec
    return vec / norm


def load_or_make_mars_disk_image(mars_image_path: str | None = None, npx: int = 900) -> np.ndarray:
    """
    Load a Mars disk image if supplied, otherwise generate a simple Mars-like disk.

    The returned image is RGBA and has a circular alpha mask so it plots as a disk.
    """
    if mars_image_path is not None and os.path.exists(mars_image_path):
        img = mpimg.imread(mars_image_path)
        img = np.asarray(img)
        if img.dtype.kind in "ui":
            img = img.astype(float) / 255.0
        else:
            img = img.astype(float)

        if img.ndim == 2:
            img = np.dstack([img, img, img])
        if img.shape[-1] == 3:
            img = np.dstack([img, np.ones(img.shape[:2])])
        elif img.shape[-1] > 4:
            img = img[..., :4]

        img = np.clip(img, 0, 1)
    else:
        # Synthetic Mars-like texture so the plot works even without an image file.
        y = np.linspace(-1, 1, npx)
        x = np.linspace(-1, 1, npx)
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)
        theta = np.arctan2(Y, X)

        texture = (
            0.50
            + 0.16*np.sin(5*theta + 3*R)
            + 0.10*np.cos(13*X)*np.cos(9*Y)
            + 0.06*np.sin(25*(X + 0.35*Y))
        )
        limb = np.clip(1.08 - 0.55*R**2, 0.45, 1.05)

        red = np.clip((0.72 + 0.20*texture)*limb, 0, 1)
        green = np.clip((0.39 + 0.11*texture)*limb, 0, 1)
        blue = np.clip((0.22 + 0.06*texture)*limb, 0, 1)
        alpha = np.ones_like(red)
        img = np.dstack([red, green, blue, alpha])

    # Apply circular alpha mask.
    h, w = img.shape[:2]
    yy = np.linspace(-1, 1, h)
    xx = np.linspace(-1, 1, w)
    XX, YY = np.meshgrid(xx, yy)
    mask = (XX**2 + YY**2) <= 1.0
    img = img.copy()
    img[..., 3] *= mask.astype(float)

    return img


def sample_detected_dots_for_fov(geometry: dict, n_dots: int | None = None) -> dict:
    """Return already FPS-sampled meteor points for FoV plotting."""
    positions = np.asarray(geometry.get("sampled_positions_km", geometry["track_positions_km"]), dtype=float)
    app_mag = np.asarray(geometry.get("sampled_app_mag", geometry.get("detected_app_mag")), dtype=float)
    abs_mag = np.asarray(geometry.get("sampled_abs_mag", geometry.get("detected_abs_mag")), dtype=float)
    alt_km = np.asarray(geometry.get("sampled_heights_km", geometry.get("detected_heights_km")), dtype=float)
    time_s = np.asarray(geometry.get("sampled_time_s", geometry.get("detected_time_s")), dtype=float)
    range_km = np.asarray(geometry.get("sampled_range_km", geometry.get("detected_range_km")), dtype=float)

    n = min(len(positions), len(app_mag), len(abs_mag), len(alt_km), len(time_s), len(range_km))
    if n == 0:
        raise ValueError("No sampled meteor points available for FoV plotting.")

    positions = positions[:n]
    app_mag = app_mag[:n]
    abs_mag = abs_mag[:n]
    alt_km = alt_km[:n]
    time_s = time_s[:n]
    range_km = range_km[:n]

    if n_dots is not None and int(n_dots) > 0 and n > int(n_dots):
        idx = np.unique(np.round(np.linspace(0, n - 1, int(n_dots))).astype(int))
    else:
        idx = np.arange(n, dtype=int)

    return {
        "indices": idx,
        "positions_km": positions[idx],
        "app_mag": app_mag[idx],
        "abs_mag": abs_mag[idx],
        "alt_km": alt_km[idx],
        "time_s": time_s[idx],
        "range_km": range_km[idx],
    }

def marker_sizes_from_absolute_magnitude(abs_mag, min_size: float = 10.0, max_size: float = 80.0) -> np.ndarray:
    """Backward-compatible alias. Use magnitude_marker_sizes with apparent magnitude for new plots."""
    return magnitude_marker_sizes(abs_mag, min_size=min_size, max_size=max_size)

def camera_frame_centered_on_mars(camera_km: np.ndarray):
    """
    Build a camera coordinate system pointing exactly at Mars center.

    x: camera horizontal angle
    y: camera vertical angle
    z/forward: line of sight to Mars center
    """
    camera_km = np.asarray(camera_km, dtype=float)
    forward = _unit_vector(-camera_km)

    up_guess = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(forward, up_guess)) > 0.95:
        up_guess = np.array([0.0, 1.0, 0.0])

    right = _unit_vector(np.cross(forward, up_guess))
    up = _unit_vector(np.cross(right, forward))

    return forward, right, up


def project_points_to_mars_centered_camera(points_km: np.ndarray, camera_km: np.ndarray):
    """
    Project 3D points into angular camera coordinates in degrees.

    The camera is centered on Mars, so (0, 0) is Mars center.
    """
    forward, right, up = camera_frame_centered_on_mars(camera_km)

    points_km = np.asarray(points_km, dtype=float)
    rel = points_km - np.asarray(camera_km, dtype=float)[None, :]

    x = rel @ right
    y = rel @ up
    z = rel @ forward

    x_deg = np.degrees(np.arctan2(x, z))
    y_deg = np.degrees(np.arctan2(y, z))

    return x_deg, y_deg


def mars_angular_radius_deg(planet_radius_km: float, camera_altitude_km: float) -> float:
    """Angular radius of Mars as seen from camera altitude."""
    distance_from_center_km = float(planet_radius_km) + float(camera_altitude_km)
    return float(np.degrees(np.arcsin(float(planet_radius_km) / distance_from_center_km)))


def camera_fov_degrees(
    planet_radius_km: float,
    camera_altitude_km: float,
    sensor_width_mm: float | None = None,
    sensor_height_mm: float | None = None,
    focal_length_mm: float | None = None,
):
    """
    Return horizontal/vertical FoV in degrees.

    Default: exactly the angular diameter of Mars, i.e. Mars fills the frame.
    If sensor and focal length are supplied:
        FoV = 2 atan(sensor_size / 2f)
    """
    default_fov = 2.0 * mars_angular_radius_deg(planet_radius_km, camera_altitude_km)

    if sensor_width_mm is None or sensor_height_mm is None or focal_length_mm is None:
        return default_fov, default_fov

    sw = float(sensor_width_mm)
    sh = float(sensor_height_mm)
    f = float(focal_length_mm)

    if sw <= 0 or sh <= 0 or f <= 0:
        return default_fov, default_fov

    fov_x = 2.0*np.degrees(np.arctan(sw/(2.0*f)))
    fov_y = 2.0*np.degrees(np.arctan(sh/(2.0*f)))

    return float(fov_x), float(fov_y)


def plot_camera_fov_mars(
    geometry: dict,
    out_path: str | Path,
    planet_radius_km: float = MARS_RADIUS_KM,
    camera_altitude_km: float = 5720.0,
    n_dots: int = 12,
    det_mag_cut: float = 2,
    sensor_width_mm: float | None = None,
    sensor_height_mm: float | None = None,
    focal_length_mm: float | None = None,
    aperture_mm: float | None = None,
    mars_image_path: str | None = None,
    title: str | None = None,
) -> str:
    """
    Plot the camera FoV centered on Mars with detected meteor dots.

    Dots are larger for brighter apparent magnitudes.
    The default FoV exactly fits the full Mars disk.
    """
    out_path = str(out_path)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    mars_img = load_or_make_mars_disk_image(mars_image_path)
    dots = sample_detected_dots_for_fov(geometry, n_dots=None)

    camera_km = np.asarray(geometry["camera_km"], dtype=float)
    x_deg, y_deg = project_points_to_mars_centered_camera(dots["positions_km"], camera_km)
    peak_x, peak_y = project_points_to_mars_centered_camera(
        np.asarray(geometry["peak_position_km"], dtype=float)[None, :],
        camera_km,
    )

    sizes = magnitude_marker_sizes(dots["app_mag"], min_size=10.0, max_size=80.0)

    mars_radius_deg = mars_angular_radius_deg(planet_radius_km, camera_altitude_km)
    fov_x_deg, fov_y_deg = camera_fov_degrees(
        planet_radius_km,
        camera_altitude_km,
        sensor_width_mm=sensor_width_mm,
        sensor_height_mm=sensor_height_mm,
        focal_length_mm=focal_length_mm,
    )

    half_x = 0.5*fov_x_deg
    half_y = 0.5*fov_y_deg

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.imshow(
        mars_img,
        extent=[-mars_radius_deg, mars_radius_deg, -mars_radius_deg, mars_radius_deg],
        origin="lower",
        zorder=1,
    )

    mars_edge = plt.Circle(
        (0, 0),
        mars_radius_deg,
        facecolor="none",
        edgecolor="black",
        linewidth=1.2,
        zorder=2,
    )
    ax.add_patch(mars_edge)

    sc = ax.scatter(
        x_deg,
        y_deg,
        s=sizes,
        c=dots["app_mag"],
        cmap="inferno_r",
        # edgecolors="white",
        linewidths=0.7,
        alpha=0.95,
        zorder=3,
        label="Detected meteor dots",
    )

    # ax.scatter(
    #     peak_x,
    #     peak_y,
    #     s=300,
    #     marker="*",
    #     color="cyan",
    #     edgecolors="black",
    #     linewidths=0.8,
    #     zorder=4,
    #     label="Peak detected",
    # )

    ax.set_xlim(-half_x, half_x)
    ax.set_ylim(-half_y, half_y)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.25)
    ax.set_xlabel("Horizontal angle from Mars center [deg]")
    ax.set_ylabel("Vertical angle from Mars center [deg]")

    if title is None:
        title = "Mars-centered camera FoV"
    ax.set_title(title)

    colorbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Apparent magnitude")

    info = [
        f"FoV x: {fov_x_deg:.2f} deg",
        f"FoV y: {fov_y_deg:.2f} deg",
        f"Mars angular diameter: {2*mars_radius_deg:.2f} deg",
        f"Detected dots shown: {len(dots['indices'])}",
        f"Camera FPS: {geometry.get('camera_fps', np.nan):.2f}",
        f"Detection: {geometry.get('detection_criteria', 'apparent magnitude')}",
        f"Peak M_abs: {geometry['peak_abs_mag']:.2f}",
        f"Peak m_app: {geometry['peak_app_mag']:.2f}",
    ]

    if focal_length_mm is not None:
        info.append(f"Focal length: {float(focal_length_mm):.2f} mm")
    if aperture_mm is not None:
        info.append(f"Aperture: {float(aperture_mm):.2f} mm")
        if focal_length_mm is not None and float(aperture_mm) > 0:
            info.append(f"f/{float(focal_length_mm)/float(aperture_mm):.2f}")

    ax.text(
        0.02,
        0.02,
        "\n".join(info),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        bbox=dict(facecolor="white", edgecolor="0.7", alpha=0.9),
    )

    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()

    print(f"Saving FoV plot to: {out_path}")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    if not os.path.exists(out_path):
        raise RuntimeError(f"FoV plot was not created: {out_path}")

    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one fitted meteor on Mars/custom atmosphere using observation_data pickle loading."
    )

    parser.add_argument(
        "--pickle",
        default=r"C:\Users\maxiv\Documents\UWO\Papers\4)Mars meteors\Fireball\EN040326_201155\EN040326_201155_trajectory.pickle",
        help="Trajectory pickle file. Camera data are loaded through observation_data(...).",
    )
    parser.add_argument(
        "--json",
        default=r"C:\Users\maxiv\Documents\UWO\Papers\4)Mars meteors\Fireball\EN040326_201155\EN040326_201155_sim_fit_latest.json",
        help="Saved best-fit MetSim JSON or saved SimulationResults JSON containing const.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Folder where the plot will be saved. Defaults to the JSON folder.",
    )
    parser.add_argument("--base-name", default=None, help="Output base name. Defaults to pickle stem.")

    # observation_data options: these are passed directly into your existing loader.
    parser.add_argument("--use-all-cameras", action="store_true", help="Pass use_all_cameras=True to observation_data.")
    parser.add_argument("--pick-position", type=float, default=0.0, help="Pass pick_position to observation_data.")
    parser.add_argument("--prior", default="", help="Prior file path passed into observation_data for mass/luminous efficiency.")
    parser.add_argument("--fps-prior", type=float, default=np.nan, help="fps_prior passed into observation_data.")
    parser.add_argument("--P-0m-prior", dest="P_0m_prior", type=float, default=np.nan, help="P_0m_prior passed into observation_data.")
    parser.add_argument("--lag-noise-prior", type=float, default=40.0, help="lag_noise_prior passed into observation_data.")
    parser.add_argument("--lum-noise-prior", type=float, default=2.5, help="lum_noise_prior passed into observation_data.")

    # New atmosphere options.
    parser.add_argument("--atm-coeff-json", default=None, help="Custom atmDensPoly coefficient JSON file.")
    parser.add_argument("--atm-coeff-npy", default=None, help="Custom atmDensPoly coefficient NPY file.")
    parser.add_argument("--atm-min-km", type=float, default=40.0, help="Minimum altitude for density matching.")
    parser.add_argument("--atm-max-km", type=float, default=180.0, help="Maximum altitude for density matching.")
    parser.add_argument("--atm-step-m", type=float, default=100.0, help="Altitude step for density/dynamic-pressure matching.")
    parser.add_argument("--planet-radius-km", type=float, default=MARS_RADIUS_KM, help="Planet radius, Mars by default.")
    parser.add_argument("--planet-g0", type=float, default=MARS_G0, help="Surface gravity, Mars by default.")
    parser.add_argument("--h-kill-km", type=float, default=6.0, help="Kill height for the new-atmosphere simulation.")
    parser.add_argument("--v-init-kms", type=float, default=None, help="Override new-atmosphere v_init. If omitted, orbit estimate is tried, otherwise Earth v_init is kept.")

    # Detected-segment and 3D view options.
    parser.add_argument("--det-mag-cut", type=float, default=2.0, help="Apparent-magnitude delta cut: keep points brighter than apparent peak + this many mag.")
    parser.add_argument("--no-delta-mag-cut", action="store_true", help="Disable the peak + delta apparent-magnitude detection cut.")
    parser.add_argument("--limiting-app-mag", type=float, default=4, help="Optional limiting apparent magnitude. Keep only points with m_app <= this value.")
    parser.add_argument("--camera-fps", type=float, default=15.0, help="Camera FPS used to sample detected points for 3D/FoV plots.")
    parser.add_argument("--include-prepeak", action="store_true", help="If set, include detected Mars points before the peak as well. By default only the segment from the peak onward is kept.")
    parser.add_argument("--camera-altitude-km", type=float, default=5720.0, help="Camera altitude above Mars surface for the 3D image.")
    parser.add_argument("--central-angle-deg", type=float, default=20.0, help="Place the meteor this many degrees away from the sub-camera point.")
    parser.add_argument("--traj-azimuth-deg", type=float, default=30.0, help="Azimuth of the local meteor ground track in the 3D view.")
    parser.add_argument("--selected-altitude-km", type=float, default=None, help="Altitude snapshot for the 3D image. Default is the detected-peak altitude.")
    parser.add_argument("--n-detected-dots", type=int, default=None, help="Optional cap on plotted dots after FPS sampling. Default: no cap.")
    parser.add_argument("--mars-image", default=None, help="Optional Mars image file. If omitted, a synthetic Mars disk is generated.")
    parser.add_argument("--sensor-width-mm", type=float, default=None, help="Camera sensor width in mm. If omitted, FoV exactly fits Mars.")
    parser.add_argument("--sensor-height-mm", type=float, default=None, help="Camera sensor height in mm. If omitted, FoV exactly fits Mars.")
    parser.add_argument("--focal-length-mm", type=float, default=None, help="Camera focal length in mm. Used with sensor size to compute FoV.")
    parser.add_argument("--aperture-mm", type=float, default=None, help="Optional aperture diameter in mm, used for annotation.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(args.json)) or "."
    base_name = args.base_name or Path(args.pickle).stem.replace("_trajectory", "")

    # This is the main correction: use your existing pickle loading treatment.
    obs_data = build_observation_from_pickle(args)
    trajectories = load_trajectory_list_from_obs(obs_data)

    earth_const = load_best_fit_constants(args.json)
    earth_const = patch_constants_from_observation(earth_const, obs_data)

    print("Running Earth best-fit simulation...")
    best_guess_obj_plot = run_model_raw(earth_const)

    print("Creating new-atmosphere same-density constants...")
    dens_co_new = atmosphere_coefficients(args)
    best_guess_cost_mars = build_density_mapped_planet_const(
        earth_const=best_guess_obj_plot.const,
        obs_data=obs_data,
        trajectories=trajectories,
        dens_co_new=dens_co_new,
        args=args,
    )

    print("Running new-atmosphere same-density simulation...")
    best_guess_obj_plot_mars = run_model_raw(best_guess_cost_mars)

    print("Creating new-atmosphere same-dynamic-pressure constants...")
    (
        best_guess_cost_mars_dyn_press,
        heightsame_dynpress_mars,
        heightsame_dynpress_change_mars,
        erosion_beg_dyn_press_change,
    ) = build_dynamic_pressure_trigger_const(
        best_guess_obj_plot=best_guess_obj_plot,
        best_guess_obj_plot_mars=best_guess_obj_plot_mars,
        best_guess_cost_mars=best_guess_cost_mars,
    )

    print("Running new-atmosphere same-dynamic-pressure simulation...")
    best_guess_obj_plot_mars_dyn_press = run_model_raw(best_guess_cost_mars_dyn_press)

    # print("\n--- Summary ---")
    # print(f"Earth h_init                      : {best_guess_obj_plot.const.h_init/1000:.2f} km")
    # print(f"Earth erosion start h_e           : {best_guess_obj_plot.const.erosion_height_start/1000:.2f} km")
    # print(f"Earth p_dyn at h_e                : {getattr(best_guess_obj_plot.const, 'erosion_beg_dyn_press', dyn_press_at_height(best_guess_obj_plot, best_guess_obj_plot.const.erosion_height_start)):.3e} Pa")
    # print(f"New atmosphere same-rho h_init    : {best_guess_cost_mars.h_init/1000:.2f} km")
    # print(f"New atmosphere same-rho h_e       : {best_guess_cost_mars.erosion_height_start/1000:.2f} km")
    # print(f"New atmosphere same-p_dyn h_e     : {best_guess_cost_mars_dyn_press.erosion_height_start/1000:.2f} km")
    # if heightsame_dynpress_change_mars is not None:
    #     print(f"New atmosphere same-p_dyn h_e2    : {heightsame_dynpress_change_mars/1000:.2f} km")
    #     print(f"Earth p_dyn at h_e2               : {erosion_beg_dyn_press_change:.3e} Pa")
    # print(f"New atmosphere v_init             : {best_guess_cost_mars_dyn_press.v_init/1000:.3f} km/s")
    # print(f"New atmosphere z_c                : {math.degrees(float(best_guess_cost_mars_dyn_press.zenith_angle)):.3f} deg")

    output_path = plot_lightcurve_earth_vs_mars_dyn_pressure(
        obs_data=obs_data,
        best_guess_obj_plot=best_guess_obj_plot,
        best_guess_obj_plot_mars=best_guess_obj_plot_mars,
        best_guess_obj_plot_mars_dyn_press=best_guess_obj_plot_mars_dyn_press,
        heightsame_dynpress_mars=heightsame_dynpress_mars,
        heightsame_dynpress_change_mars=heightsame_dynpress_change_mars,
        output_dir=output_dir,
        base_name=base_name,
    )
    print(f"Saved plot: {output_path}")

    # Build the full Mars/custom geometry first, then detect using APPARENT magnitude.
    # Apparent magnitude depends on the camera range, so the detection cut must be applied after geometry.
    maybe_integrate_luminosity(best_guess_obj_plot_mars_dyn_press, obs_data)
    simulation_segment = extract_simulation_segment(best_guess_obj_plot_mars_dyn_press)
    full_geom = build_detected_track_geometry(
        simulation_segment,
        zenith_angle_rad=float(best_guess_obj_plot_mars_dyn_press.const.zenith_angle),
        planet_radius_km=float(args.planet_radius_km),
        camera_altitude_km=float(args.camera_altitude_km),
        central_angle_deg=float(args.central_angle_deg),
        traj_azimuth_deg=float(args.traj_azimuth_deg),
        selected_altitude_km=args.selected_altitude_km,
    )
    detected_geom = apply_apparent_detection_and_camera_sampling(
        full_geom,
        det_mag_cut=args.det_mag_cut,
        limiting_app_mag=args.limiting_app_mag,
        use_delta_mag_cut=not args.no_delta_mag_cut,
        from_peak_only=not args.include_prepeak,
        camera_fps=float(args.camera_fps),
        max_dots=args.n_detected_dots,
    )
    out3d = os.path.join(output_dir, base_name + '_Mars_detected_3D.png')
    plot_mars_detected_3d_view(
        detected_geom,
        out3d,
        planet_radius_km=float(args.planet_radius_km),
        camera_altitude_km=float(args.camera_altitude_km),
        zenith_angle_rad=float(best_guess_obj_plot_mars_dyn_press.const.zenith_angle),
        det_mag_cut=float(args.det_mag_cut),
        title=f'{base_name}: detected Mars meteor',
    )
    print(f"Saved 3D plot: {out3d}")

    outfov = os.path.join(output_dir, base_name + '_Mars_detected_FoV.png')
    print(f"Creating FoV plot: {outfov}")
    plot_camera_fov_mars(
        detected_geom,
        outfov,
        planet_radius_km=float(args.planet_radius_km),
        camera_altitude_km=float(args.camera_altitude_km),
        n_dots=args.n_detected_dots,
        det_mag_cut=float(args.det_mag_cut),
        sensor_width_mm=args.sensor_width_mm,
        sensor_height_mm=args.sensor_height_mm,
        focal_length_mm=args.focal_length_mm,
        aperture_mm=args.aperture_mm,
        mars_image_path=args.mars_image,
        title=f"{base_name}: Mars-centered camera FoV",
    )
    print(f"Saved FoV plot: {outfov}")

    print(f"Peak detected M_abs: {detected_geom['peak_abs_mag']:.2f}")
    print(f"Peak detected m_app: {detected_geom['peak_app_mag']:.2f}")
    print(f"Peak detected range: {detected_geom['peak_range_km']:.1f} km")


if __name__ == "__main__":
    main()
