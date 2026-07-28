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

Folder mode also supports fit_plots/EVENT_sim_fit_dynesty_BestGuess.json and
exports a reusable batch pickle, combined plot, and flat CSV.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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

from wmpl.MetSim.GUI import FragmentationEntry
from wmpl.Utils.AtmosphereDensity import atmDensPoly

try:
    from Mars_Vel import calculate_3d_intercept_speeds
except Exception:
    calculate_3d_intercept_speeds = None


MARS_RADIUS_KM = 3389.5
MARS_G0 = 3.75
MARS_P0M = 1500.0
MARS_ORBITAL_SPEED_KMS = 24.077


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


def _is_finite_number(value: Any) -> bool:
    """Return True only for scalar values which can be converted to a finite float."""
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _fragmentation_entry_from_mapping(entry_data: dict[str, Any]) -> FragmentationEntry:
    """Rebuild a WMPL FragmentationEntry from a JSON dictionary."""
    frag_type = str(entry_data.get("frag_type", "")).strip().upper()
    if frag_type not in {"EF", "D", "M", "A", "F"}:
        raise ValueError(f"Unsupported fragmentation type {frag_type!r}")

    height = entry_data.get("height")
    if not _is_finite_number(height):
        raise ValueError(f"Fragmentation entry {frag_type!r} has no valid height: {height!r}")

    erosion_coeff = entry_data.get("erosion_coeff")
    if frag_type == "D" and erosion_coeff is None:
        erosion_coeff = 0.0

    entry = FragmentationEntry(
        frag_type,
        float(height),
        entry_data.get("number"),
        entry_data.get("mass_percent"),
        entry_data.get("sigma"),
        entry_data.get("gamma"),
        erosion_coeff,
        entry_data.get("grain_mass_min"),
        entry_data.get("grain_mass_max"),
        entry_data.get("mass_index"),
    )

    # Preserve the optional input/provenance fields used by newer WMPL JSONs.
    for field in ("id", "upward_only", "dyn_pressure"):
        if field in entry_data:
            try:
                setattr(entry, field, entry_data[field])
            except Exception:
                pass

    # Saved *_sim_fit_latest.json files contain output state from the previous
    # run (normally done=true). A reloaded event must be allowed to trigger again.
    if hasattr(entry, "done"):
        entry.done = False

    return entry


def _reset_fragmentation_entries(const: Constants) -> list[Any]:
    """Reset all loaded fragmentation events before every MetSim run."""
    entries = list(getattr(const, "fragmentation_entries", []) or [])
    rebuilt_entries = []

    for entry in entries:
        if not isinstance(entry, dict):
            entry = vars(entry)
        rebuilt_entries.append(_fragmentation_entry_from_mapping(entry))

    const.fragmentation_entries = rebuilt_entries
    const.fragmentation_on = bool(rebuilt_entries)
    return rebuilt_entries


def load_best_fit_constants(json_path: str | Path) -> Constants:
    """
    Load constants from a fitted JSON.

    First tries WMPL/MetSim loadConstants(), because that is the native route for
    MetSim JSON files. If the JSON is instead one of your saved
    SimulationResults dictionaries with a top-level ``const`` entry, it falls
    back to manually creating a Constants object from that dictionary.
    """
    json_path = str(json_path)
    data = _load_json(json_path)
    const_dict = data.get("const", data)
    if not isinstance(const_dict, dict):
        raise ValueError(f"Could not find a constants dictionary in {json_path!r}")

    try:
        const, _ = loadConstants(json_path)
    except Exception:
        const = Constants()
        for key, value in const_dict.items():
            if key == "fragmentation_entries":
                continue
            setattr(const, key, _as_numeric_array_if_possible(value))

        const.fragmentation_entries = [
            _fragmentation_entry_from_mapping(entry)
            for entry in (const_dict.get("fragmentation_entries", []) or [])
        ]

    if hasattr(const, "dens_co"):
        const.dens_co = np.asarray(const.dens_co, dtype=float)

    # The fitted JSON is the source of truth for the photometric zero point.
    # This prevents observation_data's camera default (e.g. 840 W) from silently
    # replacing the 1500 W value stored in the fitted file.
    json_p0m = const_dict.get("P_0m")
    if _is_finite_number(json_p0m):
        const.P_0m = float(json_p0m)

    _reset_fragmentation_entries(const)

    return const


def build_observation_from_pickle(args: argparse.Namespace, fitted_p0m: float) -> Any:
    """
    Build obs_data exactly through your existing observation_data class.

    This is the key change from the previous version: do not raw-open the pickle
    here and do not invent a separate adapter. observation_data.load_pickle_data()
    already uses loadPickle(*os.path.split(current_file_name)) and builds the
    camera arrays used by the plotting function.
    """
    if _is_finite_number(args.P_0m_prior):
        effective_p0m = float(args.P_0m_prior)
    else:
        effective_p0m = float(fitted_p0m)

    obs_data = observation_data(
        args.pickle,
        use_all_cameras=args.use_all_cameras,
        lag_noise_prior=args.lag_noise_prior,
        lum_noise_prior=args.lum_noise_prior,
        fps_prior=args.fps_prior,
        P_0m_prior=effective_p0m,
        pick_position=args.pick_position,
        prior_file_path=args.prior,
    )
    # Keep the observed luminosity/magnitude conversion consistent with the JSON.
    obs_data.P_0m = effective_p0m
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
    # Do not overwrite P_0m here. The fitted JSON value is intentionally kept.
    if not _is_finite_number(getattr(out, "P_0m", None)) and hasattr(obs_data, "P_0m"):
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
        inclin_val = float(getattr(orbit, "i", getattr(orbit, "incl", np.nan)))*180/math.pi
        peri_val = float(orbit.peri)*180/math.pi
        node_val = float(orbit.node)*180/math.pi
        # print(f"Estimating Mars Vinf from orbit: a={a_val}, e={e_val}, i={inclin_val}, peri={peri_val}, node={node_val}")
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
    out.m_kill = float(1e-9)
    out.dt = float(0.04)
    out.P_0m = float(args.mars_P_0m)
    
    if args.v_init_kms is not None:
        out.v_init = float(args.v_init_kms) * 1000.0
    else:
        mars_vinit = estimate_mars_vinit_from_trajectory(trajectories)
        if mars_vinit is not None and np.isfinite(mars_vinit):
            out.v_init = mars_vinit
        # Otherwise keep fitted Earth v_init. This is safer than inventing a speed.

    print(f"speed on Earth: {float(earth_const.v_init)/1000.0:.3f} km/s, speed on Mars: {float(out.v_init)/1000.0:.3f} km/s")
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
    _reset_fragmentation_entries(const)
    print(f"Running simulation with {len(const.fragmentation_entries)} fragmentation events...")
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


def _new_fragmentation_entry_at_height(fragment: Any, height_m: float) -> FragmentationEntry:
    """Copy one fragmentation input event to a new trigger height."""
    entry = FragmentationEntry(
        fragment.frag_type,
        float(height_m),
        fragment.number,
        fragment.mass_percent,
        fragment.sigma,
        fragment.gamma,
        fragment.erosion_coeff,
        fragment.grain_mass_min,
        fragment.grain_mass_max,
        fragment.mass_index,
    )

    for field in ("id", "upward_only"):
        if hasattr(fragment, field):
            try:
                setattr(entry, field, getattr(fragment, field))
            except Exception:
                pass

    if hasattr(entry, "done"):
        entry.done = False
    return entry


def remap_fragmentation_entries_by_dynamic_pressure(
    earth_result: SimulationResults,
    mars_reference_result: SimulationResults,
    mars_const: Constants,
) -> list[dict[str, float | int | str]]:
    """
    Move every JSON EF/D release to the Mars height with the same dynamic pressure.

    The saved Earth ``dyn_pressure`` is preferred. If it is absent or invalid,
    the Earth simulation profile is sampled at the original entry height.
    """
    earth_entries = list(getattr(earth_result.const, "fragmentation_entries", []) or [])
    mars_entries = []
    mapping_summary = []

    for index, fragment in enumerate(earth_entries):
        frag_type = str(getattr(fragment, "frag_type", "")).strip().upper()
        if frag_type not in {"EF", "D"}:
            raise ValueError(
                f"Unsupported fragmentation type {frag_type!r} at index {index}; "
                "this Mars conversion currently supports EF and D releases."
            )

        target_q = getattr(fragment, "dyn_pressure", None)
        if not _is_finite_number(target_q) or float(target_q) <= 0.0:
            target_q = dyn_press_at_height(earth_result, float(fragment.height))
        target_q = float(target_q)

        mars_height = nearest_height_for_dyn_press(mars_reference_result, target_q)
        matched_q = dyn_press_at_height(mars_reference_result, mars_height)
        mars_entry = _new_fragmentation_entry_at_height(fragment, mars_height)

        # Retain the Earth trigger pressure as provenance. MetSim still triggers
        # the event using the newly mapped Mars height.
        try:
            mars_entry.dyn_pressure = target_q
        except Exception:
            pass

        mars_entries.append(mars_entry)
        mapping_summary.append({
            "index": index,
            "frag_type": frag_type,
            "earth_height_m": float(fragment.height),
            "target_dyn_pressure_pa": target_q,
            "mars_height_m": mars_height,
            "matched_dyn_pressure_pa": matched_q,
        })

        print(
            f"Fragmentation {index} ({frag_type}): "
            f"Earth {float(fragment.height)/1000.0:.2f} km, q={target_q:.3f} Pa "
            f"-> Mars {mars_height/1000.0:.2f} km "
            f"(q={matched_q:.3f} Pa)"
        )

    # MetSim expects the highest event first.
    mars_entries.sort(key=lambda entry: float(entry.height), reverse=True)
    mars_const.fragmentation_entries = mars_entries
    mars_const.fragmentation_on = bool(mars_entries)
    return mapping_summary


def build_dynamic_pressure_trigger_const(
    best_guess_obj_plot: SimulationResults,
    best_guess_obj_plot_mars: SimulationResults,
    best_guess_cost_mars: Constants,
) -> tuple[Constants, float, float | None, float | None]:
    """
    Move the global erosion transitions and every EF/D release to Mars heights
    having the same dynamic pressure as the corresponding Earth transitions.

    This also supports the smaller-meteoroid Dynesty BestGuess files, which may
    contain no ``fragmentation_entries`` but still define
    ``erosion_height_start`` and ``erosion_height_change``.
    """
    best_guess_cost_mars_dyn_press = copy.deepcopy(best_guess_cost_mars)

    earth_const = best_guess_obj_plot.const
    fragmentation_entries = list(getattr(earth_const, "fragmentation_entries", []) or [])

    heightsame_dynpress_mars = float(
        getattr(best_guess_cost_mars_dyn_press, "erosion_height_start", np.nan)
    )
    heightsame_dynpress_change_mars = None
    erosion_beg_dyn_press_change = None

    mapping_metadata: dict[str, Any] = {
        "global_erosion_start": None,
        "global_erosion_change": None,
        "fragmentation_entries": [],
    }

    earth_start_height = getattr(earth_const, "erosion_height_start", None)
    has_valid_start = (
        _is_finite_number(earth_start_height)
        and float(earth_start_height) > 0.0
    )

    # The usual erosion files set erosion_on=True. The BestGuess small-body files
    # may rely on the two erosion heights while carrying an empty fragmentation
    # list, so accept that layout as well.
    should_map_global_erosion = bool(getattr(earth_const, "erosion_on", False)) or (
        not fragmentation_entries and has_valid_start
    )

    if should_map_global_erosion and has_valid_start:
        earth_start_height = float(earth_start_height)
        erosion_beg_dyn_press = getattr(earth_const, "erosion_beg_dyn_press", None)
        if not _is_finite_number(erosion_beg_dyn_press) or float(erosion_beg_dyn_press) <= 0.0:
            erosion_beg_dyn_press = dyn_press_at_height(
                best_guess_obj_plot,
                earth_start_height,
            )
        erosion_beg_dyn_press = float(erosion_beg_dyn_press)

        heightsame_dynpress_mars = nearest_height_for_dyn_press(
            best_guess_obj_plot_mars,
            erosion_beg_dyn_press,
        )
        best_guess_cost_mars_dyn_press.erosion_height_start = heightsame_dynpress_mars
        try:
            best_guess_cost_mars_dyn_press.erosion_beg_dyn_press = erosion_beg_dyn_press
        except Exception:
            pass

        mapping_metadata["global_erosion_start"] = {
            "earth_height_m": earth_start_height,
            "target_dyn_pressure_pa": erosion_beg_dyn_press,
            "mars_height_m": float(heightsame_dynpress_mars),
            "matched_dyn_pressure_pa": dyn_press_at_height(
                best_guess_obj_plot_mars,
                heightsame_dynpress_mars,
            ),
        }
        print(
            "Global erosion start: "
            f"Earth {earth_start_height/1000.0:.2f} km, "
            f"q={erosion_beg_dyn_press:.3f} Pa -> "
            f"Mars {heightsame_dynpress_mars/1000.0:.2f} km"
        )

        earth_change_height = getattr(earth_const, "erosion_height_change", None)
        if _is_finite_number(earth_change_height) and float(earth_change_height) > 0.0:
            earth_change_height = float(earth_change_height)
            try:
                erosion_beg_dyn_press_change = dyn_press_at_height(
                    best_guess_obj_plot,
                    earth_change_height,
                )
                heightsame_dynpress_change_mars = nearest_height_for_dyn_press(
                    best_guess_obj_plot_mars,
                    erosion_beg_dyn_press_change,
                )
                best_guess_cost_mars_dyn_press.erosion_height_change = (
                    heightsame_dynpress_change_mars
                )

                mapping_metadata["global_erosion_change"] = {
                    "earth_height_m": earth_change_height,
                    "target_dyn_pressure_pa": float(erosion_beg_dyn_press_change),
                    "mars_height_m": float(heightsame_dynpress_change_mars),
                    "matched_dyn_pressure_pa": dyn_press_at_height(
                        best_guess_obj_plot_mars,
                        heightsame_dynpress_change_mars,
                    ),
                }
                print(
                    "Global erosion change: "
                    f"Earth {earth_change_height/1000.0:.2f} km, "
                    f"q={erosion_beg_dyn_press_change:.3f} Pa -> "
                    f"Mars {heightsame_dynpress_change_mars/1000.0:.2f} km"
                )
            except Exception as exc:
                print(f"Could not set p_dyn erosion_height_change: {exc}")

    fragmentation_mapping = remap_fragmentation_entries_by_dynamic_pressure(
        earth_result=best_guess_obj_plot,
        mars_reference_result=best_guess_obj_plot_mars,
        mars_const=best_guess_cost_mars_dyn_press,
    )
    mapping_metadata["fragmentation_entries"] = fragmentation_mapping

    # Store the mapping on the constants object so it is retained in the per-event
    # pickle and can be flattened into the batch CSV.
    try:
        best_guess_cost_mars_dyn_press.dynamic_pressure_mapping = mapping_metadata
    except Exception:
        pass

    best_guess_cost_mars_dyn_press.P_0m = float(best_guess_cost_mars.P_0m)

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
            float(getattr(sim.const, "P_0m", obs_data.P_0m)),
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

    # maybe_integrate_luminosity(best_guess_obj_plot, obs_data)

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

    # maybe_integrate_luminosity(best_guess_obj_plot_mars, obs_data)
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

    ax.legend(fontsize=12, loc="upper right")
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
    alpha = np.deg2rad(float(central_angle_deg))
    n = np.array([np.cos(alpha), np.sin(alpha), 0.0], dtype=float)
    east = np.array([-np.sin(alpha), np.cos(alpha), 0.0], dtype=float)
    north = np.array([0.0, 0.0, 1.0], dtype=float)
    return n, east, north


def estimate_track_speed_from_segment(detected_segment: dict) -> float | None:
    """Estimate the meteor speed in km/s from the simulated length-time arrays."""
    try:
        length_m = np.asarray(detected_segment["length_m"], dtype=float)
        time_s = np.asarray(detected_segment["time_s"], dtype=float)
    except Exception:
        return None

    n = min(len(length_m), len(time_s))
    if n < 2:
        return None

    length_m = length_m[:n]
    time_s = time_s[:n]
    dt = np.diff(time_s)
    dl = np.diff(length_m)
    good = np.isfinite(dt) & np.isfinite(dl) & (np.abs(dt) > 0.0)
    if not np.any(good):
        return None

    speed_kms = np.abs(dl[good]/dt[good])/1000.0
    if speed_kms.size == 0 or not np.any(np.isfinite(speed_kms)):
        return None
    return float(np.nanmedian(speed_kms))


MARS_METEOR_MIN_SPEED_KMS = 4.926
MARS_METEOR_MAX_SPEED_KMS = 58.254

def physically_motivated_detection_longitude(
    meteor_speed_kms: float | None,
    horizon_central_angle_deg: float,
    mars_min_speed_kms: float = MARS_METEOR_MIN_SPEED_KMS,
    mars_max_speed_kms: float = MARS_METEOR_MAX_SPEED_KMS,
    limb_fraction: float = 0.92,
) -> tuple[float, float, str]:
    """
    Map Mars meteor speed to a signed longitude on the camera-facing dark side.

    low speed  -> dusk side  -> negative Y
    high speed -> dawn side  -> positive Y

    Returned angle is signed and stays within the visible hemisphere, so X stays positive.
    """
    horizon = abs(float(horizon_central_angle_deg))
    vmin = float(mars_min_speed_kms)
    vmax = float(mars_max_speed_kms)

    if meteor_speed_kms is None or not np.isfinite(float(meteor_speed_kms)):
        speed_fraction = 0.5
    else:
        speed_fraction = float(np.clip((float(meteor_speed_kms) - vmin)/(vmax - vmin), 0.0, 1.0))

    if speed_fraction <= 1.0/3.0:
        regime = "dusk-side / low-speed"
    elif speed_fraction >= 2.0/3.0:
        regime = "dawn-side / high-speed"
    else:
        regime = "intermediate night-side"

    # map [0,1] -> [-limb_fraction*horizon, +limb_fraction*horizon]
    central_angle_deg = (-limb_fraction + 2.0*limb_fraction*speed_fraction) * horizon

    # safety clip to remain on visible hemisphere
    central_angle_deg = float(np.clip(central_angle_deg, -0.995*horizon, +0.995*horizon))

    return central_angle_deg, speed_fraction, regime


def build_detected_track_geometry(
    detected_segment: dict,
    zenith_angle_rad: float,
    planet_radius_km: float = MARS_RADIUS_KM,
    camera_altitude_km: float = 5720.0,
    central_angle_deg: float = 20.0,
    traj_azimuth_deg: float = 30.0,
    selected_altitude_km: float | None = None,
    physically_motivated_longitude: bool = False,
    meteor_speed_kms: float | None = None,
    mars_orbital_speed_kms: float = MARS_ORBITAL_SPEED_KMS,
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

    horizon_central_angle_deg = np.degrees(np.arccos(R_km/(R_km + float(camera_altitude_km))))

    inferred_speed_kms = meteor_speed_kms
    if inferred_speed_kms is None or not np.isfinite(float(inferred_speed_kms)):
        inferred_speed_kms = estimate_track_speed_from_segment(detected_segment)

    longitude_mode = "manual"
    longitude_speed_fraction = np.nan
    longitude_regime = "manual"
    if physically_motivated_longitude:
        central_angle_deg, longitude_speed_fraction, longitude_regime = physically_motivated_detection_longitude(
            meteor_speed_kms,
            horizon_central_angle_deg,
            # mars_orbital_speed_kms=float(mars_orbital_speed_kms),
        )
        longitude_mode = "speed-based"

    # Rebuild the local frame after resolving the final central angle.
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
        "longitude_mode": longitude_mode,
        "longitude_speed_fraction": float(longitude_speed_fraction) if np.isfinite(longitude_speed_fraction) else np.nan,
        "longitude_regime": longitude_regime,
        "meteor_speed_kms": float(inferred_speed_kms) if inferred_speed_kms is not None and np.isfinite(float(inferred_speed_kms)) else np.nan,
        "mars_orbital_speed_kms": float(mars_orbital_speed_kms),
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


def _linear_threshold_crossing_time(
    t0: float,
    mag0: float,
    t1: float,
    mag1: float,
    threshold_mag: float,
) -> float:
    """Linearly interpolate the time at which apparent magnitude crosses a threshold."""
    values = (t0, mag0, t1, mag1, threshold_mag)
    if not all(np.isfinite(float(value)) for value in values):
        return float(t1)

    if float(mag1) == float(mag0):
        return 0.5*(float(t0) + float(t1))

    fraction = (float(threshold_mag) - float(mag0))/(float(mag1) - float(mag0))
    fraction = float(np.clip(fraction, 0.0, 1.0))
    return float(t0) + fraction*(float(t1) - float(t0))


def _camera_frame_times_with_random_phase(
    visible_start_s: float,
    visible_end_s: float,
    camera_fps: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float, float]:
    """
    Generate exact camera frame times with a random phase relative to detection onset.

    The threshold-crossing time is not itself sampled. The first frame occurs at
    ``visible_start_s + random_phase``, and all later frames are separated by
    exactly ``1/camera_fps``. The random phase is drawn from one frame interval.

    Because this routine is used only after the meteor has been classified as
    detected, a very short visible interval is conditioned to contain one frame:
    the phase is then drawn uniformly inside that shorter interval.
    """
    fps = float(camera_fps)
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"camera_fps must be positive and finite, got {camera_fps!r}")

    visible_start_s = float(visible_start_s)
    visible_end_s = float(visible_end_s)
    visible_duration_s = visible_end_s - visible_start_s
    if not np.isfinite(visible_duration_s) or visible_duration_s <= 0.0:
        return np.asarray([], dtype=float), np.nan, 1.0/fps

    frame_dt_s = 1.0/fps

    # Draw a strictly positive phase so the first camera sample is never placed
    # exactly at the threshold crossing. For events shorter than one frame, this
    # is the conditional phase distribution given that at least one frame sees it.
    phase_upper_s = min(frame_dt_s, visible_duration_s)
    epsilon_s = max(np.finfo(float).eps*max(1.0, abs(visible_start_s)), 1.0e-12)
    if phase_upper_s <= epsilon_s:
        phase_s = 0.5*phase_upper_s
    else:
        phase_s = float(rng.uniform(epsilon_s, phase_upper_s))

    first_frame_s = visible_start_s + phase_s
    if first_frame_s >= visible_end_s:
        first_frame_s = np.nextafter(visible_end_s, visible_start_s)
        phase_s = first_frame_s - visible_start_s

    # Use an integer frame counter rather than repeated nearest-neighbour picks.
    # This guarantees exact 1/FPS spacing in the requested frame-time grid.
    max_frame_index = int(np.floor((visible_end_s - first_frame_s)/frame_dt_s))
    frame_numbers = np.arange(max_frame_index + 1, dtype=float)
    frame_times_s = first_frame_s + frame_numbers*frame_dt_s

    # Do not include a frame at the exact fade-out crossing, even in the unlikely
    # case that floating-point arithmetic lands there.
    tolerance_s = max(1.0e-12, frame_dt_s*1.0e-10)
    frame_times_s = frame_times_s[frame_times_s < visible_end_s - tolerance_s]

    if frame_times_s.size == 0:
        frame_times_s = np.asarray([first_frame_s], dtype=float)

    return frame_times_s, phase_s, frame_dt_s


def _nearest_source_indices(time_s: np.ndarray, sample_times_s: np.ndarray) -> np.ndarray:
    """Return source-array indices nearest to exact interpolated camera frame times."""
    time_s = np.asarray(time_s, dtype=float)
    sample_times_s = np.asarray(sample_times_s, dtype=float)
    if sample_times_s.size == 0:
        return np.asarray([], dtype=int)

    insertion = np.searchsorted(time_s, sample_times_s, side="left")
    insertion = np.clip(insertion, 0, len(time_s) - 1)
    previous = np.clip(insertion - 1, 0, len(time_s) - 1)
    choose_previous = np.abs(sample_times_s - time_s[previous]) <= np.abs(time_s[insertion] - sample_times_s)
    return np.where(choose_previous, previous, insertion).astype(int)


def _interpolate_vector(time_s: np.ndarray, values: np.ndarray, sample_times_s: np.ndarray) -> np.ndarray:
    """Linearly interpolate a one-dimensional or vector-valued series in time."""
    time_s = np.asarray(time_s, dtype=float)
    values = np.asarray(values, dtype=float)
    sample_times_s = np.asarray(sample_times_s, dtype=float)

    if values.ndim == 1:
        return np.interp(sample_times_s, time_s, values)

    return np.column_stack([
        np.interp(sample_times_s, time_s, values[:, component])
        for component in range(values.shape[1])
    ])


def _threshold_interval_around_peak(
    time_s: np.ndarray,
    app_mag: np.ndarray,
    good: np.ndarray,
    peak_idx: int,
    threshold_mag: float,
    from_peak_only: bool = False,
) -> tuple[int, int, float, float]:
    """Return the contiguous interval around the peak satisfying a magnitude threshold."""
    mask = np.asarray(good, dtype=bool) & (np.asarray(app_mag, dtype=float) <= float(threshold_mag))

    if not mask[int(peak_idx)]:
        raise ValueError(
            f"The apparent-magnitude peak does not satisfy threshold {float(threshold_mag):.3f}."
        )

    left_idx = int(peak_idx)
    while left_idx > 0 and mask[left_idx - 1]:
        left_idx -= 1

    right_idx = int(peak_idx)
    while right_idx + 1 < len(mask) and mask[right_idx + 1]:
        right_idx += 1

    if from_peak_only:
        left_idx = int(peak_idx)
        start_s = float(time_s[peak_idx])
    elif left_idx > 0 and good[left_idx - 1]:
        start_s = _linear_threshold_crossing_time(
            time_s[left_idx - 1], app_mag[left_idx - 1],
            time_s[left_idx], app_mag[left_idx],
            threshold_mag,
        )
    else:
        start_s = float(time_s[left_idx])

    if right_idx + 1 < len(mask) and good[right_idx + 1]:
        end_s = _linear_threshold_crossing_time(
            time_s[right_idx], app_mag[right_idx],
            time_s[right_idx + 1], app_mag[right_idx + 1],
            threshold_mag,
        )
    else:
        end_s = float(time_s[right_idx])

    if end_s <= start_s:
        raise ValueError(
            f"Invalid threshold interval: onset={start_s:.9f} s, end={end_s:.9f} s."
        )

    return left_idx, right_idx, start_s, end_s


def apply_apparent_detection_and_camera_sampling(
    geometry: dict,
    det_mag_cut: float | None = 2,
    limiting_app_mag: float | None = 4.0,
    use_delta_mag_cut: bool = True,
    from_peak_only: bool = False,
    camera_fps: float = 15.0,
    max_dots: int | None = None,
    sampling_seed: int | None = None,
) -> dict:
    """
    Apply the camera limiting magnitude and sample at the exact camera FPS.

    A meteor with ``peak m_app > limiting_app_mag`` is not discarded. Instead,
    the full simulated luminous trajectory is sampled at the requested FPS using
    the same random sub-frame phase. Those samples are marked as non-detections
    so the plots can show them as open circles.

    Physical detections are controlled only by::

        m_app(t) <= limiting_app_mag

    ``det_mag_cut`` remains a separate peak-plus-delta-m design comparison and
    never changes the physical camera detection interval.
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
    source_indices = np.arange(n, dtype=int)

    good = (
        np.isfinite(h_km)
        & np.isfinite(abs_mag)
        & np.isfinite(app_mag)
        & np.isfinite(range_km)
        & np.isfinite(time_s)
        & np.all(np.isfinite(positions), axis=1)
    )
    if not np.any(good):
        raise ValueError("No finite apparent-magnitude points available for detection.")

    # Keep only finite points, sort by time, and remove duplicate timestamps.
    positions = positions[good]
    h_km = h_km[good]
    abs_mag = abs_mag[good]
    app_mag = app_mag[good]
    range_km = range_km[good]
    time_s = time_s[good]
    source_indices = source_indices[good]

    order = np.argsort(time_s, kind="stable")
    positions = positions[order]
    h_km = h_km[order]
    abs_mag = abs_mag[order]
    app_mag = app_mag[order]
    range_km = range_km[order]
    time_s = time_s[order]
    source_indices = source_indices[order]

    _, unique_idx = np.unique(time_s, return_index=True)
    unique_idx = np.sort(unique_idx)
    positions = positions[unique_idx]
    h_km = h_km[unique_idx]
    abs_mag = abs_mag[unique_idx]
    app_mag = app_mag[unique_idx]
    range_km = range_km[unique_idx]
    time_s = time_s[unique_idx]
    source_indices = source_indices[unique_idx]
    good = np.ones(len(time_s), dtype=bool)

    if len(time_s) < 2:
        raise ValueError("At least two finite simulation times are required for camera sampling.")

    peak_idx = int(np.nanargmin(app_mag))
    peak_app_mag = float(app_mag[peak_idx])

    if limiting_app_mag is None or not np.isfinite(float(limiting_app_mag)):
        raise ValueError(
            "A finite limiting_app_mag is required because physical detection must "
            "be defined by the camera limiting magnitude."
        )
    physical_threshold = float(limiting_app_mag)
    physically_detectable = bool(peak_app_mag <= physical_threshold)

    rng = np.random.default_rng(sampling_seed)

    if physically_detectable:
        # Use the contiguous above-LM interval around the peak.
        left_idx, right_idx, visible_start_s, visible_end_s = _threshold_interval_around_peak(
            time_s=time_s,
            app_mag=app_mag,
            good=good,
            peak_idx=peak_idx,
            threshold_mag=physical_threshold,
            from_peak_only=from_peak_only,
        )
        detected_sorted_indices = np.arange(left_idx, right_idx + 1, dtype=int)
        sample_interval_start_s = visible_start_s
        sample_interval_end_s = visible_end_s
        detection_status = "detected"
    else:
        # Preserve and sample the whole simulated luminous event. These are
        # hypothetical camera-frame locations, all marked as below the LM.
        detected_sorted_indices = np.asarray([], dtype=int)
        visible_start_s = None
        visible_end_s = None
        sample_interval_start_s = float(time_s[0])
        sample_interval_end_s = float(time_s[-1])
        detection_status = "not detected"

    detected_indices = source_indices[detected_sorted_indices]

    sample_times_s, random_phase_s, frame_dt_s = _camera_frame_times_with_random_phase(
        sample_interval_start_s,
        sample_interval_end_s,
        camera_fps=float(camera_fps),
        rng=rng,
    )

    # Interpolate every quantity at exact camera-frame timestamps.
    sampled_positions = _interpolate_vector(time_s, positions, sample_times_s)
    sampled_heights = _interpolate_vector(time_s, h_km, sample_times_s)
    sampled_abs_mag = _interpolate_vector(time_s, abs_mag, sample_times_s)
    sampled_app_mag = _interpolate_vector(time_s, app_mag, sample_times_s)
    sampled_range = _interpolate_vector(time_s, range_km, sample_times_s)
    sampled_sorted_indices = _nearest_source_indices(time_s, sample_times_s)
    sampled_indices = source_indices[sampled_sorted_indices]
    sampled_detected_mask = sampled_app_mag <= physical_threshold

    # In the non-detectable case this should be entirely False; calculating it
    # explicitly keeps the plotting code robust to future sampling changes.
    physical_detected_frame_count = int(np.count_nonzero(sampled_detected_mask))

    # DESIGN ASSUMPTION: calculate Delta-m statistics without using them as a trigger.
    design_enabled = (
        bool(use_delta_mag_cut)
        and det_mag_cut is not None
        and np.isfinite(float(det_mag_cut))
        and float(det_mag_cut) >= 0.0
    )

    delta_threshold = np.nan
    delta_start_s = np.nan
    delta_end_s = np.nan
    delta_duration_s = np.nan
    delta_expected_frames = np.nan
    delta_frame_mask = np.zeros(len(sample_times_s), dtype=bool)
    delta_frame_count = 0
    delta_fully_within_camera_limit = False

    if design_enabled:
        delta_mag = float(det_mag_cut)
        delta_threshold = peak_app_mag + delta_mag
        _, _, delta_start_s, delta_end_s = _threshold_interval_around_peak(
            time_s=time_s,
            app_mag=app_mag,
            good=good,
            peak_idx=peak_idx,
            threshold_mag=delta_threshold,
            from_peak_only=False,
        )
        delta_duration_s = delta_end_s - delta_start_s
        delta_expected_frames = delta_duration_s*float(camera_fps)
        delta_frame_mask = sampled_app_mag <= delta_threshold

        # Only physically detected camera frames count as observed Delta-m frames.
        delta_frame_count = int(np.count_nonzero(delta_frame_mask & sampled_detected_mask))
        delta_fully_within_camera_limit = bool(delta_threshold <= physical_threshold)

    if physically_detectable:
        print(
            f"Physical visible interval (m_app <= {physical_threshold:.2f}): "
            f"{visible_start_s:.6f}--{visible_end_s:.6f} s "
            f"({visible_end_s - visible_start_s:.6f} s)."
        )
    else:
        print(
            "Meteor is not physically detectable: "
            f"peak m_app={peak_app_mag:.3f} is fainter than LM={physical_threshold:.3f}. "
            "The full simulated event will be shown as open circles."
        )

    print(
        f"Random first-frame delay: {random_phase_s:.6f} s; "
        f"camera cadence: {frame_dt_s:.6f} s ({float(camera_fps):.3f} FPS); "
        f"sampled frames shown: {len(sample_times_s)}; "
        f"physically detected frames: {physical_detected_frame_count}."
    )

    out["detection_peak_idx"] = int(source_indices[peak_idx])
    out["peak_idx"] = int(source_indices[peak_idx])
    out["peak_position_km"] = positions[peak_idx]
    out["peak_altitude_km"] = float(h_km[peak_idx])
    out["peak_abs_mag"] = float(abs_mag[peak_idx])
    out["peak_app_mag"] = peak_app_mag
    out["peak_range_km"] = float(range_km[peak_idx])

    out["limiting_app_mag"] = physical_threshold
    out["effective_app_mag_threshold"] = physical_threshold
    out["is_physically_detectable"] = physically_detectable
    out["detection_status"] = detection_status
    out["detection_criteria"] = (
        f"m_app <= {physical_threshold:.2f} (physical camera limit)"
        if physically_detectable
        else f"not detected: peak m_app={peak_app_mag:.2f} > LM={physical_threshold:.2f}"
    )
    out["camera_fps"] = float(camera_fps)
    out["camera_frame_dt_s"] = frame_dt_s
    out["sampling_seed"] = sampling_seed
    out["sampling_phase_s"] = random_phase_s
    out["first_sample_delay_s"] = random_phase_s
    out["sample_interval_start_time_s"] = sample_interval_start_s
    out["sample_interval_end_time_s"] = sample_interval_end_s
    out["sample_interval_duration_s"] = sample_interval_end_s - sample_interval_start_s
    out["detection_onset_time_s"] = visible_start_s
    out["detection_end_time_s"] = visible_end_s
    out["visible_duration_s"] = (
        float(visible_end_s - visible_start_s) if physically_detectable else 0.0
    )
    out["physical_detected_frame_count"] = physical_detected_frame_count
    out["camera_frame_count"] = int(len(sample_times_s))
    out["requested_plot_max_dots"] = max_dots

    out["det_mag_cut"] = float(det_mag_cut) if design_enabled else None
    out["delta_app_mag_threshold"] = float(delta_threshold) if design_enabled else None
    out["design_assumption_enabled"] = design_enabled
    out["design_assumption_criteria"] = (
        f"m_app <= m_peak + {float(det_mag_cut):.2f} = {delta_threshold:.2f}"
        if design_enabled else "disabled"
    )
    out["design_delta_onset_time_s"] = float(delta_start_s) if design_enabled else None
    out["design_delta_end_time_s"] = float(delta_end_s) if design_enabled else None
    out["design_delta_duration_s"] = float(delta_duration_s) if design_enabled else None
    out["design_delta_expected_frames_at_fps"] = float(delta_expected_frames) if design_enabled else None
    out["design_delta_frame_count"] = int(delta_frame_count) if design_enabled else 0
    out["design_delta_fully_within_camera_limit"] = (
        delta_fully_within_camera_limit if design_enabled else None
    )
    out["design_delta_fraction_of_physical_frames"] = (
        float(delta_frame_count/physical_detected_frame_count)
        if design_enabled and physical_detected_frame_count > 0 else None
    )

    out["detected_indices"] = detected_indices
    out["track_positions_km"] = (
        positions[detected_sorted_indices] if physically_detectable else positions
    )
    out["detected_heights_km"] = h_km[detected_sorted_indices]
    out["detected_abs_mag"] = abs_mag[detected_sorted_indices]
    out["detected_app_mag"] = app_mag[detected_sorted_indices]
    out["detected_range_km"] = range_km[detected_sorted_indices]
    out["detected_time_s"] = time_s[detected_sorted_indices]

    out["sampled_indices"] = sampled_indices
    out["sampled_positions_km"] = sampled_positions
    out["sampled_heights_km"] = sampled_heights
    out["sampled_abs_mag"] = sampled_abs_mag
    out["sampled_app_mag"] = sampled_app_mag
    out["sampled_range_km"] = sampled_range
    out["sampled_time_s"] = sample_times_s
    out["sampled_detected_mask"] = sampled_detected_mask

    out["design_delta_sampled_mask"] = delta_frame_mask
    out["design_delta_sampled_indices"] = sampled_indices[delta_frame_mask]
    out["design_delta_sampled_time_s"] = sample_times_s[delta_frame_mask]
    out["design_delta_sampled_app_mag"] = sampled_app_mag[delta_frame_mask]

    return out


def save_detection_summary(geometry: dict, out_path: str | Path) -> str:
    """Save physical-detection and Delta-m design-comparison metrics to JSON."""
    out_path = str(out_path)
    summary = {
        "peak": {
            "absolute_magnitude": float(geometry["peak_abs_mag"]),
            "apparent_magnitude": float(geometry["peak_app_mag"]),
            "altitude_km": float(geometry["peak_altitude_km"]),
            "range_km": float(geometry["peak_range_km"]),
        },
        "physical_camera_detection": {
            "is_detectable": geometry.get("is_physically_detectable"),
            "status": geometry.get("detection_status"),
            "criterion": geometry.get("detection_criteria"),
            "limiting_apparent_magnitude": geometry.get("limiting_app_mag"),
            "onset_time_s": geometry.get("detection_onset_time_s"),
            "end_time_s": geometry.get("detection_end_time_s"),
            "duration_s": geometry.get("visible_duration_s"),
            "camera_fps": geometry.get("camera_fps"),
            "frame_interval_s": geometry.get("camera_frame_dt_s"),
            "first_frame_delay_s": geometry.get("first_sample_delay_s"),
            "detected_frame_count": geometry.get("physical_detected_frame_count"),
            "sampled_frame_count": geometry.get("camera_frame_count"),
            "sample_times_s": np.asarray(geometry.get("sampled_time_s", []), dtype=float).tolist(),
            "sampled_apparent_magnitudes": np.asarray(
                geometry.get("sampled_app_mag", []), dtype=float
            ).tolist(),
        },
        "delta_m_design_comparison": {
            "enabled": geometry.get("design_assumption_enabled"),
            "criterion": geometry.get("design_assumption_criteria"),
            "delta_m": geometry.get("det_mag_cut"),
            "faint_edge_apparent_magnitude": geometry.get("delta_app_mag_threshold"),
            "onset_time_s": geometry.get("design_delta_onset_time_s"),
            "end_time_s": geometry.get("design_delta_end_time_s"),
            "duration_s": geometry.get("design_delta_duration_s"),
            "expected_frames_duration_times_fps": geometry.get(
                "design_delta_expected_frames_at_fps"
            ),
            "actual_physical_camera_frames": geometry.get("design_delta_frame_count"),
            "fraction_of_physical_frames": geometry.get(
                "design_delta_fraction_of_physical_frames"
            ),
            "fully_within_camera_limit": geometry.get(
                "design_delta_fully_within_camera_limit"
            ),
            "sample_times_s": np.asarray(
                geometry.get("design_delta_sampled_time_s", []), dtype=float
            ).tolist(),
            "sampled_apparent_magnitudes": np.asarray(
                geometry.get("design_delta_sampled_app_mag", []), dtype=float
            ).tolist(),
        },
    }

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, allow_nan=False)

    return out_path


def _format_optional_number(value: Any, precision: int = 3, fallback: str = "n/a") -> str:
    """Format an optional finite scalar for plot annotations."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    if not np.isfinite(number):
        return fallback
    return f"{number:.{int(precision)}f}"


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


def _make_shaded_mars_facecolors(xs: np.ndarray, ys: np.ndarray, zs: np.ndarray, camera_km: np.ndarray) -> np.ndarray:
    """Return RGBA face colors with a clear shadow on the right-hand side of the visible disk."""
    camera_km = np.asarray(camera_km, dtype=float)
    _, right, _ = camera_frame_centered_on_mars(camera_km)
    # sun_dir = -_unit_vector(right)  # light from the left, shadow on the right as seen by the camera
    sun_dir = np.array([-1.0, 0.0, 0.0], dtype=float)

    normals = np.stack([xs, ys, zs], axis=-1)
    norm = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = normals/np.clip(norm, np.finfo(float).eps, None)
    illum = np.sum(normals*sun_dir[None, None, :], axis=-1)
    illum = np.clip(illum, -1.0, 1.0)

    base = np.array([0.80, 0.48, 0.26], dtype=float)
    lit = 0.90 + 0.25*np.clip(illum, 0.0, 1.0)
    dark = 0.22 + 0.20*np.clip(illum, -1.0, 0.0)  # darker on the night side
    shade = np.where(illum >= 0.0, lit, dark)

    rgb = np.clip(base[None, None, :]*shade[..., None], 0.0, 1.0)
    alpha = np.full(xs.shape, 0.92, dtype=float)
    return np.dstack([rgb, alpha])


def plot_mars_detected_3d_view(
    geometry: dict,
    out_path: str | Path,
    planet_radius_km: float = MARS_RADIUS_KM,
    camera_altitude_km: float = 5720.0,
    zenith_angle_rad: float | None = None,
    det_mag_cut: float = 2,
    title: str | None = None,):
    """Create a 3D Mars, camera, and exact-FPS meteor sampling view."""
    out_path = str(out_path)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(
        111,
        projection="3d",
        computed_zorder=False,
    )
    ax.grid(True)

    R = float(planet_radius_km)
    cam = np.asarray(geometry["camera_km"], dtype=float)
    u = np.linspace(0, 2*np.pi, 60)
    v = np.linspace(0, np.pi, 40)
    xs = R*np.outer(np.cos(u), np.sin(v))
    ys = R*np.outer(np.sin(u), np.sin(v))
    zs = R*np.outer(np.ones_like(u), np.cos(v))
    facecolors = _make_shaded_mars_facecolors(xs, ys, zs, cam)
    ax.plot_surface(
        xs, ys, zs,
        rstride=1, cstride=1, linewidth=0,
        facecolors=facecolors, shade=False,
    )

    ax.xaxis.set_pane_color((0, 0, 0, 1))
    ax.yaxis.set_pane_color((0, 0, 0, 1))
    ax.zaxis.set_pane_color((0, 0, 0, 1))

    dots = np.asarray(geometry.get("sampled_positions_km", []), dtype=float)
    dot_app_mag = np.asarray(geometry.get("sampled_app_mag", []), dtype=float)
    detected_mask = np.asarray(
        geometry.get("sampled_detected_mask", np.ones(len(dots), dtype=bool)),
        dtype=bool,
    )
    n = min(len(dots), len(dot_app_mag), len(detected_mask))
    dots = dots[:n]
    dot_app_mag = dot_app_mag[:n]
    detected_mask = detected_mask[:n]
    dot_sizes = magnitude_marker_sizes(dot_app_mag, min_size=12.0, max_size=90.0)

    peak = np.asarray(geometry["peak_position_km"], dtype=float)
    sc = None

    # Physically visible samples: filled, magnitude-coloured circles.
    if np.any(detected_mask):
        # make sure that the sactter is zorder to be alwasys visible on top of the Mars surface
        sc = ax.scatter(
            dots[detected_mask, 0], dots[detected_mask, 1], dots[detected_mask, 2],
            s=dot_sizes[detected_mask],
            c=dot_app_mag[detected_mask],
            cmap="inferno_r",
            alpha=0.95,
            depthshade=False,
            label="Detected camera frames",
            zorder=10
        )

        ax.plot(
            [cam[0], peak[0]], [cam[1], peak[1]], [cam[2], peak[2]],
            color="0.75", ls="--", lw=1.1, alpha=0.8,
        )

    # Below-LM samples remain hollow. Draw a thin white ring over a slightly
    # wider black ring so the markers are visible on both light and dark parts
    # of Mars without requiring a black plot background.
    invisible_mask = ~detected_mask
    if np.any(invisible_mask):
        ax.scatter(
            dots[invisible_mask, 0], dots[invisible_mask, 1], dots[invisible_mask, 2],
            s=dot_sizes[invisible_mask],
            facecolors="none",
            edgecolors="dimgray",
            linewidths=2.4,
            alpha=0.95,
            depthshade=False,
            label="Below camera limiting magnitude",
            zorder=9
        )
        ax.scatter(
            dots[invisible_mask, 0], dots[invisible_mask, 1], dots[invisible_mask, 2],
            s=dot_sizes[invisible_mask],
            facecolors="none",
            edgecolors="dimgray",
            linewidths=1.1,
            alpha=1.0,
            depthshade=False,
            zorder=10
        )



    if sc is not None:
        cb = plt.colorbar(sc, ax=ax, fraction=0.036, pad=0.08)
        cb.set_label("Apparent magnitude")
        cb.ax.invert_yaxis()

    ax.scatter(
        [cam[0]], [cam[1]], [cam[2]],
        color="tab:blue", edgecolors="black", linewidths=0.5,
        s=80, label="Camera",zorder=11
    )

    pad = R + float(camera_altitude_km) + 500.0
    x_limits = (-pad*0.15, pad)
    y_limits = (-pad*0.6, pad*0.6)
    z_limits = (-pad*0.45, pad*0.45)
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    ax.set_zlim(*z_limits)
    ax.set_box_aspect((
        x_limits[1] - x_limits[0],
        y_limits[1] - y_limits[0],
        z_limits[1] - z_limits[0],
    ))

    ax.set_xlabel("X [km]")
    ax.set_ylabel("Y [km]")
    ax.set_zlabel("Z [km]")

    if title is None:
        title = "Mars meteor camera sampling"
    ax.set_title(title)
    ax.view_init(elev=18, azim=-58)

    ax.legend(loc="upper right", fontsize=9)

    detectable = bool(geometry.get("is_physically_detectable", False))
    zc_deg = None if zenith_angle_rad is None else np.degrees(float(zenith_angle_rad))
    label_lines = [
        f'Detection status: {"VISIBLE" if detectable else "NOT VISIBLE"}',
        f'Camera LM: {_format_optional_number(geometry.get("limiting_app_mag"), 2)}',
        f'Peak m_app: {geometry["peak_app_mag"]:.2f}',
        f'Camera altitude: {camera_altitude_km:.0f} km',
        f'Meteor speed: {geometry.get("meteor_speed_kms", np.nan):.2f} km/s',
        f'Mars orbital speed: {geometry.get("mars_orbital_speed_kms", np.nan):.2f} km/s',
        f'Central angle: {geometry.get("central_angle_deg", np.nan):.2f} deg',
        f'Camera frames shown: {geometry.get("camera_frame_count", len(dots))}',
        f'Physically detected frames: {geometry.get("physical_detected_frame_count", 0)}',
    ]
    if zc_deg is not None:
        label_lines.insert(0, f"Zenith angle z_c: {zc_deg:.2f} deg")
    label_lines.append(f'Track azimuth: {geometry["traj_azimuth_deg"]:.1f} deg')

    fig.text(
        0.02, 0.02,
        "\n".join(label_lines),
        ha="left", va="bottom", fontsize=10, color="black",
        bbox=dict(facecolor="white", edgecolor="0.7", alpha=0.9),
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
    """Return exact-FPS meteor points and their physical visibility status."""
    positions = np.asarray(geometry.get("sampled_positions_km", []), dtype=float)
    app_mag = np.asarray(geometry.get("sampled_app_mag", []), dtype=float)
    abs_mag = np.asarray(geometry.get("sampled_abs_mag", []), dtype=float)
    alt_km = np.asarray(geometry.get("sampled_heights_km", []), dtype=float)
    time_s = np.asarray(geometry.get("sampled_time_s", []), dtype=float)
    range_km = np.asarray(geometry.get("sampled_range_km", []), dtype=float)
    detected_mask = np.asarray(
        geometry.get("sampled_detected_mask", np.ones(len(positions), dtype=bool)),
        dtype=bool,
    )

    n = min(
        len(positions), len(app_mag), len(abs_mag), len(alt_km),
        len(time_s), len(range_km), len(detected_mask),
    )
    if n == 0:
        raise ValueError("No camera-frame meteor points available for FoV plotting.")

    positions = positions[:n]
    app_mag = app_mag[:n]
    abs_mag = abs_mag[:n]
    alt_km = alt_km[:n]
    time_s = time_s[:n]
    range_km = range_km[:n]
    detected_mask = detected_mask[:n]

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
        "detected_mask": detected_mask[idx],
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
    title: str | None = None,) -> str:
    """Plot Mars FoV, camera-frame samples, and the apparent-magnitude series."""
    out_path = str(out_path)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    mars_img = load_or_make_mars_disk_image(mars_image_path)
    dots = sample_detected_dots_for_fov(geometry, n_dots=n_dots)

    camera_km = np.asarray(geometry["camera_km"], dtype=float)
    x_deg, y_deg = project_points_to_mars_centered_camera(dots["positions_km"], camera_km)
    sizes = magnitude_marker_sizes(dots["app_mag"], min_size=10.0, max_size=80.0)
    detected_mask = np.asarray(dots["detected_mask"], dtype=bool)

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

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.8, 1.05], wspace=0.15)
    ax = fig.add_subplot(gs[0, 0])
    side_gs = gs[0, 1].subgridspec(2, 1, height_ratios=[1.0, 1.35], hspace=0.12)
    ax_mag = fig.add_subplot(side_gs[0, 0])
    ax_table = fig.add_subplot(side_gs[1, 0])

    ax.set_facecolor("black")
    ax.imshow(
        mars_img,
        extent=[-mars_radius_deg, mars_radius_deg, -mars_radius_deg, mars_radius_deg],
        origin="lower",
        zorder=1,
    )
    ax.add_patch(plt.Circle(
        (0, 0), mars_radius_deg,
        facecolor="none", edgecolor="white", linewidth=1.0, zorder=2,
    ))

    sc = None
    if np.any(detected_mask):
        sc = ax.scatter(
            x_deg[detected_mask], y_deg[detected_mask],
            s=sizes[detected_mask],
            c=dots["app_mag"][detected_mask],
            cmap="inferno_r",
            linewidths=0.5,
            alpha=0.95,
            zorder=3,
            label="Detected camera frames",
        )

    invisible_mask = ~detected_mask
    if np.any(invisible_mask):
        ax.scatter(
            x_deg[invisible_mask], y_deg[invisible_mask],
            s=sizes[invisible_mask],
            facecolors="none",
            edgecolors="dimgray",
            linewidths=1.25,
            alpha=0.98,
            zorder=4,
            label="Below camera limiting magnitude",
        )

    if sc is not None:
        colorbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        colorbar.ax.invert_yaxis()
        colorbar.set_label("Apparent magnitude", fontsize=9)

    ax.set_xlim(-half_x - 1, half_x + 1)
    ax.set_ylim(-half_y - 1, half_y + 1)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.20)
    ax.set_xlabel("Horizontal angle from Mars center [deg]")
    ax.set_ylabel("Vertical angle from Mars center [deg]")
    ax.set_title(title or "Mars-centered camera FoV")

    sampled_time = np.asarray(dots["time_s"], dtype=float)
    sampled_app_mag = np.asarray(dots["app_mag"], dtype=float)

    detectable = bool(geometry.get("is_physically_detectable", False))
    visible_start = geometry.get("detection_onset_time_s")
    if detectable and visible_start is not None and np.isfinite(float(visible_start)):
        sampled_t_rel = sampled_time - float(visible_start)
        time_label = "Time since LM crossing [s]"
    else:
        sampled_t_rel = sampled_time - float(sampled_time[0])
        time_label = "Time from first sampled frame [s]"

    # Filled points for actual detections; open black circles for below-LM frames.
    if np.any(detected_mask):
        ax_mag.scatter(
            sampled_t_rel[detected_mask], sampled_app_mag[detected_mask],
            s=sizes[detected_mask],
            c=sampled_app_mag[detected_mask],
            cmap="inferno_r",
            edgecolors="black", linewidths=0.5, alpha=0.9,
            label="Detected frames",
        )
    if np.any(invisible_mask):
        ax_mag.scatter(
            sampled_t_rel[invisible_mask], sampled_app_mag[invisible_mask],
            s=sizes[invisible_mask],
            facecolors="none",
            edgecolors="dimgray",
            linewidths=1.2,
            alpha=0.95,
            label="Below LM",
        )

    # Always retain the physical camera detection-limit line.
    limiting_mag = geometry.get("limiting_app_mag")
    if limiting_mag is not None and np.isfinite(float(limiting_mag)):
        ax_mag.axhline(
            float(limiting_mag), color="black", linestyle="--", linewidth=1.2,
            label=f"LM = {float(limiting_mag):.1f}",
        )

    
    delta_threshold = geometry.get("delta_app_mag_threshold")
    if (
        delta_threshold is not None
        and limiting_mag is not None
        and np.isfinite(float(delta_threshold))
        and np.isfinite(float(limiting_mag))
        and float(delta_threshold) < float(limiting_mag)
    ):
        ax_mag.axhline(
            float(delta_threshold), color="0.45", linestyle=":", linewidth=1.2,
            label=f"Peak + Delta m = {float(delta_threshold):.1f}",
        )

    ax_mag.invert_yaxis()
    ax_mag.set_xlabel(time_label)
    # ax_mag.set_ylabel("Apparent magnitude")
    # make te ticks smaller
    ax_mag.tick_params(axis="both", which="major", labelsize=6)
    ax_mag.set_title("Sampled apparent-magnitude light curve")
    ax_mag.grid(alpha=0.3)
    ax_mag.legend(fontsize=8)

    ax_table.axis("off")
    info = [
        "FoV / Detection summary",
        f'Detection status: {"VISIBLE" if detectable else "NOT VISIBLE"}',
        f"Mars angular diameter: {2*mars_radius_deg:.2f} deg",
        f"Camera FPS: {geometry.get('camera_fps', np.nan):.2f}",
        f"Camera LM: {_format_optional_number(geometry.get('limiting_app_mag'), 2)}",
        f"Peak M_abs: {geometry['peak_abs_mag']:.2f}",
        f"Peak m_app: {geometry['peak_app_mag']:.2f}",
        f"Meteor speed: {geometry.get('meteor_speed_kms', np.nan):.2f} km/s",
        f"Visible duration (LM): {geometry.get('visible_duration_s', 0.0):.3f} s",
        f"Frame interval: {geometry.get('camera_frame_dt_s', np.nan):.4f} s",
        f"Camera frames shown: {geometry.get('camera_frame_count', len(sampled_time))}",
        f"Physically detected frames: {geometry.get('physical_detected_frame_count', 0)}",
    ]
    # add the delta-m if the delta-m is below the limiting magnitude
    if (
        delta_threshold is not None
        and limiting_mag is not None
        and np.isfinite(float(delta_threshold))
        and np.isfinite(float(limiting_mag))
        and float(delta_threshold) < float(limiting_mag)
    ):
        info.append(f"Delta-m design: {geometry.get('design_assumption_criteria', 'disabled')}")
        info.append(f"Delta-m duration: {_format_optional_number(geometry.get('design_delta_duration_s'), 3)} s")
        info.append(f"Delta-m expected frames: {_format_optional_number(geometry.get('design_delta_expected_frames_at_fps'), 2)}")
        info.append(f"Delta-m actual detected frames: {geometry.get('design_delta_frame_count', 0)}")

    if focal_length_mm is not None:
        info.append(f"Focal length: {float(focal_length_mm):.2f} mm")
    if aperture_mm is not None:
        info.append(f"Aperture: {float(aperture_mm):.2f} mm")
        if focal_length_mm is not None and float(aperture_mm) > 0:
            info.append(f"f/{float(focal_length_mm)/float(aperture_mm):.2f}")

    ax_table.text(
        0.0, 0.9, "\n".join(info),
        transform=ax_table.transAxes,
        ha="left", va="top", family="monospace", fontsize=9,
        bbox=dict(facecolor="white", edgecolor="0.7", alpha=0.95),
    )

    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="upper right", fontsize=9)

    print(f"Saving FoV plot to: {out_path}")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    if not os.path.exists(out_path):
        raise RuntimeError(f"FoV plot was not created: {out_path}")
    return out_path



def _case_insensitive_file_map(directory: str | Path) -> dict[str, Path]:
    """Return files in one directory keyed by lowercase filename."""
    directory = Path(directory)
    if not directory.is_dir():
        return {}
    return {
        path.name.lower(): path
        for path in directory.iterdir()
        if path.is_file()
    }


def _case_insensitive_child_directory(
    directory: str | Path,
    child_name: str,
) -> Path | None:
    """Return a direct child directory with a case-insensitive name match."""
    directory = Path(directory)
    if not directory.is_dir():
        return None
    target = str(child_name).lower()
    for path in directory.iterdir():
        if path.is_dir() and path.name.lower() == target:
            return path
    return None


def discover_meteor_input_pairs(input_folder: str | Path) -> list[dict[str, Any]]:
    """
    Recursively find ``*_trajectory.pickle`` files and their matching fit JSON.

    Pairing is case-insensitive. Supported layouts, in priority order, are:

    1. ``EVENT_sim_fit_latest.json`` beside the trajectory pickle;
    2. ``EVENT_sim_fit.json`` beside the trajectory pickle;
    3. ``fit_plots/EVENT_sim_fit_dynesty_BestGuess.json`` below the trajectory
       directory (used by the smaller-meteoroid Dynesty solutions);
    4. ``EVENT_sim_fit_dynesty_BestGuess.json`` beside the trajectory pickle.

    Event outputs are always written beside the trajectory pickle, not inside
    ``fit_plots``.
    """
    root = Path(input_folder).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Input folder does not exist: {root}")

    trajectory_suffix = "_trajectory.pickle"
    pairs: list[dict[str, Any]] = []

    for pickle_path in sorted(root.rglob("*")):
        if not pickle_path.is_file():
            continue
        lower_name = pickle_path.name.lower()
        if not lower_name.endswith(trajectory_suffix):
            continue

        event_name = pickle_path.name[:-len(trajectory_suffix)]
        event_lower = event_name.lower()
        event_directory = pickle_path.parent
        files_by_lower = _case_insensitive_file_map(event_directory)
        fit_plots_directory = _case_insensitive_child_directory(
            event_directory,
            "fit_plots",
        )
        fit_plots_files = _case_insensitive_file_map(fit_plots_directory) if fit_plots_directory else {}

        candidates = [
            (
                files_by_lower.get(event_lower + "_sim_fit_latest.json"),
                "sim_fit_latest",
                "event_directory",
            ),
            (
                files_by_lower.get(event_lower + "_sim_fit.json"),
                "sim_fit",
                "event_directory",
            ),
            (
                fit_plots_files.get(
                    event_lower + "_sim_fit_dynesty_bestguess.json"
                ),
                "sim_fit_dynesty_BestGuess",
                "fit_plots",
            ),
            (
                files_by_lower.get(
                    event_lower + "_sim_fit_dynesty_bestguess.json"
                ),
                "sim_fit_dynesty_BestGuess",
                "event_directory",
            ),
        ]

        json_path = None
        json_kind = None
        json_layout = None
        for candidate_path, candidate_kind, candidate_layout in candidates:
            if candidate_path is not None:
                json_path = candidate_path
                json_kind = candidate_kind
                json_layout = candidate_layout
                break

        pairs.append({
            "event_name": event_name,
            "pickle_path": pickle_path,
            "json_path": json_path,
            "json_kind": json_kind,
            "json_layout": json_layout,
            "event_directory": event_directory,
            "fit_plots_directory": fit_plots_directory,
        })

    return pairs


def _finite_float(value: Any, default: float = np.nan) -> float:
    """Convert a scalar to float, returning default for invalid values."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if np.isfinite(result) else float(default)


def _first_finite_attribute(obj: Any, names: Iterable[str], default: float = np.nan) -> float:
    """Read the first finite scalar attribute available on an object."""
    for name in names:
        if hasattr(obj, name):
            value = _finite_float(getattr(obj, name), default=np.nan)
            if np.isfinite(value):
                return value
    return float(default)


def save_pickle_data(data: Any, out_path: str | Path) -> str:
    """Save arbitrary Python data with the highest available pickle protocol."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as fh:
        pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return str(out_path)


def load_pickle_data(path: str | Path) -> Any:
    """Load a Python pickle created by this script."""
    with Path(path).open("rb") as fh:
        return pickle.load(fh)


def make_event_data_record(
    event_name: str,
    pickle_path: str | Path,
    json_path: str | Path,
    output_dir: str | Path,
    earth_const: Any,
    mars_const: Any,
    geometry: dict,
    output_paths: dict[str, str],
    json_kind: str | None = None,
    json_layout: str | None = None,
) -> dict[str, Any]:
    """Build the per-event cache record used by folder mode and summary plots."""
    initial_mass_kg = _first_finite_attribute(
        earth_const,
        ("m_init", "initial_mass", "mass_init", "m0"),
    )
    if not np.isfinite(initial_mass_kg):
        initial_mass_kg = _first_finite_attribute(
            mars_const,
            ("m_init", "initial_mass", "mass_init", "m0"),
        )

    earth_velocity_kms = _first_finite_attribute(
        earth_const, ("v_init", "initial_velocity", "v0")
    ) / 1000.0
    mars_velocity_kms = _first_finite_attribute(
        mars_const, ("v_init", "initial_velocity", "v0")
    ) / 1000.0

    detected_frames = int(geometry.get("physical_detected_frame_count", 0) or 0)
    is_detected = bool(
        geometry.get("is_physically_detectable", False)
        and detected_frames > 0
    )

    dynamic_mapping = getattr(mars_const, "dynamic_pressure_mapping", {}) or {}
    start_mapping = dynamic_mapping.get("global_erosion_start") or {}
    change_mapping = dynamic_mapping.get("global_erosion_change") or {}
    fragmentation_mapping = dynamic_mapping.get("fragmentation_entries") or []

    return {
        "record_format_version": 2,
        "event_name": str(event_name),
        "pickle_path": str(Path(pickle_path).resolve()),
        "json_path": str(Path(json_path).resolve()),
        "json_kind": json_kind,
        "json_layout": json_layout,
        "event_directory": str(Path(output_dir).resolve()),
        "initial_mass_kg": float(initial_mass_kg),
        "earth_velocity_kms": float(earth_velocity_kms),
        "mars_velocity_kms": float(mars_velocity_kms),
        "peak_absolute_magnitude": _finite_float(geometry.get("peak_abs_mag")),
        "peak_apparent_magnitude": _finite_float(geometry.get("peak_app_mag")),
        "limiting_apparent_magnitude": _finite_float(geometry.get("limiting_app_mag")),
        "is_physically_detected": is_detected,
        "detected_frame_count": detected_frames,
        "camera_frame_count": int(geometry.get("camera_frame_count", 0) or 0),
        "visible_duration_s": _finite_float(geometry.get("visible_duration_s"), 0.0),
        "central_angle_deg": _finite_float(geometry.get("central_angle_deg")),
        "longitude_regime": geometry.get("longitude_regime"),
        "fragmentation_entry_count": int(len(fragmentation_mapping)),
        "earth_erosion_height_start_km": (
            _finite_float(start_mapping.get("earth_height_m")) / 1000.0
        ),
        "mars_erosion_height_start_km": (
            _finite_float(start_mapping.get("mars_height_m")) / 1000.0
        ),
        "erosion_start_dynamic_pressure_pa": _finite_float(
            start_mapping.get("target_dyn_pressure_pa")
        ),
        "earth_erosion_height_change_km": (
            _finite_float(change_mapping.get("earth_height_m")) / 1000.0
        ),
        "mars_erosion_height_change_km": (
            _finite_float(change_mapping.get("mars_height_m")) / 1000.0
        ),
        "erosion_change_dynamic_pressure_pa": _finite_float(
            change_mapping.get("target_dyn_pressure_pa")
        ),
        "dynamic_pressure_mapping": dynamic_mapping,
        "outputs": dict(output_paths),
        # Retain all arrays and metadata needed to remake event-level figures.
        "geometry": geometry,
    }


def plot_batch_velocity_mass_summary(
    event_records: Iterable[dict[str, Any]],
    out_path: str | Path,
) -> str:
    """
    Plot Mars velocity versus initial mass for all processed meteors.

    Point colour gives peak apparent magnitude, the adjacent integer is the
    number of physically detected frames, and a red open ring marks detections.
    """
    records = list(event_records)
    valid_records = [
        record for record in records
        if np.isfinite(_finite_float(record.get("mars_velocity_kms")))
        and np.isfinite(_finite_float(record.get("initial_mass_kg")))
        and _finite_float(record.get("initial_mass_kg")) > 0.0
    ]
    if not valid_records:
        raise ValueError("No events have finite Mars velocity and positive initial mass.")

    velocity = np.asarray([
        _finite_float(record.get("mars_velocity_kms")) for record in valid_records
    ], dtype=float)
    mass = np.asarray([
        _finite_float(record.get("initial_mass_kg")) for record in valid_records
    ], dtype=float)
    peak_app_mag = np.asarray([
        _finite_float(record.get("peak_apparent_magnitude")) for record in valid_records
    ], dtype=float)
    frame_count = np.asarray([
        int(record.get("detected_frame_count", 0) or 0) for record in valid_records
    ], dtype=int)
    detected = np.asarray([
        bool(record.get("is_physically_detected", False)) for record in valid_records
    ], dtype=bool)

    fig, ax = plt.subplots(figsize=(10, 7))

    finite_colour = np.isfinite(peak_app_mag)
    scatter = None
    if np.any(finite_colour):
        scatter = ax.scatter(
            velocity[finite_colour], mass[finite_colour],
            c=peak_app_mag[finite_colour], cmap="inferno_r",
            s=75, edgecolors="0.25", linewidths=0.45,
            alpha=0.92, zorder=2,
        )
    if np.any(~finite_colour):
        ax.scatter(
            velocity[~finite_colour], mass[~finite_colour],
            color="0.65", s=75, edgecolors="0.25", linewidths=0.45,
            alpha=0.92, zorder=2,
        )

    if np.any(detected):
        ax.scatter(
            velocity[detected], mass[detected],
            s=135, facecolors="none", edgecolors="red",
            linewidths=1.7, zorder=4,
        )

    for x_value, y_value, frames in zip(velocity, mass, frame_count):
        ax.annotate(
            str(int(frames)),
            (x_value, y_value),
            xytext=(5, 5), textcoords="offset points",
            fontsize=8, color="black", zorder=5,
        )

    if scatter is not None:
        colorbar = fig.colorbar(scatter, ax=ax, pad=0.02)
        colorbar.set_label("Peak apparent magnitude")
        colorbar.ax.invert_yaxis()

    ax.set_yscale("log")
    ax.set_xlabel("Initial velocity at Mars [km/s]")
    ax.set_ylabel("Initial meteoroid mass [kg]")
    ax.set_title("Mars meteor detectability summary")
    ax.grid(True, which="both", alpha=0.25)

    legend_handles = [
        Line2D(
            [0], [0], marker="o", linestyle="none",
            markerfacecolor="none", markeredgecolor="red",
            markeredgewidth=1.7, markersize=10,
            label="At least one frame above camera LM",
        ),
        Line2D(
            [0], [0], linestyle="none", color="none",
            label="Number beside point = detected frames",
        ),
    ]
    ax.legend(handles=legend_handles, loc="best", fontsize=9)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def _csv_scalar(value: Any) -> Any:
    """Convert optional/numpy values into stable scalar CSV cells."""
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return int(bool(value))
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if np.isfinite(value) else ""
    return value


def save_batch_summary_csv(
    event_records: Iterable[dict[str, Any]],
    out_path: str | Path,
) -> str:
    """Save a flat, analysis-ready CSV containing one row per meteor."""
    columns = [
        "event_name",
        "json_kind",
        "json_layout",
        "pickle_path",
        "json_path",
        "event_directory",
        "initial_mass_kg",
        "earth_velocity_kms",
        "mars_velocity_kms",
        "peak_absolute_magnitude",
        "peak_apparent_magnitude",
        "limiting_apparent_magnitude",
        "is_physically_detected",
        "detected_frame_count",
        "camera_frame_count",
        "visible_duration_s",
        "central_angle_deg",
        "longitude_regime",
        "fragmentation_entry_count",
        "earth_erosion_height_start_km",
        "mars_erosion_height_start_km",
        "erosion_start_dynamic_pressure_pa",
        "earth_erosion_height_change_km",
        "mars_erosion_height_change_km",
        "erosion_change_dynamic_pressure_pa",
        "lightcurve_plot",
        "plot_3d",
        "fov_plot",
        "detection_summary_json",
        "event_data_pickle",
    ]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for record in event_records:
            outputs = record.get("outputs", {}) or {}
            row = {
                column: record.get(column, "")
                for column in columns
            }
            for output_name in (
                "lightcurve_plot",
                "plot_3d",
                "fov_plot",
                "detection_summary_json",
                "event_data_pickle",
            ):
                row[output_name] = outputs.get(output_name, "")
            writer.writerow({key: _csv_scalar(value) for key, value in row.items()})

    return str(out_path)


def _normalised_path_key(path: str | Path | None) -> str:
    """Return a case-insensitive absolute path key for cache matching."""
    if path is None or str(path).strip() == "":
        return ""
    try:
        return str(Path(path).expanduser().resolve()).casefold()
    except Exception:
        return str(path).casefold()


def _pair_cache_key(pickle_path: str | Path, json_path: str | Path) -> tuple[str, str]:
    return (
        _normalised_path_key(pickle_path),
        _normalised_path_key(json_path),
    )


def _record_matches_pair(record: dict[str, Any], pair: dict[str, Any]) -> bool:
    """Return True when a cached record belongs to the discovered input pair."""
    return _pair_cache_key(
        record.get("pickle_path"),
        record.get("json_path"),
    ) == _pair_cache_key(pair.get("pickle_path"), pair.get("json_path"))


def _cached_record_is_usable(record: dict[str, Any], pair: dict[str, Any]) -> bool:
    """Check pair identity and require new erosion metadata for BestGuess fits."""
    if not isinstance(record, dict) or not _record_matches_pair(record, pair):
        return False

    if pair.get("json_kind") == "sim_fit_dynesty_BestGuess":
        return (
            int(record.get("record_format_version", 0) or 0) >= 2
            and "dynamic_pressure_mapping" in record
            and "earth_erosion_height_start_km" in record
        )

    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one fitted meteor or recursively process a folder of trajectory/fit pairs "
            "on Mars/custom atmosphere."
        )
    )

    parser.add_argument(
        "--input",
        default=r"C:\Users\maxiv\Documents\UWO\Papers\5)METEORCAM-Strawman\cz_Fireball",
        help=(
            "Optional input folder for recursive batch processing, or a single "
            "*_trajectory.pickle file. Folder mode pairs files case-insensitively "
            "with *_sim_fit_latest.json or *_sim_fit.json beside it, or with "
            "fit_plots/*_sim_fit_dynesty_BestGuess.json for small-body fits."
        ),
    )
    parser.add_argument(
        "--batch-summary-name",
        default="Mars_detection_batch_summary.pkl",
        help="Filename for the reusable batch summary pickle saved in the input folder.",
    )
    parser.add_argument(
        "--batch-plot-name",
        default="Mars_detection_velocity_mass_summary.png",
        help="Filename for the combined velocity-mass detectability plot.",
    )
    parser.add_argument(
        "--batch-csv-name",
        default="Mars_detection_velocity_mass_summary.csv",
        help="Filename for the flat per-meteor CSV exported in folder mode.",
    )
    parser.add_argument(
        "--rebuild-batch",
        action="store_true",
        help="Ignore existing global and per-event pickle caches and rerun all events.",
    )

    parser.add_argument(
        "--pickle",
        default=r"C:\Users\maxiv\Documents\UWO\Papers\5)METEORCAM-Strawman\cz_Fireball\EN181125_040337\EN181125_040337_trajectory.pickle",
        help="Trajectory pickle file. Camera data are loaded through observation_data(...).",
    )
    parser.add_argument(
        "--json",
        default=r"C:\Users\maxiv\Documents\UWO\Papers\5)METEORCAM-Strawman\cz_Fireball\EN181125_040337\EN181125_040337_sim_fit_latest.json",
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
    parser.add_argument(
        "--mars-P-0m",
        dest="mars_P_0m",
        type=float,
        default=MARS_P0M,
        help="Mars meteor zero-magnitude power in watts. Default: 1500 W.",
    )
    parser.add_argument("--h-kill-km", type=float, default=6.0, help="Kill height for the new-atmosphere simulation.")
    parser.add_argument("--v-init-kms", type=float, default=None, help="Override new-atmosphere v_init. If omitted, orbit estimate is tried, otherwise Earth v_init is kept.")

    # Detected-segment and 3D view options.
    parser.add_argument("--det-mag-cut", type=float, default=2.0, help="Delta-m design-comparison window relative to the apparent peak. This does not control physical detection.")
    parser.add_argument("--no-delta-mag-cut", action="store_true", help="Disable calculation of the peak + delta-m design-comparison statistics. Physical LM detection is unchanged.")
    parser.add_argument("--limiting-app-mag", type=float, default=4.0, help="Physical camera limiting apparent magnitude. Detection uses m_app <= LM.")
    parser.add_argument("--camera-fps", type=float, default=15.0, help="Camera FPS used to sample detected points for 3D/FoV plots.")
    parser.add_argument("--sampling-seed", type=int, default=None, help="Optional random seed for the sub-frame camera phase. Omit it to draw a new phase on every run.")
    parser.add_argument("--postpeak-only", action="store_true", help="If set, keep only the detected segment from the peak onward. By default the full detected interval is kept, including the onset before peak and the fading branch after peak.")
    parser.add_argument("--camera-altitude-km", type=float, default=5720.0, help="Camera altitude above Mars surface for the 3D image.")
    parser.add_argument("--central-angle-deg", type=float, default=20.0, help="Manual central angle from the sub-camera point. Used when speed-based longitude is disabled.")
    parser.add_argument(
        "--physically-motivated-longitude",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use the speed-based signed dusk-to-dawn longitude mapping. "
            "Use --no-physically-motivated-longitude for the manual central angle."
        ),
    )
    parser.add_argument("--mars-orbital-speed-kms", type=float, default=MARS_ORBITAL_SPEED_KMS, help="Mars orbital speed used by the speed-based longitude mapping.")
    parser.add_argument("--traj-azimuth-deg", type=float, default=30.0, help="Azimuth of the local meteor ground track in the 3D view.")
    parser.add_argument("--selected-altitude-km", type=float, default=None, help="Altitude snapshot for the 3D image. Default is the detected-peak altitude.")
    parser.add_argument("--n-detected-dots", type=int, default=None, help="Optional display cap for FoV dots. It does not change the exact camera FPS sampling.")
    parser.add_argument("--mars-image", default=r"C:\Users\maxiv\WMPG-repoMAX\Code\DynNestSampl\Mars\Dark.png", help="Optional Mars image file. If omitted, a synthetic Mars disk is generated.")
    parser.add_argument("--sensor-width-mm", type=float, default=None, help="Camera sensor width in mm. If omitted, FoV exactly fits Mars.")
    parser.add_argument("--sensor-height-mm", type=float, default=None, help="Camera sensor height in mm. If omitted, FoV exactly fits Mars.")
    parser.add_argument("--focal-length-mm", type=float, default=None, help="Camera focal length in mm. Used with sensor size to compute FoV.")
    parser.add_argument("--aperture-mm", type=float, default=None, help="Optional aperture diameter in mm, used for annotation.")
    parser.add_argument("--plot", action="store_true", help="Save the lightcurve, 3D view, and FoV plots. By default they are saved.")

    return parser.parse_args()



def run_single_event(args: argparse.Namespace) -> dict[str, Any]:
    """Run the complete existing workflow for one trajectory/JSON pair."""
    if not args.pickle or not args.json:
        raise ValueError("Single-event mode requires both --pickle and --json.")

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(args.pickle)) or "."
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    base_name = args.base_name or Path(args.pickle).stem.replace("_trajectory", "")

    earth_const = load_best_fit_constants(args.json)
    if _is_finite_number(args.P_0m_prior):
        earth_const.P_0m = float(args.P_0m_prior)
    obs_data = build_observation_from_pickle(args, fitted_p0m=float(earth_const.P_0m))
    trajectories = load_trajectory_list_from_obs(obs_data)
    earth_const = patch_constants_from_observation(earth_const, obs_data)

    print(f"Using Earth/observed P_0m: {float(earth_const.P_0m):.1f} W")
    print(f"Using Mars P_0m: {float(args.mars_P_0m):.1f} W")

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
        _erosion_beg_dyn_press_change,
    ) = build_dynamic_pressure_trigger_const(
        best_guess_obj_plot=best_guess_obj_plot,
        best_guess_obj_plot_mars=best_guess_obj_plot_mars,
        best_guess_cost_mars=best_guess_cost_mars,
    )

    print("Running new-atmosphere same-dynamic-pressure simulation...")
    best_guess_obj_plot_mars_dyn_press = run_model_raw(best_guess_cost_mars_dyn_press)
    if not args.plot:
        print("Skipping lightcurve plot.")
        lightcurve_path = None
    
    else:
        lightcurve_path = plot_lightcurve_earth_vs_mars_dyn_pressure(
            obs_data=obs_data,
            best_guess_obj_plot=best_guess_obj_plot,
            best_guess_obj_plot_mars=best_guess_obj_plot_mars,
            best_guess_obj_plot_mars_dyn_press=best_guess_obj_plot_mars_dyn_press,
            heightsame_dynpress_mars=heightsame_dynpress_mars,
            heightsame_dynpress_change_mars=heightsame_dynpress_change_mars,
            output_dir=output_dir,
            base_name=base_name,
        )
        print(f"Saved plot: {lightcurve_path}")

    simulation_segment = extract_simulation_segment(best_guess_obj_plot_mars_dyn_press)
    full_geom = build_detected_track_geometry(
        simulation_segment,
        zenith_angle_rad=float(best_guess_obj_plot_mars_dyn_press.const.zenith_angle),
        planet_radius_km=float(args.planet_radius_km),
        camera_altitude_km=float(args.camera_altitude_km),
        central_angle_deg=float(args.central_angle_deg),
        traj_azimuth_deg=float(args.traj_azimuth_deg),
        selected_altitude_km=args.selected_altitude_km,
        physically_motivated_longitude=bool(args.physically_motivated_longitude),
        meteor_speed_kms=float(best_guess_obj_plot_mars_dyn_press.const.v_init)/1000.0,
        mars_orbital_speed_kms=float(args.mars_orbital_speed_kms),
    )
    detected_geom = apply_apparent_detection_and_camera_sampling(
        full_geom,
        det_mag_cut=args.det_mag_cut,
        limiting_app_mag=args.limiting_app_mag,
        use_delta_mag_cut=not args.no_delta_mag_cut,
        from_peak_only=bool(args.postpeak_only),
        camera_fps=float(args.camera_fps),
        max_dots=args.n_detected_dots,
        sampling_seed=args.sampling_seed,
    )

    detection_summary_path = os.path.join(
        output_dir, base_name + "_Mars_detection_summary.json"
    )
    save_detection_summary(detected_geom, detection_summary_path)
    print(f"Saved detection summary: {detection_summary_path}")

    out3d = os.path.join(output_dir, base_name + "_Mars_detected_3D.png")
    if not args.plot:
        print(f"Skipping 3D plot: {out3d}")
    else:
        plot_mars_detected_3d_view(
            detected_geom,
            out3d,
            planet_radius_km=float(args.planet_radius_km),
            camera_altitude_km=float(args.camera_altitude_km),
            zenith_angle_rad=float(best_guess_obj_plot_mars_dyn_press.const.zenith_angle),
            det_mag_cut=float(args.det_mag_cut),
            title=f"{base_name}: detected Mars meteor",
        )
        print(f"Saved 3D plot: {out3d}")

    outfov = os.path.join(output_dir, base_name + "_Mars_detected_FoV.png")
    if not args.plot:
        print(f"Skipping FoV plot: {outfov}")
    else:
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

    output_paths = {
        "lightcurve_plot": str(lightcurve_path),
        "plot_3d": str(out3d),
        "fov_plot": str(outfov),
        "detection_summary_json": str(detection_summary_path),
    }

    record = make_event_data_record(
        event_name=base_name,
        pickle_path=args.pickle,
        json_path=args.json,
        output_dir=output_dir,
        earth_const=best_guess_obj_plot.const,
        mars_const=best_guess_obj_plot_mars_dyn_press.const,
        geometry=detected_geom,
        output_paths=output_paths,
        json_kind=getattr(args, "json_kind", None),
        json_layout=getattr(args, "json_layout", None),
    )
    record["run_configuration"] = {
        "camera_altitude_km": float(args.camera_altitude_km),
        "camera_fps": float(args.camera_fps),
        "limiting_apparent_magnitude": float(args.limiting_app_mag),
        "delta_m_design_cut": None if args.no_delta_mag_cut else float(args.det_mag_cut),
        "mars_P_0m_W": float(args.mars_P_0m),
        "planet_radius_km": float(args.planet_radius_km),
        "planet_g0_mps2": float(args.planet_g0),
        "physically_motivated_longitude": bool(args.physically_motivated_longitude),
        "manual_central_angle_deg": float(args.central_angle_deg),
        "trajectory_azimuth_deg": float(args.traj_azimuth_deg),
        "sampling_seed": args.sampling_seed,
    }

    event_pickle_path = os.path.join(
        output_dir, base_name + "_Mars_detection_data.pkl"
    )
    record["outputs"]["event_data_pickle"] = event_pickle_path
    save_pickle_data(record, event_pickle_path)
    print(f"Saved event data: {event_pickle_path}")

    print(f"Peak detected M_abs: {detected_geom['peak_abs_mag']:.2f}")
    print(f"Peak detected m_app: {detected_geom['peak_app_mag']:.2f}")
    print(f"Peak detected range: {detected_geom['peak_range_km']:.1f} km")
    return record


def run_batch_folder(args: argparse.Namespace, input_folder: str | Path) -> dict[str, Any]:
    """
    Process or reload every valid meteor pair under an input folder.

    Existing global/per-event pickle data are reused, but discovery is always
    repeated so newly added Dynesty BestGuess events are included automatically.
    """
    root = Path(input_folder).expanduser().resolve()
    summary_path = root / args.batch_summary_name
    batch_plot_path = root / args.batch_plot_name
    batch_csv_path = root / args.batch_csv_name

    discovered = discover_meteor_input_pairs(root)
    valid_pairs = [pair for pair in discovered if pair["json_path"] is not None]
    missing_json = [pair for pair in discovered if pair["json_path"] is None]

    print(
        f"Found {len(discovered)} trajectory pickle(s): "
        f"{len(valid_pairs)} matched pair(s), {len(missing_json)} without matching JSON."
    )

    previous_records: dict[tuple[str, str], dict[str, Any]] = {}
    if summary_path.exists() and not args.rebuild_batch:
        try:
            previous_summary = load_pickle_data(summary_path)
            previous_events = (
                list(previous_summary.get("events", []))
                if isinstance(previous_summary, dict)
                else list(previous_summary)
            )
            previous_records = {
                _pair_cache_key(record.get("pickle_path"), record.get("json_path")): record
                for record in previous_events
                if isinstance(record, dict)
            }
            print(
                f"Loaded {len(previous_records)} previous event record(s) from "
                f"{summary_path}."
            )
        except Exception as exc:
            print(f"Could not load existing batch summary; rebuilding records: {exc}")

    events: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for index, pair in enumerate(valid_pairs, start=1):
        event_name = str(pair["event_name"])
        event_dir = Path(pair["event_directory"])
        event_cache = event_dir / f"{event_name}_Mars_detection_data.pkl"
        pair_key = _pair_cache_key(pair["pickle_path"], pair["json_path"])

        print(f"\n[{index}/{len(valid_pairs)}] {event_name}")
        print(f"  Pickle: {pair['pickle_path']}")
        print(f"  JSON:   {pair['json_path']}")
        print(f"  Kind:   {pair.get('json_kind')} ({pair.get('json_layout')})")

        record = None
        if not args.rebuild_batch and event_cache.exists():
            try:
                cached_record = load_pickle_data(event_cache)
                if _cached_record_is_usable(cached_record, pair):
                    record = cached_record
                    print(f"  Loaded cached event data: {event_cache}")
                elif isinstance(cached_record, dict):
                    print("  Local cache predates BestGuess erosion metadata; rerunning event.")
            except Exception as exc:
                print(f"  Could not load local cache: {exc}")

        if record is None and not args.rebuild_batch:
            cached_record = previous_records.get(pair_key)
            if cached_record is not None and _cached_record_is_usable(cached_record, pair):
                record = cached_record
                print("  Loaded event data from global summary cache.")

        if record is not None:
            # Backfill source classification for records written by an older script.
            record["json_kind"] = pair.get("json_kind")
            record["json_layout"] = pair.get("json_layout")
            events.append(record)
            continue

        event_args = copy.copy(args)
        event_args.input = None
        event_args.pickle = str(pair["pickle_path"])
        event_args.json = str(pair["json_path"])
        event_args.output_dir = str(event_dir)
        event_args.base_name = event_name
        event_args.json_kind = pair.get("json_kind")
        event_args.json_layout = pair.get("json_layout")

        try:
            record = run_single_event(event_args)
            events.append(record)
        except Exception as exc:
            failure = {
                "event_name": event_name,
                "pickle_path": str(pair["pickle_path"]),
                "json_path": str(pair["json_path"]),
                "json_kind": str(pair.get("json_kind") or ""),
                "error": f"{type(exc).__name__}: {exc}",
            }
            failures.append(failure)
            print(f"  FAILED: {failure['error']}")

    for pair in missing_json:
        failures.append({
            "event_name": str(pair["event_name"]),
            "pickle_path": str(pair["pickle_path"]),
            "json_path": "",
            "json_kind": "",
            "error": (
                "No matching *_sim_fit_latest.json, *_sim_fit.json, or "
                "fit_plots/*_sim_fit_dynesty_BestGuess.json."
            ),
        })

    summary = {
        "format_version": 2,
        "input_folder": str(root),
        "event_count": len(events),
        "failure_count": len(failures),
        "events": events,
        "failures": failures,
        "outputs": {
            "summary_pickle": str(summary_path),
            "velocity_mass_plot": str(batch_plot_path),
            "summary_csv": str(batch_csv_path),
        },
    }
    save_pickle_data(summary, summary_path)
    print(f"Saved batch summary: {summary_path}")

    if events:
        plot_batch_velocity_mass_summary(events, batch_plot_path)
        print(f"Saved batch velocity-mass plot: {batch_plot_path}")
        save_batch_summary_csv(events, batch_csv_path)
        print(f"Saved batch CSV: {batch_csv_path}")

    return summary


def main() -> None:
    args = parse_args()

    if args.input is not None:
        input_path = Path(args.input).expanduser()
        if input_path.is_dir():
            run_batch_folder(args, input_path)
            return
        if input_path.is_file():
            args.pickle = str(input_path)
            # When a single trajectory is supplied through --input, try the same
            # case-insensitive pair-discovery logic before using --json.
            pair_candidates = discover_meteor_input_pairs(input_path.parent)
            resolved = [
                pair for pair in pair_candidates
                if Path(pair["pickle_path"]).resolve() == input_path.resolve()
                and pair["json_path"] is not None
            ]
            if resolved:
                args.json = str(resolved[0]["json_path"])
                args.base_name = str(resolved[0]["event_name"])
                args.output_dir = str(input_path.parent)
                args.json_kind = resolved[0].get("json_kind")
                args.json_layout = resolved[0].get("json_layout")
        else:
            raise FileNotFoundError(f"Input path does not exist: {input_path}")

    run_single_event(args)


if __name__ == "__main__":
    main()
