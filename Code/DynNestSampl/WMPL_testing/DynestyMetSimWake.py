import os
import glob
import numpy as np
import matplotlib.pyplot as plt

# --- WMPL imports (reuse as much as possible) ---
from wmpl.Utils.Pickling import loadPickle
from wmpl.MetSim.GUI import WakeContainter, plotWakeOverview, loadWakeFile, loadConstants, SimulationResults
from wmpl.MetSim.MetSimErosion import Constants, runSimulation
# from wmpl.MetSim.GUI import (
#     altAz2RADec, geo2Cartesian, raDec2ECI, findClosestPoints, vectMag, cartesian2Geo
# )


# ============================================================
# 1) Find + load trajectory pickle (required)
# ============================================================

def _find_files(root_dir, pattern):
    return sorted(glob.glob(os.path.join(os.path.abspath(root_dir), "**", pattern), recursive=True))


def load_traj_from_dir(input_dir, verbose=True):
    cand = []
    for pat in ("**/*.pickle", "**/*.pkl", "**/*.pck", "**/*.pckl"):
        cand += _find_files(input_dir, pat)

    # newest first
    cand = sorted(set(cand), key=lambda p: os.path.getmtime(p), reverse=True)

    if verbose:
        print(f"Found {len(cand)} candidate pickle files under: {os.path.abspath(input_dir)}")

    for fp in cand:
        try:
            traj = loadPickle(*os.path.split(fp))
            if hasattr(traj, "observations") and hasattr(traj, "jdt_ref") and hasattr(traj, "rbeg_ele"):
                if verbose:
                    print(f"Loaded trajectory pickle: {fp}")
                return traj
        except Exception as e:
            if verbose:
                print(f"Not a trajectory / failed: {fp} ({e})")

    raise RuntimeError("No usable trajectory pickle found under input_dir.")

import json
import math
import numpy as np

def _to_builtin(x):
    """Convert numpy scalars/arrays + NaN/inf to JSON-safe builtin types."""
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        x = float(x)
    if isinstance(x, float):
        # JSON doesn't support NaN/inf; use None (or strings if you prefer)
        if not math.isfinite(x):
            return None
        return x
    if isinstance(x, (np.ndarray,)):
        return [_to_builtin(v) for v in x.tolist()]
    return x


def wake_container_to_dict(wc):
    """Serialize one WakeContainter into a JSON-friendly dict."""
    return {
        "site_id": int(wc.site_id),
        "frame_n": int(wc.frame_n),
        "points": [
            {
                "n": _to_builtin(p.n),
                "th": _to_builtin(p.th),
                "phi": _to_builtin(p.phi),
                "intens_sum": _to_builtin(p.intens_sum),
                "amp": _to_builtin(p.amp),
                "r": _to_builtin(p.r),
                "b": _to_builtin(p.b),
                "c": _to_builtin(p.c),
                "state_vect_dist": _to_builtin(p.state_vect_dist),
                "ht": _to_builtin(p.ht),
                "leading_frag_length": _to_builtin(getattr(p, "leading_frag_length", 0.0)),
            }
            for p in wc.points
        ],
    }


def wake_containers_to_json(wake_containers, out_path, metadata=None):
    """
    Save a list of WakeContainter objects to JSON.

    metadata: optional dict (event_name, traj_pickle_used, etc.)
    """
    payload = {
        "version": 1,
        "metadata": metadata or {},
        "wake_containers": [wake_container_to_dict(wc) for wc in wake_containers],
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return out_path


def plotWakeOverviewObservedOnly(wake_containers, plot_dir, event_name,
                                site_id=None, wake_samples=8,
                                first_height_ratio=0.1, final_height_ratio=0.75,
                                peak_region=20.0, xlim=(-200, 80), dpi=300):
    """
    Same visual style as WMPL plotWakeOverview, but observed-only.
    Labels show HEIGHT (km) from pt.ht computed during wake loading.
    """
    os.makedirs(plot_dir, exist_ok=True)

    # optional filter by site
    if site_id is not None:
        wake_containers = [wc for wc in wake_containers if int(wc.site_id) == int(site_id)]
    if not wake_containers:
        raise RuntimeError("No wake containers available (after filtering).")

    heights = np.array([wc.points[0].ht for wc in wake_containers], dtype=float)
    ht_min, ht_max = np.nanmin(heights), np.nanmax(heights)
    ht_range = ht_max - ht_min

    fracs = np.linspace(first_height_ratio, final_height_ratio, int(wake_samples))

    fig, axes = plt.subplots(figsize=(8, 8), nrows=len(fracs), sharex=True)

    for i, f in enumerate(fracs):
        ht_ref_target = ht_max - f * ht_range
        idx = int(np.nanargmin(np.abs(heights - ht_ref_target)))
        wc = wake_containers[idx]
        ht_ref = float(wc.points[0].ht)

        x = np.array([pt.leading_frag_length for pt in wc.points], dtype=float)
        y = np.array([pt.intens_sum for pt in wc.points], dtype=float)

        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        if x.size == 0:
            continue

        # normalize area near peak
        if peak_region is None:
            peak_region = np.inf
        xpk = x[np.argmax(y)]
        region = np.abs(x - xpk) < peak_region
        area = np.trapz(y[region], x[region]) if np.any(region) else np.trapz(y, x)
        if area != 0 and np.isfinite(area):
            y = y / area

        # WMPL convention: plot distance behind leading fragment with negative sign
        axes[i].plot(-x, y, color="black", linestyle="--", linewidth=1, alpha=0.85)
        axes[i].text(0.98, 0.55, f"{ht_ref/1000:.1f} km",
                     transform=axes[i].transAxes, ha="right", va="center", fontsize=8)
        axes[i].set_yticks([])

    axes[-1].set_xlabel("Distance behind leading fragment (m)", fontsize=12)
    if xlim is not None:
        axes[-1].set_xlim(*xlim)
    axes[-1].invert_xaxis()

    plt.subplots_adjust(hspace=0)

    out_path = os.path.join(plot_dir, f"{event_name}_wake_overview_obs.png")
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path

def make_wake_overview_png(input_dir, plot_dir=None, event_name=None, sr=None,
                           site_id=None, wake_samples=8,
                           first_height_ratio=0.1, final_height_ratio=0.75,
                           peak_region=20.0, verbose=True):

    input_dir = os.path.abspath(input_dir)
    if plot_dir is None:
        plot_dir = os.path.join(input_dir, "wake_plots")
    if event_name is None:
        event_name = os.path.basename(input_dir.rstrip(os.sep))

    wid_files = _find_files(input_dir, "wid_*.txt")
    if verbose:\
        print(f"Found {len(wid_files)} wid files under: {input_dir}")

    cand = []
    for pat in "**/*.pickle":
        cand += _find_files(input_dir, pat)

    # newest first
    cand = sorted(set(cand), key=lambda p: os.path.getmtime(p), reverse=True)

    if verbose:
        print(f"Found {len(cand)} candidate pickle files under: {os.path.abspath(input_dir)}")

    for fp in cand:
        try:
            traj = loadPickle(*os.path.split(fp))
            if hasattr(traj, "observations") and hasattr(traj, "jdt_ref") and hasattr(traj, "rbeg_ele"):
                if verbose:
                    print(f"Loaded trajectory pickle: {fp}")
            
            wake_containers = []
            for fp in wid_files:
                try:
                    wc = loadWakeFile(traj, fp)
                    # wc = loadWakeFile_fixed(traj, fp)
                    if wc is not None:
                        wake_containers.append(wc)
                except Exception as e:
                    if verbose:
                        print(f"\nFAILED loading {fp}: {e}")
            if wake_containers:
                break  # successfully loaded traj + some wakes

        except Exception as e:
            if verbose:
                print(f"Not a trajectory / failed: {fp} ({e})")

    if verbose:
        print(f"Loaded {len(wake_containers)} wake containers.")

    if not wake_containers:
        raise RuntimeError("No wake containers loaded (all wid files rejected or unreadable).")

    out_json = wake_containers_to_json(
        wake_containers,
        os.path.join(plot_dir, f"{event_name}_wakes.json"),
        metadata={"event": event_name}
    )
    print("Saved:", out_json)


    # check if there are .json files in the input_dir
    json_files = _find_files(input_dir, "*.json")
    if json_files and verbose:
        print(f"Note: Found {len(json_files)} .json files in input_dir. "
              f"These are not used by this function but may contain useful metadata.")
    for json_name in json_files:
        try:
            # Load the constants
            const, _ = loadConstants(json_name)
            const.dens_co = np.array(const.dens_co)

            # Run the simulation
            frag_main, results_list, wake_results = runSimulation(const, compute_wake=True)
            sr = SimulationResults(const, frag_main, results_list, wake_results)
            if verbose:
                print(f"Successfully ran MetSimErosion simulation from constants in: {json_name}")
            break  # successfully ran simulation
        except Exception as e:
            if verbose:
                print(f"Failed to run MetSimErosion simulation from {json_name}: {e}")

    # If sr provided, reuse WMPL plotWakeOverview directly
    if sr is not None:
        plotWakeOverview(
            sr, wake_containers, plot_dir, event_name,
            site_id=site_id,
            wake_samples=wake_samples,
            first_height_ratio=first_height_ratio,
            final_height_ratio=final_height_ratio,
            peak_region=peak_region,
        )
        return os.path.join(plot_dir, f"{event_name}_wake_overview.png")

    # Otherwise observed-only
    return plotWakeOverviewObservedOnly(
        wake_containers, plot_dir, event_name,
        site_id=site_id,
        wake_samples=wake_samples,
        first_height_ratio=first_height_ratio,
        final_height_ratio=final_height_ratio,
        peak_region=peak_region
    )


# ============================================================
# Example usage
# ============================================================

if __name__ == "__main__":
    input_dir = r"C:\Users\maxiv\Documents\UWO\Papers\0.4)Wake\test\20191023_091225"
    # extract the name of the event from the folder name
    name = os.path.basename(os.path.normpath(input_dir))
    out_png = make_wake_overview_png(
        input_dir=input_dir,
        plot_dir=input_dir,#os.path.join(input_dir, "wake_plots"),
        event_name=name,
        sr=None,      # pass SimulationResults here if you have it
        site_id=None,
        verbose=True
    )
    print("Saved:", out_png)
