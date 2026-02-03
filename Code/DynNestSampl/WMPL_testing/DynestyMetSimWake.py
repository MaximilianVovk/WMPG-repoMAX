import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate

# --- WMPL imports (reuse as much as possible) ---
from wmpl.Utils.Pickling import loadPickle
from wmpl.MetSim.GUI import WakeContainter, plotWakeOverview, loadWakeFile, loadConstants, SimulationResults, saveConstants
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




def WakeNormalizeAlignReduce(wake_ref, wake_container_ref, peak_region=20, max_len_shift=50, normalization_method="peak", align_method="correlate", lenMax=100, interp=True):
    """ Extract the wake from the simulation results. 
    
    Arguments:
        wake_ref: [SimulationResults object] Simulation results.
        wake_container_ref: [WakeContainer object] Wake container.

    Keyword arguments:
        peak_region: [float] Region around the peak to use for the wake normalization (m). If None, the whole
            wake will be used.
        max_len_shift: [float] Maximum length shift allowed when aligning the observed and simulated wakes (m).
            If the shift is larger than this, the wake will not be aligned.
        normalization_method: [str] Method to normalize the wake intensity. Options are "area" or "peak".
        align_method: [str] Method to align the observed and simulated wakes. Options are "correlate" or "peak".
        lenMax: [float] Maximum length of the wake to consider (m).
        interp: [bool] Whether to interpolate the simulated wake to the observed wake lengths.
    """

    # Extract the wake points from the containers
    len_ref_array = []
    wake_ref_intensity_array = []
    for wake_pt in wake_container_ref.points:
        len_ref_array.append(wake_pt.leading_frag_length)
        wake_ref_intensity_array.append(wake_pt.intens_sum)

    len_ref_array = np.array(len_ref_array)
    wake_ref_intensity_array = np.array(wake_ref_intensity_array)

    # Normalize the wake intensity so the areas are equal between the observed and simulated wakes
    if normalization_method == 'area':
        ### Normalize the wake intensity by area ###

        # Only take the points +/- peak_region m from the maximum intensity
        obslen_ref_max_intens = len_ref_array[np.argmax(wake_ref_intensity_array)]
        obslen_ref_filter = np.abs(len_ref_array - obslen_ref_max_intens) < peak_region
        obs_area_ref = np.trapz(wake_ref_intensity_array[obslen_ref_filter], len_ref_array[obslen_ref_filter])
        simlen_ref_max_intens = wake_ref.length_array[np.argmax(wake_ref.wake_luminosity_profile)]
        simlen_ref_filter = np.abs(wake_ref.length_array - simlen_ref_max_intens) < peak_region
        sim_area_ref = np.trapz(wake_ref.wake_luminosity_profile[simlen_ref_filter], wake_ref.length_array[simlen_ref_filter])

        # Normalize the observed wake intensity
        wake_ref_intensity_array = wake_ref_intensity_array*sim_area_ref/obs_area_ref

    elif normalization_method == 'peak':
        ### Normalize the wake intensity by peak luminosity ###

        simulated_peak_luminosity = np.max(wake_ref.wake_luminosity_profile)

        # Handle cases when the simulation doesn't exist
        if simulated_peak_luminosity is None:
            simulated_peak_luminosity = 0

        wake_ref_intensity_array *= simulated_peak_luminosity/np.max(wake_ref_intensity_array)


    # Align wake by peaks
    if align_method == 'peak':
        ### Align the observed and simulated wakes by their peaks ###

        # Store the max values
        simulated_peak_length = wake_ref.length_array[np.argmax(wake_ref.wake_luminosity_profile)]

        # Find the length of the peak intensity
        peak_len = len_ref_array[np.argmax(wake_ref_intensity_array)]

        # Offset lengths
        len_ref_array -= peak_len + simulated_peak_length

    elif align_method == 'correlate':
        ### Align the observed and simulated wakes by correlation ###
        
        # Interpolate the model values and sample them at observed points
        sim_wake_interp = scipy.interpolate.interp1d(wake_ref.length_array, \
            wake_ref.wake_luminosity_profile, bounds_error=False, fill_value=0)
        model_wake_obs_len_sample = sim_wake_interp(-len_ref_array)

        # Correlate the wakes and find the shift
        wake_shift = np.argmax(np.correlate(model_wake_obs_len_sample, wake_ref_intensity_array, \
            "full")) + 1

        # Find the index of the zero observed length
        obs_len_zero_indx = np.argmin(np.abs(len_ref_array))

        # Compute the length shift
        len_shift = len_ref_array[(obs_len_zero_indx + wake_shift)%len(model_wake_obs_len_sample)]

        # If the shift is larger than the maximum allowed, do not align the wakes
        if np.abs(len_shift) > max_len_shift:
            len_shift = 0

        # Add the offset to the observed length
        len_ref_array += len_shift

    sim_wake_ref_length = wake_ref.length_array
    sim_wake_ref_luminosity = wake_ref.wake_luminosity_profile

    if lenMax != 0:
        # Limit the length array to the specified maximum length
        length_filter = (len_ref_array < lenMax) & (-len_ref_array < wake_ref.length_array.max())
        # Limit the simulated wake to the specified maximum length
        length_filter_sim = sim_wake_ref_length > -lenMax

        len_ref_array = len_ref_array[length_filter]
        wake_ref_intensity_array = wake_ref_intensity_array[length_filter]
        sim_wake_ref_length = sim_wake_ref_length[length_filter_sim]
        sim_wake_ref_luminosity = sim_wake_ref_luminosity[length_filter_sim]

    if interp == True:
        # Interpolate the simulated wake to the observed wake lengths so they have the same length array
        sim_wake_interp_final = scipy.interpolate.interp1d(sim_wake_ref_length, \
            sim_wake_ref_luminosity, bounds_error=False, fill_value=0)
        sim_wake_ref_luminosity = sim_wake_interp_final(-len_ref_array)
        sim_wake_ref_length = -len_ref_array

    return (
        sim_wake_ref_length, sim_wake_ref_luminosity, # Return the simulated wake at the ref ht
        -len_ref_array, wake_ref_intensity_array # Return the observed wake at the ref ht
    )



def plotWakeOverviewOptions(sr, wake_containers, plot_dir, event_name, site_id=None, wake_samples=8,
                     first_height_ratio=0.1, final_height_ratio=0.75, peak_region=20, 
                     normalization_method="area", align_method="correlate", lenMax = 0, noise_guess=1):
    """ Plot the wake at a range of heights showing the match between the observed and simulated wake. 

    Arguments:
        sr: [SimulationResults object] Simulation results.
        wake_containers: [list of WakeContainer objects] List of wake containers.
        plot_dir: [str] Path to the directory where the plots will be saved.
        event_name: [str] Name of the event.

    Keyword arguments:
        site_id: [int] Name of the site where the meteor was observed. 1 for Tavistock and 2 for Elginfield.
            If None, both will be taken.
        wake_samples: [int] Number of wake samples to plot.
        first_height_ratio: [float] Fraction of the wake height to probe for the first sample. 0 is the height
            when tracking began and 1 is the height when the tracking stopped.
        final_height_ratio: [float] Fraction of the wake height to probe for the last sample. 0 is the height
            when tracking began and 1 is the height when the tracking stopped.
        peak_region: [float] Region around the peak to use for the wake normalization (m). If None, the whole
            wake will be used.

    """

    wake_indices = [i for i, w in enumerate(sr.wake_results) if w is not None]

    # delete the ones that are not in wake_indices
    for i in sorted(set(range(len(sr.wake_results))) - set(wake_indices), reverse=True):
        del sr.wake_results[i]
    # count the wake_indices True values
    if len(wake_indices)<wake_samples:
        wake_samples = len(wake_indices)

    # # filter the sr.wake_results only take those in wake_indices discard the rest
    # sr.wake_results = [sr.wake_results[i] for i in wake_indices]

    # Make N plots for wake_samples heights
    height_fractions = np.linspace(first_height_ratio, final_height_ratio, wake_samples)

    # Set up the plot
    fig, axes = plt.subplots(figsize=(8, 8), nrows=wake_samples, sharex=True)

    # Length at which text is plotted
    txt_len_coord = 50 # m

    if lenMax == 0:
        interp_flag = False
    else:   
        interp_flag = True

    # Loop through the heights
    for i, height_fraction in enumerate(height_fractions):
        
        if peak_region is None:
            peak_region = np.inf

        # Filter wake containers by site
        if site_id is not None:
            wake_containers = [wake_container for wake_container in wake_containers 
                            if wake_container.site_id == site_id]

        # Get a list of all heights in the wake
        wake_heights = [wake_container.points[0].ht for wake_container in wake_containers]

        if len(wake_indices) == wake_samples:
            # Compute the probing heights
            ht_ref = sr.leading_frag_height_arr[wake_indices[i]] # sr.wake_results[len(height_fractions)-1 - i].leading_frag_length

        else:  
            # Compute the range of heights
            ht_range = np.max(wake_heights) - np.min(wake_heights)

            # Compute the probing heights
            ht_ref = np.max(wake_heights) - height_fraction*ht_range

        # Find the container which are closest to reference height of the wake fraction
        ht_ref_idx = np.argmin(np.abs(np.array(wake_heights) - ht_ref))

        # Get the two containers with observations
        wake_container_ref = wake_containers[ht_ref_idx]

        # Find indices where the wake result is not None
        valid_wake_indices = [i for i, w in enumerate(sr.wake_results) if w is not None]
        
        if not valid_wake_indices:
            # Should ideally handle this gracefully, but for now fallback to previous behavior which might error or return None
            wake_res_indx_ref = np.nanargmin(np.abs(ht_ref - sr.brightest_height_arr))
        else:
            # Find the wake index closest to the given wake height, considering only valid wakes
            closest_idx_in_valid = np.argmin(np.abs(ht_ref - sr.brightest_height_arr[valid_wake_indices]))
            wake_res_indx_ref = valid_wake_indices[closest_idx_in_valid]

        # Get the wake results
        (
            wake_len_array, wake_lum_array, # Return the simulated wake at the ref ht
            obs_len_array, obs_lum_array # Return the observed wake at the ref ht
        ) = WakeNormalizeAlignReduce(sr.wake_results[wake_res_indx_ref], wake_container_ref, 
                                     normalization_method=normalization_method, align_method=align_method, 
                                     lenMax=lenMax, interp=interp_flag)

        # Plot the observed wake
        axes[i].plot(obs_len_array, obs_lum_array, color="black", linestyle="--", linewidth=1, alpha=0.75)

        # Plot the simulated wake
        axes[i].plot(wake_len_array, wake_lum_array, color="black", linestyle="solid", linewidth=1, alpha=0.75)

        # Get the height label as halfway between the peak model wake and 0
        txt_ht = np.max(wake_lum_array)/2

        if len(wake_len_array) == len(obs_len_array):
            # axes[i].scatter(obs_len_array, obs_lum_array, color="red", s=5, alpha=0.75)
            for jj in range(len(obs_len_array)):
                axes[i].plot([obs_len_array[jj],wake_len_array[jj]] , [obs_lum_array[jj], wake_lum_array[jj]], ":xr", alpha=0.75, markersize=1, linewidth=0.2)
            # compute the loglikehood between the two wakes
            log_likelihood_wake = np.nansum(-0.5*np.log(2*np.pi*noise_guess**2) - 0.5/(noise_guess**2)*(obs_lum_array - wake_lum_array) ** 2)
            txt_len_coord = -80  # m
            # Set the height label
            axes[i].text(txt_len_coord, txt_ht, "{:.1f} km\nLogL={:.2f}".format(ht_ref/1000, log_likelihood_wake), fontsize=8, ha="left", va="center")
        
        else:
            axes[i].text(txt_len_coord, txt_ht, "{:.1f} km".format(ht_ref/1000), fontsize=8, ha="right", va="center")

    # Remove Y ticks on all axes
    for ax in axes:
        ax.set_yticks([])

    # Set the X label
    axes[-1].set_xlabel("Length (m)", fontsize=12)
    
    if len(wake_len_array) == len(obs_len_array):
        # Set X axis limits
        axes[-1].set_xlim(-100, max(np.max(wake_len_array), np.max(obs_len_array)))
    else:
        # Set X axis limits
        axes[-1].set_xlim(-200, 80)

    # Invert X axis
    axes[-1].invert_xaxis()

    # Remove vertical space between subplots
    plt.subplots_adjust(hspace=0)

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Save the plot
    plt.savefig(os.path.join(plot_dir, "{:s}_wake_overview_{:s}_{:s}.png".format(event_name, normalization_method, align_method)), dpi=300, 
                bbox_inches="tight")

    # Close the plot
    plt.close(fig)

    return os.path.join(plot_dir, "{:s}_wake_overview_{:s}_{:s}.png".format(event_name, normalization_method, align_method))

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

    axes[-1].set_xlabel("Length (m)", fontsize=12)
    if xlim is not None:
        axes[-1].set_xlim(*xlim)
    axes[-1].invert_xaxis()

    plt.subplots_adjust(hspace=0)

    out_path = os.path.join(plot_dir, f"{event_name}_wake_overview_obs.png")
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path




def save_sr_wake_results_json(sr, out_json_path, metadata=None, include_fragments=True):
    """
    Save sr.wake_results (list of Wake or None) to a JSON file.

    Args:
        sr: SimulationResults-like object with sr.wake_results
        out_json_path: path to write JSON
        metadata: optional dict with extra info (event name, timestamp, etc.)
        include_fragments: if True, saves minimal per-fragment info (length, lum, ...)

    Returns:
        out_json_path
    """

    def to_json_safe(x):
        # numpy -> python
        if isinstance(x, np.integer):
            return int(x)
        if isinstance(x, np.floating):
            x = float(x)

        # arrays -> lists
        if isinstance(x, np.ndarray):
            return [to_json_safe(v) for v in x.tolist()]

        # float: NaN/inf -> None
        if isinstance(x, float):
            return x if math.isfinite(x) else None

        return x

    def extract_fragment_dict(frag):
        # Keep this minimal & robust; add fields if you want.
        d = {}
        for k in ("length", "lum", "mass", "ht", "vel", "t"):
            if hasattr(frag, k):
                d[k] = to_json_safe(getattr(frag, k))
        return d

    payload = {
        "version": 1,
        "metadata": metadata or {},
        "wake_results": []
    }

    wake_results = getattr(sr, "wake_results", None)
    if wake_results is None:
        raise ValueError("sr has no attribute 'wake_results'.")

    for i, w in enumerate(wake_results):
        if w is None:
            payload["wake_results"].append(None)
            continue

        # Extract PSF-related constants safely
        const = getattr(w, "const", None)
        wake_psf = getattr(const, "wake_psf", None) if const is not None else None
        wake_psf_weights = getattr(const, "wake_psf_weights", None) if const is not None else None

        entry = {
            "index": i,
            "leading_frag_length": to_json_safe(getattr(w, "leading_frag_length", None)),
            "length_array": to_json_safe(getattr(w, "length_array", None)),
            "wake_luminosity_profile": to_json_safe(getattr(w, "wake_luminosity_profile", None)),
            "length_points": to_json_safe(getattr(w, "length_points", None)),
            "luminosity_points": to_json_safe(getattr(w, "luminosity_points", None)),
            "const": {
                "wake_psf": to_json_safe(wake_psf),
                "wake_psf_weights": to_json_safe(wake_psf_weights),
            }
        }

        if include_fragments:
            frag_list = getattr(w, "frag_list", [])
            entry["fragments"] = [extract_fragment_dict(frag) for frag in frag_list]

        payload["wake_results"].append(entry)

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return out_json_path


def compute_wake_noise_by_altitude(
    wake_containers,
    site_id=None,
    x_threshold=-100.0,
    tail_len=100.0,
    poly_order=2,
    keep_percent=95.0,
    min_points=8,
    use_plot_convention=True,
    save_plots=False,
    output_dir=None, ):
    """
    Compute per-altitude noise from wake_containers by:
      1) selecting the 'noise region' (x <= x_threshold; else last tail_len of x-range),
      2) fitting a polynomial to intensity vs x,
      3) using trimmed median absolute residual as noise.

    Args:
        wake_containers: list of WakeContainter (WMPL)
        site_id: int or None
        x_threshold: float, e.g. -100 (meters)
        tail_len: float, used if x_threshold not covered
        poly_order: int (2 for quadratic)
        keep_percent: float, e.g. 95 -> keep central 95% residuals
        min_points: minimum points needed to fit; else fallback
        use_plot_convention: bool
            - True  => x = -leading_frag_length (matches your plotting convention in overview)
            - False => x = leading_frag_length (raw)

    Returns:
        results: list of dicts with keys:
            ht_m, ht_km, site_id, frame_n,
            noise_median_abs_resid,
            resid_percentiles, npts_total, npts_used,
            poly_coeffs (highest power first),
            region_used ("threshold" or "tail")
        overall_noise: robust median across altitudes (median of per-altitude noise)
    """

    # Filter by site if requested
    if site_id is not None:
        wake_containers = [wc for wc in wake_containers if int(wc.site_id) == int(site_id)]

    if not wake_containers:
        raise ValueError("No wake_containers available after filtering.")

    # Percentile cut for trimming outliers
    lo = (100.0 - keep_percent) / 2.0
    hi = 100.0 - lo

    results = []

    for wc in wake_containers:
        # Altitude for this wake (WMPL uses points[0].ht as container height)
        ht_m = float(wc.points[0].ht) if wc.points else np.nan

        # Build x,y
        if use_plot_convention:
            x = np.array([-pt.leading_frag_length for pt in wc.points], dtype=float)
        else:
            x = np.array([pt.leading_frag_length for pt in wc.points], dtype=float)

        y = np.array([pt.intens_sum for pt in wc.points], dtype=float)

        # Clean
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]

        if x.size < max(min_points, poly_order + 2):
            # Not enough points -> skip or fallback
            # Fallback: robust scale of y (trimmed)
            if x.size == 0:
                continue
            y_lo, y_hi = np.percentile(y, [lo, hi])
            y_trim = y[(y >= y_lo) & (y <= y_hi)]
            noise = float(np.median(np.abs(y_trim - np.median(y_trim))))
            results.append({
                "ht_m": ht_m,
                "ht_km": ht_m / 1000.0,
                "site_id": int(wc.site_id),
                "frame_n": int(wc.frame_n),
                "noise_median_abs_resid": noise,
                "resid_percentiles": (np.nan, np.nan),
                "npts_total": int(ok.sum()),
                "npts_used": int(y_trim.size),
                "poly_coeffs": None,
                "region_used": "fallback_trimmed_y",
            })
            continue

        # --- Choose noise region ---
        x_min = float(np.min(x))

        # Primary: x <= x_threshold
        mask_thr = x <= x_threshold

        if np.any(mask_thr):
            mask = mask_thr
            region_used = "threshold"
        else:
            # Fallback: last tail_len of available x-range at the most-negative end:
            # take x in [x_min, x_min + tail_len]
            mask = x <= (x_min + tail_len)
            region_used = "tail"

        xr, yr = x[mask], y[mask]

        # Ensure enough points to fit
        if xr.size < max(min_points, poly_order + 2):
            # Expand tail if needed
            mask = x <= (x_min + max(tail_len, 2 * tail_len))
            xr, yr = x[mask], y[mask]
            region_used = "tail_expanded"

        if xr.size < max(min_points, poly_order + 2):
            # Still not enough -> fallback robust on yr
            y_lo, y_hi = np.percentile(yr, [lo, hi]) if yr.size else (np.nan, np.nan)
            y_trim = yr[(yr >= y_lo) & (yr <= y_hi)] if yr.size else np.array([])
            noise = float(np.median(np.abs(y_trim - np.median(y_trim)))) if y_trim.size else np.nan
            results.append({
                "ht_m": ht_m,
                "ht_km": ht_m / 1000.0,
                "site_id": int(wc.site_id),
                "frame_n": int(wc.frame_n),
                "noise_median_abs_resid": noise,
                "resid_percentiles": (np.nan, np.nan),
                "npts_total": int(ok.sum()),
                "npts_used": int(y_trim.size),
                "poly_coeffs": None,
                "region_used": "fallback_trimmed_y",
            })
            continue

        # --- Fit polynomial on region ---
        # Sort by x for numerical stability (not required but nice)
        idx = np.argsort(xr)
        xr, yr = xr[idx], yr[idx]

        coeffs = np.polyfit(xr, yr, deg=poly_order)
        yhat = np.polyval(coeffs, xr)
        resid = yr - yhat

        # --- Trim outliers of residuals ---
        r_lo, r_hi = np.percentile(resid, [lo, hi])
        resid_trim = resid[(resid >= r_lo) & (resid <= r_hi)]

        # Noise = median absolute residual (robust)
        noise = float(np.median(np.abs(resid_trim))) if resid_trim.size else float(np.median(np.abs(resid)))

        if save_plots:
            # Diagnostic plot
            plt.figure(figsize=(6, 4))
            plt.scatter(x, y, color="gray", s=10, label="Data")
            plt.scatter(xr, yr, color="blue", s=20, label="Fit region")
            x_fit = np.linspace(np.min(xr), np.max(xr), 200)
            y_fit = np.polyval(coeffs, x_fit)
            plt.plot(x_fit, y_fit, color="red", linewidth=2, label="Poly fit")
            plt.title(f"Wake noise at ht={ht_m/1000:.2f} km\nNoise={noise:.4f}, Region={region_used}")
            plt.xlabel("Length (m)")
            plt.ylabel("Intensity sum")
            # invert x-axis
            plt.gca().invert_xaxis()
            plt.legend()
            plot_path = os.path.join(output_dir or "wake_noise_plots", f"wake_noise_ht_{int(ht_m)}m_site_{wc.site_id}_frame_{wc.frame_n}.png")
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)
            plt.savefig(plot_path, dpi=200)
            plt.close()
            print("Saved wake noise plot:", plot_path)

        results.append({
            "ht_m": ht_m,
            "ht_km": ht_m / 1000.0,
            "site_id": int(wc.site_id),
            "frame_n": int(wc.frame_n),
            "noise_median_abs_resid": noise,
            "resid_percentiles": (float(r_lo), float(r_hi)),
            "npts_total": int(ok.sum()),
            "npts_used": int(resid_trim.size),
            "poly_coeffs": [float(c) for c in coeffs],  # [a,b,c] for quadratic
            "region_used": region_used,
        })

    # Overall robust noise summary across altitudes
    noises = np.array([r["noise_median_abs_resid"] for r in results], dtype=float)
    noises = noises[np.isfinite(noises)]
    overall_noise = float(np.median(noises)) if noises.size else np.nan

    return results, overall_noise



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

    # create the plot directory
    os.makedirs(plot_dir, exist_ok=True)

    per_alt, overall_noise = compute_wake_noise_by_altitude(
        wake_containers,
        site_id=None,          # or 1 / 2
        x_threshold=-100.0,
        tail_len=100.0,
        poly_order=2,
        keep_percent=95.0,
        use_plot_convention=True,
        save_plots=False,
        output_dir=plot_dir,
    )

    print("Wake noise by altitude:")
    for alt_entry in per_alt:
        # print altitude and noise
        print(f"Altitude: {alt_entry['ht_km']:.2f} km, Noise: {alt_entry['noise_median_abs_resid']:.4f}")
    print("Overall noise (median across altitudes):", overall_noise)

    # check if there are .json files in the input_dir
    json_files = _find_files(input_dir, "*.json")
    if json_files and verbose:
        print(f"Note: Found {len(json_files)} .json files in input_dir. "
              f"These are not used by this function but may contain useful metadata.")
    for json_name in json_files:
        try:
            print(f"Trying to load constants from JSON: {json_name}")
            # check if "dt": 0.005,
            with open(json_name, "r", encoding="utf-8") as f:
                data = json.load(f)
            # check if dt is present as a key
            if "dt" not in data:
                if verbose:
                    print(f"Skipping {json_name} not correct type of file.")
                continue
            # Load the constants
            const, _ = loadConstants(json_name)
            const.dens_co = np.array(const.dens_co)
            const.wake_psf = [5] # PSF width in meters

            const.wake_heights = [102000, 101000, 100000]

            # Run the simulation
            frag_main, results_list, wake_results = runSimulation(const, compute_wake=True)
            sr = SimulationResults(const, frag_main, results_list, wake_results)
            # # print the altitudes of the wake_results
            # wake_indices = [i for i, w in enumerate(sr.wake_results) if w is not None]
            # print(f"Wake result indices: {wake_indices}")
            # # print the from results_list the leading fragment heights
            # print(f"Wake result heights (m): {sr.leading_frag_height_arr[wake_indices]}")
            if verbose:
                print(f"Successfully ran MetSimErosion simulation from constants in: {json_name}")
            break  # successfully ran simulation
        except Exception as e:
            if verbose:
                print(f"Failed to run MetSimErosion simulation from {json_name}: {e}")

    # # join plot_dir and event_name for output json
    # out_json = save_sr_wake_results_json(
    #     sr,
    #     os.path.join(plot_dir, f"{event_name}_sr_wake_results.json"),
    #     metadata={"event": event_name},
    #     include_fragments=True
    # )


    # If sr provided, reuse WMPL plotWakeOverview directly
    if sr is not None:
        return plotWakeOverviewOptions(
            sr, wake_containers, plot_dir, event_name,
            site_id=site_id,
            wake_samples=wake_samples,
            first_height_ratio=first_height_ratio,
            final_height_ratio=final_height_ratio,
            peak_region=peak_region,
            normalization_method="peak",           # or  "area", "peak"
            align_method="correlate",              # or "none","correlate","peak"
            lenMax=100,
            noise_guess=overall_noise
        )
    else:
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
    input_dir = r"C:\Users\maxiv\Documents\UWO\Papers\0.4)Wake\test\20191023_091225_combined"
    output_dir = r"C:\Users\maxiv\Documents\UWO\Papers\0.4)Wake\test_plots"
    # extract the name of the event from the folder name
    name = os.path.basename(os.path.normpath(input_dir))
    out_png = make_wake_overview_png(
        input_dir=input_dir,
        plot_dir=output_dir,#os.path.join(input_dir, "wake_plots"),
        event_name=name,
        sr=None,      # pass SimulationResults here if you have it
        site_id=None,
        verbose=True
    )
    print("Saved:", out_png)
