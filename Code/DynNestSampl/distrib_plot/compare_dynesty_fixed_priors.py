import sys
import os

import numpy as np

# Add the parent directory to the sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from DynNestSapl_metsim import *


import os
import re
import math
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

EVENT_RE = re.compile(r"(?P<eid>\d{8}[_-]\d{6})")

# -----------------------------
# Variable label maps (user's)
# -----------------------------
variable_map = {
    'v_init': r"$v_0$ [km/s]",
    'zenith_angle': r"$z_c$ [rad]",
    'm_init': r"$m_0$ [kg]",
    'rho': r"$\rho$ [kg/m$^3$]",
    'sigma': r"$\sigma$ [kg/MJ]",
    'erosion_height_start': r"$h_e$ [km]",
    'erosion_coeff': r"$\eta$ [kg/MJ]",
    'erosion_mass_index': r"$s$",
    'erosion_mass_min': r"$m_{l}$ [kg]",
    'erosion_mass_max': r"$m_{u}$ [kg]",
    'erosion_height_change': r"$h_{e2}$ [km]",
    'erosion_coeff_change': r"$\eta_{2}$ [kg/MJ]",
    'erosion_rho_change': r"$\rho_{2}$ [kg/m$^3$]",
    'erosion_sigma_change': r"$\sigma_{2}$ [kg/MJ]",
    'noise_lag': r"$\sigma_{lag}$ [m]",
    'noise_lum': r"$\sigma_{lum}$ [W]",
    'eeucs': r"$E_s$ [MJ/m$^2$]",
    'eeum': r"$E_m$ [MJ/kg]",
    'eeucs_end': r"$E_{s\,end}$ [MJ/m$^2$]",
    'eeum_end': r"$E_{m\,end}$ [MJ/kg]"
}

variable_map_plot = {
    'v_init': r"$v_0$ [m/s]",
    'zenith_angle': r"$z_c$ [rad]",
    'm_init': r"$m_0$ [kg]",
    'rho': r"$\rho$ [kg/m$^3$]",
    'sigma': r"$\sigma$ [kg/J]",
    'erosion_height_start': r"$h_e$ [m]",
    'erosion_coeff': r"$\eta$ [kg/J]",
    'erosion_mass_index': r"$s$",
    'erosion_mass_min': r"$m_{l}$ [kg]",
    'erosion_mass_max': r"$m_{u}$ [kg]",
    'erosion_height_change': r"$h_{e2}$ [m]",
    'erosion_coeff_change': r"$\eta_{2}$ [kg/J]",
    'erosion_rho_change': r"$\rho_{2}$ [kg/m$^3$]",
    'erosion_sigma_change': r"$\sigma_{2}$ [kg/J]",
    'noise_lag': r"$\sigma_{lag}$ [m]",
    'noise_lum': r"$\sigma_{lum}$ [W]",
    'eeucs': r"$E_s$ [J/m$^2$]",
    'eeum': r"$E_m$ [J/kg]",
    'eeucs_end': r"$E_{s\,end}$ [J/m$^2$]",
    'eeum_end': r"$E_{m\,end}$ [J/kg]"
}

def _find_event_id_from_text(text: str) -> Optional[str]:
    m = re.search(r"(\d{8}[_-]\d{6})", str(text))
    if not m:
        return None
    return m.group(1).replace("-", "_")

def _has_log_flag(flagval) -> bool:
    try:
        if isinstance(flagval, str):
            return "log" in flagval
        return any(isinstance(s, str) and "log" in s for s in flagval)
    except Exception:
        return False

def _transform_to_display_units(var: str, arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr, dtype=float).copy()
    if var in ("v_init", "erosion_height_start", "erosion_height_change"):
        x = x / 1e3  # m/s->km/s ; m->km
    if var in ("sigma", "erosion_sigma_change", "erosion_coeff", "erosion_coeff_change"):
        x = x * 1e6  # kg/J -> kg/MJ
    if var in ("eeucs", "eeum", "eeucs_end", "eeum_end"):
        x = x * 1e-6  # J/(m^2 or kg) -> MJ/(m^2 or kg); remove if already MJ
    return x

def _weighted_ci(values: np.ndarray, weights: np.ndarray, ci: float):
    # Use dynesty's weighted quantile for med/CI
    from dynesty import utils as dyfunc
    v = np.asarray(values, float)
    w = np.asarray(weights, float)
    m = np.isfinite(v) & np.isfinite(w) & (w >= 0)
    v = v[m]; w = w[m]
    if v.size == 0 or np.sum(w) == 0:
        return (np.nan, np.nan, np.nan)
    w = w / np.sum(w)
    lo_q = 0.5 - ci/2.0
    hi_q = 0.5 + ci/2.0
    q = dyfunc.quantile(v, [0.5, lo_q, hi_q], weights=w)
    return float(q[0]), float(q[1]), float(q[2])

# --------- NEW: load via dynesty.restore (primary path) ---------
def _load_dynesty_samples_and_weights(dynesty_path: str, verbose: bool = False):
    """
    Preferred loader: use dynesty.DynamicNestedSampler.restore(dynesty_path).
    Falls back to plain pickle if restore is unavailable.
    Returns: (samples, weights, labels_or_None)
    """
    try:
        import dynesty
        # dynesty 2.x: restore is a @classmethod on DynamicNestedSampler
        dsampler = dynesty.DynamicNestedSampler.restore(dynesty_path)
        res = dsampler.results
        # samples
        samples = None
        if hasattr(res, '__getitem__'):
            try:
                samples = np.asarray(res['samples'])
            except Exception:
                samples = None
        if samples is None and hasattr(res, 'samples'):
            samples = np.asarray(res.samples)
        if samples is None:
            raise RuntimeError("restore() succeeded but 'samples' missing in results")
        # weights
        if hasattr(res, 'importance_weights') and callable(res.importance_weights):
            weights = np.asarray(res.importance_weights())
        else:
            # rare: derive from logwt/logz
            logwt = getattr(res, 'logwt', None)
            logz = getattr(res, 'logz', None)
            if logwt is not None and logz is not None and len(logz) > 0:
                weights = np.exp(np.asarray(logwt) - np.asarray(logz)[-1])
            else:
                weights = np.ones(samples.shape[0], dtype=float)
        # labels (optional)
        labels = getattr(res, 'labels', None) or getattr(res, 'names', None)
        if labels is not None:
            labels = list(map(str, labels))
        if verbose:
            print(f"[restore] {os.path.basename(dynesty_path)} -> samples={samples.shape}, weights={weights.shape}")
        return samples, weights, labels
    except Exception as e:
        if verbose:
            print(f"[restore] failed on {dynesty_path}: {e}; trying pickle fallback...")

    # Fallback: pickle open (kept for compatibility)
    with open(dynesty_path, "rb") as f:
        obj = pickle.load(f)

    res = None
    labels = None
    if hasattr(obj, "results"):
        res = obj.results
        labels = getattr(res, "labels", None) or getattr(res, "names", None)
    elif isinstance(obj, dict) and "results" in obj:
        res = obj["results"]
        labels = res.get("labels") if isinstance(res, dict) else None
    else:
        res = obj
        labels = getattr(res, "labels", None) or getattr(res, "names", None)

    # samples
    samples = None
    for key in ("samples", "samples_u", "samples_v"):
        if isinstance(res, dict):
            if key in res:
                samples = np.asarray(res[key])
                break
        else:
            if hasattr(res, key):
                samples = np.asarray(getattr(res, key))
                break
    if samples is None:
        raise RuntimeError(f"Could not find samples in {dynesty_path}")

    # weights
    weights = None
    if hasattr(res, "importance_weights") and callable(res.importance_weights):
        try:
            weights = np.asarray(res.importance_weights())
        except Exception:
            weights = None
    if weights is None:
        logwt = None; logz = None
        if isinstance(res, dict):
            logwt = np.asarray(res.get("logwt")) if res.get("logwt") is not None else None
            logz  = np.asarray(res.get("logz"))  if res.get("logz")  is not None else None
        else:
            if hasattr(res, "logwt"): logwt = np.asarray(getattr(res, "logwt"))
            if hasattr(res, "logz"):  logz  = np.asarray(getattr(res, "logz"))
        if logwt is not None and logz is not None and logz.size > 0:
            weights = np.exp(logwt - logz[-1])
    if weights is None:
        if isinstance(res, dict) and "weights" in res:
            weights = np.asarray(res["weights"])
        elif hasattr(res, "weights"):
            weights = np.asarray(getattr(res, "weights"))
    if weights is None:
        weights = np.ones(samples.shape[0], dtype=float)

    if labels is not None:
        labels = list(map(str, labels))
    if verbose:
        print(f"[pickle] {os.path.basename(dynesty_path)} -> samples={samples.shape}, weights={weights.shape}")
    return samples, weights, labels

def _samples_for_flags(samples: np.ndarray,
                       flags_dict: Dict[str, str],
                       align_mode: str = "flags_order",
                       labels: Optional[List[str]] = None,
                       verbose: bool = False) -> np.ndarray:
    """
    Return samples aligned to flags_dict order.
    align_mode:
      - 'flags_order' (default): ignore labels; assume sample columns follow flags order.
      - 'labels': attempt name-based mapping (only if most names match), else fall back to flags_order.
    """
    var_keys = list(flags_dict.keys())
    ns, nd = samples.shape

    if align_mode == "labels" and labels is not None:
        name_to_idx = {str(n): i for i, n in enumerate(labels)}
        cols = [name_to_idx.get(v) for v in var_keys]
        match = sum(c is not None for c in cols)
        if verbose:
            print(f"  [align] labels overlap {match}/{len(var_keys)}")
        if match >= max(1, len(var_keys)//2):
            out = np.full((ns, len(var_keys)), np.nan, dtype=float)
            for j, c in enumerate(cols):
                if c is not None:
                    out[:, j] = samples[:, c]
        else:
            out = None
    else:
        out = samples[:, :len(var_keys)].astype(float, copy=True) if nd >= len(var_keys) else None

    if out is None:
        out = np.full((ns, len(var_keys)), np.nan, dtype=float)
        out[:, :min(nd, len(var_keys))] = samples[:, :min(nd, len(var_keys))]

    # Back-transform log variables
    for j, v in enumerate(var_keys):
        if _has_log_flag(flags_dict[v]):
            with np.errstate(over='ignore', invalid='ignore'):
                out[:, j] = np.power(10.0, out[:, j])

    if verbose:
        print(f"  [align] mode={align_mode} ns={ns} nd={nd} -> aligned_nd={out.shape[1]}")
    return out

def _summarize_one_event(dynesty_path: str,
                         flags_dict: Dict[str, str],
                         ci: float,
                         align_mode: str,
                         verbose: bool) -> Dict[str, Dict[str, float]]:
    samples, weights, labels = _load_dynesty_samples_and_weights(dynesty_path, verbose=verbose)
    aligned = _samples_for_flags(samples, flags_dict, align_mode=align_mode, labels=labels, verbose=verbose)
    var_keys = list(flags_dict.keys())
    summary = {}
    for j, v in enumerate(var_keys):
        med, lo, hi = _weighted_ci(aligned[:, j], weights, ci=ci)
        med, lo, hi = _transform_to_display_units(v, np.array([med, lo, hi]))
        summary[v] = {"median": float(med), "lo": float(lo), "hi": float(hi)}
    if verbose:
        finite_count = sum(np.isfinite(summary[v]['median']) for v in var_keys)
        print(f"  [summ] finite medians {finite_count}/{len(var_keys)}")
    return summary

def _union_events(dicts: List[Dict[str, dict]]) -> List[str]:
    s = set()
    for d in dicts:
        s.update(d.keys())
    return sorted(s)

def _union_vars_from_summary(dicts: List[Dict[str, dict]]) -> List[str]:
    s = set()
    for d in dicts:
        for _eid, blob in d.items():
            s.update(blob.get("summary", {}).keys())
    return sorted(s)

def _union_vars_from_flags(dicts: List[Dict[str, dict]]) -> List[str]:
    s = set()
    for d in dicts:
        for _eid, blob in d.items():
            fl = blob.get("flags", [])
            s.update(fl)
    return sorted(s)

def compare_fixed_priors_with_flags(
    input_folders: List[str],
    labels: Optional[List[str]] = None,
    colors: Optional[List[str]] = None,
    ci: float = 0.95,
    output_png: Optional[str] = None,
    output_csv: Optional[str] = None,
    figure_title: Optional[str] = None,
    verbose: bool = False,
    align_mode: str = "labels",
):
    if labels is None:
        labels = [f"Run {i+1}" for i in range(len(input_folders))]
    if colors is None:
        palette = ["red", "blue", "green", "purple", "orange", "black"]
        colors = [palette[i % len(palette)] for i in range(len(input_folders))]
    assert len(labels) == len(input_folders)
    assert len(colors) == len(input_folders)

    # Import here to avoid hard dependency at module import time
    from DynNestSapl_metsim import find_dynestyfile_and_priors

    per_folder: List[Dict[str, dict]] = []

    for folder in input_folders:
        if verbose:
            print(f"[compare] scanning folder: {folder}")
        finder = find_dynestyfile_and_priors(
            input_dir_or_file=folder,
            prior_file="",
            resume=True,
            output_dir=folder,
            use_all_cameras=False,
            pick_position=0,
        )

        event_summaries: Dict[str, dict] = {}
        for base_name, dynesty_info, prior_path, out_folder in zip(
            finder.base_names, finder.input_folder_file, finder.priors, finder.output_folders
        ):
            dynesty_file, pickle_file, bounds, flags_dict, fixed_values = dynesty_info

            eid = _find_event_id_from_text(base_name) or _find_event_id_from_text(dynesty_file) or base_name
            try:
                summary = _summarize_one_event(dynesty_file, flags_dict, ci=ci, align_mode=align_mode, verbose=verbose)
            except Exception as e:
                if verbose:
                    print(f"  [warn] {eid}: failed to summarize ({e})")
                summary = {}

            event_summaries[eid] = {
                "summary": summary,
                "variables": list(summary.keys()),
                "flags": list(flags_dict.keys())  # store for fallback plotting
            }

        per_folder.append(event_summaries)

    eids = _union_events(per_folder)
    all_vars = _union_vars_from_summary(per_folder)
    if len(all_vars) == 0:
        all_vars = _union_vars_from_flags(per_folder)

    if len(eids) == 0 or len(all_vars) == 0:
        raise RuntimeError("No events or variables found to plot. Check loaders/flags and file contents.")

    # Reorder variables strictly by variable_map (skip missing); append any unknowns at end
    ordered_keys = [k for k in variable_map.keys() if k in all_vars]
    others = [k for k in all_vars if k not in variable_map]
    all_vars = ordered_keys + others

    # Build tidy table for plotting/export
    rows = []
    for folder, label, fmap in zip(input_folders, labels, per_folder):
        for eid in eids:
            blob = fmap.get(eid, {})
            summ = blob.get("summary", {})
            for v in all_vars:
                s = summ.get(v)
                if s:
                    rows.append(dict(event=eid, variable=v, run=label, folder=folder,
                                     median=s["median"], lo=s["lo"], hi=s["hi"]))
                else:
                    rows.append(dict(event=eid, variable=v, run=label, folder=folder,
                                     median=np.nan, lo=np.nan, hi=np.nan))
    df = pd.DataFrame(rows)

    # CSV path
    if output_png is not None and output_csv is None:
        base, _ = os.path.splitext(output_png)
        output_csv = base + "_summary.csv"
    if output_csv:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False)

    var_labels = {v: variable_map.get(v, v) for v in all_vars}

    nvar = len(all_vars)
    nrows = 5
    ncols = max(1, math.ceil(nvar / nrows))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 2.5 * nrows), squeeze=False, sharey=True)
    axes = axes.flatten()

    if figure_title:
        fig.suptitle(figure_title, fontsize=20, y=0.995)

    legend_ax_idx = 2 # index of subplot to hold legend
    fontsize_y = 12
    # Subplots
    any_points = False
    log_vars = {"m_init","erosion_coeff","erosion_coeff_change","erosion_mass_min","erosion_mass_max"}
    for ax_idx, v in enumerate(all_vars):
        ax = axes[ax_idx]
        y_pos = np.arange(len(eids))
        # Only first column shows y tick labels; others share y but hide labels
        if (ax_idx % ncols) == 0:
            ax.set_yticks(y_pos)
            ax.set_yticklabels(eids, fontsize=fontsize_y)
            ax.set_ylim(-0.5, len(eids) - 0.5)  # keep full list of meteors visible
        else:
            ax.set_yticks(y_pos)
            ax.tick_params(axis='y', which='both', labelleft=False)
        ax.set_xlabel(var_labels.get(v, v), fontsize=15)
        ax.grid(True, axis="x", alpha=0.3, linestyle="--")
        if v in log_vars:
            ax.set_xscale("log")

        for fidx, (label, color) in enumerate(zip(labels, colors)):
            sub = df[(df["variable"] == v) & (df["run"] == label)]
            meds, los, his = [], [], []
            for eid in eids:
                row = sub[sub["event"] == eid]
                if row.empty:
                    meds.append(np.nan); los.append(np.nan); his.append(np.nan)
                else:
                    r = row.iloc[0]
                    meds.append(r["median"]); los.append(r["lo"]); his.append(r["hi"])
            meds = np.asarray(meds, float)
            los  = np.asarray(los, float)
            his  = np.asarray(his, float)

            xerr = np.vstack([np.abs(meds - los), np.abs(his - meds)])
            y_off = (fidx - (len(labels)-1)/2.0) * 0.2
            y_plot = y_pos + y_off

            finite = np.isfinite(meds) & np.isfinite(los) & np.isfinite(his)
            if v in log_vars:
                finite &= (meds > 0) & (los > 0) & (his > 0)
            if np.any(finite):
                ax.errorbar(
                    meds[finite], y_plot[finite],
                    xerr=xerr[:, finite],
                    fmt="o", markersize=3, elinewidth=1.1, capsize=2,
                    color=color, ecolor=color, label=label if ax_idx == legend_ax_idx else None
                )
                any_points = True

        if ax_idx == legend_ax_idx  and len(labels) > 1:
            # flip the legend order to match y-offsets
            handles, legend_labels = ax.get_legend_handles_labels()
            ax.legend(handles[::-1], legend_labels[::-1], loc="best", fontsize=12)
            # ax.legend(loc="best", fontsize=12)

    # Hide unused axes
    for j in range(nvar, len(axes)):
        axes[j].axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.98])

    if "v_init" in all_vars:
        # # Adjust these numbers to your actual ranges if needed
        # left_max  = 23.3   # upper end of the low-velocity cluster
        # right_min = 68   # lower end of the high-velocity cluster
        # margin    = 0.5    # small padding around the data in each pane
        # Adjust these numbers to your actual ranges if needed
        left_max  = 26.3   # upper end of the low-velocity cluster
        right_min = 66.2   # lower end of the high-velocity cluster
        margin    = 0.5    # small padding around the data in each pane

        vidx = all_vars.index("v_init")
        base_ax = axes[vidx]
        fig = base_ax.figure

        # Get the position of the original axis *after* tight_layout
        bbox = base_ax.get_position()
        # Hide the original axis (and its legend)
        base_ax.set_visible(False)

        width = bbox.width
        gap_frac = 0.05   # fraction of width left as empty gap between the two panels
        w1 = (width * (1 - gap_frac)) / 2.0
        w2 = w1
        gap = width - (w1 + w2)

        # Create left and right axes in the same "slot"
        ax_left = fig.add_axes([bbox.x0, bbox.y0, w1, bbox.height])
        ax_right = fig.add_axes(
            [bbox.x0 + w1 + gap, bbox.y0, w2, bbox.height],
            sharey=ax_left
        )

        # Y ticks on the left only
        y_pos = np.arange(len(eids))
        ax_left.set_yticks(y_pos)
        ax_left.set_yticklabels(eids, fontsize=fontsize_y)
        ax_left.set_ylim(-0.5, len(eids) - 0.5)

        ax_right.set_yticks(y_pos)
        ax_right.tick_params(axis='y', which='both', labelleft=False)
        ax_right.set_ylim(-0.5, len(eids) - 0.5)

        # Common cosmetics (no per-axis xlabel here)
        for ax_b in (ax_left, ax_right):
            ax_b.grid(True, axis="x", alpha=0.3, linestyle="--")

        # Figure out x-limits from the data
        sub_all = df[df["variable"] == "v_init"]
        all_meds = sub_all["median"].to_numpy(dtype=float)
        finite_all = np.isfinite(all_meds)
        if np.any(finite_all):
            low_cluster = all_meds[(all_meds <= left_max) & finite_all]
            high_cluster = all_meds[(all_meds >= right_min) & finite_all]

            if len(low_cluster) > 0:
                x1_lo = np.nanmin(low_cluster) - margin
                x1_hi = left_max + margin
            else:
                x1_lo, x1_hi = left_max - 1, left_max + 1

            if len(high_cluster) > 0:
                x2_lo = right_min - margin
                x2_hi = np.nanmax(high_cluster) + margin
            else:
                x2_lo, x2_hi = right_min - 1, right_min + 1

            ax_left.set_xlim(x1_lo, x1_hi)
            ax_right.set_xlim(x2_lo, x2_hi)

        # Re-plot the v_init points and errorbars, split between the two panes
        for fidx, (label, color) in enumerate(zip(labels, colors)):
            sub = df[(df["variable"] == "v_init") & (df["run"] == label)]

            meds, los, his = [], [], []
            for eid in eids:
                row = sub[sub["event"] == eid]
                if row.empty:
                    meds.append(np.nan); los.append(np.nan); his.append(np.nan)
                else:
                    r = row.iloc[0]
                    meds.append(r["median"])
                    los.append(r["lo"])
                    his.append(r["hi"])

            meds = np.asarray(meds, float)
            los  = np.asarray(los, float)
            his  = np.asarray(his, float)

            finite = np.isfinite(meds) & np.isfinite(los) & np.isfinite(his)
            if not np.any(finite):
                continue

            xerr = np.vstack([np.abs(meds - los), np.abs(his - meds)])
            y_off = (fidx - (len(labels)-1)/2.0) * 0.2
            y_plot = y_pos + y_off

            left_mask  = finite & (meds <= left_max)
            right_mask = finite & (meds >= right_min)

            # No legend from v_init if you're putting legend on another subplot
            if np.any(left_mask):
                ax_left.errorbar(
                    meds[left_mask], y_plot[left_mask],
                    xerr=xerr[:, left_mask],
                    fmt="o", markersize=3, elinewidth=1.1, capsize=2,
                    color=color, ecolor=color,
                    label=None,
                )

            if np.any(right_mask):
                ax_right.errorbar(
                    meds[right_mask], y_plot[right_mask],
                    xerr=xerr[:, right_mask],
                    fmt="o", markersize=3, elinewidth=1.1, capsize=2,
                    color=color, ecolor=color,
                    label=None,
                )

        x_center = bbox.x0 + width / 2.0
        pad = 0.025  # shrink this to move the label up
        y_label = bbox.y0 - pad

        fig.text(
            x_center, y_label,
            var_labels.get("v_init", "v_init"),
            ha="center", va="top", fontsize=15
        )

        # Optional: add diagonal "break" marks on the inside edges
        d = .5  # proportion for the slanted line
        kwargs = dict(marker=[(-d, -1), (d, 1)], markersize=12,
                      linestyle="none", color='k', mec='k', mew=1, clip_on=False)
        # Right side of left axis
        ax_left.plot([1, 1], [0, 1], transform=ax_left.transAxes, **kwargs)
        # visibility set to False for the y axis ticks on the right pane
        ax_right.yaxis.set_visible(False)
        # do not show the line between two plots
        ax_right.spines['left'].set_visible(False)
        ax_left.spines['right'].set_visible(False)
        # Left side of right axis
        ax_right.plot([0, 0], [0, 1], transform=ax_right.transAxes, **kwargs)
    # ------------------------------------------------------------------

    if output_png is None:
        output_png = os.path.join(os.getcwd(), "compare_fixed_priors_flags.png")
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    fig.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return dict(
        figure_path=output_png,
        csv_path=output_csv,
        n_events=len(eids),
        n_variables=nvar,
        any_points=any_points
    )

##################################### ORI CAP PLOT #####################################

# info = compare_fixed_priors_with_flags(
#     input_folders=[
#         r"C:\Users\maxiv\Documents\UWO\Papers\2)ORI-CAP-PER-DRA\Validation\Validation-lum.eff\Results\LumEff\ORI-HighLumEff",
#         r"C:\Users\maxiv\Documents\UWO\Papers\2)ORI-CAP-PER-DRA\Results\CAMO+EMCCD\ORI_radiance_new",
#         r"C:\Users\maxiv\Documents\UWO\Papers\2)ORI-CAP-PER-DRA\Validation\Validation-lum.eff\Results\LumEff\ORI-LowLumEff",
#     ],
#     labels=[r"$\tau$=3%", r"$\tau$=0.4%", r"$\tau$=0.05%"],  # legend labels
#     colors=["red", "blue", "green"],              # requested colors
#     ci=0.95,
#     output_png=r"C:\Users\maxiv\Documents\UWO\Papers\2)ORI-CAP-PER-DRA\Validation\Validation-lum.eff\Results\ORIcompare_tau_runs.png",    # figure path
#     # figure_title="ORI Sensitivity to fixed τ prior",   # (optional) title
#     verbose=True,              # << turn on once to see diagnostics
#     align_mode="labels",  # << default; uses your flags order
# )
# print(info)

# info = compare_fixed_priors_with_flags(
#     input_folders=[
#         r"C:\Users\maxiv\Documents\UWO\Papers\2)ORI-CAP-PER-DRA\Validation\Validation-lum.eff\Results\LumEff\CAP-HighLumEff",
#         r"C:\Users\maxiv\Documents\UWO\Papers\2)ORI-CAP-PER-DRA\Results\CAMO+EMCCD\CAP_radiance_new",
#         r"C:\Users\maxiv\Documents\UWO\Papers\2)ORI-CAP-PER-DRA\Validation\Validation-lum.eff\Results\LumEff\CAP-LowLumEff",
#     ],
#     labels=[r"$\tau$=3.5%", r"$\tau$=1.4%", r"$\tau$=0.12%"],  # legend labels
#     colors=["red", "blue", "green"],              # requested colors
#     ci=0.95,
#     output_png=r"C:\Users\maxiv\Documents\UWO\Papers\2)ORI-CAP-PER-DRA\Validation\Validation-lum.eff\Results\CAPcompare_tau_runs.png",    # figure path
#     # figure_title="CAP Sensitivity to fixed τ prior",   # (optional) title
#     verbose=True,              # << turn on once to see diagnostics
#     align_mode="labels",  # << default; uses your flags order
# )
# print(info)

# info = compare_fixed_priors_with_flags(
#     input_folders=[
#         r"C:\Users\maxiv\Documents\UWO\Papers\2)ORI-CAP-PER-DRA\Results\EMCCD only\ORI_radiance",
#         r"C:\Users\maxiv\Documents\UWO\Papers\2)ORI-CAP-PER-DRA\Results\CAMO+EMCCD\ORI_radiance_new",
#     ],
#     labels=[r"EMCCD only", r"EMCCD + CAMO"],  # legend labels
#     colors=["brown", "blue"],              # requested colors
#     ci=0.95,
#     output_png=r"C:\Users\maxiv\Documents\UWO\Papers\2)ORI-CAP-PER-DRA\Results\CAMO+EMCCD\ORIcompare_Camera_runs.png",    # figure path
#     # figure_title="CAP Sensitivity to fixed τ prior",   # (optional) title
#     verbose=True,              # << turn on once to see diagnostics
#     # align_mode="labels",  # << default; uses your flags order
# )
# print(info)

# info = compare_fixed_priors_with_flags(
#     input_folders=[
#         r"C:\Users\maxiv\Documents\UWO\Papers\2)ORI-CAP-PER-DRA\Results\EMCCD only\CAP_radiance",
#         r"C:\Users\maxiv\Documents\UWO\Papers\2)ORI-CAP-PER-DRA\Results\CAMO+EMCCD\CAP_radiance_new",
#     ],
#     labels=[r"EMCCD only", r"EMCCD + CAMO"],  # legend labels
#     colors=["brown", "blue"],              # requested colors
#     ci=0.95,
#     output_png=r"C:\Users\maxiv\Documents\UWO\Papers\2)ORI-CAP-PER-DRA\Results\CAMO+EMCCD\CAPcompare_Camera_runs.png",    # figure path
#     # figure_title="CAP Sensitivity to fixed τ prior",   # (optional) title
#     verbose=True,              # << turn on once to see diagnostics
#     # align_mode="labels",  # << default; uses your flags order
# )
# print(info)

# info = compare_fixed_priors_with_flags(
#     input_folders=[
#         r"C:\Users\maxiv\Documents\UWO\Papers\2)ORI-CAP-PER-DRA\Validation\Validation-grainrho\Results\ORI",
#         r"C:\Users\maxiv\Documents\UWO\Papers\2)ORI-CAP-PER-DRA\Results\CAMO+EMCCD\ORI_radiance_new",
#     ],
#     labels=[r"$\rho_g$=3500 kg/m$^3$", r"$\rho_g$=3000 kg/m$^3$"],  # legend labels
#     colors=["orange", "blue"],              # requested colors
#     ci=0.95,
#     output_png=r"C:\Users\maxiv\Documents\UWO\Papers\2)ORI-CAP-PER-DRA\Validation\Validation-grainrho\Results\ORIcompare_rhoGrains_runs.png",    # figure path
#     # figure_title="CAP Sensitivity to fixed τ prior",   # (optional) title
#     verbose=True,              # << turn on once to see diagnostics
#     # align_mode="labels",  # << default; uses your flags order
# )
# print(info)

# info = compare_fixed_priors_with_flags(
#     input_folders=[
#         r"C:\Users\maxiv\Documents\UWO\Papers\2)ORI-CAP-PER-DRA\Validation\Validation-grainrho\Results\CAP",
#         r"C:\Users\maxiv\Documents\UWO\Papers\2)ORI-CAP-PER-DRA\Results\CAMO+EMCCD\CAP_radiance_new",
#     ],
#     labels=[r"$\rho_g$=3500 kg/m$^3$", r"$\rho_g$=3000 kg/m$^3$"],  # legend labels
#     colors=["orange", "blue"],              # requested colors
#     ci=0.95,
#     output_png=r"C:\Users\maxiv\Documents\UWO\Papers\2)ORI-CAP-PER-DRA\Validation\Validation-grainrho\Results\CAPcompare_rhoGrains_runs.png",    # figure path
#     # figure_title="CAP Sensitivity to fixed τ prior",   # (optional) title
#     verbose=True,              # << turn on once to see diagnostics
#     # align_mode="labels",  # << default; uses your flags order
# )
# print(info)

##################################### Sporadic PLOT #####################################

# info = compare_fixed_priors_with_flags(
#     input_folders=[
#         r"C:\Users\maxiv\Documents\UWO\Papers\3)Sporadics\Validation_nlive\nlive250",
#         r"C:\Users\maxiv\Documents\UWO\Papers\3)Sporadics\Validation_nlive\nlive500",
#         r"C:\Users\maxiv\Documents\UWO\Papers\3)Sporadics\Validation_nlive\nlive750",
#     ],
#     labels=[r"nlive=250", r"nlive=500", r"nlive=750"],  # legend labels
#     colors=["red", "blue", "green"],              # requested colors
#     ci=0.95,
#     output_png=r"C:\Users\maxiv\Documents\UWO\Papers\3)Sporadics\Validation_nlive\FastSpor10compare_nlive_runs.png",    # figure path
#     # figure_title="CAP Sensitivity to fixed τ prior",   # (optional) title
#     verbose=True,              # << turn on once to see diagnostics
#     align_mode="labels",  # << default; uses your flags order
# )
# print(info)

# info = compare_fixed_priors_with_flags(
#     input_folders=[
#         r"C:\Users\maxiv\Documents\UWO\Papers\3)Sporadics\Validation_logrho\logUniform_Fast",
#         r"C:\Users\maxiv\Documents\UWO\Papers\3)Sporadics\Validation_logrho\Uniform_Fast",
#     ],
#     labels=[r"$\rho$ log$_{10}$ uniform", r"$\rho$ uniform"],  # legend labels
#     colors=["orange", "blue"],              # requested colors
#     ci=0.95,
#     output_png=r"C:\Users\maxiv\Documents\UWO\Papers\3)Sporadics\Validation_logrho\FastSpor10compare_logUniform_runs.png",    # figure path
#     # figure_title="CAP Sensitivity to fixed τ prior",   # (optional) title
#     verbose=True,              # << turn on once to see diagnostics
#     # align_mode="labels",  # << default; uses your flags order
# )
# print(info)

# info = compare_fixed_priors_with_flags(
#     input_folders=[
#         r"C:\Users\maxiv\Documents\UWO\Papers\3)Sporadics\Validation_logrho\logUniform_Slow",
#         r"C:\Users\maxiv\Documents\UWO\Papers\3)Sporadics\Validation_logrho\Uniform_Slow",
#     ],
#     labels=[r"$\rho$ log$_{10}$ uniform", r"$\rho$ uniform"],  # legend labels
#     colors=["orange", "blue"],              # requested colors
#     ci=0.95,
#     output_png=r"C:\Users\maxiv\Documents\UWO\Papers\3)Sporadics\Validation_logrho\SlowSpor10compare_logUniform_runs.png",    # figure path
#     # figure_title="CAP Sensitivity to fixed τ prior",   # (optional) title
#     verbose=True,              # << turn on once to see diagnostics
#     # align_mode="labels",  # << default; uses your flags order
# )
# print(info)

# ##################################### Iron PLOT #####################################

# info = compare_fixed_priors_with_flags(
#     input_folders=[
#         r"C:\Users\maxiv\Documents\UWO\Papers\3.2)Iron Letter\irons-rho_eta100-noPoros\Tau3",
#         r"C:\Users\maxiv\Documents\UWO\Papers\3.2)Iron Letter\irons-rho_eta100-noPoros\Tau03",
#         r"C:\Users\maxiv\Documents\UWO\Papers\3.2)Iron Letter\irons-rho_eta100-noPoros\Tau008",
#     ],
#     labels=[r"$\tau$=3%", r"$\tau$=0.3%", r"$\tau$=0.08%"],  # legend labels
#     colors=["red", "blue", "green"],              # requested colors
#     ci=0.95,
#     output_png=r"C:\Users\maxiv\Documents\UWO\Papers\3.2)Iron Letter\irons-rho_eta100-noPoros\Iron_compare_tau_runs_2025.png",    # figure path
#     # figure_title="CAP Sensitivity to fixed τ prior",   # (optional) title
#     verbose=True,              # << turn on once to see diagnostics
#     align_mode="labels",  # << default; uses your flags order
# )
# print(info)

##################################### ASTRA #####################################

info = compare_fixed_priors_with_flags(
    input_folders=[
        r"C:\Users\maxiv\Documents\UWO\Papers\ASTRA\LogUnif\ASTRA",
        r"C:\Users\maxiv\Documents\UWO\Papers\ASTRA\LogUnif\CAMO+EMCCD",
        r"C:\Users\maxiv\Documents\UWO\Papers\ASTRA\LogUnif\EMCCD_only",
    ],
    labels=[r"ASTRA", r"CAMO", r"EMCCD"],  # legend labels
    colors=["red", "blue", "green"],              # requested colors # 
    ci=0.95,
    output_png=r"C:\Users\maxiv\Documents\UWO\Papers\ASTRA\LogUnif\LogUnif_runs.png",    # figure path
    # figure_title="CAP Sensitivity to fixed τ prior",   # (optional) title
    verbose=True,              # << turn on once to see diagnostics
    align_mode="labels",  # << default; uses your flags order
)
print(info)

# info = compare_fixed_priors_with_flags(
#     input_folders=[
#         r"C:\Users\maxiv\Documents\UWO\Papers\3)Sporadics\3.2)Iron Letter\irons-rho_eta100-noPoros\Test",
#         r"C:\Users\maxiv\Documents\UWO\Papers\3)Sporadics\3.2)Iron Letter\irons-rho_eta100-noPoros\Tau03",
#     ],
#     labels=[r"$\rho$=7000 kg/m³", r"No porosity"],  # legend labels
#     colors=["black", "blue"],              # requested colors
#     ci=0.95,
#     output_png=r"C:\Users\maxiv\Documents\UWO\Papers\3)Sporadics\3.2)Iron Letter\irons-rho_eta100-noPoros\Iron_compare_rho_runs.png",    # figure path
#     # figure_title="CAP Sensitivity to fixed τ prior",   # (optional) title
#     verbose=True,              # << turn on once to see diagnostics
#     align_mode="labels",  # << default; uses your flags order
# )
# print(info)

###########################################################################################
############################### Command line interface #####################################
###########################################################################################

# if __name__ == "__main__":
#     import argparse
#     ap = argparse.ArgumentParser(description="Compare dynesty runs by reading only flags_dict variables.")
#     ap.add_argument("folders", nargs="+", help="Result folders to scan (recursive) via find_dynestyfile_and_priors.")
#     ap.add_argument("--labels", nargs="*", default=None, help="Legend labels for runs.")
#     ap.add_argument("--colors", nargs="*", default=None, help="Colors for runs.")
#     ap.add_argument("--ci", type=float, default=0.95, help="Credible interval (e.g. 0.68, 0.90, 0.95).")
#     ap.add_argument("--out", default="compare_fixed_priors_flags.png", help="Output PNG path.")
#     ap.add_argument("--csv", default=None, help="Optional CSV path for summaries.")
#     ap.add_argument("--title", default=None, help="Figure title.")
#     ap.add_argument("--verbose", action="store_true", help="Print progress and warnings.")
#     ap.add_argument("--align", default="flags_order", choices=["flags_order","labels"], help="Column alignment strategy.")
#     args = ap.parse_args()

#     info = compare_fixed_priors_with_flags(
#         input_folders=args.folders,
#         labels=args.labels,
#         colors=args.colors,
#         ci=args.ci,
#         output_png=args.out,
#         output_csv=args.csv,
#         figure_title=args.title,
#         verbose=args.verbose,
#         align_mode=args.align,
#     )
#     print("Saved:", info["figure_path"])
#     if info.get("csv_path"):
#         print("CSV:", info["csv_path"])
#     print("Info:", info)
