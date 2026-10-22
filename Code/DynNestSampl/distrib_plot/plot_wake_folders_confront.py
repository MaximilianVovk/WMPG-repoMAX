r"""Compare ORI results by reading two exported result folders recursively.

QUICK START
  1. Install numpy and matplotlib: python -m pip install numpy matplotlib
  2. Set BASELINE_FOLDER and WAKE_FOLDER below, then run this script.
     Or: python plot_ori_folders.py --baseline "C:\path\ORI" --wake "C:\path\ORI-wake"

INPUTS (no spreadsheets, CSV inputs or pickle files needed)
  * Recursively finds each event's *results_table.tex and reads the 2.5CI,
    Median and 97.5CI columns by header name (not column position).
  * Reads the human-readable best-fit RMSD lines directly from log_* files.
    No JSON or separate RMSD data files are needed. Both the current readable
    format and previous "Best fit wake RMSD (...) [ADU]:" lines are supported.
    Older logs with no RMSD values produce dashes, not inferred RMSD values.
  * Optional *_posterior_summary_table.tex for an overall plot, if exported.

OUTPUTS
  ORI_event_comparison.pdf/.png; ORI_wake_RMSD_comparison.tex;
  extracted_comparison.json/.csv. An ORI_overall_comparison.pdf/.png is also
  produced when both folders contain an exported overall summary.
  Overall distributions are NEVER inferred by averaging event medians.
  Masses are plotted in kg. Rounded table values retain source precision.

The event comparison adds h_e2, rho2, sigma2 and eta2 when a second erosion
height is exported for any matched event: 13 panels in a five-row layout.
Events without that stage have no point in its panels; missing values are not
filled with zeros. A second stage present in just one fit is plotted on that side.

Only matching event IDs are compared; '_combined' suffixes are ignored.
Missing events/parameters and ambiguous exports are reported. Identical rounded
summaries do NOT establish that the underlying posterior samples are identical.
Tables cannot reconstruct the synthetic KDE curves from the previous script.
"""

from pathlib import Path
import argparse
import csv
import json
import math
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter


# ======================= EDIT THESE SETTINGS ===========================
BASELINE_FOLDER = r"C:\Users\maxiv\Documents\UWO\Papers\6)Wake TEST\Results\ORI_onlylumlag_wake_plot"
WAKE_FOLDER = r"C:\Users\maxiv\Documents\UWO\Papers\6)Wake TEST\Results\ORI-wakefit"
OUTPUT_FOLDER = r"C:\Users\maxiv\Documents\UWO\Papers\6)Wake TEST\Results\ORI_comparison_plots"

# Optional exact paths, useful if a folder contains multiple conflicting runs.
# Relative paths are resolved within the corresponding result folder.
BASELINE_EVENT_FILE = None  # e.g. "20191026_065838/run_results_table.tex"
WAKE_EVENT_FILE = None
BASELINE_SUMMARY_FILE = None  # e.g. "ORI_onlylumlag_posterior_summary_table.tex"
WAKE_SUMMARY_FILE = None

LABELS = {"baseline": "Luminosity + lag", "wake": "Luminosity + lag + wake"}
COLORS = {"baseline": "#286e9d", "wake": "#c25a27"}
FONT_SIZE = 10
EVENT_FIGSIZE = (14.5, 9.6)  # Minimum size; height grows for 13 panels.
OVERALL_FIGSIZE = (12, 5.5)
SAVE_PNG = True
WRITE_LOGL_TABLE = False  # Optional legacy output, separate from RMSD.
PNG_DPI = 200
EVENT_IDS = None  # Optional list of YYYYMMDD_HHMMSS IDs, in desired order.
X_LIMITS = {}  # In displayed units, e.g. {'rho': (80, 4000), 'm_init': (1, 20)}
SECOND_STAGE_PARAMETERS = ["h_e2", "rho2", "sigma2", "eta2"]
# EVENT_PARAMETERS = ["v_init", "m_init", "rho", "sigma", "h_e", "eta", "s", "ml", "mu"] + SECOND_STAGE_PARAMETERS
# OVERALL_PARAMETERS = ["v_init", "m_init", "rho", "sigma", "h_e", "eta", "s", "ml", "mu"]
EVENT_PARAMETERS = ["m_init", "s", "ml", "mu", "h_e", "rho", "sigma", "eta", ] + SECOND_STAGE_PARAMETERS
OVERALL_PARAMETERS = ["m_init", "s", "ml", "mu", "h_e", "rho", "sigma", "eta"] + SECOND_STAGE_PARAMETERS
# =======================================================================

# key: (label, conversion from source units, logarithmic x-axis)
PARAMETERS = {
    "rho": (r"Initial density $\rho_1$ (kg m$^{-3}$)", 1, True),
    "rho_eff": (r"Effective density $\rho_{eff}$ (kg m$^{-3}$)", 1, True),
    "m_init": (r"Initial mass $m_0$ (kg)", 1, True),
    "diameter_mm": (r"Meteoroid diameter (mm)", 1, False),
    "s": (r"Grain mass index $s$", 1, False),
    "ml": (r"Lower grain mass $m_l$ (kg)", 1, True),
    "mu": (r"Upper grain mass $m_u$ (kg)", 1, True),
    "eta": (r"Initial erosion coefficient $\eta_1$ (kg MJ$^{-1}$)", 1, True),
    "sigma": (r"Initial ablation coefficient $\sigma_1$ (kg MJ$^{-1}$)", 1, True),
    "v_init": (r"Initial speed $v_0$ (km s$^{-1}$)", 1, False),
    "h_e": (r"Erosion start height $h_{e1}$ (km)", 1, False),
    "h_e2": (r"Second erosion height $h_{e2}$ (km)", 1, False),
    "rho2": (r"Second density $\rho_2$ (kg m$^{-3}$)", 1, True),
    "sigma2": (r"Second ablation coefficient $\sigma_2$ (kg MJ$^{-1}$)", 1, True),
    "eta2": (r"Second erosion coefficient $\eta_2$ (kg MJ$^{-1}$)", 1, True),
}
TEX_KEYS = {r"\rho": "rho", "m_0": "m_init", "s": "s", "m_l": "ml",
            "m_u": "mu", r"\eta": "eta", r"\sigma": "sigma", "v_0": "v_init",
            "h_e": "h_e", "h_e2": "h_e2", "h_e_2": "h_e2",
            r"\rho_2": "rho2", r"\sigma_2": "sigma2", r"\eta_2": "eta2"}
NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"


def event_id(text):
    match = re.search(r"(\d{8})[_ -](\d{6})", text.replace(r"\_", "_"))
    return "_".join(match.groups()) if match else None


def number(text):
    text = text.strip().strip("$").replace("−", "-")
    match = re.fullmatch(r"(" + NUMBER + r")\s*\\(?:times|cdot)\s*10\^\{([+-]?\d+)\}", text)
    if match:
        return float(match[1]) * 10 ** int(match[2])
    return float(text)


def interval(low, med, high):
    """Canonical storage: [lower endpoint, median, upper endpoint]."""
    q = [float(low), float(med), float(high)]
    if not all(math.isfinite(x) for x in q) or not q[0] <= q[1] <= q[2]:
        raise ValueError("Invalid interval: " + repr(q))
    return q


def tex_key(text):
    match = re.search(r"\$([^$]+)\$", text)
    symbol = match[1] if match else text
    return TEX_KEYS.get(re.sub(r"[{}\s]", "", symbol))


def read_summary(path):
    """Read endpoint columns from either an event or an overall summary."""
    result, columns = {}, None
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if "&" not in line:
            continue
        cells = [c.strip() for c in line.split(r"\\", 1)[0].split("&")]
        if "Parameter" in cells[0]:
            names = [re.sub(r"\\textbf\{([^}]+)\}", r"\1", c).strip() for c in cells]
            columns = [names.index(k) for k in ("2.5CI", "Median", "97.5CI")]
            continue
        key = tex_key(cells[0])
        if key and columns is not None:
            if len(cells) <= max(columns):
                raise ValueError("Missing interval columns in {}: {}".format(path, line))
            values = [cells[i].strip().strip("$").strip() for i in columns]
            if any(value.lower() in ("", "--", "---", "nan", "none", r"\textemdash") for value in values):
                continue  # A missing second-stage result is not a zero value.
            q = interval(*(number(value) for value in values))
            # DynestyMetSim normally exports kg, km, km/s and kg/MJ.
            # Honour explicit older units rather than silently mislabelling them.
            factor = 1.0
            unit = cells[0].split("]")[0].split("[")[-1].replace(" ", "")
            if key in ("m_init", "ml", "mu"):
                factor = {"kg": 1.0, "g": 1e-3, "mg": 1e-6}.get(unit, 1.0)
            elif key == "v_init" and unit == "m/s":
                factor = 1e-3
            elif key in ("h_e", "h_e2") and unit == "m":
                factor = 1e-3
            elif key in ("eta", "sigma", "eta2", "sigma2") and unit == "kg/J":
                factor = 1e6
            q = [v * factor for v in q]
            if key in result:
                raise ValueError("Duplicate parameter {} in {}".format(key, path))
            result[key] = q
    if not result:
        raise ValueError("No recognised parameter summary rows in " + str(path))
    return result


def candidates(folder, override, pattern):
    if override:
        path = Path(override).expanduser()
        path = path if path.is_absolute() else folder / path
        if not path.is_file():
            raise ValueError("File not found: " + str(path))
        return [path]
    return sorted(folder.rglob(pattern))


def event_from_path(path, root):
    """Prefer the filename, then the closest event-named parent directory."""
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        parts = path.parts  # Explicit override outside the scanned root.
    for part in reversed(parts):
        eid = event_id(part)
        if eid:
            return eid
    return event_id(root.name)


def read_wake_log(path):
    """Read RMSD and optional legacy wake scores from the latest best-fit block.

    Explicit unweighted and weighted values are kept separate. Older logs with
    no wake-specific output return an empty record. Appended blocks supersede
    earlier blocks, including when the newest block has no wake score.
    """
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    blocks = re.split(r"(?im)^\s*Best fit:\s*$", text)
    text = blocks[-1]
    blocks = re.split(r"(?im)^\s*Best fit logL\s*:[^\n]*$", text)
    text = blocks[-1]
    score = {}
    patterns = {
        "unweighted": r"Best fit wake logL(?:\s*\(unweighted\))?",
        "weighted": r"Best fit wake logL\s*\(weighted contribution\)",
        "alpha": r"Best fit wake alpha",
    }
    for key, label in patterns.items():
        matches = list(re.finditer(r"(?im)^\s*" + label + r"\s*:\s*(" + NUMBER + r")(?=\s|;|$)", text))
        if matches:
            val = float(matches[-1][1])
            if math.isfinite(val):
                score[key] = val
    for key, label in [("n_points", "N_wake"), ("n_heights", "N_heights")]:
        matches = re.findall(r"\b" + label + r"\s*:\s*(\d+)", text)
        if matches:
            score[key] = int(matches[-1])
    processing = re.findall(r"(?im)^\s*Best fit wake processing:\s*(.*)$", text)
    if processing:
        score["processing"] = processing[-1].strip()
    # Read only the readable RMSD summary lines; ignore embedded legacy JSON.
    groups = {}
    for line in text.splitlines():
        legacy = re.match(
            r"^\s*Best fit wake RMSD \((likelihood|diagnostic_only)\) \[ADU\]:\s*(.*)$",
            line, re.I)
        current = re.match(r"^\s*(Likelihood heights|Diagnostic-only heights):\s*(.*)$", line, re.I)
        if legacy:
            group, body = legacy[1].lower(), legacy[2]
        elif current:
            group = "likelihood" if current[1].lower() == "likelihood heights" else "diagnostic_only"
            body = current[2]
        else:
            continue
        values = {}
        match = re.match(r"(" + NUMBER + r")(?=\s|;|$)", body)
        if match:
            rmsd = float(match[1])
            if math.isfinite(rmsd) and rmsd >= 0:
                values["rmsd_adu"] = rmsd
        if legacy:
            n = re.search(r"\bN=(\d+)", body)
            count = re.search(r"\bprofiles=(\d+)", body)
        else:
            n = re.search(r"(\d+)\s+samples", body)
            count = re.search(r"(\d+)\s+profiles", body)
        if n:
            values["n_points"] = int(n[1])
        if count:
            values["n_profiles"] = int(count[1])
        groups[group] = values
    if groups:
        score["rmsd"] = {"groups": groups}
    return score


def load_folder(folder, event_file=None, summary_file=None):
    folder = Path(folder).expanduser().resolve()
    if not folder.is_dir():
        raise ValueError("Folder not found: " + str(folder))
    paths = candidates(folder, event_file, "*results_table.tex")
    events, sources, event_sources = {}, [], {}
    for path in paths:
        eid = event_from_path(path, folder)
        if eid is None:
            # Export captions also contain the event name.
            eid = event_id(path.read_text(encoding="utf-8-sig"))
        if eid is None:
            raise ValueError("Cannot identify event for " + str(path))
        record = read_summary(path)
        if eid in events and events[eid] != record:
            raise ValueError("Conflicting result tables for {}: {} and {}. "
                             "Select a folder containing one run per event.".format(
                                 eid, event_sources[eid], path))
        events[eid] = record
        sources.append(str(path))
        event_sources.setdefault(eid, []).append(str(path))
        print("Event source:", path)
    scores, log_sources = {}, {}
    for path in sorted(folder.rglob("log_*")):
        if not path.is_file() or path.suffix.lower() == ".json":
            continue
        eid = event_from_path(path, folder)
        if eid is None:
            # An unnamed run folder is usable only when it contains exactly
            # one identified event table (avoids pairing a shower-wide log).
            nearby = {e for e, files in event_sources.items()
                      if any(Path(f).parent == path.parent for f in files)}
            if len(nearby) == 1:
                eid = next(iter(nearby))
        if eid not in events:
            continue
        score = read_wake_log(path)
        log_sources.setdefault(eid, []).append(str(path))
        if not score:
            continue
        if eid in scores and scores[eid] != score:
            raise ValueError("Conflicting wake-score logs for {} in {}. "
                             "Select one run per event.".format(eid, folder))
        scores[eid] = score
    for eid in events:
        if "rmsd" not in scores.get(eid, {}):
            print("No best-fit ADU RMSD for", eid, "in", folder)
    summary, summary_sources = {}, []
    for path in candidates(folder, summary_file, "*_posterior_summary_table.tex"):
        parsed = read_summary(path)
        if summary and parsed != summary:
            raise ValueError("Multiple different overall summaries in {}. Set the exact SUMMARY_FILE path.".format(folder))
        summary = parsed
        summary_sources.append(str(path))
    if not events and not summary:
        raise ValueError("No *results_table.tex or overall summary tables found in " + str(folder))
    return {"events": events, "summary": summary, "wake_scores": scores,
            "sources": {"events": sources, "event_tables": event_sources,
                        "summary": summary_sources, "logs": log_sources}}


def write_wake_comparison(baseline, wake, ids, out):
    """Export raw wake-only scores; missing entries are never replaced by zero."""
    rows = []
    for eid in ids:
        a = baseline["wake_scores"].get(eid, {})
        b = wake["wake_scores"].get(eid, {})
        av, bv = a.get("unweighted"), b.get("unweighted")
        delta = bv-av if av is not None and bv is not None else None
        rows.append((eid, av, bv, delta, a.get("weighted"), b.get("weighted"),
                     a.get("n_points"), b.get("n_points")))
        if a.get("n_points") != b.get("n_points") and av is not None and bv is not None:
            print("Wake scores use different/unknown sample counts for", eid)
        if a.get("processing") != b.get("processing") and av is not None and bv is not None:
            print("Wake processing differs or is undocumented for", eid)
    with (out/"ORI_wake_loglikelihood_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["event", "without_wake_logL", "with_wake_logL", "delta_logL",
                         "without_wake_weighted", "with_wake_weighted",
                         "without_wake_n_points", "with_wake_n_points"])
        writer.writerows(rows)
    def fmt(value):
        return "--" if value is None else "${:.6g}$".format(value)
    lines = [r"\begin{table*}[htbp]", r"\centering", r"\small",
             r"\caption{Best-fit unweighted wake log-likelihoods for the paired Orionids. "
             r"Without and with wake refer to the observables used in the inversion. "
             r"$\Delta\log\mathcal{L}_{\mathrm{wake}}$ is with minus without; "
             r"positive values indicate a higher recorded wake score. Dashes denote "
             r"scores absent from the logs. Scores use each fit's recorded noise and "
             r"processing; differences measure wake agreement only when data, noise "
             r"scales and processing are comparable.}",
             r"\label{tab:ori_wake_loglikelihood}", r"\begin{tabular}{lrrr}", r"\hline",
             r"Event & Without wake & With wake & $\Delta\log\mathcal{L}_{\mathrm{wake}}$ \\", r"\hline"]
    for eid, av, bv, delta, *_ in rows:
        lines.append(eid.replace("_", r"\_") + " & " + " & ".join(map(fmt, (av, bv, delta))) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table*}"]
    path = out/"ORI_wake_loglikelihood_comparison.tex"
    path.write_text("\n".join(lines)+"\n", encoding="utf-8")
    print("Saved:", path)


def write_wake_rmsd_comparison(baseline, wake, ids, out):
    """Tabulate the readable group RMSD values recorded in each event log."""
    def fmt(value):
        if value is None:
            return "--"
        text = "{:.6g}".format(value)
        if "e" in text:
            mantissa, exponent = text.split("e")
            text = mantissa + r"\times10^{" + str(int(exponent)) + "}"
        return "$" + text + "$"
    lines = [r"\begin{table*}[htbp]", r"\centering", r"\small",
             r"\caption{Best-fit wake RMSD in ADU, read directly from the event logs. "
             r"L denotes likelihood heights and D diagnostic-only heights, as recorded "
             r"for each run. $\Delta$ is with minus without wake; negative values "
             r"indicate a smaller recorded RMSD. Dashes denote missing values. "
             r"Comparisons assume equivalent observed profiles, distance windows, "
             r"normalisation and alignment; these are not verified from the summary lines.}",
             r"\label{tab:ori_wake_rmsd}", r"\begin{tabular}{llrrr}", r"\hline",
             r"Event & Heights & Without wake [ADU] & With wake [ADU] & $\Delta$ [ADU] \\",
             r"\hline"]
    for eid in ids:
        for group in ("likelihood", "diagnostic_only"):
            a = baseline["wake_scores"].get(eid, {}).get("rmsd", {}).get("groups", {}).get(group, {})
            b = wake["wake_scores"].get(eid, {}).get("rmsd", {}).get("groups", {}).get(group, {})
            av, bv = a.get("rmsd_adu"), b.get("rmsd_adu")
            delta = bv-av if av is not None and bv is not None else None
            if av is None or bv is None:
                print("RMSD missing for {} ({}) in one or both logs; writing --.".format(eid, group))
            lines.append(eid.replace("_", r"\_") + " & " + ("L" if group == "likelihood" else "D")
                         + " & " + " & ".join(map(fmt, (av, bv, delta))) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table*}"]
    path = out/"ORI_wake_RMSD_comparison.tex"
    path.write_text("\n".join(lines)+"\n", encoding="utf-8")
    print("Saved:", path)


def available_parameters(wanted, records):
    # Show all four second-stage panels when any matched event has a second
    # erosion height in either run. A stage can be present on only one side.
    has_second_stage = any("h_e2" in r for pair in records for r in pair)
    keys = [key for key in wanted
            if (has_second_stage if key in SECOND_STAGE_PARAMETERS
                else any(key in a and key in b for a, b in records))]
    for key in wanted:
        if key not in keys:
            print("Skipping unavailable parameter:", key)
    if "diameter_mm" in wanted and "diameter_mm" not in keys:
        if any("eta" in a and "eta" in b for a, b in records) and "eta" not in keys:
            keys.insert(min(2, len(keys)), "eta")
            print("Using erosion coefficient instead of diameter (not exported in the LaTeX tables).")
    return keys


def draw_intervals(ax, records, key, offset, color, label):
    scale = PARAMETERS[key][1]
    for i, record in enumerate(records):
        if key not in record or (key in SECOND_STAGE_PARAMETERS and "h_e2" not in record):
            continue
        lo, med, hi = np.array(record[key]) * scale
        if PARAMETERS[key][2] and lo <= 0:
            print("Nonpositive lower bound for {}: plotted on a linear axis.".format(key))
        ax.errorbar(med, i+offset, xerr=[[med-lo], [hi-med]], fmt="o", ms=4.2,
                    capsize=2, lw=1.3, color=color, label=label if i == 0 else None)


def make_plot(a, b, ids, keys, out, name, overall=False):
    if not keys:
        print("No common parameters for", name)
        return
    ncols = min(4, len(keys))
    nrows = math.ceil(len(keys)/ncols)
    figsize = (OVERALL_FIGSIZE if overall else
               (EVENT_FIGSIZE[0], max(EVENT_FIGSIZE[1], 3.2 * nrows)))
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    for ax, key in zip(axs.flat, keys):
        draw_intervals(ax, a, key, -.14, COLORS["baseline"], LABELS["baseline"])
        draw_intervals(ax, b, key, .14, COLORS["wake"], LABELS["wake"])
        ax.set_title(PARAMETERS[key][0], fontsize=11, pad=12)
        ax.set_yticks(range(len(ids)))
        ax.set_yticklabels(ids if ax in axs[:, 0] else [""]*len(ids), fontsize=8)
        ax.set_ylim(len(ids)-.5, -.5)
        lows = [r[key][0] for r in a+b if key in r
                and (key not in SECOND_STAGE_PARAMETERS or "h_e2" in r)]
        if PARAMETERS[key][2] and lows and all(x > 0 for x in lows):
            ax.set_xscale("log")
            ax.xaxis.set_minor_formatter(NullFormatter())
            if key == "m_init":
                ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
                ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: "{:g}".format(x)))
        if key in X_LIMITS:
            ax.set_xlim(*X_LIMITS[key])
        ax.grid(axis="x", alpha=.2)
        for y in np.arange(len(ids))+.5:
            ax.axhline(y, color="#eeeeee", lw=.6, zorder=0)
    for ax in list(axs.flat)[len(keys):]:
        ax.set_visible(False)
    handles = [plt.Line2D([], [], color=COLORS[k], marker="o", label=LABELS[k]) for k in ("baseline", "wake")]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False)
    # footer = ("Exported overall medians and 95% distribution intervals" if overall
    #           else "Points: exported posterior medians; bars: 95% credible intervals")
    # fig.text(.5, .02, footer, ha="center", fontsize=9)
    fig.subplots_adjust(left=.16 if not overall else .09, right=.98, bottom=.10,
                        top=.87 if overall else .91, hspace=.4, wspace=.2)
    fig.savefig(out/(name+".pdf"))
    if SAVE_PNG:
        fig.savefig(out/(name+".png"), dpi=PNG_DPI)
    plt.close(fig)
    print("Saved:", out/(name+".pdf"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", default=BASELINE_FOLDER)
    parser.add_argument("--wake", default=WAKE_FOLDER)
    parser.add_argument("--output", default=str(OUTPUT_FOLDER))
    parser.add_argument("--baseline-event-file", default=BASELINE_EVENT_FILE)
    parser.add_argument("--wake-event-file", default=WAKE_EVENT_FILE)
    parser.add_argument("--baseline-summary-file", default=BASELINE_SUMMARY_FILE)
    parser.add_argument("--wake-summary-file", default=WAKE_SUMMARY_FILE)
    args = parser.parse_args()
    try:
        baseline = load_folder(args.baseline, args.baseline_event_file, args.baseline_summary_file)
        wake = load_folder(args.wake, args.wake_event_file, args.wake_summary_file)
        ids = sorted(set(baseline["events"]) & set(wake["events"]))
        for label, dataset in [("baseline", baseline), ("wake", wake)]:
            missing = sorted(set(dataset["events"])-set(ids))
            if missing:
                print("Only in {} (excluded from paired plot): {}".format(label, ", ".join(missing)))
        if EVENT_IDS is not None:
            absent = set(EVENT_IDS)-set(ids)
            if absent:
                raise ValueError("Requested IDs not matched in both folders: " + ", ".join(sorted(absent)))
            ids = list(EVENT_IDS)
        have_overall = bool(baseline["summary"] and wake["summary"])
        if not ids and not have_overall:
            raise ValueError("No matched events or paired overall summaries to plot.")
        out = Path(args.output).expanduser().resolve()
        out.mkdir(parents=True, exist_ok=True)
        plt.rcParams.update({"font.size": FONT_SIZE, "font.family": "DejaVu Sans",
                             "axes.spines.top": False, "axes.spines.right": False})
        if ids:
            a = [baseline["events"][e] for e in ids]
            b = [wake["events"][e] for e in ids]
            keys = available_parameters(EVENT_PARAMETERS, list(zip(a, b)))
            for e, aa, bb in zip(ids, a, b):
                missing = [k for k in keys if k not in SECOND_STAGE_PARAMETERS
                           and (k not in aa or k not in bb)]
                if missing:
                    print("Missing interval(s) for {}: {}".format(e, ", ".join(missing)))
            make_plot(a, b, ids, keys, out, "ORI_event_comparison")
            print("Matched {} events.".format(len(ids)))
        else:
            print("No paired event tables; producing the overall comparison only.")
        if have_overall:
            a, b = [baseline["summary"]], [wake["summary"]]
            keys = available_parameters(OVERALL_PARAMETERS, list(zip(a, b)))
            make_plot(a, b, ["ORI"], keys, out, "ORI_overall_comparison", overall=True)
        else:
            print("No paired overall tables; the overall comparison was skipped.")
        write_wake_rmsd_comparison(baseline, wake, ids, out)
        if WRITE_LOGL_TABLE:
            write_wake_comparison(baseline, wake, ids, out)
        extracted = {"baseline": baseline, "wake": wake, "matched_events": ids,
                     "interval_order": ["lower", "median", "upper"],
                     "note": "Display units: kg, kg/m^3, km/s, km, kg/MJ; no posterior reconstruction from summaries."}
        (out/"extracted_comparison.json").write_text(json.dumps(extracted, indent=2)+"\n", encoding="utf-8")
        with (out/"extracted_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["dataset", "scope", "event", "parameter", "lower", "median", "upper"])
            for label, data in [("baseline", baseline), ("wake", wake)]:
                for eid, record in list(data["events"].items()) + [("overall", data["summary"])]:
                    for key, q in record.items():
                        writer.writerow([label, "overall" if eid == "overall" else "event", eid, key] + q)
        print("Extracted values and source paths saved in:", out)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, "Error: " + str(exc) + "\n")


if __name__ == "__main__":
    main()
