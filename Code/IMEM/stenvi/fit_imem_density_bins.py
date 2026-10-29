#!/usr/bin/env python3
"""Fit three SEI density intervals to a CSV reference and save an overlay.

Requirements: Python >= 3.8, numpy, matplotlib.
Run without arguments to use the Windows paths in DEFAULT_SEI / DEFAULT_CSV.
Optional: --sei FILE --csv FILE --out-dir DIR --rho-min 100 --rho-max 8000
          --min-bin-width 50 --bounds centers --overwrite
          --flux-csv TABDATA.csv  (bin heights + diameter flux from IMEM tabdata)
          --size-density [STRENGTH] --density-split 4000 --ref-diameter 1e-3

Method follows the requested peak-normalized shape comparison:
  * Sum DISTSET flux by its sixth index (density), then divide by max(sum).
  * Peak-normalize CSV this_work_total_fraction.
  * Match the total area of three fixed-height steps to the reference area.
  * Resolve the remaining freedom by minimizing integrated squared error.
  * Require positive widths (default: one median CSV spacing).
The solution is global for the piecewise-linear reference, not a grid search.

Default bounds are the inferred outer edges of uniformly spaced CSV bins.
Between centres the curve is linear; on the two outer half-bins it is constant.
Use --bounds centers to reproduce the previous centre-to-centre calculation.
Explicit bounds may clip the curve or extend it only to its inferred bin edges.

--flux-csv: take the three step heights from the rows with quantity_code "dens"
(flux per density bin, averaged over orbital points, sorted by density) instead
of summing the SEI DISTSET rows. The SEI is still the template that is rewritten.

--size-density: the reference distribution holds for mm-size particles. As in
mem3_to_sei.py, densities below --density-split are shifted towards the split
for smaller diameters following the Divine (1986) radius law, and the shifted
distributions are mixed with the flux fraction of each SEI diameter bin (from
--flux-csv "dia" rows if given, else from the SEI DISTSET rows).

IMPORTANT: step heights here represent relative BIN TOTALS, not flux per unit
density. With unequal widths, their geometric area fractions differ from the
SEI bin flux fractions. Only DENSITY and DISTDEN definitions are changed;
DISTSET fluxes/indices, size bins, and all other file bytes stay unchanged.
This is a reassignment of density intervals, not a physical reweighting of flux
or a recomputation of mass/diameter consistency.
"""

import argparse
import csv
import itertools
import json
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mem3_to_sei import density_shift, overlap_matrix


DEFAULT_SEI = Path(r"C:\Users\maxiv\Documents\UWO\IMEM2_implement\1densitychange\Comparison\NewFormat\IMEM\IMEM2_sunoriented_mgt1e-06.sei")
DEFAULT_CSV = Path(r"C:\Users\maxiv\WMPG-repoMAX\Code\IMEM\stenvi\Sporadic_final_rho_full_posterior_distributions.csv")


def number(s):
    return float(s.replace(b"D", b"E").replace(b"d", b"e"))


def read_sei(path):
    flux = np.zeros(3)
    counts = np.zeros(3, dtype=int)
    dia_flux = {}
    dia_bins = {}
    intervals = {}
    density_headers = []
    with path.open("rb") as handle:
        for line_number, line in enumerate(handle, 1):
            parts = line.split()
            if not parts:
                continue
            if parts[0] == b"DISTSET":
                if len(parts) != 8:
                    raise ValueError("Expected DISTSET + six indices + flux at line %d" % line_number)
                indices = [int(v) for v in parts[1:7]]
                den = indices[5]
                value = number(parts[7])
                if den not in (1, 2, 3) or min(indices) < 1:
                    raise ValueError("Invalid DISTSET index at line %d" % line_number)
                if not np.isfinite(value) or value < 0:
                    raise ValueError("Invalid flux at line %d" % line_number)
                flux[den - 1] += value
                counts[den - 1] += 1
                dia_flux[indices[3]] = dia_flux.get(indices[3], 0.0) + value
            elif parts[0] == b"DISTDIA":
                dia_bins[int(parts[1])] = [number(parts[2]), number(parts[3])]
            elif parts[0] == b"DISTDEN":
                index = int(parts[1])
                if index in intervals:
                    raise ValueError("Duplicate DISTDEN definition")
                intervals[index] = [number(parts[2]), number(parts[3])]
            elif parts[0] == b"DENSITY":
                density_headers.append([int(parts[1]), number(parts[2]), number(parts[3])])
    if set(intervals) != {1, 2, 3} or len(density_headers) != 1 or density_headers[0][0] != 3:
        raise ValueError("This script requires one DENSITY header and DISTDEN bins 1, 2, 3.")
    old = np.array([intervals[i] for i in (1, 2, 3)])
    if np.any(old[:, 1] <= old[:, 0]) or not np.allclose(old[:-1, 1], old[1:, 0]):
        raise ValueError("Original density intervals must be ordered and contiguous.")
    if flux.max() <= 0:
        raise ValueError("The SEI contains no positive flux.")
    dia = np.array([dia_bins[i] for i in sorted(dia_bins)])  # [m]
    dia_flux = np.array([dia_flux.get(i, 0.0) for i in sorted(dia_bins)])
    return flux, counts, np.r_[old[:, 0], old[-1, 1]] * 1000.0, dia, dia_flux


def read_tabdata(path, code):
    """Mean flux over orbital points per x_value for one quantity_code of an IMEM tabdata CSV."""
    sums, counts = {}, {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row["quantity_code"] == code:
                x = float(row["x_value"])
                sums[x] = sums.get(x, 0.0) + float(row["y_value"])
                counts[x] = counts.get(x, 0) + 1
    if not sums:
        raise ValueError("No rows with quantity_code '%s' in %s." % (code, path))
    x = np.array(sorted(sums))
    return x, np.array([sums[v] / counts[v] for v in x])


def size_shifted_reference(x, y, dia, dia_flux, strength, split, ref_diameter):
    """Flux-weighted mix over diameter bins of the reference with densities below `split`
    moved by t(D) towards the split (see mem3_to_sei.population_to_bins)."""
    half = np.diff(x) / 2
    lo, hi = x - np.r_[half[0], half], x + np.r_[half, half[-1]]
    t = density_shift(dia[:, 0], dia[:, 1], strength, split, ref_diameter)
    out = np.zeros_like(y)
    for w, tk in zip(dia_flux / dia_flux.sum(), t):
        lo_k, hi_k = [np.where(e < split, e + tk * (split - e), e) for e in (lo, hi)]
        out += w * (y @ overlap_matrix(lo_k, hi_k, lo, hi))
    return out


def read_reference(path, bounds, lower, upper, size_density=None):
    data = np.genfromtxt(str(path), delimiter=",", names=True, encoding="utf-8-sig")
    names = ("rho_center_kg_m3", "this_work_total_fraction")
    if not all(n in (data.dtype.names or ()) for n in names):
        raise ValueError("CSV requires columns: " + ", ".join(names))
    x, y = (np.atleast_1d(data[n]).astype(float) for n in names)
    if len(x) < 2 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("CSV needs at least two finite density/fraction rows.")
    order = np.argsort(x)
    x, y = x[order], y[order]
    if np.any(np.diff(x) <= 0) or np.any(y < 0) or y.max() <= 0:
        raise ValueError("CSV densities must be distinct and fractions nonnegative with a positive peak.")
    if size_density is not None:
        y = size_shifted_reference(x, y, *size_density)
    dx = np.diff(x)
    spacing = float(np.median(dx))
    uniform = np.allclose(dx, spacing, rtol=1e-6, atol=1e-8)
    if bounds == "bin-edges" and not uniform:
        raise ValueError("Bin-edge inference requires uniform CSV spacing; use --bounds centers.")
    support = (x[0] - dx[0] / 2, x[-1] + dx[-1] / 2) if uniform else (x[0], x[-1])
    default = support if bounds == "bin-edges" else (x[0], x[-1])
    lo = default[0] if lower is None else lower
    hi = default[1] if upper is None else upper
    if not np.isfinite(lo + hi) or lo <= 0 or lo >= hi or lo < support[0] or hi > support[1]:
        raise ValueError("Requested bounds must be positive and within CSV support %s." % (support,))
    y = y / y.max()
    xx = np.r_[lo, x[(x > lo) & (x < hi)], hi]
    yy = np.interp(xx, x, y)
    return xx, yy, spacing


def fit_intervals(x, y, heights, min_width):
    """Intersect area constraint with width simplex; minimize exactly along it."""
    lo, hi = float(x[0]), float(x[-1])
    width = hi - lo
    if not np.isfinite(min_width) or min_width <= 0 or 3 * min_width >= width:
        raise ValueError("Minimum bin width must be positive and less than one third of the range.")
    dx = np.diff(x)
    slopes = np.diff(y) / dx
    cumulative = np.r_[0.0, np.cumsum((y[:-1] + y[1:]) * dx / 2)]
    area = float(cumulative[-1])

    def primitive(z):
        z = np.asarray(z)
        i = np.clip(np.searchsorted(x, z, side="right") - 1, 0, len(x) - 2)
        t = z - x[i]
        return cumulative[i] + y[i] * t + slopes[i] * t * t / 2

    residual_width = width - 3 * min_width
    residual_area = area - min_width * heights.sum()
    feasible = min_width * heights.sum() + residual_width * np.array([heights.min(), heights.max()])
    tolerance = 1e-10 * max(1.0, width)
    if area < feasible[0] - tolerance or area > feasible[1] + tolerance:
        raise ValueError(
            "Cannot match reference area %.9g with these fixed heights and minimum width %.9g. "
            "Feasible area range is [%.9g, %.9g]. Try a smaller --min-bin-width; "
            "if still infeasible, fixed heights cannot satisfy the requested area."
            % (area, min_width, feasible[0], feasible[1]))

    if np.ptp(heights) < 1e-14:
        return np.linspace(lo, hi, 4), area, 0.0

    vertices = []
    for i, j in itertools.combinations(range(3), 2):
        if abs(heights[i] - heights[j]) < 1e-14:
            if abs(residual_area - heights[i] * residual_width) <= tolerance:
                for k in (i, j):
                    w = np.full(3, min_width)
                    w[k] += residual_width
                    vertices.append(w)
            continue
        wi = (residual_area - heights[j] * residual_width) / (heights[i] - heights[j])
        if -tolerance <= wi <= residual_width + tolerance:
            w = np.full(3, min_width)
            w[i] += np.clip(wi, 0, residual_width)
            w[j] += residual_width - np.clip(wi, 0, residual_width)
            vertices.append(w)
    if not vertices:
        raise ValueError("Could not construct feasible density intervals.")
    # The feasible polygon intersection is a point or one line segment.
    distances = np.array([[np.sum((a - b)**2) for b in vertices] for a in vertices])
    p, q = np.unravel_index(np.argmax(distances), distances.shape)
    w0, dw = vertices[p], vertices[q] - vertices[p]
    a0, b0 = lo + w0[0], lo + w0[0] + w0[1]
    da, db = dw[0], dw[0] + dw[1]

    # Objective is piecewise quadratic; check all knots and stationary points.
    breaks = [0.0, 1.0]
    for start, delta in ((a0, da), (b0, db)):
        if abs(delta) > 1e-12:
            crossings = (x - start) / delta
            breaks.extend(crossings[(crossings > 0) & (crossings < 1)].tolist())
    knots = np.unique(breaks)
    candidates = list(knots)

    def derivative(t):
        ya = np.interp(a0 + da * t, x, y)
        yb = np.interp(b0 + db * t, x, y)
        return np.dot(heights**2, dw) - 2 * (
            (heights[0] - heights[1]) * ya * da
            + (heights[1] - heights[2]) * yb * db)

    for left, right in zip(knots[:-1], knots[1:]):
        dl, dr = derivative(left), derivative(right)
        if dr != dl:
            root = left - dl * (right - left) / (dr - dl)
            if left < root < right:
                candidates.append(root)
    ts = np.array(candidates)
    ws = w0[None, :] + ts[:, None] * dw[None, :]
    aa, bb = a0 + da * ts, b0 + db * ts
    fa, fb = primitive(aa), primitive(bb)
    objective = ws @ (heights**2) - 2 * (
        heights[0] * fa + heights[1] * (fb - fa) + heights[2] * (area - fb))
    k = int(np.argmin(objective))
    edges = np.array([lo, aa[k], bb[k], hi])
    blue_square_area = np.sum(dx * (y[:-1]**2 + y[:-1] * y[1:] + y[1:]**2) / 3)
    return edges, area, float(max(0.0, objective[k] + blue_square_area))


def sei_scientific(value):
    """SEI-style 0.xxxxxxxxE+nn; eight significant digits fit the 15-char fields."""
    mantissa, exponent = ("%.7e" % value).split("e")
    return ("%.8fE%+03d" % (float(mantissa) / 10, int(exponent) + 1)).encode("ascii")


def replace_definition(line, lower, upper, scientific):
    # Include preceding spaces so original fixed field widths are retained.
    pattern = rb"^(\s*(?:DENSITY|DISTDEN)\s+\d+)(\s+)(\S+)(\s+)(\S+)([^\r\n]*)(\r?\n)?$"
    match = re.match(pattern, line)
    if match is None:
        raise ValueError("Unsupported density definition line: %r" % line)
    g = match.groups()
    values = [sei_scientific(v) if scientific else ("%.7g" % v).encode("ascii") for v in (lower, upper)]
    widths = [len(g[1]) + len(g[2]), len(g[3]) + len(g[4])]
    if any(len(v) >= w for v, w in zip(values, widths)):
        raise ValueError("Density field too narrow for output precision.")
    return g[0] + values[0].rjust(widths[0]) + values[1].rjust(widths[1]) + g[5] + (g[6] or b"")


def rewrite_sei(source, destination, edges):
    # Round once to the exact values that the SEI will contain.
    encoded = [sei_scientific(v / 1000.0) for v in edges]
    saved_g = np.array([float(v) for v in encoded])
    if np.any(np.diff(saved_g) <= 0):
        raise ValueError("Density bins collapse at output precision; increase minimum width.")
    replaced = 0
    with source.open("rb") as inp, destination.open("wb") as out:
        for line in inp:
            parts = line.split()
            if parts and parts[0] == b"DENSITY":
                line = replace_definition(line, saved_g[0], saved_g[-1], False)
                replaced += 1
            elif parts and parts[0] == b"DISTDEN":
                i = int(parts[1]) - 1
                line = replace_definition(line, saved_g[i], saved_g[i + 1], True)
                replaced += 1
            out.write(line)
    if replaced != 4:
        raise ValueError("Expected to replace exactly four density definitions.")
    return saved_g * 1000.0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sei", type=Path, default=DEFAULT_SEI)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--bounds", choices=("bin-edges", "centers"), default="bin-edges")
    parser.add_argument("--rho-min", type=float, help="Outer lower bound in kg/m^3")
    parser.add_argument("--rho-max", type=float, help="Outer upper bound in kg/m^3")
    parser.add_argument("--min-bin-width", type=float, help="kg/m^3; default: median CSV spacing")
    parser.add_argument("--overwrite", action="store_true", help="Replace previously generated outputs")
    parser.add_argument("--flux-csv", type=Path, default=r"C:\Users\maxiv\WMPG-repoMAX\Code\IMEM\stenvi\debris-result-ISSspor10-3-10g_tabdata.csv",
                        help="IMEM tabdata CSV: step heights from quantity_code 'dens' instead of the SEI flux")
    parser.add_argument("--size-density", nargs="?", const=0.15, type=float, metavar="STRENGTH", # default=0.15,
                        help="raise reference densities below --density-split for particles smaller than "
                             "--ref-diameter (Divine 1986 radius law, as in mem3_to_sei.py); STRENGTH = max "
                             "fraction of the gap to the split (default 0.15)")
    parser.add_argument("--density-split", type=float, default=4000.0, metavar="KG_M3",
                        help="with --size-density: densities below this are shifted (default 4000 kg/m^3)")
    parser.add_argument("--ref-diameter", type=float, default=1e-3, metavar="M",
                        help="with --size-density: reference distribution holds at and above this diameter "
                             "(default 1e-3 m)")
    args = parser.parse_args()
    flux, counts, old_edges, dia, dia_flux = read_sei(args.sei)
    bar_flux = flux
    if args.flux_csv:
        bar_flux = read_tabdata(args.flux_csv, "dens")[1]
        if len(bar_flux) != 3:
            raise ValueError("Expected 3 'dens' values in --flux-csv, found %d." % len(bar_flux))
        dia_flux = read_tabdata(args.flux_csv, "dia")[1]
        if len(dia_flux) != len(dia):
            raise ValueError("--flux-csv has %d diameter values, the SEI has %d DISTDIA bins."
                             % (len(dia_flux), len(dia)))
    size_density = None
    if args.size_density is not None:
        size_density = (dia, dia_flux, args.size_density, args.density_split, args.ref_diameter)
    x, y, spacing = read_reference(args.csv, args.bounds, args.rho_min, args.rho_max, size_density)
    heights = bar_flux / bar_flux.max()
    min_width = spacing if args.min_bin_width is None else args.min_bin_width
    edges, target_area, squared_error = fit_intervals(x, y, heights, min_width)
    output_dir = args.out_dir or args.sei.parent / "density_refit"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = (args.sei.stem + ("_tab" if args.flux_csv else "")
            + ("_sizerho%g" % args.size_density if size_density else "") + "_density_refit")
    output_sei, output_plot, output_json = [output_dir / (stem + ext) for ext in (".sei", ".png", ".json")]
    if output_sei.resolve() == args.sei.resolve():
        raise ValueError("Output must not replace the input SEI.")
    for path in (output_sei, output_plot, output_json):
        if path.exists() and not args.overwrite:
            raise FileExistsError("%s already exists; use --overwrite or another --out-dir." % path)
    saved_edges = rewrite_sei(args.sei, output_sei, edges)
    # Read back the emitted header and flux sums to validate the actual output.
    flux_check, counts_check, edges_check = read_sei(output_sei)[:3]
    if not np.array_equal(flux, flux_check) or not np.array_equal(counts, counts_check):
        raise RuntimeError("Output flux validation failed.")
    if not np.allclose(saved_edges, edges_check, rtol=0, atol=1e-9):
        raise RuntimeError("Output interval validation failed.")
    saved_area = float(heights @ np.diff(saved_edges))
    if abs(saved_area - target_area) > 1e-6 * target_area:
        raise RuntimeError("SEI rounding exceeded the permitted relative area error (1e-6).")

    fig, ax = plt.subplots(figsize=(11, 5.5))
    if size_density:
        x0, y0, _ = read_reference(args.csv, args.bounds, args.rho_min, args.rho_max)
        ax.plot(x0, y0, color="lightsteelblue", linestyle="--", linewidth=1.6, label="This work: total (1 mm)")
    ax.plot(x, y, color="blue", linewidth=2.2,
            label="This work: total" + (" (size-shifted, flux-weighted)" if size_density else ""))
    ax.stairs(heights, old_edges, baseline=None, color="forestgreen", linestyle="--", linewidth=2,
              label="IMEM2: original density intervals")
    ax.stairs(heights, saved_edges, baseline=None, color="darkorange", linewidth=2.4,
              label="IMEM2: fitted intervals (matched area)")
    for i, boundary in enumerate(saved_edges[1:-1]):
        ax.axvline(boundary, color="darkorange", linestyle=":", linewidth=1, alpha=0.7)
        ax.text(boundary + 0.008 * (x[-1] - x[0]), 0.89 - i * 0.10,
                "%.2f" % boundary, color="darkorange", fontsize=10, rotation=90, va="top")
    ax.set_xlabel(r"$\rho$ [kg/m$^3$]", fontsize=16)
    ax.set_ylabel("Peak-normalized relative distribution", fontsize=13)
    ax.set_xlim(min(x[0], old_edges[0]), max(x[-1], old_edges[-1]))
    ax.set_ylim(0, 1.05)
    ax.tick_params(labelsize=12)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=10, loc="upper right")
    fig.tight_layout()
    fig.savefig(output_plot, dpi=300, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "input_sei": str(args.sei), "input_csv": str(args.csv),
        "input_flux_csv": str(args.flux_csv) if args.flux_csv else None,
        "bar_height_source": "flux CSV quantity_code 'dens'" if args.flux_csv else "SEI DISTSET sum",
        "flux_column_units": "As stated in the input SEI / flux CSV; no unit conversion applied",
        "sei_flux_by_density_bin": flux.tolist(),
        "flux_by_density_bin": bar_flux.tolist(), "flux_fractions": (bar_flux / bar_flux.sum()).tolist(),
        "peak_normalized_heights": heights.tolist(), "DISTSET_rows_by_bin": counts.tolist(),
        "original_edges_kg_m3": old_edges.tolist(), "fitted_edges_kg_m3": edges.tolist(),
        "saved_edges_kg_m3": saved_edges.tolist(), "saved_edges_g_cm3": (saved_edges / 1000).tolist(),
        "minimum_bin_width_kg_m3": min_width,
        "minimum_width_active": bool(np.any(np.isclose(np.diff(edges), min_width, atol=1e-6))),
        "reference_area": target_area, "saved_step_area": saved_area,
        "relative_area_error": (saved_area - target_area) / target_area,
        "integrated_squared_error_before_rounding": squared_error,
        "step_geometric_area_fractions": (heights * np.diff(saved_edges) / saved_area).tolist(),
        "size_density": None if size_density is None else {
            "strength": args.size_density, "split_kg_m3": args.density_split,
            "ref_diameter_m": args.ref_diameter, "diameter_bins_m": dia.tolist(),
            "diameter_flux": np.asarray(dia_flux).tolist(),
            "shift_fraction_t": density_shift(dia[:, 0], dia[:, 1], args.size_density,
                                              args.density_split, args.ref_diameter).tolist()},
        "reference_endpoint_policy": "Linear between centres; constant to inferred outer edges if included",
        "interpretation": "Requested bin-total height comparison; not a flux-per-unit-density PDF. "
                          "Flux rows are unchanged; only the density interval definitions are reassigned.",
    }
    output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("Flux (bins 1, 2, 3):      ", bar_flux, "(flux CSV)" if args.flux_csv else "(SEI)")
    print("Peak-normalized heights:  ", heights)
    print("New edges [kg/m^3]:       ", saved_edges)
    print("New edges [g/cm^3]:       ", saved_edges / 1000)
    print("Blue / saved step area: %.10g / %.10g" % (target_area, saved_area))
    if summary["minimum_width_active"]:
        print("Note: a fitted group reached the minimum width of %.6g kg/m^3." % min_width)
    print("All DISTSET flux rows preserved. Updated DENSITY and DISTDEN only.")
    for path in (output_sei, output_plot, output_json):
        print("Saved:", path)


if __name__ == "__main__":
    main()
