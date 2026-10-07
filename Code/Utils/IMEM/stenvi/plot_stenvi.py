#!/usr/bin/env python3
"""Plot STENVI .sei files as azimuth/elevation Aitoff maps.

Requires Python >= 3.9, NumPy and Matplotlib:
    python -m pip install numpy matplotlib
    python plot_stenvi.py "C:\\data\\example.sei"
    python plot_stenvi.py "C:\\data" --recursive

One figure per DISTLAT bin: velocity (viridis), diameter (plasma), density
(cividis). Each angular bin is an estimated flux-weighted arithmetic mean:
    mean(q) = sum(flux * representative_bin_value(q)) / sum(flux).
DISTSET fluxes must be total, non-cumulative contributions per bin (not
differential densities). No extra bin-width or solid-angle weighting is used.
Representatives are arithmetic bin midpoints by default; geometric diameter
midpoints can be selected explicitly. Within-bin distributions are unknown.
Zero-flux cells are omitted. Units are km/s, m and g/cm^3, respectively.
Explicit DIST* boundaries take precedence over header min/max summaries.

The three panels use Matplotlib's Aitoff all-sky projection. Azimuth and
elevation are converted to radians for plotting. Original bin colors are
preserved without a KDE, scatter overlay, or parameter interpolation.
Bin boundaries are subdivided only to render curved projected edges smoothly.
The map is centered at azimuth 0 degrees by default; --center-azimuth changes
this reference. --invert-azimuth reverses the displayed longitude direction.
This changes only the projection, not the input coordinate reference frame.

A single latitude bin produces <stem>.png; multiple bins produce
<stem>_LAT<bin_id>.png. Titles include filename, bin number/count and bounds.
Images are saved beside their input by default. Existing matching PNGs are
replaced. With --output-dir, recursive input subdirectories are preserved.
Color limits are shared across latitude bins within each input file.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize


CARDS = ("DISTAZI", "DISTELE", "DISTVEL", "DISTDIA", "DISTLAT", "DISTDEN")
HEADERS = ("AZIMUTH", "ELEVATION", "VELOCITY", "DIAMETER", "LATITUDE", "DENSITY")


@dataclass
class Environment:
    path: Path
    bins: dict
    ids: dict
    means: np.ndarray       # quantity, latitude, elevation, azimuth
    flux: np.ndarray       # latitude, elevation, azimuth
    rows: int
    limits: dict            # header card -> (min, max) from the spectrum summary lines


def number(value):
    """Accept both ordinary scientific notation and Fortran D exponents."""
    result = float(value.replace("D", "E").replace("d", "e"))
    if not np.isfinite(result):
        raise ValueError(f"non-finite number: {value}")
    return result


def read_sei(path, diameter_center="arithmetic"):
    """Read definitions, then stream flux rows without allocating a 6-D cube."""
    path = Path(path)
    bins = {card: {} for card in CARDS}
    counts = {}
    limits = {}
    with path.open(encoding="utf-8-sig") as stream:
        for line_no, line in enumerate(stream, 1):
            fields = line.split()
            if not fields or fields[0] == "DISTSET":
                continue
            card = fields[0].upper()
            try:
                if card in HEADERS:
                    counts[card] = int(fields[1])
                    if len(fields) >= 4:
                        limits[card] = (number(fields[2]), number(fields[3]))
                elif card in bins:
                    index = int(fields[1])
                    low, high = map(number, fields[2:4])
                    if index < 1 or index in bins[card] or low > high:
                        raise ValueError(f"invalid or duplicate {card} bin {index}")
                    if card in ("DISTVEL", "DISTDIA", "DISTDEN") and low < 0:
                        raise ValueError(f"negative lower boundary in {card}")
                    bins[card][index] = (low, high)
            except (ValueError, IndexError) as exc:
                raise ValueError(f"{path.name}:{line_no}: {exc}") from exc

    for card, header in zip(CARDS, HEADERS):
        if not bins[card]:
            raise ValueError(f"{path.name}: missing {card} definitions")
        if header in counts and counts[header] != len(bins[card]):
            raise ValueError(f"{path.name}: {header} declares {counts[header]} bins, "
                             f"but {len(bins[card])} are defined")

    # Spatial ordering comes from physical boundaries, never row order.
    ids = {card: sorted(bins[card], key=lambda i: (*bins[card][i], i))
           for card in CARDS}
    for card in ("DISTAZI", "DISTELE"):
        edges_for(bins[card], ids[card])
    maps = {card: {index: j for j, index in enumerate(ids[card])} for card in CARDS}
    centers = {}
    for card in ("DISTVEL", "DISTDIA", "DISTDEN"):
        centers[card] = {}
        for index, (low, high) in bins[card].items():
            if card == "DISTDIA" and diameter_center == "geometric":
                if low <= 0:
                    raise ValueError("Geometric diameter centers require positive bin bounds")
                center = np.sqrt(low * high)
            else:
                center = low + (high - low) / 2
            centers[card][index] = center

    shape = (len(ids["DISTLAT"]), len(ids["DISTELE"]), len(ids["DISTAZI"]))
    flux = np.zeros(shape)
    weighted = np.zeros((3,) + shape)
    rows = 0
    with path.open(encoding="utf-8-sig") as stream:
        for line_no, line in enumerate(stream, 1):
            fields = line.split()
            if not fields or fields[0].upper() != "DISTSET":
                continue
            try:
                if len(fields) < 8:
                    raise ValueError("DISTSET needs six bin IDs and one flux value")
                indices = tuple(map(int, fields[1:7]))
                value = number(fields[7])
                if value < 0:
                    raise ValueError("negative flux")
                # Validate every index, including rows with zero flux.
                a, e, v, d, lat, rho = [maps[c][i] for c, i in zip(CARDS, indices)]
                if value:
                    cell = (lat, e, a)
                    flux[cell] += value
                    for q, card, index in zip(range(3), ("DISTVEL", "DISTDIA", "DISTDEN"),
                                               (indices[2], indices[3], indices[5])):
                        weighted[(q,) + cell] += value * centers[card][index]
                rows += 1
            except (ValueError, IndexError, KeyError) as exc:
                raise ValueError(f"{path.name}:{line_no}: invalid DISTSET ({exc})") from exc
    if not rows:
        raise ValueError(f"{path.name}: no DISTSET rows found")
    means = np.full_like(weighted, np.nan)
    np.divide(weighted, flux[None, ...], out=means, where=flux[None, ...] > 0)
    return Environment(path, bins, ids, means, flux, rows, limits)


def edges_for(bins, ids):
    """Require a contiguous non-overlapping angular grid for pcolormesh."""
    pairs = np.array([bins[i] for i in ids])
    if np.any(pairs[:, 1] <= pairs[:, 0]):
        raise ValueError("Angular bins must have positive width")
    if len(pairs) > 1 and not np.allclose(pairs[:-1, 1], pairs[1:, 0], rtol=1e-9, atol=1e-9):
        raise ValueError("Azimuth/elevation bins must be contiguous without overlaps")
    return np.r_[pairs[:, 0], pairs[-1, 1]]


def color_norm(values, logarithmic=False):
    valid = values[np.isfinite(values)]
    if logarithmic:
        valid = valid[valid > 0]
    if not valid.size:
        return LogNorm(1e-6, 1) if logarithmic else Normalize(0, 1)
    low, high = float(valid.min()), float(valid.max())
    if np.isclose(low, high, rtol=1e-12, atol=0):
        if logarithmic:
            low, high = low / 1.05, high * 1.05
        else:
            pad = abs(low) * 0.05 or 0.5
            low, high = max(0, low - pad), high + pad
    return LogNorm(low, high) if logarithmic else Normalize(low, high)


def full_range_norm(env, card, header, logarithmic=False):
    """Color limits from the header min/max (e.g. VELOCITY 40 0.0 80.0), so files with the
    same definitions share scales; falls back to the DIST* bin range if the line is missing."""
    pairs = np.array(list(env.bins[card].values()), float)
    low, high = env.limits.get(header, (pairs.min(), pairs.max()))
    if logarithmic and low <= 0:
        low = pairs[pairs > 0].min()
    return LogNorm(low, high) if logarithmic else Normalize(low, high)


def projected_grid(env, center_azimuth=0.0, invert_azimuth=False):
    """Subdivide bin edges for curved Aitoff cells; retain exact bin values."""
    x = edges_for(env.bins["DISTAZI"], env.ids["DISTAZI"])
    y = edges_for(env.bins["DISTELE"], env.ids["DISTELE"])
    if x[-1] - x[0] > 360 + 1e-8 or y[0] < -90 or y[-1] > 90:
        raise ValueError("Aitoff requires an azimuth span <=360 and elevation within [-90, 90]")
    if not np.isfinite(center_azimuth):
        raise ValueError("The center azimuth must be finite")
    sign = -1 if invert_azimuth else 1
    wrapped_edges = (sign * (x - center_azimuth) + 180) % 360 - 180

    def subdivide(edges):
        edges = np.unique(edges)
        return np.concatenate([np.linspace(lo, hi, max(1, int(np.ceil((hi-lo)/2))) + 1)[:-1]
                               for lo, hi in zip(edges[:-1], edges[1:])] + [edges[-1:]])

    lon_edges = subdivide(np.r_[-180., wrapped_edges, 180.])
    lat_edges = subdivide(np.r_[-90., y, 90.])
    lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2
    lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2
    source_azimuth = x[0] + (center_azimuth + sign * lon_centers - x[0]) % 360
    ai = np.searchsorted(x, source_azimuth, side="right") - 1
    ei = np.searchsorted(y, lat_centers, side="right") - 1
    valid = ((ei[:, None] >= 0) & (ei[:, None] < len(y)-1)
             & (ai[None, :] >= 0) & (ai[None, :] < len(x)-1))
    ai = np.clip(ai, 0, len(x)-2)
    ei = np.clip(ei, 0, len(y)-2)
    return np.deg2rad(lon_edges), np.deg2rad(lat_edges), ai, ei, valid


def plot_environment(env, output_dir=None, dpi=180, diameter_scale="log",
                     diameter_center="arithmetic", center_azimuth=0.0, invert_azimuth=False,
                     full_range=False, density_scale="linear", show_diameter=False):
    output_dir = Path(output_dir) if output_dir else env.path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    lon, lat, ai, ei, coverage = projected_grid(env, center_azimuth, invert_azimuth)
    cards = [("DISTVEL", "VELOCITY"), ("DISTDIA", "DIAMETER"), ("DISTDEN", "DENSITY")]
    logs = [False, diameter_scale == "log", density_scale == "log"]  # velocity, diameter, density
    norms = [full_range_norm(env, card, header, logs[q]) if full_range
             else color_norm(env.means[q], logs[q])
             for q, (card, header) in enumerate(cards)]
    panels = [("Relative velocity", "viridis", "Velocity [km/s]"),
              ("Particle diameter", "plasma", "Diameter [m]"),
              ("Particle density", "cividis", "Density [g/cm³]")]
    shown = [0, 1, 2] if show_diameter else [0, 2]  # diameter panel is optional
    outputs = []
    total = len(env.ids["DISTLAT"])
    sign = -1 if invert_azimuth else 1
    ticks = np.arange(-120, 121, 60)
    tick_values = (center_azimuth + sign * ticks + 180) % 360 - 180
    for ordinal, lat_id in enumerate(env.ids["DISTLAT"], 1):
        low, high = env.bins["DISTLAT"][lat_id]
        suffix = f"_LAT{lat_id}" if total > 1 else ""
        destination = output_dir / f"{env.path.stem}{suffix}.png"
        fig, axes = plt.subplots(1, len(shown), figsize=(6.67 * len(shown), 6.3),
                                 subplot_kw={"projection": "aitoff"})
        fig.subplots_adjust(left=0.045, right=0.98, bottom=0.20, top=0.79, wspace=0.19)
        fig.suptitle(f"{env.path.name}\nLAT {lat_id} — {ordinal} of {total} | "
                     f"Argument of true latitude: {low:g}° to {high:g}°",
                     fontsize=15, fontweight="normal", y=0.96)
        for ax, q in zip(axes, shown):
            title, palette, label = panels[q]
            values = env.means[q, ordinal - 1][ei[:, None], ai[None, :]]
            masked = np.ma.masked_where(~coverage | ~np.isfinite(values), values)
            if logs[q]:
                masked = np.ma.masked_less_equal(masked, 0)
            cmap = plt.get_cmap(palette).copy()
            cmap.set_bad("#eeeeee")
            ax.set_facecolor("#eeeeee")
            ax.grid(False)
            mesh = ax.pcolormesh(lon, lat, masked, cmap=cmap, norm=norms[q],
                                 shading="flat", edgecolors="none", antialiased=False,
                                 rasterized=True, zorder=1)
            ax.set_title(title, fontsize=15, fontweight="normal", pad=16)
            ax.set_xticks(np.deg2rad(ticks))
            ax.set_xticklabels([f"{v:g}°" for v in tick_values], fontsize=10, color="white")
            ax.set_yticks(np.deg2rad([-60, -30, 0, 30, 60]))
            ax.set_yticklabels(["−60°", "−30°", "0°", "30°", "60°"], fontsize=10)
            ax.set_longitude_grid_ends(75)
            ax.grid(True, color="white", linestyle="--", linewidth=0.65, alpha=0.45)
            ax.set_axisbelow(False)
            ax.set_xlabel("Azimuth", fontsize=12, labelpad=12)
            ax.set_ylabel("Elevation", fontsize=12, labelpad=22)
            bar = fig.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.19,
                               fraction=0.06, aspect=32)
            # Ticks span exactly the color limits; decades on a log scale spanning >= 3 decades
            low, high = norms[q].vmin, norms[q].vmax
            decades = 10.0 ** np.arange(np.ceil(np.log10(low)), np.floor(np.log10(high)) + 1) if logs[q] else []
            if logs[q] and len(decades) >= 4:
                bar_ticks = np.unique(np.r_[low, decades, high])
            elif logs[q]:
                bar_ticks = np.geomspace(low, high, 6)
            else:
                bar_ticks = np.linspace(low, high, 6)
            bar.set_ticks(bar_ticks)
            bar.set_ticklabels([f"{t:.3g}" for t in bar_ticks])
            bar.ax.minorticks_off()
            bar.set_label(label + (" (log scale)" if logs[q] else ""), fontsize=12)
            bar.ax.tick_params(labelsize=10)
            bar.outline.set_linewidth(0.6)
            if not masked.count():
                ax.text(0.5, 0.5, "No positive flux / no plottable values", transform=ax.transAxes,
                        ha="center", va="center", fontsize=10)
        fig.text(0.5, 0.035, "Aitoff projection • Flux-weighted mean bin values • "
                 "Grey = no data • Color scales shared across LAT bins", ha="center", fontsize=11)
        fig.savefig(destination, dpi=dpi, facecolor="white")
        plt.close(fig)
        outputs.append(destination)
    return outputs


def find_files(path, recursive=False):
    path = Path(path).expanduser().resolve()
    if path.is_file():
        if path.suffix.lower() != ".sei":
            raise ValueError(f"Expected a .sei file: {path}")
        return [path]
    if path.is_dir():
        candidates = path.rglob("*") if recursive else path.iterdir()
        files = sorted(p for p in candidates if p.is_file() and p.suffix.lower() == ".sei")
        if files:
            return files
        raise ValueError(f"No .sei files found in {path}" + ("" if recursive else " (try --recursive)"))
    raise ValueError(f"Path does not exist: {path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", nargs="?", default=r"C:\Users\maxiv\Documents\UWO\IMEM2_implement\MEM3orbit\L2_eclip.sei", help=".sei file or folder (default: current folder)")
    parser.add_argument("-r", "--recursive", action="store_true", help="Also search subfolders")
    parser.add_argument("-o", "--output-dir", type=Path, help="Output folder (default: beside each input)")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--diameter-scale", choices=("linear", "log"), default="log")
    parser.add_argument("--show-diameter", action="store_true",
                        help="Also plot the mean particle diameter panel (default: velocity and density only)")
    parser.add_argument("--density-scale", choices=("linear", "log"), default="linear",
                        help="Density colorbar scale (log helps see differences with --full-range)")
    parser.add_argument("--diameter-center", choices=("arithmetic", "geometric"), default="arithmetic")
    parser.add_argument("--center-azimuth", type=float, default=0,
                        help="Azimuth at the center of the Aitoff map, in degrees (default: 180)")
    parser.add_argument("--invert-azimuth", action="store_true", help="Show decreasing azimuth left to right")
    parser.add_argument("--full-range", action="store_true",
                        help="Colorbar limits from the header min/max (e.g. VELOCITY 40 0.0 80.0) instead of the "
                             "data min/max, so files with the same definitions share scales")
    args = parser.parse_args(argv)
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    try:
        files = find_files(args.path, args.recursive)
    except ValueError as exc:
        parser.error(str(exc))
    root = Path(args.path).expanduser().resolve()
    failed = 0
    for path in files:
        try:
            env = read_sei(path, args.diameter_center)
            out = args.output_dir
            if out and root.is_dir():
                out = out / path.parent.relative_to(root)
            outputs = plot_environment(env, out, args.dpi, args.diameter_scale, args.diameter_center,
                                       args.center_azimuth, args.invert_azimuth, args.full_range,
                                       args.density_scale, args.show_diameter)
            print(f"{path.name}: {env.rows:,} flux rows; {len(outputs)} image(s)")
            for output in outputs:
                print(f"  {output}")
        except (ValueError, OSError) as exc:
            failed += 1
            print(f"ERROR: {exc}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
