"""
Convert a MEM 3 run (HiDensity + LoDensity populations) into a STENVI .sei file.

Binning (azimuth, elevation, velocity, diameter, latitude, density) and the
header layout are taken from a template .sei file. The Argument of True
Latitude spectrum is copied unchanged.

Method
------
MEM 3 gives, for each population p (hi, lo), the flux of particles with
m >= M0 (MEM limiting mass, here 1e-6 g) per direction/speed bin, F_p(az, el, v),
plus the fraction f_p(rho_j) of those particles in fine density bins rho_j.

The size distribution is extended to every diameter bin with the Grun (1985)
interplanetary cumulative flux g(m), assuming the directional/speed
distribution of the M0 particles holds for all sizes:

    Flux(az, el, v, D_k, den_i) = sum_p F_p(az, el, v) * W_p[i, k]
    W_p[i, k] = sum_{rho_j in den_i} f_p(rho_j) *
                [g(m(D_k,lo, rho_j)) - g(m(D_k,hi, rho_j))] / g(M0)
    m(D, rho) = rho * pi/6 * D^3

Hi and Lo contributions are then summed into a single flux per bin.
"""
import argparse
import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np


# GM [km^3/s^2] of the MEM input origins (options.txt line 2), DE430 values
MU = {"su": 1.32712440018e11, "ea": 398600.4418, "mo": 4902.800066,
      "me": 22031.78, "ve": 324858.592, "ma": 42828.375214}


def grun_cumulative_flux(m):
    """Grun et al. (1985) cumulative flux of particles with mass > m [g], in m^-2 s^-1."""
    m = np.asarray(m, float)
    return ((2.2e3 * m**0.306 + 15.0) ** -4.38
            + 1.3e-9 * (m + 1e11 * m**2 + 1e27 * m**4) ** -0.36
            + 1.3e-16 * (m + 1e6 * m**2) ** -0.85)


def rho_divine1986(radius_um):
    """Divine et al. (1986) Halley dust bulk density [g/cm^3] vs grain radius [um]:
    3.0 for small grains, falling to 0.8 for large ones."""
    a = np.asarray(radius_um, float)
    return 3.0 - 2.2 * a / (a + 2.0)


def density_shift(dia_lo, dia_hi, strength, split, ref_diameter):
    """Per-diameter-bin shift t(D) of low densities towards `split` (rho -> rho + t*(split - rho)).

    t(D) = strength * clip((f(D/2) - f(Dref/2)) / (f(0) - f(Dref/2)), 0, 1), with f the Divine
    (1986) radius law and D the geometric bin centre [m]: MEM's distribution holds for D >= Dref
    (mm-size), smaller particles move up towards the split, never crossing it."""
    radius_um = np.sqrt(dia_lo * dia_hi) * 5e5
    f_ref, f_zero = rho_divine1986(ref_diameter * 5e5), rho_divine1986(0.0)
    return strength * np.clip((rho_divine1986(radius_um) - f_ref) / (f_zero - f_ref), 0.0, 1.0)


def overlap_matrix(src_lo, src_hi, dst_lo, dst_hi):
    """M[s, d] = fraction of source bin s that falls in destination bin d (uniform within bin)."""
    lo = np.maximum(src_lo[:, None], dst_lo[None, :])
    hi = np.minimum(src_hi[:, None], dst_hi[None, :])
    return np.clip(hi - lo, 0, None) / (src_hi - src_lo)[:, None]


DIMS = [  # (DIST card, spectrum card, label in the spectrum card)
    ("DISTAZI", "AZIMUTH", "Azimuth [deg]"),
    ("DISTELE", "ELEVATION", "Elevation [deg]"),
    ("DISTVEL", "VELOCITY", "Velocity [km/s]"),
    ("DISTDIA", "DIAMETER", "Diameter [m]"),
    ("DISTLAT", "LATITUDE", "Argument of True Latitude [deg]"),
    ("DISTDEN", "DENSITY", "Density [g/cm^3]"),
]


def read_template(path):
    lines = Path(path).read_text().splitlines()
    end = next(i for i, l in enumerate(lines) if l.startswith("# Azi Ele Vel Dia Lat Den"))
    header = lines[:end + 1]
    bins = {}
    for l in header:
        tok = l.split()
        if tok and tok[0] in [d[0] for d in DIMS]:
            bins.setdefault(tok[0], []).append((float(tok[2]), float(tok[3])))
    return header, {k: np.array(v) for k, v in bins.items()}


def edges_to_bins(edges):
    edges = np.asarray(edges, float)
    return np.column_stack([edges[:-1], edges[1:]])


def read_mem_flux(path):
    """Return (phi_min, theta_min, speed_edges, flux[row, speed]) from a MEM flux_avg.txt."""
    with open(path) as fh:
        labels = next(l for l in fh if l.startswith("# PHI1")).split()[3:]
    mids = np.array(labels, float)
    half = np.diff(mids).mean() / 2
    d = np.loadtxt(path, comments="#")
    return d[:, 0], d[:, 1], mids - half, mids + half, d[:, 2:]


def jd_to_datetime(jd):
    return datetime(2000, 1, 1, 12) + timedelta(days=jd - 2451545.0)


def central_body_mu(run_dir):
    """GM of the state-vector origin, from line 2 (input origin) of the run's options.txt."""
    lines = [l.strip() for l in (Path(run_dir) / "options.txt").read_text().splitlines()
             if l.strip() and not l.startswith("#")]
    return MU[lines[1][:2].lower()]


def orbit_from_state_vectors(path, mu):
    """Mission start/end and Keplerian elements of the first state vector in a MEM input file."""
    sv = np.loadtxt(path, comments="#")
    r, v = sv[0, 1:4], sv[0, 4:7]
    h = np.cross(r, v)
    n = np.cross([0, 0, 1], h)
    e_vec = np.cross(v, h) / mu - r / np.linalg.norm(r)
    e = np.linalg.norm(e_vec)
    a = 1 / (2 / np.linalg.norm(r) - v @ v / mu)
    inc = np.degrees(np.arccos(h[2] / np.linalg.norm(h)))
    raan = np.degrees(np.arctan2(n[1], n[0])) % 360
    argp = np.degrees(np.arccos(np.clip(n @ e_vec / (np.linalg.norm(n) * e), -1, 1)))
    if e_vec[2] < 0:
        argp = 360 - argp
    return jd_to_datetime(sv[0, 0]), jd_to_datetime(sv[-1, 0]), a, e, inc, raan, argp


def mem_version(run_dir):
    first = (Path(run_dir) / "info.txt").read_text().splitlines()[0]
    m = re.search(r"version\s+([\d.]+)", first)
    return f"MEM v{m.group(1)}" if m else "MEM 3"


def limiting_mass(run_dir):
    m = re.search(r"Limiting mass:\s*10\^(-?[\d.]+)", (Path(run_dir) / "info.txt").read_text())
    return 10 ** float(m.group(1))


def population_to_bins(run_dir, pop, bins, m0, fold_density, flip_azimuth, az_offset=0.0,
                       size_density=None):
    """Directional/speed flux on the SEI grid [az, el, vel] and diameter/density weights [den, dia]."""
    phi, theta, v_lo, v_hi, flux = read_mem_flux(Path(run_dir) / f"{pop}Density" / "flux_avg.txt")

    # MEM body-fixed: theta 0 = ram (+x), measured towards +y (port); map to [-180, 180)
    # SEI azimuth has the opposite sign, so mirror about the elevation axis (az -> -az)
    dang = np.diff(np.unique(theta))[0]
    az = np.where(theta >= 180, theta - 360, theta)
    if flip_azimuth:
        az = -az - dang  # mirror bin [az, az+d) -> [-az-d, -az)
    if az_offset % dang:
        raise SystemExit(f"--az-offset must be a multiple of the MEM grid step ({dang:g} deg)")
    az = (az + az_offset + 180) % 360 - 180  # rotate about the zenith axis, wrap to [-180, 180)
    A = overlap_matrix(az, az + dang, bins["DISTAZI"][:, 0], bins["DISTAZI"][:, 1])
    E = overlap_matrix(phi, phi + dang, bins["DISTELE"][:, 0], bins["DISTELE"][:, 1])
    V = overlap_matrix(v_lo, v_hi, bins["DISTVEL"][:, 0], bins["DISTVEL"][:, 1])
    # [row, speed] -> [az, el, vel]
    dirvel = np.einsum("ra,re,rs,sv->aev", A, E, flux, V, optimize=True)

    dens = np.loadtxt(Path(run_dir) / f"{pop.lower()}density.txt", comments="#")
    frac = dens[:, 2] / dens[:, 2].sum()
    den_lo, den_hi = bins["DISTDEN"][:, 0], bins["DISTDEN"][:, 1]
    dia = bins["DISTDIA"]
    # Size-dependent density: low-density bins (edges below the split) are mapped linearly towards
    # the split per diameter bin, which transports their probability (uniform bins stay uniform)
    t = density_shift(dia[:, 0], dia[:, 1], *size_density) if size_density else np.zeros(len(dia))
    split = size_density[1] if size_density else 0.0

    def density_overlap(rho_lo, rho_hi):
        rho = (rho_lo + rho_hi) / 2
        D = overlap_matrix(rho_lo, rho_hi, den_lo, den_hi)
        if fold_density:  # put densities outside the SEI range into the edge bins
            D[rho < den_lo[0], 0] = 1.0
            D[rho > den_hi[-1], -1] = 1.0
        return rho, D

    lost = 1 - (frac @ density_overlap(dens[:, 0] / 1000.0, dens[:, 1] / 1000.0)[1]).sum()
    weights = np.zeros((len(den_lo), len(dia)))  # [den, dia]
    for k, tk in enumerate(t):
        lo, hi = [np.where(e < split, e + tk * (split - e), e) for e in (dens[:, 0] / 1000.0, dens[:, 1] / 1000.0)]
        rho, D = density_overlap(lo, hi)
        mass = rho[:, None] * np.pi / 6 * (dia[k] * 100.0) ** 3  # [rho, lo/hi] in g
        g = grun_cumulative_flux(mass)
        size_frac = (g[:, 0] - g[:, 1]) / grun_cumulative_flux(m0)  # [rho]
        weights[:, k] = (frac * size_frac) @ D
    return dirvel, weights, flux.sum(), lost


def build_header(template_header, model, begin, end, a, e, inc, raan, argp, bins, replaced):
    repl = {
        "MISSBEGIN": f"MISSBEGIN {begin:%Y %m %d %H} Begin [yyyy mm dd hh]",
        "MISSEND": f"MISSEND {end:%Y %m %d %H} End [yyyy mm dd hh]",
        "SEMIAXIS": f"SEMIAXIS {a:.1f} Semimajor axis [km]",
        "ECCENTRI": f"ECCENTRI {e:.1E} Eccentricity of the orbit [-]",
        "INCLIN": f"INCLIN {inc:.1f} Orbit inclination [deg]",
        "RAAN": f"RAAN {raan:.1f} Right ascension of ascending node [deg]",
        "ARGPERI": f"ARGPERI {argp:.1f} Argument of perigee [deg]",
    }
    for dist, card, label in DIMS:
        if dist in replaced:
            b = bins[dist]
            repl[card] = f"{card} {len(b)} {b[0, 0]:.1f} {b[-1, 1]:.1f} {label}"
    out, prev = [], ""
    for l in template_header:
        key = l.split()[0] if l.split() else ""
        if prev.startswith("# Environment Model"):
            l = model
        prev = l
        if key in replaced:
            if not out[-1].startswith(key):  # first line of the block: write the new bins
                out += [f"{key} {n} {lo:.4e} {hi:.4e}" for n, (lo, hi) in enumerate(bins[key], 1)]
            continue
        out.append(repl.get(key, l))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=r"C:\Users\maxiv\Documents\UWO\IMEM2_implement\MEM3orbit\L2sei_newrho", help="MEM 3 results folder")
    ap.add_argument("--template", default=r"C:\Users\maxiv\Documents\UWO\IMEM2_implement\sun_stenvi_2025-1T_2025-2T\sun_stenvi_2025-1T_2025-2T.sei", help="template .sei file (bins + header)")
    ap.add_argument("--out", default=r"C:\Users\maxiv\Documents\UWO\IMEM2_implement\MEM3orbit\MEM3_newrho.sei", help="output .sei file")
    ap.add_argument("--no-fold-density", action="store_true",
                    help="drop flux of densities outside the SEI density range instead of adding it to the edge bins")
    ap.add_argument("--mem-azimuth", action="store_true",
                    help="keep MEM's azimuth sign (positive towards port) instead of mirroring it for SEI")
    ap.add_argument("--az-offset", type=float, default=0.0, metavar="DEG",
                    help="shift the azimuth reference by DEG about the zenith axis (e.g. 180 puts ram at az 180, "
                         "wake at az 0; must be a multiple of 5)")
    ap.add_argument("--mem-angles", action="store_false",
                    help="use MEM's azimuth/elevation grid (5 deg) instead of the template's")
    ap.add_argument("--mem-velocity", action="store_false",
                    help="use MEM's velocity grid (2 km/s, 0-80 km/s) instead of the template's")
    ap.add_argument("--mem-density", nargs="?", const=0.05, type=float, metavar="STEP", default=0.5,   
                    help="use MEM's density grid (0.1-8.0 g/cm^3) instead of the template's; optional STEP "
                         "in g/cm^3 merges MEM's 0.05 g/cm^3 bins (default 0.05 = native)")
    ap.add_argument("--min-flux", type=float, default=0.0,
                    help="with --skip-zeros: only write bins with flux above this value [1/m^2/yr]")
    ap.add_argument("--skip-zeros", action="store_true",
                    help="write only bins with flux > min-flux (STENVI spec; automatic with any --mem-* grid)")
    ap.add_argument("--no-plot", action="store_true", help="don't plot the output with plot_stenvi.py")
    ap.add_argument("--size-density", nargs="?", const=0.15, type=float, metavar="STRENGTH",
                    help="raise densities below --density-split for particles smaller than --ref-diameter, "
                         "following the Divine (1986) radius law; STRENGTH = max fraction of the gap to the "
                         "split (default 0.15). Densities above the split are unchanged")
    ap.add_argument("--density-split", type=float, default=4000.0, metavar="KG_M3",
                    help="with --size-density: densities below this are shifted (default 4000 kg/m^3)")
    ap.add_argument("--ref-diameter", type=float, default=1e-3, metavar="M",
                    help="with --size-density: MEM's density distribution holds at and above this diameter "
                         "(default 1e-3 m, MEM densities are for mm-size particles)")
    ap.add_argument("--plot-diameter", action="store_true",
                    help="also plot the mean diameter panel (plot_stenvi --show-diameter)")
    ap.add_argument("--plot-log-density", action="store_true",
                    help="plot the density colorbar on a log scale (plot_stenvi --density-scale log)")
    ap.add_argument("--plot-full-range", action="store_true",
                    help="plot with colorbar limits from the header min/max instead of the data (plot_stenvi --full-range)")
    args = ap.parse_args()

    header, bins = read_template(args.template)
    m0 = limiting_mass(args.run)

    replaced = set()
    if args.mem_angles:
        bins["DISTAZI"] = edges_to_bins(np.arange(-180, 181, 5))
        bins["DISTELE"] = edges_to_bins(np.arange(-90, 91, 5))
        replaced |= {"DISTAZI", "DISTELE"}
    if args.mem_velocity:
        _, _, v_lo, v_hi, _ = read_mem_flux(Path(args.run) / "HiDensity" / "flux_avg.txt")
        bins["DISTVEL"] = np.column_stack([v_lo, v_hi])
        replaced.add("DISTVEL")
    if args.mem_density:
        d = np.loadtxt(Path(args.run) / "hidensity.txt", comments="#") / 1000.0
        step = max(1, round(args.mem_density / (d[0, 1] - d[0, 0])))
        bins["DISTDEN"] = edges_to_bins(np.append(d[::step, 0], d[-1, 1]))
        replaced.add("DISTDEN")
    skip_zeros = args.skip_zeros or bool(replaced)

    pops = []
    for pop in ("Hi", "Lo"):
        dirvel, w, mem_total, lost = population_to_bins(
            args.run, pop, bins, m0, not args.no_fold_density, not args.mem_azimuth, args.az_offset,
            (args.size_density, args.density_split / 1000.0, args.ref_diameter) if args.size_density else None)
        pops.append((dirvel, w))
        print(f"{pop}Density: MEM flux (m > {m0:g} g) = {mem_total:.4e} /m^2/yr; "
              f"gridded at 1e-6 g = {dirvel.sum():.4e}; density fraction dropped (outside SEI bins) = {lost:.3e}")

    begin, end, a, e, inc, raan, argp = orbit_from_state_vectors(
        Path(args.run) / "input.txt", central_body_mu(args.run))
    end = end.replace(minute=0, second=0, microsecond=0) + (timedelta(hours=1) if end.minute or end.second else timedelta())
    header = build_header(header, mem_version(args.run), begin, end, a, e, inc, raan, argp, bins, replaced)

    # Flux[az, el, vel, dia, lat, den] = sum_p dirvel_p[az, el, vel] * w_p[den, dia] / nlat,
    # written in chunks of (az, el, vel) cells to keep memory bounded on fine grids.
    nlat = len(bins["DISTLAT"])
    (h_dv, h_w), (l_dv, l_w) = pops
    cells = np.argwhere((h_dv + l_dv) > 0) if skip_zeros else np.argwhere(np.ones_like(h_dv, bool))
    nden, ndia = h_w.shape
    sub = np.indices((ndia, nlat, nden)).reshape(3, -1).T + 1  # [dia, lat, den] in file order
    n_written, total = 0, 0.0
    with open(args.out, "w") as fh:
        fh.write("\n".join(header) + "\n")
        for chunk in np.array_split(cells, max(1, len(cells) // 2000)):
            a_, e_, v_ = chunk.T
            f = (h_dv[a_, e_, v_, None, None] * h_w.T + l_dv[a_, e_, v_, None, None] * l_w.T) / nlat  # [cell, dia, den]
            f = np.repeat(f[:, :, None, :], nlat, axis=2).reshape(len(chunk), -1)
            idx = np.concatenate([np.repeat(chunk + 1, len(sub), axis=0), np.tile(sub, (len(chunk), 1))], axis=1)
            vals = f.ravel()
            keep = vals > args.min_flux if skip_zeros else slice(None)
            np.savetxt(fh, np.column_stack([idx[keep], vals[keep]]),
                       fmt=["DISTSET %d"] + ["%d"] * 5 + ["%.4e "], delimiter=" ")
            n_written += vals[keep].size
            total += vals.sum()
        fh.write("#-<EOF>" + "-" * 73 + "\n")

    print(f"Total flux in SEI file (all diameter bins): {total:.4e} /m^2/yr")
    print(f"Wrote {n_written} DISTSET lines to {args.out}")

    if not args.no_plot:
        import plot_stenvi
        plot_stenvi.main([str(args.out)] + (["--full-range"] if args.plot_full_range else [])
                         + (["--density-scale", "log"] if args.plot_log_density else [])
                         + (["--show-diameter"] if args.plot_diameter else []))


if __name__ == "__main__":
    main()
