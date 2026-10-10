"""
Re-distribute the density dimension of an IMEM STENVI (.sei) file.

For every (azimuth, elevation, velocity, diameter, latitude) cell the IMEM
flux is split into the part with rho < RHO_CUT and the part with rho > RHO_CUT
(1.8 g/cm^3 by default).  The low part is spread over the new density bins
following lodensity.txt, the high part following hidensity.txt:

    F_new[..., k] = F_lo[...] * w_lo[k] + F_hi[...] * w_hi[k]

so the flux in every direction / speed / size bin is unchanged; only the
density spectrum is replaced.  The new density grid is 0.1 -> 8.0 g/cm^3 in
0.5 steps (16 bins, last one 7.6-8.0), as in the MEM3 -> SEI files.

Size-dependent density (on by default, --no-size-density to disable): as in
mem3_to_sei.py, the MEM tables are taken to hold for mm-size particles; for
smaller diameter bins the table densities below 4000 kg/m^3 are moved towards
4000 following the Divine et al. (1986) Halley radius law, so w_lo / w_hi
become w_lo[dia, k] / w_hi[dia, k].

Usage:
    python imem_newrho.py [--sei IN.sei] [--hi hidensity.txt] [--lo lodensity.txt]
                          [--out OUT.sei] [--cut 1.8] [--step 0.5]
                          [--size-density 0.15 | --no-size-density]
                          [--density-split 4000] [--ref-diameter 1e-3]
"""
import argparse
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# DEF_SEI = os.path.join(HERE, "IMEM_stenvi_5deg", "sunoriented.sei")
# DEF_RHO = os.path.join(HERE, "L2_bodyfix", "L2sei_newrho_speedW")
DIMS = ["DISTAZI", "DISTELE", "DISTVEL", "DISTDIA", "DISTLAT", "DISTDEN"]


def read_sei(path):
    """Return header lines, bin edges per dimension and the 6-D flux array."""
    header, edges, rows = [], {d: [] for d in DIMS}, []
    with open(path) as f:
        for line in f:
            if line.startswith("DISTSET"):
                rows.append(line)
                continue
            key = line.split(" ", 1)[0]
            if key in edges:
                _, _, lo, hi = line.split()[:4]
                edges[key].append((float(lo), float(hi)))
            header.append(line)
    data = np.loadtxt(rows, usecols=range(1, 8))
    idx = data[:, :6].astype(int) - 1
    shape = tuple(len(edges[d]) for d in DIMS)
    flux = np.zeros(shape)
    flux[tuple(idx.T)] = data[:, 6]
    return header, {d: np.array(e) for d, e in edges.items()}, flux


def overlap(a_lo, a_hi, b_lo, b_hi):
    """Overlap length matrix between intervals a (rows) and b (cols)."""
    lo = np.maximum(a_lo[:, None], b_lo[None, :])
    hi = np.minimum(a_hi[:, None], b_hi[None, :])
    return np.clip(hi - lo, 0, None)


def rho_divine1986(radius_um):
    """Divine et al. (1986) Halley dust bulk density [g/cm^3] vs grain radius [um]:
    3.0 for small grains, falling to 0.8 for large ones."""
    a = np.asarray(radius_um, float)
    return 3.0 - 2.2 * a / (a + 2.0)


def density_shift(dia_lo, dia_hi, strength, split, ref_diameter):
    """Per-diameter-bin shift t(D) of low densities towards `split` (rho -> rho + t*(split - rho)).

    t(D) = strength * clip((f(D/2) - f(Dref/2)) / (f(0) - f(Dref/2)), 0, 1), with f the Divine
    (1986) radius law and D the geometric bin centre [m]: MEM's distribution holds for D >= Dref
    (mm-size), smaller particles move up towards the split, never crossing it.
    Same as mem3_to_sei.py."""
    radius_um = np.sqrt(dia_lo * dia_hi) * 5e5
    f_ref, f_zero = rho_divine1986(ref_diameter * 5e5), rho_divine1986(0.0)
    return strength * np.clip((rho_divine1986(radius_um) - f_ref) / (f_zero - f_ref), 0.0, 1.0)


def read_pdf(path, new_edges, shifts=(0.0,), split=0.0):
    """Rebin a MEM3 density fraction table (kg/m^3) onto new_edges (g/cm^3).

    One row per shift t: table edges below `split` are first moved to
    e + t*(split - e) (fractions carried with them).  Returns [len(shifts), nden]."""
    lo, hi, frac = np.loadtxt(path, comments="#", unpack=True)
    lo, hi = lo / 1000.0, hi / 1000.0
    out = []
    for t in shifts:
        l, h = [np.where(e < split, e + t * (split - e), e) for e in (lo, hi)]
        w = (overlap(l, h, new_edges[:, 0], new_edges[:, 1])
             / (h - l)[:, None] * frac[:, None]).sum(axis=0)
        out.append(w / w.sum())
    return np.array(out)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sei", default=r"C:\Users\maxiv\Documents\UWO\IMEM2_implement\IMEM2_stenvi\stenvi_5deg\sunoriented.sei")
    p.add_argument("--hi", default=r"C:\Users\maxiv\Documents\UWO\IMEM2_implement\MEM3orbit\L2_bodyfix\L2sei_newrho_speedW\hidensity.txt")
    p.add_argument("--lo", default=r"C:\Users\maxiv\Documents\UWO\IMEM2_implement\MEM3orbit\L2_bodyfix\L2sei_newrho_speedW\lodensity.txt")
    p.add_argument("--out", default=r"C:\Users\maxiv\Documents\UWO\IMEM2_implement\IMEM2_stenvi\stenvi_5deg\sunoriented_IMEM2_newrho.sei")
    p.add_argument("--cut", type=float, default=3.0, help="rho split [g/cm^3]")
    p.add_argument("--rmin", type=float, default=0.1)
    p.add_argument("--rmax", type=float, default=8.0)
    p.add_argument("--step", type=float, default=0.5)
    p.add_argument("--size-density", type=float, default=0.15, metavar="STRENGTH",
                   help="raise densities below --density-split for particles smaller than "
                        "--ref-diameter, following Divine (1986); STRENGTH = max fraction of "
                        "the gap to the split (default 0.15, as mem3_to_sei.py)")
    p.add_argument("--no-size-density", dest="size_density", action="store_const", const=0.0,
                   help="use the mm-size density distribution for every diameter bin")
    p.add_argument("--density-split", type=float, default=4000.0, metavar="KG_M3",
                   help="densities below this are shifted (default 4000 kg/m^3)")
    p.add_argument("--ref-diameter", type=float, default=1e-3, metavar="M",
                   help="MEM densities hold at and above this diameter (default 1e-3 m)")
    a = p.parse_args()
    out = a.out or os.path.splitext(a.sei)[0] + "_newrho.sei"

    print(f"Reading {a.sei}")
    header, edges, flux = read_sei(a.sei)
    print(f"  flux array {flux.shape}, total {flux.sum():.6e} /m^2/yr")

    # New density grid: 0.1, 0.6, ..., 7.6, 8.0
    b = np.append(np.arange(a.rmin, a.rmax, a.step), a.rmax)
    new_den = np.column_stack([b[:-1], b[1:]])
    nden = len(new_den)

    # Fraction of each old IMEM density bin that lies below / above the cut
    old = edges["DISTDEN"]
    f_lo = np.clip((a.cut - old[:, 0]) / (old[:, 1] - old[:, 0]), 0, 1)
    F_lo = flux @ f_lo
    F_hi = flux @ (1 - f_lo)

    # Density weights per diameter bin [dia, den] (identical rows if size density is off)
    dia = edges["DISTDIA"]
    split = a.density_split / 1000.0
    t = (density_shift(dia[:, 0], dia[:, 1], a.size_density, split, a.ref_diameter)
         if a.size_density else np.zeros(len(dia)))
    print("  size-density shift t(D): " + " ".join(f"{x:.3f}" for x in t))
    w_lo = read_pdf(a.lo, new_den, t, split)
    w_hi = read_pdf(a.hi, new_den, t, split)
    # F_* are [azi, ele, vel, dia, lat]; broadcast weights over the diameter axis
    new = (F_lo[..., None] * w_lo[:, None, :] + F_hi[..., None] * w_hi[:, None, :])
    print(f"  rho<{a.cut}: {F_lo.sum():.6e}   rho>{a.cut}: {F_hi.sum():.6e}")
    print(f"  new total {new.sum():.6e} (rel. diff "
          f"{abs(new.sum() - flux.sum()) / flux.sum():.2e})")

    # Header: swap the DENSITY definition and the DISTDEN interval block
    out_hdr = []
    for line in header:
        if line.startswith("DENSITY "):
            line = f"DENSITY {nden} {a.rmin} {a.rmax} Density [g/cm^3]\n"
        elif line.startswith("DISTDEN"):
            if line.startswith("DISTDEN 1 "):
                out_hdr += [f"DISTDEN {k + 1} {lo:.4e} {hi:.4e}\n"
                            for k, (lo, hi) in enumerate(new_den)]
            continue
        out_hdr.append(line)

    print(f"Writing {out} ({new.size} DISTSET lines)")
    idx = np.indices(new.shape).reshape(new.ndim, -1).T + 1
    with open(out, "w", newline="\n") as f:
        f.writelines(out_hdr)
        np.savetxt(f, np.column_stack([idx, new.ravel()]),
                   fmt="DISTSET %d %d %d %d %d %d %.4e ")
    print("Done.")


if __name__ == "__main__":
    main()
