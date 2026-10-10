"""Remove STENVI (.sei) flux rows below a mass threshold and rows with zero flux.

Each bin's mass is computed for a spherical particle, m = rho * pi/6 * d^3, from the
DISTDIA (diameter [m]) and DISTDEN (density [g/cm^3]) intervals in the file header.
Header definitions, bin indices, and retained rows are preserved unchanged.
The file is processed one line at a time to handle large inputs.

Usage:
    python zero_small_mass_flux.py --sei L2_eclip.sei
    python zero_small_mass_flux.py --sei L2_eclip.sei -m 1e-5 --mode upper -o out.sei
"""
import argparse
import math
import os


def bin_value(lo, hi, mode, log=False):
    if mode == "lower":
        return lo
    if mode == "upper":
        return hi
    return math.sqrt(lo * hi) if log else 0.5 * (lo + hi)


def zero_small_mass(src, dst, mass_min=1e-6, mode="center"):
    """Write a compact copy, omitting below-threshold and exactly zero flux rows."""
    if (os.path.realpath(src) == os.path.realpath(dst)
            or (os.path.exists(dst) and os.path.samefile(src, dst))):
        raise ValueError("Input and output must be different files.")
    dia, den = {}, {}
    removed_small = removed_zero = total = 0
    small = None  # set of (dia_idx, den_idx) to omit; built at the flux section

    with open(src, newline="") as fin, open(dst, "w", newline="") as fout:  # keep original line endings
        for line in fin:
            if line.startswith("DISTSET"):
                if small is None:
                    small = set()
                    for i, (dlo, dhi) in dia.items():
                        d_cm = bin_value(dlo, dhi, mode, log=True) * 100.0
                        for j, (rlo, rhi) in den.items():
                            rho = bin_value(rlo, rhi, mode)
                            if rho * math.pi / 6.0 * d_cm ** 3 < mass_min:
                                small.add((i, j))
                    print(f"{len(small)} of {len(dia) * len(den)} diameter/density bins below {mass_min:g} g")
                total += 1
                p = line.split()
                if (int(p[4]), int(p[6])) in small:
                    removed_small += 1
                    continue
                if float(p[7].replace("D", "E").replace("d", "e")) == 0.0:
                    removed_zero += 1
                    continue
            elif line.startswith("DISTDIA"):
                p = line.split()
                dia[int(p[1])] = (float(p[2]), float(p[3]))
            elif line.startswith("DISTDEN"):
                p = line.split()
                den[int(p[1])] = (float(p[2]), float(p[3]))
            fout.write(line)

    kept = total - removed_small - removed_zero
    print(f"Removed {removed_small} below-threshold rows and {removed_zero} other zero-flux rows; "
          f"kept {kept} of {total} flux entries -> {dst}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sei", default=r"C:\Users\maxiv\Documents\UWO\IMEM2_implement\IMEM2_stenvi\stenvi_5deg\sunoriented_IMEM2_newrho.sei", help="input .sei file")
    ap.add_argument("-m", "--mass", type=float, default=1e-6, help="mass threshold [g] (default 1e-6)")
    ap.add_argument("--mode", choices=["lower", "center", "upper"], default="center", # for IMEM lower for MEM center/newrho
                    help="bin edge used for diameter/density: center (geometric mean for diameter, "
                         "arithmetic for density), lower, or upper (only remove bins entirely below threshold)")
    ap.add_argument("-o", "--output", help="output file (default: <input>_mgt<mass>.sei)")
    a = ap.parse_args()
    out = a.output or f"{os.path.splitext(a.sei)[0]}_mgt{a.mass:g}.sei"
    zero_small_mass(a.sei, out, a.mass, a.mode)
