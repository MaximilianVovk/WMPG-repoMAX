"""Zero the flux of STENVI (.sei) Flux Contribution bins whose particle mass is below a threshold.

Each bin's mass is computed for a spherical particle, m = rho * pi/6 * d^3, from the
DISTDIA (diameter [m]) and DISTDEN (density [g/cm^3]) intervals in the file header.

Usage:
    python zero_small_mass_flux.py L2_eclip.sei
    python zero_small_mass_flux.py L2_eclip.sei -m 1e-5 --mode upper -o out.sei
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
    dia, den = {}, {}
    zeroed = total = 0
    small = None  # set of (dia_idx, den_idx) to zero; built when the flux section starts

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
                    zeroed += 1
                    eol = line[len(line.rstrip("\r\n")):]
                    line = f"DISTSET {' '.join(p[1:7])} 0.0000e+00 {eol}"
            elif line.startswith("DISTDIA"):
                p = line.split()
                dia[int(p[1])] = (float(p[2]), float(p[3]))
            elif line.startswith("DISTDEN"):
                p = line.split()
                den[int(p[1])] = (float(p[2]), float(p[3]))
            fout.write(line)

    print(f"Zeroed {zeroed} of {total} flux entries -> {dst}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sei", default=r"C:\Users\maxiv\Documents\UWO\IMEM2_implement\MEM3orbit\L2_sunoriented.sei", help="input .sei file")
    ap.add_argument("-m", "--mass", type=float, default=1e-6, help="mass threshold [g] (default 1e-6)")
    ap.add_argument("--mode", choices=["lower", "center", "upper"], default="center",
                    help="bin edge used for diameter/density: center (geometric mean for diameter, "
                         "arithmetic for density), lower, or upper (only zero bins entirely below threshold)")
    ap.add_argument("-o", "--output", help="output file (default: <input>_mgt<mass>.sei)")
    a = ap.parse_args()
    out = a.output or f"{os.path.splitext(a.sei)[0]}_mgt{a.mass:g}.sei"
    zero_small_mass(a.sei, out, a.mass, a.mode)
