#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import imageio as iio
from scipy.optimize import curve_fit


ARCSEC_TO_RAD = np.pi / (180.0 * 3600.0)


def byteswap_image(img: np.ndarray) -> np.ndarray:
    """
    Byte-swap pixel values. Do this by default (as requested).
    Safe no-op for 8-bit images.
    """
    # byteswap() swaps data; newbyteorder() fixes dtype endianness metadata
    return img.byteswap()


def gaussian(x, A, mu, sigma, C):
    return C + A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def two_gaussian(x, A1, mu1, s1, A2, mu2, s2, C):
    return (
        C
        + A1 * np.exp(-0.5 * ((x - mu1) / s1) ** 2)
        + A2 * np.exp(-0.5 * ((x - mu2) / s2) ** 2)
    )


def central_third(x, y):
    n = len(y)
    a = n // 3
    b = 2 * n // 3
    return x[a:b], y[a:b]


def maybe_invert(y):
    # If center is darker than background, invert so the "star" is a positive peak.
    mid = y[len(y) // 2]
    bg = np.median(y)
    if mid < bg:
        return (y.max() - y)
    return y


def fit_1g(x, y):
    C0 = np.median(y)
    A0 = float(np.max(y) - C0)
    mu0 = float(x[np.argmax(y)])
    s0 = 3.0

    p0 = [A0, mu0, s0, C0]
    bounds = ([0.0, x.min(), 0.2, -np.inf],
              [np.inf, x.max(), 100.0, np.inf])

    popt, pcov = curve_fit(gaussian, x, y, p0=p0, bounds=bounds, maxfev=20000)
    return popt, pcov


def fit_2g(x, y):
    C0 = np.median(y)
    A0 = float(np.max(y) - C0)
    mu0 = float(x[np.argmax(y)])

    # Simple starting guesses: narrow core + wider wing, both centered near mu0
    p0 = [0.7 * A0, mu0, 2.0,
          0.3 * A0, mu0, 6.0,
          C0]

    bounds = ([0.0, x.min(), 0.2,  0.0, x.min(), 0.5,  -np.inf],
              [np.inf, x.max(), 50.0, np.inf, x.max(), 200.0, np.inf])

    popt, pcov = curve_fit(two_gaussian, x, y, p0=p0, bounds=bounds, maxfev=50000)
    return popt, pcov


def variance_to_psf_m(variance_pix, arcsec_per_pix=6.0, range_m=100_000.0):
    # User-specified rule:
    # x_arcsec = arcsec_per_pix * variance_pix
    x_rad = (arcsec_per_pix * variance_pix) * ARCSEC_TO_RAD
    # 6 arc sec per pixel * Variance of gaussian = x
    # height of the meteor 100 km * tang(x) = PSF for both axes
    return range_m * np.tan(x_rad)


def process_axis(profile, axis_name, outdir, arcsec_per_pix=6.0, range_m=100_000.0):
    # Build x in pixel units centered at 0
    n = len(profile)
    x = np.arange(n, dtype=float) - (n // 2)

    y = maybe_invert(profile.astype(float))

    # # keep only the one between + and - 20 pixels
    mask = (x >= -10) & (x <= 10)
    x_c = x[mask]
    y_c = y[mask]
    
    # # Keep only central third
    # x_c, y_c = central_third(x, y)
    # x_c, y_c = central_third(x_c, y_c)

    # Fit 1 Gaussian and 2 Gaussians
    p1, _ = fit_1g(x_c, y_c)
    p2, _ = fit_2g(x_c, y_c)

    # Extract sigmas and variances (variance = sigma^2)
    sigma_1g = float(p1[2])
    var_1g = sigma_1g ** 2

    s1 = float(p2[2]); var_s1 = s1 ** 2
    s2 = float(p2[5]); var_s2 = s2 ** 2

    # Convert to PSF meters using your rule
    psf_1g_m = variance_to_psf_m(var_1g, arcsec_per_pix, range_m)
    psf_2g_m_1 = variance_to_psf_m(var_s1, arcsec_per_pix, range_m)
    psf_2g_m_2 = variance_to_psf_m(var_s2, arcsec_per_pix, range_m)

    # Make plot
    xx = np.linspace(x_c.min(), x_c.max(), 800)
    y1 = gaussian(xx, *p1)
    y2 = two_gaussian(xx, *p2)

    # Components for 2G (for visualization)
    comp1 = p2[6] + p2[0] * np.exp(-0.5 * ((xx - p2[1]) / p2[2]) ** 2)
    comp2 = p2[6] + p2[3] * np.exp(-0.5 * ((xx - p2[4]) / p2[5]) ** 2)

    plt.figure(figsize=(8, 5))
    plt.plot(x_c, y_c, ".", label=f"{axis_name} profile (central third)")
    plt.plot(xx, y1, "r-", label=f"1G fit (sigma={sigma_1g:.2f}px, PSF={psf_1g_m:.3f} m)")
    plt.plot(xx, y2, "k-", label=f"2G sum (PSF1={psf_2g_m_1:.3f} m, PSF2={psf_2g_m_2:.3f} m)")
    plt.plot(xx, comp1, ":", label=f"2G comp1 (sigma={s1:.2f}px)")
    plt.plot(xx, comp2, ":", label=f"2G comp2 (sigma={s2:.2f}px)")
    plt.xlabel("Pixels (relative to center)")
    plt.ylabel("Intensity (inverted if needed)")
    plt.title(f"PSF fit along {axis_name}-axis cut")
    plt.legend(fontsize=9)
    plt.tight_layout()

    outpath = outdir / f"psf_fit_{axis_name}.png"
    plt.savefig(outpath, dpi=200)
    plt.close()

    return {
        "axis": axis_name,
        "sigma_1g_px": sigma_1g,
        "var_1g_px2": var_1g,
        "psf_1g_m": psf_1g_m,
        "sigma_2g_px": [s1, s2],
        "var_2g_px2": [var_s1, var_s2],
        "psf_2g_m": [psf_2g_m_1, psf_2g_m_2],
        "plot": str(outpath),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=r"/srv/public/mvovk/wake_runs_dynesty/PSF/Elginfield_corr_1766282128_019302.png",
                    help="Path to *_corr.png")
    ap.add_argument("--outdir", default=None,
                    help="Output directory (default: same as image)")
    ap.add_argument("--arcsec-per-pix", type=float, default=6.0,
                    help="Camera plate scale (arcsec/pixel)")
    ap.add_argument("--range-km", type=float, default=100.0,
                    help="Assumed meteor range (km)")
    ap.add_argument("--no-byteswap", action="store_true",
                    help="Disable byte-swapping (byte-swap is ON by default).")
    args = ap.parse_args()

    img_path = Path(args.image)
    outdir = Path(args.outdir) if args.outdir else img_path.parent
    outdir.mkdir(parents=True, exist_ok=True)

    img = iio.imread(img_path)

    # Byte-swap by default (as requested)
    if not args.no_byteswap:
        print("Byte-swapping image...")
        img = byteswap_image(img)

    if img.ndim == 3:
        img = img[..., 0]  # first channel if RGB

    img = img.astype(float)

    h, w = img.shape
    cy, cx = h // 2, w // 2

    # Requested cuts:
    prof_y = img[:, cx]   # y-axis cut (middle column)
    prof_x = img[cy, :]   # x-axis cut (middle row)

    range_m = args.range_km * 1000.0

    res_y = process_axis(prof_y, "y", outdir, args.arcsec_per_pix, range_m)
    res_x = process_axis(prof_x, "x", outdir, args.arcsec_per_pix, range_m)

    def line(res):
        psf1 = res["psf_1g_m"]
        psf2a, psf2b = res["psf_2g_m"]
        return (f"{res['axis'].upper()} axis:  "
                f"PSF_1G={psf1:.4f} m,  "
                f"PSF_2G(comp1)={psf2a:.4f} m,  "
                f"PSF_2G(comp2)={psf2b:.4f} m  |  plot: {res['plot']}")

    print("\n=== PSF (meters) at range ===")
    print(line(res_y))
    print(line(res_x))
    print("")


if __name__ == "__main__":
    main()
