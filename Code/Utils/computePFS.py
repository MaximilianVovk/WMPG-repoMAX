#!/usr/bin/env python3
import argparse
from pathlib import Path
import re

import numpy as np
import matplotlib.pyplot as plt
import imageio as iio
from scipy.optimize import curve_fit
from scipy.optimize import minimize
import pandas as pd


ARCSEC_TO_RAD = np.pi / (180.0 * 3600.0)


def byteswap_image(img: np.ndarray) -> np.ndarray:
    """
    Byte-swap pixel values. Do this by default (as requested).
    Safe no-op for 8-bit images.
    """
    return img.byteswap()


def gaussian(x, A, mu, sigma, C):
    return C + A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def two_gaussian(x, A1, mu1, s1, A2, mu2, s2, C):
    return (
        C
        + A1 * np.exp(-0.5 * ((x - mu1) / s1) ** 2)
        + A2 * np.exp(-0.5 * ((x - mu2) / s2) ** 2)
    )


def maybe_invert(y):
    """
    Decide inversion without assuming the peak is at the center.
    If the strongest deviation from background is a *dip* (dark star),
    invert so the signal becomes a positive peak.
    """
    y = np.asarray(y, dtype=float)
    bg = np.median(y)

    # Compare absolute deviation above vs below background
    dev_up = np.max(y) - bg
    dev_dn = bg - np.min(y)

    if dev_dn > dev_up:
        # stronger dip than peak -> invert
        return (np.max(y) - y)
    return y


def peak_index(y, smooth_window=7):
    """
    Return index of the peak, using light smoothing to avoid noise spikes.
    """
    y = np.asarray(y, dtype=float)

    if smooth_window and smooth_window > 1:
        w = int(smooth_window)
        if w % 2 == 0:
            w += 1
        kernel = np.ones(w) / w
        y_s = np.convolve(y, kernel, mode="same")
    else:
        y_s = y

    return int(np.argmax(y_s))



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

def _sigmoid(z):
    # stable sigmoid
    z = np.clip(z, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))

def _map_to_bounds(z, lo, hi):
    # maps R -> (lo, hi)
    return lo + (hi - lo) * _sigmoid(z)

def _safe_exp(z):
    # maps R -> (0, +inf), safely
    z = np.clip(z, -60.0, 60.0)
    return np.exp(z)

def fit_1g_NelderMead(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    xmin, xmax = float(x.min()), float(x.max())

    # initial guesses in *physical* space
    C0 = float(np.median(y))
    A0 = float(np.max(y) - C0)
    mu0 = float(x[np.argmax(y)])
    s0 = 3.0

    # transform initial guesses to unconstrained variables
    # A = exp(a), mu = xmin + (xmax-xmin)*sigmoid(m), sigma = smin + (smax-smin)*sigmoid(s), C free
    smin, smax = 0.2, 100.0

    a0 = np.log(max(A0, 1e-12))
    # invert sigmoid approximately for mu/sigma initial guesses
    def inv_sigmoid(u):
        u = np.clip(u, 1e-6, 1 - 1e-6)
        return np.log(u / (1 - u))

    m0 = inv_sigmoid((mu0 - xmin) / (xmax - xmin + 1e-12))
    t0 = inv_sigmoid((s0 - smin) / (smax - smin))

    p0 = np.array([a0, m0, t0, C0], dtype=float)

    # scale for SSE
    sse_scale = float(np.mean((y - C0) ** 2) + 1e-12)

    def obj(p):
        a, m, t, C = p
        A = _safe_exp(a)
        mu = _map_to_bounds(m, xmin, xmax)
        sig = _map_to_bounds(t, smin, smax)
        model = gaussian(x, A, mu, sig, C)
        r = y - model
        return float(np.sum(r * r)) / sse_scale

    res = minimize(
        obj, p0,
        method="Nelder-Mead",
        options=dict(
            maxiter=4000,      # lower = faster
            xatol=1e-4,        # looser tolerances = faster
            fatol=1e-4,
            adaptive=True,
        )
    )

    a, m, t, C = res.x
    A = _safe_exp(a)
    mu = _map_to_bounds(m, xmin, xmax)
    sig = _map_to_bounds(t, smin, smax)
    popt = np.array([A, mu, sig, C], dtype=float)
    return popt, None


def fit_2g_NelderMead(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    xmin, xmax = float(x.min()), float(x.max())

    C0 = float(np.median(y))
    A0 = float(np.max(y) - C0)
    mu0 = float(x[np.argmax(y)])

    # bounds
    s1min, s1max = 0.2, 50.0
    s2min, s2max = 0.5, 200.0

    def inv_sigmoid(u):
        u = np.clip(u, 1e-6, 1 - 1e-6)
        return np.log(u / (1 - u))

    # initial guesses in physical space
    A10 = 0.7 * A0
    A20 = 0.3 * A0
    mu10 = mu0
    mu20 = mu0
    s10 = 2.0
    s20 = 6.0

    # transform initial guesses to unconstrained
    a10 = np.log(max(A10, 1e-12))
    a20 = np.log(max(A20, 1e-12))
    m10 = inv_sigmoid((mu10 - xmin) / (xmax - xmin + 1e-12))
    m20 = inv_sigmoid((mu20 - xmin) / (xmax - xmin + 1e-12))
    t10 = inv_sigmoid((s10 - s1min) / (s1max - s1min))
    t20 = inv_sigmoid((s20 - s2min) / (s2max - s2min))

    p0 = np.array([a10, m10, t10, a20, m20, t20, C0], dtype=float)

    sse_scale = float(np.mean((y - C0) ** 2) + 1e-12)

    def obj(p):
        a1, m1, t1, a2, m2, t2, C = p
        A1 = _safe_exp(a1)
        A2 = _safe_exp(a2)
        mu1 = _map_to_bounds(m1, xmin, xmax)
        mu2 = _map_to_bounds(m2, xmin, xmax)
        s1 = _map_to_bounds(t1, s1min, s1max)
        s2 = _map_to_bounds(t2, s2min, s2max)

        model = two_gaussian(x, A1, mu1, s1, A2, mu2, s2, C)
        r = y - model
        return float(np.sum(r * r)) / sse_scale

    res = minimize(
        obj, p0,
        method="Nelder-Mead",
        options=dict(
            maxiter=8000,
            xatol=2e-4,
            fatol=2e-4,
            adaptive=True,
        )
    )

    a1, m1, t1, a2, m2, t2, C = res.x
    A1 = _safe_exp(a1)
    A2 = _safe_exp(a2)
    mu1 = _map_to_bounds(m1, xmin, xmax)
    mu2 = _map_to_bounds(m2, xmin, xmax)
    s1 = _map_to_bounds(t1, s1min, s1max)
    s2 = _map_to_bounds(t2, s2min, s2max)

    popt = np.array([A1, mu1, s1, A2, mu2, s2, C], dtype=float)
    return popt, None


def variance_to_psf_m(variance_pix, arcsec_per_pix=6.0, range_m=100_000.0):
    x_rad = (arcsec_per_pix * variance_pix) * ARCSEC_TO_RAD
    return range_m * np.tan(x_rad)


def extract_event_id(png_path: Path) -> str:
    """
    From:
      Tavistock_corr_1766400731_006487.png
    return:
      1766400731_006487

    Falls back to the stem if pattern isn't matched.
    """
    name = png_path.name
    m = re.match(r"^(?:Elginfield|Tavistock)_corr_(.+)\.png$", name)
    if m:
        return m.group(1)
    # fallback: remove extension
    return png_path.stem


def process_axis(profile, axis_name, outdir, event_id, arcsec_per_pix=6.0, range_m=100_000.0):
    n = len(profile)

    # Make signal positive if needed (does NOT assume center)
    y = maybe_invert(profile.astype(float))

    # Find peak and recenter so peak is at x=0
    pk = peak_index(y, smooth_window=7)
    x = np.arange(n, dtype=float) - pk

    # Use full profile (or you can re-enable your masks after this)
    x_c = x
    y_c = y

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
    xx = np.linspace(x_c.min(), x_c.max(), 8000)
    y1 = gaussian(xx, *p1)
    y2 = two_gaussian(xx, *p2)

    # Components for 2G (for visualization)
    comp1 = p2[6] + p2[0] * np.exp(-0.5 * ((xx - p2[1]) / p2[2]) ** 2)
    comp2 = p2[6] + p2[3] * np.exp(-0.5 * ((xx - p2[4]) / p2[5]) ** 2)

    plt.figure(figsize=(8, 5))
    plt.plot(x_c, y_c, ".", label=f"{axis_name} profile")
    plt.plot(xx, y1, "r-", label=f"1G fit (sigma={sigma_1g:.2f}px\nA={p1[0]:.2e},\nPSF={psf_1g_m:.3f} m)")
    plt.plot(xx, y2, "k-", label=f"2G sum (\nPSF1={psf_2g_m_1:.3f} m,\nPSF2={psf_2g_m_2:.3f} m)")
    plt.plot(xx, comp1, ":", label=f"2G comp1 (sigma={s1:.2f}px\nA={p2[0]:.2e})")
    plt.plot(xx, comp2, ":", label=f"2G comp2 (sigma={s2:.2f}px\nA={p2[3]:.2e})")
    plt.xlabel("Pixels (relative to center)")
    plt.ylabel("Intensity")
    plt.title(f"PSF fit along {axis_name}-axis cut | {event_id}")
    plt.legend(fontsize=10, loc="upper right")
    plt.xlim(-10, 10)
    plt.tight_layout()

    outpath = outdir / f"{event_id}_psf_fit_{axis_name}.png"
    plt.savefig(outpath, dpi=200)
    plt.close()

    return {
        "axis": axis_name,
        "sigma_1g_px": sigma_1g,
        "var_1g_px2": var_1g,
        "psf_1g_m": psf_1g_m,
        "A_1g": p1[0],
        "sigma_2g_px": [s1, s2],
        "var_2g_px2": [var_s1, var_s2],
        "psf_2g_m": [psf_2g_m_1, psf_2g_m_2],
        "A_2g": [p2[0], p2[3]],
        "plot": str(outpath),
    }


def iter_matching_pngs(path: Path):
    """
    If path is file -> return [path]
    If path is dir  -> return all pngs containing Elginfield_corr_ or Tavistock_corr_
    """
    if path.is_file():
        return [path]

    if not path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {path}")

    patterns = ("Elginfield_corr_", "Tavistock_corr_")
    hits = []
    for p in sorted(path.rglob("*.png")):
        name = p.name
        if any(s in name for s in patterns):
            hits.append(p)
    return hits


def process_one_image(img_path: Path, outdir: Path, arcsec_per_pix: float, range_km: float, no_byteswap: bool):
    event_id = extract_event_id(img_path)

    img = iio.imread(img_path)

    if not no_byteswap:
        img = byteswap_image(img)

    if img.ndim == 3:
        img = img[..., 0]  # first channel if RGB

    img = img.astype(float)

    h, w = img.shape
    cy, cx = h // 2, w // 2

    prof_y = img[:, cx]   # y-axis cut (middle column)
    prof_x = img[cy, :]   # x-axis cut (middle row)

    range_m = range_km * 1000.0

    res_y = process_axis(prof_y, "y", outdir, event_id, arcsec_per_pix, range_m)
    res_x = process_axis(prof_x, "x", outdir, event_id, arcsec_per_pix, range_m)

    def line(res):
        psf1 = res["psf_1g_m"]
        psf2a, psf2b = res["psf_2g_m"]
        return (f"{res['axis'].upper()} axis:  "
                f"PSF_1G={psf1:.4f} m,  "
                f"PSF_2G(comp1)={psf2a:.4f} m,  "
                f"PSF_2G(comp2)={psf2b:.4f} m  |  plot: {res['plot']}")

    print(f"\n[{img_path.name}] event_id={event_id}")
    print("=== PSF (meters) at range ===")
    print(line(res_y))
    print(line(res_x))

    return res_x, res_y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--image",
        default="/srv/meteor/klingon/mirchk/20251222",
        help="Path to a *_corr.png OR a directory to scan recursively for Elginfield_corr_*.png / Tavistock_corr_*.png",
    )
    ap.add_argument(
        "--outdir",
        default=None,
        help="Output directory (default: same as image's directory if --image is a file; or the input dir if --image is a dir)",
    )
    ap.add_argument("--arcsec-per-pix", type=float, default=6.0, help="Camera plate scale (arcsec/pixel)")
    ap.add_argument("--range-km", type=float, default=100.0, help="Assumed meteor range (km)")
    ap.add_argument("--no-byteswap", action="store_true", help="Disable byte-swapping (byte-swap is ON by default).")
    args = ap.parse_args()

    in_path = Path(args.image)

    # Where this script lives
    script_dir = Path(__file__).resolve().parent

    all_res = []  # collect per-axis results across all images

    # Decide outdir
    if args.outdir:
        outdir = Path(args.outdir)
    else:
        if in_path.is_file():
            # single image -> default to script directory
            outdir = script_dir
        else:
            # directory scan -> default to a folder next to the script, named like the data folder
            outdir = script_dir / in_path.name

    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {outdir}")

    # check if in the output directory there is a psf_summary_all_images.csv file if so skip the plotting and load the results
    if (outdir / "psf_summary_all_images.csv").exists():
        print("\nFound existing psf_summary_all_images.csv in output directory, loading results from there.")
        df = pd.read_csv(outdir / "psf_summary_all_images.csv")
   
    else:
        # Collect inputs
        pngs = iter_matching_pngs(in_path)

        if in_path.is_dir():
            print(f"Scanning: {in_path}")
            print(f"Found {len(pngs)} matching PNG files (Elginfield_corr_ / Tavistock_corr_).")
            for p in pngs:
                print(f"  - {p}")
        else:
            print(f"Processing single file: {in_path}")

        if len(pngs) == 0:
            print("No matching files found. Nothing to do.")
            return

        # Process each image
        for p in pngs:
            try:
                res_x, res_y = process_one_image(
                    img_path=p,
                    outdir=outdir,
                    arcsec_per_pix=args.arcsec_per_pix,
                    range_km=args.range_km,
                    no_byteswap=args.no_byteswap,
                )

                event_id = extract_event_id(p)

                # store both axes as separate rows
                for r in (res_x, res_y):
                    all_res.append({
                        "event_id": event_id,
                        "image": str(p),
                        "axis": r["axis"],

                        # 1G
                        "sigma_1g_px": r["sigma_1g_px"],
                        "psf_1g_m": r["psf_1g_m"],
                        "A_1g": r["A_1g"],

                        # 2G (optional summaries)
                        "sigma_2g_1_px": r["sigma_2g_px"][0],
                        "sigma_2g_2_px": r["sigma_2g_px"][1],
                        "psf_2g_1_m": r["psf_2g_m"][0],
                        "psf_2g_2_m": r["psf_2g_m"][1],
                        "A_2g_1": r["A_2g"][0],
                        "A_2g_2": r["A_2g"][1],
                    })

            except Exception as e:
                print(f"\nERROR processing {p}: {e}")

        print(f"\nPlots saved in: {outdir}\n")

        # ------------------------------
        # Summary statistics (all axes, all images)
        # ------------------------------
        def summarize(values, label):
            v = np.array(values, dtype=float)
            v = v[np.isfinite(v)]
            if v.size == 0:
                print(f"{label}: no valid values")
                return
            # delete outlier values greater than 100 m
            mean = float(np.mean(v[v <= 100.0]))
            mode = float(pd.Series(np.round(v[v <= 100.0],1)).mode().iloc[0])
            p2p5, p97p5 = np.percentile(v, [2.5, 97.5])
            print(f"{label}: mean={mean:.6g}, mode={mode:.1f} | 95% range=[{p2p5:.6g}, {p97p5:.6g}] | N={v.size}")

        if len(all_res) > 0:
            print("\n==============================")
            print("GLOBAL SUMMARY (all images, both axes)")
            print("==============================")

            # 1G: sigma + PSF (treat PSF as your 'aperture' in meters)
            summarize([r["sigma_1g_px"] for r in all_res], "1G sigma [px]")
            summarize([r["sigma_2g_1_px"] for r in all_res], "2G comp1 sigma [px]")
            summarize([r["sigma_2g_2_px"] for r in all_res], "2G comp2 sigma [px]")
            summarize([r["psf_1g_m"] for r in all_res], "1G PSF [m]")
            summarize([r["psf_2g_1_m"] for r in all_res], "2G comp1 PSF [m]")
            summarize([r["psf_2g_2_m"] for r in all_res], "2G comp2 PSF [m]")

            for ax in ("x", "y"):
                print(f"\n=============({ax})==============")
                sub = [r for r in all_res if r["axis"] == ax]
                print(f"\n--- Axis {ax.upper()} only ---")
                summarize([r["sigma_1g_px"] for r in sub], f"1G sigma [px] ({ax})")
                summarize([r["sigma_2g_1_px"] for r in sub], f"2G comp1 sigma [px] ({ax})")
                summarize([r["sigma_2g_2_px"] for r in sub], f"2G comp2 sigma [px] ({ax})")
                summarize([r["psf_1g_m"] for r in sub], f"1G PSF [m] ({ax})")
                summarize([r["psf_2g_1_m"] for r in sub], f"2G comp1 PSF [m] ({ax})")
                summarize([r["psf_2g_2_m"] for r in sub], f"2G comp2 PSF [m] ({ax})")
            
            # save in a CSV
            df = pd.DataFrame(all_res)
            csv_out = outdir / "psf_summary_all_images.csv"
            df.to_csv(csv_out, index=False)
            print(f"\nSummary CSV saved to: {csv_out}")
    
    # check if df is in memory if not load it from the csv
    if 'df' in locals():

        # deletre the outlier PSF values greater than 100 m before plotting histograms
        df = df[df["psf_1g_m"] <= 100.0]
        df = df[df["psf_2g_1_m"] <= 100.0]
        df = df[df["psf_2g_2_m"] <= 1000.0]

        # creatre a plot with 3 subplots showing histograms of PSF 1G, PSF 2G comp1, PSF 2G comp2
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.hist(df["psf_1g_m"].dropna(), bins=50, color="blue", alpha=0.7)
        # put the value of the peak on the top of the histogram
        peak_1g = np.round(df["psf_1g_m"],1).mode().iloc[0]
        # plt.text(peak_1g, plt.ylim()[1]*0.9, f"Peak: {peak_1g:.1f} m", ha="left", color="black")
        # put a vertical line at the peak value
        plt.axvline(peak_1g, color="black", linestyle="--")
        plt.xlabel("PSF 1G [m]")
        plt.ylabel("Count")
        plt.title("PSF 1G = {:.1f} m".format(peak_1g))   
        plt.subplot(1, 3, 2)
        plt.hist(df["psf_2g_1_m"].dropna(), bins=50, color="green", alpha=0.7)
        # put the value of the peak on the top of the histogram
        peak_2g_1 = np.round(df["psf_2g_1_m"],0).mode().iloc[0]
        # plt.text(peak_2g_1, plt.ylim()[1]*0.9, f"Peak: {peak_2g_1:.1f} m", ha="left", color="black")
        # put a vertical line at the peak value
        plt.axvline(peak_2g_1, color="black", linestyle="--")
        plt.xlabel("PSF 2G comp1 [m]")
        plt.ylabel("Count")
        plt.title("PSF 2G comp1 = {:.1f} m".format(peak_2g_1))
        plt.subplot(1, 3, 3)
        plt.hist(df["psf_2g_2_m"].dropna(), bins=50, color="red", alpha=0.7)
        # put the value of the peak on the top of the histogram
        peak_2g_2 = np.round(df["psf_2g_2_m"],0).mode().iloc[0]
        # plt.text(peak_2g_2, plt.ylim()[1]*0.9, f"Peak: {peak_2g_2:.1f} m", ha="left", color="black")
        # put a vertical line at the peak value
        plt.axvline(peak_2g_2, color="black", linestyle="--")
        plt.xlabel("PSF 2G comp2 [m]")
        plt.ylabel("Count")
        plt.title("PSF 2G comp2 = {:.1f} m".format(peak_2g_2))
        plt.tight_layout()
        hist_out = outdir / "psf_histograms.png"
        plt.savefig(hist_out, dpi=200)
        plt.close()
        print(f"\nHistogram plot saved to: {hist_out}")

        # make the hinstogram of the A_1g, A_2g_1, A_2g_2 values in log scale along the x axis
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.hist(np.log10(df["A_1g"].dropna()), bins=50, color="blue", alpha=0.7)
        plt.xlabel("log$_{10}$(px Intensity)")
        plt.ylabel("Count")
        plt.title("A_1g")
        plt.subplot(1, 3, 2)
        plt.hist(np.log10(df["A_2g_1"].dropna()), bins=50, color="green", alpha=0.7)
        plt.xlabel("log$_{10}$(px Intensity)")
        plt.ylabel("Count")
        plt.title("A_2g comp1")
        plt.subplot(1, 3, 3)
        plt.hist(np.log10(df["A_2g_2"].dropna()), bins=50, color="red", alpha=0.7)
        plt.xlabel("log$_{10}$(px Intensity)")
        plt.ylabel("Count")
        plt.title("A_2g comp2")
        plt.tight_layout()
        hist_a_out = outdir / "a_values_histograms.png"
        plt.savefig(hist_a_out, dpi=200)
        plt.close()
        print(f"\nA values histogram plot saved to: {hist_a_out}")

        # create a scatter subplot of PSF 1G vs A_1g and PSF 2G vs A_2g for both components
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 3, 1)
        plt.scatter(df["A_1g"], df["psf_1g_m"], alpha=0.7, color="blue")
        plt.xlabel("px Intensity")
        # make it log scale
        plt.xscale("log")
        plt.ylabel("PSF 1G [m]")
        plt.title("PSF 1G vs px Intensity")
        plt.subplot(1, 3, 2)
        plt.scatter(df["A_2g_1"], df["psf_2g_1_m"], alpha=0.7, color="green")
        plt.xlabel("px Intensity")
        # make it log scale
        plt.xscale("log")
        plt.ylabel("PSF 2G comp1 [m]")
        plt.title("PSF 2G comp1 vs px Intensity")
        plt.subplot(1, 3, 3)
        plt.scatter(df["A_2g_2"], df["psf_2g_2_m"], alpha=0.7, color="red")
        plt.xlabel("px Intensity")
        # make it log scale
        plt.xscale("log")
        plt.ylabel("PSF 2G comp2 [m]")
        plt.title("PSF 2G comp2 vs px Intensity")
        plt.tight_layout()
        scatter_out = outdir / "psf_scatter_plots.png"
        plt.savefig(scatter_out, dpi=200)
        plt.close()
        print(f"\nScatter plot saved to: {scatter_out}")

        # plot the histogram of amplitude1_2G/(amplitude1_2G + amplitude2_2G)
        ratio_1g_2g = df["A_2g_1"] / (df["A_2g_1"] + df["A_2g_2"])
        plt.figure(figsize=(6, 4))
        plt.hist((ratio_1g_2g.dropna()), bins=50, color="purple", alpha=0.7)
        plt.xlabel("(A_2g_comp1 / (A_2g_comp1 + A_2g_comp2))")
        plt.ylabel("Count")
        peak_ratio = np.round(ratio_1g_2g,2).mode().iloc[0]
        plt.title("px Intensity Ratio Peak = {:.2f}".format(peak_ratio))
        ratio_out = outdir / "amplitude_ratio_histogram.png"
        plt.savefig(ratio_out, dpi=200)
        plt.close()
        print(f"\nAmplitude ratio histogram plot saved to: {ratio_out}")



    # create a plot 

    print("\nDone.")


if __name__ == "__main__":
    main()
