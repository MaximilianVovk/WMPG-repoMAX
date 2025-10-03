"""
Import all the pickle files and get the dynesty files distribution

Author: Maximilian Vovk
Date: 2025-04-16
"""

# main.py (inside my_subfolder)
import sys
import os

import numpy as np

# Add the parent directory to the sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from DynNestSapl_metsim import *

from itertools import combinations
from scipy.stats import ks_2samp, mannwhitneyu, anderson_ksamp
from scipy.stats import gaussian_kde
from dynesty import utils as dyfunc
from matplotlib.ticker import FormatStrFormatter
from matplotlib.gridspec import GridSpec
import itertools
from dynesty.utils import quantile as _quantile
from scipy.ndimage import gaussian_filter as norm_kde
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from dynesty import utils as dyfunc
from matplotlib.ticker import MaxNLocator, NullLocator, ScalarFormatter
from scipy.stats import gaussian_kde
from wmpl.Formats.WmplTrajectorySummary import loadTrajectorySummaryFast
from multiprocessing import Pool
from wmpl.MetSim.MetSimErosion import energyReceivedBeforeErosion

# try to resolve dynesty's internal _hist2d no matter how it's imported
try:
    from dynesty.plotting import _hist2d as _hist2d_func
except Exception:
    _hist2d_func = None

# avoid showing warnings
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np



# Create Results-like object for cornerplot
class CombinedResults:
    def __init__(self, samples, weights):
        self.samples = samples
        self.weights = weights

    def __getitem__(self, key):
        if key == 'samples':
            return self.samples
        raise KeyError(f"Key '{key}' not found.")

    def importance_weights(self):
        return self.weights


# --- your best-fit peak-capturing gamma-like model ---
def iron_percent(v_km_s):
    """% Iron candidates as a function of velocity (km/s)."""
    A, k, theta, m, v0 = 170.0, 2.7349189846, 0.3379023060, 0.5966146757, 8.4444529261
    z = np.maximum(np.asarray(v_km_s) - v0, 0.0)
    return A * (z**(k-1.0)) * np.exp(- (z/theta)**m)


def reweight_iron_by_velocity(results, variables, rho_threshold=4000.0):
    """
    Down-weight samples with rho > rho_threshold by the velocity-dependent
    iron fraction iron_percent(v)/100, where v is v_init in km/s.

    Parameters
    ----------
    results : CombinedResults
        Holds .samples (nsamps, ndim) and .importance_weights()
    variables : list[str]
        Names aligned with columns in results.samples
        (must contain 'v_init' [m/s] and 'rho' [kg/m^3])
    rho_threshold : float
        Density above which a sample is treated as iron-like

    Returns
    -------
    CombinedResults
        New object with adjusted weights
    """
    print("Reweighting samples for iron candidates based on Mills et al. 2021 iron distribution...")
    try:
        i_v = variables.index('v_init')
        i_rho = variables.index('rho')
    except ValueError as e:
        raise ValueError("variables must include 'v_init' and 'rho'") from e

    samples = results.samples
    w = results.importance_weights().astype(float).copy()

    v_km_s = samples[:, i_v] / 1e3  # v_init given in m/s
    rho = samples[:, i_rho]

    # scale factor in [0, ~peak%/100]; only applied to iron-like samples
    scale = iron_percent(v_km_s) / 100.0
    scale = np.clip(scale, 0.0, 1.0)

    mask_iron = rho > rho_threshold
    w[mask_iron] *= scale[mask_iron]

    # check how many got reweighted
    print(f"  {np.sum(mask_iron)} samples with rho > {rho_threshold} kg/m^3 reweighted by iron fraction.")

    # optional: small epsilon avoid all-zero weights if everything got down-weighted
    if not np.any(w > 0):
        raise RuntimeError("All weights became zero after reweighting. Check thresholds/fit.")

    # Return a new CombinedResults with adjusted weights
    return CombinedResults(samples, w)


# ---------- Weighted resampling helper ----------
def _weighted_resample(data, weights, n=None, rng=None):
    """Resample values ~ weights (with replacement). NaNs handled upstream."""
    rng = np.random.default_rng(rng)
    data = np.asarray(data, float)
    w = np.asarray(weights, float)
    s = np.nansum(w)
    if s <= 0 or data.size == 0:
        return np.array([], dtype=float)
    w = w / s
    if n is None:
        n = max(1, data.size)
    idx = rng.choice(data.size, size=n, replace=True, p=w)
    return data[idx]


def _effective_sample_size(weights):
    """ Kish's effective sample size for weights. """
    w = np.asarray(weights, float)
    w = w[np.isfinite(w) & (w > 0)]
    if w.size == 0:
        return 0.0
    s1 = np.sum(w)
    s2 = np.sum(w*w)
    return (s1*s1) / s2 if s2 > 0 else 0.0

# ---------- Main: run weighted tests for any number of groups ----------
def weighted_tests_table(
    values,
    weights,
    groups,
    resample_n=5000,
    random_seed=42,
    caption=r"Pairwise tests on $\rho$ distributions (weighted resampling).",
    label="tab:weighted_tests",
    floatfmt=3,
    save_path=None,):
    """
    Parameters
    ----------
    values : (N,) array-like
        Data values (e.g. rho_samp).
    weights : (N,) array-like
        Weights aligned with values (e.g. w_all).
    groups : dict[str, array[bool]] OR array-like of labels (len N)
        Either: mapping {group_name -> boolean mask} or a labels array of group names per sample.
    resample_n : int
        Size of each group's resample for the tests (per pair).
    random_seed : int
        Seed for reproducibility.
    caption : str
        LaTeX caption.
    label : str
        LaTeX label.
    floatfmt : int
        Decimal places for floating results.
    save_path : str or None
        If given, write the LaTeX string to this path.

    Returns
    -------
    tex_str : str
        The LaTeX table string.
    results : list of dict
        Raw results per pair.
    """
    rng_master = np.random.default_rng(random_seed)

    values = np.asarray(values, float)
    weights = np.asarray(weights, float)

    # Build group -> (clean values, clean weights)
    group_data = []

    if isinstance(groups, dict):
        for gname, mask in groups.items():
            m = np.asarray(mask, bool)
            m = m & np.isfinite(values) & np.isfinite(weights)
            v = values[m]
            w = weights[m]
            # normalize weights within group (not strictly required, but nice)
            s = np.nansum(w)
            if s > 0:
                w = w / s
            group_data.append((gname, v, w))
    else:
        # assume array-like labels
        labels = np.asarray(groups)
        finite = np.isfinite(values) & np.isfinite(weights) & (labels != None)
        labels = labels[finite].astype(str)
        v_all = values[finite]
        w_all = weights[finite]
        for gname in np.unique(labels):
            m = labels == gname
            v = v_all[m]
            w = w_all[m]
            s = np.nansum(w)
            if s > 0:
                w = w / s
            group_data.append((gname, v, w))

    # Filter out empty groups
    group_data = [(g,v,w) for (g,v,w) in group_data if v.size > 0 and np.nansum(w) > 0]

    # Prepare results
    rows = []
    for (gA, vA, wA), (gB, vB, wB) in combinations(group_data, 2):
        # effective sizes (informative)
        neffA = _effective_sample_size(wA)
        neffB = _effective_sample_size(wB)

        # resample
        # use different seeds per pair to avoid accidental identical draws
        seedA = rng_master.integers(0, 2**31 - 1)
        seedB = rng_master.integers(0, 2**31 - 1)
        rA = _weighted_resample(vA, wA, n=resample_n, rng=seedA)
        rB = _weighted_resample(vB, wB, n=resample_n, rng=seedB)

        # If either is empty (degenerate), skip
        if rA.size == 0 or rB.size == 0:
            rows.append(dict(
                A=gA, nA=float(neffA), B=gB, nB=float(neffB),
                ks_D=np.nan, ks_p=np.nan,
                mwu_U=np.nan, mwu_p=np.nan,
                ad_stat=np.nan, ad_sig=np.nan
            ))
            continue

        # KS
        ks_D, ks_p = ks_2samp(rA, rB, alternative="two-sided", mode="auto")

        # Mann-Whitney U (two-sided)
        # Note: MWU assumes continuous distributions; ties are fine but exact method may switch.
        mwu_U, mwu_p = mannwhitneyu(rA, rB, alternative="two-sided")

        # Anderson-Darling k-sample (2 groups is fine); returns approx significance level
        ad_stat, ad_crit, ad_sig = anderson_ksamp([rA, rB])

        rows.append(dict(
            A=gA, nA=float(neffA), B=gB, nB=float(neffB),
            ks_D=float(ks_D), ks_p=float(ks_p),
            mwu_U=float(mwu_U), mwu_p=float(mwu_p),
            ad_stat=float(ad_stat), ad_sig=float(ad_sig)
        ))

    # Build LaTeX table
    # Columns: A, n_eff(A), B, n_eff(B), KS D, KS p, MWU U, MWU p, AD stat, AD sig
    f = floatfmt
    def _fmt(x, sci=False):
        if x is None or not np.isfinite(x):
            return "--"
        if sci:
            return f"{x:.{f}e}"
        return f"{x:.{f}f}"
    
    # "Group A & $n_\\mathrm{eff}(A)$ & Group B & $n_\\mathrm{eff}(B)$ & "
    # f"{r['A']} & {_fmt(r['nA'])} & "
    # f"{r['B']} & {_fmt(r['nB'])} & "

    header = (
        "\\begin{table}[h!]\n"
        "\\centering\n"
        "\\small\n"
        "\\begin{adjustbox}{max width=\\textwidth}\n"
        "\\begin{tabular}{l l r r r r r r}\n"
        "\\hline\n"
        "Group A & Group B & "
        "KS $D$ & KS $p$ & MWU $U$ & MWU $p$ & AD stat & AD sig.\\\\\n"
        "\\hline\n"
    )

    lines = []
    for r in rows:
        line = (
            f"{r['A']} & "
            f"{r['B']} & "
            f"{_fmt(r['ks_D'])} & {_fmt(r['ks_p'], sci=True)} & "
            f"{_fmt(r['mwu_U'])} & {_fmt(r['mwu_p'], sci=True)} & "
            f"{_fmt(r['ad_stat'])} & {_fmt(r['ad_sig'], sci=True)} \\\\"
        )
        lines.append(line)

    footer = (
        "\\hline\n"
        "\\end{tabular}\n"
        "\\end{adjustbox}\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        "\\end{table}\n"
    )

    tex_str = header + "\n".join(lines) + "\n" + footer

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as fobj:
            fobj.write(tex_str)

    return tex_str, rows


def run_single_eeu(sim_num_and_data):
    sim_num, tot_sim, sample, obs_data, variables, fixed_values, flags_dict = sim_num_and_data

    # print(f"Running simulation {sim_num}/{tot_sim}")
    
    # Copy and transform the sample as in your loop
    guess = sample.copy()
    flag_total_rho = False

    for i, variable in enumerate(variables):
        if 'log' in flags_dict[variable]:
            guess[i] = 10**guess[i]
        if variable == 'noise_lag':
            obs_data.noise_lag = guess[i]
            obs_data.noise_vel = guess[i] * np.sqrt(2)/(1.0/32)
        if variable == 'noise_lum':
            obs_data.noise_lum = guess[i]
        if variable == 'erosion_rho_change':
            flag_total_rho = True

    # Build const_nominal (same as in your current loop)
    const_nominal = Constants()
    const_nominal.dens_co = obs_data.dens_co
    const_nominal.dt = obs_data.dt
    const_nominal.P_0m = obs_data.P_0m
    const_nominal.h_kill = obs_data.h_kill
    const_nominal.v_kill = obs_data.v_kill
    const_nominal.disruption_on = obs_data.disruption_on
    const_nominal.lum_eff_type = obs_data.lum_eff_type

    # Assign guessed and fixed parameters
    for i, var in enumerate(variables):
        const_nominal.__dict__[var] = guess[i]
    for k, v in fixed_values.items():
        const_nominal.__dict__[k] = v

    # Extract physical quantities
    try:
        eeucs, eeum = energyReceivedBeforeErosion(const_nominal)

        return (sim_num, eeucs, eeum)

    except Exception as e:
        print(f"Simulation {sim_num} failed: {e}")
        return (sim_num, np.nan, np.nan)
    

def shower_distrb_plot(input_dirfile, output_dir_show, shower_name, radiance_plot_flag=False, plot_correl_flag=False):
    """
    Function to plot the distribution of the parameters from the dynesty files and save them as a table in LaTeX format.
    """
    # Use the class to find .dynesty, load prior, and decide output folders
    finder = find_dynestyfile_and_priors(input_dir_or_file=input_dirfile,prior_file="",resume=True,output_dir=input_dirfile,use_all_cameras=False,pick_position=0)

    all_label_sets = []  # List to store sets of labels for each file
    variables = []  # List to store distributions for each file
    flags_dict_total = {}  # Dictionary to store flags for each file
    num_meteors = len(finder.base_names)  # Number of meteors
    for i, (base_name, dynesty_info, prior_path, out_folder) in enumerate(zip(finder.base_names,finder.input_folder_file,finder.priors,finder.output_folders)):
        dynesty_file, pickle_file, bounds, flags_dict, fixed_values = dynesty_info
        obs_data = finder.observation_instance(base_name)
        obs_data.file_name = pickle_file # update the file name in the observation data object

        # # Read the raw pickle bytes
        # with open(dynesty_file, "rb") as f:
        #     raw = f.read()

        # # Encode as Base64 so it’s pure text
        # b64 = base64.b64encode(raw).decode("ascii")

        # # create the json file name by just replacing the .dynesty with _dynesty.json
        # json_file = dynesty_file.replace(".dynesty", "_dynesty.json")
        # # Write that string into JSON
        # with open(json_file, "w") as f:
        #     json.dump({"dynesty_b64": b64}, f, indent=2)

        # save the lenght of the flags_dict to check if it is the same for all meteors
        # check if len(flags_dict.keys()) > len(variables) to avoid index error
        if len(flags_dict.keys()) > len(variables):
            variables = list(flags_dict.keys())
            flags_dict_total = flags_dict.copy()
            bounds_total = bounds.copy()


    # keep them in the same order distribution_list
    print(f"Shared labels: {variables}")

    ndim = len(variables)
    # Mapping of original variable names to LaTeX-style labels
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
        'noise_lum': r"$\sigma_{lum}$ [W]"
    }

    # Mapping of original variable names to LaTeX-style labels
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
        'noise_lum': r"$\sigma_{lum}$ [W]"
    }

    # check if there are variables in the flags_dict that are not in the variable_map
    for variable in variables:
        if variable not in variable_map:
            print(f"Warning: {variable} not found in variable_map")
            # Add the variable to the map with a default label
            variable_map[variable] = variable
    labels = [variable_map[variable] for variable in variables]

    for variable in variables:
        if variable not in variable_map_plot:
            print(f"Warning: {variable} not found in variable_map")
            # Add the variable to the map with a default label
            variable_map_plot[variable] = variable
    labels_plot = [variable_map_plot[variable] for variable in variables]



    def align_dynesty_samples(dsampler, all_variables, current_flags):
        """
        Aligns dsampler samples to the full list of all variables by padding missing variables with 0
        Weights remain unchanged for non-missing dimensions.
        """
        samples = dsampler.results['samples']
        weights = dsampler.results.importance_weights()
        n_samples = samples.shape[0]

        for i, variable in enumerate(current_flags):
            if 'log' in current_flags[variable]:
                samples[:, i] = 10**samples[:, i]

        # Create mapping of existing variables in current run
        flag_keys = list(current_flags.keys())
        flag_index = {v: i for i, v in enumerate(flag_keys)}

        # Prepare padded samples with NaNs for missing variables
        padded_samples = np.full((n_samples, len(all_variables)), np.nan)

        # # create a float array full of zeros (or use np.nan if you prefer)
        # padded_samples = np.zeros((n_samples, len(all_variables)), dtype=float)

        for j, var in enumerate(all_variables):
            if var in flag_index:
                padded_samples[:, j] = samples[:, flag_index[var]]

        return padded_samples, weights


    def extract_radiant_and_la_sun(report_path):
        """
        Returns:
        lg_mean, lg_err_lo,
        bg_mean, bg_err,
        la_sun_mean

        lg_err_lo/bg_err are the '+/-' values if present, else 0.0.
        """

        # storage
        lg = bg = la_sun = None
        lg_err_lo  = lg_err_hi = bg_err_lo  = bg_err_hi = None
        lg_helio = bg_helio = None
        lg_err_lo_helio = lg_err_hi_helio = bg_err_lo_helio = bg_err_hi_helio = None
        in_ecl = False
        in_ecl_helio = False

        # regexes for Lg/Bg with CI
        re_lg_ci = re.compile(r'^\s*Lg\s*=\s*([+-]?\d+\.\d+)[^[]*\[\s*([+-]?\d+\.\d+)\s*,\s*([+-]?\d+\.\d+)\s*\]')
        re_bg_ci = re.compile(r'^\s*Bg\s*=\s*([+-]?\d+\.\d+)[^[]*\[\s*([+-]?\d+\.\d+)\s*,\s*([+-]?\d+\.\d+)\s*\]')
        # look for "Lg = 246.70202 +/- 0.46473"
        re_lg_pm  = re.compile(r'^\s*Lg\s*=\s*([+-]?\d+\.\d+)\s*\+/-\s*([0-9.]+)')
        re_bg_pm  = re.compile(r'^\s*Bg\s*=\s*([+-]?\d+\.\d+)\s*\+/-\s*([0-9.]+)')
        # fallback plain values
        re_lg_val = re.compile(r'^\s*Lg\s*=\s*([+-]?\d+\.\d+)')
        re_bg_val = re.compile(r'^\s*Bg\s*=\s*([+-]?\d+\.\d+)')
        # solar longitude
        re_lasun  = re.compile(r'^\s*La Sun\s*=\s*([+-]?\d+\.\d+)')

        # regexes for Lh/Bh with CI
        re_lg_ci_h = re.compile(r'^\s*Lh\s*=\s*([+-]?\d+\.\d+)[^[]*\[\s*([+-]?\d+\.\d+)\s*,\s*([+-]?\d+\.\d+)\s*\]')
        re_bg_ci_h = re.compile(r'^\s*Bh\s*=\s*([+-]?\d+\.\d+)[^[]*\[\s*([+-]?\d+\.\d+)\s*,\s*([+-]?\d+\.\d+)\s*\]')
        # look for "Lh = 246.70202 +/- 0.46473"
        re_lg_pm_h  = re.compile(r'^\s*Lh\s*=\s*([+-]?\d+\.\d+)\s*\+/-\s*([0-9.]+)')
        re_bg_pm_h  = re.compile(r'^\s*Bh\s*=\s*([+-]?\d+\.\d+)\s*\+/-\s*([0-9.]+)')
        # fallback plain values
        re_lg_val_h = re.compile(r'^\s*Lh\s*=\s*([+-]?\d+\.\d+)')
        re_bg_val_h = re.compile(r'^\s*Bh\s*=\s*([+-]?\d+\.\d+)')

        with open(report_path, 'r', encoding='utf-8') as f:
            for line in f:
                s = line.strip()

                # enter the block
                if s.startswith('Radiant (ecliptic geocentric'):
                    in_ecl = True
                    continue

                if in_ecl:
                    # blank line → exit
                    if not s:
                        in_ecl = False
                    else:
                        # try 95CI first
                        m = re_lg_ci.match(line)
                        if m:
                            lg, lg_err_lo, lg_err_hi = map(float, m.groups())
                        else:
                            # try ± after
                            m = re_lg_pm.match(line)
                            if m:
                                lg, lg_err_lo = float(m.group(1)), float(m.group(2))
                            else:
                                # then plain
                                m = re_lg_val.match(line)
                                if m and lg is None:
                                    lg = float(m.group(1))
                        # try 95CI first
                        m = re_bg_ci.match(line)
                        if m:
                            bg, bg_err_lo, bg_err_hi = map(float, m.groups())
                        else:
                            m = re_bg_pm.match(line)
                            if m:
                                bg, bg_err_lo = float(m.group(1)), float(m.group(2))
                            else:
                                m = re_bg_val.match(line)
                                if m and bg is None:
                                    bg = float(m.group(1))

                if s.startswith('Radiant (ecliptic heliocentric'):
                    in_ecl_helio = True
                    continue

                if in_ecl_helio:
                    # blank line → exit
                    if not s:
                        in_ecl_helio = False
                    else:
                        # try 95CI first
                        m = re_lg_ci_h.match(line)
                        if m:
                            lg_helio, lg_err_lo_helio, lg_err_hi_helio = map(float, m.groups())
                        else:
                            # try ± after
                            m = re_lg_pm_h.match(line)
                            if m:
                                lg_helio, lg_err_lo_helio = float(m.group(1)), float(m.group(2))
                            else:
                                # then plain
                                m = re_lg_val_h.match(line)
                                if m and lg_helio is None:
                                    lg_helio = float(m.group(1))
                        # try 95CI first
                        m = re_bg_ci_h.match(line)
                        if m:
                            bg_helio, bg_err_lo_helio, bg_err_hi_helio = map(float, m.groups())
                        else:
                            m = re_bg_pm_h.match(line)
                            if m:
                                bg_helio, bg_err_lo_helio = float(m.group(1)), float(m.group(2))
                            else:
                                m = re_bg_val_h.match(line)
                                if m and bg_helio is None:
                                    bg_helio = float(m.group(1))


                # always grab La Sun
                if la_sun is None:
                    m = re_lasun.match(line)
                    if m:
                        la_sun = float(m.group(1))

                # stop if we have all three
                if lg is not None and bg is not None and la_sun is not None and lg_helio is not None and bg_helio is not None:
                    break

        if lg is None or bg is None:
            raise RuntimeError(f"Couldn t find Lg/Bg in {report_path!r}")
        if la_sun is None:
            raise RuntimeError(f"Couldn t find La Sun in {report_path!r}")
        if lg_err_lo is None:
            lg_err_lo = lg
            lg_err_hi = lg
        if bg_err_lo is None:
            bg_err_lo = bg
            bg_err_hi = bg
        if lg_err_hi is None:
            lg_err_hi = lg + abs(lg_err_lo)
            lg_err_lo = lg - abs(lg_err_lo)
        if bg_err_hi is None:
            bg_err_hi = bg + abs(bg_err_lo)
            bg_err_lo = bg - abs(bg_err_lo)
        if lg_err_lo_helio is None:
            lg_err_lo_helio = lg_helio
            lg_err_hi_helio = lg_helio
        if bg_err_lo_helio is None:
            bg_err_lo_helio = bg_helio
            bg_err_hi_helio = bg_helio
        if lg_err_hi_helio is None:
            lg_err_hi_helio = lg_helio + abs(lg_err_lo_helio)
            lg_err_lo_helio = lg_helio - abs(lg_err_lo_helio)
        if bg_err_hi_helio is None:
            bg_err_hi_helio = bg_helio + abs(bg_err_lo_helio)
            bg_err_lo_helio = bg_helio - abs(bg_err_lo_helio)

        print(f"Radiant (ecliptic heliocentric): Lg = {lg_helio}° 95CI [{lg_err_lo_helio:.3f}°, {lg_err_hi_helio:.3f}°], Bg = {bg_helio}° 95CI [{bg_err_lo_helio:.3f}°, {bg_err_hi_helio:.3f}°]")
        # print the results
        print(f"Radiant (ecliptic geocentric): Lg = {lg}° 95CI [{lg_err_lo:.3f}°, {lg_err_hi:.3f}°], Bg = {bg}° 95CI [{bg_err_lo:.3f}°, {bg_err_hi:.3f}°]")
        lg_lo = (lg - lg_err_lo)/1.96
        lg_hi = (lg_err_hi - lg)/1.96
        bg_lo = (bg_err_hi - bg)/1.96
        bg_hi = (bg - bg_err_lo)/1.96
        lg_helio_lo = (lg_helio - lg_err_lo_helio)/1.96
        lg_helio_hi = (lg_err_hi_helio - lg_helio)/1.96
        bg_helio_lo = (bg_err_hi_helio - bg_helio)/1.96
        bg_helio_hi = (bg_helio - bg_err_lo_helio)/1.96
        print(f"Error range: Lg = {lg}° ± {lg_lo:.3f}° / {lg_hi:.3f}°, Bg = {bg}° ± {bg_lo:.3f}° / {bg_hi:.3f}°")

        return lg, lg_lo, lg_hi, bg, bg_lo, bg_hi, la_sun, lg_helio, lg_helio_lo, lg_helio_hi, bg_helio, bg_helio_lo, bg_helio_hi


    def extract_tj_from_report(report_path):
        Tj = Tj_low = Tj_high = None
        inclin_val = Q_val = q_val = a_val = e_val = Vinf_val = None
        
        re_i_val = re.compile(
            r'^\s*i\s*=\s*'                           
            r'([+-]?\d+\.\d+)'                         
        )

        re_Vinf_val = re.compile(
            r'^\s*Vinf\s*=\s*'                           
            r'([+-]?\d+\.\d+)'                         
        )

        re_a_val = re.compile(
            r'^\s*a\s*=\s*'                           
            r'([+-]?\d+\.\d+)'                         
        )

        re_e_val = re.compile(
            r'^\s*e\s*=\s*'                           
            r'([+-]?\d+\.\d+)'                         
        )

        re_q_val = re.compile(
            r'^\s*q\s*=\s*'                           
            r'([+-]?\d+\.\d+)'                         
        )

        re_Q_val = re.compile(
            r'^\s*Q\s*=\s*'                           
            r'([+-]?\d+\.\d+)'                         
        )

        re_Vinf_val = re.compile(
            r'^\s*Vinf\s*=\s*'                           
            r'([+-]?\d+\.\d+)'                         
        )

        # 1) “CI present” (captures best‐fit, ci_low, ci_high)
        re_tj_ci = re.compile(
            r'^\s*Tj\s*=\s*'                           # “Tj = ”
            r'([+-]?\d+\.\d+)'                         # 1) best‐fit value
            r'(?:\s*\+/-\s*[0-9.]+)?'                  #    optional “± err” (we don’t need to capture it here)
            r'\s*,\s*95%\s*CI\s*\[\s*'
            r'([+-]?\d+\.\d+)\s*,\s*'                  # 2) ci_low
            r'([+-]?\d+\.\d+)\s*\]'                    # 3) ci_high
        )

        # 2) “± err” only (no CI)
        re_tj_pm = re.compile(
            r'^\s*Tj\s*=\s*'                           # “Tj = ”
            r'([+-]?\d+\.\d+)\s*'                      # 1) best‐fit value
            r'\+/-\s*([0-9.]+)'                        # 2) err
            r'\s*$'
        )

        # 3) Plain value only
        re_tj_val = re.compile(
            r'^\s*Tj\s*=\s*'                           # “Tj = ”
            r'([+-]?\d+\.\d+)'                         # 1) best‐fit value
            r'\s*$'
        )

        with open(report_path, 'r', encoding='utf-8') as f:
            for line in f:
                # Try “CI present” first
                m = re_tj_ci.match(line)
                if m:
                    Tj, Tj_low, Tj_high = map(float, m.groups())
                else:
                    # Next, try “± err” only
                    m = re_tj_pm.match(line)
                    if m:
                        val = float(m.group(1))
                        err = float(m.group(2))
                        Tj = val
                        Tj_low = val - err
                        Tj_high = val + err
                    else:
                        # Finally, try bare “Tj = <val>”
                        m = re_tj_val.match(line)
                        if m:
                            val = float(m.group(1))
                            Tj = val
                            Tj_low = val
                            Tj_high = val

                if inclin_val is None:
                    m = re_i_val.match(line)
                    if m:
                        inclin_val = float(m.group(1))
                if Vinf_val is None:
                    m = re_Vinf_val.match(line)
                    if m:
                        Vinf_val = float(m.group(1))
                if Q_val is None:
                    m = re_Q_val.match(line)
                    if m:
                        Q_val = float(m.group(1))
                if q_val is None:
                    m = re_q_val.match(line)
                    if m:
                        q_val = float(m.group(1))
                if a_val is None:
                    m = re_a_val.match(line)
                    if m:
                        a_val = float(m.group(1))
                if e_val is None:
                    m = re_e_val.match(line)
                    if m:
                        e_val = float(m.group(1))
                        

                if Tj is not None and inclin_val is not None and Vinf_val is not None and Q_val is not None and q_val is not None and a_val is not None and e_val is not None:
                    break


        if Tj is None:
            raise RuntimeError(f"Couldn’t find any Tj line in {report_path!r}")
        if inclin_val is None:
            raise RuntimeError(f"Couldn’t find inclination (i) in {report_path!r}")
        if Vinf_val is None:
            raise RuntimeError(f"Couldn’t find Vinf in {report_path!r}")

        print(f"Tj = {Tj:.6f} 95% CI = [{Tj_low:.6f}, {Tj_high:.6f}]")
        Tj_low = (Tj - Tj_low)#/1.96
        Tj_high = (Tj_high - Tj)#/1.96
        print(f"Vinf = {Vinf_val:.6f} km/s")
        print(f"a = {a_val:.6f} AU")
        print(f"e = {e_val:.6f}")
        print(f"i = {inclin_val:.6f} deg")
        print(f"Q = {Q_val:.6f} AU")
        print(f"q = {q_val:.6f} AU")

        return Tj, Tj_low, Tj_high, inclin_val, Vinf_val, Q_val, q_val, a_val, e_val


    def summarize_from_cornerplot(results, variables, labels_plot, smooth=0.02):
        """
        Summarize dynesty results, using the sample of max weight as the mode.
        """
        samples = results.samples               # shape (nsamps, ndim)
        weights = results.importance_weights()  # shape (nsamps,)

        # normalize weights
        w = weights.copy()
        w /= np.sum(w)

        # find the single sample index with highest weight
        mode_idx = np.nanargmax(w)   # index of peak-weight sample
        mode_raw = samples[mode_idx] # array shape (ndim,)

        rows = []
        for i, (var, lab) in enumerate(zip(variables, labels_plot)):
            x = samples[:, i].astype(float)
            # mask out NaNs
            mask = ~np.isnan(x)
            x_valid = x[mask]
            w_valid = w[mask]
            if x_valid.size == 0:
                rows.append((var, lab, *([np.nan]*5)))
                continue
            # renormalize
            w_valid /= np.sum(w_valid)

            # weighted quantiles
            low95, med, high95 = _quantile(x_valid,
                                        [0.025, 0.5, 0.975],
                                        weights=w_valid)
            # weighted mean
            mean_raw = np.sum(x_valid * w_valid)
            # simple mode from max-weight sample
            mode_value = mode_raw[i]

            # mode via corner logic
            lo, hi = np.min(x), np.max(x)
            if isinstance(smooth, int):
                hist, edges = np.histogram(x, bins=smooth, weights=w, range=(lo,hi))
            else:
                nbins = int(round(10. / smooth))
                hist, edges = np.histogram(x, bins=nbins, weights=w, range=(lo,hi))
                hist = norm_kde(hist, 10.0)
            centers = 0.5 * (edges[1:] + edges[:-1])
            mode_Ndim = centers[np.argmax(hist)]

            # now apply your log & unit transforms *after* computing stats
            def transform(v):
                if var in ['v_init',
                        'erosion_height_start',
                        'erosion_height_change']:
                    v = v / 1e3
                if var in ['erosion_coeff',
                        'sigma',
                        'erosion_coeff_change',
                        'erosion_sigma_change']:
                    v = v * 1e6
                return v

            rows.append((
                var,
                lab,
                transform(low95),
                transform(mode_value),
                transform(mode_Ndim),
                transform(mean_raw),
                transform(med),
                transform(high95),
            ))

        return pd.DataFrame(
            rows,
            columns=["Variable","Label","Low95","Mode","Mode_{Ndim}","Mean","Median","High95"]
        )

    def weighted_var_eros_height_change(var_start, var_heightchange, mass_before, m_init, w):
        x = var_start*(abs(m_init-mass_before) / m_init) + var_heightchange * (mass_before / m_init)
        mask = ~np.isnan(x)
        x_valid = x[mask]
        w_valid = w[mask]

        # renormalize
        w_valid /= np.sum(w_valid)

        # weighted quantiles
        rho_lo, rho, rho_hi = _quantile(x_valid, [0.025, 0.5, 0.975], weights=w_valid)
        rho_lo = (rho - rho_lo) #/1.96
        rho_hi = (rho_hi - rho) #/1.96
        return x_valid, rho, rho_lo, rho_hi

    # the on that are not variables are the one that were not used in the dynesty run give a np.nan weight to dsampler for those
    all_samples = []
    all_weights = []
    all_names = []  

    # base_name, lg_min_la_sun, bg, rho
    file_radiance_rho_dict = {}
    file_radiance_rho_dict_helio = {}
    file_obs_data_dict = {}
    file_phys_data_dict = {}
    file_eeu_dict = {}
    file_rho_jd_dict = {}
    find_worst_lag = {}
    find_worst_lum = {}
    # corrected rho
    rho_corrected = []
    eta_corrected = []
    sigma_corrected = []
    tau_corrected = []

    for i, (base_name, dynesty_info, prior_path, out_folder) in enumerate(zip(finder.base_names, finder.input_folder_file, finder.priors, finder.output_folders)):
        dynesty_file, pickle_file, bounds, flags_dict, fixed_values = dynesty_info
        print('\n', base_name)
        print(f"Processed {i+1} out of {len(finder.base_names)}")
        obs_data = finder.observation_instance(base_name)
        obs_data.file_name = pickle_file  # update the file name in the observation data object
        dsampler = dynesty.DynamicNestedSampler.restore(dynesty_file)
        # dsampler = load_dynesty_file(dynesty_file)

        # Align to the union of all variables (padding missing ones with NaN and 0 weights)
        samples_aligned, weights_aligned = align_dynesty_samples(dsampler, variables, flags_dict)

        print("number of simulations with posteriory above zero :",len(np.where(weights_aligned != 0))," out of ",len(weights_aligned))
        
        output_dir = os.path.dirname(dynesty_file)
        report_file = None
        for name in os.listdir(output_dir):
            if name.endswith("report.txt"):
                report_file = name; break
        if report_file is None:
            for name in os.listdir(output_dir):
                if name.endswith("report_sim.txt"):
                    report_file = name; break
        if report_file is None:
            raise FileNotFoundError("No report.txt or report_sim.txt found")

        report_path = os.path.join(output_dir, report_file)
        print(f"Using report file: {report_path}")
        lg, lg_lo, lg_hi, bg, bg_lo, bg_hi, la_sun, lg_helio, lg_helio_lo, lg_helio_hi, bg_helio, bg_helio_lo, bg_helio_hi = extract_radiant_and_la_sun(report_path)
        print(f"Ecliptic geocentric (J2000): Lg = {lg}°, Bg = {bg}°")
        print(f"Solar longitude:       La Sun = {la_sun}°")
        lg_min_la_sun = (lg - la_sun)%360
        lg_min_la_sun_helio = (lg_helio - la_sun)%360

        combined_results_meteor = CombinedResults(samples_aligned, weights_aligned)

        summary_df_meteor = summarize_from_cornerplot(
        combined_results_meteor,
        variables,
        labels
        )


        dynesty_run_results = dsampler.results
        weights = dynesty_run_results.importance_weights()
        w = weights / np.sum(weights)
        samples = dynesty_run_results.samples
        ndim = len(variables)
        sim_num = np.argmax(dynesty_run_results.logl)

        # copy the best guess values
        guess = dynesty_run_results.samples[sim_num].copy()
        flag_total_rho = False
        # load the variable names
        variables_sing = list(flags_dict.keys())
        for i, variable in enumerate(variables_sing):
            if 'log' in flags_dict[variable]:
                guess[i] = 10**guess[i]
                samples[:, i] = 10**samples[:, i]
            if variable == 'noise_lag':
                obs_data.noise_lag = guess[i]
                obs_data.noise_vel = guess[i] * np.sqrt(2)/(1.0/32)
            if variable == 'noise_lum':
                obs_data.noise_lum = guess[i]
            if variable == 'erosion_rho_change':
                flag_total_rho = True

        beg_height = obs_data.height_lum[0]
        end_height = obs_data.height_lum[-1]

        # vel_init = obs_data.v_init
        lenght_par = obs_data.length[-1]/1000 # convert to km
        max_lum_height = obs_data.height_lum[np.argmax(obs_data.luminosity)]
        F_par = (beg_height - max_lum_height) / (beg_height - end_height)
        kc_par = beg_height/1000 + (2.86 - 2*np.log10(summary_df_meteor['Median'].values[variables_sing.index('v_init')]))/0.0612
        time_tot = obs_data.time_lum[-1] - obs_data.time_lum[0]
        avg_vel = np.mean(obs_data.velocities)
        init_mag = obs_data.absolute_magnitudes[0]
        end_mag = obs_data.absolute_magnitudes[-1]
        max_mag = obs_data.absolute_magnitudes[np.argmax(obs_data.luminosity)]
        zenith_angle = np.rad2deg(obs_data.zenith_angle)



        # set up the observation data object
        obs_data = finder.observation_instance(base_name)
        obs_data.file_name = pickle_file # update teh file name in the observation data object

        # if the real_event has an initial velocity lower than 30000 set "dt": 0.005 to "dt": 0.01
        if obs_data.v_init < 30000:
            obs_data.dt = 0.01
            # const_nominal.erosion_bins_per_10mass = 5
        else:
            obs_data.dt = 0.005
            # const_nominal.erosion_bins_per_10mass = 10

        obs_data.disruption_on = False

        obs_data.lum_eff_type = 5

        obs_data.h_kill = np.min([obs_data.height_lum[-1],obs_data.height_lag[-1]])-1000
        # check if the h_kill is smaller than 0
        if obs_data.h_kill < 0:
            obs_data.h_kill = 1
        # check if np.min(obs_data.velocity[-1]) is smaller than v_init-10000
        if np.min(obs_data.velocities) < obs_data.v_init-10000:
            obs_data.v_kill = obs_data.v_init-10000
        else:
            obs_data.v_kill = np.min(obs_data.velocities)-5000
        # check if the v_kill is smaller than 0
        if obs_data.v_kill < 0:
            obs_data.v_kill = 1

        best_guess_obj_plot = run_simulation(guess, obs_data, variables_sing, fixed_values)

        heights = np.array(best_guess_obj_plot.leading_frag_height_arr, dtype=np.float64)[:-1]
        mass_best = np.array(best_guess_obj_plot.mass_total_active_arr, dtype=np.float64)[:-1]
        erosion_beg_dyn_press = best_guess_obj_plot.const.erosion_beg_dyn_press
        print(f"Dynamic pressure at erosion onset: {erosion_beg_dyn_press} Pa")

        if flag_total_rho:
            
            # find erosion change height
            if 'erosion_height_change' in variables_sing:
                erosion_height_change = guess[variables_sing.index('erosion_height_change')]
            if 'm_init' in variables_sing:
                m_init = guess[variables_sing.index('m_init')]

            old_mass_before = mass_best[np.argmin(np.abs(heights - erosion_height_change))]
            mass_before = best_guess_obj_plot.const.mass_at_erosion_change
            print(f"Mass before erosion change: {mass_before} kg, old mass before: {old_mass_before} kg")
            # check if mass_before is None if so use old_mass_before
            if mass_before is None:
                print("Using old mass before erosion change, the new one is none!")
                mass_before = old_mass_before

            x_valid_rho, rho, rho_lo, rho_hi = weighted_var_eros_height_change(samples[:, variables_sing.index('rho')].astype(float), samples[:, variables_sing.index('erosion_rho_change')].astype(float), mass_before, m_init, w)

            rho_corrected.append(x_valid_rho)

            x_valid_eta, eta, eta_lo, eta_hi = weighted_var_eros_height_change(samples[:, variables_sing.index('erosion_coeff')].astype(float), samples[:, variables_sing.index('erosion_coeff_change')].astype(float), mass_before, m_init, w)

            eta_corrected.append(x_valid_eta)

            # erosion_sigma_change
            x_valid_sigma, sigma, sigma_lo, sigma_hi = weighted_var_eros_height_change(samples[:, variables_sing.index('sigma')].astype(float), samples[:, variables_sing.index('erosion_sigma_change')].astype(float), mass_before, m_init, w)

            sigma_corrected.append(x_valid_sigma)

            # x = samples[:, variables_sing.index('rho')].astype(float)*(abs(m_init-mass_before) / m_init) + samples[:, variables_sing.index('erosion_rho_change')].astype(float) * (mass_before / m_init)
            # mask = ~np.isnan(x)
            # x_valid = x[mask]
            # w_valid = w[mask]

            # # renormalize
            # w_valid /= np.sum(w_valid)

            # # weighted quantiles
            # rho_lo, rho, rho_hi = _quantile(x_valid, [0.025, 0.5, 0.975], weights=w_valid)
            # rho_lo = (rho - rho_lo) #/1.96
            # rho_hi = (rho_hi - rho) #/1.96
            # rho_total = x_valid
            # rho_total 

        else:
            rho_lo = summary_df_meteor['Median'].values[variables.index('rho')] - summary_df_meteor['Low95'].values[variables.index('rho')]
            rho_hi = summary_df_meteor['High95'].values[variables.index('rho')] - summary_df_meteor['Median'].values[variables.index('rho')]
            rho = summary_df_meteor['Median'].values[variables.index('rho')]

            x = samples[:, variables_sing.index('rho')].astype(float)
            mask = ~np.isnan(x)
            x_valid_rho = x[mask] 

            rho_corrected.append(x_valid_rho)

            eta_lo = summary_df_meteor['Median'].values[variables.index('erosion_coeff')] - summary_df_meteor['Low95'].values[variables.index('erosion_coeff')]
            eta_hi = summary_df_meteor['High95'].values[variables.index('erosion_coeff')] - summary_df_meteor['Median'].values[variables.index('erosion_coeff')]
            eta = summary_df_meteor['Median'].values[variables.index('erosion_coeff')]

            x = samples[:, variables_sing.index('erosion_coeff')].astype(float)
            mask = ~np.isnan(x)
            x_valid_eta = x[mask]

            eta_corrected.append(x_valid_eta)

            sigma_lo = summary_df_meteor['Median'].values[variables.index('sigma')] - summary_df_meteor['Low95'].values[variables.index('sigma')]
            sigma_hi = summary_df_meteor['High95'].values[variables.index('sigma')] - summary_df_meteor['Median'].values[variables.index('sigma')]
            sigma = summary_df_meteor['Median'].values[variables.index('sigma')]

            x = samples[:, variables_sing.index('sigma')].astype(float)
            mask = ~np.isnan(x)
            x_valid_sigma = x[mask]

            sigma_corrected.append(x_valid_sigma)

        # rho_corrected.append(x_valid)
        # sigma_corrected.append(np.std(x_valid))
        # eta_corrected.append

        # find the index of m_init in variables
        tau = (calcRadiatedEnergy(np.array(obs_data.time_lum), np.array(obs_data.absolute_magnitudes), P_0m=obs_data.P_0m))*2/(samples[:, variables_sing.index('m_init')].astype(float)*obs_data.velocities[0]**2) * 100
        
        # calculate the weights calculate the weighted median and the 95 CI for tau
        tau_low95, tau_median, tau_high95 = _quantile(tau, [0.025, 0.5, 0.975],  weights=w)
        tau_corrected.append(tau)
    
        m_init_meteor_median = summary_df_meteor['Median'].values[variables.index('m_init')]
        m_init_meteor_lo = summary_df_meteor['Median'].values[variables.index('m_init')] - summary_df_meteor['Low95'].values[variables.index('m_init')]
        m_init_meteor_hi = summary_df_meteor['High95'].values[variables.index('m_init')] - summary_df_meteor['Median'].values[variables.index('m_init')]

        eta_meteor_begin = summary_df_meteor['Median'].values[variables.index('erosion_coeff')]
        sigma_meteor_begin = summary_df_meteor['Median'].values[variables.index('sigma')]
        v_init_meteor_median = summary_df_meteor['Median'].values[variables.index('v_init')]

        kinetic_energy_median = 1/2 * m_init_meteor_median * (v_init_meteor_median*1000)**2
        kinetic_energy_lo = kinetic_energy_median - 1/2 * summary_df_meteor['Low95'].values[variables.index('m_init')] * (summary_df_meteor['Low95'].values[variables.index('v_init')]*1000)**2
        kinetic_energy_hi = 1/2 * summary_df_meteor['High95'].values[variables.index('m_init')] * (summary_df_meteor['High95'].values[variables.index('v_init')]*1000)**2 - kinetic_energy_median

        # compute the meteoroid_diameter from a spherical shape in mm
        all_diameter_mm = (6 * samples[:, variables_sing.index('m_init')].astype(float) / (np.pi * x_valid_rho))**(1/3) * 1000
        # make the quntile base on w 
        meteoroid_diameter_mm_lo, meteoroid_diameter_mm, meteoroid_diameter_mm_hi = _quantile(all_diameter_mm, [0.025, 0.5, 0.975], weights=w)
        meteoroid_diameter_mm_lo = (meteoroid_diameter_mm - meteoroid_diameter_mm_lo) #/1.96
        meteoroid_diameter_mm_hi = (meteoroid_diameter_mm_hi - meteoroid_diameter_mm) #/1.96

        # meteoroid_diameter_mm_old = (6 * m_init_meteor_median / (np.pi * rho))**(1/3) * 1000

        print(f"rho: {rho} kg/m^3, 95% CI = [{rho_lo:.6f}, {rho_hi:.6f}]")
        print(f"intial mass {m_init_meteor_median} kg and diameter {meteoroid_diameter_mm:.6f} mm")#, old diameter {meteoroid_diameter_mm_old:.6f} mm")
        # print(f"erosion coeff: {eta} m/s, 95% CI = [{eta_lo}, {eta_hi}]")
        # print(f"sigma: {sigma} kg/m^3, 95% CI = [{sigma_lo}, {sigma_hi}]")

        # ### EROSION ENERGY CALCULATION ###

        # print("Calculating erosion energy per unit cross section and mass...")
        # # Package inputs
        # inputs = [
        #     (i, len(dynesty_run_results.samples), dynesty_run_results.samples[i], obs_data, variables_sing, fixed_values, flags_dict)
        #     for i in range(len(dynesty_run_results.samples)) # for i in np.linspace(0, len(dynesty_run_results.samples)-1, 10, dtype=int)
        # ]
        # #     for i in range(len(dynesty_run_results.samples)) # 
        # num_cores = multiprocessing.cpu_count()
        # print(f"Using {base_name}.")
        # # Run in parallel
        # with Pool(processes=num_cores) as pool:  # adjust to number of cores
        #     results = pool.map(run_single_eeu, inputs)

        # N = len(dynesty_run_results.samples)

        # erosion_energy_per_unit_cross_section_arr = np.full(N, np.nan)
        # erosion_energy_per_unit_mass_arr = np.full(N, np.nan)

        # for res in results:
        #     i, eeucs, eeum = res
        #     erosion_energy_per_unit_cross_section_arr[i] = eeucs / 1000000 # convert to MJ/m^2
        #     erosion_energy_per_unit_mass_arr[i] = eeum / 1000000 # convert to kg/MJ
        
        # weights = dynesty_run_results.importance_weights()
        # w = weights / np.sum(weights)
    
        # for i, x in enumerate([erosion_energy_per_unit_cross_section_arr, erosion_energy_per_unit_mass_arr]):
        #     # mask out NaNs
        #     mask = ~np.isnan(x)
        #     if not np.any(mask):
        #         print("Warning: All values are NaN, skipping quantile calculation.")
        #         continue
        #     mask = ~np.isnan(x)
        #     x_valid = x[mask]
        #     w_valid = w[mask]
        #     # renormalize
        #     w_valid /= np.sum(w_valid)
        #     if i == 0:
        #         # weighted quantiles
        #         eeucs_lo, eeucs, eeucs_hi = _quantile(x_valid, [0.025, 0.5, 0.975], weights=w_valid)
        #         print(f"erosion energy per unit cross section: {eeucs} J/m^2, 95% CI = [{eeucs_lo:.6f}, {eeucs_hi:.6f}]")
        #         eeucs_lo = (eeucs - eeucs_lo)
        #         eeucs_hi = (eeucs_hi - eeucs)
        #     elif i == 1:
        #         # weighted quantiles
        #         eeum_lo, eeum, eeum_hi = _quantile(x_valid, [0.025, 0.5, 0.975], weights=w_valid)
        #         print(f"erosion energy per unit mass: {eeum} J/kg, 95% CI = [{eeum_lo:.6f}, {eeum_hi:.6f}]")
        #         eeum_lo = (eeum - eeum_lo)
        #         eeum_hi = (eeum_hi - eeum)

        ### SAVE DATA ###

        # delete from base_name _combined if it exists
        if '_combined' in base_name:
            base_name = base_name.replace('_combined', '')

        file_radiance_rho_dict[base_name] = (lg_min_la_sun, bg, rho, lg_lo, lg_hi, bg_lo, bg_hi)
        file_radiance_rho_dict_helio[base_name] = (lg_min_la_sun_helio, lg_helio_lo, lg_helio_hi, bg_helio, bg_helio_lo, bg_helio_hi)

        tj, tj_lo, tj_hi, inclin_val, Vinf_val, Q_val, q_val, a_val, e_val = extract_tj_from_report(report_path)

        file_rho_jd_dict[base_name] = (rho, rho_lo,rho_hi, tj, tj_lo, tj_hi, inclin_val, Vinf_val, Q_val, q_val, a_val, e_val)
        # file_eeu_dict[base_name] = (eeucs, eeucs_lo, eeucs_hi, eeum, eeum_lo, eeum_hi,F_par, kc_par, lenght_par)
        file_obs_data_dict[base_name] = (kc_par, F_par, lenght_par, beg_height/1000, end_height/1000, max_lum_height/1000, avg_vel/1000, init_mag, end_mag, max_mag, time_tot, zenith_angle, m_init_meteor_median, meteoroid_diameter_mm, erosion_beg_dyn_press, v_init_meteor_median, kinetic_energy_median, kinetic_energy_lo, kinetic_energy_hi, tau_median, tau_low95, tau_high95)
        file_phys_data_dict[base_name] = (eta_meteor_begin, eta, eta_lo, eta_hi, sigma_meteor_begin, sigma, sigma_lo, sigma_hi, meteoroid_diameter_mm, meteoroid_diameter_mm_lo, meteoroid_diameter_mm_hi, m_init_meteor_median, m_init_meteor_lo, m_init_meteor_hi)

        find_worst_lag[base_name] = summary_df_meteor['Median'].values[variables.index('noise_lag')]
        find_worst_lum[base_name] = summary_df_meteor['Median'].values[variables.index('noise_lum')]

        all_names.append(base_name)
        all_samples.append(samples_aligned)
        all_weights.append(weights_aligned)


    print("Worst 5 lag:")
    worst_lag_items = sorted(find_worst_lag.items(), key=lambda x: x[1], reverse=True)[:5]
    for base_name, lag in worst_lag_items:
        print(f"{base_name}: {lag} m")
    print("Worst 5 lum:")
    worst_lum_items = sorted(find_worst_lum.items(), key=lambda x: x[1], reverse=True)[:5]
    for base_name, lum in worst_lum_items:
        print(f"{base_name}: {lum} W")


    # Extract data for plotting
    lg_min_la_sun = np.array([v[0] for v in file_radiance_rho_dict.values()])
    bg = np.array([v[1] for v in file_radiance_rho_dict.values()])
    rho = np.array([v[2] for v in file_radiance_rho_dict.values()])
    lg_lo = np.array([v[3] for v in file_radiance_rho_dict.values()])
    lg_hi = np.array([v[4] for v in file_radiance_rho_dict.values()])
    bg_lo = np.array([v[5] for v in file_radiance_rho_dict.values()])
    bg_hi = np.array([v[6] for v in file_radiance_rho_dict.values()])

    rho_lo = np.array([v[1] for v in file_rho_jd_dict.values()])
    rho_hi = np.array([v[2] for v in file_rho_jd_dict.values()])
    tj = np.array([v[3] for v in file_rho_jd_dict.values()])
    tj_lo = np.array([v[4] for v in file_rho_jd_dict.values()])
    tj_hi = np.array([v[5] for v in file_rho_jd_dict.values()])
    inclin_val = np.array([v[6] for v in file_rho_jd_dict.values()])
    Vinf_val = np.array([v[7] for v in file_rho_jd_dict.values()])
    Q_val = np.array([v[8] for v in file_rho_jd_dict.values()])
    q_val = np.array([v[9] for v in file_rho_jd_dict.values()])
    a_val = np.array([v[10] for v in file_rho_jd_dict.values()])
    e_val = np.array([v[11] for v in file_rho_jd_dict.values()])

    lg_min_la_sun_helio = np.array([v[0] for v in file_radiance_rho_dict_helio.values()])
    lg_helio_lo = np.array([v[1] for v in file_radiance_rho_dict_helio.values()])
    lg_helio_hi = np.array([v[2] for v in file_radiance_rho_dict_helio.values()])
    bg_helio = np.array([v[3] for v in file_radiance_rho_dict_helio.values()])
    bg_helio_lo = np.array([v[4] for v in file_radiance_rho_dict_helio.values()])
    bg_helio_hi = np.array([v[5] for v in file_radiance_rho_dict_helio.values()])

    kc_par = np.array([v[0] for v in file_obs_data_dict.values()])
    F_par = np.array([v[1] for v in file_obs_data_dict.values()])
    lenght_par = np.array([v[2] for v in file_obs_data_dict.values()])
    beg_height = np.array([v[3] for v in file_obs_data_dict.values()])
    end_height = np.array([v[4] for v in file_obs_data_dict.values()])
    max_lum_height = np.array([v[5] for v in file_obs_data_dict.values()])
    avg_vel = np.array([v[6] for v in file_obs_data_dict.values()])
    init_mag = np.array([v[7] for v in file_obs_data_dict.values()])
    end_mag = np.array([v[8] for v in file_obs_data_dict.values()])
    max_mag = np.array([v[9] for v in file_obs_data_dict.values()])
    time_tot = np.array([v[10] for v in file_obs_data_dict.values()])
    zenith_angle = np.array([v[11] for v in file_obs_data_dict.values()])
    m_init_med = np.array([v[12] for v in file_obs_data_dict.values()])
    meteoroid_diameter_mm = np.array([v[13] for v in file_obs_data_dict.values()])
    erosion_beg_dyn_press = np.array([v[14] for v in file_obs_data_dict.values()])
    v_init_meteor_median = np.array([v[15] for v in file_obs_data_dict.values()])
    kinetic_energy_median = np.array([v[16] for v in file_obs_data_dict.values()])
    kinetic_energy_lo = np.array([v[17] for v in file_obs_data_dict.values()])
    kinetic_energy_hi = np.array([v[18] for v in file_obs_data_dict.values()])
    tau_median = np.array([v[19] for v in file_obs_data_dict.values()])
    tau_low95 = np.array([v[20] for v in file_obs_data_dict.values()])
    tau_high95 = np.array([v[21] for v in file_obs_data_dict.values()])

    eta_meteor_begin = np.array([v[0] for v in file_phys_data_dict.values()])
    eta_corr = np.array([v[1] for v in file_phys_data_dict.values()])
    eta_corr_hi = np.array([v[2] for v in file_phys_data_dict.values()])
    eta_corr_lo = np.array([v[3] for v in file_phys_data_dict.values()])
    sigma_meteor_begin = np.array([v[4] for v in file_phys_data_dict.values()])
    sigma_corr = np.array([v[5] for v in file_phys_data_dict.values()])
    sigma_corr_hi = np.array([v[6] for v in file_phys_data_dict.values()])
    sigma_corr_lo = np.array([v[7] for v in file_phys_data_dict.values()])
    meteoroid_diameter_mm = np.array([v[8] for v in file_phys_data_dict.values()])
    meteoroid_diameter_mm_lo = np.array([v[9] for v in file_phys_data_dict.values()])
    meteoroid_diameter_mm_hi = np.array([v[10] for v in file_phys_data_dict.values()])
    m_init_meteor_median = np.array([v[11] for v in file_phys_data_dict.values()])
    m_init_meteor_lo = np.array([v[12] for v in file_phys_data_dict.values()])
    m_init_meteor_hi = np.array([v[13] for v in file_phys_data_dict.values()])

    leng_coszen = lenght_par * np.cos(zenith_angle * np.pi / 180)

    # found the global rho corrected values find the median and the 5th and 95th percentile
    rho_corrected = np.concatenate(rho_corrected)
    eta_corrected = np.concatenate(eta_corrected)
    sigma_corrected = np.concatenate(sigma_corrected)
    tau_corrected = np.concatenate(tau_corrected)

    # eeucs = np.array([v[0] for v in file_eeu_dict.values()])
    # eeucs_lo = np.array([v[1] for v in file_eeu_dict.values()])
    # eeucs_hi = np.array([v[2] for v in file_eeu_dict.values()])
    # eeum = np.array([v[3] for v in file_eeu_dict.values()])
    # eeum_lo = np.array([v[4] for v in file_eeu_dict.values()])
    # eeum_hi = np.array([v[5] for v in file_eeu_dict.values()])
    # F_par = np.array([v[6] for v in file_eeu_dict.values()])
    # kc_par = np.array([v[7] for v in file_eeu_dict.values()])
    # lenght_par = np.array([v[8] for v in file_eeu_dict.values()])


    # print("Iron case F len eeucs ...")

    # # plot the lenght_par against eeucs and color with F_par
    # plt.figure(figsize=(10, 6))
    # # after you’ve built your rho array:
    # norm = Normalize(vmin=0, vmax=1)
    # scatter = plt.scatter(lenght_par, eeucs, c=F_par, cmap='coolwarm_r', s=30,
    #                         norm=norm, zorder=2)
    # plt.colorbar(scatter, label='F')
    # plt.xlabel('Length (km)', fontsize=15)
    # plt.ylabel('Erosion Energy per Unit Cross Section (MJ/m²)', fontsize=15)
    # # plt.title('Erosion Energy per Unit Cross Section vs Length')
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig(os.path.join(output_dir_show, f"{shower_name}_erosion_energy_vs_length.png"), bbox_inches='tight', dpi=300)

    ### PLOT rho and error against dynamic pressure color by speed ###
    print("Plotting rho vs kinetic energy...")
    fig = plt.figure(figsize=(6, 4), constrained_layout=True)

    
    scatter_d = plt.scatter(rho, np.log10(kinetic_energy_median), c=np.log10(meteoroid_diameter_mm), cmap='coolwarm', s=30, norm=Normalize(vmin=_quantile(np.log10(meteoroid_diameter_mm), 0.025), vmax=_quantile(np.log10(meteoroid_diameter_mm), 0.975)), zorder=2)
    plt.colorbar(scatter_d, label='log$_{10}$ Diameter [mm]')
    plt.errorbar(rho, np.log10(kinetic_energy_median),
                xerr=[abs(rho_lo), abs(rho_hi)],
                # yerr=[abs(kinetic_energy_lo), abs(kinetic_energy_hi)],
                elinewidth=0.75,
            capthick=0.75,
            fmt='none',
            ecolor='black',
            capsize=3,
            zorder=1
        )
    # # plot for each the all_names close to their name
    # for i, txt in enumerate(all_names):
    #     # put th text in the plot
    #     plt.annotate(txt, (rho[i], np.log10(erosion_beg_dyn_press[i])), fontsize=8, color='black')
    # invert the y axis
    plt.xlabel("$\\rho$ [kg/m³]", fontsize=15) # log$_{10}$ 
    plt.ylabel("log$_{10}$ Kinetic Energy [J]", fontsize=15)
    # plt.yscale("log")
    # grid on
    plt.grid(True)

    plt.savefig(os.path.join(output_dir_show, f"{shower_name}_rho_vs_kinetic_energy.png"), bbox_inches='tight', dpi=300)

    ### PLOT rho and error against dynamic pressure color by speed ###
    thr = 3.1  # log10 Pa  (≈ 1.58 kPa)
    fig = plt.figure(figsize=(6, 4), constrained_layout=True)

    
    scatter_d = plt.scatter(rho, np.log10(erosion_beg_dyn_press), c=np.log10(meteoroid_diameter_mm), cmap='coolwarm', s=60, norm=Normalize(vmin=_quantile(np.log10(meteoroid_diameter_mm), 0.025), vmax=_quantile(np.log10(meteoroid_diameter_mm), 0.975)), zorder=2)
    plt.colorbar(scatter_d, label='log$_{10}$ Diameter [mm]')
    scatter = plt.scatter(rho, np.log10(erosion_beg_dyn_press), c=v_init_meteor_median, cmap='viridis', s=30, norm=Normalize(vmin=v_init_meteor_median.min(), vmax=v_init_meteor_median.max()), zorder=2)
    plt.errorbar(rho, np.log10(erosion_beg_dyn_press),
                xerr=[abs(rho_lo), abs(rho_hi)],
                elinewidth=0.75,
            capthick=0.75,
            fmt='none',
            ecolor='black',
            capsize=3,
            zorder=1
        )
    # # plot for each the all_names close to their name
    # for i, txt in enumerate(all_names):
    #     # put th text in the plot
    #     plt.annotate(txt, (rho[i], np.log10(erosion_beg_dyn_press[i])), fontsize=8, color='black')
    # invert the y axis
    plt.gca().invert_yaxis()
    plt.colorbar(scatter, label='v$_{0}$ [km/s]')
    plt.xlabel("$\\rho$ [kg/m³]", fontsize=15) # log$_{10}$ 
    plt.ylabel("log$_{10}$ Dynamic Pressure [Pa]", fontsize=15)
    # plot the thr line
    plt.axhline(y=thr, color='gray', linestyle='--', linewidth=1)
    # plt.yscale("log")
    # grid on
    plt.grid(True)

    plt.savefig(os.path.join(output_dir_show, f"{shower_name}_rho_vs_dynamic_pressure.png"), bbox_inches='tight', dpi=300)


    ### PLOT rho and error against eta pressure color by speed ###

    fig = plt.figure(figsize=(6, 4), constrained_layout=True)

    scatter = plt.scatter(np.log10(rho), np.log10(eta_meteor_begin), c=v_init_meteor_median, cmap='viridis', s=30, norm=Normalize(vmin=v_init_meteor_median.min(), vmax=v_init_meteor_median.max()), zorder=2)
    # plt.errorbar(np.log10(rho), np.log10(eta_meteor_begin),
    #             xerr=[abs(rho_lo), abs(rho_hi)],
    #             elinewidth=0.75,
    #         capthick=0.75,
    #         fmt='none',
    #         ecolor='black',
    #         capsize=3,
    #         zorder=1
    #     )
    plt.colorbar(scatter, label='v$_{0}$ [km/s]')
    plt.xlabel("log$_{10}$ $\\rho$ [kg/m³]", fontsize=15) # log$_{10}$
    plt.ylabel("log$_{10}$ $\eta$ [kg/MJ]", fontsize=15)
    # plt.yscale("log")
    # grid on
    plt.grid(True)

    plt.savefig(os.path.join(output_dir_show, f"{shower_name}_rho_vs_eta.png"), bbox_inches='tight', dpi=300)

    ### PLOT rho and error of rho agaist Q ###

    fig = plt.figure(figsize=(10, 6), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.05])

    # axes
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[0, 2])

    # first plot
    ax1.errorbar(rho, q_val,
                xerr=[abs(rho_lo), abs(rho_hi)],
                elinewidth=0.75,
            capthick=0.75,
            fmt='none',
            ecolor='black',
            capsize=3,
            zorder=1
        )
    sc = ax1.scatter(rho, q_val, c=tj, cmap='viridis',
                    norm=Normalize(vmin=tj.min(), vmax=tj.max()), s=30, zorder=2)
    ax1.axhline(0.2, color='red', linestyle='-.', linewidth=1)
    # ax1.text(100, 0.18, "Sun-approaching", color='black', fontsize=12, va='bottom')
    ax1.set_xlabel("$\\rho$ [kg/m³]", fontsize=15)
    ax1.set_ylabel("Perihelion [AU]", fontsize=15)
    ax1.set_yscale("log")

    # second plot
    ax2.errorbar(rho, Q_val,
                xerr=[abs(rho_lo), abs(rho_hi)],
            elinewidth=0.75,
            capthick=0.75,
            fmt='none',
            ecolor='black',
            capsize=3,
            zorder=1
        )
    sc = ax2.scatter(rho, Q_val, c=tj, cmap='viridis',
                    norm=Normalize(vmin=tj.min(), vmax=tj.max()), s=30, zorder=2)
    ax2.axhline(4.5, color='red', linestyle='--', linewidth=1)
    # ax2.text(100, 4.2, "AST", color='black', fontsize=12, va='bottom')
    ax2.set_xlabel("$\\rho$ [kg/m³]", fontsize=15)
    ax2.set_ylabel("Aphelion [AU]", fontsize=15)
    ax2.set_yscale("log")

    # shared colorbar
    cbar = fig.colorbar(sc, cax=cax)
    cbar.set_label(r"Tisserand parameter (T$_j$)", fontsize=12)

    plt.savefig(os.path.join(output_dir_show, f"{shower_name}_rho_vs_Q_q.png"),
                dpi=300)
    plt.close()

    ### CORELATION OBSERVABLE PLOT ###
    print('Creating Correlation plot for the observable and the rho...')


    log10_m_init= np.log10(m_init_med)

    # Define your observable names and corresponding data arrays
    observable_names = [
        "$v_{avg}$ [km/s]", "$T$ [s]", "log$_{10}$($m_0$) [kg]", "$h_{beg}$ [km]", "$h_{end}$ [km]",
        "$k_c$", "$F$", "$d$ [mm]", "$T_j$", "$z_c$ [deg]"
    ]

    observable_arrays = [
        avg_vel, time_tot, log10_m_init, beg_height, end_height,
        kc_par, F_par, meteoroid_diameter_mm, tj, zenith_angle
    ]

    # observable_names = [
    #     "$v_{avg}$ [km/s]", "$T$ [s]", "$L$ [km]", "$h_{beg}$ [km]", "$h_{end}$ [km]",
    #     "$k_c$", "$F$", "$L$/$cos(z_c)$ [km]", "$h_{peak}$ [km]", "$M_{peak}$"
    # ]

    # observable_arrays = [
    #     avg_vel, time_tot, lenght_par, beg_height, end_height,
    #     kc_par, F_par, leng_coszen, max_lum_height, max_mag
    # ]

    # Create figure with 2 rows and 5 columns
    fig, axes = plt.subplots(2, 5, figsize=(15, 5))
    axes = axes.flatten()  # flatten to easily index 0-9

    for i, (name, obs) in enumerate(zip(observable_names, observable_arrays)):
        ax = axes[i]
        
        # Plot scatter of observable vs. rho_corrected
        ax.scatter(obs, rho, alpha=0.7, s=40, edgecolor='k', linewidth=0.3, color='blue')

        # Compute and annotate correlation coefficient
        corr = np.corrcoef(obs, rho)[0, 1]
        ax.set_title(f'corr: {corr:.2f}', fontsize=15)
        # put a line of best fit
        z = np.polyfit(obs, rho, 1)
        p = np.poly1d(z)
        ax.plot(obs, p(obs), color='red', linewidth=1, label=f'corr: {corr:.2f}')

        ax.set_xlabel(f'{name}', fontsize=12)
        ax.set_ylabel(r'$\rho$ [kg/m$^3$]' if i in [0, 5] else '', fontsize=12)
        # # if $M_{peak}$ in the name invert the x axis
        # if 'M_{peak}' in name:
        #     ax.set_xlim(ax.get_xlim()[::-1])
        ax.grid(True)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir_show, f"{shower_name}_rho_vs_observables_grid.png"), dpi=300, bbox_inches='tight')
    plt.close()



    ### RADIANCE PLOT ###
    print("saving radiance plot...")

    # print(lg_lo, lg_hi, bg_lo, bg_hi)
    plt.figure(figsize=(8, 6))
    stream_lg_min_la_sun = []
    stream_bg = []
    shower_iau_no = -1

    if radiance_plot_flag == True:
        # check if "C:\Users\maxiv\WMPG-repoMAX\Code\Utils\streamfulldata2022.csv" exists
        if not os.path.exists(r"C:\Users\maxiv\Documents\UWO\Papers\2)ORI-CAP-PER-DRA\Results"):
            print("GMN File traj_summary_monthly not found. Please get the data from the GMN website or use the local files.")
        else:
            # empty pandas dataframe
            stream_data = []
            # if name has "CAP" in the shower_name, then filter the stream_data for the shower_iau_no
            print(f"Filtering stream data for shower: {shower_name}")
            shower_iau_no = -1
            if "CAP" in shower_name:
                csv_file_1 = loadTrajectorySummaryFast(r"C:\Users\maxiv\Documents\UWO\Papers\2)ORI-CAP-PER-DRA\Results","traj_summary_monthly_202407.txt","traj_summary_monthly_202407.pickle")
                # save the csv_file to a file called: "traj_summary_monthly_202408.csv"
                csv_file_1.to_csv(r"C:\Users\maxiv\Documents\UWO\Papers\2)ORI-CAP-PER-DRA\Results\traj_summary_monthly_202407.csv", index=False)
                csv_file_2 = loadTrajectorySummaryFast(r"C:\Users\maxiv\Documents\UWO\Papers\2)ORI-CAP-PER-DRA\Results","traj_summary_monthly_202408.txt","traj_summary_monthly_202408.pickle")
                # save the csv_file to a file called: "traj_summary_monthly_202408.csv"
                csv_file_2.to_csv(r"C:\Users\maxiv\Documents\UWO\Papers\2)ORI-CAP-PER-DRA\Results\traj_summary_monthly_202408.csv", index=False)
                # extend the in csv_file
                stream_data = pd.concat([csv_file_1, csv_file_2], ignore_index=True)
                shower_iau_no = 1#"00001"
            elif "PER" in shower_name:
                stream_data = loadTrajectorySummaryFast(r"C:\Users\maxiv\Documents\UWO\Papers\2)ORI-CAP-PER-DRA\Results","traj_summary_monthly_202408.txt","traj_summary_monthly_202408.pickle")
                # save the csv_file to a file called: "traj_summary_monthly_202408.csv"
                stream_data.to_csv(r"C:\Users\maxiv\Documents\UWO\Papers\2)ORI-CAP-PER-DRA\Results\traj_summary_monthly_202408.csv", index=False)
                shower_iau_no = 7#"00007"
            elif "ORI" in shower_name: 
                stream_data = loadTrajectorySummaryFast(r"C:\Users\maxiv\Documents\UWO\Papers\2)ORI-CAP-PER-DRA\Results","traj_summary_monthly_202410.txt","traj_summary_monthly_202410.pickle")
                # save the csv_file to a file called: "traj_summary_monthly_202408.csv"
                stream_data.to_csv(r"C:\Users\maxiv\Documents\UWO\Papers\2)ORI-CAP-PER-DRA\Results\traj_summary_monthly_202410.csv", index=False)
                shower_iau_no = 8#"00008"
            elif "DRA" in shower_name:  
                stream_data = loadTrajectorySummaryFast(r"C:\Users\maxiv\Documents\UWO\Papers\2)ORI-CAP-PER-DRA\Results","traj_summary_monthly_202410.txt","traj_summary_monthly_202410.pickle")
                # save the csv_file to a file called: "traj_summary_monthly_202408.csv"
                stream_data.to_csv(r"C:\Users\maxiv\Documents\UWO\Papers\2)ORI-CAP-PER-DRA\Results\traj_summary_monthly_202410.csv", index=False)
                shower_iau_no = 9#"00009"
            else:
                stream_data = loadTrajectorySummaryFast(r"C:\Users\maxiv\Documents\UWO\Papers\2)ORI-CAP-PER-DRA\Results","traj_summary_monthly_202402.txt","traj_summary_monthly_202402.pickle")
                # save the csv_file to a file called: "traj_summary_monthly_202402.csv"
                stream_data.to_csv(r"C:\Users\maxiv\Documents\UWO\Papers\2)ORI-CAP-PER-DRA\Results\traj_summary_monthly_202402.csv", index=False)
                shower_iau_no = -1


            print(f"Filtering stream data for shower IAU number: {shower_iau_no}")
            # filter the stream_data for the shower_iau_no
            stream_data = stream_data[stream_data['IAU (No)'] == shower_iau_no]
            print(f"Found {len(stream_data)} stream data points for shower IAU number: {shower_iau_no}")
            # # and take the one that have activity " annual "
            # stream_data = stream_data[stream_data['activity'].str.contains("annual", case=False, na=False)]
            # print(f"Found {len(stream_data)} stream data points for shower IAU number: {shower_iau_no} with activity 'annual'")
            # extract all LoR	S_LoR	LaR
            stream_lor = stream_data[['LAMgeo (deg)', 'BETgeo (deg)', 'Sol lon (deg)','LAMhel (deg)', 'BEThel (deg)','Vgeo (km/s)','HtBeg (km)', 'TisserandJ']].values
            # translate to double precision float
            stream_lor = stream_lor.astype(np.float64)
            # and now compute lg_min_la_sun = (lg - la_sun)%360
            stream_lg_min_la_sun = (stream_lor[:, 0] - stream_lor[:, 2]) % 360
            stream_bg = stream_lor[:, 1]
            stream_lg_min_la_sun_helio = (stream_lor[:, 3] - stream_lor[:, 2]) % 360
            stream_bg_helio = stream_lor[:, 4]
            stream_vgeo = stream_lor[:, 5]
            stream_htbeg = stream_lor[:, 6]
            stream_tj = stream_lor[:, 7]
            # print(f"Found {len(stream_lg_min_la_sun)} stream data points for shower IAU number: {shower_iau_no}")

            if shower_iau_no != -1:

                print(f"Plotting stream data for shower IAU number: {shower_iau_no}")

                spor_data = loadTrajectorySummaryFast(r"C:\Users\maxiv\Documents\UWO\Papers\2)ORI-CAP-PER-DRA\Results","traj_summary_monthly_202402.txt","traj_summary_monthly_202402.pickle")
                # save the csv_file to a file called: "traj_summary_monthly_202402.csv"
                spor_data.to_csv(r"C:\Users\maxiv\Documents\UWO\Papers\2)ORI-CAP-PER-DRA\Results\traj_summary_monthly_202402.csv", index=False)
                spor_data = spor_data[spor_data['IAU (No)'] == -1]

                spor_lor = spor_data[['LAMgeo (deg)', 'BETgeo (deg)', 'Sol lon (deg)','LAMhel (deg)', 'BEThel (deg)','Vgeo (km/s)','HtBeg (km)', 'TisserandJ']].values
                # translate to double precision float
                spor_lor = spor_lor.astype(np.float64)
                # and now compute lg_min_la_sun = (lg - la_sun)%360
                spor_lg_min_la_sun = (spor_lor[:, 0] - spor_lor[:, 2]) % 360
                spor_bg = spor_lor[:, 1]
                spor_lg_min_la_sun_helio = (spor_lor[:, 3] - spor_lor[:, 2]) % 360
                spor_bg_helio = spor_lor[:, 4]
                spor_vgeo = spor_lor[:, 5]
                spor_htbeg = spor_lor[:, 6]
                spor_tj = spor_lor[:, 7]

                ### Velocity vs begin height scaterd with rho ###
                print('Creating Velocity vs Begin Height scatter plot with stream...')
                plt.figure(figsize=(10, 6))
                scatter_GMN_spor = plt.scatter(spor_vgeo, spor_htbeg, c='black', s=1, alpha=0.5, linewidths=0, zorder=1) # c=stream_tj, cmap='inferno'
                scatter_GMN_stream = plt.scatter(stream_vgeo, stream_htbeg, c='red', s=5, alpha=0.5, linewidths=0, zorder=2) # c=stream_tj, cmap='inferno'
                # plt.colorbar(scatter_GMN, label='$T_{j}$', orientation='vertical')
                ## mass or mm diameter
                # scatter_d = plt.scatter(Vinf_val, beg_height, c=log10_m_init, cmap='coolwarm', s=60, norm=Normalize(vmin=log10_m_init.min(), vmax=log10_m_init.max()), zorder=2)
                # plt.colorbar(scatter_d, label='mass [kg]')
                scatter = plt.scatter(Vinf_val, beg_height, c=np.log10(rho), cmap='viridis', s=20, norm=Normalize(vmin=np.log10(rho.min()), vmax=np.log10(rho.max())), zorder=3)
                plt.colorbar(scatter, label='log$_{10}$ $\\rho$ [kg/m³]')

                plt.xlabel('$v_{geo}$ [km/s]', fontsize=15)
                plt.ylabel('$h_{beg}$ [km]', fontsize=15)
                plt.grid(True)

                # # x axes from 10 to 22
                # plt.xlim(11, 22)
                # # y axes from 70 to 110
                # plt.ylim(70, 110)

                plt.tight_layout()
                plt.savefig(os.path.join(output_dir_show, f"{shower_name}_velocity_vs_beg_height.png"), bbox_inches='tight', dpi=300)
                plt.close()

                ##### plot the data #####

                # after you’ve built your rho array:
                norm = Normalize(vmin=rho.min(), vmax=rho.max())
                # cmap = cm.viridis

                # your stream data arrays
                x = stream_lg_min_la_sun
                y = stream_bg
                # if shower_iau_no == -1: # revolve around the direction of motion of the Earth
                #     x = np.where(x > 180, x - 360, x)

                # build the KDE
                xy  = np.vstack([x, y])
                kde = gaussian_kde(xy)

                # sample on a grid
                xmin, xmax = x.min(), x.max()
                ymin, ymax = y.min(), y.max()
                X, Y = np.mgrid[xmin:xmax:200j, ymin:ymax:200j]
                positions = np.vstack([X.ravel(), Y.ravel()])
                Z = np.reshape(kde(positions).T, X.shape)

                # heatmap via imshow
                plt.imshow(
                    Z.T,
                    extent=(xmin, xmax, ymin, ymax),
                    origin='lower',
                    aspect='auto',
                    cmap='inferno',
                    alpha=0.6
                )

                # get the x axis limits
                xlim = plt.xlim()
                # get the y axis limits
                ylim = plt.ylim()

                if "CAP" in shower_name:
                    print("Plotting CAP shower data...")
                    plt.xlim(xlim[0], 182)
                    plt.ylim(8, 11.5)
                elif "PER" in shower_name:
                    print("Plotting PER shower data...")
                    # plt.xlim(xlim[0], 65)
                    # plt.ylim(77, 81)
                elif "ORI" in shower_name: 
                    print("Plotting ORI shower data...")
                    # put an x lim and a y lim
                    plt.xlim(xlim[0], 251)
                    plt.ylim(-9, -6)
                elif "DRA" in shower_name:  
                    print("Plotting DRA shower data...")
                    # put an x lim and a y lim
                    plt.xlim(xlim[0], 65)
                    plt.ylim(77, 80.5)

                # if shower_iau_no == -1:
                #     lg_min_la_sun = np.where(lg_min_la_sun > 180, lg_min_la_sun - 360, lg_min_la_sun)

                # then draw points on top, at zorder=2 # jet
                scatter = plt.scatter(
                    lg_min_la_sun, bg,
                    c=rho,
                    cmap='viridis',
                    norm=norm,
                    s=30,
                    zorder=2
                )

                # add the error bars for values lg_lo, lg_hi, bg_lo, bg_hi
                for i in range(len(lg_min_la_sun)):
                    # draw error bars for each point
                    plt.errorbar(
                        lg_min_la_sun[i], bg[i],
                        xerr=[[abs(lg_hi[i])], [abs(lg_lo[i])]],
                        yerr=[[abs(bg_hi[i])], [abs(bg_lo[i])]],
                        elinewidth=0.75,
                        capthick=0.75,
                        fmt='none',
                        ecolor='black',
                        capsize=3,
                        zorder=1
                    )
                
                # # annotate each point with its base_name in tiny text
                # for base_name, (x, y, z, x_lo, x_hi, y_lo, y_hi) in file_radiance_rho_dict.items():
                #     plt.annotate(
                #         base_name,
                #         xy=(x, y),
                #         xytext=(30, 5),             # 5 points vertical offset
                #         textcoords='offset points',
                #         ha='center',
                #         va='bottom',
                #         fontsize=6,
                #         alpha=0.8
                #     )

                # increase the size of the tick labels
                plt.gca().tick_params(labelsize=15)

                plt.gca().invert_xaxis()

                # increase the label size
                cbar = plt.colorbar(scatter, label='$\\rho$ [kg/m$^3$]')
                # 2. now set the label’s font size and the tick labels’ size
                cbar.set_label('$\\rho$ [kg/m$^3$]', fontsize=15)
                cbar.ax.tick_params(labelsize=15)

                plt.xlabel(r'$\lambda_{g} - \lambda_{\odot}$ (J2000)', fontsize=15)
                plt.ylabel(r'$\beta_{g}$ (J2000)', fontsize=15)
                # plt.title('Radiant Distribution of Meteors')
                plt.grid(True)
                plt.savefig(os.path.join(output_dir_show, f"{shower_name}_geo_radiant_distribution_CI.png"), bbox_inches='tight', dpi=300)
                plt.close()

            else: 
                ### Velocity vs begin height scaterd with rho ###
                print('Creating Velocity vs Begin Height scatter plot with stream...')
                plt.figure(figsize=(10, 6))
                scatter_GMN = plt.scatter(stream_vgeo, stream_htbeg, c='black', s=1, alpha=0.5, linewidths=0, zorder=1) # c=stream_tj, cmap='inferno'
                # plt.colorbar(scatter_GMN, label='$T_{j}$', orientation='vertical')
                # mass or mm diameter
                # scatter_d = plt.scatter(Vinf_val, beg_height, c=np.log10(eta_meteor_begin), cmap='coolwarm', s=60, norm=Normalize(vmin=_quantile(np.log10(eta_meteor_begin), 0.025), vmax=_quantile(np.log10(eta_meteor_begin), 0.975)), zorder=2)
                # plt.colorbar(scatter_d, label='log$_{10}$ $\\eta$ [kg/MJ]')
                # scatter_d = plt.scatter(Vinf_val, beg_height, c=np.log10(meteoroid_diameter_mm), cmap='coolwarm', s=60, norm=Normalize(vmin=_quantile(np.log10(meteoroid_diameter_mm), 0.025), vmax=_quantile(np.log10(meteoroid_diameter_mm), 0.975)), zorder=2)
                # plt.colorbar(scatter_d, label='log$_{10}$ Diameter [mm]')
                # scatter_d = plt.scatter(Vinf_val, beg_height, c=np.log10(erosion_beg_dyn_press), cmap='coolwarm', s=60, norm=Normalize(vmin=_quantile(np.log10(erosion_beg_dyn_press), 0.025), vmax=_quantile(np.log10(erosion_beg_dyn_press), 0.975)), zorder=2)
                # plt.colorbar(scatter_d, label='log$_{10}$ Dynamic Pressure [Pa]')

                # --- Dynamic pressure two-class overlay (NO gradient) ---
                logq = np.log10(erosion_beg_dyn_press)
                

                mask_hi = logq >= thr
                mask_lo = ~mask_hi

                # Hollow markers so the rho colors still show underneath
                plt.scatter(Vinf_val[mask_lo], beg_height[mask_lo],
                            facecolors='none', edgecolors='tab:blue', s=60, linewidths=1.2,
                            label=r'$\log_{10} q < 3.2$', zorder=3)

                plt.scatter(Vinf_val[mask_hi], beg_height[mask_hi],
                            facecolors='none', edgecolors='tab:red', s=60, linewidths=1.2,
                            label=r'$\log_{10} q \ge 3.2$ (q ≥ 1.58 kPa)', zorder=4)
                all_names = np.array(all_names)
                # add the names only to the mask_hi
                # for i, txt in enumerate(all_names[mask_hi]):
                #     plt.annotate(txt, (Vinf_val[mask_hi][i], beg_height[mask_hi][i]), fontsize=8, color='black')

                scatter = plt.scatter(Vinf_val, beg_height, c=np.log10(rho), cmap='viridis', s=20, norm=Normalize(vmin=np.log10(rho).min(), vmax=np.log10(rho).max()), zorder=3)
                plt.colorbar(scatter, label='log$_{10}$ $\\rho$ [kg/m³]')

                plt.xlabel('$v_{geo}$ [km/s]', fontsize=15)
                plt.ylabel('$h_{beg}$ [km]', fontsize=15)
                plt.grid(True)

                # # x axes from 10 to 22
                # plt.xlim(11, 22)
                # # y axes from 70 to 110
                # plt.ylim(70, 110)

                plt.tight_layout()
                plt.savefig(os.path.join(output_dir_show, f"{shower_name}_velocity_vs_beg_height_rho-eta.png"), bbox_inches='tight', dpi=300)
                plt.close()

                for plot_type in ['helio', 'geo']:
                    print(f"Plotting ecliptic {plot_type}centric data with GMN...")
                    if plot_type == 'geo':
                        lg_min_la_sun_plot = lg_min_la_sun
                        bg_plot = bg
                        lg_lo_plot = lg_lo
                        lg_hi_plot = lg_hi
                        bg_lo_plot = bg_lo
                        bg_hi_plot = bg_hi
                        stream_lg_min_la_sun_plot = stream_lg_min_la_sun
                        stream_bg_plot = stream_bg
                    elif plot_type == 'helio':  
                        lg_min_la_sun_plot = lg_min_la_sun_helio
                        bg_plot = bg_helio
                        lg_lo_plot = lg_helio_lo
                        lg_hi_plot = lg_helio_hi
                        bg_lo_plot = bg_helio_lo
                        bg_hi_plot = bg_helio_hi
                        stream_lg_min_la_sun_plot = stream_lg_min_la_sun_helio
                        stream_bg_plot = stream_bg_helio

                    def wrap_around_center_deg(x, center=270):
                        """Wraps angles around center to [-180, 180] and returns degrees."""
                        return (x - center + 180) % 360 - 180
                    
                    # --- Wrap stream background points ---
                    x_stream_deg_wrapped = wrap_around_center_deg(stream_lg_min_la_sun_plot)
                    x_stream_rad_wrapped = -np.deg2rad(x_stream_deg_wrapped)  # flip
                    y_stream_rad = np.deg2rad(stream_bg_plot)

                    # KDE on stream points (to get color per point)
                    xy_stream = np.vstack([x_stream_rad_wrapped, y_stream_rad])
                    kde_stream = gaussian_kde(xy_stream)
                    stream_density = kde_stream(xy_stream)  # one value per point

                    # --- Setup black Aitoff plot ---
                    fig = plt.figure(figsize=(10, 5))
                    ax = fig.add_subplot(111, projection="aitoff", facecolor="black")  # black background
                    ax.set_facecolor("black")
                    ax.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.3)

                    # --- Scatter stream points with inferno and alpha ---
                    scatter_stream = ax.scatter(
                        x_stream_rad_wrapped,
                        y_stream_rad,
                        c=stream_density,
                        cmap='inferno',
                        s=1,
                        alpha=0.5,
                        linewidths=0,
                        zorder=1
                    )
                    # --- Plot actual meteor points ---
                    # Wrap & flip heliocentric longitude
                    lg_deg_wrapped = wrap_around_center_deg(lg_min_la_sun_plot)
                    lg_rad_flipped = -np.deg2rad(lg_deg_wrapped)
                    bg_rad = np.deg2rad(bg_plot)

                    # Normalize rho for color mapping
                    norm = Normalize(vmin=np.nanmin(rho), vmax=np.nanmax(rho))
                    scatter = ax.scatter(
                        lg_rad_flipped,
                        bg_rad,
                        c=rho,
                        cmap='viridis',
                        norm=norm,
                        s=20,
                        edgecolors='k',
                        linewidths=0.3,
                        zorder=2
                    )

                    # Error bars
                    for i in range(len(lg_min_la_sun_plot)):
                        x = lg_rad_flipped[i]
                        y = bg_rad[i]
                        xerr = [[np.deg2rad(abs(lg_hi_plot[i]))], [np.deg2rad(abs(lg_lo_plot[i]))]]
                        yerr = [[np.deg2rad(abs(bg_hi_plot[i]))], [np.deg2rad(abs(bg_lo_plot[i]))]]
                        ax.errorbar(
                            x, y,
                            xerr=xerr,
                            yerr=yerr,
                            elinewidth=0.75,
                            capthick=0.0,
                            fmt='none',
                            ecolor='black',
                            capsize=3,
                            zorder=1
                        )

                    # # extract from file_radiance_rho_dict the base_name and the values
                    # # for each point in file_radiance_rho_dict
                    # file_radiance_rho_dict = {k: v for k, v in file_radiance_rho_dict.items() if k in file_radiance_rho_dict_helio}
                    # # annotate each point with its base_name in tiny text for lg_rad_flipped and bg_rad
                    # for ii in range(len(lg_rad_flipped)):
                    #     plt.annotate(
                    #         list(file_radiance_rho_dict.keys())[ii],
                    #         xy=(lg_rad_flipped[ii], bg_rad[ii]),
                    #         xytext=(30, 5),             # 5 points vertical offset
                    #         textcoords='offset points',
                    #         color = 'gray',
                    #         ha='center',
                    #         va='bottom',
                    #         fontsize=6,
                    #         alpha=0.8
                    #     )

                    # --- Add colorbar for scatter points only ---
                    cbar = plt.colorbar(scatter, orientation='horizontal', pad=0.08)
                    cbar.set_label('$\\rho$ [kg/m$^3$]', fontsize=13)
                    cbar.ax.tick_params(labelsize=11)

                    # --- Custom X-axis: centered at 270° and inverted ---
                    xticks_deg = np.arange(-150, 181, 30)  # degrees for Aitoff tick positions
                    xtick_labels = [(str(int((270 - t) % 360)) + "°") for t in xticks_deg]  # subtract instead of add = invert
                    ax.set_xticks(np.deg2rad(xticks_deg))
                    ax.set_xticklabels(xtick_labels, fontsize=12)

                    # Set all x-axis tick labels to gray
                    for label in ax.get_xticklabels():
                        label.set_color("gray")

                    # --- Labels ---
                    if plot_type == 'helio':
                        ax.set_xlabel(r"$\lambda_{h} - \lambda_{\odot}$ (J2000)", fontsize=15)
                        ax.set_ylabel(r"$\beta_{h}$ (J2000)", fontsize=15)
                    elif plot_type == 'geo':
                        plt.xlabel(r'$\lambda_{g} - \lambda_{\odot}$ (J2000)', fontsize=15)
                        plt.ylabel(r'$\beta_{g}$ (J2000)', fontsize=15)

                    # --- Save plot ---
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir_show, f"{shower_name}_{plot_type}_radiant_distribution.png"), dpi=300)
                    plt.close()

    #### Combine all samples and weights from different dynesty runs ####

    # Combine all the samples and weights into a single array
    combined_samples = np.vstack(all_samples)
    combined_weights = np.concatenate(all_weights)
    # correlate all_names with combined_samples
    all_names = np.array(all_names)

    # use a fixed seed for reproducibility
    rng = np.random.default_rng(seed=42)

    # for each variable-column:
    for i in range(combined_samples.shape[1]):
        col = combined_samples[:, i]
        miss = np.isnan(col)
        if not miss.any():
            continue

        # all the valid entries & their weights
        vals = col[~miss]
        wts  = combined_weights[~miss].astype(float)
        if wts.sum() > 0:
            wts /= wts.sum()   # normalize
            # draw replacements with the same weighted distribution
            fill = rng.choice(vals, size=miss.sum(), replace=True, p=wts)
        else:
            # fallback to unweighted draw
            fill = rng.choice(vals, size=miss.sum(), replace=True)

        col[miss] = fill

    # # save the combined_samples in an other variable
    # combined_samples_savecov = combined_samples.copy()
    # # put in log the combined_samples base on the flag
    # for j, var in enumerate(variables):
    #     # check if the flag associated to the variable is log
    #     if 'log' in flags_dict[variable]:
    #         combined_samples_savecov[:, j] = 10**combined_samples_savecov[:, j]
 
    # Create a CombinedResults object for the combined samples
    combined_results = CombinedResults(combined_samples, combined_weights)

    # combined_results_cov = CombinedResults(combined_samples_savecov, combined_weights)

    ### Apply the iron-by-velocity down-weighting ###

    # combined_results = reweight_iron_by_velocity(combined_results, variables)

    ### Apply the iron-by-velocity down-weighting ###

    summary_df = summarize_from_cornerplot(
        combined_results,
        variables,
        labels
    )

    ### CREATE A TABLE of the Combination of all samples ###

    print(summary_df.to_string(index=False))

    def summary_to_latex(summary_df, shower_name="ORI"):
        latex_lines = []

        header = r"""\begin{table}[htbp]
        \centering
        \renewcommand{\arraystretch}{1.2}
        \setlength{\tabcolsep}{4pt}
        \resizebox{\textwidth}{!}{
        \begin{tabular}{|l|c|c|c|c|c|}
        \hline
        \textbf{Parameter} & \textbf{2.5CI} & \textbf{Mode} & \textbf{Mean} & \textbf{Median} & \textbf{97.5CI} \\
        \hline"""
        latex_lines.append(header)

        # Format each row
        for _, row in summary_df.iterrows():
            param = row["Label"]
            low = f"{row['Low95']:.4g}" if not np.isnan(row['Low95']) else "---"
            mode = f"{row['Mode']:.4g}" if not np.isnan(row['Mode']) else "---"
            mean = f"{row['Mean']:.4g}" if not np.isnan(row['Mean']) else "---"
            median = f"{row['Median']:.4g}" if not np.isnan(row['Median']) else "---"
            high = f"{row['High95']:.4g}" if not np.isnan(row['High95']) else "---"

            line = f"    {param} & {low} & {mode} & {mean} & {median} & {high} \\\\"
            latex_lines.append(line)
            latex_lines.append("    \\hline")

        # if there is _ in the shower_name put a \
        shower_name_plot = shower_name.replace("_", "\\_")
        # Footer
        footer = r"    \end{tabular}}"
        footer2 = rf"""    \caption{{Overall posterior summary statistics for {num_meteors} meteors of the {shower_name_plot} shower.}}
        \label{{tab:overall_summary_{shower_name.lower()}}}
     \end{{table}}"""

        latex_lines.append(footer)
        latex_lines.append(footer2)

        return "\n".join(latex_lines)

    latex_code = summary_to_latex(summary_df, shower_name)
    print(latex_code)

    print("Saving LaTeX table...")
    # Save to file
    with open(os.path.join(output_dir_show, shower_name+"_posterior_summary_table.tex"), "w") as f:
        f.write(latex_code)


    combined_samples_copy_plot = combined_samples.copy()
    labels_plot_copy_plot = labels.copy()
    for j, var in enumerate(variables):
        if np.all(np.isnan(combined_samples_copy_plot[:, j])):
            continue
        if var in ['m_init','erosion_mass_min', 'erosion_mass_max']:
            combined_samples_copy_plot[:, j] = np.log10(combined_samples_copy_plot[:, j])
            labels_plot_copy_plot[j] =r"$\log_{10}$(" +labels_plot_copy_plot[j]+")"
        if var in ['erosion_coeff', 'erosion_coeff_change']:
            combined_samples_copy_plot[:, j] = np.log10(combined_samples_copy_plot[:, j] * 1e6)
            labels_plot_copy_plot[j] =r"$\log_{10}$(" +labels_plot_copy_plot[j]+")"
        if var in ['v_init', 'erosion_height_start', 'erosion_height_change']:
            combined_samples_copy_plot[:, j] = combined_samples_copy_plot[:, j] / 1000.0
        if var in ['sigma', 'erosion_sigma_change']:
            combined_samples_copy_plot[:, j] = combined_samples_copy_plot[:, j] * 1e6

    ### create new directory for rho plots ###

    # create a new folder for the rho plots
    output_dir_rho = os.path.join(output_dir_show, "rho_plots")
    os.makedirs(output_dir_rho, exist_ok=True)

    ### JD vs rho plot ###

    print("Distribution plots:")
    # Extract from combined_results
    samples = combined_samples_copy_plot
    # samples = combined_results.samples
    weights = combined_results.importance_weights()
    w = weights / np.sum(weights)

    rho_corrected_lo, rho_corrected_median, rho_corrected_hi = _quantile(rho_corrected, [0.025, 0.5, 0.975], weights=w)

    print("Creating combined plot T_j rho and k_c...")
    
    # print("Creating combined plot T_j rho and mass...")


    # save only the plot above as a separate plot:
    
    # Create figure
    fig = plt.figure(figsize=(8, 6))
    ax_dist = fig.add_subplot(111)

    smooth = 0.02
    lo, hi = np.min(rho_corrected), np.max(rho_corrected)
    nbins = int(round(10. / smooth))
    hist, edges = np.histogram(rho_corrected, bins=nbins, weights=w, range=(lo, hi))
    hist = norm_kde(hist, 10.0)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])

    ax_dist.fill_between(bin_centers, hist, color='black', alpha=0.6)

    # Percentile lines
    ax_dist.axvline(rho_corrected_median, color='black', linestyle='--', linewidth=1.5)
    ax_dist.axvline(rho_corrected_lo, color='black', linestyle='--', linewidth=1.5)
    ax_dist.axvline(rho_corrected_hi, color='black', linestyle='--', linewidth=1.5)

    # Title and formatting
    plus = rho_corrected_hi - rho_corrected_median
    minus = rho_corrected_median - rho_corrected_lo
    fmt = lambda v: f"{v:.4g}" if np.isfinite(v) else "---"
    title = rf"Tot N.{len(tj)} — $\rho$ [kg/m$^3$] = {fmt(rho_corrected_median)}$^{{+{fmt(plus)}}}_{{-{fmt(minus)}}}$"
    ax_dist.set_title(title, fontsize=20)
    # ax_dist.tick_params(axis='x', labelbottom=False)
    ax_dist.tick_params(axis='y', left=False, labelleft=False)
    ax_dist.set_ylabel("")
    ax_dist.set_xlabel(r'$\rho$ [kg/m$^3$]', fontsize=20)
    ax_dist.spines['left'].set_visible(False)
    ax_dist.spines['right'].set_visible(False)
    ax_dist.spines['top'].set_visible(False)
    plt.savefig(os.path.join(output_dir_show, f"{shower_name}_rho_distribution_both.png"), bbox_inches='tight')

    # Tau plot ############
    print("Creating combined plot tau...")
    tau_corrected_lo, tau_corrected_median, tau_corrected_hi = _quantile(tau_corrected, [0.025, 0.5, 0.975], weights=w)
    # Create figure tau
    fig = plt.figure(figsize=(8, 6))
    ax_dist = fig.add_subplot(111)

    smooth = 0.02
    lo, hi = np.min(tau_corrected), np.max(tau_corrected)
    nbins = int(round(10. / smooth))
    hist, edges = np.histogram(tau_corrected, bins=nbins, weights=w, range=(lo, hi))
    hist = norm_kde(hist, 10.0)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])

    ax_dist.fill_between(bin_centers, hist, color='blue', alpha=0.6)

    # Percentile lines
    ax_dist.axvline(tau_corrected_median, color='blue', linestyle='--', linewidth=1.5)
    ax_dist.axvline(tau_corrected_lo, color='blue', linestyle='--', linewidth=1.5)
    ax_dist.axvline(tau_corrected_hi, color='blue', linestyle='--', linewidth=1.5)

    # Title and formatting
    plus = tau_corrected_hi - tau_corrected_median
    minus = tau_corrected_median - tau_corrected_lo
    fmt = lambda v: f"{v:.4g}" if np.isfinite(v) else "---"
    title = rf"Tot N.{len(tj)} — $\tau$ = {fmt(tau_corrected_median)}$^{{+{fmt(plus)}}}_{{-{fmt(minus)}}}$"
    ax_dist.set_title(title, fontsize=20)
    # ax_dist.tick_params(axis='x', labelbottom=False)
    ax_dist.tick_params(axis='y', left=False, labelleft=False)
    ax_dist.set_ylabel("")
    ax_dist.set_xlabel(r'$\tau$', fontsize=20)
    ax_dist.spines['left'].set_visible(False)
    ax_dist.spines['right'].set_visible(False)
    ax_dist.spines['top'].set_visible(False)
    # x axis from 0 to tau_corrected*2
    # ax_dist.set_xlim(0, tau_corrected_hi+tau_corrected_median)
    plt.savefig(os.path.join(output_dir_show, f"{shower_name}_tau_distribution_both.png"), bbox_inches='tight')

    # Create figure
    fig = plt.figure(figsize=(8, 10))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 3] , hspace=0) # , hspace=0.05

    # Set main axes (with shared x-axis)
    ax_dist = fig.add_subplot(gs[0])
    ax_scatter = fig.add_subplot(gs[1], sharex=ax_dist)

    # --- TOP PANEL: Rho Distribution ---
    smooth = 0.02
    lo, hi = np.min(rho_corrected), np.max(rho_corrected)
    nbins = int(round(10. / smooth))
    hist, edges = np.histogram(rho_corrected, bins=nbins, weights=w, range=(lo, hi))
    hist = norm_kde(hist, 10.0)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])

    ax_dist.fill_between(bin_centers, hist, color='black', alpha=0.6)

    # Percentile lines
    ax_dist.axvline(rho_corrected_median, color='black', linestyle='--', linewidth=1.5)
    ax_dist.axvline(rho_corrected_lo, color='black', linestyle='--', linewidth=1.5)
    ax_dist.axvline(rho_corrected_hi, color='black', linestyle='--', linewidth=1.5)

    # Title and formatting
    plus = rho_corrected_hi - rho_corrected_median
    minus = rho_corrected_median - rho_corrected_lo
    fmt = lambda v: f"{v:.4g}" if np.isfinite(v) else "---"
    title = rf"Tot N.{len(tj)} — $\rho$ [kg/m$^3$] = {fmt(rho_corrected_median)}$^{{+{fmt(plus)}}}_{{-{fmt(minus)}}}$"
    ax_dist.set_title(title, fontsize=20)
    ax_dist.set_xlim(-100, 8300)
    ax_dist.tick_params(axis='x', labelbottom=False)
    ax_dist.tick_params(axis='y', left=False, labelleft=False)
    ax_dist.set_ylabel("")
    ax_dist.spines['bottom'].set_visible(False)
    ax_dist.spines['left'].set_visible(False)
    ax_dist.spines['right'].set_visible(False)
    ax_dist.spines['top'].set_visible(False)

    # --- BOTTOM PANEL: Rho vs Tj ---
    for i in range(len(tj)):
        ax_scatter.errorbar(
            rho[i], tj[i],
            xerr=[[abs(rho_lo[i])], [abs(rho_hi[i])]],
            yerr=[[abs(tj_lo[i])], [abs(tj_hi[i])]],
            elinewidth=0.75,
            capthick=0.75,
            fmt='none',
            ecolor='black',
            capsize=3,
            zorder=1
        )

    scatter = ax_scatter.scatter(
        rho, tj,
        c=np.log10(meteoroid_diameter_mm),
        # c=kc_par,
        # cmap='viridis',
        cmap='coolwarm',
        norm=Normalize(vmin=_quantile(np.log10(meteoroid_diameter_mm), 0.025), vmax=_quantile(np.log10(meteoroid_diameter_mm), 0.975)),
        # norm=Normalize(vmin=log10_m_init.min(), vmax=log10_m_init.max()),
        # norm=Normalize(vmin=kc_par.min(), vmax=kc_par.max()),
        s=30,
        zorder=2
    )

    # Add manually aligned colorbar
    # Get position of ax_scatter to align colorbar
    pos = ax_scatter.get_position()
    cbar_ax = fig.add_axes([pos.x1 + 0.01, pos.y0, 0.02, pos.height])  # [left, bottom, width, height]
    cbar = plt.colorbar(scatter, cax=cbar_ax)
    cbar.set_label('$log_{10}$ Diameter [mm]', fontsize=20)
    # cbar.set_label('$log_{10} m_0 [kg]$', fontsize=20)
    # cbar.set_label('$k_c$ parameter', fontsize=20)
    # the ticks size of the colorbar
    cbar.ax.tick_params(labelsize=20)

    # Tj markers
    if shower_iau_no == -1:
        ax_scatter.axhline(y=3.0, color='lime', linestyle=':', linewidth=1.5, zorder=1)
        ax_scatter.text(7500, 3.1, 'AST', color='black', fontsize=15, va='bottom')
        ax_scatter.axhline(y=2.0, color='lime', linestyle='--', linewidth=1.5, zorder=1)
        ax_scatter.text(7500, 2.3, 'JFC', color='black', fontsize=15, va='bottom')
        if ax_scatter.get_ylim()[0] < 1.5:
            ax_scatter.text(7500, 1.3, 'HTC', color='black', fontsize=15, va='bottom')

    # Axis labels
    ax_scatter.set_xlim(-100, 8300)
    ax_scatter.set_xlabel(r'$\rho$ [kg/m$^3$]', fontsize=20)
    ax_scatter.set_ylabel(r'Tisserand parameter (T$_{j}$)', fontsize=20)
    ax_scatter.tick_params(labelsize=20)
    # display the values on the x and y axes at 0 2000 4000 6000 8000
    ax_scatter.set_xticks(np.arange(0, 9000, 2000))
    ax_scatter.grid(True)

    # Save
    # plt.savefig(os.path.join(output_dir_show, f"{shower_name}_rho_Tj_kc_combined_plot.png"), bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(output_dir_rho, f"{shower_name}_rho_Tj_log10diam_combined_plot.png"), bbox_inches='tight', dpi=300)
    plt.close()

            


    ### plot of JFC HTC AST ###

    print("Creating JFC, HTC, AST plot...")

    # ---------- Helper to always get a 2D axes array ----------
    def _ensure_axes_2d(axes, nrows, ncols):
        """Return axes as a 2D ndarray of shape (nrows, ncols)."""
        if nrows == 1 and ncols == 1:
            axes = np.asarray([[axes]])
        elif nrows == 1:
            axes = np.asarray([axes])
        elif ncols == 1:
            axes = np.asarray([[ax] for ax in axes])
        return axes

    # ---------- Generic grid plotter (rows = cuts, cols = variables) ----------
    def plot_by_cuts_and_vars(vars_list, cuts_list, weights_all,
                            nbins=200, smooth=0.02, figsize=None,
                            bottom_xlabel_per_col=None, tight=True, dpi=300,
                            out_path=None):
        """
        Parameters
        ----------
        vars_list : list of dicts
            Each dict describes one plotted variable (i.e., one column):
            {
            "values": array of shape (N_samples,),      # required
            "label":  r"$\\rho$ (kg/m$^3$)",            # x-axis label for bottom row (optional)
            "name":   "rho",                            # short name for titles/logging (optional)
            "xlim":   (lo, hi),                         # limits, if None computed per variable (optional)
            "color":  "black"                           # fill/line color for this variable (optional)
            }
        cuts_list : list of dicts/tuples
            Each item describes one row (a selection mask and a title prefix):
            Either a tuple: (mask, title_prefix)
            or a dict: {"mask": mask, "title": "Tot N.123 AST"}.
        weights_all : array-like
            Importance weights aligned with vars_list[j]["values"] (N_samples,).
        nbins : int
            Histogram bins.
        smooth : float
            Smoothing parameter passed into your `norm_kde` call (keep as in your code).
        figsize : tuple
            Figure size.
        bottom_xlabel_per_col : list[str] | None
            If provided, length must equal len(vars_list); labels placed on bottom row.
            If None, tries to use vars_list[j]["label"] if present.
        tight : bool
            Use tight bbox when saving.
        dpi : int
            Save resolution.
        out_path : str | None
            If given, save figure there.

        Returns
        -------
        fig, axes : matplotlib Figure and 2D ndarray of Axes (rows=len(cuts_list), cols=len(vars_list))
        """

        # Local wrapper around your existing panel plotter -------------------------
        def _panel_like_top(ax, var_vals, weights, title_prefix, lo, hi, nbins, xlim, var_name="", color_plot="black"):
            # guard
            m = np.isfinite(var_vals)
            if weights is not None:
                m &= np.isfinite(weights)
            if not np.any(m):
                ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                        ha='center', va='center', fontsize=14, color='black')
                _style(ax, xlim)
                return

            r = var_vals[m]
            w = None
            if weights is not None:
                w = weights[m].astype(float)
                s = np.nansum(w)
                w = (w / s) if s > 0 else None

            hist, edges = np.histogram(r, bins=nbins, weights=w, range=(lo, hi))
            hist = norm_kde(hist, 10.0)  # keep your original smoothing kernel span
            bin_centers = 0.5 * (edges[:-1] + edges[1:])

            # Weighted percentiles
            if w is not None:
                q_lo, q_med, q_hi = _quantile(r, [0.025, 0.5, 0.975], weights=w)
            else:
                q_lo, q_med, q_hi = np.nanpercentile(r, [2.5, 50, 97.5])

            ax.fill_between(bin_centers, hist, alpha=0.6, color=color_plot)
            for q in (q_lo, q_med, q_hi):
                ax.axvline(q, linestyle='--', linewidth=1.5, color=color_plot)

            if "log_{10}" in var_name:
                r = 10**(r)
                # delete log_{10} from var_name
                var_name = var_name.replace("$\log_{10}$", "")
                # Weighted percentiles
                if w is not None:
                    q_lo, q_med, q_hi = _quantile(r, [0.025, 0.5, 0.975], weights=w)
                else:
                    q_lo, q_med, q_hi = np.nanpercentile(r, [2.5, 50, 97.5])

            plus  = q_hi - q_med
            minus = q_med - q_lo
            fmt = lambda v: f"{v:.4g}" if np.isfinite(v) else "---"
            if title_prefix == "":
                title = (rf"{var_name} = {fmt(q_med)}"
                        rf"$^{{+{fmt(plus)}}}_{{-{fmt(minus)}}}$")
            else:
                title = (rf"{title_prefix} — {var_name} = {fmt(q_med)}"
                        rf"$^{{+{fmt(plus)}}}_{{-{fmt(minus)}}}$")
            ax.set_title(title, fontsize=16)

            _style(ax, xlim)

        def _style(ax, xlim):
            # check if Nan or Inf in xlim
            if np.all(np.isfinite(xlim)):
                ax.set_xlim(*xlim)
            ax.tick_params(axis='x', labelbottom=False)
            ax.tick_params(axis='y', left=False, labelleft=False)
            ax.set_ylabel("")
            for sp in ['left', 'right', 'top']:
                ax.spines[sp].set_visible(False)

        # Normalize inputs ---------------------------------------------------------
        cuts_norm = []
        for i, item in enumerate(cuts_list):
            if isinstance(item, dict):
                cuts_norm.append((np.asarray(item["mask"], bool), str(item.get("title", ""))))
            else:
                m, t = item
                cuts_norm.append((np.asarray(m, bool), str(t)))
        cuts_list = cuts_norm

        # Figure and axes ----------------------------------------------------------
        nrows = len(cuts_list)
        ncols = len(vars_list)
        if figsize is None:
            figsize = (ncols * 8, nrows * 3)  # (8, 10)
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=False)
        axes = _ensure_axes_2d(axes, nrows, ncols)

        # Default bottom x-labels from vars_list if not provided
        if bottom_xlabel_per_col is None:
            bottom_xlabel_per_col = [
                v.get("label", v.get("name", f"var{j}")) for j, v in enumerate(vars_list)
            ]

        # Per-column global limits (if not provided) -------------------------------
        for j, vinfo in enumerate(vars_list):
            vals = np.asarray(vinfo["values"], float)
            if vinfo.get("xlim") is None:
                lo = float(np.nanmin(vals))
                hi = float(np.nanmax(vals))
                # Avoid identical limits
                if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
                    lo, hi = -1.0, 1.0
                vinfo["xlim"] = (lo, hi)

        # Plot grid ----------------------------------------------------------------
        weights_all = np.asarray(weights_all, float)
        for i, (mask, cut_title) in enumerate(cuts_list):
            for j, vinfo in enumerate(vars_list):
                ax = axes[i, j]
                vals = np.asarray(vinfo["values"], float)
                color = vinfo.get("color", "black")
                xlim  = vinfo["xlim"]
                lo, hi = xlim
                if j!=0:
                    cut_title = ""
                _panel_like_top(
                    ax,
                    vals[mask],
                    weights_all[mask] if np.ndim(weights_all) else None,
                    cut_title,
                    lo, hi, nbins, xlim,
                    var_name=vinfo.get("label", vinfo.get("name", "")),
                    color_plot=color
                )

        # X labels on bottom row only
        for j in range(ncols):
            axes[-1, j].tick_params(axis='x', labelbottom=True)
            axes[-1, j].set_xlabel(bottom_xlabel_per_col[j], fontsize=16)

        # Consistent tick label size
        for ax in axes.ravel():
            ax.tick_params(labelsize=14)

        if out_path:
            plt.savefig(out_path, bbox_inches='tight' if tight else None, dpi=dpi)
            plt.close(fig)

        return fig, axes
    
    event_names_like = all_names

    tj = np.asarray(tj, float)  # per-event Tj (same length as event_names_like)
    if tj.shape[0] != event_names_like.shape[0]:
        raise RuntimeError("Length mismatch: event_names vs tj.")

    # dict: base_name -> Tj
    tj_by_name = {str(n): float(v) for n, v in zip(event_names_like, tj)}

    # Ensure per-sample names of length N_samples
    N_samples = combined_samples.shape[0]
    all_names = np.asarray(all_names)

    if all_names.shape[0] == N_samples:
        names_per_sample = all_names.astype(str)
    elif all_names.shape[0] == len(all_samples):
        # Expand by repeating each event's name by its sample count
        names_per_sample = np.concatenate([
            np.repeat(str(name), arr.shape[0]) for name, arr in zip(all_names, all_samples)
        ])
        if names_per_sample.shape[0] != N_samples:
            raise RuntimeError("Expanded names length does not match combined_samples rows.")
    else:
        raise RuntimeError("all_names must be per-sample or per-event (same length as all_samples).")

    # Map each sample's base_name -> Tj (NaN if missing)
    tj_samples = np.array([tj_by_name.get(n, np.nan) for n in names_per_sample], dtype=float)

    # ---------- Pull rho_corrected samples & weights ----------
    rho_samp = np.asarray(rho_corrected, float)
    if rho_samp.shape[0] != N_samples:
        # Try to locate rho_corrected column in combined_samples via 'variables'
        if 'variables' in globals() and ('rho_corrected' in variables):
            rho_idx = variables.index('rho_corrected')
            rho_samp = combined_samples[:, rho_idx].astype(float)
        else:
            raise RuntimeError("rho_corrected length mismatch and no 'variables' index found.")

    w_all = np.asarray(combined_results.importance_weights(), float)
    w_all = np.where(np.isfinite(w_all), w_all, 0.0)
    if np.nansum(w_all) > 0:
        w_all = w_all / np.nansum(w_all)

    # Global range & plotting params (match your panel)
    smooth = 0.02
    lo_all = float(np.nanmin(rho_samp))
    hi_all = float(np.nanmax(rho_samp))
    nbins = int(round(10.0 / smooth))
    xlim = (-100, 8300)

    # ---------- Helper: a panel identical to your top one ----------
    def _panel_like_top(ax, rho_vals, weights, title_prefix, lo, hi, nbins, xlim, var_name="$\\rho$ [kg/m$^3$]", color_plot='black'):
        # guard
        m = np.isfinite(rho_vals)
        if weights is not None:
            m = m & np.isfinite(weights)
        if not np.any(m):
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                    ha='center', va='center', fontsize=14, color='black')
            _style(ax, xlim)
            return

        r = rho_vals[m]
        w = None
        if weights is not None:
            w = weights[m].astype(float)
            if np.nansum(w) > 0:
                w = w / np.nansum(w)
            else:
                w = None

        hist, edges = np.histogram(r, bins=nbins, weights=w, range=(lo, hi))
        hist = norm_kde(hist, 10.0)
        bin_centers = 0.5 * (edges[:-1] + edges[1:])

        # Weighted percentiles (your _quantile)
        if w is not None:
            q_lo, q_med, q_hi = _quantile(r, [0.025, 0.5, 0.975], weights=w)
        else:
            q_lo, q_med, q_hi = np.nanpercentile(r, [2.5, 50, 97.5])

        # Fill + lines (black)
        ax.fill_between(bin_centers, hist, color=color_plot, alpha=0.6)
        for q in (q_lo, q_med, q_hi):
            ax.axvline(q, color=color_plot, linestyle='--', linewidth=1.5)

        plus  = q_hi - q_med
        minus = q_med - q_lo
        fmt = lambda v: f"{v:.4g}" if np.isfinite(v) else "---"
        title = (rf"{title_prefix} — {var_name} = {fmt(q_med)}"
                rf"$^{{+{fmt(plus)}}}_{{-{fmt(minus)}}}$")
        ax.set_title(title, fontsize=20)

        _style(ax, xlim)

    def _style(ax, xlim):
        ax.set_xlim(*xlim)
        ax.tick_params(axis='x', labelbottom=False)
        ax.tick_params(axis='y', left=False, labelleft=False)
        ax.set_ylabel("")
        for sp in ['left','right','top']: # 'bottom',
            ax.spines[sp].set_visible(False)

    # ---------- Class masks at SAMPLE level ----------
    finite = np.isfinite(rho_samp) & np.isfinite(tj_samples) & np.isfinite(w_all)
    ast_m = finite & (tj_samples >= 3.0)
    jfc_m = finite & (tj_samples >= 2.0) & (tj_samples < 3.0)
    htc_m = finite & (tj_samples < 2.0)

    # find the number of tj above 3
    num_tj_above_3 = tj[tj >= 3].shape[0]
    # find the number of tj in between 2 and 3
    num_tj_between_2_and_3 = tj[(tj >= 2) & (tj < 3)].shape[0]
    # find the number of tj below 2
    num_tj_below_2 = tj[tj < 2].shape[0]

    # ---------- Figure with three stacked panels ----------
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    _panel_like_top(axes[0], rho_samp[ast_m], w_all[ast_m], "Tot N." + str(num_tj_above_3) + " AST", lo_all, hi_all, nbins, xlim)
    _panel_like_top(axes[1], rho_samp[jfc_m], w_all[jfc_m], "Tot N." + str(num_tj_between_2_and_3) + " JFC", lo_all, hi_all, nbins, xlim)
    _panel_like_top(axes[2], rho_samp[htc_m], w_all[htc_m], "Tot N." + str(num_tj_below_2) + " HTC", lo_all, hi_all, nbins, xlim) # "N° " + str(num_tj_below_2) + " HTC"

    # Bottom labels/ticks to match your style
    axes[2].tick_params(axis='x', labelbottom=True)
    axes[2].set_xlabel(r'$\rho$ [kg/m$^3$]', fontsize=20)
    axes[2].set_xticks(np.arange(0, 9000, 2000))
    for ax in axes:
        ax.tick_params(labelsize=20)
    out_path = os.path.join(output_dir_rho, f"{shower_name}_rho_by_Tj_threepanels_weighted.png")
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close()
    print("Saved:", out_path)

    # ### rho distribution plot ###

    # Build group masks (any number of groups works)
    groups = {
        "AST": ast_m,
        "JFC": jfc_m,
        "HTC": htc_m,
        # e.g., add a 4th group later:
        # "IEO": ieo_m,
    }

    tex, results = weighted_tests_table(
        values=rho_samp,
        weights=w_all,
        groups=groups,
        resample_n=8000,                 # bump up for smoother p-values
        random_seed=123,
        caption=r"Pairwise tests on $\rho$ by Tisserand class (weighted posteriors).",
        label="tab:rho_tj_weighted_tests",
        save_path=os.path.join(output_dir_rho, f"{shower_name}_rho_weighted_tests_orbit.tex"),
    )

    # --- Masks (rows) ---
    cuts = [
        (ast_m, f"Tot N.{num_tj_above_3} AST"),
        (jfc_m, f"Tot N.{num_tj_between_2_and_3} JFC"),
        (htc_m, f"Tot N.{num_tj_below_2} HTC"),
    ]

    # --- Column 1: rho ---
    rho_vals = np.asarray(rho_samp, float)
    rho_lo_all, rho_hi_all = float(np.nanmin(rho_vals)), float(np.nanmax(rho_vals))
    rho_spec = {
        "values": rho_vals,
        "label":  r"$\rho$ [kg/m$^3$]",
        "name":   "rho",
        "xlim":   (-100, 8300),   # or (rho_lo, rho_hi)
        "color":  "black",
    }

    vars_to_plot = [rho_spec]
    # --- Column 2: erosion_coeff (if present) ---
    idx_arr = np.where(np.asarray(variables) == "erosion_coeff")[0]
    if idx_arr.size:
        index_eros = int(idx_arr[0])
        eros_vals  = samples[:, index_eros].astype(float)
        # eros_vals  = np.log10(samples[:, index_eros].astype(float))
        eros_lo, eros_hi = float(np.nanmin(eros_vals)), float(np.nanmax(eros_vals))
        eros_spec = {
            "values": eros_vals,
            "label":  r"$\log_{10}$ $\eta$ [kg/MJ]",
            "name":   "erosion_coeff",
            "xlim":   (eros_lo, eros_hi),
            "color":  "blue",
        }
        vars_to_plot = vars_to_plot + [eros_spec]
        
    idx_arr = np.where(np.asarray(variables) == "sigma")[0]
    if idx_arr.size:
        index_sigma = int(idx_arr[0])
        sigma_vals  = samples[:, index_sigma].astype(float)
        sigma_lo, sigma_hi = float(np.nanmin(sigma_vals)), float(np.nanmax(sigma_vals))
        sigma_spec = {
            "values": sigma_vals,
            "label":  r"$\sigma$ [kg/MJ]",
            "name":   "sigma",
            "xlim":   (sigma_lo, sigma_hi),
            "color":  "green",
        }
        vars_to_plot = vars_to_plot + [sigma_spec]


    # add the mass index
    idx_arr = np.where(np.asarray(variables) == "erosion_mass_index")[0]
    if idx_arr.size:
        index_s = int(idx_arr[0])
        s_vals  = samples[:, index_s].astype(float)
        s_lo, s_hi = float(np.nanmin(s_vals)), float(np.nanmax(s_vals))
        s_spec = {
            "values": s_vals,
            "label":  r"$s$",
            "name":   "erosion_mass_index",
            "xlim":   (s_lo, s_hi),
            "color":  "red",
        }
        vars_to_plot = vars_to_plot + [s_spec]


    # add the mass index
    idx_arr = np.where(np.asarray(variables) == "erosion_mass_min")[0]
    if idx_arr.size:
        index_ml = int(idx_arr[0])
        ml_vals  = samples[:, index_ml].astype(float)
        ml_lo, ml_hi = float(np.nanmin(ml_vals)), float(np.nanmax(ml_vals))
        ml_spec = {
            "values": ml_vals,
            "label":   r"$\log_{10}$ $m_{l}$ [kg]",
            "name":   "erosion_mass_min",
            "xlim":   (ml_lo, ml_hi),
            "color":  "purple",
        }
        vars_to_plot = vars_to_plot + [ml_spec]

    # add the mass index
    idx_arr = np.where(np.asarray(variables) == "erosion_mass_max")[0]
    if idx_arr.size:
        index_mu = int(idx_arr[0])
        mu_vals  = samples[:, index_mu].astype(float)
        mu_lo, mu_hi = float(np.nanmin(mu_vals)), float(np.nanmax(mu_vals))
        mu_spec = {
            "values": mu_vals,
            "label":  r"$\log_{10}$ $m_{u}$ [kg]",
            "name":   "erosion_mass_max",
            "xlim":   (mu_lo, mu_hi),
            "color":  "violet",
        }
        vars_to_plot = vars_to_plot + [mu_spec]

    # --- Call the plotter ---
    out_path = os.path.join(output_dir_rho, f"{shower_name}_by_Tj_grid.png")
    fig, axes = plot_by_cuts_and_vars(
        vars_list=vars_to_plot,
        cuts_list=cuts,
        weights_all=w_all,
        nbins=int(round(10.0 / 0.02)),
        smooth=0.02,
        out_path=out_path
    )
    print("Saved:", out_path)
    plt.close(fig)
    

    ### mass change plots ###

    print("Creating mass change plots for rho...")

    m_init_med = np.asarray(m_init_med, float)  # per-event Tj (same length as event_names_like)
    if m_init_med.shape[0] != event_names_like.shape[0]:
        raise RuntimeError("Length mismatch: event_names vs tj.")

    # dict: base_name -> Tj
    m_init_med_by_name = {str(n): float(v) for n, v in zip(event_names_like, m_init_med)}

    # Map each sample's base_name -> Tj (NaN if missing)
    m_init_med_samples = np.array([m_init_med_by_name.get(n, np.nan) for n in names_per_sample], dtype=float)

    # ---------- Class masks at SAMPLE level ----------
    finite = np.isfinite(rho_samp) & np.isfinite(m_init_med_samples) & np.isfinite(w_all)
    big_kg = finite & (m_init_med_samples >= 10**(-4))
    medium_b_kg = finite & (m_init_med_samples >= 5*10**(-5)) & (m_init_med_samples < 10**(-4))
    medium_s_kg = finite & (m_init_med_samples >= 10**(-5)) & (m_init_med_samples < 5*10**(-5))
    small_kg = finite & (m_init_med_samples < 10**(-5))

    # find the number of mass
    num_big_kg = m_init_med[m_init_med >= 10**(-4)].shape[0]
    num_medium_b_kg = m_init_med[(m_init_med >= 5*10**(-5)) & (m_init_med < 10**(-4))].shape[0]
    num_medium_s_kg = m_init_med[(m_init_med >= 10**(-5)) & (m_init_med < 5*10**(-5))].shape[0]
    num_small_kg = m_init_med[m_init_med < 10**(-5)].shape[0]

    # ---------- Figure with three stacked panels ----------
    fig, axes = plt.subplots(4, 1, figsize=(10, 13), sharex=True)

    _panel_like_top(axes[0], rho_samp[big_kg], w_all[big_kg], "Tot N." + str(num_big_kg) + " above 10$^{-4}$ kg", lo_all, hi_all, nbins, xlim)
    _panel_like_top(axes[1], rho_samp[medium_b_kg], w_all[medium_b_kg], "Tot N." + str(num_medium_b_kg) + " 10$^{-4}$ - 5$\cdot$10$^{-5}$ kg", lo_all, hi_all, nbins, xlim)
    _panel_like_top(axes[2], rho_samp[medium_s_kg], w_all[medium_s_kg], "Tot N." + str(num_medium_s_kg) + " 5$\cdot$10$^{-5}$ - 10$^{-5}$ kg", lo_all, hi_all, nbins, xlim)
    _panel_like_top(axes[3], rho_samp[small_kg], w_all[small_kg], "Tot N." + str(num_small_kg) + " below 10$^{-5}$ kg", lo_all, hi_all, nbins, xlim)

    # Bottom labels/ticks to match your style
    axes[3].tick_params(axis='x', labelbottom=True)
    axes[3].set_xlabel(r'$\rho$ [kg/m$^3$]', fontsize=20)
    axes[3].set_xticks(np.arange(0, 9000, 2000))
    for ax in axes:
        ax.tick_params(labelsize=20)

    # Save
    out_path = os.path.join(output_dir_rho, f"{shower_name}_rho_by_mass_threepanels_weighted.png")
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close()
    print("Saved:", out_path)

    # plot scatter
    fig, ax = plt.subplots(figsize=(6, 8), constrained_layout=True)  # larger width, auto spacing

    sc = ax.scatter(rho, log10_m_init, c=np.log10(meteoroid_diameter_mm), cmap='viridis', s=30, norm=Normalize(vmin=np.log10(meteoroid_diameter_mm.min()), vmax=np.log10(meteoroid_diameter_mm.max())), zorder=2)

    plt.errorbar(rho, log10_m_init, 
            xerr=[abs(rho_lo), abs(rho_hi)],
            # yerr=[abs(meteoroid_diameter_mm_lo)/1.96, abs(meteoroid_diameter_mm_hi)/1.96],
            elinewidth=0.75, capthick=0.75,
            fmt='none', ecolor='black', capsize=3, zorder=1)
    
    cbar = fig.colorbar(sc, ax=ax, orientation='vertical', pad=0.08)
    cbar.set_label("$log_{10}$ Diameter [mm]", fontsize=20)
    cbar.ax.tick_params(labelsize=12)
        
    # Guide lines
    for yline in (np.log10(10**(-4)), np.log10(5*10**(-5)), np.log10(10**(-5))):
        plt.axhline(yline, linestyle=':', linewidth=1, alpha=0.5, color='slategray')

    # add the label
    plt.xlabel(r'$\rho$ [kg/m$^3$]', fontsize=20)
    # set x axis lim from (-100, 8300)
    plt.xlim(-100, 8300)
    plt.xticks(np.arange(0, 9000, 2000))
    plt.ylabel('log$_{10}(m_0$ [kg]$)$', fontsize=20)
    plt.tick_params(labelsize=14)
    plt.grid(True, alpha=0.2)

    # save the rho vs diameter plot
    plt.savefig(os.path.join(output_dir_rho, f"{shower_name}_rho_by_mass_scatter.png"), bbox_inches='tight', dpi=300)
    plt.close()

    # ### rho distribution plot ###

    # Build group masks (any number of groups works)
    groups = {
        "above 10$^{-4}$ kg": big_kg,
        "10$^{-4}$ - 5$\cdot$10$^{-5}$ kg": medium_b_kg,
        "5$\cdot$10$^{-5}$ - 10$^{-5}$ kg": medium_s_kg,
        "below 10$^{-5}$ kg": small_kg,
        # e.g., add a 4th group later:
        # "IEO": ieo_m,
    }

    tex, results = weighted_tests_table(
        values=rho_samp,
        weights=w_all,
        groups=groups,
        resample_n=8000,                 # bump up for smoother p-values
        random_seed=123,
        caption=r"Pairwise tests on $\rho$ by mass (weighted posteriors).",
        label="tab:rho_mass_weighted_tests",
        save_path=os.path.join(output_dir_rho, f"{shower_name}_rho_weighted_tests_mass.tex"),
    )


    # mass cuts
    cuts = [
        (big_kg, rf"Tot N." + str(num_big_kg) + " above 10$^{-4}$ kg"),
        (medium_b_kg, rf"Tot N." + str(num_medium_b_kg) + " 10$^{-4}$ - 5$\cdot$10$^{-5}$ kg"),
        (medium_s_kg, rf"Tot N." + str(num_medium_s_kg) + " 5$\cdot$10$^{-5}$ - 10$^{-5}$ kg"),
        (small_kg, rf"Tot N." + str(num_small_kg) + " below 10$^{-5}$ kg"),
    ]

    # --- Call the plotter ---
    out_path = os.path.join(output_dir_rho, f"{shower_name}_by_mass_grid.png")
    fig, axes = plot_by_cuts_and_vars(
        vars_list=vars_to_plot,
        cuts_list=cuts,
        weights_all=w_all,
        nbins=int(round(10.0 / 0.02)),
        smooth=0.02,
        out_path=out_path
    )
    print("Saved:", out_path)
    plt.close(fig)

    ### diameter change plots ###

    print("Creating diameter change plots for rho...")

    # same for meteoroid_diameter_mm
    meteoroid_diameter_mm = np.asarray(meteoroid_diameter_mm, float) 
    if meteoroid_diameter_mm.shape[0] != event_names_like.shape[0]:
        raise RuntimeError("Length mismatch: event_names vs meteoroid_diameter_mm.")

    # dict: base_name -> meteoroid_diameter_mm
    meteoroid_diameter_mm_by_name = {str(n): float(v) for n, v in zip(event_names_like, meteoroid_diameter_mm)}
    # Map each sample's base_name -> meteoroid_diameter_mm (NaN if missing)
    meteoroid_diameter_mm_samples = np.array([meteoroid_diameter_mm_by_name.get(n, np.nan) for n in names_per_sample], dtype=float)

    # ---------- Class masks at SAMPLE level ----------
    finite = np.isfinite(rho_samp) & np.isfinite(meteoroid_diameter_mm_samples) & np.isfinite(w_all)
    big = finite & (meteoroid_diameter_mm_samples >= 7.5)
    medium_b = finite & (meteoroid_diameter_mm_samples >= 5) & (meteoroid_diameter_mm_samples < 7.5)
    medium_s = finite & (meteoroid_diameter_mm_samples >= 2.5) & (meteoroid_diameter_mm_samples < 5)
    small = finite & (meteoroid_diameter_mm_samples < 2.5)

    # find the number of mass
    num_big = meteoroid_diameter_mm[meteoroid_diameter_mm >= 7.5].shape[0]
    num_medium_b = meteoroid_diameter_mm[(meteoroid_diameter_mm >= 5) & (meteoroid_diameter_mm < 7.5)].shape[0]
    num_medium_s = meteoroid_diameter_mm[(meteoroid_diameter_mm >= 2.5) & (meteoroid_diameter_mm < 5)].shape[0]
    num_small = meteoroid_diameter_mm[meteoroid_diameter_mm < 2.5].shape[0]

    # ---------- Figure with three stacked panels ----------
    fig, axes = plt.subplots(4, 1, figsize=(10, 13), sharex=True)

    _panel_like_top(axes[0], rho_samp[big], w_all[big], "Tot N." + str(num_big) + " above 7.5 mm", lo_all, hi_all, nbins, xlim)
    _panel_like_top(axes[1], rho_samp[medium_b], w_all[medium_b], "Tot N." + str(num_medium_b) + " 5 - 7.5 mm", lo_all, hi_all, nbins, xlim)
    _panel_like_top(axes[2], rho_samp[medium_s], w_all[medium_s], "Tot N." + str(num_medium_s) + " 2.5 - 5 mm", lo_all, hi_all, nbins, xlim)
    _panel_like_top(axes[3], rho_samp[small], w_all[small], "Tot N." + str(num_small) + " below 2.5 mm", lo_all, hi_all, nbins, xlim)

    # Bottom labels/ticks to match your style
    axes[3].tick_params(axis='x', labelbottom=True)
    axes[3].set_xlabel(r'$\rho$ [kg/m$^3$]', fontsize=20)
    axes[3].set_xticks(np.arange(0, 9000, 2000))
    for ax in axes:
        ax.tick_params(labelsize=20)

    # Save
    out_path = os.path.join(output_dir_rho, f"{shower_name}_rho_by_diameter_threepanels_weighted.png")
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close()
    print("Saved:", out_path)

    ### Create scatter plot for rho vs diameter ###

    fig, ax = plt.subplots(figsize=(6, 8), constrained_layout=True)  # larger width, auto spacing
    # gs = GridSpec(nrows=4, ncols=2, width_ratios=[1.3, 1.0], hspace=0.25, wspace=0.5, figure=fig)

    plt.errorbar(rho, meteoroid_diameter_mm, 
                xerr=[abs(rho_lo), abs(rho_hi)],
                yerr=[abs(meteoroid_diameter_mm_lo), abs(meteoroid_diameter_mm_hi)],
                elinewidth=0.75, capthick=0.75,
                fmt='none', ecolor='black', capsize=3, zorder=1)

    sc = plt.scatter(rho, meteoroid_diameter_mm,
                    c=log10_m_init, cmap='viridis',
                    norm=Normalize(vmin=log10_m_init.min(), vmax=log10_m_init.max()),
                    s=30, zorder=2)

    # Guide lines
    for yline in (2.5, 5.0, 7.5):
        plt.axhline(yline, linestyle=':', linewidth=1, alpha=0.5, color='slategray')

    plt.xlabel(r'$\rho$ [kg/m$^3$]', fontsize=20)
    # set x axis lim from (-100, 8300)
    plt.xlim(-100, 8300)
    plt.xticks(np.arange(0, 9000, 2000))
    plt.ylabel('Meteoroid diameter [mm]', fontsize=20)
    plt.tick_params(labelsize=14)
    plt.grid(True, alpha=0.2)

    # Colorbar on the right of scatter only
    cbar = fig.colorbar(sc, ax=ax, orientation='vertical', pad=0.08)
    cbar.set_label(r'$\log_{10}(m_{0}$ [kg]$)$', fontsize=20)
    cbar.ax.tick_params(labelsize=12)

    # save the rho vs diameter plot
    plt.savefig(os.path.join(output_dir_rho, f"{shower_name}_rho_by_diameter_scatter.png"), bbox_inches='tight', dpi=300)
    plt.close()

    # ### rho distribution plot ###

    # Build group masks (any number of groups works)
    groups = {
        "above 7.5 mm": big,
        "5 - 7.5 mm": medium_b,
        "2.5 - 5 mm": medium_s,
        "below 2.5 mm": small,
    }

    tex, results = weighted_tests_table(
        values=rho_samp,
        weights=w_all,
        groups=groups,
        resample_n=8000,                 # bump up for smoother p-values
        random_seed=123,
        caption=r"Pairwise tests on $\rho$ by diameter class (weighted posteriors).",
        label="tab:rho_diameter_weighted_tests",
        save_path=os.path.join(output_dir_rho, f"{shower_name}_rho_weighted_tests_diameter.tex"),
    )

    # print(tex)  # also written to file if save_path was given

    cuts = [
        (big, f"Tot N.{num_big} above 7.5 mm"),
        (medium_b, f"Tot N.{num_medium_b} 5 - 7.5 mm"),
        (medium_s, f"Tot N.{num_medium_s} 2.5 - 5 mm"),
        (small, f"Tot N.{num_small} below 2.5 mm"),
    ]

    # --- Call the plotter ---
    out_path = os.path.join(output_dir_rho, f"{shower_name}_by_diameter_grid.png")
    fig, axes = plot_by_cuts_and_vars(
        vars_list=vars_to_plot,
        cuts_list=cuts,
        weights_all=w_all,
        nbins=int(round(10.0 / 0.02)),
        smooth=0.02,
        out_path=out_path
    )
    print("Saved:", out_path)
    plt.close(fig)


    ### dyn press change plots ###

    print("Dyn press change plots for rho...")

    # --- Dynamic pressure two-class overlay (NO gradient) ---
    logq = np.log10(erosion_beg_dyn_press)
    # thr = 3.2  # log10 Pa  (≈ 1.58 kPa)

    # mask_hi = logq >= thr
    # mask_lo = ~mask_hi

    # same for meteoroid_diameter_mm
    logq = np.asarray(logq, float) 
    if logq.shape[0] != event_names_like.shape[0]:
        raise RuntimeError("Length mismatch: event_names vs meteoroid_diameter_mm.")

    # dict: base_name -> meteoroid_diameter_mm
    logq_by_name = {str(n): float(v) for n, v in zip(event_names_like, logq)}
    # Map each sample's base_name -> meteoroid_diameter_mm (NaN if missing)
    logq_samples = np.array([logq_by_name.get(n, np.nan) for n in names_per_sample], dtype=float)

    # ---------- Class masks at SAMPLE level ----------
    finite = np.isfinite(rho_samp) & np.isfinite(logq_samples) & np.isfinite(w_all)
    sturdy = finite & (logq_samples >= thr)
    fragile = finite & (logq_samples < thr)

    # find the number of mass
    num_sturdy = logq[logq >= thr].shape[0]
    num_fragile = logq[logq < thr].shape[0]

    # ---------- Figure with three stacked panels ----------
    fig, axes = plt.subplots(2, 1, figsize=(10, 13), sharex=True)

    _panel_like_top(axes[0], rho_samp[fragile], w_all[fragile], "Tot N." + str(num_fragile) + " below " + str(np.round(10**thr))+" Pa", lo_all, hi_all, nbins, xlim)
    _panel_like_top(axes[1], rho_samp[sturdy], w_all[sturdy], "Tot N." + str(num_sturdy) + " above " + str(np.round(10**thr))+" Pa", lo_all, hi_all, nbins, xlim)

    # Bottom labels/ticks to match your style
    axes[1].tick_params(axis='x', labelbottom=True)
    axes[1].set_xlabel(r'$\rho$ [kg/m$^3$]', fontsize=20)
    axes[1].set_xticks(np.arange(0, 9000, 2000))
    for ax in axes:
        ax.tick_params(labelsize=20)

    # Save
    out_path = os.path.join(output_dir_rho, f"{shower_name}_rho_by_dynpres_threepanels_weighted.png")
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close()
    print("Saved:", out_path)


    # ### rho distribution plot ###

    # Build group masks (any number of groups works)
    groups = {
        "below " + str(10**thr)+" Pa": fragile,
        "above " + str(10**thr)+" Pa": sturdy
    }

    tex, results = weighted_tests_table(
        values=rho_samp,
        weights=w_all,
        groups=groups,
        resample_n=8000,                 # bump up for smoother p-values
        random_seed=123,
        caption=r"Pairwise tests on $\rho$ by dynamic pressure class (weighted posteriors).",
        label="tab:rho_dynpres_weighted_tests",
        save_path=os.path.join(output_dir_rho, f"{shower_name}_rho_weighted_tests_dynpres.tex"),
    )

    # print(tex)  # also written to file if save_path was given

    cuts = [
        (fragile, f"Tot N.{num_fragile} below {np.round(10**thr)} Pa"),
        (sturdy, f"Tot N.{num_sturdy} above {np.round(10**thr)} Pa"),
    ]

    # --- Call the plotter ---
    out_path = os.path.join(output_dir_rho, f"{shower_name}_by_dynampres_grid.png")
    fig, axes = plot_by_cuts_and_vars(
        vars_list=vars_to_plot,
        cuts_list=cuts,
        weights_all=w_all,
        nbins=int(round(10.0 / 0.02)),
        smooth=0.02,
        out_path=out_path
    )
    print("Saved:", out_path)
    plt.close(fig)

    # Plot grid settings
    ndim = samples.shape[1]
    # ncols = 5
    # nrows = math.ceil(ndim / ncols)
    nrows = 5
    ncols = math.ceil(ndim / nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 2.5 * nrows))
    axes = axes.flatten()

    # Define smoothing value
    smooth = 0.02  # or pass it as argument

    for i, variable in enumerate(variables):
        ax = axes[i]
        x = samples[:, i].astype(float)
        mask = ~np.isnan(x)
        x_valid = x[mask]
        w_valid = w[mask]

        if x_valid.size == 0:
            ax.axis('off')
            continue

        # Compute histogram
        lo, hi = np.min(x_valid), np.max(x_valid)
        if isinstance(smooth, int):
            hist, edges = np.histogram(x_valid, bins=smooth, weights=w_valid, range=(lo, hi))
        else:
            nbins = int(round(10. / smooth))
            hist, edges = np.histogram(x_valid, bins=nbins, weights=w_valid, range=(lo, hi))
            hist = norm_kde(hist, 10.0)  # dynesty-style smoothing

        centers = 0.5 * (edges[1:] + edges[:-1])

        # Fill under the curve
        ax.fill_between(centers, hist, color='blue', alpha=0.6)

        # ax.plot(centers, hist, color='blue')
        ax.set_yticks([])
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))

        # Set label + quantile title
        row = summary_df.iloc[i]
        label = row["Label"]
        median = row["Median"]
        low = row["Low95"]
        high = row["High95"]
        minus = median - low
        plus = high - median

        if variables[i] in ['erosion_mass_min', 'erosion_mass_max','m_init','erosion_coeff', 'erosion_coeff_change']: # 'log' in flags_dict_total.get(variables[i], '') and 
            # put a dashed blue line at the median
            ax.axvline(np.log10(median), color='blue', linestyle='--', linewidth=1.5)
            # put a dashed Blue line at the 2.5 and 97.5 percentiles
            ax.axvline(np.log10(low), color='blue', linestyle='--', linewidth=1.5)
            ax.axvline(np.log10(high), color='blue', linestyle='--', linewidth=1.5)
            
        else:
            # put a dashed blue line at the median
            ax.axvline(median, color='blue', linestyle='--', linewidth=1.5)
            # put a dashed Blue line at the 2.5 and 97.5 percentiles
            ax.axvline(low, color='blue', linestyle='--', linewidth=1.5)
            ax.axvline(high, color='blue', linestyle='--', linewidth=1.5)

        fmt = lambda v: f"{v:.4g}" if np.isfinite(v) else "---"
        title = rf"{label} = {fmt(median)}$^{{+{fmt(plus)}}}_{{-{fmt(minus)}}}$"
        ax.set_title(title, fontsize=16)
        ax.set_xlabel(labels_plot_copy_plot[i], fontsize=20)
        # increase the size of the tick labels
        ax.tick_params(axis='x', labelsize=15)

    # Remove unused axes
    for j in range(ndim, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(output_dir_show, f"{shower_name}_distrib_plot.png"),
                bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    

    ### CORNER PLOT ###
    # takes forever, so run it last
    if plot_correl_flag == True:

        combined_samples_cov_plot = combined_samples.copy()
        labels_plot_copy_plot = labels.copy()
        for j, var in enumerate(variables):
            if np.all(np.isnan(combined_samples_cov_plot[:, j])):
                continue
            if var in ['v_init', 'erosion_height_start', 'erosion_height_change']:
                combined_samples_cov_plot[:, j] = combined_samples_cov_plot[:, j] / 1000.0
            if var in ['sigma', 'erosion_sigma_change','erosion_coeff', 'erosion_coeff_change']:
                combined_samples_cov_plot[:, j] = combined_samples_cov_plot[:, j] * 1e6

        print('saving corner plot...')
        combined_results_units = CombinedResults(combined_samples_cov_plot, combined_weights)

        # Ensure output folder
        cov_dir = os.path.join(output_dir_show, "Covariance")
        os.makedirs(cov_dir, exist_ok=True)

        # for i, variable in enumerate(variables):
        #     if 'log' in flags_dict_total[variable]:  
        #         labels_plot[i] =r"$\log_{10}$(" +labels_plot[i]+")"

        # Define weighted correlation
        def weighted_corr(x, y, w):
            """Weighted Pearson correlation of x and y with weights w."""
            w = np.asarray(w)
            x = np.asarray(x)
            y = np.asarray(y)
            w_sum = w.sum()
            x_mean = (w * x).sum() / w_sum
            y_mean = (w * y).sum() / w_sum
            cov_xy = (w * (x - x_mean) * (y - y_mean)).sum() / w_sum
            var_x  = (w * (x - x_mean)**2).sum() / w_sum
            var_y  = (w * (y - y_mean)**2).sum() / w_sum
            return cov_xy / np.sqrt(var_x * var_y)

        # … your existing prep code …
        fig, axes = plt.subplots(ndim, ndim, figsize=(35, 15))
        axes = axes.reshape((ndim, ndim))

        # call dynesty’s cornerplot
        fg, ax = dyplot.cornerplot(
            combined_results_units, 
            color='blue',
            show_titles=True,
            max_n_ticks=3,
            quantiles=None,
            labels=labels,
            label_kwargs={"fontsize": 10},
            title_kwargs={"fontsize": 12},
            title_fmt='.2e',
            fig=(fig, axes[:, :ndim])
        )

        # # supertitle, tick formatting, saving …
        # fg.suptitle(shower_name, fontsize=16, fontweight='bold')

        for ax_row in ax:
            for ax_ in ax_row:
                if ax_ is None:
                    continue
                ax_.tick_params(axis='both', labelsize=8, direction='in')
                for lbl in ax_.get_xticklabels(): lbl.set_rotation(0)
                for lbl in ax_.get_yticklabels(): lbl.set_rotation(45)
                if len(ax_.xaxis.get_majorticklocs())>0:
                    ax_.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.4g'))
                if len(ax_.yaxis.get_majorticklocs())>0:
                    ax_.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4g'))

        for i in range(ndim):
            for j in range(ndim):
                if ax[i, j] is None:
                    continue
                if j != 0:
                    ax[i, j].set_yticklabels([])
                if i != ndim - 1:
                    ax[i, j].set_xticklabels([])

        # Overlay weighted correlations in the upper triangle
        samples = combined_results_units['samples'].T  # shape (ndim, nsamps)
        weights = combined_results_units.importance_weights()

        cmap = plt.colormaps['coolwarm']
        norm = Normalize(vmin=-1, vmax=1)

        for i in range(ndim):
            for j in range(ndim):
                if j <= i or ax[i, j] is None:
                    continue

                panel = ax[i, j]
                x = samples[j]
                y = samples[i]
                corr_w = weighted_corr(x, y, weights)

                color = cmap(norm(corr_w))
                # paint the background patch
                panel.patch.set_facecolor(color)
                panel.patch.set_alpha(1.0)

                # fallback rectangle if needed
                panel.add_patch(
                    plt.Rectangle(
                        (0,0), 1, 1,
                        transform=panel.transAxes,
                        facecolor=color,
                        zorder=0
                    )
                )

                panel.text(
                    0.5, 0.5,
                    f"{corr_w:.2f}",
                    transform=panel.transAxes,
                    ha='center', va='center',
                    fontsize=25, color='black'
                )
                panel.set_xticks([]); panel.set_yticks([])
                for spine in panel.spines.values():
                    spine.set_visible(False)

        # final adjustments & save
        # fg.subplots_adjust(wspace=0.1, hspace=0.3)
        fg.subplots_adjust(wspace=0.1, hspace=0.3, top=0.978) # Increase spacing between plots
        plt.savefig(os.path.join(cov_dir, f"{shower_name}_correlation_plot.png"),
                    bbox_inches='tight', dpi=300)
        plt.close(fig)

        print('saving correlation matrix...')

        # Build the NxN matrix of weigh_corr_ij
        corr_mat = np.zeros((ndim, ndim))
        for i in range(ndim):
            for j in range(ndim):
                corr_mat[i, j] = weighted_corr(samples[i], samples[j], weights)

        # Wrap it in a DataFrame (so you get row/column labels)
        df_corr = pd.DataFrame(
            corr_mat,
            index=labels,
            columns=labels
        )

        def _sanitize_labels(labels, ndim):
            # coerce to strings, strip, and fill blanks
            labels = [str(x).strip() if str(x).strip() else f"p{i}" for i, x in enumerate(labels)]
            # enforce uniqueness by suffixing duplicates
            seen = {}
            out = []
            for lbl in labels:
                if lbl not in seen:
                    seen[lbl] = 1
                    out.append(lbl)
                else:
                    seen[lbl] += 1
                    out.append(f"{lbl} ({seen[lbl]})")
            # pad/truncate to match ndim if needed
            if len(out) < ndim:
                out.extend([f"p{i}" for i in range(len(out), ndim)])
            return out[:ndim]

        labels_clean = _sanitize_labels(labels, ndim)

        df_corr = pd.DataFrame(corr_mat, index=labels_clean, columns=labels_clean)

        # Save the full matrix
        outpath = os.path.join(cov_dir, f"{shower_name}_weighted_globalCorr_matrix.csv")
        df_corr.to_csv(outpath, float_format="%.4f")
        print(f"Saved weighted correlation matrix to:\n  {outpath}")

        # Upper-triangle only (exclude diagonal)
        mask = np.triu(np.ones(df_corr.shape, dtype=bool), k=1)
        upper = df_corr.where(mask)

        # Turn into a *DataFrame* so both columns always print (no MultiIndex collapse)
        pairs_df = (
            upper
            .stack()                      # drop NaNs from the lower triangle
            .rename("corr")
            .reset_index()
            .rename(columns={"level_0": "param_i", "level_1": "param_j"})
        )

        # Sort by absolute correlation and print clean tables
        pairs_df["abs_corr"] = pairs_df["corr"].abs()

        top10 = pairs_df.sort_values("abs_corr", ascending=False).head(10)
        bottom10 = pairs_df.sort_values("abs_corr", ascending=True).head(10)

        pd.set_option("display.max_colwidth", None)
        print("\nTop 10: highest correlations:")
        print(top10[["param_i", "param_j", "corr"]].to_string(index=False))

        print("\nBottom 10: lowest correlations:")
        print(bottom10[["param_i", "param_j", "corr"]].to_string(index=False))


        def _HIST2D(x, y, **kwargs):
            """Wrapper that calls dynesty's _hist2d wherever it lives."""
            # if dyplot has it as a private attr, prefer that (matches your cornerplot call site)
            if hasattr(dyplot, "_hist2d"):
                return dyplot._hist2d(x, y, **kwargs)
            if _hist2d_func is not None:
                return _hist2d_func(x, y, **kwargs)
            raise RuntimeError("Could not locate dynesty.plotting._hist2d. "
                            "Ensure dynesty>=2.x is installed or import _hist2d yourself.")

        # --- utils ---
        def _safe_name(s):
            s = re.sub(r"\$|\\[a-zA-Z]+|[\{\}\^\_]", "", str(s))
            s = re.sub(r"[^A-Za-z0-9\-\.]+", "_", s).strip("_")
            return s or "param"

        def _weighted_quantile(x, q, w):
            x = np.asarray(x); w = np.asarray(w)
            m = np.isfinite(x) & np.isfinite(w) & (w >= 0)
            x = x[m]; w = w[m]
            if x.size == 0:
                return np.nan if np.isscalar(q) else [np.nan]*len(q)
            order = np.argsort(x)
            x = x[order]; w = w[order]
            cdf = np.cumsum(w)
            cdf = (cdf - 0.5*w[order]) / np.sum(w)
            def interp(qi):
                if qi <= 0: return x[0]
                if qi >= 1: return x[-1]
                return np.interp(qi, cdf, x)
            if np.isscalar(q):
                return interp(q)
            return [interp(qi) for qi in q]

        def plot_top_covariances(results,
                                top_pairs,            # DataFrame with columns param_i,param_j,corr OR list of tuples
                                labels,          # labels in the SAME order as results['samples'] columns
                                shower_name_plot,
                                output_dir_show,
                                span_frac=0.98,
                                pad_frac=0.05,
                                smooth_frac=0.02,
                                levels=None,
                                figsize=(4.5, 4.0)):

            # Extract weighted samples EXACTLY like your cornerplot
            samples = results['samples']
            weights = results.importance_weights()
            samples = np.atleast_1d(samples)
            if len(samples.shape) == 1:
                samples = np.atleast_2d(samples)
            else:
                assert len(samples.shape) == 2, "Samples must be 1- or 2-D."
                samples = samples.T
            assert samples.shape[0] <= samples.shape[1], "More dimensions than samples!"
            ndim, nsamps = samples.shape
            assert weights.ndim == 1 and weights.shape[0] == nsamps, "Weights/samples mismatch."

            # Map label -> dimension index (use the SAME labels you used for df_corr)
            label_to_idx = {}
            for i, lbl in enumerate(labels):
                s = str(lbl)
                # first occurrence wins; if duplicates exist, later ones get suffix in labels_clean
                if s not in label_to_idx:
                    label_to_idx[s] = i

            # Ensure output folder
            cov_dir = os.path.join(output_dir_show, "Covariance")
            os.makedirs(cov_dir, exist_ok=True)

            # Normalize input to a list of tuples
            if hasattr(top_pairs, "iterrows"):  # pairs_df
                iterable = [(r["param_i"], r["param_j"], r["corr"]) for _, r in top_pairs.iterrows()]
            else:
                iterable = list(top_pairs)

            if levels is None:
                levels = [0.1, 0.4, 0.65, 0.85]

            for rank, (li, lj, corr_val) in enumerate(iterable, start=1):
                # if a label isn't found due to prior duplication/cleaning, fail loudly with context
                if str(li) not in label_to_idx or str(lj) not in label_to_idx:
                    raise KeyError(f"Label not found in labels: {li} or {lj}. "
                                f"Pass labels_clean here (the ones used in df_corr).")

                si = label_to_idx[str(li)]   # y-axis
                sj = label_to_idx[str(lj)]   # x-axis

                x = samples[sj]
                y = samples[si]
                w = weights

                qlo = (1.0 - span_frac)/2.0
                qhi = 1.0 - qlo
                xlo, xhi = _weighted_quantile(x, [qlo, qhi], w)
                ylo, yhi = _weighted_quantile(y, [qlo, qhi], w)

                xr = (xhi - xlo) or 1.0
                yr = (yhi - ylo) or 1.0
                xspan = [xlo - pad_frac*xr, xhi + pad_frac*xr]
                yspan = [ylo - pad_frac*yr, yhi + pad_frac*yr]

                sx = smooth_frac
                sy = smooth_frac

                fig, ax = plt.subplots(figsize=figsize)
                _HIST2D(x, y,
                        ax=ax,
                        span=[xspan, yspan],
                        weights=w,
                        color='black',
                        smooth=[sx, sy],
                        levels=levels,
                        fill_contours=True,
                        plot_contours=True)

                ax.set_xlim(xspan); ax.set_ylim(yspan)
                ax.xaxis.set_major_locator(MaxNLocator(5, prune="lower"))
                ax.yaxis.set_major_locator(MaxNLocator(5, prune="lower"))
                ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=False))
                ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
                ax.set_xlabel(str(lj))
                ax.set_ylabel(str(li))
                ax.set_title(f"{shower_name_plot} corr = {(float(corr_val)):.3f}", fontsize=10)

                fname = f"{rank:02d}_{_safe_name(lj)}__{_safe_name(li)}.png"
                fpath = os.path.join(cov_dir, fname)
                fig.tight_layout()
                fig.savefig(fpath, dpi=200)
                plt.close(fig)

            print(f"Saved {len(iterable)} covariance plots in:\n  {cov_dir}")

        if "ORI" in shower_name: # special case for ORI to avoid overly long names
            shower_name_short = "ORI"
        elif "CAP" in shower_name: # special case for GEM to avoid overly long names
            shower_name_short = "CAP"
        elif "DRA" in shower_name: # special case for DRA to avoid overly long names
            shower_name_short = "DRA"
        else:
            shower_name_short = ""

        print("Plotting top covariances 1vs1...")
        # Top-50 by absolute correlation (as you already computed)
        top50_df = pairs_df.sort_values("abs_corr", ascending=False).head(50)

        plot_top_covariances(
            results=combined_results_units,   # <<< was `results=`; use your combined_results object
            top_pairs=top50_df,
            shower_name_plot=shower_name_short,
            labels=labels_clean,   # <<< use the SAME cleaned labels used for df_corr
            output_dir_show=output_dir_show,
            span_frac=0.98,
            pad_frac=0.05,
            smooth_frac=0.02
        )




if __name__ == "__main__":

    import argparse
    ### COMMAND LINE ARGUMENTS
    arg_parser = argparse.ArgumentParser(description="Run dynesty with optional .prior file.")
    
    arg_parser.add_argument('--input_dir', metavar='INPUT_PATH', type=str,
                            
         default=r"C:\Users\maxiv\Documents\UWO\Papers\2)ORI-CAP-PER-DRA\Results\CAMO+EMCCD\CAP_radiance_new",
        help="Path to walk and find .pickle files.")
    
    arg_parser.add_argument('--output_dir', metavar='OUTPUT_DIR', type=str,
        default=r"",
        help="Output directory, if not given is the same as input_dir.")
    
    arg_parser.add_argument('--name', metavar='NAME', type=str,
        default=r"",
        help="Name of the input files, if not given is folders name.")

    arg_parser.add_argument('--radiance_plot', action='store_true',
        help="Flag to enable radiance plot.")

    arg_parser.add_argument('--correlation_plot', action='store_true',
        help="Flag to enable correlation plot.")

    # Parse
    cml_args = arg_parser.parse_args()

    # check if the input_dir exists
    if not os.path.exists(cml_args.input_dir):
        raise FileNotFoundError(f"Input directory {cml_args.input_dir} does not exist.")

    # check if cml_args.output_dir is empty and set it to the input_dir
    if cml_args.output_dir == "":
        cml_args.output_dir = cml_args.input_dir
    # check if the output_dir exists and create it if not
    if not os.path.exists(cml_args.output_dir):
        os.makedirs(cml_args.output_dir)

    # if name is empty set it to the input_dir
    if cml_args.name == "":
        # split base on the os.sep() and get the last element
        cml_args.name = cml_args.input_dir.split(os.sep)[-1]
        print(f"Setting name to {cml_args.name}")

    shower_distrb_plot(cml_args.input_dir, cml_args.output_dir, cml_args.name, radiance_plot_flag=True, plot_correl_flag=True) # cml_args.radiance_plot cml_args.correl_plot
