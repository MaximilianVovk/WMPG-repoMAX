"""
Import all the pickle files and get the dynesty files distribution

Author: Maximilian Vovk
Date: 2025-04-16
"""

# main.py (inside my_subfolder)
import sys
import os

from matplotlib.lines import Line2D
from nrlmsise00 import msise_model
import numpy as np


# import from Mars_AtmDens.py
from Mars_AtmDens import fitAtmPoly_mars
from Mars_Vel import calculate_3d_intercept_speeds

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
from matplotlib.colors import LinearSegmentedColormap
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
from wmpl.Utils.AtmosphereDensity import fitAtmPoly, atmDensPoly
from types import SimpleNamespace

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

# SUN constants
MU_SUN = 1.32712440018e11 # km^3/s^2
# Mars constants
MU_MARS   = 4.282837e4    # km^3/s^2
A_MARS_AU = 1.523679      # mean heliocentric distance (assume circular)
R_MARS    = 3389.5        # km
# Earth constants
MU_EARTH = 3.986004418e5  # km^3/s^2
A_EARTH_AU = 1.00000261    # mean heliocentric distance (assume circular)
R_EARTH = 6371.0          # km

AU_KM = 149_597_870.7     # km

def escape_speed_kms(mu_p=MU_MARS, r_p_km=R_MARS, h_km=180):
    """Planetary escape speed at altitude h (km/s)."""
    return math.sqrt(2.0 * mu_p / (r_p_km + h_km))

def vis_viva_speed_kms(a_au, r_au):
    """Heliocentric speed at distance r from Sun via vis-viva (km/s)."""
    a_km = a_au * AU_KM
    r_km = r_au * AU_KM
    # check if MU_SUN * (2.0/r_km - 1.0/a_km) is negative
    if MU_SUN * (2.0/r_km - 1.0/a_km) < 0:
        return -1
    else:
        return math.sqrt(MU_SUN * (2.0/r_km - 1.0/a_km))

def mars_orbital_speed_kms(a_mars_au=A_MARS_AU):
    """Assume circular Mars orbit."""
    r = a_mars_au * AU_KM
    return math.sqrt(MU_SUN / r)

V_ESC_MARS180 = escape_speed_kms(MU_MARS, R_MARS, 140) # same bulk density as Earth at 180 km
V_ESC_EARTH180 = escape_speed_kms(MU_EARTH, R_EARTH, 180)

V_ORBIT_MARS = mars_orbital_speed_kms(A_MARS_AU)
V_ORBIT_EARTH = mars_orbital_speed_kms(A_EARTH_AU)

V_PARAB_EARTHORBIT = escape_speed_kms(MU_SUN, A_EARTH_AU*AU_KM, 0)
V_PARAB_MARSORBIT = escape_speed_kms(MU_SUN, A_MARS_AU*AU_KM, 0)


def run_single_eeu(sim_num_and_data):
    sim_num, tot_sim, sample, obs_data, variables, fixed_values, flags_dict = sim_num_and_data

    # print(f"Running simulation {sim_num}/{tot_sim}")
    
    # Copy and transform the sample as in your loop
    guess = sample.copy()
    flag_total_rho = False

    for i, variable in enumerate(variables):
        if 'log' in flags_dict[variable]:
            guess[i] = 10**guess[i]

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

    except Exception as e:
        print(f"Simulation {sim_num} failed: {e}")
        eeucs, eeum = np.nan, np.nan

    const_nominal.erosion_height_start = obs_data.height_lum[-1] # calculate the erosion energy until the last height
    const_nominal.v_init = np.mean(obs_data.velocities) # calculate the erosion energy until using the mean velocity

    # Extract physical quantities
    try:
        eeucs_end, eeum_end = energyReceivedBeforeErosion(const_nominal)

        return (sim_num, eeucs, eeum, eeucs_end, eeum_end)

    except Exception as e:
        print(f"Simulation end {sim_num} failed: {e}")
        return (sim_num, eeucs, eeum, np.nan, np.nan)


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
    inclin_val = Q_val = q_val = a_val = e_val = Vinf_val = Vg_val = peri_val = node_val = None
    
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

    re_Vg_val = re.compile(
        r'^\s*Vg\s*=\s*'                           
        r'([+-]?\d+\.\d+)'                         
    )

    re_peri_val = re.compile(
        r'^\s*peri\s*=\s*'                           
        r'([+-]?\d+\.\d+)'                         
    )

    re_node_val = re.compile(
        r'^\s*node\s*=\s*'
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
            if Vg_val is None:
                m = re_Vg_val.match(line)
                if m:
                    Vg_val = float(m.group(1))
            if peri_val is None:
                m = re_peri_val.match(line)
                if m:
                    peri_val = float(m.group(1))
            if node_val is None:
                m = re_node_val.match(line)
                if m:
                    node_val = float(m.group(1))

            if Tj is not None and inclin_val is not None and Vinf_val is not None and Q_val is not None and q_val is not None and a_val is not None and e_val is not None and Vg_val is not None and peri_val is not None and node_val is not None:
                break


    if Tj is None:
        raise RuntimeError(f"Couldn’t find any Tj line in {report_path!r}")
    if inclin_val is None:
        raise RuntimeError(f"Couldn’t find inclination (i) in {report_path!r}")
    if Vinf_val is None:
        raise RuntimeError(f"Couldn’t find Vinf in {report_path!r}")
    if Vg_val is None:
        raise RuntimeError(f"Couldn’t find Vg in {report_path!r}")
    if peri_val is None:
        raise RuntimeError(f"Couldn’t find peri in {report_path!r}")
    if node_val is None:
        raise RuntimeError(f"Couldn’t find node in {report_path!r}")

    print(f"Tj = {Tj:.6f} 95% CI = [{Tj_low:.6f}, {Tj_high:.6f}]")
    Tj_low = (Tj - Tj_low)#/1.96
    Tj_high = (Tj_high - Tj)#/1.96
    print(f"Vinf = {Vinf_val:.6f} km/s")
    print(f"Vg = {Vg_val:.6f} km/s")
    print(f"a = {a_val:.6f} AU")
    print(f"e = {e_val:.6f}")
    print(f"i = {inclin_val:.6f} deg")
    print(f"peri = {peri_val:.6f} deg")
    print(f"node = {node_val:.6f} deg")
    print(f"Q = {Q_val:.6f} AU")
    print(f"q = {q_val:.6f} AU")

    return Tj, Tj_low, Tj_high, inclin_val, Vinf_val, Vg_val, Q_val, q_val, a_val, e_val, peri_val, node_val


def shower_distrb_plot(input_dirfile, output_dir_show, shower_name):
    """
    Function to plot the distribution of the parameters from the dynesty files and save them as a table in LaTeX format.
    """
    # Use the class to find .dynesty, load prior, and decide output folders
    finder = find_dynestyfile_and_priors(input_dir_or_file=input_dirfile,prior_file="",resume=True,output_dir=input_dirfile,use_all_cameras=True,pick_position=0)

    all_label_sets = []  # List to store sets of labels for each file
    variables = []  # List to store distributions for each file
    flags_dict_total = {}  # Dictionary to store flags for each file
    num_meteors = len(finder.base_names)  # Number of meteors
    for i, (base_name, dynesty_info, prior_path, out_folder) in enumerate(zip(finder.base_names,finder.input_folder_file,finder.priors,finder.output_folders)):
        dynesty_file, pickle_file, bounds, flags_dict, fixed_values = dynesty_info

        # check if len(flags_dict.keys()) > len(variables) to avoid index error
        if len(flags_dict.keys()) > len(variables):
            variables = list(flags_dict.keys())

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
        'noise_lum': r"$\sigma_{lum}$ [W]",
        'eeucs': r"$E_s$ [MJ/m$^2$]",
        'eeum': r"$E_m$ [MJ/kg]",
        'eeucs_end': r"$E_{s\,end}$ [MJ/m$^2$]",
        'eeum_end': r"$E_{m\,end}$ [MJ/kg]"
    }

    # check if there are variables in the flags_dict that are not in the variable_map
    for variable in variables:
        if variable not in variable_map:
            print(f"Warning: {variable} not found in variable_map")
            # Add the variable to the map with a default label
            variable_map[variable] = variable
    labels = [variable_map[variable] for variable in variables]

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
    mm_size_corrected = []
    mass_distr = []
    erosion_energy_per_unit_cross_section_corrected = []
    erosion_energy_per_unit_mass_corrected = []


    # check if a file with the name "log"+n_PC_in_PCA+"_"+str(len(df_sel))+"ev.txt" already exist
    if os.path.exists(output_dir_show+os.sep+"log_Mars.txt"):
        # remove the file
        os.remove(output_dir_show+os.sep+"log_Mars.txt")
    # use the Logger class to redirect the print to a file 
    sys.stdout = Logger(output_dir_show,"log_Mars.txt")

    for i, (base_name, dynesty_info, prior_path, out_folder) in enumerate(zip(finder.base_names, finder.input_folder_file, finder.priors, finder.output_folders)):
        dynesty_file, pickle_file, bounds, flags_dict, fixed_values = dynesty_info
        print('\n', base_name)
        print(f"Processed {i+1} out of {len(finder.base_names)}")
        obs_data = finder.observation_instance(base_name)
        obs_data.file_name = pickle_file  # update the file name in the observation data object
        # take the traj.rbeg_ele and the traj.orbit.zc 
        dsampler = dynesty.DynamicNestedSampler.restore(dynesty_file)
        # dsampler = load_dynesty_file(dynesty_file)

        # Align to the union of all variables (padding missing ones with NaN and 0 weights)
        samples_aligned, weights_aligned = align_dynesty_samples(dsampler, variables, flags_dict)

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

        combined_results_meteor = CombinedResults(samples_aligned, weights_aligned)

        summary_df_meteor = summarize_from_cornerplot(combined_results_meteor, variables, labels)

        dynesty_run_results = dsampler.results
        weights = dynesty_run_results.importance_weights()
        w = weights / np.sum(weights)
        samples = dynesty_run_results.samples
        ndim_single = len(variables)
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

        else:
            rho_lo = summary_df_meteor['Median'].values[variables.index('rho')] - summary_df_meteor['Low95'].values[variables.index('rho')]
            rho_hi = summary_df_meteor['High95'].values[variables.index('rho')] - summary_df_meteor['Median'].values[variables.index('rho')]
            rho = summary_df_meteor['Median'].values[variables.index('rho')]

            x = samples[:, variables_sing.index('rho')].astype(float)
            mask = ~np.isnan(x)
            x_valid_rho = x[mask] 

            rho_corrected.append(x_valid_rho)

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

        # compute the meteoroid_diameter from a spherical shape in mm
        all_diameter_mm = (6 * samples[:, variables_sing.index('m_init')].astype(float) / (np.pi * x_valid_rho))**(1/3) * 1000
        mm_size_corrected.append(all_diameter_mm)
        mass_distr.append(samples[:, variables_sing.index('m_init')].astype(float))
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

        # take from dynesty_file folder name
        folder_name = os.path.dirname(dynesty_file)

        # delete from base_name _combined if it exists
        if '_combined' in base_name:
            base_name = base_name.replace('_combined', '')

        # look if in folder_name it exist a file that ends in .dynestyres exist in 
        if any(f.endswith(".dynestyres") for f in os.listdir(folder_name)):
            print(f"Found existing results in {folder_name}.dynestyres, loading them.")

            # look for the file that ends in .dynestyres
            dynesty_res_file = [f for f in os.listdir(folder_name) if f.endswith(".dynestyres")][0]
            with open(folder_name + os.sep + dynesty_res_file, "rb") as f:
                dynesty_run_results = pickle.load(f)

            erosion_energy_per_unit_cross_section_arr = dynesty_run_results.erosion_energy_per_unit_cross_section
            erosion_energy_per_unit_mass_arr = dynesty_run_results.erosion_energy_per_unit_mass

        else:
            print(f"No existing results found in {folder_name}.dynestyres, running dynesty.")
            # dynesty_run_results = dsampler.results

            ### add MORE PARAMETERS ###

            # Package inputs
            inputs = [
                (i, len(dynesty_run_results.samples), dynesty_run_results.samples[i], obs_data, variables_sing, fixed_values, flags_dict)
                for i in range(len(dynesty_run_results.samples)) # for i in np.linspace(0, len(dynesty_run_results.samples)-1, 10, dtype=int)
            ]
            #     for i in range(len(dynesty_run_results.samples)) # 
            num_cores = multiprocessing.cpu_count()

            # Run in parallel
            with Pool(processes=num_cores) as pool:  # adjust to number of cores
                results = pool.map(run_single_eeu, inputs)

            N = len(dynesty_run_results.samples)

            erosion_energy_per_unit_cross_section_arr = np.full(N, np.nan)
            erosion_energy_per_unit_mass_arr = np.full(N, np.nan)
            # erosion_energy_per_unit_cross_section_arr_end = np.full(N, np.nan)
            # erosion_energy_per_unit_mass_arr_end = np.full(N, np.nan)

            for res in results:
                i, eeucs, eeum, eeucs_end, eeum_end = res
                erosion_energy_per_unit_cross_section_arr[i] = eeucs / 1e6  # convert to MJ/m^2
                erosion_energy_per_unit_mass_arr[i] = eeum / 1e6  # convert to MJ/kg
                # erosion_energy_per_unit_cross_section_arr_end[i] = eeucs_end / 1e6  # convert to MJ/m^2
                # erosion_energy_per_unit_mass_arr_end[i] = eeum_end / 1e6  # convert to MJ/kg


            # Create a namespace object for dot-style access
            results = SimpleNamespace(**dsampler.results.__dict__)  # load all default results

            # Add your custom attributes
            results.weights = dynesty_run_results.importance_weights()
            results.norm_weights = w
            results.erosion_energy_per_unit_cross_section = erosion_energy_per_unit_cross_section_arr
            results.erosion_energy_per_unit_mass = erosion_energy_per_unit_mass_arr
            # results.erosion_energy_per_unit_cross_section_end = erosion_energy_per_unit_cross_section_arr_end
            # results.erosion_energy_per_unit_mass_arr_end = erosion_energy_per_unit_mass_arr_end

            # Save
            with open(folder_name + os.sep + base_name+"_results.dynestyres", "wb") as f:
                pickle.dump(results, f)
                print(f"Results saved successfully in {folder_name + os.sep + base_name+'_results.dynestyres'}.")

        erosion_energy_per_unit_cross_section_corrected.append(erosion_energy_per_unit_cross_section_arr)
        erosion_energy_per_unit_mass_corrected.append(erosion_energy_per_unit_mass_arr)
        

        ### TRANSFORM TO MARS VEL ###

        tj, tj_lo, tj_hi, inclin_val, Vinf_val, Vg_val, Q_val, q_val, a_val, e_val, peri_val, node_val = extract_tj_from_report(report_path)

        print(f"Transforming velocities to Mars orbit:")
        # constants
        G0_mars = 3.75  # m/s^2
        dens_co_mars = fitAtmPoly_mars(40*1000, 180*1000)
        dens_co_earth = np.array(best_guess_obj_plot.const.dens_co)
        rho_poly_earth = []
        rho_poly_mars = []
        altitude = np.arange(40, 181, 0.1)*1000
        for alt in altitude:
            rho_poly_earth.append((atmDensPoly(alt, dens_co_earth)))
            rho_poly_mars.append((atmDensPoly(alt, dens_co_mars)))

        # start simulations at the same bulk density point
        dens_start_earth = atmDensPoly(best_guess_obj_plot.const.h_init, dens_co_earth)
        start_height_mars = altitude[np.argmin(np.abs(np.array(rho_poly_mars) - dens_start_earth))]
        print(f"Simulation start on Earth: {best_guess_obj_plot.const.h_init/1000:.2f} km")
        print(f"Simulation start on Mars: {start_height_mars/1000:.2f} km")

        # find the closest best_guess_obj_plot.const.erosion_height_start tat share the same density in mars
        dens_erosion_earth = atmDensPoly(best_guess_obj_plot.const.erosion_height_start, dens_co_earth)
        erosion_height_start_mars = altitude[np.argmin(np.abs(np.array(rho_poly_mars) - dens_erosion_earth))]
        print(f"Erosion start on Earth: {best_guess_obj_plot.const.erosion_height_start/1000:.2f} km")
        print(f"Erosion height start on Mars: {erosion_height_start_mars/1000:.2f} km")
        if flag_total_rho:
            dens_erosion_change_earth = atmDensPoly(best_guess_obj_plot.const.erosion_height_change, dens_co_earth)
            erosion_height_change_mars = altitude[np.argmin(np.abs(np.array(rho_poly_mars) - dens_erosion_change_earth))]
            # keep the same difference between erosion_height_change and erosion_height_start
            # erosion_height_change_mars = erosion_height_start_mars - abs(best_guess_obj_plot.const.erosion_height_change - best_guess_obj_plot.const.erosion_height_start)
            # print(f"Erosion change on Earth: {best_guess_obj_plot.const.erosion_height_change/1000:.2f} km")
            # print(f"Erosion height change on Mars: {erosion_height_change_mars/1000:.2f} km")
            print(f"Erosion change on Earth: {best_guess_obj_plot.const.erosion_height_change/1000:.2f} km")
            print(f"Erosion height change on Mars: {erosion_height_change_mars/1000:.2f} km")

        V_ESC_MARS = escape_speed_kms(MU_MARS, R_MARS, start_height_mars/1000)
        # print(f"Escape speed on Mars at {start_height_mars/1000:.2f} km: {V_ESC_MARS:.6f} km/s")
        # print(f"Escape speed on Mars at 180 km: {V_ESC_MARS180:.6f} km/s")

        # compute teh visvida velocity on earth
        V_val_earth = vis_viva_speed_kms(a_val, A_EARTH_AU)
        print(f"Vis viva speed on Earth orbit: {V_val_earth:.6f} km/s")
        if Q_val < 0:
            Vinf_val= V_PARAB_EARTHORBIT + V_ORBIT_EARTH
            Vg_val = V_PARAB_EARTHORBIT + V_ORBIT_EARTH 
            V_val_mars = V_PARAB_MARSORBIT
            Vg_val_mars = V_PARAB_MARSORBIT + V_ORBIT_MARS
            Vinf_val_mars = V_PARAB_MARSORBIT + V_ORBIT_MARS
            print(f"Hyperbolic orbit, setting Vinf to parabolic + orbit speeds")
            continue
        elif Q_val < A_MARS_AU:
            print(f"Orbit do not cross Mars Orbit, using equivalence method to compute Vinf on Mars")
            V_val_mars = -1  # to indicate that we are using the equivalence method
            # correct the speed on Mars base on the equivalence between V_ORBIT_EARTH+V_PARAB_EARTHORBIT and V_ESC_EARTH180
            Vg_val_mars = V_ESC_MARS + (Vg_val - V_ESC_EARTH180)/ (V_ORBIT_EARTH + V_PARAB_EARTHORBIT-V_ESC_EARTH180) * (V_ORBIT_MARS + V_PARAB_MARSORBIT - V_ESC_MARS)
            # correct the speed on Mars base on the equivalence between V_ORBIT_EARTH+V_PARAB_EARTHORBIT and V_ESC_EARTH180
            Vinf_val_mars = V_ESC_MARS + (Vinf_val - V_ESC_EARTH180)/ (V_ORBIT_EARTH + V_PARAB_EARTHORBIT-V_ESC_EARTH180) * (V_ORBIT_MARS + V_PARAB_MARSORBIT - V_ESC_MARS)
            continue
        else:
            # compute teh visvida velocity on mars
            V_val_mars = vis_viva_speed_kms(a_val, A_MARS_AU)
            print(f"Vis viva speed on Mars orbit: {V_val_mars:.6f} km/s")
            # waht is te fraction of earth orbit speed that can be attributed to Vg_val V_ORBIT_EARTH
            cos_earth =  (V_ORBIT_EARTH**2 + V_val_earth**2 - Vg_val**2)/(2*V_ORBIT_EARTH*V_val_earth)
            # print(f"Cosine angle between Vg and V_ORBIT_EARTH: {cos_earth:.6f}")
            Vg_val_mars = math.sqrt(V_val_mars**2 + V_ORBIT_MARS**2 - 2*V_val_mars*V_ORBIT_MARS*cos_earth)
            Vinf_val_mars = math.sqrt(Vg_val_mars**2 + V_ESC_MARS**2)
            V_val_mars_min_max, Vg_val_mars_min_max, Vinf_val_mars_min_max, V_val_earth_denis, Vg_val_denis, Vinf_val_denis = calculate_3d_intercept_speeds(a_val, e_val, inclin_val, peri_val, node_val)
            print(f"3D intercept method V on Mars: [{Vinf_val_mars_min_max[0]:.3f}, {Vinf_val_mars_min_max[1]:.3f}] km/s")

        print(f"Vg on Earth : {Vg_val:.6f} km/s")
        print(f"Vg on Mars  : {Vg_val_mars:.6f} km/s")

        # print the corrected Vg_val
        print(f"Corrected V for Earth (Vinf): {Vinf_val:.6f} km/s")
        print(f"Corrected V for Mars (Vinf) : {Vinf_val_mars:.6f} km/s")
        print(f"Corrected V equivalent for Mars : {V_ESC_MARS + (Vinf_val - V_ESC_EARTH180)/ (V_ORBIT_EARTH + V_PARAB_EARTHORBIT-V_ESC_EARTH180) * (V_ORBIT_MARS + V_PARAB_MARSORBIT - V_ESC_MARS):.6f} km/s")
        
        # create a deep copy of best_guess_cost
        best_guess_cost_mars = copy.deepcopy(best_guess_obj_plot.const)
        best_guess_cost_mars.h_init = start_height_mars
        best_guess_cost_mars.erosion_height_start = erosion_height_start_mars
        if flag_total_rho:
            best_guess_cost_mars.erosion_height_change = erosion_height_change_mars
        best_guess_cost_mars.v_init = Vinf_val_mars * 1000  # convert to m/s
        # PLANET PARAMETERS
        best_guess_cost_mars.G0 = G0_mars  # m/s^2
        best_guess_cost_mars.r_earth = R_MARS * 1000  # in m
        best_guess_cost_mars.dens_co = np.array(dens_co_mars)

        # ZENITH ANGLE CALCULATION
        print(f"Zenith angle Earth: {180/np.pi*best_guess_cost_mars.zenith_angle:.6f}°")
        zenith_angle_list_mars = []
        for curr_pickle_file in pickle_file:
            traj=loadPickle(*os.path.split(curr_pickle_file))
            zenith_angle_list_mars.append(zenithAngleAtSimulationBegin(start_height_mars, traj.rbeg_ele, traj.orbit.zc, best_guess_cost_mars.r_earth))
        best_guess_cost_mars.zenith_angle = np.mean(zenith_angle_list_mars)
        print(f"Zenith angle Mars: {180/np.pi*best_guess_cost_mars.zenith_angle:.6f}°")

        # Minimum height (m) for simulation termination
        best_guess_cost_mars.h_kill = 6000

        
        frag_main, results_list, wake_results = runSimulation(best_guess_cost_mars, compute_wake=False)
        best_guess_obj_plot_mars = SimulationResults(best_guess_cost_mars, frag_main, results_list, wake_results)

        # plot y axis the unique_heights_massvar vs Tot_energy_arr
        # fig, ax = plt.subplots(1,2, figsize=(12, 6))
        fig, ax = plt.subplots(figsize=(7, 6))
        station_colors = {}
        cmap = plt.get_cmap("tab10")
        # ABS MAGNITUDE
        for station in np.unique(obs_data.stations_lum):
            mask = obs_data.stations_lum == station
            if station not in station_colors:
                station_colors[station] = cmap(len(station_colors) % 10)
            ax.plot(obs_data.absolute_magnitudes[mask], obs_data.height_lum[mask] / 1000, 'x--', color=station_colors[station], label=station)
        # max_mag= np.max(obs_data.absolute_magnitudes)+1
        # take the y axis limits from the obs_data
        y_min = ax.get_ylim()[0]
        y_max = ax.get_ylim()[1]
        x_max = ax.get_xlim()[1]

        # Integrate luminosity/magnitude if needed
        if (1 / obs_data.fps_lum) > best_guess_obj_plot.const.dt:
            best_guess_obj_plot.luminosity_arr, best_guess_obj_plot.abs_magnitude = luminosity_integration(
                best_guess_obj_plot.time_arr, best_guess_obj_plot.time_arr, best_guess_obj_plot.luminosity_arr,
                best_guess_obj_plot.const.dt, obs_data.fps_lum, obs_data.P_0m
            )

        # make a first subplot with the lightcurve against height
        ax.plot(best_guess_obj_plot.abs_magnitude,best_guess_obj_plot.leading_frag_height_arr/1000, color='k', label='Best Fit Simulation')
        ax.set_ylabel('Height [km]', fontsize=15)
        ax.set_xlabel('Abs.Mag [-]', fontsize=15)
        # add a hrizontal line at y=total_energy_before_erosion
        ax.axhline(y=best_guess_obj_plot.const.erosion_height_start/1000, color='gray', linestyle='--', label='Erosion Height Start $h_{e}$')
        if flag_total_rho:
            ax.axhline(y=best_guess_obj_plot.const.erosion_height_change/1000, color='gray', linestyle='-.', label='Erosion Height Change $h_{e2}$')
        # ax.legend(fontsize=10)
        # call the Earth plot
        # ax[0].set_title('Earth', fontsize=16)

        # Integrate luminosity/magnitude if needed
        if (1 / obs_data.fps_lum) > best_guess_obj_plot.const.dt:
            best_guess_obj_plot_mars.luminosity_arr, best_guess_obj_plot_mars.abs_magnitude = luminosity_integration(
                best_guess_obj_plot_mars.time_arr, best_guess_obj_plot_mars.time_arr, best_guess_obj_plot_mars.luminosity_arr,
                best_guess_obj_plot_mars.const.dt, obs_data.fps_lum, obs_data.P_0m
            )
        # make a second subplot with the lightcurve against height for mars
        ax.plot(best_guess_obj_plot_mars.abs_magnitude,best_guess_obj_plot_mars.leading_frag_height_arr/1000, color='r', label='Best Fit Simulation (Mars)')
        ax.axhline(y=best_guess_obj_plot_mars.const.erosion_height_start/1000, color='pink', linestyle='--', label='Erosion Height Start $h_{e}$ (Mars)')
        if flag_total_rho:
            ax.axhline(y=best_guess_obj_plot_mars.const.erosion_height_change/1000, color='pink', linestyle='-.', label='Erosion Height Change $h_{e2}$ (Mars)')
        # ax[0].invert_xaxis()
        # ax[1].legend(fontsize=10)
        # ax.set_title('Mars', fontsize=16)
        # ax[1].set_ylabel('Height [km]', fontsize=15)
        ax.set_xlabel('Abs.Mag [-]', fontsize=15)
        # activate grid
        ax.grid()
        x_min = ax.get_xlim()[0]
        x_max = 8
        # put the x axis from the x_max to x_min
        ax.set_xlim(x_max, x_min)
        # for the y axis
        new_ax_min = np.min([y_min, ax.get_ylim()[0]])
        # # closest index to the x_max in best_guess_obj_plot_mars.abs_magnitude after the erosion start
        # index_min_abs_mag = np.argmin(np.abs(best_guess_obj_plot_mars.abs_magnitude[np.argmin(np.abs(best_guess_obj_plot_mars.leading_frag_height_arr - best_guess_obj_plot_mars.const.erosion_height_start)):] - x_max))
        # # closest index to the x_max in best_guess_obj_plot_mars.leading_frag_height_arr/1000
        # new_ax_min = np.max([best_guess_obj_plot_mars.leading_frag_height_arr[index_min_abs_mag]/1000, new_ax_min])
        new_ax_max = np.min([y_max, ax.get_ylim()[1]])
        new_ax_max = np.max([best_guess_obj_plot.const.erosion_height_start/1000+2, new_ax_max])
        ax.set_ylim(new_ax_min, new_ax_max)
        # ax.invert_xaxis()
        # put legend outside the plot
        # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        ax.legend(fontsize=10)
        # ax[1].grid()
        # plt.suptitle(f'Lightcurve Comparison for {base_name}', fontsize=18)
        plt.tight_layout()
        plt.savefig(output_dir + os.sep + base_name + "_Lightcurve_Earth_vs_Mars.png")
        plt.close()




        ### save the results


        file_rho_jd_dict[base_name] = (rho, rho_lo,rho_hi, tj, tj_lo, tj_hi, inclin_val, Vinf_val, Vg_val, Q_val, q_val, a_val, e_val, V_val_earth, V_val_mars, Vg_val_mars, Vinf_val_mars, Vg_val_mars_min_max, Vinf_val_mars_min_max, Vg_val_denis, Vinf_val_denis)
        # file_eeu_dict[base_name] = (eeucs, eeucs_lo, eeucs_hi, eeum, eeum_lo, eeum_hi,F_par, kc_par, lenght_par)
        file_obs_data_dict[base_name] = (kc_par, F_par, lenght_par, beg_height/1000, end_height/1000, max_lum_height/1000, avg_vel/1000, init_mag, end_mag, max_mag, time_tot, zenith_angle, m_init_meteor_median, meteoroid_diameter_mm, erosion_beg_dyn_press, v_init_meteor_median, tau_median, tau_low95, tau_high95)
        file_phys_data_dict[base_name] = (eta_meteor_begin, sigma_meteor_begin, meteoroid_diameter_mm, meteoroid_diameter_mm_lo, meteoroid_diameter_mm_hi, m_init_meteor_median, m_init_meteor_lo, m_init_meteor_hi)

        all_names.append(base_name)
        all_samples.append(samples_aligned)
        all_weights.append(weights_aligned)

    # Close the Logger to ensure everything is written to the file STOP COPY in TXT file
    sys.stdout.close()

    # Reset sys.stdout to its original value if needed
    sys.stdout = sys.__stdout__

    print("\nPlotting Vinf distribution...")
    # plot the distribution of speed and Vg_val and Vg_val_mars in a single plot one in blue for earth and one in red for mars 
    fig, axs = plt.subplots(1, 1, figsize=(8, 6))
    for name in all_names:
        rho, rho_lo, rho_hi, tj, tj_lo, tj_hi, inclin_val, Vinf_val, Vg_val, Q_val, q_val, a_val, e_val, V_val_earth, V_val_mars, Vg_val_mars, Vinf_val_mars, Vg_val_mars_min_max, Vinf_val_mars_min_max, Vg_val_denis, Vinf_val_denis = file_rho_jd_dict[name]
        axs.scatter(Vinf_val, tj, color='blue')
        # axs.scatter( Vinf_val_denis, tj, color='blue', marker='d')
        # add a asterisk next to the point if V_val_mars < 0
        if V_val_mars < 0:
            axs.scatter(Vinf_val_mars, tj, marker='x', color='red', s=50)
        else:
            axs.plot([Vinf_val_mars_min_max[0], Vinf_val_mars_min_max[1]], [tj, tj], color='red', marker='d')
            axs.scatter(Vinf_val_mars, tj, color='red')
        # # add error bars
        # axs.errorbar(Vg_val, tj, xerr=0, yerr=[[tj - tj_lo], [tj_hi - tj]], fmt='o', color='blue', alpha=0.5)
        # axs.errorbar(Vg_val_mars, tj, xerr=0, yerr=[[tj - tj_lo], [tj_hi - tj]], fmt='o', color='red', alpha=0.5)
    # add the labels and legend for a red and blue points
    axs.scatter([], [], color='blue', label='Earth')
    axs.scatter([], [], color='red', label='Mars 2D')
    # axs.scatter([], [], color='blue', marker='d', label='Earth 3D')
    axs.plot([], [], color='red', marker='d', label='Mars 3D')
    # axs.scatter([], [], color='red', marker='x', s=50, label='Not reach Mars')
    print(f"Earth Min: {V_ESC_EARTH180:.3f} km/s, Mars Min: {V_ESC_MARS180:.3f} km/s")
    print(f"Earth MAX: {V_ORBIT_EARTH+V_PARAB_EARTHORBIT:.3f} km/s, Mars MAX: {V_ORBIT_MARS+V_PARAB_MARSORBIT:.3f} km/s")
    # plot as a dashed blue line the escape speed of earth at 180 km
    axs.axvline(x=V_ESC_EARTH180, color='blue', linestyle='--', label='Minimum meteoroid speed on Earth')
    axs.axvline(x=V_ESC_MARS180, color='red', linestyle='--', label='Minimum meteoroid speed on Mars')
    # plot as a dashed blue line the maximum speed of earth at 180 km
    axs.axvline(x=V_ORBIT_EARTH+V_PARAB_EARTHORBIT, color='blue', linestyle=':', label='MAX meteoroid speed on Earth')
    axs.axvline(x=V_ORBIT_MARS+V_PARAB_MARSORBIT, color='red', linestyle=':', label='MAX meteoroid speed on Mars')
    axs.set_xlabel('$V_{\infty}$ [km/s]', fontsize=12)
    axs.set_ylabel('$T_j$', fontsize=12)
    axs.legend(fontsize=10)
    # grid on
    axs.grid()
    plt.tight_layout()
    plt.savefig(output_dir_show + os.sep + "Vinf_distribution.png")
    plt.close()

if __name__ == "__main__":

    import argparse
    ### COMMAND LINE ARGUMENTS
    arg_parser = argparse.ArgumentParser(description="Run dynesty with optional .prior file.")
    
    arg_parser.add_argument('--input_dir', metavar='INPUT_PATH', type=str,
                            # "C:\Users\maxiv\Documents\UWO\Papers\3)Sporadics\Results\Sporadics_rho-uniform\Best_irons",
                            # "C:\Users\maxiv\Documents\UWO\Papers\3)Sporadics\Results\Sporadics_rho-uniform\Fastsporad_CAMOnew+EMCCD_unif_density\fastsporad_EMCCD+CAMO_CAP"
        default=r"C:\Users\maxiv\Documents\UWO\Papers\3)Sporadics\Results\Sporadics_rho-uniform",
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

    shower_distrb_plot(cml_args.input_dir, cml_args.output_dir, cml_args.name) # cml_args.radiance_plot cml_args.correl_plot
