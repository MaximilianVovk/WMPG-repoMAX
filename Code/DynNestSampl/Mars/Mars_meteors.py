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
import seaborn as sns
import pandas as pd

# import from Mars_AtmDens.py
from Mars_AtmDens import fitAtmPoly_mars
from Mars_Vel import calculate_3d_intercept_speeds

# Add the parent directory to the sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from DynNestSapl_metsim import *

from scipy.interpolate import griddata
from scipy import ndimage
from scipy.interpolate import UnivariateSpline
from matplotlib.collections import LineCollection
from itertools import combinations
from scipy.stats import ks_2samp, mannwhitneyu, anderson_ksamp
from scipy.stats import gaussian_kde
from scipy.stats import binned_statistic_2d
from scipy import ndimage
from scipy.ndimage import gaussian_filter
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
from wmpl.Utils.AtmosphereDensity import fitAtmPoly, atmDensPoly, getAtmDensity
from wmpl.MetSim.MetSimErosionCyTools import atmDensityPoly
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

def compute_energy_per_height(best_guess_obj_plot, flag_total_rho=False, only_from_he_to_he2=False):
    """
    Compute the total energy received before erosion for the best guess object.
    """
    # extract the const from the best_guess_obj_plot
    const_nominal = best_guess_obj_plot.const

    mass_best = np.array(best_guess_obj_plot.main_mass_arr[:-1], dtype=np.float64) # mass_total_active_arr # main_mass_arr
    heights = np.array(best_guess_obj_plot.leading_frag_height_arr[:-1], dtype=np.float64)
    velocities = np.array(best_guess_obj_plot.leading_frag_vel_arr[:-1], dtype=np.float64)

    # Extract physical quantities
    eeucs_best, eeum_best = energyReceivedBeforeErosion(const_nominal)
    total_energy_before_erosion = eeum_best * best_guess_obj_plot.const.m_init / 1e3  # Total energy received before erosion in kJ from the Kinetic Energy not the one computed
    eeum_best = eeum_best / 1e6  # convert to MJ/kg
    # precise erosion tal energy calculation ########################
    erosion_height_start = best_guess_obj_plot.const.erosion_height_start
    ### get for each mass_best that is different from te previuse one get the height at which the mass loss happens
    diff_mask = np.concatenate(([True], np.diff(mass_best) != 0))
    ### only consider when mass actually changes enought
    # diff_mask = np.concatenate(([True], ~np.isclose(np.diff(mass_best), 0)))
    unique_heights_massvar = heights[diff_mask]
    mass_best_massvar = mass_best[diff_mask]
    velocities_massvar = velocities[diff_mask]
    # print("alt0",len(unique_heights_massvar))

    if only_from_he_to_he2:
        # # now delete any unique_heights_massvar and mass_best_massvar that are bigger than erosion_height_change
        mass_best_massvar = mass_best_massvar[unique_heights_massvar < erosion_height_start]
        velocities_massvar = velocities_massvar[unique_heights_massvar < erosion_height_start]
        unique_heights_massvar = unique_heights_massvar[unique_heights_massvar < erosion_height_start]
        # print("alt",len(unique_heights_massvar))

        # cut mass_best_massvar and unique_heights_massvar up to erosion_height_change
        if flag_total_rho:
            mass_best_massvar = mass_best_massvar[unique_heights_massvar >= best_guess_obj_plot.const.erosion_height_change]
            velocities_massvar = velocities_massvar[unique_heights_massvar >= best_guess_obj_plot.const.erosion_height_change]
            unique_heights_massvar = unique_heights_massvar[unique_heights_massvar >= best_guess_obj_plot.const.erosion_height_change]
            # print("alt2",len(unique_heights_massvar))

    ### normal way
    unique_heights_massvar_init = np.concatenate(([best_guess_obj_plot.const.h_init], unique_heights_massvar))
    mass_best_massvar = np.concatenate(([best_guess_obj_plot.const.m_init], mass_best_massvar))
    velocities_massvar_init = np.concatenate(([best_guess_obj_plot.const.v_init], velocities_massvar))
    # deete the last element of unique_heights_massvar_init and mass_best_massvar
    unique_heights_massvar_init = unique_heights_massvar_init[:-1]
    mass_best_massvar = mass_best_massvar[:-1]
    velocities_massvar_init = velocities_massvar_init[:-1]

    # Package inputs
    inputs = [
        (i, best_guess_obj_plot, unique_heights_massvar_init[i], unique_heights_massvar[i], mass_best_massvar[i], velocities_massvar_init[i], 1, best_guess_obj_plot.const.dens_co)
        for i in range(len(mass_best_massvar)) # for i in np.linspace(0, len(dynesty_run_results.samples)-1, 10, dtype=int)
    ]
    #     for i in range(len(dynesty_run_results.samples)) # 
    num_cores = multiprocessing.cpu_count()

    # Run in parallel
    with Pool(processes=num_cores) as pool:  # adjust to number of cores
        results = pool.map(run_total_energy_received_varNom, inputs)

    N = len(mass_best_massvar)
    # print("N energy received:", N)

    # Pre-allocate for 3 results for each simulation
    Tot_energy_arr = np.full((2, N), np.nan, dtype=float)  # kJ
    eeucs_end      = np.full((2, N), np.nan, dtype=float)   # MJ/m^2
    eeum_end       = np.full((2, N), np.nan, dtype=float)   # MJ/kg

    # Collect
    for res in results:
        i, eeucs, eeum, tot_en = res  # each of eeucs/eeum/tot_en is length-3 array
        i = int(i)
        if 0 <= i < N:
            eeucs_end[:, i] = eeucs
            eeum_end[:, i]  = eeum
            Tot_energy_arr[:, i] = tot_en / 1e3  # J -> kJ
        else:
            print(f"Warning: received invalid index {i} in total energy received results.")

    # Convert to MJ units only when you need to present/store them as MJ:
    Tot_energy_per_unit_cross_section = eeucs_end[0, :] / 1e6   # J/m^2 -> MJ/m^2
    Tot_energy_per_unit_mass  = eeum_end[0, :]  / 1e6   # J/kg  -> MJ/kg
    Tot_energy  = np.sum(Tot_energy_arr[0, :]) # J -> kJ

    Tot_energy_arr_cum = np.cumsum(Tot_energy_arr[0,:])

    # the first is the tota energy at the erosion height start
    total_energy_at_erosion_height_start = Tot_energy_arr_cum[0]
    if flag_total_rho:
        total_energy_at_erosion_height_change = Tot_energy_arr_cum[-1]
        return Tot_energy_arr_cum, total_energy_at_erosion_height_start, total_energy_at_erosion_height_change
    else:
        return Tot_energy_arr_cum, total_energy_at_erosion_height_start, None


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

def energyReceivedBeforeErosion_varNom(const, dens_co_atm_nominal, lam=1.0):
    """ Compute the energy the meteoroid receive prior to erosion +- 25%, assuming no major mass loss occured. 
    
    Arguments:
        const: [Constants]

    Keyword arguments:
        lam: [float] Heat transfter coeff. 1.0 by default.

    Return: max and min energy received prior to erosion given 25% variation
        (es, ev):
            - es: [float] Energy received per unit cross-section (J/m^2)
            - ev: [float] Energy received per unit mass (J/kg).

    """

    # Integrate atmosphere density from the beginning of simulation to beginning of erosion.
    dens_integ = scipy.integrate.quad(atmDensityPoly, const.erosion_height_start, const.h_init, \
        args=(const.dens_co))[0]

    # Integrate atmosphere density from the beginning of simulation to beginning of erosion.
    dens_integ_atm_nominal = scipy.integrate.quad(atmDensityPoly, const.erosion_height_start, const.h_init, \
        args=(dens_co_atm_nominal))[0]

    dens_integ = np.array([dens_integ, dens_integ_atm_nominal])  # dens_integ * 0.75, dens_integ * 1.25])

    # Compute the energy per unit cross-section
    es = 1/2*lam*(const.v_init**2)*dens_integ/np.cos(const.zenith_angle)

    # Compute initial shape-density coefficient
    k = const.gamma*const.shape_factor*const.rho**(-2/3.0)

    # Compute the energy per unit mass
    ev = es*k/(const.gamma*const.m_init**(1/3.0))

    return es, ev

def run_total_energy_received_varNom(sim_num_and_data):
    sim_num, best_guess_obj_plot, unique_heights_massvar_init, unique_heights_massvar, mass_best_massvar, velocities_massvar_init, lambda_val, dens_co_atm_nominal = sim_num_and_data

    # extract the const from the best_guess_obj_plot
    const_bestguess = best_guess_obj_plot.const

    const_bestguess.h_init = unique_heights_massvar_init
    const_bestguess.erosion_height_start = unique_heights_massvar
    const_bestguess.v_init = velocities_massvar_init
    const_bestguess.m_init = mass_best_massvar

    # Extract physical quantities
    try:
        eeucs, eeum = energyReceivedBeforeErosion_varNom(const_bestguess, dens_co_atm_nominal, lambda_val)
        total_energy = eeum * const_bestguess.m_init  # Total energy received before erosion in MJ

        return (sim_num, eeucs, eeum, total_energy)

    except Exception as e:
        print(f"Simulation {sim_num} failed: {e}")
        return (sim_num, np.nan, np.nan, np.nan)


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



def save_json_data(self,file_name):
    """Save the object to a JSON file."""

    # Deep copy to avoid modifying the original object
    json_self_save = copy.deepcopy(self)

    # Convert all numpy arrays in `self2` to lists
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        elif hasattr(obj, '__dict__'):  # Convert objects with __dict__
            return convert_to_serializable(obj.__dict__)
        else:
            return obj  # Leave as is if it's already serializable

    serializable_dict = convert_to_serializable(json_self_save.__dict__)

    # Define file path for saving
    json_file_save = os.path.splitext(file_name)[0] + "_saved.json"
    # # check if the file exists if so give a _1, _2, _3, etc. at the end of the file name
    # i_json = 1
    # if os.path.exists(json_file_save):
    #     while os.path.exists(json_file_save):
    #         json_file_save = os.path.splitext(file_name)[0] + f"_{i_json}_saved.json"
    #         i_json += 1

    # # if already exists, remove it
    # if os.path.exists(json_file_save):
    #     os.remove(json_file_save)

    # Write to JSON file
    with open(json_file_save, 'w') as f:
        json.dump(serializable_dict, f, indent=4)

    print("Saved fit parameters saved to:", json_file_save)

    return json_file_save



def Mars_distrb_plot(input_dirfile, output_dir_show, shower_name, new_marsmeteor=False):
    """
    Function to plot the distribution of the parameters from the dynesty files and save them as a table in LaTeX format.
    """
    print(f"Processing input: {input_dirfile}")
    if new_marsmeteor:
        print("Update the .marsmeteor file.")
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


    # the on that are not variables are the one that were not used in the dynesty run give a np.nan weight to dsampler for those
    all_samples = []
    all_weights = []
    all_names = []  

    # base_name, lg_min_la_sun, bg, rho
    file_obs_data_dict = {}
    file_phys_data_dict = {}
    file_bright_dict = {}
    file_rho_jd_dict = {}

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

        # delete from base_name _combined if it exists
        if '_combined' in base_name:
            base_name = base_name.replace('_combined', '')

        # chek for the .marsmeteor file to extract the meteor name
        marsmeteor_file = None
        for name in os.listdir(output_dir):
            if name.endswith(".marsmeteor"):
                marsmeteor_file = name; break
            
        if new_marsmeteor:
            marsmeteor_file = None

        if marsmeteor_file is None:
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

            # chek if obs_data is a None type object
            if obs_data is None:
                print(f"Observation data for {base_name} is None. Skipping this meteor.")
                continue

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
            # if start_height_mars > 100000:
            #     start_height_mars = 100000  # do not start below 100 km on mars
            print(f"Simulation start on Earth: {best_guess_obj_plot.const.h_init/1000:.2f} km")
            print(f"Simulation start on Mars: {start_height_mars/1000:.2f} km")

            # find the closest best_guess_obj_plot.const.erosion_height_start tat share the same density in mars
            dens_erosion_earth = atmDensPoly(best_guess_obj_plot.const.erosion_height_start, dens_co_earth)
            erosion_height_start_mars = altitude[np.argmin(np.abs(np.array(rho_poly_mars) - dens_erosion_earth))]
            print(f"Erosion onset on Earth: {best_guess_obj_plot.const.erosion_height_start/1000:.2f} km rho: {dens_erosion_earth:.6f} kg/m^3")
            print(f"Erosion onset on Mars: {erosion_height_start_mars/1000:.2f} km rho: {atmDensPoly(erosion_height_start_mars, dens_co_mars):.6f} kg/m^3")
            if flag_total_rho:
                dens_erosion_change_earth = atmDensPoly(best_guess_obj_plot.const.erosion_height_change, dens_co_earth)
                erosion_height_change_mars = altitude[np.argmin(np.abs(np.array(rho_poly_mars) - dens_erosion_change_earth))]
                # keep the same difference between erosion_height_change and erosion_height_start
                # erosion_height_change_mars = erosion_height_start_mars - abs(best_guess_obj_plot.const.erosion_height_change - best_guess_obj_plot.const.erosion_height_start)
                print(f"Erosion change on Earth: {best_guess_obj_plot.const.erosion_height_change/1000:.2f} km rho: {dens_erosion_change_earth:.6f} kg/m^3")
                print(f"Erosion height change on Mars: {erosion_height_change_mars/1000:.2f} km rho: {atmDensPoly(erosion_height_change_mars, dens_co_mars):.6f} kg/m^3")
                
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
                print(f"2D intercept method V on Mars: {Vinf_val_mars:.6f} km/s")
                V_val_mars_min_max, Vg_val_mars_min_max, Vinf_val_mars_min_max, V_val_earth_denis, Vg_val_denis, Vinf_val_denis = calculate_3d_intercept_speeds(a_val, e_val, inclin_val, peri_val, node_val)
                print(f"3D intercept method V on Mars: [{Vinf_val_mars_min_max[0]:.3f}, {Vinf_val_mars_min_max[1]:.3f}] km/s")
                Vinf_val_mars = np.mean(Vinf_val_mars_min_max)

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
            best_guess_cost_mars.v_kill = Vinf_val_mars * 1000 - 10000  # convert to m/s
            if best_guess_cost_mars.v_kill < 0:
                best_guess_cost_mars.v_kill = 1

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

            # # get rid of pickle_file file name and take the folder
            # out_folder_json = os.path.dirname(pickle_file[0])

            # json_file_save = out_folder_json + os.sep + base_name + "_fit_params_saved_mars.json"
            # save_json_data(best_guess_obj_plot_mars, json_file_save)

            ############## SINGLE BODY ABLATION ####################################################################

            if '_combined' in base_name:
                base_name = base_name.replace('_combined', '')

            # find in the Single body ablation with the same base_name in r"C:\Users\maxiv\Documents\UWO\Papers\4)Mars meteors\Results\SingleBody"
            single_body_folder = r"C:\Users\maxiv\Documents\UWO\Papers\4)Mars meteors\Results\SingleBody"
            # check if any folder in single_body_folder contains base_name
            matching_folders = [f for f in os.listdir(single_body_folder) if base_name in f]
            if len(matching_folders) > 0:
                print(f"Found single body ablation folder for {base_name}")
                # use the finder to load the observation data
                single_body_folder_full = os.path.join(single_body_folder, matching_folders[0])
                finder_single = find_dynestyfile_and_priors(input_dir_or_file=single_body_folder_full,prior_file="",resume=True,output_dir=single_body_folder_full,use_all_cameras=True,pick_position=0)

                # input_folder_file is a list/iterable of dynesty_info tuples; take the first one
                dynesty_info_single = finder_single.input_folder_file[0]
                dynesty_file_single, pickle_file_single, bounds_single, flags_dict_single, fixed_values_single = dynesty_info_single

                dsampler_single = dynesty.DynamicNestedSampler.restore(dynesty_file_single)
                dynesty_run_results_single = dsampler_single.results
                sim_num_single = np.argmax(dynesty_run_results_single.logl)
                # copy the best guess values
                guess_single = dynesty_run_results_single.samples[sim_num_single].copy()
                samples_single = dynesty_run_results_single.samples
                variables_single = list(flags_dict_single.keys())
                for i, variable in enumerate(variables_single):
                    if 'log' in flags_dict_single[variable]:
                        guess_single[i] = 10**guess_single[i]
                        samples_single[:, i] = 10**samples_single[:, i]
                best_guess_obj_plot_single_Earth = run_simulation(guess_single, obs_data, variables_single, fixed_values_single)
                
                # run for mars
                best_guess_cost_single_mars = copy.deepcopy(best_guess_cost_mars)#copy.deepcopy(best_guess_obj_plot_single_Earth.const)
                
                best_guess_cost_single_mars.erosion_height_start = 2
                best_guess_cost_single_mars.erosion_height_change = 1
                best_guess_cost_single_mars.rho = best_guess_obj_plot_single_Earth.const.rho
                # best_guess_cost_single_mars.erosion_coeff = best_guess_obj_plot_single_Earth.const.erosion_coeff
                best_guess_cost_single_mars.sigma = best_guess_obj_plot_single_Earth.const.sigma
                best_guess_cost_single_mars.m_init = best_guess_obj_plot_single_Earth.const.m_init
                
                # best_guess_cost_single_mars.h_init = start_height_mars
                # best_guess_cost_single_mars.v_init = Vinf_val_mars * 1000  # convert to m/s
                # # PLANET PARAMETERS
                # best_guess_cost_single_mars.G0 = G0_mars  # m/s^
                # best_guess_cost_single_mars.r_earth = R_MARS * 1000  # in m
                # best_guess_cost_single_mars.dens_co = np.array(dens_co_mars)
                # best_guess_cost_single_mars.zenith_angle = best_guess_cost_mars.zenith_angle
                # best_guess_cost_mars.v_kill = Vinf_val_mars * 1000 - 10000  # convert to m/s
                # if best_guess_cost_mars.v_kill < 0:
                #     best_guess_cost_mars.v_kill = 1

                # # Minimum height (m) for simulation termination
                # best_guess_cost_single_mars.h_kill = 6000
                frag_main_single, results_list_single, wake_results_single = runSimulation(best_guess_cost_single_mars, compute_wake=False)
                best_guess_obj_plot_single_mars = SimulationResults(best_guess_cost_single_mars, frag_main_single, results_list_single, wake_results_single)
                # check if best_guess_obj_plot_single_mars.leading_frag_dyn_press_arr[:-1] is empty
                if len(best_guess_obj_plot_single_mars.leading_frag_dyn_press_arr[:-1]) == 0:
                    print("Even single body do not work")
                    
            else:
                print(f"No single body ablation folder found for {base_name}, skipping single body Mars simulation.")
                continue

            
            
            ################## DYNAMIC PRESSURE #####################################################

            best_guess_cost_mars_dyn_press = copy.deepcopy(best_guess_cost_mars)

            # check if best_guess_obj_plot_mars.leading_frag_dyn_press_arr[:-1] is empty
            if len(best_guess_obj_plot_mars.leading_frag_dyn_press_arr[:-1]) == 0:
                print("No dynamic pressure data for Mars simulation, use same erosion heights as Mars rho.")
                continue
                    
       
            # now I want to know when it reaches the dyn_press on mars that matches the erosion_beg_dyn_press on earth
            heightsame_dynpress_mars = best_guess_obj_plot_mars.leading_frag_height_arr[np.argmin(np.abs(best_guess_obj_plot_mars.leading_frag_dyn_press_arr[:-1] - best_guess_obj_plot.const.erosion_beg_dyn_press))]
            # heightsame_dynpress_mars_single = best_guess_obj_plot_single_mars.leading_frag_height_arr[np.argmin(np.abs(best_guess_obj_plot_single_mars.leading_frag_dyn_press_arr[:-1] - best_guess_obj_plot.const.erosion_beg_dyn_press))]
            # print(f"Erosion onset dynamic pressure on Mars (single): {best_guess_obj_plot.const.erosion_beg_dyn_press} Pa at height {heightsame_dynpress_mars_single/1000:.2f} km instead of {erosion_height_start_mars/1000:.2f} km")
            # now pick the erosion height change that matches the same dynamic pressure on mars
            best_guess_cost_mars_dyn_press.erosion_height_start = heightsame_dynpress_mars
            if flag_total_rho:
                # fid te dynamic pressure on earth at the secod erosion height
                erosion_beg_dyn_press_change = best_guess_obj_plot.leading_frag_dyn_press_arr[np.argmin(np.abs(best_guess_obj_plot.leading_frag_height_arr[:-1] - best_guess_obj_plot.const.erosion_height_change))]
                heightsame_dynpress_change_mars = best_guess_obj_plot_mars.leading_frag_height_arr[np.argmin(np.abs(best_guess_obj_plot_mars.leading_frag_dyn_press_arr[:-1] - erosion_beg_dyn_press_change))]
                # heightsame_dynpress_change_mars_single = best_guess_obj_plot_single_mars.leading_frag_height_arr[np.argmin(np.abs(best_guess_obj_plot_single_mars.leading_frag_dyn_press_arr[:-1] - erosion_beg_dyn_press_change))]
                # print(f"Erosion change dynamic pressure on Mars (single): {erosion_beg_dyn_press_change} Pa at height {heightsame_dynpress_change_mars_single/1000:.2f} km instead of {erosion_height_change_mars/1000:.2f} km")
                best_guess_cost_mars_dyn_press.erosion_height_change = heightsame_dynpress_change_mars                  

            frag_main, results_list, wake_results = runSimulation(best_guess_cost_mars_dyn_press, compute_wake=False)
            best_guess_obj_plot_mars_dyn_press = SimulationResults(best_guess_cost_mars_dyn_press, frag_main, results_list, wake_results)
            
            ################## ENERGY #####################################################

            best_guess_cost_mars_energy = copy.deepcopy(best_guess_cost_mars)

            if flag_total_rho:
                Tot_energy_arr_cum, total_energy_at_erosion_height_start, total_energy_at_erosion_height_change = compute_energy_per_height(best_guess_obj_plot, flag_total_rho=True, only_from_he_to_he2=True)
                Tot_energy_arr_cum_mars, total_energy_at_erosion_height_start_mars, total_energy_at_erosion_height_change_mars = compute_energy_per_height(best_guess_obj_plot_mars, flag_total_rho=True, only_from_he_to_he2=False)
                # find in Tot_energy_arr_cum_mars the closest index to total_energy_at_erosion_height_start and total_energy_at_erosion_height_change
                height_energy_erosion_start_mars = best_guess_obj_plot_mars.leading_frag_height_arr[np.argmin(np.abs(Tot_energy_arr_cum_mars - total_energy_at_erosion_height_start))]
                height_energy_erosion_change_mars = best_guess_obj_plot_mars.leading_frag_height_arr[np.argmin(np.abs(Tot_energy_arr_cum_mars - total_energy_at_erosion_height_change))]  
                best_guess_cost_mars_energy.erosion_height_start = height_energy_erosion_start_mars
                best_guess_cost_mars_energy.erosion_height_change = height_energy_erosion_change_mars
            else:
                _, total_energy_at_erosion_height_start, _ = compute_energy_per_height(best_guess_obj_plot, flag_total_rho=False, only_from_he_to_he2=True)
                Tot_energy_arr_cum_mars, total_energy_at_erosion_height_start_mars, _ = compute_energy_per_height(best_guess_obj_plot_mars, flag_total_rho=False, only_from_he_to_he2=False)
                # find in Tot_energy_arr_cum_mars the closest index to total_energy_at_erosion_height_start
                height_energy_erosion_start_mars = best_guess_obj_plot_mars.leading_frag_height_arr[np.argmin(np.abs(Tot_energy_arr_cum_mars - total_energy_at_erosion_height_start))]
                best_guess_cost_mars_energy.erosion_height_start = height_energy_erosion_start_mars
            
            frag_main, results_list, wake_results = runSimulation(best_guess_cost_mars_energy, compute_wake=False)
            best_guess_obj_plot_mars_energy = SimulationResults(best_guess_cost_mars_energy, frag_main, results_list, wake_results)
                
            ##### PRINT RESULTS FOR MARS SIMULATIONS #####
            print("\n--- Mars Simulation Results ---")
            print(f"\nErosion start on Earth: {best_guess_obj_plot.const.erosion_height_start/1000:.2f} km")
            print("Using same initial density:")
            print(f"Erosion height start on Mars: {best_guess_cost_mars.erosion_height_start/1000:.2f} km rho: {atmDensPoly(best_guess_cost_mars.erosion_height_start, dens_co_mars):.6f} kg/m^3")
            print("Using same dynamic pressure:")
            print(f"Erosion onset dynamic pressure on Mars: {best_guess_cost_mars_dyn_press.erosion_height_start/1000:.2f} km for {best_guess_obj_plot.const.erosion_beg_dyn_press} Pa")
            print("Using same energy at erosion onset:")
            print(f"Erosion energy onset on Mars: {best_guess_cost_mars_energy.erosion_height_start/1000:.2f} km for {total_energy_at_erosion_height_start:.2e} MJ")
            if flag_total_rho:
                print(f"\nErosion change on Earth: {best_guess_obj_plot.const.erosion_height_change/1000:.2f} km")
                print("Using same initial density:")
                print(f"Erosion height change on Mars: {best_guess_cost_mars.erosion_height_change/1000:.2f} km rho: {atmDensPoly(best_guess_cost_mars.erosion_height_change, dens_co_mars):.6f} kg/m^3")
                print("Using same dynamic pressure:")
                print(f"Erosion change dynamic pressure on Mars: {best_guess_cost_mars_dyn_press.erosion_height_change/1000:.2f} km for {erosion_beg_dyn_press_change} Pa")
                print("Using same energy at erosion change:")
                print(f"Erosion height change on Mars: {best_guess_cost_mars_energy.erosion_height_change/1000:.2f} km for {total_energy_at_erosion_height_change:.2e} MJ")
            ############ PLOTTING ##############

            # plot y axis the unique_heights_massvar vs Tot_energy_arr
            # fig, ax = plt.subplots(1,2, figsize=(12, 6))
            fig, ax = plt.subplots(figsize=(10, 6))
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
            if (1 / obs_data.fps_lum) > best_guess_obj_plot_mars_dyn_press.const.dt:
                best_guess_obj_plot_mars_dyn_press.luminosity_arr, best_guess_obj_plot_mars_dyn_press.abs_magnitude = luminosity_integration(
                    best_guess_obj_plot_mars_dyn_press.time_arr, best_guess_obj_plot_mars_dyn_press.time_arr, best_guess_obj_plot_mars_dyn_press.luminosity_arr,
                    best_guess_obj_plot_mars_dyn_press.const.dt, obs_data.fps_lum, obs_data.P_0m
                )
            if (1 / obs_data.fps_lum) > best_guess_obj_plot_mars_energy.const.dt:
                best_guess_obj_plot_mars_energy.luminosity_arr, best_guess_obj_plot_mars_energy.abs_magnitude = luminosity_integration(
                    best_guess_obj_plot_mars_energy.time_arr, best_guess_obj_plot_mars_energy.time_arr, best_guess_obj_plot_mars_energy.luminosity_arr,
                    best_guess_obj_plot_mars_energy.const.dt, obs_data.fps_lum, obs_data.P_0m
                )
            # make a second subplot with the lightcurve against height for mars
            ax.plot(best_guess_obj_plot_mars.abs_magnitude,best_guess_obj_plot_mars.leading_frag_height_arr/1000, color='pink', label='Best Fit Simulation (Mars same $\\rho$)')
            ax.axhline(y=best_guess_obj_plot_mars.const.erosion_height_start/1000, color='pink', linestyle='--')
            ax.plot(best_guess_obj_plot_mars_dyn_press.abs_magnitude,best_guess_obj_plot_mars_dyn_press.leading_frag_height_arr/1000, color='plum', label='Best Fit Simulation (Mars same $p_{dyn}$)')
            ax.axhline(y=heightsame_dynpress_mars/1000, color='plum', linestyle='--')
            ax.plot(best_guess_obj_plot_mars_energy.abs_magnitude,best_guess_obj_plot_mars_energy.leading_frag_height_arr/1000, color='thistle', label='Best Fit Simulation (Mars same $E_{e}$)')
            ax.axhline(y=height_energy_erosion_start_mars/1000, color='thistle', linestyle='--')
            if flag_total_rho:
                ax.axhline(y=best_guess_obj_plot_mars.const.erosion_height_change/1000, color='pink', linestyle='-.')
                ax.axhline(y=heightsame_dynpress_change_mars/1000, color='plum', linestyle='-.')
                ax.axhline(y=height_energy_erosion_change_mars/1000, color='thistle', linestyle='-.')



            ####### plot the single body ablation model on Mars ######
            if (1 / obs_data.fps_lum) > best_guess_obj_plot_single_mars.const.dt:
                best_guess_obj_plot_single_mars.luminosity_arr, best_guess_obj_plot_single_mars.abs_magnitude = luminosity_integration(
                    best_guess_obj_plot_single_mars.time_arr, best_guess_obj_plot_single_mars.time_arr, best_guess_obj_plot_single_mars.luminosity_arr,
                    best_guess_obj_plot_single_mars.const.dt, obs_data.fps_lum, obs_data.P_0m   
                )
            ax.plot(best_guess_obj_plot_single_mars.abs_magnitude,best_guess_obj_plot_single_mars.leading_frag_height_arr/1000, color='peru', linestyle=':', label='Single Body Ablation (Mars)')
        
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
            # put it outside to the top right
            ax.legend(fontsize=10, loc='upper left', bbox_to_anchor=(1.05, 1))
            # ax[1].grid()
            # plt.suptitle(f'Lightcurve Comparison for {base_name}', fontsize=18)
            plt.tight_layout()
            plt.savefig(output_dir + os.sep + base_name + "_Lightcurve_Earth_vs_Mars.png")
            plt.close()

            ### save the results

            # save the 
            file_bright_dict[base_name] = (obs_data.absolute_magnitudes, obs_data.height_lum/ 1000,
                                        # best_guess_obj_plot.abs_magnitude,best_guess_obj_plot.leading_frag_height_arr/1000, 
                                        best_guess_obj_plot_single_mars.abs_magnitude,best_guess_obj_plot_single_mars.leading_frag_height_arr/1000, 
                                        best_guess_obj_plot_mars.abs_magnitude, best_guess_obj_plot_mars.leading_frag_height_arr/1000, 
                                        best_guess_obj_plot_mars_dyn_press.abs_magnitude,best_guess_obj_plot_mars_dyn_press.leading_frag_height_arr/1000, 
                                        best_guess_obj_plot_mars_energy.abs_magnitude,best_guess_obj_plot_mars_energy.leading_frag_height_arr/1000,
                                        best_guess_obj_plot.const,
                                        best_guess_obj_plot_single_mars.const,
                                        best_guess_obj_plot_mars.const,
                                        best_guess_obj_plot_mars_dyn_press.const,
                                        best_guess_obj_plot_mars_energy.const)

            file_rho_jd_dict[base_name] = (rho, rho_lo,rho_hi, tj, tj_lo, tj_hi, inclin_val, Vinf_val, Vg_val, Q_val, q_val, a_val, e_val, V_val_earth, V_val_mars, Vg_val_mars, Vinf_val_mars, Vg_val_mars_min_max, Vinf_val_mars_min_max, Vg_val_denis, Vinf_val_denis)
            # file_eeu_dict[base_name] = (eeucs, eeucs_lo, eeucs_hi, eeum, eeum_lo, eeum_hi,F_par, kc_par, lenght_par)
            file_obs_data_dict[base_name] = (kc_par, F_par, lenght_par, beg_height/1000, end_height/1000, max_lum_height/1000, avg_vel/1000, init_mag, end_mag, max_mag, time_tot, zenith_angle, m_init_meteor_median, meteoroid_diameter_mm, erosion_beg_dyn_press, v_init_meteor_median, tau_median, tau_low95, tau_high95)
            file_phys_data_dict[base_name] = (eta_meteor_begin, sigma_meteor_begin, meteoroid_diameter_mm, meteoroid_diameter_mm_lo, meteoroid_diameter_mm_hi, m_init_meteor_median, m_init_meteor_lo, m_init_meteor_hi)

            # save in .marsmeteor as a pickle file the results for the best guess meteor for earth and mars

            # Save ONLY this meteor's data into its .marsmeteor file
            marsmeteor_path = os.path.join(output_dir, base_name + "_res_mars.marsmeteor")

            single_meteor_payload = {
                "file_bright_dict": {base_name: file_bright_dict[base_name]},
                "file_rho_jd_dict": {base_name: file_rho_jd_dict[base_name]},
                "file_obs_data_dict": {base_name: file_obs_data_dict[base_name]},
                "file_phys_data_dict": {base_name: file_phys_data_dict[base_name]},
            }

            with open(marsmeteor_path, "wb") as f:
                pickle.dump(single_meteor_payload, f)

        else:
            
            # open the .marsmeteor file and load the dictionaries
            marsmeteor_path = os.path.join(output_dir, marsmeteor_file)
            with open(marsmeteor_path, "rb") as f:
                data = pickle.load(f)

            if base_name not in data["file_bright_dict"]:
                raise RuntimeError(
                    f"{marsmeteor_path} does not contain brightness data for {base_name}"
                )
            else:
                print(f"Loading pre-computed Mars data for {base_name} from {marsmeteor_path}")
            
            # Merge into the global dicts instead of overwriting them
            for key, val in data["file_bright_dict"].items():
                file_bright_dict[key] = val

            for key, val in data["file_rho_jd_dict"].items():
                file_rho_jd_dict[key] = val

            for key, val in data["file_obs_data_dict"].items():
                file_obs_data_dict[key] = val

            for key, val in data["file_phys_data_dict"].items():
                file_phys_data_dict[key] = val


        all_names.append(base_name)
        all_samples.append(samples_aligned)
        all_weights.append(weights_aligned)

    # Close the Logger to ensure everything is written to the file STOP COPY in TXT file
    sys.stdout.close()

    # Reset sys.stdout to its original value if needed
    sys.stdout = sys.__stdout__



    ############ PLOTTING SUMMARY ##############
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
            # axs.scatter(Vinf_val_mars, tj, color='red')
        # # add error bars
        # axs.errorbar(Vg_val, tj, xerr=0, yerr=[[tj - tj_lo], [tj_hi - tj]], fmt='o', color='blue', alpha=0.5)
        # axs.errorbar(Vg_val_mars, tj, xerr=0, yerr=[[tj - tj_lo], [tj_hi - tj]], fmt='o', color='red', alpha=0.5)
    # add the labels and legend for a red and blue points
    axs.scatter([], [], color='blue', label='Earth')
    # axs.scatter([], [], color='red', label='Mars 2D')
    # axs.scatter([], [], color='blue', marker='d', label='Earth 3D')
    axs.plot([], [], color='red', marker='d', label='Mars') #  3D
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

    #########################################

    print("\nPlotting the brightness distribution against height...")

    # ---- 1. Collect all magnitudes/heights across events for each method ----

    methods = [
        ("Earth",           "obs"),
        ("Mars single body", "single_mars"),
        ("Mars $\\rho$",       "mars_rho"),
        ("Mars $p_{dyn}$",  "mars_dyn_press"),
        ("Mars $E_{e}$",     "mars_energy"),
    ]

    # Storage for each method
    data_dict = {
        "obs":            {"mag": [], "h": []},
        "single_mars":    {"mag": [], "h": []},
        "mars_rho":           {"mag": [], "h": []},
        "mars_dyn_press": {"mag": [], "h": []},
        "mars_energy":    {"mag": [], "h": []},
    }

    # Fill from your per-meteor dictionary
    for name in all_names:
        (abs_mags_obs, heights_obs,
        abs_mags_single_mars, heights_single_mars,
        abs_mags_mars, heights_mars,
        abs_mags_mars_dyn_press, heights_mars_dyn_press,
        abs_mags_mars_energy, heights_mars_energy, 
        const_obs, const_single_mars, const_mars, const_mars_dyn_press, const_mars_energy) = file_bright_dict[name]

        data_dict["obs"]["mag"].append(abs_mags_obs)
        data_dict["obs"]["h"].append(heights_obs)

        data_dict["single_mars"]["mag"].append(abs_mags_single_mars)
        data_dict["single_mars"]["h"].append(heights_single_mars)

        data_dict["mars_rho"]["mag"].append(abs_mags_mars)
        data_dict["mars_rho"]["h"].append(heights_mars)

        data_dict["mars_dyn_press"]["mag"].append(abs_mags_mars_dyn_press)
        data_dict["mars_dyn_press"]["h"].append(heights_mars_dyn_press)

        data_dict["mars_energy"]["mag"].append(abs_mags_mars_energy)
        data_dict["mars_energy"]["h"].append(heights_mars_energy)

    # Concatenate lists into single arrays per method
    for key in data_dict:
        if len(data_dict[key]["mag"]) > 0:
            data_dict[key]["mag"] = np.concatenate(data_dict[key]["mag"])
            data_dict[key]["h"]   = np.concatenate(data_dict[key]["h"])
        else:
            data_dict[key]["mag"] = np.array([])
            data_dict[key]["h"]   = np.array([])

    # ---- 2. Define common height bins across all methods ----

    all_heights = np.concatenate([data_dict[k]["h"] for k in data_dict if data_dict[k]["h"].size > 0])

    h_min = np.nanmin(all_heights)
    if h_min < 50:
        h_min = 50.0  # limit to 50 km for better visualisation
    # h_max = np.nanmax(all_heights)
    h_max = 120.0  # limit to 120 km for better visualisation

    # Choose how many vertical bins you want
    n_hbins = 40
    height_bins = np.linspace(h_min, h_max, n_hbins + 1)

    # ---- 3. Build brightness density grid [n_methods x n_hbins] using KDE ----

    n_methods = len(methods)
    n_hbins = len(height_bins) - 1

    # Use bin centres for the KDE evaluation points
    h_centers = 0.5 * (height_bins[:-1] + height_bins[1:])

    brightness_grid = np.zeros((n_methods, n_hbins), dtype=float)

    # Gaussian kernel bandwidth in km (tune this if needed)
    bandwidth = 2.0  # smaller -> more detailed, larger -> smoother

    for m_idx, (_, key) in enumerate(methods):
        mags = data_dict[key]["mag"]
        hs   = data_dict[key]["h"]

        # Keep only finite values
        mask = np.isfinite(mags) & np.isfinite(hs)
        mags = mags[mask]
        hs   = hs[mask]

        if mags.size == 0:
            continue

        # Remove extreme outliers outside the plotting range
        in_range = (hs >= h_min) & (hs <= h_max)
        mags = mags[in_range]
        hs   = hs[in_range]

        if mags.size == 0:
            continue

        # Brightness weights (more negative mag -> higher weight)
        weights = 10.0 ** (-0.4 * mags)

        # Brightness-weighted Gaussian KDE in height:
        # at each h_center, sum_j w_j * exp(-0.5 * ((h_center - h_j)/bandwidth)^2)
        diff = h_centers[:, None] - hs[None, :]         # shape: [n_hbins, N_points]
        kern = np.exp(-0.5 * (diff / bandwidth) ** 2)   # Gaussian kernel

        kde_vals = np.sum(kern * weights[None, :], axis=1)  # shape: [n_hbins]

        brightness_grid[m_idx, :] = kde_vals

    # ---- 4. Normalise each column so colour range is comparable ----

    col_max = brightness_grid.max(axis=1, keepdims=True)
    col_max[col_max == 0] = 1.0
    brightness_grid_norm = brightness_grid / col_max  # each method in [0,1]

    # ---- 5. Plot as "5 columns" with inverse viridis ----

    fig, ax = plt.subplots(figsize=(8, 8))

    x_edges = np.arange(n_methods + 1)
    X, Y = np.meshgrid(x_edges, height_bins)

    c = ax.pcolormesh(
        X, Y, brightness_grid_norm.T,
        cmap="viridis", shading="auto"
    )

    ax.set_xticks(np.arange(n_methods) + 0.5)
    ax.set_xticklabels([m[0] for m in methods], rotation=30, ha="right")

    ax.set_ylabel("Height [km]")
    ax.set_ylim(h_min, h_max)

    cb = fig.colorbar(c, ax=ax)
    cb.set_label("Relative density")

    fig.tight_layout()
    plt.savefig(output_dir_show + os.sep + "density_AbsMag_density.png")
    plt.close()

    #############################################################

    print("\nPlotting the brightness distribution median against height...")

    # ---- 3. Compute per-bin magnitudes:
    #   - low/mid: mostly regular median
    #   - high: mostly bright-tail median
    #   with a smooth blend in between ----

    n_methods = len(methods)
    mag_grid        = np.full((n_methods, n_hbins), np.nan, dtype=float)
    count_grid      = np.zeros((n_methods, n_hbins), dtype=int)
    median_grid     = np.full((n_methods, n_hbins), np.nan, dtype=float)
    bright_grid     = np.full((n_methods, n_hbins), np.nan, dtype=float)

    # Height-bin centres
    h_centers = 0.5 * (height_bins[:-1] + height_bins[1:])

    # Define a BLEND REGION in height:
    #  - below h0: pure median
    #  - above h1: pure bright-tail
    #  - between: linear blend
    h0 = h_min + 0.55 * (h_max - h_min)   # start of blend
    h1 = h_min + 0.80 * (h_max - h_min)   # end of blend

    bright_frac = 0.3  # brightest 30% at high altitudes

    for m_idx, (_, key) in enumerate(methods):
        mags = data_dict[key]["mag"]
        hs   = data_dict[key]["h"]

        mask = np.isfinite(mags) & np.isfinite(hs)
        mags = mags[mask]
        hs   = hs[mask]

        if mags.size == 0:
            continue

        # assign each point to a height bin
        bin_idx = np.digitize(hs, height_bins) - 1
        valid   = (bin_idx >= 0) & (bin_idx < n_hbins)

        mags = mags[valid]
        bin_idx = bin_idx[valid]

        for b in range(n_hbins):
            in_bin = (bin_idx == b)
            if not np.any(in_bin):
                continue

            mag_bin = mags[in_bin]
            count_grid[m_idx, b] = mag_bin.size

            # Regular median
            med = np.median(mag_bin)
            median_grid[m_idx, b] = med

            # Bright-tail median (if enough points)
            mag_bin_sorted = np.sort(mag_bin)  # smaller = brighter
            k = int(np.ceil(bright_frac * mag_bin_sorted.size))
            k = max(1, k)
            bright_tail = mag_bin_sorted[:k]
            bright_med = np.median(bright_tail)
            bright_grid[m_idx, b] = bright_med

            # Blend weight based on height
            h_c = h_centers[b]
            if h_c <= h0:
                w = 0.0              # pure median
            elif h_c >= h1:
                w = 1.0              # pure bright-tail
            else:
                w = (h_c - h0) / (h1 - h0)  # linear ramp 0→1

            # Handle NaNs gracefully
            if np.isnan(med) and np.isnan(bright_med):
                mag_grid[m_idx, b] = np.nan
            elif np.isnan(bright_med):
                mag_grid[m_idx, b] = med
            elif np.isnan(med):
                mag_grid[m_idx, b] = bright_med
            else:
                mag_grid[m_idx, b] = (1.0 - w) * med + w * bright_med

    # ---- 3b. Remove low-altitude bins dominated by a tiny fraction of the data ----

    min_rel_count = 0.1  # 10% of the maximum bin population for that method

    for m_idx in range(n_methods):
        col_counts = count_grid[m_idx, :]
        if np.all(col_counts == 0):
            continue

        max_cnt = col_counts.max()
        if max_cnt == 0:
            continue

        dense_bins = np.where(col_counts >= min_rel_count * max_cnt)[0]
        if dense_bins.size == 0:
            continue

        lowest_dense = dense_bins[0]
        low_sparse_bins = np.arange(lowest_dense)
        mag_grid[m_idx, low_sparse_bins] = np.nan
        count_grid[m_idx, low_sparse_bins] = 0

    if np.all(np.isnan(mag_grid)):
        print("No median magnitude bins left after outlier filtering; skipping median plot.")
        return

    # ---- 4. Plot as 5 columns, colour = magnitude ----

    fig, ax = plt.subplots(figsize=(8, 8))

    x_edges = np.arange(n_methods + 1)
    X, Y = np.meshgrid(x_edges, height_bins)

    mag_min = np.nanmin(mag_grid)
    mag_max = 8.0

    c = ax.pcolormesh(
        X, Y, mag_grid.T,
        cmap="viridis_r",
        shading="auto",
        vmin=mag_min,
        vmax=mag_max
    )

    ax.set_xticks(np.arange(n_methods) + 0.5)
    ax.set_xticklabels([m[0] for m in methods], rotation=30, ha="right")

    ax.set_ylabel("Height [km]")
    ax.set_ylim(h_min, h_max)

    cb = fig.colorbar(c, ax=ax)
    cb.set_label("Median Abs.Mag [-]")

    fig.tight_layout()
    plt.savefig(output_dir_show + os.sep + "density_mean_AbsMagHeight.png")
    plt.close()

    ############ PLOTTING SUMMARY ##############
    print("\nPlotting Velocity height with brightness colorbar...")
    # plot the distribution of speed and Vg_val and Vg_val_mars in a single plot one in blue for earth and one in red for mars 
    
    brightest_mags = []
    height_brightest = []
    Earth_height = []
    Earth_single_height = []
    Earth_brightest = []
    vel_brightest = []
    height_type = []
    vel_brightest_single = []
    height_brightest_single = []
    brightest_mags_single = []
    for name in all_names:
        rho, rho_lo, rho_hi, tj, tj_lo, tj_hi, inclin_val, Vinf_val, Vg_val, Q_val, q_val, a_val, e_val, V_val_earth, V_val_mars, Vg_val_mars, Vinf_val_mars, Vg_val_mars_min_max, Vinf_val_mars_min_max, Vg_val_denis, Vinf_val_denis = file_rho_jd_dict[name]
        
        vel_brightest.append(Vinf_val_mars)
        vel_brightest.append(Vinf_val_mars)
        vel_brightest.append(Vinf_val_mars)

        (abs_mags_obs, heights_obs,
        abs_mags_single_mars, heights_single_mars,
        abs_mags_mars, heights_mars,
        abs_mags_mars_dyn_press, heights_mars_dyn_press,
        abs_mags_mars_energy, heights_mars_energy, 
        const_obs, const_single_mars, const_mars, const_mars_dyn_press, const_mars_energy) = file_bright_dict[name]

        min_mag_index_earth = np.argmin(abs_mags_obs)
        Earth_height.append(heights_obs[min_mag_index_earth])
        Earth_height.append(heights_obs[min_mag_index_earth])
        Earth_height.append(heights_obs[min_mag_index_earth])
        Earth_brightest.append(abs_mags_obs[min_mag_index_earth])
        Earth_brightest.append(abs_mags_obs[min_mag_index_earth])
        Earth_brightest.append(abs_mags_obs[min_mag_index_earth])

        Earth_single_height.append(heights_obs[min_mag_index_earth])

        # Earth_brightest.append(abs_mags_obs[min_mag_index_earth])
        # take the brightest abs_mags_mars heights_mars an plot with a colorbar
        min_mag_index = np.argmin(abs_mags_mars)
        brightest_mags.append(abs_mags_mars[min_mag_index])
        height_brightest.append(heights_mars[min_mag_index])
        height_type.append('Mars $\\rho$')

        min_mag_index_dyn = np.argmin(abs_mags_mars_dyn_press)
        brightest_mags.append(abs_mags_mars_dyn_press[min_mag_index_dyn])
        height_brightest.append(heights_mars_dyn_press[min_mag_index_dyn])
        height_type.append('Mars $p_{dyn}$')
        
        min_mag_index_energy = np.argmin(abs_mags_mars_energy)
        brightest_mags.append(abs_mags_mars_energy[min_mag_index_energy])
        height_brightest.append(heights_mars_energy[min_mag_index_energy])
        height_type.append('Mars $E_{e}$')

        vel_brightest_single.append(Vinf_val_mars)
        # clear nan values from heights_single_mars and delete the corresponding abs_mags_single_mars
        # valid_indices = ~np.isnan(heights_single_mars)
        # heights_single_mars = heights_single_mars[valid_indices]
        # abs_mags_single_mars = abs_mags_single_mars[valid_indices]
        min_mag_index_single = np.argmin(abs_mags_single_mars)
        brightest_mags_single.append(abs_mags_single_mars[min_mag_index_single])
        height_brightest_single.append(heights_single_mars[min_mag_index_single])
        # print the single body results
        # print(f"{name} Single body: Vinf: {Vinf_val_mars:.3f} km/s, Height: {heights_single_mars[min_mag_index_single]:.2f} km, Abs.Mag: {abs_mags_single_mars[min_mag_index_single]:.2f} ")
        if np.isnan(heights_single_mars[min_mag_index_single]):
            print(f"{name} Warning: Single body result is NaN, skipping...")
            # # plot abs_mags_single_mars and heights_single_mars just to show they are nan
            # fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            # ax.scatter(abs_mags_single_mars, heights_single_mars, c='black', label='Mars Single Body', s=60)
            # ax.scatter(abs_mags_single_mars[min_mag_index_single], heights_single_mars[min_mag_index_single], c='red', label='Brightest Point', s=100, edgecolors='black')
            # ax.legend(fontsize=12)
            # ax.set_xlabel('$V_{\infty}$ [km/s]', fontsize=12)
            # ax.set_ylabel('Height [km]', fontsize=12)
            # ax.tick_params(axis='both', which='major', labelsize=12)
            # ax.grid()
            # plt.tight_layout()
            # plt.show()

    fig, axs = plt.subplots(1, 1, figsize=(8, 6))
    brightest_mags_arr = np.asarray(brightest_mags, float)

    # Option A: linear scale
    vmin = np.nanpercentile(brightest_mags_arr, 2.5)
    vmax = np.nanpercentile(brightest_mags_arr, 97.5)

    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    sc = None  # to keep a handle for the colorbar
        
    for ht in set(height_type):
        indices = [i for i, x in enumerate(height_type) if x == ht]

        # the color values for these points
        cvals = brightest_mags_arr[indices]         # or log_vals[indices] if using log

        if ht == 'Mars $\\rho$':
            sc = axs.scatter(
                np.array(vel_brightest)[indices],
                np.array(height_brightest)[indices],
                c=cvals,
                cmap='plasma_r',
                norm=norm,
                edgecolors='blue',
                s=60,
                linewidth=1.5,
                label='Mars $\\rho$',
            )

        elif ht == 'Mars $p_{dyn}$':
            sc = axs.scatter(
                np.array(vel_brightest)[indices],
                np.array(height_brightest)[indices],
                c=cvals,
                cmap='plasma_r',
                norm=norm,
                edgecolors='green',
                s=60,
                linewidth=1.5,
                label='Mars $p_{dyn}$',
            )

        elif ht == 'Mars $E_{e}$':
            sc = axs.scatter(
                np.array(vel_brightest)[indices],
                np.array(height_brightest)[indices],
                c=cvals,
                cmap='plasma_r',
                norm=norm,
                edgecolors='red',
                s=60,
                linewidth=1.5,
                label='Mars $E_{e}$',
            )

    # Legend (deduplicate labels)
    handles, labels = axs.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axs.legend(by_label.values(), by_label.keys(), fontsize=15)

    # Single shared colorbar: uses the same norm and cmap as all scatters
    cbar = fig.colorbar(sc, ax=axs)
    cbar.set_label('Abs.Mag [-]', fontsize=12)
    cbar.ax.invert_yaxis()
    cbar.ax.yaxis.set_tick_params(pad=10)
    
    # add the labels and legend for a red and blue points
    axs.set_xlabel('$V_{\infty}$ [km/s]', fontsize=12)
    axs.set_ylabel('Height [km]', fontsize=12) 
    # mak the thiks bigger
    axs.tick_params(axis='both', which='major', labelsize=12)
    # grid on
    axs.grid()
    plt.tight_layout()
    plt.savefig(output_dir_show + os.sep + "Vinf_Height_AbsMag.png")
    plt.close()

    #############################################################

    # create a new figure tat shows the drop in brightness with height for the three eorion types respect to earth
    fig, axs = plt.subplots(1, 1, figsize=(8, 6))
    # find fro each speed the difference between earth and mars brightness in magnitude
    brightness_diff = np.array(Earth_brightest) - np.array(brightest_mags)
    # between max Mars and min Mars speed create a 10 km/s bin and find the mendian and 95 percentile of brightness difference in each bin
    v_bin = 10.0
    v_min = np.min([V_ESC_MARS180, np.min(vel_brightest)])
    v_max = np.max([V_ORBIT_MARS+V_PARAB_MARSORBIT, np.max(vel_brightest)])
    v_bins = np.arange(v_min, v_max + v_bin, v_bin)
    v_bin_centers = 0.5 * (v_bins[:-1] + v_bins[1:])
    median_brightness_diff = []
    p95_brightness_diff = []
    p5_brightness_diff = []
    for i in range(len(v_bins) - 1):
        bin_indices = np.where((np.array(vel_brightest) >= v_bins[i]) & (np.array(vel_brightest) < v_bins[i+1]))[0]
        if len(bin_indices) > 0:
            median_brightness_diff.append(np.median(brightness_diff[bin_indices]))
            p95_brightness_diff.append(np.percentile(brightness_diff[bin_indices], 68))
            p5_brightness_diff.append(np.percentile(brightness_diff[bin_indices], 34))
        else:
            median_brightness_diff.append(np.nan)
            p95_brightness_diff.append(np.nan)
            p5_brightness_diff.append(np.nan)
    axs.plot(v_bin_centers, median_brightness_diff, color='black', label='Median Abs.Mag Difference', linewidth=2)
    axs.fill_between(v_bin_centers, p5_brightness_diff, p95_brightness_diff, color='gray', alpha=0.2, label='1-sigma range') 
    axs.set_xlabel('$V_{\infty}$ [km/s]', fontsize=12)
    axs.set_ylabel('$\Delta$ Abs.Mag', fontsize=12) 
    axs.tick_params(axis='both', which='major', labelsize=12)
    axs.grid()
    axs.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir_show + os.sep + "Vinf_BrightnessDifference_Earth_Mars.png")
    plt.close()


    #############################################################


    print("\nPlotting interpolated Abs.Mag field in V_inf-Height space...")

    # Use the same arrays you already built
    vel_arr = np.asarray(vel_brightest, float)
    h_arr   = np.asarray(height_brightest, float)
    mag_arr = np.asarray(brightest_mags_arr, float)

    # Clean NaNs
    mask = np.isfinite(vel_arr) & np.isfinite(h_arr) & np.isfinite(mag_arr)
    vel_arr = vel_arr[mask]
    h_arr   = h_arr[mask]
    mag_arr = mag_arr[mask]

    # ---- 1) Coarse binning: median Abs.Mag per (V_inf, height) bin ----
    n_v_bins = 25
    n_h_bins = 25

    v_min, v_max = vel_arr.min(), vel_arr.max()
    h_min, h_max = h_arr.min(),  h_arr.max()

    v_edges = np.linspace(v_min, v_max, n_v_bins + 1)
    h_edges = np.linspace(h_min, h_max, n_h_bins + 1)

    # median magnitude per bin
    stat, v_edges_out, h_edges_out, _ = binned_statistic_2d(
        vel_arr, h_arr, mag_arr,
        statistic="median",
        bins=[v_edges, h_edges],
    )

    # how many points per bin (to ignore very empty bins)
    count, _, _, _ = binned_statistic_2d(
        vel_arr, h_arr, mag_arr,
        statistic="count",
        bins=[v_edges, h_edges],
    )

    # transpose to [y, x] for plotting
    stat = stat.T
    count = count.T

    # ignore bins with too few points
    min_pts_per_bin = 2
    stat[count < min_pts_per_bin] = np.nan

    # ---- 2) Nearest–neighbour infill + mild smoothing ----
    nan_mask = np.isnan(stat)

    # nearest-neighbour fill inside data region
    dist, idx = ndimage.distance_transform_edt(
        nan_mask,
        return_distances=True,
        return_indices=True,
    )
    stat_filled = stat[tuple(idx)]

    # don’t extrapolate too far from real data (in bin units)
    max_pix_dist = 5.0  # smaller = less interpolation
    stat_filled[dist > max_pix_dist] = np.nan

    # mild Gaussian blur over bins to avoid blocky look
    stat_smooth = ndimage.gaussian_filter(stat_filled, sigma=0.7)
    stat_smooth = np.ma.masked_invalid(stat_smooth)

    # Use bin *centers* and contourf with many levels

    v_centers = 0.5 * (v_edges_out[:-1] + v_edges_out[1:])
    h_centers = 0.5 * (h_edges_out[:-1] + h_edges_out[1:])
    Vc, Hc = np.meshgrid(v_centers, h_centers)   # shapes match stat_smooth

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # filled contour plot: z = median Abs.Mag
    cont = ax.contourf(
        Vc, Hc, stat_smooth,
        levels=100,                 # like sns.kdeplot levels=100
        cmap='plasma_r',
        vmin=vmin,                  # same vmin/vmax as your scatter
        vmax=vmax,
    )

    # (optional) overlay the points faintly; comment out if you want *only* the contours
    ax.scatter(
        vel_arr,
        h_arr,
        c=mag_arr,
        cmap='plasma_r',
        norm=norm,
        s=15,
        edgecolors='none',
        alpha=0.3,
    )

    # Colorbar: z-axis = Abs.Mag
    cbar = fig.colorbar(cont, ax=ax)
    cbar.set_label('Abs.Mag [-]', fontsize=12)
    cbar.ax.invert_yaxis()                # brighter (more negative) at top
    cbar.ax.yaxis.set_tick_params(pad=10)

    ax.set_xlabel('$V_{\\infty}$ [km/s]', fontsize=12)
    ax.set_ylabel('Height [km]', fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid()

    plt.tight_layout()
    plt.savefig(output_dir_show + os.sep + "Vinf_Height_AbsMag_field_contour.png")
    plt.close()


    ############## Velocity delta height with brightness colorbar ##############


    print("\nPlotting interpolated Abs.Mag field in V_inf-Height space...")

    # Use the same arrays you already built
    vel_arr = np.asarray(vel_brightest, float)
    h_arr   = np.asarray(height_brightest, float)
    mag_arr = np.asarray(brightest_mags_arr, float)

    # Clean NaNs
    mask = np.isfinite(vel_arr) & np.isfinite(h_arr) & np.isfinite(mag_arr)
    vel_arr = vel_arr[mask]
    h_arr   = h_arr[mask]
    mag_arr = mag_arr[mask]

    # ---- 1) Coarse binning: median Abs.Mag per (V_inf, height) bin ----
    # adjust these if you want coarser/finer resolution
    n_v_bins = 25
    n_h_bins = 25

    v_min, v_max = vel_arr.min(), vel_arr.max()
    h_min, h_max = h_arr.min(),  h_arr.max()

    v_edges = np.linspace(v_min, v_max, n_v_bins + 1)
    h_edges = np.linspace(h_min, h_max, n_h_bins + 1)

    # median magnitude per bin
    stat, v_edges_out, h_edges_out, _ = binned_statistic_2d(
        vel_arr, h_arr, mag_arr,
        statistic="median",
        bins=[v_edges, h_edges],
    )

    # how many points per bin (to ignore very empty bins)
    count, _, _, _ = binned_statistic_2d(
        vel_arr, h_arr, mag_arr,
        statistic="count",
        bins=[v_edges, h_edges],
    )

    # transpose to [y, x] for plotting
    stat = stat.T
    count = count.T

    # ignore bins with too few points
    min_pts_per_bin = 2
    stat[count < min_pts_per_bin] = np.nan

    # ---- 2) Nearest–neighbour infill + mild smoothing ----
    nan_mask = np.isnan(stat)

    # nearest-neighbour fill inside data region
    dist, idx = ndimage.distance_transform_edt(
        nan_mask,
        return_distances=True,
        return_indices=True,
    )
    stat_filled = stat[tuple(idx)]

    # don’t extrapolate too far from real data (in bin units)
    max_pix_dist = 3.0  # tweak: smaller = less interpolation
    stat_filled[dist > max_pix_dist] = np.nan

    # mild Gaussian blur over bins to avoid blocky look
    # (small sigma so it’s not too smooth)
    stat_smooth = ndimage.gaussian_filter(stat_filled, sigma=0.7)
    stat_smooth = np.ma.masked_invalid(stat_smooth)

    # ---- 3) KDE-like filled contour plot (Abs.Mag as z) ----
    # Use bin centres for the contour grid
    v_centers = 0.5 * (v_edges_out[:-1] + v_edges_out[1:])
    h_centers = 0.5 * (h_edges_out[:-1] + h_edges_out[1:])
    Vc, Hc = np.meshgrid(v_centers, h_centers)   # shapes match stat_smooth

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # many levels + fill=True → visually similar to sns.kdeplot(..., fill=True, levels=100)
    cont = ax.contourf(
        Vc,
        Hc,
        stat_smooth,
        levels=100,                # like levels=100 in sns.kdeplot
        cmap='plasma_r',
        vmin=vmin,                 # same vmin/vmax/norm as your scatter
        vmax=vmax,
    )

    # OPTIONAL: if you want contour lines on top of the fill
    # ax.contour(Vc, Hc, stat_smooth, levels=10, colors='k', linewidths=0.5, alpha=0.5)

    # Colorbar: values are *absolute magnitude*
    cbar = fig.colorbar(cont, ax=ax)
    cbar.set_label('Abs.Mag [-]', fontsize=12)
    cbar.ax.invert_yaxis()        # brighter (more negative) at the top
    cbar.ax.yaxis.set_tick_params(pad=10)

    ax.set_xlabel('$V_{\\infty}$ [km/s]', fontsize=12)
    ax.set_ylabel('Height [km]', fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid()

    plt.tight_layout()
    plt.savefig(output_dir_show + os.sep + "Vinf_Height_AbsMag_field_contour.png")
    plt.close()


    #############################################################

    print("\nPlotting Velocity delta height with brightness colorbar...")
    # compute the difference in height between earth and mars for the brightest point in mars for each method
    fig, axs = plt.subplots(1, 1, figsize=(8, 6))
    # do the difference in height between earth and mars for the brightest point in mars for each method
    delta_height_brightest = []
    for i in range(len(height_brightest)):
        delta_height_brightest.append(height_brightest[i] - Earth_height[i])
        delta_height_single = np.array(height_brightest_single) - np.array(Earth_single_height)

    # make circles around the points based on the height_type so we have s=60 and a different color for each type
    for ht in set(height_type):
        indices = [i for i, x in enumerate(height_type) if x == ht]
        if ht == 'Mars $\\rho$':
            axs.scatter(
                np.array(vel_brightest)[indices],
                np.array(delta_height_brightest)[indices],
                c=np.array(brightest_mags)[indices],
                cmap='plasma_r',
                s=60,
                edgecolors='blue',
                linewidth=1.5,
                label='Mars $\\rho$',
            )
        elif ht == 'Mars $p_{dyn}$':
            axs.scatter(
                np.array(vel_brightest)[indices],
                np.array(delta_height_brightest)[indices],
                c=np.array(brightest_mags)[indices],
                cmap='plasma_r',
                s=60,
                edgecolors='green',
                linewidth=1.5,
                label='Mars $p_{dyn}$',
            )
        elif ht == 'Mars $E_{e}$':
            axs.scatter(
                np.array(vel_brightest)[indices],
                np.array(delta_height_brightest)[indices],
                c=np.array(brightest_mags)[indices],
                cmap='plasma_r',
                s=60,
                edgecolors='red',
                linewidth=1.5,
                label='Mars $E_{e}$',
            )
            
    # show legend only once for each type
    handles, labels = axs.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axs.legend(by_label.values(), by_label.keys(), fontsize=15)
    cbar = plt.colorbar(sc, ax=axs)
    cbar.set_label('Abs.Mag [-]', fontsize=12)
    cbar.ax.invert_yaxis()
    cbar.ax.yaxis.set_tick_params(pad=10)
    # add the labels and legend for a red and blue points
    axs.set_xlabel('$V_{\infty}$ [km/s]', fontsize=12)
    axs.set_ylabel('$\Delta$ Height [km]', fontsize=12)
    # mak the thiks bigger
    axs.tick_params(axis='both', which='major', labelsize=12)
    # put a thik horizontal line at y=0
    axs.axhline(y=0, color='darkgray', linestyle='-', linewidth=2)
    # put a text at the top of this line saying "Brigtest height on Earth" y =0 and x = min vel_brightest
    axs.text(min(vel_brightest), -1.5, "Brightest height on Earth", color='darkgray', fontsize=12, verticalalignment='bottom', horizontalalignment='left')
    # grid on
    axs.grid()
    plt.tight_layout()
    plt.savefig(output_dir_show + os.sep + "Vinf_DeltaHeight_AbsMag.png")
    plt.close()

    ############## 3 plots showing the distribution Velocity height with brightness with a shared colorbar ##############

    print("\nPlotting Velocity height with brightness colorbar - separate plots...")
    fig, axs = plt.subplots(1, 4, figsize=(16, 6), sharex=True, sharey=True)
    axs = axs.ravel()

    for ht in set(height_type):
        indices = [i for i, x in enumerate(height_type) if x == ht]

        if ht == 'Mars $\\rho$':
            sc = axs[0].scatter(
                np.array(vel_brightest)[indices],
                np.array(height_brightest)[indices],
                c=np.array(brightest_mags)[indices],
                cmap='plasma_r',
                s=40,
                edgecolors='black'#'blue'
            )
            axs[0].set_title('Mars $\\rho$', fontsize=15)

        elif ht == 'Mars $p_{dyn}$':
            sc = axs[1].scatter(
                np.array(vel_brightest)[indices],
                np.array(height_brightest)[indices],
                c=np.array(brightest_mags)[indices],
                cmap='plasma_r',
                s=40,
                edgecolors='black'#'green'
            )
            axs[1].set_title('Mars $p_{dyn}$', fontsize=15)

        elif ht == 'Mars $E_{e}$':
            sc = axs[2].scatter(
                np.array(vel_brightest)[indices],
                np.array(height_brightest)[indices],
                c=np.array(brightest_mags)[indices],
                cmap='plasma_r',
                s=40,
                edgecolors='black'#'red'
            )
            axs[2].set_title('Mars $E_{e}$', fontsize=15)

    # add the single body plot as the last subplot
    sc = axs[3].scatter(
        np.array(vel_brightest_single),
        np.array(height_brightest_single),
        c=np.array(brightest_mags_single),
        cmap='plasma_r',
        s=40,
        edgecolors='black'#'red'
    )
    axs[3].set_title('Mars Single Body', fontsize=15)

    # First tighten only the subplots
    fig.subplots_adjust(right=0.86)  # leave room on the right

    # Create a dedicated axis for the colorbar
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])  # [left, bottom, width, height] in figure coords
    cbar = fig.colorbar(sc, cax=cbar_ax)
    cbar.set_label('Abs.Mag [-]', fontsize=14)
    cbar.ax.invert_yaxis()
    cbar.ax.yaxis.set_tick_params(pad=14, labelsize=14)

    for ax in axs:
        ax.set_xlabel('$V_{\\infty}$ [km/s]', fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.grid()

    axs[0].set_ylabel('Height [km]', fontsize=14)

    plt.savefig(output_dir_show + os.sep + "Vinf_Height_AbsMag_separate.png")
    plt.close()

    ################################################
    # same plots but for delta height
    print("\nPlotting Velocity delta height with brightness colorbar - separate plots...")
    fig, axs = plt.subplots(1, 4, figsize=(16, 6), sharex=True, sharey=True)
    axs = axs.ravel()
    for ht in set(height_type):
        indices = [i for i, x in enumerate(height_type) if x == ht]

        if ht == 'Mars $\\rho$':
            sc = axs[0].scatter(
                np.array(vel_brightest)[indices],
                np.array(delta_height_brightest)[indices],
                c=np.array(brightest_mags)[indices],
                cmap='plasma_r',
                s=40,
                edgecolors='black'#'blue'
            )
            axs[0].set_title('Mars $\\rho$', fontsize=15)

        elif ht == 'Mars $p_{dyn}$':
            sc = axs[1].scatter(
                np.array(vel_brightest)[indices],
                np.array(delta_height_brightest)[indices],
                c=np.array(brightest_mags)[indices],
                cmap='plasma_r',
                s=40,
                edgecolors='black'#'green'
            )
            axs[1].set_title('Mars $p_{dyn}$', fontsize=15)

        elif ht == 'Mars $E_{e}$':
            sc = axs[2].scatter(
                np.array(vel_brightest)[indices],
                np.array(delta_height_brightest)[indices],
                c=np.array(brightest_mags)[indices],
                cmap='plasma_r',
                s=40,
                edgecolors='black'#'red'
            )
            axs[2].set_title('Mars $E_{e}$', fontsize=15)
    
    # add the single body plot as the last subplot
    sc = axs[3].scatter(
        np.array(vel_brightest_single),
        np.array(delta_height_single),
        c=np.array(brightest_mags_single),
        cmap='plasma_r',
        s=40,
        edgecolors='black'#'red'
    )
    axs[3].set_title('Mars Single Body', fontsize=15)

    # First tighten only the subplots
    fig.subplots_adjust(right=0.86)  # leave room on the right
    # Create a dedicated axis for the colorbar
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])  # [left, bottom, width, height] in figure coords
    cbar = fig.colorbar(sc, cax=cbar_ax)
    cbar.set_label('Abs.Mag [-]', fontsize=14)
    cbar.ax.invert_yaxis()
    cbar.ax.yaxis.set_tick_params(pad=14, labelsize=14)
    # put the horizontal line at y=0 in each subplot
    for ax in axs:
        ax.axhline(0, color='darkgray', linestyle='-', linewidth=2)
        ax.set_xlabel('$V_{\\infty}$ [km/s]', fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.grid()
    # only on the first subplot add the ylabel
    axs[0].set_ylabel('$\Delta$ Height [km]', fontsize=14)
    # only on the first sublopt wite that that is the brightest height put it below the
    axs[0].text(min(vel_brightest), -1.5, "Brightest height on Earth", color='gray', fontsize=12, verticalalignment='bottom', horizontalalignment='left')
    plt.savefig(output_dir_show + os.sep + "Vinf_DeltaHeight_AbsMag_separate.png")
    plt.close()

    #############################################################

    print("\nPlotting median Abs.Mag vs height for Earth and Mars (brightest erosion cases)...")

    Earth_heights = []
    Earth_mags    = []

    Mars_heights  = []
    Mars_mags     = []

    for name in all_names:
        (
            abs_mags_obs, heights_obs,
            abs_mags_single_mars, heights_single_mars,
            abs_mags_mars, heights_mars,
            abs_mags_mars_dyn_press, heights_mars_dyn_press,
            abs_mags_mars_energy, heights_mars_energy,
            const_obs, const_single_mars, const_mars,
            const_mars_dyn_press, const_mars_energy
        ) = file_bright_dict[name]

        # --- Earth: brightest point (min Abs.Mag) along the Earth curve ---
        if len(abs_mags_obs) > 0:
            idx_e = np.argmin(abs_mags_obs)
            Earth_heights.append(heights_obs[idx_e])
            Earth_mags.append(abs_mags_obs[idx_e])

        # --- Mars: brightest over all erosion cases (rho, p_dyn, E_e), excluding single body ---
        mars_candidates_mags    = []
        mars_candidates_heights = []

        if len(abs_mags_mars) > 0:
            idx_m = np.argmin(abs_mags_mars)
            mars_candidates_mags.append(abs_mags_mars[idx_m])
            mars_candidates_heights.append(heights_mars[idx_m])

        if len(abs_mags_mars_dyn_press) > 0:
            idx_dp = np.argmin(abs_mags_mars_dyn_press)
            mars_candidates_mags.append(abs_mags_mars_dyn_press[idx_dp])
            mars_candidates_heights.append(heights_mars_dyn_press[idx_dp])

        if len(abs_mags_mars_energy) > 0:
            idx_en = np.argmin(abs_mags_mars_energy)
            mars_candidates_mags.append(abs_mags_mars_energy[idx_en])
            mars_candidates_heights.append(heights_mars_energy[idx_en])

        # pick the brightest among the three erosion cases, if any exist
        if mars_candidates_mags:
            j = int(np.argmin(mars_candidates_mags))
            Mars_mags.append(mars_candidates_mags[j])
            Mars_heights.append(mars_candidates_heights[j])

    Earth_heights = np.asarray(Earth_heights, float)
    Earth_mags    = np.asarray(Earth_mags, float)
    Mars_heights  = np.asarray(Mars_heights, float)
    Mars_mags     = np.asarray(Mars_mags, float)

    # Mask NaNs just in case
    mask_e = np.isfinite(Earth_heights) & np.isfinite(Earth_mags)
    Earth_heights = Earth_heights[mask_e]
    Earth_mags    = Earth_mags[mask_e]

    mask_m = np.isfinite(Mars_heights) & np.isfinite(Mars_mags)
    Mars_heights = Mars_heights[mask_m]
    Mars_mags    = Mars_mags[mask_m]

    # --- Bin by height and compute median + 1σ/2σ bands in each bin for Earth and Mars ---
    n_h_bins = 20

    h_min = min(Earth_heights.min(), Mars_heights.min())
    h_max = max(Earth_heights.max(), Mars_heights.max())

    h_edges   = np.linspace(h_min, h_max, n_h_bins + 1)
    h_centers = 0.5 * (h_edges[:-1] + h_edges[1:])

    # Gaussian-equivalent quantiles
    q_med = 0.50
    q_1lo, q_1hi = 0.158655, 0.841345   # ~ ±1σ
    q_2lo, q_2hi = 0.022750, 0.977250   # ~ ±2σ

    Earth_q50 = np.full(n_h_bins, np.nan, dtype=float)
    Earth_q1L = np.full(n_h_bins, np.nan, dtype=float)
    Earth_q1H = np.full(n_h_bins, np.nan, dtype=float)
    Earth_q2L = np.full(n_h_bins, np.nan, dtype=float)
    Earth_q2H = np.full(n_h_bins, np.nan, dtype=float)

    Mars_q50  = np.full(n_h_bins, np.nan, dtype=float)
    Mars_q1L  = np.full(n_h_bins, np.nan, dtype=float)
    Mars_q1H  = np.full(n_h_bins, np.nan, dtype=float)
    Mars_q2L  = np.full(n_h_bins, np.nan, dtype=float)
    Mars_q2H  = np.full(n_h_bins, np.nan, dtype=float)

    for i in range(n_h_bins):
        lo, hi = h_edges[i], h_edges[i+1]

        # Earth bin
        m_e = (Earth_heights >= lo) & (Earth_heights < hi)
        if np.any(m_e):
            vals = Earth_mags[m_e]
            Earth_q50[i] = np.quantile(vals, q_med)
            Earth_q1L[i] = np.quantile(vals, q_1lo)  # brighter side (lower mag)
            Earth_q1H[i] = np.quantile(vals, q_1hi)  # dimmer side (higher mag)
            Earth_q2L[i] = np.quantile(vals, q_2lo)
            Earth_q2H[i] = np.quantile(vals, q_2hi)

        # Mars bin
        m_m = (Mars_heights >= lo) & (Mars_heights < hi)
        if np.any(m_m):
            vals = Mars_mags[m_m]
            Mars_q50[i] = np.quantile(vals, q_med)
            Mars_q1L[i] = np.quantile(vals, q_1lo)
            Mars_q1H[i] = np.quantile(vals, q_1hi)
            Mars_q2L[i] = np.quantile(vals, q_2lo)
            Mars_q2H[i] = np.quantile(vals, q_2hi)

    # --- Plot: median + 1σ/2σ shaded bands ---
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))

    valid_e = np.isfinite(Earth_q50) & np.isfinite(Earth_q1L) & np.isfinite(Earth_q1H) & np.isfinite(Earth_q2L) & np.isfinite(Earth_q2H)
    valid_m = np.isfinite(Mars_q50)  & np.isfinite(Mars_q1L)  & np.isfinite(Mars_q1H)  & np.isfinite(Mars_q2L)  & np.isfinite(Mars_q2H)

    # delete the one with the lowest height (your existing logic)
    if np.any(valid_m):
        valid_m[np.argmin(h_centers[valid_m])] = False

    # Colors + alpha (tune these)
    earth_col = "tab:blue"
    mars_col  = "tab:orange"
    alpha_2s = 0.18   # lighter band
    alpha_1s = 0.35   # darker band

    # Earth bands (x = magnitude range, y = height)
    ax.fill_betweenx(
        h_centers[valid_e], Earth_q2L[valid_e], Earth_q2H[valid_e],
        color=earth_col, alpha=alpha_2s, linewidth=0, label="Earth ±2σ"
    )
    ax.fill_betweenx(
        h_centers[valid_e], Earth_q1L[valid_e], Earth_q1H[valid_e],
        color=earth_col, alpha=alpha_1s, linewidth=0, label="Earth ±1σ"
    )
    ax.plot(
        Earth_q50[valid_e], h_centers[valid_e],
        marker="o", linestyle="-", color=earth_col,
        label="Earth median"
    )

    # Mars bands
    ax.fill_betweenx(
        h_centers[valid_m], Mars_q2L[valid_m], Mars_q2H[valid_m],
        color=mars_col, alpha=alpha_2s, linewidth=0, label="Mars ±2σ"
    )
    ax.fill_betweenx(
        h_centers[valid_m], Mars_q1L[valid_m], Mars_q1H[valid_m],
        color=mars_col, alpha=alpha_1s, linewidth=0, label="Mars ±1σ"
    )
    ax.plot(
        Mars_q50[valid_m], h_centers[valid_m],
        marker="s", linestyle="-", color=mars_col,
        label="Mars median"
    )

    # Magnitude: brighter = more negative; invert so brighter appears to the right
    ax.invert_xaxis()

    ax.set_xlabel("Abs.Mag [-]", fontsize=12)
    ax.set_ylabel("Height [km]", fontsize=12)
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.grid(True)

    # Optional: reduce legend clutter by keeping only median labels
    # (comment out if you want all band labels)
    handles, labels = ax.get_legend_handles_labels()
    keep = [i for i, lab in enumerate(labels) if ("median" in lab)]
    ax.legend([handles[i] for i in keep], [labels[i] for i in keep], fontsize=11)

    plt.tight_layout()
    plt.savefig(output_dir_show + os.sep + "Median_AbsMag_vs_Height_Earth_Mars_uncert.png", dpi=200)
    plt.close()


    #############################################################

    print("\nPlotting Earth vs Mars median brightness vs height (pcolormesh style)...")

    # ---------- 1. Build per-event Earth & Mars "brightest" samples ----------

    Earth_heights = []
    Earth_mags    = []

    Mars_heights  = []
    Mars_mags     = []

    for name in all_names:
        (
            abs_mags_obs, heights_obs,
            abs_mags_single_mars, heights_single_mars,
            abs_mags_mars, heights_mars,
            abs_mags_mars_dyn_press, heights_mars_dyn_press,
            abs_mags_mars_energy, heights_mars_energy,
            const_obs, const_single_mars, const_mars,
            const_mars_dyn_press, const_mars_energy
        ) = file_bright_dict[name]

        # --- Earth: brightest point along the Earth light curve ---
        if len(abs_mags_obs) > 0:
            idx_e = np.argmin(abs_mags_obs)
            Earth_heights.append(heights_obs[idx_e])
            Earth_mags.append(abs_mags_obs[idx_e])

        # --- Mars: brightest among the three erosion cases (rho, p_dyn, E_e) ---
        mars_cand_mags    = []
        mars_cand_heights = []

        if len(abs_mags_mars) > 0:
            idx_m = np.argmin(abs_mags_mars)
            mars_cand_mags.append(abs_mags_mars[idx_m])
            mars_cand_heights.append(heights_mars[idx_m])

        if len(abs_mags_mars_dyn_press) > 0:
            idx_dp = np.argmin(abs_mags_mars_dyn_press)
            mars_cand_mags.append(abs_mags_mars_dyn_press[idx_dp])
            mars_cand_heights.append(heights_mars_dyn_press[idx_dp])

        if len(abs_mags_mars_energy) > 0:
            idx_en = np.argmin(abs_mags_mars_energy)
            mars_cand_mags.append(abs_mags_mars_energy[idx_en])
            mars_cand_heights.append(heights_mars_energy[idx_en])

        if mars_cand_mags:
            j = int(np.argmin(mars_cand_mags))
            Mars_mags.append(mars_cand_mags[j])
            Mars_heights.append(mars_cand_heights[j])

    Earth_heights = np.asarray(Earth_heights, float)
    Earth_mags    = np.asarray(Earth_mags, float)
    Mars_heights  = np.asarray(Mars_heights, float)
    Mars_mags     = np.asarray(Mars_mags, float)

    # Mask NaNs
    mask_e = np.isfinite(Earth_heights) & np.isfinite(Earth_mags)
    Earth_heights = Earth_heights[mask_e]
    Earth_mags    = Earth_mags[mask_e]

    mask_m = np.isfinite(Mars_heights) & np.isfinite(Mars_mags)
    Mars_heights = Mars_heights[mask_m]
    Mars_mags    = Mars_mags[mask_m]

    # ---------- 2. Put into "methods/data_dict" format like your existing code ----------

    methods = [
        ("Earth", "earth"),
        ("Mars",  "mars"),
    ]

    data_dict = {
        "earth": {
            "h":   Earth_heights,
            "mag": Earth_mags,
        },
        "mars": {
            "h":   Mars_heights,
            "mag": Mars_mags,
        },
    }

    # ---------- 3. Define height bins and grids ----------

    # Use combined min/max so both series share the same vertical range
    h_min = min(Earth_heights.min(), Mars_heights.min())
    h_max = max(Earth_heights.max(), Mars_heights.max())

    n_methods = len(methods)
    n_hbins   = 30  # tweak as you like

    height_bins = np.linspace(h_min, h_max, n_hbins + 1)
    h_centers   = 0.5 * (height_bins[:-1] + height_bins[1:])

    mag_grid   = np.full((n_methods, n_hbins), np.nan, dtype=float)
    count_grid = np.zeros((n_methods, n_hbins), dtype=int)

    # ---------- 4. Compute median magnitude per (method, height bin) ----------

    for m_idx, (_, key) in enumerate(methods):
        mags = data_dict[key]["mag"]
        hs   = data_dict[key]["h"]

        mask = np.isfinite(mags) & np.isfinite(hs)
        mags = mags[mask]
        hs   = hs[mask]

        if mags.size == 0:
            continue

        bin_idx = np.digitize(hs, height_bins) - 1
        valid   = (bin_idx >= 0) & (bin_idx < n_hbins)

        mags    = mags[valid]
        bin_idx = bin_idx[valid]

        for b in range(n_hbins):
            in_bin = (bin_idx == b)
            if not np.any(in_bin):
                continue

            mag_bin = mags[in_bin]
            count_grid[m_idx, b] = mag_bin.size
            mag_grid[m_idx, b]   = np.median(mag_bin)

    # ---------- 5. Optional: remove very sparse low-altitude bins (like your code) ----------

    min_rel_count = 0.1  # 10% of max population for that method

    for m_idx in range(n_methods):
        col_counts = count_grid[m_idx, :]
        if np.all(col_counts == 0):
            continue

        max_cnt = col_counts.max()
        if max_cnt == 0:
            continue

        dense_bins = np.where(col_counts >= min_rel_count * max_cnt)[0]
        if dense_bins.size == 0:
            continue

        lowest_dense = dense_bins[0]
        low_sparse_bins = np.arange(lowest_dense)
        mag_grid[m_idx, low_sparse_bins]   = np.nan
        count_grid[m_idx, low_sparse_bins] = 0

    if np.all(np.isnan(mag_grid)):
        print("No median magnitude bins left after filtering; skipping Earth/Mars median plot.")
    else:
        # ---------- 6. Plot as 2 columns (Earth, Mars), colour = median Abs.Mag ----------

        fig, ax = plt.subplots(figsize=(6, 8))

        x_edges = np.arange(n_methods + 1)  # 0..2
        X, Y = np.meshgrid(x_edges, height_bins)

        mag_min = np.nanmin(mag_grid)
        mag_max = 8.0  # or np.nanmax(mag_grid), or a fixed value you prefer

        c = ax.pcolormesh(
            X, Y, mag_grid.T,
            cmap="viridis_r",
            shading="auto",
            vmin=mag_min,
            vmax=mag_max,
        )

        ax.set_xticks(np.arange(n_methods) + 0.5)
        ax.set_xticklabels([m[0] for m in methods], rotation=0, ha="center")

        ax.set_ylabel("Height [km]")
        ax.set_ylim(h_min, h_max)

        cb = fig.colorbar(c, ax=ax)
        cb.set_label("Median Abs.Mag [-]")
        # invert the colorbar so that brighter (more negative) is at the top
        cb.ax.invert_yaxis()

        fig.tight_layout()
        plt.savefig(output_dir_show + os.sep + "Median_AbsMagHeight_Earth_Mars.png")
        plt.close()




    ###########################################################

    print("\nPlotting the brightness distribution MAX against height...")

    # Use combined min/max so both series share the same vertical range
    h_min = 50
    h_max = 120

    n_methods = len(methods)
    n_hbins   = 40  # tweak as you like

    height_bins = np.linspace(h_min, h_max, n_hbins + 1)

    def _to_1d_float(x):
        """Deep-flatten any nested list/array structure into a 1D float array."""
        if x is None:
            return np.empty(0, dtype=float)

        arr = np.asarray(x)

        if arr.size == 0:
            return np.empty(0, dtype=float)

        if arr.dtype == object:
            parts = []
            for item in arr.ravel():
                p = _to_1d_float(item)
                if p.size:
                    parts.append(p)
            return np.concatenate(parts) if parts else np.empty(0, dtype=float)

        return arr.astype(float, copy=False).ravel()


    def _flatten_pairs(mag_list, h_list):
        mags_all, hs_all = [], []
        for m, h in zip(mag_list, h_list):
            m = _to_1d_float(m)
            h = _to_1d_float(h)

            n = min(m.size, h.size)
            if n == 0:
                continue

            mags_all.append(m[:n])
            hs_all.append(h[:n])

        if mags_all:
            return np.concatenate(mags_all).astype(float, copy=False), np.concatenate(hs_all).astype(float, copy=False)

        return np.empty(0, dtype=float), np.empty(0, dtype=float)
    

    methods = [
        ("Earth", "obs"),
        ("Mars", "mars"),
    ]

    # Storage for each method
    data_dict = {
        "obs":            {"mag": [], "h": []},
        "mars":    {"mag": [], "h": []},
    }

    for name in all_names:
        (abs_mags_obs, heights_obs,
        abs_mags_single_mars, heights_single_mars,
        abs_mags_mars, heights_mars,
        abs_mags_mars_dyn_press, heights_mars_dyn_press,
        abs_mags_mars_energy, heights_mars_energy,
        const_obs, const_single_mars, const_mars, const_mars_dyn_press, const_mars_energy) = file_bright_dict[name]

        # Earth (deep-flatten + align)
        m_obs = _to_1d_float(abs_mags_obs)
        h_obs = _to_1d_float(heights_obs)
        n = min(m_obs.size, h_obs.size)
        data_dict["obs"]["mag"].append(m_obs[:n])
        data_dict["obs"]["h"].append(h_obs[:n])

        # Mars (deep-flatten each component, then concatenate)
        m1 = _to_1d_float(abs_mags_mars)
        m2 = _to_1d_float(abs_mags_mars_dyn_press)
        m3 = _to_1d_float(abs_mags_mars_energy)

        h1 = _to_1d_float(heights_mars)
        h2 = _to_1d_float(heights_mars_dyn_press)
        h3 = _to_1d_float(heights_mars_energy)

        mars_mags    = np.concatenate([m1, m2, m3])
        mars_heights = np.concatenate([h1, h2, h3])

        n = min(mars_mags.size, mars_heights.size)
        data_dict["mars"]["mag"].append(mars_mags[:n])
        data_dict["mars"]["h"].append(mars_heights[:n])


    # ---- 3. Compute "most likely" magnitude in each bin: median mag ----

    n_methods = len(methods)
    mag_grid = np.full((n_methods, n_hbins), np.nan, dtype=float)  # store median mags
    count_grid = np.zeros((n_methods, n_hbins), dtype=int)         # how many samples per bin (optional)

    for m_idx, (_, key) in enumerate(methods):
        mags, hs = _flatten_pairs(data_dict[key]["mag"], data_dict[key]["h"])

        mask = np.isfinite(mags) & np.isfinite(hs)
        mags = mags[mask]
        hs   = hs[mask]

        if mags.size == 0:
            continue

        bin_idx = np.digitize(hs, height_bins) - 1
        valid   = (bin_idx >= 0) & (bin_idx < n_hbins)

        mags    = mags[valid]
        bin_idx = bin_idx[valid]

        for b in range(n_hbins):
            in_bin = (bin_idx == b)
            if not np.any(in_bin):
                continue

            mag_bin = mags[in_bin]
            mag_grid[m_idx, b] = np.quantile(mag_bin, 0.05)
            count_grid[m_idx, b] = mag_bin.size


    # ---- 3b. Remove low-altitude bins dominated by a tiny fraction of the data ----

    min_rel_count = 0.1  # 10% of the maximum bin population for that method

    for m_idx in range(n_methods):
        col_counts = count_grid[m_idx, :]
        if np.all(col_counts == 0):
            continue

        max_cnt = col_counts.max()
        if max_cnt == 0:
            continue

        dense_bins = np.where(col_counts >= min_rel_count * max_cnt)[0]
        if dense_bins.size == 0:
            continue

        lowest_dense = dense_bins[0]
        low_sparse_bins = np.arange(lowest_dense)
        mag_grid[m_idx, low_sparse_bins] = np.nan
        count_grid[m_idx, low_sparse_bins] = 0

    if np.all(np.isnan(mag_grid)):
        print("No magnitude bins left after outlier filtering; skipping median plot.")
        return

    # ---- 4. Plot as 5 columns, colour = "bright-tail" magnitude ----

    fig, ax = plt.subplots(figsize=(5, 8))

    x_edges = np.arange(n_methods + 1)
    X, Y = np.meshgrid(x_edges, height_bins)

    mag_min = 1 # np.nanmin(mag_grid)
    mag_max = 8.0

    c = ax.pcolormesh(
        X, Y, mag_grid.T,
        cmap="plasma_r",
        shading="auto",
        vmin=mag_min,
        vmax=mag_max
    )

    ax.set_xticks(np.arange(n_methods) + 0.5)
    ax.set_xticklabels([m[0] for m in methods], rotation=30, ha="right")

    ax.set_ylabel("Height [km]", fontsize=14)
    ax.set_ylim(h_min, h_max)

    cb = fig.colorbar(c, ax=ax)
    cb.set_label("Bright-tail Abs.Mag [-]", fontsize=16)
    # invet the color bar so that brighter (more negative) is at the top
    cb.ax.invert_yaxis()

    # inrse the size of the ticks
    cb.ax.tick_params(labelsize=14)
    # inease te label size x
    ax.tick_params(axis='x', which='major', labelsize=14)
    ax.tick_params(axis='y', which='major', labelsize=14)

    fig.tight_layout()
    plt.savefig(output_dir_show + os.sep + "density_MAX_AbsMagHeight_Earth_Mars.png")
    plt.close()

    ################################################

    print("\nPlotting median Abs.Mag vs height for Earth and Mars (all data)...")
        
    # Flatten list-of-arrays into 1D arrays (and keep mag/height paired per event)
    Earth_mags,  Earth_heights = _flatten_pairs(data_dict["obs"]["mag"],  data_dict["obs"]["h"])
    Mars_mags,   Mars_heights  = _flatten_pairs(data_dict["mars"]["mag"], data_dict["mars"]["h"])

    # Mask NaNs just in case
    mask_e = np.isfinite(Earth_heights) & np.isfinite(Earth_mags)
    Earth_heights = Earth_heights[mask_e]
    Earth_mags    = Earth_mags[mask_e]
    # delete anything that is above 8
    mask_e = Earth_mags <= 5
    Earth_heights = Earth_heights[mask_e]
    Earth_mags    = Earth_mags[mask_e]

    mask_m = np.isfinite(Mars_heights) & np.isfinite(Mars_mags)
    Mars_heights = Mars_heights[mask_m]
    Mars_mags    = Mars_mags[mask_m]
    mask_m = Mars_mags <= 5
    Mars_heights = Mars_heights[mask_m]
    Mars_mags    = Mars_mags[mask_m]

    # --- Bin by height and compute median + 1σ/2σ bands in each bin for Earth and Mars ---
    n_h_bins = 50

    h_min = 60
    h_max = max(Earth_heights.max(), Mars_heights.max())

    h_edges   = np.linspace(h_min, h_max, n_h_bins + 1)
    h_centers = 0.5 * (h_edges[:-1] + h_edges[1:])

    # Gaussian-equivalent quantiles
    q_med = 0.50
    q_1lo, q_1hi = 0.158655, 0.841345   # ~ ±1σ
    q_2lo, q_2hi = 0.022750, 0.977250   # ~ ±2σ

    Earth_q50 = np.full(n_h_bins, np.nan, dtype=float)
    Earth_q1L = np.full(n_h_bins, np.nan, dtype=float)
    Earth_q1H = np.full(n_h_bins, np.nan, dtype=float)
    Earth_q2L = np.full(n_h_bins, np.nan, dtype=float)
    Earth_q2H = np.full(n_h_bins, np.nan, dtype=float)

    Mars_q50  = np.full(n_h_bins, np.nan, dtype=float)
    Mars_q1L  = np.full(n_h_bins, np.nan, dtype=float)
    Mars_q1H  = np.full(n_h_bins, np.nan, dtype=float)
    Mars_q2L  = np.full(n_h_bins, np.nan, dtype=float)
    Mars_q2H  = np.full(n_h_bins, np.nan, dtype=float)

    for i in range(n_h_bins):
        lo, hi = h_edges[i], h_edges[i+1]

        # Earth bin
        m_e = (Earth_heights >= lo) & (Earth_heights < hi)
        if np.any(m_e):
            vals = Earth_mags[m_e]
            Earth_q50[i] = np.quantile(vals, q_med)
            Earth_q1L[i] = np.quantile(vals, q_1lo)  # brighter side (lower mag)
            Earth_q1H[i] = np.quantile(vals, q_1hi)  # dimmer side (higher mag)
            Earth_q2L[i] = np.quantile(vals, q_2lo)
            Earth_q2H[i] = np.quantile(vals, q_2hi)

        # Mars bin
        m_m = (Mars_heights >= lo) & (Mars_heights < hi)
        if np.any(m_m):
            vals = Mars_mags[m_m]
            Mars_q50[i] = np.quantile(vals, q_med)
            Mars_q1L[i] = np.quantile(vals, q_1lo)
            Mars_q1H[i] = np.quantile(vals, q_1hi)
            Mars_q2L[i] = np.quantile(vals, q_2lo)
            Mars_q2H[i] = np.quantile(vals, q_2hi)

    # --- Plot: median + 1σ/2σ shaded bands ---
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))

    valid_e = np.isfinite(Earth_q50) & np.isfinite(Earth_q1L) & np.isfinite(Earth_q1H) & np.isfinite(Earth_q2L) & np.isfinite(Earth_q2H)
    valid_m = np.isfinite(Mars_q50)  & np.isfinite(Mars_q1L)  & np.isfinite(Mars_q1H)  & np.isfinite(Mars_q2L)  & np.isfinite(Mars_q2H)

    # delete the one with the lowest height (your existing logic)
    if np.any(valid_m):
        valid_e[np.argmin(h_centers[valid_e])] = False
        valid_m[np.argmin(h_centers[valid_m])] = False
    
    # put to false all the onest below 75 km for valid_e
    if np.any(valid_e):
        valid_e[h_centers < 75] = False

    # Colors + alpha (tune these)
    earth_col = "blue" # "tab:blue"
    mars_col  = "red" # "tab:orange"
    alpha_2s = 0.18   # lighter band
    alpha_1s = 0.35   # darker band

    # # Earth bands (x = magnitude range, y = height)
    # ax.fill_betweenx(
    #     h_centers[valid_e], Earth_q2L[valid_e], Earth_q2H[valid_e],
    #     color=earth_col, alpha=alpha_2s, linewidth=0, label="Earth ±2σ"
    # )
    # ax.fill_betweenx(
    #     h_centers[valid_e], Earth_q1L[valid_e], Earth_q1H[valid_e],
    #     color=earth_col, alpha=alpha_1s, linewidth=0, label="Earth ±1σ"
    # )
    ax.plot(
        Earth_q50[valid_e], h_centers[valid_e], # marker="o",
         linestyle="-", color=earth_col, linewidth=2,
        label="Earth median"
    )

    # # Mars bands
    # ax.fill_betweenx(
    #     h_centers[valid_m], Mars_q2L[valid_m], Mars_q2H[valid_m],
    #     color=mars_col, alpha=alpha_2s, linewidth=0, label="Mars ±2σ"
    # )
    # ax.fill_betweenx(
    #     h_centers[valid_m], Mars_q1L[valid_m], Mars_q1H[valid_m],
    #     color=mars_col, alpha=alpha_1s, linewidth=0, label="Mars ±1σ"
    # )
    ax.plot(
        Mars_q50[valid_m], h_centers[valid_m], # marker="d", 
        linestyle="-", color=mars_col, linewidth=2,
        label="Mars median"
    )
    
    # I need a cover letter where I express strong interest for this position mention that I am an Aerospce engineer and I have a Master where for my thesis  work extesivelly with MASTER and DRAMA and even the mre preise rentry mel SCARAB. I now work on this researc funded by  both NASA and ESA to chearterize the reis of impcat wit meteroid in order to better. mention that I work extesivlly with Python thought my studies andesearch.
    
    # Magnitude: brighter = more negative; invert so brighter appears to the right
    ax.invert_xaxis()

    ax.set_xlabel("Abs.Mag [-]", fontsize=14)
    ax.set_ylabel("Height [km]", fontsize=14)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.grid(True)

    # Optional: reduce legend clutter by keeping only median labels
    # (comment out if you want all band labels)
    handles, labels = ax.get_legend_handles_labels()
    keep = [i for i, lab in enumerate(labels) if ("median" in lab)]
    ax.legend([handles[i] for i in keep], [labels[i] for i in keep], fontsize=14)
    # put the ticks to 14
    ax.tick_params(axis="both", which="major", labelsize=14)

    plt.tight_layout()
    plt.savefig(output_dir_show + os.sep + "Median_AbsMag_vs_Height_Earth_Mars_uncert_all.png", dpi=200)
    plt.close()


if __name__ == "__main__":

    import argparse
    ### COMMAND LINE ARGUMENTS
    arg_parser = argparse.ArgumentParser(description="Run dynesty with optional .prior file.")
    
    arg_parser.add_argument('--input_dir', metavar='INPUT_PATH', type=str,
                            # "C:\Users\maxiv\Documents\UWO\Papers\3)Sporadics\Results\Sporadics_rho-uniform\Best_irons",
                            # "C:\Users\maxiv\Documents\UWO\Papers\3)Sporadics\Results\Sporadics_rho-uniform\Fastsporad_CAMOnew+EMCCD_unif_density\fastsporad_EMCCD+CAMO_CAP"
                            # "C:\Users\maxiv\Documents\UWO\Papers\3)Sporadics\Results\Sporadics_rho-uniform"
                            # "C:\Users\maxiv\Documents\UWO\Papers\4)Mars meteors\Results\CAMO-EMCCDonly"
        default=r"C:\Users\maxiv\Documents\UWO\Papers\3)Sporadics\Results\Sporadics_rho-uniform",
        help="Path to walk and find .pickle files.")
    
    arg_parser.add_argument('--output_dir', metavar='OUTPUT_DIR', type=str,
        default=r"C:\Users\maxiv\Documents\UWO\Papers\4)Mars meteors\Results\Sporaics_Mars",
        help="Output directory, if not given is the same as input_dir.")
    
    arg_parser.add_argument('--name', metavar='NAME', type=str,
        default=r"",
        help="Name of the input files, if not given is folders name.")

    arg_parser.add_argument('-new','--new_marsmeteor',
        help="recompute the .marsmeteor files even if they exist.",
        action="store_true")

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

    Mars_distrb_plot(cml_args.input_dir, cml_args.output_dir, cml_args.name, cml_args.new_marsmeteor) # cml_args.radiance_plot cml_args.correl_plot
