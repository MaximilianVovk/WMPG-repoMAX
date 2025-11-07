"""
Import all the pickle files and extract physicla prap and other properties from the dynesty files.

Author: Maximilian Vovk
Date: 2025-06-11
"""

# main.py (inside my_subfolder)
import sys
import os

# Add the parent directory to the sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from DynNestSapl_metsim import *

from scipy.stats import gaussian_kde
from dynesty import utils as dyfunc
from matplotlib.ticker import FormatStrFormatter
import itertools
from dynesty.utils import quantile as _quantile
from scipy.ndimage import gaussian_filter as norm_kde
import matplotlib.cm as cm
from matplotlib import ticker as mticker
from matplotlib.colors import Normalize
from dynesty import utils as dyfunc
from matplotlib.ticker import MaxNLocator, NullLocator, ScalarFormatter
from scipy.stats import gaussian_kde
from wmpl.Formats.WmplTrajectorySummary import loadTrajectorySummaryFast
from wmpl.MetSim.MetSimErosion import energyReceivedBeforeErosion
from wmpl.Utils.AtmosphereDensity import getAtmDensity, atmDensPoly
from wmpl.Utils.TrajConversions import jd2Date
from datetime import datetime, timedelta
from types import SimpleNamespace
from wmpl.MetSim.MetSimErosionCyTools import atmDensityPoly
from multiprocessing import Pool, cpu_count
from functools import partial

# (optional) avoid oversubscription if nrlmsise uses OpenMP
os.environ.setdefault("OMP_NUM_THREADS", "1")

# avoid showing warnings
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Define Julian epoch
JULIAN_EPOCH = datetime(2000, 1, 1, 12) # J2000.0 noon

def date2JD(year, month, day, hour, minute, second, millisecond=0, UT_corr=0.0):
    """ Convert date and time to Julian Date in J2000.0. 
    
    Arguments:
        year: [int] year
        month: [int] month
        day: [int] day of the date
        hour: [int] hours
        minute: [int] minutes
        second: [int] seconds

    Kwargs:
        millisecond: [int] milliseconds (optional)
        UT_corr: [float] UT correction in hours (difference from local time to UT)
    
    Return:
        [float] julian date, J2000.0 epoch
    """

    # Convert all input arguments to integer (except milliseconds)
    year, month, day, hour, minute, second = map(int, (year, month, day, hour, minute, second))

    # Create datetime object of current time
    dt = datetime(year, month, day, hour, minute, second, int(millisecond*1000))

    # Calculate Julian date
    julian = dt - JULIAN_EPOCH + J2000_JD - timedelta(hours=UT_corr)
    
    # Convert seconds to day fractions
    return julian.days + (julian.seconds + julian.microseconds/1000000.0)/86400.0

def _build_wide_jd_sample(center_jd, years_span=11, diurnal_hours=(0, 6, 12, 18)):
    """
    Build a broad JD set around a center JD:
      - Monthly samples across ±years_span
      - 4 diurnal times per day (default)
    This covers seasonal + solar-cycle variability reasonably well.
    """
    print("Building wide JD sample...")
    center_dt = jd2Date(center_jd, dt_obj=True)
    jds = []
    # From (year - years_span) .. (year + years_span), months 1..12, mid-month day=15
    for dy in range(-years_span, years_span + 1):
        for m in range(1, 13):
            # Clamp day=15 safely for all months
            day = 15
            for hr in diurnal_hours:
                try:
                    dt = datetime(center_dt.year + dy, m, day, hr, 0, 0)
                except ValueError:
                    # Fallback: last day of month if needed
                    if m == 2:
                        day_fallback = 28
                    elif m in (4, 6, 9, 11):
                        day_fallback = 30
                    else:
                        day_fallback = 31
                    dt = datetime(center_dt.year + dy, m, day_fallback, hr, 0, 0)
                jds.append(date2JD(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, dt.microsecond/1000.0))
    return np.array(jds, dtype=float)


def _dens_row_for_jd(jd_i, lat, lon, height_arr):
    # top-level function so it's picklable
    return [getAtmDensity(lat, lon, h, jd_i) for h in height_arr]

def compute_dens_matrix_pool(lat, lon, height_arr, jds, processes=None):
    processes = processes or cpu_count()
    func = partial(_dens_row_for_jd, lat=lat, lon=lon, height_arr=height_arr)

    # chunksize helps performance when jds is large
    chunksize = max(1, len(jds) // (processes * 4))
    print(f"Computing density matrix using {processes} processes with chunksize={chunksize}... (takes a while)")
    with Pool(processes=processes) as pool:
        dens_list = pool.map(func, jds, chunksize=chunksize)
    print("Density matrix computation complete.")
    dens_matrix = np.asarray(dens_list, dtype=float)
    return dens_matrix


def fitAtmPoly_variance(
    lat, lon,
    height_min_m, height_max_m,
    jd_or_jds,
    order=7,
    n_heights=200,
    use_median_nominal=True,
    auto_span_if_scalar=True,
    years_span=11,
    diurnal_hours=(0, 6, 12, 18),
    ):
    """
    Build a height-resolved nominal density and variance from NRLMSISE-00 over many JDs.

    Parameters
    ----------
    lat, lon : float
        Geodetic latitude/longitude in radians.
    height_min_m, height_max_m : float
        Height range in meters (e.g., 80e3 to 180e3).
    jd_or_jds : float or array-like
        Single Julian Date or an iterable of JDs to sample.
        If a scalar and auto_span_if_scalar=True, a wide JD set is generated
        around it (monthly over ±years_span, with diurnal_hours per day).
    order : int
        Polynomial order (in height [km]) to fit the *log nominal density*.
    n_heights : int
        Number of height samples between min and max.
    use_median_nominal : bool
        If True, nominal = exp(median(ln rho)); else exp(mean(ln rho)).
        (Median is robust; mean is classical log-normal μ.)
    auto_span_if_scalar : bool
        If True and jd_or_jds is scalar, auto-generate a broad JD set.
    years_span : int
        Half-span of years for auto-JD generation (±years_span).
    diurnal_hours : tuple
        Local hours used in auto-JD sampling.

    Returns
    -------
    result : dict
        {
          'height_m'        : (n_heights,) heights in meters,
          'height_km'       : (n_heights,) heights in km,
          'rho_nominal'     : (n_heights,) nominal density (kg/m^3),
          'mu_ln'           : (n_heights,) mean of ln(rho),
          'sigma_ln'        : (n_heights,) std of ln(rho),
          'poly_coeffs_ln'  : (order+1,) np.polyfit coeffs for ln(rho_nominal) vs height_km,
          'rho_1sigma_low'  : (n_heights,) = rho_nominal * exp(-sigma_ln),
          'rho_1sigma_high' : (n_heights,) = rho_nominal * exp(+sigma_ln),
          'rho_2sigma_low'  : ...
          'rho_2sigma_high' : ...
          'rho_3sigma_low'  : ...
          'rho_3sigma_high' : ...
          'var_linear'      : (n_heights,) variance of rho using log-normal moments,
          'dens_matrix'     : (n_times, n_heights) raw densities used [kg/m^3] (for diagnostics)
        }
    """
    print("Fitting atmospheric density polynomial with variance estimation...")
    # Heights
    height_arr = np.linspace(height_min_m, height_max_m, int(n_heights))

    atm_densities = np.array([getAtmDensity(lat, lon, ht, jd_or_jds) for ht in height_arr])

    # JD set
    if np.isscalar(jd_or_jds) and auto_span_if_scalar:
        jds = _build_wide_jd_sample(float(jd_or_jds), years_span=years_span, diurnal_hours=diurnal_hours)
    else:
        jds = np.atleast_1d(jd_or_jds).astype(float)

    dens_matrix = compute_dens_matrix_pool(lat, lon, height_arr, jds)

    # # Evaluate densities: shape (n_times, n_heights)
    # dens_list = []
    # for ii,jd_i in enumerate(jds):
    #     print(f"  [{ii+1}/{len(jds)}] Evaluating densities for JD={jd_i:.5f}...")
    #     dens_list.append([getAtmDensity(lat, lon, h, jd_i) for h in height_arr])
    # dens_matrix = np.array(dens_list, dtype=float)

    # Work in log-space
    ln_rho = np.log(np.clip(dens_matrix, 1e-300, None))  # guard tiny values

    # Statistics in log-space
    if use_median_nominal:
        mu_ln = np.median(ln_rho, axis=0)    # robust center
    else:
        mu_ln = np.mean(ln_rho, axis=0)      # classical log-normal μ

    sigma_ln = np.std(ln_rho, axis=0, ddof=1) if ln_rho.shape[0] > 1 else np.zeros_like(mu_ln)

    # Nominal density profile (kg/m^3)
    rho_nominal = np.exp(mu_ln)

    # # Log-normal moments for variance in linear space:
    # # Var[ρ] = (exp(σ^2) - 1) * exp(2μ + σ^2)
    # var_linear = (np.exp(sigma_ln**2) - 1.0) * np.exp(2.0 * mu_ln + sigma_ln**2)

    atm_densities_log_nominal = np.log10(rho_nominal)
    # print("atm_densities_log_nominal:", rho_nominal)
    # print("atm_densities_log_nominal sigma:", rho_nominal * np.exp(+sigma_ln))

    def atmDensPolyLog(height_arr, *dens_co):
        return np.log10(atmDensPoly(height_arr, dens_co))

    # Fit the 7th order polynomial
    dens_co_atm_nominal, _ = scipy.optimize.curve_fit(atmDensPolyLog, height_arr, atm_densities_log_nominal, \
        p0=np.zeros(7), maxfev=10000)

    atm_densities_log_sigma = np.log10(rho_nominal * np.exp(+sigma_ln))
    # Fit the 7th order polynomial
    dens_co_sigma, _ = scipy.optimize.curve_fit(atmDensPolyLog, height_arr, atm_densities_log_sigma, \
        p0=np.zeros(7), maxfev=10000)

    return {
        'height_arr': height_arr,
        'rho_true': atm_densities,
        'rho_nominal': rho_nominal,
        'sigma': np.exp(sigma_ln),
        'dens_co_atm_nominal': dens_co_atm_nominal,
        # 'dens_co_sigma': dens_co_sigma,
        # # 'var_linear': var_linear,
        'dens_matrix': dens_matrix,
    }


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


def compute_temperature_profile_iron(
    heights_m,               # array [m] (top->bottom). Will be truncated to >= h_ero_start
    velocities_ms,           # array [m/s], same length as heights_m
    masses_kg,               # array [kg], same length as heights_m
    const,                   # object holding: zenith_angle [rad], erosion_height_start [m], dens_co (if using poly), etc.
    rho_a_fn,                # callable: rho_a_fn(h_m) -> air density [kg/m^3]
    T_a_fn=lambda h: 280.0,  # callable: T_a(h) -> atmospheric T [K]; default 280 K (as in the paper)
    Tm0=280.0,               # initial meteoroid temperature [K]
    emissivity=0.80,         # iron emissivity (oxidized/hot surface ~0.7–0.9) DRAMA iron: 0.7
    c_spec=450.0,            # iron specific heat [J/kg/K] (room-T value; rises with T, but this is a good baseline)
    lambda_val=1,       # heat transfer coefficient (same form as paper; aerodynamic, not material)
    sigma_B=5.670374419e-8   # Stefan–Boltzmann constant
):
    """
    Returns:
        h_used [m], t_used [s] (elapsed), Tm [K], dT_per_step [K], q_conv [W], q_rad [W]
        where q_* are per-step power terms (evaluated at the beginning of each step).
    """
    h_ero = float(const.erosion_height_start)

    rho_m = float(const.rho_grain)

    A_shape = float(const.shape_factor)

    # Truncate the profile to stop at erosion height
    # heights usually decrease; we keep indices where h >= h_ero
    keep = heights_m >= h_ero
    if not np.any(keep):
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    # Ensure we include the last index just above erosion height.
    last_idx = np.where(keep)[0][-1]
    h = np.asarray(heights_m[:last_idx + 1], dtype=float)
    v = np.asarray(velocities_ms[:last_idx + 1], dtype=float)
    m = np.asarray(masses_kg[:last_idx + 1], dtype=float)
    # put all the masses equal to the initial mass
    m[:] = float(const.m_init) # m[0]

    # Time base from heights and zenith angle (dt ~ dh / (v * cos(z)))
    cosz = np.cos(float(const.zenith_angle))
    # Guard against pathological values
    cosz = 1e-6 if np.isclose(cosz, 0.0) else cosz

    # Preallocate
    n = len(h)
    dt = np.full(n-1, float(const.dt), dtype=float)
    Tm = np.empty(n, dtype=float)
    Tm[0] = float(Tm0)
    dT = np.zeros(n, dtype=float)
    q_conv = np.zeros(n, dtype=float)  # aerodynamic heating power [W]
    q_rad  = np.zeros(n, dtype=float)  # radiative loss power [W]

    E_conv = np.zeros(n, dtype=float)  # J absorbed by convection
    E_rad  = np.zeros(n, dtype=float)  # J emitted via radiation (kept positive)
    E_net  = np.zeros(n, dtype=float)  # J net gain = E_conv - E_rad

    # Helper: reference area factor S = A_shape * (m / rho_m)^(2/3)
    # (NOTE: this is proportional to surface area; consistent with your equation form)
    def area_factor(mi):
        return A_shape * (mi / rho_m) ** (2.0 / 3.0)

    # Step the ODE with explicit Euler (small frame dt is fine here)
    for i in range(n - 1):
        mi = m[i]
        vi = v[i]
        hi = h[i]
        Ti = Tm[i]
        Sa = area_factor(mi)

        rho_a = float(rho_a_fn(hi))
        Ta = float(T_a_fn(hi))

        # Aerodynamic (convective) heating power term:
        #   (ϕ * 1/2 * ρ_a * v^3) * S
        q_in = lambda_val * 0.5 * rho_a * (vi ** 3) * Sa

        # Radiative loss power term:
        #   4 * σB * ε * (T_m^4 - T_a^4) * S
        q_out = 4.0 * sigma_B * emissivity * (Ti ** 4 - Ta ** 4) * Sa

        # dT/dt = (q_in - q_out) / (c * m)   (NO mass-loss term)
        dTdt = (q_in - q_out) / (c_spec * mi)

        Tm[i + 1] = Ti + dTdt * dt[i]
        dT[i + 1] = Tm[i + 1] - Ti
        q_conv[i] = q_in
        q_rad[i]  = q_out

        # Energy integrals (left Riemann)
        E_conv[i+1] = E_conv[i] + q_in * dt[i]
        # keep E_rad as cumulative *magnitude* of radiative exchange outward
        # If you want strictly "loss", use positive when q_out>0, otherwise add 0.
        E_rad[i+1]  = E_rad[i] + abs(q_out) * dt[i]
        # Net energy gained by the meteoroid
        E_net[i+1]  = E_net[i] + (q_in - q_out) * dt[i]


    # Last step power terms (for completeness, evaluate at final state)
    rho_a_last = float(rho_a_fn(h[-1]))
    Ta_last = float(T_a_fn(h[-1]))
    Sa_last = area_factor(m[-1])
    q_conv[-1] = lambda_val * 0.5 * rho_a_last * (v[-1] ** 3) * Sa_last
    q_rad[-1]  = 4.0 * sigma_B * emissivity * (Tm[-1] ** 4 - Ta_last ** 4) * Sa_last

    return h, Tm, dT, q_conv, q_rad, E_conv, E_rad, E_net


def run_single_eeu(sim_num_and_data):
    sim_num, tot_sim, sample, obs_data, variables, fixed_values, flags_dict, lambda_val = sim_num_and_data

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
        eeucs, eeum = energyReceivedBeforeErosion(const_nominal, lambda_val)

    except Exception as e:
        print(f"Simulation {sim_num} failed: {e}")
        eeucs, eeum = np.nan, np.nan

    const_nominal.erosion_height_start = obs_data.height_lum[-1] # calculate the Kinetic Energy until the last height
    const_nominal.v_init = np.mean(obs_data.velocities) # calculate the Kinetic Energy until using the mean velocity

    # Extract physical quantities
    try:
        eeucs_end, eeum_end = energyReceivedBeforeErosion(const_nominal, lambda_val)

        return (sim_num, eeucs, eeum, eeucs_end, eeum_end)

    except Exception as e:
        print(f"Simulation end {sim_num} failed: {e}")
        return (sim_num, eeucs, eeum, np.nan, np.nan)

def extract_other_prop(input_dirfile, output_dir_show, name_distr="", lambda_val=1, recompute_eenres=False):
    """
    Function to plot the distribution of the parameters from the dynesty files and save them as a table in LaTeX format.
    """
    # Use the class to find .dynesty, load prior, and decide output folders
    finder = find_dynestyfile_and_priors(input_dir_or_file=input_dirfile,prior_file="",resume=True,output_dir=input_dirfile,use_all_cameras=True,pick_position=0)

    num_meteors = len(finder.base_names)  # Number of meteors
    file_eeu_dict = {}
    print(f"Found {num_meteors} meteors in {input_dirfile}.")
    for i, (base_name, dynesty_info, prior_path, out_folder) in enumerate(zip(finder.base_names,finder.input_folder_file,finder.priors,finder.output_folders)):
        dynesty_file, pickle_file, bounds, flags_dict, fixed_values = dynesty_info
        # split dynesty_file in folder and file name
        folder_name, _ = os.path.split(dynesty_file)
        print(f"\nProcessing {i+1}/{num_meteors}: {base_name} in {folder_name}")

        if i == 0:
            # check if .pikle file exists where the python script is located
            local_dir_script = os.path.dirname(os.path.abspath(__file__))
            # look for files that end in atm.pickle
            pickle_files_atm = [f for f in os.listdir(local_dir_script) if f.endswith('atm.pickle')]
            if len(pickle_files_atm) == 0:
                print("No atmospheric density variance pickle file found in the script directory. Running density variance computation and saving the pickle file...")
                # check if dynesty_file is an array of files
                if isinstance(pickle_file, (list, tuple)):
                    curr_lum_file = pickle_file[0]
                else:
                    curr_lum_file = pickle_file
                # take the m_init_list from the trajectory file
                traj=loadPickle(*os.path.split(curr_lum_file))

                # now find the zenith angle mass v_init and dens_co
                dens_fit_ht_beg = 180000
                dens_fit_ht_end = 50000
                # dens_fit_ht_end = traj.rend_ele - 5000
                # if dens_fit_ht_end < 14000:
                #     dens_fit_ht_end = 14000

                lat_mean = np.mean([traj.rbeg_lat, traj.rend_lat])
                lon_mean = meanAngle([traj.rbeg_lon, traj.rend_lon])
                jd_dat=traj.jdt_ref

                # Fit the polynomail describing the density
                dens_co_var = fitAtmPoly_variance(lat_mean, lon_mean, dens_fit_ht_end, dens_fit_ht_beg, jd_dat)
                
                # plot the density profile with variance
                heights = dens_co_var['height_arr']
                rho_nominal = dens_co_var['rho_nominal']
                rho_true = dens_co_var['rho_true']
                sigma_mult = dens_co_var['sigma']

                # Log-normal bands (multiplicative)
                rho_1_low, rho_1_high = rho_nominal / sigma_mult,            rho_nominal * sigma_mult
                rho_2_low, rho_2_high = rho_nominal / (sigma_mult**2),       rho_nominal * (sigma_mult**2)
                rho_3_low, rho_3_high = rho_nominal / (sigma_mult**3),       rho_nominal * (sigma_mult**3)

                # compute the percentage difference at the heights
                percent_diff_1sigm = abs(np.mean(((rho_nominal - rho_1_high) / rho_1_high) * 100))
                percent_diff_2sigm = abs(np.mean(((rho_nominal - rho_2_high) / rho_2_high) * 100))
                percent_diff_3sigm = abs(np.mean(((rho_nominal - rho_3_high) / rho_3_high) * 100))

                fig, ax = plt.subplots(figsize=(8,6))
                ax.plot(rho_nominal, heights/1000, label='Nominal Density', color='blue')
                ax.fill_betweenx(heights/1000, rho_1_low, rho_1_high, color='blue', alpha=0.3, label=f'±1σ {percent_diff_1sigm:.2f}%')
                ax.fill_betweenx(heights/1000, rho_2_low, rho_2_high, color='blue', alpha=0.2, label=f'±2σ {percent_diff_2sigm:.2f}%')
                ax.fill_betweenx(heights/1000, rho_3_low, rho_3_high, color='blue', alpha=0.1, label=f'±3σ {percent_diff_3sigm:.2f}%')
                ax.plot(rho_true, heights/1000, label='Sampled Densities', color='black')
                ax.set_xscale('log')
                ax.set_xlabel('Atmospheric Density [kg/m³]')
                ax.set_ylabel('Height [km]')
                # ax.set_title('Fitted Atmospheric Density Profile with Variance')
                ax.legend()
                plt.grid(True, which="both", ls="--", lw=0.5)
                # save the figure
                fig.savefig(local_dir_script + os.sep + f"atmospheric_density_profile_variance_{name_distr}.png", dpi=300)

                # save in a pickle file in local_dir_script
                atm_variance_pickle_file = local_dir_script + os.sep + f"{name_distr}_variance_rho_atm.pickle"
                with open(atm_variance_pickle_file, "wb") as f:
                    pickle.dump(dens_co_var, f)
                print(f"Saved atmospheric density variance data to {atm_variance_pickle_file}.")
            else:
                atm_variance_pickle_file = local_dir_script + os.sep + pickle_files_atm[0]
                print(f"Loaded atmospheric density variance data from pickle file: {atm_variance_pickle_file}.")

            # load the pickle file
            with open(atm_variance_pickle_file, "rb") as f:
                dens_co_nominal_and_var = pickle.load(f)
                sigma_mult_atm = dens_co_nominal_and_var['sigma']
                heights_atm = dens_co_nominal_and_var['height_arr']
                dens_co_atm_nominal = dens_co_nominal_and_var['dens_co_atm_nominal']

            

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

        # load the dynesty file
        dsampler = dynesty.DynamicNestedSampler.restore(dynesty_file)

        # load the variable names
        variables = list(flags_dict.keys())

        if recompute_eenres:
            print(f"\nRecomputing .eenres for {folder_name}.")
            check_flag = False  
        else:
            check_flag = True

        for variable in variables:
            if variable == 'erosion_rho_change':
                flag_total_rho = True

        # look if in folder_name it exist a file that ends in .eenres exist in 
        if any(f.endswith(".eenres") for f in os.listdir(folder_name)) and check_flag:
            print(f"\nFound existing results in {folder_name}.eenres, loading them.")

            # look for the file that ends in .eenres
            dynesty_res_file = [f for f in os.listdir(folder_name) if f.endswith(".eenres")][0]
            with open(folder_name + os.sep + dynesty_res_file, "rb") as f:
                dynesty_run_results = pickle.load(f)

            erosion_energy_per_unit_cross_section_arr = dynesty_run_results.erosion_energy_per_unit_cross_section
            erosion_energy_per_unit_mass_arr = dynesty_run_results.erosion_energy_per_unit_mass
            # erosion_energy_per_unit_cross_section_arr_end = dynesty_run_results.erosion_energy_per_unit_cross_section_end
            # erosion_energy_per_unit_mass_arr_end = dynesty_run_results.erosion_energy_per_unit_mass_arr_end
            total_energy_before_erosion = dynesty_run_results.total_energy_before_erosion
            total_energy_before_second_erosion = dynesty_run_results.total_energy_before_second_erosion
            energy_per_mass_before_first_erosion = dynesty_run_results.energy_per_mass_before_first_erosion
            energy_per_mass_before_second_erosion = dynesty_run_results.energy_per_mass_before_second_erosion
            Tot_energy = dynesty_run_results.Tot_energy
            rho_total_arr = dynesty_run_results.rho_total
            # check if lambda_val exists in the dynesty_run_results
            if hasattr(dynesty_run_results, 'lambda_val'):
                lambda_val = dynesty_run_results.lambda_val
            else:
                lambda_val = 1
            # check if erosion_energy_total_end exists

            samples = dynesty_run_results.samples

            weights = dynesty_run_results.weights
            w = weights / np.sum(weights)

        else:
            if not recompute_eenres:
                print(f"\nNo existing results found in {folder_name} .eenres, running dynesty.")
            dynesty_run_results = dsampler.results

            weights = dynesty_run_results.importance_weights()
            w = weights / np.sum(weights)

            # for attr in dir(dynesty_run_results):
            #     if not attr.startswith("_") and not callable(getattr(dynesty_run_results, attr)):
            #         print(attr, ":", type(getattr(dynesty_run_results, attr)))

            ### PLOT THE DISTRIBUTION OF THE PARAMETERS and how it gets better ###

            # for sim_num in between 0 and the number of samples in dynesty_run_results.samples but just pick 10 among wich is the best and the worst samples
            # for sim_num in np.linspace(0, len(dynesty_run_results.samples)-1, 10, dtype=int):
            #     guess = dynesty_run_results.samples[sim_num]
            #     # for variable in variables: for 
            #     for i, variable in enumerate(variables):
            #         if 'log' in flags_dict[variable]:  
            #             guess[i] = 10**(guess[i])
            #         if 'noise_lag' == variable:
            #             obs_data.noise_lag = guess[i]
            #             obs_data.noise_vel = guess[i]*np.sqrt(2)/(1.0/32)
            #         if 'noise_lum' == variable:
            #             obs_data.noise_lum = guess[i]

            #     best_guess_obj_plot = run_simulation(guess, obs_data, variables, fixed_values)

            #     # Plot the data with residuals and the best fit
            #     plot_data_with_residuals_and_real(obs_data, best_guess_obj_plot, output_dir_show, base_name + "_fit_"+str(sim_num)+"_-LnZ_"+str(abs(np.round(dynesty_run_results.logvol[sim_num],2))), color_sim='black', label_sim="sim "+str(sim_num))

            ### add MORE PARAMETERS ###

            # Package inputs
            inputs = [
                (i, len(dynesty_run_results.samples), dynesty_run_results.samples[i], obs_data, variables, fixed_values, flags_dict, lambda_val)
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
            erosion_energy_per_unit_cross_section_arr_end = np.full(N, np.nan)
            erosion_energy_per_unit_mass_arr_end = np.full(N, np.nan)
            # rho_total_arr = np.full(N, np.nan)
            # kc_par_arr = np.full(N, np.nan)
            # F_par_arr = np.full(N, np.nan)
            # trail_length_arr = np.full(N, np.nan)

            for res in results:
                i, eeucs, eeum, eeucs_end, eeum_end = res
                erosion_energy_per_unit_cross_section_arr[i] = eeucs / 1e6  # convert to MJ/m^2
                erosion_energy_per_unit_mass_arr[i] = eeum / 1e6  # convert to MJ/kg
                # also get the end values
                erosion_energy_per_unit_cross_section_arr_end[i] = eeucs_end / 1e6  # convert to MJ/m^2
                erosion_energy_per_unit_mass_arr_end[i] = eeum_end / 1e6  # convert to MJ/kg
                # rho_total_arr[i] = rho
                # kc_par_arr[i] = kc
                # F_par_arr[i] = F
                # trail_length_arr[i] = tl

            # get total erosion_energy from samples
            total_energy_before_erosion = erosion_energy_per_unit_mass_arr * dynesty_run_results.samples[:, variables.index('m_init')]  # Total energy received before erosion in MJ

            sim_num = np.argmax(dynesty_run_results.logl)
            # best_guess_obj_plot = dynesty_run_results.samples[sim_num]
            # create a copy of the best guess
            best_guess = dynesty_run_results.samples[sim_num].copy()
            samples = dynesty_run_results.samples
            # for variable in variables: for 
            for i, variable in enumerate(variables):
                if 'log' in flags_dict[variable]:
                    # print(f"Transforming {variable} from log scale to linear scale.{best_guess[i]}")  
                    best_guess[i] = 10**(best_guess[i])
                    # print(f"Transforming {variable} from log scale to linear scale.{best_guess[i]}")
                    samples[:, i] = 10**(samples[:, i])  # also transform all samples
                if variable == 'noise_lag':
                    obs_data.noise_lag = best_guess[i]
                    obs_data.noise_vel = best_guess[i] * np.sqrt(2)/(1.0/32)
                if variable == 'noise_lum':
                    obs_data.noise_lum = best_guess[i]

            constjson_bestfit = Constants()

            constjson_bestfit.__dict__['P_0m'] = obs_data.P_0m
            constjson_bestfit.__dict__['lum_eff_type'] = obs_data.lum_eff_type
            constjson_bestfit.__dict__['disruption_on'] = obs_data.disruption_on
            constjson_bestfit.__dict__['dens_co'] = obs_data.dens_co
            constjson_bestfit.__dict__['dt'] = obs_data.dt
            constjson_bestfit.__dict__['h_kill'] = obs_data.h_kill
            constjson_bestfit.__dict__['v_kill'] = obs_data.v_kill

            # change Constants that have the same variable names and the one fixed
            for variable in variables:
                if variable in constjson_bestfit.__dict__.keys():
                    constjson_bestfit.__dict__[variable] = best_guess[variables.index(variable)]

            # do te same for the fixed values
            for variable in fixed_values.keys():
                if variable in constjson_bestfit.__dict__.keys():
                    constjson_bestfit.__dict__[variable] = fixed_values[variable]
                    print(f"Fixed value for {variable}: {fixed_values[variable]}")

            best_guess_obj_plot = run_simulation(best_guess, obs_data, variables, fixed_values)

            # extract the const from the best_guess_obj_plot
            const_nominal = best_guess_obj_plot.const

            # const_nominal.h_init = unique_heights_massvar_init
            # const_nominal.erosion_height_start = unique_heights_massvar
            # const_nominal.v_init = velocities_massvar_init
            # const_nominal.m_init = mass_best_massvar

            mass_best = np.array(best_guess_obj_plot.main_mass_arr[:-1], dtype=np.float64) # mass_total_active_arr # main_mass_arr
            heights = np.array(best_guess_obj_plot.leading_frag_height_arr[:-1], dtype=np.float64)
            velocities = np.array(best_guess_obj_plot.leading_frag_vel_arr[:-1], dtype=np.float64)

            # Extract physical quantities
            eeucs_best, eeum_best = energyReceivedBeforeErosion(const_nominal, lambda_val)
            total_energy_before_erosion = eeum_best * best_guess_obj_plot.const.m_init / 1e3  # Total energy received before erosion in kJ from the Kinetic Energy not the one computed
            eeum_best = eeum_best / 1e6  # convert to MJ/kg
            # precise erosion tal energy calculation ########################
            erosion_height_start = best_guess_obj_plot.const.erosion_height_start+1000
            ### get for each mass_best that is different from te previuse one get the height at which the mass loss happens
            diff_mask = np.concatenate(([True], np.diff(mass_best) != 0))
            ### only consider when mass actually changes enought
            # diff_mask = np.concatenate(([True], ~np.isclose(np.diff(mass_best), 0)))
            unique_heights_massvar = heights[diff_mask]
            mass_best_massvar = mass_best[diff_mask]
            velocities_massvar = velocities[diff_mask]

            # Example density function using your existing polynomial (adjust to your implementation)
            def rho_a_poly(h_m):
                # If you have something like atmDensityPoly(h, const.dens_co):
                return atmDensityPoly(h_m, const_nominal.dens_co)

            # Run the temperature integration (iron properties baked in)
            h_used, Tm, dT, q_conv, q_rad, E_conv, E_rad, E_net = compute_temperature_profile_iron(
                heights_m=unique_heights_massvar,
                velocities_ms=velocities_massvar,
                masses_kg=mass_best_massvar,
                const=best_guess_obj_plot.const,
                lambda_val=lambda_val,
                rho_a_fn=rho_a_poly,              # or your own atmosphere function
                T_a_fn=lambda h: 280.0,           # constant Ta per the paper assumption
                Tm0=280.0
            )

            fig, ax = plt.subplots(1,2, figsize=(12, 6))
            # plot the temperature profile
            ax[0].plot(Tm, h_used/1000, color='k')
            # create 
            ax[0].set_xlabel('Meteoroid Temperature [K]', fontsize=15)
            ax[0].set_ylabel('Height [km]', fontsize=15)
            # put a red dot at the height when Tm reaches 1811 K the melting point of iron
            melting_point_iron = 1811  # K
            # find the first index where Tm is higher than melting_point_iron
            melting_idx = np.where(Tm >= melting_point_iron)[0]
            if len(melting_idx) > 0:
                melting_height = h_used[melting_idx[0]] / 1000  # in km
                # point out the height when the melting point is reached
                ax[0].plot(melting_point_iron, melting_height, 'ro', label=f'Melting Point Reached{f" at {melting_height:.1f} km" if melting_height >= 0 else ""}')
                ax[0].legend()
            # plot the convective and radiative power
            ax[1].plot(q_conv/1e3, h_used/1000, color='tab:blue', label='Convective Power')
            ax[1].plot(q_rad/1e3, h_used/1000, color='tab:orange', label='Radiative Power')
            ax[1].set_xlabel('Power [kW]', fontsize=15)
            # ax[1].set_ylabel('Height [km]', fontsize=15)
            # put te line when the erosion starts
            ax[0].axhline(y=best_guess_obj_plot.const.erosion_height_start/1000, color='gray', linestyle='--', label='Erosion Height Start $h_{e}$')
            ax[1].axhline(y=best_guess_obj_plot.const.erosion_height_start/1000, color='gray', linestyle='--', label='Erosion Height Start $h_{e}$')
            ax[1].legend()
            # put the x axis in log scale
            # ax[0].set_xscale('log')
            ax[1].set_xscale('log')
            # grid on
            ax[0].grid()
            ax[1].grid()
            # plt.suptitle(f"Temperature and Power Profile for Best Fit: {base_name}")
            plt.tight_layout()
            plt.savefig(folder_name + os.sep + base_name + "_temperature_profile.png")
            plt.close()

            # # now delete any unique_heights_massvar and mass_best_massvar that are bigger than erosion_height_change
            mass_best_massvar = mass_best_massvar[unique_heights_massvar < erosion_height_start]
            velocities_massvar = velocities_massvar[unique_heights_massvar < erosion_height_start]
            unique_heights_massvar = unique_heights_massvar[unique_heights_massvar < erosion_height_start]
            
            # add at the begnning the m_init to mass_best_massvar and h_init_best to unique_heights_massvar
            ### same as Kinetic Energy at erosion heiht
            # unique_heights_massvar_init = np.concatenate(([best_guess_obj_plot.const.h_init,best_guess_obj_plot.const.erosion_height_start], unique_heights_massvar[1:]))
            # unique_heights_massvar[0] = best_guess_obj_plot.const.erosion_height_start
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
                (i, best_guess_obj_plot, unique_heights_massvar_init[i], unique_heights_massvar[i], mass_best_massvar[i], velocities_massvar_init[i], lambda_val, dens_co_atm_nominal)
                for i in range(len(mass_best_massvar)) # for i in np.linspace(0, len(dynesty_run_results.samples)-1, 10, dtype=int)
            ]
            #     for i in range(len(dynesty_run_results.samples)) # 
            num_cores = multiprocessing.cpu_count()

            # Run in parallel
            with Pool(processes=num_cores) as pool:  # adjust to number of cores
                results = pool.map(run_total_energy_received_varNom, inputs)

            N = len(mass_best_massvar)

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

            # print(f"Check: {total_energy_before_erosion} MJ")
            # print(f"Precise total Kinetic Energy before first erosion: {Tot energy_arr[0]} MJ")

            # plot y axis the unique_heights_massvar vs Tot energy_arr
            fig, ax = plt.subplots(1,2, figsize=(12, 6))
            station_colors = {}
            cmap = plt.get_cmap("tab10")
            # ABS MAGNITUDE
            for station in np.unique(obs_data.stations_lum):
                mask = obs_data.stations_lum == station
                if station not in station_colors:
                    station_colors[station] = cmap(len(station_colors) % 10)
                ax[0].plot(obs_data.absolute_magnitudes[mask], obs_data.height_lum[mask] / 1000, 'x--', color=station_colors[station], label=station)

            # Integrate luminosity/magnitude if needed
            if (1 / obs_data.fps_lum) > best_guess_obj_plot.const.dt:
                best_guess_obj_plot.luminosity_arr, best_guess_obj_plot.abs_magnitude = luminosity_integration(
                    best_guess_obj_plot.time_arr, best_guess_obj_plot.time_arr, best_guess_obj_plot.luminosity_arr,
                    best_guess_obj_plot.const.dt, obs_data.fps_lum, obs_data.P_0m
                )

            # make a first subplot with the lightcurve against height
            ax[0].plot(best_guess_obj_plot.abs_magnitude,best_guess_obj_plot.leading_frag_height_arr/1000, color='k', label='Best Fit Simulation')
            ax[0].set_ylabel('Height [km]', fontsize=15)
            ax[0].set_xlabel('Abs.Mag [-]', fontsize=15)
            # make the Tot energy_arr the sum of the previous values
            Tot_energy_arr_cum = np.cumsum(Tot_energy_arr[0,:])
            Tot_energy_arr_cum_atmNom = np.cumsum(Tot_energy_arr[1,:])
            # Tot_energy_arr_cum_075 = np.cumsum(Tot_energy_arr[1,:])
            # Tot_energy_arr_cum_125 = np.cumsum(Tot_energy_arr[2,:])

            # interpolate the sigma vaues at each unique_heights_massvar
            sigma_mult_interp = np.interp(unique_heights_massvar, heights_atm, sigma_mult_atm)
            # create the sigma from the nominal value Log-normal bands (multiplicative)
            Tot_energy_arr_sigma1_low, Tot_energy_arr_sigma1_high = Tot_energy_arr_cum_atmNom / sigma_mult_interp,            Tot_energy_arr_cum_atmNom * sigma_mult_interp
            Tot_energy_arr_sigma2_low, Tot_energy_arr_sigma2_high = Tot_energy_arr_cum_atmNom / (sigma_mult_interp**2),       Tot_energy_arr_cum_atmNom * (sigma_mult_interp**2)
            Tot_energy_arr_sigma3_low, Tot_energy_arr_sigma3_high = Tot_energy_arr_cum_atmNom / (sigma_mult_interp**3),       Tot_energy_arr_cum_atmNom * (sigma_mult_interp**3)

            # # create the atmpsperic uncertanty as a shaded area between Tot_energy_arr_cum_075 and Tot_energy_arr_cum_125
            # # ax[1].fill_betweenx(unique_heights_massvar/1000, Tot_energy_arr_cum_075, Tot_energy_arr_cum_125, color='lightgray', alpha=0.3, label='Receive Energy Atm. Uncert. (±25%)')
            # # ax[1].plot(Tot_energy_arr_cum_atmNom, unique_heights_massvar/1000, color='gray', linestyle='-', label='Receive Energy Profile (Nominal Atm.)')
            # ax[1].fill_betweenx(unique_heights_massvar/1000, Tot_energy_arr_sigma1_low, Tot_energy_arr_sigma1_high, color='darkgray', alpha=0.3, label='Atm. Density Uncert. (±1σ)')
            # ax[1].fill_betweenx(unique_heights_massvar/1000, Tot_energy_arr_sigma2_low, Tot_energy_arr_sigma2_high, color='gray', alpha=0.2, label='Atm. Density Uncert. (±2σ)')
            # ax[1].fill_betweenx(unique_heights_massvar/1000, Tot_energy_arr_sigma3_low, Tot_energy_arr_sigma3_high, color='lightgray', alpha=0.1, label='Atm. Density Uncert. (±3σ)')
            ax[1].plot(Tot_energy_arr_cum, unique_heights_massvar/1000, color='k', label='Receive Energy Profile')
            # # add a hrizontal line at y=total_energy_before_erosion
            ax[1].axhline(y=best_guess_obj_plot.const.erosion_height_start/1000, color='gray', linestyle='--', label='Erosion Height Start $h_{e}$')
            ax[0].axhline(y=best_guess_obj_plot.const.erosion_height_start/1000, color='gray', linestyle='--')

            # found the index of erosion_height_change in variables
            erosion_height_best = constjson_bestfit.erosion_height_start
            # find the closest unique_heights_massvar to erosion_height_change_best
            idx_closest = (np.abs(unique_heights_massvar - erosion_height_best)).argmin()
            # summ all of the Tot energy_arr until idx_closest
            total_energy_before_first_erosion = Tot_energy_arr[0,:idx_closest].sum()
            energy_per_mass_before_first_erosion = eeum_best # total_energy_before_first_erosion/(best_guess_obj_plot.const.m_init-mass_best_massvar[idx_closest])/1000 # MJ/kg
            print(f"Unit energy before first erosion: {energy_per_mass_before_first_erosion} MJ/kg")
            energy_per_mass_before_second_erosion = energy_per_mass_before_first_erosion
            # found the index of erosion_height_change in variables
            # print(f"Total energy before first erosion: {total_energy_before_first_erosion} kJ")
            total_energy_before_second_erosion = total_energy_before_first_erosion
            if flag_total_rho:
                # found the index of erosion_height_change in variables
                erosion_height_change_best = constjson_bestfit.erosion_height_change
                # find the closest unique_heights_massvar to erosion_height_change_best
                idx_closest_2 = (np.abs(unique_heights_massvar - erosion_height_change_best)).argmin()
                # summ all of the Tot energy_arr until idx_closest
                total_energy_before_second_erosion = Tot_energy_arr[0,:idx_closest_2].sum()
                energy_per_mass_before_second_erosion = total_energy_before_second_erosion/(best_guess_obj_plot.const.m_init-mass_best_massvar[idx_closest_2])/1000 # MJ/kg
                print(f"Unit energy before second erosion: {energy_per_mass_before_second_erosion} MJ/kg")
                # print(f"Total energy before second erosion: {total_energy_before_second_erosion} kJ")
                # print the constjson_bestfit.erosion_height_change as a line
                ax[1].axhline(y=erosion_height_change_best/1000, color='gray', linestyle='-.', label='Erosion Height Change $h_{e_2}$') # , label='Erosion Height Change $h_{e_2}$')
                ax[0].axhline(y=erosion_height_change_best/1000, color='gray', linestyle='-.')

            print(f"Precise total Kinetic Energy: {Tot_energy} kJ")
            # temperature line when is 1811 K
            if len(melting_idx) > 0:
                ax[1].axhline(y=melting_height, color='peru', linestyle='--', linewidth=1.5)
                ax[0].axhline(y=melting_height, color='peru', linestyle='--', linewidth=1.5, label=f'Melting Point Reached{f" at {melting_height:.1f} km" if melting_height >= 0 else ""}')
            # ax[1].set_ylabel('Height (km)', fontsize=15)
            ax[1].set_xlabel('Energy [kJ]', fontsize=15)
            # put the x axis in log
            ax[1].set_xscale('log')
            Fe_mol = 55.845  # g/mol
            Hentalpy_fusion_iron = 13.81  # kJ/mol
            Hentalpy_vapor_iron = 340.0  # kJ/mol
            # 0.000449 J/gK
            specific_heat_capacity_iron = 0.000449  # J/gK
            print(f"Fusion energy iron: {Hentalpy_fusion_iron * best_guess_obj_plot.const.m_init*1000 / Fe_mol} kJ")
            print(f"Energy to bring iron from 275K to 1811K: {(1811 - 275) * specific_heat_capacity_iron * best_guess_obj_plot.const.m_init*1000} kJ") # specific heat capacity of iron 0.449 J/gK
            print(f"Total energy for fusion of iron: {(Hentalpy_fusion_iron/ Fe_mol + (1811 - 275) * specific_heat_capacity_iron) * best_guess_obj_plot.const.m_init*1000} kJ")
            print(f"Total Radiative energy: {E_rad[-1]/1e3} kJ")
            # print(f"Vaporization energy iron: {Hentalpy_vapor_iron * best_guess_obj_plot.const.m_init*1000 / Fe_mol} kJ")   
            # create a red line with the energy of fusion of iron 0.272 MJ/kg * m_init
            fusion_energy_iron_kJ = best_guess_obj_plot.const.m_init*1000 * (Hentalpy_fusion_iron / Fe_mol + (1811 - 275) * specific_heat_capacity_iron) # MJ/kg# kJ
            # Hentalpy_vapor_iron_kJ = Hentalpy_vapor_iron * best_guess_obj_plot.const.m_init*1000 / Fe_mol # kJ
            ax[1].axvline(x=fusion_energy_iron_kJ+E_rad[-1]/1e3, color='purple', linestyle=':', label='Tot.Energy for Fusion of Iron + radiation')
            #find the index where Tot_energy_arr_cum is closest to fusion_energy_iron_kJ+E_rad[-1]/1e3 from the interpolated values to have a more precise value
            # increase by an order of magnitude the number of Tot_energy_arr_cum points by interpolation
            target_E = fusion_energy_iron_kJ + E_rad[-1]/1e3  # kJ
            E_min = Tot_energy_arr_cum[0]
            E_max = Tot_energy_arr_cum[-1]

            if target_E >= E_min and target_E <= E_max:
                height_at_target_m = np.interp( target_E, Tot_energy_arr_cum, unique_heights_massvar )
                # put a red line at the height_at_target_m on both plots
                ax[0].axhline(y=height_at_target_m/1000, color='red', linestyle=':', linewidth=2.5, label=f'Fusion Energy Reached at {height_at_target_m/1000:.1f} km')
                ax[1].axhline(y=height_at_target_m/1000, color='red', linestyle=':', linewidth=2.5) # , label=f'Fusion Energy reached at {height_at_target_m/1000:.1f} km'
                ax[1].plot( target_E,  height_at_target_m/1000, 'o', color='purple') # , label=f'Fusion Energy reached at {height_at_target_m/1000:.1f} km'
            else:
                print("⚠ target energy outside cumulative range, not plotting intersection.")

            # take the value of the y axis max and min value from ax[1] 
            y_min = ax[1].get_ylim()[0]
            y_max = ax[1].get_ylim()[1]
            x_min = ax[1].get_xlim()[0]
            x_max = ax[1].get_xlim()[1]
            ax[0].set_ylim([y_min, y_max])
            ax[1].set_xlim([x_min, x_max])
            ax[1].set_ylim([y_min, y_max])
            # tilt by 45 degrees the tics of x
            ax[1].tick_params(axis='x', rotation=45)
            # also the minor tics tilt them by 45 degrees
            ax[1].tick_params(axis='x', which='minor', rotation=45)
            # put the x axis form 8 to the max value of abs magnitude
            x_min = ax[0].get_xlim()[0]
            ax[0].set_xlim([x_min, 8])
            # flip the x axis
            ax[0].invert_xaxis()
            # # # plot the net energy in dashed line
            # ax[1].plot(E_net/1e3, h_used/1000, color='black', linestyle='--', label='Net Energy (convective - radiative)')
            # ax[1].plot(E_conv/1e3, h_used/1000, color='black', linestyle='-.', label='Energy convective')
            # ax[1].plot(E_rad/1e3, h_used/1000, color='black', linestyle=':', label='Energy radiative')
            # add grid
            ax[0].grid()
            ax[1].grid()
            # put the legend in the right lower corner
            ax[1].legend(loc='lower left')
            ax[0].legend(loc='lower right')
            # plt.show()
            # save in folder_name as base_name+"_erosion_energy_profile.png"
            fig.tight_layout()
            fig.savefig(folder_name + os.sep + base_name+"_Lambda"+str(lambda_val)+"_erosion_energy_profile.png", bbox_inches='tight', dpi=300)
            plt.close(fig)

            # # precise erosion tal energy calculation ########################

            # mass_before = mass_best[np.argmin(np.abs(heights - erosion_height_change))]

            best_guess_obj_plot = run_simulation(best_guess, obs_data, variables, fixed_values)

            # find erosion change height
            if 'erosion_height_change' in variables:
                erosion_height_change = best_guess[variables.index('erosion_height_change')]
            if 'm_init' in variables:
                m_init = best_guess[variables.index('m_init')]

            heights = np.array(best_guess_obj_plot.leading_frag_height_arr, dtype=np.float64)[:-1]
            mass_best = np.array(best_guess_obj_plot.mass_total_active_arr, dtype=np.float64)[:-1]

            mass_before = best_guess_obj_plot.const.mass_at_erosion_change
            if mass_before is None:
                mass_before = mass_best[np.argmin(np.abs(heights - erosion_height_change))]

            if 'erosion_rho_change' in variables:
                rho_total_arr = samples[:, variables.index('rho')].astype(float)*(abs(m_init-mass_before) / m_init) + samples[:, variables.index('erosion_rho_change')].astype(float) * (mass_before / m_init)
            else:
                rho_total_arr = samples[:, variables.index('rho')].astype(float)

            rho_total_arr = np.array(rho_total_arr, dtype=np.float64)

            # Create a namespace object for dot-style access
            results = SimpleNamespace(**dsampler.results.__dict__)  # load all default results

            # Add your custom attributes
            results.weights = dynesty_run_results.importance_weights()
            results.norm_weights = w
            results.erosion_energy_per_unit_cross_section = erosion_energy_per_unit_cross_section_arr
            results.erosion_energy_per_unit_mass = erosion_energy_per_unit_mass_arr
            # results.erosion_energy_per_unit_cross_section_end = Tot_energy_per_unit_cross_section
            # results.erosion_energy_per_unit_mass_arr_end = Tot_energy_per_unit_mass
            results.total_energy_before_erosion = total_energy_before_erosion
            results.total_energy_before_second_erosion = total_energy_before_second_erosion
            results.energy_per_mass_before_first_erosion = energy_per_mass_before_first_erosion
            results.energy_per_mass_before_second_erosion = energy_per_mass_before_second_erosion
            results.Tot_energy = Tot_energy
            results.rho_total = rho_total_arr
            results.lambda_val = lambda_val

            # delete from base_name _combined if it exists
            if '_combined' in base_name:
                base_name = base_name.replace('_combined', '')

            # Save
            with open(folder_name + os.sep + base_name+"_results.eenres", "wb") as f:
                pickle.dump(results, f)
                print(f"Results saved successfully in {folder_name + os.sep + base_name+'_results.eenres'}.")

        for i, x in enumerate([erosion_energy_per_unit_cross_section_arr, erosion_energy_per_unit_mass_arr, rho_total_arr, samples[:, variables.index('v_init')].astype(float), samples[:, variables.index('m_init')].astype(float)]):
            # mask out NaNs
            mask = ~np.isnan(x)
            if not np.any(mask):
                print("Warning: All values are NaN, skipping quantile calculation.")
                continue
            mask = ~np.isnan(x)
            x_valid = x[mask]
            w_valid = w[mask]
            # renormalize
            w_valid /= np.sum(w_valid)
            if i == 0:
                # weighted quantiles
                eeucs_lo, eeucs, eeucs_hi = _quantile(x_valid, [0.025, 0.5, 0.975], weights=w_valid)
                print(f"Kinetic Energy per unit cross section: {eeucs} MJ/m^2, 95% CI = [{eeucs_lo:.6f}, {eeucs_hi:.6f}]")
                eeucs_lo = (eeucs - eeucs_lo)
                eeucs_hi = (eeucs_hi - eeucs)
            elif i == 1:
                # weighted quantiles
                eeum_lo, eeum, eeum_hi = _quantile(x_valid, [0.025, 0.5, 0.975], weights=w_valid)
                print(f"Kinetic Energy per unit mass: {eeum} MJ/kg, 95% CI = [{eeum_lo:.6f}, {eeum_hi:.6f}]")
                eeum_lo = (eeum - eeum_lo)
                eeum_hi = (eeum_hi - eeum)
            elif i == 2:
                # weighted quantiles
                rho_total_lo, rho_total, rho_total_hi = _quantile(x_valid, [0.025, 0.5, 0.975], weights=w_valid)
                print(f"rho total: {rho_total} kg/m^3, 95% CI = [{rho_total_lo:.6f}, {rho_total_hi:.6f}]")
                rho_total_lo = (rho_total - rho_total_lo)
                rho_total_hi = (rho_total_hi - rho_total)
            elif i == 3:
                # weighted quantiles
                v_init_lo, v_init, v_init_hi = _quantile(x_valid, [0.025, 0.5, 0.975], weights=w_valid)
                print(f"v init: {v_init} m/s, 95% CI = [{v_init_lo:.6f}, {v_init_hi:.6f}]")
                v_init_lo = (v_init - v_init_lo)
                v_init_hi = (v_init_hi - v_init)
            elif i == 4:
                # weighted quantiles
                m_init_lo, m_init, m_init_hi = _quantile(x_valid, [0.025, 0.5, 0.975], weights=w_valid)
                print(f"m init: {m_init} kg, 95% CI = [{m_init_lo:.6f}, {m_init_hi:.6f}]")
                m_init_lo = (m_init - m_init_lo)
                m_init_hi = (m_init_hi - m_init)


        max_logl_index = np.argmax(dynesty_run_results.logl)
        # get the best guess object erosion height
        best_eros_height = samples[max_logl_index, variables.index('erosion_height_start')].astype(float)/1000
        best_eros_height_change = best_eros_height
        if flag_total_rho:
            # get the one with the highest loglikelyhood erosion_height_change
            best_eros_height_change = samples[max_logl_index, variables.index('erosion_height_change')].astype(float)/1000




        beg_height = obs_data.height_lum[0]
        end_height = obs_data.height_lum[-1]

        # vel_init = obs_data.v_init
        lenght_par = obs_data.length[-1]/1000 # convert to km
        max_lum_height = obs_data.height_lum[np.argmax(obs_data.luminosity)]
        F_par = (beg_height - max_lum_height) / (beg_height - end_height)
        kc_par = beg_height/1000 + (2.86 - 2*np.log(v_init/1000))/0.0612
        zenith_angle = obs_data.zenith_angle
        print(f"beg_height: {beg_height} m")
        print(f"end_height: {end_height} m")
        print(f"max_lum_height: {max_lum_height} m")
        print(f"F_par: {F_par}")
        print(f"lenght_par: {lenght_par} km")
        print(f"kc_par: {kc_par}")
        print(f"rho_total: {rho_total} kg/m^3")
        print(f"zenith_angle: {zenith_angle}°")
        print(f"m_init: {m_init} kg")

        # save the results for the file
        file_eeu_dict[base_name] = (eeucs, eeum, F_par, kc_par, lenght_par, rho_total, zenith_angle, \
                                    energy_per_mass_before_first_erosion, energy_per_mass_before_second_erosion, \
                                    total_energy_before_erosion, total_energy_before_second_erosion, Tot_energy, m_init, v_init, best_eros_height, best_eros_height_change)

        # with open(folder_name + os.sep + base_name+"_dynesty_results_only.dynestyres", "rb") as f:
        #     results = pickle.load(f)

        # print("Results saved successfully.")
        # print(results.rho_total)

    eeucs = np.array([v[0] for v in file_eeu_dict.values()])
    eeum = np.array([v[1] for v in file_eeu_dict.values()])
    F_par = np.array([v[2] for v in file_eeu_dict.values()])
    kc_par = np.array([v[3] for v in file_eeu_dict.values()])
    lenght_par = np.array([v[4] for v in file_eeu_dict.values()])
    rho_total = np.array([v[5] for v in file_eeu_dict.values()])
    zenith_angle = np.array([v[6] for v in file_eeu_dict.values()])
    eeum_best = np.array([v[7] for v in file_eeu_dict.values()])
    eeum_best2 = np.array([v[8] for v in file_eeu_dict.values()])
    eebest_first = np.array([v[9] for v in file_eeu_dict.values()])
    eebest_second = np.array([v[10] for v in file_eeu_dict.values()])
    Tot_energy = np.array([v[11] for v in file_eeu_dict.values()])
    m_init = np.array([v[12] for v in file_eeu_dict.values()])
    v_init = np.array([v[13] for v in file_eeu_dict.values()])
    best_eros_height = np.array([v[14] for v in file_eeu_dict.values()])
    best_eros_height_change = np.array([v[15] for v in file_eeu_dict.values()])
    
    ###########################################################################################################

    # save i a .tex file the results in a table for ID F lenght_par eeucs
    with open(os.path.join(output_dir_show, name_distr + "_erosion_energy_results_table.tex"), "w") as f:
        f.write("\\begin{tabular}{lcccccccccc}\n")
        f.write("\\hline\n")
        f.write("ID & $E_{erosion}/A$ (MJ/m$^2$) & $E_{erosion}/m$ (MJ/kg) & F & $h_{kc}$ (km) & Length (km) & $\\rho$ (kg/m$^3$) & Zenith Angle (°) & $m_{init}$ (kg) & $E_{erosion}$ to $h_e$ (MJ) & $E_{erosion}$ to $h_{e2}$ (MJ) & Tot $E_{erosion}$ \\\\\n")
        f.write("\\hline\n")
        for base_name, (eeucs_1, eeum_1, F_par_1, kc_par_1, lenght_par_1, rho_total_1, zenith_angle_1, \
                        eeum_b1, eeum_b2, eebest_first_1, eebest_second_1, Tot_energy_1, m_init_1, v_init_1, best_eros_height_1, best_eros_height_change_1) in file_eeu_dict.items():
            f.write(f"{base_name} & {eeucs_1:.2f} & {eeum_1:.2f} & {F_par_1:.2f} & {kc_par_1:.2f} & {lenght_par_1:.2f} & {rho_total_1:.2f} & {zenith_angle_1:.2f} & {m_init_1:.2f} & {eebest_first_1:.2f} & {eebest_second_1:.2f} & {Tot_energy_1:.2f} \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
    print(f"Results table saved successfully in {os.path.join(output_dir_show, name_distr + '_erosion_energy_results_table.tex')}.")

    print("\nPlots:")

    ###########################################################################################################

    print("Bar plot for ID and 3 bar for eebest_first, eebest_second, Tot_energy...")

    # create a bar plot with the ID on the x axis and 3 bars for eebest_first, eebest_second, Tot_energy
    x = np.arange(len(file_eeu_dict))  # the label locations
    width = 0.25  # the width of the bars  
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width, eebest_first, width, label='Received Energy before $h_e$', color='tab:blue')
    bars2 = ax.bar(x, eebest_second, width, label='Received Energy before $h_{e2}$', color='tab:orange')
    bars3 = ax.bar(x + width, Tot_energy, width, label='Total Received Energy', color='tab:green')
    # add the value on top of each bar
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3g}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Energy [kJ]', fontsize=15)
    # set the y axis in log
    ax.set_yscale('log')
    # ax.set_title('Kinetic Energy before and after Erosion Height Change', fontsize=15)
    ax.set_xticks(x)
    ax.set_xticklabels(file_eeu_dict.keys(), rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir_show, f"{name_distr}_Lambda{lambda_val}_erosion_energy_bar_plot.png"), bbox_inches='tight', dpi=300)
    # close the plot
    plt.close()

    ###########################################################################################################

    # create a bar plot with the ID on the y axis and 2 bars one for the first Kinetic Energy and one for the second Kinetic Energy and put the line of the total fusion energy of iron
    print("Bar plot for ID and 2 bar for eeum_best, eeum_best2 with fusion energy of iron...")
    fusion_energy_iron_MJ = (13.81 / 55.845 + (1811 - 275) * 0.000449)  # MJ/kg
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, eeum_best, width, label='Received Energy per unit mass before $h_e$', color='tab:blue')
    bars2 = ax.bar(x + width/2, eeum_best2, width, label='Received Energy per unit mass before $h_{e2}$', color='tab:red')
    # add the value on top of each bar
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3g}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Energy per Unit mass [MJ/kg]', fontsize=15)
    # set the y axis in log
    ax.set_yscale('log')
    ax.axhline(y=fusion_energy_iron_MJ, color='purple', linestyle='-.', label='Tot.energy per unit mass for Fusion of Iron')
    ax.text(len(file_eeu_dict)-1, fusion_energy_iron_MJ*1.1, f'{fusion_energy_iron_MJ:.3g} MJ/kg', color='gray', fontsize=10, ha='right')
    # ax.set_title('Kinetic Energy per Unit Mass before and after Erosion Height Change', fontsize=15)
    ax.set_xticks(x)
    ax.set_xticklabels(file_eeu_dict.keys(), rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir_show, f"{name_distr}_Lambda{lambda_val}_erosion_energy_per_unit_mass_bar_plot.png"), bbox_inches='tight', dpi=300)
    # close the plot
    plt.close()

    ###########################################################################################################

    # create a pathch plot with the ID on the x axis and the eeucs on the y axis
    print("Patch plot for ID and best_eros_height and unit energy at best_eros_height...")
    fig, ax1 = plt.subplots(figsize=(12, 6))
    # for eah Id put a fill_between 
    for i, (base_name, (eeucs_1, eeum_1, F_par_1, kc_par_1, lenght_par_1, rho_total_1, zenith_angle_1, \
                        eeum_b1, eeum_b2, eebest_first_1, eebest_second_1, Tot_energy_1, m_init_1, v_init_1, best_eros_height_1, best_eros_height_change_1)) in enumerate(file_eeu_dict.items()):
        # cycle the colors
        color = plt.cm.tab20(i / len(file_eeu_dict))
        ax1.fill_between([eeum_best[i], eeum_best2[i]], best_eros_height_1, best_eros_height_change_1, color=color, alpha=0.5, label=base_name)
    ax1.set_xlabel('Energy per Unit mass [MJ/kg]', fontsize=15)
    ax1.set_ylabel('Height [km]', fontsize=15)
    # ax1.set_title('Erosion Height Range vs Energy per Unit Mass', fontsize=15)
    ax1.axvline(x=fusion_energy_iron_MJ, color='purple', linestyle='-.', label='Tot.energy per unit\nmass for Fusion of Iron')
    ax1.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    # grid on
    ax1.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir_show, f"{name_distr}_Lambda{lambda_val}_Patch_plot_erosion_height_energy_per_unit_mass.png"), bbox_inches='tight', dpi=300)
    # close the plot
    plt.close()


    ###########################################################################################################

    # plot the velocity in x axis and the mass in the y axis in log scale and color with rho_total
    print("Scatter plot for initial velocity vs initial mass ...")
    plt.figure(figsize=(10, 6))
    # after you’ve built your rho array:
    norm = Normalize(vmin=np.min((rho_total)), vmax=np.max((rho_total)))
    scatter = plt.scatter(v_init/1000, m_init, c=(rho_total), cmap='viridis', s=30,
                            norm=norm, zorder=2)
    # make the  bar color log scale
    plt.colorbar(scatter, label=r'$\rho$ (kg/m$^3$)')
    plt.gca().set_yscale('log')
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.xlabel('Initial Velocity (km/s)', fontsize=15)
    plt.ylabel('Initial Mass (kg)', fontsize=15)
    # increase the size of the tick labels
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tick_params(axis='both', which='minor', labelsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir_show, f"{name_distr}_initial_velocity_vs_initial_mass.png"), bbox_inches='tight', dpi=300)
    # close the plot
    plt.close()

    ###########################################################################################################

    # plot the distribution of rho_total

    print("Iron case F len erosion_energy_per_unit_cross_section ...")

    # plot the lenght_par against eeucs and color with F_par
    plt.figure(figsize=(10, 6))
    # after you’ve built your rho array:
    norm = Normalize(vmin=0, vmax=1)
    scatter = plt.scatter(lenght_par, eeucs, c=F_par, cmap='coolwarm_r', s=30,
                            norm=norm, zorder=2)
    plt.colorbar(scatter, label='F')
    plt.xlabel('Length (km)', fontsize=15)
    plt.ylabel('Received Energy per Unit Cross Section (MJ/m²)', fontsize=15)
    # increase the size of the tick labels
    plt.gca().tick_params(labelsize=15)

    # # annotate each point with its base_name in tiny text
    # for base_name, (eeucs_1, eeum_1, F_par_1, kc_par_1, lenght_par_1, rho_total_1) in file_eeu_dict.items():
    #     # print(f"Annotating {base_name} at length {lenght_par} km with eeucs {eeucs} MJ/m²")
    #     plt.annotate(
    #         base_name,
    #         xy=(lenght_par_1,eeucs_1),
    #         xytext=(30, 5),             # 5 points vertical offset
    #         textcoords='offset points',
    #         ha='center',
    #         va='bottom',
    #         fontsize=6,
    #         alpha=0.8
    #     )
    # invert the y axis to have the highest energy at the top
    plt.gca().invert_yaxis()
    # plt.title('Kinetic Energy per Unit Cross Section vs Length')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir_show, f"{name_distr}_Lambda{lambda_val}_erosion_energy_vs_length.png"), bbox_inches='tight', dpi=300)
    # close the plot
    plt.close()

    ###########################################################################################################

    # plot the lenght_par against eeucs and color with rho_total
    print("Iron case rho_total len erosion_energy_per_unit_cross_section ...")
    plt.figure(figsize=(10, 6))
    # after you’ve built your rho array:
    norm = Normalize(vmin=np.min(np.log10(rho_total)), vmax=np.max(np.log10(rho_total)))
    scatter = plt.scatter(lenght_par, eeucs, c=np.log10(rho_total), cmap='viridis', s=30,
                            norm=norm, zorder=2)
    plt.colorbar(scatter, label='log$_{10}$ $\\rho$ (kg/m³)')
    plt.xlabel('Length (km)', fontsize=15)
    plt.ylabel('Received Energy per Unit Cross Section (MJ/m²)', fontsize=15)
    # increase the size of the tick labels
    plt.gca().tick_params(labelsize=15)

    # # annotate each point with its base_name in tiny text
    # for base_name, (eeucs_1, eeum_1, F_par_1, kc_par_1, lenght_par_1, rho_total_1) in file_eeu_dict.items():
    #     # print(f"Annotating {base_name} at length {lenght_par} km with eeucs {eeucs} MJ/m²")
    #     plt.annotate(
    #         base_name,
    #         xy=(lenght_par_1,eeucs_1),
    #         xytext=(30, 5),             # 5 points vertical offset
    #         textcoords='offset points',
    #         ha='center',
    #         va='bottom',
    #         fontsize=6,
    #         alpha=0.8
    #     )
    # invert the y axis to have the highest energy at the top
    plt.gca().invert_yaxis()
    # plt.title('Kinetic Energy per Unit Cross Section vs Length')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir_show, f"{name_distr}_Lambda{lambda_val}_erosion_energy_vs_length_rho_total.png"), bbox_inches='tight', dpi=300)
    # close the plot
    plt.close()

    ###########################################################################################################

    # plot the distribution of rho_total

    print("Until end and energy up to erosion unit mass plot...")

    # # do the negative log of the m_initt 
    # m_init = abs(np.log10(m_init))
    # plot the lenght_par against eeucs and color with F_par
    plt.figure(figsize=(10, 10))
    # after you’ve built your rho array:
    scatter = plt.scatter(eeum, Tot_energy, c=np.log10(rho_total), cmap='viridis', s=30,
                            norm=norm, zorder=2)
    plt.colorbar(scatter, label='log$_{10}$ $\\rho$ (kg/m³)')
    plt.xlabel('Received Energy per Unit Mass before erosion (MJ/kg)', fontsize=15)
    plt.ylabel('Total Energy for complete ablation (kJ)', fontsize=15)
    # increase the size of the tick labels
    plt.gca().tick_params(labelsize=15)

    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir_show, f"{name_distr}_Lambda{lambda_val}_Tot_energy_vs_erosion_energy_x_unitMass.png"), bbox_inches='tight', dpi=300)
    # close the plot
    plt.close()

    ###########################################################################################################

    # plot the distribution of rho_total

    print("Until end and energy up to erosion unit area plot...")

    # plot the lenght_par against eeucs and color with F_par
    plt.figure(figsize=(10, 10))
    # after you’ve built your rho array:
    scatter = plt.scatter(eeucs, Tot_energy, c=np.log10(rho_total), cmap='viridis', s=30,
                            norm=norm, zorder=2)
    plt.colorbar(scatter, label='log$_{10}$ $\\rho$ (kg/m³)')
    plt.xlabel('Received Energy per Unit Cross Section before erosion (MJ/m²)', fontsize=15)
    plt.ylabel('Total Energy for complete ablation (kJ)', fontsize=15)
    # increase the size of the tick labels
    plt.gca().tick_params(labelsize=15)

    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir_show, f"{name_distr}_Lambda{lambda_val}_Tot_energy_vs_erosion_energy_x_unitCrossSec.png"), bbox_inches='tight', dpi=300)
    # close the plot
    plt.close()
    


if __name__ == "__main__":

    import argparse
    ### COMMAND LINE ARGUMENTS
    arg_parser = argparse.ArgumentParser(description="Run dynesty with optional .prior file.")
    # C:\Users\maxiv\Documents\UWO\Papers\3)Sporadics\Validation_nlive\nlive500
    # C:\Users\maxiv\Documents\UWO\Papers\3)Sporadics\3.2)Iron Letter\irons-rho_eta100-noPoros\Tau03
    arg_parser.add_argument('--input_dir', metavar='INPUT_PATH', type=str,
        default=r"C:\Users\maxiv\Documents\UWO\Papers\3.2)Iron Letter\irons-rho_eta100-noPoros\Best_iron-fit",
        help="Path to walk and find .pickle files.")
    
    arg_parser.add_argument('--output_dir', metavar='OUTPUT_DIR', type=str,
        default=r"",
        help="Output directory, if not given is the same as input_dir.")
    
    arg_parser.add_argument('--name', metavar='NAME', type=str,
        default=r"",
        help="Name of the input files, if not given is folders name.")

    arg_parser.add_argument('--lam', metavar='LAMBDA', type=float, default=1,
        help="Heat transfer coefficient.")

    arg_parser.add_argument('-new', '--new_eenres',
        help="Recompute the .eenres files, even if they exist.",
        action="store_true")

    # Parse
    cml_args = arg_parser.parse_args()

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

    extract_other_prop(cml_args.input_dir, cml_args.output_dir, cml_args.name, cml_args.lam, True)
    