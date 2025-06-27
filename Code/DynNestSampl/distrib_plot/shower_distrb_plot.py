"""
Import all the pickle files and get the dynesty files distribution

Author: Maximilian Vovk
Date: 2025-04-16
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
from matplotlib.colors import Normalize
from dynesty import utils as dyfunc
from matplotlib.ticker import MaxNLocator, NullLocator, ScalarFormatter
from scipy.stats import gaussian_kde
from wmpl.Formats.WmplTrajectorySummary import loadTrajectorySummaryFast
from multiprocessing import Pool
from wmpl.MetSim.MetSimErosion import energyReceivedBeforeErosion


# avoid showing warnings
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


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
    

def shower_distrb_plot(input_dirfile, output_dir_show, shower_name):
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
        'noise_lag': r"$\varepsilon_{lag}$ [m]",
        'noise_lum': r"$\varepsilon_{lum}$ [W]"
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
        'noise_lag': r"$\varepsilon_{lag}$ [m]",
        'noise_lum': r"$\varepsilon_{lum}$ [W]"
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
        inclin_val = None
        
        re_i_val = re.compile(
            r'^\s*i\s*=\s*'                           
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

                if Tj is not None and inclin_val is not None:
                    break


        if Tj is None:
            raise RuntimeError(f"Couldn’t find any Tj line in {report_path!r}")
        if inclin_val is None:
            raise RuntimeError(f"Couldn’t find inclination (i) in {report_path!r}")

        print(f"Tj = {Tj:.6f} 95% CI = [{Tj_low:.6f}, {Tj_high:.6f}]")
        Tj_low = (Tj - Tj_low)#/1.96
        Tj_high = (Tj_high - Tj)#/1.96
        print(f"i = {inclin_val:.6f} deg")

        return Tj, Tj_low, Tj_high, inclin_val

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


    # the on that are not variables are the one that were not used in the dynesty run give a np.nan weight to dsampler for those
    all_samples = []
    all_weights = []

    # base_name, lg_min_la_sun, bg, rho
    file_radiance_rho_dict = {}
    file_radiance_rho_dict_helio = {}
    file_eeu_dict = {}
    file_rho_jd_dict = {}
    find_worst_lag = {}
    find_worst_lum = {}

    for i, (base_name, dynesty_info, prior_path, out_folder) in enumerate(zip(finder.base_names, finder.input_folder_file, finder.priors, finder.output_folders)):
        dynesty_file, pickle_file, bounds, flags_dict, fixed_values = dynesty_info
        print(base_name)
        obs_data = finder.observation_instance(base_name)
        obs_data.file_name = pickle_file  # update the file name in the observation data object
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
        F_par = beg_height - max_lum_height / (beg_height - end_height)
        kc_par = beg_height/1000 + (2.86 - 2*np.log(summary_df_meteor['Median'].values[variables_sing.index('v_init')]/1000))/0.0612


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


        if flag_total_rho:
            
            # find erosion change height
            if 'erosion_height_change' in variables_sing:
                erosion_height_change = guess[variables_sing.index('erosion_height_change')]
            if 'm_init' in variables_sing:
                m_init = guess[variables_sing.index('m_init')]


            best_guess_obj_plot = run_simulation(guess, obs_data, variables_sing, fixed_values)

            heights = np.array(best_guess_obj_plot.leading_frag_height_arr, dtype=np.float64)[:-1]
            mass_best = np.array(best_guess_obj_plot.mass_total_active_arr, dtype=np.float64)[:-1]

            mass_before = mass_best[np.argmin(np.abs(heights - erosion_height_change))]

            x = samples[:, variables_sing.index('rho')].astype(float)*(abs(m_init-mass_before) / m_init) + samples[:, variables_sing.index('erosion_rho_change')].astype(float) * (mass_before / m_init)
            mask = ~np.isnan(x)
            x_valid = x[mask]
            w_valid = w[mask]

            # renormalize
            w_valid /= np.sum(w_valid)

            # weighted quantiles
            rho_lo, rho, rho_hi = _quantile(x_valid, [0.025, 0.5, 0.975], weights=w_valid)
            rho_lo = (rho - rho_lo) #/1.96
            rho_hi = (rho_hi - rho) #/1.96

        else:
            rho_lo = summary_df_meteor['Median'].values[variables.index('rho')] - summary_df_meteor['Low95'].values[variables.index('rho')]
            rho_hi = summary_df_meteor['High95'].values[variables.index('rho')] - summary_df_meteor['Median'].values[variables.index('rho')]
            rho = summary_df_meteor['Median'].values[variables.index('rho')]
        
        print(f"rho: {rho} kg/m^3, 95% CI = [{rho_lo:.6f}, {rho_hi:.6f}]")
        
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
        #     erosion_energy_per_unit_mass_arr[i] = eeum / 1000000 # convert to MJ/kg
        
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

        tj, tj_lo, tj_hi, inclin_val = extract_tj_from_report(report_path)

        file_rho_jd_dict[base_name] = (rho, rho_lo,rho_hi, tj, tj_lo, tj_hi, inclin_val)
        # file_eeu_dict[base_name] = (eeucs, eeucs_lo, eeucs_hi, eeum, eeum_lo, eeum_hi,F_par, kc_par, lenght_par)

        find_worst_lag[base_name] = summary_df_meteor['Median'].values[variables.index('noise_lag')]
        find_worst_lum[base_name] = summary_df_meteor['Median'].values[variables.index('noise_lum')]

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

    lg_min_la_sun_helio = np.array([v[0] for v in file_radiance_rho_dict_helio.values()])
    lg_helio_lo = np.array([v[1] for v in file_radiance_rho_dict_helio.values()])
    lg_helio_hi = np.array([v[2] for v in file_radiance_rho_dict_helio.values()])
    bg_helio = np.array([v[3] for v in file_radiance_rho_dict_helio.values()])
    bg_helio_lo = np.array([v[4] for v in file_radiance_rho_dict_helio.values()])
    bg_helio_hi = np.array([v[5] for v in file_radiance_rho_dict_helio.values()])

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

    print("saving radiance plot...")

    # print(lg_lo, lg_hi, bg_lo, bg_hi)
    plt.figure(figsize=(8, 6))
    stream_lg_min_la_sun = []
    stream_bg = []

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
        stream_lor = stream_data[['LAMgeo (deg)', 'BETgeo (deg)', 'Sol lon (deg)','LAMhel (deg)', 'BEThel (deg)']].values
        # translate to double precision float
        stream_lor = stream_lor.astype(np.float64)
        # and now compute lg_min_la_sun = (lg - la_sun)%360
        stream_lg_min_la_sun = (stream_lor[:, 0] - stream_lor[:, 2]) % 360
        stream_bg = stream_lor[:, 1]
        stream_lg_min_la_sun_helio = (stream_lor[:, 3] - stream_lor[:, 2]) % 360
        stream_bg_helio = stream_lor[:, 4]
        # print(f"Found {len(stream_lg_min_la_sun)} stream data points for shower IAU number: {shower_iau_no}")

        if shower_iau_no != -1:
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
            cbar = plt.colorbar(scatter, label='Median density (kg/m$^3$)')
            # 2. now set the label’s font size and the tick labels’ size
            cbar.set_label('Median density (kg/m$^3$)', fontsize=15)
            cbar.ax.tick_params(labelsize=15)

            plt.xlabel(r'$\lambda_{g} - \lambda_{\odot}$ (J2000)', fontsize=15)
            plt.ylabel(r'$\beta_{g}$ (J2000)', fontsize=15)
            # plt.title('Radiant Distribution of Meteors')
            plt.grid(True)
            plt.savefig(os.path.join(output_dir_show, f"{shower_name}_geo_radiant_distribution_CI.png"), bbox_inches='tight', dpi=300)
            plt.close()

        else: 
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
                cbar.set_label('Median density (kg/m$^3$)', fontsize=13)
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

    ### JD vs rho plot ###
    print("saving rho vs JD plot...")
    # plot rho_lo and rho_hi as error bars and rho as points for tj and the error bars for tj_lo and tj_hi
    plt.figure(figsize=(8, 6))

    for i in range(len(tj)):
        # draw error bars for each point
        plt.errorbar(
            rho[i], tj[i],
            xerr=[[abs(rho_lo[i])],[abs(rho_hi[i])]],
            yerr=[[abs(tj_lo[i])], [abs(tj_hi[i])]],
            elinewidth=0.75,
            capthick=0.75,
            fmt='none',
            ecolor='black',
            capsize=3,
            zorder=1
        )

    # then draw points on top, at zorder=2 with black color
    scatter = plt.scatter(
        rho, tj,
        c=inclin_val,
        cmap='viridis',
        norm=Normalize(vmin=inclin_val.min(), vmax=inclin_val.max()),
        s=30,
        zorder=2
    )
    # increase the size of the tick labels
    plt.gca().tick_params(labelsize=15)
    # annotate each point with its base_name in tiny text
    # for base_name, (rho_val, rho_lo_val, rho_hi_val, tj_val, tj_lo_val, tj_hi_val, inclin_val) in file_rho_jd_dict.items():
    #     plt.annotate(
    #         base_name,
    #         xy=(rho_val, tj_val),
    #         xytext=(30, 5),             # 5 points vertical offset
    #         textcoords='offset points',
    #         ha='center',
    #         va='bottom',
    #         fontsize=6,
    #         alpha=0.8
    #     )

    # increase the label size
    cbar = plt.colorbar(scatter, label='Orbital inclination (deg)')

    # take the x axis limits
    xlim = plt.xlim()
    # take the y axis limits
    ylim = plt.ylim()

    if shower_iau_no == -1:
        # put a green horizontal line at Tj = 3.0
        plt.axhline(y=3.0, color='lime', linestyle=':', linewidth=1.5, zorder=1) # label='Tj = 3.0', 
        # write AST on the left side of the line
        plt.text(7500, 3.1, 'AST', color='black', fontsize=15, va='bottom')
        # put a red horizontal line at Tj = 2.0
        plt.axhline(y=2.0, color='lime', linestyle='--', linewidth=1.5, zorder=1) # label='Tj = 2.0', 
        # write APT on the left side of the line
        plt.text(7500, 2.3, 'JFC', color='black', fontsize=15, va='bottom')
        # # write below at 1.5 'HTC'
        # if the lowest ylim is below 1.5, then put a horizontal line at Tj = 1.5
        if ylim[0] < 1.5:
            plt.text(7500, 1.5, 'HTC', color='black', fontsize=15, va='bottom')

    # incrrease the x limits
    plt.xlim(-100, 8300)
    # increase the label size
    plt.xlabel(r'$\rho$ (kg/m$^3$)', fontsize=15)
    plt.ylabel(r'Tisserand parameter (T$_{j}$)', fontsize=15)
    # plt.title('rho vs Tj')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir_show, f"{shower_name}_rho_vs_Tj_CI.png"), bbox_inches='tight', dpi=300)

    #### Combine all samples and weights from different dynesty runs ####

    # Combine all the samples and weights into a single array
    combined_samples = np.vstack(all_samples)
    combined_weights = np.concatenate(all_weights)

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
 
    # Create a CombinedResults object for the combined samples
    combined_results = CombinedResults(combined_samples, combined_weights)

    summary_df = summarize_from_cornerplot(
        combined_results,
        variables,
        labels
    )


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
        if var in ['v_init', 'erosion_height_start', 'erosion_height_change']:
            combined_samples_copy_plot[:, j] = combined_samples_copy_plot[:, j] / 1000.0
        if var in ['erosion_coeff', 'sigma', 'erosion_coeff_change', 'erosion_sigma_change']:
            combined_samples_copy_plot[:, j] = combined_samples_copy_plot[:, j] * 1e6


    print('saving distribution plot...')

    # Extract from combined_results
    samples = combined_samples_copy_plot
    # samples = combined_results.samples
    weights = combined_results.importance_weights()
    w = weights / np.sum(weights)
    ndim = samples.shape[1]

    # Plot grid settings
    ncols = 5
    nrows = math.ceil(ndim / ncols)
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

        if variables[i] in ['erosion_mass_min', 'erosion_mass_max','m_init']: # 'log' in flags_dict_total.get(variables[i], '') and 
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
    # # takes forever, so run it last

    # print('saving corner plot...')

    # for i, variable in enumerate(variables):
    #     if 'log' in flags_dict_total[variable]:  
    #         labels_plot[i] =r"$\log_{10}$(" +labels_plot[i]+")"

    # # Define weighted correlation
    # def weighted_corr(x, y, w):
    #     """Weighted Pearson correlation of x and y with weights w."""
    #     w = np.asarray(w)
    #     x = np.asarray(x)
    #     y = np.asarray(y)
    #     w_sum = w.sum()
    #     x_mean = (w * x).sum() / w_sum
    #     y_mean = (w * y).sum() / w_sum
    #     cov_xy = (w * (x - x_mean) * (y - y_mean)).sum() / w_sum
    #     var_x  = (w * (x - x_mean)**2).sum() / w_sum
    #     var_y  = (w * (y - y_mean)**2).sum() / w_sum
    #     return cov_xy / np.sqrt(var_x * var_y)

    # # … your existing prep code …
    # fig, axes = plt.subplots(ndim, ndim, figsize=(35, 15))
    # axes = axes.reshape((ndim, ndim))

    # # call dynesty’s cornerplot
    # fg, ax = dyplot.cornerplot(
    #     combined_results, 
    #     color='blue',
    #     show_titles=True,
    #     max_n_ticks=3,
    #     quantiles=None,
    #     labels=labels_plot,
    #     label_kwargs={"fontsize": 10},
    #     title_kwargs={"fontsize": 12},
    #     title_fmt='.2e',
    #     fig=(fig, axes[:, :ndim])
    # )

    # # # supertitle, tick formatting, saving …
    # # fg.suptitle(shower_name, fontsize=16, fontweight='bold')

    # for ax_row in ax:
    #     for ax_ in ax_row:
    #         if ax_ is None:
    #             continue
    #         ax_.tick_params(axis='both', labelsize=8, direction='in')
    #         for lbl in ax_.get_xticklabels(): lbl.set_rotation(0)
    #         for lbl in ax_.get_yticklabels(): lbl.set_rotation(45)
    #         if len(ax_.xaxis.get_majorticklocs())>0:
    #             ax_.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.4g'))
    #         if len(ax_.yaxis.get_majorticklocs())>0:
    #             ax_.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4g'))

    # for i in range(ndim):
    #     for j in range(ndim):
    #         if ax[i, j] is None:
    #             continue
    #         if j != 0:
    #             ax[i, j].set_yticklabels([])
    #         if i != ndim - 1:
    #             ax[i, j].set_xticklabels([])

    # # Overlay weighted correlations in the upper triangle
    # samples = combined_results['samples'].T  # shape (ndim, nsamps)
    # weights = combined_results.importance_weights()

    # cmap = plt.colormaps['coolwarm']
    # norm = Normalize(vmin=-1, vmax=1)

    # for i in range(ndim):
    #     for j in range(ndim):
    #         if j <= i or ax[i, j] is None:
    #             continue

    #         panel = ax[i, j]
    #         x = samples[j]
    #         y = samples[i]
    #         corr_w = weighted_corr(x, y, weights)

    #         color = cmap(norm(corr_w))
    #         # paint the background patch
    #         panel.patch.set_facecolor(color)
    #         panel.patch.set_alpha(1.0)

    #         # fallback rectangle if needed
    #         panel.add_patch(
    #             plt.Rectangle(
    #                 (0,0), 1, 1,
    #                 transform=panel.transAxes,
    #                 facecolor=color,
    #                 zorder=0
    #             )
    #         )

    #         panel.text(
    #             0.5, 0.5,
    #             f"{corr_w:.2f}",
    #             transform=panel.transAxes,
    #             ha='center', va='center',
    #             fontsize=25, color='black'
    #         )
    #         panel.set_xticks([]); panel.set_yticks([])
    #         for spine in panel.spines.values():
    #             spine.set_visible(False)

    # # final adjustments & save
    # # fg.subplots_adjust(wspace=0.1, hspace=0.3)
    # fg.subplots_adjust(wspace=0.1, hspace=0.3, top=0.978) # Increase spacing between plots
    # plt.savefig(os.path.join(output_dir_show, f"{shower_name}_correlation_plot.png"),
    #             bbox_inches='tight', dpi=300)
    # plt.close(fig)

    # print('saving correlation matrix...')

    # # Build the NxN matrix of weigh_corr_ij
    # corr_mat = np.zeros((ndim, ndim))
    # for i in range(ndim):
    #     for j in range(ndim):
    #         corr_mat[i, j] = weighted_corr(samples[i], samples[j], weights)

    # # Wrap it in a DataFrame (so you get row/column labels)
    # df_corr = pd.DataFrame(
    #     corr_mat,
    #     index=labels_plot,
    #     columns=labels_plot
    # )

    # # Save to CSV (or TSV, whichever you prefer)
    # outpath = os.path.join(
    #     output_dir_show, f"{shower_name}_weighted_correlation_matrix.csv"
    # )
    # df_corr.to_csv(outpath, float_format="%.4f")
    # print(f"Saved weighted correlation matrix to:\n  {outpath}")

    # # Create a mask for the strict upper triangle (i<j), diagonal excluded
    # mask = np.triu(np.ones(df_corr.shape, dtype=bool), k=1)

    # # Keep only those entries
    # upper = df_corr.where(mask)

    # # Stack into a Series of (row, col) → corr_ij
    # pairs = upper.stack()

    # # For “top 10” by absolute strength:
    # top10 = pairs.sort_values(key=lambda x: x.abs(), ascending=False).head(10)
    # print("\nTop 10: highest correlations:")
    # print(top10)

    # # If you want the “bottom 10” (i.e. the smallest absolute correlations):
    # bottom10 = pairs.sort_values(key=lambda x: x.abs(), ascending=True).head(10)
    # print("\nBottom 10: lowest correlations:")
    # print(bottom10)



if __name__ == "__main__":

    import argparse
    ### COMMAND LINE ARGUMENTS
    arg_parser = argparse.ArgumentParser(description="Run dynesty with optional .prior file.")
    
    arg_parser.add_argument('--input_dir', metavar='INPUT_PATH', type=str,
         default=r"C:\Users\maxiv\Documents\UWO\Papers\3)Sporadics\Slow_sporadics",
        help="Path to walk and find .pickle files.")
    
    arg_parser.add_argument('--output_dir', metavar='OUTPUT_DIR', type=str,
        default=r"",
        help="Output directory, if not given is the same as input_dir.")
    
    arg_parser.add_argument('--name', metavar='NAME', type=str,
        default=r"",
        help="Name of the input files, if not given is folders name.")

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

    shower_distrb_plot(cml_args.input_dir, cml_args.output_dir, cml_args.name)
    