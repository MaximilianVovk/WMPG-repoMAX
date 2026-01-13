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
from matplotlib.colors import Normalize
from dynesty import utils as dyfunc
from matplotlib.ticker import MaxNLocator, NullLocator, ScalarFormatter
from scipy.stats import gaussian_kde
from wmpl.Formats.WmplTrajectorySummary import loadTrajectorySummaryFast
from wmpl.MetSim.MetSimErosion import energyReceivedBeforeErosion
from types import SimpleNamespace
from wmpl.MetSim.MetSimErosionCyTools import atmDensityPoly

import numpy as np
from multiprocessing import Pool
from types import SimpleNamespace

# avoid showing warnings
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

def compute_temperature_profile_iron(
    heights_m,               # array [m] (top->bottom). Will be truncated to >= h_ero_start
    velocities_ms,           # array [m/s], same length as heights_m
    masses_kg,               # array [kg], same length as heights_m
    h_ero,
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
    # h_ero = float(const.erosion_height_start)

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


def extract_other_prop(input_dirfile_all, output_dir_show, lambda_val=1):
    """
    Function to plot the distribution of the parameters from the dynesty files and save them as a table in LaTeX format.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    for i in range(len(input_dirfile_all)):
        input_dirfile = input_dirfile_all[i]
        print(f"\nProcessing input: {input_dirfile}")

        # Use the class to find .dynesty, load prior, and decide output folders
        finder = find_dynestyfile_and_priors(input_dir_or_file=input_dirfile,prior_file="",resume=True,output_dir=input_dirfile,use_all_cameras=True,pick_position=0)

        num_meteors = len(finder.base_names)  # Number of meteors
        file_eeu_dict = {}
        print(f"Found {num_meteors} meteors in {input_dirfile}.")
        for ii, (base_name, dynesty_info, prior_path, out_folder) in enumerate(zip(finder.base_names,finder.input_folder_file,finder.priors,finder.output_folders)):
            dynesty_file, pickle_file, bounds, flags_dict, fixed_values = dynesty_info
            # split dynesty_file in folder and file name
            folder_name, _ = os.path.split(dynesty_file)
            print(f"\nProcessing {i+1}/{len(input_dirfile_all)}: {base_name} in {folder_name}")

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

            print(f"\nNo existing results found in {folder_name} .eenres, running dynesty.")
            dynesty_run_results = dsampler.results

            weights = dynesty_run_results.importance_weights()
            w = weights / np.sum(weights)

            sim_num = np.argmax(dynesty_run_results.logl)
            # best_guess_obj_plot = dynesty_run_results.samples[sim_num]
            # create a copy of the best guess
            best_guess = dynesty_run_results.samples[sim_num].copy()
            samples = dynesty_run_results.samples
            # for variable in variables: for 
            for jj, variable in enumerate(variables):
                if 'log' in flags_dict[variable]:
                    # print(f"Transforming {variable} from log scale to linear scale.{best_guess[jj]}")  
                    best_guess[jj] = 10**(best_guess[jj])
                    # print(f"Transforming {variable} from log scale to linear scale.{best_guess[jj]}")
                    samples[:, jj] = 10**(samples[:, jj])  # also transform all samples
                if variable == 'noise_lag':
                    obs_data.noise_lag = best_guess[jj]
                    obs_data.noise_vel = best_guess[jj] * np.sqrt(2)/(1.0/32)
                if variable == 'noise_lum':
                    obs_data.noise_lum = best_guess[jj]
                if variable == 'erosion_rho_change':
                    flag_total_rho = True

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
            # erosion_height_start = best_guess_obj_plot.const.erosion_height_start+1000
            ### get for each mass_best that is different from te previuse one get the height at which the mass loss happens
            diff_mask = np.concatenate(([True], np.diff(mass_best) != 0))
            ### only consider when mass actually changes enought
            # diff_mask = np.concatenate(([True], ~np.isclose(np.diff(mass_best), 0)))
            unique_heights_massvar = heights[diff_mask]
            mass_best_massvar = mass_best[diff_mask]
            velocities_massvar = velocities[diff_mask]

            if i==0:
                erosion_height_start = best_guess_obj_plot.const.erosion_height_start

            # Example density function using your existing polynomial (adjust to your implementation)
            def rho_a_poly(h_m):
                # If you have something like atmDensityPoly(h, const.dens_co):
                return atmDensityPoly(h_m, const_nominal.dens_co)

            # Run the temperature integration (iron properties baked in)
            h_used, Tm, dT, q_conv, q_rad, E_conv, E_rad, E_net = compute_temperature_profile_iron(
                heights_m=unique_heights_massvar,
                velocities_ms=velocities_massvar,
                masses_kg=mass_best_massvar,
                h_ero=erosion_height_start,
                const=best_guess_obj_plot.const,
                lambda_val=lambda_val,
                rho_a_fn=rho_a_poly,              # or your own atmosphere function
                T_a_fn=lambda h: 280.0,           # constant Ta per the paper assumption
                Tm0=280.0
            )

            # cycle through the color and use the same color for each tau value
            color_map = {0.08: 'green', 0.3: 'blue', 3: 'red', 1.0: 'purple'}

            
            # plot the temperature profile
            ax.plot(Tm, h_used/1000, color=color_map[constjson_bestfit.lum_eff], label=f'$\\tau$={constjson_bestfit.lum_eff}%')
            # create 
            ax.set_xlabel('Meteoroid Temperature [K]', fontsize=15)
            ax.set_ylabel('Height [km]', fontsize=15)
            # put a red dot at the height when Tm reaches 1811 K the melting point of iron
            melting_point_iron = 1811  # K
            # find the first index where Tm is higher than melting_point_iron
            melting_idx = np.where(Tm >= melting_point_iron)[0]
            if len(melting_idx) > 0:
                melting_height = h_used[melting_idx[0]] / 1000  # in km
                # point out the height when the melting point is reached
                ax.plot(melting_point_iron, melting_height, 'o', color=color_map[constjson_bestfit.lum_eff], label=f'1811 K Reached at {melting_height:.1f} km')
            # grid on
            ax.grid()
            
            # # plt.suptitle(f"Temperature and Power Profile for Best Fit: {base_name}")
            # plt.tight_layout()
    ax.axhline(y=erosion_height_start/1000, color='gray', linestyle='--', label=f'Erosion Height Start {erosion_height_start/1000:.1f} km') # $h_e$ = 
    ax.legend(fontsize=12)
    # take the current y axis values
    y_limits = ax.get_ylim()
    ax.set_ylim(y_limits[0], 120)  # set the y axis limit to 120 km
    plt.tight_layout()
    plt.savefig(output_dir_show + os.sep + base_name + "_temperature_change_tau.png")
    print(f"Saved temperature profile plot to {output_dir_show + os.sep + base_name + '_temperature_change_tau.png'}")
    plt.close()



if __name__ == "__main__":

    import argparse
    ### COMMAND LINE ARGUMENTS
    arg_parser = argparse.ArgumentParser(description="Run dynesty with optional .prior file.")
    # C:\Users\maxiv\Documents\UWO\Papers\3)Sporadics\Validation_nlive\nlive500
    # C:\Users\maxiv\Documents\UWO\Papers\3)Sporadics\3.2)Iron Letter\irons-rho_eta100-noPoros\Tau03
    arg_parser.add_argument('--input_dir', metavar='INPUT_PATH', type=str,
        default=[r"C:\Users\maxiv\Documents\UWO\Papers\2.1)Iron Letter\irons-rho_eta100-noPoros\Tau3\20210813_022604",
                 r"C:\Users\maxiv\Documents\UWO\Papers\2.1)Iron Letter\irons-rho_eta100-noPoros\Tau03\20210813_022604",
                 r"C:\Users\maxiv\Documents\UWO\Papers\2.1)Iron Letter\irons-rho_eta100-noPoros\Tau008\20210813_022604"],
        help="Path to walk and find .pickle files.")
    
    arg_parser.add_argument('--output_dir', metavar='OUTPUT_DIR', type=str,
        default=r"C:\Users\maxiv\Documents\UWO\Papers\2.1)Iron Letter\irons-rho_eta100-noPoros\Temp_plot",
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
        cml_args.output_dir = cml_args.input_dir[0]
    # check if the output_dir exists and create it if not
    if not os.path.exists(cml_args.output_dir):
        os.makedirs(cml_args.output_dir)

    extract_other_prop(cml_args.input_dir, cml_args.output_dir, cml_args.lam)
    