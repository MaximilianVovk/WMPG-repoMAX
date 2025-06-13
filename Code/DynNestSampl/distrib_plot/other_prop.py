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

import numpy as np
from multiprocessing import Pool
from types import SimpleNamespace

# avoid showing warnings
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

def run_single_sim(sim_num_and_data):
    sim_num, tot_sim, sample, obs_data, variables, fixed_values, flags_dict = sim_num_and_data

    print(f"Running simulation {sim_num}/{tot_sim}")
    
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

    # Try to simulate
    try:
        frag_main, results_list, wake_results = runSimulation(const_nominal, compute_wake=False)
        sim_obj = SimulationResults(const_nominal, frag_main, results_list, wake_results)
    except ZeroDivisionError:
        sim_obj = SimulationResults(const_nominal, *runSimulation(Constants(), compute_wake=False))

    # Extract physical quantities
    try:
        eeucs, eeum = energyReceivedBeforeErosion(const_nominal)

        heights = np.array(sim_obj.leading_frag_height_arr, dtype=np.float64)
        vel = np.array(sim_obj.leading_frag_vel_arr, dtype=np.float64)
        lum = np.array(sim_obj.luminosity_arr, dtype=np.float64)
        length = np.array(sim_obj.leading_frag_length_arr, dtype=np.float64)

        # Remove NaNs
        valid_heights = heights[:-1]
        valid_vel = vel[:-1]
        valid_lum = lum[:-1]
        valid_length = length[:-1]

        beg_height = obs_data.height_lum[0]
        end_height = obs_data.height_lum[-1]

        kc_par = beg_height/1000 + (2.86 - 2*np.log(valid_vel[np.argmin(np.abs(valid_heights - beg_height))]/1000))/0.0612
        max_lum_height = valid_heights[np.argmax(valid_lum)]
        F_par = beg_height - max_lum_height / (beg_height - end_height)
        if F_par < 0:
            F_par = 0
        if F_par > 1:
            F_par = 1
        trail_length = abs(valid_length[np.argmin(np.abs(valid_heights - beg_height))] - valid_length[np.argmin(np.abs(valid_heights - end_height))])

        rho_total = const_nominal.rho
        if flag_total_rho:
            mass_before = sim_obj.mass_total_active_arr[np.argmin(np.abs(valid_heights - const_nominal.erosion_height_change))]
            rho_total = (
                const_nominal.rho * ((const_nominal.m_init - mass_before) / const_nominal.m_init) +
                const_nominal.erosion_rho_change * (mass_before / const_nominal.m_init)
            )

        return (sim_num, eeucs, eeum, rho_total, kc_par, F_par, trail_length)

    except Exception as e:
        print(f"Simulation {sim_num} failed: {e}")
        return (sim_num, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)


def extract_other_prop(input_dirfile, output_dir_show):
    """
    Function to plot the distribution of the parameters from the dynesty files and save them as a table in LaTeX format.
    """
    # Use the class to find .dynesty, load prior, and decide output folders
    finder = find_dynestyfile_and_priors(input_dir_or_file=input_dirfile,prior_file="",resume=True,output_dir=input_dirfile,use_all_cameras=False,pick_position=0)

    num_meteors = len(finder.base_names)  # Number of meteors
    print(f"Found {num_meteors} meteors in {input_dirfile}.")
    for i, (base_name, dynesty_info, prior_path, out_folder) in enumerate(zip(finder.base_names,finder.input_folder_file,finder.priors,finder.output_folders)):
        dynesty_file, pickle_file, bounds, flags_dict, fixed_values = dynesty_info
        # split dynesty_file in folder and file name
        folder_name, _ = os.path.split(dynesty_file)
        print(f"Processing {i+1}/{num_meteors}: {base_name} in {folder_name}")

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
        dynesty_run_results = dsampler.results

        # load the variable names
        variables = list(flags_dict.keys())

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
            (i, len(dynesty_run_results.samples), dynesty_run_results.samples[i], obs_data, variables, fixed_values, flags_dict)
            for i in range(len(dynesty_run_results.samples)) # for i in np.linspace(0, len(dynesty_run_results.samples)-1, 10, dtype=int)
        ]
        #     for i in range(len(dynesty_run_results.samples)) # 
        num_cores = multiprocessing.cpu_count()

        # Run in parallel
        with Pool(processes=num_cores) as pool:  # adjust to number of cores
            results = pool.map(run_single_sim, inputs)

        N = len(dynesty_run_results.samples)

        erosion_energy_per_unit_cross_section_arr = np.full(N, np.nan)
        erosion_energy_per_unit_mass_arr = np.full(N, np.nan)
        rho_total_arr = np.full(N, np.nan)
        kc_par_arr = np.full(N, np.nan)
        F_par_arr = np.full(N, np.nan)
        trail_length_arr = np.full(N, np.nan)

        for res in results:
            i, eeucs, eeum, rho, kc, F, tl = res
            erosion_energy_per_unit_cross_section_arr[i] = eeucs
            erosion_energy_per_unit_mass_arr[i] = eeum
            rho_total_arr[i] = rho
            kc_par_arr[i] = kc
            F_par_arr[i] = F
            trail_length_arr[i] = tl

        # Create a namespace object for dot-style access
        results = SimpleNamespace(**dsampler.results.__dict__)  # load all default results

        # Add your custom attributes
        results.weights = dynesty_run_results.importance_weights()
        results.norm_weights = w
        results.erosion_energy_per_unit_cross_section = erosion_energy_per_unit_cross_section_arr
        results.erosion_energy_per_unit_mass = erosion_energy_per_unit_mass_arr
        results.rho_total = rho_total_arr
        results.kc_par = kc_par_arr
        results.F_par = F_par_arr
        results.trail_length = trail_length_arr

        # Save
        with open(folder_name + os.sep + base_name+"_dynesty_results_only.dynestyres", "wb") as f:
            pickle.dump(results, f)

        # with open(folder_name + os.sep + base_name+"_dynesty_results_only.dynestyres", "rb") as f:
        #     results = pickle.load(f)

        # print("Results saved successfully.")
        # print(results.rho_total)



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

    # check if cml_args.output_dir is empty and set it to the input_dir
    if cml_args.output_dir == "":
        cml_args.output_dir = cml_args.input_dir
    # check if the output_dir exists and create it if not
    if not os.path.exists(cml_args.output_dir):
        os.makedirs(cml_args.output_dir)

    extract_other_prop(cml_args.input_dir, cml_args.output_dir)
    