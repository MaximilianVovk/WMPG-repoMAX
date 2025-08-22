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

# def run_total_energy_received(sim_num_and_data):
#     sim_num, best_guess_obj_plot, unique_heights_massvar_init, unique_heights_massvar, mass_best_massvar, velocities_massvar_init = sim_num_and_data

#     # extract the const from the best_guess_obj_plot
#     const_nominal = best_guess_obj_plot.const

#     const_nominal.h_init = unique_heights_massvar_init
#     const_nominal.erosion_height_start = unique_heights_massvar
#     const_nominal.v_init = velocities_massvar_init
#     const_nominal.m_init = mass_best_massvar

#     # Extract physical quantities
#     try:
#         _, eeum = energyReceivedBeforeErosion(const_nominal)
#         total_energy = eeum * const_nominal.m_init  # Total energy received before erosion in MJ

#         return (sim_num, total_energy)

#     except Exception as e:
#         print(f"Simulation {sim_num} failed: {e}")
#         return (sim_num, np.nan)

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

def extract_other_prop(input_dirfile, output_dir_show):
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

        # load the variable names
        variables = list(flags_dict.keys())
        
        # look if in folder_name it exist a file that ends in .dynestyres exist in 
        if any(f.endswith(".dynestyres") for f in os.listdir(folder_name)):
            print(f"\nFound existing results in {folder_name}.dynestyres, loading them.")

            # look for the file that ends in .dynestyres
            dynesty_res_file = [f for f in os.listdir(folder_name) if f.endswith(".dynestyres")][0]
            with open(folder_name + os.sep + dynesty_res_file, "rb") as f:
                dynesty_run_results = pickle.load(f)

            erosion_energy_per_unit_cross_section_arr = dynesty_run_results.erosion_energy_per_unit_cross_section
            erosion_energy_per_unit_mass_arr = dynesty_run_results.erosion_energy_per_unit_mass
            erosion_energy_per_unit_cross_section_arr_end = dynesty_run_results.erosion_energy_per_unit_cross_section_end
            erosion_energy_per_unit_mass_arr_end = dynesty_run_results.erosion_energy_per_unit_mass_arr_end
            rho_total_arr = dynesty_run_results.rho_total

            samples = dynesty_run_results.samples

            weights = dynesty_run_results.weights
            w = weights / np.sum(weights)

        else:
            print(f"\nNo existing results found in {folder_name}.dynestyres, running dynesty.")
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
                (i, len(dynesty_run_results.samples), dynesty_run_results.samples[i], obs_data, variables, fixed_values, flags_dict)
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
            best_guess_obj_plot = run_simulation(best_guess, obs_data, variables, fixed_values)

            # find erosion change height
            if 'erosion_height_change' in variables:
                erosion_height_change = best_guess[variables.index('erosion_height_change')]
            if 'm_init' in variables:
                m_init = best_guess[variables.index('m_init')]

            heights = np.array(best_guess_obj_plot.leading_frag_height_arr, dtype=np.float64)[:-1]
            mass_best = np.array(best_guess_obj_plot.mass_total_active_arr, dtype=np.float64)[:-1]

            mass_before = mass_best[np.argmin(np.abs(heights - erosion_height_change))]


            # # precise erosion tal energy calculation ########################
            # velocities = np.array(best_guess_obj_plot.leading_frag_vel_arr, dtype=np.float64)[:-1]
            # erosion_height_start = best_guess_obj_plot.const.erosion_height_start
            # # get for each mass_best that is different from te previuse one get the height at which the mass loss happens
            # # diff_mask = np.concatenate(([True], np.diff(mass_best) != 0))
            # diff_mask = np.concatenate(([True], ~np.isclose(np.diff(mass_best), 0)))
            # unique_heights_massvar = heights[diff_mask]
            # mass_best_massvar = mass_best[diff_mask]
            # velocities_massvar = velocities[diff_mask]
            # # now delete any unique_heights_massvar and mass_best_massvar that are bigger than erosion_height_change
            # unique_heights_massvar = unique_heights_massvar[unique_heights_massvar <= erosion_height_start]
            # mass_best_massvar = mass_best_massvar[:len(unique_heights_massvar)]
            # velocities_massvar = velocities_massvar[:len(unique_heights_massvar)]
            # # print the unique_heights_massvar and next to it the mass_best_massvar
            # # add at the begnning the m_init to mass_best_massvar and h_init_best to unique_heights_massvar
            # unique_heights_massvar_init = np.concatenate(([best_guess_obj_plot.const.h_init], unique_heights_massvar))
            # mass_best_massvar = np.concatenate(([m_init], mass_best_massvar))
            # velocities_massvar_init = np.concatenate(([best_guess_obj_plot.const.v_init], velocities_massvar))
            # # deete the last element of unique_heights_massvar_init and mass_best_massvar
            # unique_heights_massvar_init = unique_heights_massvar_init[:-1]
            # mass_best_massvar = mass_best_massvar[:-1]
            # velocities_massvar_init = velocities_massvar_init[:-1]

            # # Package inputs
            # inputs = [
            #     (i, best_guess_obj_plot, unique_heights_massvar_init[i], unique_heights_massvar[i], mass_best_massvar[i], velocities_massvar_init[i])
            #     for i in range(len(mass_best_massvar)) # for i in np.linspace(0, len(dynesty_run_results.samples)-1, 10, dtype=int)
            # ]
            # #     for i in range(len(dynesty_run_results.samples)) # 
            # num_cores = multiprocessing.cpu_count()

            # # Run in parallel
            # with Pool(processes=num_cores) as pool:  # adjust to number of cores
            #     results = pool.map(run_total_energy_received, inputs)

            # N = len(mass_best_massvar)

            # Tot_energy_arr = np.full(N, np.nan)
            # for res in results:
            #     i, tot_en = res
            #     Tot_energy_arr[i] = tot_en / 1e3  # convert to kJ
            # # now sum Tot_energy
            # Tot_energy = np.sum(Tot_energy_arr)
            
            # # precise erosion tal energy calculation ########################

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
            results.erosion_energy_per_unit_cross_section_end = erosion_energy_per_unit_cross_section_arr_end
            results.erosion_energy_per_unit_mass_arr_end = erosion_energy_per_unit_mass_arr_end
            results.rho_total = rho_total_arr

            # delete from base_name _combined if it exists
            if '_combined' in base_name:
                base_name = base_name.replace('_combined', '')

            # Save
            with open(folder_name + os.sep + base_name+"_results.dynestyres", "wb") as f:
                pickle.dump(results, f)
                print(f"Results saved successfully in {folder_name + os.sep + base_name+'_results.dynestyres'}.")

        for i, x in enumerate([erosion_energy_per_unit_cross_section_arr, erosion_energy_per_unit_mass_arr, rho_total_arr, samples[:, variables.index('v_init')].astype(float), erosion_energy_per_unit_cross_section_arr_end, erosion_energy_per_unit_mass_arr_end, samples[:, variables.index('m_init')].astype(float)]):
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
                print(f"erosion energy per unit cross section: {eeucs} MJ/m^2, 95% CI = [{eeucs_lo:.6f}, {eeucs_hi:.6f}]")
                eeucs_lo = (eeucs - eeucs_lo)
                eeucs_hi = (eeucs_hi - eeucs)
            elif i == 1:
                # weighted quantiles
                eeum_lo, eeum, eeum_hi = _quantile(x_valid, [0.025, 0.5, 0.975], weights=w_valid)
                print(f"erosion energy per unit mass: {eeum} MJ/kg, 95% CI = [{eeum_lo:.6f}, {eeum_hi:.6f}]")
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
                eeucs_end_lo, eeucs_end, eeucs_end_hi = _quantile(x_valid, [0.025, 0.5, 0.975], weights=w_valid)
                print(f"erosion energy per unit cross section end: {eeucs_end} MJ/m², 95% CI = [{eeucs_end_lo:.6f}, {eeucs_end_hi:.6f}]")
                eeucs_end_lo = (eeucs_end - eeucs_end_lo)
                eeucs_end_hi = (eeucs_end_hi - eeucs_end)
            elif i == 5:
                # weighted quantiles
                eeum_end_lo, eeum_end, eeum_end_hi = _quantile(x_valid, [0.025, 0.5, 0.975], weights=w_valid)
                print(f"erosion energy per unit mass end: {eeum_end} MJ/kg, 95% CI = [{eeum_end_lo:.6f}, {eeum_end_hi:.6f}]")
                eeum_end_lo = (eeum_end - eeum_end_lo)
                eeum_end_hi = (eeum_end_hi - eeum_end)
            elif i == 6:
                # weighted quantiles
                m_init_lo, m_init, m_init_hi = _quantile(x_valid, [0.025, 0.5, 0.975], weights=w_valid)
                print(f"m init: {m_init} kg, 95% CI = [{m_init_lo:.6f}, {m_init_hi:.6f}]")
                m_init_lo = (m_init - m_init_lo)
                m_init_hi = (m_init_hi - m_init)


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

        # save the results for the file
        file_eeu_dict[base_name] = (eeucs, eeum, F_par, kc_par, lenght_par, rho_total, zenith_angle, eeucs_end, eeum_end, m_init)

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
    eeucs_end = np.array([v[7] for v in file_eeu_dict.values()])
    eeum_end = np.array([v[8] for v in file_eeu_dict.values()])
    m_init = np.array([v[9] for v in file_eeu_dict.values()])
    
    ###########################################################################################################

    # plot the distribution of rho_total

    print("\nIron case F len erosion_energy_per_unit_cross_section ...")

    # plot the lenght_par against eeucs and color with F_par
    plt.figure(figsize=(10, 6))
    # after you’ve built your rho array:
    norm = Normalize(vmin=0, vmax=1)
    scatter = plt.scatter(lenght_par, eeucs, c=F_par, cmap='coolwarm_r', s=30,
                            norm=norm, zorder=2)
    plt.colorbar(scatter, label='F')
    plt.xlabel('Length (km)', fontsize=15)
    plt.ylabel('Erosion Energy per Unit Cross Section (MJ/m²)', fontsize=15)
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
    # plt.title('Erosion Energy per Unit Cross Section vs Length')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir_show, f"erosion_energy_vs_length.png"), bbox_inches='tight', dpi=300)
    # close the plot
    plt.close()

    ###########################################################################################################

    # plot the lenght_par against eeucs and color with rho_total
    print("Iron case rho_total len erosion_energy_per_unit_cross_section ...")
    plt.figure(figsize=(10, 6))
    # after you’ve built your rho array:
    norm = Normalize(vmin=np.min(rho_total), vmax=np.max(rho_total))
    scatter = plt.scatter(lenght_par, eeucs, c=rho_total, cmap='viridis', s=30,
                            norm=norm, zorder=2)
    plt.colorbar(scatter, label='$\\rho$ (kg/m³)')
    plt.xlabel('Length (km)', fontsize=15)
    plt.ylabel('Erosion Energy per Unit Cross Section (MJ/m²)', fontsize=15)
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
    # plt.title('Erosion Energy per Unit Cross Section vs Length')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir_show, f"erosion_energy_vs_length_rho_total.png"), bbox_inches='tight', dpi=300)
    # close the plot
    plt.close()

    ###########################################################################################################

    # plot the distribution of rho_total

    print("Until end and energy up to erosion  unit mass plot...")

    # # do the negative log of the m_initt 
    # m_init = abs(np.log10(m_init))
    # plot the lenght_par against eeucs and color with F_par
    plt.figure(figsize=(10, 10))
    # after you’ve built your rho array:
    scatter = plt.scatter(eeum, eeum_end, c=rho_total, cmap='viridis', s=30,
                            norm=norm, zorder=2)
    plt.colorbar(scatter, label='$\\rho$ (kg/m³)')
    plt.xlabel('Erosion Energy per Unit Mass before erosion (MJ/kg)', fontsize=15)
    plt.ylabel('Total Energy for complete ablation per Unit Mass (MJ/kg)', fontsize=15)
    # increase the size of the tick labels
    plt.gca().tick_params(labelsize=15)

    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir_show, f"Tot_energy_vs_erosion_energy_x_unitMass.png"), bbox_inches='tight', dpi=300)
    # close the plot
    plt.close()

    ###########################################################################################################

    # plot the distribution of rho_total

    print("Until end and energy up to erosion unit area plot...")

    # plot the lenght_par against eeucs and color with F_par
    plt.figure(figsize=(10, 10))
    # after you’ve built your rho array:
    scatter = plt.scatter(eeucs, eeucs_end, c=rho_total, cmap='viridis', s=30,
                            norm=norm, zorder=2)
    plt.colorbar(scatter, label='$\\rho$ (kg/m³)')
    plt.xlabel('Erosion Energy per Unit Cross Section before erosion (MJ/m²)', fontsize=15)
    plt.ylabel('Total Energy for complete ablation per Unit Cross Section (MJ/m²)', fontsize=15)
    # increase the size of the tick labels
    plt.gca().tick_params(labelsize=15)

    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir_show, f"Tot_energy_vs_erosion_energy_x_unitCrossSec.png"), bbox_inches='tight', dpi=300)
    # close the plot
    plt.close()
    


if __name__ == "__main__":

    import argparse
    ### COMMAND LINE ARGUMENTS
    arg_parser = argparse.ArgumentParser(description="Run dynesty with optional .prior file.")
    
    arg_parser.add_argument('--input_dir', metavar='INPUT_PATH', type=str,
        default=r"C:\Users\maxiv\Documents\UWO\Papers\3)Sporadics\Slow_sporadics_with_EMCCD",
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
    