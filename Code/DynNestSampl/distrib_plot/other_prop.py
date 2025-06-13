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


def run_single_sim(sim_num_and_data):
    sim_num, sample, obs_data, variables, fixed_values, flags_dict = sim_num_and_data
    
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
        valid_heights = heights[~np.isnan(heights)]
        valid_vel = vel[~np.isnan(vel)]
        valid_lum = lum[~np.isnan(lum)]
        valid_length = length[~np.isnan(length)]

        beg_height = obs_data.height_lum[0]
        end_height = obs_data.height_lum[-1]

        kc_par = beg_height/1000 + (2.86 - 2*np.log(valid_vel[np.argmin(np.abs(valid_heights - beg_height))]/1000))/0.0612
        max_lum_height = valid_heights[np.argmax(valid_lum)]
        F_par = (beg_height - max_lum_height) / (beg_height - end_height)
        trail_length = valid_length[np.argmin(np.abs(valid_heights - beg_height))] - valid_length[np.argmin(np.abs(valid_heights - end_height))]

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
        print(f"Processing {i+1}/{num_meteors}: {base_name} in {out_folder}")

        dynesty_file, pickle_file, bounds, flags_dict, fixed_values = dynesty_info

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

        for attr in dir(dynesty_run_results):
            if not attr.startswith("_") and not callable(getattr(dynesty_run_results, attr)):
                print(attr, ":", type(getattr(dynesty_run_results, attr)))






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
        
        beg_height = obs_data.height_lum[0]
        end_height = obs_data.height_lum[-1]

        flag_total_rho = False
        erosion_energy_per_unit_cross_section_arr = np.zeros(len(dynesty_run_results.samples))
        erosion_energy_per_unit_mass_arr = np.zeros(len(dynesty_run_results.samples))
        rho_total_arr = np.zeros(len(dynesty_run_results.samples))
        kc_par_arr = np.zeros(len(dynesty_run_results.samples))
        F_par_arr = np.zeros(len(dynesty_run_results.samples))
        trail_length_arr = np.zeros(len(dynesty_run_results.samples))
        # for sim_num in range(len(dynesty_run_results.samples)):
        #
        for sim_num in np.linspace(0, len(dynesty_run_results.samples)-1, 10, dtype=int):
            print(f"{base_name}: Processing simulation {sim_num+1}/{len(dynesty_run_results.samples)}")
            guess = dynesty_run_results.samples[sim_num]
            # for variable in variables: for 
            for i, variable in enumerate(variables):
                if 'log' in flags_dict[variable]:  
                    guess[i] = 10**(guess[i])
                if 'noise_lag' == variable:
                    obs_data.noise_lag = guess[i]
                    obs_data.noise_vel = guess[i]*np.sqrt(2)/(1.0/32)
                if 'noise_lum' == variable:
                    obs_data.noise_lum = guess[i]
                if 'erosion_rho_change' == variable:
                    flag_total_rho = True



            ##### CREATE THE SIMULATION OBJECT #####
            parameter_guess, real_event, var_names, fix_var = guess, obs_data, variables, fixed_values

            # Load the nominal simulation parameters
            const_nominal = Constants()
            const_nominal.dens_co = np.array(const_nominal.dens_co)

            dens_co=real_event.dens_co

            ### Calculate atmosphere density coeffs (down to the bottom observed height, limit to 15 km) ###

            # Assign the density coefficients
            const_nominal.dens_co = dens_co

            # # Turn on plotting of LCs of individual fragments 
            # const_nominal.fragmentation_show_individual_lcs = True

            # for loop for the var_cost that also give a number from 0 to the length of the var_cost
            for i, var in enumerate(var_names):
                const_nominal.__dict__[var] = parameter_guess[i]

            # first chack if fix_var is not {}
            if fix_var:
                var_names_fix = list(fix_var.keys())
                # for loop for the fix_var that also give a number from 0 to the length of the fix_var
                for i, var in enumerate(var_names_fix):
                    const_nominal.__dict__[var] = fix_var[var]

            # if the real_event has an initial velocity lower than 30000 set "dt": 0.005 to "dt": 0.01
            const_nominal.dt = real_event.dt
            # const_nominal.erosion_bins_per_10mass = 5

            const_nominal.P_0m = real_event.P_0m

            const_nominal.disruption_on = real_event.disruption_on

            const_nominal.lum_eff_type = real_event.lum_eff_type

            # Minimum height [m]
            const_nominal.h_kill = real_event.h_kill 

            # minim velocity [m/s]
            const_nominal.v_kill = real_event.v_kill
            
            # # Initial meteoroid height [m]
            # const_nominal.h_init = 180000

            try:
                # Run the simulation
                frag_main, results_list, wake_results = runSimulation(const_nominal, compute_wake=False)
                simulation_MetSim_object = SimulationResults(const_nominal, frag_main, results_list, wake_results)
            except ZeroDivisionError as e:
                print(f"Error during simulation: {e}")
                # run again with the nominal values to avoid the error
                const_nominal = Constants()
                # Run the simulation
                frag_main, results_list, wake_results = runSimulation(const_nominal, compute_wake=False)
                simulation_MetSim_object = SimulationResults(const_nominal, frag_main, results_list, wake_results)

            best_guess_obj_plot = simulation_MetSim_object




            ##### EXTRACT from OBJECT #####

            '''
            dict_keys(['const', 'frag_main', 'time_arr', 'luminosity_arr', 'luminosity_main_arr', 'luminosity_eroded_arr', 
            'electron_density_total_arr', 'tau_total_arr', 'tau_main_arr', 'tau_eroded_arr', 'brightest_height_arr', 
            'brightest_length_arr', 'brightest_vel_arr', 'leading_frag_height_arr', 'leading_frag_length_arr', 
            'leading_frag_vel_arr', 'leading_frag_dyn_press_arr', 'mass_total_active_arr', 'main_mass_arr', 
            'main_height_arr', 'main_length_arr', 'main_vel_arr', 'main_dyn_press_arr', 'abs_magnitude', 
            'abs_magnitude_main', 'abs_magnitude_eroded', 'wake_results', 'wake_max_lum'])

            in const

            dict_keys(['dt', 'total_time', 'n_active', 'm_kill', 'v_kill', 'h_kill', 'len_kill', 'h_init', 'P_0m', 
            'dens_co', 'r_earth', 'total_fragments', 'wake_psf', 'wake_extension', 'rho', 'm_init', 'v_init', 
            'shape_factor', 'sigma', 'zenith_angle', 'gamma', 'rho_grain', 'lum_eff_type', 'lum_eff', 'mu', 
            'erosion_on', 'erosion_bins_per_10mass', 'erosion_height_start', 'erosion_coeff', 'erosion_height_change', 
            'erosion_coeff_change', 'erosion_rho_change', 'erosion_sigma_change', 'erosion_mass_index', 'erosion_mass_min', 
            'erosion_mass_max', 'disruption_on', 'compressive_strength', 'disruption_height', 'disruption_erosion_coeff', 
            'disruption_mass_index', 'disruption_mass_min_ratio', 'disruption_mass_max_ratio', 'disruption_mass_grain_ratio', 
            'fragmentation_on', 'fragmentation_show_individual_lcs', 'fragmentation_entries', 'fragmentation_file_name', 
            'electron_density_meas_ht', 'electron_density_meas_q', 'erosion_beg_vel', 'erosion_beg_mass', 'erosion_beg_dyn_press', 
            'mass_at_erosion_change', 'energy_per_cs_before_erosion', 'energy_per_mass_before_erosion', 'main_mass_exhaustion_ht', 'main_bottom_ht'])
            '''

            erosion_energy_per_unit_cross_section, erosion_energy_per_unit_mass = energyReceivedBeforeErosion(const_nominal)

            # erosion_energy_per_unit_mass / 1000000 MJ/kg
            # erosion_energy_per_unit_cross_section / 1000000 MJ/m^2
            # print(f"Erosion energy per unit mass: {erosion_energy_per_unit_mass/1000000:.2f} MJ/kg")
            # print(f"Erosion energy per unit cross section: {erosion_energy_per_unit_cross_section/1000000:.2f} MJ/m^2")

            best_guess_obj_plot.fps_lum = obs_data.fps_lum 
            best_guess_obj_plot.P_0m = obs_data.P_0m 
            # if ((1/obs_data.fps_lum) > best_guess_obj_plot.const.dt):
            #     # integration time step lumionosity
            #     best_guess_obj_plot.luminosity_arr, abs_magnitude = luminosity_integration(best_guess_obj_plot.time_arr,best_guess_obj_plot.time_arr,best_guess_obj_plot.luminosity_arr,best_guess_obj_plot.const.dt,best_guess_obj_plot.fps_lum,best_guess_obj_plot.P_0m)

            # Convert to numpy array if not already
            heights = np.array(best_guess_obj_plot.leading_frag_height_arr, dtype=np.float64)
            # Remove NaNs
            valid_heights = heights[~np.isnan(heights)]
            # Convert to numpy array if not already
            vel = np.array(best_guess_obj_plot.leading_frag_vel_arr, dtype=np.float64)
            # Remove NaNs
            valid_vel = vel[~np.isnan(vel)]
            # Convert to numpy array if not already
            lum = np.array(best_guess_obj_plot.luminosity_arr, dtype=np.float64)
            # Remove NaNs
            valid_lum = lum[~np.isnan(lum)]
            # Conveert to numpy array if not already leading_frag_length_arr
            leading_frag_length_arr = np.array(best_guess_obj_plot.leading_frag_length_arr, dtype=np.float64)
            # Remove NaNs
            valid_length = leading_frag_length_arr[~np.isnan(leading_frag_length_arr)]


            kc_par = beg_height/1000 + (2.86 - 2*np.log(valid_vel[np.argmin(np.abs(valid_heights - beg_height))]/1000))/0.0612
            
            # find the heigh were there is the maximum luminosity
            max_lum_height = heights[np.argmax(valid_lum)]
            F_par = (beg_height - (max_lum_height)) / (beg_height - end_height)

            # trail length
            trail_length = abs(valid_length[np.argmin(np.abs(valid_heights - beg_height))] - valid_length[np.argmin(np.abs(valid_heights - end_height))])
            
            if flag_total_rho:
                # find the closest best_guess_obj_plot.leading_frag_height_arr to the erosion_height_change and pick the associated best_guess_obj_plot.mass_total_active_arr
                mass_before_second_erosion = best_guess_obj_plot.mass_total_active_arr[np.argmin(np.abs(valid_heights - const_nominal.erosion_height_change))]
                # crate a weighted average of the const_nominal.erosion_height_change and the mass_before_erosion
                rho_total = const_nominal.rho* ((const_nominal.m_init-mass_before_second_erosion)/const_nominal.m_init) + const_nominal.erosion_rho_change * (mass_before_second_erosion/const_nominal.m_init)
                # print('##########################################')
                # print('mass initial ',const_nominal.m_init,'kg mass_before_second_erosion ', mass_before_second_erosion, 'kg')
                # print(f"second erosion heigh: {const_nominal.erosion_height_change} m, closest height: {valid_heights[np.argmin(np.abs(valid_heights - const_nominal.erosion_height_change))]} m")
                # print(f"Total density before second erosion: {const_nominal.rho} kg/m^3 with erosion density change: {const_nominal.erosion_rho_change} kg/m^3")
                # print(f"Total density : {rho_total} kg/m^3")
                # print('##########################################')
            else:
                rho_total = const_nominal.rho

            # add to the arrays to the sim_num position
            erosion_energy_per_unit_cross_section_arr[sim_num] = erosion_energy_per_unit_cross_section
            erosion_energy_per_unit_mass_arr[sim_num] = erosion_energy_per_unit_mass
            rho_total_arr[sim_num] = rho_total
            kc_par_arr[sim_num] = kc_par
            F_par_arr[sim_num] = F_par   
            trail_length_arr[sim_num] = trail_length         
            
            # print(base_name + "_fitN."+str(sim_num)+" -LnZ : "+str(abs(np.round(dynesty_run_results.logvol[sim_num],2)))+" weights : "+str((w[sim_num]))," logwt : "+ str(abs(np.round(dynesty_run_results.logwt[sim_num],2))))
            # plot_data_with_residuals_and_real(obs_data, best_guess_obj_plot, output_dir_show, base_name + "_fit_"+str(sim_num+1)+"_-LnZ_"+str(abs(np.round(dynesty_run_results.logvol[sim_num],2))), color_sim='black', label_sim="sim "+str(sim_num+1))

        ### save the simulation object to the dsampler ###

        print(f"Saving results for {base_name} to {output_dir_show}")
        print(erosion_energy_per_unit_cross_section_arr)
        print(erosion_energy_per_unit_mass_arr)
        print(rho_total_arr)
        print(kc_par_arr)
        print(F_par_arr)
        print(trail_length_arr)


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
        with open(output_dir_show + os.sep + "dynesty_results_only.dynestyres", "wb") as f:
            pickle.dump(results, f)

        # with open(output_dir_show + os.sep + "dynesty_results_only.dynestyres", "rb") as f:
        #     results = pickle.load(f)

        # print("Results saved successfully.")
        # print(results.rho_total)

        # # Extract results as a dict (do this FIRST)
        # results_dict = dsampler.results.__dict__.copy()  # now it's a mutable regular dict

        # # Add new item
        # results_dict['weights'] = dynesty_run_results.importance_weights()
        # results_dict['norm_weights'] = w
        # results_dict['erosion_energy_per_unit_cross_section'] = erosion_energy_per_unit_cross_section_arr
        # results_dict['erosion_energy_per_unit_mass'] = erosion_energy_per_unit_mass_arr
        # results_dict['rho_total'] = rho_total_arr
        # results_dict['kc_par'] = kc_par_arr
        # results_dict['F_par'] = F_par_arr
        # results_dict['trail_length'] = trail_length_arr

        # print("Saving results to file...")
        # # Save it
        # output_path = os.path.join(output_dir_show, "dynesty_results_only.dynestyres")
        # with open(output_path, "wb") as f:
        #     pickle.dump(results_dict, f)

        # with open(output_dir_show + os.sep + "dynesty_results_only.dynestyres", "rb") as f:
        #     results = pickle.load(f)

        # # Access examples
        # print(results['samples'].shape)
        # print(results['logz'])
        # print(results['weights'])
        # print(results['erosion_energy_per_unit_cross_section'])
        # print(results['erosion_energy_per_unit_mass'])


if __name__ == "__main__":

    import argparse
    ### COMMAND LINE ARGUMENTS
    arg_parser = argparse.ArgumentParser(description="Run dynesty with optional .prior file.")
    
    arg_parser.add_argument('--input_dir', metavar='INPUT_PATH', type=str,
         default=r"C:\Users\maxiv\Documents\UWO\Papers\3)Sporadics\iron_erosion_coeff\results\20190704_072615_iron_2fr",
        help="Path to walk and find .pickle files.")
    
    arg_parser.add_argument('--output_dir', metavar='OUTPUT_DIR', type=str,
        default=r"C:\Users\maxiv\Documents\UWO\Papers\3)Sporadics\iron_erosion_coeff",
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
    