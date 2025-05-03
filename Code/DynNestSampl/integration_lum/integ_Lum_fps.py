"""
Import all the pickle files and get the luminosity of the first fram of all the files

Author: Maximilian Vovk
Date: 2025-04-08
"""

# main.py (inside my_subfolder)
import sys
import os

# Add the parent directory to the sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from DynNestSapl_metsim import *
from scipy import stats

from scipy.optimize import minimize
import numpy as np
import scipy.optimize as opt
import matplotlib.gridspec as gridspec
from scipy import stats

input_dirfile = r"C:\Users\maxiv\WMPG-repoMAX\Code\DynNestSampl\integration_lum\CAP_new_2frg"

# Use the class to find .dynesty, load prior, and decide output folders
finder = find_dynestyfile_and_priors(input_dir_or_file=input_dirfile,prior_file="",resume=True,output_dir=input_dirfile,use_all_cameras=False,pick_position=0)

for i, (base_name, dynesty_info, prior_path, out_folder) in enumerate(zip(finder.base_names,finder.input_folder_file,finder.priors,finder.output_folders)):
    dynesty_file, pickle_file, bounds, flags_dict, fixed_values = dynesty_info
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

    dsampler = dynesty.DynamicNestedSampler.restore(dynesty_file)
    # dsampler = dynesty.DynamicNestedSampler.restore(dynesty_file)
    dynesty_run_results = dsampler.results.copy()
    # make a deep copy of the results
    file_name = base_name

    # check if _combined in the file_name
    if '_combined' in file_name:
        # remove _combined from the file_name
        file_name = file_name.replace('_combined', '')

    variables = list(flags_dict.keys())
    sim_num = np.argmax(dynesty_run_results.logl)   
    best_guess = copy.deepcopy(dynesty_run_results.samples[sim_num])
    # for variable in variables: for 
    for i, variable in enumerate(variables):
        if 'log' in flags_dict[variable]:  
            best_guess[i] = 10**(best_guess[i])

    sim_data = run_simulation(best_guess, obs_data, variables, fixed_values)

    fig = plt.figure(figsize=(12,6), dpi=300)

    # Use the user-specified GridSpec
    gs_main = gridspec.GridSpec(1, 2, figure=fig)

    # Define colormap
    cmap = plt.get_cmap("tab10")
    station_colors = {}  # Dictionary to store colors assigned to stations

    # Adjust subplots: give some horizontal space (wspace) so there's separation between the pairs (5 & 6)
    # We'll later manually remove space between (4,5) and (6,7) by adjusting positions.
    fig.subplots_adjust(wspace=0.3, hspace=0.3)

    # integration time step in self.const.dt for luminosity integration and abs_magnitude_integration
    if (1/obs_data.fps_lum) > sim_data.const.dt:
        integr_luminosity_arr, integr_abs_magnitude = luminosity_integration(sim_data.time_arr,sim_data.time_arr,sim_data.luminosity_arr,sim_data.const.dt,obs_data.fps_lum,obs_data.P_0m)

    ax0 = fig.add_subplot(gs_main[0,0])  # height vs luminosity
    ax1 = fig.add_subplot(gs_main[0,1])  # luminosity vs time

    # --- Plotting Data --- #

    # for each station in obs_data_plot
    for station in np.unique(obs_data.stations_lum):
        # Assign a unique color if not already used
        if station not in station_colors:
            station_colors[station] = cmap(len(station_colors) % 10)  # Use modulo to cycle colors
        # plot the height vs. absolute_magnitudes
        ax0.plot(obs_data.luminosity[np.where(obs_data.stations_lum == station)], \
                    obs_data.height_lum[np.where(obs_data.stations_lum == station)]/1000, 'x--', \
                    color=station_colors[station], label=station)
        # plot the luminosity vs. time
        ax1.plot(obs_data.time_lum[np.where(obs_data.stations_lum == station)], \
                    obs_data.luminosity[np.where(obs_data.stations_lum == station)], 'x--', \
                    color=station_colors[station], label=station)

    # save the y-axis limits
    ylim_lum = ax0.get_ylim()
    # fix the y-axis limits to ylim_lum
    ax0.set_ylim(ylim_lum)

    # save the x-axis limits
    xlim_time = ax1.get_xlim()
    # fix the x-axis limits to xlim_time
    ax1.set_xlim(xlim_time)

    the_fontsize = 12

    # now plot the simulated data
    ax0.set_xlabel('Luminosity [W]', fontsize=the_fontsize)
    # ax4.tick_params(axis='x', rotation=45)
    ax0.set_ylabel('Height [km]', fontsize=the_fontsize)

    ax1.set_ylabel('Luminosity [W]', fontsize=the_fontsize)
    # ax4.tick_params(axis='x', rotation=45)
    ax1.set_xlabel('Time [s]', fontsize=the_fontsize)

    simulated_time = np.interp(obs_data.height_lum, 
                                       np.flip(sim_data.leading_frag_height_arr), 
                                       np.flip(sim_data.time_arr))

    # all simulated time is the time_arr subtract the first time of the simulation
    all_simulated_time = sim_data.time_arr-simulated_time[0]

    # plot the simulated data
    ax0.plot(sim_data.luminosity_arr, sim_data.leading_frag_height_arr/1000, ':', color='darkgray', label='Simulated',linewidth=2)
    ax1.plot(all_simulated_time, sim_data.luminosity_arr, ':', color='darkgray', label='Simulated',linewidth=2)

    # save the x-axis limits
    xlim_lum = ax0.get_xlim()
    # fix the x-axis limits to xlim_lum
    ax0.set_xlim(xlim_lum)

    # save the y-axis limits
    ylim_time = ax1.get_ylim()
    # fix the y-axis limits to ylim_time
    ax1.set_ylim(ylim_time)

    # plot the simulated data with the integration time step in self.const.dt for luminosity integration and abs_magnitude_integration
    if (1/obs_data.fps_lum) > sim_data.const.dt:
        ax0.plot(integr_luminosity_arr, sim_data.leading_frag_height_arr/1000, color='black', label='Simulated Integr.',linewidth=2)
        ax1.plot(all_simulated_time, integr_luminosity_arr, color='black', label='Simulated Integr.',linewidth=2)

    ax1.legend()

    ax0.grid()
    ax1.grid()

    # # redduce the space between the two plots
    # plt.subplots_adjust(hspace=0.01)

    plt.savefig(os.path.join(input_dirfile, file_name+'_sim_and_integ_sim.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved figure for {os.path.join(input_dirfile, file_name+'_sim_and_integ_sim.png')}")
