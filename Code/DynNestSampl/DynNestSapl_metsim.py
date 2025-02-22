import numpy as np
import pandas as pd
import pickle
import sys
import json
import os
import copy
import matplotlib.pyplot as plt
import dynesty
from dynesty import plotting as dyplot
import time
from matplotlib.ticker import ScalarFormatter
import scipy
from scipy.stats import norm
from scipy.stats import multivariate_normal
import warnings
import re
import matplotlib.gridspec as gridspec

from wmpl.MetSim.GUI import loadConstants, saveConstants,SimulationResults
from wmpl.MetSim.MetSimErosion import runSimulation, Constants, zenithAngleAtSimulationBegin
from wmpl.MetSim.ML.GenerateSimulations import generateErosionSim,saveProcessedList,MetParam
from wmpl.Utils.Math import lineFunc, mergeClosePoints, findClosestPoints, vectMag, vectNorm, lineFunc, meanAngle
from wmpl.Utils.Physics import calcMass, dynamicPressure, calcRadiatedEnergy
from wmpl.Utils.TrajConversions import J2000_JD, date2JD
from wmpl.Utils.AtmosphereDensity import fitAtmPoly
from wmpl.Utils.Pickling import loadPickle

import signal


# Plotting function
def plot_data_with_residuals_and_real(obs_data, sim_data=None, output_folder='',file_name=''):

    # Create the figure and main GridSpec with specified height ratios
    fig = plt.figure(figsize=(14, 6))
    gs_main = gridspec.GridSpec(2, 4, figure=fig, height_ratios=[1, 1], width_ratios=[1, 1, 1, 1])

    # Define colormap
    cmap = plt.get_cmap("tab10")
    station_colors = {}  # Dictionary to store colors assigned to stations

    ### ABSOLUTE MAGNITUDES PLOT ###

    # Create a sub GridSpec for Plot 0 and Plot 1 with width ratios
    gs01 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_main[0, 0:2], wspace=0, width_ratios=[3, 1])

    # Plot 0 and 1: Side by side, sharing the y-axis
    ax0 = fig.add_subplot(gs01[0])
    ax1 = fig.add_subplot(gs01[1], sharey=ax0)

    # for each station in obs_data_plot
    for station in np.unique(obs_data.stations_lum):
        # Assign a unique color if not already used
        if station not in station_colors:
            station_colors[station] = cmap(len(station_colors) % 10)  # Use modulo to cycle colors
        # plot the height vs. absolute_magnitudes
        ax0.plot(obs_data.absolute_magnitudes[np.where(obs_data.stations_lum == station)], \
                 obs_data.height_lum[np.where(obs_data.stations_lum == station)]/1000, 'x--', \
                 color=station_colors[station], label=station)

    ax0.set_xlabel('Absolute Magnitudes')
    # flip the x-axis
    ax0.invert_xaxis()
    ax0.legend()
    # ax0.tick_params(axis='x', rotation=45)
    ax0.set_ylabel('Height (km)')
    ax0.grid(True, linestyle='--', color='lightgray')

    

    ax1.fill_betweenx(obs_data.height_lum/1000, -obs_data.noise_mag, obs_data.noise_mag, color='darkgray', alpha=0.2)
    ax1.fill_betweenx(obs_data.height_lum/1000, -obs_data.noise_mag * 2, obs_data.noise_mag * 2, color='lightgray', alpha=0.2)
    ax1.plot([0, 0], [obs_data.height_lum[0]/1000, obs_data.height_lum[-1]/1000],color='lightgray')
    ax1.set_xlabel('Res.Mag')
    # flip the x-axis
    ax1.invert_xaxis()
    # ax1.tick_params(axis='x', rotation=45)
    ax1.tick_params(labelleft=False)  # Hide y-axis tick labels
    ax1.grid(True, linestyle='--', color='lightgray')

    ### LUMINOSITY PLOT ###

    # Create a sub GridSpec for Plot 0 and Plot 1 with width ratios
    gs02 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_main[1, 0:2], wspace=0, width_ratios=[3, 1])

    # Plot 0 and 1: Side by side, sharing the y-axis
    ax4 = fig.add_subplot(gs02[0])
    ax5 = fig.add_subplot(gs02[1], sharey=ax4)

    # for each station in obs_data_plot
    for station in np.unique(obs_data.stations_lum):
        # Assign a unique color if not already used
        if station not in station_colors:
            station_colors[station] = cmap(len(station_colors) % 10)  # Use modulo to cycle colors
        # plot the height vs. absolute_magnitudes
        ax4.plot(obs_data.luminosity[np.where(obs_data.stations_lum == station)], \
                 obs_data.height_lum[np.where(obs_data.stations_lum == station)]/1000, 'x--', \
                 color=station_colors[station], label=station)
    ax4.set_xlabel('Luminosity [J/s]')
    # ax4.tick_params(axis='x', rotation=45)
    ax4.set_ylabel('Height (km)')
    ax4.grid(True, linestyle='--', color='lightgray')

    ax5.fill_betweenx(obs_data.height_lum/1000, -obs_data.noise_lum, obs_data.noise_lum, color='darkgray', alpha=0.2)
    ax5.fill_betweenx(obs_data.height_lum/1000, -obs_data.noise_lum * 2, obs_data.noise_lum * 2, color='lightgray', alpha=0.2)
    ax5.plot([0, 0], [obs_data.height_lum[0]/1000, obs_data.height_lum[-1]/1000],color='lightgray')
    ax5.set_xlabel('Res.Lum [J/s]')
    # ax5.tick_params(axis='x', rotation=45)
    ax5.tick_params(labelleft=False)  # Hide y-axis tick labels
    ax5.grid(True, linestyle='--', color='lightgray')

    ### VELOCITY PLOT ###

    # Plot 2 and 6: Vertically stacked, sharing the x-axis (Time) with height ratios
    gs_col2 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_main[:, 2], hspace=0, height_ratios=[3, 1])
    ax2 = fig.add_subplot(gs_col2[0, 0])
    ax6 = fig.add_subplot(gs_col2[1, 0], sharex=ax2)

    # for each station in obs_data_plot
    for station in np.unique(obs_data.stations_lag):
        # Assign a unique color if not already used
        if station not in station_colors:
            station_colors[station] = cmap(len(station_colors) % 10)  # Use modulo to cycle colors
        # plot the height vs. absolute_magnitudes
        ax2.plot(obs_data.time_lag[np.where(obs_data.stations_lag == station)], \
                 obs_data.velocities[np.where(obs_data.stations_lag == station)]/1000, '.', \
                 color=station_colors[station], label=station)
    ax2.set_ylabel('Velocity [km/s]')
    ax2.legend()
    ax2.tick_params(labelbottom=False)  # Hide x-axis tick labels
    ax2.grid(True, linestyle='--', color='lightgray')

    # Plot 6: Res.Vel vs. Time
    ax6.fill_between(obs_data.time_lag, -obs_data.noise_vel/1000, obs_data.noise_vel/1000, color='darkgray', alpha=0.2)
    ax6.fill_between(obs_data.time_lag, -obs_data.noise_vel * 2/1000, obs_data.noise_vel * 2/1000, color='lightgray', alpha=0.2)
    ax6.plot([obs_data.time_lag[0], obs_data.time_lag[-1]], [0, 0], color='lightgray')
    ax6.set_xlabel('Time [s]')
    ax6.set_ylabel('Res.Vel [km/s]')
    ax6.grid(True, linestyle='--', color='lightgray')

    ### LAG PLOT ###

    # Plot 3 and 7: Vertically stacked, sharing the x-axis (Time) with height ratios
    gs_col3 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_main[:, 3], hspace=0, height_ratios=[3, 1])
    ax3 = fig.add_subplot(gs_col3[0, 0])
    ax7 = fig.add_subplot(gs_col3[1, 0], sharex=ax3)

    # for each station in obs_data_plot
    for station in np.unique(obs_data.stations_lag):
        # Assign a unique color if not already used
        if station not in station_colors:
            station_colors[station] = cmap(len(station_colors) % 10)  # Use modulo to cycle colors
        # plot the height vs. absolute_magnitudes
        ax3.plot(obs_data.time_lag[np.where(obs_data.stations_lag == station)], \
                 obs_data.lag[np.where(obs_data.stations_lag == station)], 'x:', \
                 color=station_colors[station], label=station)
    ax3.set_ylabel('Lag [m]')
    ax3.tick_params(labelbottom=False)  # Hide x-axis tick labels
    ax3.grid(True, linestyle='--', color='lightgray')

    # Plot 7: Res.Vel vs. Time
    ax7.fill_between(obs_data.time_lag, -obs_data.noise_lag, obs_data.noise_lag, color='darkgray', alpha=0.2)
    ax7.fill_between(obs_data.time_lag, -obs_data.noise_lag * 2, obs_data.noise_lag * 2, color='lightgray', alpha=0.2)
    ax7.plot([obs_data.time_lag[0], obs_data.time_lag[-1]], [0, 0], color='lightgray')
    ax7.set_xlabel('Time [s]')
    ax7.set_ylabel('Res.Lag [m]')
    ax7.grid(True, linestyle='--', color='lightgray')

    # Adjust the overall layout to prevent overlap
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    # make the suptitle
    # fig.suptitle(file_name)

    # Save the plot
    print('file saved: '+out_folder +os.sep+ file_name+'.png')
    fig.savefig(output_folder +os.sep+ file_name +'.png', dpi=300)

    # Display the plot
    plt.close(fig)

    # Check if sim_data was provided
    if sim_data is None:
        print("Warning: No simulation data provided. Proceeding with only observation data.")
        return
    
    # ax0.fill_betweenx(height_km_err, abs_mag_sim_err - mag_noise, abs_mag_sim_err + mag_noise, color='darkgray', alpha=0.2)
    # ax0.fill_betweenx(height_km_err, abs_mag_sim_err - mag_noise * real_original['z_score'], abs_mag_sim_err + mag_noise * real_original['z_score'], color='lightgray', alpha=0.2)

    # ax2.fill_between(residual_time_pos, vel_kms_err - vel_noise, vel_kms_err + vel_noise, color='darkgray', alpha=0.2)
    # ax2.fill_between(residual_time_pos, vel_kms_err - vel_noise * real_original['z_score'], vel_kms_err + vel_noise * real_original['z_score'], color='lightgray', alpha=0.2)


###############################################################################
# Function: read_prior_to_bounds
###############################################################################
def read_prior_to_bounds(object_meteor,file_path=""):
    # Default bounds
    default_bounds = {
        "v_init": (500, np.nan),
        "zenith_angle": (0.01, np.nan),
        "m_init": (np.nan, np.nan),
        "rho": (10, 4000),  # log transformation applied later
        "sigma": (0.008 / 1e6, 0.03 / 1e6),
        "erosion_height_start": (np.nan, np.nan),
        "erosion_coeff": (1 / 1e12, 2 / 1e6),  # log transformation applied later
        "erosion_mass_index": (1, 3),
        "erosion_mass_min": (5e-12, 1e-9),  # log transformation applied later
        "erosion_mass_max": (1e-10, 1e-7),  # log transformation applied later
    }

    default_flags = {
        "v_init": ["norm"],
        "zenith_angle": ["norm"],
        "m_init": [],
        "rho": ["log"],
        "sigma": [],
        "erosion_height_start": [],
        "erosion_coeff": ["log"],
        "erosion_mass_index": [],
        "erosion_mass_min": ["log"],
        "erosion_mass_max": ["log"]
        }

    # Default values if no file path is provided
    if file_path == "":
        # delete from the default_bounds the zenith_angle
        default_bounds.pop("zenith_angle")
        bounds = [default_bounds[key] for key in default_bounds]
        flags_dict = {key: default_flags.get(key, []) for key in default_bounds}
        # for the one that have log transformation, apply it
        for i, key in enumerate(default_bounds):
            if "log" in flags_dict[key]:
                bounds[i] = np.log10(bounds[i][0]), np.log10(bounds[i][1])
        fixed_values = {
            "zenith_angle": np.nan,
        }

    else:
        bounds = []
        flags_dict = {}
        fixed_values = {}

        def safe_eval(value):
            try:    
                return eval(value, {"__builtins__": {"np": np}}, {})
            except Exception:
                return value  # Return as string if evaluation fails
        
        # Read .prior file, ignoring comment lines
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if line.startswith('#') or not line:
                    continue
                parts = line.split('#')[0].strip().split(',')  # Remove inline comments
                name = parts[0].strip()
                # Handle fixed values
                if "fix" in parts:
                    val = parts[1].strip() if len(parts) > 1 else "nan"
                    fixed_values[name] = safe_eval(val) if val.lower() != "nan" else np.nan
                    print(fixed_values[name])
                    if np.isnan(fixed_values[name]) and name in object_meteor.__dict__:
                        fixed_values[name] = object_meteor.__dict__[name]
                    if np.isnan(fixed_values[name]) and name == "erosion_height_start":
                        fixed_values[name] = object_meteor.height_lum[0]
                    if np.isnan(fixed_values[name]):
                        fixed_values[name] = np.mean(default_bounds[name])
                    continue
                min_val = parts[1].strip() if len(parts) > 1 else "nan"
                max_val = parts[2].strip() if len(parts) > 2 else "nan"
                flags = [flag.strip() for flag in parts[3:]] if len(parts) > 3 else []
                
                # Handle NaN values and default replacement
                min_val = safe_eval(min_val) if min_val.lower() != "nan" else default_bounds.get(name, (np.nan, np.nan))[0]
                max_val = safe_eval(max_val) if max_val.lower() != "nan" else default_bounds.get(name, (np.nan, np.nan))[1]

                # check if name=='v_init' or zenith_angle or m_init or erosion_height_start values are np.nan and replace them with the object_meteor values
                if np.isnan(min_val) and name in object_meteor.__dict__ and "norm" in flags:
                    if "norm" in default_flags[name]:
                        min_val = default_bounds.get(name, (np.nan, np.nan))[0]
                    else:
                        min_val = object_meteor.__dict__[name]/10/2
                if np.isnan(min_val) and name in object_meteor.__dict__:
                    # if norm in default_flags[name] then divide by 10
                    if "norm" in default_flags[name]:
                        min_val = object_meteor.__dict__[name] + default_bounds.get(name, (np.nan, np.nan))[0]
                    else:
                        min_val = object_meteor.__dict__[name] - object_meteor.__dict__[name]/10/2
                elif np.isnan(min_val) and name == "erosion_height_start":
                    min_val = object_meteor.height_lum[0] - 1000
                    if "norm" in flags:
                        min_val = min_val/10

                if np.isnan(max_val) and name in object_meteor.__dict__ and "norm" in flags:
                    max_val = object_meteor.__dict__[name]
                if np.isnan(max_val) and name in object_meteor.__dict__:
                    # if norm in default_flags[name] then divide by 10
                    if "norm" in default_flags[name]:
                        max_val = object_meteor.__dict__[name] + default_bounds.get(name, (np.nan, np.nan))[0]
                    else:
                        max_val = object_meteor.__dict__[name] + object_meteor.__dict__[name]/10/2
                elif np.isnan(max_val) and name == "erosion_height_start":
                    max_val = object_meteor.height_lum[0] + 9000
                    if "norm" in flags:
                        max_val = object_meteor.height_lum[0]
                
                # check if min_val > max_val, then swap them cannot have negative values
                if min_val > max_val:
                    print(f"Min/sigma > MAX/mean : Swapping {min_val} and {max_val} for {name}")
                    min_val, max_val = max_val, min_val

                # Store flags
                flags_dict[name] = flags
                            
                # Apply log10 transformation if needed
                if "log" in flags:
                    # check if any values is 0 and if it is, replace it with the default value
                    if min_val == 0:
                        min_val = 1/1e12
                    # Apply log10 transformation
                    min_val, max_val = np.log10(min_val), np.log10(max_val)
                
                bounds.append((min_val, max_val))
    
    # check if the bounds the len(bounds) + len(fixed_values) =>10
    if len(bounds) + len(fixed_values) < 10:
        raise ValueError("The number of bounds and fixed values should 10 or above")

    return bounds, flags_dict, fixed_values


class observation_data:
    def __init__(self, obs_file_path,use_CAMO_data):
        self.file = obs_file_path
        # check if the file is a json file
        if obs_file_path.endswith('.pickle'):
            self.load_pickle_data(use_CAMO_data)
        # elif obs_file_path.endswith('.json'):
        #     self.load_json_data()
        else:
            # file type not supported
            raise ValueError("File type not supported, only .json and .pickle files are supported")

    def load_pickle_data(self,use_CAMO_data):
        print('Loading pickle file:',self.file)
        # load the pickle file
        traj=loadPickle(*os.path.split(self.file))
        # get the trajectory
        # v_avg = traj.v_avg
        self.v_init=traj.orbit.v_init+100
        self.stations = []
        obs_data_CAMO = []
        obs_data_EMCCD = []
        peak_abs_mag_CAMO = None
        flag_there_is_CAMO_data = False
        flag_there_is_EMCCD_data = False
        for obs in traj.observations:
            if (obs.station_id == "01T" or obs.station_id == "02T" or obs.station_id == "01T'-Mirfit" or obs.station_id == "02T'-Mirfit"):
                if flag_there_is_CAMO_data is None:
                    peak_abs_mag_CAMO = np.min(obs_data_CAMO.absolute_magnitudes)
                elif peak_abs_mag_CAMO > np.min(obs_data_CAMO.absolute_magnitudes):
                    peak_abs_mag_CAMO = np.min(obs_data_CAMO.absolute_magnitudes)

            # check if among obs.station_id there is one of the following 01T or 02T
            if (obs.station_id == "01T" or obs.station_id == "02T" or obs.station_id == "01T'-Mirfit" or obs.station_id == "02T'-Mirfit") and use_CAMO_data==True:
                P_0m = 840
                obs_dict_CAMO = {
                    # make an array that is long as len(obs.model_ht) and has only obs.station_id
                    'flag_station': np.array([obs.station_id]*len(obs.model_ht)),
                    'height': np.array(obs.model_ht), # m
                    'absolute_magnitudes': np.array(obs.absolute_magnitudes),
                    'luminosity': np.array(P_0m*(10 ** (obs.absolute_magnitudes/(-2.5)))), # const.P_0m)
                    'time': np.array(obs.time_data), # s
                    'ignore_list': np.array(obs.ignore_list),
                    'velocities': np.array(obs.velocities), # m/s
                    'lag': np.array(obs.lag), # m
                    'length': np.array(obs.state_vect_dist), # m
                    'time_lag': np.array(obs.time_data), # s
                    'height_lag': np.array(obs.model_ht) # m
                    }
                obs_dict_CAMO['velocities'][0] = obs.v_init
                self.stations.append(obs.station_id)
                obs_data_CAMO.append(obs_dict_CAMO)
                flag_there_is_CAMO_data = True
            elif obs.station_id == "01G" or obs.station_id == "02G" or obs.station_id == "01F" or obs.station_id == "02F" or obs.station_id == "1G" or obs.station_id == "2G" or obs.station_id == "1F" or obs.station_id == "2F":
                P_0m = 935
                obs_dict_EMCCD = {
                    # make an array that is long as len(obs.model_ht) and has only obs.station_id
                    'flag_station': np.array([obs.station_id]*len(obs.model_ht)),
                    'height': np.array(obs.model_ht), # m
                    'absolute_magnitudes': np.array(obs.absolute_magnitudes),
                    'luminosity': np.array(P_0m*(10 ** (obs.absolute_magnitudes/(-2.5)))), # const.P_0m)
                    'time': np.array(obs.time_data), # s
                    'ignore_list': np.array(obs.ignore_list),
                    'velocities': np.array(obs.velocities), # m/s
                    'lag': np.array(obs.lag), # m
                    'length': np.array(obs.state_vect_dist), # m
                    'time_lag': np.array(obs.time_data), # s
                    'height_lag': np.array(obs.model_ht) # m
                    }
                obs_dict_EMCCD['velocities'][0] = obs.v_init
                self.stations.append(obs.station_id)
                obs_data_EMCCD.append(obs_dict_EMCCD)
                flag_there_is_EMCCD_data = True
            else:
                print(obs.station_id,'Station data not considered')
                continue
        
        print('Stations:',self.stations)
        
        # Combine all observations
        combined_obs_CAMO = {}
        combined_obs_EMCCD = {}

        if flag_there_is_EMCCD_data:

            # Combine obs1 and obs2
            for key in obs_dict_EMCCD.keys():
                combined_obs_EMCCD[key] = np.concatenate([obs[key] for obs in obs_data_EMCCD])

            sorted_indices = np.argsort(combined_obs_EMCCD['time'])
            for key in obs_dict_EMCCD.keys():
                combined_obs_EMCCD[key] = combined_obs_EMCCD[key][sorted_indices]

            # check if any value is below 8 absolute_magnitudes and print find values below 8 absolute_magnitudes
            if np.any(combined_obs_EMCCD['absolute_magnitudes'] > 8):
                print('Found values below 8 absolute magnitudes:', combined_obs_EMCCD['absolute_magnitudes'][combined_obs_EMCCD['absolute_magnitudes'] > 8])
                # delete any values above 8 absolute_magnitudes and delete the corresponding values in the other arrays
                combined_obs_EMCCD = {key: combined_obs_EMCCD[key][combined_obs_EMCCD['absolute_magnitudes'] < 8] for key in combined_obs_EMCCD.keys()}

            self.P_0m = 935
            self.height_lum = combined_obs_EMCCD['height']
            self.absolute_magnitudes = combined_obs_EMCCD['absolute_magnitudes']
            self.luminosity = combined_obs_EMCCD['luminosity']
            self.time_lum = combined_obs_EMCCD['time']
            self.stations_lum = combined_obs_EMCCD['flag_station']

        if flag_there_is_CAMO_data and use_CAMO_data:

            # Combine obs1 and obs2
            for key in obs_dict_CAMO.keys():
                combined_obs_CAMO[key] = np.concatenate([obs[key] for obs in obs_data_CAMO])

            # sort the indices
            sorted_indices = np.argsort(combined_obs_CAMO['time'])
            for key in obs_dict_CAMO.keys():
                combined_obs_CAMO[key] = combined_obs_CAMO[key][sorted_indices]

            # if there is, use the CAMO data for position and velocity and the ignore_list == 0
            self.velocities = combined_obs_CAMO['velocities'][combined_obs_CAMO['ignore_list'] == 0]
            self.lag = combined_obs_CAMO['lag'][combined_obs_CAMO['ignore_list'] == 0]
            self.length = combined_obs_CAMO['length'][combined_obs_CAMO['ignore_list'] == 0]
            self.height_lag = combined_obs_CAMO['height_lag'][combined_obs_CAMO['ignore_list'] == 0]
            self.time_lag = combined_obs_CAMO['time_lag'][combined_obs_CAMO['ignore_list'] == 0]
            self.stations_lag = combined_obs_CAMO['flag_station'][combined_obs_CAMO['ignore_list'] == 0]
            self.fps = 80
            self.noise_lag = 5
            self.noise_vel = self.noise_lag*np.sqrt(2)/(1.0/self.fps)
            
            if flag_there_is_EMCCD_data==False:

                # check if any value is below 8 absolute_magnitudes and print find values below 8 absolute_magnitudes
                if np.any(combined_obs_CAMO['absolute_magnitudes'] > 8):
                    print('Found values below 8 absolute magnitudes:', combined_obs_CAMO['absolute_magnitudes'][combined_obs_CAMO['absolute_magnitudes'] > 8])
                
                # delete any values above 8 absolute_magnitudes and delete the corresponding values in the other arrays
                combined_obs_CAMO = {key: combined_obs_CAMO[key][combined_obs_CAMO['absolute_magnitudes'] < 8] for key in combined_obs_CAMO.keys()}

                # check if any value is below 8 absolute_magnitudes and print find values below 8 absolute_magnitudes
                if np.any(combined_obs_CAMO['absolute_magnitudes'] > 8):
                    print('Found values below 8 absolute magnitudes:', combined_obs_CAMO['absolute_magnitudes'][combined_obs_CAMO['absolute_magnitudes'] > 8])
                    # delete any values above 8 absolute_magnitudes and delete the corresponding values in the other arrays
                    combined_obs_CAMO = {key: combined_obs_CAMO[key][combined_obs_CAMO['absolute_magnitudes'] < 8] for key in combined_obs_CAMO.keys()}

                # if there is not, use the EMCCD data for position and velocity
                self.P_0m = 840
                self.height_lum = combined_obs_CAMO['height']
                self.absolute_magnitudes = combined_obs_CAMO['absolute_magnitudes']
                self.luminosity = combined_obs_CAMO['luminosity']
                self.time_lum = combined_obs_CAMO['time']
                self.stations_lum = combined_obs_CAMO['flag_station']
        else:
            # if there is not, use the EMCCD data for position and velocity
            self.velocities = combined_obs_EMCCD['velocities'][combined_obs_EMCCD['ignore_list'] == 0]
            self.lag = combined_obs_EMCCD['lag'][combined_obs_EMCCD['ignore_list'] == 0]
            self.length = combined_obs_EMCCD['length'][combined_obs_EMCCD['ignore_list'] == 0]
            self.height_lag = combined_obs_EMCCD['height_lag'][combined_obs_EMCCD['ignore_list'] == 0]
            self.time_lag = combined_obs_EMCCD['time_lag'][combined_obs_EMCCD['ignore_list'] == 0]
            self.stations_lag = combined_obs_EMCCD['flag_station'][combined_obs_EMCCD['ignore_list'] == 0]
            self.fps = 32
            self.noise_lag = 40
            self.noise_vel = self.noise_lag*np.sqrt(2)/(1.0/self.fps)

        if flag_there_is_EMCCD_data==False and flag_there_is_CAMO_data==False:
            raise ValueError("No data found for EMCCD and CAMO in the pickle file")
        if flag_there_is_EMCCD_data==False and flag_there_is_CAMO_data==True and use_CAMO_data==False:
            raise ValueError("No data found for EMCCD in the pickle file but CAMO data is available (set use_CAMO_data=True)")
        
        # for lag measurements start from 0 for length and time
        self.length = self.length-self.length[0]
        self.time_lag = self.time_lag-self.time_lag[0]

        self.noise_lum = 2.5
        self.noise_mag = 0.1

        # Ensure there are exactly two unique stations before proceeding
        unique_stations = np.unique(self.stations_lum)
        if len(unique_stations) == 2:
            self._photometric_adjustment(unique_stations,peak_abs_mag_CAMO)

        dens_fit_ht_beg = 180000
        dens_fit_ht_end = traj.rend_ele - 5000
        if dens_fit_ht_end < 14000:
            dens_fit_ht_end = 14000

        lat_mean = np.mean([traj.rbeg_lat, traj.rend_lat])
        lon_mean = meanAngle([traj.rbeg_lon, traj.rend_lon])
        jd_dat=traj.jdt_ref

        # Fit the polynomail describing the density
        self.dens_co = fitAtmPoly(lat_mean, lon_mean, dens_fit_ht_end, dens_fit_ht_beg, jd_dat)

        const=Constants()
        self.zenith_angle=zenithAngleAtSimulationBegin(const.h_init, traj.rbeg_ele, traj.orbit.zc, const.r_earth)

        time_mag_arr = []
        avg_t_diff_max = 0
        # Extract time vs. magnitudes from the trajectory pickle file
        for obs in traj.observations:

            # If there are not magnitudes for this site, skip it
            if obs.absolute_magnitudes is None:
                continue

            # Compute average time difference
            avg_t_diff_max = max(avg_t_diff_max, np.median(obs.time_data[1:] - obs.time_data[:-1]))

            for t, mag in zip(obs.time_data[obs.ignore_list == 0], \
                obs.absolute_magnitudes[obs.ignore_list == 0]):

                if (mag is not None) and (not np.isnan(mag)) and (not np.isinf(mag)):
                    time_mag_arr.append([t, mag])


        print("NOTE: The mass was computing using a constant luminous efficiency 0.7!")

        # Sort array by time
        time_mag_arr = np.array(sorted(time_mag_arr, key=lambda x: x[0]))

        time_arr, mag_arr = time_mag_arr.T

        # Average out the magnitudes
        time_arr, mag_arr = mergeClosePoints(time_arr, mag_arr, avg_t_diff_max, method='avg')

        # # Calculate the radiated energy
        # radiated_energy = calcRadiatedEnergy(np.array(time_arr), np.array(mag_arr), P_0m=self.P_0m)

        # Compute the photometric mass
        photom_mass = calcMass(np.array(time_arr), np.array(mag_arr), traj.orbit.v_avg, tau=0.7/100, P_0m=self.P_0m)

        self.m_init = photom_mass


    def _photometric_adjustment(self,unique_stations,peak_abs_mag_CAMO):
        
        # Find indices of each station in self.stations_lum
        station_0_indices = np.where(self.stations_lum == unique_stations[0])[0]
        station_1_indices = np.where(self.stations_lum == unique_stations[1])[0]

        # Extract corresponding time arrays
        time_0 = self.time_lum[station_0_indices]
        time_1 = self.time_lum[station_1_indices]

        # Define a common time grid (union of both time arrays, sorted)
        common_time = np.linspace(max(time_0.min(), time_1.min()), min(time_0.max(), time_1.max()), num=100)

        # Interpolate absolute magnitudes onto the common time grid
        absolute_magnitudes_0 = np.interp(common_time, time_0, self.absolute_magnitudes[station_0_indices])
        absolute_magnitudes_1 = np.interp(common_time, time_1, self.absolute_magnitudes[station_1_indices])

        # Compute the mean
        avg_mag_0 = np.mean(absolute_magnitudes_0)
        avg_mag_1 = np.mean(absolute_magnitudes_1)
        # take the 4 smallest values from the absolute_magnitudes_0 and absolute_magnitudes_1
        small4_0 = np.mean(np.sort(absolute_magnitudes_0)[:4])
        small4_1 = np.mean(np.sort(absolute_magnitudes_1)[:4])
        # Compute the average difference between the two stations
        avg_diff = np.sqrt(np.mean((absolute_magnitudes_0 - absolute_magnitudes_1) ** 2))

        if peak_abs_mag_CAMO is not None:
            diff_0 = np.abs(peak_abs_mag_CAMO - small4_0)
            diff_1 = np.abs(peak_abs_mag_CAMO - small4_1)
            if diff_0 < diff_1 and avg_mag_0 < avg_mag_1:
                self.absolute_magnitudes[station_1_indices] -= avg_diff
                # Update luminosity for station_1
                self.luminosity[station_1_indices] = self.P_0m * (10 ** (self.absolute_magnitudes[station_1_indices] / (-2.5)))
            elif diff_0 < diff_1 and avg_mag_0 > avg_mag_1:
                self.absolute_magnitudes[station_1_indices] += avg_diff
                # Update luminosity for station_1
                self.luminosity[station_1_indices] = self.P_0m * (10 ** (self.absolute_magnitudes[station_1_indices] / (-2.5)))
            elif diff_0 > diff_1 and avg_mag_0 > avg_mag_1:
                self.absolute_magnitudes[station_0_indices] -= avg_diff
                # Update luminosity for station_0
                self.luminosity[station_0_indices] = self.P_0m * (10 ** (self.absolute_magnitudes[station_0_indices] / (-2.5)))
            elif diff_0 > diff_1 and avg_mag_0 < avg_mag_1:
                self.absolute_magnitudes[station_0_indices] += avg_diff
                # Update luminosity for station_0
                self.luminosity[station_0_indices] = self.P_0m * (10 ** (self.absolute_magnitudes[station_0_indices] / (-2.5)))
        else: 
            # Apply correction to station_0's absolute magnitudes
            if avg_mag_0 > avg_mag_1:
                self.absolute_magnitudes[station_0_indices] -= avg_diff
                # Update luminosity for station_0
                self.luminosity[station_0_indices] = self.P_0m * (10 ** (self.absolute_magnitudes[station_0_indices] / (-2.5)))
            else:
                self.absolute_magnitudes[station_1_indices] -= avg_diff
                # Update luminosity for station_1
                self.luminosity[station_1_indices] = self.P_0m * (10 ** (self.absolute_magnitudes[station_1_indices] / (-2.5)))

    # def load_json_data(self):

    #     '''
    #     dict_keys(['const', 'frag_main', 'time_arr', 'luminosity_arr', 'luminosity_main_arr', 'luminosity_eroded_arr', 
    #     'electron_density_total_arr', 'tau_total_arr', 'tau_main_arr', 'tau_eroded_arr', 'brightest_height_arr', 
    #     'brightest_length_arr', 'brightest_vel_arr', 'leading_frag_height_arr', 'leading_frag_length_arr', 
    #     'leading_frag_vel_arr', 'leading_frag_dyn_press_arr', 'mass_total_active_arr', 'main_mass_arr', 
    #     'main_height_arr', 'main_length_arr', 'main_vel_arr', 'main_dyn_press_arr', 'abs_magnitude', 
    #     'abs_magnitude_main', 'abs_magnitude_eroded', 'wake_results', 'wake_max_lum'])

    #     in const

    #     dict_keys(['dt', 'total_time', 'n_active', 'm_kill', 'v_kill', 'h_kill', 'len_kill', 'h_init', 'P_0m', 
    #     'dens_co', 'r_earth', 'total_fragments', 'wake_psf', 'wake_extension', 'rho', 'm_init', 'v_init', 
    #     'shape_factor', 'sigma', 'zenith_angle', 'gamma', 'rho_grain', 'lum_eff_type', 'lum_eff', 'mu', 
    #     'erosion_on', 'erosion_bins_per_10mass', 'erosion_height_start', 'erosion_coeff', 'erosion_height_change', 
    #     'erosion_coeff_change', 'erosion_rho_change', 'erosion_sigma_change', 'erosion_mass_index', 'erosion_mass_min', 
    #     'erosion_mass_max', 'disruption_on', 'compressive_strength', 'disruption_height', 'disruption_erosion_coeff', 
    #     'disruption_mass_index', 'disruption_mass_min_ratio', 'disruption_mass_max_ratio', 'disruption_mass_grain_ratio', 
    #     'fragmentation_on', 'fragmentation_show_individual_lcs', 'fragmentation_entries', 'fragmentation_file_name', 
    #     'electron_density_meas_ht', 'electron_density_meas_q', 'erosion_beg_vel', 'erosion_beg_mass', 'erosion_beg_dyn_press', 
    #     'mass_at_erosion_change', 'energy_per_cs_before_erosion', 'energy_per_mass_before_erosion', 'main_mass_exhaustion_ht', 'main_bottom_ht'])
    #     '''

    #     with open(self.file, 'r') as f:
    #         self.obs_data = json.load(f)
        
    #     # Sample the time according to the FPS from one camera
    #     time_sampled_cam1 = np.arange(np.min(time_visible), np.max(time_visible), 1.0/params.fps)

    #     # Simulate sampling of the data from a second camera, with a random phase shift
    #     time_sampled_cam2 = time_sampled_cam1 + np.random.uniform(-1.0/params.fps, 1.0/params.fps)

    #     # The second camera will only capture 50 - 100% of the data, simulate this
    #     cam2_portion = np.random.uniform(0.5, 1.0)
    #     cam2_start = np.random.uniform(0, 1.0 - cam2_portion)
    #     cam2_start_index = int(cam2_start*len(time_sampled_cam2))
    #     cam2_end_index = int((cam2_start + cam2_portion)*len(time_sampled_cam2))

    #     # Cut the cam2 time to the portion of the data it will capture
    #     time_sampled_cam2 = time_sampled_cam2[cam2_start_index:cam2_end_index]

    #     # Cut the time array to the length of the visible data
    #     time_sampled_cam2 = time_sampled_cam2[(time_sampled_cam2 >= np.min(time_visible)) 
    #                                         & (time_sampled_cam2 <= np.max(time_visible))]

    #     # Combine the two camera time arrays
    #     time_sampled = np.sort(np.concatenate([time_sampled_cam1, time_sampled_cam2]))


###############################################################################
# Class: find_dynestyfile_and_priors
###############################################################################
class find_dynestyfile_and_priors:
    """
    1. If `input_dir_or_file` is a file:
       - Strip extension => add ".dynesty"
       - If `prior_file` is a valid file, use it (read_prior_to_bounds).
         Otherwise, look for a .prior in the same folder.
         If still none, use defaults.

    2. If `input_dir_or_file` is a directory:
       - For each .pickle found:
         - Decide .dynesty name (check existing, apply resume logic).
         - If `prior_file` is valid, use it.
           Otherwise, look for a .prior in that folder.
           If none, default.

    3. `resume=True`: Reuse existing .dynesty if it exists.
       `resume=False`: Create new .dynesty name (append _n1, _n2, etc.) if found.

    4. If `output_dir != ""`, create a subfolder named after the .dynesty base name.
       Otherwise, store results in the same folder as the .dynesty.

    5. Stored results:
       - `self.input_folder_file`: List of (dynesty_file, bounds, flags_dict, fixed_values)
       - `self.priors`: List of the .prior paths used (or "") for each entry
       - `self.output_folders`: Where results are stored for each entry
    """

    def __init__(self, input_dir_or_file, prior_file, resume, output_dir="", use_CAMO_data=False):
        self.input_dir_or_file = input_dir_or_file
        self.prior_file = prior_file
        self.resume = resume
        self.output_dir = output_dir
        self.use_CAMO_data = use_CAMO_data

        # Prepare placeholders
        self.base_names = []        # [base_name, ...] 
        self.input_folder_file = [] # [(dynesty_file, bounds, flags_dict, fixed_values), ...]
        self.priors = []            # [used_prior_path_or_empty_string, ...]
        self.output_folders = []    # [output_folder_for_this_dynesty, ...]

        # Kick off processing
        self._process_input()

    def _process_input(self):
        """Decide if input is file or directory, build .dynesty, figure out prior, and store results."""
        if os.path.isfile(self.input_dir_or_file):
            # Single file case
            root, _ = os.path.splitext(self.input_dir_or_file)
            dynesty_file = root + ".dynesty"

            self.observation_obj = observation_data(self.input_dir_or_file, self.use_CAMO_data)

            # If user gave a valid .prior path, read it once.
            # We'll reuse this for every .dynesty discovered.
            if os.path.isfile(self.prior_file):
                self.user_prior = read_prior_to_bounds(self.observation_obj,self.prior_file)
                # We already read a user-provided prior
                bounds, flags_dict, fixed_values = self.user_prior
                prior_path = self.prior_file
            else:
                # If no user-prior, check for a local .prior
                folder = os.path.dirname(self.input_dir_or_file)
                prior_path = self._find_prior_in_folder(folder)
                if prior_path:
                    bounds, flags_dict, fixed_values = read_prior_to_bounds(self.observation_obj,prior_path)
                else:
                    # default
                    prior_path = ""
                    bounds, flags_dict, fixed_values = read_prior_to_bounds(self.observation_obj)

            # Decide output folder
            output_folder = self._decide_output_folder(dynesty_file)

            # Store in class variables
            self.base_names.append(self._extract_base_name(self.input_dir_or_file))
            self.input_folder_file.append((dynesty_file, bounds, flags_dict, fixed_values))
            self.priors.append(prior_path)
            self.output_folders.append(output_folder)

        else:
            # Directory case
            for root, dirs, files in os.walk(self.input_dir_or_file):
                pickle_files = [f for f in files if f.endswith('.pickle')]
                if not pickle_files:
                    continue

                for pf in pickle_files:
                    base_name = os.path.splitext(pf)[0]
                    possible_dynesty = os.path.join(root, base_name + ".dynesty")

                    # Check for existing .dynesty in the same folder
                    existing_dynesty_list = [f for f in files if f.endswith(".dynesty")]

                    if existing_dynesty_list:
                        # There is at least one .dynesty in this folder
                        if os.path.exists(possible_dynesty):
                            # Matches the .pickle base
                            if self.resume:
                                dynesty_file = possible_dynesty
                            else:
                                dynesty_file = self._build_new_dynesty_name(possible_dynesty)
                        else:
                            # There's a .dynesty, but not matching the base name
                            first_dynesty_found = os.path.join(root, existing_dynesty_list[0])
                            if self.resume:
                                dynesty_file = first_dynesty_found
                            else:
                                dynesty_file = self._build_new_dynesty_name(first_dynesty_found)
                    else:
                        # No .dynesty => create from .pickle base name
                        dynesty_file = possible_dynesty

                    self.observation_obj = observation_data(pf, self.use_CAMO_data)

                    # If user gave a valid .prior path, read it once.
                    # We'll reuse this for every .dynesty discovered.
                    if os.path.isfile(self.prior_file):
                        bounds, flags_dict, fixed_values = read_prior_to_bounds(self.observation_obj,self.prior_file)
                    else:
                        # Look for local .prior
                        prior_path = self._find_prior_in_folder(root)
                        if prior_path:
                            bounds, flags_dict, fixed_values = read_prior_to_bounds(self.observation_obj,prior_path)
                        else:
                            # default
                            prior_path = ""
                            bounds, flags_dict, fixed_values = read_prior_to_bounds(self.observation_obj)

                    # Decide the output folder
                    output_folder = self._decide_output_folder(dynesty_file)

                    # Store results
                    self.base_names.append(self._extract_base_name(pf))
                    self.input_folder_file.append((dynesty_file, bounds, flags_dict, fixed_values))
                    self.priors.append(prior_path)
                    self.output_folders.append(output_folder)

    def observation_instance(self):
        """Return the PriorHandler instance for further use."""
        return self.observation_obj  # This returns the instance of the other class

    def _extract_base_name(self, file_path):
        """
        Extracts a base name from the file:
        - If the filename starts with "YYYYMMDD_HHMMSS" (e.g., 20230811_082648_trajectory.pickle),
          return only the first 15 characters: "20230811_082648"
        - Otherwise, return the filename without extension.
        """
        filename = os.path.basename(file_path)  # Extract just the file name
        name_without_ext, _ = os.path.splitext(filename)  # Remove extension

        # Define regex pattern for YYYYMMDD_HHMMSS
        pattern = r"^(\d{8}_\d{6})"

        # Try to match the pattern
        match = re.match(pattern, name_without_ext)
        if match:
            return match.group(1)  # Return only the matched timestamp
        else:
            return name_without_ext  # Return the full name if no match

    def _find_prior_in_folder(self, folder):
        """Return the first .prior file found, or None if none."""
        for f in os.listdir(folder):
            if f.endswith(".prior"):
                return os.path.join(folder, f)
        return None

    def _build_new_dynesty_name(self, existing_dynesty_path):
        """
        If resume=False and we want to avoid overwriting an existing .dynesty file,
        append _n1, _n2, etc. until no file collision occurs.
        """
        folder = os.path.dirname(existing_dynesty_path)
        base = os.path.splitext(os.path.basename(existing_dynesty_path))[0]
        ext = ".dynesty"

        counter = 1
        while True:
            new_name = f"{base}_n{counter}{ext}"
            new_path = os.path.join(folder, new_name)
            if not os.path.exists(new_path):
                return new_path
            counter += 1

    def _decide_output_folder(self, dynesty_file_path):
        """
        Decide where to store results:
          - If self.output_dir is empty, use the .dynesty file's folder.
          - Otherwise, create a subdirectory named after the .dynesty base name.
        """
        if not self.output_dir:
            return os.path.dirname(dynesty_file_path)

        # If output_dir is given, create a named subdirectory
        subfolder_name = os.path.splitext(os.path.basename(dynesty_file_path))[0]
        final_output_dir = os.path.join(self.output_dir, subfolder_name+"_dynesty")
        # os.makedirs(final_output_dir, exist_ok=True)
        return final_output_dir


###############################################################################
# EXAMPLE MAIN (as you provided), illustrating usage
###############################################################################
if __name__ == "__main__":

    import argparse
    import sys
    import warnings

    ### COMMAND LINE ARGUMENTS
    arg_parser = argparse.ArgumentParser(description="Run dynesty with optional .prior file.")

    arg_parser.add_argument('--input_dir', metavar='INPUT_PATH', type=str,
        default=r"/home/mvovk/WMPG-repoMAX/Code/DynNestSampl/20240305_031927_trajectory.pickle",
        help="Path to walk and find .pickle file or specific single file .pickle or .json file divided by ',' in between.")

    arg_parser.add_argument('--output_dir', metavar='OUTPUT_DIR', type=str,
        default=r"",
        help="Where to store results. If empty, store next to each .dynesty.")

    arg_parser.add_argument('--prior', metavar='PRIOR', type=str,
        default=r"/home/mvovk/WMPG-repoMAX/Code/DynNestSampl/stony_meteoroid.prior",
        help="Path to a .prior file. If blank, we look in the .dynesty folder or default to built-in bounds.")
    
    arg_parser.add_argument('--use_CAMO_data', metavar='USE_CAMO_DATA', type=bool, default=True,
        help="If True, use only CAMO data for lag if present in pickle file. If False, do not use CAMO data (by default is False).")

    arg_parser.add_argument('--resume', metavar='RESUME', type=bool, default=True,
        help="If True, resume from existing .dynesty if found. If False, create a new version.")

    arg_parser.add_argument('--cores', metavar='CORES', type=int, default=None,
        help="Number of cores to use. Default = all available.")

    # Parse
    cml_args = arg_parser.parse_args()

    # Optional: suppress warnings
    # warnings.filterwarnings('ignore')

    # If no core count given, use all
    if cml_args.cores is None:
        import multiprocessing
        cml_args.cores = multiprocessing.cpu_count()

    # If user specified a non-empty prior but the file doesn't exist, exit
    if cml_args.prior != "" and not os.path.isfile(cml_args.prior):
        print(f"File {cml_args.prior} not found.")
        print("Specify a valid .prior path or leave it empty.")
        sys.exit()

    # Handle comma-separated input paths
    if ',' in cml_args.input_dir:
        cml_args.input_dir = cml_args.input_dir.split(',')
        print('Number of input directories/files:', len(cml_args.input_dir))
    else:
        cml_args.input_dir = [cml_args.input_dir]

    # Process each input path
    for input_dirfile in cml_args.input_dir:
        print(f"Processing {input_dirfile} (this may take a while if subdirectories are large)")
        print("--------------------------------------------------")

        # Use the class to find .dynesty, load prior, and decide output folders
        finder = find_dynestyfile_and_priors(
            input_dir_or_file=input_dirfile,
            prior_file=cml_args.prior,
            resume=cml_args.resume,
            output_dir=cml_args.output_dir,
            use_CAMO_data=cml_args.use_CAMO_data
        )

        # Each discovered or created .dynesty is in input_folder_file
        # with its matching prior info
        for i, (base_name, dynesty_info, prior_path, out_folder) in enumerate(zip(
            finder.base_names,
            finder.input_folder_file,
            finder.priors,
            finder.output_folders
        )):
            dynesty_file, bounds, flags_dict, fixed_values = dynesty_info
            print(f"Entry #{i+1}:", base_name)
            print("  Dynesty file: ", dynesty_file)
            print("  Prior file:   ", prior_path)
            print("  Output folder:", out_folder)
            print("  Bounds:       ", bounds)
            print("  Flags:        ", flags_dict)
            print("  Fixed Values: ", fixed_values)
            print("--------------------------------------------------")
            obs_data = finder.observation_instance()
            # print("  Observation data:")
            # print("    v_init:", obs_data.v_init)
            # print("    zenith_angle:", obs_data.zenith_angle)
            # print("    m_init:", obs_data.m_init)
            # print("    height_lum:", obs_data.height_lum)
            # print("    absolute_magnitudes:", obs_data.absolute_magnitudes)
            # print("    luminosity:", obs_data.luminosity)
            # print("    time_lum:", obs_data.time_lum)
            # print("    velocities:", obs_data.velocities)
            # print("    lag:", obs_data.lag)
            # print("    length:", obs_data.length)
            # print("    time_lag:", obs_data.time_lag)
            # print("    height_lag:", obs_data.height_lag)
            # print("    stations_lag:", obs_data.stations_lag)
            # print("    stations_lum:", obs_data.stations_lum)
            # print("--------------------------------------------------")
            os.makedirs(out_folder, exist_ok=True)
            plot_data_with_residuals_and_real(obs_data, output_folder=out_folder,file_name=base_name)


