"""
Nested sampling with Dynesty for MetSim meteor data, generate plots and tables 
for given trajectory.pickle file or generate new observation from metsim.json solution file
for EMCCD and CAMO cameras.

Author: Maximilian Vovk
Date: 2025-02-25
"""

import numpy as np
import pandas as pd
import pickle
import sys
import json
import os
import io
import copy
import matplotlib.pyplot as plt
import dynesty
from dynesty import plotting as dyplot
import time
from matplotlib.ticker import ScalarFormatter
import scipy
import warnings
import re
import matplotlib.gridspec as gridspec
from scipy.optimize import minimize
from scipy.stats import norm, invgamma
import shutil
import matplotlib.ticker as ticker
import multiprocessing

from wmpl.MetSim.GUI import loadConstants, SimulationResults
from wmpl.MetSim.MetSimErosion import runSimulation, Constants, zenithAngleAtSimulationBegin
from wmpl.Utils.Math import lineFunc, mergeClosePoints, meanAngle
from wmpl.MetSim.ML.GenerateSimulations import generateErosionSim,saveProcessedList,MetParam
from wmpl.Utils.Physics import calcMass, dynamicPressure, calcRadiatedEnergy
from wmpl.Utils.TrajConversions import J2000_JD, date2JD
from wmpl.Utils.AtmosphereDensity import fitAtmPoly
from wmpl.Utils.Pickling import loadPickle

import signal

class TimeoutException(Exception):
    """Custom exception for timeouts."""
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Function execution timed out")

# create a txt file where you save averithing that has been printed
class Logger(object):
    def __init__(self, directory=".", filename="log.txt"):
        self.terminal = sys.stdout
        # Ensure the directory exists
        if not os.path.exists(directory):
            os.makedirs(directory)
        # Combine the directory and filename to create the full path
        filepath = os.path.join(directory, filename)
        self.log = open(filepath, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # This might be necessary as stdout could call flush
        self.terminal.flush()

    def close(self):
        # Close the log file when done
        self.log.close()

###############################################################################
# Function: plotting function
###############################################################################

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
    # chek if np.unique(obs_data.stations_lag) and np.unique(obs_data.stations_lum) are the same
    if not np.array_equal(np.unique(obs_data.stations_lag), np.unique(obs_data.stations_lum)):
        # print a horizonal along the x axis at the height_lag[0] darkgray
        ax0.axhline(y=obs_data.height_lag[0]/1000, color='gray', linestyle='-.', linewidth=1, label=f"{', '.join(np.unique(obs_data.stations_lag))}", zorder=2)

    ax0.set_xlabel('Absolute Magnitudes')
    # flip the x-axis
    ax0.invert_xaxis()
    ax0.legend()
    # ax0.tick_params(axis='x', rotation=45)
    ax0.set_ylabel('Height (km)')
    ax0.grid(True, linestyle='--', color='lightgray')
    # save the x-axis limits
    xlim_abs_mag = ax0.get_xlim()
    # fix the x-axis limits to xlim_abs_mag
    ax0.set_xlim(xlim_abs_mag)
    # save the y-axis limits
    ylim_abs_mag = ax0.get_ylim()
    # fix the y-axis limits to ylim_abs_mag
    ax0.set_ylim(ylim_abs_mag)
    

    ax1.fill_betweenx(obs_data.height_lum/1000, -obs_data.noise_mag, obs_data.noise_mag, color='darkgray', alpha=0.2)
    ax1.fill_betweenx(obs_data.height_lum/1000, -obs_data.noise_mag * 2, obs_data.noise_mag * 2, color='lightgray', alpha=0.2)
    ax1.plot([0, 0], [obs_data.height_lum[0]/1000, obs_data.height_lum[-1]/1000],color='lightgray')
    ax1.set_xlabel('Res.Mag')
    # flip the x-axis
    # ax1.invert_xaxis()
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
    # chek if np.unique(obs_data.stations_lag) and np.unique(obs_data.stations_lum) are the same
    if not np.array_equal(np.unique(obs_data.stations_lag), np.unique(obs_data.stations_lum)):
        # print a horizonal along the x axis at the height_lag[0] darkgray
        ax4.axhline(y=obs_data.height_lag[0]/1000, color='gray', linestyle='-.', linewidth=1, label=f"{', '.join(np.unique(obs_data.stations_lag))}", zorder=2)
    ax4.set_xlabel('Luminosity [J/s]')
    # ax4.tick_params(axis='x', rotation=45)
    ax4.set_ylabel('Height (km)')
    ax4.grid(True, linestyle='--', color='lightgray')
    # save the x-axis limits
    xlim_lum = ax4.get_xlim()
    # fix the x-axis limits to xlim_lum
    ax4.set_xlim(xlim_lum)
    # save the y-axis limits
    ylim_lum = ax4.get_ylim()
    # fix the y-axis limits to ylim_lum
    ax4.set_ylim(ylim_lum)

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
    # save the x-axis limits
    xlim_vel = ax2.get_xlim()
    # fix the x-axis limits to xlim_vel
    ax2.set_xlim(xlim_vel)
    # save the y-axis limits
    ylim_vel = ax2.get_ylim()
    # fix the y-axis limits to ylim_vel
    ax2.set_ylim(ylim_vel)

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
    # save the x-axis limits
    xlim_lag = ax3.get_xlim()
    # fix the x-axis limits to xlim_lag
    ax3.set_xlim(xlim_lag)
    # save the y-axis limits
    ylim_lag = ax3.get_ylim()
    # fix the y-axis limits to ylim_lag
    ax3.set_ylim(ylim_lag)

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

    # check if 'const' in the object obs_data.keys()
    if hasattr(obs_data, 'const'):

        # plot abs_magnitude_arr vs leading_frag_height_arr
        ax0.plot(obs_data.abs_magnitude, obs_data.leading_frag_height_arr/1000, '--', color='black', linewidth=0.5, label='No Noise', zorder=2)
        ax0.legend()

        # inerpoate the abs_magnitude_arr to the leading_frag_height_arr
        no_noise_mag = np.interp(obs_data.height_lum, 
                                        np.flip(obs_data.leading_frag_height_arr), 
                                        np.flip(obs_data.abs_magnitude))
        
        # make the difference between the no_noise_mag and the obs_data.abs_magnitude
        diff_mag = no_noise_mag - obs_data.absolute_magnitudes
        ax1.plot(diff_mag, obs_data.height_lum/1000, '.', markersize=3, color='black', label='No Noise')
        
        # # for ax5 add a noise that changes for the left and right side of the curve base on the -2.5*np.log10((self.luminosity_arr+self.noise_lum)/self.P_0m) and 2.5*np.log10((self.luminosity_arr+self.noise_lum)/self.P_0m)
        # ax1.fill_betweenx(obs_data.leading_frag_height_arr/1000, \
        #                   -2.5*(np.log10((obs_data.luminosity_arr-obs_data.noise_lum)/obs_data.P_0m)-np.log10((obs_data.luminosity_arr)/obs_data.P_0m)), \
        #                   -2.5*(np.log10((obs_data.luminosity_arr+obs_data.noise_lum)/obs_data.P_0m)-np.log10((obs_data.luminosity_arr)/obs_data.P_0m)), \
        #                     color='darkgray', alpha=0.2)
        # ax1.fill_betweenx(obs_data.leading_frag_height_arr/1000, \
        #                   -2.5*(np.log10((obs_data.luminosity_arr-2*obs_data.noise_lum)/obs_data.P_0m)-np.log10((obs_data.luminosity_arr)/obs_data.P_0m)), \
        #                   -2.5*(np.log10((obs_data.luminosity_arr+2*obs_data.noise_lum)/obs_data.P_0m)-np.log10((obs_data.luminosity_arr)/obs_data.P_0m)), \
        #                     color='lightgray', alpha=0.2)

        # plot luminosity_arr vs leading_frag_height_arr
        ax4.plot(obs_data.luminosity_arr, obs_data.leading_frag_height_arr/1000, '--', color='black', linewidth=0.5, label='No Noise', zorder=2)

        # interpolate to make sure they are the same length and discard points after height starts increasing if it does at any point obs_metsim_obj.traj.observations[0].model_ht
        no_noise_lum = np.interp(obs_data.height_lum, 
                                        np.flip(obs_data.leading_frag_height_arr), 
                                        np.flip(obs_data.luminosity_arr))
        
        # make the difference between the no_noise_intensity and the obs_data.luminosity_arr
        diff_lum = no_noise_lum - obs_data.luminosity
        ax5.plot(diff_lum, obs_data.height_lum/1000, '.', markersize=3, color='black', label='No Noise')


        # find the obs_data.leading_frag_height_arr index is close to obs_data.height_lum[0] wihouth nan
        index = np.argmin(np.abs(obs_data.leading_frag_height_arr[~np.isnan(obs_data.leading_frag_height_arr)]-obs_data.height_lag[0]))
        # plot velocity_arr vs leading_frag_time_arr
        ax2.plot(obs_data.time_arr-obs_data.time_arr[index], \
                 obs_data.leading_frag_vel_arr/1000, '--', color='black', linewidth=0.5, label='No Noise', zorder=2)
        ax2.legend()

        # inerpoate the velocity_arr to the leading_frag_time_arr
        no_noise_vel = np.interp(obs_data.height_lag,
                                    np.flip(obs_data.leading_frag_height_arr),
                                    np.flip(obs_data.leading_frag_vel_arr))
        
        # make the difference between the no_noise_vel and the obs_data.velocities
        diff_vel = no_noise_vel - obs_data.velocities
        ax6.plot(obs_data.time_lag, diff_vel/1000, '.', markersize=3, color='black', label='No Noise')

        # plot lag_arr vs leading_frag_time_arr withouth nan values
        lag_no_noise = (obs_data.leading_frag_length_arr-obs_data.leading_frag_length_arr[index])\
              - ((obs_data.velocities[0])*(obs_data.time_arr-obs_data.time_arr[index]))
        lag_no_noise -= lag_no_noise[index]
        # plot lag_arr vs leading_frag_time_arr
        ax3.plot(obs_data.time_arr-obs_data.time_arr[index], \
                 lag_no_noise, '--', color='black', linewidth=0.5, label='No Noise', zorder=2)

        # inerpoate the lag_arr to the leading_frag_time_arr
        no_noise_lag = np.interp(obs_data.height_lag,
                                    np.flip(obs_data.leading_frag_height_arr),
                                    np.flip(lag_no_noise))
        
        # make the difference between the no_noise_lag and the obs_data.lag
        diff_lag = no_noise_lag - obs_data.lag
        ax7.plot(obs_data.time_lag, diff_lag, '.', markersize=3, color='black', label='No Noise')


    # Check if sim_data was provided
    if sim_data is not None:

        # Plot simulated data
        ax0.plot(sim_data.abs_magnitude, sim_data.leading_frag_height_arr/1000, color='black', label='Best guess')
        ax0.legend()
        
        # inerpoate the abs_magnitude_arr to the leading_frag_height_arr
        sim_mag = np.interp(obs_data.height_lum, 
                                        np.flip(sim_data.leading_frag_height_arr), 
                                        np.flip(sim_data.abs_magnitude))
        
        # make the difference between the no_noise_mag and the obs_data.abs_magnitude
        sim_diff_mag = sim_mag - obs_data.absolute_magnitudes
        # for each station in obs_data_plot
        for station in np.unique(obs_data.stations_lum):
            # plot the height vs. absolute_magnitudes
            ax1.plot(sim_diff_mag[np.where(obs_data.stations_lag == station)], \
                    obs_data.height_lum[np.where(obs_data.stations_lag == station)]/1000, '.', \
                    color=station_colors[station], label=station)

        ax4.plot(sim_data.luminosity_arr, sim_data.leading_frag_height_arr/1000, color='black', label='Best guess') 

        # interpolate to make sure they are the same length and discard points after height starts increasing if it does at any point obs_metsim_obj.traj.observations[0].model_ht
        sim_lum = np.interp(obs_data.height_lum, 
                                        np.flip(sim_data.leading_frag_height_arr), 
                                        np.flip(sim_data.luminosity_arr))
        
        # make the difference between the no_noise_intensity and the obs_data.luminosity_arr
        sim_diff_lum = sim_lum - obs_data.luminosity
        # for each station in obs_data_plot
        for station in np.unique(obs_data.stations_lum):
            # plot the height vs. absolute_magnitudes
            ax5.plot(sim_diff_lum[np.where(obs_data.stations_lag == station)], \
                    obs_data.height_lum[np.where(obs_data.stations_lag == station)]/1000, '.', \
                    color=station_colors[station], label=station)

        # find the obs_data.leading_frag_height_arr index is close to obs_data.height_lum[0] wihouth nan
        index = np.argmin(np.abs(sim_data.leading_frag_height_arr[~np.isnan(sim_data.leading_frag_height_arr)]-obs_data.height_lag[0]))
        # plot velocity_arr vs leading_frag_time_arr
        ax2.plot(sim_data.time_arr-sim_data.time_arr[index], sim_data.leading_frag_vel_arr/1000, color='black', label='Best guess')
        ax2.legend()

        # inerpoate the velocity_arr to the leading_frag_time_arr
        sim_vel = np.interp(obs_data.height_lag,
                                    np.flip(sim_data.leading_frag_height_arr),
                                    np.flip(sim_data.leading_frag_vel_arr))
        
        # make the difference between the no_noise_vel and the obs_data.velocities
        sim_diff_vel = sim_vel - obs_data.velocities

        # for each station in obs_data_plot
        for station in np.unique(obs_data.stations_lag):
            # plot the height vs. absolute_magnitudes
            ax6.plot(obs_data.time_lag[np.where(obs_data.stations_lag == station)], \
                    sim_diff_vel[np.where(obs_data.stations_lag == station)]/1000, '.', \
                        color=station_colors[station], label=station)

        # plot lag_arr vs leading_frag_time_arr withouth nan values
        sim_lag = (sim_data.leading_frag_length_arr-sim_data.leading_frag_length_arr[index])\
              - ((obs_data.velocities[0])*(sim_data.time_arr-sim_data.time_arr[index]))
        
        sim_lag -= sim_lag[index]
        # plot lag_arr vs leading_frag_time_arr
        ax3.plot(sim_data.time_arr-sim_data.time_arr[index], sim_lag, color='black', label='Best guess')

        # inerpoate the lag_arr to the leading_frag_time_arr
        sim_lag = np.interp(obs_data.height_lag,
                                    np.flip(sim_data.leading_frag_height_arr),
                                    np.flip(sim_lag))
        
        # make the difference between the no_noise_lag and the obs_data.lag
        sim_diff_lag = sim_lag - obs_data.lag
        
        # for each station in obs_data_plot
        for station in np.unique(obs_data.stations_lag):
            # plot the height vs. absolute_magnitudes
            ax7.plot(obs_data.time_lag[np.where(obs_data.stations_lag == station)], \
                    sim_diff_lag[np.where(obs_data.stations_lag == station)], '.', \
                        color=station_colors[station], label=station)

    # ax0.fill_betweenx(height_km_err, abs_mag_sim_err - mag_noise, abs_mag_sim_err + mag_noise, color='darkgray', alpha=0.2)
    # ax0.fill_betweenx(height_km_err, abs_mag_sim_err - mag_noise * real_original['z_score'], abs_mag_sim_err + mag_noise * real_original['z_score'], color='lightgray', alpha=0.2)

    # ax2.fill_between(residual_time_pos, vel_kms_err - vel_noise, vel_kms_err + vel_noise, color='darkgray', alpha=0.2)
    # ax2.fill_between(residual_time_pos, vel_kms_err - vel_noise * real_original['z_score'], vel_kms_err + vel_noise * real_original['z_score'], color='lightgray', alpha=0.2)
    
    #### avoid overlapping of the x-axis labels ####

    # # After plotting, get the current ticks that matplotlib chose
    # current_ticks = ax1.get_xticks()
    # # Suppose you want to ensure 3 ticks, keep the two rightmost plus zero,
    # # i.e., if 0 is not among the rightmost two, force it in.
    # # A simple approach (though not always perfect):
    # rightmost_two = current_ticks[1:]       # the last 2 ticks
    # if 0 not in rightmost_two:
    #     new_ticks = [0] + list(rightmost_two)
    # else:
    #     # 0 is already there, so just keep the last 3:
    #     new_ticks = current_ticks[1:]
    # # Now set them
    # ax1.set_xticks(new_ticks)
    # # If you want them labeled as-is:
    # ax1.set_xticklabels([str(tk) for tk in new_ticks])

    # # After plotting, get the current ticks that matplotlib chose
    # current_ticks = ax5.get_xticks()
    # # Suppose you want to ensure 3 ticks, keep the two rightmost plus zero,
    # # i.e., if 0 is not among the rightmost two, force it in.
    # # A simple approach (though not always perfect):
    # rightmost_two = current_ticks[-2:]       # the last 2 ticks
    # if 0 not in rightmost_two:
    #     new_ticks = [0] + list(rightmost_two)
    # else:
    #     # 0 is already there, so just keep the last 3:
    #     new_ticks = current_ticks[-3:]
    # # Now set them
    # ax5.set_xticks(new_ticks)
    # # If you want them labeled as-is:
    # ax5.set_xticklabels([str(tk) for tk in new_ticks])

    # Save the plot
    print('file saved: '+output_folder +os.sep+ file_name+'_best_fit_plot.png')
    fig.savefig(output_folder +os.sep+ file_name +'_best_fit_plot.png', dpi=300)

    # Display the plot
    plt.close(fig)


# Plotting function dynesty
def plot_dynesty(dynesty_run_results, obs_data, flags_dict, fixed_values, output_folder='', file_name=''):

    print(dynesty_run_results.summary())
    print('information gain:', dynesty_run_results.information[-1])
    print('niter i.e number of metsim simulated events\n')

    fig, axes = dyplot.runplot(dynesty_run_results,label_kwargs={"fontsize": 10})  # Reduce title font size)
    plt.savefig(output_folder +os.sep+ file_name +'_dynesty_runplot.png', dpi=300)
    plt.close(fig)

    variables = list(flags_dict.keys())

    logwt = dynesty_run_results.logwt

    # Subtract the maximum logwt for numerical stability
    logwt_shifted = logwt - np.max(logwt)
    weights = np.exp(logwt_shifted)

    # Normalize so that sum(weights) = 1
    weights /= np.sum(weights)

    samples_equal = dynesty.utils.resample_equal(dynesty_run_results.samples, weights)

    # Mapping of original variable names to LaTeX-style labels
    variable_map = {
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
        'noise_lag': r"$\varepsilon_{lag}$ [m]",
        'noise_lum': r"$\varepsilon_{lum}$ [J/s]"
    }

    # check if there are variables in the flags_dict that are not in the variable_map
    for variable in variables:
        if variable not in variable_map:
            print(f"Warning: {variable} not found in variable_map")
            # Add the variable to the map with a default label
            variable_map[variable] = variable
    labels = [variable_map[variable] for variable in variables]
    labels_plot = labels. copy() # list for plot labels

    ndim = len(variables)
    sim_num = -1
    # copy the best guess values
    best_guess = copy.deepcopy(dynesty_run_results.samples[sim_num])
    # for variable in variables: for 
    for i, variable in enumerate(variables):
        if 'log' in flags_dict[variable]:  
            samples_equal[:, i] = 10**(samples_equal[:, i])
            best_guess[i] = 10**(best_guess[i])
            labels_plot[i] =r"$\log_{10}$(" +labels_plot[i]+")"

    print('Best fit:')
    # write the best fit variable names and then the best guess values
    for i in range(len(best_guess)):
        print(variables[i],':\t', best_guess[i])
    print('logL:', dynesty_run_results.logl[sim_num])
    real_logL = None
    diff_logL = None
    if hasattr(obs_data, 'const'):
        simulated_lc_intensity = np.interp(obs_data.height_lum, 
                                        np.flip(obs_data.leading_frag_height_arr), 
                                        np.flip(obs_data.luminosity_arr))

        lag_sim = obs_data.leading_frag_length_arr - (obs_data.v_init * obs_data.time_arr)
        simulated_lag = np.interp(obs_data.height_lag, 
                                np.flip(obs_data.leading_frag_height_arr), 
                                np.flip(lag_sim))
        lag_sim = simulated_lag - simulated_lag[0]

        ### Log Likelihood ###
        log_likelihood_lum = np.nansum(-0.5 * np.log(2*np.pi*obs_data.noise_lum**2) - 0.5 / (obs_data.noise_lum**2) * (obs_data.luminosity - simulated_lc_intensity) ** 2)
        log_likelihood_lag = np.nansum(-0.5 * np.log(2*np.pi*obs_data.noise_lag**2) - 0.5 / (obs_data.noise_lag**2) * (obs_data.lag - lag_sim) ** 2)
        real_logL = log_likelihood_lum + log_likelihood_lag
        diff_logL = dynesty_run_results.logl[sim_num] - real_logL
        # use log_likelihood_dynesty to compute the logL
        print('REAL logL:', real_logL)
        print('DIFF logL:', diff_logL)
    ### PLOT best fit ###

    best_guess_obj_plot = run_simulation(best_guess, obs_data, variables, fixed_values)

    # Plot the data with residuals and the best fit
    # plot_data_with_residuals_and_real(obs_data, best_guess_obj_plot, output_folder, file_name + "_best_fit")
    plot_data_with_residuals_and_real(obs_data, best_guess_obj_plot, output_folder, file_name)



    ### TABLE OF POSTERIOR SUMMARY STATISTICS ###

    # Posterior mean (per dimension)
    posterior_mean = np.mean(samples_equal, axis=0)      # shape (ndim,)

    # Posterior median (per dimension)
    posterior_median = np.median(samples_equal, axis=0)  # shape (ndim,)

    # 95% credible intervals (2.5th and 97.5th percentiles)
    lower_95 = np.percentile(samples_equal, 2.5, axis=0)   # shape (ndim,)
    upper_95 = np.percentile(samples_equal, 97.5, axis=0)  # shape (ndim,)

    # Function to approximate mode using histogram binning
    def approximate_mode_1d(samples):
        hist, bin_edges = np.histogram(samples, bins='auto', density=True)
        idx_max = np.argmax(hist)
        return 0.5 * (bin_edges[idx_max] + bin_edges[idx_max + 1])

    approx_modes = [approximate_mode_1d(samples_equal[:, d]) for d in range(ndim)]

    truth_values_plot = {}
    # if 'dynesty_run_results has const
    if hasattr(obs_data, 'const'):

        truth_values_plot = {}
        # if 'noise_lag' take it from obs_data.noise_lag
        if 'noise_lag' in flags_dict.keys():
            truth_values_plot['noise_lag'] = obs_data.noise_lag
        # if 'noise_mag' take it from obs_data.noise_mag
        if 'noise_lum' in flags_dict.keys():
            truth_values_plot['noise_lum'] = obs_data.noise_lum

        # Extract values from dictionary
        for variable in variables:
            if variable in obs_data.const:  # Use dictionary lookup instead of hasattr()
                truth_values_plot[variable] = obs_data.const[variable]
            else:
                print(f"Warning: {variable} not found in obs_data.const")

        # Convert to array safely
        truths = np.array([truth_values_plot.get(variable, np.nan) for variable in variables])

        # Apply log10 safely if needed
        for variable in variables:
            if 'log' in flags_dict.get(variable, []):
                if variable in truth_values_plot:
                    truth_values_plot[variable] = np.log10(truth_values_plot[variable]) #np.log10(np.maximum(truth_values_plot[variable], 1e-10))
                else:
                    print(f"Skipping {variable}: Missing from truth_values_plot")

        # Compare to true theta
        bias = posterior_mean - truths
        abs_error = np.abs(bias) * 100
        rel_error = abs_error / np.abs(truths) * 100

        # Coverage check
        coverage_mask = (truths >= lower_95) & (truths <= upper_95)
        print("Coverage mask per dimension:", coverage_mask)
        print("Fraction of dimensions covered:", coverage_mask.mean())

        # Generate LaTeX table
        latex_str = r"""\begin{table}[htbp]
    \centering
    \renewcommand{\arraystretch}{1.2} % Increase row height for readability
    \setlength{\tabcolsep}{4pt} % Adjust column spacing
    \resizebox{\textwidth}{!}{ % Resizing table to fit page width
    \begin{tabular}{|l|c|c|c|c|c|c||c|c||c|}
    \hline
    Parameter & 2.5CI & True Value & Mean & Median & Mode & 97.5CI & Abs.Error\% & Rel.Error\% & Cover \\
    \hline
        """
        # & Mode
        # {approx_modes[i]:.4g} &
        for i, label in enumerate(labels):
            coverage_val = "\ding{51}" if coverage_mask[i] else "\ding{55}"  # Use checkmark/x for coverage
            latex_str += (f"    {label} & {lower_95[i]:.4g} & {truths[i]:.4g} & {posterior_mean[i]:.4g} "
                        f"& {posterior_median[i]:.4g} & {approx_modes[i]:.4g} & {upper_95[i]:.4g} "
                        f"& {abs_error[i]:.4g} & {rel_error[i]:.4g} & {coverage_val} \\\\\n    \hline\n")

    else:
        # Generate LaTeX table
        latex_str = r"""\begin{table}[htbp]
    \centering
    \renewcommand{\arraystretch}{1.2} % Increase row height for readability
    \setlength{\tabcolsep}{4pt} % Adjust column spacing
    \resizebox{\textwidth}{!}{ % Resizing table to fit page width
    \begin{tabular}{|l|c|c|c|c|c|}
    \hline
    Parameter & 2.5CI & Mean & Median & Mode & 97.5CI\\
    \hline
        """
        # & Mode
        # {approx_modes[i]:.4g} &
        for i, label in enumerate(labels):
            latex_str += (f"    {label} & {lower_95[i]:.4g} & {posterior_mean[i]:.4g} "
                        f"& {posterior_median[i]:.4g} & {approx_modes[i]:.4g} & {upper_95[i]:.4g} \\\\\n    \hline\n")

    latex_str += r"""
    \end{tabular}}
    \caption{Posterior summary statistics comparing estimated values with the true values. The cover column indicates whether the true value is within the 95\% confidence interval.}
    \label{tab:posterior_summary}
\end{table}"""

    # Capture the printed output of summary()
    summary_buffer = io.StringIO()
    sys.stdout = summary_buffer  # Redirect stdout
    dynesty_run_results.summary()  # This prints to the buffer instead of stdout
    sys.stdout = sys.__stdout__  # Reset stdout
    summary_str = summary_buffer.getvalue()  # Get the captured text

    # Save to a .tex file
    with open(output_folder+os.sep+file_name+"_results_table.tex", "w") as f:
        f.write(summary_str+'\n')
        f.write("H info.gain:"+str(dynesty_run_results.information[-1])+'\n')
        f.write("niter i.e number of metsim simulated events\n\n")
        f.write("Best fit:\n")
        for i in range(len(best_guess)):
            f.write(variables[i]+':\t'+str(best_guess[i])+'\n')
        f.write('logL:'+str(dynesty_run_results.logl[sim_num])+'\n')
        if diff_logL is not None:
            f.write('REAL logL:'+str(real_logL)+'\n')
            f.write('diff logL:'+str(diff_logL)+'\n')
        f.write("\n")
        f.write(latex_str)
        f.close()

    # Print LaTeX code for quick copy-pasting
    print(latex_str)

    ### Plot the trace plot ###

    print('saving trace plot...')

    if hasattr(obs_data, 'const'):
        # 25310it [5:59:39,  1.32s/it, batch: 0 | bound: 10 | nc: 30 | ncall: 395112 | eff(%):  6.326 | loglstar:   -inf < -16256.467 <    inf | logz: -16269.475 +/-  0.049 | dlogz: 15670.753 >  0.010]
        truth_plot = np.array([truth_values_plot[variable] for variable in variables])

        fig, axes = dyplot.traceplot(dynesty_run_results, truths=truth_plot, labels=labels_plot,
                                    label_kwargs={"fontsize": 10},  # Reduce axis label size
                                    title_kwargs={"fontsize": 10},  # Reduce title font size
                                    title_fmt='.2e',  # Scientific notation for titles
                                    truth_color='black', show_titles=True,
                                    trace_cmap='viridis', connect=True,
                                    connect_highlight=range(5))

    else:

        fig, axes = dyplot.traceplot(dynesty_run_results, labels=labels_plot,
                                    label_kwargs={"fontsize": 10},  # Reduce axis label size
                                    title_kwargs={"fontsize": 10},  # Reduce title font size
                                    title_fmt='.2e',  # Scientific notation for titles
                                    show_titles=True,
                                    trace_cmap='viridis', connect=True,
                                    connect_highlight=range(5))

    # Adjust spacing and tick label size
    fig.subplots_adjust(hspace=0.5)  # Increase spacing between plots

    # save the figure
    plt.savefig(output_folder+os.sep+file_name+'_trace_plot.png', dpi=300)

    # show the trace plot
    # plt.show()
    plt.close(fig)

    ### Plot the corner plot ###

    print('saving corner plot...')

    # Trace Plots
    fig, axes = plt.subplots(ndim, ndim, figsize=(35, 15))
    axes = axes.reshape((ndim, ndim))  # reshape axes

    if hasattr(obs_data, 'const'):
        # Increase spacing between subplots
        fg, ax = dyplot.cornerplot(
            dynesty_run_results, 
            color='blue', 
            truths=truth_plot,  # Use the defined truth values
            truth_color='black', 
            show_titles=True, 
            max_n_ticks=3, 
            quantiles=None, 
            labels=labels_plot,  # Update axis labels
            label_kwargs={"fontsize": 10},  # Reduce axis label size
            title_kwargs={"fontsize": 10},  # Reduce title font size
            title_fmt='.2e',  # Scientific notation for titles
            fig=(fig, axes[:, :ndim])
        )
    else:

        # Increase spacing between subplots
        fg, ax = dyplot.cornerplot(
            dynesty_run_results, 
            color='blue', 
            show_titles=True, 
            max_n_ticks=3, 
            quantiles=None, 
            labels=labels_plot,  # Update axis labels
            label_kwargs={"fontsize": 10},  # Reduce axis label size
            title_kwargs={"fontsize": 10},  # Reduce title font size
            title_fmt='.2e',  # Scientific notation for titles
            fig=(fig, axes[:, :ndim])
        )

    # # Reduce tick size
    # for ax_row in ax:
    #     for ax_ in ax_row:
    #         ax_.tick_params(axis='both', labelsize=6)  # Reduce tick number size

    # Apply scientific notation and horizontal tick labels
    for ax_row in ax:
        for ax_ in ax_row:
            ax_.tick_params(axis='both', labelsize=10, direction='in')

            # Set tick labels to be horizontal
            for label in ax_.get_xticklabels():
                label.set_rotation(0)
            for label in ax_.get_yticklabels():
                label.set_rotation(45)

            if ax_ is None:
                continue  # if cornerplot left some entries as None
            
            # Get the actual major tick locations.
            x_locs = ax_.xaxis.get_majorticklocs()
            y_locs = ax_.yaxis.get_majorticklocs()

            # Only update the formatter if we actually have tick locations:
            if len(x_locs) > 0:
                ax_.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.4g'))
            if len(y_locs) > 0:
                ax_.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4g'))

    for i in range(ndim):
        for j in range(ndim):
            # In some corner-plot setups, the upper-right triangle can be None
            if ax[i, j] is None:
                continue
            
            # Remove y-axis labels (numbers) on the first column (j==0)
            if j != 0:
                ax[i, j].set_yticklabels([])  
                # or ax[i, j].tick_params(labelleft=False) if you prefer

            # Remove x-axis labels (numbers) on the bottom row (i==ndim-1)
            if i != ndim - 1:
                ax[i, j].set_xticklabels([])  
                # or ax[i, j].tick_params(labelbottom=False)

    # Adjust spacing and tick label size
    fg.subplots_adjust(wspace=0.1, hspace=0.3)  # Increase spacing between plots

    # save the figure
    plt.savefig(output_folder+os.sep+file_name+'_corner_plot.png', dpi=300)

    # close the figure
    plt.close(fig)

###############################################################################
# Function: read prior to generate bounds
###############################################################################
def read_prior_to_bounds(object_meteor,file_path=""):
    # Default bounds
    default_bounds = {
        "v_init": (500, np.nan),
        "zenith_angle": (0.01, np.nan),
        "m_init": (np.nan, np.nan),
        "rho": (10, 4000),  # log transformation applied later
        "sigma": (0.001 / 1e6, 0.05 / 1e6),
        "erosion_height_start": (\
            object_meteor.height_lum[0] -100- (object_meteor.height_lum[0]-object_meteor.height_lum[np.argmax(object_meteor.luminosity)])/2,\
            object_meteor.height_lum[0] +100+ (object_meteor.height_lum[0]-object_meteor.height_lum[np.argmax(object_meteor.luminosity)])/2),
        "erosion_coeff": (1 / 1e12, 2 / 1e6),  # log transformation applied later
        "erosion_mass_index": (1, 3),
        "erosion_mass_min": (5e-12, 1e-9),  # log transformation applied later
        "erosion_mass_max": (1e-10, 1e-7),  # log transformation applied later
        "rho_grain": (3000, 3500),
        "erosion_height_change": (\
            object_meteor.height_lum[-1]+100+ (object_meteor.height_lum[0]-object_meteor.height_lum[np.argmax(object_meteor.luminosity)])/2,\
            object_meteor.height_lum[0] -100- (object_meteor.height_lum[0]-object_meteor.height_lum[np.argmax(object_meteor.luminosity)])/2),
        "erosion_coeff_change": (1 / 1e12, 2 / 1e6),  # log transformation applied later
        "erosion_rho_change": (10, 4000),  # log transformation applied later
        "erosion_sigma_change": (0.001 / 1e6, 0.05 / 1e6),
        "noise_lag": (10, object_meteor.noise_lag), # more of a peak around the real value
        "noise_lum": (3, object_meteor.noise_lum) # look for more values at higher uncertainty can be because of the noise
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
        "erosion_mass_max": ["log"],
        "rho_grain": [],
        "erosion_height_change": [],
        "erosion_coeff_change": ["log"],
        "erosion_rho_change": ["log"],
        "erosion_sigma_change": [],
        "noise_lag": ["invgamma"],
        "noise_lum": ["invgamma"]
        }

    rho_grain_real = 3000
    # check if object_meteor.const.rho_grain exist
    if hasattr(object_meteor, 'const'):
        rho_grain_real = object_meteor.const["rho_grain"]

    # Default values if no file path is provided
    if file_path == "":
        # delete from the default_bounds the "zenith_angle","rho_grain","erosion_height_change","erosion_coeff_change","erosion_rho_change","erosion_sigma_change"
        default_bounds.pop("zenith_angle")
        default_bounds.pop("rho_grain")
        default_bounds.pop("erosion_height_change")
        default_bounds.pop("erosion_coeff_change")
        default_bounds.pop("erosion_rho_change")
        default_bounds.pop("erosion_sigma_change")
        
        bounds = [default_bounds[key] for key in default_bounds]
        flags_dict = {key: default_flags.get(key, []) for key in default_bounds}
        # for the one that have log transformation, apply it
        for i, key in enumerate(default_bounds):
            if "log" in flags_dict[key]:
                bounds[i] = np.log10(bounds[i][0]), np.log10(bounds[i][1])

        # check if any of the values are np.nan and replace them with the object_meteor values
        for i, key in enumerate(default_bounds):
            bounds[i] = list(bounds[i])  # Convert tuple to list
            # now check if the values are np.nan and if the flag key is 'norm' then divide by 10
            if np.isnan(bounds[i][0]) and key in object_meteor.__dict__:
                bounds[i][0] = object_meteor.__dict__[key] - object_meteor.__dict__[key]/10/2

            if np.isnan(bounds[i][1]) and key in object_meteor.__dict__ and "norm" in flags_dict[key]:
                bounds[i][1] = object_meteor.__dict__[key]
            elif np.isnan(bounds[i][1]) and key in object_meteor.__dict__:
                bounds[i][1] = object_meteor.__dict__[key] + object_meteor.__dict__[key]/10/2
            bounds[i] = tuple(bounds[i])  # Convert back to tuple if needed
        # checck if stil any bounds are np.nan and raise an error
        for i, key in enumerate(default_bounds):
            if np.isnan(bounds[i][0]) or np.isnan(bounds[i][1]):
                raise ValueError(f"The value for {key} is np.nan and it is not in the dictionary")

        fixed_values = {
            "zenith_angle": object_meteor.zenith_angle,
            "rho_grain": rho_grain_real,
            "erosion_height_change": 1, # deactivate the erosion height change
            "erosion_coeff_change": 1 / 1e7,
            "erosion_rho_change": rho_grain_real,
            "erosion_sigma_change": 0.001 / 1e6
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
                min_val = safe_eval(min_val) if min_val.lower() != "nan" else np.nan
                max_val = safe_eval(max_val) if max_val.lower() != "nan" else np.nan

                #### vel, mass, zenith ####
                # check if name=='v_init' or zenith_angle or m_init or erosion_height_start values are np.nan and replace them with the object_meteor values
                if np.isnan(min_val) and name in object_meteor.__dict__ and ("norm" in flags or "invgamma" in flags):
                    if "norm" in default_flags[name] or "invgamma" in default_flags[name]:
                        min_val = default_bounds.get(name, (np.nan, np.nan))[0]
                    else:
                        min_val = object_meteor.__dict__[name]/10/2
                if np.isnan(min_val) and name in object_meteor.__dict__:
                    # if norm in default_flags[name] then divide by 10
                    if "norm" in default_flags[name] or "invgamma" in default_flags[name]:
                        min_val = object_meteor.__dict__[name] + default_bounds.get(name, (np.nan, np.nan))[0]
                    else:
                        min_val = object_meteor.__dict__[name] - object_meteor.__dict__[name]/10/2

                if np.isnan(max_val) and name in object_meteor.__dict__ and ("norm" in flags or "invgamma" in flags):
                    max_val = object_meteor.__dict__[name]
                if np.isnan(max_val) and name in object_meteor.__dict__:
                    # if norm in default_flags[name] then divide by 10
                    if "norm" in default_flags[name] or "invgamma" in default_flags[name]:
                        max_val = object_meteor.__dict__[name] + default_bounds.get(name, (np.nan, np.nan))[0]
                    else:
                        max_val = object_meteor.__dict__[name] + object_meteor.__dict__[name]/10/2
                
                #### rest of variables ####
                if np.isnan(min_val) and ("norm" in flags or "invgamma" in flags):
                    if "norm" in default_flags[name] or "invgamma" in default_flags[name]:
                        min_val = default_bounds.get(name, (np.nan, np.nan))[0]
                    else:
                        min_val = (default_bounds.get(name, (np.nan, np.nan))[1]-default_bounds.get(name, (np.nan, np.nan))[0])/10/2
                elif np.isnan(min_val):
                    min_val = default_bounds.get(name, (np.nan, np.nan))[0]
                
                if np.isnan(max_val) and ("norm" in flags or "invgamma" in flags):
                    if "norm" in default_flags[name] or "invgamma" in default_flags[name]:
                        max_val = default_bounds.get(name, (np.nan, np.nan))[1]
                    else:
                        max_val = np.mean([default_bounds.get(name,(np.nan, np.nan))[1],default_bounds.get(name,(np.nan, np.nan))[0]])
                elif np.isnan(max_val):
                    max_val = default_bounds.get(name, (np.nan, np.nan))[1]
                
                # check if min_val > max_val, then swap them cannot have negative values
                if min_val > max_val and "invgamma" not in flags:
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

                # check if any of the values are np.nan raise an error
                if np.isnan(min_val) or np.isnan(max_val):
                    raise ValueError(f"The value for {name} is np.nan and it is not in the dictionary")
                # check if inf in the values and raise an error
                if np.isinf(min_val) or np.isinf(max_val):
                    raise ValueError(f"The value for {name} is inf and it is not in the dictionary")

                bounds.append((min_val, max_val))
    
    # take the key of the fixed_values and append flags_dict
    variable_loaded = list(fixed_values.keys()) + list(flags_dict.keys())
    # check if among the variable_loaded if there is 'v_init' or 'zenith_angle' or 'm_init' in case are not in the variable_loaded put it in to fixed_values
    if 'v_init' not in variable_loaded:
        fixed_values['v_init'] = object_meteor.v_init
    if 'zenith_angle' not in variable_loaded:
        fixed_values['zenith_angle'] = object_meteor.zenith_angle
    if 'm_init' not in variable_loaded:
        fixed_values['m_init'] = object_meteor.m_init
    if 'rho_grain' not in variable_loaded:
        fixed_values['rho_grain'] = rho_grain_real
    if 'erosion_height_change' not in variable_loaded:
        fixed_values['erosion_height_change'] = 1

    # check if the bounds the len(bounds) + len(fixed_values) =>10
    if len(bounds) + len(fixed_values) < 10:
        raise ValueError("The number of bounds and fixed values should 10 or above")

    return bounds, flags_dict, fixed_values


###############################################################################
# Load observation data
###############################################################################
class observation_data:
    def __init__(self, obs_file_path,use_CAMO_data):
        self.file_name = obs_file_path
        # check if the file is a json file
        if obs_file_path.endswith('.pickle'):
            self.load_pickle_data(use_CAMO_data)
        elif obs_file_path.endswith('.json'):
            self.load_json_data(use_CAMO_data)
        else:
            # file type not supported
            raise ValueError("File type not supported, only .json and .pickle files are supported")

    def load_pickle_data(self,use_CAMO_data):
        print('Loading pickle file:',self.file_name)
        # load the pickle file
        traj=loadPickle(*os.path.split(self.file_name))
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
                if peak_abs_mag_CAMO is None:
                    peak_abs_mag_CAMO = np.min(obs.absolute_magnitudes)
                elif peak_abs_mag_CAMO > np.min(obs.absolute_magnitudes):
                    peak_abs_mag_CAMO = np.min(obs.absolute_magnitudes)

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
        
        print("NOTE: Applying photometric adjustment to mach the luminosity of the two stations")

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



    def load_json_data(self,use_CAMO_data):

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

        print(f"Loading json file: {self.file_name}")

        # Read the JSON file
        with open(self.file_name, 'r') as f:
            data_dict = json.load(f)

        # check if data_dict has the key time_lag
        if 'time_lag' in data_dict.keys():

            # Convert lists back to numpy arrays where necessary
            def restore_data(obj):
                if isinstance(obj, dict):
                    return {k: restore_data(v) for k, v in obj.items()}

                elif isinstance(obj, list):
                    # If all items are numeric, convert to np.array of floats
                    if all(isinstance(i, (int, float)) for i in obj):
                        return np.array(obj)

                    # If all items are strings, convert to np.array of strings
                    if all(isinstance(i, str) for i in obj):
                        return np.array(obj, dtype=str)

                    # Otherwise, recurse in case it's a nested list
                    return [restore_data(v) for v in obj]

                else:
                    return obj

            restored_dict = restore_data(data_dict)
            self.__dict__.update(restored_dict)

        else:

            # Load the constants
            const, _ = loadConstants(self.file_name)
            const.dens_co = np.array(const.dens_co)

            # const_nominal.P_0m = 935

            # const.disruption_on = False

            const.lum_eff_type = 5

            # Run the simulation
            frag_main, results_list, wake_results = runSimulation(const, compute_wake=False)
            simulation_MetSim_object = SimulationResults(const, frag_main, results_list, wake_results)

            # Store results in the object
            self.__dict__.update(simulation_MetSim_object.__dict__)

            self.noise_lum = 2.5
            self.noise_mag = 0.1

            # add a gausian noise to the luminosity of 2.5
            lum_obs_data = self.luminosity_arr + np.random.normal(loc=0, scale=self.noise_lum, size=len(self.luminosity_arr))

            # Identify indices where lum_obs_data > 0
            positive_indices = np.where(lum_obs_data > 0)[0]  # Get only valid indices

            # If no positive values exist, return empty list
            if len(positive_indices) == 0:
                indices_visible = []
            else:
                # Find differences between consecutive indices
                diff = np.diff(positive_indices)

                # Identify breaks (where difference is more than 1)
                breaks = np.where(diff > 1)[0]

                # Split the indices into uninterrupted sequences
                sequences = np.split(positive_indices, breaks + 1)

                # Find the longest sequence
                indices_visible = max(sequences, key=len)

            # Store the constants
            self.v_init = self.const.v_init
            self.zenith_angle = self.const.zenith_angle
            self.m_init = self.const.m_init

            self.P_0m = self.const.P_0m
            self.dens_co = np.array(self.const.dens_co) 

            # Compute absolute magnitudes
            absolute_magnitudes_check = -2.5 * np.log10(lum_obs_data / self.P_0m)

            # Check if absolute_magnitudes_check exceeds 8
            if len(indices_visible) > 0:
                mask = absolute_magnitudes_check[indices_visible] > 8  # Only check relevant indices
                if np.any(mask):
                    print('Found values below 8 absolute magnitudes:', absolute_magnitudes_check[indices_visible][mask])

                    # Remove invalid indices
                    indices_visible = indices_visible[~mask]

                    # If gaps exist, extract longest continuous segment
                    indices_visible = np.sort(indices_visible)
                    diff = np.diff(indices_visible)
                    breaks = np.where(diff > 1)[0]
                    sequences = np.split(indices_visible, breaks + 1)
                    indices_visible = max(sequences, key=len) if sequences else []
                
            # Select time, magnitude, height, and length above the visibility limit
            time_visible = self.time_arr[indices_visible]
            # the rest of the arrays are the same length as time_arr
            lum_visible = lum_obs_data[indices_visible]
            ht_visible   = self.brightest_height_arr[indices_visible]
            len_visible  = self.brightest_length_arr[indices_visible]
            # mag_visible  = self.abs_magnitude[indices_visible]
            # vel_visible  = self.leading_frag_vel_arr[indices_visible]

            # Resample the time to the system FPS
            lum_interpol = scipy.interpolate.CubicSpline(time_visible, lum_visible)
            ht_interpol  = scipy.interpolate.CubicSpline(time_visible, ht_visible)
            len_interpol = scipy.interpolate.CubicSpline(time_visible, len_visible)
            # mag_interpol = scipy.interpolate.CubicSpline(time_visible, mag_visible)
            # vel_interpol = scipy.interpolate.CubicSpline(time_visible, vel_visible)

            fps_lum = 32
            if use_CAMO_data:
                self.fps = 80
                self.stations = ['01G','02G','01T','02T']
                self.noise_lag = 5
                self.noise_vel = self.noise_lag*np.sqrt(2)/(1.0/self.fps)
                # multiply by a number between 0.6 and 0.4 for the time to track for CAMO
                time_to_track = (time_visible[-1]-time_visible[0])*np.random.uniform(0.4,0.6)
                time_sampled_lag, stations_array_lag = self.mimic_fps_camera(time_visible,time_to_track,self.fps,self.stations[2],self.stations[3])
                time_sampled_lum, stations_array_lum = self.mimic_fps_camera(time_visible,0,fps_lum,self.stations[0],self.stations[1])
            else:
                self.fps = 32
                self.stations = ['01F','02F']
                self.noise_lag = 40
                self.noise_vel = self.noise_lag*np.sqrt(2)/(1.0/self.fps)
                time_to_track = 0
                time_sampled_lag, stations_array_lag = self.mimic_fps_camera(time_visible,time_to_track,self.fps,self.stations[0],self.stations[1])
                time_sampled_lum, stations_array_lum = time_sampled_lag, stations_array_lag

            # Create new mag, height and length arrays at FPS frequency
            self.stations_lum = stations_array_lum
            self.height_lum = ht_interpol(time_sampled_lum)
            self.time_lum = time_sampled_lum - time_sampled_lum[0]
            self.luminosity = lum_interpol(time_sampled_lum)
            self.absolute_magnitudes = -2.5*np.log10(self.luminosity/self.P_0m) # P_0m*(10 ** (obs.absolute_magnitudes/(-2.5)))
            
            # mag_sampled = mag_interpol(time_sampled_lum)
            self.stations_lag = stations_array_lag
            self.height_lag = ht_interpol(time_sampled_lag)
            self.time_lag = time_sampled_lag - time_sampled_lag[0]

            # Create new length and velocity arrays at FPS frequency and add noise
            self.length = len_interpol(time_sampled_lag) 
            self.length = self.length - self.length[0]
            self.length = self.length + np.random.normal(loc=0, scale=self.noise_lag, size=len(time_sampled_lag))

            # # Find the index where time_arr is closest to self.time_arr[np.min(indices_visible)] + time_to_track
            closest_index_vel = np.argmin(np.abs(self.brightest_height_arr - self.height_lag[0]))
            
            # find the velcity at the beginning of the smallest index in observed_index
            v_first_frame = self.leading_frag_vel_arr[closest_index_vel]
            # make an empty list for the velocities with zeros for the length of the time_sampled_lag
            velocities_noise = np.zeros(len(time_sampled_lag))
            # for each of the unique stations in the stations_lag divided by the time_lag of the station
            for station in np.unique(self.stations_lag):
                # find the indices of the station in the stations_lag
                station_indices = np.where(self.stations_lag == station)
                # make the difference between the length of the station_indices
                diff_length = np.diff(self.length[station_indices])
                # make the difference between the time of the station_indices
                diff_time = np.diff(self.time_lag[station_indices])
                # calculate the velocity of the station
                velocity = np.divide(diff_length,diff_time)
                # put the first equal to v_first_frame and push the rest of the values
                velocity = np.insert(velocity,0,v_first_frame)
                # put in the index of the station_indices the velocities_noise
                velocities_noise[station_indices] = velocity
            # concatenate the velocities
            self.velocities = velocities_noise

            optimized_v_first_frame = v_first_frame

            optimized_v_first_frame = self.find_optimal_v_first_frame(v_first_frame)

            # Now use the optimized velocity for computing lag
            self.lag = self.length - (optimized_v_first_frame * self.time_lag)

            self._save_json_data()


    def find_optimal_v_first_frame(self, v_first_frame_initial=60000, bounds=(10000, 72000)):
        """
        Optimizes v_first_frame to minimize the difference between observed lag and simulated lag.

        Parameters:
        v_first_frame_initial (float): Initial guess for v_first_frame.
        bounds (tuple): Lower and upper bounds for v_first_frame.

        Returns:
        float: Optimized v_first_frame.
        """

        def objective(v_first_frame):
            """Objective function to minimize the difference in lag."""
            # Compute the lag based on v_first_frame
            computed_lag = self.length - (v_first_frame * self.time_lag)

            # Find the index closest to the first height without NaNs
            index = np.argmin(np.abs(self.leading_frag_height_arr[~np.isnan(self.leading_frag_height_arr)]
                                      - self.height_lag[0]))

            # Compute the theoretical lag (without noise)
            lag_no_noise = (self.leading_frag_length_arr - self.leading_frag_length_arr[index]) - \
                           (self.velocities[0] * (self.time_arr - self.time_arr[index]))
            lag_no_noise -= lag_no_noise[index]

            # Interpolate to align with observed height_lag
            no_noise_lag = np.interp(self.height_lag,
                                     np.flip(self.leading_frag_height_arr),
                                     np.flip(lag_no_noise))

            # Compute the squared error between computed and theoretical lag
            diff_lag = no_noise_lag - computed_lag
            return np.sum(diff_lag**2)  # Sum of squared differences

        # Use minimize (which supports an initial guess)
        result = minimize(objective, x0=[v_first_frame_initial], bounds=[bounds], method='L-BFGS-B')

        print(f"Optimized first frame velocity for lag : {result.x[0]:.2f}")
        return result.x[0]



    def mimic_fps_camera(self, time_visible, time_to_track, fps, station1, station2):

        # Sample the time according to the FPS from one camera
        time_sampled_cam1 = np.arange(np.min(time_visible)+time_to_track, np.max(time_visible), 1.0/fps)

        # Simulate sampling of the data from a second camera, with a random phase shift
        time_sampled_cam2 = time_sampled_cam1 + np.random.uniform(-1.0/fps, 1.0/fps)

        # The second camera will only capture 50 - 100% of the data, simulate this
        cam2_portion = np.random.uniform(0.5, 1.0)
        cam2_start = np.random.uniform(0, 1.0 - cam2_portion)
        cam2_start_index = int(cam2_start*len(time_sampled_cam2))
        cam2_end_index = int((cam2_start + cam2_portion)*len(time_sampled_cam2))

        # Cut the cam2 time to the portion of the data it will capture
        time_sampled_cam2 = time_sampled_cam2[cam2_start_index:cam2_end_index]

        # Cut the time array to the length of the visible data
        time_sampled_cam2 = time_sampled_cam2[(time_sampled_cam2 >= np.min(time_visible)) 
                                            & (time_sampled_cam2 <= np.max(time_visible))]

        # Combine the two camera time arrays
        time_sampled = np.sort(np.concatenate([time_sampled_cam1, time_sampled_cam2]))

        # # find the index of the time_sampled_cam1 in time_sampled
        # index_cam1 = np.searchsorted(time_sampled,time_sampled_cam1)
        # find the index of the time_sampled_cam2 in time_sampled
        index_cam2 = np.searchsorted(time_sampled,time_sampled_cam2)
        # create a array with self.station[-1] for the length of time_sampled
        stations = np.array([station1]*len(time_sampled))
        # replace the values of the index_cam1 with self.stations[0]
        stations[index_cam2] = station2
        
        return time_sampled,stations


    def _save_json_data(self):
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
        json_file_save = os.path.splitext(self.file_name)[0] + "_with_noise.json"
        # check if the file exists if so give a _1, _2, _3, etc. at the end of the file name
        i_json = 1
        if os.path.exists(json_file_save):
            while os.path.exists(json_file_save):
                json_file_save = os.path.splitext(self.file_name)[0] + f"_{i_json}_with_noise.json"
                i_json += 1

        # update the file name
        self.file_name = json_file_save

        # Write to JSON file
        with open(json_file_save, 'w') as f:
            json.dump(serializable_dict, f, indent=4)

        print("Saved fit parameters with noise to:", json_file_save)



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

            if os.path.exists(dynesty_file):
                # Matches the .dynesty base
                if self.resume==False:
                    dynesty_file = self._build_new_dynesty_name(dynesty_file)

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
                    # print the prior path has been found
                    print(f"Prior file found in the same folder as the observation file: {prior_path}")
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
                            # print the prior path has been found
                            print(f"Prior file found in the same folder as the observation file: {prior_path}")
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
# Function: dynesty
###############################################################################

class TimeoutException(Exception):
    """Custom exception for timeouts."""
    pass

def run_simulation_wrapper(guess_var, obs_metsim_obj, var_names, fix_var, queue):
    """Wrapper function for multiprocessing to run the simulation."""
    try:
        result = run_simulation(guess_var, obs_metsim_obj, var_names, fix_var)
        queue.put(result)  # Send result back through queue
    except Exception as e:
        print(f"Error during simulation: {e}")
        queue.put(None)  # Indicate failure


def run_simulation(parameter_guess, real_event, var_names, fix_var):
    '''
        path_and_file = must be a json file generated file from the generate_simulationsm function or from Metsim file
    '''

    # Load the nominal simulation parameters
    const_nominal = Constants()
    const_nominal.dens_co = np.array(const_nominal.dens_co)

    dens_co=real_event.dens_co

    ### Calculate atmosphere density coeffs (down to the bottom observed height, limit to 15 km) ###

    # Assign the density coefficients
    const_nominal.dens_co = dens_co

    # Turn on plotting of LCs of individual fragments 
    const_nominal.fragmentation_show_individual_lcs = True

    # for loop for the var_cost that also give a number from 0 to the length of the var_cost
    for i, var in enumerate(var_names):
        const_nominal.__dict__[var] = parameter_guess[i]

    # first chack if fix_var is not {}
    if fix_var:
        var_names_fix = list(fix_var.keys())
        # for loop for the fix_var that also give a number from 0 to the length of the fix_var
        for i, var in enumerate(var_names_fix):
            const_nominal.__dict__[var] = fix_var[var]

    const_nominal.P_0m = real_event.P_0m

    const_nominal.disruption_on = False

    const_nominal.lum_eff_type = 5

    # # Minimum height [m]
    # const_nominal.h_kill = 60000

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

    return simulation_MetSim_object


def log_likelihood_dynesty(guess_var, obs_metsim_obj, flags_dict, fix_var, timeout=10):
    """
    Runs the simulation with a timeout in a separate process to allow parallel execution.
    """

    var_names = list(flags_dict.keys())
    # check for each var_name in flags_dict if there is "log" in the flags_dict
    for i, var_name in enumerate(var_names):
        if 'log' in flags_dict[var_name]:
            guess_var[i] = 10 ** guess_var[i]
        if var_name == 'noise_lag':
            obs_metsim_obj.noise_lag = guess_var[i]
        if var_name == 'noise_lum':
            obs_metsim_obj.noise_lum = guess_var[i]

    # check if among the var_names there is a "erosion_mass_max" and if there is a "erosion_mass_min"
    if 'erosion_mass_max' in var_names and 'erosion_mass_min' in var_names:
        # check if the guess_var of the erosion_mass_max is smaller than the guess_var of the erosion_mass_min
        if guess_var[var_names.index('erosion_mass_max')] < guess_var[var_names.index('erosion_mass_min')]:
            # # if so, set the guess_var of the erosion_mass_max to the guess_var of the erosion_mass_min
            # guess_var[var_names.index('erosion_mass_max')] = guess_var[var_names.index('erosion_mass_min')]
            # unphysical values, return -np.inf
            return -np.inf  # immediately return -np.inf if times out

    ### LINUX ###

    # Set timeout handler
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)  # Start the timer for timeout
    # get simulated LC intensity onthe object
    try: # try to run the simulation
        simulation_results = run_simulation(guess_var, obs_metsim_obj, var_names, fix_var)
    except TimeoutException:
        print('timeout')
        return -np.inf  # immediately return -np.inf if times out
    finally:
        signal.alarm(0)  # Cancel alarm
    
    ### LINUX ###

    ### WINDOWS ### does not work...

    # queue = multiprocessing.Queue()
    # process = multiprocessing.Process(target=run_simulation_wrapper, args=(guess_var, obs_metsim_obj, var_names, fix_var, queue))

    # process.start()
    # process.join(timeout)  # Wait up to `timeout` seconds

    # if process.is_alive():
    #     process.terminate()  # Kill process if it's still running
    #     print("Timeout occurred")
    #     return -np.inf  # Return negative infinity if timed out

    # simulation_results = queue.get()  # Retrieve results from queue

    # if simulation_results is None:
    #     return -np.inf

    ### WINDOWS ### does not work...

    # find the time_arr index in simulation_results that are above the np.min(obs_metsim_obj.luminosity) and are after height_lum[0] (the leading_frag_height_arr[-1] is nan)
    indices_visible = np.where((simulation_results.luminosity_arr[:-1] > np.min(obs_metsim_obj.luminosity)) & (simulation_results.leading_frag_height_arr[:-1] < obs_metsim_obj.height_lum[0]))[0]
    # check if indices_visible is empty
    if len(indices_visible) == 0:
        return -np.inf
    real_time_visible = obs_metsim_obj.time_lum[-1]-obs_metsim_obj.time_lum[0]
    simulated_time_visible = simulation_results.time_arr[indices_visible][-1]-simulation_results.time_arr[indices_visible][0]
    # check if is too short and the time difference is smaller than 60% of the real time difference
    if simulated_time_visible < 0.6*real_time_visible:
        return -np.inf
    
    simulated_lc_intensity = np.interp(obs_metsim_obj.height_lum, 
                                       np.flip(simulation_results.leading_frag_height_arr), 
                                       np.flip(simulation_results.luminosity_arr))

    lag_sim = simulation_results.leading_frag_length_arr - (obs_metsim_obj.v_init * simulation_results.time_arr)

    simulated_lag = np.interp(obs_metsim_obj.height_lag, 
                              np.flip(simulation_results.leading_frag_height_arr), 
                              np.flip(lag_sim))

    lag_sim = simulated_lag - simulated_lag[0]

    ### Log Likelihood ###

    log_likelihood_lum = np.nansum(-0.5 * np.log(2*np.pi*obs_metsim_obj.noise_lum**2) - 0.5 / (obs_metsim_obj.noise_lum**2) * (obs_metsim_obj.luminosity - simulated_lc_intensity) ** 2)
    log_likelihood_lag = np.nansum(-0.5 * np.log(2*np.pi*obs_metsim_obj.noise_lag**2) - 0.5 / (obs_metsim_obj.noise_lag**2) * (obs_metsim_obj.lag - lag_sim) ** 2)

    log_likelihood_tot = log_likelihood_lum + log_likelihood_lag

    ### Chi Square ###

    # chi_square_lum = - 0.5/(obs_metsim_obj.noise_lum**2) * np.nansum((obs_metsim_obj.luminosity_arr - simulated_lc_intensity) ** 2)  # add the error
    # chi_square_lag = - 0.5/(obs_metsim_obj.noise_lag**2) * np.nansum((obs_metsim_obj.lag - lag_sim) ** 2)  # add the error

    # log_likelihood_tot = chi_square_lum + chi_square_lag

    return log_likelihood_tot


def prior_dynesty(cube,bounds,flags_dict):
    """
    Transform the unit cube to a uniform prior
    """
    x = np.array(cube)  # Copy u to avoid modifying it directly
    param_names = list(flags_dict.keys())
    i_prior=0
    for (min_or_sigma, MAX_or_mean), param_name in zip(bounds, param_names):
        # check if the flags_dict at index i is empty
        if 'norm' in flags_dict[param_name]:
            x[i_prior] = norm.ppf(cube[i_prior], loc=MAX_or_mean, scale=min_or_sigma)
        elif 'invgamma' in flags_dict[param_name]:
            x[i_prior] = invgamma.ppf(cube[i_prior], min_or_sigma, scale=MAX_or_mean * (min_or_sigma + 1))
        else:
            x[i_prior] = cube[i_prior] * (MAX_or_mean - min_or_sigma) + min_or_sigma  # Scale and shift
        i_prior += 1

    return x



def main_dynestsy(dynesty_file, obs_data, bounds, flags_dict, fixed_values, n_core=1, output_folder="", file_name=""):
    """
    Main function to run dynesty.
    """

    print("Starting dynesty run...")  
    # get variable names
    var_names = list(flags_dict.keys())
    # get the number of parameters
    ndim = len(var_names)
    print("Number of parameters:", ndim)

    # first chack if fix_var is not {}
    if fixed_values:
        var_names_fix = list(fixed_values.keys())
        # check if among the noise_lum and noise_lag there is a "noise_lum" and if there is a "noise_lag"
        if 'noise_lum' in var_names_fix:
            # if so, set the noise_lum to the fixed value
            obs_data.noise_lum = fixed_values['noise_lum']
            print("Fixed noise in luminosity to:", fixed_values['noise_lum'])
        if 'noise_lag' in var_names_fix:
            # if so, set the noise_lag to the fixed value
            obs_data.noise_lag = fixed_values['noise_lag']
            print("Fixed noise in lag to:", fixed_values['noise_lag'])

    # check if file exists
    if not os.path.exists(dynesty_file):
        print("Starting new run:")
        # Start new run
        with dynesty.pool.Pool(n_core, log_likelihood_dynesty, prior_dynesty,
                               logl_args=(obs_data, flags_dict, fixed_values, 10),
                               ptform_args=(bounds, flags_dict)) as pool:
            ### NEW RUN
            dsampler = dynesty.DynamicNestedSampler(pool.loglike, pool.prior_transform, ndim, pool = pool)
            dsampler.run_nested(print_progress=True, checkpoint_file=dynesty_file)

    else:
        print("Resuming previous run:")
        print('Warning: make sure the number of parameters and the bounds are the same as the previous run!')
        # Resume previous run
        with dynesty.pool.Pool(n_core, log_likelihood_dynesty, prior_dynesty,
                               logl_args=(obs_data, flags_dict, fixed_values, 10),
                               ptform_args=(bounds, flags_dict)) as pool:
            ### RESUME:
            dsampler = dynesty.DynamicNestedSampler.restore(dynesty_file, pool = pool)
            dsampler.run_nested(resume=True, print_progress=True, checkpoint_file=dynesty_file)

    print('SUCCESS: dynesty results ready!\n')

    # check if output_folder is different from the dynesty_file folder
    if output_folder != os.path.dirname(dynesty_file):
        print("Copying dynesty file to output folder...")
        shutil.copy(dynesty_file, output_folder)
        print("dynesty file copied to:", output_folder)
    
    # dsampler = dynesty.DynamicNestedSampler.restore(filename)
    plot_dynesty(dsampler.results, obs_data, flags_dict, fixed_values, output_folder, file_name)






###############################################################################
if __name__ == "__main__":

    import argparse

    ### COMMAND LINE ARGUMENTS
    arg_parser = argparse.ArgumentParser(description="Run dynesty with optional .prior file.")
    # r"C:\Users\maxiv\WMPG-repoMAX\Code\DynNestSampl\Shower\CAMO\ORI_mode\ORI_mode_CAMO_with_noise.json,C:\Users\maxiv\WMPG-repoMAX\Code\DynNestSampl\Shower\EMCCD\ORI_mode\ORI_mode_EMCCD_with_noise.json,C:\Users\maxiv\WMPG-repoMAX\Code\DynNestSampl\Shower\CAMO\CAP_mode\CAP_mode_CAMO_with_noise.json,C:\Users\maxiv\WMPG-repoMAX\Code\DynNestSampl\Shower\EMCCD\DRA_mode\DRA_mode_EMCCD_with_noise.json,C:\Users\maxiv\WMPG-repoMAX\Code\DynNestSampl\Shower\EMCCD\CAP_mode\CAP_mode_EMCCD_with_noise.json"
    # r"/home/mvovk/WMPG-repoMAX/Code/DynNestSampl/Shower/CAMO/ORI_mode/ORI_mode_CAMO_with_noise.json,/home/mvovk/WMPG-repoMAX/Code/DynNestSampl/Shower/EMCCD/ORI_mode/ORI_mode_EMCCD_with_noise.json,/home/mvovk/WMPG-repoMAX/Code/DynNestSampl/Shower/CAMO/CAP_mode/CAP_mode_CAMO_with_noise.json,/home/mvovk/WMPG-repoMAX/Code/DynNestSampl/Shower/EMCCD/CAP_mode/CAP_mode_EMCCD_with_noise.json,/home/mvovk/WMPG-repoMAX/Code/DynNestSampl/Shower/EMCCD/DRA_mode/DRA_mode_EMCCD_with_noise.json"
    arg_parser.add_argument('--input_dir', metavar='INPUT_PATH', type=str,
        default=r"/home/mvovk/WMPG-repoMAX/Code/DynNestSampl/Shower/EMCCD/ORI_mode/ORI_mode_EMCCD_with_noise.json,/home/mvovk/WMPG-repoMAX/Code/DynNestSampl/Shower/CAMO/ORI_mode/ORI_mode_CAMO_with_noise.json,/home/mvovk/WMPG-repoMAX/Code/DynNestSampl/Shower/EMCCD/CAP_mode/CAP_mode_EMCCD_with_noise.json,/home/mvovk/WMPG-repoMAX/Code/DynNestSampl/Shower/CAMO/CAP_mode/CAP_mode_CAMO_with_noise.json,/home/mvovk/WMPG-repoMAX/Code/DynNestSampl/Shower/EMCCD/DRA_mode/DRA_mode_EMCCD_with_noise.json",
        help="Path to walk and find .pickle file or specific single file .pickle or .json file divided by ',' in between.")
    # /home/mvovk/Results/Results_Nested/validation/
    arg_parser.add_argument('--output_dir', metavar='OUTPUT_DIR', type=str,
        default=r"/home/mvovk/Results/Results_Nested/validation/",
        help="Where to store results. If empty, store next to each .dynesty.")
    # /home/mvovk/WMPG-repoMAX/Code/DynNestSampl/stony_meteoroid.prior
    arg_parser.add_argument('--prior', metavar='PRIOR', type=str,
        default=r"/home/mvovk/WMPG-repoMAX/Code/DynNestSampl/stony_meteoroid.prior",
        help="Path to a .prior file. If blank, we look in the .dynesty folder or default to built-in bounds.")
    
    arg_parser.add_argument('--use_CAMO_data', metavar='USE_CAMO_DATA', type=bool, default=False,
        help="If True, use only CAMO data for lag if present in pickle file, or generate json file with CAMO noise. If False, do not use/generate CAMO data (by default is False).")

    arg_parser.add_argument('--resume', metavar='RESUME', type=bool, default=True,
        help="If True, resume from existing .dynesty if found. If False, create a new version.")
    
    arg_parser.add_argument('--only_plot', metavar='ONLY_PLOT', type=bool, default=False,
        help="If True, only plot the results of the dynesty run. If False, run dynesty.")

    arg_parser.add_argument('--cores', metavar='CORES', type=int, default=None,
        help="Number of cores to use. Default = all available.")

    # Parse
    cml_args = arg_parser.parse_args()

    # Optional: suppress warnings
    # warnings.filterwarnings('ignore')

    # If no core count given, use all
    if cml_args.cores is None:
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
        print(f"Processing {input_dirfile} look for all files...")

        # Use the class to find .dynesty, load prior, and decide output folders
        finder = find_dynestyfile_and_priors(
            input_dir_or_file=input_dirfile,
            prior_file=cml_args.prior,
            resume=cml_args.resume,
            output_dir=cml_args.output_dir,
            use_CAMO_data=cml_args.use_CAMO_data
        )

        # check if finder is empty
        if not finder.base_names:
            print("No files found in the input directory.")
            continue

        # Each discovered or created .dynesty is in input_folder_file
        # with its matching prior info
        for i, (base_name, dynesty_info, prior_path, out_folder) in enumerate(zip(
            finder.base_names,
            finder.input_folder_file,
            finder.priors,
            finder.output_folders
        )):
            dynesty_file, bounds, flags_dict, fixed_values = dynesty_info
            obs_data = finder.observation_instance()
            print("--------------------------------------------------")
            # check if a file with the name "log"+n_PC_in_PCA+"_"+str(len(df_sel))+"ev.txt" already exist
            if os.path.exists(out_folder+os.sep+"log_"+base_name+".txt"):
                # remove the file
                os.remove(out_folder+os.sep+"log_"+base_name+".txt")
            sys.stdout = Logger(out_folder,"log_"+base_name+".txt") # 
            print(f"Meteor:", base_name)
            print("  File name:    ", obs_data.file_name)
            print("  Dynesty file: ", dynesty_file)
            print("  Prior file:   ", prior_path)
            print("  Output folder:", out_folder)
            print("  Bounds:")
            param_names = list(flags_dict.keys())
            for (low_val, high_val), param_name in zip(bounds, param_names):
                print(f"    {param_name}: [{low_val}, {high_val}] flags={flags_dict[param_name]}")
            print("  Fixed Values: ", fixed_values)
            # Close the Logger to ensure everything is written to the file STOP COPY in TXT file
            sys.stdout.close()
            # Reset sys.stdout to its original value if needed
            sys.stdout = sys.__stdout__
            print("--------------------------------------------------")
            # Run the dynesty sampler
            os.makedirs(out_folder, exist_ok=True)
            plot_data_with_residuals_and_real(obs_data, output_folder=out_folder, file_name=base_name)

            # if prior_path is not in the output directory and is not "" then copy the prior_path to the output directory
            if prior_path != "":
                # check if there is a prior file with the same name in the output_folder
                prior_file_output = os.path.join(out_folder,os.path.basename(prior_path))
                if not os.path.exists(prior_file_output):
                    shutil.copy(prior_path, out_folder)
                    print("prior file copied to output folder:", prior_file_output)
            # check if obs_data.file_name is not in the output directory
            if not os.path.exists(os.path.join(out_folder,os.path.basename(obs_data.file_name))) and os.path.isfile(obs_data.file_name):
                shutil.copy(obs_data.file_name, out_folder)
                print("observation file copied to output folder:", os.path.join(out_folder,os.path.basename(obs_data.file_name)))
            elif not os.path.isfile(obs_data.file_name):
                print("original observation file not found, not copied:",obs_data.file_name)
            
            if not cml_args.only_plot:
                main_dynestsy(dynesty_file, obs_data, bounds, flags_dict, fixed_values, cml_args.cores, output_folder=out_folder, file_name=base_name)
            
            elif cml_args.only_plot and os.path.isfile(dynesty_file): 
                print("Only plotting requested. Skipping dynesty run.")
                dsampler = dynesty.DynamicNestedSampler.restore(dynesty_file)
                # dsampler = dynesty.DynamicNestedSampler.restore(dynesty_file)
                plot_dynesty(dsampler.results, obs_data, flags_dict, fixed_values, out_folder, base_name)

            else:
                print("Fail to generate dynesty plots, dynasty file not found:",dynesty_file)
                print("If you want to run the dynasty file set only_plot to False")
