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
import datetime
import re
import matplotlib.gridspec as gridspec
from scipy.optimize import minimize
from scipy.stats import norm, invgamma
import shutil
import matplotlib.ticker as ticker
import multiprocessing
import math

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


# 100^((m-6.96)/5) =(114007.25/100000)^2
# math.log((114007.25/100000)^2, 100)*5+6.96
def meteor_abs_magnitude_to_apparent(abs_mag, distance):
    apparent_mag = []
    # check if it is an array
    if isinstance(abs_mag, np.ndarray):
        for ii in range(len(abs_mag)):
            apparent_mag.append(math.log((distance[ii]/100000)**2, 100)*5+abs_mag[ii])
    else: 
        apparent_mag = math.log((distance/100000)**2, 100)*5+abs_mag
    return apparent_mag

###############################################################################
# Function: plotting function
###############################################################################

# Plotting function
def plot_data_with_residuals_and_real(obs_data, sim_data=None, output_folder='',file_name='', color_sim='black', label_sim='Best guess'):
    ''' Plot the data with residuals and real data '''

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
    # print('testing unique stations plot',np.unique(obs_data.stations_lag), np.unique(obs_data.stations_lum))
    if not np.array_equal(np.unique(obs_data.stations_lag), np.unique(obs_data.stations_lum)):
        # take the one that are not in the other in lag
        stations_lag = np.setdiff1d(np.unique(obs_data.stations_lag), np.unique(obs_data.stations_lum))
        if len(stations_lag) != 0:
            # take the one that are shared between lag and lum
            # stations_lag = np.intersect1d(np.unique(obs_data.stations_lag), np.unique(obs_data.stations_lum))
            # print('stations_lag',stations_lag)
            # Suppose stations_lag is your array of station IDs you care about
            mask = np.isin(obs_data.stations_lag, stations_lag)
            # Filter heights for only those stations
            filtered_heights = obs_data.height_lag[mask]
            # Get the maximum of that subset
            max_height_lag = filtered_heights.max()
            # print a horizonal along the x axis at the height_lag[0] darkgray
            ax0.axhline(y=max_height_lag/1000, color='gray', linestyle='-.', linewidth=1, label=f"{', '.join(stations_lag)}", zorder=2)

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
        if len(stations_lag) != 0:
            # print a horizonal along the x axis at the height_lag[0] darkgray
            ax4.axhline(y=max_height_lag/1000, color='gray', linestyle='-.', linewidth=1, label=f"{', '.join(stations_lag)}", zorder=2)
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
              - ((obs_data.v_init)*(obs_data.time_arr-obs_data.time_arr[index]))
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
        ax0.plot(sim_data.abs_magnitude, sim_data.leading_frag_height_arr/1000, color=color_sim, label=label_sim)
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
            ax1.plot(sim_diff_mag[np.where(obs_data.stations_lum == station)], \
                    obs_data.height_lum[np.where(obs_data.stations_lum == station)]/1000, '.', \
                    color=station_colors[station], label=station)

        ax4.plot(sim_data.luminosity_arr, sim_data.leading_frag_height_arr/1000, color=color_sim, label=label_sim) 

        # interpolate to make sure they are the same length and discard points after height starts increasing if it does at any point obs_metsim_obj.traj.observations[0].model_ht
        sim_lum = np.interp(obs_data.height_lum, 
                                        np.flip(sim_data.leading_frag_height_arr), 
                                        np.flip(sim_data.luminosity_arr))
        
        # make the difference between the no_noise_intensity and the obs_data.luminosity_arr
        sim_diff_lum = sim_lum - obs_data.luminosity
        # for each station in obs_data_plot
        for station in np.unique(obs_data.stations_lum):
            # plot the height vs. absolute_magnitudes
            ax5.plot(sim_diff_lum[np.where(obs_data.stations_lum == station)], \
                    obs_data.height_lum[np.where(obs_data.stations_lum == station)]/1000, '.', \
                    color=station_colors[station], label=station)

        # find the obs_data.leading_frag_height_arr index is close to obs_data.height_lum[0] wihouth nan
        index = np.argmin(np.abs(sim_data.leading_frag_height_arr[~np.isnan(sim_data.leading_frag_height_arr)]-obs_data.height_lag[0]))
        # plot velocity_arr vs leading_frag_time_arr
        ax2.plot(sim_data.time_arr-sim_data.time_arr[index], sim_data.leading_frag_vel_arr/1000, color=color_sim, label=label_sim)
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
              - ((obs_data.v_init)*(sim_data.time_arr-sim_data.time_arr[index]))
        
        sim_lag -= sim_lag[index]
        # plot lag_arr vs leading_frag_time_arr
        ax3.plot(sim_data.time_arr-sim_data.time_arr[index], sim_lag, color=color_sim, label=label_sim)

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

    # # Save the plot
    # print('file saved: '+output_folder +os.sep+ file_name+'_best_fit_plot.png')
    # fig.savefig(output_folder +os.sep+ file_name +'_best_fit_plot.png', dpi=300)

    # Save the plot
    print('file saved: '+output_folder +os.sep+ file_name+'_LumLag_plot.png')
    fig.savefig(output_folder +os.sep+ file_name +'_LumLag_plot.png', dpi=300)

    # Display the plot
    plt.close(fig)


# Plotting function dynesty
def plot_dynesty(dynesty_run_results, obs_data, flags_dict, fixed_values, output_folder='', file_name='', log_file=''):
    ''' Plot the dynesty results '''

    if log_file == '':
        log_file = os.path.join(output_folder, f"log_{file_name}.txt")

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
    all_samples = dynesty.utils.resample_equal(dynesty_run_results.samples, weights)

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
            all_samples[:, i] = 10**(all_samples[:, i])
            best_guess[i] = 10**(best_guess[i])
            labels_plot[i] =r"$\log_{10}$(" +labels_plot[i]+")"
        # check variable is 'v_init' or 'erosion_height_start' divide by 1000
        if variable == 'v_init' or variable == 'erosion_height_start' or variable == 'erosion_height_change':
            samples_equal[:, i] = samples_equal[:, i] / 1000
        # check variable is 'erosion_coeff' or 'sigma' divide by 1e6
        if variable == 'erosion_coeff' or variable == 'sigma' or variable == 'erosion_coeff_change' or variable == 'erosion_sigma_change':
            samples_equal[:, i] = samples_equal[:, i] * 1e6

    constjson_bestfit = Constants()
    # change Constants that have the same variable names and the one fixed
    for variable in variables:
        if variable in constjson_bestfit.__dict__.keys():
            constjson_bestfit.__dict__[variable] = best_guess[variables.index(variable)]

    # do te same for the fixed values
    for variable in fixed_values.keys():
        if variable in constjson_bestfit.__dict__.keys():
            constjson_bestfit.__dict__[variable] = fixed_values[variable]

    constjson_bestfit.__dict__['P_0m'] = obs_data.P_0m
    constjson_bestfit.__dict__['lum_eff_type'] = obs_data.lum_eff_type
    constjson_bestfit.__dict__['dens_co'] = obs_data.dens_co
    constjson_bestfit.__dict__['dt'] = obs_data.dt
    constjson_bestfit.__dict__['h_kill'] = obs_data.h_kill

    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif obj is None:
            return None
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj

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

    # create a folder to save the fit plots
    if not os.path.exists(output_folder +os.sep+ 'fit_plots'):
        os.makedirs(output_folder +os.sep+ 'fit_plots')

    best_guess_obj_plot = run_simulation(best_guess, obs_data, variables, fixed_values)

    # Plot the data with residuals and the best fit
    plot_data_with_residuals_and_real(obs_data, best_guess_obj_plot, output_folder +os.sep+ 'fit_plots', file_name + "_best_fit")

    # Prepare dictionary and convert all values to serializable format
    constjson_bestfit_dict = {key: convert_to_serializable(value) for key, value in constjson_bestfit.__dict__.items()}

    # Save as JSON
    output_path = os.path.join(output_folder +os.sep+ 'fit_plots', file_name + '_sim_fit_dynesty_BestGuess.json')
    with open(output_path, 'w') as f:
        json.dump(constjson_bestfit_dict, f, indent=4)

    ### TABLE OF POSTERIOR SUMMARY STATISTICS ###

    # Posterior mean (per dimension)
    posterior_mean = np.mean(samples_equal, axis=0)      # shape (ndim,)
    all_samples_mean = np.mean(all_samples, axis=0)      # shape (ndim,)

    # Posterior median (per dimension)
    posterior_median = np.median(samples_equal, axis=0)  # shape (ndim,)
    all_samples_median = np.median(all_samples, axis=0)  # shape (ndim,)

    # 95% credible intervals (2.5th and 97.5th percentiles)
    lower_95 = np.percentile(samples_equal, 2.5, axis=0)   # shape (ndim,)
    upper_95 = np.percentile(samples_equal, 97.5, axis=0)  # shape (ndim,)

    # Function to approximate mode using histogram binning
    def approximate_mode_1d(samples):
        hist, bin_edges = np.histogram(samples, bins='auto', density=True)
        idx_max = np.argmax(hist)
        return 0.5 * (bin_edges[idx_max] + bin_edges[idx_max + 1])

    approx_modes = [approximate_mode_1d(samples_equal[:, d]) for d in range(ndim)]
    approx_modes_all = [approximate_mode_1d(all_samples[:, d]) for d in range(ndim)]

    ### MODE PLOT and json ###

    approx_mode_obj_plot = run_simulation(approx_modes_all, obs_data, variables, fixed_values)

    for variable in variables:
        if variable in constjson_bestfit.__dict__.keys():
            constjson_bestfit.__dict__[variable] = approx_modes_all[variables.index(variable)]

    # Prepare dictionary and convert all values to serializable format
    constjson_mode_dict = {key: convert_to_serializable(value) for key, value in constjson_bestfit.__dict__.items()}

    # Save as JSON
    with open(os.path.join(output_folder +os.sep+ 'fit_plots', file_name + '_sim_fit_dynesty_mode.json'), 'w') as f:
        json.dump(constjson_mode_dict, f, indent=4)

    ### MEAN PLOT and json ###

    # Plot the data with residuals and the best fit
    plot_data_with_residuals_and_real(obs_data, approx_mode_obj_plot, output_folder +os.sep+ 'fit_plots', file_name+'_mode','red', 'Mode')

    mean_obj_plot = run_simulation(all_samples_mean, obs_data, variables, fixed_values)

    for variable in variables:
        if variable in constjson_bestfit.__dict__.keys():
            constjson_bestfit.__dict__[variable] = all_samples_mean[variables.index(variable)]

    # Prepare dictionary and convert all values to serializable format
    constjson_mean_dict = {key: convert_to_serializable(value) for key, value in constjson_bestfit.__dict__.items()}

    # Save as JSON
    with open(os.path.join(output_folder +os.sep+ 'fit_plots', file_name + '_sim_fit_dynesty_mean.json'), 'w') as f:
        json.dump(constjson_mean_dict, f, indent=4)

    ### MEDIAN PLOT and json ###

    # Plot the data with residuals and the best fit
    plot_data_with_residuals_and_real(obs_data, mean_obj_plot, output_folder +os.sep+ 'fit_plots', file_name+'_mean','blue', 'Mean')

    median_obj_plot = run_simulation(all_samples_median, obs_data, variables, fixed_values)

    # Plot the data with residuals and the best fit
    plot_data_with_residuals_and_real(obs_data, median_obj_plot, output_folder +os.sep+ 'fit_plots', file_name+'_median','cornflowerblue', 'Median')

    for variable in variables:
        if variable in constjson_bestfit.__dict__.keys():
            constjson_bestfit.__dict__[variable] = all_samples_median[variables.index(variable)]

    # Prepare dictionary and convert all values to serializable format
    constjson_median_dict = {key: convert_to_serializable(value) for key, value in constjson_bestfit.__dict__.items()}

    # Save as JSON
    with open(os.path.join(output_folder +os.sep+ 'fit_plots', file_name + '_sim_fit_dynesty_median.json'), 'w') as f:
        json.dump(constjson_median_dict, f, indent=4)

    truth_values_plot = {}
    # if 'dynesty_run_results has const
    if hasattr(obs_data, 'const'):

        truth_values_plot = {}

        # Extract values from dictionary
        for variable in variables:
            if variable in obs_data.const:  # Use dictionary lookup instead of hasattr()
                truth_values_plot[variable] = obs_data.const[variable]
            else:
                print(f"Warning: {variable} not found in obs_data.const")

        # if 'noise_lag' take it from obs_data.noise_lag
        if 'noise_lag' in flags_dict.keys():
            truth_values_plot['noise_lag'] = obs_data.noise_lag
        # if 'noise_mag' take it from obs_data.noise_mag
        if 'noise_lum' in flags_dict.keys():
            truth_values_plot['noise_lum'] = obs_data.noise_lum

        # Convert to array safely
        truths = np.array([truth_values_plot.get(variable, np.nan) for variable in variables])

        # Apply log10 safely if needed
        for variable in variables:
            if 'log' in flags_dict.get(variable, []):
                if variable in truth_values_plot:
                    truth_values_plot[variable] = np.log10(truth_values_plot[variable]) #np.log10(np.maximum(truth_values_plot[variable], 1e-10))
                else:
                    print(f"Skipping {variable}: Missing from truth_values_plot")

        for i, variable in enumerate(variables): 
            # check variable is 'v_init' or 'erosion_height_start' divide by 1000
            if variable == 'v_init' or variable == 'erosion_height_start' or variable == 'erosion_height_change':
                truths[i] = truths[i] / 1000
            # check variable is 'erosion_coeff' or 'sigma' divide by 1e6
            if variable == 'erosion_coeff' or variable == 'sigma' or variable == 'erosion_coeff_change' or variable == 'erosion_sigma_change':
                truths[i] = truths[i] * 1e6

        # Compare to true theta
        # bias = posterior_mean - truths
        bias = approx_modes - truths
        abs_error = np.abs(bias)
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
    Parameter & 2.5CI & True Value & Mode & Mean & Median & 97.5CI & Abs.Error & Rel.Error\% & Cover \\
    \hline
        """
        # & Mode
        # {approx_modes[i]:.4g} &
        for i, label in enumerate(labels):
            coverage_val = "\ding{51}" if coverage_mask[i] else "\ding{55}"  # Use checkmark/x for coverage
            latex_str += (f"    {label} & {lower_95[i]:.4g} & {truths[i]:.4g} & {approx_modes[i]:.4g}"
                        f"& {posterior_mean[i]:.4g} & {posterior_median[i]:.4g} & {upper_95[i]:.4g} "
                        f"& {abs_error[i]:.4g} & {rel_error[i]:.4g}\% & {coverage_val} \\\\\n    \hline\n")

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
    \caption{"""

    # check if the file_name has a _ in it if so put \ before it
    if '_' in file_name:
        file_name_caption = file_name.replace('_', '\_')
    else:
        file_name_caption = file_name

    if hasattr(obs_data, 'const'):
        latex_str += f"Posterior summary statistics for {file_name_caption} simulation. Absolute and relative errors are calculated based on the mode. The Cover column indicates whether the true value lies within the 95\% CI."
    else:
        latex_str += f"Posterior summary statistics for {file_name_caption} meteor."
    latex_str += r"""}
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
        f.write(latex_str)
        f.close()

    # add to the log_file
    with open(log_file, "a") as f:
        f.write('\n'+summary_str)
        f.write("H info.gain: "+str(np.round(dynesty_run_results.information[-1],3))+'\n')
        f.write("niter i.e number of metsim simulated events\n")
        f.write("ncall i.e. number of likelihood evaluations\n")
        f.write("eff(%) i.e. (niter/ncall)*100 eff. of the logL call \n")
        f.write("logz i.e. final estimated evidence\n")
        f.write("H info.gain i.e. big H very small peak posterior, low H broad posterior distribution no need or a lot of live points\n")
        f.write("\nBest fit:\n")
        for i in range(len(best_guess)):
            f.write(variables[i]+':\t'+str(best_guess[i])+'\n')
        f.write('\nBest fit logL: '+str(dynesty_run_results.logl[sim_num])+'\n')
        if diff_logL is not None:
            f.write('REAL logL: '+str(real_logL)+'\n')
            f.write('Diff logL (Best fit - REAL): '+str(diff_logL)+'\n')
            f.write('Rel.Error % diff logL: '+str(abs(diff_logL/real_logL)*100)+'%\n')
            f.write('\nCoverage mask per dimension: '+str(coverage_mask)+'\n')
            f.write('Fraction of dimensions covered: '+str(coverage_mask.mean())+'\n')

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
    ''' Read the prior file and generate the bounds for the dynesty sampler '''
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
        # check if object_meteor.const.rho_grain exist
        if hasattr(object_meteor.const, 'rho_grain'):
            rho_grain_real = object_meteor.const.rho_grain
        else:
            if "rho_grain" in object_meteor.const:
                rho_grain_real = object_meteor.const["rho_grain"]
            # chack if object_meteor.const.rho_grain exist as key
            if "rho_grain" in object_meteor.const.keys():
                rho_grain_real = object_meteor.const["rho_grain"]

    # Default values if no file path is provided
    if file_path == "":
        print("No prior file provided. Using default bounds.")
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
                bounds[i][0] = object_meteor.__dict__[key] - 10**int(np.floor(np.log10(abs(object_meteor.__dict__[key])))) #object_meteor.__dict__[key]/10/2

            if np.isnan(bounds[i][1]) and key in object_meteor.__dict__ and "norm" in flags_dict[key]:
                bounds[i][1] = object_meteor.__dict__[key]
            elif np.isnan(bounds[i][1]) and key in object_meteor.__dict__:
                bounds[i][1] = object_meteor.__dict__[key] + 10**int(np.floor(np.log10(abs(object_meteor.__dict__[key])))) #object_meteor.__dict__[key]/10/2
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
                min_val = safe_eval(min_val) if isinstance(min_val, str) and min_val.lower() != "nan" else default_bounds.get(name, (np.nan, np.nan))[0]
                max_val = safe_eval(max_val) if isinstance(max_val, str) and max_val.lower() != "nan" else default_bounds.get(name, (np.nan, np.nan))[1]

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
                        # min_val = object_meteor.__dict__[name] - 10**int(np.floor(np.log10(abs(object_meteor.__dict__[name]))))#object_meteor.__dict__[name]/10/2
                        min_val = 10**int(np.floor(np.log10(abs(object_meteor.__dict__[name]))) - 1)#object_meteor.__dict__[name]/10/2

                if np.isnan(max_val) and name in object_meteor.__dict__ and ("norm" in flags or "invgamma" in flags):
                    max_val = object_meteor.__dict__[name]
                if np.isnan(max_val) and name in object_meteor.__dict__:
                    # if norm in default_flags[name] then divide by 10
                    if "norm" in default_flags[name] or "invgamma" in default_flags[name]:
                        max_val = object_meteor.__dict__[name] + default_bounds.get(name, (np.nan, np.nan))[0]
                    else:
                        # max_val = object_meteor.__dict__[name] + 10**int(np.floor(np.log10(abs(object_meteor.__dict__[name]))))#object_meteor.__dict__[name]/10/2
                        max_val = 2 * 10**int(np.floor(np.log10(abs(object_meteor.__dict__[name]))) + 1)#object_meteor.__dict__[name]/10/2
                                
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
# Load observation data and create an object
###############################################################################
class observation_data:
    ''' class to load the observation data and create an object '''
    def __init__(self, obs_file_path,use_all_cameras=False, lag_noise_prior=40, lum_noise_prior=2.5):
        self.noise_lag = lag_noise_prior
        self.noise_lum = lum_noise_prior
        self.file_name = obs_file_path

        # check obs_file_path is a list if so take the first element
        if isinstance(obs_file_path, list):
            obs_file_path = obs_file_path[0]

        # check if the file is a json file
        if obs_file_path.endswith('.pickle'):
            self.load_pickle_data(use_all_cameras)
        elif obs_file_path.endswith('.json'):
            self.load_json_data(use_all_cameras)
        else:
            # file type not supported
            raise ValueError("File type not supported, only .json and .pickle files are supported")


    def load_pickle_data(self, use_all_cameras=False):
        """
        Load the pickle file(s) and create a dictionary keyed by each file name.
        Each files data (e.g., list of station IDs, dens_co, zenith_angle, etc.)
        goes into a sub-dict.
        """
        print('Loading pickle file(s):', self.file_name)
        
        # Top-level dictionary.
        combined_obs_dict = {}
        const = Constants()
        # check if it is not an array
        if not isinstance(self.file_name, list):
            self.file_name = [self.file_name]     

        obs_dict = []  # accumulates everything from all files   
        # Loop over each pickle file
        for current_file_name in self.file_name:
            traj = loadPickle(*os.path.split(current_file_name))

            # Skip if there's no .orbit attribute
            if not hasattr(traj, 'orbit'):
                print(f"Trajectory data not found in file: {current_file_name}")
                continue

            # self.v_init=traj.orbit.v_init+100
            # self.stations = []

            obs_data_dict = []
            for obs in traj.observations:
                # print('Station:', obs.station_id)

                # check if among obs.station_id there is one of the following 01T or 02T
                if "1T" in obs.station_id or "2T" in obs.station_id:
                    P_0m = 840
                elif "1K" in obs.station_id or "2K" in obs.station_id:
                    P_0m = 840
                elif "1G" in obs.station_id or "2G" in obs.station_id or "1F" in obs.station_id or "2F" in obs.station_id:
                    P_0m = 935
                else:
                    print(obs.station_id,'Station uknown\nMake sure the station is either EMCCD or CAMO (i.e. contains in the name 1T, 2T, 1K, 2K, 1G, 2G, 1F, 2F)')
                    continue

                # check if obs.absolute_magnitudes is a 'NoneType' object
                if obs.absolute_magnitudes is None:
                    # create an array with the same length as obs.model_ht and fill it with 15
                    obs.absolute_magnitudes = np.array([15.5]*len(obs.model_ht))

                obs_data_camera = {
                    # make an array that is long as len(obs.model_ht) and has only obs.station_id
                    'flag_station': np.array([obs.station_id]*len(obs.model_ht)),
                    'flag_file': np.array([current_file_name]*len(obs.model_ht)),
                    'height': np.array(obs.model_ht), # m
                    'absolute_magnitudes': np.array(obs.absolute_magnitudes),
                    'luminosity': np.array(P_0m*(10 ** (obs.absolute_magnitudes/(-2.5)))), # const.P_0m)
                    'time': np.array(obs.time_data), # s
                    'ignore_list': np.array(obs.ignore_list),
                    'velocities': np.array(obs.velocities), # m/s
                    'lag': np.array(obs.lag), # m
                    'length': np.array(obs.state_vect_dist), # m
                    'time_lag': np.array(obs.time_data), # s
                    'height_lag': np.array(obs.model_ht), # m
                    'apparent_magnitudes': np.array(meteor_abs_magnitude_to_apparent(np.array(obs.absolute_magnitudes), np.array(obs.meas_range))) # model_range
                    }
                obs_data_camera['velocities'][0] = obs.v_init
                obs_data_dict.append(obs_data_camera)
            
                obs_dict.extend(obs_data_dict)  # Add this file's data to the big list
            
        # ceck if obs_dict is empty
        if len(obs_dict) == 0:
            print('No valid station data found')
            return

        # Combine obs1 and obs2
        for key in obs_dict[0].keys():
            combined_obs_dict[key] = np.concatenate([obs[key] for obs in obs_dict])

        sorted_indices = np.argsort(combined_obs_dict['time'])
        for key in obs_dict[0].keys():
            combined_obs_dict[key] = combined_obs_dict[key][sorted_indices]

        # take all the unique values of the flag_station
        unique_stations = np.unique(combined_obs_dict['flag_station'])
        # print('Unique stations:', unique_stations)

        # check if among the unique_stations there is one of the following 01T or 02T
        if use_all_cameras==False:
            # check if among unique_stations there is one of the following 01T or 02T  
            if any(("1T" in station) or ("2T" in station) for station in unique_stations):
                # find the name of the camera that has 1T or 2T
                camera_name_lag = [camera for camera in unique_stations if "1T" in camera or "2T" in camera]
                lag_data, lag_files = self.extract_lag_data(combined_obs_dict, camera_name_lag)
                self.fps = 80
            elif any(("1G" in station) or ("2G" in station) or ("1F" in station) or ("2F" in station) for station in unique_stations):
                # find the name of the camera that has 1G or 2G or 1F or 2F
                camera_name_lag = [camera for camera in unique_stations if "1G" in camera or "2G" in camera or "1F" in camera or "2F" in camera]
                lag_data, lag_files = self.extract_lag_data(combined_obs_dict, camera_name_lag)
                self.fps = 32
            elif any(("1K" in station) or ("2K" in station) for station in unique_stations):
                # find the name of the camera that has 1K or 2K
                camera_name_lag = [camera for camera in unique_stations if "1K" in camera or "2K" in camera]
                lag_data, lag_files = self.extract_lag_data(combined_obs_dict, camera_name_lag)
                self.fps = 80
            else:
                # print the unique_stations
                print(unique_stations,'no known camera found')
                return
            
            if any(("1G" in station) or ("2G" in station) or ("1F" in station) or ("2F" in station) for station in unique_stations):
                camera_name_lag = [camera for camera in unique_stations if "1G" in camera or "2G" in camera or "1F" in camera or "2F" in camera]
                lum_data, lum_files = self.extract_lum_data(combined_obs_dict, camera_name_lag)
                self.P_0m = 935
            elif any(("1K" in station) or ("2K" in station) or ("1T" in station) or ("2T" in station) for station in unique_stations):
                camera_name_lag = [camera for camera in unique_stations if "1K" in camera or "2K" in camera or "1T" in camera or "2T" in camera]
                lum_data, lum_files = self.extract_lum_data(combined_obs_dict, camera_name_lag)
                self.P_0m = 840
            else:
                # print the unique_stations
                print(unique_stations,'no known camera found')
                return
        else:
            lag_data = combined_obs_dict
            lum_data = combined_obs_dict
            lum_files = self.file_name
            lag_files = self.file_name

            # if it is a list of files consider a warning
            if len(lag_files) > 1:
                print('WARNING: Multiple files detected. Using all cameras for lag, the recorded data might have different starting time.')

            if any(("1K" in station) or ("2K" in station) or ("1T" in station) or ("2T" in station) for station in unique_stations):
                self.P_0m = 840
                self.fps = 80
            elif any(("1G" in station) or ("2G" in station) or ("1F" in station) or ("2F" in station) for station in unique_stations):
                self.P_0m = 935
                self.fps = 32
            else:
                print(unique_stations,'no known camera found')
                return
            
        # for the lum data delete all the keys that have values above 8
        if np.any(lum_data['absolute_magnitudes'] > 8):
            print(obs.station_id,'Found values below 8 absolute magnitudes :', lum_data['absolute_magnitudes'][lum_data['absolute_magnitudes'] > 8])
            # delete any values above 8 absolute_magnitudes and delete the corresponding values in the other arrays
            lum_data = {key: lum_data[key][lum_data['absolute_magnitudes'] < 8] for key in lum_data.keys()}

        # # print all the keys in the lag_data
        # print('Keys in lag_data:',lag_data.keys())
        # # print all the keys in the lum_data
        # print('Keys in lum_data:',lum_data.keys())
        # print(lum_files)
        # print(lag_files)
        
        self.lum_files = lum_files
        self.lag_files = lag_files

        # put all the lag_data in the object
        self.velocities = lag_data['velocities']
        self.lag = lag_data['lag']
        self.length = lag_data['length']
        self.height_lag = lag_data['height']
        self.time_lag = lag_data['time']
        self.stations_lag = lag_data['flag_station']
        # put all the lum_data in the object
        self.height_lum = lum_data['height']
        self.absolute_magnitudes = lum_data['absolute_magnitudes']
        self.luminosity = lum_data['luminosity']
        self.time_lum = lum_data['time']
        self.stations_lum = lum_data['flag_station']
        self.apparent_magnitudes = lum_data['apparent_magnitudes']

        # for lag measurements start from 0 for length and time
        self.length = self.length-self.length[0]
        self.time_lag = self.time_lag-self.time_lag[0]
        self.time_lum = self.time_lum-self.time_lum[0]

        v_init_list = []
        for curr_lag_file in lag_files:
            # take the v_init from the trajectory file
            traj=loadPickle(*os.path.split(curr_lag_file))
            # get the trajectory
            # v_avg = traj.v_avg
            v_init_list.append(traj.orbit.v_init+100)
        # do the mean of the v_init_list
        self.v_init = np.mean(v_init_list)

        # usually noise_lum = 2.5
        if np.isnan(self.noise_lum):
            self.noise_lum = self.define_SNR_lum_noise()
            print('Assumed Noise in luminosity based on SNR:',self.noise_lum)
        self.noise_mag = 0.1
        
        # usually noise_lag = 40 or 5 for CAMO
        if np.isnan(self.noise_lag):
            self.noise_lag = self.define_polyn_fit_lag_noise()
            print('Assumed Noise in lag based on polynomial fit:',self.noise_lag)
        self.noise_vel = self.noise_lag*np.sqrt(2)/(1.0/self.fps)

        zenith_angle_list = []
        m_init_list = []
        for curr_lum_file in lum_files:
            # take the m_init_list from the trajectory file
            traj=loadPickle(*os.path.split(curr_lum_file))

            # now find the zenith angle mass v_init and dens_co
            dens_fit_ht_beg = 180000
            dens_fit_ht_end = traj.rend_ele - 5000
            if dens_fit_ht_end < 14000:
                dens_fit_ht_end = 14000

            lat_mean = np.mean([traj.rbeg_lat, traj.rend_lat])
            lon_mean = meanAngle([traj.rbeg_lon, traj.rend_lon])
            jd_dat=traj.jdt_ref

            # Fit the polynomail describing the density
            self.dens_co = fitAtmPoly(lat_mean, lon_mean, dens_fit_ht_end, dens_fit_ht_beg, jd_dat)

            zenith_angle_list.append(zenithAngleAtSimulationBegin(const.h_init, traj.rbeg_ele, traj.orbit.zc, const.r_earth))

            time_mag_arr = []
            avg_t_diff_max = 0
            # take only the stations that are unique in the stations_lum
            lum_stations = np.unique(self.stations_lum)
            # Extract time vs. magnitudes from the trajectory pickle file
            for obs in traj.observations:

                # check if the station_id is not in the lum_stations and continue
                if obs.station_id not in lum_stations:
                    continue

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

            # check if not enough values to unpack (expected 2, got 0)
            if len(time_mag_arr) == 0:
                print('No valid luminosity data found')
                continue

            time_arr, mag_arr = time_mag_arr.T

            # Average out the magnitudes
            time_arr, mag_arr = mergeClosePoints(time_arr, mag_arr, avg_t_diff_max, method='avg')

            # # Calculate the radiated energy
            # radiated_energy = calcRadiatedEnergy(np.array(time_arr), np.array(mag_arr), P_0m=self.P_0m)

            # Compute the photometric mass
            photom_mass = calcMass(np.array(time_arr), np.array(mag_arr), traj.orbit.v_avg, tau=0.7/100, P_0m=self.P_0m)

            m_init_list.append(photom_mass)

        # do the mean of the zenith_angle_list and m_init_list
        self.zenith_angle = np.mean(zenith_angle_list)
        self.m_init = np.mean(m_init_list)
                    
    def extract_lag_data(self, combined_obs_dict, camera_name_lag):
        lag_dict = []
        combined_lag_dict = {}
        lag_files = []
        # now for each of the camera_name save it in the lag_data
        for camera in camera_name_lag:
            lag_file=np.unique(combined_obs_dict['flag_file'][combined_obs_dict['flag_station'] == camera])[0]
            # the velocities, lag, height, time, flag_station and the unique flag_file
            lag_data = {
                'velocities': combined_obs_dict['velocities'][combined_obs_dict['flag_station'] == camera],
                'lag': combined_obs_dict['lag'][combined_obs_dict['flag_station'] == camera],
                'length': combined_obs_dict['length'][combined_obs_dict['flag_station'] == camera],
                'height': combined_obs_dict['height_lag'][combined_obs_dict['flag_station'] == camera],
                'time': combined_obs_dict['time_lag'][combined_obs_dict['flag_station'] == camera],
                'flag_station': combined_obs_dict['flag_station'][combined_obs_dict['flag_station'] == camera],
                'ignore_list': combined_obs_dict['ignore_list'][combined_obs_dict['flag_station'] == camera]
            }
            # Create a mask of all rows which have ignore_list == 0
            ignore_mask = (lag_data['ignore_list'] == 0)

            # Now rebuild lag_data so that each array is filtered by ignore_mask
            lag_data = {key: lag_data[key][ignore_mask] for key in lag_data.keys()}

            lag_dict.append(lag_data)
            lag_files.append(lag_file)

        # sort the indices by time in lag_dict
        for key in lag_dict[0].keys():
            combined_lag_dict[key] = np.concatenate([obs[key] for obs in lag_dict])

        sorted_indices = np.argsort(combined_lag_dict['time'])
        for key in lag_dict[0].keys():
            combined_lag_dict[key] = combined_lag_dict[key][sorted_indices]

        return combined_lag_dict, lag_files
    
    def extract_lum_data(self, combined_obs_dict, camera_name_lum):
        # consider that has to have height_lum absolute_magnitudes luminosity time_lum stations_lum apparent_magnitudes
        lum_dict = []
        combined_lum_dict = {}
        lum_files = []
        # now for each of the camera_name save it in the lum_data
        for camera in camera_name_lum:
            lum_file=np.unique(combined_obs_dict['flag_file'][combined_obs_dict['flag_station'] == camera])[0]
            # the velocities, lag, height, time, flag_station and the unique flag_file
            lum_data = {
                'height': combined_obs_dict['height'][combined_obs_dict['flag_station'] == camera],
                'absolute_magnitudes': combined_obs_dict['absolute_magnitudes'][combined_obs_dict['flag_station'] == camera],
                'luminosity': combined_obs_dict['luminosity'][combined_obs_dict['flag_station'] == camera],
                'time': combined_obs_dict['time'][combined_obs_dict['flag_station'] == camera],
                'flag_station': combined_obs_dict['flag_station'][combined_obs_dict['flag_station'] == camera],
                'apparent_magnitudes': combined_obs_dict['apparent_magnitudes'][combined_obs_dict['flag_station'] == camera]
            }
            lum_dict.append(lum_data)
            lum_files.append(lum_file)
        # sort the indices by time in lum_dict
        for key in lum_dict[0].keys():
            combined_lum_dict[key] = np.concatenate([obs[key] for obs in lum_dict])

        sorted_indices = np.argsort(combined_lum_dict['time'])
        for key in lum_dict[0].keys():
            combined_lum_dict[key] = combined_lum_dict[key][sorted_indices]

        return combined_lum_dict, lum_files
        
    def define_polyn_fit_lag_noise(self):
        ''' Define the lag fit noise '''
        # Fit a polynomial to the lag data

        def polyn_lag(t, a, b, c, t0):
            """
            Quadratic lag function.
            """

            # Only take times <= t0
            t_before = t[t <= t0]

            # Only take times > t0
            t_after = t[t > t0]

            # Compute the lag linearly before t0
            l_before = np.zeros_like(t_before) # +c

            # Compute the lag quadratically after t0
            l_after = -abs(a)*(t_after - t0)**3 - abs(b)*(t_after - t0)**2

            c = 0

            lag_funct = np.concatenate((l_before, l_after))

            lag_funct = lag_funct - lag_funct[0]

            return lag_funct
        
        def lag_residual_polyn(params, t_time, l_data):
            """
            Residual function for the optimization.
            """

            return np.sum((l_data - polyn_lag(t_time, *params))**2)

        # initial guess of deceleration decel equal to linear fit of velocity
        p0 = [np.mean(self.lag), 0, 0, np.mean(self.time_lag)]

        opt_res = minimize(lag_residual_polyn, p0, args=(np.array(self.time_lag), np.array(self.lag)), method='Nelder-Mead')

        # sample the fit for the velocity and acceleration
        a_t0, b_t0, c_t0, t0 = opt_res.x
        fitted_lag_t0 = polyn_lag(self.time_lag, a_t0, b_t0, c_t0, t0)
        residuals_t0 = self.lag - fitted_lag_t0
        # avg_residual = np.mean(abs(residuals))
        rmsd_lag_polyn = np.sqrt(np.mean(residuals_t0**2))

        return rmsd_lag_polyn
    
    def define_SNR_lum_noise(self):
        ''' Define the SNR luminosity noise '''
        # Compute the SNR of the luminosity data
        shower_code = self.find_IAU_code()
        # check if shower_code is None
        if shower_code is None:
            print("No IAU code found")
        else:
            print("Shower code:", shower_code)

        const = np.nan
        # if the last 3 letters of dir_pickle_files are DRA set const 8.0671
        if shower_code == 'DRA':
            const = 8.0671
        elif shower_code == 'CAP':
            const = 7.8009
        elif shower_code == 'ORI':
            const = 7.3346

        if np.isnan(const):
            # inverse polinomial fit
            velocities = np.array([20, 23, 66])*1000 # km/s
            offsets = np.array([8.0671, 7.8009, 7.3346]) # constant values
            
            log_velocities = np.log(velocities)
            log_offsets = np.log(offsets)

            b, log_a = np.polyfit(log_velocities, log_offsets, 1)
            a = np.exp(log_a)
        
            const = a * self.v_init**b

        apparent_mag = np.max(self.apparent_magnitudes)
        # find the index of the apparent magnitude in the list
        index = np.where(np.array(self.apparent_magnitudes) == apparent_mag)[0][0]

        lum_noise = self.luminosity[index]/10**((apparent_mag-const)/(-2.5))

        return lum_noise

    def find_IAU_code(self):
        # check if self.file_name is a array or a string
        if not isinstance(self.file_name, str):
            # take the first element of the array
            file_name_IAU = self.file_name[0]
        else:
            file_name_IAU = self.file_name
        # Get the directory where self.file_name is stored
        file_dir = os.path.dirname(file_name_IAU)
        
        # Define the filenames to look for
        report_file = None
        for file_name in os.listdir(file_dir):
            if file_name.endswith("report.txt"):
                print("Found report.txt file to extract IAU code")
                report_file = file_name
                break
        if report_file is None:
            for file_name in os.listdir(file_dir):
                if file_name.endswith("report_sim.txt"):
                    print("Found report_sim.txt file to extract IAU code")
                    report_file = file_name
                    break
        
        # If no report file is found, return None
        if report_file is None:
            print("No report .txt file found in the directory")
            return None
        
        # Open and read the report file
        report_path = os.path.join(file_dir, report_file)
        with open(report_path, 'r', encoding='utf-8') as file:
            for line in file:
                match = re.search(r"IAU code =\s+(\S+)", line)
                if match:
                    return match.group(1)  # Extracted IAU code
        
        return None  # Return None if no match is found

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



    def load_json_data(self,use_all_cameras):

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

            # self.noise_lum = 2.5
            if np.isnan(self.noise_lum):
                self.noise_lum = 2.5
                print('Assumed default Noise in luminosity:',self.noise_lum)
            self.noise_mag = 0.1

            # add a gausian noise to the luminosity of 2.5
            lum_obs_data = self.luminosity_arr + np.random.normal(loc=0, scale=self.noise_lum, size=len(self.luminosity_arr))

            # Identify indices where lum_obs_data > 0
            positive_indices = np.where(lum_obs_data > 0)[0]  # Get only valid indices

            # If no positive values exist, return empty list
            if len(positive_indices) == 0:
                indices_visible = []
                return
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
            
            # extra filter that ensures no NaNs remain in the height array:
            finite_mask = np.isfinite(self.leading_frag_height_arr[indices_visible])
            indices_visible = indices_visible[finite_mask]

            # Select time, magnitude, height, and length above the visibility limit
            time_visible = self.time_arr[indices_visible]
            # the rest of the arrays are the same length as time_arr
            lum_visible = lum_obs_data[indices_visible]
            ht_visible   = self.leading_frag_height_arr[indices_visible]
            len_visible  = self.leading_frag_length_arr[indices_visible]
            # mag_visible  = self.abs_magnitude[indices_visible]
            vel_visible  = self.leading_frag_vel_arr[indices_visible]

            # Resample the time to the system FPS
            lum_interpol = scipy.interpolate.CubicSpline(time_visible, lum_visible)
            ht_interpol  = scipy.interpolate.CubicSpline(time_visible, ht_visible)
            len_interpol = scipy.interpolate.CubicSpline(time_visible, len_visible)
            # mag_interpol = scipy.interpolate.CubicSpline(time_visible, mag_visible)
            vel_interpol = scipy.interpolate.CubicSpline(time_visible, vel_visible)

            fps_lum = 32
            if use_all_cameras:
                self.fps = 80
                self.stations = ['01G','02G','01T','02T']
                # self.noise_lag = 5
                if np.isnan(self.noise_lag):
                    self.noise_lag = 5
                self.noise_vel = self.noise_lag*np.sqrt(2)/(1.0/self.fps)
                # multiply by a number between 0.6 and 0.4 for the time to track for CAMO
                time_to_track = (time_visible[-1]-time_visible[0])*np.random.uniform(0.4,0.6)
                time_sampled_lag, stations_array_lag = self.mimic_fps_camera(time_visible,time_to_track,self.fps,self.stations[2],self.stations[3])
                time_sampled_lum, stations_array_lum = self.mimic_fps_camera(time_visible,0,fps_lum,self.stations[0],self.stations[1])
            else:
                self.fps = 32
                self.stations = ['01F','02F']
                # self.noise_lag = 40
                if np.isnan(self.noise_lag):
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

            # Find the index closest to the first height without NaNs
            index = np.argmin(np.abs(self.leading_frag_height_arr[~np.isnan(self.leading_frag_height_arr)]
                                      - self.height_lag[0]))

            # Compute the theoretical lag (without noise)
            lag_no_noise = (self.leading_frag_length_arr - self.leading_frag_length_arr[index]) - \
                           (self.v_init * (self.time_arr - self.time_arr[index]))
            lag_no_noise -= lag_no_noise[index]

            # Interpolate to align with observed height_lag
            self.lag = np.interp(self.height_lag, np.flip(self.leading_frag_height_arr), np.flip(lag_no_noise)) + np.random.normal(loc=0, scale=self.noise_lag, size=len(time_sampled_lag))

            self.length = len_interpol(time_sampled_lag) 
            self.length = self.length - self.length[0]
            self.length = self.length + np.random.normal(loc=0, scale=self.noise_lag, size=len(time_sampled_lag))

            # velocity noise
            self.velocities = vel_interpol(time_sampled_lag) + np.random.normal(loc=0, scale=self.noise_vel, size=len(time_sampled_lag))
            
            # Make const behave like a dict
            if hasattr(self, 'const'):
                self.const = self.const.__dict__

            self.new_json_file_save = self._save_json_data()


    def mimic_fps_camera(self, time_visible, time_to_track, fps, station1, station2):

        # Sample the time according to the FPS from one camera
        time_sampled_cam1 = np.arange(np.min(time_visible)+time_to_track, np.max(time_visible), 1.0/fps)

        # Simulate sampling of the data from a second camera, with a random phase shift
        time_sampled_cam2 = time_sampled_cam1 + np.random.uniform(-1.0/fps, 1.0/fps)

        # Ensure second camera does not start before time_visible[0]
        time_sampled_cam2 = time_sampled_cam2[time_sampled_cam2 >= np.min(time_visible)]

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

        return json_file_save



###############################################################################
# find dynestyfile and priors
###############################################################################

def setup_folder_and_run_dynesty(input_dir, output_dir='', prior='', resume=True, use_all_cameras=True, only_plot=True, cores=None):
    """
    Create the output folder if it doesn't exist.
    """

    # initlize cml_args
    class cml_args:
        pass

    cml_args.input_dir = input_dir
    cml_args.output_dir = output_dir
    cml_args.prior = prior
    cml_args.use_all_cameras = use_all_cameras
    cml_args.resume = resume
    cml_args.only_plot = only_plot
    cml_args.cores = cores

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
            use_all_cameras=cml_args.use_all_cameras
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
            dynesty_file, pickle_file, bounds, flags_dict, fixed_values = dynesty_info
            obs_data = finder.observation_instance(base_name)
            obs_data.file_name = pickle_file # update teh file name in the observation data object

            # update the log file with the error join out_folder,"log_"+base_name+".txt"
            log_file_path = os.path.join(out_folder, f"log_{base_name}.txt")

            dynesty_file = setup_dynesty_output_folder(out_folder, obs_data, bounds, flags_dict, fixed_values, pickle_file, dynesty_file, prior_path, base_name, log_file_path)
            
            ### set up obs_data const values to run same simultaions in run_simulation #################

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

            ##################################################################################################

            if not cml_args.only_plot: 

                # start a timer to check how long it takes to run dynesty
                start_time = time.time()
                # Run dynesty
                try:
                    main_dynestsy(dynesty_file, obs_data, bounds, flags_dict, fixed_values, cml_args.cores, output_folder=out_folder, file_name=base_name, log_file_path=log_file_path)
                except Exception as e:
                    # Open the file in append mode and write the error message
                    with open(log_file_path, "a") as log_file:
                        log_file.write(f"\nError encountered in dynestsy run: {e}")
                    print(f"Error encountered in dynestsy run: {e}")
                    # now try and plot the dynesty file results
                    try:
                        dsampler = dynesty.DynamicNestedSampler.restore(dynesty_file)
                        plot_dynesty(dsampler.results, obs_data, flags_dict, fixed_values, out_folder, base_name,log_file_path)
                    except Exception as e:
                        with open(log_file_path, "a") as log_file:
                            log_file.write(f"Error encountered in dynestsy plot: {e}")
                        print(f"Error encountered in dynestsy plot: {e}")
                        
                    # take only the name of the log file and the path
                    path_log_file, log_file_name = os.path.split(log_file_path)
                    # chenge the name log_file_name of the log_file_path to log_file_path_error adding error_ at the beginning
                    log_file_path_error = os.path.join(path_log_file, f"error_{log_file_name}")
                    # rename the log_file_path to log_file_path_error
                    os.rename(log_file_path, log_file_path_error)
                    print(f"Log file renamed to {log_file_path_error}")
                    log_file_path = log_file_path_error

                # Save the time it took to run dynesty
                end_time = time.time()
                elapsed_time = datetime.timedelta(seconds=end_time - start_time)

                # Add this time to the log file (use the correct log file path)
                with open(log_file_path, "a") as log_file:
                    log_file.write(f"\nTime to run dynesty: {elapsed_time}")

                # Print the time to run dynesty in hours, minutes, and seconds
                print(f"Time to run dynesty: {elapsed_time}")

            elif cml_args.only_plot and os.path.isfile(dynesty_file): 
                print("Only plotting requested. Skipping dynesty run.")
                dsampler = dynesty.DynamicNestedSampler.restore(dynesty_file)
                # dsampler = dynesty.DynamicNestedSampler.restore(dynesty_file)
                plot_dynesty(dsampler.results, obs_data, flags_dict, fixed_values, out_folder, base_name,log_file_path)

            else:
                print("Fail to generate dynesty plots, dynasty file not found:",dynesty_file)
                print("If you want to run the dynasty file set only_plot to False")


def setup_dynesty_output_folder(out_folder, obs_data, bounds, flags_dict, fixed_values, pickle_files='', dynesty_file='', prior_path='', base_name='', log_file_path=''):

    if log_file_path == '':
        log_file_path = os.path.join(out_folder, f"log_{base_name}.txt")
        base_name_log = "log_{base_name}.txt"
    else:
        base_name_log = os.path.basename(log_file_path)

    print("--------------------------------------------------")
    # check if a file with the name "log"+n_PC_in_PCA+"_"+str(len(df_sel))+"ev.txt" already exist
    if os.path.exists(log_file_path):
        # remove the file
        os.remove(log_file_path)
    sys.stdout = Logger(out_folder,base_name_log) # 
    print(f"Meteor:", base_name)
    print("  File name:    ", pickle_files)
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

    # create output folder and put the image
    os.makedirs(out_folder, exist_ok=True)
    plot_data_with_residuals_and_real(obs_data, output_folder=out_folder, file_name=base_name)
    
    dynesty_file_in_output_path = os.path.join(out_folder,os.path.basename(dynesty_file))
    # copy the dynesty file to the output folder if not already there
    if not os.path.exists(dynesty_file_in_output_path) and os.path.isfile(dynesty_file):
        shutil.copy(dynesty_file, out_folder)
        print("dynesty file copied to output folder:", dynesty_file_in_output_path)

    # add the prior pr
    if prior_path != "":
        # check if there is a prior file with the same name in the output_folder
        prior_file_output = os.path.join(out_folder,os.path.basename(prior_path))
        if not os.path.exists(prior_file_output):
            shutil.copy(prior_path, out_folder)
            print("prior file copied to output folder:", prior_file_output)
        # # folder where is stored the pickle_file
        # prior_file_input = os.path.join(os.path.dirname(pickle_file),os.path.basename(prior_path))
        # if not os.path.exists(prior_file_input):
        #     shutil.copy(prior_path, os.path.dirname(pickle_file))
        #     print("prior file copied to input folder:", prior_file_input)

    # Dictionary to track count of each base name
    base_name_counter = {}

    for pickle_file in pickle_files:
        
        # Check that the file actually exists
        if not os.path.isfile(pickle_file):
            print("Original observation file not found, not copied:", pickle_file)
            continue
        
        # Extract the base filename
        base_name = os.path.basename(pickle_file)
        
        # Check if we've seen this filename before
        if base_name in base_name_counter:
            base_name_counter[base_name] += 1
            # Insert a suffix to differentiate
            root, ext = os.path.splitext(base_name)
            new_base_name = f"{root}_{base_name_counter[base_name]}{ext}"
        else:
            # First time seeing this base_name
            base_name_counter[base_name] = 0
            new_base_name = base_name

        # Compute the destination path
        dest_path = os.path.join(out_folder, new_base_name)

        # check if pickle_file and dest_path are the same
        if pickle_file != dest_path:
            # Copy the file
            shutil.copy(pickle_file, dest_path)
            print(f"Copied {pickle_file} to {dest_path}")

    # # look at all the fill pickle_files and copy them to the output folder conider the pickle_files to be a list
    # for pickle_file in pickle_files:
    #     # check if pickle_file is not in the output directory
    #     if not os.path.exists(os.path.join(out_folder,os.path.basename(pickle_file))) and os.path.isfile(pickle_file):
    #         shutil.copy(pickle_file, out_folder)
    #         print("observation file copied to output folder:", os.path.join(out_folder,os.path.basename(pickle_file)))
    #     elif not os.path.isfile(pickle_file):
    #         print("original observation file not found, not copied:",pickle_file)
                    
    # # check if pickle_file is not in the output directory
    # if not os.path.exists(os.path.join(out_folder,os.path.basename(pickle_file))) and os.path.isfile(pickle_file):
    #     shutil.copy(pickle_file, out_folder)
    #     print("observation file copied to output folder:", os.path.join(out_folder,os.path.basename(pickle_file)))
    # elif not os.path.isfile(pickle_file):
    #     print("original observation file not found, not copied:",pickle_file)

    return dynesty_file_in_output_path


def read_prior_noise(file_path):
    """
    chack if present and read the prior file and return the bounds, flags, and fixed values.
    """

    # defealut noise values
    noise_lag_prior = np.nan
    noise_lum_prior = np.nan

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

            # check if name is noise_lag
            if name != "noise_lag" and name != "noise_lum":
                continue
            else:
                # Handle fixed values
                if "fix" in parts:
                    val = parts[1].strip() if len(parts) > 1 else "nan"
                    val_fixed = safe_eval(val) if val.lower() != "nan" else np.nan
                    if not np.isnan(val_fixed):
                        if name == "noise_lag":
                            noise_lag_prior = val_fixed
                        elif name == "noise_lum":
                            noise_lum_prior = val_fixed
                                                    
                min_val = parts[1].strip() if len(parts) > 1 else "nan"
                max_val = parts[2].strip() if len(parts) > 2 else "nan"
                flags = [flag.strip() for flag in parts[3:]] if len(parts) > 3 else []
                
                # Handle NaN values and default replacement
                min_val = safe_eval(min_val) if min_val.lower() != "nan" else np.nan
                max_val = safe_eval(max_val) if max_val.lower() != "nan" else np.nan

                #### vel, mass, zenith ####
                # check if name=='v_init' or zenith_angle or m_init or erosion_height_start values are np.nan and replace them with the object_meteor values
                if np.isnan(max_val):
                    continue
                elif ("norm" in flags or "invgamma" in flags):
                    if name == "noise_lag":
                        noise_lag_prior = max_val
                    elif name == "noise_lum":
                        noise_lum_prior = max_val
                else:
                    if name == "noise_lag":
                        # do the mean of the max and min values
                        noise_lag_prior = np.mean([min_val,max_val])
                    elif name == "noise_lum":
                        # do the mean of the max and min values
                        noise_lum_prior = np.mean([min_val,max_val])
                
    return noise_lag_prior, noise_lum_prior


class find_dynestyfile_and_priors:
    """
    This class automates the setup of .dynesty output files and associated prior configurations 
    for observation data. It handles both single file inputs and directories containing multiple 
    .pickle files, applying a consistent naming and resume logic to avoid file collisions.

    1. Input Handling:
       - **Single File Input:**  
         - When `input_dir_or_file` is a file, its extension is stripped and replaced with ".dynesty".
         - If the provided `prior_file` exists, it is used to extract bounds, flags, and fixed values.
         - Otherwise, the class searches the file's folder for a .prior file; if none is found, defaults are used.
       - **Directory Input:**  
         - The class recursively traverses the directory, looking for files ending in ".pickle".
         - For each .pickle file, it constructs a corresponding .dynesty filename based on the file’s base name.
         - The same prior selection logic is applied: use the provided `prior_file` if valid, otherwise search 
           the current folder for a .prior file, defaulting if needed.

    2. Dynesty File Naming and Resume Logic:
       - If a .dynesty file with the derived name already exists in the target folder:
         - With `resume=True`, the existing file is reused.
         - With `resume=False`, a new .dynesty filename is generated by appending suffixes (e.g., _n1, _n2) 
           to prevent overwriting.
       - If no .dynesty file exists, one is created from the base name of the corresponding .pickle file.

    3. Output Directory Management:
       - If an `output_dir` is specified, a subfolder (named after the .dynesty base name) is created to store results.
       - If no output directory is provided, results are stored in the same folder as the .dynesty file.

    4. Result Storage:
       - The class collects and stores key information:
         - `self.input_folder_file`: A list of tuples containing (dynesty file path, input file path, bounds, flags, and fixed values).
         - `self.priors`: A list of the .prior file paths used (or an empty string if defaults were applied).
         - `self.output_folders`: The output folder for each processed input.
       - An observation instance is created for each input file, accessible via the `observation_instance(base_name)` method.

    Overall, this design provides a flexible and automated approach to preparing observation data for further 
    processing, ensuring that each input is properly paired with a .dynesty configuration and the relevant prior settings.
    """

    def __init__(self, input_dir_or_file, prior_file="", resume=False, output_dir="", use_all_cameras=False):
        self.input_dir_or_file = input_dir_or_file
        self.prior_file = prior_file
        self.resume = resume
        self.output_dir = output_dir
        self.use_all_cameras = use_all_cameras

        # Prepare placeholders
        self.base_names = []        # [base_name, ...] (no extension)
        self.input_folder_file = [] # [(dynesty_file, input_file, bounds, flags_dict, fixed_values), ...]
        self.priors = []            # [used_prior_path_or_empty_string, ...]
        self.output_folders = []    # [output_folder_for_this_dynesty, ...]
        self.observation_objects = {}  # {base_name: observation_instance}

        # Kick off processing
        self._process_input()

    def _process_input(self):
        """Decide if input is file or directory, build .dynesty, figure out prior, and store results."""
        if os.path.isfile(self.input_dir_or_file):
            # Single file case
            input_file = self.input_dir_or_file
            root = os.path.dirname(input_file)
            # take all the files in the folder
            files = os.listdir(root)
            self._main_folder_costructor(input_file,root,files)

        else:

            all_pickle_files = []

            # Walk through all subdirectories and find pickle files
            for root, dirs, files in os.walk(self.input_dir_or_file):
                pickle_files = [f for f in files if f.endswith('.pickle')]
                if not pickle_files:
                    continue

                # Flatten list using extend (instead of appending a list inside a list)
                all_pickle_files.extend(os.path.join(root, f) for f in pickle_files)

            print(all_pickle_files)

            # Call function to process found pickle files
            clusters = self._combine_meteor_camera_pickle(all_pickle_files)

            print(f"Found {len(clusters)} meteors in {self.input_dir_or_file}")
            for i, cluster_info in enumerate(clusters, start=1):
                # print(f"Cluster #{i}")
                print("meteor:", cluster_info['cluster_name'])
                # print("Filenames:", cluster_info['filenames'])
                print("Union stations:", cluster_info['union_stations'])
                # print("Time range:", cluster_info['jd_range'])
                # print("-----------------")

                # now check if only a single file is found in the cluster
                if len(cluster_info['filenames']) == 1:
                    input_file = cluster_info['filenames'][0]
                    root = os.path.dirname(input_file)
                    files = os.listdir(root)
                    self._main_folder_costructor(input_file,root,files)

                elif len(cluster_info['filenames']) > 1:
                    # Combine multiple files into a single observation instance
                    print("Multiple files found in the cluster. Combine them into a single observation instance.")
                    # chek if all cluster_info['filenames'] are in the same folder already
                    if all(os.path.dirname(cluster_info['filenames'][0]) == os.path.dirname(f) for f in cluster_info['filenames']):
                        root = os.path.dirname(cluster_info['filenames'][0])
                        files = os.listdir(root)
                        self._main_folder_costructor(cluster_info['filenames'],root,files,cluster_info['cluster_name'])
                    else:
                        # join self.input_dir_or_file and cluster_info['cluster_name'] to create a new folder
                        new_combined_input_folder = os.path.join(self.input_dir_or_file, cluster_info['cluster_name'])
                        # check if the folder exists
                        if os.path.exists(new_combined_input_folder):
                            root = new_combined_input_folder
                            files = os.listdir(root)
                            self._main_folder_costructor(cluster_info['filenames'],root,files,cluster_info['cluster_name'])
                        else:
                            self._main_folder_costructor(cluster_info['filenames'],new_combined_input_folder,[],cluster_info['cluster_name'])



    def _combine_meteor_camera_pickle(self, all_pickle_files, time_threshold=1/86400):
        """
        Group the given pickle files by time (within `time_threshold` in JD).
        
        For each group (cluster):
        1) Compute the union of all stations (cameras) in that cluster.
        2) If at least one file in the cluster already has ALL stations, 
            pick ONLY that/those file(s). 
            Otherwise, pick all unique station sets (removing exact duplicates).
        3) Return a list of dicts, each with:
            'filenames':       the .pickle files selected for that cluster
            'union_stations':  sorted list of all stations
            'jd_range':        (min_jd, max_jd)
            'cluster_name':    string like "YYYYMMDD_HHMMSS.sss" from avg(min_jd, max_jd)
        """
        data = []
        for fullpath in all_pickle_files:
            folder, fname = os.path.split(fullpath)
            try:
                traj = loadPickle(folder, fname)  # your existing load function
            except Exception as e:
                print(f"Cannot load pickle {fullpath}: {e}")
                continue
            
            if not hasattr(traj, 'orbit'):
                print(f"Trajectory data not found in {fullpath}")
                continue
            
            jdt_ref = getattr(traj, 'jdt_ref', None)
            if jdt_ref is None:
                print(f"No jdt_ref found in {fullpath}")
                continue
            
            station_ids = []
            for obs in getattr(traj, 'observations', []):
                station_ids.append(obs.station_id)
            
            data.append({
                'filename': fullpath,
                'jdt_ref': jdt_ref,
                'stations': frozenset(station_ids)
            })
        
        if not data:
            print("No valid trajectory data found.")
            return []

        # Sort by time
        df = pd.DataFrame(data).sort_values('jdt_ref').reset_index(drop=True)

        # ---------------------------------------------
        # Cluster by checking consecutive files' times
        # ---------------------------------------------
        clusters_raw = []
        current_cluster = [df.iloc[0]]

        for i in range(1, len(df)):
            curr_row = df.iloc[i]
            prev_row = current_cluster[-1]
            if abs(curr_row['jdt_ref'] - prev_row['jdt_ref']) <= time_threshold:
                current_cluster.append(curr_row)
            else:
                clusters_raw.append(current_cluster)
                current_cluster = [curr_row]

        if current_cluster:
            clusters_raw.append(current_cluster)

        # -------------------------------------------------
        # Build final clusters, check for "all cameras" file
        # -------------------------------------------------
        clusters_result = []
        for cluster_rows in clusters_raw:
            # 1) Compute the union of all stations in this cluster
            union_stations = set()
            jd_values = []
            for row in cluster_rows:
                union_stations |= row['stations']
                jd_values.append(row['jdt_ref'])

            min_jd = min(jd_values)
            max_jd = max(jd_values)
            cluster_time = []
            for jd_value in jd_values:
                # transform the jd_value to a datetime object
                timestamp = (jd_value - 2440587.5) * 86400.0
                dt = datetime.datetime.utcfromtimestamp(timestamp)
                base_str = dt.strftime("%Y%m%d_%H%M%S")
                msec = dt.microsecond // 1000
                cluster_time.append(f"{base_str}.{msec:03d}")
            # put jd_values in 
            avg_jd = np.mean(jd_values)
            timestamp = (avg_jd - 2440587.5) * 86400.0
            avg_dt = datetime.datetime.utcfromtimestamp(timestamp)

            base_str = avg_dt.strftime("%Y%m%d_%H%M%S")
            msec = avg_dt.microsecond // 1000
            # cluster_name = f"{base_str}-{msec:03d}_combined"
            cluster_name = f"{base_str}_combined"

            # 2) See if any file in cluster_rows has ALL stations
            #    i.e., row['stations'] == union_stations
            files_with_all = [r for r in cluster_rows if r['stations'] == union_stations]

            if files_with_all:
                # Keep only the FIRST file that has the entire station set
                # If you prefer the last, do [-1] instead
                chosen = files_with_all[0]  
                cluster_filenames = [chosen['filename']]
            else:
                # Otherwise, keep all unique station sets
                used_station_sets = set()
                cluster_filenames = []
                for row in cluster_rows:
                    if row['stations'] not in used_station_sets:
                        used_station_sets.add(row['stations'])
                        cluster_filenames.append(row['filename'])

            clusters_result.append({
                'cluster_name': cluster_name,
                'filenames': cluster_filenames,
                'union_stations': sorted(union_stations),
                'jd_range': (cluster_time)
            })

        return clusters_result


    def _main_folder_costructor(self, input_file, root, files, base_name=""):
        """ Main function to return the observation instance """

        lag_noise_prior = np.nan
        lum_noise_prior = np.nan
        # If user gave a valid .prior path, read it once.
        if os.path.isfile(self.prior_file):
            prior_path_noise = self.prior_file
            lag_noise_prior, lum_noise_prior = read_prior_noise(self.prior_file)
        else:
            # Look for local .prior
            existing_prior_list = [f for f in files if f.endswith(".prior")]
            if existing_prior_list:
                prior_path_noise = os.path.join(root, existing_prior_list[0])
                lag_noise_prior, lum_noise_prior = read_prior_noise(prior_path_noise)
        # if np.isnan(lag_noise_prior) or np.isnan(lum_noise_prior):
        #     if np.isnan(lag_noise_prior):
        #         print("NO NOISE values found in prior file for lag.")
        #     if np.isnan(lum_noise_prior):
        #         print("NO NOISE values found in prior file for lum.")
        # else:
        #     print("Found noise in prior file: lag",lag_noise_prior,"m, lum",lum_noise_prior,"J/s")
        if not (np.isnan(lag_noise_prior) or np.isnan(lum_noise_prior)):
            print("Found noise in prior file: lag",lag_noise_prior,"m, lum",lum_noise_prior,"J/s")

        observation_instance = observation_data(input_file, self.use_all_cameras, lag_noise_prior, lum_noise_prior)

        # check if new_json_file_save is present in observation_instance
        if hasattr(observation_instance, 'new_json_file_save'):
            # change the input_file to the new_json_file_save
            input_file = observation_instance.new_json_file_save

        # check if input_file is a list
        if isinstance(input_file, list):
            # take the first file in the list
            file_name_no_ext = base_name
            input_files_save = input_file
        else:
            file_name_no_ext = os.path.splitext(input_file)[0]
            input_files_save = [input_file]
        
        possible_dynesty = os.path.join(root, file_name_no_ext + ".dynesty")

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
                dynesty_file = possible_dynesty
        else:
            # No .dynesty => create from .pickle base name
            dynesty_file = possible_dynesty

        # If user gave a valid .prior path, read it once.
        if os.path.isfile(self.prior_file):
            prior_path = self.prior_file
            bounds, flags_dict, fixed_values = read_prior_to_bounds(observation_instance,self.prior_file)
        else:

            # Look for local .prior
            existing_prior_list = [f for f in files if f.endswith(".prior")]
            if existing_prior_list:
                prior_path = os.path.join(root, existing_prior_list[0])
                # print the prior path has been found
                print(f"Take the first Prior file found in the same folder as the observation file: {prior_path}")
                bounds, flags_dict, fixed_values = read_prior_to_bounds(observation_instance,prior_path)
            else:
                # default
                prior_path = ""
                bounds, flags_dict, fixed_values = read_prior_to_bounds(observation_instance)

        if base_name == "":
            base_name = self._extract_base_name(input_file)

        # # Check if base_name already exists, and skip if it does
        # if base_name in self.observation_objects:
        #     # Convert to sets for comparison
        #     existing_stations = set(self.observation_objects[base_name].stations)
        #     new_stations = set(observation_instance.stations)
        #     # Find stations that are unique to each list (not in both)
        #     unique_stations = existing_stations.symmetric_difference(new_stations)

        #     if unique_stations:
        #         # Sort to ensure deterministic order (optional)
        #         new_stations = sorted(new_stations)
        #         base_name = base_name + "_" + "_".join(new_stations)
        #         print(f"Updated base name: {base_name}")
        #     else:
        #         print(f"Skipping duplicate entry for {base_name}")
        #         return  # Exit function to prevent duplicate insertion
            
        if self.output_dir=="":
            # if root do not exist create it
            if not os.path.exists(root):
                os.makedirs(root)
            # Output folder is not specified
            output_folder = root
        else:
            # Output folder is specified
            output_folder = os.path.join(self.output_dir, base_name)
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

        # Store results
        self.base_names.append(base_name)
        self.input_folder_file.append((dynesty_file, input_files_save, bounds, flags_dict, fixed_values))
        self.priors.append(prior_path)
        self.output_folders.append(output_folder)
        self.observation_objects[base_name] = observation_instance


    def observation_instance(self, base_name):
        """Return the observation instance corresponding to a specific base name."""
        return self.observation_objects.get(base_name, None)

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


###############################################################################
# Function: dynesty
###############################################################################

class TimeoutException(Exception):
    """Custom exception for timeouts."""
    pass

def RunSimulationWrapper(guess_var, obs_metsim_obj, var_names, fix_var, queue):
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
    
    if 'erosion_mass_max' in var_names and 'm_init' in var_names:
        # check if the guess_var of the erosion_mass_max is smaller than the guess_var of the m_init
        if guess_var[var_names.index('erosion_mass_max')] > guess_var[var_names.index('m_init')]:
            return -np.inf

    ### ONLY on LINUX ###

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
    
    ### ONLY on LINUX ###

    # # time constait
    # # find the time_arr index in simulation_results that are above the np.min(obs_metsim_obj.luminosity) and are after height_lum[0] (the leading_frag_height_arr[-1] is nan)
    # indices_visible = np.where((simulation_results.luminosity_arr[:-1] > np.min(obs_metsim_obj.luminosity)) & (simulation_results.leading_frag_height_arr[:-1] < obs_metsim_obj.height_lum[0]))[0]
    # # check if indices_visible is empty
    # if len(indices_visible) == 0:
    #     return -np.inf
    # real_time_visible = obs_metsim_obj.time_lum[-1]-obs_metsim_obj.time_lum[0]
    # simulated_time_visible = simulation_results.time_arr[indices_visible][-1]-simulation_results.time_arr[indices_visible][0]
    # # check if is too short and the time difference is smaller than 60% of the real time difference
    # if simulated_time_visible < 0.6*real_time_visible:
    #     return -np.inf
    
    simulated_lc_intensity = np.interp(obs_metsim_obj.height_lum, 
                                       np.flip(simulation_results.leading_frag_height_arr), 
                                       np.flip(simulation_results.luminosity_arr))

    lag_sim = simulation_results.leading_frag_length_arr - (obs_metsim_obj.v_init * simulation_results.time_arr)

    simulated_lag = np.interp(obs_metsim_obj.height_lag, 
                              np.flip(simulation_results.leading_frag_height_arr), 
                              np.flip(lag_sim))

    lag_sim = simulated_lag - simulated_lag[0]

    # check if the length of the simulated_lc_intensity is the same as the length of the obs_metsim_obj.luminosity
    if np.sum(~np.isnan(simulated_lc_intensity)) != np.sum(~np.isnan(obs_metsim_obj.luminosity)):
        return -np.inf
    # check if the length of the lag_sim is the same as the length of the obs_metsim_obj.lag
    if np.sum(~np.isnan(lag_sim)) != np.sum(~np.isnan(obs_metsim_obj.lag)):
        return -np.inf

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



def main_dynestsy(dynesty_file, obs_data, bounds, flags_dict, fixed_values, n_core=1, output_folder="", file_name="",log_file_path=""):
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
            dsampler = dynesty.DynamicNestedSampler(pool.loglike, 
                                                    pool.prior_transform, ndim,
                                                    pool = pool)
            dsampler.run_nested(print_progress=True, dlogz_init=0.001, checkpoint_file=dynesty_file)

    else:
        print("Resuming previous run:")
        print('Warning: make sure the number of parameters and the bounds are the same as the previous run!')
        # Resume previous run
        with dynesty.pool.Pool(n_core, log_likelihood_dynesty, prior_dynesty,
                               logl_args=(obs_data, flags_dict, fixed_values, 10),
                               ptform_args=(bounds, flags_dict)) as pool:
            ### RESUME:
            dsampler = dynesty.DynamicNestedSampler.restore(dynesty_file, 
                                                            pool = pool)
            dsampler.run_nested(resume=True, print_progress=True, dlogz_init=0.001, checkpoint_file=dynesty_file)
    # nlive=350, 
    # sample='rslice', 
    print('SUCCESS: dynesty results ready!\n')

    # dsampler.run_nested(
    #     nlive_init=500,         # or override if you prefer a different baseline
    #     maxiter=100000,         # overall max number of iterations
    #     maxcall=2000000,        # overall max number of likelihood calls
    #     dlogz_init=0.001,       # tighten from default 0.01 
    #     print_progress=True,
    #     checkpoint_file='mcmc_checkpoint.dynesty',
    #     checkpoint_every=60,    # how often (in seconds) to write checkpoint
    # )

    # check if output_folder is different from the dynesty_file folder
    if output_folder != os.path.dirname(dynesty_file):
        print("Copying dynesty file to output folder...")
        shutil.copy(dynesty_file, output_folder)
        print("dynesty file copied to:", output_folder)
    
    # dsampler = dynesty.DynamicNestedSampler.restore(filename)
    plot_dynesty(dsampler.results, obs_data, flags_dict, fixed_values, output_folder, file_name,log_file_path)







###############################################################################
if __name__ == "__main__":

    import argparse

    ### COMMAND LINE ARGUMENTS
    arg_parser = argparse.ArgumentParser(description="Run dynesty with optional .prior file.")
    # r"C:\Users\maxiv\WMPG-repoMAX\Code\DynNestSampl\Shower\CAMO\ORI_mode\ORI_mode_CAMO_with_noise.json,C:\Users\maxiv\WMPG-repoMAX\Code\DynNestSampl\Shower\EMCCD\ORI_mode\ORI_mode_EMCCD_with_noise.json,C:\Users\maxiv\WMPG-repoMAX\Code\DynNestSampl\Shower\CAMO\CAP_mode\CAP_mode_CAMO_with_noise.json,C:\Users\maxiv\WMPG-repoMAX\Code\DynNestSampl\Shower\EMCCD\DRA_mode\DRA_mode_EMCCD_with_noise.json,C:\Users\maxiv\WMPG-repoMAX\Code\DynNestSampl\Shower\EMCCD\CAP_mode\CAP_mode_EMCCD_with_noise.json"
    # r"/home/mvovk/WMPG-repoMAX/Code/DynNestSampl/Shower/CAMO/ORI_mode/ORI_mode_CAMO_with_noise.json,/home/mvovk/WMPG-repoMAX/Code/DynNestSampl/Shower/EMCCD/ORI_mode/ORI_mode_EMCCD_with_noise.json,/home/mvovk/WMPG-repoMAX/Code/DynNestSampl/Shower/CAMO/CAP_mode/CAP_mode_CAMO_with_noise.json,/home/mvovk/WMPG-repoMAX/Code/DynNestSampl/Shower/EMCCD/CAP_mode/CAP_mode_EMCCD_with_noise.json,/home/mvovk/WMPG-repoMAX/Code/DynNestSampl/Shower/EMCCD/DRA_mode/DRA_mode_EMCCD_with_noise.json"
    arg_parser.add_argument('--input_dir', metavar='INPUT_PATH', type=str,
        default=r"/home/mvovk/WMPG-repoMAX/Code/DynNestSampl/Shower_testcase/EMCCD/ORI_mode/EMCCD_ORI_mode_with_noise.json,/home/mvovk/WMPG-repoMAX/Code/DynNestSampl/Shower_testcase/EMCCD/ORI_mean/EMCCD_ORI_mean_with_noise.json,/home/mvovk/WMPG-repoMAX/Code/DynNestSampl/Shower_testcase/CAMO/ORI_mode/CAMO_ORI_mode_with_noise.json,/home/mvovk/WMPG-repoMAX/Code/DynNestSampl/Shower_testcase/CAMO/ORI_mean/CAMO_ORI_mean_with_noise.json,/home/mvovk/WMPG-repoMAX/Code/DynNestSampl/Shower_testcase/EMCCD/CAP_mean/EMCCD_CAP_mean_with_noise.json,/home/mvovk/WMPG-repoMAX/Code/DynNestSampl/Shower_testcase/EMCCD/CAP_mode/EMCCD_CAP_mode_with_noise.json,/home/mvovk/WMPG-repoMAX/Code/DynNestSampl/Shower_testcase/CAMO/CAP_mean/CAMO_CAP_mean_with_noise.json,/home/mvovk/WMPG-repoMAX/Code/DynNestSampl/Shower_testcase/CAMO/CAP_mode/CAMO_CAP_mode_with_noise.json,/home/mvovk/WMPG-repoMAX/Code/DynNestSampl/Shower_testcase/EMCCD/DRA_mean/EMCCD_DRA_mean_with_noise.json,/home/mvovk/WMPG-repoMAX/Code/DynNestSampl/Shower_testcase/EMCCD/DRA_mode/EMCCD_DRA_mode_with_noise.json",
        # "/home/mvovk/WMPG-repoMAX/Code/DynNestSampl/Shower_testcase/ORI-mass/Mode_5e-6kg/ORI_mode_with_noise5e-6kg.json,/home/mvovk/WMPG-repoMAX/Code/DynNestSampl/Shower_testcase/ORI-mass/Mode_3e-6kg/ORI_mode_with_noise3e-6kg.json,/home/mvovk/WMPG-repoMAX/Code/DynNestSampl/Shower_testcase/ORI-mass/Mode_1e-6kg/ORI_mode_with_noise1e-6kg.json,/home/mvovk/WMPG-repoMAX/Code/DynNestSampl/Shower_testcase/ORI-mass/Mode_8e-7kg/ORI_mode_with_noise8e-7kg.json,/home/mvovk/WMPG-repoMAX/Code/DynNestSampl/Shower_testcase/ORI-mass/Mode_5e-7kg/ORI_mode_with_noise5e-7kg.json",
        help="Path to walk and find .pickle file or specific single file .pickle or .json file divided by ',' in between.")
    # /home/mvovk/Results/Results_Nested/validation/
    arg_parser.add_argument('--output_dir', metavar='OUTPUT_DIR', type=str,
        default=r"/home/mvovk/Results/Results_Nested/Validation_CAMO-EMCCD/",
        help="Where to store results. If empty, store next to each .dynesty.")
    # /home/mvovk/WMPG-repoMAX/Code/DynNestSampl/stony_meteoroid.prior
    arg_parser.add_argument('--prior', metavar='PRIOR', type=str,
        default=r"",
        help="Path to a .prior file. If blank, we look in the .dynesty folder or default to built-in bounds.")
    
    arg_parser.add_argument('--use_all_cameras', metavar='USE_ALL_CAMERAS', type=bool, default=True,
        help="If True, use only CAMO data for lag if present in pickle file, or generate json file with CAMO noise. If False, do not use/generate CAMO data (by default is False).")

    arg_parser.add_argument('--resume', metavar='RESUME', type=bool, default=True,
        help="If True, resume from existing .dynesty if found. If False, create a new version.")
    
    arg_parser.add_argument('--only_plot', metavar='ONLY_PLOT', type=bool, default=False,
        help="If True, only plot the results of the dynesty run. If False, run dynesty.")

    arg_parser.add_argument('--cores', metavar='CORES', type=int, default=None,
        help="Number of cores to use. Default = all available.")


    # Optional: suppress warnings
    # warnings.filterwarnings('ignore')

    # Parse
    cml_args = arg_parser.parse_args()

    setup_folder_and_run_dynesty(cml_args.input_dir, cml_args.output_dir, cml_args.prior, cml_args.resume, cml_args.use_all_cameras, cml_args.only_plot, cml_args.cores)

    print("\nDONE: Completed processing of all files in the input directory.\n")    
