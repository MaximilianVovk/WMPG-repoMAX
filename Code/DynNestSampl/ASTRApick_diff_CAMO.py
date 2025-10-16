# main.py (inside my_subfolder)
import sys
import os

import numpy as np

# Add the parent directory to the sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from DynNestSapl_metsim import *

import numpy as np

def _is_1T2T(name: str) -> bool:
    s = str(name)
    return ("1T" in s) or ("2T" in s)

def _nearest_indices(src_h: np.ndarray, dst_h: np.ndarray) -> np.ndarray:
    """
    For each height in dst_h, return the index into src_h of the nearest-by-absolute-difference height.
    """
    # Ensure 1D float arrays
    src = np.asarray(src_h, dtype=float).ravel()
    dst = np.asarray(dst_h, dtype=float).ravel()
    # Compute |src[:,None] - dst[None,:]| and argmin over src
    return np.abs(src[:, None] - dst[None, :]).argmin(axis=0)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

# --- Helpers (keep once in your module) ---
def _nearest_indices(src_h: np.ndarray, dst_h: np.ndarray) -> np.ndarray:
    src = np.asarray(src_h, dtype=float).ravel()
    dst = np.asarray(dst_h, dtype=float).ravel()
    return np.abs(src[:, None] - dst[None, :]).argmin(axis=0)

def _is_01T02T(name: str) -> bool:
    s = str(name)
    return ("01T" in s) or ("02T" in s)

def plot_vel_lag_residuals_CAMO_EMCCD(obs_data, output_folder: str = "", file_name: str = ""):
    """
    Plot only Velocity & Lag (top row) and their residuals (bottom row).
    Residuals are (other - reference) where reference is the union of stations
    whose name contains '01T' or '02T'. Pair points by nearest HEIGHT, and keep
    only points whose heights fall within the reference height span.
    """

    # ---- Figure layout: 2 cols (Velocity, Lag) × 2 rows (data, residuals)
    fig = plt.figure(figsize=(8, 6))
    gs = gridspec.GridSpec(
        2, 2, figure=fig,
        height_ratios=[3, 1], width_ratios=[1, 1],
        hspace=0.0, wspace=0.35  # hspace=0 attaches top/bottom axes
    )

    ax_vel      = fig.add_subplot(gs[0, 0])                 # velocity vs time
    ax_vel_res  = fig.add_subplot(gs[1, 0], sharex=ax_vel)  # Δv vs time (shares x)
    ax_lag      = fig.add_subplot(gs[0, 1])                 # lag vs time
    ax_lag_res  = fig.add_subplot(gs[1, 1], sharex=ax_lag)  # Δlag vs time (shares x)

    # Hide top-row x tick labels so the axes meet cleanly
    ax_vel.tick_params(labelbottom=False)
    ax_lag.tick_params(labelbottom=False)

    # Consistent colors per station (use stations present in lag arrays)
    cmap = plt.get_cmap("tab10")
    station_colors = {}
    for st in np.unique(obs_data.stations_lag):
        if st not in station_colors:
            station_colors[st] = cmap(len(station_colors) % 10)

    # ---- Plot raw Velocity & Lag
    for station in np.unique(obs_data.stations_lag):
        m = (obs_data.stations_lag == station)
        if not np.any(m):
            continue

        t  = np.asarray(obs_data.time_lag[m], dtype=float)
        v  = np.asarray(obs_data.velocities[m], dtype=float) / 1000.0  # km/s
        lg = np.asarray(obs_data.lag[m], dtype=float)

        ax_vel.plot(t, v, '.', label=station, color=station_colors[station])
        ax_lag.plot(t, lg, 'x:', label=station, color=station_colors[station])

    ax_vel.set_ylabel('Velocity [km/s]')
    ax_vel.grid(True, linestyle='--', color='lightgray')
    ax_lag.set_ylabel('Lag [m]')
    ax_lag.grid(True, linestyle='--', color='lightgray')

    # ---- Build 01T/02T reference pool from lag arrays
    stations_lag_arr = np.asarray(obs_data.stations_lag)
    ref_mask = np.array([_is_01T02T(s) for s in stations_lag_arr])
    if ref_mask.any():
        ref_h   = np.asarray(obs_data.height_lag[ref_mask], dtype=float)
        ref_vel = np.asarray(obs_data.velocities[ref_mask], dtype=float)
        ref_lag = np.asarray(obs_data.lag[ref_mask], dtype=float)

        # Height coverage window for reference
        hmin, hmax = np.nanmin(ref_h), np.nanmax(ref_h)

        # Residuals for non-reference stations only where heights are within [hmin, hmax]
        other_stations = np.unique(stations_lag_arr[~ref_mask])
        for other_st in other_stations:
            m = (stations_lag_arr == other_st)
            if not np.any(m):
                continue

            h_other = np.asarray(obs_data.height_lag[m], dtype=float)
            t_other = np.asarray(obs_data.time_lag[m], dtype=float)
            v_other = np.asarray(obs_data.velocities[m], dtype=float)
            l_other = np.asarray(obs_data.lag[m], dtype=float)

            keep = (h_other >= hmin) & (h_other <= hmax)
            if not np.any(keep):
                continue

            h_ok = h_other[keep]
            t_ok = t_other[keep]
            v_ok = v_other[keep]
            l_ok = l_other[keep]

            # Pair each kept point to nearest ref height (no extra tolerance)
            idx = _nearest_indices(ref_h, h_ok)

            res_v = (v_ok - ref_vel[idx]) / 1000.0  # km/s
            res_l = (l_ok - ref_lag[idx])

            ax_vel_res.plot(t_ok, res_v, '.', markersize=3,
                            label=f'{other_st} Δv', color=station_colors[other_st])
            ax_lag_res.plot(t_ok, res_l, '.', markersize=3,
                            label=f'{other_st} Δlag', color=station_colors[other_st])

    # Zero lines on residuals
    ax_vel_res.axhline(0, color='lightgray')
    ax_lag_res.axhline(0, color='lightgray')

    ax_vel_res.set_xlabel('Time [s]')
    ax_vel_res.set_ylabel('Res.Vel [km/s]')
    ax_vel_res.grid(True, linestyle='--', color='lightgray')

    ax_lag_res.set_xlabel('Time [s]')
    ax_lag_res.set_ylabel('Res.Lag [m]')
    ax_lag_res.grid(True, linestyle='--', color='lightgray')

    # Deduplicate legends on top axes
    for ax in (ax_vel, ax_lag):
        h, l = ax.get_legend_handles_labels()
        if l:
            uniq = dict(zip(l, h))
            ax.legend(uniq.values(), uniq.keys(), fontsize=8)

    # Save
    outpath = os.path.join(output_folder or "", f"{file_name}_VelLag_res_plot.png")
    print("file saved:", outpath)
    fig.savefig(outpath, bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close(fig)

# Plotting function
def plot_data_residual_CAMO_EMCCD(obs_data, output_folder='', file_name=''):
    ''' Plot the data with residuals and real data '''

    # Create the figure and main GridSpec with specified height ratios
    fig = plt.figure(figsize=(14, 6))
    gs_main = gridspec.GridSpec(2, 4, figure=fig, height_ratios=[1, 1], width_ratios=[1, 1, 1, 1])
    plt.subplots_adjust(wspace=0.35, hspace=0.4)

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
    ax0.set_ylabel('Height [km]')
    ax0.grid(True, linestyle='--', color='lightgray')
    # save the x-axis limits
    xlim_abs_mag = ax0.get_xlim()
    # fix the x-axis limits to xlim_abs_mag
    ax0.set_xlim(xlim_abs_mag)
    # save the y-axis limits
    ylim_abs_mag = ax0.get_ylim()
    # fix the y-axis limits to ylim_abs_mag
    ax0.set_ylim(ylim_abs_mag)
    

    # ax1.fill_betweenx(obs_data.height_lum/1000, -obs_data.noise_mag, obs_data.noise_mag, color='darkgray', alpha=0.2)
    # ax1.fill_betweenx(obs_data.height_lum/1000, -obs_data.noise_mag * 2, obs_data.noise_mag * 2, color='lightgray', alpha=0.2)
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
    ax4.set_xlabel('Luminosity [W]')
    # ax4.tick_params(axis='x', rotation=45)
    ax4.set_ylabel('Height [km]')
    ax4.grid(True, linestyle='--', color='lightgray')
    # save the x-axis limits
    xlim_lum = ax4.get_xlim()
    # fix the x-axis limits to xlim_lum
    ax4.set_xlim(xlim_lum)
    # save the y-axis limits
    ylim_lum = ax4.get_ylim()
    # fix the y-axis limits to ylim_lum
    ax4.set_ylim(ylim_lum)

    # ax5.fill_betweenx(obs_data.height_lum/1000, -obs_data.noise_lum, obs_data.noise_lum, color='darkgray', alpha=0.2)
    # ax5.fill_betweenx(obs_data.height_lum/1000, -obs_data.noise_lum * 2, obs_data.noise_lum * 2, color='lightgray', alpha=0.2)
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
    # ax2.yaxis.labelpad = 1  # Increase space between y-label and y-axis
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

    # # Plot 6: Res.Vel vs. Time
    # ax6.fill_between(obs_data.time_lag, -obs_data.noise_vel/1000, obs_data.noise_vel/1000, color='darkgray', alpha=0.2)
    # ax6.fill_between(obs_data.time_lag, -obs_data.noise_vel * 2/1000, obs_data.noise_vel * 2/1000, color='lightgray', alpha=0.2)
    ax6.plot([obs_data.time_lag[0], obs_data.time_lag[-1]], [0, 0], color='lightgray')
    ax6.set_xlabel('Time [s]')
    ax6.set_ylabel('Res.Vel [km/s]')
    # ax6.yaxis.labelpad = 1  # Increase space between y-label and y-axis
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
    # ax7.fill_between(obs_data.time_lag, -obs_data.noise_lag, obs_data.noise_lag, color='darkgray', alpha=0.2)
    # ax7.fill_between(obs_data.time_lag, -obs_data.noise_lag * 2, obs_data.noise_lag * 2, color='lightgray', alpha=0.2)
    ax7.plot([obs_data.time_lag[0], obs_data.time_lag[-1]], [0, 0], color='lightgray')
    ax7.set_xlabel('Time [s]')
    ax7.set_ylabel('Res.Lag [m]')
    ax7.grid(True, linestyle='--', color='lightgray')

    # all_stations = np.unique(np.concatenate([obs_data.stations_lum, obs_data.stations_lag]))
    # for st in all_stations:
    #     if st not in station_colors:
    #         station_colors[st] = cmap(len(station_colors) % 10)

    def _is_01T02T(name: str) -> bool:
        s = str(name)
        return ("01T" in s) or ("02T" in s) 
    
    stations_lum_arr = np.asarray(obs_data.stations_lum)

    ref_mask_lum = np.array([_is_01T02T(s) for s in stations_lum_arr])
    if ref_mask_lum.any():
        # Reference pool (only 01T/02T variants e.g., "01T-mir")
        ref_h   = np.asarray(obs_data.height_lum[ref_mask_lum], dtype=float)
        ref_mag = np.asarray(obs_data.absolute_magnitudes[ref_mask_lum], dtype=float)
        ref_lum = np.asarray(obs_data.luminosity[ref_mask_lum], dtype=float)

        # Height span where reference has data
        hmin, hmax = np.nanmin(ref_h), np.nanmax(ref_h)

        # Non-reference stations
        other_mask_lum = ~ref_mask_lum
        other_stations_lum = np.unique(stations_lum_arr[other_mask_lum])

        for other_st in other_stations_lum:
            m = (stations_lum_arr == other_st)
            h_other  = np.asarray(obs_data.height_lum[m], dtype=float)
            mag_other = np.asarray(obs_data.absolute_magnitudes[m], dtype=float)
            lum_other = np.asarray(obs_data.luminosity[m], dtype=float)

            if h_other.size == 0:
                continue

            # Keep only points within the reference height coverage
            keep = (h_other >= hmin) & (h_other <= hmax)
            if not np.any(keep):
                continue

            h_ok   = h_other[keep]
            mag_ok = mag_other[keep]
            lum_ok = lum_other[keep]

            # Pair each kept point to nearest ref height
            idx = _nearest_indices(ref_h, h_ok)

            # Residuals: other - reference at nearest height
            res_mag = mag_ok - ref_mag[idx]
            res_lum = lum_ok - ref_lum[idx]

            ax1.plot(res_mag, h_ok/1000.0, '.', markersize=3,
                    label=f"{other_st} Δmag", color=station_colors[other_st])
            ax5.plot(res_lum, h_ok/1000.0, '.', markersize=3,
                    label=f"{other_st} Δlum", color=station_colors[other_st])
            

        
    # ---- Residuals for VEL and LAG against 01T/02T reference ----
    stations_lag_arr = np.asarray(obs_data.stations_lag)

    ref_mask_lag = np.array([_is_01T02T(s) for s in stations_lag_arr])
    if ref_mask_lag.any():
        ref_h_lag = np.asarray(obs_data.height_lag[ref_mask_lag], dtype=float)
        ref_vel   = np.asarray(obs_data.velocities[ref_mask_lag], dtype=float)
        ref_lag   = np.asarray(obs_data.lag[ref_mask_lag], dtype=float)

        hmin, hmax = np.nanmin(ref_h_lag), np.nanmax(ref_h_lag)

        other_mask_lag = ~ref_mask_lag
        other_stations_lag = np.unique(stations_lag_arr[other_mask_lag])

        for other_st in other_stations_lag:
            m = (stations_lag_arr == other_st)
            h_other   = np.asarray(obs_data.height_lag[m], dtype=float)
            vel_other = np.asarray(obs_data.velocities[m], dtype=float)
            lag_other = np.asarray(obs_data.lag[m], dtype=float)

            if h_other.size == 0:
                continue

            keep = (h_other >= hmin) & (h_other <= hmax)
            if not np.any(keep):
                continue

            h_ok     = h_other[keep]
            vel_ok   = vel_other[keep]
            lag_ok   = lag_other[keep]
            t_ok     = np.asarray(obs_data.time_lag[m], dtype=float)[keep]

            idx = _nearest_indices(ref_h_lag, h_ok)

            res_vel = (vel_ok - ref_vel[idx]) / 1000.0   # km/s
            res_lag = (lag_ok - ref_lag[idx])

            # Plot residuals vs time so traces start only when both have data
            ax6.plot(t_ok, res_vel, '.', markersize=3,
                    label=f"{other_st} Δv", color=station_colors[other_st])
            ax7.plot(t_ok, res_lag, '.', markersize=3,
                    label=f"{other_st} Δlag", color=station_colors[other_st])




    # Save the plot
    print('file saved: '+output_folder +os.sep+ file_name+'_LumLag_EMCCD_CAMO_res_plot.png')
    # fig.savefig(output_folder +os.sep+ file_name +'_LumLag_plot.png', dpi=300)

    # save the figure
    fig.savefig(output_folder +os.sep+ file_name +'_LumLag_EMCCD_CAMO_res_plot.png', 
            bbox_inches='tight',
            pad_inches=0.1,       # a little padding around the edge
            dpi=300)

    # Display the plot
    plt.close(fig)

def diff_plot_CAMO_EMCCD(input_dir, output_dir):    
    # Use the class to find .dynesty, load prior, and decide output folders
    finder = find_dynestyfile_and_priors(input_dir_or_file=input_dir, prior_file="", resume=True, output_dir=output_dir, use_all_cameras=True, pick_position=0)

    for i, (base_name, dynesty_info, prior_path, out_folder) in enumerate(zip(finder.base_names,finder.input_folder_file,finder.priors,finder.output_folders)):
        dynesty_file, pickle_file, bounds, flags_dict, fixed_values = dynesty_info
        print('\n', base_name)
        print(f"Processed {i+1} out of {len(finder.base_names)}")
        obs_data = finder.observation_instance(base_name)
        plot_data_residual_CAMO_EMCCD(obs_data, output_folder=output_dir, file_name=base_name)
        plot_vel_lag_residuals_CAMO_EMCCD(obs_data, output_folder=output_dir, file_name=base_name)


if __name__ == "__main__":

    import argparse
    ### COMMAND LINE ARGUMENTS
    arg_parser = argparse.ArgumentParser(description="Run dynesty with optional .prior file.")
    
    arg_parser.add_argument('--input_dir', metavar='INPUT_PATH', type=str,
                            
        default=r"/srv/public/mvovk/ASTRA/Justin_paper/20221022_075829_skyfit2_ZB/20221022-075828.925593_manEMCCD+CAMO",
        help="Path to walk and find .pickle files.")
    
    arg_parser.add_argument('--output_dir', metavar='OUTPUT_DIR', type=str,
        default=r"",
        help="Output directory, if not given is the same as input_dir.")
    
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

    # # if name is empty set it to the input_dir
    # if cml_args.name == "":
    #     # split base on the os.sep() and get the last element
    #     cml_args.name = cml_args.input_dir.split(os.sep)[-1]
    #     print(f"Setting name to {cml_args.name}")

    diff_plot_CAMO_EMCCD(cml_args.input_dir, cml_args.output_dir)