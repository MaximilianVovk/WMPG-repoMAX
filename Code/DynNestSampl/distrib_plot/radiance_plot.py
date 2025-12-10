#!/usr/bin/env python
"""
Walk a root directory of CAMO meteors, find all YYYYMMDD_hhmmss_report.txt,
extract Tj and orbital elements, and then:

- highlight meteors that have a report in the RADIANCE plot (green points),
- optionally also in the velocity vs begin height plot.

Assumptions:
- You already have code that builds:
    file_radiance_rho_dict[base_name] = (
        lg_min_la_sun, bg, rho, lg_lo, lg_hi, bg_lo, bg_hi
    )
  and a matching file_rho_jd_dict[base_name] = (rho, rho_lo, rho_hi, ...)
  from your dynesty outputs.

- GMN trajectory summaries are read using WMPL:
    from wmpl.Formats.WmplTrajectorySummary import loadTrajectorySummaryFast

- The shower is inferred from the `reports_root` path name (contains GEM, ORI, PER, CAP, DRA).
"""

# main.py (inside my_subfolder)
import sys
import os

from matplotlib.lines import Line2D
import numpy as np

from shower_distrb_plot import *

# Add the parent directory to the sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from DynNestSapl_metsim import *

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.stats import gaussian_kde

from wmpl.Formats.WmplTrajectorySummary import loadTrajectorySummaryFast



def infer_shower_name_from_path(path_str):
    u = path_str.upper()
    for tag in ["GEM", "PER", "ORI", "CAP", "DRA"]:
        if tag in u:
            return tag
    # fallback
    return "MIXED"


def infer_shower_iau_no(shower_name):
    """
    Map shower short name to GMN IAU number (as in your original code).
    """
    if "CAP" in shower_name:
        return 1
    if "GEM" in shower_name:
        return 4
    if "PER" in shower_name:
        return 7
    if "ORI" in shower_name:
        return 8
    if "DRA" in shower_name:
        return 9
    return -1


def infer_gmn_files(shower_name):
    """
    Return the GMN traj_summary_monthly_* filenames for the given shower.
    Uses the same mapping you had before.
    """
    if "CAP" in shower_name:
        return ["traj_summary_monthly_202407.txt", "traj_summary_monthly_202408.txt"]
    if "GEM" in shower_name:
        return ["traj_summary_monthly_202412.txt"]
    if "PER" in shower_name:
        return ["traj_summary_monthly_202408.txt"]
    if "ORI" in shower_name:
        return ["traj_summary_monthly_202410.txt"]
    if "DRA" in shower_name:
        return ["traj_summary_monthly_202410.txt"]
    # default
    return ["traj_summary_monthly_202402.txt"]



if __name__ == "__main__":

    import argparse
    import pandas as pd  # used only here for GMN frames

    parser = argparse.ArgumentParser(
        description="Parse *_report.txt to get Tj and highlight those meteors in radiance plots."
    )
    parser.add_argument(
        "--reports_root",
        default=r"C:\Users\maxiv\Documents\UWO\Papers\0.3)Phaethon\Reduction-GEM\CAMO",
        help="Root folder containing CAMO subfolders with YYYYMMDD_hhmmss_report.txt"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for plots; defaults to reports_root"
    )
    parser.add_argument(
        "--gmn-dir",
        default=r"C:\Users\maxiv\Documents\UWO\Papers\2)ORI-CAP-PER-DRA\Results",
        help="Directory with GMN traj_summary_monthly_*.txt / *.pickle (default is your usual path)"
    )

    args = parser.parse_args()

    reports_root = os.path.abspath(args.reports_root)
    output_dir_show = args.output_dir if args.output_dir is not None else reports_root
    gmn_dir = os.path.abspath(args.gmn_dir)

    shower_name = infer_shower_name_from_path(reports_root)
    iau_no = infer_shower_iau_no(shower_name)
    gmn_files = infer_gmn_files(shower_name)

    print(f"Reports root: {reports_root}")
    print(f"Inferred shower: {shower_name} (IAU = {iau_no})")
    print(f"GMN dir: {gmn_dir}")
    print(f"GMN files used: {gmn_files}")
    print(f"Output dir: {output_dir_show}")

    # ------------------------------------------------------------------
    # PLACEHOLDER: build these two dicts from your dynesty / CAMO pipeline
    # ------------------------------------------------------------------

    # Use the class to find .dynesty, load prior, and decide output folders
    finder = find_dynestyfile_and_priors(input_dir_or_file=reports_root,prior_file="",resume=True,output_dir=reports_root,use_all_cameras=True,pick_position=0)

    file_rho_jd_dict = {}
    file_radiance_rho_dict = {}
    file_radiance_rho_dict_helio = {}

    num_meteors = len(finder.base_names)  # Number of meteors
    for i, (base_name, dynesty_info, prior_path, out_folder) in enumerate(zip(finder.base_names, finder.input_folder_file, finder.priors, finder.output_folders)):
        dynesty_file, pickle_file, bounds, flags_dict, fixed_values = dynesty_info
        print('\n', base_name)
        print(f"Processed {i+1} out of {len(finder.base_names)}")
        obs_data = finder.observation_instance(base_name)
        obs_data.file_name = pickle_file  # update the file name in the observation data object
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
        
        # from base_name delete _combined if present
        if "_combined" in base_name:
            base_name = base_name.replace("_combined", "")
        report_path = os.path.join(output_dir, report_file)
        tj, tj_lo, tj_hi, inclin_val, Vg_val, Q_val, q_val, a_val, e_val = extract_tj_from_report(report_path)
        file_rho_jd_dict[base_name] = (0,0,0, tj, tj_lo, tj_hi, inclin_val, Vg_val, Q_val, q_val, a_val, e_val)
        
        lg, lg_lo, lg_hi, bg, bg_lo, bg_hi, la_sun, lg_helio, lg_helio_lo, lg_helio_hi, bg_helio, bg_helio_lo, bg_helio_hi = extract_radiant_and_la_sun(report_path)
        print(f"Ecliptic geocentric (J2000): Lg = {lg}°, Bg = {bg}°")
        print(f"Solar longitude:       La Sun = {la_sun}°")
        lg_min_la_sun = (lg - la_sun)%360
        lg_min_la_sun_helio = (lg_helio - la_sun)%360

        file_radiance_rho_dict[base_name] = (lg_min_la_sun, bg, 0, lg_lo, lg_hi, bg_lo, bg_hi)
        file_radiance_rho_dict_helio[base_name] = (lg_min_la_sun_helio, lg_helio_lo, lg_helio_hi, bg_helio, bg_helio_lo, bg_helio_hi)


    rho_lo = np.array([v[1] for v in file_rho_jd_dict.values()])
    rho_hi = np.array([v[2] for v in file_rho_jd_dict.values()])
    tj = np.array([v[3] for v in file_rho_jd_dict.values()])
    tj_lo = np.array([v[4] for v in file_rho_jd_dict.values()])
    tj_hi = np.array([v[5] for v in file_rho_jd_dict.values()])
    inclin_val = np.array([v[6] for v in file_rho_jd_dict.values()])
    Vg_val = np.array([v[7] for v in file_rho_jd_dict.values()])
    Q_val = np.array([v[8] for v in file_rho_jd_dict.values()])
    q_val = np.array([v[9] for v in file_rho_jd_dict.values()])
    a_val = np.array([v[10] for v in file_rho_jd_dict.values()])
    e_val = np.array([v[11] for v in file_rho_jd_dict.values()])

    # Extract data for plotting
    lg_min_la_sun = np.array([v[0] for v in file_radiance_rho_dict.values()])
    bg = np.array([v[1] for v in file_radiance_rho_dict.values()])
    rho = np.array([v[2] for v in file_radiance_rho_dict.values()])
    lg_lo = np.array([v[3] for v in file_radiance_rho_dict.values()])
    lg_hi = np.array([v[4] for v in file_radiance_rho_dict.values()])
    bg_lo = np.array([v[5] for v in file_radiance_rho_dict.values()])
    bg_hi = np.array([v[6] for v in file_radiance_rho_dict.values()])

    lg_min_la_sun_helio = np.array([v[0] for v in file_radiance_rho_dict_helio.values()])
    lg_helio_lo = np.array([v[1] for v in file_radiance_rho_dict_helio.values()])
    lg_helio_hi = np.array([v[2] for v in file_radiance_rho_dict_helio.values()])
    bg_helio = np.array([v[3] for v in file_radiance_rho_dict_helio.values()])
    bg_helio_lo = np.array([v[4] for v in file_radiance_rho_dict_helio.values()])
    bg_helio_hi = np.array([v[5] for v in file_radiance_rho_dict_helio.values()])


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
        elif "GEM" in shower_name:
            stream_data = loadTrajectorySummaryFast(r"C:\Users\maxiv\Documents\UWO\Papers\2)ORI-CAP-PER-DRA\Results","traj_summary_monthly_202412.txt","traj_summary_monthly_202412.pickle")
            # save the csv_file to a file called: "traj_summary_monthly_202408.csv"
            stream_data.to_csv(r"C:\Users\maxiv\Documents\UWO\Papers\2)ORI-CAP-PER-DRA\Results\traj_summary_monthly_202412.csv", index=False)
            shower_iau_no = 4#"00007"
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
        stream_lor = stream_data[['LAMgeo (deg)', 'BETgeo (deg)', 'Sol lon (deg)','LAMhel (deg)', 'BEThel (deg)','Vgeo (km/s)','HtBeg (km)', 'TisserandJ']].values
        # translate to double precision float
        stream_lor = stream_lor.astype(np.float64)
        # and now compute lg_min_la_sun = (lg - la_sun)%360
        stream_lg_min_la_sun = (stream_lor[:, 0] - stream_lor[:, 2]) % 360
        stream_bg = stream_lor[:, 1]
        stream_lg_min_la_sun_helio = (stream_lor[:, 3] - stream_lor[:, 2]) % 360
        stream_bg_helio = stream_lor[:, 4]
        stream_vgeo = stream_lor[:, 5]
        stream_htbeg = stream_lor[:, 6]
        stream_tj = stream_lor[:, 7]
        # print(f"Found {len(stream_lg_min_la_sun)} stream data points for shower IAU number: {shower_iau_no}")

        if shower_iau_no != -1:
            ############### PLOTTING ###############
            # prepare the data for plotting

            norm = Normalize(vmin=0, vmax=1)
            
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
            elif "GEM" in shower_name:
                print("Plotting GEM shower data...")
                # plt.xlim(xlim[0], 331)
                # plt.ylim(32, 36)
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
                c='limegreen',
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
            
            # annotate each point with its base_name in tiny text
            for base_name, (x, y, z, x_lo, x_hi, y_lo, y_hi) in file_radiance_rho_dict.items():
                plt.annotate(
                    base_name,
                    xy=(x, y),
                    xytext=(30, 5),             # 5 points vertical offset
                    textcoords='offset points',
                    ha='center',
                    va='bottom',
                    fontsize=6,
                    alpha=0.8
                )

            # increase the size of the tick labels
            plt.gca().tick_params(labelsize=15)

            plt.gca().invert_xaxis()

            # # increase the label size
            # cbar = plt.colorbar(scatter, label='$\\rho$ [kg/m$^3$]')
            # # 2. now set the label’s font size and the tick labels’ size
            # cbar.set_label('$\\rho$ [kg/m$^3$]', fontsize=15)
            # cbar.ax.tick_params(labelsize=15)

            plt.xlabel(r'$\lambda_{g} - \lambda_{\odot}$ (J2000)', fontsize=15)
            plt.ylabel(r'$\beta_{g}$ (J2000)', fontsize=15)
            # plt.title('Radiant Distribution of Meteors')
            plt.grid(True)
            plt.savefig(os.path.join(output_dir_show, f"{shower_name}_geo_radiant_distribution_CI.png"), bbox_inches='tight', dpi=300)
            plt.close()

            # plot the size again the rho_corrected with the weights
            # if a label isn't found due to prior duplication/cleaning, fail loudly with context


