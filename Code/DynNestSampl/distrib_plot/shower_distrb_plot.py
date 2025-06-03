"""
Import all the pickle files and get the dynesty files distribution

Author: Maximilian Vovk
Date: 2025-04-16
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

def shower_distrb_plot(input_dirfile, output_dir_show, shower_name):
    """
    Function to plot the distribution of the parameters from the dynesty files and save them as a table in LaTeX format.
    """
    # Use the class to find .dynesty, load prior, and decide output folders
    finder = find_dynestyfile_and_priors(input_dir_or_file=input_dirfile,prior_file="",resume=True,output_dir=input_dirfile,use_all_cameras=False,pick_position=0)

    all_label_sets = []  # List to store sets of labels for each file
    variables = []  # List to store distributions for each file
    flags_dict_total = {}  # Dictionary to store flags for each file
    num_meteors = len(finder.base_names)  # Number of meteors
    for i, (base_name, dynesty_info, prior_path, out_folder) in enumerate(zip(finder.base_names,finder.input_folder_file,finder.priors,finder.output_folders)):
        dynesty_file, pickle_file, bounds, flags_dict, fixed_values = dynesty_info
        obs_data = finder.observation_instance(base_name)
        obs_data.file_name = pickle_file # update the file name in the observation data object

        # # Read the raw pickle bytes
        # with open(dynesty_file, "rb") as f:
        #     raw = f.read()

        # # Encode as Base64 so it’s pure text
        # b64 = base64.b64encode(raw).decode("ascii")

        # # create the json file name by just replacing the .dynesty with _dynesty.json
        # json_file = dynesty_file.replace(".dynesty", "_dynesty.json")
        # # Write that string into JSON
        # with open(json_file, "w") as f:
        #     json.dump({"dynesty_b64": b64}, f, indent=2)

        # save the lenght of the flags_dict to check if it is the same for all meteors
        # check if len(flags_dict.keys()) > len(variables) to avoid index error
        if len(flags_dict.keys()) > len(variables):
            variables = list(flags_dict.keys())
            flags_dict_total = flags_dict.copy()
            bounds_total = bounds.copy()


    # keep them in the same order distribution_list
    print(f"Shared labels: {variables}")

    ndim = len(variables)

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
        'noise_lum': r"$\varepsilon_{lum}$ [W]"
    }

    # Mapping of original variable names to LaTeX-style labels
    variable_map_plot = {
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
        'erosion_height_change': r"$h_{e2}$ [m]",
        'erosion_coeff_change': r"$\eta_{2}$ [kg/J]",
        'erosion_rho_change': r"$\rho_{2}$ [kg/m$^3$]",
        'erosion_sigma_change': r"$\sigma_{2}$ [kg/J]",
        'noise_lag': r"$\varepsilon_{lag}$ [m]",
        'noise_lum': r"$\varepsilon_{lum}$ [W]"
    }

    # check if there are variables in the flags_dict that are not in the variable_map
    for variable in variables:
        if variable not in variable_map:
            print(f"Warning: {variable} not found in variable_map")
            # Add the variable to the map with a default label
            variable_map[variable] = variable
    labels = [variable_map[variable] for variable in variables]

    for variable in variables:
        if variable not in variable_map_plot:
            print(f"Warning: {variable} not found in variable_map")
            # Add the variable to the map with a default label
            variable_map_plot[variable] = variable
    labels_plot = [variable_map_plot[variable] for variable in variables]


    def align_dynesty_samples(dsampler, all_variables, current_flags):
        """
        Aligns dsampler samples to the full list of all variables by padding missing variables with 0
        Weights remain unchanged for non-missing dimensions.
        """
        samples = dsampler.results['samples']
        weights = dsampler.results.importance_weights()
        n_samples = samples.shape[0]

        # Create mapping of existing variables in current run
        flag_keys = list(current_flags.keys())
        flag_index = {v: i for i, v in enumerate(flag_keys)}

        # Prepare padded samples with NaNs for missing variables
        padded_samples = np.full((n_samples, len(all_variables)), np.nan)

        # # create a float array full of zeros (or use np.nan if you prefer)
        # padded_samples = np.zeros((n_samples, len(all_variables)), dtype=float)

        for j, var in enumerate(all_variables):
            if var in flag_index:
                padded_samples[:, j] = samples[:, flag_index[var]]

        return padded_samples, weights


    def extract_radiant_and_la_sun(report_path):
        """
        Returns:
        lg_mean, lg_err_lo,
        bg_mean, bg_err,
        la_sun_mean

        lg_err_lo/bg_err are the '+/-' values if present, else 0.0.
        """

        # storage
        lg = bg = la_sun = None
        lg_err_lo  = lg_err_hi = bg_err_lo  = bg_err_hi = None
        in_ecl = False

        # regexes for Lg/Bg with CI
        re_lg_ci = re.compile(r'^\s*Lg\s*=\s*([+-]?\d+\.\d+)[^[]*\[\s*([+-]?\d+\.\d+)\s*,\s*([+-]?\d+\.\d+)\s*\]')
        re_bg_ci = re.compile(r'^\s*Bg\s*=\s*([+-]?\d+\.\d+)[^[]*\[\s*([+-]?\d+\.\d+)\s*,\s*([+-]?\d+\.\d+)\s*\]')
        # look for "Lg = 246.70202 +/- 0.46473"
        re_lg_pm  = re.compile(r'^\s*Lg\s*=\s*([+-]?\d+\.\d+)\s*\+/-\s*([0-9.]+)')
        re_bg_pm  = re.compile(r'^\s*Bg\s*=\s*([+-]?\d+\.\d+)\s*\+/-\s*([0-9.]+)')
        # fallback plain values
        re_lg_val = re.compile(r'^\s*Lg\s*=\s*([+-]?\d+\.\d+)')
        re_bg_val = re.compile(r'^\s*Bg\s*=\s*([+-]?\d+\.\d+)')
        # solar longitude
        re_lasun  = re.compile(r'^\s*La Sun\s*=\s*([+-]?\d+\.\d+)')

        with open(report_path, 'r', encoding='utf-8') as f:
            for line in f:
                s = line.strip()

                # enter the block
                if s.startswith('Radiant (ecliptic geocentric'):
                    in_ecl = True
                    continue

                if in_ecl:
                    # blank line → exit
                    if not s:
                        in_ecl = False
                    else:
                        # try 95CI first
                        m = re_lg_ci.match(line)
                        if m:
                            lg, lg_err_lo, lg_err_hi = map(float, m.groups())
                        else:
                            # try ± after
                            m = re_lg_pm.match(line)
                            if m:
                                lg, lg_err_lo = float(m.group(1)), float(m.group(2))
                            else:
                                # then plain
                                m = re_lg_val.match(line)
                                if m and lg is None:
                                    lg = float(m.group(1))
                        # try 95CI first
                        m = re_bg_ci.match(line)
                        if m:
                            bg, bg_err_lo, bg_err_hi = map(float, m.groups())
                        else:
                            m = re_bg_pm.match(line)
                            if m:
                                bg, bg_err_lo = float(m.group(1)), float(m.group(2))
                            else:
                                m = re_bg_val.match(line)
                                if m and bg is None:
                                    bg = float(m.group(1))

                # always grab La Sun
                if la_sun is None:
                    m = re_lasun.match(line)
                    if m:
                        la_sun = float(m.group(1))

                # stop if we have all three
                if lg is not None and bg is not None and la_sun is not None:
                    break

        if lg is None or bg is None:
            raise RuntimeError(f"Couldn t find Lg/Bg in {report_path!r}")
        if la_sun is None:
            raise RuntimeError(f"Couldn t find La Sun in {report_path!r}")
        if lg_err_lo is None:
            lg_err_lo = lg
            lg_err_hi = lg
        if bg_err_lo is None:
            bg_err_lo = bg
            bg_err_hi = bg
        if lg_err_hi is None:
            lg_err_hi = lg + abs(lg_err_lo)
            lg_err_lo = lg - abs(lg_err_lo)
        if bg_err_hi is None:
            bg_err_hi = bg + abs(bg_err_lo)
            bg_err_lo = bg - abs(bg_err_lo)
        
        print(f"Radiant: Lg = {lg}° 95CI [{lg_err_lo:.3f}°, {lg_err_hi:.3f}°], Bg = {bg}° 95CI [{bg_err_lo:.3f}°, {bg_err_hi:.3f}°]")
        lg_lo = (lg - lg_err_lo)/1.96
        lg_hi = (lg_err_hi - lg)/1.96
        bg_lo = (bg_err_hi - bg)/1.96
        bg_hi = (bg - bg_err_lo)/1.96
        print(f"Error range: Lg = {lg}° ± {lg_lo:.3f}° / {lg_hi:.3f}°, Bg = {bg}° ± {bg_lo:.3f}° / {bg_hi:.3f}°")

        return lg, lg_lo, lg_hi, bg, bg_lo, bg_hi, la_sun


    def summarize_from_cornerplot(results, variables, labels_plot, flags_dict_total, smooth=0.02):
        """
        Summarize dynesty results, using the sample of max weight as the mode.
        """
        samples = results.samples               # shape (nsamps, ndim)
        weights = results.importance_weights()  # shape (nsamps,)

        # normalize weights
        w = weights.copy()
        w /= np.sum(w)

        # find the single sample index with highest weight
        mode_idx = np.nanargmax(w)   # index of peak-weight sample
        mode_raw = samples[mode_idx] # array shape (ndim,)

        rows = []
        for i, (var, lab) in enumerate(zip(variables, labels_plot)):
            x = samples[:, i].astype(float)
            # mask out NaNs
            mask = ~np.isnan(x)
            x_valid = x[mask]
            w_valid = w[mask]
            if x_valid.size == 0:
                rows.append((var, lab, *([np.nan]*5)))
                continue
            # renormalize
            w_valid /= np.sum(w_valid)

            # weighted quantiles
            low95, med, high95 = _quantile(x_valid,
                                        [0.025, 0.5, 0.975],
                                        weights=w_valid)
            # weighted mean
            mean_raw = np.sum(x_valid * w_valid)
            # simple mode from max-weight sample
            mode_value = mode_raw[i]

            # mode via corner logic
            lo, hi = np.min(x), np.max(x)
            if isinstance(smooth, int):
                hist, edges = np.histogram(x, bins=smooth, weights=w, range=(lo,hi))
            else:
                nbins = int(round(10. / smooth))
                hist, edges = np.histogram(x, bins=nbins, weights=w, range=(lo,hi))
                hist = norm_kde(hist, 10.0)
            centers = 0.5 * (edges[1:] + edges[:-1])
            mode_Ndim = centers[np.argmax(hist)]

            # now apply your log & unit transforms *after* computing stats
            def transform(v):
                if 'log' in flags_dict_total.get(var, ''):
                    v = 10**v
                if var in ['v_init',
                        'erosion_height_start',
                        'erosion_height_change']:
                    v = v / 1e3
                if var in ['erosion_coeff',
                        'sigma',
                        'erosion_coeff_change',
                        'erosion_sigma_change']:
                    v = v * 1e6
                return v

            rows.append((
                var,
                lab,
                transform(low95),
                transform(mode_value),
                transform(mode_Ndim),
                transform(mean_raw),
                transform(med),
                transform(high95),
            ))

        return pd.DataFrame(
            rows,
            columns=["Variable","Label","Low95","Mode","Mode_{Ndim}","Mean","Median","High95"]
        )


    # Create Results-like object for cornerplot
    class CombinedResults:
        def __init__(self, samples, weights):
            self.samples = samples
            self.weights = weights

        def __getitem__(self, key):
            if key == 'samples':
                return self.samples
            raise KeyError(f"Key '{key}' not found.")

        def importance_weights(self):
            return self.weights


    # the on that are not variables are the one that were not used in the dynesty run give a np.nan weight to dsampler for those
    all_samples = []
    all_weights = []

    # base_name, lg_min_la_sun, bg, rho
    file_radiance_rho_dict = {}

    for i, (base_name, dynesty_info, prior_path, out_folder) in enumerate(zip(finder.base_names, finder.input_folder_file, finder.priors, finder.output_folders)):
        dynesty_file, pickle_file, bounds, flags_dict, fixed_values = dynesty_info
        print(base_name)
        obs_data = finder.observation_instance(base_name)
        obs_data.file_name = pickle_file  # update the file name in the observation data object
        dsampler = dynesty.DynamicNestedSampler.restore(dynesty_file)
        # dsampler = load_dynesty_file(dynesty_file)

        # Align to the union of all variables (padding missing ones with NaN and 0 weights)
        samples_aligned, weights_aligned = align_dynesty_samples(dsampler, variables, flags_dict)
        
        
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

        report_path = os.path.join(output_dir, report_file)
        lg, lg_lo, lg_hi, bg, bg_lo, bg_hi, la_sun = extract_radiant_and_la_sun(report_path)
        print(f"Ecliptic geocentric (J2000): Lg = {lg}°, Bg = {bg}°")
        print(f"Solar longitude:       La Sun = {la_sun}°")
        lg_min_la_sun = (lg - la_sun)%360

        combined_results_meteor = CombinedResults(samples_aligned, weights_aligned)

        summary_df_meteor = summarize_from_cornerplot(
        combined_results_meteor,
        variables,
        labels,
        flags_dict_total
        )

        # delete from base_name _combined if it exists
        if '_combined' in base_name:
            base_name = base_name.replace('_combined', '')
        file_radiance_rho_dict[base_name] = (lg_min_la_sun, bg, summary_df_meteor['Median'].values[variables.index('rho')], lg_lo, lg_hi, bg_lo, bg_hi)

        all_samples.append(samples_aligned)
        all_weights.append(weights_aligned)

    print("saving radiance plot...")

    # Extract data for plotting
    lg_min_la_sun = np.array([v[0] for v in file_radiance_rho_dict.values()])
    bg = np.array([v[1] for v in file_radiance_rho_dict.values()])
    rho = np.array([v[2] for v in file_radiance_rho_dict.values()])
    lg_lo = np.array([v[3] for v in file_radiance_rho_dict.values()])
    lg_hi = np.array([v[4] for v in file_radiance_rho_dict.values()])
    bg_lo = np.array([v[5] for v in file_radiance_rho_dict.values()])
    bg_hi = np.array([v[6] for v in file_radiance_rho_dict.values()])

    # print(lg_lo, lg_hi, bg_lo, bg_hi)







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
            csv_file_1 = loadTrajectorySummaryFast(r"C:\Users\maxiv\Documents\UWO\Papers\2)ORI-CAP-PER-DRA\Results","traj_summary_monthly_202407.txt","traj_summary_monthly_202407.csv")
            # save the csv_file to a file called: "traj_summary_monthly_202408.csv"
            csv_file_1.to_csv(r"C:\Users\maxiv\Downloads\traj_summary_monthly_202407.csv", index=False)
            csv_file_2 = loadTrajectorySummaryFast(r"C:\Users\maxiv\Documents\UWO\Papers\2)ORI-CAP-PER-DRA\Results","traj_summary_monthly_202408.txt","traj_summary_monthly_202408.csv")
            # save the csv_file to a file called: "traj_summary_monthly_202408.csv"
            csv_file_2.to_csv(r"C:\Users\maxiv\Downloads\traj_summary_monthly_202408.csv", index=False)
            # extend the in csv_file
            stream_data = pd.concat([csv_file_1, csv_file_2], ignore_index=True)
            shower_iau_no = 1#"00001"
        elif "PER" in shower_name:
            stream_data = loadTrajectorySummaryFast(r"C:\Users\maxiv\Documents\UWO\Papers\2)ORI-CAP-PER-DRA\Results","traj_summary_monthly_202408.txt","traj_summary_monthly_202408.csv")
            # save the csv_file to a file called: "traj_summary_monthly_202408.csv"
            stream_data.to_csv(r"C:\Users\maxiv\Downloads\traj_summary_monthly_202408.csv", index=False)
            shower_iau_no = 7#"00007"
        elif "ORI" in shower_name: 
            stream_data = loadTrajectorySummaryFast(r"C:\Users\maxiv\Documents\UWO\Papers\2)ORI-CAP-PER-DRA\Results","traj_summary_monthly_202410.txt","traj_summary_monthly_202410.csv")
            # save the csv_file to a file called: "traj_summary_monthly_202408.csv"
            stream_data.to_csv(r"C:\Users\maxiv\Downloads\traj_summary_monthly_202410.csv", index=False)
            shower_iau_no = 8#"00008"
        elif "DRA" in shower_name:  
            stream_data = loadTrajectorySummaryFast(r"C:\Users\maxiv\Documents\UWO\Papers\2)ORI-CAP-PER-DRA\Results","traj_summary_monthly_202410.txt","traj_summary_monthly_202410.csv")
            # save the csv_file to a file called: "traj_summary_monthly_202408.csv"
            stream_data.to_csv(r"C:\Users\maxiv\Downloads\traj_summary_monthly_202410.csv", index=False)
            shower_iau_no = 9#"00009"
        else:
            shower_iau_no = -1
        
        if shower_iau_no != -1:
            print(f"Filtering stream data for shower IAU number: {shower_iau_no}")
            # filter the stream_data for the shower_iau_no
            stream_data = stream_data[stream_data['IAU (No)'] == shower_iau_no]
            print(f"Found {len(stream_data)} stream data points for shower IAU number: {shower_iau_no}")
            # # and take the one that have activity " annual "
            # stream_data = stream_data[stream_data['activity'].str.contains("annual", case=False, na=False)]
            # print(f"Found {len(stream_data)} stream data points for shower IAU number: {shower_iau_no} with activity 'annual'")
            # extract all LoR	S_LoR	LaR
            stream_lor = stream_data[['LAMgeo (deg)', 'BETgeo (deg)', 'Sol lon (deg)']].values
            # translate to double precision float
            stream_lor = stream_lor.astype(np.float64)
            # and now compute lg_min_la_sun = (lg - la_sun)%360
            stream_lg_min_la_sun = (stream_lor[:, 0] - stream_lor[:, 2]) % 360
            stream_bg = stream_lor[:, 1]
            print(f"Found {len(stream_lg_min_la_sun)} stream data points for shower IAU number: {shower_iau_no}")

    # after you’ve built your rho array:
    norm = Normalize(vmin=rho.min(), vmax=rho.max())
    # cmap = cm.viridis

    plt.figure(figsize=(8, 6))
    # your stream data arrays
    x = stream_lg_min_la_sun
    y = stream_bg

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

    # then draw points on top, at zorder=2 # jet
    scatter = plt.scatter(
        lg_min_la_sun, bg,
        c=rho,
        cmap='viridis',
        norm=norm,
        s=30,
        zorder=2
    )

    # increase the size of the tick labels
    plt.gca().tick_params(labelsize=15)

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


    # increase the label size
    cbar = plt.colorbar(scatter, label='Median density (kg/m$^3$)')
    # 2. now set the label’s font size and the tick labels’ size
    cbar.set_label('Median density (kg/m$^3$)', fontsize=15)
    cbar.ax.tick_params(labelsize=15)
    # now chack if stream_lg_min_la_sun and stream_bg are not empty

    # invert the x axis
    plt.gca().invert_xaxis()

    plt.xlabel(r'$\lambda_{g} - \lambda_{\odot}$ (J2000)', fontsize=15)
    plt.ylabel(r'$\beta_{g}$ (J2000)', fontsize=15)
    # plt.title('Radiant Distribution of Meteors')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir_show, f"{shower_name}_radiant_distribution_CI.png"), bbox_inches='tight', dpi=300)
    plt.close()

    # Combine all the samples and weights into a single array
    combined_samples = np.vstack(all_samples)
    combined_weights = np.concatenate(all_weights)

    # use a fixed seed for reproducibility
    rng = np.random.default_rng(seed=42)

    # for each variable-column:
    for i in range(combined_samples.shape[1]):
        col = combined_samples[:, i]
        miss = np.isnan(col)
        if not miss.any():
            continue

        # all the valid entries & their weights
        vals = col[~miss]
        wts  = combined_weights[~miss].astype(float)
        if wts.sum() > 0:
            wts /= wts.sum()   # normalize
            # draw replacements with the same weighted distribution
            fill = rng.choice(vals, size=miss.sum(), replace=True, p=wts)
        else:
            # fallback to unweighted draw
            fill = rng.choice(vals, size=miss.sum(), replace=True)

        col[miss] = fill
 
    # Create a CombinedResults object for the combined samples
    combined_results = CombinedResults(combined_samples, combined_weights)

    summary_df = summarize_from_cornerplot(
        combined_results,
        variables,
        labels,
        flags_dict_total
    )


    print(summary_df.to_string(index=False))

    def summary_to_latex(summary_df, shower_name="ORI"):
        latex_lines = []

        header = r"""\begin{table}[htbp]
        \centering
        \renewcommand{\arraystretch}{1.2}
        \setlength{\tabcolsep}{4pt}
        \resizebox{\textwidth}{!}{
        \begin{tabular}{|l|c|c|c|c|c|}
        \hline
        \textbf{Parameter} & \textbf{2.5CI} & \textbf{Mode} & \textbf{Mean} & \textbf{Median} & \textbf{97.5CI} \\
        \hline"""
        latex_lines.append(header)

        # Format each row
        for _, row in summary_df.iterrows():
            param = row["Label"]
            low = f"{row['Low95']:.4g}" if not np.isnan(row['Low95']) else "---"
            mode = f"{row['Mode']:.4g}" if not np.isnan(row['Mode']) else "---"
            mean = f"{row['Mean']:.4g}" if not np.isnan(row['Mean']) else "---"
            median = f"{row['Median']:.4g}" if not np.isnan(row['Median']) else "---"
            high = f"{row['High95']:.4g}" if not np.isnan(row['High95']) else "---"

            line = f"    {param} & {low} & {mode} & {mean} & {median} & {high} \\\\"
            latex_lines.append(line)
            latex_lines.append("    \\hline")

        # if there is _ in the shower_name put a \
        shower_name_plot = shower_name.replace("_", "\\_")
        # Footer
        footer = r"    \end{tabular}}"
        footer2 = rf"""    \caption{{Overall posterior summary statistics for {num_meteors} meteors of the {shower_name_plot} shower.}}
        \label{{tab:overall_summary_{shower_name.lower()}}}
    \end{{table}}"""

        latex_lines.append(footer)
        latex_lines.append(footer2)

        return "\n".join(latex_lines)

    latex_code = summary_to_latex(summary_df, shower_name)
    print(latex_code)

    print("Saving LaTeX table...")
    # Save to file
    with open(os.path.join(output_dir_show, shower_name+"_posterior_summary_table.tex"), "w") as f:
        f.write(latex_code)


    combined_samples_copy_plot = combined_samples.copy()
    labels_plot_copy_plot = labels.copy()
    for j, var in enumerate(variables):
        if np.all(np.isnan(combined_samples_copy_plot[:, j])):
            continue
        if 'log' in flags_dict_total.get(var, '') and not var in ['erosion_mass_min', 'erosion_mass_max']:
            combined_samples_copy_plot[:, j] = 10 ** combined_samples_copy_plot[:, j]
        if not 'log' in flags_dict_total.get(var, '') and var in ['m_init']:
            combined_samples_copy_plot[:, j] = np.log10(combined_samples_copy_plot[:, j])
        if var in ['m_init','erosion_mass_min', 'erosion_mass_max']:
            labels_plot_copy_plot[j] =r"$\log_{10}$(" +labels_plot_copy_plot[j]+")"
        if var in ['v_init', 'erosion_height_start', 'erosion_height_change']:
            combined_samples_copy_plot[:, j] = combined_samples_copy_plot[:, j] / 1000.0
        if var in ['erosion_coeff', 'sigma', 'erosion_coeff_change', 'erosion_sigma_change']:
            combined_samples_copy_plot[:, j] = combined_samples_copy_plot[:, j] * 1e6


    for i, variable in enumerate(variables):
        if 'log' in flags_dict_total[variable]:  
            labels_plot[i] =r"$\log_{10}$(" +labels_plot[i]+")"


    print('saving distribution plot...')

    # Extract from combined_results
    samples = combined_samples_copy_plot
    # samples = combined_results.samples
    weights = combined_results.importance_weights()
    w = weights / np.sum(weights)
    ndim = samples.shape[1]

    # Plot grid settings
    ncols = 5
    nrows = math.ceil(ndim / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 2.5 * nrows))
    axes = axes.flatten()

    # Define smoothing value
    smooth = 0.02  # or pass it as argument

    for i in range(ndim):
        ax = axes[i]
        x = samples[:, i].astype(float)
        mask = ~np.isnan(x)
        x_valid = x[mask]
        w_valid = w[mask]

        if x_valid.size == 0:
            ax.axis('off')
            continue

        # Compute histogram
        lo, hi = np.min(x_valid), np.max(x_valid)
        if isinstance(smooth, int):
            hist, edges = np.histogram(x_valid, bins=smooth, weights=w_valid, range=(lo, hi))
        else:
            nbins = int(round(10. / smooth))
            hist, edges = np.histogram(x_valid, bins=nbins, weights=w_valid, range=(lo, hi))
            hist = norm_kde(hist, 10.0)  # dynesty-style smoothing

        centers = 0.5 * (edges[1:] + edges[:-1])

        # Fill under the curve
        ax.fill_between(centers, hist, color='blue', alpha=0.6)

        # ax.plot(centers, hist, color='blue')
        ax.set_yticks([])
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))

        # Set label + quantile title
        row = summary_df.iloc[i]
        label = row["Label"]
        median = row["Median"]
        low = row["Low95"]
        high = row["High95"]
        minus = median - low
        plus = high - median

        if variables[i] in ['erosion_mass_min', 'erosion_mass_max','m_init']: # 'log' in flags_dict_total.get(variables[i], '') and 
            # put a dashed blue line at the median
            ax.axvline(np.log10(median), color='blue', linestyle='--', linewidth=1.5)
            # put a dashed Blue line at the 2.5 and 97.5 percentiles
            ax.axvline(np.log10(low), color='blue', linestyle='--', linewidth=1.5)
            ax.axvline(np.log10(high), color='blue', linestyle='--', linewidth=1.5)
            
        else:
            # put a dashed blue line at the median
            ax.axvline(median, color='blue', linestyle='--', linewidth=1.5)
            # put a dashed Blue line at the 2.5 and 97.5 percentiles
            ax.axvline(low, color='blue', linestyle='--', linewidth=1.5)
            ax.axvline(high, color='blue', linestyle='--', linewidth=1.5)

        fmt = lambda v: f"{v:.4g}" if np.isfinite(v) else "---"
        title = rf"{label} = {fmt(median)}$^{{+{fmt(plus)}}}_{{-{fmt(minus)}}}$"
        ax.set_title(title, fontsize=16)
        ax.set_xlabel(labels_plot_copy_plot[i], fontsize=20)
        # increase the size of the tick labels
        ax.tick_params(axis='x', labelsize=15)

    # Remove unused axes
    for j in range(ndim, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(output_dir_show, f"{shower_name}_distrib_plot.png"),
                bbox_inches='tight', dpi=300)
    plt.close(fig)
    

    ### CORNER PLOT ###
    # # takes forever, so run it last

    # print('saving corner plot...')

    # # Define weighted correlation
    # def weighted_corr(x, y, w):
    #     """Weighted Pearson correlation of x and y with weights w."""
    #     w = np.asarray(w)
    #     x = np.asarray(x)
    #     y = np.asarray(y)
    #     w_sum = w.sum()
    #     x_mean = (w * x).sum() / w_sum
    #     y_mean = (w * y).sum() / w_sum
    #     cov_xy = (w * (x - x_mean) * (y - y_mean)).sum() / w_sum
    #     var_x  = (w * (x - x_mean)**2).sum() / w_sum
    #     var_y  = (w * (y - y_mean)**2).sum() / w_sum
    #     return cov_xy / np.sqrt(var_x * var_y)

    # # … your existing prep code …
    # fig, axes = plt.subplots(ndim, ndim, figsize=(35, 15))
    # axes = axes.reshape((ndim, ndim))

    # # call dynesty’s cornerplot
    # fg, ax = dyplot.cornerplot(
    #     combined_results, 
    #     color='blue',
    #     show_titles=True,
    #     max_n_ticks=3,
    #     quantiles=None,
    #     labels=labels_plot,
    #     label_kwargs={"fontsize": 10},
    #     title_kwargs={"fontsize": 12},
    #     title_fmt='.2e',
    #     fig=(fig, axes[:, :ndim])
    # )

    # # # supertitle, tick formatting, saving …
    # # fg.suptitle(shower_name, fontsize=16, fontweight='bold')

    # for ax_row in ax:
    #     for ax_ in ax_row:
    #         if ax_ is None:
    #             continue
    #         ax_.tick_params(axis='both', labelsize=8, direction='in')
    #         for lbl in ax_.get_xticklabels(): lbl.set_rotation(0)
    #         for lbl in ax_.get_yticklabels(): lbl.set_rotation(45)
    #         if len(ax_.xaxis.get_majorticklocs())>0:
    #             ax_.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.4g'))
    #         if len(ax_.yaxis.get_majorticklocs())>0:
    #             ax_.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4g'))

    # for i in range(ndim):
    #     for j in range(ndim):
    #         if ax[i, j] is None:
    #             continue
    #         if j != 0:
    #             ax[i, j].set_yticklabels([])
    #         if i != ndim - 1:
    #             ax[i, j].set_xticklabels([])

    # # Overlay weighted correlations in the upper triangle
    # samples = combined_results['samples'].T  # shape (ndim, nsamps)
    # weights = combined_results.importance_weights()

    # cmap = plt.colormaps['coolwarm']
    # norm = Normalize(vmin=-1, vmax=1)

    # for i in range(ndim):
    #     for j in range(ndim):
    #         if j <= i or ax[i, j] is None:
    #             continue

    #         panel = ax[i, j]
    #         x = samples[j]
    #         y = samples[i]
    #         corr_w = weighted_corr(x, y, weights)

    #         color = cmap(norm(corr_w))
    #         # paint the background patch
    #         panel.patch.set_facecolor(color)
    #         panel.patch.set_alpha(1.0)

    #         # fallback rectangle if needed
    #         panel.add_patch(
    #             plt.Rectangle(
    #                 (0,0), 1, 1,
    #                 transform=panel.transAxes,
    #                 facecolor=color,
    #                 zorder=0
    #             )
    #         )

    #         panel.text(
    #             0.5, 0.5,
    #             f"{corr_w:.2f}",
    #             transform=panel.transAxes,
    #             ha='center', va='center',
    #             fontsize=25, color='black'
    #         )
    #         panel.set_xticks([]); panel.set_yticks([])
    #         for spine in panel.spines.values():
    #             spine.set_visible(False)

    # # final adjustments & save
    # # fg.subplots_adjust(wspace=0.1, hspace=0.3)
    # fg.subplots_adjust(wspace=0.1, hspace=0.3, top=0.978) # Increase spacing between plots
    # plt.savefig(os.path.join(output_dir_show, f"{shower_name}_correlation_plot.png"),
    #             bbox_inches='tight', dpi=300)
    # plt.close(fig)

    # print('saving correlation matrix...')

    # # Build the NxN matrix of weigh_corr_ij
    # corr_mat = np.zeros((ndim, ndim))
    # for i in range(ndim):
    #     for j in range(ndim):
    #         corr_mat[i, j] = weighted_corr(samples[i], samples[j], weights)

    # # Wrap it in a DataFrame (so you get row/column labels)
    # df_corr = pd.DataFrame(
    #     corr_mat,
    #     index=labels_plot,
    #     columns=labels_plot
    # )

    # # Save to CSV (or TSV, whichever you prefer)
    # outpath = os.path.join(
    #     output_dir_show, f"{shower_name}_weighted_correlation_matrix.csv"
    # )
    # df_corr.to_csv(outpath, float_format="%.4f")
    # print(f"Saved weighted correlation matrix to:\n  {outpath}")

    # # Create a mask for the strict upper triangle (i<j), diagonal excluded
    # mask = np.triu(np.ones(df_corr.shape, dtype=bool), k=1)

    # # Keep only those entries
    # upper = df_corr.where(mask)

    # # Stack into a Series of (row, col) → corr_ij
    # pairs = upper.stack()

    # # For “top 10” by absolute strength:
    # top10 = pairs.sort_values(key=lambda x: x.abs(), ascending=False).head(10)
    # print("\nTop 10: highest correlations:")
    # print(top10)

    # # If you want the “bottom 10” (i.e. the smallest absolute correlations):
    # bottom10 = pairs.sort_values(key=lambda x: x.abs(), ascending=True).head(10)
    # print("\nBottom 10: lowest correlations:")
    # print(bottom10)



if __name__ == "__main__":

    import argparse

    ### COMMAND LINE ARGUMENTS
    arg_parser = argparse.ArgumentParser(description="Run dynesty with optional .prior file.")
    
    arg_parser.add_argument('--input_dir', metavar='INPUT_PATH', type=str,
        default=r"C:\Users\maxiv\Documents\UWO\Papers\2)ORI-CAP-PER-DRA\Results\CAMO+EMCCD\CAP_radiance",
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

    # if name is empty set it to the input_dir
    if cml_args.name == "":
        # split base on the os.sep() and get the last element
        cml_args.name = cml_args.input_dir.split(os.sep)[-1]
        print(f"Setting name to {cml_args.name}")

    shower_distrb_plot(cml_args.input_dir, cml_args.output_dir, cml_args.name)
    