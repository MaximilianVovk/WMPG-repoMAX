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

        # Read the raw pickle bytes
        with open(dynesty_file, "rb") as f:
            raw = f.read()

        # Encode as Base64 so it’s pure text
        b64 = base64.b64encode(raw).decode("ascii")

        # create the json file name by just replacing the .dynesty with _dynesty.json
        json_file = dynesty_file.replace(".dynesty", "_dynesty.json")
        # Write that string into JSON
        with open(json_file, "w") as f:
            json.dump({"dynesty_b64": b64}, f, indent=2)

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

    # the on that are not variables are the one that were not used in the dynesty run give a np.nan weight to dsampler for those
    all_samples = []
    all_weights = []

    for i, (base_name, dynesty_info, prior_path, out_folder) in enumerate(zip(finder.base_names, finder.input_folder_file, finder.priors, finder.output_folders)):
        dynesty_file, pickle_file, bounds, flags_dict, fixed_values = dynesty_info
        obs_data = finder.observation_instance(base_name)
        obs_data.file_name = pickle_file  # update the file name in the observation data object
        # dsampler = dynesty.DynamicNestedSampler.restore(dynesty_file)
        dsampler = load_dynesty_file(dynesty_file)

        # Align to the union of all variables (padding missing ones with NaN and 0 weights)
        samples_aligned, weights_aligned = align_dynesty_samples(dsampler, variables, flags_dict)

        all_samples.append(samples_aligned)
        all_weights.append(weights_aligned)

    # Combine
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

    combined_results = CombinedResults(combined_samples, combined_weights)

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


    for i, variable in enumerate(variables):
        if 'log' in flags_dict_total[variable]:  
            labels_plot[i] =r"$\log_{10}$(" +labels_plot[i]+")"

    # for j, var in enumerate(variables):
    #     if np.all(np.isnan(combined_samples[:, j])):
    #         continue
    #     # if 'log' in flags_dict_total.get(var, ''):
    #     #     combined_samples[:, j] = 10 ** combined_samples[:, j]
    #     #     labels_plot[j] = r"$\log_{10}($" + labels_plot[j] + r"$)$"
    #     if var in ['v_init', 'erosion_height_start', 'erosion_height_change']:
    #         combined_samples[:, j] = combined_samples[:, j] / 1000.0
    #     if var in ['erosion_coeff', 'sigma', 'erosion_coeff_change', 'erosion_sigma_change']:
    #         combined_samples[:, j] = combined_samples[:, j] * 1e6

    print('saving corner plot...')

    # Define weighted correlation
    def weighted_corr(x, y, w):
        """Weighted Pearson correlation of x and y with weights w."""
        w = np.asarray(w)
        x = np.asarray(x)
        y = np.asarray(y)
        w_sum = w.sum()
        x_mean = (w * x).sum() / w_sum
        y_mean = (w * y).sum() / w_sum
        cov_xy = (w * (x - x_mean) * (y - y_mean)).sum() / w_sum
        var_x  = (w * (x - x_mean)**2).sum() / w_sum
        var_y  = (w * (y - y_mean)**2).sum() / w_sum
        return cov_xy / np.sqrt(var_x * var_y)

    # … your existing prep code …
    fig, axes = plt.subplots(ndim, ndim, figsize=(35, 15))
    axes = axes.reshape((ndim, ndim))

    # call dynesty’s cornerplot
    fg, ax = dyplot.cornerplot(
        combined_results, 
        color='blue',
        show_titles=True,
        max_n_ticks=3,
        quantiles=None,
        labels=labels_plot,
        label_kwargs={"fontsize": 10},
        title_kwargs={"fontsize": 12},
        title_fmt='.2e',
        fig=(fig, axes[:, :ndim])
    )

    # # supertitle, tick formatting, saving …
    # fg.suptitle(shower_name, fontsize=16, fontweight='bold')

    for ax_row in ax:
        for ax_ in ax_row:
            if ax_ is None:
                continue
            ax_.tick_params(axis='both', labelsize=8, direction='in')
            for lbl in ax_.get_xticklabels(): lbl.set_rotation(0)
            for lbl in ax_.get_yticklabels(): lbl.set_rotation(45)
            if len(ax_.xaxis.get_majorticklocs())>0:
                ax_.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.4g'))
            if len(ax_.yaxis.get_majorticklocs())>0:
                ax_.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4g'))

    for i in range(ndim):
        for j in range(ndim):
            if ax[i, j] is None:
                continue
            if j != 0:
                ax[i, j].set_yticklabels([])
            if i != ndim - 1:
                ax[i, j].set_xticklabels([])

    # Overlay weighted correlations in the upper triangle
    samples = combined_results['samples'].T  # shape (ndim, nsamps)
    weights = combined_results.importance_weights()

    cmap = plt.colormaps['coolwarm']
    norm = Normalize(vmin=-1, vmax=1)

    for i in range(ndim):
        for j in range(ndim):
            if j <= i or ax[i, j] is None:
                continue

            panel = ax[i, j]
            x = samples[j]
            y = samples[i]
            corr_w = weighted_corr(x, y, weights)

            color = cmap(norm(corr_w))
            # paint the background patch
            panel.patch.set_facecolor(color)
            panel.patch.set_alpha(1.0)

            # fallback rectangle if needed
            panel.add_patch(
                plt.Rectangle(
                    (0,0), 1, 1,
                    transform=panel.transAxes,
                    facecolor=color,
                    zorder=0
                )
            )

            panel.text(
                0.5, 0.5,
                f"{corr_w:.2f}",
                transform=panel.transAxes,
                ha='center', va='center',
                fontsize=25, color='black'
            )
            panel.set_xticks([]); panel.set_yticks([])
            for spine in panel.spines.values():
                spine.set_visible(False)

    # final adjustments & save
    # fg.subplots_adjust(wspace=0.1, hspace=0.3)
    fg.subplots_adjust(wspace=0.1, hspace=0.3, top=0.978) # Increase spacing between plots
    plt.savefig(os.path.join(output_dir_show, f"{shower_name}_correlation_plot.png"),
                bbox_inches='tight', dpi=300)

    # Build the NxN matrix of weigh_corr_ij
    corr_mat = np.zeros((ndim, ndim))
    for i in range(ndim):
        for j in range(ndim):
            corr_mat[i, j] = weighted_corr(samples[i], samples[j], weights)

    # Wrap it in a DataFrame (so you get row/column labels)
    df_corr = pd.DataFrame(
        corr_mat,
        index=labels_plot,
        columns=labels_plot
    )

    # Create a mask for the strict upper triangle (i<j), diagonal excluded
    mask = np.triu(np.ones(df_corr.shape, dtype=bool), k=1)

    # Keep only those entries
    upper = df_corr.where(mask)

    # Stack into a Series of (row, col) → corr_ij
    pairs = upper.stack()

    # For “top 10” by absolute strength:
    top10 = pairs.sort_values(key=lambda x: x.abs(), ascending=False).head(10)
    print("\nTop 10: highest correlations:")
    print(top10)

    # If you want the “bottom 10” (i.e. the smallest absolute correlations):
    bottom10 = pairs.sort_values(key=lambda x: x.abs(), ascending=True).head(10)
    print("\nBottom 10: lowest correlations:")
    print(bottom10)



if __name__ == "__main__":

    import argparse

    ### COMMAND LINE ARGUMENTS
    arg_parser = argparse.ArgumentParser(description="Run dynesty with optional .prior file.")
    
    arg_parser.add_argument('input_dir', metavar='INPUT_PATH', type=str,
        default=r"C:\Users\maxiv\Documents\UWO\Papers\2)ORI-CAP-PER-DRA\Results\CAMO+EMCCD\ORI",
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
    