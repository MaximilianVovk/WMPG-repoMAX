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

input_dirfile = r"C:\Users\maxiv\Documents\UWO\Papers\2)ORI-CAP-PER-DRA\Results\ORI"

# take th shower name from the input_dirfile aking the last "\" and the folder name after it
shower_name = input_dirfile.split("\\")[-1]

# the output directory for the latex table is the same as
output_dir_show = r"C:\Users\maxiv\WMPG-repoMAX\Code\DynNestSampl\distrib_plot" #input_dirfile


# Use the class to find .dynesty, load prior, and decide output folders
finder = find_dynestyfile_and_priors(input_dir_or_file=input_dirfile,prior_file="",resume=True,output_dir=input_dirfile,use_all_cameras=False,pick_position=0)

all_label_sets = []  # List to store sets of labels for each file
variables = []  # List to store distributions for each file
flags_dict_total = {}  # Dictionary to store flags for each file
num_meteors = len(finder.base_names)  # Number of meteors
for i, (base_name, dynesty_info, prior_path, out_folder) in enumerate(zip(finder.base_names,finder.input_folder_file,finder.priors,finder.output_folders)):
    dynesty_file, pickle_file, bounds, flags_dict, fixed_values = dynesty_info
    obs_data = finder.observation_instance(base_name)
    obs_data.file_name = pickle_file # update teh file name in the observation data object

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


def approximate_mode_1d(samples):
    """Approximate mode using histogram binning."""
    samples = samples[~np.isnan(samples)]
    if samples.size == 0:
        return np.nan
    hist, bin_edges = np.histogram(samples, bins='auto', density=True)
    idx_max = np.argmax(hist)
    return 0.5 * (bin_edges[idx_max] + bin_edges[idx_max + 1])

def summarize_all_meteors(finder, variables, labels_plot, flags_dict_total):
    all_samples = []

    for i, (base_name, dynesty_info, prior_path, out_folder) in enumerate(
        zip(finder.base_names, finder.input_folder_file, finder.priors, finder.output_folders)):

        dynesty_file, pickle_file, bounds, flags_dict, fixed_values = dynesty_info
        dsampler = dynesty.DynamicNestedSampler.restore(dynesty_file)
        results = dsampler.results

        try:
            logwt = results['logwt']
            samples = results['samples']
        except KeyError:
            continue

        logwt_shifted = logwt - np.max(logwt)
        weights = np.exp(logwt_shifted)
        weights /= np.sum(weights)

        try:
            samples_equal = dyfunc.resample_equal(samples, weights)
        except ValueError:
            samples_equal = np.full((1, len(variables)), np.nan)

        sample_matrix = np.full((samples_equal.shape[0], len(variables)), np.nan)
        var_index = {v: i for i, v in enumerate(flags_dict.keys())}
        for j, var in enumerate(variables):
            if var in var_index:
                sample_matrix[:, j] = samples_equal[:, var_index[var]]

        for j, var in enumerate(variables):
            if np.all(np.isnan(sample_matrix[:, j])):
                continue
            if 'log' in flags_dict_total.get(var, ''):
                sample_matrix[:, j] = 10 ** sample_matrix[:, j]
            if var in ['v_init', 'erosion_height_start', 'erosion_height_change']:
                sample_matrix[:, j] = sample_matrix[:, j] / 1000.0
            if var in ['erosion_coeff', 'sigma', 'erosion_coeff_change', 'erosion_sigma_change']:
                sample_matrix[:, j] = sample_matrix[:, j] * 1e6

        all_samples.append(sample_matrix)

    if len(all_samples) == 0:
        return pd.DataFrame(columns=["Variable", "Label", "Median", "Mean", "Mode", "Low95", "High95"])

    combined = np.vstack(all_samples)

    summary = []
    for i, var in enumerate(variables):
        col = combined[:, i]
        if np.all(np.isnan(col)):
            summary.append((var, labels_plot[i], np.nan, np.nan, np.nan, np.nan, np.nan))
        else:
            summary.append((
                var,
                labels_plot[i],
                np.nanmedian(col),
                np.nanmean(col),
                approximate_mode_1d(col),
                np.nanpercentile(col, 2.5),
                np.nanpercentile(col, 97.5)
            ))

    return pd.DataFrame(summary, columns=["Variable", "Label", "Median", "Mean", "Mode", "Low95", "High95"])

summary_df = summarize_all_meteors(finder, variables, labels_plot, flags_dict_total)
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

    # Footer
    footer = r"    \end{tabular}}"
    footer2 = rf"""    \caption{{Overall posterior summary statistics for {num_meteors} meteors of the {shower_name} shower.}}
    \label{{tab:overall_summary_{shower_name.lower()}}}
\end{{table}}"""

    latex_lines.append(footer)
    latex_lines.append(footer2)

    return "\n".join(latex_lines)

latex_code = summary_to_latex(summary_df, shower_name)
print(latex_code)

# Optional: Save to file
with open(os.path.join(output_dir_show, "posterior_summary_table.tex"), "w") as f:
    f.write(latex_code)






def align_dynesty_samples(dsampler, all_variables, current_flags):
    """
    Aligns dsampler samples to the full list of all variables by padding missing variables with np.nan.
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
    dsampler = dynesty.DynamicNestedSampler.restore(dynesty_file)

    # Align to the union of all variables (padding missing ones with NaN and 0 weights)
    samples_aligned, weights_aligned = align_dynesty_samples(dsampler, variables, flags_dict)

    all_samples.append(samples_aligned)
    all_weights.append(weights_aligned)

# Combine
combined_samples = np.vstack(all_samples)
combined_weights = np.concatenate(all_weights)

# Fill NaNs with neutral values (e.g., median of existing values for that variable)
for i in range(combined_samples.shape[1]):
    col = combined_samples[:, i]
    if np.any(~np.isnan(col)):
        median_val = np.nanmedian(col)
        col[np.isnan(col)] = median_val
    else:
        col[:] = 0.0  # optional fallback if nothing is available

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

# Trace Plots
fig, axes = plt.subplots(ndim, ndim, figsize=(35, 15))
axes = axes.reshape((ndim, ndim))  # reshape axes

span = []
for var in variables:
    if var in bounds_total:
        span.append(tuple(bounds_total[var]))
    else:
        span.append(0.999999426697)  # default 5σ auto range if missing

# Increase spacing between subplots
fg, ax = dyplot.cornerplot(
    combined_results, 
    color='blue', 
    show_titles=True, 
    max_n_ticks=3, 
    quantiles=None, 
    labels=labels_plot,  # Update axis labels
    label_kwargs={"fontsize": 8},  # Reduce axis label size
    title_kwargs={"fontsize": 8},  # Reduce title font size
    title_fmt='.2e',  # Scientific notation for titles
    fig=(fig, axes[:, :ndim])
)
# add a super title
fg.suptitle(shower_name, fontsize=16, fontweight='bold')  # Adjust y for better spacing

# Apply scientific notation and horizontal tick labels
for ax_row in ax:
    for ax_ in ax_row:
        ax_.tick_params(axis='both', labelsize=8, direction='in')

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

# # plot the corner plot
# plt.show()

# save the corner plot
plt.savefig(os.path.join(output_dir_show, "corner_plot.png"), bbox_inches='tight', dpi=300)

