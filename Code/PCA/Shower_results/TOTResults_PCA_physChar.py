import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sys
import matplotlib.artist as Artist

# Usage
directory = r'C:\Users\maxiv\Documents\UWO\Papers\2)PCA_ORI-CAP-PER-DRA\Solutions_10000\PER'  # Change this to the directory you want to process
shower = 'PER'

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

def process_files(directory):
    selected_results = []
    simulations = []

    # Walk through the directory and subdirectories
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('_sim_sel_results.csv'):
                file_path = os.path.join(root, file)
                meteor_value = file[:15]
                df = pd.read_csv(file_path)
                df['meteor'] = meteor_value
                selected_results.append(df)
            elif file.endswith('_sim.csv'):
                file_path = os.path.join(root, file)
                meteor_value = file[:15]
                df = pd.read_csv(file_path)
                # df['meteor'] = meteor_value
                simulations.append(df)

    # Concatenate all DataFrames into single DataFrames for each type
    selected_results_df = pd.concat(selected_results, ignore_index=True) if selected_results else pd.DataFrame()
    simulations_df = pd.concat(simulations, ignore_index=True) if simulations else pd.DataFrame()

    return selected_results_df, simulations_df


def find_closest_index(array, values):
    array = np.asarray(array)
    values = np.asarray(values)
    indices = np.searchsorted(array, values, side='left')
    indices = np.clip(indices, 0, len(array) - 1)
    return indices


selected_results_df, simulations_df = process_files(directory)

print(selected_results_df)

print(simulations_df)

# # Optionally save the DataFrames to CSV files
# selected_results_df.to_csv('selected_results.csv', index=False)
# simulations_df.to_csv('simulations.csv', index=False)

print("DataFrames created.")

output_dir = directory

df_sim_shower_small = simulations_df.copy()
df_sel_shower = selected_results_df.copy()

if len(df_sim_shower_small) > 10000:
    df_sim_shower_small = df_sim_shower_small.sample(n=10000)
    if 'MetSim' not in df_sim_shower_small['type'].values:
        df_sim_shower_small = pd.concat([df_sim_shower_small.iloc[[0]], df_sim_shower_small])

log_file = os.path.join(output_dir, f"log_{shower}_CI_PCA.txt")
if os.path.exists(log_file):
    os.remove(log_file)
sys.stdout = Logger(output_dir, f"log_{shower}_CI_PCA.txt")

curr_df_sim_sel = pd.concat([df_sim_shower_small, df_sel_shower], axis=0)

curr_df_sim_sel['erosion_coeff'] = curr_df_sim_sel['erosion_coeff'] * 1000000
curr_df_sim_sel['sigma'] = curr_df_sim_sel['sigma'] * 1000000
curr_df_sim_sel['erosion_energy_per_unit_cross_section'] = curr_df_sim_sel['erosion_energy_per_unit_cross_section'] / 1000000
curr_df_sim_sel['erosion_energy_per_unit_mass'] = curr_df_sim_sel['erosion_energy_per_unit_mass'] / 1000000

group_mapping = {'Simulation_sel': 'selected', 'MetSim': 'simulated', 'Simulation': 'simulated'}
curr_df_sim_sel['group'] = curr_df_sim_sel['type'].map(group_mapping)

curr_df_sim_sel['num_group'] = curr_df_sim_sel.groupby('meteor')['meteor'].transform('size')
curr_df_sim_sel['weight_type'] = 1 / curr_df_sim_sel['num_group']

curr_df_sim_sel['num_type'] = curr_df_sim_sel.groupby('type')['type'].transform('size')
curr_df_sim_sel['weight'] = 1 / curr_df_sim_sel['num_type']

curr_sel = curr_df_sim_sel[curr_df_sim_sel['group'] == 'selected'].copy()

to_plot = ['mass', 'rho', 'sigma', 'erosion_height_start', 'erosion_coeff', 'erosion_mass_index', 'erosion_mass_min', 'erosion_mass_max', 'erosion_range', 'erosion_energy_per_unit_mass', 'erosion_energy_per_unit_cross_section']
# to_plot_unit = ['mass [kg]', 'rho [kg/m^3]', 'sigma [s^2/km^2]', 'erosion height start [km]', 'erosion coeff [s^2/km^2]', 'erosion mass index [-]', 'log eros. mass min [kg]', 'log eros. mass max [kg]', 'log eros. mass range [-]', 'erosion energy per unit mass [MJ/kg]', 'erosion energy per unit cross section [MJ/m^2]', 'erosion energy per unit cross section [MJ/m^2]']
to_plot_unit = [r'log($m_0$) [-]', r'$\rho$ [kg/m$^3$]', r'$\sigma$ [s$^2$/km$^2$]', r'$h_{e}$ [km]', r'$\eta$ [s$^2$/km$^2$]', r'$s$ [-]', r'log($m_{l}$) [-]', r'log($m_{u}$) [-]',r'log($m_{u}$)-log($m_{l}$) [-]', r'$E_{S}$ [MJ/m$^2$]', r'$E_{V}$ [MJ/kg]']


print(shower,'distribution of',len(df_sel_shower['meteor'].unique()),'mteors\n')

print('\\hline')
# print('Variables & 95\\%CIlow & Mode & & Mean & Median & 95\\%CIup \\\\')
print('Variables & 95\\%CIlow & Mode & & Mean & 95\\%CIup \\\\')

fig, axs = plt.subplots(4, 3, figsize=(15, 15)) 
# from 2 numbers to one numbr for the subplot axs
axs = axs.flatten()

# ii_densest=0        
for i in range(12):

    if i == 11:
        # Plot only the legend
        axs[i].axis('off')  # Turn off the axis

        # Create custom legend entries
        import matplotlib.patches as mpatches
        from matplotlib.lines import Line2D

        # Define the legend elements
        # prior_patch = mpatches.Patch(color='blue', label='Priors', alpha=0.5, edgecolor='black')
        # sel_events_patch = mpatches.Patch(color='darkorange', label='Selected Events', alpha=0.5, edgecolor='red')

        # Add legend elements for result_number
        result_numbers = curr_df_sim_sel['meteor'].unique()
        colors = sns.color_palette("muted", len(result_numbers))
        
        # Add legend elements for result_number
        legend_elements = [
            mpatches.Patch(color=colors[j], label=result_number, alpha=0.8) # , alpha=0.5
            for j, result_number in enumerate(result_numbers)
        ]

        # if 'MetSim' in curr_df_sim_sel['type'].values:
        #     metsim_line = Line2D([0], [0], color='black', linewidth=2, label='Metsim Solution')
        # elif 'Real' in curr_df_sim_sel['type'].values:
        #     metsim_line = Line2D([0], [0], color='green', linestyle='--', linewidth=2, label='Real Solution')
        mode_line = Line2D([0], [0], color='red', linestyle='-.', label='Mode')
        mean_line = Line2D([0], [0], color='blue', linestyle='--', label='Mean')
        legend_elements += [mean_line, mode_line]

        axs[i].legend(handles=legend_elements, loc='upper left') # , fontsize=5

        # Remove axes ticks and labels
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        axs[i].set_xlabel('')
        axs[i].set_ylabel('')
        continue  # Skip to next iteration
    else:
        # put legendoutside north
        plotvar=to_plot[i]


    if plotvar == 'mass' or plotvar == 'erosion_mass_min' or plotvar == 'erosion_mass_max':

        # sns.histplot(curr_df_sim_sel, x=np.log10(curr_df_sim_sel[plotvar]), weights=curr_df_sim_sel['weight'], hue='group', ax=axs[i], palette='bright', bins=20, binrange=[np.log10(np.min(curr_df_sim_sel[plotvar])),np.log10(np.max(curr_df_sim_sel[plotvar]))])
        # # add the kde to the plot as a probability density function
        sns.histplot(curr_sel, x=np.log10(curr_sel[plotvar]), weights=curr_sel['weight'], bins=20, ax=axs[i],  multiple="stack", fill=False, edgecolor=False, color='r', kde=True, binrange=[np.log10(np.min(curr_df_sim_sel[plotvar])),np.log10(np.max(curr_df_sim_sel[plotvar]))])
        # axs[i].get_legend().remove()
        # sns.histplot(curr_df_sim_sel, x=curr_df_sim_sel[plotvar], weights=curr_df_sim_sel['weight'],hue='solution_id_dist', ax=axs[i], multiple="stack", kde=True, bins=20, binrange=[np.min(df_sel_save[plotvar]),np.max(df_sel_save[plotvar])])
        sns.histplot(curr_df_sim_sel, x=np.log10(curr_df_sim_sel[plotvar]), weights=curr_df_sim_sel['weight'],hue='meteor', ax=axs[i], palette="muted", multiple="stack", bins=20, binrange=[np.log10(np.min(curr_df_sim_sel[plotvar])),np.log10(np.max(curr_df_sim_sel[plotvar]))])

        kde_line = axs[i].lines[-1]
        # activate the grid
        axs[i].grid(True)

        # find the mean and median
        mean = np.mean(np.log10(curr_sel[plotvar]))
        # median = np.median(np.log10(curr_sel[plotvar]))

    else:

        # if plotvar == 'rho':

        #     # sns.histplot(curr_df_sim_sel, x=curr_df_sim_sel[plotvar], weights=curr_df_sim_sel['weight'], hue='group', ax=axs[i], palette='bright', bins=20, binrange=[np.min(curr_df_sim_sel[plotvar]),np.max(curr_df_sim_sel[plotvar])])
        #     # # add the kde to the plot as a probability density function
        #     sns.histplot(curr_sel, x=curr_sel[plotvar], weights=curr_sel['weight'], bins=20, ax=axs[i],  multiple="stack", fill=False, edgecolor=False, color='r', kde=True, binrange=[0, 2000])
        #     # axs[i].get_legend().remove()
        #     # sns.histplot(curr_df_sim_sel, x=curr_df_sim_sel[plotvar], weights=curr_df_sim_sel['weight'],hue='solution_id_dist', ax=axs[i], multiple="stack", kde=True, bins=20, binrange=[np.min(df_sel_save[plotvar]),np.max(df_sel_save[plotvar])])
        #     sns.histplot(curr_df_sim_sel, x=curr_df_sim_sel[plotvar], weights=curr_df_sim_sel['weight'], hue='meteor', ax=axs[i], multiple="stack", bins=20, binrange=[0, 2000])
            
        # else:
            
        # sns.histplot(curr_df_sim_sel, x=curr_df_sim_sel[plotvar], weights=curr_df_sim_sel['weight'], hue='group', ax=axs[i], palette='bright', bins=20, binrange=[np.min(curr_df_sim_sel[plotvar]),np.max(curr_df_sim_sel[plotvar])])
        # # add the kde to the plot as a probability density function
        sns.histplot(curr_sel, x=curr_sel[plotvar], weights=curr_sel['weight'], bins=20, ax=axs[i],  multiple="stack", fill=False, edgecolor=False, color='r', kde=True, binrange=[np.min(curr_df_sim_sel[plotvar]),np.max(curr_df_sim_sel[plotvar])])
        # axs[i].get_legend().remove()
        # sns.histplot(curr_df_sim_sel, x=curr_df_sim_sel[plotvar], weights=curr_df_sim_sel['weight'],hue='solution_id_dist', ax=axs[i], multiple="stack", kde=True, bins=20, binrange=[np.min(df_sel_save[plotvar]),np.max(df_sel_save[plotvar])])
        sns.histplot(curr_df_sim_sel, x=curr_df_sim_sel[plotvar], weights=curr_df_sim_sel['weight'], hue='meteor', palette="muted", ax=axs[i], multiple="stack", bins=20, binrange=[np.min(curr_df_sim_sel[plotvar]),np.max(curr_df_sim_sel[plotvar])])
        
        kde_line = axs[i].lines[-1]
        # activate the grid
        axs[i].grid(True)

        # find the mean and median
        mean = np.mean(curr_sel[plotvar])
        # median = np.median(curr_sel[plotvar])


    axs[i].set_ylabel('probability')
    axs[i].set_xlabel(to_plot_unit[i])

    if i==0:
        # place the xaxis exponent in the bottom right corner
        axs[i].xaxis.get_offset_text().set_x(1.10)
        
    sigma_95 = np.percentile(curr_sel[plotvar], 95)
    sigma_84 = np.percentile(curr_sel[plotvar], 84.13)
    sigma_15 = np.percentile(curr_sel[plotvar], 15.87)
    sigma_5 = np.percentile(curr_sel[plotvar], 5)

    kde_line_Xval = kde_line.get_xdata()
    kde_line_Yval = kde_line.get_ydata()


    max_index = np.argmax(kde_line_Yval)

    # if i != 8:
    #     # axs[i].plot(kde_line_Xval[max_index], kde_line_Yval[max_index], 'ro')
    #     # put a vertical line at the kde_line_Xval[max_index]
    #     axs[i].axvline(x=kde_line_Xval[max_index], color='red', linestyle='-.')
    # size of the line 2
    # axs[i].axvline(x=kde_line_Xval[max_index], color='g', linestyle='--', linewidth=2, label='Real')
    axs[i].axvline(x=kde_line_Xval[max_index], color='red', linestyle='-.', linewidth=3, label='Mode')

    if i != 11:    # # delete from the plot the axs[i].lines[-1]
        axs[i].lines[-2].remove()
        axs[i].get_legend().remove()
    else:
        # add to the legend the -. line as mode
        # axs[i].legend(['Real','Mode'], loc='upper right')
        axs[i].lines[-1].remove()

    # plot the mean and median
    axs[i].axvline(x=mean, color='blue', linestyle='--', linewidth=3, label='Mean')
    # axs[i].axvline(x=median, color='orange', linestyle='--', linewidth=2, label='Median')

    x_10mode = kde_line_Xval[max_index]
    if plotvar == 'mass' or plotvar == 'erosion_mass_min' or plotvar == 'erosion_mass_max':
        x_10mode = 10**kde_line_Xval[max_index]
        mean = 10**mean
        # median = 10**median
    if plotvar == 'mass':
        to_plot_unit[i] = r'$m_0$ [kg]'
    if plotvar == 'erosion_mass_min':
        to_plot_unit[i] = r'$m_{l}$ [kg]'
    if plotvar == 'erosion_mass_max':
        to_plot_unit[i] = r'$m_{u}$ [kg]'


    if i < 11:
        print('\\hline')
        # print(f"{to_plot_unit[i]} & {'{:.4g}'.format(sigma_5)} & {'{:.4g}'.format(x_10mode)} & {'{:.4g}'.format(mean)} & {'{:.4g}'.format(median)} & {'{:.4g}'.format(sigma_95)} \\\\")
        print(f"{to_plot_unit[i]} & {'{:.4g}'.format(sigma_5)} & {'{:.4g}'.format(x_10mode)} & {'{:.4g}'.format(mean)} & {'{:.4g}'.format(sigma_95)} \\\\")

print('\\hline')

# Close the Logger to ensure everything is written to the file STOP COPY in TXT file
sys.stdout.close()

# Reset sys.stdout to its original value if needed
sys.stdout = sys.__stdout__

# # make backgound of the legend is white
# axs[i].legend().get_frame().set_facecolor('white')

# add the super title
fig.suptitle(f'{shower}', fontsize=16)

fig.tight_layout()
plt.savefig(os.path.join(output_dir, f"{shower}_CI_shower.png"), dpi=300)
# plt.show()
plt.close()