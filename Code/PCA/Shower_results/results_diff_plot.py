import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import sys

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

def read_csv_files(base_folder, result_type='Real'):
    data_frames = []
    
    for root, dirs, files in os.walk(base_folder):
        for file in files:
            if file.endswith("results.csv"):
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)
                
                subfolder = os.path.basename(root)
                
                # Use regex to find and remove 'n' followed by numbers
                match = re.search(r'n\d+', subfolder)
                if match:
                    result_name = re.sub(r'n\d+', '', subfolder).strip('_')  # Remove 'n' + numbers and trailing underscores
                else:
                    result_name = subfolder
                
                # check if the solution_id column has / or \ and then split it and take the last element
                if '/' in df['solution_id'].iloc[0]:
                    solution_id_file = df['solution_id'].apply(lambda x: x.split('/')[-1])
                elif '\\' in df['solution_id'].iloc[0]:
                    solution_id_file = df['solution_id'].apply(lambda x: x.split('\\')[-1])
                
                # Ensure solution_id exists and parse for '/' or '\\'
                if 'solution_id' in df.columns:
                    df['solution_id'] = df['solution_id'].fillna('')  # Safeguard against NaN values
                    solution_id_file = df['solution_id'].apply(
                        lambda x: x.split('/')[-1] if '/' in x else x.split('\\')[-1] if '\\' in x else x
                    )
                else:
                    solution_id_file = pd.Series([''] * len(df))  # Default empty series if column missing
                
                # delete from result_name 'Results'
                subfolder = subfolder.replace('Results', '')

                # delete from result_name 'Results'
                result_name = result_name.replace('Results', '')

                # Find rows where `result_name` matches the beginning of `solution_id_file`
                idx = solution_id_file.str.startswith(result_name)
                
                # Update `type` column for matching rows
                if 'type' in df.columns:
                    df.loc[idx, 'type'] = result_type
                else:
                    df['type'] = result_type if idx.any() else ''
                
                
                df['result'] = result_name
                df['result_number'] = subfolder

                # put these two column after solution_id column
                cols = df.columns.tolist()
                cols = cols[:1] + cols[-2:] + cols[1:-2]
                df = df[cols]
                
                data_frames.append(df)
    
    combined_df = pd.concat(data_frames, ignore_index=True)
    return combined_df



def PhysicalPropPLOT_results(df_sel_shower_real, output_dir, file_name, save_log=True):
    curr_df_sim_sel = df_sel_shower_real.copy()

    if save_log:
        # check if a file with the name "log"+n_PC_in_PCA+"_"+str(len(df_sel))+"ev.txt" already exist
        if os.path.exists(output_dir + os.sep + "log_" + file_name + "_ConfInterval.txt"):
            # remove the file
            os.remove(output_dir + os.sep + "log_" + file_name + "_ConfInterval.txt")
        sys.stdout = Logger(output_dir, "log_" + file_name + "_ConfInterval.txt")  # _30var_99%_13PC

    # multiply the erosion coeff by 1000000 to have it in km/s
    curr_df_sim_sel['erosion_coeff'] = curr_df_sim_sel['erosion_coeff'] * 1000000
    curr_df_sim_sel['sigma'] = curr_df_sim_sel['sigma'] * 1000000
    curr_df_sim_sel['erosion_energy_per_unit_cross_section'] = curr_df_sim_sel['erosion_energy_per_unit_cross_section'] / 1000000
    curr_df_sim_sel['erosion_energy_per_unit_mass'] = curr_df_sim_sel['erosion_energy_per_unit_mass'] / 1000000

    group_mapping = {
        'Simulation_sel': 'selected',
        'MetSim': 'selected',
        'Real': 'selected',
        'Simulation': 'simulated'
    }
    curr_df_sim_sel['group'] = curr_df_sim_sel['type'].map(group_mapping)

    curr_df_sim_sel['num_group'] = curr_df_sim_sel.groupby('group')['group'].transform('size')
    curr_df_sim_sel['weight'] = 1 / curr_df_sim_sel['num_group']

    curr_df_sim_sel['num_type'] = curr_df_sim_sel.groupby('type')['type'].transform('size')
    curr_df_sim_sel['weight_type'] = 1 / curr_df_sim_sel['num_type']

    curr_sim = curr_df_sim_sel[curr_df_sim_sel['group'] == 'simulated'].copy()

    # delete the last two rows of the curr_df_sim_sel dataframe
    curr_df_sim_sel = curr_df_sim_sel.iloc[:-2]
    

    to_plot = ['mass', 'rho', 'sigma', 'erosion_height_start', 'erosion_coeff', 'erosion_mass_index', 'erosion_mass_min', 'erosion_mass_max', 'erosion_range']
    to_plot_unit = [r'$m_0$ [kg]', r'$\rho$ [kg/m$^3$]', r'$\sigma$ [s$^2$/km$^2$]', r'$h_{e}$ [km]', r'$\eta$ [s$^2$/km$^2$]', r'$s$ [-]', r'log($m_{l}$) [-]', r'log($m_{u}$) [-]', r'log($m_{u}$)-log($m_{l}$) [-]']

    fig, axs = plt.subplots(3, 3)
    axs = axs.flatten()

    print('\\hline')
    # check if the type is MetSim or Real exist in the dataframe
    if 'MetSim' in curr_df_sim_sel['type'].values:
        print('Variables & Metsim & 95\\%CIlow & Mean & Mode & 95\\%CIup \\\\')
    elif 'Real' in curr_df_sim_sel['type'].values:
        print('Variables & Real & 95\\%CIlow & Mean & Mode & 95\\%CIup \\\\')

    # order by 'result_number'
    curr_df_sim_sel = curr_df_sim_sel.sort_values(by='result_number')

    for i in range(9):
        plotvar = to_plot[i]

        if i == 8:
            # Plot only the legend
            axs[i].axis('off')  # Turn off the axis

            # Create custom legend entries
            import matplotlib.patches as mpatches
            from matplotlib.lines import Line2D

            # Define the legend elements
            # prior_patch = mpatches.Patch(color='blue', label='Priors', alpha=0.5, edgecolor='black')
            # sel_events_patch = mpatches.Patch(color='darkorange', label='Selected Events', alpha=0.5, edgecolor='red')

            # Add legend elements for result_number
            result_numbers = curr_df_sim_sel['result_number'].unique()
            colors = sns.color_palette("flare", len(result_numbers))
            
            # Add legend elements for result_number
            legend_elements = [
                mpatches.Patch(color=colors[j], label=result_number, alpha=0.5)
                for j, result_number in enumerate(result_numbers)
            ]

            if 'MetSim' in curr_df_sim_sel['type'].values:
                metsim_line = Line2D([0], [0], color='black', linewidth=2, label='Metsim Solution')
            else:
                metsim_line = Line2D([0], [0], color='green', linestyle='--', linewidth=2, label='Real Solution')
            mode_line = Line2D([0], [0], color='red', linestyle='-.', label='Mode')
            mean_line = Line2D([0], [0], color='blue', linestyle='--', label='Mean')
            legend_elements += [metsim_line, mean_line, mode_line]

            axs[i].legend(handles=legend_elements, loc='upper left', fontsize=5, bbox_to_anchor=(0, 1.2))

            # Remove axes ticks and labels
            axs[i].set_xticks([])
            axs[i].set_yticks([])
            axs[i].set_xlabel('')
            axs[i].set_ylabel('')
            continue  # Skip to next iteration

        if plotvar == 'erosion_mass_min' or plotvar == 'erosion_mass_max':
            # take the log of the erosion_mass_min and erosion_mass_max
            curr_df_sim_sel[plotvar] = np.log10(curr_df_sim_sel[plotvar])
            curr_sim[plotvar] = np.log10(curr_sim[plotvar])

        sns.histplot(curr_df_sim_sel, x=curr_df_sim_sel[plotvar], weights=curr_df_sim_sel['weight'], hue='result_number', ax=axs[i], multiple="stack", palette='flare', bins=20, binrange=[np.min(curr_sim[plotvar]), np.max(curr_sim[plotvar])])
        sns.histplot(curr_df_sim_sel, x=curr_df_sim_sel[plotvar], weights=curr_df_sim_sel['weight'], bins=20, ax=axs[i], fill=False, edgecolor=False, color='r', kde=True, binrange=[np.min(curr_sim[plotvar]), np.max(curr_sim[plotvar])])
        kde_line = axs[i].lines[-1]
        axs[i].lines[-1].remove()

        axs[i].axvline(x=np.mean(curr_df_sim_sel[curr_df_sim_sel['group'] == 'selected'][plotvar]), color='blue', linestyle='--')

        if kde_line is not None:
            # Get the x and y data from the KDE line
            kde_line_Xval = kde_line.get_xdata()
            kde_line_Yval = kde_line.get_ydata()

            # Find the index of the maximum y value (mode)
            max_index = np.argmax(kde_line_Yval)
            # Plot a vertical line at the mode
            axs[i].axvline(x=kde_line_Xval[max_index], color='red', linestyle='-.')

            x_10mode = kde_line_Xval[max_index]
            if plotvar == 'erosion_mass_min' or plotvar == 'erosion_mass_max':
                x_10mode = 10 ** kde_line_Xval[max_index]

        if 'MetSim' in curr_df_sim_sel['type'].values:
            axs[i].axvline(x=curr_df_sim_sel[curr_df_sim_sel['type'] == 'MetSim'][plotvar].values[0], color='k', linewidth=2)
            find_type = 'MetSim'
        elif 'Real' in curr_df_sim_sel['type'].values:
            axs[i].axvline(x=curr_df_sim_sel[curr_df_sim_sel['type'] == 'Real'][plotvar].values[0], color='g', linewidth=2, linestyle='--')
            find_type = 'Real'

        if plotvar == 'erosion_mass_min' or plotvar == 'erosion_mass_max':
            # Convert back from log scale
            curr_df_sim_sel[plotvar] = 10 ** curr_df_sim_sel[plotvar]
            curr_sim[plotvar] = 10 ** curr_sim[plotvar]

        # Calculate percentiles
        sigma_95 = np.percentile(curr_df_sim_sel[plotvar], 95)
        sigma_5 = np.percentile(curr_df_sim_sel[plotvar], 5)

        mean_values_sel = np.mean(curr_df_sim_sel[plotvar])

        axs[i].set_ylabel('Probability')
        axs[i].set_xlabel(to_plot_unit[i])

        # Adjust y-axis limit
        if axs[i].get_ylim()[1] > 1:
            axs[i].set_ylim(0, 1)

        # Remove individual legends
        axs[i].get_legend().remove()

        if i == 0:
            # Adjust x-axis offset text
            axs[i].xaxis.get_offset_text().set_x(1.10)

        if i < 9:
            print('\\hline')
            print(f"{to_plot_unit[i]} & {'{:.4g}'.format(curr_df_sim_sel[curr_df_sim_sel['type'] == find_type][plotvar].values[0])} & {'{:.4g}'.format(sigma_5)} & {'{:.4g}'.format(mean_values_sel)} & {'{:.4g}'.format(x_10mode)} & {'{:.4g}'.format(sigma_95)} \\\\")


    # add the super title
    # plt.suptitle(file_name+' Physical Properties') #  fontsize=16
    plt.tight_layout()
    print('\\hline')

    fig.savefig(output_dir + os.sep + file_name + '_PhysicProp_' + str(len(curr_df_sim_sel)) + 'ev.png', dpi=300)
    plt.close()

    if save_log:
        sys.stdout.close()
        sys.stdout = sys.__stdout__



output_dir = '/home/mvovk/Documents/json_test'
base_folder = '/home/mvovk/Documents/json_test/Results_1000_30'
type_result = 'Real'
result_df = read_csv_files(base_folder, type_result)
# print(result_df)    
# save csv file
# result_df.to_csv('/home/mvovk/Documents/json_test/results_1000.csv', index=False)
# Define the parameter ranges
param_ranges = {
    'erosion_coeff': (0.0, 1/1e6),  # s^2/m^2
    'erosion_mass_index': (1.5, 2.5),
    'erosion_mass_min': (5e-12, 1e-10),  # kg
    'erosion_mass_max': (1e-10, 5e-8),   # kg
    'rho': (100, 1000),  # kg/m^3
    'sigma': (0.008/1e6, 0.03/1e6),  # s^2/m^2
}

# Loop through each result
for result in result_df['result'].unique():
    # Take all the rows with the same result and make a copy
    df_sel_shower_real = result_df[result_df['result'] == result].copy()
    
    # Take the index of the type_result
    idx = df_sel_shower_real[df_sel_shower_real['type'] == type_result].index
    
    # Delete the row with the type_result except the first one
    if len(idx) > 1:
        df_sel_shower_real.drop(idx[1:], inplace=True)
    
    # Extract the row with Real or MetSim values to base MIN and MAX on
    reference_row = df_sel_shower_real[df_sel_shower_real['type'].isin(['Real', 'MetSim'])].iloc[0]
    
    # Initialize MIN and MAX rows as copies of the reference row
    min_row = reference_row.copy()
    max_row = reference_row.copy()
    
    for param, (min_val, max_val) in param_ranges.items():
        min_row[param] = min_val
        max_row[param] = max_val

    # Handle mass
    mass = reference_row['mass']
    order = int(np.floor(np.log10(mass)))
    min_row['mass'] = mass - (10**order)/2
    max_row['mass'] = mass + (10**order)/2

    # Handle erosion height start
    erosion_height_start = reference_row['erosion_height_start']
    min_row['erosion_height_start'] = erosion_height_start - 2
    max_row['erosion_height_start'] = erosion_height_start + 2

    # Assign 'Simulation' type and a 'solution_id'
    min_row['type'] = 'Simulation'
    max_row['type'] = 'Simulation'
    min_row['solution_id'] = 'MIN'
    max_row['solution_id'] = 'MAX'

    # Append the new rows to df_sel_shower_real
    df_sel_shower_real = pd.concat(
        [df_sel_shower_real, pd.DataFrame([min_row, max_row])],
        ignore_index=True
    )

    # substitute to the value of -2 and -1 of  'mass', 'rho', 'sigma', 'erosion_height_start', 'erosion_coeff', 'erosion_mass_index', 'erosion_mass_min', 'erosion_mass_max', 'erosion_range' for each max and min
    df_sel_shower_real.loc[df_sel_shower_real['mass'] == -2, 'mass'] = df_sel_shower_real['mass'].min()
    df_sel_shower_real.loc[df_sel_shower_real['mass'] == -1, 'mass'] = df_sel_shower_real['mass'].max()
    df_sel_shower_real.loc[df_sel_shower_real['rho'] == -2, 'rho'] = df_sel_shower_real['rho'].min()
    df_sel_shower_real.loc[df_sel_shower_real['rho'] == -1, 'rho'] = df_sel_shower_real['rho'].max()
    df_sel_shower_real.loc[df_sel_shower_real['sigma'] == -2, 'sigma'] = df_sel_shower_real['sigma'].min()
    df_sel_shower_real.loc[df_sel_shower_real['sigma'] == -1, 'sigma'] = df_sel_shower_real['sigma'].max()
    df_sel_shower_real.loc[df_sel_shower_real['erosion_height_start'] == -2, 'erosion_height_start'] = df_sel_shower_real['erosion_height_start'].min()
    df_sel_shower_real.loc[df_sel_shower_real['erosion_height_start'] == -1, 'erosion_height_start'] = df_sel_shower_real['erosion_height_start'].max()
    df_sel_shower_real.loc[df_sel_shower_real['erosion_coeff'] == -2, 'erosion_coeff'] = df_sel_shower_real['erosion_coeff'].min()
    df_sel_shower_real.loc[df_sel_shower_real['erosion_coeff'] == -1, 'erosion_coeff'] = df_sel_shower_real['erosion_coeff'].max()
    df_sel_shower_real.loc[df_sel_shower_real['erosion_mass_index'] == -2, 'erosion_mass_index'] = df_sel_shower_real['erosion_mass_index'].min
    df_sel_shower_real.loc[df_sel_shower_real['erosion_mass_index'] == -1, 'erosion_mass_index'] = df_sel_shower_real['erosion_mass_index'].max
    df_sel_shower_real.loc[df_sel_shower_real['erosion_mass_min'] == -2, 'erosion_mass_min'] = df_sel_shower_real['erosion_mass_min'].min()
    df_sel_shower_real.loc[df_sel_shower_real['erosion_mass_min'] == -1, 'erosion_mass_min'] = df_sel_shower_real['erosion_mass_min'].max()
    df_sel_shower_real.loc[df_sel_shower_real['erosion_mass_max'] == -2, 'erosion_mass_max'] = df_sel_shower_real['erosion_mass_max'].min()
    df_sel_shower_real.loc[df_sel_shower_real['erosion_mass_max'] == -1, 'erosion_mass_max'] = df_sel_shower_real['erosion_mass_max'].max()
    df_sel_shower_real.loc[df_sel_shower_real['erosion_range'] == -2, 'erosion_range'] = df_sel_shower_real['erosion_range'].min()
    df_sel_shower_real.loc[df_sel_shower_real['erosion_range'] == -1, 'erosion_range'] = df_sel_shower_real['erosion_range'].max()
    
    # Save or process df_sel_shower_real further
    # df_sel_shower_real.to_csv(f'/home/mvovk/Documents/json_test/{result}_updated.csv', index=False)

    # plot the results
    PhysicalPropPLOT_results(df_sel_shower_real, output_dir, result, save_log=True)

    
