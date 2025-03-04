"""
Import all the pickle files and get the luminosity of the first fram of all the files

Author: Maximilian Vovk
Date: 2025-03-04
"""

from DynNestSapl_metsim import *

dir_pickle_files = r'C:\Users\maxiv\Documents\UWO\Papers\2)ORI-CAP-PER-DRA\Reductions\DRA'

# Use the class to find .dynesty, load prior, and decide output folders
finder = find_dynestyfile_and_priors(input_dir_or_file=dir_pickle_files)

# check if finder is empty
if not finder.base_names:
    print("No files found in the input directory.")
    sys.exit()

lum_list = []
abs_mag_list = []
base_name_list = []

print(f"Pickle files in {dir_pickle_files}")
for i, (base_name, dynesty_info, prior_path, out_folder) in enumerate(zip(
    finder.base_names,
    finder.input_folder_file,
    finder.priors,
    finder.output_folders
)):
    dynesty_file, pickle_file, bounds, flags_dict, fixed_values = dynesty_info
    obs_data = finder.observation_instance(base_name)  # Get the correct instance
    
    if obs_data:  # Ensure the instance exists
        obs_data.file_name = pickle_file  # Update file name if needed

        lum_list.append(obs_data.luminosity[0])
        abs_mag_list.append(obs_data.absolute_magnitudes[0])
        base_name_list.append(base_name)

        print(f"File {i+1}: {base_name} has 1st frame luminosity {obs_data.luminosity[0]} and absolute magnitude {obs_data.absolute_magnitudes[0]}")
    else:
        print(f"Warning: No observation instance found for {base_name}")



