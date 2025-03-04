"""
Import all the pickle files and get the luminosity of the first fram of all the files

Author: Maximilian Vovk
Date: 2025-02-25
"""

from DynNestSapl_metsim import *

dir_pickle_files = '/home/mavovk/Documents/Research/Projects/2021-02-25_DynNestSampl/Code/DynNestSampl/pickle_files/'
print(f"Looking for pickle files in {dir_pickle_files}")
# Use the class to find .dynesty, load prior, and decide output folders
finder = find_dynestyfile_and_priors(input_dir_or_file=dir_pickle_files)

# check if finder is empty
if not finder.base_names:
    print("No files found in the input directory.")
    sys.exit()

lum_list = []
abs_mag_list = []
base_name_list = []

# Each discovered or created .dynesty is in input_folder_file
# with its matching prior info
for i, (base_name, dynesty_info, prior_path, out_folder) in enumerate(zip(
    finder.base_names,
    finder.input_folder_file,
    finder.priors,
    finder.output_folders
)):
    dynesty_file, pickle_file, bounds, flags_dict, fixed_values = dynesty_info
    obs_data = finder.observation_instance()
    obs_data.file_name = pickle_file # update teh file name in the observation_data object

    # take the luminosity of the first frame and the absolute magnitude
    lum_list.append(obs_data.luminosity[0])
    abs_mag_list.append(obs_data.absolute_magnitudes[0])
    base_name_list.append(base_name)
    print(f"File {i+1}: {base_name} has 1st frame luminosity {obs_data.luminosity[0]} and absolute magnitude {obs_data.absolute_magnitudes[0]}")




