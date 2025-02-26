"""
Nested sampling with Dynesty for MetSim meteor data ONLY generate plots and tables
and genreate noise.

Author: Maximilian Vovk
Date: 2025-02-25
"""

import sys
import os
import shutil
from DynNestSapl_metsim import *


if __name__ == "__main__":

    import argparse

    ### COMMAND LINE ARGUMENTS
    arg_parser = argparse.ArgumentParser(description="Run dynesty with optional .prior file.")
    # r"C:\Users\maxiv\WMPG-repoMAX\Code\DynNestSampl\Shower\CAMO\ORI_mode\ORI_mode_CAMO_with_noise.json,C:\Users\maxiv\WMPG-repoMAX\Code\DynNestSampl\Shower\EMCCD\ORI_mode\ORI_mode_EMCCD_with_noise.json,C:\Users\maxiv\WMPG-repoMAX\Code\DynNestSampl\Shower\CAMO\CAP_mode\CAP_mode_CAMO_with_noise.json,C:\Users\maxiv\WMPG-repoMAX\Code\DynNestSampl\Shower\EMCCD\DRA_mode\DRA_mode_EMCCD_with_noise.json,C:\Users\maxiv\WMPG-repoMAX\Code\DynNestSampl\Shower\EMCCD\CAP_mode\CAP_mode_EMCCD_with_noise.json"
    # r"/home/mvovk/WMPG-repoMAX/Code/DynNestSampl/Shower/CAMO/ORI_mode/ORI_mode_CAMO_with_noise.json,/home/mvovk/WMPG-repoMAX/Code/DynNestSampl/Shower/EMCCD/ORI_mode/ORI_mode_EMCCD_with_noise.json,/home/mvovk/WMPG-repoMAX/Code/DynNestSampl/Shower/CAMO/CAP_mode/CAP_mode_CAMO_with_noise.json,/home/mvovk/WMPG-repoMAX/Code/DynNestSampl/Shower/EMCCD/CAP_mode/CAP_mode_EMCCD_with_noise.json,/home/mvovk/WMPG-repoMAX/Code/DynNestSampl/Shower/EMCCD/DRA_mode/DRA_mode_EMCCD_with_noise.json"
    arg_parser.add_argument('--input_dir', metavar='INPUT_PATH', type=str,
        default=r"C:\Users\maxiv\Documents\UWO\Papers\2)ORI-CAP-PER-DRA\Validation\ORI_mode_EMCCD_with_noise_dynesty\ORI_mode_EMCCD_with_noise.json",
        help="Path to walk and find .pickle file or specific single file .pickle or .json file divided by ',' in between.")
    # /home/mvovk/Results/Results_Nested/validation/
    arg_parser.add_argument('--output_dir', metavar='OUTPUT_DIR', type=str,
        default=r"C:\Users\maxiv\Documents\UWO\Papers\2)ORI-CAP-PER-DRA\validation",
        help="Where to store results. If empty, store next to each .dynesty.")
    # /home/mvovk/WMPG-repoMAX/Code/DynNestSampl/stony_meteoroid.prior
    arg_parser.add_argument('--prior', metavar='PRIOR', type=str,
        default=r"",
        help="Path to a .prior file. If blank, we look in the .dynesty folder or default to built-in bounds.")
    
    arg_parser.add_argument('--use_CAMO_data', metavar='USE_CAMO_DATA', type=bool, default=False,
        help="If True, use only CAMO data for lag if present in pickle file, or generate json file with CAMO noise. If False, do not use/generate CAMO data (by default is False).")

    # Parse
    cml_args = arg_parser.parse_args()

    # Optional: suppress warnings
    # warnings.filterwarnings('ignore')

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
            resume=True,
            output_dir=cml_args.output_dir,
            use_CAMO_data=cml_args.use_CAMO_data
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
            dynesty_file, bounds, flags_dict, fixed_values = dynesty_info
            obs_data = finder.observation_instance()
            print("--------------------------------------------------")
            # check if a file with the name "log"+n_PC_in_PCA+"_"+str(len(df_sel))+"ev.txt" already exist
            if os.path.exists(out_folder+os.sep+"log_"+base_name+".txt"):
                # remove the file
                os.remove(out_folder+os.sep+"log_"+base_name+".txt")
            sys.stdout = Logger(out_folder,"log_"+base_name+".txt") # 
            print(f"Meteor:", base_name)
            print("  File name:    ", obs_data.file_name)
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
            # Run the dynesty sampler
            os.makedirs(out_folder, exist_ok=True)
            plot_data_with_residuals_and_real(obs_data, output_folder=out_folder, file_name=base_name)

            # if prior_path is not in the output directory and is not "" then copy the prior_path to the output directory
            if prior_path != "":
                # check if there is a prior file with the same name in the output_folder
                prior_file_output = os.path.join(out_folder,os.path.basename(prior_path))
                if not os.path.exists(prior_file_output):
                    shutil.copy(prior_path, out_folder)
                    print("prior file copied to output folder:", prior_file_output)
            # check if obs_data.file_name is not in the output directory
            if not os.path.exists(os.path.join(out_folder,os.path.basename(obs_data.file_name))) and os.path.isfile(obs_data.file_name):
                shutil.copy(obs_data.file_name, out_folder)
                print("observation file copied to output folder:", os.path.join(out_folder,os.path.basename(obs_data.file_name)))
            elif not os.path.isfile(obs_data.file_name):
                print("original observation file not found, not copied:",obs_data.file_name)
            
            if os.path.isfile(dynesty_file): 
                print("Only plotting requested.")
                dsampler = dynesty.DynamicNestedSampler.restore(dynesty_file)
                # dsampler = dynesty.DynamicNestedSampler.restore(dynesty_file)
                plot_dynesty(dsampler.results, obs_data, flags_dict, fixed_values, out_folder, base_name)

            else:
                print("No generate dynesty plots, dynasty file not found:",dynesty_file)
