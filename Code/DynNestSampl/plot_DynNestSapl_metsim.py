"""
Nested sampling with Dynesty for MetSim meteor data ONLY generate plots and tables
and genreate noise.

Author: Maximilian Vovk
Date: 2025-02-25
"""

from DynNestSapl_metsim import *


if __name__ == "__main__":

    import argparse

    ### COMMAND LINE ARGUMENTS
    arg_parser = argparse.ArgumentParser(description="Run dynesty with optional .prior file.")
    # r"C:\Users\maxiv\WMPG-repoMAX\Code\DynNestSampl\Shower\CAMO\ORI_mode\ORI_mode_CAMO_with_noise.json,C:\Users\maxiv\WMPG-repoMAX\Code\DynNestSampl\Shower\EMCCD\ORI_mode\ORI_mode_EMCCD_with_noise.json,C:\Users\maxiv\WMPG-repoMAX\Code\DynNestSampl\Shower\CAMO\CAP_mode\CAP_mode_CAMO_with_noise.json,C:\Users\maxiv\WMPG-repoMAX\Code\DynNestSampl\Shower\EMCCD\DRA_mode\DRA_mode_EMCCD_with_noise.json,C:\Users\maxiv\WMPG-repoMAX\Code\DynNestSampl\Shower\EMCCD\CAP_mode\CAP_mode_EMCCD_with_noise.json"
    # r"/home/mvovk/WMPG-repoMAX/Code/DynNestSampl/Shower/CAMO/ORI_mode/ORI_mode_CAMO_with_noise.json,/home/mvovk/WMPG-repoMAX/Code/DynNestSampl/Shower/EMCCD/ORI_mode/ORI_mode_EMCCD_with_noise.json,/home/mvovk/WMPG-repoMAX/Code/DynNestSampl/Shower/CAMO/CAP_mode/CAP_mode_CAMO_with_noise.json,/home/mvovk/WMPG-repoMAX/Code/DynNestSampl/Shower/EMCCD/CAP_mode/CAP_mode_EMCCD_with_noise.json,/home/mvovk/WMPG-repoMAX/Code/DynNestSampl/Shower/EMCCD/DRA_mode/DRA_mode_EMCCD_with_noise.json"
    arg_parser.add_argument('input_dir', metavar='INPUT_PATH', type=str,
        default=r"/home/mvovk/WMPG-repoMAX/Code/DynNestSampl/Shower/EMCCD/ORI_mode/ORI_mode_EMCCD_with_noise.json",
        help="Path to walk and find .pickle file or specific single file .pickle or .json file divided by ',' in between.")
    # /home/mvovk/Results/Results_Nested/validation/
    arg_parser.add_argument('--output_dir', metavar='OUTPUT_DIR', type=str,
        default=r"/home/mvovk/Results/Results_Nested/Validation",
        help="Where to store results. If empty, store next to each .dynesty.")
    # /home/mvovk/WMPG-repoMAX/Code/DynNestSampl/stony_meteoroid.prior
    arg_parser.add_argument('--prior', metavar='PRIOR', type=str,
        default=r"",
        help="Path to a .prior file. If blank, we look in the .dynesty folder or default to built-in bounds.")
    
    arg_parser.add_argument('--use_CAMO_data', metavar='USE_CAMO_DATA', type=bool, default=False,
        help="If True, use only CAMO data for lag if present in pickle file, or generate json file with CAMO noise. If False, do not use/generate CAMO data (by default is False).")

    # Parse
    cml_args = arg_parser.parse_args()

    setup_folder_and_run_dynesty(cml_args.input_dir, cml_args.output_dir, cml_args.prior, use_CAMO_data=cml_args.use_CAMO_data, only_plot=True)
    