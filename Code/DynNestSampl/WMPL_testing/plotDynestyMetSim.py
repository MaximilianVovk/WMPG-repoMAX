"""
Nested sampling with Dynesty for MetSim meteor data ONLY generate plots and tables
and genreate noise.

Author: Maximilian Vovk
Date: 2025-02-25
"""

from DynestyMetSim import *


if __name__ == "__main__":

    import argparse

    ### COMMAND LINE ARGUMENTS
    arg_parser = argparse.ArgumentParser(description="Run dynesty with optional .prior file.")
    
    arg_parser.add_argument('--input_dir', metavar='INPUT_PATH', type=str,
        default=r"C:\Users\maxiv\Documents\UWO\Papers\0.4)Wake\test\20191023_091225", 
        help="Path to walk and find .pickle file or specific single file .pickle or .json file divided by ',' in between.")
    
    arg_parser.add_argument('--output_dir', metavar='OUTPUT_DIR', type=str,
        default=r"",
        help="Where to store results. If empty, store next to each .dynesty.")
    
    arg_parser.add_argument('--prior', metavar='PRIOR', type=str,
        default=r"",
        help="Path to a .prior file. If blank, we look in the .dynesty folder or default to built-in bounds.")
    
    arg_parser.add_argument('--extraprior', metavar='EXTRAPRIOR', type=str, 
        default=r"",
        help="Path to an .extraprior file these are used to add more FragmentationEntry or diferent types of fragmentations. " \
        "If blank, no extraprior file will be used so will only use the prior file.")
    
    arg_parser.add_argument('-all','--all_cameras',
        help="If active use all data, if not only CAMO data for lag if present in pickle file, " \
        "or generate json file with CAMO noise. If False, do not use/generate CAMO data (by default is False).",
        action="store_true")

    # Parse
    cml_args = arg_parser.parse_args()

    setupDirAndRunDynesty(cml_args.input_dir, 
        cml_args.output_dir, cml_args.prior, use_all_cameras=cml_args.all_cameras, 
        extraprior_file=cml_args.extraprior, save_backup=False, 
        only_plot=True)
    