"""
Nested sampling with Dynesty for MetSim meteor data run dynsty and plot everithing using MPIPool 
in order to run dynesty simulaions with  multiple node cluster.

Author: Maximilian Vovk
Date: 2025-04-01
"""

from DynNestSapl_metsim import *
from schwimmbad import MPIPool

if __name__ == "__main__":

    print("Using MPI for parallelization of multiple nodes")
    # Create the MPIPool at the top
    pool_MPI = MPIPool()

    # If *not* master, all worker processes wait and exit right here
    if not pool_MPI.is_master():
        pool_MPI.wait()
        sys.exit(0)

    import argparse

    ### COMMAND LINE ARGUMENTS
    arg_parser = argparse.ArgumentParser(description="Run dynesty with optional .prior file.")
    
    arg_parser.add_argument('input_dir', metavar='INPUT_PATH', type=str,
        default=r"/home/mvovk/WMPG-repoMAX/Code/DynNestSampl/Shower_testcase/EMCCD/ORI_mode/EMCCD_ORI_mode_with_noise.json,/home/mvovk/WMPG-repoMAX/Code/DynNestSampl/Shower_testcase/EMCCD/ORI_mean/EMCCD_ORI_mean_with_noise.json,/home/mvovk/WMPG-repoMAX/Code/DynNestSampl/Shower_testcase/CAMO/ORI_mode/CAMO_ORI_mode_with_noise.json,/home/mvovk/WMPG-repoMAX/Code/DynNestSampl/Shower_testcase/CAMO/ORI_mean/CAMO_ORI_mean_with_noise.json,/home/mvovk/WMPG-repoMAX/Code/DynNestSampl/Shower_testcase/EMCCD/CAP_mean/EMCCD_CAP_mean_with_noise.json,/home/mvovk/WMPG-repoMAX/Code/DynNestSampl/Shower_testcase/EMCCD/CAP_mode/EMCCD_CAP_mode_with_noise.json,/home/mvovk/WMPG-repoMAX/Code/DynNestSampl/Shower_testcase/CAMO/CAP_mean/CAMO_CAP_mean_with_noise.json,/home/mvovk/WMPG-repoMAX/Code/DynNestSampl/Shower_testcase/CAMO/CAP_mode/CAMO_CAP_mode_with_noise.json,/home/mvovk/WMPG-repoMAX/Code/DynNestSampl/Shower_testcase/EMCCD/DRA_mean/EMCCD_DRA_mean_with_noise.json,/home/mvovk/WMPG-repoMAX/Code/DynNestSampl/Shower_testcase/EMCCD/DRA_mode/EMCCD_DRA_mode_with_noise.json",
        help="Path to walk and find .pickle file or specific single file .pickle or .json file divided by ',' in between.")
    
    arg_parser.add_argument('--output_dir', metavar='OUTPUT_DIR', type=str,
        default=r"",
        help="Where to store results. If empty, store next to each .dynesty.")
    
    arg_parser.add_argument('--prior', metavar='PRIOR', type=str,
        default=r"",
        help="Path to a .prior file. If blank, we look in the .dynesty folder or default to built-in bounds.")

    arg_parser.add_argument('-all','--all_cameras',
        help="If active use all data, if not only CAMO data for lag if present in pickle file, or generate json file with CAMO noise. If False, do not use/generate CAMO data (by default is False).",
        action="store_true")

    arg_parser.add_argument('-new','--new_dynesty',
        help="If active restart a new dynesty run if not resume from existing .dynesty if found. If False, create a new version.",
        action="store_false")
    
    arg_parser.add_argument('-plot','--only_plot',
        help="If active only plot the results of the dynesty run, if not run dynesty.", 
        action="store_true")

    arg_parser.add_argument('--cores', metavar='CORES', type=int, default=None,
        help="Number of cores to use. Default = all available.")


    # Optional: suppress warnings
    # warnings.filterwarnings('ignore')

    # Parse
    cml_args = arg_parser.parse_args()

    setup_folder_and_run_dynesty(cml_args.input_dir, cml_args.output_dir, cml_args.prior, cml_args.new_dynesty, cml_args.all_cameras, cml_args.only_plot, cml_args.cores, pool_MPI)

    # Close the pool if using MPI
    if pool_MPI is not None:
        pool_MPI.close()
        # pool.join()

    print("\nDONE: Completed processing of all files in the input directory.\n")    
    