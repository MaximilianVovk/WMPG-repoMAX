from DynestyMetSim import *


setupDirAndCheckData


###############################################################################
if __name__ == "__main__":

    import argparse

    ### COMMAND LINE ARGUMENTS
    arg_parser = argparse.ArgumentParser(description="Run dynesty with optional .prior file.")
    
    arg_parser.add_argument('input_dir', metavar='INPUT_PATH', type=str,
        help="Path to walk and find .pickle file or specific single file .pickle or .json file."
        "If you want multiple specific folder or files just divided them by ',' in between.")
    
    arg_parser.add_argument('--output_dir', metavar='OUTPUT_DIR', type=str,
        default=r"",
        help="Where to store results. If empty, store in the input directory.")
    
    arg_parser.add_argument('--prior', metavar='PRIOR', type=str,
        default=r"",
        help="Path to a .prior file. If blank, we look in the .dynesty folder for other .prior files. " \
        "If no data given and none found not present resort to default built-in bounds.")

    arg_parser.add_argument('--extraprior', metavar='EXTRAPRIOR', type=str, 
        default=r"",
        help="Path to an .extraprior file these are used to add more FragmentationEntry or diferent types of fragmentations. " \
        "If blank, no extraprior file will be used so will only use the prior file.")

    arg_parser.add_argument('-all','--all_cameras',
        help="If active use all data, if not only CAMO data for lag if present in pickle file. " \
        "If False, use CAMO data only for deceleration (by default is False). " \
        "When gnerating json simulations filr if False create a combination EMCCD CAMO data and if True EMCCD only",
        action="store_true")

    arg_parser.add_argument('-new','--new_dynesty',
        help="If active restart a new dynesty run if not resume from existing .dynesty if found. " \
        "If False, create a new dynesty version.",
        action="store_false")
    
    arg_parser.add_argument('-NoBackup','--not_backup',
        help="Run all the simulation agin at th end saves the weighted mass bulk density and save a back with all the data" \
        "and creates the distribution plot takes, in general 10 more minute or more base on the number of cores available.",
        action="store_false")
    
    arg_parser.add_argument('-plot','--only_plot',
        help="If active only plot the results of the dynesty run, if not run dynesty and then plot all when finish.", 
        action="store_true")

    arg_parser.add_argument('--pick_pos', metavar='PICK_POSITION_REAL', type=int, default=0,
        help="corretion for pick postion in the meteor frame raging from from 0 to 1, " \
        "for leading edge picks is 0 for the centroid on the entire meteor is 0.5.")

    arg_parser.add_argument('--cores', metavar='CORES', type=int, default=None,
        help="Number of cores to use. Default = all available.")

    # Optional: suppress warnings
    # warnings.filterwarnings('ignore')

    # Parse
    cml_args = arg_parser.parse_args()

    # check if the pick position is between 0 and 1
    if cml_args.pick_pos < 0 or cml_args.pick_pos > 1:
        raise ValueError("pick_position must be between 0 and 1, 0 leading edge, 0.5 centroid full meteor, 1 trailing edge.")

    setupDirAndRunDynesty(cml_args.input_dir, output_dir=cml_args.output_dir, prior=cml_args.prior, resume=cml_args.new_dynesty, use_all_cameras=cml_args.all_cameras, only_plot=cml_args.only_plot, cores=cml_args.cores, pick_position=cml_args.pick_pos, extraprior_file=cml_args.extraprior, save_backup=cml_args.save_backup)

    print("\nDONE: Completed processing of all files in the input directory.\n")