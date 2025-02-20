import numpy as np
import pandas as pd
import pickle
import sys
import json
import os
import copy
import matplotlib.pyplot as plt
import dynesty
from dynesty import plotting as dyplot
import time
from matplotlib.ticker import ScalarFormatter
import scipy
from scipy.stats import norm
from scipy.stats import multivariate_normal
import warnings

from wmpl.MetSim.GUI import loadConstants, saveConstants,SimulationResults
from wmpl.MetSim.MetSimErosion import runSimulation, Constants, zenithAngleAtSimulationBegin
from wmpl.MetSim.ML.GenerateSimulations import generateErosionSim,saveProcessedList,MetParam
from wmpl.Utils.TrajConversions import J2000_JD, date2JD
from wmpl.Utils.Math import meanAngle
from wmpl.Utils.AtmosphereDensity import fitAtmPoly
from wmpl.Utils.Physics import dynamicPressure

import signal


###############################################################################
# Function: read_prior_to_bounds
###############################################################################
def read_prior_to_bounds(object_meteor,file_path=""):
    # Default bounds
    default_bounds = {
        "v_init": (np.nan, 500),
        "zenith_angle": (np.nan, 0.01),
        "m_init": (np.nan, np.nan),
        "rho": (10, 4000),  # log transformation applied later
        "sigma": (0.008 / 1e6, 0.03 / 1e6),
        "erosion_height_start": (np.nan, np.nan),
        "erosion_coeff": (1 / 1e12, 2 / 1e6),  # log transformation applied later
        "erosion_mass_index": (1, 3),
        "erosion_mass_min": (5e-12, 1e-9),  # log transformation applied later
        "erosion_mass_max": (1e-10, 1e-7),  # log transformation applied later
    }

    default_flags = {
        "v_init": ["norm"],
        "zenith_angle": ["norm"],
        "m_init": [],
        "rho": ["log"],
        "sigma": [],
        "erosion_height_start": [],
        "erosion_coeff": ["log"],
        "erosion_mass_index": [],
        "erosion_mass_min": ["log"],
        "erosion_mass_max": ["log"]
        }

    # Default values if no file path is provided
    if file_path == "":
        # delete from the default_bounds the zenith_angle
        default_bounds.pop("zenith_angle")
        bounds = [default_bounds[key] for key in default_bounds]
        flags_dict = {key: default_flags.get(key, []) for key in default_bounds}
        # for the one that have log transformation, apply it
        for i, key in enumerate(default_bounds):
            if "log" in flags_dict[key]:
                bounds[i] = np.log10(bounds[i][0]), np.log10(bounds[i][1])
        fixed_values = {
            "zenith_angle": np.nan,
        }

    else:
        bounds = []
        flags_dict = {}
        fixed_values = {}

        def safe_eval(value):
            try:    
                return eval(value, {"__builtins__": {"np": np}}, {})
            except Exception:
                return value  # Return as string if evaluation fails
        
        # Read .prior file, ignoring comment lines
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if line.startswith('#') or not line:
                    continue
                parts = line.split('#')[0].strip().split(',')  # Remove inline comments
                name = parts[0].strip()
                # Handle fixed values
                if "fix" in parts:
                    fixed_values[name] = safe_eval(parts[1].strip() if len(parts) > 1 else "nan")
                    continue
                min_val = parts[1].strip() if len(parts) > 1 else "nan"
                max_val = parts[2].strip() if len(parts) > 2 else "nan"
                flags = [flag.strip() for flag in parts[3:]] if len(parts) > 3 else []
                
                # Handle NaN values and default replacement
                min_val = safe_eval(min_val) if min_val.lower() != "nan" else default_bounds.get(name, (np.nan, np.nan))[0]
                max_val = safe_eval(max_val) if max_val.lower() != "nan" else default_bounds.get(name, (np.nan, np.nan))[1]

                # check if min_val is greater than max_val and if it is, swap the values
                if min_val != np.nan and max_val != np.nan:
                    if min_val > max_val:
                        min_val, max_val = max_val, min_val
                if min_val == max_val and min_val != np.nan:
                    fixed_values[name] = min_val
                    continue
                
                # check if both min_val and max_val are np.nan, if they are, replace them with the default values
                if np.isnan(min_val) and np.isnan(max_val):
                    min_val, max_val = default_bounds.get(name, (np.nan, np.nan))
                    # take he default flag
                    flags = default_flags.get(name, [])
                elif np.isnan(min_val) and "norm" in flags:
                    min_val = default_bounds.get(name, (np.nan, np.nan))[0]
                    # if it is not a nan, make the mean of the values in default_bounds
                    if not np.isnan(min_val):
                        # make the mean of the values in default_bounds
                        min_val = np.mean(default_bounds.get(name, (np.nan, np.nan)))
                elif np.isnan(max_val) and "norm" in flags:
                    max_val = default_bounds.get(name, (np.nan, np.nan))[1]
                    # if it is not a nan, make the mean of the values in default_bounds
                    if not np.isnan(max_val) and default_flags.get(name, [])!=["norm"]:
                        # make the mean of the values in default_bounds
                        max_val = (min_val - max_val)/2
                elif np.isnan(min_val) and default_flags.get(name, [])!=["norm"]:
                    min_val = default_bounds.get(name, (np.nan, np.nan))[0]
                elif np.isnan(max_val) and default_flags.get(name, [])!=["norm"]:
                    max_val = default_bounds.get(name, (np.nan, np.nan))[1]
                
                # Store flags
                flags_dict[name] = flags

                # TO DO: use object_meteor to get the np.nan a value
                #####################################
                # substtute nan with the estimated values
                #####################################

                            
                # Apply log10 transformation if needed
                if "log" in flags:
                    # check if any values is 0 and if it is, replace it with the default value
                    if min_val == 0:
                        min_val = 1/1e12
                    # Apply log10 transformation
                    min_val, max_val = np.log10(min_val), np.log10(max_val)
                
                bounds.append((min_val, max_val))
    
    # check if the bounds the len(bounds) + len(fixed_values) =>10
    if len(bounds) + len(fixed_values) < 10:
        raise ValueError("The number of bounds and fixed values should 10 or above")

    return bounds, flags_dict, fixed_values


###############################################################################
# Class: find_dynestyfile_and_priors
###############################################################################
class find_dynestyfile_and_priors:
    """
    1. If `input_dir_or_file` is a file:
       - Strip extension => add ".dynesty"
       - If `prior_file` is a valid file, use it (read_prior_to_bounds).
         Otherwise, look for a .prior in the same folder.
         If still none, use defaults.

    2. If `input_dir_or_file` is a directory:
       - For each .pickle found:
         - Decide .dynesty name (check existing, apply resume logic).
         - If `prior_file` is valid, use it.
           Otherwise, look for a .prior in that folder.
           If none, default.

    3. `resume=True`: Reuse existing .dynesty if it exists.
       `resume=False`: Create new .dynesty name (append _n1, _n2, etc.) if found.

    4. If `output_dir != ""`, create a subfolder named after the .dynesty base name.
       Otherwise, store results in the same folder as the .dynesty.

    5. Stored results:
       - `self.input_folder_file`: List of (dynesty_file, bounds, flags_dict, fixed_values)
       - `self.priors`: List of the .prior paths used (or "") for each entry
       - `self.output_folders`: Where results are stored for each entry
    """

    def __init__(self, input_dir_or_file, prior_file, resume, output_dir=""):
        self.input_dir_or_file = input_dir_or_file
        self.prior_file = prior_file
        self.resume = resume
        self.output_dir = output_dir

        # Prepare placeholders
        self.input_folder_file = [] # [(dynesty_file, bounds, flags_dict, fixed_values), ...]
        self.priors = []            # [used_prior_path_or_empty_string, ...]
        self.output_folders = []    # [output_folder_for_this_dynesty, ...]

        # If user gave a valid .prior path, read it once.
        # We'll reuse this for every .dynesty discovered.
        if self.prior_file and os.path.isfile(self.prior_file):
            self.user_prior = read_prior_to_bounds(self.prior_file)
        else:
            self.user_prior = None  # Means we have to search or default for each .dynesty

        # Kick off processing
        self._process_input()

    def _process_input(self):
        """Decide if input is file or directory, build .dynesty, figure out prior, and store results."""
        if os.path.isfile(self.input_dir_or_file):
            # Single file case
            root, _ = os.path.splitext(self.input_dir_or_file)
            dynesty_file = root + ".dynesty"

            # Determine prior for this single file:
            if self.user_prior is not None:
                # We already read a user-provided prior
                bounds, flags_dict, fixed_values = self.user_prior
                prior_path = self.prior_file
            else:
                # If no user-prior, check for a local .prior
                folder = os.path.dirname(self.input_dir_or_file)
                prior_path = self._find_prior_in_folder(folder)
                if prior_path:
                    bounds, flags_dict, fixed_values = read_prior_to_bounds(prior_path)
                else:
                    # default
                    prior_path = ""
                    bounds, flags_dict, fixed_values = read_prior_to_bounds("")

            # Decide output folder
            output_folder = self._decide_output_folder(dynesty_file)

            # Store in class variables
            self.input_folder_file.append((dynesty_file, bounds, flags_dict, fixed_values))
            self.priors.append(prior_path)
            self.output_folders.append(output_folder)

        else:
            # Directory case
            for root, dirs, files in os.walk(self.input_dir_or_file):
                pickle_files = [f for f in files if f.endswith('.pickle')]
                if not pickle_files:
                    continue

                for pf in pickle_files:
                    base_name = os.path.splitext(pf)[0]
                    possible_dynesty = os.path.join(root, base_name + ".dynesty")

                    # Check for existing .dynesty in the same folder
                    existing_dynesty_list = [f for f in files if f.endswith(".dynesty")]

                    if existing_dynesty_list:
                        # There is at least one .dynesty in this folder
                        if os.path.exists(possible_dynesty):
                            # Matches the .pickle base
                            if self.resume:
                                dynesty_file = possible_dynesty
                            else:
                                dynesty_file = self._build_new_dynesty_name(possible_dynesty)
                        else:
                            # There's a .dynesty, but not matching the base name
                            first_dynesty_found = os.path.join(root, existing_dynesty_list[0])
                            if self.resume:
                                dynesty_file = first_dynesty_found
                            else:
                                dynesty_file = self._build_new_dynesty_name(first_dynesty_found)
                    else:
                        # No .dynesty => create from .pickle base name
                        dynesty_file = possible_dynesty

                    # Determine prior
                    if self.user_prior is not None:
                        # Use the single user-supplied prior
                        bounds, flags_dict, fixed_values = self.user_prior
                        prior_path = self.prior_file
                    else:
                        # Look for local .prior
                        prior_path = self._find_prior_in_folder(root)
                        if prior_path:
                            bounds, flags_dict, fixed_values = read_prior_to_bounds(prior_path)
                        else:
                            # default
                            prior_path = ""
                            bounds, flags_dict, fixed_values = read_prior_to_bounds("")

                    # Decide the output folder
                    output_folder = self._decide_output_folder(dynesty_file)

                    # Store results
                    self.input_folder_file.append((dynesty_file, bounds, flags_dict, fixed_values))
                    self.priors.append(prior_path)
                    self.output_folders.append(output_folder)

    def _find_prior_in_folder(self, folder):
        """Return the first .prior file found, or None if none."""
        for f in os.listdir(folder):
            if f.endswith(".prior"):
                return os.path.join(folder, f)
        return None

    def _build_new_dynesty_name(self, existing_dynesty_path):
        """
        If resume=False and we want to avoid overwriting an existing .dynesty file,
        append _n1, _n2, etc. until no file collision occurs.
        """
        folder = os.path.dirname(existing_dynesty_path)
        base = os.path.splitext(os.path.basename(existing_dynesty_path))[0]
        ext = ".dynesty"

        counter = 1
        while True:
            new_name = f"{base}_n{counter}{ext}"
            new_path = os.path.join(folder, new_name)
            if not os.path.exists(new_path):
                return new_path
            counter += 1

    def _decide_output_folder(self, dynesty_file_path):
        """
        Decide where to store results:
          - If self.output_dir is empty, use the .dynesty file's folder.
          - Otherwise, create a subdirectory named after the .dynesty base name.
        """
        if not self.output_dir:
            return os.path.dirname(dynesty_file_path)

        # If output_dir is given, create a named subdirectory
        subfolder_name = os.path.splitext(os.path.basename(dynesty_file_path))[0]
        final_output_dir = os.path.join(self.output_dir, subfolder_name+"_dynesty")
        # os.makedirs(final_output_dir, exist_ok=True)
        return final_output_dir


###############################################################################
# EXAMPLE MAIN (as you provided), illustrating usage
###############################################################################
if __name__ == "__main__":

    import argparse
    import sys
    import warnings

    ### COMMAND LINE ARGUMENTS
    arg_parser = argparse.ArgumentParser(description="Run dynesty with optional .prior file.")

    arg_parser.add_argument('--input_dir', metavar='INPUT_PATH', type=str,
        default=r"C:\Users\maxiv\WMPG-repoMAX\Code\DynNestSampl\PER_v59_heavy.json",
        help="Path to walk and find .pickle file or specific single file .pickle or .json file divided by ',' in between.")

    arg_parser.add_argument('--output_dir', metavar='OUTPUT_DIR', type=str,
        default=r"C:\Users\maxiv\WMPG-repoMAX\Code\DynNestSampl",
        help="Where to store results. If empty, store next to each .dynesty.")

    arg_parser.add_argument('--prior', metavar='PRIOR', type=str,
        default=r"C:\Users\maxiv\WMPG-repoMAX\Code\DynNestSampl\stony_meteoroid.prior",
        help="Path to a .prior file. If blank, we look in the .dynesty folder or default to built-in bounds.")

    arg_parser.add_argument('--resume', metavar='RESUME', type=bool, default=True,
        help="If True, resume from existing .dynesty if found. If False, create a new version.")

    arg_parser.add_argument('--cores', metavar='CORES', type=int, default=None,
        help="Number of cores to use. Default = all available.")

    # Parse
    cml_args = arg_parser.parse_args()

    # Optional: suppress warnings
    warnings.filterwarnings('ignore')

    # If no core count given, use all
    if cml_args.cores is None:
        import multiprocessing
        cml_args.cores = multiprocessing.cpu_count()

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
        print(f"Processing {input_dirfile} (this may take a while if subdirectories are large)")
        print("--------------------------------------------------")

        # Use the class to find .dynesty, load prior, and decide output folders
        finder = find_dynestyfile_and_priors(
            input_dir_or_file=input_dirfile,
            prior_file=cml_args.prior,
            resume=cml_args.resume,
            output_dir=cml_args.output_dir
        )

        # Each discovered or created .dynesty is in input_folder_file
        # with its matching prior info
        for i, (dynesty_info, prior_path, out_fold) in enumerate(zip(
            finder.input_folder_file,
            finder.priors,
            finder.output_folders
        )):
            dynesty_file, bounds, flags_dict, fixed_values = dynesty_info
            print(f"Entry #{i+1}:")
            print("  Dynesty file: ", dynesty_file)
            print("  Prior file:   ", prior_path)
            print("  Output folder:", out_fold)
            print("  Bounds:       ", bounds)
            print("  Flags:        ", flags_dict)
            print("  Fixed Values: ", fixed_values)
            print("--------------------------------------------------")
