from __future__ import print_function, division, absolute_import, unicode_literals

import os
import copy
import numpy as np
import matplotlib.pyplot as plt
import json

from datetime import datetime
from wmpl.MetSim.MetSimErosion import runSimulation as runSimulationErosion
from wmpl.Utils.Pickling import savePickle, loadPickle

# Find all the pickle files in the directory
def find_all_pickle_files(directory):
    pickle_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".pickle"):
                pickle_files.append(file)
    return pickle_files


# Convert numpy arrays to lists and handle nested structures
def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif hasattr(obj, '__dict__'):
        # Convert custom objects to dictionaries
        return {k: convert_to_serializable(v) for k, v in obj.__dict__.items()}
    return obj


# Save the pickle file as a JSON file
def save_as_json(output_dir,pickle_file):
    sim_res = loadPickle(output_dir,pickle_file)

    # Make a deepcopy of the object to avoid modifying the original
    sim_res_copy = copy.deepcopy(sim_res)

    # Convert all relevant attributes to serializable formats
    sim_res_dict = convert_to_serializable(sim_res_copy)

    # Determine the output JSON file path
    json_file_path = os.path.join(output_dir, os.path.basename(pickle_file).replace(".pickle", ".json"))

    # Save the converted object as a JSON file
    with open(json_file_path, 'w') as f:
        json.dump(sim_res_dict, f, indent=4)

    print(f"Saved {pickle_file} as {json_file_path}")



if __name__ == "__main__":
    # Change the current directory it reads all the pickle files in the current directory and trnslates them to JSON
    curr_dir = r'C:\Users\maxiv\WMPG-repoMAX\Code\Utils\met2ecsv_test_cases\match\20240305-031927.292039_ZB'

    # Find all the pickle files in the current directory
    pickle_files = find_all_pickle_files(curr_dir)

    # For each pickle file, load it and save it as a JSON file
    for pickle_file in pickle_files:
        print("Loading pickle file:", pickle_file)
        save_as_json(curr_dir,pickle_file)