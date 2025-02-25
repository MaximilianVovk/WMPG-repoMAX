import os 
import re
import json
import csv
import statistics
import numpy as np
import math
from wmpl.MetSim.MetSimErosion import Constants
from wmpl.MetSim.GUI import loadConstants

target_dir = r"C:\Users\maxiv\Documents\UWO\Papers\2)ORI-CAP-PER-DRA\Reductions\CAP"

# Define the pattern: 8 digits, underscore, 6 digits, then _sim_fit.json
pattern = re.compile(r'^\d{8}_\d{6}_sim_fit\.json$')

# List of parameters to extract from the JSON files
parameters = [
    "rho", "m_init", "v_init",
    "sigma", "zenith_angle",
    "rho_grain", "lum_eff_type", "lum_eff",
    "erosion_bins_per_10mass", "erosion_height_start",
    "erosion_coeff", "erosion_height_change", "erosion_coeff_change",
    "erosion_rho_change", "erosion_sigma_change", "erosion_mass_index",
    "erosion_mass_min", "erosion_mass_max", "compressive_strength",
    "disruption_erosion_coeff", "disruption_mass_index",
    "disruption_mass_min_ratio", "disruption_mass_max_ratio",
    "disruption_mass_grain_ratio"
]

filename_base = os.path.basename(target_dir)
# Use the target directory's base name to form the CSV file name.
csv_filename = os.path.basename(os.path.normpath(target_dir)) + ".csv"

# This list will hold a row (dictionary) for each file found.
rows = []

# Walk through all subdirectories of the target directory.
for root, dirs, files in os.walk(target_dir):
    for filename in files:
        if pattern.match(filename):
            filepath = os.path.join(root, filename)
            # Determine the first folder relative to the target directory.
            rel_path = os.path.relpath(root, target_dir)
            folder_name = rel_path.split(os.sep)[0] if rel_path != '.' else os.path.basename(os.path.normpath(target_dir))
            
            # Read the JSON file.
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
                continue

            # Create a row with the folder and file info plus all parameter values.
            row = {"folder": folder_name, "file": filename}
            for param in parameters:
                row[param] = data.get(param, None)
            rows.append(row)

# Function to approximate mode using histogram binning.
def approximate_mode_1d(samples):
    hist, bin_edges = np.histogram(samples, bins='auto', density=True)
    idx_max = np.argmax(hist)
    return 0.5 * (bin_edges[idx_max] + bin_edges[idx_max + 1])

# Custom JSON serializer to handle numpy arrays and numpy numbers.
def json_serial(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    raise TypeError(f"Type {type(obj)} not serializable")

# Calculate the mean and mode for each parameter across all files.
mean_row = {"folder": "mean", "file": ""}
mode_row = {"folder": "mode", "file": ""}

# # Create instances of Constants to hold the default values (which we will override)
# json_file_default_mean = Constants()
# json_file_default_mode = Constants()

# Load the nominal simulation parameters
json_file_default_mean, _ = loadConstants(filepath)
json_file_default_mean.dens_co = np.array(json_file_default_mean.dens_co)
json_file_default_mode, _ = loadConstants(filepath)
json_file_default_mode.dens_co = np.array(json_file_default_mode.dens_co)

for param in parameters:
    # Get all values (ignoring missing ones).
    values = [row[param] for row in rows if row[param] is not None]
    # For numerical calculations, filter to int and float.
    numeric_values = [v for v in values if isinstance(v, (int, float))]
    if numeric_values:
        try:
            mean_value = statistics.mean(numeric_values)
        except statistics.StatisticsError:
            mean_value = None
        try:
            mode_value = approximate_mode_1d(numeric_values)
        except statistics.StatisticsError:
            mode_value = None
    else:
        mean_value = None
        mode_value = None

    mean_row[param] = mean_value
    mode_row[param] = mode_value

    # Store the computed values in the Constants instances.
    json_file_default_mean.__dict__[param] = mean_value
    json_file_default_mode.__dict__[param] = mode_value

# Write the results to the CSV file.
fieldnames = ["folder", "file"] + parameters
with open(csv_filename, "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    # Write the mean and mode rows at the top.
    writer.writerow(mean_row)
    writer.writerow(mode_row)
    # Write one row per file.
    for row in rows:
        writer.writerow(row)

print(f"CSV file created: {csv_filename}")

# Now, dump the computed mean and mode values into two new JSON files.
mean_json_filename = filename_base + "_mean.json"
mode_json_filename = filename_base + "_mode.json"

with open(mean_json_filename, "w") as f:
    json.dump(json_file_default_mean.__dict__, f, indent=4, default=json_serial)
with open(mode_json_filename, "w") as f:
    json.dump(json_file_default_mode.__dict__, f, indent=4, default=json_serial)

print(f"Mean JSON file created: {mean_json_filename}")
print(f"Mode JSON file created: {mode_json_filename}")
