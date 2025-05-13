import os
import re
import datetime
import statistics

# Directory containing the log files
directory = r"C:\Users\maxiv\Documents\UWO\Papers\2)ORI-CAP-PER-DRA\Results\CAMO+EMCCD\ORI-1frag-0417"

# Function to extract time from "Time to run dynesty" in the log files
def extract_time_from_log(file_path):
    time_pattern = r"Time to run dynesty: (\d{2}:\d{2}:\d{2}.\d+)"
    with open(file_path, 'r') as file:
        content = file.read()
        match = re.search(time_pattern, content)
        if match:
            time_str = match.group(1)
            return datetime.datetime.strptime(time_str, "%H:%M:%S.%f")
    return None

# List to hold file names and times
log_files = []

# Walk through all subdirectories in the given directory
for root, dirs, files in os.walk(directory):
    for file_name in files:
        if file_name.startswith("log_") and file_name.endswith(".txt"):
            file_path = os.path.join(root, file_name)
            run_time = extract_time_from_log(file_path)
            if run_time:
                log_files.append((file_name, run_time))

# Convert times to timedelta for proper calculations
times_in_timedelta = [run_time - datetime.datetime.min for _, run_time in log_files]

# Sort the files based on run time (from highest to lowest)
log_files.sort(key=lambda x: x[1], reverse=True)

# Calculate the average and mode of the times
average_time = sum(times_in_timedelta, datetime.timedelta()) / len(times_in_timedelta)
mode_time = statistics.mode(times_in_timedelta)

# Display results
print("Average Time: ", average_time)
print("Mode Time: ", mode_time)
print("\nFiles sorted by Time to run dynesty:")
for file, time in log_files:
    print(f"{file}: {time}")
