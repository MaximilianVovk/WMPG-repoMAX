import os
import re
import datetime
import statistics

directory = r"C:\Users\maxiv\Documents\UWO\Papers\3)Sporadics\Slow_sporadics"

# Function to extract and parse "Time to run dynesty" from a log file
def extract_timedelta_from_log(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

        # Try matching full format: "X days, HH:MM:SS.ffffff"
        match = re.search(r"Time to run dynesty: (\d+) days, (\d{1,2}:\d{2}:\d{2}\.\d+)", content)
        if match:
            days = int(match.group(1))
            time_part = match.group(2)
            t = datetime.datetime.strptime(time_part, "%H:%M:%S.%f")
            return datetime.timedelta(days=days, hours=t.hour, minutes=t.minute, seconds=t.second, microseconds=t.microsecond)

        # Try matching short format: "HH:MM:SS.ffffff"
        match = re.search(r"Time to run dynesty: (\d{1,2}:\d{2}:\d{2}\.\d+)", content)
        if match:
            time_part = match.group(1)
            t = datetime.datetime.strptime(time_part, "%H:%M:%S.%f")
            return datetime.timedelta(hours=t.hour, minutes=t.minute, seconds=t.second, microseconds=t.microsecond)

    return None

# Gather log files and their parsed runtimes
log_files = []
for root, dirs, files in os.walk(directory):
    for file_name in files:
        if file_name.startswith("log_") and file_name.endswith(".txt"):
            file_path = os.path.join(root, file_name)
            run_time = extract_timedelta_from_log(file_path)
            if run_time:
                log_files.append((file_name, run_time))

# Sort by runtime (longest first)
log_files.sort(key=lambda x: x[1], reverse=True)

# Extract just the durations
durations = [rt for _, rt in log_files]

# Calculate average and mode
average_time = sum(durations, datetime.timedelta()) / len(durations)
# mode_time = statistics.mode(durations)
median_time = statistics.median(durations)

# Print results
print("Average Time: ", average_time)
# print("Mode Time: ", mode_time)
print("Median Time: ", median_time)
print("\nFiles sorted by Time to run dynesty:")
for file, duration in log_files:
    print(f"{file}: {duration}")
