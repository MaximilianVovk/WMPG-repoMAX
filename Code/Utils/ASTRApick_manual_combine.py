import os
import pandas as pd
from datetime import datetime
import re
from io import StringIO  # add this at the top of your script



def find_files(folder1, folder2):
    folder1_matches = []
    folder2_matches = []

    for root, _, files in os.walk(folder1):
        for file in files:
            if file.endswith("_RMS_01G.ecsv") or file.endswith("_RMS_02G.ecsv"):
                folder1_matches.append(os.path.join(root, file))

    for root, _, files in os.walk(folder2):
        for file in files:
            if file.startswith("ev_") and (file.endswith("_01G.ecsv") or file.endswith("_02G.ecsv")):
                folder2_matches.append(os.path.join(root, file))

    return folder1_matches, folder2_matches

def parse_timestamp_from_filename(filename):
    match = re.search(r"(\d{4})[-_](\d{2})[-_](\d{2})T(\d{2})[_](\d{2})[_](\d{2})", filename)
    if match:
        return datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S")
    match = re.search(r"(\d{8})_(\d{6})", filename)
    if match:
        return datetime.strptime(match.group(1) + match.group(2), "%Y%m%d%H%M%S")
    return None

def update_rms_with_ev_and_save_new(folder1_files, folder2_files):
    print(f"Found {len(folder1_files)} RMS files and {len(folder2_files)} EV files.")
    
    for file1 in folder1_files:
        ts1 = parse_timestamp_from_filename(os.path.basename(file1))
        cam1 = re.search(r"_RMS_(\d{2}[A-Z])\.ecsv", file1)
        if ts1 is None or cam1 is None:
            continue
        cam1 = cam1.group(1)

        for file2 in folder2_files:
            ts2 = parse_timestamp_from_filename(os.path.basename(file2))
            cam2 = re.search(r"ev_\d+_\d+_(\d{2}[A-Z])\.ecsv", file2)
            if ts2 is None or cam2 is None:
                continue
            cam2 = cam2.group(1)

            # Only proceed if timestamp close AND camera ID matches
            if abs((ts1 - ts2).total_seconds()) <= 2 and cam1 == cam2:
                print(f"Processing files for Camera {cam1}:\n  RMS: {file1}\n  EV : {file2}")

                with open(file1, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                header_end_index = 0
                for i, line in enumerate(lines):
                    if not line.startswith('#'):
                        header_end_index = i
                        break

                header = lines[:header_end_index]
                data_csv = "".join(lines[header_end_index:])
                df_rms = pd.read_csv(StringIO(data_csv))
                df_ev = pd.read_csv(file2, comment='#')

                if 'datetime' not in df_rms.columns or 'datetime' not in df_ev.columns:
                    continue

                df_rms['datetime'] = pd.to_datetime(df_rms['datetime'])
                df_ev['datetime'] = pd.to_datetime(df_ev['datetime'])

                common_times = df_rms['datetime'].isin(df_ev['datetime'])
                matched_rms = df_rms[common_times].copy()
                matched_ev = df_ev[df_ev['datetime'].isin(matched_rms['datetime'])]

                for col in ['ra', 'dec', 'azimuth', 'altitude', 'x_image', 'y_image']:
                    df_rms.loc[common_times, col] = matched_ev.set_index('datetime').loc[
                        matched_rms['datetime'].values, col
                    ].values

                # Modify the camera_id line in header
                new_header = []
                for line in header:
                    if "camera_id" in line:
                        match = re.search(r"'(\d{2}[A-Z])'", line)
                        if match:
                            cam_id = match.group(1)
                            new_line = line.replace(cam_id, f"{cam_id}astraCombine")
                            new_header.append(new_line)
                        else:
                            new_header.append(line)
                    else:
                        new_header.append(line)

                # Save updated data
                folder_path = os.path.dirname(file1)
                base_name = os.path.basename(file1).replace(".ecsv", "_ASTRA.ecsv")
                new_file_path = os.path.join(folder_path, base_name)

                df_rms['datetime'] = df_rms['datetime'].dt.strftime("%Y-%m-%dT%H:%M:%S.%f")

                with open(new_file_path, 'w', encoding='utf-8') as out_f:
                    out_f.writelines(new_header)
                    df_rms.to_csv(out_f, index=False)

                print(f"Saved updated file to: {new_file_path}")
                break

# Example usage:
folder1_path_RMS_files = "/srv/public/mvovk/3rdPaper/iron/Test_combineASTRA/Test"
folder2_path_ASTRA_files = "/srv/public/mvovk/3rdPaper/iron/Test_combineASTRA/Test/Re_re-picking these EMCCD iron meteors"

folder1_files, folder2_files = find_files(folder1_path_RMS_files, folder2_path_ASTRA_files)
update_rms_with_ev_and_save_new(folder1_files, folder2_files)
