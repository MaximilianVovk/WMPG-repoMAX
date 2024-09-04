"""
This script is designed to help with the rapid extraction and organization of data from the SkyFit2 reduction process.
It scans through a source directory containing folders with exactly one underscore (_) in their names. Inside these folders, 
it searches for subfolders whose names start with a number. Once found, the script copies these numbered subfolders to a 
specified target directory. This process is especially useful when handling large volumes of SkyFit2 reduction data, ensuring 
that important results are efficiently organized and transferred.

How it works:
1. The script opens each folder in the source directory that has exactly one underscore (_) in its name.
2. It then checks if there are any subfolders with a number at the beginning of their name.
3. If such subfolders are found, they are copied into the specified target directory.
4. The script ensures that no duplicate folders are copied into the target directory.
"""

import os
import shutil
import re

def copy_folders_with_numbered_subfolders(source_folder, target_folder):
    print(f"Source folder: {source_folder}")
    # List all folders in the source directory
    source_folders = [f for f in os.listdir(source_folder) if os.path.isdir(os.path.join(source_folder, f))]

    # Find folders with exactly one underscore in their name
    underscore_folders = [f for f in source_folders if f.count('_') == 2]

    # Loop over the folders with one underscore
    for folder in underscore_folders:
        print(f"Processing folder '{folder}'...")
        folder_path = os.path.join(source_folder, folder)
        
        # List subfolders inside the current folder
        subfolders = [sf for sf in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, sf))]

        # Find subfolders that have a number at the beginning of their name
        numbered_subfolders = [sf for sf in subfolders if re.match(r'^\d', sf)]

        # Copy each numbered subfolder into the target folder
        for subfolder in numbered_subfolders:
            src_subfolder_path = os.path.join(folder_path, subfolder)
            dest_subfolder_path = os.path.join(target_folder, subfolder)

            # Only copy if the folder doesn't already exist in the target directory
            if not os.path.exists(dest_subfolder_path):
                shutil.copytree(src_subfolder_path, dest_subfolder_path)
                print(f"Copied '{subfolder}' from '{folder_path}' to '{target_folder}'")
            else:
                print(f"Subfolder '{subfolder}' already exists in '{target_folder}'")

# Usage example
source_folder = r'N:\eharmos\reduction\Skyfit2\ORI'  # Replace with your source directory path
target_folder = r'C:\Users\maxiv\Documents\UWO\Papers\2)PCA_ORI-CAP-PER\Reductions\ORI'  # Replace with your target directory path
copy_folders_with_numbered_subfolders(source_folder, target_folder)
