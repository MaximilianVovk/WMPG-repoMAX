import os
import pandas as pd
import matplotlib.pyplot as plt
import re
import datetime
import matplotlib.dates as mdates

def extract_time_from_png(filename):
    """Extracts date and time from a PNG filename, ignoring 'all_data' files."""
    if "all_data" in filename:
        return None, None
    match = re.search(r'_(\d{8})_(\d{6})_', filename)
    if match:
        date, time = match.groups()
        return date, f"{time[:2]}:{time[2:4]}:{time[4:]}"  # Format as HH:MM:SS
    return None, None

def process_phot_file(phot_path):
    """Reads a .phot file and extracts time (HH:MM:SS) and poffset."""
    data = []
    with open(phot_path, 'r') as file:
        for line in file:
            if line.startswith("#") or "reject" in line:
                continue
            parts = line.split()
            if len(parts) >= 5:
                time = parts[2]
                poffset = float(parts[3])
                data.append((time, poffset))

    df = pd.DataFrame(data, columns=['time', 'poffset'])
    
    if not df.empty:
        df['time'] = pd.to_datetime(df['time'], format='%H:%M:%S')  # Convert to datetime
    return df

def plot_photometry(input_folder):
    """Processes phot files inside input_folder and plots time vs. poffset as scatter points."""
    phot_dir = os.path.join(input_folder, "phot")
    png_files = [f for f in os.listdir(input_folder) if f.endswith(".png") and "all_data" not in f]
    phot_files = [f for f in os.listdir(phot_dir) if f.endswith(".phot")]

    plt.figure(figsize=(10, 6))

    color_map = {}  # Store colors for each phot file
    camera_scatter = {}  # Track scatter plots per camera for legend

    # Process each phot file
    for phot_file in phot_files:
        phot_path = os.path.join(phot_dir, phot_file)
        camera = phot_file.split('_')[-1].split('.')[0]  # Extract camera ID
        date = phot_file.split('_')[1]  # Extract date in yyyymmdd format
        df = process_phot_file(phot_path)

        if not df.empty:
            # Scatter plot instead of lines
            scatter = plt.scatter(df['time'], df['poffset'], label=f"{date} {camera}", alpha=0.8)
            color_map[(date, camera)] = scatter.get_facecolor()[0]  # Store color
            camera_scatter[(date, camera)] = scatter  # Store for legend handling

        # Find corresponding png file with matching date and camera
        for png_file in png_files:
            png_date, png_time = extract_time_from_png(png_file)
            if png_date and png_time and png_date == date:
                png_time_obj = datetime.datetime.strptime(png_time, '%H:%M:%S')

                # Check if there are multiple cameras at the same time
                same_time_cameras = [key for key in color_map.keys() if key[0] == png_date]
                if len(same_time_cameras) > 1:
                    linestyle = '--'  # Differentiate overlapping camera detections
                else:
                    linestyle = '-'

                # Draw vertical line for detection time
                plt.axvline(png_time_obj, color=color_map.get((date, camera), 'gray'), linestyle=linestyle, alpha=0.7, linewidth=3)

    plt.xlabel("Time (HH:MM:SS)")
    plt.ylabel("Poffset")
    plt.title("Poffset vs Time")

    # Format x-axis to show HH:MM:SS properly
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())  # Auto-adjust ticks for readability
    plt.xticks(rotation=45)

    plt.grid(True, linestyle='--', alpha=0.6)

    # Show legend with distinct cameras
    plt.legend(handles=camera_scatter.values(), labels=camera_scatter.keys(), loc='best')

    # make tight layout
    plt.tight_layout()

    # plt.show()
    # save it with 300 dpi
    plt.savefig(os.path.join(input_folder, "photometry_plot.png"), dpi=300)

# Example usage
input_folder = "/home/mvovk/WMPG-repoMAX/Code/DynNestSampl/lum_noise/plots_DRA"  # Change this to your actual input folder
plot_photometry(input_folder)
