import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math

def find_and_append_csv_files(base_directory):
    # Initialize an empty DataFrame to store all data
    big_df = pd.DataFrame()
    
    # Walk through all subdirectories and files
    for root, dirs, files in os.walk(base_directory):
        for file in files:
            if file.endswith('_AltAzerr_manual.csv'):
                # Create the full path to the file
                file_path = os.path.join(root, file)
                
                # Read the CSV file into a DataFrame
                df = pd.read_csv(file_path)
                
                # Extract the folder name (Camera_Sat) from the path
                folder_name = os.path.basename(root)
                
                # Add the Camera_Sat column to the DataFrame
                df['Camera_Sat'] = folder_name
                
                # Append the DataFrame to the big DataFrame
                big_df = big_df.append(df, ignore_index=True)
    
    return big_df

def calculate_total_error(altitude_error, azimuth_error):
    # Convert degrees to radians
    alt = math.radians(altitude_error)
    az = math.radians(azimuth_error)
    
    # Calculate the magnitude of the vector using the given formula
    magnitude = math.sqrt(math.cos(alt)**2 * math.cos(az)**2 +
                          math.cos(alt)**2 * math.sin(az)**2 +
                          math.sin(alt)**2)
    
    # Convert the magnitude back to degrees
    total_error_deg = math.degrees(magnitude)
    
    return total_error_deg

def angular_distance(alt1, az1, alt2, az2):
    # Convert degrees to radians
    alt1, az1, alt2, az2 = map(math.radians, [alt1, az1, alt2, az2])
    
    # Haversine formula for angular distance on the sphere
    daz = az2 - az1
    dalt = alt2 - alt1
    a = math.sin(dalt/2)**2 + math.cos(alt1) * math.cos(alt2) * math.sin(daz/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return math.degrees(c)

# Usage example
base_directory = r'C:\Users\maxiv\Documents\UWO\Space Situational Awareness DRDC\25mm'
big_dataframe = find_and_append_csv_files(base_directory)

# # Calculate total error using the calculate_total_error function
# big_dataframe['total_error'] = big_dataframe.apply(
#     lambda row: calculate_total_error(row['altitude_error'], row['azimuth_error']), axis=1)

# Calculate total error using the angular_distance function
big_dataframe['total_error'] = big_dataframe.apply(
    lambda row: angular_distance(0, 0, row['altitude_error'], row['azimuth_error']), axis=1)


# put the azimuth_error adn altitude_error in abs
big_dataframe['azimuth_error'] = big_dataframe['azimuth_error'].abs()
big_dataframe['altitude_error'] = big_dataframe['altitude_error'].abs()

# Create subplots using seaborn
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# First subplot: Azimuth vs. Azimuth Error make the border black
sns.scatterplot(ax=axes[0], data=big_dataframe, x='azimuth', y='azimuth_error', hue='Camera_Sat', palette='bright', s=100,  legend='full', edgecolor='black', alpha=0.7)
sns.scatterplot(ax=axes[0], data=big_dataframe, x='azimuth', y='total_error', hue='Camera_Sat', palette='bright', s=100,  marker='s', legend=False, edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Azimuth (deg)')
axes[0].set_ylabel('Error (deg)')
axes[0].set_title('Azimuth and Total Angle Errors (square=total.err, circle=azimuth.err)')
axes[0].grid(True)
# delete the legend
axes[0].get_legend().remove()

# Second subplot: Altitude vs. Altitude Error
sns.scatterplot(ax=axes[1], data=big_dataframe, x='altitude', y='altitude_error', hue='Camera_Sat', palette='bright', s=100,  legend='full', edgecolor='black', alpha=0.7)
sns.scatterplot(ax=axes[1], data=big_dataframe, x='altitude', y='total_error', hue='Camera_Sat', palette='bright', s=100,  marker='s', legend=False, edgecolor='black', alpha=0.7)
axes[1].set_xlabel('Altitude (deg)')
axes[1].set_ylabel('Error (deg)')
axes[1].set_title('Altitude and Total Angle Errors (square=total.err, circle=altitude.err)')
axes[1].grid(True)

plt.tight_layout()
# plt.show()
# save the plot as a file
plt.savefig(base_directory+os.sep+'AzAltMag_errCamera.png', dpi=300)
plt.close()



# Calculate total error using the angular_distance function
big_dataframe['opt.total_error'] = big_dataframe.apply(
    lambda row: angular_distance(0, 0, row['opt.altitude_error'], row['opt.azimuth_error']), axis=1)

# put the azimuth_error adn altitude_error in abs
big_dataframe['opt.azimuth_error'] = big_dataframe['opt.azimuth_error'].abs()
big_dataframe['opt.altitude_error'] = big_dataframe['opt.altitude_error'].abs()

# Create subplots using seaborn
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# First subplot: Azimuth vs. Azimuth Error make the border black
sns.scatterplot(ax=axes[0], data=big_dataframe, x='azimuth', y='opt.azimuth_error', hue='Camera_Sat', palette='bright', s=100,  legend='full', edgecolor='black', alpha=0.7)
sns.scatterplot(ax=axes[0], data=big_dataframe, x='azimuth', y='opt.total_error', hue='Camera_Sat', palette='bright', s=100,  marker='s', legend=False, edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Azimuth (deg)')
axes[0].set_ylabel('Error (deg)')
axes[0].set_title('OPTIMIZED Azimuth and Total Angle Errors (square=total.err, circle=azimuth.err)')
axes[0].grid(True)
# delete the legend
axes[0].get_legend().remove()

# Second subplot: Altitude vs. Altitude Error
sns.scatterplot(ax=axes[1], data=big_dataframe, x='altitude', y='opt.altitude_error', hue='Camera_Sat', palette='bright', s=100,  legend='full', edgecolor='black', alpha=0.7)
sns.scatterplot(ax=axes[1], data=big_dataframe, x='altitude', y='opt.total_error', hue='Camera_Sat', palette='bright', s=100,  marker='s', legend=False, edgecolor='black', alpha=0.7)
axes[1].set_xlabel('Altitude (deg)')
axes[1].set_ylabel('Error (deg)')
axes[1].set_title('OPTIMIZED Altitude and Total Angle Errors (square=total.err, circle=altitude.err)')
axes[1].grid(True)

plt.tight_layout()
# plt.show()
# save the plot as a file
plt.savefig(base_directory+os.sep+'OPTIMIZED_AzAltMag_errCamera.png', dpi=300)
plt.close()