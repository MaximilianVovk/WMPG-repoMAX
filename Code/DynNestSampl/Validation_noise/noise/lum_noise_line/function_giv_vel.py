import numpy as np
import matplotlib.pyplot as plt
import os

# Given function to fit inverse power relation
def fit_inverse_power():
    velocities = np.array([20, 23, 66])
    offsets = np.array([8.0671, 7.8009, 7.3346])
    
    log_velocities = np.log(velocities)
    log_offsets = np.log(offsets)

    b, log_a = np.polyfit(log_velocities, log_offsets, 1)
    a = np.exp(log_a)
    
    return a, b

def offset_for_velocity(v):
    """Compute offset for a given velocity using inverse power fit"""
    a, b = fit_inverse_power()
    return a * v**b

# Known data points
known_velocities = np.array([20, 23, 66])
known_offsets = np.array([8.0671, 7.8009, 7.3346])
known_errors = np.array([0.1540, 0.1968, 0.2406])  # Standard deviations

# Define velocities for which we will plot the interpolated lines
velocities_to_plot = np.array([20, 30, 40, 50, 60, 70])

# Generate x values for -2.5 * log10(SNR)
x_values = np.linspace(-5, 0, 100)  # Cover a reasonable range

# Get corresponding offset values for given velocities
offsets = np.array([offset_for_velocity(v) for v in velocities_to_plot])

# Generate colors
colors = plt.cm.viridis(np.linspace(0, 1, len(velocities_to_plot)))

# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Left subplot: Lines for apparent magnitude vs. -2.5 log10(SNR)
for i, (v, c) in enumerate(zip(velocities_to_plot, offsets)):
    y_values = x_values + c
    axes[0].plot(x_values, y_values, label=f"v = {v} km/s", color=colors[i])

axes[0].set_xlabel("SNR",  fontsize=15)
axes[0].set_ylabel("Apparent Meteor Magnitude",  fontsize=15)
axes[0].legend(fontsize=15)
# axes[0].set_title("Apparent Magnitude vs. SNR")
# increase the tick label size
axes[0].tick_params(axis='both', which='major', labelsize=12)

# Get current x-axis tick locations
log_x_ticks = axes[0].get_xticks()

# Convert log tick values to SNR
snr_ticks = 10**(log_x_ticks / (-2.5))

# Update tick labels with SNR values
axes[0].set_xticklabels([f'{snr:.2f}' for snr in snr_ticks])

# Change the x-axis label to "SNR"
axes[0].set_xlabel("SNR",  fontsize=15)  # Fix incorrect xlabel placement
# grid lines
axes[0].grid(True)

# Right subplot: Inverse power function fit for c(v)
vel_range = np.linspace(15, 75, 100)  # Smooth range for function plot
c_values = [offset_for_velocity(v) for v in vel_range]

# Plot the fitted inverse power function
axes[1].plot(vel_range, c_values, label=r"$c(v) = a v^b$", color="black")

# Plot colored sample points from the computed function
axes[1].scatter(velocities_to_plot, offsets, color=colors, label="Sampled Points", zorder=3)

# Plot known data points as black dots with error bars
axes[1].errorbar(known_velocities, known_offsets, yerr=known_errors, fmt='o', color='black', capsize=5, label="Known Data w/ Errors", zorder=4)

for i, v in enumerate(velocities_to_plot):
    axes[1].text(v, offsets[i], f"{v} km/s", color=colors[i], fontsize=10, verticalalignment='bottom')

axes[1].set_xlabel("Velocity (km/s)",  fontsize=15)
axes[1].set_ylabel("Offset c(v)",  fontsize=15)
# axes[1].set_title("Offset Function")
axes[1].legend(fontsize=15)
# increase the tick label size
axes[1].tick_params(axis='both', which='major', labelsize=12)
# grid lines
axes[1].grid(True)

plt.tight_layout()
# save with dpi=300 for higher resolution
output_path = r"C:\Users\maxiv\WMPG-repoMAX\Code\DynNestSampl\Validation_noise\noise\lum_noise_line"  # Define your output path here
plt.savefig(output_path + os.sep + "inverse_power_fit.png", dpi=300)
# plt.savefig("inverse_power_fit.png", dpi=300)

