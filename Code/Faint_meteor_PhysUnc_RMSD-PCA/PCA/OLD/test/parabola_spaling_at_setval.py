import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

def adaptive_sampling_parabola(a, b, c, x_start, x_end, threshold):
    # Define fine and coarse sampling rates
    fine_sampling_rate = 0.1
    coarse_sampling_rate = 1.0

    # Initial x range with fine sampling
    x_fine = np.arange(x_start, x_end, fine_sampling_rate)
    y_fine = a * x_fine**2 + b * x_fine + c  # Main parabola
    y_shifted = a * (x_fine - 2)**2 + b * (x_fine - 2) + c  # Shifted parabola by 1 along x-axis

    # Upper and lower threshold lines based on the shifted function
    y_upper_threshold = y_shifted + threshold
    y_lower_threshold = y_shifted - threshold

    # Start building the combined x and y values
    x_combined = []
    y_combined = []

    # Go through the x values with fine sampling
    i = 0
    while i < len(x_fine) - 1:
        x_combined.append(x_fine[i])
        y_combined.append(y_fine[i])

        # Check if current y difference is within the threshold range
        if y_lower_threshold[i] <= y_fine[i] <= y_upper_threshold[i]:
            # Add coarse sampling within this threshold region
            j = i
            while j < len(x_fine) - 1 and y_lower_threshold[j] <= y_fine[j] <= y_upper_threshold[j]:
                x_combined.append(x_fine[j])
                y_combined.append(y_fine[j])
                j += int(coarse_sampling_rate / fine_sampling_rate)  # Step to next coarse point
            i = min(j, len(x_fine) - 1)  # Move to the next point outside the threshold region or end
        else:
            # Continue with fine sampling when outside the threshold range
            i += 1

    # Convert to numpy arrays
    x_combined = np.array(x_combined)
    y_combined = np.array(y_combined)

    # Remove duplicates from x_combined and corresponding values in y_combined
    unique_indices = np.unique(x_combined, return_index=True)[1]
    x_combined = x_combined[unique_indices]
    y_combined = y_combined[unique_indices]

    # Interpolation to smooth the data for plotting
    interp_func = interp1d(x_combined, y_combined, kind='cubic', fill_value="extrapolate")
    x_dense = np.linspace(x_start, x_combined[-1], 1000)  # Ensure x_dense is within valid range
    y_dense = interp_func(x_dense)

    # Plotting the results
    plt.figure(figsize=(10, 5))
    plt.plot(x_dense, y_dense, label="Interpolated Main Parabola")
    plt.scatter(x_combined, y_combined, color="red", label="Sampled Points")
    plt.plot(x_fine, y_shifted, color="black", label="Shifted Parabola (Threshold Center)", linestyle="--")
    plt.plot(x_fine, y_upper_threshold, color="gray", linestyle="--", label="Threshold")
    plt.plot(x_fine, y_lower_threshold, color="gray", linestyle="--")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.legend()
    plt.title("Adaptive Sampling of Parabolic Function with Thresholds Around Shifted Parabola")
    plt.show()

    return x_combined, y_combined

# Example usage
a, b, c = 1, 0, 0  # Coefficients of the parabola
x_start, x_end = -10, 10  # Range of x
threshold = 20  # Threshold for y values

x_sampled, y_sampled = adaptive_sampling_parabola(a, b, c, x_start, x_end, threshold)
