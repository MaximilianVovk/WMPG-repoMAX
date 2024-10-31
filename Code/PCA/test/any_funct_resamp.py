import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

def generate_random_smooth_function(x_fine, scale=20):
    # Create a random smooth function using cumulative sum for smoothness
    y_random = np.cumsum(np.random.randn(len(x_fine)))  # Cumulative sum for smoothness
    y_random -= np.mean(y_random)  # Center the function around zero
    y_random /= np.max(np.abs(y_random))  # Normalize
    y_random *= scale  # Scale to desired range
    return y_random

def adaptive_sampling_random_function(x_start, x_end, threshold):
    # Define fine and coarse sampling rates
    fine_sampling_rate = 0.1
    coarse_sampling_rate = 1.0

    # Initial x range with fine sampling
    x_fine = np.arange(x_start, x_end, fine_sampling_rate)
    
    # Generate the main random smooth function to be sampled
    y_fine = generate_random_smooth_function(x_fine, scale=20)

    # # Generate the a line with x_fine point with value 100 
    # y_fine = np.full_like(x_fine, 40)
    
    # Generate the threshold random smooth function
    y_threshold = generate_random_smooth_function(x_fine, scale=15)

    y_fine = y_threshold

    # Upper and lower threshold lines based on the threshold random function
    y_upper_threshold = y_threshold + threshold
    y_lower_threshold = y_threshold - threshold

    # Start building the combined x and y values
    x_combined = []
    y_combined = []

    # check if all point in y_fine are below the y_threshold, if so resample with a coarse sampling rate
    if not np.all(y_fine <= y_threshold):
        i = 0
        while i < len(x_fine):
            # Add the fine sampled point
            x_combined.append(x_fine[i])
            y_combined.append(y_fine[i])

            if y_fine[i] > y_upper_threshold[i]:
                # Crossing from above into the threshold zone
                j = i + 1
                while j < len(x_fine) and y_lower_threshold[j] <= y_fine[j] <= y_upper_threshold[j]:
                    if j == i + 1:
                        # Add the initial crossing point to start coarse sampling
                        x_combined.append(x_fine[j])
                        y_combined.append(y_fine[j])

                    # Move by coarse sampling rate
                    next_index = j + int(coarse_sampling_rate / fine_sampling_rate)

                    if next_index < len(x_fine) and y_lower_threshold[next_index] <= y_fine[next_index] <= y_upper_threshold[next_index]:
                        # Add the coarse sampled point if it is still within the threshold
                        x_combined.append(x_fine[next_index])
                        y_combined.append(y_fine[next_index])
                        j = next_index
                    else:
                        # If the next coarse point is outside the threshold, stop coarse sampling
                        break

                # Move to the next point after finishing coarse sampling or exiting the threshold
                i = j
            else:
                # Continue with fine sampling when outside the threshold range or crossing from below
                i += 1

        # Convert to numpy arrays
        x_combined = np.array(x_combined)
        y_combined = np.array(y_combined)

        # Remove duplicates from x_combined and corresponding values in y_combined
        unique_indices = np.unique(x_combined, return_index=True)[1]
        x_combined = x_combined[unique_indices]
        y_combined = y_combined[unique_indices]

    else:
        # If all points are below the threshold, sample with coarse sampling rate by using interp1d
        interp_func = interp1d(x_fine, y_fine, kind='linear', fill_value="extrapolate")
        x_combined = np.arange(x_start, x_end, coarse_sampling_rate)
        y_combined = interp_func(x_combined)
        
    # Interpolation to smooth the data for plotting
    interp_func = interp1d(x_combined, y_combined, kind='cubic', fill_value="extrapolate")
    x_dense = np.linspace(x_start, x_combined[-1], 1000)  # Ensure x_dense is within valid range
    y_dense = interp_func(x_dense)

    # Plotting the results
    plt.figure(figsize=(10, 5))
    plt.plot(x_dense, y_dense, label="Interpolated Random Function")
    plt.scatter(x_combined, y_combined, color="red", label="Sampled Points")
    plt.plot(x_fine, y_threshold, label="Threshold Center (Random Function)", linestyle="--")
    plt.plot(x_fine, y_upper_threshold, color="gray", linestyle="--", label="Upper Threshold")
    plt.plot(x_fine, y_lower_threshold, color="gray", linestyle="--", label="Lower Threshold")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.legend()
    plt.title("Adaptive Sampling of Random Smooth Function with Threshold Based on Another Random Function")
    plt.show()

    return x_combined, y_combined

# Example usage
x_start, x_end = -10, 10  # Range of x
threshold = 5  # Threshold for y values

x_sampled, y_sampled = adaptive_sampling_random_function(x_start, x_end, threshold)
