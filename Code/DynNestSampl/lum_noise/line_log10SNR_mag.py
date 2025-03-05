import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import subprocess
# Make sure you have scipy installed: pip install scipy
from scipy.stats import zscore

def run_skyfit_on_all_states(root_dir):
    """
    Recursively searches 'root_dir' for every 'skyFitMR_latest.state' file
    and runs:
        python -m Utils.SkyFit2 "<path_to_state_file>" --config .
    in that file's directory.
    
    This should generate SNR_values.txt (and other outputs) for each found .state file.
    """
    # Recursively find all 'skyFitMR_latest.state' files
    state_files = glob.glob(
        os.path.join(root_dir, '**', 'skyFitMR_latest.state'),
        recursive=True
    )

    if not state_files:
        # print(f"No skyFitMR_latest.state files were found in '{root_dir}'!")
        return

    for state_file in state_files:
        # The directory where the state file is located
        state_dir = os.path.dirname(state_file)
        # print(f"\nFound: {state_file}\nRunning SkyFit2 in: {state_dir}")

        # Command to run
        cmd = [
            'python', '-m', 'Utils.SkyFit2',
            state_file,
            '--config', '.'
        ]
        # Run the command in the same directory (so it uses the local config)
        # subprocess.run(cmd, cwd=state_dir)
        # merger = ' '.join(cmd)
        print(' '.join(cmd))





def process_snr_files(input_dir):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)

    inclin_vals = []
    const_vals = []

    snr_files = glob.glob(os.path.join(input_dir, '**', '*SNR_values.txt'), recursive=True)
    if not snr_files:
        print(f"No files found under '{input_dir}'.")
        return

    all_linefuncts_path = os.path.join(output_dir, "all_linefuncts.txt")
    with open(all_linefuncts_path, 'w') as f:
        f.write("# All regression line functions (after outlier removal):\n\n")

    zscore_threshold = 3

    for snr_file in snr_files:
        x_vals = []
        y_vals = []

        with open(snr_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = [p.strip() for p in line.split(',')]
                if len(parts) == 4:
                    # parse columns
                    frame_val = float(parts[0])
                    snr_val   = float(parts[1])
                    x_logsnr  = float(parts[2])
                    y_mag     = float(parts[3].replace('+',''))
                    x_vals.append(x_logsnr)
                    y_vals.append(y_mag)

        x_vals = np.array(x_vals)
        y_vals = np.array(y_vals)

        if len(x_vals) == 0:
            print(f"No valid data in {snr_file}. Skipping.")
            continue

        # Initial fit
        m_init, c_init = np.polyfit(x_vals, y_vals, 1)
        y_fit_init = m_init * x_vals + c_init

        # Residuals and z-scores
        residuals = y_vals - y_fit_init
        residuals_z = zscore(residuals)

        # Inliers if |zscore| < threshold
        inlier_mask = np.abs(residuals_z) < zscore_threshold

        # Count how many inliers vs outliers
        num_inliers = np.count_nonzero(inlier_mask)
        num_total = len(x_vals)
        num_outliers = num_total - num_inliers
        print(f"{snr_file}: Found {num_outliers} outliers, {num_inliers} inliers.")

        # If everything is an inlier, that means no outliers at this threshold
        if num_inliers == 0:
            print("All points were outliers, skipping plot.")
            continue

        x_inliers = x_vals[inlier_mask]
        y_inliers = y_vals[inlier_mask]

        # Final fit on inliers
        m_final, c_final = np.polyfit(x_inliers, y_inliers, 1)
        y_fit_final = m_final * x_inliers + c_final
        inclin_vals.append(m_final)
        const_vals.append(c_final)

        # Plot
        plt.figure()
        plt.scatter(x_inliers, y_inliers, label='Inliers')
        plt.plot(x_inliers, y_fit_final, label=f'y = {m_final:.4f}x + {c_final:.4f}')
        plt.xlabel('log10SNR')
        plt.ylabel('mag_data')
        plt.title('log10SNR vs mag_data (Z-score outlier removal)')
        plt.legend()
        plt.grid(True)

        base_name = os.path.basename(snr_file)
        file_root, _ = os.path.splitext(base_name)
        plot_filename = os.path.join(output_dir, f"{file_root}.png")

        plt.savefig(plot_filename)
        plt.close()

        # Write line function
        with open(all_linefuncts_path, 'a') as f:
            f.write(f"{base_name}: y = {m_final:.4f}x + {c_final:.4f} the SNR=1 is {m_final * 1 + c_final}\n")

        print(f"Processed {snr_file} -> {plot_filename}")

    # delete the outlier from the inclin_vals and const_vals
    inclin_vals = np.array(inclin_vals)
    const_vals = np.array(const_vals)

    # remove the outliers using zscore
    inclin_vals = inclin_vals[np.abs(zscore(inclin_vals)) < zscore_threshold]
    const_vals = const_vals[np.abs(zscore(const_vals)) < zscore_threshold]

    # calculate the mean and standard deviation
    mean_inclin = np.mean(inclin_vals)  
    mean_const = np.mean(const_vals)
    std_inclin = np.std(inclin_vals)
    std_const = np.std(const_vals)

    # write the mean and standard deviation to the file
    with open(all_linefuncts_path, 'a') as f:
        f.write(f"\nMean of inclin: {mean_inclin:.4f} ± {std_inclin:.4f}\n")
        f.write(f"Mean of const: {mean_const:.4f} ± {std_const:.4f}\n")
        f.write(f"avg SNR=1 is {mean_inclin * 1 + mean_const}\n")


    print(f"\nAll line functions saved to: {all_linefuncts_path}")




if __name__ == "__main__":
    # Example usage:
    # Replace 'path/to/directory' with the directory containing your SNR_values.txt files
    directory_with_txt = "/srv/meteor/reductions/emccd/mvovk_pca_project/CAP"
    run_skyfit_on_all_states(directory_with_txt)
    process_snr_files(directory_with_txt)
