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
    sigma_threshold = 2

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

        # check if all the values are the same
        if np.all(x_vals == x_vals[0]) or np.all(y_vals == y_vals[0]):
            print(f"All values are the same in {snr_file}. Skipping.")
            continue

        # delete inf values of x_inliers on both x and y
        y_vals = y_vals[np.isfinite(x_vals)]
        x_vals = x_vals[np.isfinite(x_vals)]
        # delete nan values of y_inliers on both x and y
        x_vals = x_vals[np.isfinite(y_vals)]
        y_vals = y_vals[np.isfinite(y_vals)]

        # make an inlier_mask that is all True
        inlier_mask = np.ones(len(x_vals), dtype=bool)

        ################# zscore remove outliers #################

        # # Initial fit
        # m_init, c_init = np.polyfit(x_vals, y_vals, 1)
        # y_fit_init = m_init * x_vals + c_init

        # # Residuals and z-scores
        # residuals = y_vals - y_fit_init
        # residuals_z = zscore(residuals)

        # # Inliers if |zscore| < threshold
        # inlier_mask = np.abs(residuals_z) < zscore_threshold

        # # Count how many inliers vs outliers
        # num_inliers = np.count_nonzero(inlier_mask)
        # num_total = len(x_vals)
        # num_outliers = num_total - num_inliers
        # print(f"{snr_file}: Found {num_outliers} outliers, {num_inliers} inliers.")

        # # If everything is an inlier, that means no outliers at this threshold
        # if num_inliers == 0:
        #     print("All points were outliers, skipping plot.")
        #     continue

        # x_inliers = x_vals[inlier_mask]
        # y_inliers = y_vals[inlier_mask]

        # # x_inliers = -2.5*np.array(x_vals)
        # x_inliers = np.array(x_vals)
        # y_inliers = np.array(y_vals)

        ################# line sigma remove outliers #################

        # # 1) Initial linear regression
        # m_init, c_init = np.polyfit(x_vals, y_vals, 1)
        # y_fit_init = m_init * x_vals + c_init

        # # 2) Identify outliers
        # residuals = y_vals - y_fit_init
        # std_res = np.std(residuals)
        # # Indices of points that are within the threshold
        # inlier_mask = np.abs(residuals) <= sigma_threshold * std_res

        # # Filter data to remove outliers
        # x_inliers = x_vals[inlier_mask]
        # y_inliers = y_vals[inlier_mask]

        # # If everything got removed as outliers, skip
        # if len(x_inliers) == 0:
        #     print(f"All points were outliers in {snr_file}. Skipping plot.")
        #     continue

        # # 3) Final linear regression on inliers only
        # m_final, c_final = np.polyfit(x_inliers, y_inliers, 1)
        # y_fit_final = m_final * x_inliers + c_final

        ###########################################################

        x_inliers = x_vals[inlier_mask]
        y_inliers = y_vals[inlier_mask]

        # Final fit on inliers
        m_final, c_final = np.polyfit(x_inliers, y_inliers, 1)
        y_fit_final = m_final * x_inliers + c_final
        inclin_vals.append(m_final)
        const_vals.append(c_final)

        # Plot
        plt.figure()
        plt.scatter(x_inliers, y_inliers, label='Data')
        plt.plot(x_inliers, y_fit_final, label=f'y = {m_final:.4f}x + {c_final:.4f}')
        # use the latex format for the labels for the x and y axis
        # plt.xlabel(r'$-2.5\times log_{10}(SNR)$')
        plt.xlabel(r'$log_{10}(SNR)$')
        plt.ylabel('Apparent Magnitude')
        # take the file name from snr_file
        file_name = os.path.basename(snr_file)
        # for the title take the name of the file all the way with either tavis or elgin isode like for 2019-07-26_03_08_46elgin_SNR_values.png only 2019-07-26_03_08_46elgin
        title_name = ""
        if "tavis" in file_name:
            title_name = file_name.split("tavis")[0]
            title_name = title_name + "tavis"
        elif "elgin" in file_name:
            title_name = file_name.split("elgin")[0]
            title_name = title_name + "elgin"
        # delete the - from the title_name
        title_name = title_name.replace("-", "")
        # delete all the _ but keep the first one
        title_name = title_name.replace("_", "")
        # in between the 8 and the 9 character if is long enougth put a _ to separate the date from the time
        if len(title_name) > 8:
            title_name = title_name[:8] + "_" + title_name[8:]
        plt.title(title_name)
        # plt.title('log10SNR vs mag_data')
        plt.legend()
        plt.grid(True)

        base_name = os.path.basename(snr_file)
        file_root, _ = os.path.splitext(base_name)
        plot_filename = os.path.join(output_dir, f"{file_root}.png")

        plt.savefig(plot_filename)
        plt.close()

        # Write line function
        with open(all_linefuncts_path, 'a') as f:
            f.write(f"{base_name}: y = {m_final:.4f}x + {c_final:.4f}\n")

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
        f.write(f"\nMean of slope and stand.dev.: {mean_inclin:.4f} ± {std_inclin:.4f}\n")
        f.write(f"Mean of const and stand.dev.: {mean_const:.4f} ± {std_const:.4f}\n")


    print(f"\nAll line functions saved to: {all_linefuncts_path}")




if __name__ == "__main__":
    # Example usage:
    # Replace 'path/to/directory' with the directory containing your SNR_values.txt files
    directory_with_txt = "/srv/meteor/reductions/emccd/EMCCD_showers/DRA"
    run_skyfit_on_all_states(directory_with_txt)
    process_snr_files(directory_with_txt)
