import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import pandas as pd
import seaborn as sns
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
        merger = ' '.join(cmd)
        # delete skyFitMR_latest.state from the merger
        merger = merger.replace("skyFitMR_latest.state", "")
        print(merger)





def process_snr_files(input_dir):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)

    inclin_vals = []
    const_vals = []

    # snr_files = glob.glob(os.path.join(input_dir, '**', '*SNR_values.txt'), recursive=True)
    snr_files = glob.glob(os.path.join(input_dir, '**', '*_manual.txt'), recursive=True)
    if not snr_files:
        print(f"No files found under '{input_dir}'.")
        return

    all_linefuncts_path = os.path.join(output_dir, "all_linefuncts.txt")
    with open(all_linefuncts_path, 'w') as f:
        f.write("# All regression line functions (after outlier removal):\n\n")

    zscore_threshold = 3
    sigma_threshold = 2

    # create a pandas dataframe to store the data
    all_data = pd.DataFrame(columns=[r'$-2.5 \, log_{10}(SNR)$', 'Apparent Meteor Magnitude', 'file'])

    for snr_file in snr_files:

        magnitudes, snrs = parse_rms_file(snr_file)
        
        x_vals = np.array(-2.5*np.log10(snrs))
        # x_vals = np.array(snrs)
        y_vals = np.array(magnitudes)

        # delete the values from y_vals and x_vals where x_vals are -2.5*np.log10(99)
        y_vals = y_vals[x_vals != -2.5*np.log10(99.99)]
        x_vals = x_vals[x_vals != -2.5*np.log10(99.99)]

        # x_vals = []
        # y_vals = []

        # with open(snr_file, 'r') as f:
        #     for line in f:
        #         line = line.strip()
        #         if not line or line.startswith('#'):
        #             continue
        #         parts = [p.strip() for p in line.split(',')]
        #         if len(parts) == 4:
        #             # parse columns
        #             frame_val = float(parts[0])
        #             snr_val   = float(parts[1])
        #             x_logsnr  = float(parts[2])
        #             y_mag     = float(parts[3].replace('+',''))
        #             x_vals.append(x_logsnr)
        #             y_vals.append(y_mag)

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
        # plt.xlabel(r'$-2.5\, -2.5 \, log_{10}(SNR)$')
        plt.xlabel(r'$-2.5 \, log_{10}(SNR)$')
        plt.ylabel('Apparent Meteor Magnitude')
        # take the file name from snr_file
        file_name = os.path.basename(snr_file)
        # for the title take the name of the file all the way with either tavis or elgin isode like for 2019-07-26_03_08_46elgin_SNR_values.png only 2019-07-26_03_08_46elgin
        title_name = ""
        if "tavis" in file_name:
            title_name = file_name.split("tavis")[0]
            title_name = title_name + "tavis"
            # delete the - from the title_name
            title_name = title_name.replace("-", "")
            # delete all the _ but keep the first one
            title_name = title_name.replace("_", "")
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

        if title_name == "":
            title_name = file_name
            if "detectinfo_" in file_name:
                title_name = title_name.split("detectinfo_")[1]
            if "_0000000_manual" in file_name:
                title_name = title_name.split("_0000000_manual")[0]

        # Example new data
        new_data = pd.DataFrame({
            r'$-2.5 \, log_{10}(SNR)$': x_inliers, 
            'Apparent Meteor Magnitude': y_inliers, 
            'file': title_name
        })

        # Append new data
        all_data = pd.concat([all_data, new_data], ignore_index=True)

        plt.title(title_name)
        # plt.title('log10SNR vs mag_data')
        plt.legend()
        plt.grid(True)

        base_name = os.path.basename(snr_file)
        file_root, _ = os.path.splitext(base_name)
        plot_filename = os.path.join(output_dir, f"{file_root}.png")

        plt.savefig(plot_filename, dpi=300)
        plt.close()

        # Write line function
        with open(all_linefuncts_path, 'a') as f:
            f.write(f"{base_name}: y = {m_final:.4f}x + {c_final:.4f}\n")

        print(f"Processed {snr_file} -> {plot_filename}")

    # check if the all_data is empty
    if all_data.empty:
        print("No valid data to plot.")
        return
    # create a plot with for the pandas dataframe 
    plt.figure()    
    sns.lmplot(data=all_data, x=r'$-2.5 \, log_{10}(SNR)$', y='Apparent Meteor Magnitude', hue='file', fit_reg=False)
    # # add the regression line
    # sns.regplot(data=all_data, x=r'$-2.5 \, log_{10}(SNR)$', y='Apparent Meteor Magnitude', scatter=False)
    # keep the fit to HAVE INCLINATION OF 1 AND FIND THE CONSTANT
    inclin=1
    # Calculate the best-fit constant (intercept)
    x_values = all_data[r'$-2.5 \, log_{10}(SNR)$']
    y_values = all_data['Apparent Meteor Magnitude']
    c_all = np.mean(y_values) - inclin * np.mean(x_values)

    # Calculate standard deviation of residuals
    residuals = y_values - (inclin * x_values + c_all)
    std_all = np.std(residuals)

    # Plot the regression line with slope=1 and calculated intercept
    x_range = np.linspace(x_values.min(), x_values.max(), 100)
    y_fit = inclin * x_range + c_all
    plt.plot(x_range, y_fit, color='black', linestyle='--', label=f"y = {inclin}x + {c_all:.4f}")

    # Add text with the calculated intercept and standard deviation
    plt.text(1.05, 0.9, f"y = x + {c_all:.4f} \nstand.dev {std_all:.4f}", 
            transform=plt.gca().transAxes, color='black')
    # grid on
    plt.grid(True)
    # cal the regresion line slope and constant and the standard deviation
    m_all, c_all = np.polyfit(all_data[r'$-2.5 \, log_{10}(SNR)$'], all_data['Apparent Meteor Magnitude'], 1)
    # calc standard deviation
    std_all = np.std(all_data['Apparent Meteor Magnitude'] - (m_all * all_data[r'$-2.5 \, log_{10}(SNR)$'] + c_all))
    # write the value of the slope and the constant and the standard deviation
    # plt.text(1.05, 0.9, f"y = {m_all:.4f}x + {c_all:.4f} \nstand.dev {std_all:.4f}", transform=plt.gca().transAxes, color='tab:blue')
    # # write the value of the slope and the constant
    # plt.text(0.05, 0.95, f"Mean of slope and stand.dev.: {mean_inclin:.4f} ± {std_inclin:.4f}\nMean of const and stand.dev.: {mean_const:.4f} ± {std_const:.4f}", transform=plt.gca().transAxes)
    plt.savefig(os.path.join(output_dir, "all_data.png"), dpi=300)
    plt.close()

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
        # write the value of the slope and the constant for the all data fit
        f.write(f"\nAll data fit: y = {m_all:.4f}x + {c_all:.4f} \nstand.dev {std_all:.4f}\n")


    print(f"\nAll line functions saved to: {all_linefuncts_path}")


def parse_rms_file(lines_or_path):
    """
    Parses RMS output lines to extract 'Mag' and 'SNR' columns.
    
    Parameters
    ----------
    lines_or_path : str or list of str
        A file path to read from or a list of lines to parse.

    Returns
    -------
    magnitudes : list of float
        The list of Magnitude values extracted.
    snrs : list of float
        The list of SNR values extracted.
    """

    # If the input is a filename, read all lines from the file first.
    if isinstance(lines_or_path, str):
        with open(lines_or_path, 'r') as f:
            lines = f.readlines()
    else:
        # Otherwise, assume it's already a list of lines
        lines = lines_or_path

    magnitudes = []
    snrs = []

    for line in lines:
        # Strip whitespace; skip empty or divider lines
        stripped = line.strip()
        if not stripped:
            continue
        
        # Split the line by whitespace
        parts = stripped.split()
        
        # Look for lines that have exactly 12 columns
        if len(parts) == 12:
            try:
                # By the data layout, 9th column is Mag, 11th column is SNR
                # parts index: 0=Frame#, 1=Col, 2=Row, 3=RA, 4=Dec, 5=Azim, 6=Elev,
                #             7=Inten, 8=Mag, 9=Bcknd, 10=SNR, 11=NSatPx
                mag = float(parts[8])
                snr = float(parts[10])
                
                magnitudes.append(mag)
                snrs.append(snr)
            except ValueError:
                # If conversion fails, skip this line
                continue

    return magnitudes, snrs


if __name__ == "__main__":
    # Example usage:
    # Replace 'path/to/directory' with the directory containing your SNR_values.txt files
    directory_with_txt = "/home/mvovk/Documents/2ndPaper/Reductions/CAP"
    run_skyfit_on_all_states(directory_with_txt)
    process_snr_files(directory_with_txt)
