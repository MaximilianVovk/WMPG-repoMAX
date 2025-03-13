"""
Import all the pickle files and get the luminosity of the first fram of all the files

Author: Maximilian Vovk
Date: 2025-03-04
"""

from DynNestSapl_metsim import *
from scipy import stats

from scipy.optimize import minimize
import numpy as np
import scipy.optimize as opt
import matplotlib.gridspec as gridspec
from scipy import stats

dir_pickle_files = r'C:\Users\maxiv\Documents\UWO\Papers\2)ORI-CAP-PER-DRA\Reductions\ORI'

const=0
shower=''
# if the last 3 letters of dir_pickle_files are DRA set const 8.0671
if dir_pickle_files[-3:] == 'DRA':
    const = 8.0671
    shower = 'DRA'
if dir_pickle_files[-3:] == 'CAP':
    const = 7.8009
    shower = 'CAP'  
if dir_pickle_files[-3:] == 'ORI':
    const = 7.3346
    shower = 'ORI'

# outputput fodler is C:\Users\maxiv\WMPG-repoMAX\Code\DynNestSampl\Validation\lum_noise\plots_+shower
out_folder = r'C:\Users\maxiv\WMPG-repoMAX\Code\DynNestSampl\Validation\noise_test\plots_'+shower

# create the output folder if it does not exist
if not os.path.exists(out_folder):
    os.makedirs(out_folder)





def cubic_lag(t, a, b, c, t0):
    """
    Quadratic lag function.
    """

    # Only take times <= t0
    t_before = t[t <= t0]

    # Only take times > t0
    t_after = t[t > t0]

    # Compute the lag linearly before t0
    l_before = np.zeros_like(t_before) # +c

    # Compute the lag quadratically after t0
    l_after = -abs(a)*(t_after - t0)**3 - abs(b)*(t_after - t0)**2

    c = 0

    lag_funct = np.concatenate((l_before, l_after))

    lag_funct = lag_funct - lag_funct[0]

    return lag_funct


def lag_residual_cub(params, t_time, l_data):
    """
    Residual function for the optimization.
    """

    return np.sum((l_data - cubic_lag(t_time, *params))**2)

def jacchia_Lag(t, a1, a2):
    return -np.abs(a1) * np.exp(np.abs(a2) * t)

def lag_residual(params, t_time, l_data):
    """
    Residual function for the optimization.
    """
    return np.sum((l_data - jacchia_Lag(t_time, *params))**2)

def fit_lag(lag_data,time_data):

    # Initial guess for the parameters
    initial_guess = [0.005,	10]
    result = minimize(lag_residual, initial_guess, args=(time_data, lag_data))
    fitted_params = result.x
    fitted_lag = jacchia_Lag(time_data, *fitted_params)
    residuals = lag_data - fitted_lag
    # avg_residual = np.mean(abs(residuals)) #RMSD
    rmsd = np.sqrt(np.mean(residuals**2))

    # initial guess of deceleration decel equal to linear fit of velocity
    p0 = [np.mean(lag_data), 0, 0, np.mean(time_data)]

    opt_res = opt.minimize(lag_residual_cub, p0, args=(np.array(time_data), np.array(lag_data)), method='Nelder-Mead')

    # sample the fit for the velocity and acceleration
    a_t0, b_t0, c_t0, t0 = opt_res.x
    fitted_lag_t0 = cubic_lag(time_data, a_t0, b_t0, c_t0, t0)
    residuals_t0 = lag_data - fitted_lag_t0
    # avg_residual = np.mean(abs(residuals))
    rmsd_t0 = np.sqrt(np.mean(residuals_t0**2))

    # intrpoate the lag to the time_data that goes from 0 to the max time for 1000 points
    time_data_1000 = np.linspace(0, max(time_data), 1000)
    # interpolate the lag base on the new time
    fitted_lag_t0 = cubic_lag(time_data_1000, a_t0, b_t0, c_t0, t0)


    # if rmsd_t0<rmsd:
    #     return fitted_lag_t0, residuals_t0, rmsd_t0,'Polin t0'
    # else:
    #     return spline_fit, residuals, rmsd,'Jacchia Fit'

    # # Prepare data
    # x_train = np.array(time_data).reshape(-1, 1)
    # y_train = np.array(lag_data/1000)

    # # Instantiate and train the model
    # reg = GradientBoostingRegressor()
    # reg.fit(x_train, y_train)

    # # Predict
    # y_fit = reg.predict(x_train)*1000
    
    # residuals_gradboost = y_train*1000 - y_fit
    # # avg_residual = np.mean(abs(residuals))
    # rmsd_gradboost = np.sqrt(np.mean(residuals_gradboost**2))

    # return y_fit, residuals_gradboost, rmsd_gradboost,'Polin t0'

    # return spline_fit, residuals, rmsd,'jacchia Fit'
    return fitted_lag_t0, residuals_t0, rmsd_t0, time_data_1000,'Polin t0'


def plot_side_by_side(obs_data, file_name='', output=''):
    fig = plt.figure(figsize=(14,6), dpi=300)

    # Use the user-specified GridSpec
    gs_main = gridspec.GridSpec(2, 3, figure=fig,
                                height_ratios=[3, 1],
                                width_ratios=[1, 0.3, 1])

    # Define colormap
    cmap = plt.get_cmap("tab10")
    station_colors = {}  # Dictionary to store colors assigned to stations

    # Adjust subplots: give some horizontal space (wspace) so there's separation between the pairs (5 & 6)
    # We'll later manually remove space between (4,5) and (6,7) by adjusting positions.
    fig.subplots_adjust(wspace=0.3, hspace=0.3)

    # Top row
    ax0 = fig.add_subplot(gs_main[0,0])  # Lag
    ax1 = fig.add_subplot(gs_main[0,1])  # Empty
    ax2 = fig.add_subplot(gs_main[0,2])  # Empty

    # Bottom row
    # Residual lag and hist lag
    ax4 = fig.add_subplot(gs_main[1,0])        # Lag residual
    ax5 = fig.add_subplot(gs_main[1,1], sharey=ax4)  # Lag hist
    ax6 = fig.add_subplot(gs_main[1,2])        # Mag residual

    # --- Plotting Data --- #

    # for each station in obs_data_plot
    for station in np.unique(obs_data.stations_lum):
        # Assign a unique color if not already used
        if station not in station_colors:
            station_colors[station] = cmap(len(station_colors) % 10)  # Use modulo to cycle colors
        # plot the height vs. absolute_magnitudes
        ax2.plot(obs_data.apparent_magnitudes[np.where(obs_data.stations_lum == station)], \
                 obs_data.height_lum[np.where(obs_data.stations_lum == station)]/1000, 'x--', \
                 color=station_colors[station], label=station)

    apparent_mag = np.max(obs_data.apparent_magnitudes)
    # find the index of the apparent magnitude in the list
    index = np.where(np.array(obs_data.apparent_magnitudes) == apparent_mag)[0][0]
    SNR_case = 10**((apparent_mag-const)/(-2.5))    
    lum_noise = obs_data.luminosity[index]/SNR_case   

    # put a circe around the max value
    # ax2.plot(np.max(obs_data.apparent_magnitudes), obs_data.height_lum[np.where(obs_data.apparent_magnitudes == np.max(obs_data.apparent_magnitudes))[0][0]]/1000,
    #           'ro', markersize=10, label=f'Max App.Mag. {apparent_mag:.2f}\nMin Lum {obs_data.luminosity[index]:.2f}\nSNR {SNR_case:.2f}\nLum/SNR {lum_noise:.2f}')
    
    ax2.scatter(np.max(obs_data.apparent_magnitudes), obs_data.height_lum[np.where(obs_data.apparent_magnitudes == np.max(obs_data.apparent_magnitudes))[0][0]]/1000,
                          s=200, edgecolors='red', facecolors='none', linewidth=2, label=f'Min Lum {obs_data.luminosity[index]:.2f}\nSNR {SNR_case:.2f}\nLum/SNR {lum_noise:.2f}')

    ax2.set_xlabel('App. Meteor Mag.')
    ax2.set_ylabel('Height [km]')
    ax2.legend()
    ax2.invert_xaxis()
    ax2.grid(True)


    fitted_lag, residuals_lag, avg_residual_lag, time_data, labels = fit_lag(obs_data.lag, obs_data.time_lag)
    # for each station in obs_data_plot
    for station in np.unique(obs_data.stations_lag):
        # Assign a unique color if not already used
        if station not in station_colors:
            station_colors[station] = cmap(len(station_colors) % 10)  # Use modulo to cycle colors
        # plot the height vs. absolute_magnitudes
        ax0.plot(obs_data.time_lag[np.where(obs_data.stations_lag == station)], \
                 obs_data.lag[np.where(obs_data.stations_lag == station)], 'x:', \
                 color=station_colors[station], label=station)
        
    ax0.plot(time_data, fitted_lag, 'k-', label='Model\nRMSD: {:.1f} m'.format(avg_residual_lag))
    ax0.set_xlabel('Time [s]')
    ax0.set_ylabel('Lag [m]')
    ax0.legend()
    ax0.grid(True)

    # Empty plot top row
    ax1.axis('off')


    # put the zer line in dark gray
    ax4.axhline(0, color='darkgray', linewidth=1.5)
    # Lag residual
    ax4.plot(obs_data.time_lag, residuals_lag, 'k.')
    ax4.set_xlabel('Time [s]')
    ax4.set_ylabel('Residual [m]')
    ax4.grid(True)

    # Lag histogram
    ax5.hist(residuals_lag, bins=20, orientation='horizontal', color='k')
    ax5.set_xlabel('Count')
    ax5.set_ylabel('')  # no label
    # We'll remove tick labels later with tick_params
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)
    # ax5.spines['left'].set_visible(False)
    ax5.grid(True)

    # Empty plot top row
    ax6.axis('off')

    # --- Manually adjust positions to remove space between (4,5) and (6,7) --- #
    pos4 = ax4.get_position()
    pos5 = ax5.get_position()
    # ax5.set_position([pos4.x1, pos5.y0, pos5.width, pos5.height])  # no space between 4 & 5
    # put 0.01 space between the plots
    ax5.set_position([pos4.x1+0.02, pos5.y0, pos5.width, pos5.height])


    # --- Adjust tick labels at the end ---
    # For ax4 and ax6, we want y tick labels:
    ax4.tick_params(labelleft=True)

    # For ax5 and ax7, no y tick labels:
    ax5.tick_params(labelleft=False)

    plt.savefig(os.path.join(output, file_name+'_fit_noise_std_dev.png'))
    plt.close()

    return avg_residual_lag




# Use the class to find .dynesty, load prior, and decide output folders
finder = find_dynestyfile_and_priors(input_dir_or_file=dir_pickle_files,use_CAMO_data=True)

# check if finder is empty
if not finder.base_names:
    print("No files found in the input directory.")
    sys.exit()

lum_list = []
abs_mag_list = []
base_name_list = []
lag_RMSD_list_EMCCD = []
lag_RMSD_list_CAMO = []

# check if a file with the name "log"+n_PC_in_PCA+"_"+str(len(df_sel))+"ev.txt" already exist
if os.path.exists(out_folder+os.sep+"log_lum_"+shower+".txt"):
    # remove the file
    os.remove(out_folder+os.sep+"log_lum_"+shower+".txt")
sys.stdout = Logger(out_folder,"log_lum_"+shower+".txt") # 

avg_residuals = []
print(shower)
print('linear fit x:-2.5log10(SNR) y:Apparent Meteor Magnitude')
print('y = x +',const)
print('')
print('\hline')
print('Name & RMSD lag [m] & MAX App.Mag. & MIN Lum [J/s] & SNR & Lum/SNR [J/s] \\\\')
print('\hline')

print(f"Pickle files in {dir_pickle_files}")
for i, (base_name, dynesty_info, prior_path, out_new_folder) in enumerate(zip(
    finder.base_names,
    finder.input_folder_file,
    finder.priors,
    finder.output_folders
)):
    dynesty_file, pickle_file, bounds, flags_dict, fixed_values = dynesty_info
    obs_data = finder.observation_instance(base_name)  # Get the correct instance
    
    if obs_data:  # Ensure the instance exists
        obs_data.file_name = pickle_file  # Update file name if needed

        # this is the lowest apparent magnitude and I have to get the same index in the luminosity list
        apparent_mag = np.max(obs_data.apparent_magnitudes)
        # find the index of the apparent magnitude in the list
        index = np.where(np.array(obs_data.apparent_magnitudes) == apparent_mag)[0][0]
        lum_list.append(obs_data.luminosity[index])
        abs_mag_list.append(apparent_mag)
        base_name_list.append(base_name)

        # check if among the np.unique(obs_data.stations_lag) there is sothing like 01T or 02T
        lag_camers = np.unique(obs_data.stations_lag)
        if any(item.endswith('1T') or item.endswith('2T') for item in lag_camers):
            # create a panda dataframe with obs1 and then obs2 only where
            RMSD_lag = plot_side_by_side(obs_data, 'CAMO_'+base_name, out_folder)
            lag_RMSD_list_CAMO.append(RMSD_lag)
            print(f'CAMO ${base_name}$ & {RMSD_lag:.1f} & {apparent_mag:.2f} & {obs_data.luminosity[index]:.2f} & {10**((apparent_mag-const)/(-2.5)):.2f} & {obs_data.luminosity[index]/10**((apparent_mag-const)/(-2.5)):.2f} \\\\')
        else:
            # create a panda dataframe with obs1 and then obs2 only where
            RMSD_lag = plot_side_by_side(obs_data, base_name, out_folder)
            lag_RMSD_list_EMCCD.append(RMSD_lag)
            print(f'${base_name}$ & {RMSD_lag:.1f} & {apparent_mag:.2f} & {obs_data.luminosity[index]:.2f} & {10**((apparent_mag-const)/(-2.5)):.2f} & {obs_data.luminosity[index]/10**((apparent_mag-const)/(-2.5)):.2f} \\\\')
        # print('lag noise',obs_data.noise_lag,'lum noise',obs_data.noise_lum)
        print('\hline') 

    else:
        print(f"Warning: No observation instance found for {base_name}")

# make the mean of the luminosity and apparent magnitude
mean_lum = np.mean(lum_list)
mean_abs_mag = np.mean(abs_mag_list)

print("")
print("linear fit x:-2.5log10(SNR) y:Apparent Meteor Magnitude")
print(f"y = x + {const}")
print("")
print("MEAN:")
print(f"Mean apparent magnitude of the dimmest frame: {mean_abs_mag}")
print(f"Mean luminosity of the dimmest frame: {mean_lum}")
SNR_val_mean = 10**((mean_abs_mag-const)/(-2.5))
print(f"SNR from line: {SNR_val_mean}")
print(f"MEAN lum noise : {mean_lum/SNR_val_mean}")
print(f'MEAN lag EMCCD RMSD mean: {np.mean(lag_RMSD_list_EMCCD)}')
if lag_RMSD_list_CAMO :
    print(f'MEAN lag CAMO RMSD mean: {np.mean(lag_RMSD_list_CAMO)}')


# find the kde mode of the abs_mag_list and lum_list
kde_abs_mag = stats.gaussian_kde(abs_mag_list)
kde_lum = stats.gaussian_kde(lum_list)
mode_abs_mag = abs_mag_list[np.argmax(kde_abs_mag(abs_mag_list))]
mode_lum = lum_list[np.argmax(kde_lum(lum_list))]

# do the mode of the lag_RMSD_list
kde_lag_RMSD_EMCCD = stats.gaussian_kde(lag_RMSD_list_EMCCD)
mode_lag_RMSD_CAMO = lag_RMSD_list_EMCCD[np.argmax(kde_lag_RMSD_EMCCD(lag_RMSD_list_EMCCD))]

print("")
print("MODE:")
print(f"Mode of apparent magnitude: {mode_abs_mag}")
print(f"Mode of luminosity: {mode_lum}")
SNR_val_mode = 10**((mode_abs_mag-const)/(-2.5))
print(f"SNR from line: {SNR_val_mode}")
print(f"MODE lum noise : {mode_lum/SNR_val_mode}")
print(f'MODE lag EMCCD RMSD mode: {mode_lag_RMSD_CAMO}')

# check if not epty the lag_RMSD_list_CAMO and more than 2 elements
if lag_RMSD_list_CAMO and len(lag_RMSD_list_CAMO)>2:
    kde_lag_RMSD_CAMO = stats.gaussian_kde(lag_RMSD_list_CAMO)
    mode_lag_RMSD_CAMO = lag_RMSD_list_CAMO[np.argmax(kde_lag_RMSD_CAMO(lag_RMSD_list_CAMO))]
    print(f'MODE lag CAMO RMSD mode: {mode_lag_RMSD_CAMO}')
elif lag_RMSD_list_CAMO and len(lag_RMSD_list_CAMO)<=2:
    print(f'MODE (only 2 or less) lag CAMO RMSD mode: {np.mean(lag_RMSD_list_CAMO)}')


# Close the Logger to ensure everything is written to the file STOP COPY in TXT file
sys.stdout.close()
# Reset sys.stdout to its original value if needed
sys.stdout = sys.__stdout__



