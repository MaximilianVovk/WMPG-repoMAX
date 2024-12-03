import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

from scipy.interpolate import UnivariateSpline
from scipy.interpolate import CubicSpline, make_interp_spline
from scipy.optimize import minimize
import numpy as np
import numpy as np
import scipy.optimize as opt
from sklearn.ensemble import GradientBoostingRegressor

def read_and_process_pickle(file_path):
    with open(file_path, 'rb') as f:
        traj = pickle.load(f, encoding='latin1')

    v_avg = traj.v_avg
    
    obs_data = []
    for obs in traj.observations[:2]:
        if obs.station_id == '01G' or obs.station_id == '02G' or obs.station_id == '01F' or obs.station_id == '02F':
            obs_dict = {
                'v_init': obs.v_init,
                'velocities': np.array(obs.velocities),
                'model_ht': np.array(obs.model_ht),
                'absolute_magnitudes': np.array(obs.absolute_magnitudes),
                'lag': np.array(obs.lag),
                'length': np.array(obs.length),
                'time_data': np.array(obs.time_data),
                'station_id': obs.station_id
            }
            obs_dict['velocities'][0] = obs_dict['v_init']
            obs_data.append(obs_dict)
            # delete any data that absolute_magnitudes is above 8
            index_abs_mag = [i for i in range(len(obs_dict['absolute_magnitudes'])) if obs_dict['absolute_magnitudes'][i] > 8]
            # delete from all the lists
            for key in ['velocities', 'model_ht', 'absolute_magnitudes', 'lag', 'length', 'time_data']:
                obs_dict[key] = np.delete(obs_dict[key], index_abs_mag)
    
    # print the station_id
    # print('station_id:', obs_data[0]['station_id'], obs_data[1]['station_id'])

    # Save distinct values for the two observations
    obs1, obs2 = obs_data[0], obs_data[1]
    
    # Extend and concatenate the observations
    for key in ['velocities', 'model_ht', 'absolute_magnitudes', 'lag', 'length', 'time_data']:
        obs1[key] = np.concatenate((obs1[key], obs2[key]))

    # Order the observations based on time
    sorted_indices = np.argsort(obs1['time_data'])
    for key in ['time_data', 'velocities', 'model_ht', 'absolute_magnitudes', 'lag', 'length']:
        obs1[key] = obs1[key][sorted_indices]

    return obs1, obs2


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


def cubic_acceleration(t, a, b, t0):
    """
    Quadratic acceleration function.
    """

    # Only take times <= t0
    t_before = t[t <= t0]

    # Only take times > t0
    t_after = t[t > t0]

    # No deceleration before t0
    a_before = np.zeros_like(t_before)

    # Compute the acceleration quadratically after t0
    a_after = -6*abs(a)*(t_after - t0) - 2*abs(b)

    return np.concatenate((a_before, a_after))

def lag_residual_parab(params, t_time, l_data):
    """
    Residual function for the optimization.
    """

    return np.sum((l_data - cubic_lag(t_time, *params))**2)


def fit_spline(data, time_data,spli=''):
    if spli == '':
        spline = UnivariateSpline(time_data, data)
    else:
        spline = UnivariateSpline(time_data, data, s=spli)
    spline_fit = spline(time_data)
    residuals = data - spline_fit
    # avg_residual = np.mean(np.abs(residuals))
    rmsd = np.sqrt(np.mean(residuals**2))

    # Select the data up to the minimum value
    x1 = time_data[:np.argmin(data)]
    y1 = data[:np.argmin(data)]

    # Fit the first parabolic curve
    coeffs1 = np.polyfit(x1, y1, 2)
    fit1 = np.polyval(coeffs1, x1)

    # Select the data from the minimum value onwards
    x2 = time_data[np.argmin(data):]
    y2 = data[np.argmin(data):]

    # Fit the second parabolic curve
    coeffs2 = np.polyfit(x2, y2, 2)
    fit2 = np.polyval(coeffs2, x2)

    # concatenate fit1 and fit2
    fit1=np.concatenate((fit1, fit2))

    residuals_pol = data - fit1
    # avg_residual_pol = np.mean(abs(residuals_pol))
    rmsd_pol = np.sqrt(np.mean(residuals_pol**2))

    # intrpoate the lag to the time_data that goes from 0 to the max time for 1000 points
    time_data_1000_1 = np.linspace(0, max(x1), 500)
    # Fit the first parabolic curve
    fit1_1000 = np.polyval(coeffs1, time_data_1000_1)

    time_data_1000_2 = np.linspace(min(x2), max(x2), 500)
    # Fit the second parabolic curve
    fit2_1000 = np.polyval(coeffs2, time_data_1000_2)

    # concatenate fit1 and fit2
    fit1_1000=np.concatenate((fit1_1000, fit2_1000))
    time_data_1000=np.concatenate((time_data_1000_1, time_data_1000_2))

    # if rmsd_pol<rmsd:
    #     return fit1, residuals_pol, rmsd_pol,'Polinomial Fit'
    # else:
    #     return spline_fit, residuals, rmsd,'Spline Fit'

    # return spline_fit, residuals, rmsd,'Spline Fit'
    return fit1_1000, residuals_pol, rmsd_pol, time_data_1000,'Polinomial Fit'

def jacchia_Lag(t, a1, a2):
    return -np.abs(a1) * np.exp(np.abs(a2) * t)

def lag_residual(params, t_time, l_data):
    """
    Residual function for the optimization.
    """
    return np.sum((l_data - jacchia_Lag(t_time, *params))**2)

def fit_lag(lag_data,time_data,spli = ''):

    if spli == '':
        spline = UnivariateSpline(time_data, lag_data)
    else:
        spline = UnivariateSpline(time_data, lag_data, s=spli)

    spline_fit = spline(time_data)
    residuals = lag_data - spline_fit
    # avg_residual = np.mean(np.abs(residuals))
    rmsd = np.sqrt(np.mean(residuals**2))

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

    opt_res = opt.minimize(lag_residual_parab, p0, args=(np.array(time_data), np.array(lag_data)), method='Nelder-Mead')

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

def fit_cubic_spline(data, time_data):
    spline = CubicSpline(time_data, data)
    spline_fit = spline(time_data)
    residuals = data - spline_fit
    # avg_residual = np.mean(abs(residuals))
    rmsd = np.sqrt(np.mean(residuals**2))
    return spline_fit, residuals, rmsd

def fit_interp_spline(data, time_data, k=3):
    spline = make_interp_spline(time_data, data, k=k)
    spline_fit = spline(time_data)
    residuals = data - spline_fit
    # avg_residual = np.mean(abs(residuals))
    rmsd = np.sqrt(np.mean(residuals**2))
    return spline_fit, residuals, rmsd

def plot_side_by_side(obs1, obs2,file,num):
    # Plot the simulation results
    fig, ax = plt.subplots(2, 4, figsize=(14, 6),gridspec_kw={'height_ratios': [ 3, 1],'width_ratios': [ 3, 0.5, 3, 0.5]}, dpi=300) #  figsize=(10, 5), dpi=300 0.5, 3, 3, 0.5

    # flat the ax
    ax = ax.flatten()

    ax[0].plot(obs1['time_data'], obs1['lag'], 'o', label=f'{obs1["station_id"]}')
    ax[0].plot(obs2['time_data'], obs2['lag'], 'o', label=f'{obs2["station_id"]}')
    fitted_lag, residuals_lag, avg_residual_lag, time_data, labels = fit_lag(obs1['lag'],obs1['time_data'],spli=100000)
    ax[0].plot(time_data, fitted_lag, 'k-', label=labels)
    # spline_fit, residuals, avg_residual_lag = fit_spline(obs1['lag'] / 1000 , obs1['time_data'])
    # plt.plot(obs1['time_data'], spline_fit*1000, 'k-', label='Spline Fit')
    # spline_fit, residuals_lag, avg_residual_lag = fit_cubic_spline(obs1['lag'] / 1000, obs1['time_data'])
    # plt.plot(obs1['time_data'], spline_fit, 'r-', label='Cubic Spline Fit')
    # spline_fit, residuals, avg_residual_lag = fit_interp_spline(obs1['lag'] / 1000, obs1['time_data'])
    # plt.plot(obs1['time_data'], spline_fit, 'g-', label='Interpolated Spline Fit')
    ax[0].set_xlabel('Time [s]')
    ax[0].set_ylabel('Lag [m]')
    ax[0].title.set_text(f'Lag - RMSD: {avg_residual_lag:.2f}')
    ax[0].legend()
    ax[0].grid()

    # delete the plot in the middle
    ax[1].axis('off')

    ax[2].plot(obs1['time_data'], obs1['absolute_magnitudes'], 'o', label=f'{obs1["station_id"]}')
    ax[2].plot(obs2['time_data'], obs2['absolute_magnitudes'], 'o', label=f'{obs2["station_id"]}')
    spline_fit, residuals_mag, avg_residual,time_data,label_fit = fit_spline(obs1['absolute_magnitudes'], obs1['time_data'])
    ax[2].plot(time_data, spline_fit, 'k-', label=label_fit)
    ax[2].set_xlabel('Time [s]')
    ax[2].set_ylabel('Absolute Magnitude [-]')
    # flip the y-axis
    ax[2].invert_yaxis()
    ax[2].title.set_text(f'Absolute Magnitude - RMSD: {avg_residual:.2f}')
    ax[2].legend()
    ax[2].grid()
    # delete the plot in the middle
    ax[3].axis('off')
    name= file.replace('_trajectory.pickle','')
    # put as the super title the name
    plt.suptitle(name)


    # plot the residuals against time
    ax[4].plot(obs1['time_data'], residuals_lag, 'ko', label=f'{obs1["station_id"]}')
    ax[4].set_xlabel('Time [s]')
    ax[4].set_ylabel('Residual [m]')
    # ax[2].title(f'Lag Residuals')
    # ax[2].legend()
    ax[4].grid()

    # plot the distribution of the residuals along the y axis
    ax[5].hist(residuals_lag, bins=20, orientation='horizontal', color='k')
    ax[5].set_xlabel('N.data')
    ax[5].set_ylabel('Residual [m]')
    # delete the the the line at the top ad the right
    ax[5].spines['top'].set_visible(False)
    ax[5].spines['right'].set_visible(False)
    # do not show the y ticks
    # ax[5].set_yticks([])
    # # show the zero line
    # ax[5].axhline(0, color='k', linewidth=0.5)
    # grid on
    ax[5].grid()

    # plot the residuals against time
    ax[6].plot(obs1['time_data'], residuals_mag, 'ko', label=f'{obs1["station_id"]}')
    ax[6].set_xlabel('Time [s]')
    ax[6].set_ylabel('Residual [-]')
    ax[6].invert_yaxis()
    # ax[3].title(f'Absolute Magnitude Residuals')
    # ax[3].legend()
    ax[6].grid()

    # plot the distribution of the residuals along the y axis
    ax[7].hist(residuals_mag, bins=20, orientation='horizontal', color='k')
    ax[7].set_xlabel('N.data')
    # invert the y axis
    ax[7].invert_yaxis()
    ax[7].set_ylabel('Residual [-]')
    # delete the the the line at the top ad the right
    ax[7].spines['top'].set_visible(False)
    ax[7].spines['right'].set_visible(False)
    # do not show the y ticks
    # ax[7].set_yticks([])
    # # show the zero line
    # ax[7].axhline(0, color='k', linewidth=0.5)
    # grid on
    ax[7].grid()



    plt.tight_layout()
    
    # ouput
    output=r'C:\Users\maxiv\WMPG-repoMAX\Code\PCA\manual_reduce'

    # change the name of the file to be able to save it
    name_for_table=name.replace("_","\\_")
    # print the name with an \ close to _ to be able to copy and paste in latex
    print(f'${name_for_table}$ & {avg_residual_lag:.2f} & {avg_residual:.2f} \\\\')
    print('\hline')

    # Save the figure as file with instead of _trajectory.pickle it has file+std_dev.png on the desktop
    plt.savefig(os.path.join(output, file.replace('_trajectory.pickle', '_n'+str(num)+'_std_dev.png')))

    plt.close()
    return avg_residual_lag, avg_residual

    

# Define the directory path
directory = r'C:\Users\maxiv\Desktop\test_pickl'
directory = r'C:\Users\maxiv\Documents\UWO\Papers\1)PCA\Reductions\manual_reductions'
directory = r'C:\Users\maxiv\Documents\UWO\Papers\1)PCA\Reductions\PER_EMCCD_centroid'
# Prepare a list to store average residuals
avg_residuals = []
print('\hline')
print('Name & RMSD lag [m] & RMSD mag [-] \\\\')
print('\hline')

all_residuals_lag = []
all_residuals_mag = []
ii=0
# Walk through the directory and find pickle files
for root, dirs, files in os.walk(directory):
    for file in files:
        if file.endswith('_trajectory.pickle'):
            # print('-----------------')
            # print('file:', file)
            file_path = os.path.join(root, file)
            combined_data, obs2 = read_and_process_pickle(file_path)
            
            time_data = combined_data['time_data']
            ii+=1
            # Plot lag and absolute magnitudes side by side
            avg_residual_lag, avg_residual = plot_side_by_side(combined_data, obs2,file,ii)
            all_residuals_lag.append(avg_residual_lag)
            all_residuals_mag.append(avg_residual)

# print the average of the residuals
print('Average of the residuals')
print('RMSD lag:', np.mean(all_residuals_lag))
print('RMSD mag:', np.mean(all_residuals_mag))

