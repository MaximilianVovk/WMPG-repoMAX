"""
The code is used to extract the physical properties of the simulated showers from observations
by selecting the most similar simulated events using a montecarlo method. 
The code is used to :
- Generate the simulated meteors for given observations
- Extract the physical properties of the most similar simulated showers from observations
"""

import json
import copy
import re
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.gridspec as gridspec
# import matplotlib
# matplotlib.use('Agg')
from numpy.linalg import inv
import numpy as np
import subprocess
import glob
import os
import pickle
import seaborn as sns
import scipy.spatial.distance
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import wmpl
import shutil
from wmpl.Utils.OSTools import mkdirP
from matplotlib.ticker import ScalarFormatter
from scipy.stats import gaussian_kde
from scipy.stats import norm
from scipy.stats import chi2
from wmpl.Utils.PyDomainParallelizer import domainParallelizer
from scipy.linalg import svd
from wmpl.MetSim.GUI import loadConstants, saveConstants,SimulationResults
from wmpl.MetSim.MetSimErosion import runSimulation, Constants, zenithAngleAtSimulationBegin
from scipy.interpolate import interp1d
from matplotlib.colors import Normalize
from scipy.optimize import minimize
import scipy.optimize as opt
import sys
from scipy.stats import zscore
import scipy.spatial
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from wmpl.Utils.AtmosphereDensity import fitAtmPoly
from sklearn.cluster import KMeans
import scipy.stats as stats
from sklearn.preprocessing import PowerTransformer
from wmpl.MetSim.ML.GenerateSimulations import generateErosionSim,saveProcessedList,MetParam
from wmpl.Utils.TrajConversions import J2000_JD, date2JD
import warnings
import itertools
import time
from multiprocessing import Pool
from multiprocessing import cpu_count

# CONSTANTS ###########################################################################################

NAME_SUFX_GENSIM = "_GenSim"
NAME_SUFX_CSV_OBS = "_obs.csv"
NAME_SUFX_CSV_SIM = "_sim.csv"
NAME_SUFX_CSV_SIM_NEW = "_sim_new.csv"
NAME_SUFX_CSV_CURRENT_FIT = "_fit_sim.csv"
NAME_SUFX_CSV_PHYSICAL_FIT_RESULTS = "_physical_prop.csv"

SAVE_SELECTION_FOLDER='Selection'
VAR_SEL_DIR_SUFX = '_sel_var_vs_physProp'
PCA_SEL_DIR_SUFX = '_sel_PCA_vs_physProp'

# these may change though the script
SAVE_RESULTS_FINAL_FOLDER='Results'

# sensistivity lvl mag of camera
CAMERA_SENSITIVITY_LVL_MAG = np.float64(0.1)
# sensistivity lvl mag of camera
CAMERA_SENSITIVITY_LVL_LEN = np.float64(0.005)*1000
# Length of data that will be used as an input during training
DATA_LENGTH = 256
# Default number of minimum frames for simulation
MIN_FRAMES_VISIBLE = 4

# Define the maximum difference in magnitude allowed
MAX_MAG_DIFF = 1
# Penalty thresholds
TIME_THRESHOLD = 1  # frames
HEIGHT_THRESHOLD = 1  # km

# python -m EMCCD_PCA_Shower_PhysProp "C:\Users\maxiv\Documents\UWO\Papers\1)PCA\PCA_Error_propagation\TEST" "PER" "C:\Users\maxiv\Documents\UWO\Papers\1)PCA\PCA_Error_propagation" 1000
# python -m EMCCD_PCA_Shower_PhysProp "C:\Users\maxiv\Documents\UWO\Papers\1)PCA\PCA_Error_propagation\TEST" "PER" "C:\Users\maxiv\Documents\UWO\Papers\1)PCA\PCA_Error_propagation" 1000 > output.txt    

# MATH FUNCTIONS ###########################################################################################

def find_closest_index(time_arr, time_sampled):
    closest_indices = []
    for sample in time_sampled:
        closest_index = min(range(len(time_arr)), key=lambda i: abs(time_arr[i] - sample))
        closest_indices.append(closest_index)
    return closest_indices


def cubic_lag(t, a, b, c, t0):
    """
    Quadratic lag function.
    """

    # Only take times <= t0
    t_before = t[t <= t0]

    # Only take times > t0
    t_after = t[t > t0]

    # Compute the lag linearly before t0
    l_before = np.zeros_like(t_before)#+c

    # Compute the lag quadratically after t0
    l_after = -abs(a)*(t_after - t0)**3 - abs(b)*(t_after - t0)**2 #+ c

    c=0

    return np.concatenate((l_before, l_after))


def cubic_velocity(t, a, b, v0, t0):
    """
    Quadratic velocity function.
    """

    # Only take times <= t0
    t_before = t[t <= t0]

    # Only take times > t0
    t_after = t[t > t0]

    # Compute the velocity linearly before t0
    v_before = np.ones_like(t_before)*v0

    # Compute the velocity quadratically after t0 lag_sampled=len_sampled-(vel_sampled[0]*time_sampled+len_sampled[0])
    v_after = -3*abs(a)*(t_after - t0)**2 - 2*abs(b)*(t_after - t0) + v0

    return np.concatenate((v_before, v_after))


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


def lag_residual(params, t_time, l_data):
    """
    Residual function for the optimization.
    """

    return np.sum((l_data - cubic_lag(t_time, *params))**2)


def vel_residual(params, t_time, l_data):
    """
    Residual function for the optimization.
    """

    return np.sum((l_data - cubic_velocity(t_time, *params))**2)


def fit_mag_polin2_RMSD(data_mag, time_data):

    # Select the data up to the minimum value
    x1 = time_data[:np.argmin(data_mag)]
    y1 = data_mag[:np.argmin(data_mag)]

    # Fit the first parabolic curve
    coeffs1 = np.polyfit(x1, y1, 2)
    fit1 = np.polyval(coeffs1, x1)

    # Select the data from the minimum value onwards
    x2 = time_data[np.argmin(data_mag):]
    y2 = data_mag[np.argmin(data_mag):]

    # Fit the second parabolic curve
    coeffs2 = np.polyfit(x2, y2, 2)
    fit2 = np.polyval(coeffs2, x2)

    # concatenate fit1 and fit2
    fit1=np.concatenate((fit1, fit2))

    residuals_pol = data_mag - fit1
    # avg_residual_pol = np.mean(abs(residuals_pol))
    rmsd_pol = np.sqrt(np.mean(residuals_pol**2))

    return fit1, residuals_pol, rmsd_pol,'Polinomial Fit'


def fit_lag_t0_RMSD(lag_data, time_data, velocity_data):
    v_init = velocity_data[0]
    # initial guess of deceleration decel equal to linear fit of velocity
    p0 = [np.mean(lag_data), 0, 0, np.mean(time_data)]
    opt_res = opt.minimize(lag_residual, p0, args=(np.array(time_data), np.array(lag_data)), method='Nelder-Mead')
    a_t0, b_t0, c_t0, t0 = opt_res.x
    fitted_lag_t0 = cubic_lag(np.array(time_data), a_t0, b_t0, c_t0, t0)
    
    # Optimize velocity residual based on initial guess from lag residual
    opt_res_vel = opt.minimize(vel_residual, [a_t0, b_t0, v_init, t0], args=(np.array(time_data), np.array(velocity_data)), method='Nelder-Mead')
    a_t0_vel, b_t0_vel, v_init_vel, t0_vel = opt_res_vel.x
    fitted_vel_t0_vel = cubic_velocity(np.array(time_data), a_t0_vel, b_t0_vel, v_init_vel, t0_vel)

    fitted_vlag_t0_vel = cubic_lag(np.array(time_data), a_t0_vel, b_t0_vel, c_t0, t0_vel)
    
    # # Compute fitted velocity from original lag optimization
    # fitted_vel_t0_lag = cubic_velocity(np.array(time_data), a_t0, b_t0, v_init, t0)

    # Compute fitted velocity from original lag optimization
    fitted_vel_t0_lag = cubic_velocity(np.array(time_data), a_t0, b_t0, v_init_vel, t0)

    # # Compute fitted velocity from original lag optimization
    # fitted_vel_t0_lag_vel = cubic_velocity(np.array(time_data), a_t0, b_t0, v_init_vel, t0)
    
    # Calculate residuals
    residuals_vel_vel = velocity_data - fitted_vel_t0_vel
    residuals_vel_lag = velocity_data - fitted_vel_t0_lag
    
    rmsd_vel_vel = np.sqrt(np.mean(residuals_vel_vel ** 2))
    rmsd_vel_lag = np.sqrt(np.mean(residuals_vel_lag ** 2))
    
    # Choose the best fitted velocity based on RMSD
    if rmsd_vel_vel < rmsd_vel_lag:
        best_fitted_vel_t0 = fitted_vel_t0_vel
        best_a_t0, best_b_t0, best_t0 = a_t0_vel, b_t0_vel, t0_vel
    else:
        best_fitted_vel_t0 = fitted_vel_t0_lag
        best_a_t0, best_b_t0, best_t0 = a_t0, b_t0, t0

    # # plot the two curves of lag and velocity
    # fig, ax = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    # # flat the ax
    # ax = ax.flatten()
    # ax[0].plot(time_data, lag_data, 'go', label='Observation')
    # ax[0].plot(time_data, fitted_lag_t0, 'k--', label='Cubic Fit lag')
    # ax[0].plot(time_data, fitted_vlag_t0_vel, 'r--', label='Cubic Fit vel')
    # ax[0].set_xlabel('Time (s)')
    # ax[0].set_ylabel('Lag [m]')
    # ax[0].legend()
    # ax[1].plot(time_data, velocity_data, 'go', label='Observation')
    # ax[1].plot(time_data, fitted_vel_t0_lag, 'k--', label='Cubic Fit lag')
    # ax[1].plot(time_data, fitted_vel_t0_vel, 'r--', label='Cubic Fit vel')
    # ax[1].set_ylabel('Velocity (m/s)')
    # ax[1].set_xlabel('Time (s)')
    # ax[1].legend()
    # plt.show()

    fitted_acc_t0 = cubic_acceleration(np.array(time_data), best_a_t0, best_b_t0, best_t0)
    # lag can be wrong for short meteors but stil the RMSD will be the same as the scatter WILL NOT CHANGE
    residuals_t0 = lag_data - fitted_lag_t0
    rmsd_t0 = np.sqrt(np.mean(residuals_t0 ** 2))

    # # lag can be wrong for short meteors where velocity drops suddenly
    # fitted_lag_t0 = cubic_lag(np.array(time_data), best_a_t0, best_b_t0, c_t0, best_t0)

    return fitted_lag_t0, residuals_t0, rmsd_t0, 'Cubic Fit', best_fitted_vel_t0, residuals_vel_vel, fitted_acc_t0


def find_noise_of_data(data, fps=32, plot_case=False, output_folder='', file_name=''):
    '''
        Find the noise of the data
    '''
    # make a copy of data_obs
    data_obs = copy.deepcopy(data)

    fitted_lag_t0_lag, residuals_t0_lag, rmsd_t0_lag, fit_type_lag, fitted_vel_t0, residuals_t0_vel, fitted_acc_t0 = fit_lag_t0_RMSD(data_obs['lag'],data_obs['time'], data_obs['velocities'])
    # now do it for fit_mag_polin2_RMSD
    fit_pol_mag, residuals_pol_mag, rmsd_pol_mag, fit_type_mag = fit_mag_polin2_RMSD(data_obs['absolute_magnitudes'],data_obs['time'])

    # lag_sampled=len_sampled-(vel_sampled[0]*time_sampled+len_sampled[0])

    len_t0_extr= fitted_lag_t0_lag + (fitted_vel_t0[0]*data_obs['time'])

    # create a pd dataframe with fit_pol_mag and fitted_vel_t0 and time and height
    fit_funct = {
        'velocities': fitted_vel_t0,
        'height': data_obs['height'],
        'absolute_magnitudes': fit_pol_mag,
        'time': data_obs['time'],
        'lag': fitted_lag_t0_lag,
        'length': len_t0_extr,
        'rmsd_len' : rmsd_t0_lag/1000,
        'rmsd_mag' : rmsd_pol_mag,
        'rmsd_vel' : rmsd_t0_lag/1000*np.sqrt(2)/(1.0/fps),
        'fps': fps
    }
    
    data_obs['res_absolute_magnitudes'] = residuals_pol_mag
    data_obs['res_lag'] = residuals_t0_lag
    data_obs['res_velocities'] = residuals_t0_vel/1000

    # data['name'] is a path and I need aonly the name of the file
    plot_data_with_residuals_and_real(rmsd_pol_mag, rmsd_t0_lag/1000*np.sqrt(2)/(1.0/fps), rmsd_t0_lag/1000, fit_funct, data_obs, label_real=data['name'].split(os.sep)[-1], file_name=data['name'].split(os.sep)[-1]+'_fit_t0_polin_curve.png', output_dir = output_folder)

    if plot_case:
        fig, ax = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
        # flat the ax
        ax = ax.flatten()
        plot_side_by_side(data,fig, ax,'go','Obsevation')

        plot_side_by_side(fit_funct,fig, ax,'k--','fit')

        return rmsd_t0_lag, rmsd_pol_mag, fit_pol_mag, fitted_lag_t0_lag, fit_funct, fig, ax
    else:
        return rmsd_t0_lag, rmsd_pol_mag, fit_pol_mag, fitted_lag_t0_lag, fit_funct


#### Generate Observation #########################################################################

def generate_observation_realization(data, rmsd_lag, rmsd_mag, fit_funct, name='', fps=32, fig='', ax='', plot_case=False):

    # print a . so that the next will be on the same line
    print('.', end='')
    # make a copy of data_obs
    data_obs = copy.deepcopy(data)
    fit_pol_mag = copy.deepcopy(fit_funct['absolute_magnitudes'])
    fitted_lag_t0_lag = copy.deepcopy(fit_funct['lag'])
    fitted_lag_t0_vel = copy.deepcopy(fit_funct['velocities'])

    if name!='':
        # print(name)
        data_obs['name']=name

    data_obs['type']='Realization'

    ### ADD NOISE ###

    # Add noise to magnitude data (Gaussian noise) for each realization
    fit_pol_mag += np.random.normal(loc=0.0, scale=rmsd_mag, size=len(data_obs['absolute_magnitudes']))
    data_obs['absolute_magnitudes']=fit_pol_mag
    # Add noise to length data (Gaussian noise) for each realization
    fitted_lag_t0_lag += np.random.normal(loc=0.0, scale=rmsd_lag, size=len(data_obs['length']))
    data_obs['lag']=fitted_lag_t0_lag
    # add noise to velocity data considering the noise as rmsd_lag/(1.0/fps)
    # fitted_lag_t0_vel += np.random.normal(loc=0.0, scale=rmsd_lag/(1.0/fps), size=len(data_obs['velocities']))
    fitted_lag_t0_vel += np.random.normal(loc=0.0, scale=rmsd_lag*np.sqrt(2)/(1.0/fps), size=len(data_obs['velocities']))
    data_obs['velocities']=fitted_lag_t0_vel

    ### ###

    # data_obs['lag']=np.array(data_obs['length'])-(data_obs['v_init']*np.array(data_obs['time'])+data_obs['length'][0])
    data_obs['length']= np.array(data_obs['lag'])+(data_obs['v_init']*np.array(data_obs['time'])+data_obs['length'][0])

    # # get the new velocity with noise
    # for vel_ii in range(1,len(data_obs['time'])-1):
    #     diff_1=abs((data_obs['time'][vel_ii]-data_obs['time'][vel_ii-1])-1.0/fps)
    #     diff_2=abs((data_obs['time'][vel_ii+1]-data_obs['time'][vel_ii-1])-1.0/fps)

    #     if diff_1<diff_2:
    #         data_obs['velocities'][vel_ii]=(data_obs['length'][vel_ii]-data_obs['length'][vel_ii-1])/(data_obs['time'][vel_ii]-data_obs['time'][vel_ii-1])
    #     else:
    #         data_obs['velocities'][vel_ii+1]=(data_obs['length'][vel_ii+1]-data_obs['length'][vel_ii-1])/(data_obs['time'][vel_ii+1]-data_obs['time'][vel_ii-1])

    if plot_case:
        plot_side_by_side(data_obs,fig, ax)

    # compute the initial velocity
    data_obs['v_init']=data_obs['velocities'][0] # m/s
    # compute the average velocity
    data_obs['v_avg']=np.mean(data_obs['velocities']) # m/s

    # data_obs['v_avg']=data_obs['v_avg']*1000 # km/s

    pd_datfram_PCA = array_to_pd_dataframe_PCA(data_obs)

    return pd_datfram_PCA

#### No Metsim Initial guess #########################################################################


# # Given parameters
# A = -12.59
# B = 5.58
# C = -0.17
# D = -1.21

# def compute_mass(v, tau):
#     """
#     Compute the mass of the meteoroid given velocity and luminous efficiency.
#     """
#     # Calculate S
#     S = np.log(tau) - A - B * np.log(v) - C * (np.log(v))**3
#     # Compute tanh argument
#     tanh_arg = S / D
#     # Ensure tanh_arg is within the valid range (-1 + ε, 1 - ε)
#     epsilon = 1e-10
#     tanh_arg = np.clip(tanh_arg, -1 + epsilon, 1 - epsilon)
#     # Compute ln(m × 10^6)
#     ln_m_times_1e6 = np.arctanh(tanh_arg) / 0.2
#     # Calculate the mass m
#     m = np.exp(ln_m_times_1e6) / 1e6
#     return m

# def compute_tau(v, m):
#     """
#     Compute the luminous efficiency given velocity and mass.
#     """
#     ln_tau = A + B * np.log(v) + C * (np.log(v))**3 + D * np.tanh(0.2 * np.log(m * 1e6))
#     tau = np.exp(ln_tau)
#     return tau

# def assess_mass(v, tau_old, tol=1e-8, max_iter=100):
#     """
#     Iteratively assess the meteoroid mass until the luminous efficiency converges.
#     """
#     for iteration in range(max_iter):
#         m = compute_mass(v, tau_old)
#         tau_new = compute_tau(v, m)
#         print(f"Iteration {iteration}: τ = {tau_new:.6f}, m = {m:.6e} kg")
#         if abs(tau_new - tau_old) < tol:
#             return m
#         tau_old = tau_new
#     print("Maximum iterations reached without convergence.")
#     return None



# def Find_init_mass_and_vel(ang_init, h_obs, v_obs, m_guess):
#     '''
#         path_and_file = must be a json file generated file from the generate_simulationsm function or from Metsim file
#     '''
    
#     const_nominal = Constants()

#     # Minimum height [m]
#     const_nominal.h_kill = h_obs
#     const_nominal.zenith_angle = ang_init
#     const_nominal.erosion_on = False

#     p0 = [v_obs, m_init]
#     opt_res = opt.minimize(fit_residual, p0, args=(np.array(time_data), np.array(lag_data)), method='Nelder-Mead')
#     a_t0, b_t0, c_t0, t0 = opt_res.x

#     try:
#         # Run the simulation
#         frag_main, results_list, wake_results = runSimulation(const_nominal, compute_wake=False)
#         simulation_MetSim_object = SimulationResults(const_nominal, frag_main, results_list, wake_results)
#     except ZeroDivisionError as e:
#         print(f"Error during simulation: {e}")
#         const_nominal = Constants()
#         # Run the simulation
#         frag_main, results_list, wake_results = runSimulation(const_nominal, compute_wake=False)
#         simulation_MetSim_object = SimulationResults(const_nominal, frag_main, results_list, wake_results)

#     # # print the column of simulation_MetSim_object to see what is inside
#     # print(simulation_MetSim_object.__dict__.keys())
#     # print(simulation_MetSim_object.const.__dict__.keys())
  
#     # ax[0].plot(sr_nominal_1D_KDE.abs_magnitude, sr_nominal_1D_KDE.leading_frag_height_arr/1000, label="Mode", color='r')
#     # ax[2].plot(sr_nominal.leading_frag_vel_arr/1000, sr_nominal.leading_frag_height_arr/1000, color='k', abel="Simulated")

#     gensim_data_metsim = read_RunSim_output(simulation_MetSim_object, real_event, path_and_file_MetSim)

#     pd_Metsim  = array_to_pd_dataframe_PCA(gensim_data_metsim, fit_funct)

#     return simulation_MetSim_object, gensim_data_metsim, pd_Metsim


#### Generate Simulations #########################################################################

class ErosionSimParametersEMCCD_Comet(object):
    def __init__(self):
        """ Range of physical parameters for the erosion model, EMCCD system for Perseids. """

        self.dt = 0.005

        # Define the reference time for the atmosphere density model as J2000
        self.jdt_ref = date2JD(2020, 8, 10, 10, 0, 0)


        ## Atmosphere density ##
        #   Use the atmosphere density for the time at J2000 and coordinates of Elginfield
        self.dens_co = fitAtmPoly(np.radians(43.19301), np.radians(-81.315555), 60000, 180000, self.jdt_ref)

        ##


        # List of simulation parameters
        self.param_list = []



        ## System parameters ##

        # System limiting magnitude (given as a range)
        self.lim_mag_faintest = +5.49    # change the startng height
        self.lim_mag_brightest = +5.48   # change the startng height
        self.lim_mag_len_end_faintest = +5.61
        self.lim_mag_len_end_brightest = +5.60

        # Power of a zero-magnitude meteor (Watts)
        self.P_0m = 935

        # System fps
        self.fps = 32

        # Time lag of length measurements (range in seconds) - accomodate CAMO tracking delay of 8 frames
        #   This should be 0 for all other systems except for the CAMO mirror tracking system
        self.len_delay_min = 0
        self.len_delay_max = 0

        # Simulation height range [m] that will be used to map the output to a grid
        self.sim_height = MetParam(70000, 130000)


        ##


        ## Physical parameters

        # Mass range (kg)
        self.m_init = MetParam(1e-6, 2e-6)
        self.param_list.append("m_init") # change

        # Initial velocity range (m/s)
        self.v_init = MetParam(60000, 60200) # 60091.41691
        self.param_list.append("v_init") # change

        # Zenith angle range
        self.zenith_angle = MetParam(np.radians(43.35), np.radians(43.55)) # 43.466538
        self.param_list.append("zenith_angle") # change

        # Density range (kg/m^3)
        self.rho = MetParam(100, 1000)
        self.param_list.append("rho")

        # Intrinsic ablation coeff range (s^2/m^2)
        self.sigma = MetParam(0.008/1e6, 0.03/1e6) 
        self.param_list.append("sigma")
        # self.sigma = MetParam(0.005/1e6, 0.5/1e6)


        ##


        ## Erosion parameters ##
        ## Assumes no change in erosion once it starts!

        # Erosion height range
        self.erosion_height_start = MetParam(115000, 119000)
        self.param_list.append("erosion_height_start")

        # Erosion coefficient (s^2/m^2)
        self.erosion_coeff = MetParam(0.0, 1/1e6)
        self.param_list.append("erosion_coeff")

        # Mass index
        self.erosion_mass_index = MetParam(1.5, 2.5)
        self.param_list.append("erosion_mass_index")

        # Minimum mass for erosion
        self.erosion_mass_min = MetParam(5e-12, 1e-10)
        self.param_list.append("erosion_mass_min")

        # Maximum mass for erosion
        self.erosion_mass_max = MetParam(1e-10, 5e-8)
        self.param_list.append("erosion_mass_max")

        ## 


        ### Simulation quality checks ###

        # Minimum time above the limiting magnitude (10 frames)
        #   This is a minimum for both magnitude and length! 
        # self.visibility_time_min = 4/self.fps # DUMMY VARIABLE # THIS IS A USELLES VARIABLE

        ### ###


        ### Added noise ###

        # Standard deviation of the magnitude Gaussian noise
        self.mag_noise = 0.1

        # SD of noise in length [m]
        self.len_noise = 20.0

        ### ###


        ### Fit parameters ###

        # Length of input data arrays that will be given to the neural network
        self.data_length = DATA_LENGTH

        ### ###


        ### Output normalization range ###

        # Height range [m]
        self.ht_min = 70000
        self.ht_max = 130000

        # Magnitude range
        self.mag_faintest = +9
        self.mag_brightest = -2


        # Compute length range
        self.len_min = 0
        self.len_max = self.v_init.max*self.data_length/self.fps


        ### ###

# List of classed that can be used for data generation and postprocessing
SIM_CLASSES = [ErosionSimParametersEMCCD_Comet]
SIM_CLASSES_NAMES = [c.__name__ for c in SIM_CLASSES]

def run_simulation(path_and_file_MetSim, real_event, fit_funct):
    '''
        path_and_file = must be a json file generated file from the generate_simulationsm function or from Metsim file
    '''

    # Load the nominal simulation parameters
    const_nominal, _ = loadConstants(path_and_file_MetSim)
    const_nominal.dens_co = np.array(const_nominal.dens_co)

    dens_co=np.array(const_nominal.dens_co)

    ### Calculate atmosphere density coeffs (down to the bottom observed height, limit to 15 km) ###

    # Assign the density coefficients
    const_nominal.dens_co = dens_co

    # Turn on plotting of LCs of individual fragments 
    const_nominal.fragmentation_show_individual_lcs = True

    # # Minimum height [m]
    # const_nominal.h_kill = 60000

    # # Initial meteoroid height [m]
    # const_nominal.h_init = 180000

    try:
        # Run the simulation
        frag_main, results_list, wake_results = runSimulation(const_nominal, compute_wake=False)
        simulation_MetSim_object = SimulationResults(const_nominal, frag_main, results_list, wake_results)
    except ZeroDivisionError as e:
        print(f"Error during simulation: {e}")
        const_nominal = Constants()
        # Run the simulation
        frag_main, results_list, wake_results = runSimulation(const_nominal, compute_wake=False)
        simulation_MetSim_object = SimulationResults(const_nominal, frag_main, results_list, wake_results)

    # # print the column of simulation_MetSim_object to see what is inside
    # print(simulation_MetSim_object.__dict__.keys())
    # print(simulation_MetSim_object.const.__dict__.keys())
  
    # ax[0].plot(sr_nominal_1D_KDE.abs_magnitude, sr_nominal_1D_KDE.leading_frag_height_arr/1000, label="Mode", color='r')
    # ax[2].plot(sr_nominal.leading_frag_vel_arr/1000, sr_nominal.leading_frag_height_arr/1000, color='k', abel="Simulated")

    gensim_data_metsim = read_RunSim_output(simulation_MetSim_object, real_event, path_and_file_MetSim)

    pd_Metsim  = array_to_pd_dataframe_PCA(gensim_data_metsim, real_event)

    return simulation_MetSim_object, gensim_data_metsim, pd_Metsim

def safe_generate_erosion_sim(params):
    try:
        return generateErosionSim(*params)
    except Exception as e:
        print(f"Error in generateErosionSim: {e}")
        return None

def generate_simulations(real_data,simulation_MetSim_object,gensim_data,numb_sim,output_folder,file_name, fps, plot_case=False, flag_manual_metsim=True, CI_physical_param=''):
    '''
        Generate simulations for the given real data
    '''

    # Init simulation parameters with the given class name
    erosion_sim_params = SIM_CLASSES[SIM_CLASSES_NAMES.index('ErosionSimParametersEMCCD_Comet')]()

    erosion_sim_params.fps = fps

    # get from real_data the beg_abs_mag value of the first row and set it as the lim_mag_faintest value
    erosion_sim_params.lim_mag_faintest = real_data['beg_abs_mag'].iloc[0]+0.01
    erosion_sim_params.lim_mag_brightest = real_data['beg_abs_mag'].iloc[0]-0.01
    erosion_sim_params.lim_mag_len_end_faintest = real_data['end_abs_mag'].iloc[0]+0.01
    erosion_sim_params.lim_mag_len_end_brightest = real_data['end_abs_mag'].iloc[0]-0.01
    print('lim_mag_faintest',erosion_sim_params.lim_mag_faintest,'lim_mag_brightest',erosion_sim_params.lim_mag_brightest)
    print('lim_mag_len_end_faintest',erosion_sim_params.lim_mag_len_end_faintest,'lim_mag_len_end_brightest',erosion_sim_params.lim_mag_len_end_brightest)
    
    if flag_manual_metsim:
        mass_sim = simulation_MetSim_object.const.m_init
        # find the at what is the order of magnitude of the real_data['mass'][0]
        order = int(np.floor(np.log10(mass_sim)))
        # create a MetParam object with the mass range that is above and below the real_data['mass'][0] by 2 orders of magnitude
        erosion_sim_params.m_init = MetParam(mass_sim-(10**order)/2, mass_sim+(10**order)/2)
        # erosion_sim_params.m_init = MetParam(mass_sim/2, mass_sim*2)
        v_init_180km = simulation_MetSim_object.const.v_init # in m/s
        # Initial velocity range (m/s) 
        erosion_sim_params.v_init = MetParam(v_init_180km-real_data['rmsd_len'].iloc[0]*np.sqrt(2)/(1/fps)*1000, v_init_180km+real_data['rmsd_len'].iloc[0]*np.sqrt(2)/(1/fps)*1000) # 60091.41691
        # erosim_sim_params.erosion_height_start
        erosion_sim_params.erosion_height_start = MetParam(simulation_MetSim_object.const.erosion_height_start-2000, simulation_MetSim_object.const.erosion_height_start+2000)
            
    else:
        v_init_180km = real_data['vel_init_norot']
        # Initial velocity range (m/s) 
        erosion_sim_params.v_init = MetParam(real_data['vel_init_norot']-real_data['rmsd_len'].iloc[0]*np.sqrt(2)/(1/fps)*2, real_data['vel_init_norot']+real_data['rmsd_len'].iloc[0]*np.sqrt(2)/(1/fps)*2)
        # Mass range (kg)
        erosion_sim_params.m_init = MetParam(10**(-7), 10**(-4))
        # erosim_sim_params.erosion_height_start
        erosion_sim_params.erosion_height_start = MetParam(real_data['begin_height'].iloc[0]*1000-1000, real_data['begin_height'].iloc[0]*1000+9000)
        
    erosion_sim_params.dt = 0.005
    # if v_init_180km>60000:
    #     erosion_sim_params.dt = 0.005
    # elif v_init_180km<20000:
    #     erosion_sim_params.dt = 0.01
    # else:
    #     erosion_sim_params.dt = (-1)*0.000000125*v_init_180km+0.0125


    # Zenith angle range
    erosion_sim_params.zenith_angle = MetParam(np.radians(real_data['zenith_angle'].iloc[0]-0.01), np.radians(real_data['zenith_angle'].iloc[0]+0.01)) # 43.466538
    
    # # erosion_sim_params.erosion_height_start = MetParam(real_data['peak_mag_height'].iloc[0]*1000+(real_data['begin_height'].iloc[0]-real_data['peak_mag_height'].iloc[0])*1000/2, real_data['begin_height'].iloc[0]*1000+(real_data['begin_height'].iloc[0]-real_data['peak_mag_height'].iloc[0])*1000/2) # 43.466538
    # erosion_sim_params.erosion_height_start = MetParam(real_data['begin_height'].iloc[0]*1000-1000, real_data['begin_height'].iloc[0]*1000+4000) # 43.466538


    if CI_physical_param!='':
        erosion_sim_params.v_init = MetParam(CI_physical_param['v_init_180km'][0], CI_physical_param['v_init_180km'][1]) # 60091.41691
        erosion_sim_params.zenith_angle = MetParam(np.radians(CI_physical_param['zenith_angle'][0]), np.radians(CI_physical_param['zenith_angle'][1])) # 43.466538
        erosion_sim_params.m_init = MetParam(CI_physical_param['mass'][0], CI_physical_param['mass'][1])
        erosion_sim_params.rho = MetParam(CI_physical_param['rho'][0], CI_physical_param['rho'][1])
        erosion_sim_params.sigma = MetParam(CI_physical_param['sigma'][0], CI_physical_param['sigma'][1])
        erosion_sim_params.erosion_height_start = MetParam(CI_physical_param['erosion_height_start'][0], CI_physical_param['erosion_height_start'][1])
        erosion_sim_params.erosion_coeff = MetParam(CI_physical_param['erosion_coeff'][0], CI_physical_param['erosion_coeff'][1])
        erosion_sim_params.erosion_mass_index = MetParam(CI_physical_param['erosion_mass_index'][0], CI_physical_param['erosion_mass_index'][1])
        erosion_sim_params.erosion_mass_min = MetParam(CI_physical_param['erosion_mass_min'][0], CI_physical_param['erosion_mass_min'][1])
        erosion_sim_params.erosion_mass_max = MetParam(CI_physical_param['erosion_mass_max'][0], CI_physical_param['erosion_mass_max'][1])

        # check if a file with the name "log"+n_PC_in_PCA+"_"+str(len(df_sel))+"ev.txt" already exist
        if os.path.exists(output_folder+os.sep+"log_"+file_name[:15]+"_GenereateSimulations_range_NEW.txt"):
            # remove the file
            os.remove(output_folder+os.sep+"log_"+file_name[:15]+"_GenereateSimulations_range_NEW.txt")
        sys.stdout = Logger(output_folder,"log_"+file_name[:15]+"_GenereateSimulations_range_NEW.txt") # _30var_99%_13PC
    else:
        # check if a file with the name "log"+n_PC_in_PCA+"_"+str(len(df_sel))+"ev.txt" already exist
        if os.path.exists(output_folder+os.sep+"log_"+file_name[:15]+"_GenereateSimulations_range.txt"):
            # remove the file
            os.remove(output_folder+os.sep+"log_"+file_name[:15]+"GenereateSimulations_range.txt")
        sys.stdout = Logger(output_folder,"log_"+file_name[:15]+"GenereateSimulations_range.txt") # _30var_99%_13PC


    print('Run',numb_sim,'simulations with :')
    # to_plot_unit=['mass [kg]','rho [kg/m^3]','sigma [s$^2$/km$^2$]','erosion height start [km]','erosion coeff [s$^2$/km$^2$]','erosion mass index [-]','eros. mass min [kg]','eros. mass max [kg]']
    print('\\hline') #df_sel_save[df_sel_save['solution_id']==only_select_meteors_from][plotvar].values[0]
    print('Variables & min.val. & MAX.val. \\\\')

    print('\\hline')
    # - velocity: min 58992.19459103218 - MAX 60992.19459103218
    # print('- velocity: min',erosion_sim_params.v_init.min,'- MAX',erosion_sim_params.v_init.max)
    print(f"Velocity [km/s] & {'{:.4g}'.format(erosion_sim_params.v_init.min/1000)} & {'{:.4g}'.format(erosion_sim_params.v_init.max/1000)} \\\\")
    
    print('\\hline')
    # - zenith angle: min 28.736969960110045 - MAX 28.75696996011005
    # print('- zenith angle: min',np.degrees(erosion_sim_params.zenith_angle.min),'- MAX',np.degrees(erosion_sim_params.zenith_angle.max))
    print(f"Zenith ang. [deg] & {'{:.4g}'.format(np.degrees(erosion_sim_params.zenith_angle.min))} & {'{:.4g}'.format(np.degrees(erosion_sim_params.zenith_angle.max))} \\\\")

    print('\\hline') 
    # - Initial mag: min 5.45949291900601 - MAX 5.43949291900601
    # print('- Initial mag: min',erosion_sim_params.lim_mag_faintest,'- MAX',erosion_sim_params.lim_mag_brightest)
    print(f"Init. mag [-] & {'{:.4g}'.format(erosion_sim_params.lim_mag_faintest)} & {'{:.4g}'.format(erosion_sim_params.lim_mag_brightest)} \\\\")

    print('\\hline')
    # - Final mag: min 6.0268141526507435 - MAX 6.006814152650744
    # print('- Final mag: min',erosion_sim_params.lim_mag_len_end_faintest,'- MAX',erosion_sim_params.lim_mag_len_end_brightest)
    print(f"Fin. mag [-] & {'{:.4g}'.format(erosion_sim_params.lim_mag_len_end_faintest)} & {'{:.4g}'.format(erosion_sim_params.lim_mag_len_end_brightest)} \\\\")

    print('\\hline')
    # - Mass: min 5.509633400654068e-07 - MAX 1.5509633400654067e-06
    # print('- Mass: min',erosion_sim_params.m_init.min,'- MAX',erosion_sim_params.m_init.max)
    print(f"Mass [kg] & {'{:.4g}'.format(erosion_sim_params.m_init.min)} & {'{:.4g}'.format(erosion_sim_params.m_init.max)} \\\\")

    print('\\hline')
    # - rho : min 100 - MAX 1000
    # print('- rho : min',erosion_sim_params.rho.min,'- MAX',erosion_sim_params.rho.max)
    print(f"Rho [kg/m^3] & {'{:.4g}'.format(erosion_sim_params.rho.min)} & {'{:.4g}'.format(erosion_sim_params.rho.max)} \\\\")

    print('\\hline')
    # - sigma : min 8e-09 - MAX 3e-08
    # print('- sigma : min',erosion_sim_params.sigma.min,'- MAX',erosion_sim_params.sigma.max)
    print(f"sigma [s^2/km^2] & {'{:.4g}'.format(erosion_sim_params.sigma.min*1000000)} & {'{:.4g}'.format(erosion_sim_params.sigma.max*1000000)} \\\\")

    print('\\hline')
    # - erosion_height_start : min 107622.04437691614 - MAX 117622.04437691614
    # print('- erosion_height_start : min',erosion_sim_params.erosion_height_start.min,'- MAX',erosion_sim_params.erosion_height_start.max)
    print(f"Eros.height [km] & {'{:.4g}'.format(erosion_sim_params.erosion_height_start.min/1000)} & {'{:.4g}'.format(erosion_sim_params.erosion_height_start.max/1000)} \\\\")

    print('\\hline')
    # - erosion_coeff : min 0.0 - MAX 1e-06
    # print('- erosion_coeff : min',erosion_sim_params.erosion_coeff.min,'- MAX',erosion_sim_params.erosion_coeff.max)
    print(f"Eros.coeff. [s^2/km^2] & {'{:.4g}'.format(erosion_sim_params.erosion_coeff.min*1000000)} & {'{:.4g}'.format(erosion_sim_params.erosion_coeff.max*1000000)} \\\\")

    print('\\hline')
    # - erosion_mass_index : min 1.5 - MAX 2.5
    # print('- erosion_mass_index : min',erosion_sim_params.erosion_mass_index.min,'- MAX',erosion_sim_params.erosion_mass_index.max)
    print(f"Eros.mass index [-] & {'{:.4g}'.format(erosion_sim_params.erosion_mass_index.min)} & {'{:.4g}'.format(erosion_sim_params.erosion_mass_index.max)} \\\\")

    print('\\hline')
    # - erosion_mass_min : min 5e-12 - MAX 1e-10
    # print('- erosion_mass_min : min',erosion_sim_params.erosion_mass_min.min,'- MAX',erosion_sim_params.erosion_mass_min.max)
    print(f"Eros.mass min [kg] & {'{:.4g}'.format(erosion_sim_params.erosion_mass_min.min)} & {'{:.4g}'.format(erosion_sim_params.erosion_mass_min.max)} \\\\")

    print('\\hline')
    # - erosion_mass_max : min 1e-10 - MAX 5e-08
    # print('- erosion_mass_max : min',erosion_sim_params.erosion_mass_max.min,'- MAX',erosion_sim_params.erosion_mass_max.max)
    print(f"Eros.mass max [kg] & {'{:.4g}'.format(erosion_sim_params.erosion_mass_max.min)} & {'{:.4g}'.format(erosion_sim_params.erosion_mass_max.max)} \\\\")

    print('\\hline')


    # Close the Logger to ensure everything is written to the file STOP COPY in TXT file
    sys.stdout.close()

    # Reset sys.stdout to its original value if needed
    sys.stdout = sys.__stdout__

    input_list = [(output_folder, copy.deepcopy(erosion_sim_params), np.random.randint(0, 2**31 - 1), MIN_FRAMES_VISIBLE) for _ in range(numb_sim)]
    with Pool(cml_args.cores) as pool:
        results_list = pool.map(safe_generate_erosion_sim, input_list)

    count_none = sum(res is None for res in results_list)
    saveProcessedList(output_folder, results_list, erosion_sim_params.__class__.__name__, MIN_FRAMES_VISIBLE)

    print('Resulted simulations:', numb_sim - count_none)
    print('Failed simulations:', count_none)
    print('Saved', numb_sim - count_none, 'simulations in', output_folder)

    #########################
    # # Generate simulations using multiprocessing
    # input_list = [[output_folder, copy.deepcopy(erosion_sim_params), \
    #     np.random.randint(0, 2**31 - 1),MIN_FRAMES_VISIBLE] for _ in range(numb_sim)]
    # results_list = domainParallelizer(input_list, generateErosionSim, cores=cml_args.cores)

    # # print(results_list)

    # # count how many None are in the results_list
    # count_none=0
    # for res in results_list:
    #     if res is None:
    #         count_none+=1
    #         continue
        
    # saveProcessedList(output_folder, results_list, erosion_sim_params.__class__.__name__, \
    # MIN_FRAMES_VISIBLE)
    
    # print('Resulted simulations:', numb_sim-count_none)
    # print('Failed siulations', len(results_list)/100*count_none,'%')
    # print('Saved',numb_sim-count_none,'simulations in',output_folder)
    #########################

    # plot the pickle files data that are not none in the results_list
    # do not plot more than 10 curves
    if plot_case:

        fig, ax = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
        # flat the ax
        ax = ax.flatten()

        jj_plots_curve=0
        for res in results_list:
            if res is not None:
                if jj_plots_curve>100:
                    # stop if too many curves are plotted
                    break
                
                if res[0] is not None:
                    # change res[0] extension to .json
                    res[0] = res[0].replace('.pickle', '.json')
                    print(res[0]) 
                    # get the first value of res
                    gensim_data_sim = read_GenerateSimulations_output(res[0])

                    plot_side_by_side(gensim_data_sim, fig, ax, 'b-')
                    jj_plots_curve += 1
                
        plot_side_by_side(gensim_data,fig, ax,'go','Obsevation')

        return fig, ax

    

#### Plot #############################################################################


def check_axis_inversion(ax):
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    is_x_inverted = x_max < x_min
    is_y_inverted = y_max < y_min
    return is_x_inverted, is_y_inverted

def plot_data_with_residuals_and_real(rmsd_mag, rmsd_vel, rmsd_len, fit_funct_original, real_original, label_real='', file_name='', output_dir = '', data_original='', label_data='', data_opt_or_desns_original='', label_opt_or_desns=''):

    # copy the data
    fit_funct = copy.deepcopy(fit_funct_original)
    real = copy.deepcopy(real_original)
    data = copy.deepcopy(data_original)
    data_opt_or_desns = copy.deepcopy(data_opt_or_desns_original)

    if fit_funct['height'][1] > 1000:
        fit_funct['velocities'] = fit_funct['velocities']/1000
        fit_funct['height'] = fit_funct['height']/1000

    if real['height'][1] > 1000:
        real['velocities'] = real['velocities']/1000
        real['height'] = real['height']/1000

    if data != '':
        if data['height'][1] > 1000:
            data['velocities'] = data['velocities']/1000
            data['height'] = data['height']/1000
    if data_opt_or_desns != '':
        if data_opt_or_desns['height'][1] > 1000:
            data_opt_or_desns['velocities'] = data_opt_or_desns['velocities']/1000
            data_opt_or_desns['height'] = data_opt_or_desns['height']/1000

    def line_and_color_plot(label,color_line1=None):
        if label=='Mode':
            return '','-','r'
        elif label=='Metsim':
            return '','-','k'
        elif label=='Dens.point':
            return '','-','b'
        elif label=='Optimized':
            return 'x',':', color_line1
        else:
            return '','-',None

    # Create the figure and main GridSpec with specified height ratios
    fig = plt.figure(figsize=(14, 6))
    gs_main = gridspec.GridSpec(2, 4, figure=fig, height_ratios=[3, 0.5], width_ratios=[1, 1, 1, 1])

    # Create a sub GridSpec for Plot 0 and Plot 1 with width ratios
    gs01 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_main[0, 0:2], wspace=0, width_ratios=[3, 1])

    # Plot 0 and 1: Side by side, sharing the y-axis
    ax0 = fig.add_subplot(gs01[0])
    ax1 = fig.add_subplot(gs01[1], sharey=ax0)

    # Insert fill_between for magnitude
    height_km_err = real['height']
    abs_mag_sim_err = fit_funct['absolute_magnitudes']
    mag_noise = rmsd_mag
    ax0.fill_betweenx(height_km_err, abs_mag_sim_err - mag_noise, abs_mag_sim_err + mag_noise, color='darkgray', alpha=0.2)
    ax0.fill_betweenx(height_km_err, abs_mag_sim_err - mag_noise * 1.96, abs_mag_sim_err + mag_noise * 1.96, color='lightgray', alpha=0.2)
    ax0.plot(real['absolute_magnitudes'], real['height'], 'go')
    if data != '':
        line1, = ax0.plot(data['absolute_magnitudes'], data['height'])
        _, _, color_line1= line_and_color_plot(label_data)
        if color_line1!=None:
            # set the color of line1 to color_line1
            line1.set_color(color_line1)
        # get line1 color
        color_line1 = line1.get_color()
        if data_opt_or_desns!='':
            line2, = ax0.plot(data_opt_or_desns['absolute_magnitudes'], data_opt_or_desns['height'])
            line_marker2, line_sty2, color_line2 = line_and_color_plot(label_opt_or_desns,color_line1)
            if color_line2!=None:
                # set the color of line2 to color_line2
                line2.set_color(color_line2)
            # set the linestyle of line2 to line_sty2
            line2.set_linestyle(line_sty2)
            # set the marker of line2 to line_marker2
            line2.set_marker(line_marker2)
    else:
        ax0.plot(fit_funct['absolute_magnitudes'], fit_funct['height'], 'k--')
    ax0.set_xlabel('Absolute Magnitudes [-]')
    # flip the x-axis
    ax0.invert_xaxis()
    # ax0.tick_params(axis='x', rotation=45)
    ax0.set_ylabel('Height (km)')
    ax0.grid(True, linestyle='--', color='lightgray')

    ax1.fill_betweenx(height_km_err, -mag_noise, mag_noise, color='darkgray', alpha=0.2)
    ax1.fill_betweenx(height_km_err, -mag_noise * 1.96, mag_noise * 1.96, color='lightgray', alpha=0.2)
    ax1.plot([0, 0], [fit_funct['height'][0], fit_funct['height'][-1]],color='lightgray')
    # Plot 1: Height vs. Res.Mag, without y-axis tick labels    
    if data != '':
        # Plot 0: Height vs. Absolute Magnitudes with two lines
        ax1.plot(data['res_absolute_magnitudes'], real['height'],'.',color=color_line1)
        if data_opt_or_desns!='':
            if line_marker2!='':
                ax1.plot(data_opt_or_desns['res_absolute_magnitudes'], real['height'],line_marker2,color=color_line2)
            else:
                ax1.plot(data_opt_or_desns['res_absolute_magnitudes'], real['height'],'.',color=color_line2)
    else:
        ax1.plot(real['res_absolute_magnitudes'], real['height'], 'g.')
    ax1.set_xlabel('Res.Mag [-]')
    # flip the x-axis
    ax1.invert_xaxis()
    # ax1.tick_params(axis='x', rotation=45)
    ax1.tick_params(labelleft=False)  # Hide y-axis tick labels
    ax1.grid(True, linestyle='--', color='lightgray')


    # Plot 4: Custom legend for Plot 0 with two columns
    ax4 = fig.add_subplot(gs_main[1, 0])
    ax4.axis('off')
    if data_opt_or_desns!='':
        label_line1= label_data+' mag$_{RMSD}$ '+str(round(data['rmsd_mag'],3))+' len$_{RMSD}$ '+str(round(data['rmsd_len']*1000,1))+'m\n\
mag$\chi^2_{red}$'+str(round(data['chi2_red_mag'],2))+' len$\chi^2_{red}$'+str(round(data['chi2_red_lag'],2))+'\n\
$m_0$:'+str('{:.2e}'.format(data['mass'],1))+'kg $\\rho$:'+str(round(data['rho']))+'kg/m$^3$\n\
$\sigma$:'+str(round(data['sigma']*1000000,4))+'s$^2$/km$^2$ $\eta$:'+str(round(data['erosion_coeff']*1000000,3))+'s$^2$/km$^2$\n\
$h_e$:'+str(round(data['erosion_height_start'],1))+'km $s$:'+str(round(data['erosion_mass_index'],2))+'\n\
$m_l$:'+str('{:.2e}'.format(data['erosion_mass_min'],1))+'kg $m_u$:'+str('{:.2e}'.format(data['erosion_mass_max'],1))+'kg'
        label_line2 = label_opt_or_desns+' mag$_{RMSD}$ '+str(round(data_opt_or_desns['rmsd_mag'],3))+' len$_{RMSD}$ '+str(round(data_opt_or_desns['rmsd_len']*1000,1))+'m\n\
mag$\chi^2_{red}$'+str(round(data_opt_or_desns['chi2_red_mag'],2))+' len$\chi^2_{red}$'+str(round(data_opt_or_desns['chi2_red_lag'],2))+'\n\
$m_0$:'+str('{:.2e}'.format(data_opt_or_desns['mass'],1))+'kg $\\rho$:'+str(round(data_opt_or_desns['rho']))+'kg/m$^3$\n\
$\sigma$:'+str(round(data_opt_or_desns['sigma']*1000000,1))+'s$^2$/km$^2$ $\eta$:'+str(round(data_opt_or_desns['erosion_coeff']*1000000,3))+'s$^2$/km$^2$\n\
$h_e$:'+str(round(data_opt_or_desns['erosion_height_start'],1))+'km $s$:'+str(round(data_opt_or_desns['erosion_mass_index'],2))+'\n\
$m_l$:'+str('{:.2e}'.format(data_opt_or_desns['erosion_mass_min'],1))+'kg $m_u$:'+str('{:.2e}'.format(data_opt_or_desns['erosion_mass_max'],1))+'kg'
        ax4.legend([line1, line2], [label_line1, label_line2], loc='center', ncol=2, fontsize=7)
    elif data!='':
        label_line1=label_data+' mag$_{RMSD}$ '+str(round(data['rmsd_mag'],3))+' len$_{RMSD}$ '+str(round(data['rmsd_len']*1000,1))+'m\n\
mag$\chi^2_{red}$'+str(round(data['chi2_red_mag'],2))+' len$\chi^2_{red}$'+str(round(data['chi2_red_lag'],2))+'\n\
$m_0$:'+str('{:.2e}'.format(data['mass'],1))+'kg $\\rho$:'+str(round(data['rho']))+'kg/m$^3$\n\
$\sigma$:'+str(round(data['sigma']*1000000,4))+'s$^2$/km$^2$ $\eta$:'+str(round(data['erosion_coeff']*1000000,3))+'s$^2$/km$^2$\n\
$h_e$:'+str(round(data['erosion_height_start'],1))+'km $s$:'+str(round(data['erosion_mass_index'],2))+'\n\
$m_l$:'+str('{:.2e}'.format(data['erosion_mass_min'],1))+'kg $m_u$:'+str('{:.2e}'.format(data['erosion_mass_max'],1))+'kg'
        ax4.legend([line1], [label_line1], loc='center left', ncol=1)


    # Plot 5: Custom legend with green dot, dashed line, and shaded areas
    ax5 = fig.add_subplot(gs_main[1, 1])
    ax5.axis('off')
    ax5.plot([], [], 'go', label=label_real[:15]+'\nmag$_{RMSD}$ '+str(round(rmsd_mag,3))+'\nvel$_{RMSD}$ '+str(round(rmsd_vel,3))+'km/s\nlen$_{RMSD}$ '+str(round(rmsd_len*1000,1))+'m')  # Green dot
    if data == '':
        ax5.plot([], [], 'k--', label='Fit')  # Black dashed line
    ax5.fill_between([], [], [], color='darkgray', alpha=0.2, label='1$\sigma$')
    ax5.fill_between([], [], [], color='lightgray', alpha=0.2, label='2$\sigma$')
    ax5.legend(loc='right', fontsize=8) # upper right


    # Plot 2 and 6: Vertically stacked, sharing the x-axis (Time) with height ratios
    gs_col2 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_main[:, 2], hspace=0, height_ratios=[3, 1])
    ax2 = fig.add_subplot(gs_col2[0, 0])
    ax6 = fig.add_subplot(gs_col2[1, 0], sharex=ax2)


    # Remaining subplots with fill_between
    residual_time_pos = real['time']
    vel_kms_err = fit_funct['velocities']
    vel_noise = rmsd_vel
    ax2.fill_between(residual_time_pos, vel_kms_err - vel_noise, vel_kms_err + vel_noise, color='darkgray', alpha=0.2)
    ax2.fill_between(residual_time_pos, vel_kms_err - vel_noise * 1.96, vel_kms_err + vel_noise * 1.96, color='lightgray', alpha=0.2)
    # Plot 2: Velocity vs. Time, without x-axis tick labels
    ax2.plot(real['time'], real['velocities'], 'go')
    if data != '':
        ax2.plot(data['time'], data['velocities'], color=color_line1)
        if data_opt_or_desns!='':
            ax2.plot(data_opt_or_desns['time'], data_opt_or_desns['velocities'], line_marker2+line_sty2, color=color_line2)
    else:
        ax2.plot(fit_funct['time'], fit_funct['velocities'], 'k--')
    ax2.set_ylabel('Velocity [km/s]')
    ax2.tick_params(labelbottom=False)  # Hide x-axis tick labels
    ax2.grid(True, linestyle='--', color='lightgray')

    # Plot 6: Res.Vel vs. Time
    ax6.fill_between(residual_time_pos, -vel_noise, vel_noise, color='darkgray', alpha=0.2)
    ax6.fill_between(residual_time_pos, -vel_noise * 1.96, vel_noise * 1.96, color='lightgray', alpha=0.2)
    ax6.plot([fit_funct['time'][0], fit_funct['time'][-1]], [0, 0], color='lightgray')
    if data != '':
        ax6.plot(real['time'], data['res_velocities'], '.', color=color_line1)
        if data_opt_or_desns!='':
            if line_marker2!='':
                ax6.plot(real['time'], data_opt_or_desns['res_velocities'], line_marker2, color=color_line2)
            else:
                ax6.plot(real['time'], data_opt_or_desns['res_velocities'], '.', color=color_line2)
    else:
        ax6.plot(real['time'], real['res_velocities'], 'g.')
    ax6.set_xlabel('Time [s]')
    ax6.set_ylabel('Res.Vel [km/s]')
    ax6.grid(True, linestyle='--', color='lightgray')

    # Plot 3 and 7: Vertically stacked, sharing the x-axis (Time) with height ratios
    gs_col3 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_main[:, 3], hspace=0, height_ratios=[3, 1])
    ax3 = fig.add_subplot(gs_col3[0, 0])
    ax7 = fig.add_subplot(gs_col3[1, 0], sharex=ax3)

    lag_km_err = fit_funct['lag']
    lag_noise = rmsd_len * 1000
    ax3.fill_between(residual_time_pos, lag_km_err - lag_noise, lag_km_err + lag_noise, color='darkgray', alpha=0.2)
    ax3.fill_between(residual_time_pos, lag_km_err - lag_noise * 1.96, lag_km_err + lag_noise * 1.96, color='lightgray', alpha=0.2)
    # Plot 2: Velocity vs. Time, without x-axis tick labels
    ax3.plot(real['time'], real['lag'], 'go')
    if data != '':
        ax3.plot(data['time'], data['lag'], color=color_line1)
        if data_opt_or_desns!='':
            ax3.plot(data_opt_or_desns['time'], data_opt_or_desns['lag'], line_marker2+line_sty2, color=color_line2)
    else:
        ax3.plot(fit_funct['time'], fit_funct['lag'], 'k--')
    ax3.set_ylabel('Lag [m]')
    ax3.tick_params(labelbottom=False)  # Hide x-axis tick labels
    ax3.grid(True, linestyle='--', color='lightgray')

    # Plot 7: Res.Vel vs. Time
    ax7.fill_between(residual_time_pos, -lag_noise, lag_noise, color='darkgray', alpha=0.2)
    ax7.fill_between(residual_time_pos, -lag_noise * 1.96, lag_noise * 1.96, color='lightgray', alpha=0.2)
    ax7.plot([fit_funct['time'][0], fit_funct['time'][-1]], [0, 0], color='lightgray')
    if data != '':
        ax7.plot(real['time'], data['res_lag'], '.', color=color_line1)
        if data_opt_or_desns!='':
            if line_marker2!='':
                ax7.plot(real['time'], data_opt_or_desns['res_lag'], line_marker2, color=color_line2)
            else:
                ax7.plot(real['time'], data_opt_or_desns['res_lag'], '.', color=color_line2)
    else:
        ax7.plot(real['time'], real['res_lag'], 'g.')
    ax7.set_xlabel('Time [s]')
    ax7.set_ylabel('Res.Lag [m]')
    ax7.grid(True, linestyle='--', color='lightgray')

    # Adjust the overall layout to prevent overlap
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    select_data=''
    if data!='':
        if data['rmsd_mag']<rmsd_mag and data['rmsd_len']<rmsd_len:
            select_data=label_data+' SELECTED'
        else:
            select_data=label_data+' NOT SELECTED'
    if data_opt_or_desns !='':
        if data_opt_or_desns['rmsd_mag']<rmsd_mag and data_opt_or_desns['rmsd_len']<rmsd_len:
            select_data=select_data+' '+label_opt_or_desns+' SELECTED'
        else:
            select_data=select_data+' '+label_opt_or_desns+' NOT SELECTED'

    file_name_title=file_name
    #check if the file_name has a '.' in it if so rake file_name[:-5]
    if '.pickle' in file_name:
        file_name_title=file_name[:15]
    elif '.json' in file_name:
        # find in which position is '.json'
        pos=file_name.find('.json')
        # delete the '.json' from the file_name and all the characters after it
        file_name_title=file_name[:pos]
    elif '.png' in file_name:
        file_name_title=file_name[:-4]
    fig.suptitle(file_name_title+' '+select_data)

    # Save the plot
    print('file saved: '+output_dir +os.sep+ file_name)
    fig.savefig(output_dir +os.sep+ file_name, dpi=300)

    # Display the plot
    plt.close(fig)


def plot_side_by_side(data1, fig='', ax='', colorline1='.', label1='', residuals_mag='', residuals_vel='', residual_time_pos='', residual_height_pos='', residuals_lag='', fit_funct='', mag_noise='', vel_noise='',lag_noise='', sim_lag='', sim_time=''):

    # check if data1 is None
    if data1 is None:
        print("Warning: data1 is None. Skipping plot.")
        return
    
    # check if it is in km/s or in m/s
    obs1 = copy.deepcopy(data1)
    if 'velocities' not in obs1 or 'height' not in obs1:
        print("Warning: Required keys missing in obs1. Skipping plot.")
        return

    # check if it is in km/s or in m/s
    obs1= copy.deepcopy(data1)
    if np.mean(obs1['velocities'])>1000:
        # convert to km/s
        obs1['velocities'] = np.array(obs1['velocities'])/1000
        obs1['height'] = np.array(obs1['height'])/1000


    # Plot the simulation results
    if residuals_mag != '' and residuals_vel != '' and residual_time_pos!='' and residual_height_pos!='':

        residual_time_pos_err=residual_time_pos
        if len(residual_time_pos) != len(obs1['velocities']):
            # interpolate from residual_time_pos[0] to residual_time_pos[-1] with len(obs1['velocities'])
            residual_time_pos = obs1['time'] # np.linspace(residual_time_pos[0], residual_time_pos[-1], len(obs1['velocities'])) 


        if fig=='' and ax=='':
            fig, ax = plt.subplots(2, 3, figsize=(14, 6),gridspec_kw={'height_ratios': [ 3, 1],'width_ratios': [ 3, 0.5, 3]}) #  figsize=(10, 5), dpi=300 0.5, 3, 3, 0.5
            # flat the ax
            ax = ax.flatten()
            return fig, ax
        
        if fit_funct!='' and mag_noise!='' and vel_noise!='':
            obs_time_err=np.array(fit_funct['time'])
            abs_mag_sim_err=np.array(fit_funct['absolute_magnitudes'])
            height_km_err=np.array(fit_funct['height'])
            vel_kms_err=np.array(fit_funct['velocities'])
            len_km_err=np.array(fit_funct['length'])
            lag_km_err=np.array(fit_funct['lag'])
            #lag_kms_err=len_km_err - (obs1['velocities'][0]/1000*obs_time_err)
            #_err=lag_kms_err - lag_kms_err[0]
            # from list to array
            if np.mean(fit_funct['height'])>1000:
                # convert to km/s
                height_km_err=np.array(fit_funct['height'])/1000
                vel_kms_err=np.array(fit_funct['velocities'])/1000

            # plot noisy area around vel_kms for vel_noise for the fix height_km
            ax[0].fill_betweenx(height_km_err, abs_mag_sim_err-mag_noise, abs_mag_sim_err+mag_noise, color='darkgray', alpha=0.2)
            ax[0].fill_betweenx(height_km_err, abs_mag_sim_err-mag_noise*1.96, abs_mag_sim_err+mag_noise*1.96, color='lightgray', alpha=0.2)
            ax[0].plot(abs_mag_sim_err,height_km_err, 'k--')

            # plot noisy area around vel_kms for vel_noise for the fix height_km
            ax[1].fill_betweenx(height_km_err, -mag_noise, mag_noise, color='darkgray', alpha=0.2)
            ax[1].fill_betweenx(height_km_err, -mag_noise*1.96, mag_noise*1.96, color='lightgray', alpha=0.2)

            if lag_noise != '':
                lag_noise = lag_noise * 1000

                # plot noisy area around vel_kms for vel_noise for the fix height_km
                ax[2].fill_between(residual_time_pos, vel_kms_err-vel_noise, vel_kms_err+vel_noise, color='darkgray', alpha=0.2)
                ax[2].fill_between(residual_time_pos, vel_kms_err-vel_noise*1.96, vel_kms_err+vel_noise*1.96, color='lightgray', alpha=0.2)
                ax[2].plot(residual_time_pos, vel_kms_err, 'k--')

                # plot noisy area around vel_kms for vel_noise for the fix height_km
                ax[3].fill_between(residual_time_pos, lag_km_err-lag_noise, lag_km_err+lag_noise, color='darkgray', alpha=0.2, label='1$\sigma$')
                ax[3].fill_between(residual_time_pos, lag_km_err-lag_noise*1.96, lag_km_err+lag_noise*1.96, color='lightgray', alpha=0.2, label='2$\sigma$')
                ax[3].plot(residual_time_pos, lag_km_err, 'k--', label='Fit')

                # plot noisy area around vel_kms for vel_noise for the fix height_km
                ax[6].fill_between(residual_time_pos, -vel_noise, vel_noise, color='darkgray', alpha=0.2)
                ax[6].fill_between(residual_time_pos, -vel_noise*1.96, vel_noise*1.96, color='lightgray', alpha=0.2)

                # plot noisy area around vel_kms for vel_noise for the fix height_km
                ax[7].fill_between(residual_time_pos, -lag_noise, lag_noise, color='darkgray', alpha=0.2)
                ax[7].fill_between(residual_time_pos, -lag_noise*1.96, lag_noise*1.96, color='lightgray', alpha=0.2)

            else:
                # plot noisy area around vel_kms for vel_noise for the fix height_km
                ax[2].fill_between(residual_time_pos, vel_kms_err-vel_noise, vel_kms_err+vel_noise, color='darkgray', alpha=0.2, label='1$\sigma$')
                ax[2].fill_between(residual_time_pos, vel_kms_err-vel_noise*1.96, vel_kms_err+vel_noise*1.96, color='lightgray', alpha=0.2, label='2$\sigma$')
                ax[2].plot(residual_time_pos, vel_kms_err, 'k--', label='Fit')

                # plot noisy area around vel_kms for vel_noise for the fix height_km
                ax[5].fill_between(residual_time_pos, -vel_noise, vel_noise, color='lightgray', alpha=0.5)

        ax[0].plot(obs1['absolute_magnitudes'],obs1['height'], colorline1)
        ax[0].set_xlabel('Absolute Magnitude [-]')
        ax[0].set_ylabel('Height [km]')
        # grid on on both subplot with -- as linestyle and light gray color
        ax[0].grid(True)
        ax[0].grid(linestyle='--',color='lightgray')

        # flip the y-axis
        is_x_inverted, _ =check_axis_inversion(ax[0])
        if is_x_inverted==False:
            ax[0].invert_xaxis()

        # Get the color of the last plotted line in graph 0
        line_color = ax[0].get_lines()[-1].get_color()

        # if line_color == '#2ca02c':
        #     line_color='m'
        #     ax[0].plot(obs1['absolute_magnitudes'],obs1['height'], colorline1, color='m')

        # plot the residuals against time
        if fit_funct=='' and mag_noise=='' and vel_noise=='':
            ax[1].plot(residuals_mag, residual_height_pos, '.', color=line_color)
        # ax[1].set_ylabel('Height [km]')
        ax[1].set_xlabel('Res.mag [-]')
        ax[1].tick_params(axis='x', rotation=45)

        # flip the y-axis
        is_x_inverted, _ =check_axis_inversion(ax[1])
        if is_x_inverted==False:
            ax[1].invert_xaxis()

        # ax[1].title(f'Lag Residuals')
        # ax[1].legend()
        is_x_inverted, _ =check_axis_inversion(ax[1])
        if is_x_inverted==False:
            ax[1].invert_xaxis()
        ax[1].grid(True)
        ax[1].grid(linestyle='--',color='lightgray')
        ax[1].set_ylim(ax[0].get_ylim())


        if residuals_lag!='':
            if sim_time!='':
                ax[2].plot(sim_time, obs1['velocities'], colorline1, color=line_color)
            else:
                ax[2].plot(residual_time_pos, obs1['velocities'], colorline1, color=line_color)

            ax[2].set_xlabel('Time [s]')
            ax[2].set_ylabel('Velocity [km/s]')
            ax[2].grid(True)
            ax[2].grid(linestyle='--',color='lightgray')

            if label1!='':
                if sim_lag!='':
                    if sim_time!='':
                        ax[3].plot(sim_time, sim_lag*1000, colorline1, color=line_color, label=label1)
                    else:
                        ax[3].plot(residual_time_pos, sim_lag*1000, colorline1, color=line_color, label=label1)
                else:
                    if sim_time!='':
                        ax[3].plot(sim_time, obs1['lag'], colorline1, color=line_color, label=label1)
                    else:
                        ax[3].plot(residual_time_pos, obs1['lag'], colorline1, color=line_color, label=label1)
            else:
                if sim_lag!='':
                    if sim_time!='':
                        ax[3].plot(sim_time, sim_lag*1000, colorline1, color=line_color)
                    else:
                        ax[3].plot(residual_time_pos, sim_lag*1000, colorline1, color=line_color)
                else:
                    if sim_time!='':
                        ax[3].plot(sim_time, obs1['lag'], colorline1, color=line_color)
                    else:
                        ax[3].plot(residual_time_pos, obs1['lag'], colorline1, color=line_color)

            # show the legend
            if label1 != '':
                ax[3].legend()

            ax[3].set_xlabel('Time [s]')
            ax[3].set_ylabel('Lag [m]')
            ax[3].grid(True)
            ax[3].grid(linestyle='--',color='lightgray')

            # delete the plot in the middle
            ax[4].axis('off')
            
            # # put as the super title the name
            # plt.suptitle(name)
            ax[5].axis('off')

            # plot the residuals against time
            if fit_funct=='' and mag_noise=='' and vel_noise=='':
                ax[6].plot(residual_time_pos_err, residuals_vel, '.', color=line_color)
            ax[6].set_ylabel('Res.vel [km/s]')
            ax[6].grid(True)
            ax[6].grid(linestyle='--',color='lightgray')
            # use the same limits of ax[3]
            ax[6].set_xlim(ax[2].get_xlim())

            # plot the residuals against time
            if fit_funct=='' and mag_noise=='' and vel_noise=='':
                ax[7].plot(residual_time_pos_err, residuals_lag*1000, '.', color=line_color)
            ax[7].set_ylabel('Res.lag [m]')
            ax[7].grid(True)
            ax[7].grid(linestyle='--',color='lightgray')
            # use the same limits of ax[3]
            ax[7].set_xlim(ax[3].get_xlim())

        else:

            if label1!='':
                if sim_time!='':
                    ax[2].plot(sim_time, obs1['velocities'], colorline1, color=line_color, label=label1)
                else:
                    ax[2].plot(residual_time_pos, obs1['velocities'], colorline1, color=line_color, label=label1)
            else:
                if sim_time!='':
                    ax[2].plot(sim_time, obs1['velocities'], colorline1, color=line_color)
                else:
                    ax[2].plot(residual_time_pos, obs1['velocities'], colorline1, color=line_color)
            # show the legend
            if label1 != '':
                ax[2].legend()

            ax[2].set_xlabel('Time [s]')
            ax[2].set_ylabel('Velocity [km/s]')
            ax[2].grid(True)
            ax[2].grid(linestyle='--',color='lightgray')

            # delete the plot in the middle
            ax[3].axis('off')
            
            # # put as the super title the name
            # plt.suptitle(name)
            ax[4].axis('off')

            # plot the residuals against time
            if fit_funct=='' and mag_noise=='' and vel_noise=='':
                ax[5].plot(residual_time_pos_err, residuals_vel, '.', color=line_color)
            ax[5].set_ylabel('Res.vel [km/s]')
            ax[5].grid(True)
            ax[5].grid(linestyle='--',color='lightgray')
            # use the same limits of ax[3]
            ax[5].set_xlim(ax[2].get_xlim())


    else :
        if fig=='' and ax=='':
            fig, ax = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
            # flat the ax
            ax = ax.flatten()
            return fig, ax
        
        # plot the magnitude curve with height
        ax[0].plot(obs1['absolute_magnitudes'],obs1['height'], colorline1)

        ax[0].set_xlabel('Absolute Magnitude [-]')
        ax[0].set_ylabel('Height [km]')
        # check if the axis is inverted
        is_x_inverted, _ =check_axis_inversion(ax[0])
        if is_x_inverted==False:
            ax[0].invert_xaxis()
        # grid on
        ax[0].grid(True)

        # plot 
        if label1 == '':
            ax[1].plot(obs1['time'], obs1['velocities'], colorline1)
        else:
            ax[1].plot(obs1['time'], obs1['velocities'], colorline1, label=label1)

        # show the legend
        if label1 != '':
            ax[1].legend()

        ax[1].set_xlabel('Time [s]')
        ax[1].set_ylabel('Velocity [km/s]')
        ax[1].grid(True)

        # grid on on both subplot with -- as linestyle and light gray color
        ax[1].grid(linestyle='--',color='lightgray')
        # grid on
        ax[0].grid(linestyle='--',color='lightgray')

    plt.tight_layout()

    



#### Reader #############################################################################


def read_GenerateSimulations_output_to_PCA(file_path, name='', fit_funct='', real_event='', flag_for_PCA=False):
    real_event_copy = copy.deepcopy(real_event)
    if name!='':   
        print(name) 
    gensim_data = read_GenerateSimulations_output(file_path, real_event_copy, flag_for_PCA)
    if gensim_data is None:
        return None
    else:
        pd_datfram_PCA = array_to_pd_dataframe_PCA(gensim_data, real_event_copy)
        return pd_datfram_PCA


def read_GenerateSimulations_output(file_path, real_event, flag_for_PCA=False):

    f = open(file_path,"r")
    data = json.loads(f.read())

    # show processed event
    print(file_path)

    # check if there is 'ht_sampled' in the data
    if 'ht_sampled' not in data:
        print("Warning: 'ht_sampled' not in data. Skipping.")
        return None
    if data['ht_sampled']!= None: 

        vel_sim=data['simulation_results']['leading_frag_vel_arr'][:-1]#['brightest_vel_arr']#['leading_frag_vel_arr']#['main_vel_arr']
        ht_sim=data['simulation_results']['leading_frag_height_arr'][:-1]#['brightest_height_arr']['leading_frag_height_arr']['main_height_arr']
        time_sim=data['simulation_results']['time_arr'][:-1]#['main_time_arr']
        abs_mag_sim=data['simulation_results']['abs_magnitude'][:-1]
        len_sim=data['simulation_results']['leading_frag_length_arr'][:-1]#['brightest_length_arr']
        Dynamic_pressure= data['simulation_results']['leading_frag_dyn_press_arr'][:-1]
        
        # ht_obs=data['ht_sampled']
        # try:
        #     index_ht_sim=next(x for x, val in enumerate(ht_sim) if val <= ht_obs[0])
        # except StopIteration:
        #     # index_ht_sim = None
        #     print('The first element of the observation is not in the simulation')
        #     return None

        # try:
        #     index_ht_sim_end=next(x for x, val in enumerate(ht_sim) if val <= ht_obs[-1])
        # except StopIteration:
        #     # index_ht_sim_end = None
        #     print('The last element of the observation is not in the simulation')
        #     return None
        
        # if real_event!= '':
        #     mag_obs=real_event['absolute_magnitudes']
        # else:
        #     mag_obs=data['mag_sampled']

        mag_obs=real_event['absolute_magnitudes']

        # print('read_GenerateSimulations_output mag',mag_obs[0],'-',mag_obs[-1])

        try:
            # find the index of the first element of abs_mag_sim that is smaller than the first element of mag_obs
            index_abs_mag_sim_start = next(i for i, val in enumerate(abs_mag_sim) if val <= mag_obs[0])
            if flag_for_PCA:
                index_abs_mag_sim_start = index_abs_mag_sim_start - 1 + np.random.randint(2)
            else:
                index_abs_mag_sim_start = index_abs_mag_sim_start - 1 # + np.random.randint(2)
        except StopIteration:
            print("The first observation height is not within the simulation data range.")
            return None
        try:   
            index_abs_mag_sim_end = next(i for i, val in enumerate(abs_mag_sim[::-1]) if val <= mag_obs[-1])
            if flag_for_PCA:
                index_abs_mag_sim_end = len(abs_mag_sim) - index_abs_mag_sim_end + 1 - np.random.randint(2)
            else:
                index_abs_mag_sim_end = len(abs_mag_sim) - index_abs_mag_sim_end + 1        
        except StopIteration:
            print("The first observation height is not within the simulation data range.")
            return None
        
        # print('mag',index_abs_mag_sim_start,'-',index_abs_mag_sim_end,'\nheight',index_ht_sim,'-',index_ht_sim_end)
            
        abs_mag_sim = abs_mag_sim[index_abs_mag_sim_start:index_abs_mag_sim_end]
        vel_sim = vel_sim[index_abs_mag_sim_start:index_abs_mag_sim_end]
        time_sim = time_sim[index_abs_mag_sim_start:index_abs_mag_sim_end]
        ht_sim = ht_sim[index_abs_mag_sim_start:index_abs_mag_sim_end]
        len_sim = len_sim[index_abs_mag_sim_start:index_abs_mag_sim_end]
        Dynamic_pressure = Dynamic_pressure[index_abs_mag_sim_start:index_abs_mag_sim_end]



        # abs_mag_sim=abs_mag_sim[index_ht_sim:index_ht_sim_end]
        # vel_sim=vel_sim[index_ht_sim:index_ht_sim_end]
        # time_sim=time_sim[index_ht_sim:index_ht_sim_end]
        # ht_sim=ht_sim[index_ht_sim:index_ht_sim_end]
        # len_sim=len_sim[index_ht_sim:index_ht_sim_end]

        # closest_indices = find_closest_index(ht_sim, ht_obs)

        # Dynamic_pressure= data['simulation_results']['leading_frag_dyn_press_arr']
        # Dynamic_pressure= Dynamic_pressure[index_ht_sim:index_ht_sim_end]
        # Dynamic_pressure=[Dynamic_pressure[jj_index_cut] for jj_index_cut in closest_indices]

        # abs_mag_sim=[abs_mag_sim[jj_index_cut] for jj_index_cut in closest_indices]
        # vel_sim=[vel_sim[jj_index_cut] for jj_index_cut in closest_indices]
        # time_sim=[time_sim[jj_index_cut] for jj_index_cut in closest_indices]
        # ht_sim=[ht_sim[jj_index_cut] for jj_index_cut in closest_indices]
        # len_sim=[len_sim[jj_index_cut] for jj_index_cut in closest_indices]

        # divide the vel_sim by 1000 considering is a list
        time_sim = [i-time_sim[0] for i in time_sim]
        # vel_sim = [i/1000 for i in vel_sim]
        len_sim = [i-len_sim[0] for i in len_sim]
        # ht_sim = [i/1000 for i in ht_sim]

        # Load the constants
        const, _ = loadConstants(file_path)
        const.dens_co = np.array(const.dens_co)

        # Compute the erosion energies
        erosion_energy_per_unit_cross_section, erosion_energy_per_unit_mass = wmpl.MetSim.MetSimErosion.energyReceivedBeforeErosion(const)

        gensim_data = {
        'name': file_path,
        'type': 'Simulation',
        'v_init': vel_sim[0], # m/s
        'velocities': np.array(vel_sim), # m/s
        'height': np.array(ht_sim), # m
        'absolute_magnitudes': np.array(abs_mag_sim),
        'lag': np.array(len_sim-(vel_sim[0]*np.array(time_sim))), # m +len_sim[0]
        'length': np.array(len_sim), # m
        'time': np.array(time_sim), # s
        'v_avg': np.mean(vel_sim), # m/s
        'v_init_180km': data['params']['v_init']['val'], # m/s
        'Dynamic_pressure_peak_abs_mag': Dynamic_pressure[np.argmin(abs_mag_sim)],
        'zenith_angle': data['params']['zenith_angle']['val']*180/np.pi,
        'mass': data['params']['m_init']['val'],
        'rho': data['params']['rho']['val'],
        'sigma': data['params']['sigma']['val'],
        'erosion_height_start': data['params']['erosion_height_start']['val']/1000,
        'erosion_coeff': data['params']['erosion_coeff']['val'],
        'erosion_mass_index': data['params']['erosion_mass_index']['val'],
        'erosion_mass_min': data['params']['erosion_mass_min']['val'],
        'erosion_mass_max': data['params']['erosion_mass_max']['val'],
        'erosion_range': np.log10(data['params']['erosion_mass_max']['val']) - np.log10(data['params']['erosion_mass_min']['val']),
        'erosion_energy_per_unit_cross_section': erosion_energy_per_unit_cross_section,
        'erosion_energy_per_unit_mass': erosion_energy_per_unit_mass
        }

        return gensim_data
    
    else:
        return None


def Old_GenSym_json_get_vel_lag(data, fps=32):

    ht_sim=data['simulation_results']['leading_frag_height_arr'][:-1]#['brightest_height_arr']['leading_frag_height_arr']['main_height_arr']
    ht_obs=data['ht_sampled']
    time_sampled = np.array(data['time_sampled'])
    len_sampled = np.array(data['len_sampled'])

    closest_indices = find_closest_index(ht_sim, ht_obs)

    vel_sim=data['simulation_results']['leading_frag_vel_arr'][:-1]#['brightest_vel_arr']#['leading_frag_vel_arr']#['main_vel_arr']
    vel_sim=[vel_sim[jj_index_cut] for jj_index_cut in closest_indices]

    # get the new velocity with noise
    for vel_ii in range(1,len(time_sampled)):
        if time_sampled[vel_ii]-time_sampled[vel_ii-1]<1.0/fps:
        # if time_sampled[vel_ii] % 0.03125 < 0.000000001:
            if vel_ii+1<len(len_sampled):
                vel_sim[vel_ii+1]=(len_sampled[vel_ii+1]-len_sampled[vel_ii-1])/(time_sampled[vel_ii+1]-time_sampled[vel_ii-1])
        else:
            vel_sim[vel_ii]=(len_sampled[vel_ii]-len_sampled[vel_ii-1])/(time_sampled[vel_ii]-time_sampled[vel_ii-1])

    data['vel_sampled']=vel_sim
    
    lag_sim=len_sampled-(vel_sim[0]*time_sampled) #+len_sampled[0]

    data['lag_sampled']=lag_sim.tolist()

    return data


def read_with_noise_GenerateSimulations_output(file_path, fps=32):

    f = open(file_path,"r")
    data = json.loads(f.read())

    if data['ht_sampled']!= None: 

        ht_sim=data['simulation_results']['leading_frag_height_arr']#['brightest_height_arr']['leading_frag_height_arr']['main_height_arr']

        ht_obs=data['ht_sampled']

        closest_indices = find_closest_index(ht_sim, ht_obs)

        Dynamic_pressure= data['simulation_results']['leading_frag_dyn_press_arr']
        Dynamic_pressure=[Dynamic_pressure[jj_index_cut] for jj_index_cut in closest_indices]

        # Load the constants
        const, _ = loadConstants(file_path)
        const.dens_co = np.array(const.dens_co)

        # Compute the erosion energies
        erosion_energy_per_unit_cross_section, erosion_energy_per_unit_mass = wmpl.MetSim.MetSimErosion.energyReceivedBeforeErosion(const)

        # if is not present the vel_sampled in the data
        if 'vel_sampled' not in data:
            data = Old_GenSym_json_get_vel_lag(data, fps)

        gensim_data = {
        'name': file_path,
        'type': 'Observation_sim',
        'v_init_180km': data['params']['v_init']['val'], # m/s
        'v_init': data['vel_sampled'][0], # m/s
        'velocities': np.array(data['vel_sampled']), # m/s
        'height': np.array(data['ht_sampled']), # m
        'absolute_magnitudes': np.array(data['mag_sampled']),
        'lag': np.array(data['lag_sampled']), # m
        'length': np.array(data['len_sampled']), # m
        'time': np.array(data['time_sampled']), # s
        'v_avg': np.mean(data['vel_sampled']), # m/s
        'Dynamic_pressure_peak_abs_mag': Dynamic_pressure[np.argmin(data['mag_sampled'])],
        'zenith_angle': data['params']['zenith_angle']['val']*180/np.pi,
        'mass': data['params']['m_init']['val'],
        'rho': data['params']['rho']['val'],
        'sigma': data['params']['sigma']['val'],
        'erosion_height_start': data['params']['erosion_height_start']['val']/1000,
        'erosion_coeff': data['params']['erosion_coeff']['val'],
        'erosion_mass_index': data['params']['erosion_mass_index']['val'],
        'erosion_mass_min': data['params']['erosion_mass_min']['val'],
        'erosion_mass_max': data['params']['erosion_mass_max']['val'],
        'erosion_range': np.log10(data['params']['erosion_mass_max']['val']) - np.log10(data['params']['erosion_mass_min']['val']),
        'erosion_energy_per_unit_cross_section': erosion_energy_per_unit_cross_section,
        'erosion_energy_per_unit_mass': erosion_energy_per_unit_mass
        }

        return gensim_data
    
    else:
        return None


def read_RunSim_output(simulation_MetSim_object, real_event, MetSim_phys_file_path):
    '''
    dict_keys(['const', 'frag_main', 'time_arr', 'luminosity_arr', 'luminosity_main_arr', 'luminosity_eroded_arr', 
    'electron_density_total_arr', 'tau_total_arr', 'tau_main_arr', 'tau_eroded_arr', 'brightest_height_arr', 
    'brightest_length_arr', 'brightest_vel_arr', 'leading_frag_height_arr', 'leading_frag_length_arr', 
    'leading_frag_vel_arr', 'leading_frag_dyn_press_arr', 'mass_total_active_arr', 'main_mass_arr', 
    'main_height_arr', 'main_length_arr', 'main_vel_arr', 'main_dyn_press_arr', 'abs_magnitude', 
    'abs_magnitude_main', 'abs_magnitude_eroded', 'wake_results', 'wake_max_lum'])

    in const

    dict_keys(['dt', 'total_time', 'n_active', 'm_kill', 'v_kill', 'h_kill', 'len_kill', 'h_init', 'P_0m', 
    'dens_co', 'r_earth', 'total_fragments', 'wake_psf', 'wake_extension', 'rho', 'm_init', 'v_init', 
    'shape_factor', 'sigma', 'zenith_angle', 'gamma', 'rho_grain', 'lum_eff_type', 'lum_eff', 'mu', 
    'erosion_on', 'erosion_bins_per_10mass', 'erosion_height_start', 'erosion_coeff', 'erosion_height_change', 
    'erosion_coeff_change', 'erosion_rho_change', 'erosion_sigma_change', 'erosion_mass_index', 'erosion_mass_min', 
    'erosion_mass_max', 'disruption_on', 'compressive_strength', 'disruption_height', 'disruption_erosion_coeff', 
    'disruption_mass_index', 'disruption_mass_min_ratio', 'disruption_mass_max_ratio', 'disruption_mass_grain_ratio', 
    'fragmentation_on', 'fragmentation_show_individual_lcs', 'fragmentation_entries', 'fragmentation_file_name', 
    'electron_density_meas_ht', 'electron_density_meas_q', 'erosion_beg_vel', 'erosion_beg_mass', 'erosion_beg_dyn_press', 
    'mass_at_erosion_change', 'energy_per_cs_before_erosion', 'energy_per_mass_before_erosion', 'main_mass_exhaustion_ht', 'main_bottom_ht'])
    '''

    vel_sim=simulation_MetSim_object.leading_frag_vel_arr #main_vel_arr
    ht_sim=simulation_MetSim_object.leading_frag_height_arr #main_height_arr
    time_sim=simulation_MetSim_object.time_arr
    abs_mag_sim=simulation_MetSim_object.abs_magnitude
    len_sim=simulation_MetSim_object.leading_frag_length_arr #main_length_arr
    Dynamic_pressure=simulation_MetSim_object.leading_frag_dyn_press_arr # main_dyn_press_arr
    
    mag_obs=real_event['absolute_magnitudes']

    # print('read_RunSim_output mag',mag_obs[0],'-',mag_obs[-1])

    try:
        # find the index of the first element of abs_mag_sim that is smaller than the first element of mag_obs
        index_abs_mag_sim_start = next(i for i, val in enumerate(abs_mag_sim) if val <= mag_obs[0])
        index_abs_mag_sim_start = index_abs_mag_sim_start - 1 # + np.random.randint(2)
    except StopIteration:
        print("The first observation height is not within the simulation data range.")
        return None
    try:   
        index_abs_mag_sim_end = next(i for i, val in enumerate(abs_mag_sim[::-1]) if val <= mag_obs[-1])
        index_abs_mag_sim_end = len(abs_mag_sim) - index_abs_mag_sim_end + 1 # - 1           
    except StopIteration:
        print("The first observation height is not within the simulation data range.")
        return None
        
    abs_mag_sim = abs_mag_sim[index_abs_mag_sim_start:index_abs_mag_sim_end]
    vel_sim = vel_sim[index_abs_mag_sim_start:index_abs_mag_sim_end]
    time_sim = time_sim[index_abs_mag_sim_start:index_abs_mag_sim_end]
    ht_sim = ht_sim[index_abs_mag_sim_start:index_abs_mag_sim_end]
    len_sim = len_sim[index_abs_mag_sim_start:index_abs_mag_sim_end]
    Dynamic_pressure = Dynamic_pressure[index_abs_mag_sim_start:index_abs_mag_sim_end]

    # ht_obs=real_event['height']
    # try:
    #     # find the index of the first element of the simulation that is equal to the first element of the observation
    #     index_ht_sim = next(x for x, val in enumerate(ht_sim) if val <= ht_obs[0])
    # except StopIteration:
    #     print("The first observation height is not within the simulation data range.")
    #     index_ht_sim = 0

    # try:
    #     # find the index of the last element of the simulation that is equal to the last element of the observation
    #     index_ht_sim_end = next(x for x, val in enumerate(ht_sim) if val <= ht_obs[-1])
    # except StopIteration:
    #     print("The last observation height is not within the simulation data range.")
    #     index_ht_sim_end = len(ht_sim) - 2 # at -1 there is Nan in some sim value


    # abs_mag_sim=abs_mag_sim[index_ht_sim:index_ht_sim_end]
    # vel_sim=vel_sim[index_ht_sim:index_ht_sim_end]
    # time_sim=time_sim[index_ht_sim:index_ht_sim_end]
    # ht_sim=ht_sim[index_ht_sim:index_ht_sim_end]
    # len_sim=len_sim[index_ht_sim:index_ht_sim_end]
    # Dynamic_pressure= Dynamic_pressure[index_ht_sim:index_ht_sim_end]

    # closest_indices = find_closest_index(ht_sim, ht_obs)

    # abs_mag_sim=[abs_mag_sim[jj_index_cut] for jj_index_cut in closest_indices]
    # vel_sim=[vel_sim[jj_index_cut] for jj_index_cut in closest_indices]
    # time_sim=[time_sim[jj_index_cut] for jj_index_cut in closest_indices]
    # ht_sim=[ht_sim[jj_index_cut] for jj_index_cut in closest_indices]
    # len_sim=[len_sim[jj_index_cut] for jj_index_cut in closest_indices]
    # Dynamic_pressure=[Dynamic_pressure[jj_index_cut] for jj_index_cut in closest_indices]

    # divide the vel_sim by 1000 considering is a list
    time_sim = [i-time_sim[0] for i in time_sim]
    # vel_sim = [i/1000 for i in vel_sim]
    len_sim = [i-len_sim[0] for i in len_sim]
    # ht_sim = [i/1000 for i in ht_sim]

    output_phys = read_MetSim_phyProp_output(MetSim_phys_file_path)

    gensim_data = {
        'name': MetSim_phys_file_path,
        'type': 'MetSim',
        'v_init': vel_sim[0], # m/s
        'velocities': np.array(vel_sim), # m/s
        'v_init_180km': simulation_MetSim_object.const.v_init, # m/s
        'height': np.array(ht_sim), # m
        'absolute_magnitudes': np.array(abs_mag_sim),
        'lag': np.array(len_sim-(vel_sim[0]*np.array(time_sim))), # m +len_sim[0]
        'length': np.array(len_sim), # m
        'time': np.array(time_sim), # s
        'v_avg': np.mean(vel_sim), # m/s
        'Dynamic_pressure_peak_abs_mag': Dynamic_pressure[np.argmin(abs_mag_sim)],
        'zenith_angle': simulation_MetSim_object.const.zenith_angle*180/np.pi,
        'mass': output_phys[0],
        'rho': output_phys[1],
        'sigma': output_phys[2],
        'erosion_height_start': output_phys[3],
        'erosion_coeff': output_phys[4],
        'erosion_mass_index': output_phys[5],
        'erosion_mass_min': output_phys[6],
        'erosion_mass_max': output_phys[7],
        'erosion_range': output_phys[8],
        'erosion_energy_per_unit_cross_section': output_phys[9],
        'erosion_energy_per_unit_mass': output_phys[10]
        }

    return gensim_data


def read_pickle_reduction_file(file_path, MetSim_phys_file_path='', obs_sep=False):


    with open(file_path, 'rb') as f:
        traj = pickle.load(f, encoding='latin1')

    v_avg = traj.v_avg
    v_init=traj.orbit.v_init
    jd_dat=traj.jdt_ref
    obs_data = []
    # obs_init_vel = []
    for obs in traj.observations:
        if obs.station_id == "01G" or obs.station_id == "02G" or obs.station_id == "01F" or obs.station_id == "02F" or obs.station_id == "1G" or obs.station_id == "2G" or obs.station_id == "1F" or obs.station_id == "2F":
            obs_dict = {
                'v_init': obs.v_init, # m/s
                'velocities': np.array(obs.velocities), # m/s
                # 'velocities': np.array(obs.velocities)[1:], # m/s
                'height': np.array(obs.model_ht), # m
                # pick all except the first element
                # 'height' : np.array(obs.model_ht)[1:],
                'absolute_magnitudes': np.array(obs.absolute_magnitudes),
                # 'absolute_magnitudes': np.array(obs.absolute_magnitudes)[1:],
                'lag': np.array(obs.lag), # m
                # 'lag': np.array(obs.lag)[1:],
                'length': np.array(obs.state_vect_dist), # m
                # 'length': np.array(obs.state_vect_dist)[1:],
                'time': np.array(obs.time_data) # s
                # 'time': np.array(obs.time_data)[1:]
                # 'station_id': obs.station_id
                # 'elev_data':  np.array(obs.elev_data)
            }
            
            obs_dict['velocities'][0] = obs_dict['v_init']
            obs_data.append(obs_dict)

            # obs_init_vel.append(obs.v_init)
                
            lat_dat=obs.lat
            lon_dat=obs.lon

        else:
            print(obs.station_id,'Station not in the list of stations')
            continue
    
    
    
    # Save distinct values for the two observations
    obs1, obs2 = obs_data[0], obs_data[1]

    # # do the average of the two obs_init_vel
    # v_init_vel = np.mean(obs_init_vel)

    # save time of each observation
    obs1_time = np.array(obs1['time'])
    obs2_time = np.array(obs2['time'])
    obs1_length = np.array(obs1['length'])
    obs2_length = np.array(obs2['length'])
    obs1_height = np.array(obs1['height'])
    obs2_height = np.array(obs2['height'])
    obs1_velocities = np.array(obs1['velocities'])
    obs2_velocities = np.array(obs2['velocities'])
    obs1_absolute_magnitudes = np.array(obs1['absolute_magnitudes'])
    obs2_absolute_magnitudes = np.array(obs2['absolute_magnitudes'])
    obs1_lag = np.array(obs1['lag'])
    obs2_lag = np.array(obs2['lag'])
    
    # Combine obs1 and obs2
    combined_obs = {}
    for key in ['velocities', 'height', 'absolute_magnitudes', 'lag', 'length', 'time']: #, 'elev_data']:
        combined_obs[key] = np.concatenate((obs1[key], obs2[key]))

    # Order the combined observations based on time
    sorted_indices = np.argsort(combined_obs['time'])
    for key in ['time', 'velocities', 'height', 'absolute_magnitudes', 'lag', 'length']: #, 'elev_data']:
        combined_obs[key] = combined_obs[key][sorted_indices]

    # check if any value is below 10 absolute_magnitudes and print find values below 8 absolute_magnitudes
    if np.any(combined_obs['absolute_magnitudes'] > 8):
        print('Found values below 8 absolute magnitudes:', combined_obs['absolute_magnitudes'][combined_obs['absolute_magnitudes'] > 8])
    
    # delete any values above 10 absolute_magnitudes and delete the corresponding values in the other arrays
    combined_obs = {key: combined_obs[key][combined_obs['absolute_magnitudes'] < 8] for key in combined_obs.keys()}

    Dynamic_pressure_peak_abs_mag=(wmpl.Utils.Physics.dynamicPressure(lat_dat, lon_dat, combined_obs['height'][np.argmin(combined_obs['absolute_magnitudes'])], jd_dat, combined_obs['velocities'][np.argmin(combined_obs['absolute_magnitudes'])]))
    const=Constants()
    zenith_angle=zenithAngleAtSimulationBegin(const.h_init, traj.rbeg_ele, traj.orbit.zc, const.r_earth)

    if MetSim_phys_file_path != '':
        output_phys = read_MetSim_phyProp_output(MetSim_phys_file_path)
        type_sim='MetSim'
        
    else:
        # if no data on weight is 0
        mass=(0)
        rho=(0)
        sigma=(0)
        erosion_height_start=(0)
        erosion_coeff=(0)
        erosion_mass_index=(0)
        erosion_mass_min=(0)
        erosion_mass_max=(0)
        erosion_range=(0)
        erosion_energy_per_unit_cross_section_arr=(0)
        erosion_energy_per_unit_mass_arr=(0)
        v_180km=v_init

        type_sim='Observation'

        # put all the varible in a array mass, rho, sigma, erosion_height_start, erosion_coeff, erosion_mass_index, erosion_mass_min, erosion_mass_max, erosion_range, erosion_energy_per_unit_cross_section_arr, erosion_energy_per_unit_mass_arr
        output_phys = [mass, rho, sigma, erosion_height_start, erosion_coeff, erosion_mass_index, erosion_mass_min, erosion_mass_max, erosion_range, erosion_energy_per_unit_cross_section_arr, erosion_energy_per_unit_mass_arr, v_180km]

    # # delete the elev_data from the combined_obs
    # del combined_obs['elev_data']

    # add to combined_obs the avg velocity and the peak dynamic pressure and all the physical parameters
    combined_obs['name'] = file_path    
    combined_obs['v_init'] = v_init
    combined_obs['v_init_180km'] = output_phys[11]
    combined_obs['obs1_time'] = obs1_time
    combined_obs['obs2_time'] = obs2_time
    combined_obs['obs1_length'] = obs1_length   
    combined_obs['obs2_length'] = obs2_length
    combined_obs['obs1_height'] = obs1_height
    combined_obs['obs2_height'] = obs2_height
    combined_obs['obs1_velocities'] = obs1_velocities
    combined_obs['obs2_velocities'] = obs2_velocities
    combined_obs['obs1_absolute_magnitudes'] = obs1_absolute_magnitudes
    combined_obs['obs2_absolute_magnitudes'] = obs2_absolute_magnitudes
    combined_obs['obs1_lag'] = obs1_lag
    combined_obs['obs2_lag'] = obs2_lag
    combined_obs['type'] = type_sim
    combined_obs['v_avg'] = v_avg
    combined_obs['Dynamic_pressure_peak_abs_mag'] = Dynamic_pressure_peak_abs_mag
    combined_obs['zenith_angle'] = zenith_angle*180/np.pi
    combined_obs['mass'] = output_phys[0]
    combined_obs['rho'] = output_phys[1]
    combined_obs['sigma'] = output_phys[2]
    combined_obs['erosion_height_start'] = output_phys[3]
    combined_obs['erosion_coeff'] = output_phys[4]
    combined_obs['erosion_mass_index'] = output_phys[5]
    combined_obs['erosion_mass_min'] = output_phys[6]
    combined_obs['erosion_mass_max'] = output_phys[7]
    combined_obs['erosion_range'] = output_phys[8]
    combined_obs['erosion_energy_per_unit_cross_section'] = output_phys[9]
    combined_obs['erosion_energy_per_unit_mass'] = output_phys[10]

    if obs_sep:
        return combined_obs, obs1, obs2
    else:
        return combined_obs


def read_MetSim_phyProp_output(MetSim_phys_file_path):

    # check if in os.path.join(root, name_file) present and then open the .json file with the same name as the pickle file with in stead of _trajectory.pickle it has _sim_fit_latest.json
    if os.path.isfile(MetSim_phys_file_path):
        with open(MetSim_phys_file_path,'r') as json_file: # 20210813_061453_sim_fit.json
            print('Loading Physical Characteristics MetSim file:', MetSim_phys_file_path)
            data = json.load(json_file)
            mass=(data['m_init'])
            # add also rho	sigma	erosion_height_start	erosion_coeff	erosion_mass_index	erosion_mass_min	erosion_mass_max	erosion_range	erosion_energy_per_unit_cross_section	erosion_energy_per_unit_mass
            v_180km=(data['v_init'])
            rho=(data['rho'])
            sigma=(data['sigma'])
            erosion_height_start=(data['erosion_height_start']/1000)
            erosion_coeff=(data['erosion_coeff'])
            erosion_mass_index=(data['erosion_mass_index'])
            erosion_mass_min=(data['erosion_mass_min'])
            erosion_mass_max=(data['erosion_mass_max'])

            # Compute the erosion range
            erosion_range=(np.log10(data['erosion_mass_max']) - np.log10(data['erosion_mass_min']))

            cost_path = os.path.join(MetSim_phys_file_path)

            # Load the constants
            const, _ = loadConstants(cost_path)
            const.dens_co = np.array(const.dens_co)

            # Compute the erosion energies
            erosion_energy_per_unit_cross_section, erosion_energy_per_unit_mass = wmpl.MetSim.MetSimErosion.energyReceivedBeforeErosion(const)
            erosion_energy_per_unit_cross_section_arr=(erosion_energy_per_unit_cross_section)
            erosion_energy_per_unit_mass_arr=(erosion_energy_per_unit_mass)

    else:
        print('No json file:',MetSim_phys_file_path)

        # if no data on weight is 0
        mass=(0)
        rho=(0)
        sigma=(0)
        erosion_height_start=(0)
        erosion_coeff=(0)
        erosion_mass_index=(0)
        erosion_mass_min=(0)
        erosion_mass_max=(0)
        erosion_range=(0)
        erosion_energy_per_unit_cross_section_arr=(0)
        erosion_energy_per_unit_mass_arr=(0)
        v_180km=(0)

    # put all the varible in a array mass, rho, sigma, erosion_height_start, erosion_coeff, erosion_mass_index, erosion_mass_min, erosion_mass_max, erosion_range, erosion_energy_per_unit_cross_section_arr, erosion_energy_per_unit_mass_arr
    output_phys = [mass, rho, sigma, erosion_height_start, erosion_coeff, erosion_mass_index, erosion_mass_min, erosion_mass_max, erosion_range, erosion_energy_per_unit_cross_section_arr, erosion_energy_per_unit_mass_arr, v_180km]
    
    return output_phys



def array_to_pd_dataframe_PCA(data, test_data=[]):

    if data is None:
        # Handle the None case, maybe log an error or return an empty DataFrame
        print(f"Warning: 'data' is None for source returning an empty DataFrame.")
        return pd.DataFrame()  # or any other appropriate action
    # do a copy of data_array
    data_array = data.copy()

    # compute the linear regression
    data_array['v_init'] = data_array['v_init']/1000
    data_array['v_avg'] = data_array['v_avg']/1000
    data_array['velocities'] = [i/1000 for i in data_array['velocities']] # convert m/s to km/s
    data_array['height'] = [i/1000 for i in data_array['height']]
    data_array['lag']=[i/1000 for i in data_array['lag']]
    v0=data_array['v_init']

    # from 'time_sampled' extract the last element and save it in a list
    duration = data_array['time'][-1]
    begin_height = data_array['height'][0]
    end_height = data_array['height'][-1]
    peak_abs_mag = data_array['absolute_magnitudes'][np.argmin(data_array['absolute_magnitudes'])]
    F_param = (begin_height - (data_array['height'][np.argmin(data_array['absolute_magnitudes'])])) / (begin_height - end_height)
    peak_mag_height = data_array['height'][np.argmin(data_array['absolute_magnitudes'])]
    beg_abs_mag	= data_array['absolute_magnitudes'][0]
    end_abs_mag	= data_array['absolute_magnitudes'][-1]
    trail_len = data_array['length'][-1]
    avg_lag = np.mean(data_array['lag'])


    kc_par = begin_height + (2.86 - 2*np.log(data_array['v_init']))/0.0612
    
    try:
        # fit a line to the throught the vel_sim and ht_sim
        a, b = np.polyfit(data_array['time'],data_array['velocities'], 1)
        acceleration_lin = a

        t0 = np.mean(data_array['time'])

        # initial guess of deceleration decel equal to linear fit of velocity
        p0 = [a, 0, 0, t0]

        opt_res = opt.minimize(lag_residual, p0, args=(np.array(data_array['time']), np.array(data_array['lag'])), method='Nelder-Mead')

        # sample the fit for the velocity and acceleration
        a_t0, b_t0, c_t0, t0 = opt_res.x

        # compute reference decelearation
        t_decel_ref = (t0 + np.max(data_array['time']))/2
        decel_t0 = cubic_acceleration(t_decel_ref, a_t0, b_t0, t0)[0]

        a_t0=-abs(a_t0)
        b_t0=-abs(b_t0)

        acceleration_parab_t0=a_t0*6 + b_t0*2

        a3, b3, c3 = np.polyfit(data_array['time'],data_array['velocities'], 2)
        acceleration_parab=a3*2 + b3

        # Assuming the jacchiaVel function is defined as:
        def jacchiaVel(t, a1, a2, v_init):
            return v_init - np.abs(a1) * np.abs(a2) * np.exp(np.abs(a2) * t)

        # Generating synthetic observed data for demonstration
        t_observed = np.array(data_array['time'])  # Observed times

        # Residuals function for optimization
        def residuals(params):
            a1, a2 = params
            predicted_velocity = jacchiaVel(t_observed, a1, a2, v0)
            return np.sum((data_array['velocities'] - predicted_velocity)**2)

        # Initial guess for a1 and a2
        initial_guess = [0.005,	10]

        # Apply minimize to the residuals
        result = minimize(residuals, initial_guess)

        # Results
        jac_a1, jac_a2 = abs(result.x)

        acc_jacchia = abs(jac_a1)*abs(jac_a2)**2
    except Exception as e:
        # Handle exceptions and provide default values
        print(f"Error in computation: {e}, filling with default zeros.")
        acceleration_lin = 0
        a_t0 = b_t0 = c_t0 = t0 = 0
        decel_t0 = 0
        acceleration_parab_t0 = 0
        a3 = b3 = c3 = 0
        acceleration_parab = 0
        jac_a1 = jac_a2 = acc_jacchia = 0

    try:
        # fit a line to the throught the obs_vel and ht_sim
        index_ht_peak = next(x for x, val in enumerate(data_array['height']) if val <= peak_mag_height)
    except StopIteration:
        # Handle the case where no height is less than or equal to peak_mag_height
        index_ht_peak = len(data_array['height']) // 2

    # Check if the arrays are non-empty before fitting the polynomial
    if len(data_array['height'][:index_ht_peak]) > 0 and len(data_array['absolute_magnitudes'][:index_ht_peak]) > 0:
        a3_Inabs, b3_Inabs, c3_Inabs = np.polyfit(data_array['height'][:index_ht_peak], data_array['absolute_magnitudes'][:index_ht_peak], 2)
    else:
        # Handle the case of empty input arrays
        a3_Inabs, b3_Inabs, c3_Inabs = 0, 0, 0

    # Check if the arrays are non-empty before fitting the polynomial
    if len(data_array['height'][index_ht_peak:]) > 0 and len(data_array['absolute_magnitudes'][index_ht_peak:]) > 0:
        a3_Outabs, b3_Outabs, c3_Outabs = np.polyfit(data_array['height'][index_ht_peak:], data_array['absolute_magnitudes'][index_ht_peak:], 2)
    else:
        # Handle the case of empty input arrays
        a3_Outabs, b3_Outabs, c3_Outabs = 0, 0, 0

    # # check if the ht_obs[:index_ht_peak] and abs_mag_obs[:index_ht_peak] are empty
    # a3_Inabs, b3_Inabs, c3_Inabs = np.polyfit(data_array['height'][:index_ht_peak], data_array['absolute_magnitudes'][:index_ht_peak], 2)

    # # check if the ht_obs[index_ht_peak:] and abs_mag_obs[index_ht_peak:] are empty
    # a3_Outabs, b3_Outabs, c3_Outabs = np.polyfit(data_array['height'][index_ht_peak:], data_array['absolute_magnitudes'][index_ht_peak:], 2)


    ######## RMSD ###############
    # print('fit_funct RMSD mag',fit_funct['rmsd_mag'],' vel',fit_funct['rmsd_vel'], ' lag',fit_funct['rmsd_len'])
    if test_data == []:
        rmsd_lag = 9999
        rmsd_mag = 9999
        chi2_red_mag = 9999
        chi2_red_lag = 9999
    else:
        # Compute the residuals
        rmsd_mag, rmsd_vel, rmsd_lag, _, _, _, _, _, _ = RMSD_calc_diff(data, test_data)
        _, _, res_chi_p, _, _, _, _, _, _ = chiSquare_test_sim_real(data, test_data) 
        chi2_red_mag = res_chi_p['chi2_red_mag']
        chi2_red_lag = res_chi_p['chi2_red_lag']

    # print(data_array['name'],'rmsd_mag',rmsd_mag,'rmsd_vel',rmsd_vel,'rmsd_len',rmsd_lag)

    ################################# 

    

    # Data to populate the dataframe
    data_picklefile_pd = {
        'solution_id': [data_array['name']],
        'type': [data_array['type']],
        'rmsd_mag': [rmsd_mag],
        'rmsd_len': [rmsd_lag],
        'chi2_red_mag': [chi2_red_mag],
        'chi2_red_lag': [chi2_red_lag],
        'vel_init_norot': [data_array['v_init']],
        'vel_avg_norot': [data_array['v_avg']],
        'v_init_180km': [data_array['v_init_180km']],
        'duration': [duration],
        'peak_mag_height': [peak_mag_height],
        'begin_height': [begin_height],
        'end_height': [end_height],
        'peak_abs_mag': [peak_abs_mag],
        'beg_abs_mag': [beg_abs_mag],
        'end_abs_mag': [end_abs_mag],
        'F': [F_param],
        'trail_len': [trail_len],
        't0': [t0],
        'deceleration_lin': [acceleration_lin],
        'deceleration_parab': [acceleration_parab],
        'decel_parab_t0': [acceleration_parab_t0],
        'decel_t0': [decel_t0],
        'decel_jacchia': [acc_jacchia],
        'zenith_angle': [data_array['zenith_angle']],
        'avg_lag': [avg_lag],
        'kc': [kc_par], 
        'Dynamic_pressure_peak_abs_mag': [data_array['Dynamic_pressure_peak_abs_mag']],
        'a_acc': [a3],
        'b_acc': [b3],
        'c_acc': [c3],
        'a_t0': [a_t0],
        'b_t0': [b_t0],
        'c_t0': [c_t0],
        'a1_acc_jac': [jac_a1],
        'a2_acc_jac': [jac_a2],
        'a_mag_init': [a3_Inabs],
        'b_mag_init': [b3_Inabs],
        'c_mag_init': [c3_Inabs],
        'a_mag_end': [a3_Outabs],
        'b_mag_end': [b3_Outabs],
        'c_mag_end': [c3_Outabs],
        'mass': [data_array['mass']],
        'rho': [data_array['rho']],
        'sigma': [data_array['sigma']],
        'erosion_height_start': [data_array['erosion_height_start']],
        'erosion_coeff': [data_array['erosion_coeff']],
        'erosion_mass_index': [data_array['erosion_mass_index']],
        'erosion_mass_min': [data_array['erosion_mass_min']],
        'erosion_mass_max': [data_array['erosion_mass_max']],
        'erosion_range': [data_array['erosion_range']],
        'erosion_energy_per_unit_cross_section': [data_array['erosion_energy_per_unit_cross_section']],
        'erosion_energy_per_unit_mass': [data_array['erosion_energy_per_unit_mass']]
    }

    # Create the dataframe
    panda_dataframe_PCA = pd.DataFrame(data_picklefile_pd)

    if data_array['mass']==0:
        # delete the mass 
        panda_dataframe_PCA = panda_dataframe_PCA.drop(columns=['mass', 'rho', 'sigma', 'erosion_height_start', 'erosion_coeff', 'erosion_mass_index', 'erosion_mass_min', 'erosion_mass_max', 'erosion_range', 'erosion_energy_per_unit_cross_section', 'erosion_energy_per_unit_mass'])

    return panda_dataframe_PCA





########## Utils ##########################

# create a txt file where you save averithing that has been printed
class Logger(object):
    def __init__(self, directory=".", filename="log.txt"):
        self.terminal = sys.stdout
        # Ensure the directory exists
        if not os.path.exists(directory):
            os.makedirs(directory)
        # Combine the directory and filename to create the full path
        filepath = os.path.join(directory, filename)
        self.log = open(filepath, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # This might be necessary as stdout could call flush
        self.terminal.flush()

    def close(self):
        # Close the log file when done
        self.log.close()


# update solution_id directory saved in CSV files
def update_solution_ids(base_dir, new_base_dir):
    # Iterate through all subdirectories
    for root, dirs, files in os.walk(new_base_dir):
        for file in files:
            # Only process CSV files
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                print(f"Processing file: {file_path}")
                
                try:
                    # Load the CSV file as DataFrame
                    df = pd.read_csv(file_path)

                    # Check if 'solution_id' column exists
                    if 'solution_id' in df.columns:
                        # Update each value in 'solution_id'
                        df['solution_id'] = df['solution_id'].apply(
                            lambda x: x.replace(base_dir, new_base_dir) if isinstance(x, str) else x
                        )
                        
                        # Write updated DataFrame back to CSV
                        df.to_csv(file_path, index=False)
                        print(f"Updated solution_id in: {file_path}")

                except Exception as e:
                    print(f"Failed to process file {file_path} due to: {e}")


class SetUpObservationFolders:
    def __init__(self, input_folder, metsim_json):
        """
        Loads the observation data from the given folder or file and MetSim json file.
        The observation data can be in the form of a CSV file, pickle file, or JSON file.

        Parameters:
        input_folder: str - The path to the folder containing the observation data.
        metsim_json: str - JSON file extension for MetSim constants, default '_sim_fit_latest.json'.
        """
        self.input_folder = input_folder
        self.metsim_json = metsim_json
        self.input_folder_file = self._get_observation_files()

    def __repr__(self):
        return f"SetUpObservationFolders({self.input_folder}, {self.metsim_json})"

    def __str__(self):
        return f"SetUpObservationFolders: input folder={self.input_folder}, MetSim json file end={self.metsim_json}"

    def _get_observation_files(self):
        """
        Determines if the input is a directory or a single file and processes accordingly.
        """
        if os.path.isdir(self.input_folder):
            return self._find_trajectory_files(self.input_folder)
        elif os.path.isfile(self.input_folder):
            return self._process_single_file(self.input_folder)
        else:
            print('The provided path or file does not exist')
            sys.exit()

    def _process_single_file(self, filepath):
        """
        Processes a single file, extracts relevant information, and determines the output folder and MetSim file path.
        """
        trajectory_files = [filepath]
        file_name = os.path.splitext(os.path.basename(filepath))[0]
        input_folder = os.path.dirname(filepath)
        output_folder = os.path.splitext(filepath)[0] + '_GenSim'

        # Get the MetSim file path, or create it if the input is a JSON file
        if filepath.endswith('.json'):
            with open(filepath) as json_file:
                const_part = json.load(json_file)['const']
                metsim_path = os.path.join(input_folder, f'{file_name}{self.metsim_json}')
                with open(metsim_path, 'w') as outfile:
                    json.dump(const_part, outfile, indent=4)
            return [[trajectory_files[0], file_name, input_folder, output_folder, metsim_path]]
        else:
            metsim_path = self._get_metsim_file(input_folder, file_name)
            return [[trajectory_files[0], file_name, input_folder, output_folder, metsim_path]]

    def _find_trajectory_files(self, directory):
        """
        Walks through the directory to find and process trajectory files.
        """
        trajectory_files, file_names, input_folders, output_folders, metsim_files = [], [], [], [], []

        for root, _, files in os.walk(directory):
            # Skip folders with the name 'GenSim'
            if 'GenSim' in root:
                continue

            csv_found = False
            # Look for CSV files first
            for file in files:
                if file.endswith('_obs.csv'):
                    csv_found = True
                    self._process_csv_file(root, file, trajectory_files, file_names, input_folders, output_folders, metsim_files)
                    break

            # If no CSV file is found, look for pickle files
            if not csv_found:
                for file in files:
                    if file.endswith('_trajectory.pickle'):
                        self._process_pickle_file(root, file, trajectory_files, file_names, input_folders, output_folders, metsim_files)

        return [[trajectory_files[i], file_names[i], input_folders[i], output_folders[i], metsim_files[i]] for i in range(len(trajectory_files))]

    def _process_csv_file(self, root, file, trajectory_files, file_names, input_folders, output_folders, metsim_files):
        """
        Processes a CSV file to extract relevant information and determine the output folder and MetSim file path.
        """
        real_data = pd.read_csv(os.path.join(root, file))
        solution_id = real_data['solution_id'][0]
        if root not in solution_id:
            print('The solution_id in the CSV file does not match the folder name:', root)
            return

        _, file_from_csv = os.path.split(solution_id)
        base_name = os.path.splitext(file_from_csv)[0]
        variable_name, output_folder_name = self._get_variable_and_output(base_name)

        # Get the MetSim file path, or create it if the input is a JSON file
        metsim_path = self._get_metsim_file(root, variable_name)
        if file_from_csv.endswith('.json'):
            with open(os.path.join(root, file_from_csv)) as json_file:
                const_part = json.load(json_file)['const']
                metsim_path = os.path.join(root, output_folder_name, f'{variable_name}_sim_fit.json')
                os.makedirs(os.path.join(root, output_folder_name), exist_ok=True)
                with open(metsim_path, 'w') as outfile:
                    json.dump(const_part, outfile, indent=4)

        self._add_file_details(root, file, variable_name, output_folder_name, metsim_path, trajectory_files, file_names, input_folders, output_folders, metsim_files)

    def _process_pickle_file(self, root, file, trajectory_files, file_names, input_folders, output_folders, metsim_files):
        """
        Processes a pickle file to extract relevant information and determine the output folder and MetSim file path.
        """
        base_name = os.path.splitext(file)[0]
        variable_name, output_folder_name = self._get_variable_and_output(base_name)
        metsim_path = self._get_metsim_file(root, variable_name)
        self._add_file_details(root, file, variable_name, output_folder_name, metsim_path, trajectory_files, file_names, input_folders, output_folders, metsim_files)

    def _get_variable_and_output(self, base_name):
        """
        Determines the variable name and output folder name based on the base file name.
        """
        if base_name.endswith('_trajectory'):
            variable_name = base_name.replace('_trajectory', '')
            output_folder_name = f'{variable_name}_GenSim'
        else:
            variable_name = base_name
            output_folder_name = f'{base_name}_GenSim'
        return variable_name, output_folder_name

    def _get_metsim_file(self, folder, variable_name):
        """
        Gets the path to the MetSim file, falling back to a default if necessary.
        """
        metsim_path = os.path.join(folder, f'{variable_name}{self.metsim_json}')
        if os.path.isfile(metsim_path):
            return metsim_path
        default_path = os.path.join(folder, f'{variable_name}_sim_fit_latest.json')
        if os.path.isfile(default_path):
            print(f'{variable_name}: No MetSim file with the given extension {self.metsim_json}, reverting to default extension _sim_fit_latest.json')
            return default_path
        print(f'{variable_name}: No MetSim file found, create a first guess.')
        const_nominal = Constants()
        const_dict = const_nominal.to_dict()
        first_guess = os.path.join(folder, f'{variable_name}_first_guess.json')
        with open(first_guess, 'w') as outfile:
            json.dump(const_dict, outfile, indent=4)
        return first_guess

    def _add_file_details(self, root, file, variable_name, output_folder_name, metsim_path, trajectory_files, file_names, input_folders, output_folders, metsim_files):
        """
        Adds the file details to the respective lists if the MetSim file path is valid.
        """
        if metsim_path:
            trajectory_files.append(os.path.join(root, file))
            file_names.append(variable_name)
            input_folders.append(root)
            output_folders.append(os.path.join(root, output_folder_name))
            metsim_files.append(metsim_path)


def update_sigma_values(file_path, mag_sigma, len_sigma, More_complex_fit=False, Custom_refinement=False):
    with open(file_path, 'r') as file:
        content = file.read()
        
    # Modify mag_sigma and len_sigma
    content = re.sub(r'"mag_sigma":\s*[\d.]+', f'"mag_sigma": {mag_sigma}', content)
    content = re.sub(r'"len_sigma":\s*[\d.]+', f'"len_sigma": {len_sigma}', content)
    
    if More_complex_fit:
        # Enable "More complex fit - overall fit"
        content = re.sub(
            r'(# More complex fit - overall fit\s*\{[^{}]*"enabled":\s*)false', 
            r'\1true',
            content
        )
    else:
        # Enable "More complex fit - overall fit"
        content = re.sub(
            r'(# More complex fit - overall fit\s*\{[^{}]*"enabled":\s*)true', 
            r'\1false',
            content
        )
    
    if Custom_refinement:
        # Enable "Custom refinement of erosion parameters - improves wake"
        content = re.sub(
            r'(# Custom refinement of erosion parameters - improves wake\s*\{[^{}]*"enabled":\s*)false', 
            r'\1true',
            content
        )
    else:
        # Enable "Custom refinement of erosion parameters - improves wake"
        content = re.sub(
            r'(# Custom refinement of erosion parameters - improves wake\s*\{[^{}]*"enabled":\s*)true', 
            r'\1false',
            content
        )

    # Save the modified content back to the file
    with open(file_path, 'w') as file:
        file.write(content)
    
    print('modified options file:', file_path)




########## Distance ##########################

# Function to find the knee of the distance plot
def find_knee_dist_index(data_meteor_pd, window_of_smothing_avg=3, std_multip_threshold=1, output_path='', around_meteor='', N_sim_sel_force=0, data_meteor_pd_sel='distance_meteor', find_closest_results=False):
    
    # check if 'distance_meteor' is in the columns of data_for_meteor
    # if 'distance_meteor' in data_meteor_pd.columns:
    #     dist_for_meteor=np.array(data_meteor_pd['distance_meteor'])
    if data_meteor_pd_sel in data_meteor_pd.columns:
        dist_for_meteor=np.array(data_meteor_pd[data_meteor_pd_sel])
    else:
        print('Neither distance_meteor nor',data_meteor_pd_sel,'in the columns')
        return 0

    #make subtraction of the next element and the previous element of data_for_meteor["distance_meteor"]
    # diff_distance_meteor = np.diff(dist_for_meteor[:int(len(dist_for_meteor)/10)])
    diff_distance_meteor = np.diff(dist_for_meteor)

    # histogram plot of the difference with the count on the x axis and diff_distance_meteor on the y axis 
    indices = np.arange(len(diff_distance_meteor))

    # create the cumulative sum of the diff_distance_meteor
    cumsum_diff_distance_meteor = np.cumsum(diff_distance_meteor)
    # check if any diff_distance_meteor is negative if so take the abs
    # if np.any(diff_distance_meteor<0):
    #     # diff_distance_meteor = dist_for_meteor-min(dist_for_meteor)
    #     # # delete the 0 values
    #     # diff_distance_meteor = diff_distance_meteor[diff_distance_meteor!=0]

    #     diff_distance_meteor = np.abs(diff_distance_meteor)

    # normalize the diff_distance_meteor xnormalized = (x - xminimum) / range of x
    diff_distance_meteor_normalized = (diff_distance_meteor - np.min(diff_distance_meteor)) / (np.max(diff_distance_meteor) - np.min(diff_distance_meteor))

    def moving_average_smoothing(data, window_size):
        smoothed_data = np.convolve(data, np.ones(window_size)/window_size, mode='same')
        return smoothed_data

    # apply the smoothing function
    smoothed_diff_distance_meteor = moving_average_smoothing(diff_distance_meteor_normalized, window_of_smothing_avg)
    
    if find_closest_results==False:
        # fid the first value of the smoothed_diff_distance_meteor that is smaller than the std of the smoothed_diff_distance_meteor
        index10percent = np.where(smoothed_diff_distance_meteor < np.std(smoothed_diff_distance_meteor)*std_multip_threshold)[0][0]
    elif find_closest_results==True:
        # oppositely we want a way to get the most close to each other results
        # find the first value of the smoothed_diff_distance_meteor that is above than the std of the smoothed_diff_distance_meteor
        index10percent = np.where(smoothed_diff_distance_meteor > np.std(smoothed_diff_distance_meteor)*std_multip_threshold)[0][0]
        
    # if data_meteor_pd_sel=='distance_meteor':
    #     index10percent = index10percent-2
    # else:
    #     index10percent = index10percent-1
    index10percent = index10percent-1

    if N_sim_sel_force!=0:
        index10percent = N_sim_sel_force

    if index10percent<0: # below does not work problem with finding the mode on KDE later on
        index10percent=0

    if output_path!='':

        # Define a custom palette
        custom_palette_orange = {
            'Real': "darkorange",
            'Simulation': "blue",
            'Simulation_sel': "darkorange",
            'MetSim': "darkorange",
            'Realization': "darkorange",
            'Observation': "darkorange"
        }

        # dimension of the plot 15,5
        plt.figure(figsize=(15,5))
        if data_meteor_pd_sel!='distance_meteor':
            data_meteor_pd['type'] = 'Simulation'

        plt.subplot(1,2,2)
        sns.histplot(data_meteor_pd, x=data_meteor_pd_sel, hue="type", kde=True, cumulative=True, bins=len(dist_for_meteor), palette=custom_palette_orange) # , stat='density' to have probability        
        if data_meteor_pd_sel=='distance_meteor':
            plt.title('Cumulative distance in PCA space')
            plt.xlabel('Distance in PCA space')
            plt.axvline(x=(dist_for_meteor[index10percent]), color="darkorange", linestyle='--', label='Knee distance')
        else:
            plt.title('Cumulative '+data_meteor_pd_sel)
            plt.xlabel(data_meteor_pd_sel)
            plt.axvline(x=(dist_for_meteor[index10percent]), color="blue", linestyle='--', label='Knee distance')
        plt.ylabel('Number of events')            

        if len(dist_for_meteor)>100:
            plt.ylim(0,100) 
        elif len(dist_for_meteor)>50:
            plt.ylim(0,50)
        
        plt.legend()
        # delete the legend
        plt.legend().remove()


        plt.subplot(1,2,1)
        # sns.histplot(diff_distance_meteor_normalized, kde=True, bins=len(distance_meteor_sel_save))
        #make the bar plot 0.5 transparency
        if data_meteor_pd_sel=='distance_meteor':
            plt.bar(indices, diff_distance_meteor_normalized,color="darkorange", alpha=0.5, edgecolor='black')
            plt.title('Distance difference Normalized')
            # put a horizontal line at len(curr_sel['distance_meteor'])
            plt.axvline(x=index10percent, color="darkorange", linestyle='--') 
        else:
            plt.bar(indices, diff_distance_meteor_normalized,color="blue", alpha=0.5, edgecolor='black')
            plt.title(data_meteor_pd_sel+' Normalized from the mean of the selected shower')
            # put a horizontal line at len(curr_sel['distance_meteor'])
            plt.axvline(x=index10percent, color="blue", linestyle='--') 
        plt.xlabel('Number of events')
        plt.ylabel('Normalized difference')

        if len(dist_for_meteor)>100:
            plt.xlim(-1,100) 
        elif len(dist_for_meteor)>50:
            plt.xlim(-1,50)

        # find the mean of the first 100 elements of diff_distance_meteor_normalized and put a horizontal line
        # plt.axhline(y=np.std(smoothed_diff_distance_meteor), color="darkorange", linestyle='--')

        # set a sup title
        plt.suptitle(around_meteor)

        # give more space
        plt.tight_layout()  
        # plt.show()
        if data_meteor_pd_sel == 'distance_meteor':
            plt.savefig(output_path+os.sep+around_meteor+os.sep+around_meteor+'_knee'+str(index10percent+1)+'ev_MAXdist'+str(np.round(dist_for_meteor[index10percent],2))+'.png', dpi=300)
        else:
            # save the figure maximized and with the right name
            plt.savefig(output_path+os.sep+around_meteor+'_knee'+str(index10percent+1)+'ev_MAXdist'+str(np.round(dist_for_meteor[index10percent],2))+'.png', dpi=300)

        # close the figure
        plt.close()

    return index10percent

# function to use the mahaloby distance and from the mean of the selected shower
def dist_PCA_space_select_sim(df_sim_PCA, shower_current_PCA_single, cov_inv, meanPCA_current, df_sim_shower, shower_current_single, N_sim_sel_force=0, output_dir=''):
    N_sim_sel_all=100
    print('calculate distance for',shower_current_single['solution_id'])

    df_sim_PCA_for_now = df_sim_PCA.drop(['type'], axis=1).values

    distance_current = []
    for i_sim in range(len(df_sim_PCA_for_now)):
        distance_current.append(mahalanobis_distance(df_sim_PCA_for_now[i_sim], shower_current_PCA_single, cov_inv))

    # create an array with lenght equal to the number of simulations and set it to shower_current_PCA['solution_id'][i_shower]
    solution_id_dist = [shower_current_single['solution_id']] * len(df_sim_PCA_for_now)
    df_sim_shower['solution_id_dist'] = solution_id_dist
    df_sim_shower['distance_meteor'] = distance_current
    # sort the distance and select the n_selected closest to the meteor
    df_sim_shower_dis = df_sim_shower.sort_values(by=['distance_meteor']).reset_index(drop=True)
    df_sim_selected = df_sim_shower_dis[:N_sim_sel_all].drop(['type'], axis=1)
    df_sim_selected['type'] = 'Simulation_sel'

    # create a dataframe with the selected simulated shower characteristics
    df_sim_PCA_dist = df_sim_PCA
    df_sim_PCA_dist['distance_meteor'] = distance_current
    df_sim_PCA_dist = df_sim_PCA_dist.sort_values(by=['distance_meteor']).reset_index(drop=True)
    # delete the shower code
    df_sim_selected_PCA = df_sim_PCA_dist[:N_sim_sel_all].drop(['type','distance_meteor'], axis=1)

    # make df_sim_selected_PCA an array
    df_sim_selected_PCA = df_sim_selected_PCA.values
    distance_current_mean = []
    for i_shower in range(len(df_sim_selected)):
        distance_current_mean.append(scipy.spatial.distance.euclidean(meanPCA_current, df_sim_selected_PCA[i_shower]))
    df_sim_selected['distance_mean']=distance_current_mean # from the mean of the selected shower

    df_curr_sel_curv = df_sim_selected.copy()

    around_meteor=shower_current_single['solution_id']
    # check if around_meteor is a file in a folder
    if os.path.exists(around_meteor):
        # split in file and directory
        _, around_meteor = os.path.split(around_meteor)
        around_meteor = around_meteor[:15]

    mkdirP(output_dir+os.sep+around_meteor)
    window_of_smothing_avg=3
    std_multip_threshold=1
    if N_sim_sel_force!=0:
        print(around_meteor,'select the best',N_sim_sel_force,'simulations')
        dist_to_cut=find_knee_dist_index(df_curr_sel_curv,window_of_smothing_avg,std_multip_threshold, output_dir, around_meteor, N_sim_sel_force)
        # change of curvature print
        df_curr_sel_curv=df_curr_sel_curv.iloc[:dist_to_cut]
    else:
        dist_to_cut=find_knee_dist_index(df_curr_sel_curv,window_of_smothing_avg,std_multip_threshold, output_dir, around_meteor)
        print(around_meteor,'index of the knee distance',dist_to_cut+1)
        # change of curvature print
        df_curr_sel_curv=df_curr_sel_curv.iloc[:dist_to_cut+1]

    return df_sim_selected, df_curr_sel_curv





#### Matrix function ############################################################################

# Function to perform Varimax rotation
def varimax(Phi, gamma=1.0, q=20, tol=1e-6):
    p, k = Phi.shape
    R = np.eye(k)
    d = 0
    for i in range(q):
        d_old = d
        Lambda = np.dot(Phi, R)
        u, s, vh = svd(np.dot(Phi.T, np.asarray(Lambda) ** 3 - (gamma / p) * np.dot(Lambda, np.diag(np.diag(np.dot(Lambda.T, Lambda))))))
        R = np.dot(u, vh)
        d = np.sum(s)
        if d_old != 0 and d / d_old < 1 + tol:
            break
    return np.dot(Phi, R)

# Function to perform mahalanobis distance
def mahalanobis_distance(x, mean, cov_inv):
    diff = x - mean
    return np.sqrt(np.dot(np.dot(diff, cov_inv), diff.T))



def process_pca_variables(variable_PCA, No_var_PCA, df_obs_shower, df_sim_shower, OUT_PUT_PATH, file_name_obs, PCA_pairplot=False):
    # if variable_PCA is not empty
    if variable_PCA != []:
        # add to variable_PCA array 'type','solution_id'
        variable_PCA = ['solution_id','type'] + variable_PCA
        if No_var_PCA != []:
            # remove from variable_PCA the variables in No_var_PCA
            for var in No_var_PCA:
                variable_PCA.remove(var)

    else:
        # put in variable_PCA all the variables except mass
        variable_PCA = list(df_obs_shower.columns)
        # check if mass is in the variable_PCA
        if 'mass' in variable_PCA:
            # remove mass from variable_PCA
            variable_PCA.remove('mass')
        # if No_var_PCA is not empty
        if No_var_PCA != []:
            # remove from variable_PCA the variables in No_var_PCA
            for var in No_var_PCA:
                # check if the variable is in the variable_PCA
                if var in variable_PCA:
                    variable_PCA.remove(var)

    scaled_sim=df_sim_shower[variable_PCA].copy()
    scaled_sim=scaled_sim.drop(['type','solution_id'], axis=1)

    print(len(scaled_sim.columns),'Variables :\n',scaled_sim.columns)

    # Standardize each column separately
    scaler = StandardScaler()
    df_sim_var_sel_standardized = scaler.fit_transform(scaled_sim)
    df_sim_var_sel_standardized = pd.DataFrame(df_sim_var_sel_standardized, columns=scaled_sim.columns)

    # Identify outliers using Z-score method on standardized data
    z_scores = np.abs(zscore(df_sim_var_sel_standardized))
    threshold = 3
    outliers = (z_scores > threshold).any(axis=1)

    # outlier number 0 has alway to be the False
    if outliers[0]==True:
        print('The MetSim reduction is an outlier') # , still keep it for the PCA analysis
        outliers[0]=False

    # Assign df_sim_shower to the version without outliers
    df_sim_shower = df_sim_shower[~outliers].copy()


    if PCA_pairplot:
        # Mapping of original variable names to LaTeX-style labels
        variable_map = {
            'vel_init_norot': r"$v_i$ [km/s]",
            'vel_avg_norot': r"$v_{avg}[km/s]$",
            'v_init_180km': r"$v_{180km}$ [km/s]",
            'chi2_red_mag': r"$mag\chi^2_{red}$",
            'chi2_red_lag': r"$lag\chi^2_{red}$",
            'duration': r"$t$ [s]",
            'peak_mag_height': r"$h_{p}$ [km]",
            'begin_height': r"$h_{beg}$ [km]",
            'end_height': r"$h_{end}$ [km]",
            'peak_abs_mag': r"$M_{p}$",
            'beg_abs_mag': r"$M_{beg}$",
            'end_abs_mag': r"$M_{end}$",
            'F': r"$F$",
            'trail_len': r"$L$ [km]",
            't0': r"$t_0$ [s]",
            'deceleration_lin': r"$dAcc_{lin}$ [m/s$^2$]",
            'deceleration_parab': r"$dAcc_{par}$ [m/s$^2$]",
            'decel_parab_t0': r"$dAcc_{p_{t_0}}$ [m/s$^2$]",
            'decel_t0': r"$dAcc_{p1_{t_0}}$ [m/s$^2$]",
            'decel_jacchia': r"$dAcc_{jac}$ [m/s$^2$]",
            'zenith_angle': r"$\zeta$ [deg]",
            'avg_lag': r"$lag_{avg}$ [km]",
            'kc': r"$k_c$",
            'Dynamic_pressure_peak_abs_mag': r"$P_p$ [Pa]",
            'a_mag_init': r"$Mfit_{a_{int}}$",
            'b_mag_init': r"$Mfit_{b_{int}}$",
            'a_mag_end': r"$Mfit_{a_{fin}}$",
            'b_mag_end': r"$Mfit_{b_{fin}}$"
        }
        
        # Convert variable names to LaTeX-style labels
        latex_labels = [variable_map.get(var, var) for var in variable_PCA[2:]]
        
        # Prepare data for plotting
        df_sim_var_sel = df_sim_shower[variable_PCA].copy().drop(['type', 'solution_id'], axis=1)
        
        # Sample 10,000 events if the dataset is large
        if len(df_sim_var_sel) > 10000:
            print('Number of events in the simulated:', len(df_sim_var_sel))
            df_sim_var_sel = df_sim_var_sel.sample(n=10000)

        # Setup the plot grid
        fig, axs = plt.subplots(int(np.ceil(len(latex_labels) / 5)), 5, figsize=(20, 15))
        axs = axs.flatten()

        for i, (var, label) in enumerate(zip(variable_PCA[2:], latex_labels)):
            if var == 'v_init_180km' or var == 'trail_len':
                sns.histplot(df_sim_var_sel[var]/1000, kde=True, ax=axs[i], color='b', alpha=0.5, bins=20)
                axs[i].axvline(df_obs_shower[var].values[0]/1000, color='limegreen', linestyle='--', linewidth=5)
            else:
                sns.histplot(df_sim_var_sel[var], kde=True, ax=axs[i], color='b', alpha=0.5, bins=20)
                axs[i].axvline(df_obs_shower[var].values[0], color='limegreen', linestyle='--', linewidth=5)
            axs[i].set_xlabel(label)
            if i % 5 != 0:
                axs[i].set_ylabel('N.events')

        # Remove unused subplots
        for i in range(len(latex_labels), len(axs)):
            fig.delaxes(axs[i])

        plt.tight_layout()
        
        # Save and close the figure
        plt.savefig(os.path.join(OUT_PUT_PATH, f"{file_name_obs}_var_hist_real.png"))
        plt.close()

    return df_sim_shower, variable_PCA, outliers




# PCA ####################################################################################

def PCASim(df_sim_shower, df_obs_shower, OUT_PUT_PATH, PCA_percent=99, N_sim_sel=0, variable_PCA=[], No_var_PCA=['rmsd_mag', 'rmsd_len', 'avg_lag','a1_acc_jac','a2_acc_jac','a_acc','b_acc','c_acc','c_mag_init','c_mag_end','a_t0', 'b_t0', 'c_t0'], file_name_obs='', cores_parallel=None, PCA_pairplot=False, esclude_real_solution_from_selection=False):
    '''
    This function generate the simulated shower from the erosion model and apply PCA.
    The function read the json file in the folder and create a csv file with the simulated shower and take the data from GenerateSimulation.py folder.
    The function return the dataframe of the selected simulated shower.

    'solution_id','type','vel_init_norot','vel_avg_norot','duration',
    'mass','peak_mag_height','begin_height','end_height','t0','peak_abs_mag','beg_abs_mag','end_abs_mag',
    'F','trail_len','deceleration_lin','deceleration_parab','decel_jacchia','decel_t0','zenith_angle', 
    'kc','Dynamic_pressure_peak_abs_mag',
    'a_acc','b_acc','c_acc','a1_acc_jac','a2_acc_jac','a_mag_init','b_mag_init','c_mag_init','a_mag_end','b_mag_end','c_mag_end',
    'rho','sigma','erosion_height_start','erosion_coeff', 'erosion_mass_index',
    'erosion_mass_min','erosion_mass_max','erosion_range',
    'erosion_energy_per_unit_cross_section', 'erosion_energy_per_unit_mass'

    '''


    df_sim_shower, variable_PCA, outliers = process_pca_variables(variable_PCA, No_var_PCA, df_obs_shower, df_sim_shower, OUT_PUT_PATH, file_name_obs, False)

    variable_PCA_initial = variable_PCA.copy()

    ##################################### delete var that are not in the 5 and 95 percentile of the simulated shower #####################################

    # check if a file with the name "log"+n_PC_in_PCA+"_"+str(len(df_sel))+"ev.txt" already exist
    if os.path.exists(OUT_PUT_PATH+os.sep+"log_"+file_name_obs[:15]+"_"+str(len(variable_PCA)-2)+"var_"+str(PCA_percent)+"%.txt"):
        # remove the file
        os.remove(OUT_PUT_PATH+os.sep+"log_"+file_name_obs[:15]+"_"+str(len(variable_PCA)-2)+"var_"+str(PCA_percent)+"%.txt")
    sys.stdout = Logger(OUT_PUT_PATH,"log_"+file_name_obs[:15]+"_"+str(len(variable_PCA)-2)+"var_"+str(PCA_percent)+"%.txt") # _30var_99%_13PC

    df_all = pd.concat([df_sim_shower[variable_PCA],df_obs_shower[variable_PCA]], axis=0, ignore_index=True)
    # delete nan
    df_all = df_all.dropna()

    # create a copy of df_sim_shower for the resampling
    df_sim_shower_resample=df_sim_shower.copy()
    # df_obs_shower_resample=df_obs_shower.copy()
    No_var_PCA_perc=[]
    # check that all the df_obs_shower for variable_PCA is within th 5 and 95 percentie of df_sim_shower of variable_PCA
    for var in variable_PCA:
        if var != 'type' and var != 'solution_id':
            # check if the variable is in the df_obs_shower
            if var in df_obs_shower.columns:
                # check if the variable is in the df_sim_shower
                if var in df_sim_shower.columns:

                    print(var)

                    ii_all=0
                    for i_var in range(len(df_obs_shower[var])):
                        # check if all the values are outside the 5 and 95 percentile of the df_sim_shower if so delete the variable from the variable_PCA
                        if df_obs_shower[var][i_var] < np.percentile(df_sim_shower[var], 1) or df_obs_shower[var][i_var] > np.percentile(df_sim_shower[var], 99):
                            ii_all=ii_all+1

                    if ii_all==len(df_obs_shower[var]):
                        print('The observed and all realization',var,'are not within the 1 and 99 percentile of the simulated meteors')
                        print('The variable',var,'is deleted from the PCA analysis!!!')

                        # delete the variable from the variable_PCA
                        variable_PCA.remove(var)
                        # save the var deleted in a variable
                        No_var_PCA_perc.append(var)

                        df_all = df_all.drop(var, axis=1)
                    else:
                        shapiro_test = stats.shapiro(df_all[var])
                        print("Initial Shapiro-Wilk Test:", shapiro_test.statistic,"p-val", shapiro_test.pvalue)

                        if var=='v_init_180km':
                            # # do the cosine of the zenith angle
                            # df_all[var]=np.cos(np.radians(df_all[var]))
                            # # df_all[var]=transform_to_gaussian(df_all[var])
                            # df_sim_shower_resample[var]=np.cos(np.radians(df_sim_shower_resample[var]))
                            print('Variable ',var,' is not transformed')

                        if var=='zenith_angle':
                            # # do the cosine of the zenith angle
                            # df_all[var]=np.cos(np.radians(df_all[var]))
                            # # df_all[var]=transform_to_gaussian(df_all[var])
                            # df_sim_shower_resample[var]=np.cos(np.radians(df_sim_shower_resample[var]))
                            print('Variable ',var,' is not transformed')

                        elif var=='vel_init_norot':
                            # do the cosine of the zenith angle
                            # df_all[var]=transform_to_gaussian(df_all[var])
                            print('Variable ',var,' is not transformed')

                        else:

                            pt = PowerTransformer(method='yeo-johnson')
                            df_all[var]=pt.fit_transform(df_all[[var]])
                            df_sim_shower_resample[var]=pt.fit_transform(df_sim_shower_resample[[var]])

                        shapiro_test = stats.shapiro(df_all[var])
                        print("NEW Shapiro-Wilk Test:", shapiro_test.statistic,"p-val", shapiro_test.pvalue)
                    
                    print()

                else:
                    print('Variable ',var,' is not in the simulated shower')
            else:
                print('Variable ',var,' is not in the observed shower')

    # check if the variable_PCA is empty FAILSAFE
    if variable_PCA == []:
        print('All the variables are not within the 1 and 99 percentile of the simulated meteors!!!')
        # add the variable_PCA_initial
        variable_PCA=variable_PCA_initial


    # if PCA_pairplot:
    df_all_nameless_plot=df_all.copy()

    # Store the values for vertical lines before sampling
    vertical_line_values = {}
    for var in variable_PCA[2:]:
        vertical_line_values[var] = df_all_nameless_plot[var].values[len(df_sim_shower[variable_PCA])]


    if len(df_all_nameless_plot)>10000:
        # pick randomly 10000 events
        print('Number of events in the simulated:',len(df_all_nameless_plot))
        df_all_nameless_plot=df_all_nameless_plot.sample(n=10000)
        # add the last len(df_sim_shower[variable_PCA])

    # make a subplot of the rho againist each variable_PCA as a scatter plot
    fig, axs = plt.subplots(int(np.ceil(len(variable_PCA[2:])/5)), 5, figsize=(20, 15))
    # flat it
    axs = axs.flatten()
    for i, var in enumerate(variable_PCA[2:]):
        # plot the distribution of the variable
        sns.histplot(df_all_nameless_plot[var].values[:len(df_sim_shower[variable_PCA])], kde=True, ax=axs[i], color='b', alpha=0.5, bins=20)
        # axs[i//4, i%4].set_title('Distribution of '+var)
        # put a vertical line for the df_obs_shower[var] value
        # print(df_all_nameless_plot['solution_id'].values[len(df_sim_shower[variable_PCA])])
        axs[i].axvline(vertical_line_values[var], color='limegreen', linestyle='--', linewidth=5)      
        # x axis
        axs[i].set_xlabel(var)
        # # grid
        # axs[i//5, i%5].grid()
        if i != 0 and i != 5 and i != 10 and i != 15 and i != 20:
            # delete the y axis
            axs[i].set_ylabel('')
    
    # delete the plot that are not used
    for i in range(len(variable_PCA[2:]), len(axs)):
        fig.delaxes(axs[i])

    # space between the subplots
    plt.tight_layout()

    # save the figure
    plt.savefig(OUT_PUT_PATH+os.sep+file_name_obs+'_var_hist_yeo-johnson.png')
    # close the figure
    plt.close()

    ####################################################################################################################

    # Now we have all the data and we apply PCA to the dataframe
    df_all_nameless=df_all.drop(['type','solution_id'], axis=1)

    # print the data columns names
    df_all_columns_names=(df_all_nameless.columns)

    # Separating out the features
    scaled_df_all = df_all_nameless[df_all_columns_names].values

    # performing preprocessing part so to make it readeble for PCA
    scaled_df_all = StandardScaler().fit_transform(scaled_df_all)


    #################################
    # Applying PCA function on the data for the number of components
    pca = PCA(PCA_percent/100) #PCA_percent
    # pca = PCA() #PCA_percent
    all_PCA = pca.fit_transform(scaled_df_all) # fit the data and transform it

    #count the number of PC
    print('Number of PC:',pca.n_components_)

    ################################# Apply Varimax rotation ####################################
    loadings = pca.components_.T

    rotated_loadings = varimax(loadings)

    # # chage the loadings to the rotated loadings in the pca components
    pca.components_ = rotated_loadings.T

    # Transform the original PCA scores with the rotated loadings ugly PC space but same results
    # all_PCA = np.dot(all_PCA, rotated_loadings.T[:pca.n_components_, :pca.n_components_])

    ############### PCR ########################################################################################

    # Example limits for the physical variables (adjust these based on your domain knowledge)
    limits = {
        'mass': (np.min(df_sim_shower['mass']), np.max(df_sim_shower['mass'])),  # Example limits
        'rho': (np.min(df_sim_shower['rho']), np.max(df_sim_shower['rho'])),
        'sigma': (np.min(df_sim_shower['sigma']), np.max(df_sim_shower['sigma'])),
        'erosion_height_start': (np.min(df_sim_shower['erosion_height_start']), np.max(df_sim_shower['erosion_height_start'])),
        'erosion_coeff': (np.min(df_sim_shower['erosion_coeff']), np.max(df_sim_shower['erosion_coeff'])),
        'erosion_mass_index': (np.min(df_sim_shower['erosion_mass_index']), np.max(df_sim_shower['erosion_mass_index'])),
        'erosion_mass_min': (np.min(df_sim_shower['erosion_mass_min']), np.max(df_sim_shower['erosion_mass_min'])),
        'erosion_mass_max': (np.min(df_sim_shower['erosion_mass_max']), np.max(df_sim_shower['erosion_mass_max']))
    }

    exclude_columns = ['type', 'solution_id']
    physical_vars = ['mass','rho','sigma','erosion_height_start','erosion_coeff','erosion_mass_index','erosion_mass_min','erosion_mass_max'] #, 'erosion_range', 'erosion_energy_per_unit_cross_section', 'erosion_energy_per_unit_mass'

    # Delete specific columns from variable_PCA
    variable_PCA_no_info = [col for col in variable_PCA if col not in exclude_columns]

    # # Scale the data
    # scaled_sim = pd.DataFrame(scaler.fit_transform(df_sim_shower[variable_PCA_no_info + physical_vars]), columns=variable_PCA_no_info + physical_vars)

    # Define X and y (now y contains only the PCA observable parameters)
    X = df_sim_shower_resample[variable_PCA_no_info]
    y = df_sim_shower_resample[physical_vars]

    # Split the data into training and testing sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)

    # Loop over the number of principal components
    print("PCR Predictions with "+str(pca.n_components_)+"PC :")

    pca_copy=copy.deepcopy(pca)
    # PCR: Principal Component Regression inform that the predicted variable is always positive
    pcr = make_pipeline(StandardScaler(), pca_copy, LinearRegression())

    pcr.fit(X_train, y_train)
    # Predict using the models
    y_pred_pcr = pcr.predict(df_sim_shower_resample[variable_PCA_no_info])
    # to_plot_unit=['mass [kg]','rho [kg/m^3]','sigma [s$^2$/km$^2$]','erosion height start [km]','erosion coeff [s$^2$/km$^2$]','erosion mass index [-]','eros. mass min [kg]','eros. mass max [kg]']
    to_plot_unit = [r'$m_0$ [kg]', r'$\rho$ [kg/m$^3$]', r'$\sigma$ [s$^2$/km$^2$]', r'$h_{e}$ [km]', r'$\eta$ [s$^2$/km$^2$]', r'$s$ [-]', r'$m_{l}$ [kg]', r'$m_{u}$ [kg]'] #,r'log($m_{u}$)-log($m_{l}$) [-]']
    # multiply y_pred_pcr that has the 'erosion_coeff'*1000000 and 'sigma'*1000000
    y_pred_pcr[:,4]=y_pred_pcr[:,4]*1000000
    y_pred_pcr[:,2]=y_pred_pcr[:,2]*1000000
    # Get the real values
    real_values = df_sim_shower_resample[physical_vars].iloc[0].values
    # multiply the real_values
    real_values[4]=real_values[4]*1000000
    real_values[2]=real_values[2]*1000000

    # # Apply limits to the predictions
    # for i, var in enumerate(physical_vars):
    #     y_pred_pcr[:, i] = np.clip(y_pred_pcr[:, i], limits[var][0], limits[var][1])

    # Print the predictions alongside the real values
    print("Predicted vs Real Values:")
            # print(output_dir+os.sep+'PhysicProp'+n_PC_in_PCA+'_'+str(len(curr_sel))+'ev_dist'+str(np.round(np.min(curr_sel['distance_meteor']),2))+'-'+str(np.round(np.max(curr_sel['distance_meteor']),2))+'.png')
    for i, unit in enumerate(to_plot_unit):
        y_pred_pcr[0, i]= abs(y_pred_pcr[0, i])
        print(f'{unit}: Predicted: {y_pred_pcr[0, i]:.4g}, Real: {real_values[i]:.4g}')

    pcr_results_physical_param = y_pred_pcr.copy()
    print('--------------------------')

    ############### PCR ########################################################################################


    # # select only the column with in columns_PC with the same number of n_components
    columns_PC = ['PC' + str(x) for x in range(1, pca.n_components_+1)]

    # create a dataframe with the PCA space
    df_all_PCA = pd.DataFrame(data = all_PCA, columns = columns_PC)

    ### plot var explained by each PC bar

    percent_variance = np.round(pca.explained_variance_ratio_* 100, decimals =2)

    # plot the explained variance ratio of each principal componenets base on the number of column of the original dimension
    plt.bar(x= range(1,len(percent_variance)+1), height=percent_variance, tick_label=columns_PC, color='black')
    # ad text at the top of the bar with the percentage of variance explained
    for i in range(1,len(percent_variance)+1):
        # reduce text size
        plt.text(i, percent_variance[i-1], str(percent_variance[i-1])+'%', ha='center', va='bottom', fontsize=5)

    plt.ylabel('Percentance of Variance Explained')
    plt.xlabel('Principal Component')
    # save the figure
    plt.savefig(OUT_PUT_PATH+os.sep+file_name_obs+'PCAexplained_variance_ratio_'+str(len(variable_PCA)-2)+'var_'+str(PCA_percent)+'%_'+str(pca.n_components_)+'PC.png')
    # close the figure
    plt.close()
    # plt.show()

    ### plot covariance matrix

    # make the image big as the screen
    # plt.figure(figsize=(20, 20))

    # Compute the correlation coefficients
    # cov_data = pca.components_.T
    # varimax rotation
    cov_data = rotated_loadings

    # Plot the correlation matrix
    img = plt.matshow(cov_data.T, cmap=plt.cm.coolwarm, vmin=-1, vmax=1)
    plt.colorbar(img)

    # Mapping of original variable names to LaTeX-style labels
    variable_map = {
        'vel_init_norot': r"$v_i$",
        'vel_avg_norot': r"$v_{avg}$",
        'v_init_180km': r"$v_{180km}$",
        'duration': r"$t$",
        'peak_mag_height': r"$h_{p}$",
        'begin_height': r"$h_{beg}$",
        'end_height': r"$h_{end}$",
        'peak_abs_mag': r"$M_{p}$",
        'beg_abs_mag': r"$M_{beg}$",
        'end_abs_mag': r"$M_{end}$",
        'F': r"$F$",
        'trail_len': r"$L$",
        't0': r"$t_0$",
        'deceleration_lin': r"$dAcc_{lin}$",
        'deceleration_parab': r"$dAcc_{par}$",
        'decel_parab_t0': r"$dAcc_{p_{t_0}}$",
        'decel_t0': r"$dAcc_{p1_{t_0}}$",
        'decel_jacchia': r"$dAcc_{jac}$",
        'zenith_angle': r"$\zeta$",
        'avg_lag': r"$lag_{avg}$",
        'kc': r"$k_c$",
        'Dynamic_pressure_peak_abs_mag': r"$P_p$",
        'a_mag_init': r"$Mfit_{a_{int}}$",
        'b_mag_init': r"$Mfit_{b_{int}}$",
        'a_mag_end': r"$Mfit_{a_{fin}}$",
        'b_mag_end': r"$Mfit_{b_{fin}}$"
    }
    
    # Convert the given array to LaTeX-style labels
    latex_labels = [variable_map.get(var, var) for var in variable_PCA]

    rows_8 = [x for x in latex_labels]

    # add to the columns the PC number the percent_variance
    columns_PC_with_var = ['PC' + str(x) + ' (' + str(percent_variance[x-1]) + '%)' for x in range(1, pca.n_components_+1)]

    # Add the variable names as labels on the x-axis and y-axis
    plt.xticks(range(len(rows_8)-2), rows_8[2:], rotation=90)
    # yticks with variance explained
    plt.yticks(range(len(columns_PC_with_var)), columns_PC_with_var)

    # plot the influence of each component on the original dimension
    for i in range(cov_data.shape[0]):
        for j in range(cov_data.shape[1]):
            plt.text(i, j, "{:.1f}".format(cov_data[i, j]), size=5, color='black', ha="center", va="center")   
    # save the figure
    plt.savefig(OUT_PUT_PATH+os.sep+file_name_obs+'PCAcovariance_matrix_'+str(len(variable_PCA)-2)+'var_'+str(PCA_percent)+'%_'+str(pca.n_components_)+'PC.png')
    # close the figure
    plt.close()
    # plt.show()
    ###

    # print the number of simulation selected
    print('PCA run for', len(df_sim_shower),'simulations, delete ',len(outliers)-len(df_sim_shower),' outliers')

    # if len(No_var_PCA_perc) > 0:
    #     for No_var_PCA_perc in No_var_PCA_perc:
    #         print('Observable data variable [',No_var_PCA_perc,'] is not within the 5 and 95 percentile of the simulated shower')

    # print the name of the variables used in PCA
    print('Variables used in PCA: ',df_all_nameless.columns)

    print("explained variance ratio: \n",percent_variance)

    print(str(len(variable_PCA)-2)+' var = '+str(PCA_percent)+'% of the variance explained by ',pca.n_components_,' PC')


    # add the shower code to the dataframe
    df_all_PCA['type'] = df_all['type'].values

    # delete the lines after len(df_sim_shower) to have only the simulated shower
    df_sim_PCA = df_all_PCA.drop(df_all_PCA.index[len(df_sim_shower):])
    df_obs_PCA = df_all_PCA.drop(df_all_PCA.index[:len(df_sim_shower)])

    
    ########### Distance metric takes in to account varinace explained ####################################################################

    if esclude_real_solution_from_selection:
        df_all_PCA_cov = df_all_PCA[df_all_PCA['type'] != 'Real'].copy()
    else:
        # delete the type Real from
        df_all_PCA_cov = df_all_PCA.copy()

    # Get explained variances of principal components
    explained_variance = pca.explained_variance_ratio_

    # Calculate mean and inverse covariance matrix for Mahalanobis distance
    cov_matrix = df_all_PCA_cov.drop(['type'], axis=1).cov()

    # Modify covariance matrix based on explained variances
    for i in range(len(explained_variance)):
        cov_matrix.iloc[i, :] /= explained_variance[i]

    # # Modify covariance matrix to positively reflect variance explained
    # for i in range(len(explained_variance)):
    #     cov_matrix.iloc[i, :] *= explained_variance[i]

    cov_inv = inv(cov_matrix)

    ############## SELECTION ###############################################

    # group them by Observation, Realization type and the other group by MetSim, Simulation
    # meanPCA = df_all_PCA.groupby('type').mean() # does not work

    df_all_PCA['solution_id'] = df_all['solution_id']
    # Create a new column to group by broader categories
    group_mapping = {
        'Observation': 'obs',
        'Realization': 'obs',
        'Real': 'sim',
        'MetSim': 'sim',
        'Simulation': 'sim'
    }
    df_all_PCA['group'] = df_all_PCA['type'].map(group_mapping)
    df_obs_shower['group'] = df_obs_shower['type'].map(group_mapping)
    df_obs_PCA['group'] = df_obs_PCA['type'].map(group_mapping)

    # # Group by the new column and calculate the mean
    # meanPCA = df_all_PCA.groupby('group').mean()

    # # drop the sim column
    # meanPCA = meanPCA.drop(['sim'], axis=0)

    # Ensure that only numeric columns are used in the mean calculation
    df_numeric = df_all_PCA.select_dtypes(include=[np.number])

    # Group by the new column and calculate the mean only for numeric columns
    meanPCA = df_numeric.groupby(df_all_PCA['group']).mean()

    # Drop the 'sim' row if it exists
    meanPCA = meanPCA.drop(['sim'], axis=0, errors='ignore')

    # print(meanPCA)

    meanPCA_current = meanPCA.loc[(meanPCA.index == 'obs')].values.flatten()
    # take only the value of the mean of the first row
    shower_current = df_obs_shower[df_obs_shower['group'] == 'obs']
    shower_current_PCA = df_obs_PCA[df_obs_PCA['group'] == 'obs']

    # trasform the dataframe in an array
    shower_current_PCA = shower_current_PCA.drop(['type','group'], axis=1).values

    # define the distance
    mkdirP(OUT_PUT_PATH+os.sep+SAVE_SELECTION_FOLDER)      
    if esclude_real_solution_from_selection:
        # delete the type Real from
        input_list_obs_dist = [[df_sim_PCA[df_sim_PCA['type'] != 'Real'], shower_current_PCA[ii], cov_inv, meanPCA_current, df_sim_shower[df_sim_shower['type'] != 'Real'], shower_current.iloc[ii], N_sim_sel, OUT_PUT_PATH+os.sep+SAVE_SELECTION_FOLDER] for ii in range(len(shower_current))]
        df_sim_selected_both_df = domainParallelizer(input_list_obs_dist, dist_PCA_space_select_sim, cores=cores_parallel)

    else:  
        input_list_obs_dist = [[df_sim_PCA, shower_current_PCA[ii], cov_inv, meanPCA_current, df_sim_shower, shower_current.iloc[ii], N_sim_sel, OUT_PUT_PATH+os.sep+SAVE_SELECTION_FOLDER] for ii in range(len(shower_current))]
        df_sim_selected_both_df = domainParallelizer(input_list_obs_dist, dist_PCA_space_select_sim, cores=cores_parallel)


    # separet df_sim_selected the '<class 'tuple'>' to a list of dataframe called df_sim_selected_all and df_sim_selected_knee
    df_sim_selected_all = []
    df_sim_selected_knee = []
    for item in df_sim_selected_both_df:
        if isinstance(item, tuple):
            df_sim_selected_all.append(item[0])
            df_sim_selected_knee.append(item[1])

    df_sim_selected_all = pd.concat(df_sim_selected_all)
    df_sel_shower = pd.concat(df_sim_selected_knee)

    # DELETE ALL INDEX

    # Insert the column at the first position
    df_sim_selected_all.insert(1, 'distance_mean', df_sim_selected_all.pop('distance_mean'))
    df_sim_selected_all.insert(1, 'distance_meteor', df_sim_selected_all.pop('distance_meteor'))
    df_sim_selected_all.insert(1, 'solution_id_dist', df_sim_selected_all.pop('solution_id_dist'))
    df_sim_selected_all.insert(1, 'type', df_sim_selected_all.pop('type'))

    df_sim_selected_all.reset_index(drop=True, inplace=True)

    df_sim_selected_all.to_csv(OUT_PUT_PATH+os.sep+file_name_obs+'_sim_sel.csv', index=False)

    # Insert the column at the first position
    df_sel_shower.insert(1, 'distance_mean', df_sel_shower.pop('distance_mean'))
    df_sel_shower.insert(1, 'distance_meteor', df_sel_shower.pop('distance_meteor'))
    df_sel_shower.insert(1, 'solution_id_dist', df_sel_shower.pop('solution_id_dist'))
    df_sel_shower.insert(1, 'type', df_sel_shower.pop('type'))

    df_sel_shower.reset_index(drop=True, inplace=True)

    df_sel_shower.to_csv(OUT_PUT_PATH+os.sep+file_name_obs+'_sim_sel_bf_knee.csv', index=False)

    if isinstance(df_sel_shower, tuple):
        df_sel_shower = df_sel_shower[0]
    if isinstance(df_sim_selected_all, tuple):
        df_sim_selected_all = df_sim_selected_all[0]

    # DELETE ALL old INDEX

    # Create the new DataFrame by filtering df_sim_PCA
    df_sel_PCA = df_all_PCA[df_all_PCA['solution_id'].isin(df_sel_shower['solution_id'])]
    # change all df_sel_PCA 'type' to Simulation_sel
    df_sel_PCA['type'] = 'Simulation_sel'
    # reset the index
    df_sel_PCA.reset_index(drop=True, inplace=True)

    # df_sel_shower_no_repetitions = df_sim_shower[df_sim_shower['solution_id'].isin(df_sel_shower['solution_id'])]
    # # change all df_sel_PCA 'type' to Simulation_sel
    # df_sel_shower_no_repetitions['type'] = 'Simulation_sel'
    # # reset the index
    # df_sel_shower_no_repetitions.reset_index(drop=True, inplace=True)
    
    df_sel_shower_no_repetitions = df_sel_shower.copy()

    # group by solution_id_dist and keep only n_confront_sel from each group
    df_sel_shower_no_repetitions = df_sel_shower_no_repetitions.groupby('solution_id_dist').head(len(df_sel_shower_no_repetitions))

    # order by distance_meteor
    df_sel_shower_no_repetitions = df_sel_shower_no_repetitions.sort_values('distance_meteor')

    # count duplicates and add a column for the number of duplicates
    df_sel_shower_no_repetitions['num_duplicates'] = df_sel_shower_no_repetitions.groupby('solution_id')['solution_id'].transform('size') 
    
    df_sel_shower_no_repetitions['solution_id_dist'] = df_obs_shower['solution_id'].values[0]

    df_sel_shower_no_repetitions.drop_duplicates(subset='solution_id', keep='first', inplace=True)            

    # save df_sel_shower_real to disk
    df_sel_shower_no_repetitions.to_csv(OUT_PUT_PATH+os.sep+SAVE_SELECTION_FOLDER+os.sep+file_name_obs+'_sim_sel_to_optimize.csv', index=False)



    print('\nSUCCESS: the simulated meteor have been selected\n')

    # Close the Logger to ensure everything is written to the file STOP COPY in TXT file
    sys.stdout.close()

    # Reset sys.stdout to its original value if needed
    sys.stdout = sys.__stdout__

    ########### save dist to observed shower ########################################

    # # save dist also on selected shower
    # distance_current = []
    # for i_shower in range(len(shower_current)):
    #     distance_current.append(scipy.spatial.distance.euclidean(meanPCA_current, shower_current_PCA[i_shower]))
    # shower_current['distance_mean']=distance_current # from the mean of the selected shower
    # shower_current.to_csv(OUT_PUT_PATH+os.sep+file_name_obs+'_obs_and_dist.csv', index=False)

    # PLOT the selected simulated shower ########################################

    # dataframe with the simulated and the selected meteors in the PCA space
    # df_sim_sel_PCA = pd.concat([df_sim_PCA,df_sel_PCA], axis=0)
    if PCA_pairplot:

        # Copy the DataFrame
        df_sim_shower_small = df_sim_shower.copy()

        # Store necessary values before sampling
        # For example, store the first value of var_phys
        physical_vars = ['mass', 'rho', 'sigma', 'erosion_height_start', 'erosion_coeff', 'erosion_mass_index', 'erosion_mass_min', 'erosion_mass_max']
        var_phys_values = {}
        for var_phys in physical_vars:
            var_phys_values[var_phys] = df_sim_shower[var_phys].values[0]

        # if len(df_sim_shower_small) >10000:  # Avoid long plotting times
        #     # Randomly sample 10,000 events
        #     df_sim_shower_small = df_sim_shower_small.sample(n=10000)

        if len(df_sim_shower_small) > 10000:  # Limit to 10,000 rows for performance
            # Separate rows with 'MetSim' or 'Real' types
            metsim_or_real_rows = df_sim_shower_small[df_sim_shower_small['type'].isin(['MetSim', 'Real'])]

            # Sample the remaining rows excluding 'MetSim' and 'Real'
            other_rows = df_sim_shower_small[~df_sim_shower_small['type'].isin(['MetSim', 'Real'])]
            sampled_other_rows = other_rows.sample(n=10000 - len(metsim_or_real_rows), random_state=42)

            # Combine the sampled rows with 'MetSim' or 'Real' rows
            df_sim_shower_small = pd.concat([metsim_or_real_rows, sampled_other_rows], axis=0)

        print('Generating selected simulation histogram plot...')

        # Define a custom palette
        custom_palette = {
            'Real': "r",
            'Simulation': "b",
            'Simulation_sel': "darkorange",
            'MetSim': "k",
            'Realization': "mediumaquamarine",
            'Observation': "limegreen"
        }

        # Concatenate DataFrames
        curr_df = pd.concat([df_sim_shower_small, df_sel_shower, df_obs_shower], axis=0)

        # Compute weights
        curr_df['num_type'] = curr_df.groupby('type')['type'].transform('size')
        curr_df['weight'] = 1 / curr_df['num_type']

        # Plotting
        fig, axs = plt.subplots(int(np.ceil(len(variable_PCA[2:]) / 5)), 5, figsize=(20, 15))
        axs = axs.flatten()

        for ii, var in enumerate(variable_PCA[2:]):
            sns.histplot(curr_df, x=var, weights=curr_df['weight'], hue='type', ax=axs[ii], kde=True, palette=custom_palette, bins=20)
            axs[ii].set_xticks([np.round(np.min(curr_df[var]), 2), np.round(np.max(curr_df[var]), 2)])

            # Invert x-axis for specific variables
            if var in ['beg_abs_mag', 'peak_abs_mag', 'end_abs_mag']:
                axs[ii].invert_xaxis()

            # Format x-axis
            axs[ii].xaxis.set_major_formatter(ScalarFormatter())
            axs[ii].ticklabel_format(useOffset=False, style='plain', axis='x')

            axs[ii].set_ylabel('Probability')
            axs[ii].set_xlabel(var)
            axs[ii].get_legend().remove()
            axs[ii].set_yscale('log')
            axs[ii].set_ylim(0.01, 1)

        plt.tight_layout()
        fig.savefig(OUT_PUT_PATH + os.sep + file_name_obs + '_Histograms_' + str(len(variable_PCA) - 2) + 'var_' + str(PCA_percent) + '%_' + str(pca.n_components_) + 'PC.png', dpi=300)
        plt.close()

        # Sampling df_sim_PCA consistently
        if len(df_sim_PCA) >10000:
            # Use the same indices as in df_sim_shower_small
            df_sim_PCA = df_sim_PCA.loc[df_sim_shower_small.index]

        print('Generating PCA space plot...')

        df_sim_sel_PCA = pd.concat([df_sim_PCA, df_sel_PCA, df_obs_PCA], axis=0)

        # Select only numeric columns
        numeric_columns = df_sim_sel_PCA.select_dtypes(include=[np.number]).columns

        # Map point sizes
        df_sim_sel_PCA['point_size'] = df_sim_sel_PCA['type'].map({
            'Simulation_sel': 5,
            'Simulation': 5,
            'MetSim': 20,
            'Realization': 20,
            'Observation': 40
        })

        # Create the pair plot
        fig = sns.pairplot(
            df_sim_sel_PCA[numeric_columns.append(pd.Index(['type']))],
            hue='type',
            corner=True,
            palette=custom_palette,
            diag_kind='kde',
            plot_kws={'s': 5, 'edgecolor': 'k'}
        )

        # Overlay scatter plots with custom point sizes
        for i in range(len(fig.axes)):
            for j in range(len(fig.axes)):
                if i > j:
                    ax = fig.axes[i, j]
                    sns.scatterplot(
                        data=df_sim_sel_PCA,
                        x=df_sim_sel_PCA.columns[j],
                        y=df_sim_sel_PCA.columns[i],
                        hue='type',
                        size='point_size',
                        sizes=(5, 40),
                        ax=ax,
                        legend=False,
                        edgecolor='k',
                        palette=custom_palette
                    )

        plt.subplots_adjust(hspace=0.3, wspace=0.3)
        fig.savefig(OUT_PUT_PATH + os.sep + file_name_obs + 'PCAspace_sim_sel_real_' + str(len(variable_PCA) - 2) + 'var_' + str(PCA_percent) + '%_' + str(pca.n_components_) + 'PC.png')
        plt.close()

        print('Generating result variable plot...')

        output_folder = OUT_PUT_PATH + os.sep + file_name_obs + VAR_SEL_DIR_SUFX
        if not os.path.isdir(output_folder):
            mkdirP(output_folder)

        # Loop over physical variables
        for var_phys in physical_vars:
            # Create subplots
            fig, axs = plt.subplots(int(np.ceil(len(variable_PCA[2:]) / 5)), 5, figsize=(20, 15))
            axs = axs.flatten()

            for i, var in enumerate(variable_PCA[2:]):
                # Plot simulation data
                axs[i].scatter(df_sim_shower_small[var], df_sim_shower_small[var_phys], c='b')

                # Plot selected data
                axs[i].scatter(df_sel_shower[var], df_sel_shower[var_phys], c='orange')

                # Plot vertical line using stored value
                axs[i].axvline(shower_current[var].values[0], color='limegreen', linestyle='--', linewidth=5)

                # Plot horizontal line using stored value
                axs[i].axhline(var_phys_values[var_phys], color='k', linestyle='-', linewidth=2)

                if i % 5 == 0:
                    axs[i].set_ylabel(var_phys)

                axs[i].set_xlabel(var)
                axs[i].grid()

                # Log scale for specific variables
                if var_phys in ['erosion_mass_min', 'erosion_mass_max']:
                    axs[i].set_yscale('log')

            # Remove unused subplots
            for i in range(len(variable_PCA[2:]), len(axs)):
                fig.delaxes(axs[i])

            plt.tight_layout()
            plt.savefig(output_folder + os.sep + file_name_obs + var_phys + '_vs_var_select_PCA.png')
            plt.close()

        print('Generating PCA position plot...')

        output_folder = OUT_PUT_PATH + os.sep + file_name_obs + PCA_SEL_DIR_SUFX
        if not os.path.isdir(output_folder):
            mkdirP(output_folder)

        # Loop over physical variables
        for var_phys in physical_vars:
            fig, axs = plt.subplots(int(np.ceil(len(columns_PC) / 5)), 5, figsize=(20, 15))
            axs = axs.flatten()

            for i, var in enumerate(columns_PC):
                # Plot simulation data
                axs[i].scatter(df_sim_PCA[var], df_sim_shower_small[var_phys], c='b')

                # Plot selected data
                axs[i].scatter(df_sel_PCA[var], df_sel_shower_no_repetitions[var_phys], c='orange')

                # Plot vertical line
                axs[i].axvline(df_obs_PCA[var].values[0], color='limegreen', linestyle='--', linewidth=5)

                # Plot horizontal line using stored value
                axs[i].axhline(var_phys_values[var_phys], color='k', linestyle='-', linewidth=2)

                if i % 5 == 0:
                    axs[i].set_ylabel(var_phys)

                axs[i].set_xlabel(var)
                axs[i].grid()

                # Log scale for specific variables
                if var_phys in ['erosion_mass_min', 'erosion_mass_max']:
                    axs[i].set_yscale('log')

            # Remove unused subplots
            for i in range(len(columns_PC), len(axs)):
                fig.delaxes(axs[i])

            plt.tight_layout()
            plt.savefig(output_folder + os.sep + file_name_obs + var_phys + '_vs_var_select_PC_space.png')
            plt.close()


    return df_sel_shower, df_sel_shower_no_repetitions, df_sim_selected_all, pcr_results_physical_param, pca.n_components_






def PCAcorrelation_selPLOT(curr_sim_init, curr_sel, output_dir='', pca_N_comp=0):

    curr_sim=curr_sim_init.copy()
    # if len(curr_sim)>10000:
    #     # pick randomly 10000 events
    #     print('Number of events in the simulated :',len(curr_sim))
    #     curr_sim=curr_sim.sample(n=10000).copy()

    if len(curr_sim) > 10000:  # Limit to 10,000 rows for performance
        # Separate rows with 'MetSim' or 'Real' types
        metsim_or_real_rows = curr_sim[curr_sim['type'].isin(['MetSim', 'Real'])]

        # Sample the remaining rows excluding 'MetSim' and 'Real'
        other_rows = curr_sim[~curr_sim['type'].isin(['MetSim', 'Real'])]
        sampled_other_rows = other_rows.sample(n=10000 - len(metsim_or_real_rows), random_state=42)

        # Combine the sampled rows with 'MetSim' or 'Real' rows
        curr_sim = pd.concat([metsim_or_real_rows, sampled_other_rows], axis=0)
        

    curr_sel=curr_sel.copy()
    curr_sel = curr_sel.drop_duplicates(subset='solution_id')
    curr_df_sim_sel=pd.concat([curr_sim,curr_sel], axis=0, ignore_index=True)

    # Define your label mappings
    label_mappings = {
        'mass': '$m_0$ [kg]',
        'rho': '$\\rho$ [kg/m$^3$]',
        'sigma': '$\sigma$ [s$^2$/km$^2$]',
        'erosion_height_start': '$h_e$ [km]',
        'erosion_coeff': '$\eta$ [s$^2$/km$^2$]',
        'erosion_mass_index': '$s$ [-]',
        'erosion_mass_min': '$m_{l}$ [kg]',
        'erosion_mass_max': '$m_{u}$ [kg]'
    }

    # to_plot_unit = [r'$m_0$ [kg]', r'$\rho$ [kg/m$^3$]', r'$\sigma$ [s$^2$/km$^2$]', r'$h_{e}$ [km]', r'$\eta$ [s$^2$/km$^2$]', r'$s$ [-]', r'log($m_{l}$) [-]', r'log($m_{u}$) [-]',r'log($m_{u}$)-log($m_{l}$) [-]']

    # Define a custom palette
    custom_palette = {
        'Real': "r",
        'Simulation': "b",
        'Simulation_sel': "darkorange",
        'MetSim': "k",
        'Realization': "mediumaquamarine",
        'Observation': "limegreen"
    }

    to_plot8 = ['type', 'mass', 'rho', 'sigma', 'erosion_height_start', 'erosion_coeff', 'erosion_mass_index', 'erosion_mass_min', 'erosion_mass_max']
    hue_column = 'type'


    # Create a PairGrid
    pairgrid = sns.PairGrid(curr_df_sim_sel[to_plot8], hue=hue_column, palette=custom_palette)

    # Map the plots
    pairgrid.map_lower(sns.scatterplot, edgecolor='k', palette=custom_palette)
    # for the upper triangle delete x and y axis
    # pairgrid.map_diag(sns.kdeplot)
    # pairgrid.map_diag(sns.histplot, kde=True, color='k', edgecolor='k')
    # pairgrid.add_legend()

    # Update the labels
    for ax in pairgrid.axes.flatten():
        if ax is not None:  # Check if the axis exists
            xlabel = ax.get_xlabel()
            ylabel = ax.get_ylabel()
            if ylabel in label_mappings:
                ax.set_ylabel(label_mappings[ylabel])
            if xlabel in label_mappings:
                ax.set_xlabel(label_mappings[xlabel])
            if ylabel in ['erosion_mass_min', 'erosion_mass_max']:#'sigma', 
                ax.set_yscale('log')
            if xlabel in ['erosion_mass_min', 'erosion_mass_max']: #'sigma', 
                ax.set_xscale('log')

    # # Calculate the correlation matrix
    # corr = curr_df_sim_sel[to_plot8[1:]].corr()

    corr = curr_sel[to_plot8[1:]].corr()

    # Find the min and max correlation values
    vmin = corr.values.min()
    vmax = corr.values.max()
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = sns.color_palette('coolwarm', as_cmap=True)

    # Fill the upper triangle plots with the correlation matrix values and color it with the coolwarm cmap
    for i, row in enumerate(to_plot8[1:]):
        for j, col in enumerate(to_plot8[1:]):
            if i < j:
                ax = pairgrid.axes[i, j]  # Adjust index to fit the upper triangle
                corr_value = corr.loc[row, col]
                ax.text(0.5, 0.5, f'{corr_value:.2f}', horizontalalignment='center', verticalalignment='center', fontsize=12, color='black', transform=ax.transAxes)
                ax.set_facecolor(cmap(norm(corr_value)))
                # cmap = sns.color_palette('coolwarm', as_cmap=True)
                # ax.set_facecolor(cmap(corr_value))

                # Remove the axis labels
                ax.xaxis.set_visible(False)
                ax.yaxis.set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
            if i == j:
                ax = pairgrid.axes[i, j]
                ax.set_axis_off()

    # Adjust layout
    plt.tight_layout()
    
    if pca_N_comp!=0:
        # Save the figure
        plt.savefig(output_dir+os.sep+'PCA'+str(pca_N_comp)+'PC_MixPhysicPropPairPlot_'+str(len(curr_sel))+'ev.png', dpi=300)
    else:
        # Save the figure
        plt.savefig(output_dir+os.sep+'MixPhysicPropPairPlot_'+str(len(curr_sel))+'ev.png', dpi=300)

    # Close the figure
    plt.close()

    # Calculate the correlation matrix
    corr = curr_sel[to_plot8[1:]].corr()

    # Saving correlation matrix to a text file
    if pca_N_comp!=0:
        corr_filename = os.path.join(output_dir, f'correlation_matrix_PCA.txt')
    else:
        corr_filename = os.path.join(output_dir, f'correlation_matrix.txt')
    corr.to_csv(corr_filename, sep='\t', float_format="%.2f")  # Save as a tab-separated file with 2 decimal precision
    print(f"Correlation matrix saved to: {corr_filename}")

    ##########################################################################
    ##########################################################################


# Custom objective function with time-based limit
class TimeLimitedObjective:
    def __init__(self, func, time_limit):
        self.func = func
        self.start_time = None
        self.time_limit = time_limit

    def __call__(self, x):
        if self.start_time is None:
            self.start_time = time.time()
        elif time.time() - self.start_time > self.time_limit:
            raise TimeoutError("Time limit exceeded during optimization.")
        return self.func(x)


def PCA_PhysicalPropPLOT(df_sel_shower_real, df_sim_shower, output_dir, file_name, Min_KDE_point='', save_log=True, pca_N_comp=0):
    df_sim_shower_small = df_sim_shower.copy()
    df_sel_shower = df_sel_shower_real.copy()

    # if len(df_sim_shower_small) > 10000:  # w/o takes forever to plot
    #     # pick randomly 10000 events
    #     df_sim_shower_small = df_sim_shower_small.sample(n=10000)
    #     if 'MetSim' not in df_sim_shower_small['type'].values and 'Real' not in df_sim_shower_small['type'].values:
    #         df_sim_shower_small = pd.concat([df_sim_shower_small.iloc[[0]], df_sim_shower_small])
    
    if len(df_sim_shower_small) > 10000:  # Limit to 10,000 rows for performance
        # Separate rows with 'MetSim' or 'Real' types
        metsim_or_real_rows = df_sim_shower_small[df_sim_shower_small['type'].isin(['MetSim', 'Real'])]

        # Sample the remaining rows excluding 'MetSim' and 'Real'
        other_rows = df_sim_shower_small[~df_sim_shower_small['type'].isin(['MetSim', 'Real'])]
        sampled_other_rows = other_rows.sample(n=10000 - len(metsim_or_real_rows), random_state=42)

        # Combine the sampled rows with 'MetSim' or 'Real' rows
        df_sim_shower_small = pd.concat([metsim_or_real_rows, sampled_other_rows], axis=0)

    if save_log:
        # check if a file with the name "log"+n_PC_in_PCA+"_"+str(len(df_sel))+"ev.txt" already exist
        if os.path.exists(output_dir + os.sep + "log_" + file_name[:15] + "_ConfInterval.txt"):
            # remove the file
            os.remove(output_dir + os.sep + "log_" + file_name[:15] + "_ConfInterval.txt")
        sys.stdout = Logger(output_dir, "log_" + file_name[:15] + "_ConfInterval.txt")  # _30var_99%_13PC

    curr_df_sim_sel = pd.concat([df_sim_shower_small, df_sel_shower], axis=0)

    # multiply the erosion coeff by 1000000 to have it in km/s
    curr_df_sim_sel['erosion_coeff'] = curr_df_sim_sel['erosion_coeff'] * 1000000
    curr_df_sim_sel['sigma'] = curr_df_sim_sel['sigma'] * 1000000
    curr_df_sim_sel['erosion_energy_per_unit_cross_section'] = curr_df_sim_sel['erosion_energy_per_unit_cross_section'] / 1000000
    curr_df_sim_sel['erosion_energy_per_unit_mass'] = curr_df_sim_sel['erosion_energy_per_unit_mass'] / 1000000

    group_mapping = {
        'Simulation_sel': 'selected',
        'MetSim': 'simulated',
        'Real': 'simulated',
        'Simulation': 'simulated'
    }
    curr_df_sim_sel['group'] = curr_df_sim_sel['type'].map(group_mapping)

    curr_df_sim_sel['num_group'] = curr_df_sim_sel.groupby('group')['group'].transform('size')
    curr_df_sim_sel['weight'] = 1 / curr_df_sim_sel['num_group']

    curr_df_sim_sel['num_type'] = curr_df_sim_sel.groupby('type')['type'].transform('size')
    curr_df_sim_sel['weight_type'] = 1 / curr_df_sim_sel['num_type']

    curr_sel = curr_df_sim_sel[curr_df_sim_sel['group'] == 'selected'].copy()

    to_plot = ['mass', 'rho', 'sigma', 'erosion_height_start', 'erosion_coeff', 'erosion_mass_index', 'erosion_mass_min', 'erosion_mass_max', 'erosion_range']
    to_plot_unit = [r'$m_0$ [kg]', r'$\rho$ [kg/m$^3$]', r'$\sigma$ [s$^2$/km$^2$]', r'$h_{e}$ [km]', r'$\eta$ [s$^2$/km$^2$]', r'$s$ [-]', r'log($m_{l}$) [-]', r'log($m_{u}$) [-]', r'log($m_{u}$)-log($m_{l}$) [-]']

    fig, axs = plt.subplots(3, 3)
    axs = axs.flatten()

    print('\\hline')
    if len(Min_KDE_point) > 0:
        # delete the first two elements
        Min_KDE_point = Min_KDE_point[2:]
        # print('flag_',Min_KDE_point)
        print('Variables & ' + str(df_sim_shower['type'].iloc[0]) + ' & 95\\%CIlow & Mean & Mode & Dens.Point & 95\\%CIup \\\\')
    else:
        print('Variables & ' + str(df_sim_shower['type'].iloc[0]) + ' & 95\\%CIlow & Mean & Mode & 95\\%CIup \\\\')

    ii_densest = 0
    for i in range(9):
        plotvar = to_plot[i]

        if i == 8:
            # Plot only the legend
            axs[i].axis('off')  # Turn off the axis

            # Create custom legend entries
            import matplotlib.patches as mpatches
            from matplotlib.lines import Line2D

            # Define the legend elements
            # Define the legend elements
            prior_patch = mpatches.Patch(color='blue', label='Priors', alpha=0.5, edgecolor='black')
            sel_events_patch = mpatches.Patch(color='darkorange', label='Selected Events', alpha=0.5, edgecolor='red')
            if 'MetSim' in curr_df_sim_sel['type'].values:
                metsim_line = Line2D([0], [0], color='black', linewidth=2, label='Metsim Solution')
            else:
                metsim_line = Line2D([0], [0], color='green', linestyle='--', linewidth=2, label='Real Solution')
            mode_line = Line2D([0], [0], color='red', linestyle='-.', label='Mode')
            mean_line = Line2D([0], [0], color='blue', linestyle='--', label='Mean')
            if len(Min_KDE_point) > 0:
                dens_point_line = Line2D([0], [0], color='blue', linestyle='-.', label='Densest Point')
                # Create the legend
                legend_elements = [prior_patch, sel_events_patch, metsim_line, mean_line, mode_line, dens_point_line]
            else:
                # Create the legend
                legend_elements = [prior_patch, sel_events_patch, metsim_line, mean_line, mode_line]
            
            axs[i].legend(handles=legend_elements, loc='upper right', fontsize='small')

            # Remove axes ticks and labels
            axs[i].set_xticks([])
            axs[i].set_yticks([])
            axs[i].set_xlabel('')
            axs[i].set_ylabel('')
            continue  # Skip to next iteration

        if plotvar == 'erosion_mass_min' or plotvar == 'erosion_mass_max':
            # take the log of the erosion_mass_min and erosion_mass_max
            curr_df_sim_sel[plotvar] = np.log10(curr_df_sim_sel[plotvar])
            curr_sel[plotvar] = np.log10(curr_sel[plotvar])
            if len(Min_KDE_point) > 0:
                Min_KDE_point[ii_densest] = np.log10(Min_KDE_point[ii_densest])

        sns.histplot(curr_df_sim_sel, x=curr_df_sim_sel[plotvar], weights=curr_df_sim_sel['weight'], hue='group', ax=axs[i], palette='bright', bins=20)
        unique_values_count = curr_sel[plotvar].nunique()
        if unique_values_count > 1:
            # Add the KDE to the plot
            sns.histplot(curr_sel, x=curr_sel[plotvar], weights=curr_sel['weight'], bins=20, ax=axs[i], fill=False, edgecolor=False, color='r', kde=True, binrange=[np.min(curr_df_sim_sel[plotvar]), np.max(curr_df_sim_sel[plotvar])])
            kde_line = axs[i].lines[-1]
            axs[i].lines[-1].remove()
        else:
            kde_line = None

        axs[i].axvline(x=np.mean(curr_df_sim_sel[curr_df_sim_sel['group'] == 'selected'][plotvar]), color='blue', linestyle='--')

        if 'MetSim' in curr_df_sim_sel['type'].values:
            axs[i].axvline(x=curr_df_sim_sel[curr_df_sim_sel['type'] == 'MetSim'][plotvar].values[0], color='k', linewidth=2)
            find_type = 'MetSim'
        elif 'Real' in curr_df_sim_sel['type'].values:
            axs[i].axvline(x=curr_df_sim_sel[curr_df_sim_sel['type'] == 'Real'][plotvar].values[0], color='g', linewidth=2, linestyle='--')
            find_type = 'Real'

        if plotvar == 'erosion_mass_min' or plotvar == 'erosion_mass_max':
            # Convert back from log scale
            curr_df_sim_sel[plotvar] = 10 ** curr_df_sim_sel[plotvar]
            curr_sel[plotvar] = 10 ** curr_sel[plotvar]

        # Calculate percentiles
        sigma_95 = np.percentile(curr_sel[plotvar], 95)
        sigma_5 = np.percentile(curr_sel[plotvar], 5)

        mean_values_sel = np.mean(curr_sel[plotvar])

        if kde_line is not None:
            # Get the x and y data from the KDE line
            kde_line_Xval = kde_line.get_xdata()
            kde_line_Yval = kde_line.get_ydata()

            # Find the index of the maximum y value (mode)
            max_index = np.argmax(kde_line_Yval)
            # Plot a vertical line at the mode
            axs[i].axvline(x=kde_line_Xval[max_index], color='red', linestyle='-.')

            x_10mode = kde_line_Xval[max_index]
            if plotvar == 'erosion_mass_min' or plotvar == 'erosion_mass_max':
                x_10mode = 10 ** kde_line_Xval[max_index]

            if len(Min_KDE_point) > 0:
                if len(Min_KDE_point) > ii_densest:
                    densest_index = find_closest_index(kde_line_Xval, [Min_KDE_point[ii_densest]])
                    axs[i].axvline(x=Min_KDE_point[ii_densest], color='blue', linestyle='-.')
                    if plotvar == 'erosion_mass_min' or plotvar == 'erosion_mass_max':
                        Min_KDE_point[ii_densest] = 10 ** (Min_KDE_point[ii_densest])

                    if i < 9:
                        print('\\hline') 
                        print(f"{to_plot_unit[i]} & {'{:.4g}'.format(curr_df_sim_sel[curr_df_sim_sel['type'] == find_type][plotvar].values[0])} & {'{:.4g}'.format(sigma_5)} & {'{:.4g}'.format(mean_values_sel)} & {'{:.4g}'.format(x_10mode)} & {'{:.4g}'.format(Min_KDE_point[ii_densest])} & {'{:.4g}'.format(sigma_95)} \\\\")
                    ii_densest += 1
            else:
                if i < 9:
                    print('\\hline')
                    print(f"{to_plot_unit[i]} & {'{:.4g}'.format(curr_df_sim_sel[curr_df_sim_sel['type'] == find_type][plotvar].values[0])} & {'{:.4g}'.format(sigma_5)} & {'{:.4g}'.format(mean_values_sel)} & {'{:.4g}'.format(x_10mode)} & {'{:.4g}'.format(sigma_95)} \\\\")
        else:
            if i < 9:
                print('\\hline')
                print(f"{to_plot_unit[i]} & {'{:.4g}'.format(curr_df_sim_sel[curr_df_sim_sel['type'] == find_type][plotvar].values[0])} & {'{:.4g}'.format(sigma_5)} & {'{:.4g}'.format(mean_values_sel)} & {'{:.4g}'.format(sigma_95)} \\\\")

        axs[i].set_ylabel('Probability')
        axs[i].set_xlabel(to_plot_unit[i])

        # Adjust y-axis limit
        if axs[i].get_ylim()[1] > 1:
            axs[i].set_ylim(0, 1)

        # Remove individual legends
        axs[i].get_legend().remove()

        if i == 0:
            # Adjust x-axis offset text
            axs[i].xaxis.get_offset_text().set_x(1.10)

    plt.tight_layout()
    print('\\hline')

    if pca_N_comp != 0:
        # Save the figure
        fig.savefig(output_dir + os.sep + 'PCA' + str(pca_N_comp) + 'PC_'+ file_name +'_PhysicProp_' + str(len(curr_sel)) + 'ev.png', dpi=300)
    else:
        # Save the figure
        fig.savefig(output_dir + os.sep + file_name + '_PhysicProp_' + str(len(curr_sel)) + 'ev.png', dpi=300)
    plt.close()

    if save_log:
        sys.stdout.close()
        sys.stdout = sys.__stdout__

    # # Additional plotting for realizations (if applicable)
    # ii_densest = 0
    # if 'solution_id_dist' in df_sel_shower_real.columns:
    #     if len(df_sel_shower_real['solution_id_dist'].unique()) < 60 and len(df_sel_shower_real['solution_id_dist'].unique()) > 1:
    #         print('Plot the distribution of the Realizations', len(df_sel_shower_real['solution_id_dist'].unique()))
    #         fig, axs = plt.subplots(3, 3)
    #         axs = axs.flatten()

    #         for i in range(9):
    #             plotvar = to_plot[i]

    #             if i == 8:
    #                 # Plot only the legend
    #                 axs[i].axis('off')

    #                 import matplotlib.patches as mpatches
    #                 from matplotlib.lines import Line2D

    #                 prior_patch = mpatches.Patch(color='blue', label='Priors')
    #                 sel_events_patch = mpatches.Patch(color='orange', label='Selected Events')
    #                 if 'MetSim' in curr_df_sim_sel['type'].values:
    #                     metsim_line = Line2D([0], [0], color='black', linewidth=2, label='Metsim Solution')
    #                 else:
    #                     metsim_line = Line2D([0], [0], color='green', linestyle='--', linewidth=2, label='Real Solution')
    #                 mode_line = Line2D([0], [0], color='red', linestyle='--', label='Mode')

    #                 legend_elements = [prior_patch, sel_events_patch, metsim_line, mode_line]
    #                 axs[i].legend(handles=legend_elements, loc='center')

    #                 axs[i].set_xticks([])
    #                 axs[i].set_yticks([])
    #                 axs[i].set_xlabel('')
    #                 axs[i].set_ylabel('')
    #                 continue

    #             if plotvar == 'erosion_mass_min' or plotvar == 'erosion_mass_max':
    #                 sns.histplot(curr_df_sim_sel, x=np.log10(curr_df_sim_sel[plotvar]), weights=curr_df_sim_sel['weight'], hue='group', ax=axs[i], palette='bright', bins=20, binrange=[np.log10(np.min(curr_df_sim_sel[plotvar])), np.log10(np.max(curr_df_sim_sel[plotvar]))])
    #                 sns.histplot(curr_df_sim_sel, x=np.log10(curr_df_sim_sel[plotvar]), weights=curr_df_sim_sel['weight'], hue='solution_id_dist', ax=axs[i], multiple="stack", bins=20, binrange=[np.log10(np.min(curr_df_sim_sel[plotvar])), np.log10(np.max(curr_df_sim_sel[plotvar]))])
    #                 sns.histplot(curr_sel, x=np.log10(curr_sel[plotvar]), weights=curr_sel['weight'], bins=20, ax=axs[i], multiple="stack", fill=False, edgecolor=False, color='r', kde=True, binrange=[np.log10(np.min(curr_df_sim_sel[plotvar])), np.log10(np.max(curr_df_sim_sel[plotvar]))])

    #                 kde_line = axs[i].lines[-1]
    #                 axs[i].lines[-1].remove()

    #                 if 'MetSim' in curr_df_sim_sel['type'].values:
    #                     axs[i].axvline(x=np.log10(curr_df_sim_sel[curr_df_sim_sel['type'] == 'MetSim'][plotvar].values[0]), color='k', linewidth=2)
    #                 elif 'Real' in curr_df_sim_sel['type'].values:
    #                     axs[i].axvline(x=np.log10(curr_df_sim_sel[curr_df_sim_sel['type'] == 'Real'][plotvar].values[0]), color='g', linewidth=2, linestyle='--')

    #             else:
    #                 sns.histplot(curr_df_sim_sel, x=curr_df_sim_sel[plotvar], weights=curr_df_sim_sel['weight'], hue='group', ax=axs[i], palette='bright', bins=20, binrange=[np.min(curr_df_sim_sel[plotvar]), np.max(curr_df_sim_sel[plotvar])])
    #                 sns.histplot(curr_df_sim_sel, x=curr_df_sim_sel[plotvar], weights=curr_df_sim_sel['weight'], hue='solution_id_dist', ax=axs[i], multiple="stack", bins=20, binrange=[np.min(curr_df_sim_sel[plotvar]), np.max(curr_df_sim_sel[plotvar])])
    #                 sns.histplot(curr_sel, x=curr_sel[plotvar], weights=curr_sel['weight'], bins=20, ax=axs[i], multiple="stack", fill=False, edgecolor=False, color='r', kde=True, binrange=[np.min(curr_df_sim_sel[plotvar]), np.max(curr_df_sim_sel[plotvar])])

    #                 kde_line = axs[i].lines[-1]
    #                 axs[i].lines[-1].remove()

    #                 if 'MetSim' in curr_df_sim_sel['type'].values:
    #                     axs[i].axvline(x=curr_df_sim_sel[curr_df_sim_sel['type'] == 'MetSim'][plotvar].values[0], color='k', linewidth=2)
    #                 elif 'Real' in curr_df_sim_sel['type'].values:
    #                     axs[i].axvline(x=curr_df_sim_sel[curr_df_sim_sel['type'] == 'Real'][plotvar].values[0], color='g', linewidth=2, linestyle='--')

    #             axs[i].set_ylabel('Probability')
    #             axs[i].set_xlabel(to_plot_unit[i])

    #             if axs[i].get_ylim()[1] > 1:
    #                 axs[i].set_ylim(0, 1)

    #             axs[i].get_legend().remove()

    #         plt.tight_layout()

    #         if pca_N_comp != 0:
    #             # Save the figure
    #             fig.savefig(output_dir + os.sep + 'PCA' + str(pca_N_comp) + 'PC_'+ file_name + '_PhysicProp_Reliazations_' + str(len(curr_sel)) + 'ev.png', dpi=300)
    #         else:
    #             # Save the figure
    #             fig.savefig(output_dir + os.sep + file_name + '_PhysicProp_Reliazations_' + str(len(curr_sel)) + 'ev.png', dpi=300)

    #         plt.close()




def PCA_LightCurveCoefPLOT(df_sel_shower_real, df_obs_shower, output_dir, fit_funct, gensim_data_obs='', mag_noise_real=0.1, len_noise_real=20.0, fps=32, file_name_obs='', trajectory_Metsim_file='', output_folder_of_csv='', pca_N_comp=0):
    """
    Plots the light curve coefficients and includes a table with parameters for each colored curve.

    Parameters:
    - df_sel_shower_real: DataFrame with selected shower real data.
    - df_obs_shower: DataFrame with observed shower data.
    - output_dir: Directory to save the output plot.
    - fit_funct: Fitting function data.
    - gensim_data_obs: Generated simulation observation data (optional).
    - mag_noise_real: Magnitude noise (default 0.1).
    - len_noise_real: Length noise in meters (default 20.0).
    - fps: Frames per second (default 32).
    - file_name_obs: File name for observations (optional).
    - trajectory_Metsim_file: Metsim trajectory file (optional).
    - output_folder_of_csv: Output folder for CSV (optional).
    """

    # Number of observations and selections to plot
    n_confront_obs = 1
    n_confront_sel = 10

    # Flags for additional fits (set to False as default)
    with_noise = True
    noise_data_input = False
    jacchia_fit = False
    parabolic_fit = False
    t0_fit = False
    mag_fit = False

    # Convert length noise to km and calculate velocity noise
    len_noise = len_noise_real / 1000
    vel_noise = (len_noise * np.sqrt(2) / (1 / fps))

    # Increase figure size to provide more space for the table
    fig = plt.figure(figsize=(22, 6))  # Increased figure width
    # Adjust width_ratios to allocate more space to the table
    gs = GridSpec(1, 3, width_ratios=[1, 1, 1])  # Allocated equal space to the table

    # Create axes for the two plots
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    df_sel_shower = df_sel_shower_real.copy()

    # Adjust units for erosion coefficients
    df_sel_shower['erosion_coeff'] = df_sel_shower['erosion_coeff'] * 1e6
    df_sel_shower['sigma'] = df_sel_shower['sigma'] * 1e6

    # Limit observations and selections if necessary
    if n_confront_obs < len(df_obs_shower):
        df_obs_shower = df_obs_shower.head(n_confront_obs)

    if n_confront_sel < len(df_sel_shower):
        df_sel_shower = df_sel_shower.head(n_confront_sel)

    # Concatenate observation and selection DataFrames
    curr_sel = pd.concat([df_obs_shower, df_sel_shower], axis=0).reset_index(drop=True)

    # Initialize data for the table
    table_data = []
    row_colors = []

    # Define headers for the table
    headers = [
        '',  # This will be the color column
        r'$\mathbf{mag$\chi^2_{red}$}$',
        r'$\mathbf{lag$\chi^2_{red}$}$',
        r'$\mathbf{m_0 \ [kg]}$',
        r'$\mathbf{\rho \ [kg/m^3]}$',
        r'$\mathbf{\sigma \ [s^2/km^2]}$',
        r'$\mathbf{\eta \ [s^2/km^2]}$',
        r'$\mathbf{h_e \ [km]}$',
        r'$\mathbf{s}$',
        r'$\mathbf{m_l \ [kg]}$',
        r'$\mathbf{m_u \ [kg]}$'
    ]

    # Loop over the observations and selected simulations
    for ii in range(len(curr_sel)):
        namefile_sel = curr_sel.iloc[ii]['solution_id']
        Metsim_flag = False
        print('real', trajectory_Metsim_file, '- sel', namefile_sel)

        # Check if the file exists
        if not os.path.isfile(namefile_sel):
            print('file ' + namefile_sel + ' not found')
            continue
        else:
            # Read the appropriate data file
            if namefile_sel.endswith('.pickle'):
                data_file = read_pickle_reduction_file(namefile_sel)
                data_file_real = data_file.copy()

            elif namefile_sel.endswith('.json'):
                with open(namefile_sel, "r") as f:
                    data = json.loads(f.read())
                if 'ht_sampled' in data:
                    if ii == 0:
                        data_file = read_with_noise_GenerateSimulations_output(namefile_sel, fps)
                        data_file_real = data_file.copy()
                    else:
                        data_file = read_GenerateSimulations_output(namefile_sel, gensim_data_obs)
                        data_file_real = data_file.copy()
                else:
                    if trajectory_Metsim_file == '':
                        print('no data for the Metsim file')
                        continue

                    trajectory_Metsim_file_name = trajectory_Metsim_file.split(os.sep)[-1]
                    namefile_sel_name = namefile_sel.split(os.sep)[-1]

                    if trajectory_Metsim_file_name == namefile_sel_name:
                        _, data_file, _ = run_simulation(trajectory_Metsim_file, gensim_data_obs, fit_funct)
                        Metsim_flag = True
                    else:
                        _, data_file, _ = run_simulation(namefile_sel, gensim_data_obs, fit_funct)
            
            if ii == 0:
                # give the name of the file
                file_name_only = os.path.basename(namefile_sel)

            # Extract necessary data from the data file
            height_km = np.array(data_file['height']) / 1000
            abs_mag_sim = np.array(data_file['absolute_magnitudes'])
            obs_time = np.array(data_file['time'])
            vel_kms = np.array(data_file['velocities']) / 1000
            lag_km = np.array(data_file['lag']) / 1000

        if ii == 0:
            # Plotting the observed data (green line)
            if with_noise and fit_funct != '':
                height_km_err = np.array(fit_funct['height']) / 1000
                abs_mag_sim_err = np.array(fit_funct['absolute_magnitudes'])

                # Plot confidence intervals (filled areas)
                ax0.fill_betweenx(
                    height_km_err,
                    abs_mag_sim_err - mag_noise_real,
                    abs_mag_sim_err + mag_noise_real,
                    color='darkgray',
                    label='1$\sigma$ '+str(np.round(mag_noise_real,3)),
                    alpha=0.2
                )
                ax0.fill_betweenx(
                    height_km_err,
                    abs_mag_sim_err - mag_noise_real * 1.96,
                    abs_mag_sim_err + mag_noise_real * 1.96,
                    color='lightgray',
                    alpha=0.2
                )

                obs_time_err = np.array(fit_funct['time'])
                vel_kms_err = np.array(fit_funct['velocities']) / 1000

                # Plot velocity confidence intervals
                ax1.fill_between(
                    obs_time_err,
                    vel_kms_err - vel_noise,
                    vel_kms_err + vel_noise,
                    color='darkgray',
                    label='1$\sigma$ '+str(np.round(len_noise*1000,1))+' m',
                    alpha=0.2
                )
                ax1.fill_between(
                    obs_time_err,
                    vel_kms_err - vel_noise * 1.96,
                    vel_kms_err + vel_noise * 1.96,
                    color='lightgray',
                    alpha=0.2
                )

            # Store real observation data
            real_time = obs_time
            real_abs_mag = abs_mag_sim
            real_height_km = height_km

            # Plot the observed data (green markers)
            ax0.plot(abs_mag_sim, height_km, 'o', color='g')
            ax1.plot(obs_time, vel_kms, 'o', color='g')

            # Optionally, include observed data in the table
            # Uncomment the following lines if you want to include observed data
            # curve_data = [
            #     '',  # Placeholder for color
            #     'N/A',  # mag$_{RMSD}$
            #     'N/A',  # len$_{RMSD}$
            #     'N/A',  # m0
            #     'N/A',  # rho
            #     'N/A',  # sigma
            #     'N/A',  # eta
            #     'N/A',  # he
            #     'N/A',  # s
            #     'N/A',  # ml
            #     'N/A'   # mu
            # ]
            # row_colors.append('g')  # Color of the observed data
            # table_data.append(curve_data)

        else:
            # Limit the number of selections plotted
            if ii > n_confront_sel:
                break  # Exit the loop if we've reached the desired number of selections

            # Interpolate time positions based on height
            interp_ht_time = interp1d(
                real_height_km,
                real_time,
                kind='linear',
                bounds_error=False,
                fill_value='extrapolate'
            )
            residual_time_pos = interp_ht_time(height_km)

            # Plot the selected simulation data
            if Metsim_flag:
                # For Metsim data, plot in black
                line_sel0, = ax0.plot(abs_mag_sim, height_km, color='k')
                line, = ax1.plot(residual_time_pos, vel_kms, color='k')
                line_color = 'k'
            else:
                line_sel0, = ax0.plot(abs_mag_sim, height_km)
                line_color = line_sel0.get_color()
                if line_color == '#2ca02c':
                    line_color='m'
                    # change the color of line_sel0
                    line_sel0.set_color('m')
                line, = ax1.plot(residual_time_pos, vel_kms, color=line_color)

            # Collect data for the table
            curve_data = [
                '',  # Placeholder for color, will be replaced later
                round(curr_sel.iloc[ii]['chi2_red_mag'], 3) if 'chi2_red_mag' in curr_sel.columns else 'N/A',
                round(curr_sel.iloc[ii]['chi2_red_lag'] * 1000, 1) if 'chi2_red_lag' in curr_sel.columns else 'N/A',
                '{:.2e}'.format(curr_sel.iloc[ii]['mass']) if 'mass' in curr_sel.columns else 'N/A',
                round(curr_sel.iloc[ii]['rho']) if 'rho' in curr_sel.columns else 'N/A',
                round(curr_sel.iloc[ii]['sigma'], 4) if 'sigma' in curr_sel.columns else 'N/A',
                round(curr_sel.iloc[ii]['erosion_coeff'], 3) if 'erosion_coeff' in curr_sel.columns else 'N/A',
                round(curr_sel.iloc[ii]['erosion_height_start'], 1) if 'erosion_height_start' in curr_sel.columns else 'N/A',
                round(curr_sel.iloc[ii]['erosion_mass_index'], 2) if 'erosion_mass_index' in curr_sel.columns else 'N/A',
                '{:.2e}'.format(curr_sel.iloc[ii]['erosion_mass_min']) if 'erosion_mass_min' in curr_sel.columns else 'N/A',
                '{:.2e}'.format(curr_sel.iloc[ii]['erosion_mass_max']) if 'erosion_mass_max' in curr_sel.columns else 'N/A'
            ]

            # Append the data and color
            row_colors.append(line_color)
            table_data.append(curve_data)

    # Check if table_data is empty
    if not table_data:
        print("No data available to display in the table.")
        plt.close()  # Close the plot
        return  # Exit the function or skip table creation

    # Adjust the plot styles and axes
    ax0.invert_xaxis()
    ax1.grid(linestyle='--', color='lightgray')
    ax0.grid(linestyle='--', color='lightgray')

    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Velocity [km/s]')
    ax0.set_xlabel('Absolute Magnitude [-]')
    ax0.set_ylabel('Height [km]')

    # Remove legends from both plots if any
    if ax0.get_legend() is not None:
        ax0.get_legend().remove()
    if ax1.get_legend() is not None:
        ax1.get_legend().remove()

    # Adjust layout to make room for the table on the right
    # plt.subplots_adjust(right=0.75)  # Adjust right as needed

    # # Adjust layout to make room for the table on the far right
    plt.subplots_adjust(left=0.05, right=0.7)  # Increase the 'right' value to detach the table

    # Adjust the GridSpec to create more space between the second plot and the table
    gs = GridSpec(1, 3, width_ratios=[1, 1, 0.97])  # Reduce the width of the table column

    # Create a new axis for the table
    ax_table = fig.add_subplot(gs[0, 2])
    ax_table.axis('off')  # Hide the axis lines and ticks

    # Create the table in ax_table
    # Include color patches in the first column
    cell_text = []
    for idx, row in enumerate(table_data):
        # Replace the placeholder with the color patch
        row[0] = ''
        cell_text.append(row)

    # Create the table
    table = ax_table.table(
        cellText=cell_text,
        colLabels=headers,
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)  # Increased font size for better readability

    # Loop through each header cell to set a different font size
    for col_idx in range(len(headers)):
        header_cell = table[(0, col_idx)]  # Access the header row cells
        header_cell.set_fontsize(6)        # Set a smaller font size for the header
        # header_cell.set_fontweight('bold') # Optional: make the header bold

    # Adjust the table column widths to fit labels
    n_cols = len(headers)
    col_widths = [0.1] + [0.13] * (n_cols - 1)  # Increased column widths
    for col_idx, width in enumerate(col_widths):
        for row_idx in range(len(table_data) + 1):  # +1 for header row
            cell = table[(row_idx, col_idx)]
            cell.set_width(width)

    # Set the cell colors for the first column
    for row_idx, color in enumerate(row_colors):
        cell = table[row_idx + 1, 0]  # +1 to skip header row
        cell.set_facecolor(color)
        # Optionally, set text color to improve readability
        if color == 'k':
            cell.get_text().set_color('white')
        else:
            cell.get_text().set_color('black')

    # Adjust the cell heights to ensure labels fit
    n_rows = len(table_data) + 1  # +1 for header row
    for row_idx in range(n_rows):
        for col_idx in range(n_cols):
            cell = table[(row_idx, col_idx)]
            cell.set_height(1 / n_rows)

    fig.suptitle(
        file_name_only + r' - mag$_{RMSD}$ ' + str(round(curr_sel.iloc[0]['rmsd_mag'], 3)) +
        r' len$_{RMSD}$ ' + str(round(curr_sel.iloc[0]['rmsd_len']*1000, 1)) + ' m',
        fontsize=12,     # Adjust font size as needed
        ha='left',       # Align text to the left
        x=0.05,           # Adjust x to move it to the left (0 is far left, 1 is far right)
        y=0.95           # Adjust y to move it up (0 is bottom, 1 is top)
    )

    if pca_N_comp != 0:
        plt.savefig(output_dir + os.sep + 'PCA'+str(pca_N_comp)+'PC_'+file_name_obs + '_Heigh_MagVelCoef.png', bbox_inches='tight')
    else:
        # Save and close the plot
        plt.savefig(output_dir + os.sep + file_name_obs + '_Heigh_MagVelCoef.png', bbox_inches='tight')
    plt.close()

    # Save the DataFrame with RMSD
    if output_folder_of_csv == '':
        df_sel_shower_real.to_csv(output_dir + os.sep +'PCA_'+file_name_obs + '_sim_sel.csv', index=False)
    else:
        df_sel_shower_real.to_csv(output_folder_of_csv, index=False)




# OPTIMIZATION ####################################################################################

def PCA_LightCurveRMSDPLOT_optimize(df_sel_shower, df_obs_shower, data_file_real, output_dir, fit_funct='', gen_Metsim='', mag_noise_real = 0.1, len_noise_real = 20.0, mag_RMSD = 0.1, len_RMSD = 20.0, fps=32, file_name_obs='', save_results_folder_events_plots='', run_optimization=True):
    # try:
        # merge curr_sel and curr_obs
        curr_sel = df_sel_shower.copy()

        pd_datafram_PCA_selected_optimized=pd.DataFrame()

        mag_noise = mag_noise_real.copy()
        len_noise = len_noise_real.copy()

        len_noise= len_noise/1000
        # velocity noise 1 sigma km/s
        vel_noise = (len_noise*np.sqrt(2)/(1/fps))

        curr_sel['erosion_coeff']=curr_sel['erosion_coeff']*1000000
        curr_sel['sigma']=curr_sel['sigma']*1000000
        
        #     data_file_real = read_pickle_reduction_file(df_obs_shower.iloc[0]['solution_id'])
        if df_obs_shower.iloc[0]['solution_id'].endswith('.json'):
            # data_file_real = read_with_noise_GenerateSimulations_output(df_obs_shower.iloc[0]['solution_id'], fps)
            print('json file NO optimization possible:', df_obs_shower.iloc[0]['solution_id'])
            run_optimization=False

        _, _, _, residuals_mag_real, residuals_vel_real, residuals_len_real, residual_time_pos_real, residual_height_pos_real , _ = RMSD_calc_diff(fit_funct, data_file_real)

        # Get the default color cycle
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

        # Create an infinite cycle of colors
        infinite_color_cycle = itertools.cycle(color_cycle)

        for ii in range(len(curr_sel)):
            
            # pick the ii element of the solution_id column 
            namefile_sel=curr_sel.iloc[ii]['solution_id']
            Metsim_flag=False

            # chec if the file exist
            if not os.path.isfile(namefile_sel):
                print('file '+namefile_sel+' not found')
                if ii == len(curr_sel)-1:
                    print('no more files to optimize')
                    plt.close()
                continue
            else:
                if namefile_sel.endswith('.pickle'):
                    data_file = read_pickle_reduction_file(namefile_sel)

                elif namefile_sel.endswith('.json'):
                    # open the json file with the name namefile_sel 
                    f = open(namefile_sel,"r")
                    data = json.loads(f.read())
                    if 'ht_sampled' in data:
                        data_file = read_GenerateSimulations_output(namefile_sel, data_file_real)

                    else:
                        if gen_Metsim == '':
                            print('no data for the Metsim file')
                            # check if ii is the last element
                            if ii == len(curr_sel)-1:
                                print('no more files to optimize')
                                plt.close()
                            continue

                        else:
                            # make a copy of gen_Metsim
                            data_file = gen_Metsim.copy()
                            # file metsim
                            Metsim_flag=True

                rmsd_mag, rmsd_vel, rmsd_lag, residuals_mag, residuals_vel, residuals_len, residual_time_pos, residual_height_pos, lag_km_sim = RMSD_calc_diff(data_file, data_file_real)
                BOOLchi_p_mag, BOOLchi_p_lag, chi_p_results, residuals_mag, residuals_vel, residuals_len, residual_time_pos, residual_height_pos , lag_km_sim = chiSquare_test_sim_real(data_file, data_file_real)


            # Interpolation on the fit data's height grid
            interp_ht_time = interp1d(data_file_real['height'], data_file_real['time'], kind='linear', bounds_error=False, fill_value='extrapolate')
            # Interpolated fit on data grid
            sim_time_pos = interp_ht_time(data_file['height'])

            color_line=next(infinite_color_cycle)

            rmsd_mag = curr_sel.iloc[ii]['rmsd_mag']
            rmsd_lag = curr_sel.iloc[ii]['rmsd_len']

            color_line=next(infinite_color_cycle)

            image_name = namefile_sel.split(os.sep)[-1]
            if '.pickle' in image_name:
                image_name=image_name[:15]
            if '.json' in image_name:
                image_name=image_name[:-5]
            image_name = image_name+'.png'

            # split the name from the path
            _, file_name_title = os.path.split(curr_sel.iloc[ii]['solution_id'])
            
            file_json_save_phys_NOoptimized=output_dir+os.sep+SAVE_SELECTION_FOLDER+os.sep+file_name_title
            if Metsim_flag:
                file_json_save_phys=output_dir+os.sep+SAVE_SELECTION_FOLDER+os.sep+file_name_title[:23]+'_fitted.json'
                file_json_save_results=output_dir+os.sep+save_results_folder_events_plots+os.sep+file_name_title[:23]+'_fitted.json'
            else:
                file_json_save_phys=output_dir+os.sep+SAVE_SELECTION_FOLDER+os.sep+file_name_obs[:15]+'_'+file_name_title[:-5]+'_fitted.json'
                file_json_save_results=output_dir+os.sep+save_results_folder_events_plots+os.sep+file_name_obs[:15]+'_'+file_name_title[:-5]+'_fitted.json'

            try:
                # copy the file over to the selected folder
                shutil.copy(namefile_sel, file_json_save_phys_NOoptimized)
            except shutil.SameFileError:
                # log the event or handle it in another way if needed
                print(f"Skipping copy operation because '{namefile_sel}' and '{file_json_save_phys_NOoptimized}' are the same file.")

            # copy the data to the mode
            data_file_sim = data_file.copy()
            data_file_sim['time'] = sim_time_pos
            data_file_sim['res_absolute_magnitudes'] = residuals_mag
            data_file_sim['res_velocities'] = residuals_vel
            data_file_sim['res_lag'] = residuals_len * 1000
            data_file_sim['lag'] = lag_km_sim * 1000
            data_file_sim['rmsd_mag'] = rmsd_mag
            data_file_sim['rmsd_vel'] = rmsd_vel
            data_file_sim['rmsd_len'] = rmsd_lag
            data_file_sim['chi2_red_mag'] = chi_p_results['chi2_red_mag']
            data_file_sim['chi2_red_lag'] = chi_p_results['chi2_red_lag']

            chi2_mag = data_file_sim['chi2_red_mag']
            chi2_lag = data_file_sim['chi2_red_lag']

            if Metsim_flag:
                plot_data_with_residuals_and_real(mag_RMSD, len_RMSD*np.sqrt(2)/(1.0/fps), len_RMSD, fit_funct, data_file_real, data_file_real['name'].split(os.sep)[-1], image_name, output_dir+os.sep+SAVE_SELECTION_FOLDER, data_file_sim, 'Metsim')
            else:
                plot_data_with_residuals_and_real(mag_RMSD, len_RMSD*np.sqrt(2)/(1.0/fps), len_RMSD, fit_funct, data_file_real, data_file_real['name'].split(os.sep)[-1], image_name, output_dir+os.sep+SAVE_SELECTION_FOLDER, data_file_sim)

            if run_optimization:
                # check if file_json_save_phys is present
                if not os.path.isfile(file_json_save_phys):
                    output_dir_optimized = output_dir+os.sep+SAVE_SELECTION_FOLDER+os.sep+'Optimization_'+file_name_title[:-5]
                    mkdirP(output_dir_optimized)

                    if Metsim_flag:
                        const_nominal, _ = loadConstants(namefile_sel)
                        saveConstants(const_nominal,output_dir_optimized,file_name_obs+'_sim_fit.json')
                    else:
                        # from namefile_sel json file open the json file and save the namefile_sel.const part as file_name_obs+'_sim_fit.json'
                        with open(namefile_sel) as json_file:
                            data = json.load(json_file)
                            const_part = data['const']
                            with open(output_dir_optimized+os.sep+file_name_obs+'_sim_fit.json', 'w') as outfile:
                                json.dump(const_part, outfile, indent=4)

                    if curr_sel.iloc[ii]['rmsd_mag']<mag_RMSD and curr_sel.iloc[ii]['rmsd_len']<len_RMSD:
                        print(curr_sel.iloc[ii]['solution_id'],'below sigma noise, SAVED')

                        shutil.copy(output_dir_optimized+os.sep+file_name_obs+'_sim_fit.json', file_json_save_results)
                        shutil.copy(output_dir_optimized+os.sep+file_name_obs+'_sim_fit.json', file_json_save_phys)
                        pd_selected_low_RMSD = curr_sel.iloc[ii].copy()
                        pd_selected_low_RMSD['solution_id']=file_json_save_phys

                        shutil.copy(df_obs_shower.iloc[0]['solution_id'], output_dir_optimized+os.sep+os.path.basename(df_obs_shower.iloc[0]['solution_id']))

                        shutil.copy(output_dir+os.sep+SAVE_SELECTION_FOLDER+os.sep+image_name , output_dir+os.sep+save_results_folder_events_plots+os.sep+image_name)

                        pd_datafram_PCA_selected_optimized = pd.concat([pd_datafram_PCA_selected_optimized, pd_selected_low_RMSD], axis=0)

                        continue

                    elif curr_sel.iloc[ii]['rmsd_mag']<mag_RMSD*1.1 and curr_sel.iloc[ii]['rmsd_len']<len_RMSD*1.1: # 95CI 1.96 z-score
                        print(curr_sel.iloc[ii]['solution_id'],'A little bit above the RMSD, try minor OPTIMIZATION')
                        shutil.copy(output_dir+os.sep+'AutoRefineFit_options.txt', output_dir_optimized+os.sep+'AutoRefineFit_options.txt')
                        update_sigma_values(output_dir_optimized+os.sep+'AutoRefineFit_options.txt', mag_noise_real, len_noise_real, False, False) # More_complex_fit=False, Custom_refinement=False

                    elif curr_sel.iloc[ii]['rmsd_mag']<mag_RMSD*2 and curr_sel.iloc[ii]['rmsd_len']<len_RMSD*2: # 99.99CI 1.96*2 z-score
                        print(curr_sel.iloc[ii]['solution_id'],'between 2 and 4 times the RMSD threshold, try major OPTIMIZATION')
                        shutil.copy(output_dir+os.sep+'AutoRefineFit_options.txt', output_dir_optimized+os.sep+'AutoRefineFit_options.txt')
                        update_sigma_values(output_dir_optimized+os.sep+'AutoRefineFit_options.txt', mag_noise_real, len_noise_real, True, True) # More_complex_fit=False, Custom_refinement=False

                    else:
                        print(curr_sel.iloc[ii]['solution_id'],'4 times above RMSD threshold, NO OPTIMIZATION and NOT SAVED')
                        
                        shutil.copy(output_dir_optimized+os.sep+file_name_obs+'_sim_fit.json', file_json_save_phys)

                        continue

                    shutil.copy(df_obs_shower.iloc[0]['solution_id'], output_dir_optimized+os.sep+os.path.basename(df_obs_shower.iloc[0]['solution_id']))

                    print('runing the optimization...')
                    # this creates a ew file called output_dir+os.sep+file_name_obs+'_sim_fit_fitted.json'
                    subprocess.run(
                        ['python', '-m', 'wmpl.MetSim.AutoRefineFit', 
                        output_dir_optimized, 'AutoRefineFit_options.txt', '-x'], 
                        # stdout=subprocess.PIPE, 
                        # stderr=subprocess.PIPE, 
                        text=True
                    )

                    # # save the 20230811_082648_sim_fit_fitted.json as a json file in the output_dir+os.sep+SAVE_SELECTION_FOLDER+os.sep+file_name_title[:23]+'_sim_fit_fitted.json'
                    # shutil.copy(output_dir_optimized+os.sep+file_name_obs+'_sim_fit_fitted.json', file_json_save_phys)

                    # Check if the output file exists before copying
                    if os.path.exists(output_dir_optimized+os.sep+file_name_obs+'_sim_fit_fitted.json'):
                        try:
                            shutil.copy(output_dir_optimized+os.sep+file_name_obs+'_sim_fit_fitted.json', file_json_save_phys)
                        except Exception as e:
                            # logging.exception(f"Failed to copy {optimized_file} to {file_json_save_phys}")
                            print(f"Failed to copy optimized file: {e}")
                            plt.close()
                            continue
                    else:
                        # logging.warning(f"Optimized file {optimized_file} not found.")
                        print(f"Optimized file not found. Skipping copy for {file_json_save_phys}")
                        plt.close()
                        continue

                else:
                    print('file '+file_json_save_phys+' already exist, read it...')

                _, gensim_data_optimized, pd_datafram_PCA_sim_optimized = run_simulation(file_json_save_phys, data_file_real, fit_funct)


                rmsd_mag, rmsd_vel, rmsd_lag, residuals_mag, residuals_vel, residuals_len, residual_time_pos, residual_height_pos , lag_km_sim = RMSD_calc_diff(gensim_data_optimized, data_file_real)
                BOOLchi_p_mag, BOOLchi_p_lag, chi_p_results, residuals_mag, residuals_vel, residuals_len, residual_time_pos, residual_height_pos , lag_km_sim = chiSquare_test_sim_real(gensim_data_optimized, data_file_real)

                # Interpolation on the fit data's height grid
                interp_ht_time = interp1d(data_file_real['height'], data_file_real['time'], kind='linear', bounds_error=False, fill_value='extrapolate')
                # Interpolated fit on data grid
                sim_time_pos = interp_ht_time(gensim_data_optimized['height'])

                # copy the data to the mode
                data_file_sim_opt = data_file.copy()
                data_file_sim_opt['time'] = sim_time_pos
                data_file_sim_opt['res_absolute_magnitudes'] = residuals_mag
                data_file_sim_opt['res_velocities'] = residuals_vel
                data_file_sim_opt['res_lag'] = residuals_len * 1000
                data_file_sim_opt['lag'] = lag_km_sim * 1000
                data_file_sim_opt['rmsd_mag'] = rmsd_mag
                data_file_sim_opt['rmsd_vel'] = rmsd_vel
                data_file_sim_opt['rmsd_len'] = rmsd_lag
                data_file_sim_opt['chi2_red_mag'] = chi_p_results['chi2_red_mag']
                data_file_sim_opt['chi2_red_lag'] = chi_p_results['chi2_red_lag']

                chi2_mag = data_file_sim_opt['chi2_red_mag']
                chi2_lag = data_file_sim_opt['chi2_red_lag']

                if Metsim_flag:
                    plot_data_with_residuals_and_real(mag_RMSD, len_RMSD*np.sqrt(2)/(1.0/fps), len_RMSD, fit_funct, data_file_real, data_file_real['name'].split(os.sep)[-1], image_name, output_dir+os.sep+SAVE_SELECTION_FOLDER, data_file_sim, 'Metsim', data_file_sim_opt, 'Optimized')
                else:
                    plot_data_with_residuals_and_real(mag_RMSD, len_RMSD*np.sqrt(2)/(1.0/fps), len_RMSD, fit_funct, data_file_real, data_file_real['name'].split(os.sep)[-1], image_name, output_dir+os.sep+SAVE_SELECTION_FOLDER, data_file_sim,'', data_file_sim_opt, 'Optimized')

            if chi2_mag >= 0.5 and chi2_mag <= 1.5 and chi2_lag >= 0.5 and chi2_lag <= 1.5: # rmsd_mag<mag_RMSD and rmsd_lag<len_RMSD:

                shutil.copy(output_dir+os.sep+SAVE_SELECTION_FOLDER+os.sep+image_name , output_dir+os.sep+save_results_folder_events_plots+os.sep+image_name) # _mag$_{RMSD}$'+str(round(rmsd_mag,2))+'_RMSDlen'+str(round(rmsd_lag,2))+'_Heigh_MagVelCoef
                
                if run_optimization:
                    if not os.path.isfile(file_json_save_phys):
                        # output_dir+os.sep+file_name_obs+'_sim_fit.json'
                        shutil.copy(output_dir_optimized+os.sep+file_name_obs+'_sim_fit_fitted.json', file_json_save_results)
                        shutil.copy(output_dir_optimized+os.sep+file_name_obs+'_sim_fit_fitted.json', file_json_save_phys)
                    # change solution_id of pd_datafram_PCA_sim_optimized to file_json_save_results
                    pd_datafram_PCA_sim_optimized['solution_id']=file_json_save_phys
                    pd_datafram_PCA_selected_optimized = pd.concat([pd_datafram_PCA_selected_optimized, pd_datafram_PCA_sim_optimized], axis=0)
                else:
                    pd_datafram_PCA_sim = array_to_pd_dataframe_PCA(data_file, data_file_real)
                    pd_datafram_PCA_sim['solution_id']=file_json_save_phys_NOoptimized
                    shutil.copy(file_json_save_phys_NOoptimized, output_dir+os.sep+save_results_folder_events_plots+os.sep+file_name_title)
                    # remove curr_sel.iloc[[ii]].drop(columns=['rmsd_mag', 'rmsd_len', 'solution_id_dist', 'distance_meteor', 'distance_mean']) rmsd_mag	rmsd_len solution_id_dist	distance_meteor	distance_mean
                    pd_datafram_PCA_selected_optimized = pd.concat([pd_datafram_PCA_selected_optimized, pd_datafram_PCA_sim], axis=0)

            else:
                print(curr_sel.iloc[ii]['solution_id'],'above the noise, NOT SAVED')

        return pd_datafram_PCA_selected_optimized

    # except FileNotFoundError as fnf_error:
    #     print(f"File not found: {fnf_error}")
    #     # Handle the error or log it
    #     # Optionally, continue or exit the function
    #     return pd.DataFrame()
 
    # except Exception as e:
    #     print(f"An unexpected error occurred: {e}")
    #     # Handle other exceptions
    #     return pd.DataFrame()



# RMSD ###########################################################################################


def RMSD_calc_diff(sim_file_data, real_funct_data):

    # copy the data
    sim_file = copy.deepcopy(sim_file_data)
    real_funct = copy.deepcopy(real_funct_data)
    
    # Check if data_file and fit_funct are not None
    if sim_file is None or real_funct is None:
        print('Error: data_file or fit_funct is None')
        return 9999, 9999, 9999, 9999, 9999, 9999, 0, 100, 0

    # Check if required keys are present in data_file and fit_funct
    required_keys = ['height', 'absolute_magnitudes', 'time', 'velocities', 'lag']
    for key in required_keys:
        if key not in sim_file or key not in real_funct:
            print(f'Error: Missing key {key} in data_file or fit_funct')
            return 9999, 9999, 9999, 9999, 9999, 9999, 0, 100, 0

    # Convert lists to arrays and adjust units
    height_km_sim = np.array(sim_file['height']) / 1000
    abs_mag_sim = np.array(sim_file['absolute_magnitudes'])
    time_sim= np.array(sim_file['time'])
    vel_kms_sim = np.array(sim_file['velocities']) / 1000
    len_km_sim = np.array(sim_file['length']) / 1000
    lag_kms_sim = np.array(sim_file['lag']) / 1000


    # Convert lists to arrays and adjust units
    height_km_real = np.array(real_funct['height']) / 1000
    abs_mag_real = np.array(real_funct['absolute_magnitudes'])
    time_real = np.array(real_funct['time'])
    vel_kms_real = np.array(real_funct['velocities']) / 1000
    len_km_real = np.array(real_funct['length']) / 1000
    # lag_kms_real = len_km_real - (vel_kms_sim[0] * time_real)
    # wrong_lag = np.array(real_funct['lag']) / 1000
    lag_kms_real = np.array(real_funct['lag']) / 1000
    # # start from 0
    # lag_kms_real = lag_kms_real - lag_kms_real[0]

    if 'v_init' in sim_file:
        lag_kms_sim = len_km_sim - (real_funct['v_init']/1000 * time_sim)
    else:
        lag_kms_sim = len_km_sim - (vel_kms_real[0] * time_sim)

    # Define the overlapping range for time
    common_height_min = max(height_km_sim.min(), height_km_real.min())
    common_height_max = min(height_km_sim.max(), height_km_real.max())

    if common_height_min >= common_height_max:
        print('No overlap in time')
        return 9999, 9999, 9999, 9999, 9999, 9999, time_real[0], height_km_real[0], 0

    # Restrict fit_funct data to the overlapping time range
    valid_fit_indices = (height_km_real >= common_height_min) & (height_km_real <= common_height_max)
    if not np.any(valid_fit_indices):
        print('No valid fit data in overlapping time range')
        return 9999, 9999, 9999, 9999, 9999, 9999, time_real[0], height_km_real[0], 0


    # Interpolation on the fit data's height grid
    interp_ht_absmag= interp1d(height_km_sim, abs_mag_sim, kind='linear', bounds_error=False, fill_value='extrapolate')
    interp_ht_time = interp1d(height_km_sim, time_sim, kind='linear', bounds_error=False, fill_value='extrapolate')
    # Interpolated fit on data grid
    abs_mag_sim_interp = interp_ht_absmag(height_km_real)
    time_sim_interp = interp_ht_time(height_km_real)

    magnitude_differences = abs_mag_real - abs_mag_sim_interp

    # Interpolation on the fit data's time grid
    interp_t_vel = interp1d(time_sim, vel_kms_sim, kind='linear', bounds_error=False, fill_value='extrapolate')
    interp_t_lag = interp1d(time_sim, lag_kms_sim, kind='linear', bounds_error=False, fill_value='extrapolate')
    # Interpolated fit on data grid
    vel_kms_sim_interp = interp_t_vel(time_sim_interp)
    lag_kms_sim_interp = interp_t_lag(time_sim_interp)

    velocity_differences = vel_kms_real - vel_kms_sim_interp
    lag_differences = lag_kms_real - lag_kms_sim_interp

    residual_time_pos = time_sim_interp
    residual_height_pos = height_km_real
        
    # copute RMSD
    rmsd_mag = np.sqrt(np.mean(magnitude_differences**2))
    rmsd_vel = np.sqrt(np.mean(velocity_differences**2))
    rmsd_lag = np.sqrt(np.mean(lag_differences**2))
    
    # check if threshold_mag exists
    if 'rmsd_mag' in real_funct:
        threshold_mag = real_funct['rmsd_mag']
    else:
        threshold_mag = 0
    if 'rmsd_vel' in real_funct:
        threshold_vel = real_funct['rmsd_vel']
    else:
        threshold_vel = 0
    if 'rmsd_len' in real_funct:
        threshold_lag = real_funct['rmsd_len']
        # print('threshold_lag',threshold_lag)
        # print('lag_differences',lag_differences)
        # exceeds_threshold = np.abs(lag_differences) > threshold_lag*3
        # if np.any(exceeds_threshold):
        #     exceeding_values = lag_differences[exceeds_threshold]
        #     print(f'Lag differences exceeding {threshold_lag*3} found: {len(exceeding_values)}')
        #     rmsd_lag = 9999  
    else:
        threshold_lag = 0
    if 'fps' in real_funct:
        fps = real_funct['fps']
    else:
        fps = 32
    
    max_diff_threshold = MAX_MAG_DIFF
    # Identify which differences exceed the maximum allowed difference
    if threshold_mag*2 > MAX_MAG_DIFF:
        max_diff_threshold = threshold_mag*2
        exceeds_threshold = np.abs(magnitude_differences) > max_diff_threshold
    else:
        exceeds_threshold = np.abs(magnitude_differences) > max_diff_threshold

    if np.any(exceeds_threshold):
        exceeding_values = magnitude_differences[exceeds_threshold]
        print(f'Magnitude differences exceeding {max_diff_threshold} found: {len(exceeding_values)}')
        rmsd_mag = 9999                                                              

    # Handle NaNs in RMSD calculations
    if np.isnan(rmsd_mag):
        rmsd_mag = 9999
    if np.isnan(rmsd_vel):
        rmsd_vel = 9999
    if np.isnan(rmsd_lag):
        rmsd_lag = 9999


    # sigma values estimate from the data
    sigma_abs_mag = np.std(abs_mag_real - abs_mag_sim_interp)
    sigma_vel = np.std(vel_kms_real - vel_kms_sim_interp)
    sigma_lag = np.std(lag_kms_real - lag_kms_sim_interp)
        
    # Compute the chi-squared statistics
    chi2_mag = np.sum((magnitude_differences / sigma_abs_mag) ** 2)
    chi2_vel = np.sum((velocity_differences / sigma_vel) ** 2)
    chi2_lag = np.sum((lag_differences / sigma_lag) ** 2)

    # Degrees of freedom (assuming no parameters estimated from data)
    dof_mag = len(abs_mag_real) - 0  # Adjust if you have fitted parameters
    dof_vel = len(vel_kms_real) - 0
    dof_lag = len(lag_kms_real) - 0

    # Reduced chi-squared
    chi2_red_mag = chi2_mag / dof_mag
    chi2_red_vel = chi2_vel / dof_vel
    chi2_red_lag = chi2_lag / dof_lag

    # Interpret the results
    print(f'Magnitude reduced chi-squared: {chi2_red_mag:.2f}')
    print(f'Velocity reduced chi-squared: {chi2_red_vel:.2f}')
    print(f'Lag reduced chi-squared: {chi2_red_lag:.2f}')

    p_value_mag = 1 - chi2.cdf(chi2_mag, dof_mag)
    p_value_vel = 1 - chi2.cdf(chi2_vel, dof_vel)
    p_value_lag = 1 - chi2.cdf(chi2_lag, dof_lag)

    # Define the significance level (alpha)
    alpha = 0.05  # Corresponds to 95% confidence level

    # Define thresholds
    chi2_red_threshold_lower = 0.5  # Lower bound for reduced chi-squared
    chi2_red_threshold_upper = 1.5  # Upper bound for reduced chi-squared

    # check if any is nan and if so substitute tha with 9999
    if np.isnan(chi2_red_mag):
        chi2_red_mag = 9999
    if np.isnan(chi2_red_vel):
        chi2_red_vel = 9999
    if np.isnan(chi2_red_lag):
        chi2_red_lag = 9999
    if np.isnan(p_value_mag):
        p_value_mag = 9999
    if np.isnan(p_value_vel):
        p_value_vel = 9999
    if np.isnan(p_value_lag):
        p_value_lag = 9999

    # Initialize results dictionary
    chi2_results = {
        'chi2_mag': chi2_mag,
        'chi2_red_mag': chi2_red_mag,
        'p_value_mag': p_value_mag,
        'chi2_vel': chi2_vel,
        'chi2_red_vel': chi2_red_vel,
        'p_value_vel': p_value_vel,
        'chi2_lag': chi2_lag,
        'chi2_red_lag': chi2_red_lag,
        'p_value_lag': p_value_lag,
        'fit_quality_mag': False,
        'fit_quality_vel': False,
        'fit_quality_lag': False
    }

    # Function to evaluate fit quality
    def evaluate_fit(chi2_red, p_value):
        if (chi2_red >= chi2_red_threshold_lower and chi2_red <= chi2_red_threshold_upper): #  and (p_value >= alpha)
            return True
        else:
            return False

    # Evaluate fit quality for magnitude
    chi2_results['fit_quality_mag'] = evaluate_fit(chi2_red_mag, p_value_mag)
    # # Evaluate fit quality for velocity
    chi2_results['fit_quality_vel'] = evaluate_fit(chi2_red_vel, p_value_vel)
    # Evaluate fit quality for lag
    chi2_results['fit_quality_lag'] = evaluate_fit(chi2_red_lag, p_value_lag)

    # print(f'RMSD Magnitude: {rmsd_mag:.4f}')
    # print(f'RMSD Velocity: {rmsd_vel:.4f}')
    # print(f'RMSD Lag: {rmsd_lag:.4f}')
    # # Plotting the results
    # plt.figure(figsize=(10, 5))
    # # create 3 subplots
    # plt.subplot(1, 3, 1)
    # # plot both the abs_mag_data_interp against the height and the vel_kms_data_interp angainst time and the lag_sampled_adj gainst time
    # plt.plot(abs_mag_sim_interp, height_km_real, 'k-', label='simulated')
    # # against the abs_mag_fit and the height_km_fit
    # plt.plot(abs_mag_real, height_km_real, 'b.', label='real')
    # # put also the range of the threshold_mag
    # plt.plot(abs_mag_sim_interp-threshold_mag, height_km_real, 'g--', label='threshold_mag')
    # plt.plot(abs_mag_sim_interp+threshold_mag, height_km_real, 'g--')
    # plt.ylabel('Height [km]')
    # plt.xlabel('Absolute Magnitude')
    # plt.grid()
    # plt.legend()

    # # plt.subplot(1, 3, 2)
    # # plt.plot(vel_kms_sim, height_km_sim, 'k-', label='simulated')
    # # plt.plot(vel_kms_real, height_km_real, 'b.', label='real')
    # # # Plot the range of the threshold_vel
    # # plt.plot(vel_kms_sim - threshold_vel, height_km_sim, 'g--', label='threshold_vel')
    # # plt.plot(vel_kms_sim + threshold_vel, height_km_sim, 'g--')
    # # plt.xlabel('Velocity [km/s]')
    # # plt.ylabel('Height [s]')
    # # plt.grid()
    # # plt.legend()

    # # plt.subplot(1, 3, 3)
    # # plt.plot(lag_kms_sim, height_km_sim, 'k-', label='simulated')
    # # plt.plot(lag_kms_real, height_km_real, 'b.', label='new_real')
    # # plt.plot(wrong_lag, height_km_real, 'rx', label='old_real')
    # # # Plot the range of the threshold_lag
    # # plt.plot(lag_kms_sim - threshold_lag, height_km_sim, 'g--', label='threshold_lag')
    # # plt.plot(lag_kms_sim + threshold_lag, height_km_sim, 'g--')
    # # plt.xlabel('Lag [km]')
    # # plt.ylabel('Height [s]')
    # # plt.grid()
    # # plt.legend()


    # plt.subplot(1, 3, 2)
    # plt.plot(time_real, vel_kms_sim_interp,'k-', label='simulated')
    # plt.plot(time_real, vel_kms_real, 'b.', label='real')
    # # put also the range of the threshold_vel
    # plt.plot(time_real, vel_kms_sim_interp-threshold_vel, 'g--', label='threshold_vel')
    # plt.plot(time_real, vel_kms_sim_interp+threshold_vel, 'g--')
    # plt.xlabel('Time [s]')
    # plt.ylabel('Velocity [km/s]')
    # plt.grid()
    # plt.legend()

    # plt.subplot(1, 3, 3)
    # # plt.plot(time_data, lag_sampled_adj, 'k-', label='lag_sampled_adj')
    # plt.plot(time_real, lag_kms_sim_interp, 'k-', label='simulated')
    # plt.plot(time_real, lag_kms_real, 'b.', label='new_real')
    # # plt.plot(time_real, wrong_lag, 'rx', label='old_real')
    # # put also the range of the threshold_lag 
    # plt.plot(time_real, lag_kms_sim_interp-threshold_lag, 'g--', label='threshold_lag')
    # plt.plot(time_real, lag_kms_sim_interp+threshold_lag, 'g--')
    # plt.xlabel('Time [s]')
    # plt.ylabel('Lag [km]')
    # plt.grid()
    # plt.legend()

    # # more space between the plots
    # plt.tight_layout()
    # # show the plot
    # plt.show()

    return rmsd_mag, rmsd_vel, rmsd_lag, magnitude_differences, velocity_differences, lag_differences, residual_time_pos, residual_height_pos, lag_kms_sim


def compute_chi2_red_thresholds(confidence_level, degrees_of_freedom): # 0.95, len(residuals_mag)
    # Significance level
    alpha = 1 - confidence_level  # e.g., 0.10 for 90% confidence level
    
    # Lower and upper percentiles
    lower_percentile = alpha / 2
    upper_percentile = 1 - (alpha / 2)
    
    # Critical chi-squared values
    chi2_lower = chi2.ppf(lower_percentile, degrees_of_freedom)
    chi2_upper = chi2.ppf(upper_percentile, degrees_of_freedom)
    
    # Thresholds for reduced chi-squared
    chi2_red_threshold_lower = chi2_lower / degrees_of_freedom
    chi2_red_threshold_upper = chi2_upper / degrees_of_freedom
    
    return chi2_red_threshold_lower, chi2_red_threshold_upper


def chiSquare_test_sim_real(sim_file_data, real_funct_data):

    # copy the data
    sim_file = copy.deepcopy(sim_file_data)
    real_funct = copy.deepcopy(real_funct_data)
    
    # Check if data_file and fit_funct are not None
    if sim_file is None or real_funct is None:
        print('Error: data_file or fit_funct is None')
        return 9999, 9999, 9999, 9999, 9999, 9999, 0, 100, 0

    # Check if required keys are present in data_file and fit_funct
    required_keys = ['height', 'absolute_magnitudes', 'time', 'velocities', 'lag']
    for key in required_keys:
        if key not in sim_file or key not in real_funct:
            print(f'Error: Missing key {key} in data_file or fit_funct')
            return 9999, 9999, 9999, 9999, 9999, 9999, 0, 100, 0

    # Convert lists to arrays and adjust units
    height_km_sim = np.array(sim_file['height']) / 1000
    abs_mag_sim = np.array(sim_file['absolute_magnitudes'])
    time_sim= np.array(sim_file['time'])
    vel_kms_sim = np.array(sim_file['velocities']) / 1000
    len_km_sim = np.array(sim_file['length']) / 1000
    lag_kms_sim = np.array(sim_file['lag']) / 1000


    # Convert lists to arrays and adjust units
    height_km_real = np.array(real_funct['height']) / 1000
    abs_mag_real = np.array(real_funct['absolute_magnitudes'])
    time_real = np.array(real_funct['time'])
    vel_kms_real = np.array(real_funct['velocities']) / 1000
    len_km_real = np.array(real_funct['length']) / 1000
    # lag_kms_real = len_km_real - (vel_kms_sim[0] * time_real)
    # wrong_lag = np.array(real_funct['lag']) / 1000
    lag_kms_real = np.array(real_funct['lag']) / 1000
    # # start from 0
    # lag_kms_real = lag_kms_real - lag_kms_real[0]

    if 'v_init' in sim_file:
        lag_kms_sim = len_km_sim - (real_funct['v_init']/1000 * time_sim)
    else:
        lag_kms_sim = len_km_sim - (vel_kms_real[0] * time_sim)

    # Define the overlapping range for time
    common_height_min = max(height_km_sim.min(), height_km_real.min())
    common_height_max = min(height_km_sim.max(), height_km_real.max())

    if common_height_min >= common_height_max:
        print('No overlap in time')
        return 9999, 9999, 9999, 9999, 9999, 9999, time_real[0], height_km_real[0], 0

    # Restrict fit_funct data to the overlapping time range
    valid_fit_indices = (height_km_real >= common_height_min) & (height_km_real <= common_height_max)
    if not np.any(valid_fit_indices):
        print('No valid fit data in overlapping time range')
        return 9999, 9999, 9999, 9999, 9999, 9999, time_real[0], height_km_real[0], 0


    # Interpolation on the fit data's height grid
    interp_ht_absmag= interp1d(height_km_sim, abs_mag_sim, kind='linear', bounds_error=False, fill_value='extrapolate')
    interp_ht_time = interp1d(height_km_sim, time_sim, kind='linear', bounds_error=False, fill_value='extrapolate')
    # Interpolated fit on data grid
    abs_mag_sim_interp = interp_ht_absmag(height_km_real)
    time_sim_interp = interp_ht_time(height_km_real)

    magnitude_differences = abs_mag_real - abs_mag_sim_interp

    # Interpolation on the fit data's time grid
    interp_t_vel = interp1d(time_sim, vel_kms_sim, kind='linear', bounds_error=False, fill_value='extrapolate')
    interp_t_lag = interp1d(time_sim, lag_kms_sim, kind='linear', bounds_error=False, fill_value='extrapolate')
    # Interpolated fit on data grid
    vel_kms_sim_interp = interp_t_vel(time_sim_interp)
    lag_kms_sim_interp = interp_t_lag(time_sim_interp)

    velocity_differences = vel_kms_real - vel_kms_sim_interp
    lag_differences = lag_kms_real - lag_kms_sim_interp

    residual_time_pos = time_sim_interp
    residual_height_pos = height_km_real

    # sigma values estimate from the data
    sigma_abs_mag = np.std(abs_mag_real - abs_mag_sim_interp)
    sigma_vel = np.std(vel_kms_real - vel_kms_sim_interp)
    sigma_lag = np.std(lag_kms_real - lag_kms_sim_interp)
        
    # Compute the chi-squared statistics
    chi2_mag = np.sum((magnitude_differences / sigma_abs_mag) ** 2)
    chi2_vel = np.sum((velocity_differences / sigma_vel) ** 2)
    chi2_lag = np.sum((lag_differences / sigma_lag) ** 2)

    # Degrees of freedom (assuming no parameters estimated from data)
    dof_mag = len(abs_mag_real) - 0  # Adjust if you have fitted parameters
    dof_vel = len(vel_kms_real) - 0
    dof_lag = len(lag_kms_real) - 0

    # Reduced chi-squared
    chi2_red_mag = chi2_mag / dof_mag
    chi2_red_vel = chi2_vel / dof_vel
    chi2_red_lag = chi2_lag / dof_lag

    # Interpret the results
    print(f'Magnitude reduced chi-squared: {chi2_red_mag:.2f}')
    print(f'Velocity reduced chi-squared: {chi2_red_vel:.2f}')
    print(f'Lag reduced chi-squared: {chi2_red_lag:.2f}')

    p_value_mag = 1 - chi2.cdf(chi2_mag, dof_mag)
    p_value_vel = 1 - chi2.cdf(chi2_vel, dof_vel)
    p_value_lag = 1 - chi2.cdf(chi2_lag, dof_lag)

    # Define the significance level (alpha)
    alpha = 0.05  # Corresponds to 95% confidence level

    # Define thresholds
    chi2_red_threshold_lower = 0.5  # Lower bound for reduced chi-squared
    chi2_red_threshold_upper = 1.5  # Upper bound for reduced chi-squared

    # check if any is nan and if so substitute tha with 9999
    if np.isnan(chi2_red_mag):
        chi2_red_mag = 9999
    if np.isnan(chi2_red_vel):
        chi2_red_vel = 9999
    if np.isnan(chi2_red_lag):
        chi2_red_lag = 9999
    if np.isnan(p_value_mag):
        p_value_mag = 9999
    if np.isnan(p_value_vel):
        p_value_vel = 9999
    if np.isnan(p_value_lag):
        p_value_lag = 9999

    # Initialize results dictionary
    test_results = {
        'chi2_mag': chi2_mag,
        'chi2_red_mag': chi2_red_mag,
        'p_value_mag': p_value_mag,
        'chi2_vel': chi2_vel,
        'chi2_red_vel': chi2_red_vel,
        'p_value_vel': p_value_vel,
        'chi2_lag': chi2_lag,
        'chi2_red_lag': chi2_red_lag,
        'p_value_lag': p_value_lag,
        'fit_quality_mag': False,
        # 'fit_quality_vel': False,
        'fit_quality_lag': False
    }

    # Function to evaluate fit quality
    def evaluate_fit(chi2_red, p_value):
        if (chi2_red >= chi2_red_threshold_lower and chi2_red <= chi2_red_threshold_upper): #  and (p_value >= alpha)
            return True
        else:
            return False

    # Evaluate fit quality for magnitude
    test_results['fit_quality_mag'] = evaluate_fit(chi2_red_mag, p_value_mag)
    # # Evaluate fit quality for velocity
    # test_results['fit_quality_vel'] = evaluate_fit(chi2_red_vel, p_value_vel)
    # Evaluate fit quality for lag
    test_results['fit_quality_lag'] = evaluate_fit(chi2_red_lag, p_value_lag)

    return test_results['fit_quality_mag'], test_results['fit_quality_lag'], test_results, magnitude_differences, velocity_differences, lag_differences, residual_time_pos, residual_height_pos, lag_kms_sim






# MAIN FUNCTION ##################################################################################


def main_PhysUncert(trajectory_file, file_name, input_folder, output_folder, trajectory_Metsim_file, cml_args_user):
    #copy cml_args_user
    cml_args = copy.deepcopy(cml_args_user)

    print('processing file:',file_name)
    print(trajectory_file)
    print(input_folder)
    print(output_folder)
    print(trajectory_Metsim_file)

    if cml_args.delete_all:
        # if presen the output_folder then delete all the files in the folder
        if os.path.isdir(output_folder):
            # remove all the files in the folder
            shutil.rmtree(output_folder)
            print('All files in the output folder have been deleted.')
    elif cml_args.delete_old:

        # Regex pattern for folders named vXX
        folder_pattern = re.compile(r"^v\d{2}$")

        # Patterns for files to keep
        file_patterns = ["_obs.csv", "_sim.csv"]

        # Traverse the directory
        for item in os.listdir(output_folder):
            item_path = os.path.join(output_folder, item)
            
            # Check if the item is a folder and matches the pattern
            if os.path.isdir(item_path):
                if not folder_pattern.match(item):
                    # Delete folder if it doesn't match the pattern
                    shutil.rmtree(item_path)
            elif os.path.isfile(item_path):
                # Check if the file ends with any of the allowed patterns
                if not any(item.endswith(pattern) for pattern in file_patterns):
                    # Delete file if it doesn't match the allowed patterns
                    os.remove(item_path)

        print("Cleanup completed!")

    # add to save_res_fin_folder the file_name
    save_res_fin_folder=SAVE_RESULTS_FINAL_FOLDER+file_name
    save_results_folder=save_res_fin_folder
    save_results_folder_events_plots = save_results_folder+os.sep+'events_plots'

    flag_manual_metsim=True
    # check if it ends with _first_guess.json
    if trajectory_Metsim_file.endswith('_first_guess.json'):
        flag_manual_metsim=False

    start_time = time.time()

    # chek if input_folder+os.sep+file_name+NAME_SUFX_CSV_OBS exist
    if os.path.isfile(output_folder+os.sep+file_name+NAME_SUFX_CSV_OBS):
        # read the csv file
        trajectory_file = output_folder+os.sep+file_name+NAME_SUFX_CSV_OBS

    # check if the output_folder exists
    if not os.path.isdir(output_folder):
        mkdirP(output_folder)

    # check if the input_folder exists if the csv file has been already created
    if trajectory_file.endswith('.csv'):
        # read the csv file
        pd_dataframe_PCA_obs_real = pd.read_csv(trajectory_file)
        # check the column name solution_id	and see if it matches a file i the folder
        if not input_folder in pd_dataframe_PCA_obs_real['solution_id'][0]:
            # if the solution_id is in the name of the file then the file is the real data
            print('The folder of the csv file is different')
            # check if the file is present in the folder
            if not os.path.isfile(pd_dataframe_PCA_obs_real['solution_id'][0]):
                print()
                print('--- MODIFY OLD CSV FILE PATH ---')
                # take the first element pd_dataframe_PCA_obs_real['solution_id'][0] and take only the path
                old_input_folder = os.path.split(pd_dataframe_PCA_obs_real['solution_id'][0])[0]
                # run the update_solution_ids function
                print('old_input_folder',old_input_folder)
                update_solution_ids(old_input_folder, input_folder)

    print()

    ######################### OBSERVATION ###############################
    print('--- OBSERVATION ---')

    mkdirP(output_folder+os.sep+save_results_folder)
    mkdirP(output_folder+os.sep+save_results_folder_events_plots)
    mkdirP(output_folder+os.sep+SAVE_SELECTION_FOLDER)

    # check the extension of the file if it already present the csv file meas it has been aleady processed
    if trajectory_file.endswith('.csv'):
        # read the csv file
        pd_dataframe_PCA_obs_real = pd.read_csv(trajectory_file)

        if pd_dataframe_PCA_obs_real['type'][0] != 'Observation' and pd_dataframe_PCA_obs_real['type'][0] != 'Observation_sim':
            # raise an error saing that the type is wrong and canot be processed by PCA
            raise ValueError('Type of the csv file is wrong and canot be processed by script.')
    
        if pd_dataframe_PCA_obs_real['solution_id'][0].endswith('.pickle'):
            # read the pickle file
            gensim_data_obs = read_pickle_reduction_file(pd_dataframe_PCA_obs_real['solution_id'][0])

        # json file
        elif pd_dataframe_PCA_obs_real['solution_id'][0].endswith('.json') and pd_dataframe_PCA_obs_real['type'][0] != 'Observation_sim': 
            # read the json file with noise
            gensim_data_obs = read_with_noise_GenerateSimulations_output(pd_dataframe_PCA_obs_real['solution_id'][0], fps)

        else:
            # raise an error if the file is not a csv, pickle or json file
            raise ValueError('File format not supported. Please provide a csv, pickle or json file.')

        rmsd_t0_lag, rmsd_pol_mag, fit_pol_mag, fitted_lag_t0_lag, fit_funct = find_noise_of_data(gensim_data_obs, cml_args.fps, False, output_folder+os.sep+save_results_folder, file_name)

        print('read the csv file:',trajectory_file)

    else:

        if trajectory_file.endswith('.pickle'):
            # read the pickle file
            gensim_data_obs = read_pickle_reduction_file(trajectory_file) #,trajectory_Metsim_file

        # json file
        elif trajectory_file.endswith('.json'): 
            # read the json file with noise
            gensim_data_obs = read_with_noise_GenerateSimulations_output(trajectory_file, fps)
            
        else:
            # raise an error if the file is not a csv, pickle or json file
            raise ValueError('File format not supported. Please provide a csv, pickle or json file.')

        if cml_args.save_test_plot:
            # run generate observation realization with the gensim_data_obs
            rmsd_t0_lag, rmsd_pol_mag, fit_pol_mag, fitted_lag_t0_lag, fit_funct, fig, ax = find_noise_of_data(gensim_data_obs, cml_args.fps, cml_args.save_test_plot, output_folder+os.sep+save_results_folder, file_name)
            # make the results_list to incorporate all rows of pd_dataframe_PCA_obs_real
            results_list = []
            for ii in range(cml_args.nobs):
                results_pd = generate_observation_realization(gensim_data_obs, rmsd_t0_lag, rmsd_pol_mag, fit_funct,'realization_'+str(ii+1), fps, fig, ax, cml_args.save_test_plot) 
                results_list.append(results_pd)
                print()

            # plot noisy area around vel_kms for vel_noise for the fix height_km
            ax[0].fill_betweenx(np.array(fit_funct['height'])/1000, np.array(fit_funct['absolute_magnitudes'])-rmsd_pol_mag, np.array(fit_funct['absolute_magnitudes'])+rmsd_pol_mag, color='lightgray', alpha=0.5)

            # plot noisy area around vel_kms for vel_noise for the fix height_km
            # ax[1].fill_between(np.array(fit_funct['time']), np.array(fit_funct['velocities'])/1000-(rmsd_t0_lag/1000/(1/fps)), np.array(fit_funct['velocities'])/1000+(rmsd_t0_lag/1000/(1/fps)), color='lightgray', alpha=0.5, label='Std.dev. realizations')
            ax[1].fill_between(np.array(fit_funct['time']), np.array(fit_funct['velocities'])/1000-(rmsd_t0_lag/1000*np.sqrt(2)/(1/fps)), np.array(fit_funct['velocities'])/1000+(rmsd_t0_lag/1000*np.sqrt(2)/(1/fps)), color='lightgray', alpha=0.5, label='Std.dev. realizations')

            ax[0].plot(np.array(fit_funct['absolute_magnitudes']), np.array(fit_funct['height'])/1000, 'k--')
            ax[1].plot(np.array(fit_funct['time']), np.array(fit_funct['velocities'])/1000, 'k--', label='Fit')

            # Save the figure as file with instead of _trajectory.pickle it has file+std_dev.png on the desktop
            plt.savefig(output_folder+os.sep+file_name+'obs_realizations.png', dpi=300)

            plt.close()

        else:      
            rmsd_t0_lag, rmsd_pol_mag, fit_pol_mag, fitted_lag_t0_lag, fit_funct = find_noise_of_data(gensim_data_obs, cml_args.fps, False, output_folder+os.sep+save_results_folder, file_name)       
            input_list_obs = [[gensim_data_obs, rmsd_t0_lag, rmsd_pol_mag, fit_funct,'realization_'+str(ii+1), fps] for ii in range(cml_args.nobs)]
            results_list = domainParallelizer(input_list_obs, generate_observation_realization, cores=cml_args.cores)
            print()
        
        pd_dataframe_PCA_obs_real = array_to_pd_dataframe_PCA(gensim_data_obs)
        pd_dataframe_PCA_obs_real['type'] = 'Observation'

        df_obs_realiz = pd.concat(results_list)
        pd_dataframe_PCA_obs_real = pd.concat([pd_dataframe_PCA_obs_real, df_obs_realiz])
        # re index the dataframe
        pd_dataframe_PCA_obs_real.reset_index(drop=True, inplace=True)

        # check if there is a column with the name 'mass'
        if 'mass' in pd_dataframe_PCA_obs_real.columns:
            #delete from the real_data panda dataframe mass rho sigma
            pd_dataframe_PCA_obs_real = pd_dataframe_PCA_obs_real.drop(columns=['mass', 'rho', 'sigma', 'erosion_height_start', 'erosion_coeff', 'erosion_mass_index', 'erosion_mass_min', 'erosion_mass_max', 'erosion_range', 'erosion_energy_per_unit_cross_section', 'erosion_energy_per_unit_mass'])
        
        # add to all the rows the rmsd_mag_obs, rmsd_lag_obs, RMSD cannot work for noisy simulations because of interp
        pd_dataframe_PCA_obs_real['rmsd_mag'] = rmsd_pol_mag
        pd_dataframe_PCA_obs_real['rmsd_len'] = rmsd_t0_lag/1000
        if flag_manual_metsim:
            simulation_MetSim_object, gensim_data_Metsim, pd_datafram_PCA_sim_Metsim = run_simulation(trajectory_Metsim_file, gensim_data_obs, fit_funct)
            # add pd_datafram_PCA_sim_Metsim['v_init_180km'] to pd_dataframe_PCA_obs_real
            pd_dataframe_PCA_obs_real['v_init_180km'] = pd_datafram_PCA_sim_Metsim['v_init_180km'].iloc[0]

        pd_dataframe_PCA_obs_real.to_csv(output_folder+os.sep+file_name+NAME_SUFX_CSV_OBS, index=False)
        # print saved csv file
        print()
        print('saved obs csv file:',output_folder+os.sep+file_name+NAME_SUFX_CSV_OBS)


    # if flag_manual_metsim==False:
    #     # load the first guess file and add "P_0m": 935.0 and the initial velocity
    #     load the json file
    #     with open(trajectory_Metsim_file) as f:
    #         gensim_data_obs = json.load(f)
    #     # add the P_0m and v_init_180km
    #         gensim_data_obs['P_0m'] = 935.0
    #         gensim_data_obs['v_init_180km'] = 180000.0
    #         # save the file
    #         with open(trajectory_Metsim_file, 'w') as f:
    #             json.dump(gensim_data_obs, f, indent=4)
    

    # rmsd
    # print('before rmsd_pol_mag', rmsd_pol_mag)
    # print('before rmsd_t0_lag', rmsd_t0_lag/1000)
    # rmsd_pol_mag, rmsd_vel, rmsd_t0_lag, _, _, _, _, _ , _ = RMSD_calc_diff(gensim_data_obs, fit_funct)
    # print('after rmsd_pol_mag', rmsd_pol_mag)
    # print('after rmsd_t0_lag', rmsd_t0_lag)

    alpha = 1 - cml_args.conf_lvl/100
    cml_args.conf_lvl=cml_args.conf_lvl/100
    z_score = norm.ppf(1 - (1 - cml_args.conf_lvl) / 2)
    print('z_score:',z_score)
    if cml_args.mag_rmsd != 0:
        # set the value of rmsd_pol_mag=rmsd_mag_obs and len_RMSD=rmsd_lag_obs*conf_lvl
        rmsd_pol_mag = cml_args.mag_rmsd
    if cml_args.len_rmsd != 0:
        if rmsd_t0_lag>1:
            # set the value of rmsd_t0_lag=rmsd_mag_obs and len_RMSD=rmsd_lag_obs*conf_lvl
            rmsd_t0_lag = cml_args.len_rmsd
        else:
            # keep it in m instead of km
            rmsd_t0_lag = cml_args.len_rmsd*1000

    if rmsd_pol_mag<CAMERA_SENSITIVITY_LVL_MAG:
        # rmsd_pol_mag if below 0.1 print the value and set it to 0.1
        print('below the sensitivity level RMSD required, real RMSD mag:',rmsd_pol_mag)
        print('set the RMSD mag to',CAMERA_SENSITIVITY_LVL_MAG)
        rmsd_pol_mag = CAMERA_SENSITIVITY_LVL_MAG

    if rmsd_t0_lag<CAMERA_SENSITIVITY_LVL_LEN:
        # rmsd_pol_mag if below 0.1 print the value and set it to 0.1
        print('below the sensitivity level RMSD required, real RMSD len:',rmsd_t0_lag)
        print('set the RMSD len to',CAMERA_SENSITIVITY_LVL_LEN)
        rmsd_t0_lag = CAMERA_SENSITIVITY_LVL_LEN

    gensim_data_obs['fps'] = cml_args.fps
    gensim_data_obs['rmsd_mag'] = rmsd_pol_mag
    gensim_data_obs['rmsd_len'] = rmsd_t0_lag/1000
    gensim_data_obs['rmsd_vel'] = rmsd_t0_lag/1000*np.sqrt(2)/(1.0/fps)

    # set the value of mag_RMSD=rmsd_mag_obs*conf_lvl and len_RMSD=rmsd_lag_obs*conf_lvl
    mag_RMSD = rmsd_pol_mag*z_score
    # check if in km or m
    if rmsd_t0_lag>1:
        len_RMSD = rmsd_t0_lag/1000*z_score
    else:
        len_RMSD = rmsd_t0_lag*z_score
        # ned in m instead of km
        rmsd_t0_lag=rmsd_t0_lag*1000

    # # Calculate the cumulative probability for the z-value, the confidence level is the percentage of the area within ±z_value
    CONFIDENCE_LEVEL = (2 * stats.norm.cdf(z_score) - 1)*100
    print('CONFIDENCE LEVEL required : '+str(np.round(CONFIDENCE_LEVEL,3))+'%')
    print('mag_RMSD:',mag_RMSD)
    print('len_RMSD:',len_RMSD,'km')

    print()




    ######################## SIMULATIONTS ###############################
    print('--- SIMULATIONS ---')

    print('Run MetSim file:',trajectory_Metsim_file)

    simulation_MetSim_object, gensim_data_Metsim, pd_datafram_PCA_sim_Metsim = run_simulation(trajectory_Metsim_file, gensim_data_obs, fit_funct)

    # open the folder and extract all the json files
    os.chdir(input_folder)

    # enough simulations in the folder
    flag_enough_sim = False
    # check in directory if it exist a csv file with input_folder+os.sep+file_name+NAME_SUFX_CSV_SIM
    if os.path.isfile(output_folder+os.sep+file_name+NAME_SUFX_CSV_SIM):
        # read the csv file
        pd_datafram_PCA_sim = pd.read_csv(output_folder+os.sep+file_name+NAME_SUFX_CSV_SIM)
        print('read the csv file:',output_folder+os.sep+file_name+NAME_SUFX_CSV_SIM)
        # print len of the csv file
        print('Number of simulations in the csv file:',len(pd_datafram_PCA_sim))
        flag_enough_sim = True

        if len(pd_datafram_PCA_sim) < cml_args.nsim:
            print('Add',cml_args.nsim - len(pd_datafram_PCA_sim),' json files')
            flag_enough_sim = False

        if cml_args.fix_n_sim:
            flag_enough_sim = True
        
        if cml_args.resample_sim:
            print('Number of simulations in the csv file is more than the number of simulations to run')
            print('Resample the simulations taking only', cml_args.nsim, 'simulations')
            
            # Extract and keep the first row as a separate DataFrame
            pd_datafram_PCA_sim_0 = pd_datafram_PCA_sim.iloc[[0]]  # Keep as DataFrame, not Series
            # Remove the first row from the DataFrame
            pd_datafram_PCA_sim = pd_datafram_PCA_sim.drop([0])
            # Sample the remaining rows with exactly cml_args.nsim - 1 rows
            pd_datafram_PCA_sim = pd_datafram_PCA_sim.sample(n=cml_args.nsim - 1)
            # Concatenate the first row back to the sampled DataFrame
            pd_datafram_PCA_sim = pd.concat([pd_datafram_PCA_sim_0, pd_datafram_PCA_sim], ignore_index=True)
            
            # # Verify the final count
            # print('Number of simulations in the csv file:', len(pd_datafram_PCA_sim)) 
            # print(pd_datafram_PCA_sim['solution_id'][0])
            


    if flag_enough_sim == False:

        # open the folder and extract all the json files
        os.chdir(output_folder)

        extension = 'json'
        # walk thorought the directories and find all the json files inside each folder inside the directory
        all_jsonfiles = [i for i in glob.glob('**/*.{}'.format(extension), recursive=True)]

        # delete from all_jsonfiles any file that has 'Selection' or 'Results' or 'mode' or 'DensPoint' in it
        all_jsonfiles = [file for file in all_jsonfiles if 'Selection' not in file and 'Results' not in file and 'mode' not in file and 'DensPoint' not in file]

        if len(all_jsonfiles) == 0 or len(all_jsonfiles) < cml_args.nsim:
            if len(all_jsonfiles) != 0:
                print('In the sim folder there are already',len(all_jsonfiles),'json files')
                print('Add',cml_args.nsim - len(all_jsonfiles),' json files')
            number_sim_to_run_and_simulation_in_folder = cml_args.nsim - len(all_jsonfiles)
            
            # run the new simulations
            if cml_args.save_test_plot:
                fig, ax = generate_simulations(pd_dataframe_PCA_obs_real,simulation_MetSim_object,gensim_data_obs,number_sim_to_run_and_simulation_in_folder,output_folder,file_name,fps,cml_args.save_test_plot, flag_manual_metsim)
                
                if flag_manual_metsim:
                    # plot gensim_data_Metsim
                    plot_side_by_side(gensim_data_Metsim,fig, ax,'k-',str(pd_datafram_PCA_sim_Metsim['type'].iloc[0]))

                # plot noisy area around vel_kms for vel_noise for the fix height_km
                ax[0].fill_betweenx(np.array(fit_funct['height'])/1000, np.array(fit_funct['absolute_magnitudes'])-rmsd_pol_mag, np.array(fit_funct['absolute_magnitudes'])+rmsd_pol_mag, color='lightgray', alpha=0.5)

                # plot noisy area around vel_kms for vel_noise for the fix height_km
                # ax[1].fill_between(np.array(fit_funct['time']), np.array(fit_funct['velocities'])/1000-(rmsd_t0_lag/1000/(1/fps)), np.array(fit_funct['velocities'])/1000+(rmsd_t0_lag/1000/(1/fps)), color='lightgray', alpha=0.5, label='Std.dev. realizations')
                ax[1].fill_between(np.array(fit_funct['time']), np.array(fit_funct['velocities'])/1000-(rmsd_t0_lag/1000*np.sqrt(2)/(1/fps)), np.array(fit_funct['velocities'])/1000+(rmsd_t0_lag/1000*np.sqrt(2)/(1/fps)), color='lightgray', alpha=0.5, label='Std.dev. realizations')

                ax[0].plot(fit_funct['absolute_magnitudes'],fit_funct['height']/1000, 'k--')
                ax[1].plot(fit_funct['time'],fit_funct['velocities']/1000, 'k--', label='Fit')

                # save the plot
                plt.savefig(output_folder+os.sep+file_name+'_obs_sim.png', dpi=300)
                # close the plot
                plt.close()
                # print saved csv file
                print('saved image '+output_folder+os.sep+file_name+'_obs_sim.png')

                # walk thorought the directories and find all the json files inside each folder inside the directory
                all_jsonfiles = [i for i in glob.glob('**/*.{}'.format(extension), recursive=True)]

                # delete from all_jsonfiles any file that has 'Selection' or 'Results' or 'mode' or 'DensPoint' in it
                all_jsonfiles = [file for file in all_jsonfiles if 'Selection' not in file and 'Results' not in file and 'mode' not in file and 'DensPoint' not in file]

            else:
                generate_simulations(pd_dataframe_PCA_obs_real,simulation_MetSim_object,gensim_data_obs,number_sim_to_run_and_simulation_in_folder,output_folder,file_name,fps,cml_args.save_test_plot, flag_manual_metsim)
                # walk thorought the directories and find all the json files inside each folder inside the directory
                all_jsonfiles = [i for i in glob.glob('**/*.{}'.format(extension), recursive=True)]

                # delete from all_jsonfiles any file that has 'Selection' or 'Results' or 'mode' or 'DensPoint' in it
                all_jsonfiles = [file for file in all_jsonfiles if 'Selection' not in file and 'Results' not in file and 'mode' not in file and 'DensPoint' not in file]

                
        print('start reading the json files')

        # add the output_folder to all_jsonfiles
        all_jsonfiles = [output_folder+os.sep+file for file in all_jsonfiles]

        # open the folder and extract all the json files
        os.chdir(input_folder)

        print('Number of simulated files: ',len(all_jsonfiles))

        input_list = [[all_jsonfiles[ii], 'simulation_'+str(ii+1), fit_funct, gensim_data_obs, True] for ii in range(len(all_jsonfiles))]
        results_list = domainParallelizer(input_list, read_GenerateSimulations_output_to_PCA, cores=cml_args.cores)
        
        # if no read the json files in the folder and create a new csv file
        pd_datafram_PCA_sim = pd.concat(results_list)

        if flag_manual_metsim:
            # concatenate the two dataframes
            pd_datafram_PCA_sim = pd.concat([pd_datafram_PCA_sim_Metsim, pd_datafram_PCA_sim])
        # print(df_sim_shower)
        pd_datafram_PCA_sim.reset_index(drop=True, inplace=True)
        
        # if pd_dataframe_PCA_obs_real['solution_id'].iloc[0].endswith('.json'): 
        #     print('REAL json file:',trajectory_Metsim_file)
        #     # change the type column to Real
        #     pd_datafram_PCA_sim['type'].iloc[0] = 'Real'

        pd_datafram_PCA_sim.to_csv(output_folder+os.sep+file_name+NAME_SUFX_CSV_SIM, index=False)
        # print saved csv file
        print('saved sim csv file:',output_folder+os.sep+file_name+NAME_SUFX_CSV_SIM)

        
    if pd_dataframe_PCA_obs_real['solution_id'].iloc[0].endswith('.json'): 
        print('REAL json file:',trajectory_Metsim_file)
        # change the type column to Real
        pd_datafram_PCA_sim['type'].iloc[0] = 'Real'


    # save the trajectory_file in the output_folder
    shutil.copy(pd_dataframe_PCA_obs_real['solution_id'][0], output_folder)

    # delete any file that end with _good_files.txt in the output_folder
    files = [f for f in os.listdir(output_folder) if f.endswith('_good_files.txt')]
    for file in files:
        os.remove(os.path.join(output_folder, file))
    
    print()
        


    ######################## PCA SELECTION ###############################

    pd_datafram_PCA_selected_lowRMSD = pd.DataFrame()
    pd_datafram_PCA_selected_before_knee_NO_repetition = pd.DataFrame()

    PCAn_comp = 0

    _,_,_=process_pca_variables(cml_args.YesPCA, cml_args.NoPCA, pd_dataframe_PCA_obs_real, pd_datafram_PCA_sim, output_folder+os.sep+save_results_folder, file_name, True)
    

    # if cml_args.use_PCA:

    #     print('--- PCA SELECTION ---')

    #     save_results_folder_PCA = save_results_folder+os.sep+'PCA'
    #     mkdirP(output_folder+os.sep+save_results_folder_PCA)

    #     pd_datafram_PCA_selected_before_knee, pd_datafram_PCA_selected_before_knee_NO_repetition, pd_datafram_PCA_selected_all, pcr_results_physical_param, PCAn_comp = PCASim(pd_datafram_PCA_sim, pd_dataframe_PCA_obs_real, output_folder, cml_args.PCA_percent, cml_args.nsel_forced, cml_args.YesPCA, cml_args.NoPCA, file_name, cml_args.cores, cml_args.save_test_plot, cml_args.esclude_real_solution_from_selection)

    #     print('PLOT: best 10 simulations selected and add the RMSD value to csv selected')
    #     # plot of the best 9 selected simulations and add the RMSD value to csv selected
    #     PCA_LightCurveCoefPLOT(pd_datafram_PCA_selected_before_knee_NO_repetition, pd_dataframe_PCA_obs_real, output_folder+os.sep+save_results_folder_PCA, fit_funct, gensim_data_obs, rmsd_pol_mag, rmsd_t0_lag, fps, file_name, trajectory_Metsim_file, pca_N_comp=PCAn_comp)

    #     print('PLOT: the physical characteristics of the selected simulations Mode and KDE')
    #     PCA_PhysicalPropPLOT(pd_datafram_PCA_selected_before_knee, pd_datafram_PCA_sim, output_folder+os.sep+save_results_folder_PCA, file_name, pca_N_comp=PCAn_comp)

    #     print('PLOT: correlation of the selected simulations (takes a long time)')
    #     # plot correlation function of the selected simulations
    #     PCAcorrelation_selPLOT(pd_datafram_PCA_sim, pd_datafram_PCA_selected_before_knee_NO_repetition, output_folder+os.sep+save_results_folder_PCA, pca_N_comp=PCAn_comp)

    #     print('Selected simulations and generate KDE and MODE plot')
    #     input_list_obs = [[pd_datafram_PCA_sim, pd_dataframe_PCA_obs_real.iloc[[ii]].reset_index(drop=True), pd_datafram_PCA_selected_before_knee[pd_datafram_PCA_selected_before_knee['solution_id_dist'] == pd_dataframe_PCA_obs_real['solution_id'].iloc[ii]], gensim_data_obs, fit_funct, rmsd_pol_mag, rmsd_t0_lag, mag_RMSD, len_RMSD, fps, trajectory_Metsim_file, file_name, pd_dataframe_PCA_obs_real['solution_id'].iloc[0], output_folder, save_results_folder_events_plots] for ii in range(len(pd_dataframe_PCA_obs_real))]
    #     results_list = domainParallelizer(input_list_obs, PCA_physicalProp_KDE_MODE_PLOT, cores=cml_args.cores)
        
    #     # if no read the json files in the folder and create a new csv file
    #     pd_datafram_PCA_selected_mode_min_KDE = pd.concat(results_list)

    #     pd_datafram_PCA_selected_mode_min_KDE_TOT = PCA_physicalProp_KDE_MODE_PLOT(pd_datafram_PCA_sim, pd_dataframe_PCA_obs_real, pd_datafram_PCA_selected_before_knee, gensim_data_obs, fit_funct, rmsd_pol_mag, rmsd_t0_lag, mag_RMSD, len_RMSD, fps, trajectory_Metsim_file, file_name, pd_dataframe_PCA_obs_real['solution_id'].iloc[0], output_folder, save_results_folder_events_plots,True, True)
        
    #     # concatenate the two dataframes
    #     pd_datafram_PCA_selected_lowRMSD = pd.concat([pd_datafram_PCA_selected_mode_min_KDE_TOT, pd_datafram_PCA_selected_mode_min_KDE])
    #     # reset index
    #     pd_datafram_PCA_selected_lowRMSD.reset_index(drop=True, inplace=True)

    #     pd_datafram_PCA_selected_lowRMSD['type'] = 'Simulation_sel'
    
    #     # # save df_sel_shower_real to disk add the RMSD
    #     # pd_datafram_PCA_selected_lowRMSD.to_csv(output_folder+os.sep+save_results_folder+os.sep+file_name+'_sim_sel_results.csv', index=False)

    #     # print('PLOT: the physical characteristics of the selected simulations with no repetitions')
    #     # PCA_PhysicalPropPLOT(pd_datafram_PCA_selected_before_knee_NO_repetition, pd_datafram_PCA_sim, output_folder, file_name)


    #     # print(pd_datafram_PCA_selected_lowRMSD)
    #     # split in directory and filename
    #     filename_list = []
    #     # print(pd_datafram_PCA_selected_lowRMSD['solution_id'].values)
    #     if 'solution_id' in pd_datafram_PCA_selected_lowRMSD.columns:
    #         # check if in pd_datafram_PCA_selected_lowRMSD there is any json file that is not in the selected simulations
    #         for solution_id in pd_datafram_PCA_selected_lowRMSD['solution_id'].values:
    #             directory, filename = os.path.split(solution_id)
    #             filename_list.append(filename)
    #         # print(filename_list)
    #         json_files = [f for f in os.listdir(output_folder+os.sep+save_results_folder_events_plots) if f.endswith('.json')]
    #         for json_file in json_files:
    #             folder_and_jsonfile_result = output_folder+os.sep+save_results_folder_events_plots+os.sep+json_file
    #             if json_file not in filename_list:
    #                 # print that is found a json file that is not in the selected simulations
    #                 print(folder_and_jsonfile_result,'\njson file found in the Results directory that is not in '+file_name+'_sim_sel_results.csv')
    #                 f = open(folder_and_jsonfile_result,"r")
    #                 data = json.loads(f.read())
    #                 if 'ht_sampled' in data:
    #                     data_file = read_GenerateSimulations_output(folder_and_jsonfile_result, gensim_data_obs)
    #                     pd_datafram_PCA_sim_resulsts=array_to_pd_dataframe_PCA(data_file, gensim_data_obs)
    #                 else:
    #                     _, _, pd_datafram_PCA_sim_resulsts = run_simulation(folder_and_jsonfile_result, gensim_data_obs, fit_funct)
                    
    #                 pd_datafram_PCA_sim_resulsts['type'] = 'Simulation_sel'
    #                 # delete every pd_datafram_PCA_sim_resulsts with RMSD above mag_RMSD and len_RMSD
    #                 if pd_datafram_PCA_sim_resulsts['rmsd_mag'].iloc[0] < mag_RMSD and pd_datafram_PCA_sim_resulsts['rmsd_len'].iloc[0] < len_RMSD:
    #                     # Add the simulation results to pd_datafram_PCA_selected_lowRMSD
    #                     pd_datafram_PCA_selected_lowRMSD = pd.concat([pd_datafram_PCA_selected_lowRMSD, pd_datafram_PCA_sim_resulsts])

    #                 # # Add the simulation results to pd_datafram_PCA_selected_lowRMSD
    #                 # pd_datafram_PCA_selected_lowRMSD = pd.concat([pd_datafram_PCA_selected_lowRMSD, pd_datafram_PCA_sim_resulsts])

    #         pd_datafram_PCA_selected_lowRMSD.reset_index(drop=True, inplace=True)
    #     else:
    #         print('No Mode and Densest point solutions for the selected simulations')

    #     print()

    ######################## OPTIMIZATION ###############################

    print('--- RMSD CHECK ---')

    ### ORDER BASE ON BOTH RMSD ### chi2_red_mag	chi2_red_len

    # deep copy pd_datafram_PCA_sim
    pd_datafram_check_below_RMSD = pd_datafram_PCA_sim.copy(deep=True)

    pd_datafram_check_below_RMSD['chi2_red_mag_0'] = abs(pd_datafram_check_below_RMSD['chi2_red_mag']-1) # chi2_red_mag	chi2_red_len
    pd_datafram_check_below_RMSD['chi2_red_lag_0'] = abs(pd_datafram_check_below_RMSD['chi2_red_lag']-1) # chi2_red_mag	chi2_red_len

    # Normalize the columns to bring them to the same scale
    pd_datafram_check_below_RMSD['chi2_red_mag_norm'] = pd_datafram_check_below_RMSD['chi2_red_mag_0'] / pd_datafram_check_below_RMSD['chi2_red_mag_0'].max()
    pd_datafram_check_below_RMSD['chi2_red_lag_norm'] = pd_datafram_check_below_RMSD['chi2_red_lag_0'] / pd_datafram_check_below_RMSD['chi2_red_lag_0'].max()

    # Compute the combined metric (e.g., sum of absolute normalized values)
    pd_datafram_check_below_RMSD['combined_chi2_red_metric'] = abs(pd_datafram_check_below_RMSD['chi2_red_mag_norm']) + abs(pd_datafram_check_below_RMSD['chi2_red_lag_norm'])

    # Sort the DataFrame based on the combined metric
    pd_datafram_check_below_RMSD = pd_datafram_check_below_RMSD.sort_values(by='combined_chi2_red_metric')

    # Reset index if needed
    pd_datafram_check_below_RMSD = pd_datafram_check_below_RMSD.reset_index(drop=True)

    # check if the following values are also smaller than mag_RMSD and len_RMSD and save the index
    for i in range(0, len(pd_datafram_check_below_RMSD)):
        # if pd_datafram_check_below_RMSD['rmsd_mag'].iloc[i] > mag_RMSD or pd_datafram_check_below_RMSD['rmsd_len'].iloc[i] > len_RMSD:
        if pd_datafram_check_below_RMSD['chi2_red_mag'].iloc[i] >= 0.5 and pd_datafram_check_below_RMSD['chi2_red_mag'].iloc[i] <= 1.5 and pd_datafram_check_below_RMSD['chi2_red_lag'].iloc[i] >= 0.5 and pd_datafram_check_below_RMSD['chi2_red_lag'].iloc[i] <= 1.5:
            index = i
            break
    
    if index==0 and not 'solution_id' in pd_datafram_PCA_selected_lowRMSD.columns:
        cml_args.optimize=True
        if cml_args.number_optimized==0:
            cml_args.number_optimized=cpu_count()           
        pd_datafram_check_below_RMSD = pd_datafram_check_below_RMSD.head(cml_args.number_optimized)
        print('No simulations below RMSD, run the optimization for the first',cml_args.number_optimized,'simulations')
        if not os.path.isfile(cml_args.ref_opt_path):
            # If the file is not found, check in the parent directory
            parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            cml_args.ref_opt_path = os.path.join(parent_directory, 'AutoRefineFit_options.txt')
            if not os.path.isfile(cml_args.ref_opt_path):
                print('file '+cml_args.ref_opt_path+' not found')
                print("You need to specify the correct path and name of the AutoRefineFit_options.txt file in --ref_opt_path, like: C:\\path\\AutoRefineFit_options.txt")
                sys.exit()
        # copy the file to the output_folder
        shutil.copy(cml_args.ref_opt_path, output_folder+os.sep+'AutoRefineFit_options.txt')
    else:
        # take the head of the dataframe with the index
        pd_datafram_check_below_RMSD = pd_datafram_check_below_RMSD.head(index+1)
    
    print('index:',index)

    # Drop the auxiliary columns if they are no longer needed
    pd_datafram_check_below_RMSD = pd_datafram_check_below_RMSD.drop(columns=['chi2_red_mag_0', 'chi2_red_mag_0', 'chi2_red_mag_norm', 'chi2_red_lag_norm', 'combined_chi2_red_metric'])

    # print the RMSD column
    # print('RMSD column:',pd_datafram_check_below_RMSD[['solution_id','rmsd_mag', 'rmsd_len']])
    
    ##############################

    # check if in pd_datafram_PCA_sim there are any where rmsd_mag and rmsd_len are below mag$_{RMSD}$ and len$_{RMSD}$
    print('Number of simulations below RMSD:', len(pd_datafram_check_below_RMSD))
    print('Threshold mag_RMSD:', mag_RMSD,'len_RMSD:', len_RMSD)
    # print the first mag_RMSD and len_RMSD
    print('Best simulation mag_RMSD:', pd_datafram_check_below_RMSD['rmsd_mag'].iloc[0],'len_RMSD:', pd_datafram_check_below_RMSD['rmsd_len'].iloc[0])

    # add a column with solution_id_dist to 9999
    pd_datafram_check_below_RMSD['solution_id_dist'] = pd_datafram_check_below_RMSD['solution_id'].iloc[0]
    # set type equal 'Simulation_sel'
    pd_datafram_check_below_RMSD['type'] = 'Simulation_sel'
    # add column distance_meteor=9999 and distance_mean=9999 and num_duplicates=0
    pd_datafram_check_below_RMSD['distance_meteor'] = 9999
    pd_datafram_check_below_RMSD['distance_mean'] = 9999
    pd_datafram_check_below_RMSD['num_duplicates'] = 0

    if cml_args.use_PCA:
        # delete any rows from pd_datafram_check_below_RMSD that has the same ['solution_id'] as pd_datafram_PCA_selected_before_knee_NO_repetition['solution_id']
        pd_datafram_check_below_RMSD = pd_datafram_check_below_RMSD[~pd_datafram_check_below_RMSD['solution_id'].isin(pd_datafram_PCA_selected_before_knee_NO_repetition['solution_id'])]

    # reset index
    pd_datafram_check_below_RMSD.reset_index(drop=True, inplace=True)
        
    # concatenate the two dataframes
    pd_datafram_check_below_RMSD = pd.concat([pd_datafram_check_below_RMSD, pd_datafram_PCA_selected_before_knee_NO_repetition])
    pd_datafram_check_below_RMSD.to_csv(output_folder+os.sep+SAVE_SELECTION_FOLDER+os.sep+file_name+'_sim_sel_to_optimize.csv', index=False)

    # check if cml_args.number_optimized is negative and take the abs
    if cml_args.number_optimized < 0:
        cml_args.number_optimized = 0
    # check if cml_args.number_optimized is bigger than the number of selected simulations
    if cml_args.number_optimized > len(pd_datafram_check_below_RMSD):
        cml_args.number_optimized = 0

    # # plot the values and find the RMSD of each of them
    if cml_args.number_optimized == 0:
        input_list_obs = [[pd_datafram_check_below_RMSD.iloc[[ii]].reset_index(drop=True), pd_dataframe_PCA_obs_real, gensim_data_obs, output_folder, fit_funct, gensim_data_Metsim, rmsd_pol_mag, rmsd_t0_lag, mag_RMSD, len_RMSD, fps, file_name, save_results_folder_events_plots, cml_args.optimize] for ii in range(len(pd_datafram_check_below_RMSD))]
        results_list = domainParallelizer(input_list_obs, PCA_LightCurveRMSDPLOT_optimize, cores=cml_args.cores)
        # check if the results_list is empty
        if len(results_list)==0:
            pd_datafram_PCA_selected_optimized = pd.DataFrame()
        else:
            pd_datafram_PCA_selected_optimized = pd.concat(results_list)
    elif cml_args.number_optimized > 0 and cml_args.optimize==True:
        # repeat for the rest of the selected events but no Optimization
        input_list_obs = [[pd_datafram_check_below_RMSD.iloc[[ii]].reset_index(drop=True), pd_dataframe_PCA_obs_real, gensim_data_obs, output_folder, fit_funct, gensim_data_Metsim, rmsd_pol_mag, rmsd_t0_lag, mag_RMSD, len_RMSD, fps, file_name, save_results_folder_events_plots, False] for ii in range(cml_args.number_optimized,len(pd_datafram_check_below_RMSD))]
        results_list = domainParallelizer(input_list_obs, PCA_LightCurveRMSDPLOT_optimize, cores=cml_args.cores)
        # check if is.empty(results_list)
        if len(results_list)==0:
            pd_datafram_PCA_selected_NO_optimizad = pd.DataFrame()
        else:
            pd_datafram_PCA_selected_NO_optimizad = pd.concat(results_list)        
        # take only the first cml_args.number_optimized
        input_list_obs = [[pd_datafram_check_below_RMSD.iloc[[ii]].reset_index(drop=True), pd_dataframe_PCA_obs_real, gensim_data_obs, output_folder, fit_funct, gensim_data_Metsim, rmsd_pol_mag, rmsd_t0_lag, mag_RMSD, len_RMSD, fps, file_name, save_results_folder_events_plots, True] for ii in range(cml_args.number_optimized)]
        results_list = domainParallelizer(input_list_obs, PCA_LightCurveRMSDPLOT_optimize, cores=cml_args.cores)
        # check if is.empty(results_list)
        if len(results_list)==0:
            pd_datafram_PCA_selected_optimized = pd.DataFrame()
        else:
            pd_datafram_PCA_selected_optimized = pd.concat(results_list)
        # pd_datafram_PCA_selected_optimized = PCA_LightCurveRMSDPLOT_optimize(pd_datafram_check_below_RMSD.head(cml_args.number_optimized), pd_dataframe_PCA_obs_real, output_folder, fit_funct, gensim_data_Metsim, rmsd_pol_mag, rmsd_t0_lag, mag_RMSD, len_RMSD, fps, file_name, save_results_folder_events_plots, True)
        pd_datafram_PCA_selected_lowRMSD = pd.concat([pd_datafram_PCA_selected_optimized, pd_datafram_PCA_selected_NO_optimizad])

    
    # concatenate the two dataframes
    pd_datafram_PCA_selected_lowRMSD = pd.concat([pd_datafram_PCA_selected_optimized, pd_datafram_PCA_selected_lowRMSD])

    # check if trajectory_Metsim_file is among the pd_datafram_check_below_RMSD
    if flag_manual_metsim and trajectory_Metsim_file not in pd_datafram_check_below_RMSD['solution_id'].values:
        print('Check the manual reduction')
        # check also the manual reduction
        pd_datafram_PCA_selected_optimized_Metsim = PCA_LightCurveRMSDPLOT_optimize(pd_datafram_PCA_sim_Metsim, pd_dataframe_PCA_obs_real, gensim_data_obs, output_folder, fit_funct, gensim_data_Metsim, rmsd_pol_mag, rmsd_t0_lag, mag_RMSD, len_RMSD, fps, file_name, save_results_folder_events_plots, False) # file_name, trajectory_Metsim_file, 
            
        # concatenate the two dataframes
        pd_datafram_PCA_selected_lowRMSD = pd.concat([pd_datafram_PCA_selected_optimized_Metsim, pd_datafram_PCA_selected_lowRMSD])

    # get all the json file in output_folder+os.sep+save_results_folder_events_plots
    json_files_results = [f for f in os.listdir(output_folder+os.sep+save_results_folder_events_plots) if f.endswith('.json')]
    # check if output_folder+os.sep+save_res_fin_folder+'events_plots' exist
    if os.path.isdir(output_folder+os.sep+save_res_fin_folder+os.sep+'events_plots'):
        # get all the json file in output_folder+os.sep+save_res_fin_folder+'events_plots'
        json_files_results = json_files_results + [f for f in os.listdir(output_folder+os.sep+save_res_fin_folder+os.sep+'events_plots') if f.endswith('.json')]

    # check if any json_files_results is in pd_datafram_PCA_selected_lowRMSD['solution_id'].values
    if 'solution_id' in pd_datafram_PCA_selected_lowRMSD.columns:
        for json_file in json_files_results:
            if json_file not in pd_datafram_PCA_selected_lowRMSD['solution_id'].values:
                # print that is found a json file that is not in the selected simulations
                print(output_folder+os.sep+save_results_folder_events_plots+os.sep+json_file,'\njson file found in the Results directory that is not in '+file_name+'_sim_sel_results.csv')
                f = open(output_folder+os.sep+save_results_folder_events_plots+os.sep+json_file,"r")
                data = json.loads(f.read())
                if 'ht_sampled' in data:
                    data_file = read_GenerateSimulations_output(output_folder+os.sep+save_results_folder_events_plots+os.sep+json_file, gensim_data_obs)
                    pd_datafram_PCA_sim_resulsts=array_to_pd_dataframe_PCA(data_file, gensim_data_obs)
                else:
                    _, data_file, pd_datafram_PCA_sim_resulsts = run_simulation(output_folder+os.sep+save_results_folder_events_plots+os.sep+json_file, gensim_data_obs, fit_funct)
                
                rmsd_mag, rmsd_vel, rmsd_lag, _, _, _, _, _ , _ = RMSD_calc_diff(data_file, gensim_data_obs)
                # Add the simulation results that have a rmsd_mag and rmsd_len that is below RMSD to pd_datafram_PCA_selected_lowRMSD
                if rmsd_mag <= mag_RMSD and rmsd_lag <= len_RMSD:
                    # print to added to the selected simulations pd_datafram_PCA_sim_resulsts['solution_id'].values[0]
                    print('Added to the selected simulations:',output_folder+os.sep+save_results_folder_events_plots+os.sep+json_file)
                    pd_datafram_PCA_selected_lowRMSD = pd.concat([pd_datafram_PCA_selected_lowRMSD, pd_datafram_PCA_sim_resulsts])
                    # pd_datafram_PCA_selected_lowRMSD['type'] = 'Simulation_sel'
                    # pd_datafram_PCA_selected_lowRMSD.reset_index(drop=True, inplace=True)
    else:
        print('No solutions below the RMSD for the selected simulations')

    print()

    ######################## ITERATIVE RESEARCH ###############################

    print('--- ITERATIVE RESEARCH ---')

    # move the txt file starting with log_RMSD_ to the save_results_folder
    files = [f for f in os.listdir(output_folder) if f.startswith('log_RMSD_')]
    for file in files:
        shutil.move(os.path.join(output_folder, file), os.path.join(output_folder, save_results_folder, file))

    # move the image file starting with Combined RMSD metric to the save_results_folder
    img_files = [f for f in os.listdir(output_folder) if f.startswith('Combined RMSD metric')]
    for img_file in img_files:
        shutil.move(os.path.join(output_folder, img_file), os.path.join(output_folder, save_results_folder, img_file))

    flag_fail = False
    old_results_number = 0
    result_number = 0
    ii_repeat = 0
    pd_results = pd.DataFrame()
    # while cml_args.min_nresults > result_number:
    print(cml_args.min_nresults,'simulated to find:')
    while cml_args.min_nresults > result_number:

        # reset index
        pd_datafram_PCA_selected_lowRMSD.reset_index(drop=True, inplace=True)

        pd_datafram_PCA_selected_lowRMSD['type'] = 'Simulation_sel'

        # delete any row from the csv file that has the same value of mass, rho, sigma, erosion_height_start, erosion_coeff, erosion_mass_index, erosion_mass_min, erosion_mass_max, erosion_range, erosion_energy_per_unit_cross_section, erosion_energy_per_unit_mass
        if 'mass' in pd_datafram_PCA_selected_lowRMSD.columns:                  
            # Drop duplicate rows based on the specified columns
            pd_datafram_PCA_selected_lowRMSD = pd_datafram_PCA_selected_lowRMSD.drop_duplicates(subset=[
                'mass', 'rho', 'sigma', 'erosion_height_start', 'erosion_coeff', 
                'erosion_mass_index', 'erosion_mass_min', 'erosion_mass_max', 
                'erosion_range', 'erosion_energy_per_unit_cross_section', 
                'erosion_energy_per_unit_mass'
            ])
            pd_datafram_PCA_selected_lowRMSD.reset_index(drop=True, inplace=True)

        pd_results = pd.concat([pd_results, pd_datafram_PCA_selected_lowRMSD])

        # save and update the disk 
        pd_results.to_csv(output_folder+os.sep+save_results_folder+os.sep+file_name+'_sim_sel_results.csv', index=False)

        
        if 'solution_id' in pd_results.columns:
            print('PLOT: the physical characteristics results')
            PCA_PhysicalPropPLOT(pd_results, pd_datafram_PCA_sim, output_folder+os.sep+save_results_folder, file_name)
            print('PLOT: correlation matrix of the results (takes a long time)')
            PCAcorrelation_selPLOT(pd_datafram_PCA_sim, pd_results, output_folder+os.sep+save_results_folder)
            print('PLOT: best 10 results and add the RMSD value to csv selected')
            # deep copy pd_datafram_PCA_sim
            pd_results['chi2_red_mag_0'] = abs(pd_results['chi2_red_mag']-1) # chi2_red_mag	chi2_red_len
            pd_results['chi2_red_lag_0'] = abs(pd_results['chi2_red_lag']-1) # chi2_red_mag	chi2_red_len

            # Normalize the columns to bring them to the same scale
            pd_results['chi2_red_mag_norm'] = pd_results['chi2_red_mag_0'] / pd_results['chi2_red_mag_0'].max()
            pd_results['chi2_red_lag_norm'] = pd_results['chi2_red_lag_0'] / pd_results['chi2_red_lag_0'].max()

            # Compute the combined metric (e.g., sum of absolute normalized values)
            pd_results['combined_chi2_red_metric'] = abs(pd_results['chi2_red_mag_norm']) + abs(pd_results['chi2_red_lag_norm'])

            # Sort the DataFrame based on the combined metric
            pd_results = pd_results.sort_values(by='combined_chi2_red_metric')
            # Reset index if needed
            pd_results = pd_results.reset_index(drop=True)
            pd_results = pd_results.drop(columns=['chi2_red_mag_0','chi2_red_lag_0','chi2_red_mag_norm', 'chi2_red_lag_norm', 'combined_chi2_red_metric'])
            PCA_LightCurveCoefPLOT(pd_results, pd_dataframe_PCA_obs_real, output_folder+os.sep+save_results_folder, fit_funct, gensim_data_obs, rmsd_pol_mag, rmsd_t0_lag, fps, file_name, trajectory_Metsim_file,output_folder+os.sep+save_results_folder+os.sep+file_name+'_sim_sel_results.csv')
            print()
            print('SUCCES: the physical characteristics range is in the results folder')
        else:
            # print('FAIL: Not found any result below magRMSD',rmsd_pol_mag,'and lenRMSD',rmsd_t0_lag/1000)
            print('FAIL: Not found any result below magRMSD',mag_RMSD,'and lenRMSD',len_RMSD)
            flag_fail = True
            break


        # check if only 1 in len break
        if len(pd_results) == 1:
            print('Only one result found')
            # create a dictionary with the physical parameters
            CI_physical_param = {
                'v_init_180km': [pd_results['v_init_180km'].values[0], pd_results['v_init_180km'].values[0]],
                'zenith_angle': [pd_results['zenith_angle'].values[0], pd_results['zenith_angle'].values[0]],
                'mass': [pd_results['mass'].values[0], pd_results['mass'].values[0]],
                'rho': [pd_results['rho'].values[0], pd_results['rho'].values[0]],
                'sigma': [pd_results['sigma'].values[0], pd_results['sigma'].values[0]],
                'erosion_height_start': [pd_results['erosion_height_start'].values[0], pd_results['erosion_height_start'].values[0]],
                'erosion_coeff': [pd_results['erosion_coeff'].values[0], pd_results['erosion_coeff'].values[0]],
                'erosion_mass_index': [pd_results['erosion_mass_index'].values[0], pd_results['erosion_mass_index'].values[0]],
                'erosion_mass_min': [pd_results['erosion_mass_min'].values[0], pd_results['erosion_mass_min'].values[0]],
                'erosion_mass_max': [pd_results['erosion_mass_max'].values[0], pd_results['erosion_mass_max'].values[0]]
            }

        else:
            print('Number of results found:',len(pd_results))
            columns_physpar = ['v_init_180km','zenith_angle','mass', 'rho', 'sigma', 'erosion_height_start', 'erosion_coeff', 
                'erosion_mass_index', 'erosion_mass_min', 'erosion_mass_max']
            
            if ii_repeat > 1 and old_results_number == result_number:
            ###############################################################################
                # try to focus on the one that have good results
                
                # Calculate the quantiles
                quantiles = pd_results[columns_physpar].quantile([0.2, 0.8])

                # Convert the quantiles to a dictionary
                CI_physical_param = {col: quantiles[col].tolist() for col in columns_physpar}

            ###############################################################################
            else:
                # try and look for other results that might be around

                # Calculate the quantiles
                quantiles = pd_results[columns_physpar].quantile([0.1, 0.9])

                # Get the minimum and maximum values
                min_val = pd_results[columns_physpar].min()
                max_val = pd_results[columns_physpar].max()

                # Calculate the extended range using the logic provided
                extended_min = min_val - (quantiles.loc[0.1] - min_val)
                # consider the value extended_min<0 Check each column in extended_min and set to min_val if negative
                for col in columns_physpar:
                    if extended_min[col] < 0:
                        extended_min[col] = min_val[col]
                extended_max = max_val + (max_val - quantiles.loc[0.9])

                # Convert the extended range to a dictionary
                CI_physical_param = {col: [extended_min[col], extended_max[col]] for col in columns_physpar}
            
            ###############################################################################


        # check if v_init_180km are the same value
        if CI_physical_param['v_init_180km'][0] == CI_physical_param['v_init_180km'][1]:
            CI_physical_param['v_init_180km'] = [CI_physical_param['v_init_180km'][0] - CI_physical_param['v_init_180km'][0]/1000, CI_physical_param['v_init_180km'][1] + CI_physical_param['v_init_180km'][1]/1000]
        if CI_physical_param['zenith_angle'][0] == CI_physical_param['zenith_angle'][1]:
            CI_physical_param['zenith_angle'] = [CI_physical_param['zenith_angle'][0] - CI_physical_param['zenith_angle'][0]/10000, CI_physical_param['zenith_angle'][1] + CI_physical_param['zenith_angle'][1]/10000]
        if CI_physical_param['mass'][0] == CI_physical_param['mass'][1]:
            CI_physical_param['mass'] = [CI_physical_param['mass'][0] - CI_physical_param['mass'][0]/10, CI_physical_param['mass'][1] + CI_physical_param['mass'][1]/10]
        if np.round(CI_physical_param['rho'][0]/100) == np.round(CI_physical_param['rho'][1]/100):
            CI_physical_param['rho'] = [CI_physical_param['rho'][0] - CI_physical_param['rho'][0]/5, CI_physical_param['rho'][1] + CI_physical_param['rho'][1]/5]
        if CI_physical_param['sigma'][0] == CI_physical_param['sigma'][1]:
            CI_physical_param['sigma'] = [CI_physical_param['sigma'][0] - CI_physical_param['sigma'][0]/10, CI_physical_param['sigma'][1] + CI_physical_param['sigma'][1]/10]
        if CI_physical_param['erosion_height_start'][0] == CI_physical_param['erosion_height_start'][1]:
            CI_physical_param['erosion_height_start'] = [CI_physical_param['erosion_height_start'][0] - CI_physical_param['erosion_height_start'][0]/100, CI_physical_param['erosion_height_start'][1] + CI_physical_param['erosion_height_start'][1]/100]
        if CI_physical_param['erosion_coeff'][0] == CI_physical_param['erosion_coeff'][1]:
            CI_physical_param['erosion_coeff'] = [CI_physical_param['erosion_coeff'][0] - CI_physical_param['erosion_coeff'][0]/10, CI_physical_param['erosion_coeff'][1] + CI_physical_param['erosion_coeff'][1]/10]
        if CI_physical_param['erosion_mass_index'][0] == CI_physical_param['erosion_mass_index'][1]:
            CI_physical_param['erosion_mass_index'] = [CI_physical_param['erosion_mass_index'][0] - CI_physical_param['erosion_mass_index'][0]/10, CI_physical_param['erosion_mass_index'][1] + CI_physical_param['erosion_mass_index'][1]/10]
        if CI_physical_param['erosion_mass_min'][0] == CI_physical_param['erosion_mass_min'][1]:
            CI_physical_param['erosion_mass_min'] = [CI_physical_param['erosion_mass_min'][0] - CI_physical_param['erosion_mass_min'][0]/10, CI_physical_param['erosion_mass_min'][1] + CI_physical_param['erosion_mass_min'][1]/10]
        if CI_physical_param['erosion_mass_max'][0] == CI_physical_param['erosion_mass_max'][1]:
            CI_physical_param['erosion_mass_max'] = [CI_physical_param['erosion_mass_max'][0] - CI_physical_param['erosion_mass_max'][0]/10, CI_physical_param['erosion_mass_max'][1] + CI_physical_param['erosion_mass_max'][1]/10]
            

        # Multiply the 'erosion_height_start' values by 1000
        CI_physical_param['erosion_height_start'] = [x * 1000 for x in CI_physical_param['erosion_height_start']]

        print('CI_physical_param:',CI_physical_param)

        result_number = len(pd_results)

        if cml_args.min_nresults <= result_number:
            # print the number of results found
            print('SUCCES: Number of results found:',result_number)
            break
        else:
            if old_results_number == result_number:
                print('Same number of results found:',result_number)
                ii_repeat+=1
            if ii_repeat==cml_args.ntry:
                print('STOP: After '+str(cml_args.ntry)+' failed attempt')
                print('STOP: No new simulation below magRMSD',mag_RMSD,'and lenRMSD',len_RMSD)
                print('STOP: Number of results found:',result_number)
                flag_fail = True
                break
            old_results_number = result_number
            print('regenerate new simulation in the CI range')
            generate_simulations(pd_dataframe_PCA_obs_real, simulation_MetSim_object, gensim_data_obs, cml_args.nsim_refine_step, output_folder, file_name,fps, False, True, CI_physical_param)
            
            # look for the good_files = glob.glob(os.path.join(output_folder, '*_good_files.txt'))
            good_files = [f for f in os.listdir(output_folder) if f.endswith('_good_files.txt')]                

            # Construct the full path to the good file
            good_file_path = os.path.join(output_folder, good_files[0])

            # Read the file, skipping the first line
            df_good_files = pd.read_csv(good_file_path, skiprows=1)

            # Rename the columns
            df_good_files.columns = ["File name", "lim mag", "lim mag length", "length delay (s)"]

            # Extract the first column into an array
            file_names = df_good_files["File name"].to_numpy()

            # Change the file extension to .json
            all_jsonfiles = [file_name.replace('.pickle', '.json') for file_name in file_names]

            # open the folder and extract all the json files
            os.chdir(input_folder)

            # print('Number of simulated files : ',len(all_jsonfiles),' LEN confidence interval', conf_len,'% MAG confidence interval', conf_mag,'%')

            input_list = [[all_jsonfiles[ii], 'simulation_'+str(ii+1), fit_funct, gensim_data_obs] for ii in range(len(all_jsonfiles))]
            results_list = domainParallelizer(input_list, read_GenerateSimulations_output_to_PCA, cores=cml_args.cores)
            
            # if no read the json files in the folder and create a new csv file
            pd_datafram_NEWsim_good = pd.concat(results_list)

            pd_datafram_NEWsim_good.to_csv(output_folder+os.sep+file_name+NAME_SUFX_CSV_SIM_NEW, index=False)
            # print saved csv file
            print('saved sim csv file:',output_folder+os.sep+file_name+NAME_SUFX_CSV_SIM_NEW)

            input_list_obs = [[pd_datafram_NEWsim_good.iloc[[ii]].reset_index(drop=True), pd_dataframe_PCA_obs_real, gensim_data_obs, output_folder, fit_funct, gensim_data_Metsim, rmsd_pol_mag, rmsd_t0_lag, mag_RMSD, len_RMSD, fps, file_name, save_results_folder_events_plots, False] for ii in range(len(pd_datafram_NEWsim_good))]
            results_list = domainParallelizer(input_list_obs, PCA_LightCurveRMSDPLOT_optimize, cores=cml_args.cores)

            # base on the one selected
            pd_datafram_PCA_selected_lowRMSD = pd.concat(results_list)

    print()




    ######################## FINAL RESULTS ###############################

    # check if a file with the name "log"+n_PC_in_PCA+"_"+str(len(df_sel))+"ev.txt" already exist print('processing file:',file_name)
    if os.path.exists(output_folder+os.sep+save_results_folder+os.sep+"log_"+file_name[:15]+"_results.txt"):
        # remove the file
        os.remove(output_folder+os.sep+save_results_folder+os.sep+"log_"+file_name[:15]+"_results.txt")
    sys.stdout = Logger(output_folder+os.sep+save_results_folder,"log_"+file_name[:15]+"_results.txt") # _30var_99%_13PC

    print('--- FINAL RESULTS ---')

    print()

    if flag_fail == False:
        print('Save RESULTS!')
        print('Number of RESULTS found:',result_number)
        print('SUCCES: Number of RESULTS found:',result_number)
    else:
        print('FAIL: Not found enough result below magRMSD',mag_RMSD,'and lenRMSD',len_RMSD)
        print('FAIL: Number of results found:',result_number)
        print('INCREASE the intial SIMULATIONS or ADD an other FRAGMENTATION!')

    print('The results are in the folder:',output_folder+os.sep+save_results_folder)
    # print the RMSD and the rmsd_pol_mag, rmsd_t0_lag/1000 and the CONFIDENCE_LEVEL
    print('real data RMSD mag:'+str(rmsd_pol_mag)+'[-] RMSD len:'+str(rmsd_t0_lag/1000)+'[km]')
    if rmsd_pol_mag==CAMERA_SENSITIVITY_LVL_MAG:
        print('real data RMSD mag at the limit of sensistivity, automatically set to '+str(CAMERA_SENSITIVITY_LVL_MAG)+'[-]')
    if rmsd_t0_lag==CAMERA_SENSITIVITY_LVL_LEN:
        print('real data RMSD len at the limit of sensistivity, automatically set to '+str(CAMERA_SENSITIVITY_LVL_LEN)+'[m]')
    print('Confidence level: '+str(CONFIDENCE_LEVEL)+'% and z-factor: '+str(z_score))
    print('real data RMSD * z-factor = RMSD')
    print('RMSD mag:'+str(mag_RMSD)+'[-] RMSD len:'+str(len_RMSD)+'[km]')

    print(len(gensim_data_obs['time']),'data points for the observed meteor')

    # if cml_args.delete_sim then delete the folder that contains the simulations
    if cml_args.delete_sim:
        # Initialize a set to store unique 'vXX' directories
        directories_to_delete = set()

        # Loop over each solution_id in your DataFrame
        for solution_id in pd_datafram_PCA_sim['solution_id']:
            # Get the directory of the file
            dir_path = os.path.dirname(solution_id)
            # Split the path into its components
            path_parts = dir_path.split(os.sep)
            # Iterate over the parts to find the 'vXX' directory
            for idx, part in enumerate(path_parts):
                if part.startswith('v') and len(part) == 3 and part[1:].isdigit():
                    # Reconstruct the path up to the 'vXX' directory
                    v_dir = os.sep.join(path_parts[:idx+1])
                    directories_to_delete.add(v_dir)
                    break  # Stop after finding the 'vXX' directory

        # Delete each 'vXX' directory if it exists
        for dir_path in directories_to_delete:
            if os.path.isdir(dir_path):
                shutil.rmtree(dir_path)
                print(f"Deleted directory: {dir_path}")


    if cml_args.save_results_dir != r'':
        # Ensure the destination path includes the same folder name as the source
        dest_path_with_name = os.path.join(cml_args.save_results_dir, save_results_folder)
        print(f"Directory copied from {output_folder+os.sep+save_results_folder} to {dest_path_with_name} (if it exists)")

    # Timing end
    end_time = time.time()
    
    # Compute elapsed time
    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    # print('Elapsed time in seconds:',elapsed_time)
    print(f"Elapsed time: {int(hours):02}:{int(minutes):02}:{int(seconds):02}")

    # Close the Logger to ensure everything is written to the file STOP COPY in TXT file
    sys.stdout.close()

    # Reset sys.stdout to its original value if needed
    sys.stdout = sys.__stdout__

    # Check if the destination directory exists
    if cml_args.save_results_dir != r'':

        # check if cml_args.save_results_dir exist if not create it
        if not os.path.exists(cml_args.save_results_dir):
            print(f"Directory {cml_args.save_results_dir} does not exist, create it.")
            # os.makedirs(cml_args.save_results_dir)
            mkdirP(cml_args.save_results_dir)

        # Ensure the destination path includes the same folder name as the source
        dest_path_with_name = os.path.join(cml_args.save_results_dir, save_results_folder)
        if os.path.exists(dest_path_with_name):
            print(f"Directory {dest_path_with_name} already exists.")
            # Generate a unique name by appending a counter if the folder already exists
            counter = 1
            while os.path.exists(dest_path_with_name + f"_n{counter}"):
                counter += 1
            dest_path_with_name += f"_n{counter}"
            print(f"Using new directory name: {dest_path_with_name}")

            # Copy the entire directory to the new destination
            shutil.copytree(os.path.join(output_folder, save_results_folder), dest_path_with_name)
        else:
            # Copy the entire directory to the new destination
            shutil.copytree(os.path.join(output_folder, save_results_folder), dest_path_with_name)
        print(f"Directory copied from {os.path.join(output_folder, save_results_folder)} to {dest_path_with_name}")

    print()








if __name__ == "__main__":

    import argparse

    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="From Observation and simulated data we select the most likely through Montecarlo method and store results to disk.")
    # C:\Users\maxiv\Desktop\RunTest\TRUEerosion_sim_v59.84_m1.33e-02g_rho0209_z39.8_abl0.014_eh117.3_er0.636_s1.61.json
    # C:\Users\maxiv\Desktop\20230811-082648.931419
    # 'C:\Users\maxiv\Desktop\jsontest\Simulations_PER_v65_fast\TRUEerosion_sim_v65.00_m7.01e-04g_rho0709_z51.7_abl0.015_eh115.2_er0.483_s2.46.json'
    # '/home/mvovk/Documents/json_test/Simulations_PER_v57_slow/PER_v57_slow.json,/home/mvovk/Documents/json_test/Simulations_PER_v59_heavy/PER_v59_heavy.json,/home/mvovk/Documents/json_test/Simulations_PER_v60_heavy_shallow/PER_v61_heavy_shallow.json,/home/mvovk/Documents/json_test/Simulations_PER_v60_heavy_steep/PER_v60_heavy_steep.json,/home/mvovk/Documents/json_test/Simulations_PER_v60_light/PER_v60_light.json,/home/mvovk/Documents/json_test/Simulations_PER_v61_shallow/PER_v61_shallow.json,/home/mvovk/Documents/json_test/Simulations_PER_v62_steep/PER_v62_steep.json,/home/mvovk/Documents/json_test/Simulations_PER_v65_fast/PER_v65_fast.json'
    arg_parser.add_argument('--input_dir', metavar='INPUT_PATH', type=str, default=r'/home/mvovk/Documents/json_test/Simulations_PER_v59_heavy/PER_v59_heavy.json', \
       help="Path were are store both simulated and observed shower .csv file.")
    # arg_parser.add_argument('input_dir', metavar='INPUT_PATH', type=str, \
    #     help="Path were are store both simulated and observed shower .csv file.")
    
    arg_parser.add_argument('--save_results_dir', metavar='SAVE_OUTPUT_PATH', type=str, default=r'/home/mvovk/Documents/json_test/Results',\
        help="Path were to store the results, by default the same as the input_dir.")

    arg_parser.add_argument('--repeate_research', metavar='REPEATE_RESEARCH', type=int, default=1, \
        help="By default 1 (no re computation), check the consistency of the result by re-trying multiple times creating new simulation to test the precision of the results, this set delete_all to True.")

    arg_parser.add_argument('--fps', metavar='FPS', type=int, default=32, \
        help="Number of frames per second of the video, by default 32 like EMCCD.")
    
    arg_parser.add_argument('--delete_all', metavar='DELETE_ALL', type=bool, default=False, \
        help="By default set to False, if set to True delete all directories and files.")
    
    arg_parser.add_argument('--delete_old', metavar='DELETE_OLD', type=bool, default=False, \
        help="By default set to False, if set to True delete Slected and Results directory and all files except for the sim and obs csv file and the Simulations folder.")
    
    arg_parser.add_argument('--MetSim_json', metavar='METSIM_JSON', type=str, default='_sim_fit_latest.json', \
        help="json file extension where are stored the MetSim constats, by default _sim_fit_latest.json.")   

    arg_parser.add_argument('--nobs', metavar='OBS_NUM', type=int, default=50, \
        help="Number of Observation that will be resampled.")
    
    arg_parser.add_argument('--nsim', metavar='SIM_NUM', type=int, default=10000, \
        help="Number of simulations to generate.")
    
    arg_parser.add_argument('--nsim_refine_step', metavar='SIM_NUM_REFINE', type=int, default=1000, \
        help="Minimum number of results that are in the CI that have to be found.")

    arg_parser.add_argument('--min_nresults', metavar='SIM_RESULTS', type=int, default=100, \
        help="Minimum number of results that are in the CI that have to be found.")
    
    arg_parser.add_argument('--ntry', metavar='NUM_TRY', type=int, default=5, \
        help="Number of failed attemp allowed to generate the set number of similar simulations, by default 3.")

    arg_parser.add_argument('--fix_n_sim', metavar='FIX_NUM_SIM', type=bool, default=True, \
        help="do not change the number of simularions if the csv file is smaller than the SIM_RESULTS.")

    arg_parser.add_argument('--resample_sim', metavar='RESAMPLE_SIM', type=bool, default=False, \
        help="if the number of simulations in the csv file is above SIM_NUM then resample the csv file base on the simulations.")
    
    arg_parser.add_argument('--delete_sim', metavar='DELETE_SIM', type=bool, default=False, \
        help="Delete the simulations after the entire run.")
    
    arg_parser.add_argument('--mag_rmsd', metavar='mag_RMSD', type=float, default=0, \
        help="Minimum absolute Magnitude RMSD = mag_rmsd*conf_lvl.")
    
    arg_parser.add_argument('--len_rmsd', metavar='len_RMSD', type=float, default=0, \
        help="Minimum lenght RMSD = len_rmsd*conf_lvl.")

    arg_parser.add_argument('--conf_lvl', metavar='CONF_LVL', type=float, default=95, \
        help="Confidene level that multiply the RMSD mag and len, by default set to 95%.")

    arg_parser.add_argument('--use_PCA', metavar='USE_PCA', type=bool, default=False, \
        help="Use PCA method to initially estimate possible candidates.")

    arg_parser.add_argument('--nsel_forced', metavar='SEL_NUM_FORCED', type=int, default=0, \
        help="Number of selected simulations forced to consider instead of choosing the knee of the distance function.")
    
    arg_parser.add_argument('--PCA_percent', metavar='PCA_PERCENT', type=int, default=99, \
        help="Percentage of the variance explained by the PCA.")

    arg_parser.add_argument('--YesPCA', metavar='YESPCA', type=str, default=[], \
        help="Use specific variable to considered in PCA.")

    arg_parser.add_argument('--NoPCA', metavar='NOPCA', type=str, default=['rmsd_mag', 'rmsd_len', 'avg_lag','a1_acc_jac','a2_acc_jac','a_acc','b_acc','c_acc','c_mag_init','c_mag_end','a_t0', 'b_t0', 'c_t0'], \
        help="Use specific variable NOT considered in PCA.")

    arg_parser.add_argument('--save_test_plot', metavar='SAVE_TEST_PLOT', type=bool, default=False, \
        help="save test plots of the realization and the simulations and more plots in PCA control plots.")
    
    arg_parser.add_argument('--optimize', metavar='OPTIMIZE', type=bool, default=False, \
        help="Run optimization step to have more precise results but increase the computation time, automatically active if no solution found for 5 events.")

    arg_parser.add_argument('--number_optimized', metavar='NUMBER_OPTIMZED', type=int, default=0, \
        help="ONLY ACTIVE IF OPTIMIZE=True, Number of optimized simulations that have to be optimized starting starting from the best, 0 means all, by default 0.")

    arg_parser.add_argument('--esclude_real_solution_from_selection', metavar='ESCLUDE_REAL_SOLUTION_FROM_SELECTION', type=bool, default=False, \
        help="When use a generate simulation you can select to exclude the real result with True or also consider it in the distance calculations with False.")
    
    arg_parser.add_argument('--ref_opt_path', metavar='REF_OPT_PATH', type=str, default=r'/home/mvovk/WesternMeteorPyLib/wmpl/MetSim/AutoRefineFit_options.txt', \
        help="path and name of like C: path + AutoRefineFit_options.txt")

    arg_parser.add_argument('--cores', metavar='CORES', type=int, default=None, \
        help="Number of cores to use. All by default.")

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    # needed to add fps when multiprocessing
    fps = cml_args.fps

    #########################
    warnings.filterwarnings('ignore')

    if cml_args.optimize:
        # check if the file exist
        if not os.path.isfile(cml_args.ref_opt_path):
            # If the file is not found, check in the parent directory
            parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            cml_args.ref_opt_path = os.path.join(parent_directory, 'AutoRefineFit_options.txt')
            if not os.path.isfile(cml_args.ref_opt_path):
                print('file '+cml_args.ref_opt_path+' not found')
                print("You need to specify the correct path and name of the AutoRefineFit_options.txt file in --ref_opt_path, like: C:\\path\\AutoRefineFit_options.txt")
                sys.exit()

    if cml_args.repeate_research <= 1:
        cml_args.repeate_research = 1
    else:
        print('Number of repeating results search:',cml_args.repeate_research)
        cml_args.delete_all = True

    # check if the input_dir has a comma if so split the string and create a list
    if ',' in cml_args.input_dir:
        cml_args.input_dir = cml_args.input_dir.split(',')
        print('Number of input directories or files:',len(cml_args.input_dir))
    else:
        cml_args.input_dir = [cml_args.input_dir]

    for ii in range(cml_args.repeate_research):

        for input_dir_or_file in cml_args.input_dir:

            # set up observation folder
            Class_folder_files=SetUpObservationFolders(input_dir_or_file, cml_args.MetSim_json)
            input_folder_file=Class_folder_files.input_folder_file

            # print only the file name in the directory split the path and take the last element
            print('Number of trajectory.pickle files found:',len(input_folder_file))
            # print every trajectory_file 
            print('List of trajectory files:')
            # print them line by line and not in a single array [trajectory_file for trajectory_file, file_name, input_folder, output_folder, trajectory_Metsim_file in input_folder_file]
            print('\n'.join([trajectory_file for trajectory_file, file_name, input_folder, output_folder, trajectory_Metsim_file in input_folder_file]))
            print()

            for trajectory_file, file_name, input_folder, output_folder, trajectory_Metsim_file in input_folder_file:
                # run the main function
                main_PhysUncert(trajectory_file, file_name, input_folder, output_folder, trajectory_Metsim_file, cml_args)
