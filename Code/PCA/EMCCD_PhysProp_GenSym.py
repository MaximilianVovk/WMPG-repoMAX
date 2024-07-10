"""
The code is used to extract the physical properties of the simulated showers from EMCCD observations
by selecting the most similar simulated events in the PC space using:
- Mode of the siumulated events
- The min of the KDE esults
- Principal Component Regression (PCR)
"""

import json
import copy
import pandas as pd
import matplotlib.pyplot as plt
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
# from heapq import nsmallest
import wmpl
from wmpl.MetSim.GUI import loadConstants
import shutil
from scipy.stats import kurtosis, skew
from wmpl.Utils.OSTools import mkdirP
import math
from scipy.stats import gaussian_kde
from wmpl.Utils.PyDomainParallelizer import domainParallelizer
from scipy.linalg import svd
from wmpl.MetSim.GUI import loadConstants, SimulationResults
from wmpl.MetSim.MetSimErosion import runSimulation, Constants
# from scipy.optimize import curve_fit # faster 
# from scipy.optimize import basinhopping # slower but more accurate
from matplotlib.colors import Normalize
from scipy.optimize import minimize
import scipy.optimize as opt
import sys
from scipy.interpolate import UnivariateSpline
from scipy.stats import zscore
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import scipy.spatial
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from wmpl.Utils.AtmosphereDensity import fitAtmPoly
from sklearn.cluster import KMeans
import scipy.stats as stats
import statsmodels.api as sm
from sklearn.preprocessing import PowerTransformer
from wmpl.MetSim.ML.GenerateSimulations import generateErosionSim,saveProcessedList,MetParam
from wmpl.Utils.TrajConversions import J2000_JD, date2JD
import warnings
import itertools


# CONSTANTS ###########################################################################################

FPS = 32
NAME_SUFX_GENSIM = "_GenSim"
NAME_SUFX_CSV_OBS = "_obs.csv"
NAME_SUFX_CSV_SIM = "_sim.csv"
NAME_SUFX_CSV_CURRENT_FIT = "_fit_sim.csv"
NAME_SUFX_CSV_PHYSICAL_FIT_RESULTS = "_physical_prop.csv"

# Length of data that will be used as an input during training
DATA_LENGTH = 256
# Default number of minimum frames for simulation
MIN_FRAMES_VISIBLE = 4

# python -m EMCCD_PCA_Shower_PhysProp "C:\Users\maxiv\Documents\UWO\Papers\1)PCA\PCA_Error_propagation\TEST" "PER" "C:\Users\maxiv\Documents\UWO\Papers\1)PCA\PCA_Error_propagation" 1000
# python -m EMCCD_PCA_Shower_PhysProp "C:\Users\maxiv\Documents\UWO\Papers\1)PCA\PCA_Error_propagation\TEST" "PER" "C:\Users\maxiv\Documents\UWO\Papers\1)PCA\PCA_Error_propagation" 1000 > output.txt    

# FUNCTIONS ###########################################################################################

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
    l_before = np.zeros_like(t_before)+c

    # Compute the lag quadratically after t0
    l_after = -abs(a)*(t_after - t0)**3 - abs(b)*(t_after - t0)**2 + c

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

    # Compute the velocity quadratically after t0
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


def fit_lag_t0_RMSD(lag_data,time_data,velocity_data):
    v_init=velocity_data[0]
    # initial guess of deceleration decel equal to linear fit of velocity
    p0 = [np.mean(lag_data), 0, 0, np.mean(time_data)]
    opt_res = opt.minimize(lag_residual, p0, args=(np.array(time_data), np.array(lag_data)), method='Nelder-Mead')
    a_t0, b_t0, c_t0, t0 = opt_res.x
    fitted_lag_t0 = cubic_lag(np.array(time_data), a_t0, b_t0, c_t0, t0)
    
    opt_res_vel = opt.minimize(vel_residual, [a_t0, b_t0, v_init, t0], args=(np.array(time_data), np.array(velocity_data)), method='Nelder-Mead')
    a_t0, b_t0, v_init_new, t0 = opt_res_vel.x
    fitted_vel_t0 = cubic_velocity(np.array(time_data), a_t0, b_t0, v_init, t0)

    fitted_acc_t0 = cubic_acceleration(np.array(time_data), a_t0, b_t0, t0)
    residuals_t0 = lag_data - fitted_lag_t0
    rmsd_t0 = np.sqrt(np.mean(residuals_t0 ** 2))

    return fitted_lag_t0, residuals_t0, rmsd_t0, 'Cubic Fit', fitted_vel_t0, fitted_acc_t0


def find_noise_of_data(data, plot_case=False):
    # make a copy of data_obs
    data_obs = copy.deepcopy(data)

    fitted_lag_t0_lag, residuals_t0_lag, rmsd_t0_lag, fit_type_lag, fitted_vel_t0, fitted_acc_t0 = fit_lag_t0_RMSD(data_obs['lag'],data_obs['time'], data_obs['velocities'])
    # now do it for fit_mag_polin2_RMSD
    fit_pol_mag, residuals_pol_mag, rmsd_pol_mag, fit_type_mag = fit_mag_polin2_RMSD(data_obs['absolute_magnitudes'],data_obs['time'])

    # create a pd dataframe with fit_pol_mag and fitted_vel_t0 and time and height
    fit_funct = {
        'velocities': fitted_vel_t0,
        'height': data_obs['height'],
        'absolute_magnitudes': fit_pol_mag,
        'time': data_obs['time'],
    }

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

def generate_observation_realization(data, rmsd_lag, rmsd_mag, fit_pol_mag_real, fitted_lag_t0_lag_real,name='', fig='', ax='', plot_case=False):

    # print a . so that the next will be on the same line
    print('.', end='')
    # make a copy of data_obs
    data_obs = copy.deepcopy(data)
    fit_pol_mag = copy.deepcopy(fit_pol_mag_real)
    fitted_lag_t0_lag = copy.deepcopy(fitted_lag_t0_lag_real)

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

    ### ###

    # data_obs['lag']=np.array(data_obs['length'])-(data_obs['v_init']*np.array(data_obs['time'])+data_obs['length'][0])
    data_obs['length']= np.array(data_obs['lag'])+(data_obs['v_init']*np.array(data_obs['time'])+data_obs['length'][0])

    # get the new velocity with noise
    for vel_ii in range(1,len(data_obs['time'])-1):
        diff_1=abs((data_obs['time'][vel_ii]-data_obs['time'][vel_ii-1])-1.0/FPS)
        diff_2=abs((data_obs['time'][vel_ii+1]-data_obs['time'][vel_ii-1])-1.0/FPS)

        if diff_1<diff_2:
            data_obs['velocities'][vel_ii]=(data_obs['length'][vel_ii]-data_obs['length'][vel_ii-1])/(data_obs['time'][vel_ii]-data_obs['time'][vel_ii-1])
        else:
            data_obs['velocities'][vel_ii+1]=(data_obs['length'][vel_ii+1]-data_obs['length'][vel_ii-1])/(data_obs['time'][vel_ii+1]-data_obs['time'][vel_ii-1])
    
    if plot_case:
        plot_side_by_side(data_obs,fig, ax)

    # compute the average velocity
    data_obs['v_avg']=np.mean(data_obs['velocities']) # m/s

    # data_obs['v_avg']=data_obs['v_avg']*1000 # km/s

    pd_datfram_PCA = array_to_pd_dataframe_PCA(data_obs)

    return pd_datfram_PCA


#### Generate Simulations #########################################################################

class ErosionSimParametersEMCCD_Comet(object):
    def __init__(self):
        """ Range of physical parameters for the erosion model, EMCCD system for Perseids. """


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
        self.P_0M = 935

        # System FPS
        self.fps = 32

        # Time lag of length measurements (range in seconds) - accomodate CAMO tracking delay of 8 frames
        #   This should be 0 for all other systems except for the CAMO mirror tracking system
        self.len_delay_min = 0
        self.len_delay_max = 0

        # Simulation height range (m) that will be used to map the output to a grid
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

        # SD of noise in length (m)
        self.len_noise = 20.0

        ### ###


        ### Fit parameters ###

        # Length of input data arrays that will be given to the neural network
        self.data_length = DATA_LENGTH

        ### ###


        ### Output normalization range ###

        # Height range (m)
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

def run_simulation(path_and_file_MetSim, real_event):
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

    # # Minimum height (m)
    # const_nominal.h_kill = 60000

    # # Initial meteoroid height (m)
    # const_nominal.h_init = 180000

    # Run the simulation
    frag_main, results_list, wake_results = runSimulation(const_nominal, \
        compute_wake=False)

    simulation_MetSim_object = SimulationResults(const_nominal, frag_main, results_list, wake_results)

    # # print the column of simulation_MetSim_object to see what is inside
    # print(simulation_MetSim_object.__dict__.keys())
    # print(simulation_MetSim_object.const.__dict__.keys())
  
    # ax[0].plot(sr_nominal_1D_KDE.abs_magnitude, sr_nominal_1D_KDE.leading_frag_height_arr/1000, label="Mode", color='r')
    # ax[2].plot(sr_nominal.leading_frag_vel_arr/1000, sr_nominal.leading_frag_height_arr/1000, color='k', abel="Simulated")

    gensim_data_metsim = read_RunSim_output(simulation_MetSim_object, real_event, path_and_file_MetSim)

    pd_Metsim  = array_to_pd_dataframe_PCA(gensim_data_metsim)

    return simulation_MetSim_object, gensim_data_metsim, pd_Metsim


def generate_simulations(real_data,simulation_MetSim_object,gensim_data,numb_sim,output_folder, plot_case=False):

    if real_data['solution_id'].iloc[0].endswith('.json'):
        mass_sim=gensim_data['mass']
        v_init_180km =gensim_data['vel_180km']

    else:
        mass_sim= simulation_MetSim_object.const.m_init
        # print('mass_sim',mass_sim)

        v_init_180km = simulation_MetSim_object.const.v_init # in m/s
        # print('v_init_130km',v_init_130km)

    # Init simulation parameters with the given class name
    erosion_sim_params = SIM_CLASSES[SIM_CLASSES_NAMES.index('ErosionSimParametersEMCCD_Comet')]()
        
    # get from real_data the beg_abs_mag value of the first row and set it as the lim_mag_faintest value
    erosion_sim_params.lim_mag_faintest = real_data['beg_abs_mag'].iloc[0]+0.01
    erosion_sim_params.lim_mag_brightest = real_data['beg_abs_mag'].iloc[0]-0.01
    erosion_sim_params.lim_mag_len_end_faintest = real_data['end_abs_mag'].iloc[0]+0.01
    erosion_sim_params.lim_mag_len_end_brightest = real_data['end_abs_mag'].iloc[0]-0.01

    # find the at what is the order of magnitude of the real_data['mass'][0]
    order = int(np.floor(np.log10(mass_sim)))
    # create a MetParam object with the mass range that is above and below the real_data['mass'][0] by 2 orders of magnitude
    erosion_sim_params.m_init = MetParam(mass_sim-10**order, mass_sim+10**order)

    # Initial velocity range (m/s) 
    erosion_sim_params.v_init = MetParam(v_init_180km-100, v_init_180km+100) # 60091.41691

    # Zenith angle range
    erosion_sim_params.zenith_angle = MetParam(np.radians(real_data['zenith_angle'].iloc[0]-0.01), np.radians(real_data['zenith_angle'].iloc[0]+0.01)) # 43.466538

    print('Run',numb_sim,'simulations with :')
    # print all the modfiend values
    print('- velocity: min',erosion_sim_params.v_init.min,'- MAX',erosion_sim_params.v_init.max)

    print('- zenith angle: min',np.degrees(erosion_sim_params.zenith_angle.min),'- MAX',np.degrees(erosion_sim_params.zenith_angle.max))

    print('- Initial mag: min',erosion_sim_params.lim_mag_faintest,'- MAX',erosion_sim_params.lim_mag_brightest)

    print('- Final mag: min',erosion_sim_params.lim_mag_len_end_faintest,'- MAX',erosion_sim_params.lim_mag_len_end_brightest)
 
    print('- Mass: min',erosion_sim_params.m_init.min,'- MAX',erosion_sim_params.m_init.max)

    print('- rho : min',erosion_sim_params.rho.min,'- MAX',erosion_sim_params.rho.max)

    print('- sigma : min',erosion_sim_params.sigma.min,'- MAX',erosion_sim_params.sigma.max)

    print('- erosion_height_start : min',erosion_sim_params.erosion_height_start.min,'- MAX',erosion_sim_params.erosion_height_start.max)

    print('- erosion_coeff : min',erosion_sim_params.erosion_coeff.min,'- MAX',erosion_sim_params.erosion_coeff.max)

    print('- erosion_mass_index : min',erosion_sim_params.erosion_mass_index.min,'- MAX',erosion_sim_params.erosion_mass_index.max)

    print('- erosion_mass_min : min',erosion_sim_params.erosion_mass_min.min,'- MAX',erosion_sim_params.erosion_mass_min.max)

    print('- erosion_mass_max : min',erosion_sim_params.erosion_mass_max.min,'- MAX',erosion_sim_params.erosion_mass_max.max)

    # print('\\hline') #df_sel_save[df_sel_save['solution_id']==only_select_meteors_from][plotvar].values[0]
    # print(f"{to_plot_unit[i]} & {'{:.4g}'.format(df_sel_save[df_sel_save['solution_id']==only_select_meteors_from][plotvar].values[0])} & {'{:.4g}'.format(x_10mode)} & {'{:.4g}'.format(densest_point[i])} & {'{:.4g}'.format(sigma_2)} & {'{:.4g}'.format(sigma_97)} \\\\")
    # ii_densest=ii_densest+1

    # Generate simulations using multiprocessing
    input_list = [[output_folder, copy.deepcopy(erosion_sim_params), \
        np.random.randint(0, 2**31 - 1),MIN_FRAMES_VISIBLE] for _ in range(numb_sim)]
    results_list = domainParallelizer(input_list, generateErosionSim, cores=cml_args.cores)

    # print(results_list)

    # count how many None are in the results_list
    count_none=0
    for res in results_list:
        if res is None:
            count_none+=1
            continue
        
    saveProcessedList(output_folder, results_list, erosion_sim_params.__class__.__name__, \
    MIN_FRAMES_VISIBLE)


    
    print('Resulted simulations:', numb_sim-count_none)
    print('Failed siulations', len(results_list)/100*count_none,'%')
    print('Saved',numb_sim-count_none,'simulations in',output_folder)

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
                
                # change res[0] extension to .json
                res[0]=res[0].replace('.pickle','.json')
                print(res[0]) 
                # get the first value of res
                gensim_data_sim=read_GenerateSimulations_output(res[0])

                plot_side_by_side(gensim_data_sim,fig, ax, 'b-')
                jj_plots_curve+=1
                
        plot_side_by_side(gensim_data,fig, ax,'go','Obsevation')

        return fig, ax

    

#### Plot #############################################################################


def check_axis_inversion(ax):
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    is_x_inverted = x_max < x_min
    is_y_inverted = y_max < y_min
    return is_x_inverted, is_y_inverted


def plot_side_by_side(data1, fig, ax, colorline1='.', label1='', residual=False, data2=''):
    # check if it is in km/s or in m/s
    obs1= copy.deepcopy(data1)
    if np.mean(obs1['velocities'])>1000:
        # convert to km/s
        obs1['velocities'] = np.array(obs1['velocities'])/1000
        obs1['height'] = np.array(obs1['height'])/1000


    # Plot the simulation results
    if residual == True:
        fig, ax = plt.subplots(2, 4, figsize=(14, 6),gridspec_kw={'height_ratios': [ 3, 1],'width_ratios': [ 3, 0.5, 3, 0.5]}, dpi=300) #  figsize=(10, 5), dpi=300 0.5, 3, 3, 0.5
        
        # flat the ax
        ax = ax.flatten()

        ax[0].plot(obs1['time'], obs1['lag'], 'o', label=f'{obs1["station_id"]}')
        ax[0].plot(obs2['tmie'], obs2['lag'], 'k-', label=f'{obs2["station_id"]}')
        ax[0].set_xlabel('Time [s]')
        ax[0].set_ylabel('Lag [m]')
        ax[0].title.set_text(f'Lag - RMSD: {avg_residual_lag:.2f}')
        ax[0].legend()
        ax[0].grid()

        # delete the plot in the middle
        ax[1].axis('off')

        ax[2].plot(obs1['time_data'], obs1['absolute_magnitudes'], 'o', label=f'{obs1["station_id"]}')
        ax[2].plot(obs2['time_data'], obs2['absolute_magnitudes'], 'o', label=f'{obs2["station_id"]}')

        ax[2].plot(obs1['time_data'], spline_fit, 'k-', label=label_fit)
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


    else :
        
        # plot the magnitude curve with height
        if label1 == '':
            ax[0].plot(obs1['absolute_magnitudes'],obs1['height'], colorline1)
        else:
            ax[0].plot(obs1['absolute_magnitudes'],obs1['height'], colorline1, label=label1)
        # show the legend
        if label1 != '':
            ax[0].legend()
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
            ax[1].plot(obs1['time'], obs1['velocities'], colorline1)
        ax[1].set_xlabel('Time [s]')
        ax[1].set_ylabel('Velocity [km/s]')
        ax[1].grid(True)

    plt.tight_layout()

    

#### Reader #############################################################################

def read_GenerateSimulations_output_to_PCA(file_path, name=''):
    if name!='':   
        print(name) 
    gensim_data = read_GenerateSimulations_output(file_path)
    if gensim_data is None:
        return None
    else:
        pd_datfram_PCA = array_to_pd_dataframe_PCA(gensim_data)
        return pd_datfram_PCA


def read_GenerateSimulations_output(file_path):

    f = open(file_path,"r")
    data = json.loads(f.read())

    # show processed event
    print(file_path)

    if data['ht_sampled']!= None: 

        vel_sim=data['simulation_results']['leading_frag_vel_arr']#['brightest_vel_arr']#['leading_frag_vel_arr']#['main_vel_arr']
        ht_sim=data['simulation_results']['leading_frag_height_arr']#['brightest_height_arr']['leading_frag_height_arr']['main_height_arr']
        time_sim=data['simulation_results']['time_arr']#['main_time_arr']
        abs_mag_sim=data['simulation_results']['abs_magnitude']
        len_sim=data['simulation_results']['brightest_length_arr']#['brightest_length_arr']
        
        ht_obs=data['ht_sampled']

        # # find the index of the first element of the simulation that is equal to the first element of the observation
        index_ht_sim=next(x for x, val in enumerate(ht_sim) if val <= ht_obs[0])
        # find the index of the last element of the simulation that is equal to the last element of the observation
        index_ht_sim_end=next(x for x, val in enumerate(ht_sim) if val <= ht_obs[-1])

        abs_mag_sim=abs_mag_sim[index_ht_sim:index_ht_sim_end]
        vel_sim=vel_sim[index_ht_sim:index_ht_sim_end]
        time_sim=time_sim[index_ht_sim:index_ht_sim_end]
        ht_sim=ht_sim[index_ht_sim:index_ht_sim_end]
        len_sim=len_sim[index_ht_sim:index_ht_sim_end]

        closest_indices = find_closest_index(ht_sim, ht_obs)

        Dynamic_pressure= data['simulation_results']['leading_frag_dyn_press_arr']
        Dynamic_pressure= Dynamic_pressure[index_ht_sim:index_ht_sim_end]
        Dynamic_pressure=[Dynamic_pressure[jj_index_cut] for jj_index_cut in closest_indices]

        abs_mag_sim=[abs_mag_sim[jj_index_cut] for jj_index_cut in closest_indices]
        vel_sim=[vel_sim[jj_index_cut] for jj_index_cut in closest_indices]
        time_sim=[time_sim[jj_index_cut] for jj_index_cut in closest_indices]
        ht_sim=[ht_sim[jj_index_cut] for jj_index_cut in closest_indices]
        len_sim=[len_sim[jj_index_cut] for jj_index_cut in closest_indices]

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
        'velocities': vel_sim, # m/s
        'height': ht_sim, # m
        'absolute_magnitudes': abs_mag_sim,
        'lag': len_sim-(vel_sim[0]*np.array(time_sim)+len_sim[0]), # m
        'length': len_sim, # m
        'time': time_sim, # s
        'v_avg': np.mean(vel_sim), # m/s
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


def read_with_noise_GenerateSimulations_output(file_path):

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

        gensim_data = {
        'name': file_path,
        'type': 'Observation_sim',
        'vel_180km': data['params']['v_init']['val'], # m/s
        'v_init': data['vel_sampled'][0], # m/s
        'velocities': data['vel_sampled'], # m/s
        'height': data['ht_sampled'], # m
        'absolute_magnitudes': data['mag_sampled'],
        'lag': data['lag_sampled'], # m
        'length': data['len_sampled'], # m
        'time': data['time_sampled'], # s
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
    
    ht_obs=real_event['height']

    # find the index of the first element of the simulation that is equal to the first element of the observation
    index_ht_sim=next(x for x, val in enumerate(ht_sim) if val <= ht_obs[0])
    # find the index of the last element of the simulation that is equal to the last element of the observation
    index_ht_sim_end=next(x for x, val in enumerate(ht_sim) if val <= ht_obs[-1])

    abs_mag_sim=abs_mag_sim[index_ht_sim:index_ht_sim_end]
    vel_sim=vel_sim[index_ht_sim:index_ht_sim_end]
    time_sim=time_sim[index_ht_sim:index_ht_sim_end]
    ht_sim=ht_sim[index_ht_sim:index_ht_sim_end]
    len_sim=len_sim[index_ht_sim:index_ht_sim_end]
    Dynamic_pressure= Dynamic_pressure[index_ht_sim:index_ht_sim_end]

    closest_indices = find_closest_index(ht_sim, ht_obs)

    abs_mag_sim=[abs_mag_sim[jj_index_cut] for jj_index_cut in closest_indices]
    vel_sim=[vel_sim[jj_index_cut] for jj_index_cut in closest_indices]
    time_sim=[time_sim[jj_index_cut] for jj_index_cut in closest_indices]
    ht_sim=[ht_sim[jj_index_cut] for jj_index_cut in closest_indices]
    len_sim=[len_sim[jj_index_cut] for jj_index_cut in closest_indices]
    Dynamic_pressure=[Dynamic_pressure[jj_index_cut] for jj_index_cut in closest_indices]

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
        'velocities': vel_sim, # m/s
        'height': ht_sim, # m
        'absolute_magnitudes': abs_mag_sim,
        'lag': len_sim-(vel_sim[0]*np.array(time_sim)+len_sim[0]), # m
        'length': len_sim, # m
        'time': time_sim, # s
        'v_avg': np.mean(vel_sim), # m/s
        'Dynamic_pressure_peak_abs_mag': Dynamic_pressure[np.argmin(abs_mag_sim)],
        'zenith_angle': real_event['zenith_angle'],
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
    jd_dat=traj.jdt_ref
    obs_data = []
    for obs in traj.observations:
        if obs.station_id == "01G" or obs.station_id == "02G" or obs.station_id == "01F" or obs.station_id == "02F" or obs.station_id == "1G" or obs.station_id == "2G" or obs.station_id == "1F" or obs.station_id == "2F":
            obs_dict = {
                'v_init': obs.v_init, # m/s
                'velocities': np.array(obs.velocities), # m/s
                'height': np.array(obs.model_ht), # m
                'absolute_magnitudes': np.array(obs.absolute_magnitudes),
                'lag': np.array(obs.lag), # m
                'length': np.array(obs.length), # m
                'time': np.array(obs.time_data), # s
                # 'station_id': obs.station_id
                'elev_data':  np.array(obs.elev_data)
            }
            obs_dict['velocities'][0] = obs_dict['v_init']
            obs_data.append(obs_dict)
                
            lat_dat=obs.lat
            lon_dat=obs.lon

        else:
            print(obs.station_id,'Station not in the list of stations')
            continue
    
    # Save distinct values for the two observations
    obs1, obs2 = obs_data[0], obs_data[1]
    
    # Combine obs1 and obs2
    combined_obs = {}
    for key in ['velocities', 'height', 'absolute_magnitudes', 'lag', 'length', 'time', 'elev_data']:
        combined_obs[key] = np.concatenate((obs1[key], obs2[key]))

    # Order the combined observations based on time
    sorted_indices = np.argsort(combined_obs['time'])
    for key in ['time', 'velocities', 'height', 'absolute_magnitudes', 'lag', 'length', 'elev_data']:
        combined_obs[key] = combined_obs[key][sorted_indices]

    Dynamic_pressure_peak_abs_mag=(wmpl.Utils.Physics.dynamicPressure(lat_dat, lon_dat, combined_obs['height'][np.argmin(combined_obs['absolute_magnitudes'])], jd_dat, combined_obs['velocities'][np.argmin(combined_obs['absolute_magnitudes'])]))
    zenith_angle=(90 - combined_obs['elev_data'][0]*180/np.pi)

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

        type_sim='Observation'

        # put all the varible in a array mass, rho, sigma, erosion_height_start, erosion_coeff, erosion_mass_index, erosion_mass_min, erosion_mass_max, erosion_range, erosion_energy_per_unit_cross_section_arr, erosion_energy_per_unit_mass_arr
        output_phys = [mass, rho, sigma, erosion_height_start, erosion_coeff, erosion_mass_index, erosion_mass_min, erosion_mass_max, erosion_range, erosion_energy_per_unit_cross_section_arr, erosion_energy_per_unit_mass_arr]

    # delete the elev_data from the combined_obs
    del combined_obs['elev_data']

    # add to combined_obs the avg velocity and the peak dynamic pressure and all the physical parameters
    combined_obs['name'] = file_path    
    combined_obs['v_init'] = combined_obs['velocities'][0]
    combined_obs['type'] = type_sim
    combined_obs['v_avg'] = v_avg
    combined_obs['Dynamic_pressure_peak_abs_mag'] = Dynamic_pressure_peak_abs_mag
    combined_obs['zenith_angle'] = zenith_angle
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
            # mass=(data['m_init'])
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

    # put all the varible in a array mass, rho, sigma, erosion_height_start, erosion_coeff, erosion_mass_index, erosion_mass_min, erosion_mass_max, erosion_range, erosion_energy_per_unit_cross_section_arr, erosion_energy_per_unit_mass_arr
    output_phys = [mass, rho, sigma, erosion_height_start, erosion_coeff, erosion_mass_index, erosion_mass_min, erosion_mass_max, erosion_range, erosion_energy_per_unit_cross_section_arr, erosion_energy_per_unit_mass_arr]
    
    return output_phys



def array_to_pd_dataframe_PCA(data):

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

    # fit a line to the throught the obs_vel and ht_sim
    index_ht_peak = next(x for x, val in enumerate(data_array['height']) if val <= peak_mag_height)

    # check if the ht_obs[:index_ht_peak] and abs_mag_obs[:index_ht_peak] are empty
    a3_Inabs, b3_Inabs, c3_Inabs = np.polyfit(data_array['height'][:index_ht_peak], data_array['absolute_magnitudes'][:index_ht_peak], 2)

    # check if the ht_obs[index_ht_peak:] and abs_mag_obs[index_ht_peak:] are empty
    a3_Outabs, b3_Outabs, c3_Outabs = np.polyfit(data_array['height'][index_ht_peak:], data_array['absolute_magnitudes'][index_ht_peak:], 2)


    ######## SKEW KURT ################ 
    # create a new array with the same values as time_pickl
    index=[]
    # if the distance between two index is smalle than 0.05 delete the second one
    for i in range(len(data_array['time'])-1):
        if data_array['time'][i+1]-data_array['time'][i] < 0.01:
            # save the index as an array
            index.append(i+1)
    # delete the index from the list
    time_pickl = np.delete(data_array['time'], index)
    abs_mag_pickl = np.delete(data_array['time'], index)

    abs_mag_pickl = [0 if math.isnan(x) else x for x in abs_mag_pickl]

    # subrtract the max value of the mag to center it at the origin
    mag_sampled_norm = (-1)*(abs_mag_pickl - np.max(abs_mag_pickl))
    # check if there is any negative value and add the absolute value of the min value to all the values
    mag_sampled_norm = mag_sampled_norm + np.abs(np.min(mag_sampled_norm))
    # normalize the mag so that the sum is 1
    time_sampled_norm= time_pickl - np.mean(time_pickl)
    # mag_sampled_norm = mag_sampled_norm/np.sum(mag_sampled_norm)
    mag_sampled_norm = mag_sampled_norm/np.max(mag_sampled_norm)
    # substitute the nan values with zeros
    mag_sampled_norm = np.nan_to_num(mag_sampled_norm)

    # create an array with the number the ammount of same number equal to the value of the mag
    mag_sampled_distr = []
    mag_sampled_array=np.asarray(mag_sampled_norm*1000, dtype = 'int')
    for i in range(len(abs_mag_pickl)):
        # create an integer form the array mag_sampled_array[i] and round of the given value
        numbs=mag_sampled_array[i]
        # invcrease the array number by the mag_sampled_distr numbs 
        # array_nu=(np.ones(numbs+1)*i_pos).astype(int)
        array_nu=(np.ones(numbs+1)*time_sampled_norm[i])
        mag_sampled_distr=np.concatenate((mag_sampled_distr, array_nu))
    
    # # # plot the mag_sampled_distr as an histogram
    # plt.hist(mag_sampled_distr)
    # plt.show()

    # kurtosyness.append(kurtosis(mag_sampled_distr))
    # skewness.append(skew(mag_sampled_distr))
    kurtosyness=kurtosis(mag_sampled_distr)
    skewness=skew(mag_sampled_distr)

    ################################# 

    

    # Data to populate the dataframe
    data_picklefile_pd = {
        'solution_id': [data_array['name']],
        'type': [data_array['type']],
        'vel_init_norot': [data_array['v_init']],
        'vel_avg_norot': [data_array['v_avg']],
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
        'kurtosis': [kurtosyness],
        'skew': [skewness],
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

# Function to get trajectory data folder
def find_and_extract_trajectory_files(directory, MetSim_extention):
    trajectory_files = []
    file_names = []
    output_folders = []
    input_folders = []
    trajectory_Metsim_file = []

    for root, dirs, files in os.walk(directory):

        # go in each folder and find the file with the end _trajectory.pickle but skip the folder with the name GenSim
        if 'GenSim' in root:
            continue
        
        csv_file_found=False

        for file in files:
            if file.endswith(NAME_SUFX_CSV_OBS):
                # open
                csv_file_found=True
                real_data = pd.read_csv(os.path.join(root, file))
                if root not in real_data['solution_id'][0]:
                    print('The solution_id in the csv file is not the same as the folder name or does not exist in the folder name:', root)
                    continue
                # split real_data['solution_id'][0] in the directory and the name of the file
                _ , file_from_csv = os.path.split(real_data['solution_id'][0])
                
                base_name = os.path.splitext(file_from_csv)[0]  # Remove the file extension
                #check if the file_from_csv endswith "_trajectory" if yes then extract the number 20230405_010203
                if base_name.endswith("_trajectory"):
                    variable_name = base_name.replace("_trajectory", "")  # Extract the number 20230405_010203
                    output_folder_name = base_name.replace("_trajectory", NAME_SUFX_GENSIM) # _GenSim folder whre all generated simulations are stored
                else:
                    variable_name = base_name
                    output_folder_name = base_name + NAME_SUFX_GENSIM
                

                if file_from_csv.endswith("json"):
                    MetSim_phys_file_path = os.path.join(root, file_from_csv)
                else:
                    # check if MetSim_phys_file_path exist
                    if os.path.isfile(os.path.join(root, variable_name + MetSim_extention)):
                        # print did not find with th given extention revert to default
                        MetSim_phys_file_path = os.path.join(root, variable_name + MetSim_extention)
                    elif os.path.isfile(os.path.join(root, variable_name + '_sim_fit_latest.json')):
                        print(base_name,': No MetSim file with the given extention', MetSim_extention,'reverting to default extention _sim_fit_latest.json')
                        MetSim_phys_file_path = os.path.join(root, variable_name + '_sim_fit_latest.json')
                    else:
                        # do not save the rest of the files
                        print(base_name,': No MetSim file with the given extention', MetSim_extention,'do not consider the folder')
                        continue


                input_folders.append(root)
                trajectory_files.append(os.path.join(root, file))
                file_names.append(variable_name)
                output_folders.append(os.path.join(root, output_folder_name))
                trajectory_Metsim_file.append(MetSim_phys_file_path)

                

        if csv_file_found==False:   
            for file in files:
                if file.endswith("_trajectory.pickle"):
                    base_name = os.path.splitext(file)[0]  # Remove the file extension
                    variable_name = base_name.replace("_trajectory", "")  # Extract the number 20230405_010203
                    output_folder_name = base_name.replace("_trajectory", NAME_SUFX_GENSIM) # _GenSim folder whre all generated simulations are stored

                    # check if MetSim_phys_file_path exist
                    if os.path.isfile(os.path.join(root, variable_name + MetSim_extention)):
                        # print did not find with th given extention revert to default
                        MetSim_phys_file_path = os.path.join(root, variable_name + MetSim_extention)
                    elif os.path.isfile(os.path.join(root, variable_name + '_sim_fit_latest.json')):
                        print(base_name,': No MetSim file with the given extention', MetSim_extention,'reverting to default extention _sim_fit_latest.json')
                        MetSim_phys_file_path = os.path.join(root, variable_name + '_sim_fit_latest.json')
                    else:
                        # do not save the rest of the files
                        print(base_name,': No MetSim file with the given extention', MetSim_extention,'do not consider the folder')
                        continue

                    input_folders.append(root)
                    trajectory_files.append(os.path.join(root, file))
                    file_names.append(variable_name)
                    output_folders.append(os.path.join(root, output_folder_name))
                    trajectory_Metsim_file.append(MetSim_phys_file_path)

    
    input_list = [[trajectory_files[ii], file_names[ii], input_folders[ii], output_folders[ii], trajectory_Metsim_file[ii]] for ii in range(len(trajectory_files))]

    return input_list


########## Distance ##########################


# Function to find the knee of the distance plot
def diff_dist_index(data_for_meteor, window_of_smothing_avg=3, std_multip_threshold=1):
    #make subtraction of the next element and the previous element of data_for_meteor["distance_meteor"]
    diff_distance_meteor = np.diff(data_for_meteor["distance_meteor"][:int(len(data_for_meteor["distance_meteor"])/10)])
    # histogram plot of the difference with the count on the x axis and diff_distance_meteor on the y axis 
    indices = np.arange(len(diff_distance_meteor))
    # create the cumulative sum of the diff_distance_meteor
    cumsum_diff_distance_meteor = np.cumsum(diff_distance_meteor)
    # normalize the diff_distance_meteor xnormalized = (x - xminimum) / range of x
    diff_distance_meteor_normalized = (diff_distance_meteor - np.min(diff_distance_meteor)) / (np.max(diff_distance_meteor) - np.min(diff_distance_meteor))

    def moving_average_smoothing(data, window_size):
        smoothed_data = np.convolve(data, np.ones(window_size)/window_size, mode='same')
        return smoothed_data

    # apply the smoothing finction
    smoothed_diff_distance_meteor = moving_average_smoothing(diff_distance_meteor_normalized, window_of_smothing_avg)
    
    # fid the first value of the smoothed_diff_distance_meteor that is smaller than the std of the smoothed_diff_distance_meteor
    index10percent = np.where(smoothed_diff_distance_meteor < np.std(smoothed_diff_distance_meteor)*std_multip_threshold)[0][0]-2
    if index10percent<0:
        index10percent=0

   
    # # dimension of the plot 15,5
    # plt.figure(figsize=(15,5))
    # plt.subplot(1,3,1)
    # plt.bar(indices, diff_distance_meteor_normalized,color='blue', edgecolor='black')
    # plt.xlabel('Index')
    # plt.ylabel('Difference')
    # plt.title('Diff normalized')

    # plt.subplot(1,3,2)
    # plt.bar(indices, cumsum_diff_distance_meteor,color='blue', edgecolor='black')
    # plt.xlabel('Index')
    # plt.ylabel('Cumulative sum')
    # plt.title('Cumulative sum diff')

    # plt.subplot(1,3,3)
    # sns.histplot(data_for_meteor, x=data_for_meteor["distance_meteor"][:100], kde=True, cumulative=True, bins=len(data_for_meteor["distance_meteor"]))
    # plt.ylabel('Index')
    # plt.xlabel('distance')
    # plt.title('Dist')  
    # # give more space
    # plt.tight_layout()  
    # plt.show()

    return index10percent

# function to use the mahaloby distance and from the mean of the selected shower
def dist_PCA_space_select_sim(df_sim_PCA, shower_current_PCA_single, cov_inv, meanPCA_current, df_sim_shower, shower_current_single, N_sim_sel):

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
    df_sim_selected = df_sim_shower_dis[:N_sim_sel].drop(['type'], axis=1)
    df_sim_selected['type'] = 'Simulation_sel'

    # create a dataframe with the selected simulated shower characteristics
    df_sim_PCA_dist = df_sim_PCA
    df_sim_PCA_dist['distance_meteor'] = distance_current
    df_sim_PCA_dist = df_sim_PCA_dist.sort_values(by=['distance_meteor']).reset_index(drop=True)
    # delete the shower code
    df_sim_selected_PCA = df_sim_PCA_dist[:N_sim_sel].drop(['type','distance_meteor'], axis=1)

    # make df_sim_selected_PCA an array
    df_sim_selected_PCA = df_sim_selected_PCA.values
    distance_current_mean = []
    for i_shower in range(len(df_sim_selected)):
        distance_current_mean.append(scipy.spatial.distance.euclidean(meanPCA_current, df_sim_selected_PCA[i_shower]))
    df_sim_selected['distance_mean']=distance_current_mean # from the mean of the selected shower

    return df_sim_selected



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



# PCA ####################################################################################

def PCASim(df_sim_shower,df_obs_shower, OUT_PUT_PATH, PCA_percent=99, N_sim_sel=1000, variable_PCA=[], No_var_PCA=['kurtosis','skew','a1_acc_jac','a2_acc_jac','a_acc','b_acc','c_acc','c_mag_init','c_mag_end','a_t0', 'b_t0', 'c_t0'], file_name_obs='', cores_parallel=None, PCA_pairplot=False):
    '''
    This function generate the simulated shower from the erosion model and apply PCA.
    The function read the json file in the folder and create a csv file with the simulated shower and take the data from GenerateSimulation.py folder.
    The function return the dataframe of the selected simulated shower.

    'solution_id','type','vel_init_norot','vel_avg_norot','duration',
    'mass','peak_mag_height','begin_height','end_height','t0','peak_abs_mag','beg_abs_mag','end_abs_mag',
    'F','trail_len','deceleration_lin','deceleration_parab','decel_jacchia','decel_t0','zenith_angle', 'kurtosis','skew',
    'kc','Dynamic_pressure_peak_abs_mag',
    'a_acc','b_acc','c_acc','a1_acc_jac','a2_acc_jac','a_mag_init','b_mag_init','c_mag_init','a_mag_end','b_mag_end','c_mag_end',
    'rho','sigma','erosion_height_start','erosion_coeff', 'erosion_mass_index',
    'erosion_mass_min','erosion_mass_max','erosion_range',
    'erosion_energy_per_unit_cross_section', 'erosion_energy_per_unit_mass'

    '''

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
                variable_PCA.remove(var)

    scaled_sim=df_sim_shower[variable_PCA].copy()
    scaled_sim=scaled_sim.drop(['type','solution_id'], axis=1)

    print(len(scaled_sim.columns),'Variables for PCA:\n',scaled_sim.columns)

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
        print('The MetSim reduction is an outlier, still keep it for the PCA analysis')
        outliers[0]=False

    # Assign df_sim_shower to the version without outliers
    df_sim_shower = df_sim_shower[~outliers].copy()


    if PCA_pairplot:

        output_folder=OUT_PUT_PATH+os.sep+file_name_obs+'_var_real'
        # check if the output_folder exists
        if not os.path.isdir(output_folder):
            mkdirP(output_folder)

        # scale the data so to be easily plot against each other with the same scale
        df_sim_var_sel = df_sim_shower[variable_PCA].copy()
        df_sim_var_sel = df_sim_var_sel.drop(['type','solution_id'], axis=1)

        if len(df_sim_var_sel)>10000:
            # pick randomly 10000 events
            print('Number of events in the simulated :',len(df_sim_var_sel))
            df_sim_var_sel=df_sim_var_sel.sample(n=10000)

        # # loop all pphysical variables
        # physical_vars = ['mass','rho','sigma','erosion_height_start','erosion_coeff','erosion_mass_index','erosion_mass_min','erosion_mass_max']
        # for var_phys in physical_vars:
        #     # # make a subplot of the rho againist each variable_PCA as a scatter plot
        #     # fig, axs = plt.subplots(5, 5, figsize=(20, 15))
            # # make a subplot of the rho againist each variable_PCA as a scatter plot
            # fig, axs = plt.subplots(int(np.ceil(len(physical_vars)/5)), 5, figsize=(20, 15))
            # # flat it
            # axs = axs.flatten()
        #     # flatten the axs array
        #     axs = axs.flatten()
        #     for i, var in enumerate(variable_PCA[2:]):
        #         # plot the rho againist the variable with black borders
        #         axs[i].scatter(df_sim_shower[var], df_sim_shower[var_phys], c='b') #, edgecolors='k', alpha=0.5
        #         # put a green vertical line for the df_obs_shower[var] value
        #         axs[i].axvline(df_obs_shower[var].values[0], color='limegreen', linestyle='--', linewidth=5)
        #         # put a horizontal line for the rho of the first df_sim_shower
        #         axs[i].axhline(df_sim_shower[var_phys].values[0], color='k', linestyle='-', linewidth=2)
        #         # axs[i].set_title(var)
        #         # # as a suptitle put the variable_PCA
        #         # fig.suptitle(var_phys)
        #         if i == 0 or i == 5 or i == 10 or i == 15 or i == 20:
        #             # as a suptitle put the variable_PCA
        #             axs[i].set_ylabel(var_phys)
        #         # x axis
        #         axs[i].set_xlabel(var)

        #         # grid
        #         axs[i].grid()
        #         # make y axis log if the variable is 'erosion_mass_min' 'erosion_mass_max'
        #         if var_phys == 'erosion_mass_min' or var_phys == 'erosion_mass_max':
        #             axs[i].set_yscale('log')
                
        #     # space between the subplots
        #     plt.tight_layout()
        #     # save the figure
        #     plt.savefig(output_folder+os.sep+file_name_obs+var_phys+'_vs_select_PCA.png')
        #     # close the figure
        #     plt.close()


        # make a subplot of the distribution of the variables
        fig, axs = plt.subplots(int(np.ceil(len(variable_PCA[2:])/5)), 5, figsize=(20, 15))
        # flat it
        axs = axs.flatten()
        for i, var in enumerate(variable_PCA[2:]):
            # plot the distribution of the variable
            sns.histplot(df_sim_var_sel[var], kde=True, ax=axs[i], color='b', alpha=0.5, bins=20)
            # axs[i//4, i%4].set_title('Distribution of '+var)
            # put a vertical line for the df_obs_shower[var] value
            axs[i].axvline(df_obs_shower[var].values[0], color='limegreen', linestyle='--', linewidth=5)
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
        plt.savefig(output_folder+os.sep+file_name_obs+'_var_hist_real.png')
        # close the figure
        plt.close()
        


    ##################################### delete var that are not in the 5 and 95 percentile of the simulated shower #####################################

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

                    ii_all=0
                    for i_var in range(len(df_obs_shower[var])):
                        # check if all the values are outside the 5 and 95 percentile of the df_sim_shower if so delete the variable from the variable_PCA
                        if df_obs_shower[var][i_var] < np.percentile(df_sim_shower[var], 1) or df_obs_shower[var][i_var] > np.percentile(df_sim_shower[var], 99):
                            ii_all=+ii_all

                    print(var)

                    if ii_all==len(df_obs_shower[var]):
                        print('The observed and all realization',var,'are not within the 1 and 99 percentile of the simulated meteors')
                        # delete the variable from the variable_PCA
                        variable_PCA.remove(var)
                        # save the var deleted in a variable
                        No_var_PCA_perc.append(var)

                        df_all = df_all.drop(var, axis=1)
                    else:
                        shapiro_test = stats.shapiro(df_all[var])
                        print("Initial Shapiro-Wilk Test:", shapiro_test.statistic,"p-val", shapiro_test.pvalue)

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
                        
                else:
                    print('Variable ',var,' is not in the simulated shower')
            else:
                print('Variable ',var,' is not in the observed shower')



    if PCA_pairplot:
        df_all_nameless_plot=df_all.copy()

        if len(df_all_nameless_plot)>10000:
            # pick randomly 10000 events
            print('Number of events in the simulated:',len(df_all_nameless_plot))
            df_all_nameless_plot=df_all_nameless_plot.sample(n=10000)

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
            axs[i].axvline(df_all_nameless_plot[var].values[len(df_sim_shower[variable_PCA])], color='limegreen', linestyle='--', linewidth=5)       
            # x axis
            axs[i].set_xlabel(var)
            # # grid
            # axs[i//5, i%5].grid()
            if i != 0 and i != 5 and i != 10 and i != 15 and i != 20:
                # delete the y axis
                axs[i].set_ylabel('')
        
        # space between the subplots
        plt.tight_layout()

        # save the figure
        plt.savefig(output_folder+os.sep+file_name_obs+'_var_hist_yeo-johnson.png')
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

    # check if a file with the name "log"+n_PC_in_PCA+"_"+str(len(df_sel))+"ev.txt" already exist
    if os.path.exists(OUT_PUT_PATH+os.sep+file_name_obs+"log_"+str(len(variable_PCA))+"var_"+str(PCA_percent)+"%_"+str(pca.n_components_)+"PC.txt"):
        # remove the file
        os.remove(OUT_PUT_PATH+os.sep+file_name_obs+"log_"+str(len(variable_PCA))+"var_"+str(PCA_percent)+"%_"+str(pca.n_components_)+"PC.txt")
    sys.stdout = Logger(OUT_PUT_PATH,file_name_obs+"log_"+str(len(variable_PCA))+"var_"+str(PCA_percent)+"%_"+str(pca.n_components_)+"PC.txt") # _30var_99%_13PC

    ################################# Apply Varimax rotation ####################################
    loadings = pca.components_.T

    rotated_loadings = varimax(loadings)

    # # chage the loadings to the rotated loadings in the pca components
    pca.components_ = rotated_loadings.T

    # Transform the original PCA scores with the rotated loadings ugly PC space but same results
    # all_PCA = np.dot(all_PCA, rotated_loadings.T[:pca.n_components_, :pca.n_components_])

    ############### PCR ########################################################################################


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
    to_plot_unit=['mass [kg]','rho [kg/m^3]','sigma [s^2/km^2]','erosion height start [km]','erosion coeff [s^2/km^2]','erosion mass index [-]','eros. mass min [kg]','eros. mass max [kg]']
    # multiply y_pred_pcr that has the 'erosion_coeff'*1000000 and 'sigma'*1000000
    y_pred_pcr[:,4]=y_pred_pcr[:,4]*1000000
    y_pred_pcr[:,2]=y_pred_pcr[:,2]*1000000
    # Get the real values
    real_values = df_sim_shower_resample[physical_vars].iloc[0].values
    # multiply the real_values
    real_values[4]=real_values[4]*1000000
    real_values[2]=real_values[2]*1000000


    # Print the predictions alongside the real values
    print("Predicted vs Real Values:")
            # print(output_dir+os.sep+'PhysicProp'+n_PC_in_PCA+'_'+str(len(curr_sel))+'ev_dist'+str(np.round(np.min(curr_sel['distance_meteor']),2))+'-'+str(np.round(np.max(curr_sel['distance_meteor']),2))+'.png')
    for i, unit in enumerate(to_plot_unit):
        print(f'{unit}: Predicted: {y_pred_pcr[0, i]:.4g}, Real: {real_values[i]:.4g}')

    pcr_results = y_pred_pcr.copy()
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
    rows=variable_PCA

    # delete after the 8th cararacter of the string in columns_PC
    rows_8 = [x[:10] for x in rows]

    # Add the variable names as labels on the x-axis and y-axis
    plt.xticks(range(len(rows_8)-2), rows_8[2:], rotation=90)
    plt.yticks(range(len(columns_PC)), columns_PC)

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


    # Get explained variances of principal components
    explained_variance = pca.explained_variance_ratio_

    # Calculate mean and inverse covariance matrix for Mahalanobis distance
    cov_matrix = df_all_PCA.drop(['type'], axis=1).cov()

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
        'MetSim': 'sim',
        'Simulation': 'sim'
    }
    df_all_PCA['group'] = df_all_PCA['type'].map(group_mapping)
    df_obs_shower['group'] = df_obs_shower['type'].map(group_mapping)
    df_obs_PCA['group'] = df_obs_PCA['type'].map(group_mapping)

    # Group by the new column and calculate the mean
    meanPCA = df_all_PCA.groupby('group').mean()

    # drop the sim column
    meanPCA = meanPCA.drop(['sim'], axis=0)

    # print(meanPCA)

    meanPCA_current = meanPCA.loc[(meanPCA.index == 'obs')].values.flatten()
    # take only the value of the mean of the first row
    shower_current = df_obs_shower[df_obs_shower['group'] == 'obs']
    shower_current_PCA = df_obs_PCA[df_obs_PCA['group'] == 'obs']

    # trasform the dataframe in an array
    shower_current_PCA = shower_current_PCA.drop(['type','group'], axis=1).values
        
    # define the distance
    input_list_obs_dist = [[df_sim_PCA, shower_current_PCA[ii], cov_inv, meanPCA_current, df_sim_shower, shower_current.iloc[ii], N_sim_sel] for ii in range(len(shower_current))]
    df_sim_selected = domainParallelizer(input_list_obs_dist, dist_PCA_space_select_sim, cores=cores_parallel)

    df_sel_shower = pd.concat(df_sim_selected)

    # Insert the column at the first position
    df_sel_shower.insert(1, 'distance_mean', df_sel_shower.pop('distance_mean'))
    df_sel_shower.insert(1, 'distance_meteor', df_sel_shower.pop('distance_meteor'))
    df_sel_shower.insert(1, 'solution_id_dist', df_sel_shower.pop('solution_id_dist'))
    df_sel_shower.insert(1, 'type', df_sel_shower.pop('type'))

    df_sel_shower.reset_index(drop=True, inplace=True)

    df_sel_shower.to_csv(OUT_PUT_PATH+os.sep+file_name_obs+'_sim_sel.csv', index=False)

    # print('Selected shower:\n',df_sel_shower)
    # for the name of observation shower check if it has ben selected one with the same name unique for the solution_id_dist ['solution_id_dist'].unique()
    changed_csv=False
    for len_shower in range(len(shower_current)):
        sol_id_dist_search_OG = df_sim_shower['solution_id'][0]
        sol_id_dist_search = shower_current['solution_id'][len_shower]
        # get all the data with the same solution_id_dist
        all_sol_id_dist = df_sel_shower[df_sel_shower['solution_id_dist'] == sol_id_dist_search]
        # check if among all_sol_id_dist there is one with the same solution_id as the sol_id_dist_search
        if len(all_sol_id_dist[all_sol_id_dist['solution_id'] == sol_id_dist_search_OG]) == 0:
            # copy the first row of df_sim_shower and create df_sel_shower_real
            df_sel_shower_real = df_sim_shower.iloc[0].copy()

            # Add the new columns and values directly to the Series
            sol_id_dist_search = 'example_id'  # Replace this with the actual value

            df_sel_shower_real['solution_id_dist'] = sol_id_dist_search
            df_sel_shower_real['distance_meteor'] = 9999
            df_sel_shower_real['type'] = 'Simulation_sel'
            df_sel_shower_real['distance_mean'] = 9999
            # df_sel_shower_real['solution_id'] = sol_id_dist_search

            # add the row to the df_sel_shower
            df_sel_shower = pd.concat([df_sel_shower, df_sel_shower_real])
            changed_csv=True
        # else:
        #     # change the solution_id of the selected shower to the solution_id of the observation shower
        #     df_sel_shower.loc[(df_sel_shower['solution_id_dist'] == sol_id_dist_search) & (df_sel_shower['solution_id'] == sol_id_dist_search_OG), 'solution_id'] = sol_id_dist_search

    if changed_csv:
        df_sel_shower.reset_index(drop=True, inplace=True)
        # save the dataframe to a csv file withouth the index
        df_sel_shower.to_csv(OUT_PUT_PATH+os.sep+file_name_obs+'_sim_sel.csv', index=False)

    # No repetitions

    # Create the new DataFrame by filtering df_sim_PCA
    df_sel_PCA = df_all_PCA[df_all_PCA['solution_id'].isin(df_sel_shower['solution_id'])]
    # change all df_sel_PCA 'type' to Simulation_sel
    df_sel_PCA['type'] = 'Simulation_sel'
    # reset the index
    df_sel_PCA.reset_index(drop=True, inplace=True)

    df_sel_shower_no_repetitions = df_sim_shower[df_sim_shower['solution_id'].isin(df_sel_shower['solution_id'])]
    # change all df_sel_PCA 'type' to Simulation_sel
    df_sel_shower_no_repetitions['type'] = 'Simulation_sel'
    # reset the index
    df_sel_shower_no_repetitions.reset_index(drop=True, inplace=True)

    print('\nSUCCESS: the simulated meteor have been selected')


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

        print('generating PCA space plot...')

        if len(df_sim_PCA)>10000:
            # pick randomly 10000 events
            df_sim_PCA=df_sim_PCA.sample(n=10000)

        df_sim_sel_PCA = pd.concat([df_sim_PCA,df_sel_PCA,df_obs_PCA], axis=0)

        # Select only the numeric columns for percentile calculations
        numeric_columns = df_sim_sel_PCA.select_dtypes(include=[np.number]).columns

        # Create a new column for point sizes
        df_sim_sel_PCA['point_size'] = df_sim_sel_PCA['type'].map({
            'Simulation_sel': 5,
            'Simulation': 5,
            'MetSim': 20,
            'Realization': 20,    
            'Observation': 40
        })

        # Define a custom palette
        custom_palette = {
            'Simulation': "b",
            'Simulation_sel': "darkorange",
            'MetSim': "k",
            'Realization': "mediumaquamarine",
            'Observation': "limegreen"
        }
        

        # open a new figure to plot the pairplot
        fig = plt.figure(figsize=(10, 10), dpi=300)

        # # fig = sns.pairplot(df_sim_sel_PCA, hue='type', plot_kws={'alpha': 0.6, 's': 5, 'edgecolor': 'k'},corner=True)
        # fig = sns.pairplot(df_sim_sel_PCA, hue='type',corner=True, palette='bright', diag_kind='kde', plot_kws={'s': 5, 'edgecolor': 'k'})
        # # plt.show()

        # Create the pair plot without points initially
        fig = sns.pairplot(df_sim_sel_PCA[numeric_columns.append(pd.Index(['type']))], hue='type', corner=True, palette=custom_palette, diag_kind='kde', plot_kws={'s': 5, 'edgecolor': 'k'})

        # Overlay scatter plots with custom point sizes
        for i in range(len(fig.axes)):
            for j in range(len(fig.axes)):
                if i > j:
                    # check if the variable is in the list of the numeric_columns and set the axis limit
                    if df_sim_sel_PCA.columns[j] in numeric_columns and df_sim_sel_PCA.columns[i] in numeric_columns:

                        ax = fig.axes[i, j]
                        sns.scatterplot(data=df_sim_sel_PCA, x=df_sim_sel_PCA.columns[j], y=df_sim_sel_PCA.columns[i], hue='type', size='point_size', sizes=(5, 40), ax=ax, legend=False, edgecolor='k', palette=custom_palette)

                        # ax.set_xlim(percentiles_1[df_sim_sel_PCA.columns[j]], percentiles_99[df_sim_sel_PCA.columns[j]])
                        # ax.set_ylim(percentiles_1[df_sim_sel_PCA.columns[i]], percentiles_99[df_sim_sel_PCA.columns[i]])

        # delete the last row of the plot
        # fig.axes[-1, -1].remove()
        # Hide the last row of plots
        # for ax in fig.axes[-1]:
        #     ax.remove()

        # Adjust the subplots layout parameters to give some padding
        plt.subplots_adjust(hspace=0.3, wspace=0.3)
        # plt.show()
        
        # save the figure
        fig.savefig(OUT_PUT_PATH+os.sep+file_name_obs+'PCAspace_sim_sel_real_'+str(len(variable_PCA)-2)+'var_'+str(PCA_percent)+'%_'+str(pca.n_components_)+'PC.png')
        # close the figure
        plt.close()

        print('generating result variable plot...')

        output_folder=OUT_PUT_PATH+os.sep+file_name_obs+'_var_real'
        # check if the output_folder exists
        if not os.path.isdir(output_folder):
            mkdirP(output_folder)

        # df_sim_PCA,df_sel_PCA,df_obs_PCA
        # print(df_sim_shower)
        # loop all physical variables
        physical_vars = ['mass','rho','sigma','erosion_height_start','erosion_coeff','erosion_mass_index','erosion_mass_min','erosion_mass_max']
        for var_phys in physical_vars:
            # make a subplot of the rho againist each variable_PCA as a scatter plot
            fig, axs = plt.subplots(int(np.ceil(len(variable_PCA[2:])/5)), 5, figsize=(20, 15))
            # flat it
            axs = axs.flatten()

            for i, var in enumerate(variable_PCA[2:]):
                # plot the rho againist the variable with black borders
                axs[i].scatter(df_sim_shower[var], df_sim_shower[var_phys], c='b') #, edgecolors='k', alpha=0.5

                axs[i].scatter(df_sel_shower[var], df_sel_shower[var_phys], c='orange') #, edgecolors='k', alpha=0.5
                # put a green vertical line for the df_obs_shower[var] value
                axs[i].axvline(shower_current[var].values[0], color='limegreen', linestyle='--', linewidth=5)
                # put a horizontal line for the rho of the first df_sim_shower
                axs[i].axhline(df_sim_shower[var_phys].values[0], color='k', linestyle='-', linewidth=2)
                # axs[i].set_title(var)
                # as a suptitle put the variable_PCA
                # fig.suptitle(var_phys)
                if i == 0 or i == 5 or i == 10 or i == 15 or i == 20:
                    # as a suptitle put the variable_PCA
                    axs[i].set_ylabel(var_phys)

                # x axis
                axs[i].set_xlabel(var)

                # grid
                axs[i].grid()
                # make y axis log if the variable is 'erosion_mass_min' 'erosion_mass_max'
                if var_phys == 'erosion_mass_min' or var_phys == 'erosion_mass_max':
                    axs[i].set_yscale('log')

            plt.tight_layout()
            # save the figure
            plt.savefig(output_folder+os.sep+file_name_obs+var_phys+'_vs_var_select_PCA.png')
            # close the figure
            plt.close()

        print('generating PCA position plot...')

        output_folder=OUT_PUT_PATH+os.sep+file_name_obs+'_sel_PCA'
        # check if the output_folder exists
        if not os.path.isdir(output_folder):
            mkdirP(output_folder)

        # loop all pphysical variables
        physical_vars = ['mass','rho','sigma','erosion_height_start','erosion_coeff','erosion_mass_index','erosion_mass_min','erosion_mass_max']
        for var_phys in physical_vars:

            # make a subplot of the rho againist each variable_PCA as a scatter plot
            fig, axs = plt.subplots(int(np.ceil(len(columns_PC)/5)), 5, figsize=(20, 15))

            # flatten the axs array
            axs = axs.flatten()
            for i, var in enumerate(columns_PC):
                # plot the rho againist the variable with black borders
                axs[i].scatter(df_sim_PCA[var], df_sim_shower[var_phys], c='b') #, edgecolors='k', alpha=0.5

                axs[i].scatter(df_sel_PCA[var], df_sel_shower_no_repetitions[var_phys], c='orange') #, edgecolors='k', alpha=0.5
                # put a green vertical line for the df_obs_shower[var] value
                axs[i].axvline(df_obs_PCA[var].values[0], color='limegreen', linestyle='--', linewidth=5)
                # put a horizontal line for the rho of the first df_sim_shower
                axs[i].axhline(df_sim_shower[var_phys].values[0], color='k', linestyle='-', linewidth=2)
                # axs[i].set_title(var)
                # # as a suptitle put the variable_PCA
                # fig.suptitle(var_phys)
                if i == 0 or i == 5 or i == 10 or i == 15 or i == 20:
                    # as a suptitle put the variable_PCA
                    axs[i].set_ylabel(var_phys)
                # axis x
                axs[i].set_xlabel(var)
                # grid
                axs[i].grid()
                # make y axis log if the variable is 'erosion_mass_min' 'erosion_mass_max'
                if var_phys == 'erosion_mass_min' or var_phys == 'erosion_mass_max':
                    axs[i].set_yscale('log')

            # delete the subplot that are not used
            for i in range(len(columns_PC), len(axs)):
                fig.delaxes(axs[i])

            plt.tight_layout()
            # save the figure
            plt.savefig(output_folder+os.sep+file_name_obs+var_phys+'_vs_var_select_PC_space.png')
            # close the figure
            plt.close()
        

    window_of_smothing_avg=3
    std_multip_threshold=1
    df_app=[]
    for around_meteor in df_sel_shower['solution_id_dist'].unique():
        # select the data with distance less than dist_select and check if there are more than n_select
        df_curr_sel_curv = df_sel_shower[df_sel_shower['solution_id_dist']==around_meteor]
        dist_to_cut=diff_dist_index(df_curr_sel_curv,window_of_smothing_avg,std_multip_threshold)
        # # change of curvature print
        dist_to_cut=df_curr_sel_curv.iloc[:dist_to_cut+1]

        # print(dist_to_cut)
        df_app.append(dist_to_cut)
    df_sel_dist=pd.concat(df_app)


    return df_sel_shower




def PCA_confrontPLOT(df_sim, df_obs, df_sel, output_dir, n_PC_in_PCA=10 , only_select_meteors_from='', do_not_select_meteor='', true_file='', true_path=''):
    # Set the shower name (can be multiple) e.g. 'GEM' or ['GEM','PER', 'ORI', 'ETA', 'SDA', 'CAP']
    # Shower=['GEM', 'PER', 'ORI', 'ETA', 'SDA', 'CAP']
    # Shower=['PER']#['CAP']

    # number of selected events selected
    n_select=10
    dist_select=np.array([10000000000000])
    # dist_select=np.ones(9)*10000000000000

    # weight factor for the distance
    distance_weight_fact=0

    Sim_data_distribution=True

    curvature_selection_diff=True
    window_of_smothing_avg=3
    std_multip_threshold=1

    plot_dist=True

    plot_var=True


    # FUNCTIONS ###########################################################################################




    df_obs['weight']=1/len(df_obs)
    # append the observed shower to the list
    df_obs_shower.append(df_obs)
    

    # check in the current folder there is a csv file with the name of the simulated shower
    # df_sim = pd.read_csv(input_dir+os.sep+'Simulated_'+current_shower+'.csv')
    print('simulation: '+str(len(df_sim)))
    # simulation with acc positive
    df_sim['weight']=1/len(df_sim)

    # df_PCA_columns = pd.read_csv(input_dir+os.sep+'Simulated_'+current_shower+'_select_PCA.csv')
    # # fid the numbr of columns
    # n_PC_in_PCA=str(len(df_PCA_columns.columns)-1)+'PC'
    # # print the number of selected events
    # print('The PCA space has '+str(n_PC_in_PCA))

    # append the simulated shower to the list
    df_sim_shower.append(df_sim)

    # check in the current folder there is a csv file with the name of the simulated shower
    df_sel_save = df_sel.copy()
    df_sel_save_dist = df_sel_save

    n_sample_noise=len(df_sel['solution_id_dist'].unique())

    flag_remove=False
    # check if the do_not_select_meteor any of the array value is in the solution_id of the df_sel if yes remove it
    for i in range(len(do_not_select_meteor)):
        if do_not_select_meteor[i] in df_sel['solution_id'].values:
            df_sel=df_sel[df_sel['solution_id']!=do_not_select_meteor[i]]
            df_sel_save_dist=df_sel_save[df_sel_save['solution_id']!=do_not_select_meteor[i]]
            print('removed: '+do_not_select_meteor[i])
            flag_remove=True

    if curvature_selection_diff==True:

        if Sim_data_distribution==True or Sim_data_distribution==False and n_sample_noise==1:
            # if there only_select_meteors_from is equal to any solution_id_dist
            if only_select_meteors_from in df_sel['solution_id_dist'].values:
                # select only the one with the similar name as only_select_meteors_from in solution_id_dist for df_sel
                df_sel=df_sel[df_sel['solution_id_dist']==only_select_meteors_from]
                df_sel_save=df_sel_save[df_sel_save['solution_id_dist']==only_select_meteors_from]
                df_sel_save_dist=df_sel_save_dist[df_sel_save_dist['solution_id_dist']==only_select_meteors_from]
            #     print('selected events for : '+only_select_meteors_from)
            # print(len(df_sel))
            dist_to_cut=diff_dist_index(df_sel,window_of_smothing_avg,std_multip_threshold)

            # change of curvature print  
            # print('Change of curvature at:'+str(dist_to_cut))

            # get the data from df_sel upto the dist_to_cut
            df_sel=df_sel.iloc[:dist_to_cut+1]  

        elif Sim_data_distribution==False:
            # create a for loop for each different solution_id_dist in df_sel
            df_app=[]
            for around_meteor in df_sel['solution_id_dist'].unique():
                # select the data with distance less than dist_select and check if there are more than n_select
                df_curr_sel_curv = df_sel[df_sel['solution_id_dist']==around_meteor]
                dist_to_cut=diff_dist_index(df_curr_sel_curv,window_of_smothing_avg,std_multip_threshold)
                # # change of curvature print
                # print(around_meteor)
                # print('- Curvature change in the first '+str(dist_to_cut)+' at a distance of: '+str(df_curr_sel_curv['distance_meteor'].iloc[dist_to_cut]))
                # get the data from df_sel upto the dist_to_cut
                dist_to_cut=df_curr_sel_curv.iloc[:dist_to_cut+1]

                # print(dist_to_cut)
                df_app.append(dist_to_cut)
            df_sel=pd.concat(df_app)
            # print(df_sel['solution_id_dist'])
            # print(df_sel["solution_id_dist"].unique())
            # print(df_sel_save["solution_id_dist"].unique())


    else:
        if Sim_data_distribution==True:
            n_sample_noise=1
            # if there only_select_meteors_from is equal to any solution_id_dist
            if only_select_meteors_from in df_sel['solution_id_dist'].values:
                # select only the one with the similar name as only_select_meteors_from in solution_id_dist for df_sel
                df_sel=df_sel[df_sel['solution_id_dist']==only_select_meteors_from]
                df_sel_save=df_sel_save[df_sel_save['solution_id_dist']==only_select_meteors_from]
                print('selected events for : '+only_select_meteors_from)

            if len(df_sel)>n_select:
                df_sel=df_sel.head(n_select)
        elif Sim_data_distribution==False:
            # pick the first n_select for each set of solution_id_dist selected event
            df_sel=df_sel.groupby('solution_id_dist').head(n_select)
            # print the number of selected events
            print('selected events for each case below the value: '+str(len(df_sel)))

    df_sel['weight']=1/len(df_sel)
    # df_sel['weight']=0.00000000000000000000001
    # df_sel=df_sel[df_sel['acceleration']>0]
    # df_sel=df_sel[df_sel['acceleration']<100]
    # df_sel=df_sel[df_sel['trail_len']<50]
    # append the simulated shower to the list
    df_sel_shower.append(df_sel)

    ########## txt file for the print ############################################################

    # check if a file with the name "log"+n_PC_in_PCA+"_"+str(len(df_sel))+"ev.txt" already exist
    if os.path.exists(output_dir+os.sep+"log"+n_PC_in_PCA+"_"+str(len(df_sel))+"ev.txt"):
        # remove the file
        os.remove(output_dir+os.sep+"log"+n_PC_in_PCA+"_"+str(len(df_sel))+"ev.txt")
    sys.stdout = Logger(output_dir, "log"+n_PC_in_PCA+"_"+str(len(df_sel))+"ev.txt")

    ########## txt file for the print ############################################################

    if flag_remove==True:
        print('TOT simulations : '+str(len(df_sim)))            
        print('removed: '+do_not_select_meteor[i])
    else:
        print('TOT simulations : '+str(len(df_sim)))
        print('removed nothing')
    
    for around_meteor in df_sel['solution_id_dist'].unique():
        # select the data with distance less than dist_select and check if there are more than n_select
        df_curr_sel_curv = df_sel[df_sel['solution_id_dist']==around_meteor]
        # change of curvature print
        print()
        print(around_meteor)
        print('- Curvature change in the first '+str(len(df_curr_sel_curv['distance_meteor']))+' at a distance of: '+str(df_curr_sel_curv['distance_meteor'].iloc[-1]))

    # concatenate all the simulated shower in a single dataframe
    df_sim_shower = pd.concat(df_sim_shower)

    # concatenate all the EMCCD observed showers in a single dataframe
    df_obs_shower = pd.concat(df_obs_shower)

    # concatenate all the selected simulated showers in a single dataframe
    df_sel_shower = pd.concat(df_sel_shower)

    # # correlation matrix observed and fit parameters
    # corr = df_sel_shower.drop(['weight','distance_meteor'], axis=1).corr() # need solar longitude
    # sns.heatmap(corr,
    #             xticklabels=corr.columns.values,
    #             yticklabels=corr.columns.values,
    #             cmap="coolwarm")
    # plt.title('Correlation Matrix')
    # # shift the plot to the right
    # plt.show()



    curr_sim=df_sim_shower[df_sim_shower['shower_code']=='sim_'+current_shower]
    curr_obs=df_obs_shower[df_obs_shower['shower_code']==current_shower]
    curr_obs['shower_code']=current_shower+'_obs'
    curr_sel=df_sel_shower[df_sel_shower['shower_code']==current_shower+'_sel']
    curr_sel_save=df_sel_save[df_sel_save['shower_code']==current_shower+'_sel']
    curr_sel_save_dist=df_sel_save_dist[df_sel_save_dist['shower_code']==current_shower+'_sel']

    if curvature_selection_diff==False:
        if n_sample_noise>1:

            if len(dist_select)<n_sample_noise:
                dist_select=np.ones(n_sample_noise)*10000000000000

            # Extract unique locations
            meteors_IDs = curr_sel_save["solution_id_dist"].unique()

            # # split the pd for the different solution_id_dist
            # curr_sel_split = [curr_sel[curr_sel["solution_id_dist"] == around_meteor] for around_meteor in meteors_IDs]
            sel_split_curr=[]
            # for each meteors_IDs consider the dist_select to cut the data
            for i, around_meteor in enumerate(meteors_IDs):

                curr_sel_for_meteor = curr_sel[curr_sel["solution_id_dist"] == around_meteor]

                # for each meteors_IDs consider the dist_select to cut the data
                if np.min(curr_sel_for_meteor['distance_meteor'])>dist_select[i]:
                    forprint=curr_sel_for_meteor
                    sel_split_curr.append(curr_sel_for_meteor)
                    print(around_meteor)
                    print(str(i+1)+') No selected event below the given minimum distance :'+str(len(forprint)))
                    print('SEL = MAX dist: '+str(np.round(np.max(forprint['distance_meteor']),2)) +' min dist:'+str(np.round(np.min(forprint['distance_meteor']),2)))
                else:
                    forprint=curr_sel_for_meteor[curr_sel_for_meteor['distance_meteor']<dist_select[i]]
                    # delete the data with the same around_meteor ["solution_id_dist"] that have distance_meteor bigger than dist_select[i]
                    sel_split_curr.append(curr_sel_for_meteor[curr_sel_for_meteor['distance_meteor']<dist_select[i]])
                    print(around_meteor)
                    # print the number of selected events
                    print(str(i+1)+') selected events below the value: '+str(len(forprint)))
                    print('SEL = MAX dist: '+str(np.round(np.max(forprint['distance_meteor']),2)) +' min dist:'+str(np.round(np.min(forprint['distance_meteor']),2)))
                    
            curr_sel=pd.concat(sel_split_curr)
            print('selected events below the distances give : '+str(len(curr_sel)))
        
    curr_df=pd.concat([curr_sim.drop(['rho','sigma','erosion_height_start','erosion_coeff','erosion_mass_index','erosion_mass_min','erosion_mass_max','erosion_energy_per_unit_cross_section',  'erosion_energy_per_unit_mass', 'erosion_range'], axis=1),curr_sel.drop(['distance_meteor','solution_id_dist','rho','sigma','erosion_height_start','erosion_coeff','erosion_mass_index','erosion_mass_min','erosion_mass_max','erosion_energy_per_unit_cross_section',  'erosion_energy_per_unit_mass','distance','erosion_range'], axis=1)], axis=0, ignore_index=True)
    curr_df=pd.concat([curr_df,curr_obs.drop(['distance'], axis=1)], axis=0, ignore_index=True)
    curr_df=curr_df.dropna()
    if Sim_data_distribution==True:
        if len(curr_sim)>10000:
            # pick randomly 10000 events
            curr_sim=curr_sim.sample(n=10000)
            curr_sim['weight']=1/len(curr_sim)
        curr_df_sim_sel=pd.concat([curr_sim,curr_sel.drop(['distance'], axis=1)], axis=0, ignore_index=True)
        
        curr_sel['erosion_coeff']=curr_sel['erosion_coeff']*1000000
        curr_sel['sigma']=curr_sel['sigma']*1000000
        curr_sel['erosion_energy_per_unit_cross_section']=curr_sel['erosion_energy_per_unit_cross_section']/1000000
        curr_sel['erosion_energy_per_unit_mass']=curr_sel['erosion_energy_per_unit_mass']/1000000

    elif Sim_data_distribution==False:
        curr_df_sim_sel=curr_sel
    

    if plot_var==True:
        fig, axs = plt.subplots(4, 3)
        # flatten the axs
        axs = axs.flatten()
        # fig.suptitle('Data between the 5 and 95 percentile of the variable values')
        # with color based on the shower but skip the first 2 columns (shower_code, shower_id)
        ii=0

        to_plot_unit=['init vel [km/s]','avg vel [km/s]','duration [s]','begin height [km]','peak height [km]','end height [km]','begin abs mag [-]','peak abs mag [-]','end abs mag [-]','F parameter [-]','zenith angle [deg]','deceleration [km/s^2]','trail lenght [km]','kurtosis','skew']

        to_plot=['vel_init_norot','vel_avg_norot','duration','begin_height','peak_mag_height','end_height','beg_abs_mag','peak_abs_mag','end_abs_mag','F','zenith_angle','decel_parab_t0','trail_len','kurtosis','skew']

        # deleter form curr_df the mass
        #curr_df=curr_df.drop(['mass'], axis=1)
        for ii in range(len(axs)):
            plotvar=to_plot[ii]
            # plot x within the 5 and 95 percentile of curr_df[plotvar] 
            x_plot=curr_df[plotvar]

            if plotvar in ['decel_parab_t0','decel_t0']:
                sns.histplot(curr_df, x=x_plot[x_plot>-500], weights=curr_df['weight'][x_plot>-500],hue='shower_code', ax=axs[ii], kde=True, palette='bright', bins=20)
                axs[ii].set_xticks([np.round(np.min(x_plot[x_plot>-500]),2),np.round(np.max(x_plot[x_plot>-500]),2)])
            
            else:
                # cut the x axis after the 95 percentile and 5 percentile for the one that have a shower_code Shower+'_sim'
                # x_plot= x_plot[(x_plot > np.percentile(curr_df[plotvar], 5)) & (x_plot < np.percentile(curr_df[plotvar], 95))]
                sns.histplot(curr_df, x=x_plot, weights=curr_df['weight'],hue='shower_code', ax=axs[ii], kde=True, palette='bright', bins=20)
                axs[ii].set_xticks([np.round(np.min(x_plot),2),np.round(np.max(x_plot),2)])

            # if beg_abs_mag','peak_abs_mag','end_abs_mag inver the x axis
            if plotvar in ['beg_abs_mag','peak_abs_mag','end_abs_mag']:
                axs[ii].invert_xaxis()

            # Set the x-axis formatter to ScalarFormatter
            axs[ii].xaxis.set_major_formatter(ScalarFormatter())
            axs[ii].ticklabel_format(useOffset=False, style='plain', axis='x')
            # Set the number of x-axis ticks to 3
            # axs[ii].xaxis.set_major_locator(MaxNLocator(nbins=3))

            axs[ii].set_ylabel('probability')
            axs[ii].set_xlabel(to_plot_unit[ii])
            axs[ii].get_legend().remove()
            # check if there are more than 3 ticks and if yes only use the first and the last

            # put y axis in log scale
            axs[ii].set_yscale('log')
            axs[ii].set_ylim(0.01,1)

            

                
        # more space between the subplots
        plt.tight_layout()
        # # full screen
        # figManager = plt.get_current_fig_manager()
        # figManager.window.showMaximized()

        print(output_dir+os.sep+'HistogramsVar_'+n_PC_in_PCA+'_'+str(len(curr_sel))+'ev_dist'+str(np.round(np.min(curr_sel['distance_meteor']),2))+'-'+str(np.round(np.max(curr_sel['distance_meteor']),2))+'.png')
        # save the figure
        fig.savefig(output_dir+os.sep+'Histograms'+str(len(axs))+'Var_'+n_PC_in_PCA+'_'+str(len(curr_sel))+'ev_MAXdist-'+str(np.round(np.max(curr_sel['distance_meteor']),2))+'.png', dpi=300)
        plt.close()

######### DISTANCE PLOT ##################################################
    if plot_dist==True:
        if n_sample_noise>1 and Sim_data_distribution==False:

            # Extract unique locations
            meteors_IDs = curr_sel_save["solution_id_dist"].unique()
            
            # save the distance_meteor from df_sel_save
            distance_meteor_sel_save=curr_sel_save['distance_meteor']
            # save the distance_meteor from df_sel_save
            distance_meteor_sel=curr_sel['distance_meteor']

            # Plotting
            fig, axes = plt.subplots(nrows=3, ncols=3)
            axes = axes.flatten()  # Flatten the array for easier iteration

            # use the default matpotlib default color cycle for the plots
            # print(plt.rcParams['axes.prop_cycle'].by_key()['color'])
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

            # if above meteors_IDs is bigger than 9 cut it to 9
            if len(meteors_IDs)>9:
                meteors_IDs=meteors_IDs[:9]

            for i, around_meteor in enumerate(meteors_IDs):
                    
                # Filter data for the current location
                data_for_meteor = curr_sel_save_dist[curr_sel_save_dist["solution_id_dist"] == around_meteor]
                data_for_meteor_sel = curr_sel[curr_sel["solution_id_dist"] == around_meteor]
                # use the default matpotlib default color cycle
                sns.histplot(data_for_meteor, x=data_for_meteor["distance_meteor"], kde=True, cumulative=True, bins=len(data_for_meteor["distance_meteor"]), color=colors[i], ax=axes[i])
                
                data_range = [data_for_meteor["distance_meteor"].min(), data_for_meteor["distance_meteor"].max()]

                # sns.histplot(data_for_meteor, x=data_for_meteor["distance_meteor"], kde=True, cumulative=True, bins=len(data_for_meteor["distance_meteor"]))
                # sns.kdeplot(data_for_meteor, x=data_for_meteor["distance_meteor"], cumulative=True, bw_adjust=sensib, clip=data_range, color='k',ax=axes[i])
                
                # axes[i].set_title(around_meteor[-12:-8])
                axes[i].set_xlabel('Dist. PCA space')  # Remove x label for clarity
                axes[i].set_ylabel('No.events')
                # axes[i].tick_params(labelrotation=45)  # Rotate labels for better readability
                # check if distance_meteor_sel have any value
                if len(data_for_meteor_sel)>0:
                    axes[i].axvline(x=np.max(data_for_meteor_sel["distance_meteor"]), color=colors[i], linestyle='--')
                else:
                    axes[i].axvline(x=distance_meteor_sel, color=colors[i], linestyle='--')
                # plot a dasced line with the max distance_meteor_sel
                #axes[i].axvline(x=np.max(distance_meteor_sel), color='k', linestyle='--')
                # pu a y lim .ylim(0,100) 
                axes[i].set_ylim(0,100)
                # axes[i].set_ylim(0,0.01)
                # if len(distance_meteor_sel)<1000:
                #     axes[i].set_ylim(0,100)

            # Hide unused subplots if there are any
            for ax in axes[len(meteors_IDs):]:
                ax.set_visible(False)

            plt.tight_layout()
            # plt.show()
            # save the figure maximized and with the right name
            plt.savefig(output_dir+os.sep+'DistributionDist'+n_PC_in_PCA+'_'+str(len(df_sel))+'ev_MAXdist'+str(np.round(np.max(distance_meteor_sel),2))+'.png', dpi=300)

            # close the figure
            plt.close()
        else:

            # Extract unique locations
            meteors_IDs = curr_sel_save["solution_id_dist"].unique()

            curr_sel_save_dist=curr_sel_save_dist[curr_sel_save_dist["solution_id_dist"] == meteors_IDs[0]]

            # save the distance_meteor from df_sel_save
            distance_meteor_sel_save=curr_sel_save_dist['distance_meteor']
            # save the distance_meteor from df_sel_save
            distance_meteor_sel=curr_sel['distance_meteor']
            # delete the index
            distance_meteor_sel_save=distance_meteor_sel_save.reset_index(drop=True)

            ###################################################
            # # check if distance_meteor_sel_save index is bigger than the index distance_meteor_sel+50
            # sns.histplot(distance_meteor_sel_save, kde=True, cumulative=True, bins=len(distance_meteor_sel_save)) # , stat='density' to have probability
            # # plt.ylim(0,len(distance_meteor_sel_save))
            # if len(distance_meteor_sel)<100:
            #     plt.ylim(0,100) 
            # # axis label
            # plt.xlabel('Distance in PCA space')
            # plt.ylabel('Number of events')

            # colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

            # # plot a dasced line with the max distance_meteor_sel
            # plt.axvline(x=np.max(distance_meteor_sel), color=colors[0], linestyle='--')

            # # make the y axis logarithmic
            # # plt.xscale('log')
            
            # # show
            # # plt.show()
            ####################################################

            #make subtraction of the next element and the previous element of distance_meteor_sel_save
            diff_distance_meteor = np.diff(distance_meteor_sel_save)
            # histogram plot of the difference with the count on the x axis and diff_distance_meteor on the y axis 
            indices = np.arange(len(diff_distance_meteor))
            # create the cumulative sum of the diff_distance_meteor
            cumsum_diff_distance_meteor = np.cumsum(diff_distance_meteor)
            # normalize the diff_distance_meteor xnormalized = (x - xminimum) / range of x
            diff_distance_meteor_normalized = (diff_distance_meteor - np.min(diff_distance_meteor)) / (np.max(diff_distance_meteor) - np.min(diff_distance_meteor))

            # find the index equal to 1 in diff_distance_meteor_normalized
            # index1 = np.where(diff_distance_meteor_normalized == 1)[0]
            # check when the diff_distance_meteor is two nxt to eac other are smaller than 0.1 starting from the first element
        
            # dimension of the plot 15,5
            plt.figure(figsize=(15,5))

            plt.subplot(1,2,2)
            sns.histplot(distance_meteor_sel_save, kde=True, cumulative=True, bins=len(distance_meteor_sel_save)) # , stat='density' to have probability
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
            plt.xlabel('Distance in PCA space')
            plt.ylabel('Number of events')
            plt.title('Cumulative distance in PCA space')
            plt.axvline(x=np.max(distance_meteor_sel), color=colors[0], linestyle='--')
            if len(distance_meteor_sel)<100:
                plt.ylim(0,100) 
            

            plt.subplot(1,2,1)
            # sns.histplot(diff_distance_meteor_normalized, kde=True, bins=len(distance_meteor_sel_save))
            #make the bar plot 0.5 transparency
            
            plt.bar(indices, diff_distance_meteor_normalized,color=colors[0], alpha=0.5, edgecolor='black')
            plt.xlabel('Number of events')
            plt.ylabel('Normalized difference')
            plt.title('Distance difference Normalized')
            # put a horizontal line at len(curr_sel['distance_meteor'])
            plt.axvline(x=len(distance_meteor_sel)-1, color=colors[0], linestyle='--') 
            if len(distance_meteor_sel)<100:
                plt.xlim(-1,100) 

            # find the mean of the first 100 elements of diff_distance_meteor_normalized and put a horizontal line
            # plt.axhline(y=np.std(moving_average_smoothing(diff_distance_meteor_normalized, window_size=3)), color=colors[0], linestyle='--')


            # plt.subplot(1,3,3)
            # # sns.histplot(diff_distance_meteor_normalized, kde=True, cumulative=True, bins=len(distance_meteor_sel_save))
            # plt.bar(indices, cumsum_diff_distance_meteor,color=colors[0], alpha=0.5, edgecolor='black')
            # # sns.histplot(data_for_meteor, x=data_for_meteor["distance_meteor"][:100], kde=True, cumulative=True, bins=len(data_for_meteor["distance_meteor"]))
            # plt.ylabel('Cumulative sum')
            # plt.xlabel('Number of events')
            # plt.title('Cumulative sum diff Normalized') 
            # # put a horizontal line at len(curr_sel['distance_meteor'])
            # plt.axvline(x=len(distance_meteor_sel), color=colors[0], linestyle='--')  
            # if len(distance_meteor_sel)<100:
            #     plt.xlim(-1,101) 
            # give more space
            plt.tight_layout()  
            # plt.show()

            # save the figure maximized and with the right name
            plt.savefig(output_dir+os.sep+'DistributionDist'+n_PC_in_PCA+'_'+str(len(df_sel))+'ev_MAXdist'+str(np.round(np.max(distance_meteor_sel),2))+'.png', dpi=300)

            # close the figure
            plt.close()

    ############################################################################

    # multiply the erosion coeff by 1000000 to have it in km/s
    curr_df_sim_sel['erosion_coeff']=curr_df_sim_sel['erosion_coeff']*1000000
    curr_df_sim_sel['sigma']=curr_df_sim_sel['sigma']*1000000
    df_sel_save['erosion_coeff']=df_sel_save['erosion_coeff']*1000000
    df_sel_save['sigma']=df_sel_save['sigma']*1000000
    curr_df_sim_sel['erosion_energy_per_unit_cross_section']=curr_df_sim_sel['erosion_energy_per_unit_cross_section']/1000000
    curr_df_sim_sel['erosion_energy_per_unit_mass']=curr_df_sim_sel['erosion_energy_per_unit_mass']/1000000
    df_sel_save['erosion_energy_per_unit_cross_section']=df_sel_save['erosion_energy_per_unit_cross_section']/1000000
    df_sel_save['erosion_energy_per_unit_mass']=df_sel_save['erosion_energy_per_unit_mass']/1000000
    # # pick the one with shower_code==current_shower+'_sel'
    # Acurr_df_sel=curr_df_sim_sel[curr_df_sim_sel['shower_code']==current_shower+'_sel']
    # Acurr_df_sim=curr_df_sim_sel[curr_df_sim_sel['shower_code']=='sim_'+current_shower]

    var_kde=['mass','rho','sigma','erosion_height_start','erosion_coeff','erosion_mass_index','erosion_mass_min','erosion_mass_max']

    # create the dataframe with the selected variable
    curr_sel_data = curr_sel[var_kde].values

    # check if leng of curr_sel_data is bigger than 1
    if len(curr_sel_data)>8:
        kde = gaussian_kde(dataset=curr_sel_data.T)  # Note the transpose to match the expected input shape

        # Negative of the KDE function for optimization
        def neg_density(x):
            return -kde(x)

        # Bounds for optimization within all the sim space
        # data_sim = df_sim[var_kde].values
        bounds = [(np.min(curr_sel_data[:,i]), np.max(curr_sel_data[:,i])) for i in range(curr_sel_data.shape[1])]

        # Initial guesses: curr_sel_data mean, curr_sel_data median, and KMeans centroids
        mean_guess = np.mean(curr_sel_data, axis=0)
        median_guess = np.median(curr_sel_data, axis=0)

        # KMeans centroids as additional guesses
        kmeans = KMeans(n_clusters=5, n_init='auto').fit(curr_sel_data)  # Adjust n_clusters based on your understanding of the curr_sel_data
        centroids = kmeans.cluster_centers_

        # Combine all initial guesses
        initial_guesses = [mean_guess, median_guess] + centroids.tolist()

        # Perform optimization from each initial guess
        results = [minimize(neg_density, x0, method='L-BFGS-B', bounds=bounds) for x0 in initial_guesses]

        # Filter out unsuccessful optimizations and find the best result
        successful_results = [res for res in results if res.success]
        if successful_results:
            best_result = min(successful_results, key=lambda x: x.fun)
            densest_point = best_result.x
            print("Densest point using KMeans centroid:", densest_point)
        else:
            # raise ValueError('Optimization was unsuccessful. Consider revising the strategy.')
            print('Optimization was unsuccessful. Consider revising the strategy.')
            # revise the optimization strategy
            print('Primary optimization strategies were unsuccessful. Trying fallback strategy (Grid Search).')
            # Fallback strategy: Grid Search
            grid_size = 5  # Define the grid size for the search
            grid_points = [np.linspace(bound[0], bound[1], grid_size) for bound in bounds]
            grid_combinations = list(itertools.product(*grid_points))

            best_grid_point = None
            best_grid_density = -np.inf

            for point in grid_combinations:
                density = kde(point)
                if density > best_grid_density:
                    best_grid_density = density
                    best_grid_point = point

            if best_grid_point is not None:
                print("Densest point found using Grid Search:", best_grid_point)
                return best_grid_point
            else:
                raise ValueError('Grid Search was unsuccessful. None of the strategy worked.')

    else:
        print('Not enough data to perform the KDE need more than 8 meteors')
        # raise ValueError('The data is ill-conditioned. Consider a bigger number of elements.')




# if pickle change the extension and the code ##################################################################################################
    
    # check if the file is a pickle
    if true_file.endswith('.pickle'):
        # Load the trajectory file
        # traj = loadPickle(true_path, true_file)
        sim_fit_json_nominal = os.path.join(true_path, true_file)

        with open(sim_fit_json_nominal, 'rb') as f:
            traj = pickle.load(f, encoding='latin1')

        # look for the 20230811_082648_sim_fit.json in the same folder
        sim_fit_json_nominal = sim_fit_json_nominal.replace('.pickle', '_sim_fit.json')
        # check if the file exist
        if not os.path.exists(sim_fit_json_nominal):
            # open any file with the json extension
            sim_fit_json_nominal = os.path.join(true_path, [f for f in os.listdir(true_path) if f.endswith('.json')][0])
            # print the file name
            print(sim_fit_json_nominal)

    else:
        # Load the nominal simulation
        sim_fit_json_nominal = os.path.join(true_path, true_file)
        traj = None

    # Load the nominal simulation parameters
    const_nominal, _ = loadConstants(sim_fit_json_nominal)
    const_nominal.dens_co = np.array(const_nominal.dens_co)

    dens_co=np.array(const_nominal.dens_co)

    # print(const_nominal.__dict__)

    ### Calculate atmosphere density coeffs (down to the bottom observed height, limit to 15 km) ###

    # Determine the height range for fitting the density
    dens_fit_ht_beg = const_nominal.h_init
    # dens_fit_ht_end = const_nominal.h_final

    # Assign the density coefficients
    const_nominal.dens_co = dens_co

    # Turn on plotting of LCs of individual fragments 
    const_nominal.fragmentation_show_individual_lcs = True

    # # change the sigma of the fragmentation
    # const_nominal.sigma = 1.0

    # 'rho': 209.27575861617834, 'm_init': 1.3339843905562902e-05, 'v_init': 59836.848805126894, 'shape_factor': 1.21, 'sigma': 1.387556841276162e-08, 'zenith_angle': 0.6944268835985749, 'gamma': 1.0, 'rho_grain': 3000, 'lum_eff_type': 5, 'lum_eff': 0.7, 'mu': 3.8180000000000003e-26, 'erosion_on': True, 'erosion_bins_per_10mass': 10, 'erosion_height_start': 117311.48011974395, 'erosion_coeff': 6.356639734390828e-07, 'erosion_height_change': 0, 'erosion_coeff_change': 3.3e-07, 'erosion_rho_change': 3700, 'erosion_sigma_change': 2.3e-08, 'erosion_mass_index': 1.614450928834309, 'erosion_mass_min': 4.773894502090459e-11, 'erosion_mass_max': 7.485333377052805e-10, 'disruption_on': False, 'compressive_strength': 2000, 

# create a copy of the const_nominal
    const_nominal_1D_KDE = copy.deepcopy(const_nominal)
    const_nominal_allD_KDE = copy.deepcopy(const_nominal)

    var_cost=['m_init','rho','sigma','erosion_height_start','erosion_coeff','erosion_mass_index','erosion_mass_min','erosion_mass_max']
    # print for each variable the kde
    percent_diff_1D=[]
    percent_diff_allD=[]
    for i in range(len(var_kde)):

        x=curr_sel[var_kde[i]]

        # Compute KDE
        kde = gaussian_kde(x)
        
        # Define the range for which you want to compute KDE values, with more points for higher accuracy
        kde_x = np.linspace(x.min(), x.max(), 1000)
        kde_values = kde(kde_x)
        
        # Find the mode (x-value where the KDE curve is at its maximum)
        mode_index = np.argmax(kde_values)
        mode = kde_x[mode_index]
        
        real_val=df_sel_save[df_sel_save['solution_id']==only_select_meteors_from][var_kde[i]]
        print(real_val)
        # put it from Series.__format__ to double format
        real_val=real_val.values[0]

        print()
        #     var_kde=['mass','rho','sigma','erosion_height_start','erosion_coeff','erosion_mass_index','erosion_mass_min','erosion_mass_max']
        # Print the mode
        print(f"Real value {var_kde[i]}: {'{:.4g}'.format(real_val)}")
        print(f"1D Mode of KDE for {var_kde[i]}: {'{:.4g}'.format(mode)} percent diff: {'{:.4g}'.format(abs((real_val-mode)/(real_val+mode))/2*100)}%")
        percent_diff_1D.append(abs((real_val-mode)/(real_val+mode))/2*100)
        if len(curr_sel_data)>8:
            print(f"Mult.dim. KDE densest {var_kde[i]}:  {'{:.4g}'.format(densest_point[i])} percent diff: {'{:.4g}'.format(abs((real_val-densest_point[i])/(real_val+densest_point[i]))/2*100)}%")
            percent_diff_allD.append(abs((real_val-densest_point[i])/(real_val+densest_point[i]))/2*100)
        # print the value of const_nominal
        # print(f"const_nominal {var_cost[i]}:  {'{:.4g}'.format(const_nominal.__dict__[var_cost[i]])}")

        if var_cost[i] == 'sigma' or var_cost[i] == 'erosion_coeff':
            # put it back as it was
            const_nominal_1D_KDE.__dict__[var_cost[i]]=mode/1000000
            if len(curr_sel_data)>8:
                const_nominal_allD_KDE.__dict__[var_cost[i]]=densest_point[i]/1000000
        elif var_cost[i] == 'erosion_height_start':
            # put it back as it was
            const_nominal_1D_KDE.__dict__[var_cost[i]]=mode*1000
            if len(curr_sel_data)>8:
                const_nominal_allD_KDE.__dict__[var_cost[i]]=densest_point[i]*1000
        else:
            # add each to const_nominal_1D_KDE and const_nominal_allD_KDE
            const_nominal_1D_KDE.__dict__[var_cost[i]]=mode
            if len(curr_sel_data)>8:
                const_nominal_allD_KDE.__dict__[var_cost[i]]=densest_point[i]

    # # Close the Logger to ensure everything is written to the file STOP COPY in TXT file
    # sys.stdout.close()

    # # Reset sys.stdout to its original value if needed
    # sys.stdout = sys.__stdout__

    # Run the simulation
    frag_main, results_list, wake_results = runSimulation(const_nominal, \
        compute_wake=False)

    sr_nominal = SimulationResults(const_nominal, frag_main, results_list, wake_results)

    # Run the simulation
    frag_main, results_list, wake_results = runSimulation(const_nominal_1D_KDE, \
        compute_wake=False)

    sr_nominal_1D_KDE = SimulationResults(const_nominal_1D_KDE, frag_main, results_list, wake_results)

    if len(curr_sel_data)>8:
        # Run the simulation
        frag_main, results_list, wake_results = runSimulation(const_nominal_allD_KDE, \
            compute_wake=False)

        sr_nominal_allD_KDE = SimulationResults(const_nominal_allD_KDE, frag_main, results_list, wake_results)

    # const_nominal = sr_nominal.const

    # open the json file with the name namefile_sel
    f = open(sim_fit_json_nominal,"r")
    data = json.loads(f.read())

    if traj == None:
        zenith_angle= data['params']['zenith_angle']['val']*180/np.pi

        vel_sim_brigh=data['simulation_results']['brightest_vel_arr']#['brightest_vel_arr']#['leading_frag_vel_arr']#['main_vel_arr']
        vel_sim=data['simulation_results']['leading_frag_vel_arr']#['brightest_vel_arr']#['leading_frag_vel_arr']#['main_vel_arr']
        ht_sim=data['simulation_results']['leading_frag_height_arr']#['brightest_height_arr']['leading_frag_height_arr']['main_height_arr']
        time_sim=data['simulation_results']['time_arr']#['main_time_arr']
        abs_mag_sim=data['simulation_results']['abs_magnitude']
        len_sim=data['simulation_results']['brightest_length_arr']#['brightest_length_arr']

        erosion_height_start = data['params']['erosion_height_start']['val']/1000

        ht_obs=data['ht_sampled']

        v0 = vel_sim[0]/1000

        # find the index of the first element of the simulation that is equal to the first element of the observation
        index_ht_sim=next(x for x, val in enumerate(ht_sim) if val <= ht_obs[0])
        # find the index of the last element of the simulation that is equal to the last element of the observation
        index_ht_sim_end=next(x for x, val in enumerate(ht_sim) if val <= ht_obs[-1])

        abs_mag_sim=abs_mag_sim[index_ht_sim:index_ht_sim_end]
        vel_sim=vel_sim[index_ht_sim:index_ht_sim_end]
        time_sim=time_sim[index_ht_sim:index_ht_sim_end]
        ht_sim=ht_sim[index_ht_sim:index_ht_sim_end]
        len_sim=len_sim[index_ht_sim:index_ht_sim_end]

        # divide the vel_sim by 1000 considering is a list
        time_sim = [i-time_sim[0] for i in time_sim]
        vel_sim = [i/1000 for i in vel_sim]
        len_sim = [(i-len_sim[0])/1000 for i in len_sim]
        ht_sim = [i/1000 for i in ht_sim]

        ht_obs=[x/1000 for x in ht_obs]

        closest_indices = find_closest_index(ht_sim, ht_obs)

        abs_mag_sim=[abs_mag_sim[jj_index_cut] for jj_index_cut in closest_indices]
        vel_sim=[vel_sim[jj_index_cut] for jj_index_cut in closest_indices]
        time_sim=[time_sim[jj_index_cut] for jj_index_cut in closest_indices]
        ht_sim=[ht_sim[jj_index_cut] for jj_index_cut in closest_indices]
        len_sim=[len_sim[jj_index_cut] for jj_index_cut in closest_indices]

    ############ add noise to the simulation

        obs_time=data['time_sampled']
        obs_length=data['len_sampled']
        abs_mag_sim=data['mag_sampled']
        obs_length=[x/1000 for x in obs_length]

        # vel_sim=[v0]
        # # append from vel_sampled the rest by the difference of the first element of obs_length divided by the first element of obs_time
        # rest_vel_sampled=[(obs_length[vel_ii]-obs_length[vel_ii-1])/(obs_time[vel_ii]-obs_time[vel_ii-1]) for vel_ii in range(1,len(obs_length))]
        # # append the rest_vel_sampled to vel_sampled
        # vel_sim.extend(rest_vel_sampled)


        # create a list of the same length of obs_time with the value of the first element of vel_sim
        obs_vel=vel_sim

        for vel_ii in range(1,len(obs_time)):
            if obs_time[vel_ii]-obs_time[vel_ii-1]<0.03125:
            # if obs_time[vel_ii] % 0.03125 < 0.000000001:
                if vel_ii+1<len(obs_length):
                    obs_vel[vel_ii+1]=(obs_length[vel_ii+1]-obs_length[vel_ii-1])/(obs_time[vel_ii+1]-obs_time[vel_ii-1])
            else:
                obs_vel[vel_ii]=(obs_length[vel_ii]-obs_length[vel_ii-1])/(obs_time[vel_ii]-obs_time[vel_ii-1])

        vel_sim=obs_vel
        
        data_index_2cam = pd.DataFrame(list(zip(obs_time, ht_obs, obs_vel, abs_mag_sim)), columns =['time_sampled', 'ht_sampled', 'vel_sampled', 'mag_sampled'])

        # find in the index of camera 1 and camera 2 base if time_sampled % 0.03125 < 0.000000001 ==cam1 and the rest cam2
        time_cam1= [i for i in obs_time if i % 0.03125 < 0.000000001]
        time_cam2= [i for i in obs_time if i % 0.03125 > 0.000000001]


    else:

        obs_vel=[]
        obs_time=[]
        abs_mag_sim=[]
        ht_obs=[]
        lag_total=[]
        elg_pickl=[]
        tav_pickl=[]


        jj=0
        for obs in traj.observations:
            # find all the differrnt names of the variables in the pickle files
            # print(obs.__dict__.keys())
            jj+=1
            if jj==1:
                tav_pickl=obs.velocities[1:int(len(obs.velocities)/4)]
                # if tav_pickl is empty append the first value of obs.velocities
                if len(tav_pickl)==0:
                    tav_pickl=obs.velocities[1:2]
                
                vel_01=obs.velocities
                time_01=obs.time_data
                abs_mag_01=obs.absolute_magnitudes
                height_01=obs.model_ht

            elif jj==2:
                elg_pickl=obs.velocities[1:int(len(obs.velocities)/4)]
                if len(elg_pickl)==0:
                    elg_pickl=obs.velocities[1:2]
                
                vel_02=obs.velocities
                time_02=obs.time_data
                abs_mag_02=obs.absolute_magnitudes
                height_02=obs.model_ht

            # put it at the end obs.velocities[1:] at the end of vel_pickl list
            obs_vel.extend(obs.velocities)
            obs_time.extend(obs.time_data)
            abs_mag_sim.extend(obs.absolute_magnitudes)
            ht_obs.extend(obs.model_ht)
            lag_total.extend(obs.lag)

        # compute the linear regression
        obs_vel = [i/1000 for i in obs_vel] # convert m/s to km/s
        obs_time = [i for i in obs_time]
        abs_mag_sim = [i for i in abs_mag_sim]
        ht_obs = [i/1000 for i in ht_obs]
        lag_total = [i/1000 for i in lag_total]

        time_cam1 = [i for i in time_01]

        time_cam2 = [i for i in time_02]


        # find the height when the velocity start dropping from the initial value 
        v0 = (np.mean(elg_pickl)+np.mean(tav_pickl))/2/1000

        # find all the values of the velocity that are equal to 0 and put them to v0
        obs_vel = [v0 if x==0 else x for x in obs_vel]
        vel_01[0]=v0
        vel_02[0]=v0

        #####order the list by time
        obs_vel = [x for _,x in sorted(zip(obs_time,obs_vel))]
        abs_mag_sim = [x for _,x in sorted(zip(obs_time,abs_mag_sim))]
        ht_obs = [x for _,x in sorted(zip(obs_time,ht_obs))]
        lag_total = [x for _,x in sorted(zip(obs_time,lag_total))]
        # length_pickl = [x for _,x in sorted(zip(time_pickl,length_pickl))]
        obs_time = sorted(obs_time)

        vel_sim=obs_vel
        ht_sim=ht_obs

        erosion_height_start = data['erosion_height_start']/1000

        data_index_2cam = pd.DataFrame(list(zip(obs_time, ht_obs, obs_vel, abs_mag_sim)), columns =['time_sampled', 'ht_sampled', 'vel_sampled', 'mag_sampled'])



    # find the index of the camera 1 and camera 2 in the dataframe
    index_cam1_df= data_index_2cam[data_index_2cam['time_sampled'].isin(time_cam1)].index
    index_cam2_df= data_index_2cam[data_index_2cam['time_sampled'].isin(time_cam2)].index



    ############ plot the simulation
    # multiply ht_sim by 1000 to have it in m
    ht_sim_meters=[x*1000 for x in ht_sim]

    # find for the index of sr_nominal.leading_frag_height_arr with the same values as sr_nominal_1D_KDE.leading_frag_height_arr
    closest_indices_1D = find_closest_index(sr_nominal_1D_KDE.leading_frag_height_arr, ht_sim_meters )
    # make the subtraction of the closest_indices between sr_nominal.abs_magnitude and sr_nominal_1D_KDE.abs_magnitude
    diff_mag_1D=[(sr_nominal.abs_magnitude[jj_index_cut]-sr_nominal_1D_KDE.abs_magnitude[jj_index_cut]) for jj_index_cut in closest_indices_1D]
    diff_vel_1D=[(sr_nominal.leading_frag_vel_arr[jj_index_cut]-sr_nominal_1D_KDE.leading_frag_vel_arr[jj_index_cut])/1000 for jj_index_cut in closest_indices_1D]
    # do the same for the sr_nominal_allD_KDE
    if len(curr_sel_data)>8:
        closest_indices_allD = find_closest_index(sr_nominal_allD_KDE.leading_frag_height_arr, ht_sim_meters )
        diff_mag_allD=[(sr_nominal.abs_magnitude[jj_index_cut]-sr_nominal_allD_KDE.abs_magnitude[jj_index_cut]) for jj_index_cut in closest_indices_allD]
        diff_vel_allD=[(sr_nominal.leading_frag_vel_arr[jj_index_cut]-sr_nominal_allD_KDE.leading_frag_vel_arr[jj_index_cut])/1000 for jj_index_cut in closest_indices_allD]

    # Plot the simulation results
    fig, ax = plt.subplots(2, 2, figsize=(8, 10),gridspec_kw={'width_ratios': [ 3, 1]}, dpi=300) #  figsize=(10, 5), dpi=300 0.5, 3, 3, 0.5

    # flat the ax
    ax = ax.flatten()

    # plot a line plot in the first subplot the magnitude vs height dashed with x markers
    ax[0].plot(data_index_2cam['mag_sampled'][index_cam1_df], data_index_2cam['ht_sampled'][index_cam1_df], linestyle='dashed', marker='x', label='1')
    ax[0].plot(data_index_2cam['mag_sampled'][index_cam2_df], data_index_2cam['ht_sampled'][index_cam2_df], linestyle='dashed', marker='x', label='2')
    

    # add the erosion_height_start as a horizontal line in the first subplot grey dashed
    ax[0].axhline(y=erosion_height_start, color='grey', linestyle='dashed')
    # add the name on the orizontal height line
    ax[0].text(max(abs_mag_sim)+1, erosion_height_start, 'Erosion height', color='grey')

    # plot a scatter plot in the second subplot the velocity vs height
    # ax[2].scatter(vel_sim, ht_sim, marker='.', label='1')
    # use the . maker and none linestyle
    ax[2].plot(data_index_2cam['vel_sampled'][index_cam1_df], data_index_2cam['ht_sampled'][index_cam1_df], marker='.', linestyle='none', label='1')
    ax[2].plot(data_index_2cam['vel_sampled'][index_cam2_df], data_index_2cam['ht_sampled'][index_cam2_df], marker='.', linestyle='none', label='2')

    # set the xlim and ylim of the first subplot
    ax[0].set_xlim([min(abs_mag_sim)-1, max(abs_mag_sim)+1])
    # check if the max(ht_sim) is greater than the erosion_height_start and set the ylim of the first subplot

    # set the xlim and ylim of the second subplot
    ax[2].set_xlim([min(vel_sim)-1, max(vel_sim)+1])

    # Plot the height vs magnitude
    ax[0].plot(sr_nominal.abs_magnitude, sr_nominal.leading_frag_height_arr/1000, label="Simulated", \
        color='k')
    
    ax[0].plot(sr_nominal_1D_KDE.abs_magnitude, sr_nominal_1D_KDE.leading_frag_height_arr/1000, label="Mode", color='r')
    
    if len(curr_sel_data)>8:
        ax[0].plot(sr_nominal_allD_KDE.abs_magnitude, sr_nominal_allD_KDE.leading_frag_height_arr/1000, label="Min KDE", color='b')

    # velocity vs height

    # # height vs velocity
    # ax[2].plot(sr_nominal.brightest_vel_arr/1000, sr_nominal.brightest_height_arr/1000, label="Simulated - brightest", \
    #     color='k', alpha=0.75)  
    
    # # Plot the velocity of the main mass
    # ax[2].plot(sr_nominal.leading_frag_vel_arr/1000, sr_nominal.leading_frag_height_arr/1000, color='k', \
    #     linestyle='dashed', label="Simulated - leading")        
    ax[2].plot(sr_nominal.leading_frag_vel_arr/1000, sr_nominal.leading_frag_height_arr/1000, color='k', \
        label="Simulated")
    
    # ax[2].plot(sr_nominal_1D_KDE.brightest_vel_arr/1000, sr_nominal_1D_KDE.brightest_height_arr/1000, \
    #             label="Mode - brightest", alpha=0.75)

    # # keep the same color and use a dashed line
    # ax[2].plot(sr_nominal_1D_KDE.leading_frag_vel_arr/1000, sr_nominal_1D_KDE.leading_frag_height_arr/1000, \
    #     linestyle='dashed', label="Mode - leading", color=ax[2].lines[-1].get_color())
    ax[2].plot(sr_nominal_1D_KDE.leading_frag_vel_arr/1000, sr_nominal_1D_KDE.leading_frag_height_arr/1000, \
        label="Mode", color='r')
    

    ax[1].scatter(diff_mag_1D,sr_nominal.leading_frag_height_arr[closest_indices_1D]/1000, color=ax[2].lines[-1].get_color(), marker='.')
    ax[3].scatter(diff_vel_1D,sr_nominal.leading_frag_height_arr[closest_indices_1D]/1000, color=ax[2].lines[-1].get_color(), marker='.')

    # ax[2].plot(sr_nominal_allD_KDE.brightest_vel_arr/1000, sr_nominal_allD_KDE.brightest_height_arr/1000, \
    #             label="Min KDE - brightest")

    # # keep the same color and use a dashed line
    # ax[2].plot(sr_nominal_allD_KDE.leading_frag_vel_arr/1000, sr_nominal_allD_KDE.leading_frag_height_arr/1000, \
    #     linestyle='dashed', label="Min KDE - leading", color=ax[2].lines[-1].get_color())
    if len(curr_sel_data)>8:
        ax[2].plot(sr_nominal_allD_KDE.leading_frag_vel_arr/1000, sr_nominal_allD_KDE.leading_frag_height_arr/1000, \
            label="Min KDE", color='b')
    
        ax[1].scatter(diff_mag_allD,sr_nominal.leading_frag_height_arr[closest_indices_allD]/1000, color=ax[2].lines[-1].get_color(), marker='.')
        ax[3].scatter(diff_vel_allD,sr_nominal.leading_frag_height_arr[closest_indices_allD]/1000, color=ax[2].lines[-1].get_color(), marker='.')

    if max(ht_sim)>erosion_height_start:
        ax[1].set_ylim([min(ht_sim)-1, max(ht_sim)+1])
        ax[0].set_ylim([min(ht_sim)-1, max(ht_sim)+1])
        ax[2].set_ylim([min(ht_sim)-1, max(ht_sim)+1])
        ax[3].set_ylim([min(ht_sim)-1, max(ht_sim)+1])
    else:
        ax[1].set_ylim([min(ht_sim)-1, erosion_height_start+2])
        ax[0].set_ylim([min(ht_sim)-1, erosion_height_start+2])
        ax[2].set_ylim([min(ht_sim)-1, erosion_height_start+2])
        ax[3].set_ylim([min(ht_sim)-1, erosion_height_start+2])
    
    # set the xlabel and ylabel of the subplots

    # on ax[1] the sides of the plot put the error in the magnitude as a value with one axis
    ax[1].set_xlabel('abs.mag.err')
    # set the same y axis as the plot above
    # ax[1].set_ylim(ax[0].get_ylim())
    # place the y axis along the zero
    ax[1].spines['left'].set_position(('data', 0))
    # place the ticks along the zero
    ax[1].yaxis.set_ticks_position('left')
    # delete the numbers from the y axis
    ax[1].yaxis.set_tick_params(labelleft=False)
    # invert the y axis
    ax[1].invert_xaxis()
    # delte the border of the plot
    ax[1].spines['right'].set_color('none')
    ax[1].spines['top'].set_color('none')
    
    
    if len(curr_sel_data)>8:
        # append diff_vel_allD to diff_vel_1D
        diff_mag_1D.extend(diff_mag_allD)
    
    # delete any nan or inf from the list
    diff_mag_1D = [x for x in diff_mag_1D if str(x) != 'nan' and str(x) != 'inf']
    # put the ticks in the x axis to -1*max(abs(np.array(diff_mag_allD))), max(abs(np.array(diff_mag_allD)) with only 2 significant digits
    # ax[1].set_xticks([-1*max(abs(np.array(diff_mag_1D))), max(abs(np.array(diff_mag_1D)))])

    ax[1].axvspan(-mag_noise, mag_noise, color='lightgray', alpha=0.5)
    # Rotate tick labels
    ax[1].tick_params(axis='x', rotation=45)
    # rotate that by 45 degrees
    ax[1].set_xlim([-1*max(abs(np.array(diff_mag_1D)))-max(abs(np.array(diff_mag_1D)))/4, max(abs(np.array(diff_mag_1D)))+max(abs(np.array(diff_mag_1D)))/4])
    # add more ticks
    ax[1].xaxis.set_major_locator(plt.MaxNLocator(5))
    # add grid to the plot
    ax[1].grid(linestyle='dashed')
    
    # on ax[3] the sides of the plot put the error in the velocity as a value with one axis
    # ax[3].set_xlabel('vel.lead.err [km/s]')
    ax[3].set_xlabel('vel.err [km/s]')
    # set the same y axis as the plot above
    # ax[3].set_ylim(ax[2].get_ylim())
    # place the y axis along the zero
    ax[3].spines['right'].set_position(('data', 0))
    # place the ticks along the zero
    ax[3].yaxis.set_ticks_position('right')
    # delete the numbers from the y axis
    ax[3].yaxis.set_tick_params(labelright=False)
    # delte the border of the plot
    ax[3].spines['left'].set_color('none')
    ax[3].spines['top'].set_color('none')
    if len(curr_sel_data)>8:
        # append diff_vel_allD to diff_vel_1D
        diff_vel_1D.extend(diff_vel_allD)

    # delete any nan or inf from the list
    diff_vel_1D = [x for x in diff_vel_1D if str(x) != 'nan' and str(x) != 'inf']
    # x limit of the plot equal to max of the absolute magnitude
    # ax[3].set_xticks([-1*max(abs(np.array(diff_vel_1D))), max(abs(np.array(diff_vel_1D)))])

    ax[3].axvspan(-vel_noise, vel_noise, color='lightgray', alpha=0.5)
    # Rotate tick labels
    ax[3].tick_params(axis='x', rotation=45)
    ax[3].set_xlim([-1*max(abs(np.array(diff_vel_1D)))-max(abs(np.array(diff_vel_1D)))/4, max(abs(np.array(diff_vel_1D)))+max(abs(np.array(diff_vel_1D)))/4])
    # add more ticks
    ax[3].xaxis.set_major_locator(plt.MaxNLocator(5))
    # add grid to the plot
    ax[3].grid(linestyle='dashed')
    
    # put the grid in the subplots and make it dashed
    ax[0].grid(linestyle='dashed')
    ax[2].grid(linestyle='dashed')
    # add the legend
    ax[0].legend()
    ax[2].legend()

    # add the labels
    ax[0].set_ylabel('Height [km]')
    ax[0].set_xlabel('Absolute Magnitude')
    # invert the x axis
    ax[0].invert_xaxis()

    # put the ticks on the right
    # ax[2].yaxis.tick_right()
    ax[2].set_ylabel('Height [km]')
    ax[2].set_xlabel('Velocity [km/s]')
    # put the labels on the right
    # ax[2].yaxis.set_label_position("right")


    # make more space between the subplots
    plt.tight_layout()

    # make the plot visible
    # plt.show()
    print(output_dir+os.sep+'BestFitKDE'+n_PC_in_PCA+'_'+str(len(curr_sel))+'ev_dist'+str(np.round(np.min(curr_sel['distance_meteor']),2))+'-'+str(np.round(np.max(curr_sel['distance_meteor']),2))+'.png')
    # save the figure maximized and with the right name
    fig.savefig(output_dir+os.sep+'BestFitKDE'+n_PC_in_PCA+'_'+str(len(curr_sel))+'ev_dist'+str(np.round(np.min(curr_sel['distance_meteor']),2))+'-'+str(np.round(np.max(curr_sel['distance_meteor']),2))+'.png', dpi=300)
    
    # close the figure
    plt.close()
    
##########################################################################

    # open a new figure to plot the pairplot
    fig = plt.figure(figsize=(10, 10), dpi=300)

    # Define your label mappings
    label_mappings = {
        'mass': 'mass [kg]',
        'rho': 'rho [kg/m^3]',
        'sigma': 'sigma [s^2/km^2]',
        'erosion_height_start': 'erosion height start [km]',
        'erosion_coeff': 'erosion coeff [s^2/km^2]',
        'erosion_mass_index': 'erosion mass index [-]',
        'erosion_mass_min': 'log eros. mass min [kg]',
        'erosion_mass_max': 'log eros. mass max [kg]'
    }

    if Sim_data_distribution==True:
        to_plot8=['shower_code','mass','rho','sigma','erosion_height_start','erosion_coeff','erosion_mass_index','erosion_mass_min','erosion_mass_max']
        # crete the triangular plot to of the selected variables with seaborn with the simulated and the selected events and plot it in the figure
        fig = sns.pairplot(curr_df_sim_sel[to_plot8], hue='shower_code', palette='bright', diag_kind='kde', corner=True, plot_kws={'edgecolor': 'k'}) 
        fig._legend.remove()
    elif Sim_data_distribution==False:
        to_plot8=['solution_id_dist','mass','rho','sigma','erosion_height_start','erosion_coeff','erosion_mass_index','erosion_mass_min','erosion_mass_max']
        # crete the triangular plot to of the selected variables with seaborn with the simulated and the selected events
        fig = sns.pairplot(curr_df_sim_sel[to_plot8], hue='solution_id_dist', palette='bright', diag_kind='kde', corner=True, plot_kws={'edgecolor': 'k'})
        fig._legend.remove()
    
    label_plot=['mass [kg]','rho [kg/m^3]','sigma [s^2/km^2]','erosion height start [km]','erosion coeff [s^2/km^2]','erosion mass index [-]','log eros. mass min [kg]','log eros. mass max [kg]']
    # change the x and y labels of the plot
    # Update the labels
    for ax in fig.axes.flatten():
        if ax is not None:  # Check if the axis exists
            xlabel = ax.get_xlabel()
            ylabel = ax.get_ylabel()
            if ylabel=='erosion_mass_min' or ylabel=='erosion_mass_max': # ylabel == 'mass' or xlabel == 'sigma'
                # set it to log scale
                ax.set_yscale('log')
            if xlabel=='erosion_mass_min' or xlabel=='erosion_mass_max': # xlabel=='erosion_coeff'
                # set it to log sca
                ax.set_xscale('log')
            if xlabel in label_mappings:
                ax.set_xlabel(label_mappings[xlabel])
            if ylabel in label_mappings:
                ax.set_ylabel(label_mappings[ylabel])
    # plt.tight_layout()
    # plt.show()

    print(output_dir+os.sep+'PhysicPropPairPlot'+n_PC_in_PCA+'_'+str(len(curr_sel))+'ev_dist'+str(np.round(np.min(curr_sel['distance_meteor']),2))+'-'+str(np.round(np.max(curr_sel['distance_meteor']),2))+'.png')
    # save the figure maximized and with the right name
    fig.savefig(output_dir+os.sep+'PhysicPropPairPlot'+n_PC_in_PCA+'_'+str(len(curr_sel))+'ev_dist'+str(np.round(np.min(curr_sel['distance_meteor']),2))+'-'+str(np.round(np.max(curr_sel['distance_meteor']),2))+'.png', dpi=300)
    # close the figure
    plt.close()

##########################################################################

    # Define your label mappings
    label_mappings = {
        'mass': 'mass [kg]',
        'rho': 'rho [kg/m^3]',
        'sigma': 'sigma [s^2/km^2]',
        'erosion_height_start': 'erosion height start [km]',
        'erosion_coeff': 'erosion coeff [s^2/km^2]',
        'erosion_mass_index': 'erosion mass index [-]',
        'erosion_mass_min': 'log eros. mass min [kg]',
        'erosion_mass_max': 'log eros. mass max [kg]'
    }

    # Choose which columns to plot based on condition
    if Sim_data_distribution:
        to_plot8 = ['shower_code', 'mass', 'rho', 'sigma', 'erosion_height_start', 'erosion_coeff', 'erosion_mass_index', 'erosion_mass_min', 'erosion_mass_max']
        hue_column = 'shower_code'
    else:
        to_plot8 = ['solution_id_dist', 'mass', 'rho', 'sigma', 'erosion_height_start', 'erosion_coeff', 'erosion_mass_index', 'erosion_mass_min', 'erosion_mass_max']
        hue_column = 'solution_id_dist'

    # Create a PairGrid
    pairgrid = sns.PairGrid(curr_df_sim_sel[to_plot8], hue=hue_column, palette='bright')

    # Map the plots
    pairgrid.map_lower(sns.scatterplot, edgecolor='k', palette='bright')
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

    if Sim_data_distribution==True:
        corr = curr_sel[to_plot8[1:]].corr()
    if Sim_data_distribution==False:
        corr = curr_df_sim_sel[to_plot8[1:]].corr()

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

    fig_name = (output_dir+os.sep+'MixPhysicPropPairPlot'+n_PC_in_PCA+'_'+str(len(curr_sel))+'ev_dist'+str(np.round(np.min(curr_sel['distance_meteor']),2))+'-'+str(np.round(np.max(curr_sel['distance_meteor']),2))+'.png')
    plt.savefig(fig_name, dpi=300)

    # Close the figure
    plt.close()

##########################################################################

    to_plot8=['mass','rho','sigma','erosion_height_start','erosion_coeff','erosion_mass_index','erosion_mass_min','erosion_mass_max']
    # labe for the variables
    label_plot=['mass','rho','sigma','er.height','er.coeff','er.mass index','er.mass min','er.mass max']

    # # create a covarariance matrix plot of the selected variables
    # scaler = StandardScaler()
    # scaled_data = scaler.fit_transform(curr_df_sim_sel[to_plot8])  # Assuming 'df' and 'selected_columns' from your context
    # scaled_df = pd.DataFrame(scaled_data, columns=to_plot8)

    # # Now compute covariance on scaled data
    # cov_matrix = scaled_df.cov()

    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
    # Compute the correlation coefficients
    # corr_matrix = curr_df_sim_sel[to_plot8].corr()

    # create covariance matrix plot of the selected variables
    # sns.heatmap(curr_df_sim_sel[to_plot8].cov(), annot=True, cmap='coolwarm', ax=ax)
    # create a heatmap of the selected variables base on the covariance matrix corr()
    if Sim_data_distribution==True:
        sns.heatmap(curr_sel[to_plot8].corr(), annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
    if Sim_data_distribution==False:
        sns.heatmap(curr_df_sim_sel[to_plot8].corr(), annot=True, fmt=".2f", cmap='coolwarm', ax=ax)

    # sns.heatmap(curr_df_sim_sel[to_plot8].cov(), annot=True, fmt=".2f", cmap='coolwarm', ax=ax, vmin=-1, vmax=1)
    # sns.heatmap(cov_matrix, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
    # and put the values of the correlation in the heatmap rounded to 2 decimals in the center of the square
    for t in ax.texts: t.set_text(np.round(float(t.get_text()), 2))
    # use the label_plot as the xticks and yticks
    ax.set_xticklabels(label_plot, rotation=45)
    ax.set_yticklabels(label_plot, rotation=0)

    # plt.show()
    # save the figure maximized and with the right name
    fig.savefig(output_dir+os.sep+'PhysicPropCovar'+n_PC_in_PCA+'_'+str(len(curr_sel))+'ev_dist'+str(np.round(np.min(curr_sel['distance_meteor']),2))+'-'+str(np.round(np.max(curr_sel['distance_meteor']),2))+'.png', dpi=300)

    # close the figure
    plt.close()

##########################################################################

    # with color based on the shower but skip the first 2 columns (shower_code, shower_id)
    to_plot=['mass','rho','sigma','erosion_height_start','erosion_coeff','erosion_mass_index','erosion_mass_min','erosion_mass_max','erosion_range','erosion_energy_per_unit_mass','erosion_energy_per_unit_cross_section','erosion_energy_per_unit_cross_section']
    to_plot_unit=['mass [kg]','rho [kg/m^3]','sigma [s^2/km^2]','erosion height start [km]','erosion coeff [s^2/km^2]','erosion mass index [-]','log eros. mass min [kg]','log eros. mass max [kg]','log eros. mass range [-]','erosion energy per unit mass [MJ/kg]','erosion energy per unit cross section [MJ/m^2]','erosion energy per unit cross section [MJ/m^2]']
    
    # # multiply the erosion coeff by 1000000 to have it in km/s
    # curr_df_sim_sel['erosion_coeff']=curr_df_sim_sel['erosion_coeff']*1000000
    # curr_df_sim_sel['sigma']=curr_df_sim_sel['sigma']*1000000
    # df_sel_save['erosion_coeff']=df_sel_save['erosion_coeff']*1000000
    # df_sel_save['sigma']=df_sel_save['sigma']*1000000
    # curr_df_sim_sel['erosion_energy_per_unit_cross_section']=curr_df_sim_sel['erosion_energy_per_unit_cross_section']/1000000
    # curr_df_sim_sel['erosion_energy_per_unit_mass']=curr_df_sim_sel['erosion_energy_per_unit_mass']/1000000
    # df_sel_save['erosion_energy_per_unit_cross_section']=df_sel_save['erosion_energy_per_unit_cross_section']/1000000
    # df_sel_save['erosion_energy_per_unit_mass']=df_sel_save['erosion_energy_per_unit_mass']/1000000
    # # pick the one with shower_code==current_shower+'_sel'
    # Acurr_df_sel=curr_df_sim_sel[curr_df_sim_sel['shower_code']==current_shower+'_sel']
    # Acurr_df_sim=curr_df_sim_sel[curr_df_sim_sel['shower_code']=='sim_'+current_shower]
    
    fig, axs = plt.subplots(3, 3)
    # from 2 numbers to one numbr for the subplot axs
    axs = axs.flatten()

    print('\\hline')
    # print('var & $real$ & $1D_{KDE}$ & $1D_{KDE}\\%_{dif}$ & $allD_{KDE}$ & $allD_{KDE}\\%_{dif}$\\\\')
    # print('var & real & mode & min$_{KDE}$ & -1\\sigma/+1\\sigma & -2\\sigma/+2\\sigma \\\\')
    print('Variables & Real & Mode & Min$_{KDE}$ & 95\\%CIlow & 95\\%CIup \\\\')

    ii_densest=0        
    for i in range(9):
        # put legendoutside north
        plotvar=to_plot[i]


        if Sim_data_distribution==True:
            if plotvar == 'erosion_mass_min' or plotvar == 'erosion_mass_max':
                # take the log of the erosion_mass_min and erosion_mass_max
                curr_df_sim_sel[plotvar]=np.log10(curr_df_sim_sel[plotvar])
                df_sel_save[plotvar]=np.log10(df_sel_save[plotvar])
                curr_sel[plotvar]=np.log10(curr_sel[plotvar])
                if len(curr_sel_data)>8:
                    densest_point[ii_densest]=np.log10(densest_point[ii_densest])
                    # densest_point[ii_densest-1]=np.log10(densest_point[ii_densest-1])
            # sns.histplot(curr_df_sim_sel, x=curr_df_sim_sel[plotvar], weights=curr_df_sim_sel['weight'],hue='shower_code', ax=axs[i], kde=True, palette='bright', bins=20)
            sns.histplot(curr_df_sim_sel, x=curr_df_sim_sel[plotvar], weights=curr_df_sim_sel['weight'],hue='shower_code', ax=axs[i], palette='bright', bins=20)
            # # add the kde to the plot probability density function
            sns.histplot(curr_sel, x=curr_sel[plotvar], weights=curr_sel['weight'], bins=20, ax=axs[i], fill=False, edgecolor=False, color='r', kde=True, binrange=[np.min(curr_df_sim_sel[plotvar]),np.max(curr_df_sim_sel[plotvar])])
            kde_line = axs[i].lines[-1]

            # if the only_select_meteors_from is equal to any curr_df_sim_sel plot the observed event value as a vertical red line
            if only_select_meteors_from in df_sel_save['solution_id'].values:
                axs[i].axvline(x=df_sel_save[df_sel_save['solution_id']==only_select_meteors_from][plotvar].values[0], color='k', linewidth=2)

            if plotvar == 'erosion_mass_min' or plotvar == 'erosion_mass_max':
                # put it back as it was
                curr_df_sim_sel[plotvar]=10**curr_df_sim_sel[plotvar]
                df_sel_save[plotvar]=10**df_sel_save[plotvar]
                curr_sel[plotvar]=10**curr_sel[plotvar]
            
        elif Sim_data_distribution==False:

                if plotvar == 'erosion_mass_min' or plotvar == 'erosion_mass_max':
                    # sns.histplot(curr_df_sim_sel, x=curr_df_sim_sel[plotvar], weights=curr_df_sim_sel['weight'],hue='solution_id_dist', ax=axs[i], multiple="stack", kde=True, bins=20, binrange=[np.min(df_sel_save[plotvar]),np.max(df_sel_save[plotvar])])
                    sns.histplot(curr_df_sim_sel, x=np.log10(curr_df_sim_sel[plotvar]), weights=curr_df_sim_sel['weight'],hue='solution_id_dist', ax=axs[i], multiple="stack", bins=20, binrange=[np.log10(np.min(df_sel_save[plotvar])),np.log10(np.max(df_sel_save[plotvar]))])
                    # # add the kde to the plot as a probability density function
                    sns.histplot(curr_df_sim_sel, x=np.log10(curr_df_sim_sel[plotvar]), weights=curr_df_sim_sel['weight'], bins=20, ax=axs[i],  multiple="stack", fill=False, edgecolor=False, color='r', kde=True, binrange=[np.log10(np.min(df_sel_save[plotvar])),np.log10(np.max(df_sel_save[plotvar]))])
                    
                    kde_line = axs[i].lines[-1]
                    
                    # if the only_select_meteors_from is equal to any curr_df_sim_sel plot the observed event value as a vertical red line
                    if only_select_meteors_from in df_sel_save['solution_id'].values:
                        axs[i].axvline(x=np.log10(df_sel_save[df_sel_save['solution_id']==only_select_meteors_from][plotvar].values[0]), color='k', linewidth=2)
                    if len(curr_sel_data)>8:
                        densest_point[ii_densest]=np.log10(densest_point[ii_densest])
                        # densest_point[ii_densest-1]=np.log10(densest_point[ii_densest-1])
                
                else:
                    # sns.histplot(curr_df_sim_sel, x=curr_df_sim_sel[plotvar], weights=curr_df_sim_sel['weight'],hue='solution_id_dist', ax=axs[i], multiple="stack", kde=True, bins=20, binrange=[np.min(df_sel_save[plotvar]),np.max(df_sel_save[plotvar])])
                    sns.histplot(curr_df_sim_sel, x=curr_df_sim_sel[plotvar], weights=curr_df_sim_sel['weight'],hue='solution_id_dist', ax=axs[i], multiple="stack", bins=20, binrange=[np.min(df_sel_save[plotvar]),np.max(df_sel_save[plotvar])])
                    # # add the kde to the plot as a probability density function
                    sns.histplot(curr_df_sim_sel, x=curr_df_sim_sel[plotvar], weights=curr_df_sim_sel['weight'], bins=20, ax=axs[i],  multiple="stack", fill=False, edgecolor=False, color='r', kde=True, binrange=[np.min(df_sel_save[plotvar]),np.max(df_sel_save[plotvar])])
                    
                    kde_line = axs[i].lines[-1]

                    # if the only_select_meteors_from is equal to any curr_df_sim_sel plot the observed event value as a vertical red line
                    if only_select_meteors_from in df_sel_save['solution_id'].values:
                        axs[i].axvline(x=df_sel_save[df_sel_save['solution_id']==only_select_meteors_from][plotvar].values[0], color='k', linewidth=2)
                        # put the value of diff_percent_1d at th upper left of the line
                        

        # Get the x and y data from the KDE line
        x = kde_line.get_xdata()
        y = kde_line.get_ydata()

        # \hline
        # $h_{init}$ & $v_{init}$ & $\vartheta_z$ &size  & $\tau$ & Peak Mag\\
        # \hline
        # 130 km & 7 to 9 km/s & 88° to 85° &5 to 20 cm & 0.001 to 0.1 & 0 to -5 \\
        # \hline

        # Find the index of the maximum y value
        max_index = np.argmax(y)
        if i!=8:
            # Plot a dot at the maximum point
            axs[i].plot(x[max_index], y[max_index], 'ro')  # 'ro' for red dot

        if len(curr_sel_data)>8:        
            if len(densest_point)>ii_densest:                    
            
                # print(densest_point[ii_densest])
                # Find the index with the closest value to densest_point[ii_dense] to all y values
                densest_index = find_closest_index(x, [densest_point[ii_densest]])

                # add also the densest_point[i] as a blue dot
                axs[i].plot(densest_point[ii_densest], y[densest_index[0]], 'bo')
                # get te 97.72nd percentile and the 2.28th percentile of curr_sel[plotvar] and call them sigma_97 and sigma_2
                sigma_97=np.percentile(curr_sel[plotvar], 95)
                sigma_84=np.percentile(curr_sel[plotvar], 84.13)
                sigma_15=np.percentile(curr_sel[plotvar], 15.87)
                sigma_2=np.percentile(curr_sel[plotvar], 5)
                
                x_10mode=x[max_index]
                if plotvar == 'erosion_mass_min' or plotvar == 'erosion_mass_max':
                    densest_point[ii_densest]=10**(densest_point[ii_densest])
                    x_10mode=10**x[max_index]

                if i<9:
                    print('\\hline') #df_sel_save[df_sel_save['solution_id']==only_select_meteors_from][plotvar].values[0]
                    # print(f"{to_plot_unit[i]} & ${'{:.4g}'.format(df_sel_save[df_sel_save['solution_id']==only_select_meteors_from][plotvar].values[0])}$ & ${'{:.4g}'.format(x_10mode)}$ & $ {'{:.2g}'.format(percent_diff_1D[i])}$\\% & $ {'{:.4g}'.format(densest_point[i])}$ & $ {'{:.2g}'.format(percent_diff_allD[i])}$\\% \\\\")
                    # print(to_plot_unit[i]+'& $'+str(x[max_index])+'$ & $'+str(percent_diff_1D[i])+'$\\% & $'+str(densest_point[ii_densest])+'$ & $'+str(percent_diff_allD[i])+'\\% \\\\')
                    # print(f"{to_plot_unit[i]} & ${'{:.4g}'.format(df_sel_save[df_sel_save['solution_id']==only_select_meteors_from][plotvar].values[0])}$ & ${'{:.4g}'.format(x_10mode)}$ & $ {'{:.2g}'.format(percent_diff_1D[i])}$\\% & $ {'{:.4g}'.format(densest_point[i])}$ & $ {'{:.2g}'.format(percent_diff_allD[i])}$\\% \\\\")
                    # print(f"{to_plot_unit[i]} & {'{:.4g}'.format(df_sel_save[df_sel_save['solution_id']==only_select_meteors_from][plotvar].values[0])} & {'{:.4g}'.format(x_10mode)} & {'{:.4g}'.format(densest_point[i])} & {'{:.4g}'.format(sigma_15)} / {'{:.4g}'.format(sigma_84)} & {'{:.4g}'.format(sigma_2)} / {'{:.4g}'.format(sigma_97)} \\\\")
                    print(f"{to_plot_unit[i]} & {'{:.4g}'.format(df_sel_save[df_sel_save['solution_id']==only_select_meteors_from][plotvar].values[0])} & {'{:.4g}'.format(x_10mode)} & {'{:.4g}'.format(densest_point[i])} & {'{:.4g}'.format(sigma_2)} & {'{:.4g}'.format(sigma_97)} \\\\")
                ii_densest=ii_densest+1

        axs[i].set_ylabel('probability')
        axs[i].set_xlabel(to_plot_unit[i])
        
        # # plot the legend outside the plot
        # axs[i].legend()
        axs[i].get_legend().remove()
            

        if i==0:
            # place the xaxis exponent in the bottom right corner
            axs[i].xaxis.get_offset_text().set_x(1.10)

    # # more space between the subplots erosion_coeff sigma
    plt.tight_layout()

    print('\\hline')
    
    # plt.show()
    print(output_dir+os.sep+'PhysicProp'+n_PC_in_PCA+'_'+str(len(curr_sel))+'ev_dist'+str(np.round(np.min(curr_sel['distance_meteor']),2))+'-'+str(np.round(np.max(curr_sel['distance_meteor']),2))+'.png')
    # save the figure maximized and with the right name
    fig.savefig(output_dir+os.sep+'PhysicProp'+n_PC_in_PCA+'_'+str(len(curr_sel))+'ev_dist'+str(np.round(np.min(curr_sel['distance_meteor']),2))+'-'+str(np.round(np.max(curr_sel['distance_meteor']),2))+'.png', dpi=300)

    # close the figure
    plt.close()

    # Close the Logger to ensure everything is written to the file STOP COPY in TXT file
    sys.stdout.close()

    # Reset sys.stdout to its original value if needed
    sys.stdout = sys.__stdout__

                # # cumulative distribution histogram of the distance wihouth considering the first two elements
                # sns.histplot(curr_df_sim_sel, x=curr_df_sim_sel[plotvar][2:], weights=curr_df_sim_sel['weight'][2:],hue='shower_code', ax=axs[i], kde=True, palette='bright', bins=20, cumulative=True, stat='density')
    # # cumulative distribution histogram of the distance wihouth considering the first two elements
    # sns.histplot(curr_df_sim_sel, x=curr_df_sim_sel[plotvar][2:], weights=curr_df_sim_sel['weight'][2:],hue='shower_code', ax=axs[i], kde=True, palette='bright', bins=20, cumulative=True, stat='density')

    ##########################################################################################################




def PCA_LightCurveCoefPLOT(df_sel_shower, df_obs_shower, output_dir, fit_funct='', gen_Metsim='', mag_noise = 0.1, len_noise = 20.0, file_name_obs=''):

    # number to confront
    n_confront_obs=1
    n_confront_sel=9

    # number of PC in PCA
    with_noise=True

    # is the input data noisy
    noise_data_input=False

    # activate jachia
    jacchia_fit=False

    # activate parabolic fit
    parabolic_fit=False

    t0_fit=False

    mag_fit=False

    # 5 sigma confidence interval
    # five_sigma=False

    # Standard deviation of the magnitude Gaussian noise 1 sigma
    # SD of noise in length (m) 1 sigma in km
    len_noise= len_noise/1000
    # velocity noise 1 sigma km/s
    vel_noise = (len_noise*np.sqrt(2)/(1/FPS))

    # put the first plot in 2 sublots
    fig, ax = plt.subplots(1, 2, figsize=(17, 5))

    # group by solution_id_dist and keep only n_confront_sel from each group
    df_sel_shower = df_sel_shower.groupby('solution_id_dist').head(n_confront_sel)
    print(df_sel_shower)

    # order by distance_meteor
    df_sel_shower = df_sel_shower.sort_values('distance_meteor')

    # count duplicates and add a column for the number of duplicates
    df_sel_shower['num_duplicates'] = df_sel_shower.groupby('solution_id')['solution_id'].transform('size')

    df_sel_shower.drop_duplicates(subset='solution_id', keep='first', inplace=True)

    df_sel_shower['erosion_coeff']=df_sel_shower['erosion_coeff']*1000000
    df_sel_shower['sigma']=df_sel_shower['sigma']*1000000

    if n_confront_obs<len(df_obs_shower):
        df_obs_shower=df_obs_shower.head(n_confront_obs)
    
    if n_confront_sel<len(df_sel_shower):
        df_sel_shower=df_sel_shower.head(n_confront_sel)  

    # merge curr_sel and curr_obs
    curr_sel = pd.concat([df_obs_shower,df_sel_shower], axis=0)


    for ii in range(len(curr_sel)):
        # pick the ii element of the solution_id column 
        namefile_sel=curr_sel.iloc[ii]['solution_id']
        Metsim_flag=False

        # chec if the file exist
        if not os.path.isfile(namefile_sel):
            print('file '+namefile_sel+' not found')
            continue
        else:
            if namefile_sel.endswith('.pickle'):
                data_file = read_pickle_reduction_file(namefile_sel)

            elif namefile_sel.endswith('.json'):
                # open the json file with the name namefile_sel 
                f = open(namefile_sel,"r")
                data = json.loads(f.read())
                if 'ht_sampled' in data:
                    if noise_data_input == False:
                        data_file = read_GenerateSimulations_output(namefile_sel)
                    elif noise_data_input == True:
                        data_file = read_with_noise_GenerateSimulations_output(namefile_sel)
                else:
                    if gen_Metsim == '':
                        print('no data for the Metsim file')
                        continue

                    else:
                        # make a copy of gen_Metsim
                        data_file = gen_Metsim.copy()
                        # file metsim
                        Metsim_flag=True

            height_km=np.array(data_file['height'])/1000
            abs_mag_sim=np.array(data_file['absolute_magnitudes'])
            obs_time=np.array(data_file['time'])
            vel_kms=np.array(data_file['velocities'])/1000

        if ii==0:
            
            if with_noise==True and fit_funct!='':
                # from list to array
                height_km_err=np.array(fit_funct['height'])/1000
                abs_mag_sim_err=np.array(fit_funct['absolute_magnitudes'])

                # plot noisy area around vel_kms for vel_noise for the fix height_km
                ax[0].fill_betweenx(height_km_err, abs_mag_sim_err-mag_noise, abs_mag_sim_err+mag_noise, color='lightgray', alpha=0.5)

                obs_time_err=np.array(fit_funct['time'])
                vel_kms_err=np.array(fit_funct['velocities'])/1000

                # plot noisy area around vel_kms for vel_noise for the fix height_km
                ax[1].fill_between(obs_time_err, vel_kms_err-vel_noise, vel_kms_err+vel_noise, color='lightgray', alpha=0.5)

            ax[0].plot(abs_mag_sim,height_km)

            ax[1].plot(obs_time, vel_kms,label=file_name_obs[:15])
        else:
            ax[0].plot(abs_mag_sim,height_km)
            
            if Metsim_flag:
                ax[1].plot(obs_time, vel_kms,label='Manual MetSim reduction\n\
        N°duplic. '+str(round(curr_sel.iloc[ii]['num_duplicates']))+' min dist:'+str(round(curr_sel.iloc[ii]['distance_meteor'],2))+'\n\
        m:'+str('{:.2e}'.format(curr_sel.iloc[ii]['mass'],1))+' F:'+str(round(curr_sel.iloc[ii]['F'],2))+'\n\
        rho:'+str(round(curr_sel.iloc[ii]['rho']))+' sigma:'+str(round(curr_sel.iloc[ii]['sigma'],4))+'\n\
        er.height:'+str(round(curr_sel.iloc[ii]['erosion_height_start'],2))+' er.log:'+str(round(curr_sel.iloc[ii]['erosion_range'],1))+'\n\
        er.coeff:'+str(round(curr_sel.iloc[ii]['erosion_coeff'],3))+' er.index:'+str(round(curr_sel.iloc[ii]['erosion_mass_index'],2)))         
            else:
                ax[1].plot(obs_time, vel_kms,label='N°duplic. '+str(round(curr_sel.iloc[ii]['num_duplicates']))+' min dist:'+str(round(curr_sel.iloc[ii]['distance_meteor'],2))+'\n\
        m:'+str('{:.2e}'.format(curr_sel.iloc[ii]['mass'],1))+' F:'+str(round(curr_sel.iloc[ii]['F'],2))+'\n\
        rho:'+str(round(curr_sel.iloc[ii]['rho']))+' sigma:'+str(round(curr_sel.iloc[ii]['sigma'],4))+'\n\
        er.height:'+str(round(curr_sel.iloc[ii]['erosion_height_start'],2))+' er.log:'+str(round(curr_sel.iloc[ii]['erosion_range'],1))+'\n\
        er.coeff:'+str(round(curr_sel.iloc[ii]['erosion_coeff'],3))+' er.index:'+str(round(curr_sel.iloc[ii]['erosion_mass_index'],2)))


#############ADD COEF#############################################
        if mag_fit==True:

            index_ht_peak = np.argmin(abs_mag_sim)

            ax[0].plot(curr_sel.iloc[ii]['a_mag_init']*np.array(height_km[:index_ht_peak])**2+curr_sel.iloc[ii]['b_mag_init']*np.array(height_km[:index_ht_peak])+curr_sel.iloc[ii]['c_mag_init'],height_km[:index_ht_peak], color=ax[0].lines[-1].get_color(), linestyle='None', marker='<')# , markersize=5

            ax[0].plot(curr_sel.iloc[ii]['a_mag_end']*np.array(height_km[index_ht_peak:])**2+curr_sel.iloc[ii]['b_mag_end']*np.array(height_km[index_ht_peak:])+curr_sel.iloc[ii]['c_mag_end'],height_km[index_ht_peak:], color=ax[0].lines[-1].get_color(), linestyle='None', marker='>')# , markersize=5
            
        if parabolic_fit==True:
            ax[1].plot(obs_time,curr_sel.iloc[ii]['a_acc']*np.array(obs_time)**2+curr_sel.iloc[ii]['b_acc']*np.array(obs_time)+curr_sel.iloc[ii]['c_acc'], color=ax[1].lines[-1].get_color(), linestyle='None', marker='o')# , markersize=5
        
        # Assuming the jacchiaVel function is defined as:
        def jacchiaVel(t, a1, a2, v_init):
            return v_init - np.abs(a1) * np.abs(a2) * np.exp(np.abs(a2) * t)
        if jacchia_fit==True:
            ax[1].plot(obs_time, jacchiaVel(np.array(obs_time), curr_sel.iloc[ii]['a1_acc_jac'], curr_sel.iloc[ii]['a2_acc_jac'],vel_kms[0]), color=ax[1].lines[-1].get_color(), linestyle='None', marker='d') 

        if t0_fit==True: # quadratic_velocity(t, a, v0, t0) 
            ax[1].plot(obs_time, cubic_velocity(np.array(obs_time), curr_sel.iloc[ii]['a_t0'], curr_sel.iloc[ii]['b_t0'], curr_sel.iloc[ii]['vel_init_norot'], curr_sel.iloc[ii]['t0']), color=ax[1].lines[-1].get_color(), linestyle='None', marker='s') 


    # change the first plotted line style to be a dashed line
    ax[0].lines[0].set_linestyle("None")
    ax[1].lines[0].set_linestyle("None")
    # change the first plotted marker to be a x
    ax[0].lines[0].set_marker("x")
    ax[1].lines[0].set_marker("x")
    # change first line color
    ax[0].lines[0].set_color('black')
    ax[1].lines[0].set_color('black')
    # change the zorder=-1 of the first line
    ax[0].lines[0].set_zorder(n_confront_sel)
    ax[1].lines[0].set_zorder(n_confront_sel)


    # change dot line color
    if mag_fit==True:
        ax[0].lines[1].set_color('black')
        ax[0].lines[2].set_color('black')


# check how many of the jacchia_fit and parabolic_fit and t0_fit are set to true
    numcheck=0
    if jacchia_fit==True:
        numcheck+=1
    if parabolic_fit==True:
        numcheck+=1
    if t0_fit==True:
        numcheck+=1

    if numcheck==1:
        ax[1].lines[1].set_color('black')
        ax[1].lines[1].set_zorder(n_confront_sel)
    if numcheck==2:
        ax[1].lines[1].set_color('black')
        ax[1].lines[2].set_color('black')
        ax[1].lines[1].set_zorder(n_confront_sel)
        ax[1].lines[2].set_zorder(n_confront_sel)
    if numcheck==3:
        ax[1].lines[1].set_color('black')
        ax[1].lines[2].set_color('black')
        ax[1].lines[3].set_color('black')
        ax[1].lines[1].set_zorder(n_confront_sel)
        ax[1].lines[2].set_zorder(n_confront_sel)
        ax[1].lines[3].set_zorder(n_confront_sel)

    # change the zorder=-1 of the first line
    ax[0].lines[1].set_zorder(n_confront_sel)
    ax[0].lines[2].set_zorder(n_confront_sel)


    # grid on on both subplot with -- as linestyle and light gray color
    ax[1].grid(linestyle='--',color='lightgray')
    # grid on
    ax[0].grid(linestyle='--',color='lightgray')
    # ax[0].set_title(current_shower+' height vs mag')


    if n_confront_sel <= 5:
        # pu the leggend putside the plot and adjust the plot base on the screen size
        ax[-1].legend(bbox_to_anchor=(1.05, 1.1), loc='upper left', borderaxespad=0.)
        # the legend do not fit in the plot, so adjust the plot
        plt.subplots_adjust(right=0.8)
    else:
        # pu the leggend putside the plot and adjust the plot base on the screen size
        ax[-1].legend(bbox_to_anchor=(1.05, 1.1), loc='upper left', borderaxespad=0.,fontsize="10",ncol=2)
        # the legend do not fit in the plot, so adjust the plot
        plt.subplots_adjust(right=.6)
        # push the two subplots left
        # plt.subplots_adjust(left=-.0001)
        plt.subplots_adjust(wspace=0.2)


    # invert the x axis
    ax[0].invert_xaxis()

    ax[1].set_xlabel('Time [s]')
    ax[1].set_ylabel('Velocity [km/s]')
    ax[0].set_xlabel('Absolute Magnitude [-]')
    ax[0].set_ylabel('Height [km]')

    plt.savefig(output_dir+os.sep+file_name_obs+'_Heigh_MagVelCoef.png')

    # close the plot
    plt.close()





if __name__ == "__main__":

    import argparse


    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Fom Observation and simulated data weselect the most likely through PCA, run it, and store results to disk.")
  
    arg_parser.add_argument('--input_dir', metavar='INPUT_PATH', type=str, default=r'C:\Users\maxiv\Desktop\20230811-082648.931419', \
        help="Path were are store both simulated and observed shower .csv file.")
    
    arg_parser.add_argument('--MetSim_json', metavar='METSIM_JSON', type=str, default='_sim_fit_latest.json', \
        help="json file extension where are stored the MetSim constats, by default _sim_fit_latest.json.")   

    arg_parser.add_argument('--nobs', metavar='OBS_NUM', type=int, default=9, \
        help="Number of Observation that will be resampled.")
    
    arg_parser.add_argument('--nsim', metavar='SIM_NUM', type=int, default=100, \
        help="Number of simulations to run.")

    arg_parser.add_argument('--nsel', metavar='SEL_NUM', type=int, default=10, \
        help="Number of selected simulations to consider.")
    
    arg_parser.add_argument('--PCA_percent', metavar='PCA_PERCENT', type=int, default=99, \
        help="Percentage of the variance explained by the PCA.")

    arg_parser.add_argument('--YesPCA', metavar='YESPCA', type=str, default=[], \
        help="Use specific variable to considered in PCA.")

    arg_parser.add_argument('--NoPCA', metavar='NOPCA', type=str, default=['kurtosis','skew','a1_acc_jac','a2_acc_jac','a_acc','b_acc','c_acc','c_mag_init','c_mag_end','a_t0', 'b_t0', 'c_t0'], \
        help="Use specific variable NOT considered in PCA.")

    arg_parser.add_argument('--save_plot', metavar='SAVE_PLOT', type=bool, default=True, \
        help="save the plots.")
    
    # arg_parser.add_argument('--save_csv', metavar='SAVE_CSV', type=bool, default=True, \
    #     help="save the csv file.")

    arg_parser.add_argument('--cores', metavar='CORES', type=int, default=None, \
        help="Number of cores to use. All by default.")

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################
    warnings.filterwarnings('ignore')

    # check if is a file or a directory
    if os.path.isdir(cml_args.input_dir):
        # pack the 3 lists in a tuple
        input_folder_file = find_and_extract_trajectory_files(cml_args.input_dir, cml_args.MetSim_json)
        
    else:
        # check if the file exists
        if not os.path.isfile(cml_args.input_dir):
            print('The file does not exist')
            sys.exit()
        else:
            # split the dir and file
            trajectory_files = [cml_args.input_dir]
            # for the output folder delete the extension of the file and add _GenSim
            output_folders = [os.path.splitext(cml_args.input_dir)[0]+NAME_SUFX_GENSIM]
            file_names = [os.path.splitext(os.path.split(cml_args.input_dir)[1])[0]]
            input_folders = [os.path.split(cml_args.input_dir)[0]]

            input_folder_file = [[trajectory_files[0], file_names[0], input_folders[0], output_folders[0], trajectory_files[0]]]
    

    # print only the file name in the directory split the path and take the last element
    print('Number of trajectory.pickle files find',len(input_folder_file))
    for trajectory_file, file_name, input_folder, output_folder, trajectory_Metsim_file in input_folder_file:
        print('processing file:',file_name)
        # print(trajectory_file)
        # print(input_folder)
        # print(output_folder)
        # print(trajectory_Metsim_file)

        # chek if input_folder+os.sep+file_name+NAME_SUFX_CSV_OBS exist
        if os.path.isfile(output_folder+os.sep+file_name+NAME_SUFX_CSV_OBS):
            # read the csv file
            trajectory_file = output_folder+os.sep+file_name+NAME_SUFX_CSV_OBS

        # check if the output_folder exists
        if not os.path.isdir(output_folder):
            mkdirP(output_folder)

        print()

        ######################### OBSERVATION ###############################
        print('--- OBSERVATION ---')

        # check the extension of the file
        if trajectory_file.endswith('.csv'):
            # read the csv file
            pd_dataframe_PCA_obs_real = pd.read_csv(trajectory_file)
            # check the column name solution_id	and see if it matches a file i the folder
            if not input_folder in pd_dataframe_PCA_obs_real['solution_id'][0]:
                # if the solution_id is in the name of the file then the file is the real data
                print('The folder of the csv file is different')

            if pd_dataframe_PCA_obs_real['type'][0] != 'Observation' and pd_dataframe_PCA_obs_real['type'][0] != 'Observation_sim':
                # raise an error saing that the type is wrong and canot be processed by PCA
                raise ValueError('Type of the csv file is wrong and canot be processed by script.')

            if pd_dataframe_PCA_obs_real['solution_id'][0].endswith('.pickle'):
                # read the pickle file
                gensim_data_obs = read_pickle_reduction_file(pd_dataframe_PCA_obs_real['solution_id'][0])

            # json file
            elif pd_dataframe_PCA_obs_real['solution_id'][0].endswith('.json'): 
                # read the json file with noise
                gensim_data_obs = read_with_noise_GenerateSimulations_output(pd_dataframe_PCA_obs_real['solution_id'][0])

            else:
                # raise an error if the file is not a csv, pickle or json file
                raise ValueError('File format not supported. Please provide a csv, pickle or json file.')

            rmsd_t0_lag, rmsd_pol_mag, fit_pol_mag, fitted_lag_t0_lag, fit_funct = find_noise_of_data(gensim_data_obs)

            print('read the csv file:',trajectory_file)

        else:

            if trajectory_file.endswith('.pickle'):
                # read the pickle file
                gensim_data_obs = read_pickle_reduction_file(trajectory_file) #,trajectory_Metsim_file

            # json file
            elif trajectory_file.endswith('.json'): 
                # read the json file with noise
                gensim_data_obs = read_with_noise_GenerateSimulations_output(trajectory_file)
                
            else:
                # raise an error if the file is not a csv, pickle or json file
                raise ValueError('File format not supported. Please provide a csv, pickle or json file.')
            
            pd_dataframe_PCA_obs_real = array_to_pd_dataframe_PCA(gensim_data_obs)

            if cml_args.save_plot:
                # run generate_observation_realization with the gensim_data_obs
                rmsd_t0_lag, rmsd_pol_mag, fit_pol_mag, fitted_lag_t0_lag, fit_funct, fig, ax = find_noise_of_data(gensim_data_obs,cml_args.save_plot)
                # make the results_list to incorporate all rows of pd_dataframe_PCA_obs_real
                results_list = []
                for ii in range(cml_args.nobs):
                    results_pd = generate_observation_realization(gensim_data_obs, rmsd_t0_lag, rmsd_pol_mag, fit_pol_mag, fitted_lag_t0_lag,'realization_'+str(ii+1), fig, ax, cml_args.save_plot) 
                    results_list.append(results_pd)

                # Save the figure as file with instead of _trajectory.pickle it has file+std_dev.png on the desktop
                plt.savefig(output_folder+os.sep+file_name+'obs_realizations.png', dpi=300)

                plt.close()

            else:      
                rmsd_t0_lag, rmsd_pol_mag, fit_pol_mag, fitted_lag_t0_lag, fit_funct = find_noise_of_data(gensim_data_obs)       
                input_list_obs = [[gensim_data_obs, rmsd_t0_lag, rmsd_pol_mag, fit_pol_mag, fitted_lag_t0_lag,'realization_'+str(ii+1)] for ii in range(cml_args.nobs)]
                results_list = domainParallelizer(input_list_obs, generate_observation_realization, cores=cml_args.cores)
            
            df_obs_realiz = pd.concat(results_list)
            pd_dataframe_PCA_obs_real = pd.concat([pd_dataframe_PCA_obs_real, df_obs_realiz])
            # re index the dataframe
            pd_dataframe_PCA_obs_real.reset_index(drop=True, inplace=True)

            # check if there is a column with the name 'mass'
            if 'mass' in pd_dataframe_PCA_obs_real.columns:
                #delete from the real_data panda dataframe mass rho sigma
                pd_dataframe_PCA_obs_real = pd_dataframe_PCA_obs_real.drop(columns=['mass', 'rho', 'sigma', 'erosion_height_start', 'erosion_coeff', 'erosion_mass_index', 'erosion_mass_min', 'erosion_mass_max', 'erosion_range', 'erosion_energy_per_unit_cross_section', 'erosion_energy_per_unit_mass'])

            pd_dataframe_PCA_obs_real.to_csv(output_folder+os.sep+file_name+NAME_SUFX_CSV_OBS, index=False)
            # print saved csv file
            print()
            print('saved obs csv file:',output_folder+os.sep+file_name+NAME_SUFX_CSV_OBS)
        
        print('Run MetSim file:',trajectory_Metsim_file)
        # use Metsim_file in 
        simulation_MetSim_object, gensim_data_Metsim, pd_datafram_PCA_sim_Metsim = run_simulation(trajectory_Metsim_file, gensim_data_obs)
        
        print()




        ######################## SIMULATIONTS ###############################
        print('--- SIMULATIONS ---')

        # open the folder and extract all the json files
        os.chdir(input_folder)

        # chek in directory if it exist a csv file with input_folder+os.sep+file_name+NAME_SUFX_CSV_SIM
        if os.path.isfile(output_folder+os.sep+file_name+NAME_SUFX_CSV_SIM):
            # read the csv file
            pd_datafram_PCA_sim = pd.read_csv(output_folder+os.sep+file_name+NAME_SUFX_CSV_SIM)
            print('read the csv file:',output_folder+os.sep+file_name+NAME_SUFX_CSV_SIM)

        else:

            # open the folder and extract all the json files
            os.chdir(output_folder)

            extension = 'json'
            # walk thorought the directories and find all the json files inside each folder inside the directory
            all_jsonfiles_check = [i for i in glob.glob('**/*.{}'.format(extension), recursive=True)]

            if len(all_jsonfiles_check) == 0 or len(all_jsonfiles_check) < cml_args.nsim:
                if len(all_jsonfiles_check) != 0:
                    print('In the sim folder there are already',len(all_jsonfiles_check),'json files')
                    print('Add',cml_args.nsim - len(all_jsonfiles_check),' json files')
                number_sim_to_run_and_simulation_in_folder = cml_args.nsim - len(all_jsonfiles_check)
                
                # run the new simulations
                if cml_args.save_plot:
                    fig, ax = generate_simulations(pd_dataframe_PCA_obs_real,simulation_MetSim_object,gensim_data_obs,number_sim_to_run_and_simulation_in_folder,output_folder,cml_args.save_plot)
                    # plot gensim_data_Metsim
                    plot_side_by_side(gensim_data_Metsim,fig, ax,'k-','MetSim')

                    # save the plot
                    plt.savefig(output_folder+os.sep+file_name+'_obs_sim.png', dpi=300)
                    # close the plot
                    plt.close()
                    # print saved csv file
                    print('saved image '+output_folder+os.sep+file_name+'_obs_sim.png')
                else:
                    generate_simulations(pd_dataframe_PCA_obs_real,simulation_MetSim_object,gensim_data_obs,number_sim_to_run_and_simulation_in_folder,output_folder,cml_args.save_plot)
                    
            print('start reading the json files')

            all_jsonfiles = [i for i in glob.glob('**/*.{}'.format(extension), recursive=True)]

            # add the output_folder to all_jsonfiles
            all_jsonfiles = [output_folder+os.sep+file for file in all_jsonfiles]

            # open the folder and extract all the json files
            os.chdir(input_folder)

            print('Number of simulated files: ',len(all_jsonfiles))

            input_list = [[all_jsonfiles[ii], 'simulation_'+str(ii+1)] for ii in range(len(all_jsonfiles))]
            results_list = domainParallelizer(input_list, read_GenerateSimulations_output_to_PCA, cores=cml_args.cores)
            
            # if no read the json files in the folder and create a new csv file
            pd_datafram_PCA_sim = pd.concat(results_list)
            
            # concatenate the two dataframes
            pd_datafram_PCA_sim = pd.concat([pd_datafram_PCA_sim_Metsim, pd_datafram_PCA_sim])
            
            # print(df_sim_shower)
            pd_datafram_PCA_sim.reset_index(drop=True, inplace=True)

            pd_datafram_PCA_sim.to_csv(output_folder+os.sep+file_name+NAME_SUFX_CSV_SIM, index=False)
            # print saved csv file
            print('saved sim csv file:',output_folder+os.sep+file_name+NAME_SUFX_CSV_SIM)

        print()
            
        ######################## SELECTION ###############################

        print('--- SELECTION ---')
        
        pd_datafram_PCA_selected = PCASim(pd_datafram_PCA_sim, pd_dataframe_PCA_obs_real, output_folder, cml_args.PCA_percent, cml_args.nsel, cml_args.YesPCA, cml_args.NoPCA, file_name, cml_args.cores, cml_args.save_plot)

        PCA_LightCurveCoefPLOT(pd_datafram_PCA_selected, pd_dataframe_PCA_obs_real, output_folder, fit_funct, gensim_data_Metsim, rmsd_pol_mag, rmsd_t0_lag, file_name)


