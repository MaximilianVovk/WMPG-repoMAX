
"""
The code is used to extract the physical properties of the simulated showers from EMCCD observations
by selecting the most similar simulated shower.
The code is divided in three parts:
    1. from GenerateSimulations.py output folder extract the simulated showers observable and physiscal characteristics
    2. extract from the EMCCD solution_table.json file the observable property of the shower
    3. select the simulated meteors similar to the EMCCD meteor observations and extract their physical properties
latest update: 2021-05-25
"""

import json
# import copy
# import multiprocessing
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import seaborn as sns
import scipy.spatial.distance
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# from heapq import nsmallest
import wmpl
# from wmpl.MetSim.GUI import loadConstants
import shutil
from scipy.stats import kurtosis, skew
from wmpl.Utils.OSTools import mkdirP
import math
from wmpl.Utils.PyDomainParallelizer import domainParallelizer
# from scipy.optimize import curve_fit # faster 
# from scipy.optimize import basinhopping # slower but more accurate
from scipy.optimize import minimize


add_json_noise = False

PCA_percent = 99

# CONST #######################################

class Constants(object):
    def __init__(self):
        """ Constant parameters for the ablation modelling. """

        ### Simulation parameters ###

        # Time step
        self.dt = 0.005

        # Time elapsed since the beginning
        self.total_time = 0

        # Number of active fragments
        self.n_active = 0

        # Minimum possible mass for ablation (kg)
        self.m_kill = 1e-14

        # Minimum ablation velocity (m/s)
        self.v_kill = 3000

        # Minimum height (m)
        self.h_kill = 60000

        # Initial meteoroid height (m)
        self.h_init = 180000

        # Power of a 0 magnitude meteor
        self.P_0m = 840

        # Atmosphere density coefficients
        self.dens_co = np.array([6.96795507e+01, -4.14779163e+03, 9.64506379e+04, -1.16695944e+06, \
            7.62346229e+06, -2.55529460e+07, 3.45163318e+07])
        
        # Radius of the Earth (m)
        self.r_earth = 6_371_008.7714

        self.total_fragments = 0

        ### ###


        ### Wake parameters ###

        # PSF stddev (m)
        self.wake_psf = 3.0

        # Wake extension from the leading fragment (m)
        self.wake_extension = 200

        ### ###



        ### Main meteoroid properties ###

        # Meteoroid bulk density (kg/m^3)
        self.rho = 1000

        # Initial meteoroid mass (kg)
        self.m_init = 2e-5

        # Initial meteoroid veocity (m/s)
        self.v_init = 23570

        # Shape factor (1.21 is sphere)
        self.shape_factor = 1.21

        # Main fragment ablation coefficient (s^2/km^2)
        self.sigma = 0.023/1e6

        # Zenith angle (radians)
        self.zenith_angle = math.radians(45)

        # Drag coefficient
        self.gamma = 1.0

        # Grain bulk density (kg/m^3)
        self.rho_grain = 3000


        # Luminous efficiency type
        #   0 - Constant
        #   1 - TDB
        #   2 - TDB ...
        self.lum_eff_type = 0

        # Constant luminous efficiency (percent)
        self.lum_eff = 0.7

        # Mean atomic mass of a meteor atom, kg (Jones 1997)
        self.mu = 23*1.66*1e-27

        ### ###


        ### Erosion properties ###

        # Toggle erosion on/off
        self.erosion_on = True


        # Bins per order of magnitude mass
        self.erosion_bins_per_10mass = 10
        
        # Height at which the erosion starts (meters)
        self.erosion_height_start = 102000

        # Erosion coefficient (s^2/m^2)
        self.erosion_coeff = 0.33/1e6

        
        # Height at which the erosion coefficient changes (meters)
        self.erosion_height_change = 90000

        # Erosion coefficient after the change (s^2/m^2)
        self.erosion_coeff_change = 0.33/1e6

        # Density after erosion change (density of small chondrules by default)
        self.erosion_rho_change = 3700

        # Ablation coeff after erosion change
        self.erosion_sigma_change = self.sigma


        # Grain mass distribution index
        self.erosion_mass_index = 2.5

        # Mass range for grains (kg)
        self.erosion_mass_min = 1.0e-11
        self.erosion_mass_max = 5.0e-10

        ###


        ### Disruption properties ###

        # Toggle disruption on/off
        self.disruption_on = True

        # Meteoroid compressive strength (Pa)
        self.compressive_strength = 2000

        # Height of disruption (will be assigned when the disruption occures)
        self.disruption_height = None

        # Erosion coefficient to use after disruption
        self.disruption_erosion_coeff = self.erosion_coeff

        # Disruption mass distribution index
        self.disruption_mass_index = 2.0


        # Mass ratio for disrupted fragments as the ratio of the disrupted mass
        self.disruption_mass_min_ratio = 1.0/100
        self.disruption_mass_max_ratio = 10.0/100

        # Ratio of mass that will disrupt into grains
        self.disruption_mass_grain_ratio = 0.25

        ### ###


        ### Complex fragmentation behaviour ###

        # Indicate if the complex fragmentation is used
        self.fragmentation_on = False

        # Track light curves of individual fragments
        self.fragmentation_show_individual_lcs = False

        # A list of fragmentation entries
        self.fragmentation_entries = []

        # Name of the fragmentation file
        self.fragmentation_file_name = "metsim_fragmentation.txt"

        ### ###


        ### Radar measurements ###

        # Height at which the electron line density is measured (m)
        self.electron_density_meas_ht = -1000

        # Measured electron line density (e-/m)
        self.electron_density_meas_q = -1

        ### ###


        
        ### OUTPUT PARAMETERS ###

        # Velocity at the beginning of erosion
        self.erosion_beg_vel = None

        # Mass at the beginning of erosion
        self.erosion_beg_mass = None

        # Dynamic pressure at the beginning of erosion
        self.erosion_beg_dyn_press = None

        # Mass of main fragment at erosion change
        self.mass_at_erosion_change = None

        # Energy received per unit cross section prior to to erosion begin
        self.energy_per_cs_before_erosion = None

        # Energy received per unit mass prior to to erosion begin
        self.energy_per_mass_before_erosion = None

        # Height at which the main mass was depleeted
        self.main_mass_exhaustion_ht = None


        ### ###


# FUNCTIONS ###########################################################################################


def loadConstants(sim_fit_json):
    """ Load the simulation constants from a JSON file. 
        
    Arguments:
        sim_fit_json: [str] Path to the sim_fit JSON file.

    Return:
        (const, const_json): 
            - const: [Constants object]
            - const_json: [dict]

    """

    # Init the constants
    const = Constants()


    # Load the nominal simulation
    with open(sim_fit_json) as f:
        const_json = json.load(f)


        # Fill in the constants
        for key in const_json:
            setattr(const, key, const_json[key])
            
    if 'const' in const_json:
        # Open the constants parameter part of .json file for simulaitons
        for key in const_json['const']:
            setattr(const, key, const_json['const'][key])


    if 'fragmentation_entries' in const_json:

        # Convert fragmentation entries from dictionaties to objects
        frag_entries = []
        if len(const_json['fragmentation_entries']) > 0:
            for frag_entry_dict in const_json['fragmentation_entries']:

                # Only take entries which are variable names for the FragmentationEntry class
                frag_entry_dict = {key:frag_entry_dict[key] for key in frag_entry_dict \
                    if key in FragmentationEntry.__init__.__code__.co_varnames}

                frag_entry = FragmentationEntry(**frag_entry_dict)
                frag_entries.append(frag_entry)

        const.fragmentation_entries = frag_entries


    return const, const_json

def find_closest_index(time_arr, time_sampled):
    closest_indices = []
    for sample in time_sampled:
        closest_index = min(range(len(time_arr)), key=lambda i: abs(time_arr[i] - sample))
        closest_indices.append(closest_index)
    return closest_indices

def quadratic_lag(t, a, t0):
    """
    Quadratic lag function.
    """

    # Only take times <= t0
    t_before = t[t <= t0]

    # Only take times > t0
    t_after = t[t > t0]

    # Compute the lag linearly before t0
    l_before = np.zeros_like(t_before)

    # Compute the lag quadratically after t0
    l_after = -abs(a)*(t_after - t0)**3

    return np.concatenate((l_before, l_after))

def lag_residual(params, t_time, l_data):
    """
    Residual function for the optimization.
    """

    return np.sum((l_data - quadratic_lag(t_time, *params))**2)

def read_GenerateSimulations_folder_output(shower_folder,Shower='', data_id=None):
    ''' 
    It reads the GenerateSimulations.py output json files from the shower_folder and extract the observable and physical property
    The values are given in a dataframe format and if requestd are saved in a .csv file called Shower+".csv"
    Keyword arguments:
    shower_folder:  folder of the simulated meteors.
    Shower:         Shower name, by default there is no name.
    save_it:        Boolean - save the extracted dataframe in a .csv, by default it is not saved.
    '''


    if data_id is None:
        # open the folder and extract all the json files
        os.chdir(shower_folder)
        directory=os.getcwd()
        extension = 'json'
        all_jsonfiles = [i for i in glob.glob('*.{}'.format(extension))]
    else:
        os.chdir(shower_folder)
        directory=os.getcwd()
        all_jsonfiles=data_id

    # save all the variables in a list ths is used to initialized the values of the dataframe
    dataList = [['','', 0, 0, 0,\
        0, 0, 0, 0, 0, 0, 0, 0,\
        0, 0, 0, 0, 0, 0, 0, 0, 0,\
        0, 0,\
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\
        0, 0, 0, 0, 0,\
        0, 0, 0,\
        0, 0]]

    # create a dataframe to store the data
    df_json = pd.DataFrame(dataList, columns=['solution_id','shower_code','vel_init_norot','vel_avg_norot','duration',\
    'mass','peak_mag_height','begin_height','end_height','t0','peak_abs_mag','beg_abs_mag','end_abs_mag',\
    'F','trail_len','deceleration_lin','deceleration_parab','decel_jacchia','decel_t0','zenith_angle', 'kurtosis','skew',\
    'kc','Dynamic_pressure_peak_abs_mag',\
    'a_acc','b_acc','c_acc','a1_acc_jac','a2_acc_jac','a_mag_init','b_mag_init','c_mag_init','a_mag_end','b_mag_end','c_mag_end',\
    'rho','sigma','erosion_height_start','erosion_coeff', 'erosion_mass_index',\
    'erosion_mass_min','erosion_mass_max','erosion_range',\
    'erosion_energy_per_unit_cross_section', 'erosion_energy_per_unit_mass'])

    There_is_data=False

    # open the all file and extract the data
    for i in range(len(all_jsonfiles)):
    # for i in range(len(all_jsonfiles[1:100])):
        f = open(all_jsonfiles[i],"r")
        data = json.loads(f.read())

        # show the current processed file and the number of files left to process
        # print(all_jsonfiles[i]+' - '+str(len(all_jsonfiles)-i)+' left')
        print(all_jsonfiles[i])

        if data['ht_sampled']!= None: 
            # from 'params' extract the observable parameters and save them in a list
            # get only the .json file name
            name=all_jsonfiles[i]
            shower_code = 'sim_'+Shower

            zenith_angle= data['params']['zenith_angle']['val']*180/np.pi

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

            # divide the vel_sim by 1000 considering is a list
            time_sim = [i-time_sim[0] for i in time_sim]
            vel_sim = [i/1000 for i in vel_sim]
            len_sim = [(i-len_sim[0])/1000 for i in len_sim]
            ht_sim = [i/1000 for i in ht_sim]
            
            ht_obs=[x/1000 for x in ht_obs]

            closest_indices = find_closest_index(ht_sim, ht_obs)

            v0 = vel_sim[0]

            Dynamic_pressure= data['simulation_results']['leading_frag_dyn_press_arr']
            Dynamic_pressure= Dynamic_pressure[index_ht_sim:index_ht_sim_end]
            Dynamic_pressure=[Dynamic_pressure[jj_index_cut] for jj_index_cut in closest_indices]

            # read the gaussian noise from the json file or not
            if add_json_noise == False:  
                abs_mag_sim=[abs_mag_sim[jj_index_cut] for jj_index_cut in closest_indices]
                vel_sim=[vel_sim[jj_index_cut] for jj_index_cut in closest_indices]
                time_sim=[time_sim[jj_index_cut] for jj_index_cut in closest_indices]
                ht_sim=[ht_sim[jj_index_cut] for jj_index_cut in closest_indices]
                len_sim=[len_sim[jj_index_cut] for jj_index_cut in closest_indices]

                obs_time=time_sim
                ht_obs = ht_sim
                abs_mag_obs=abs_mag_sim
                obs_vel=vel_sim
                obs_length=len_sim
               
            elif add_json_noise == True:
                obs_time=data['time_sampled']
                obs_length=data['len_sampled']
                abs_mag_obs=data['mag_sampled']
                obs_vel=[v0]
                obs_length=[x/1000 for x in obs_length]
                obs_height=[x/1000 for x in obs_height]
                # append from vel_sampled the rest by the difference of the first element of obs_length divided by the first element of obs_time
                rest_vel_sampled=[(obs_length[vel_ii]-obs_length[vel_ii-1])/(obs_time[vel_ii]-obs_time[vel_ii-1]) for vel_ii in range(1,len(obs_length))]
                # append the rest_vel_sampled to vel_sampled
                obs_vel.extend(rest_vel_sampled)

                
            # create the lag array as the difference betyween the lenght and v0*time+len_sim[0]    
            obs_lag=obs_length-(v0*np.array(obs_time)+obs_length[0])
            Dynamic_pressure_peak_abs_mag = Dynamic_pressure[np.argmin(abs_mag_obs)]

            # from 'time_sampled' extract the last element and save it in a list
            duration = obs_time[-1]
            begin_height = ht_obs[0]
            end_height = ht_obs[-1]
            peak_abs_mag = abs_mag_obs[np.argmin(abs_mag_obs)]
            F = (begin_height - (ht_obs[np.argmin(abs_mag_obs)])) / (begin_height - end_height)
            peak_mag_height = ht_obs[np.argmin(abs_mag_obs)]
            beg_abs_mag	= abs_mag_obs[0]
            end_abs_mag	= abs_mag_obs[-1]
            trail_len = obs_length[-1]
            vel_avg_norot = trail_len / duration

            kc_par = begin_height + (2.86 - 2*np.log(v0))/0.0612

            # fit a line to the throught the vel_sim and ht_sim
            a, b = np.polyfit(obs_time,obs_vel, 1)
            acceleration_lin = a

            t0 = np.mean(obs_time)

            # initial guess of deceleration decel equal to linear fit of velocity
            p0 = [a, t0]

            opt_res = opt.minimize(lag_residual, p0, args=(np.array(obs_time), np.array(obs_lag)), method='Nelder-Mead')

            # sample the fit for the velocity and acceleration
            decel_t0, t0 = opt_res.x

            decel_t0=-abs(decel_t0)


            a3, b3, c3 = np.polyfit(obs_time,obs_vel, 2)
            acceleration_parab=a3*2 + b3

            # Assuming the jacchiaVel function is defined as:
            def jacchiaVel(t, a1, a2, v_init):
                return v_init - np.abs(a1) * np.abs(a2) * np.exp(np.abs(a2) * t)

            # Generating synthetic observed data for demonstration
            t_observed = np.array(obs_time)  # Observed times

            # Residuals function for optimization
            def residuals(params):
                a1, a2 = params
                predicted_velocity = jacchiaVel(t_observed, a1, a2, v0)
                return np.sum((obs_vel - predicted_velocity)**2)

            # Initial guess for a1 and a2
            initial_guess = [0.005,	10]

            # Apply minimize to the residuals
            result = minimize(residuals, initial_guess)

            # Results
            jac_a1, jac_a2 = abs(result.x)

            acc_jacchia = abs(jac_a1)*abs(jac_a2)

            # fit a line to the throught the obs_vel and ht_sim
            index_ht_peak = next(x for x, val in enumerate(ht_obs) if val <= peak_mag_height)

            # check if the ht_obs[:index_ht_peak] and abs_mag_obs[:index_ht_peak] are empty
            if ht_obs[:index_ht_peak] == [] or abs_mag_obs[:index_ht_peak] == []:
                a3_Inabs, b3_Inabs, c3_Inabs = 0, 0, 0
            else:
                a3_Inabs, b3_Inabs, c3_Inabs = np.polyfit(ht_obs[:index_ht_peak], abs_mag_obs[:index_ht_peak], 2)

            # check if the ht_obs[index_ht_peak:] and abs_mag_obs[index_ht_peak:] are empty
            if ht_obs[index_ht_peak:] == [] or abs_mag_obs[index_ht_peak:] == []:
                a3_Outabs, b3_Outabs, c3_Outabs = 0, 0, 0
            else:
                a3_Outabs, b3_Outabs, c3_Outabs = np.polyfit(ht_obs[index_ht_peak:], abs_mag_obs[index_ht_peak:], 2)

            # # find the index of the first element of the simulation that is equal to the first element of the observation
            mag_sampled_norm = [0 if math.isnan(x) else x for x in abs_mag_obs]
            # normalize the fuction with x data['time_sampled'] and y abs_mag_obs and center it at the origin
            time_sampled_norm= data['time_sampled'] - np.mean(data['time_sampled'])
            # subrtract the max value of the mag to center it at the origin
            mag_sampled_norm = (-1)*(mag_sampled_norm - np.max(mag_sampled_norm))
            # normalize the mag so that the sum is 1
            # mag_sampled_norm = mag_sampled_norm/np.sum(mag_sampled_norm)
            mag_sampled_norm = mag_sampled_norm/np.max(mag_sampled_norm)

            # trasform abs_mag_obs[i] value 'numpy.float64' to int
            # abs_mag_obs = abs_mag_obs.astype(int)

            # create an array with the number the ammount of same number equal to the value of the mag
            mag_sampled_distr = []
            mag_sampled_array=np.asarray(mag_sampled_norm*1000, dtype = 'int')
            # i_pos=(-1)*np.round(len(abs_mag_obs)/2)
            for ii in range(len(abs_mag_obs)):
                # create an integer form the array mag_sampled_array[i] and round of the given value
                numbs=mag_sampled_array[ii]
                # invcrease the array number by the mag_sampled_distr numbs 
                array_nu=(np.ones(numbs+1)*time_sampled_norm[ii])#.astype(int)
                mag_sampled_distr=np.concatenate((mag_sampled_distr, array_nu))
                # i_pos=i_pos+1

            kurtosyness=kurtosis(mag_sampled_distr)
            skewness=skew(mag_sampled_distr)


            ##################################################################################################

            # Physical parameters
            mass = data['params']['m_init']['val']
            rho = data['params']['rho']['val']
            sigma = data['params']['sigma']['val']
            erosion_height_start = data['params']['erosion_height_start']['val']/1000
            erosion_coeff = data['params']['erosion_coeff']['val']
            erosion_mass_index = data['params']['erosion_mass_index']['val']
            erosion_mass_min = data['params']['erosion_mass_min']['val']
            erosion_mass_max = data['params']['erosion_mass_max']['val']

            # Compute the erosion range
            erosion_range = np.log10(erosion_mass_max) - np.log10(erosion_mass_min)

            cost_path = os.path.join(directory, name)

            # Load the constants
            const, _ = loadConstants(cost_path)
            const.dens_co = np.array(const.dens_co)

            # Compute the erosion energies
            erosion_energy_per_unit_cross_section, erosion_energy_per_unit_mass = wmpl.MetSim.MetSimErosion.energyReceivedBeforeErosion(const)

            ##################################################################################################


            # add a new line in dataframe
            df_json.loc[len(df_json)] = [name,shower_code, v0, vel_avg_norot, duration,\
            mass, peak_mag_height,begin_height, end_height, t0, peak_abs_mag, beg_abs_mag, end_abs_mag,\
            F, trail_len, acceleration_lin, acceleration_parab, acc_jacchia, decel_t0, zenith_angle, kurtosyness,skewness,\
            kc_par, Dynamic_pressure_peak_abs_mag,\
            a3, b3, c3, jac_a1, jac_a2, a3_Inabs, b3_Inabs, c3_Inabs, a3_Outabs, b3_Outabs, c3_Outabs, rho, sigma,\
            erosion_height_start, erosion_coeff, erosion_mass_index,\
            erosion_mass_min, erosion_mass_max, erosion_range,\
            erosion_energy_per_unit_cross_section, erosion_energy_per_unit_mass]

            There_is_data=True

    os.chdir('..')

    # delete the first line of the dataframe that is empty
    df_json = df_json.drop([0])

    # df_json=df_json[df_json['acceleration']>0]
    # df_json=df_json[df_json['acceleration']<100]
    # df_json=df_json[df_json['trail_len']<50]

    # save the dataframe in a csv file in the same folder of the code withouth the index
    # if save_it == True:
    #     df_json.to_csv(os.getcwd()+r'\Simulated_'+Shower+'.csv', index=False)

    f.close()

    if There_is_data==True:
        return df_json


def read_solution_table_json(Shower=''):
    '''
    It reads the solution_table.json file and extract the observable property from the EMCCD camera results
    The values are given in a dataframe format and if requestd are saved in .csv file called "Simulated_"+Shower+".csv"
    Keyword arguments:
    df_simul:       dataframe of the simulated shower
    Shower:         Shower name, by default there is no name.
    save_it:        Boolean - save the extracted dataframe in a .csv, by default it is not saved.
    '''

    # open the solution_table.json file
    f = open('solution_table.json',"r")
    data = json.loads(f.read())
    # create a dataframe to store the data
    df = pd.DataFrame(data, columns=['solution_id','shower_code','vel_init_norot','vel_avg_norot','vel_init_norot_err','beg_fov','end_fov','elevation_norot','duration','mass','begin_height','end_height','peak_abs_mag','beg_abs_mag','end_abs_mag','F'])

    df_shower_EMCCD = df.loc[(df.shower_code == Shower) & (df.beg_fov) & (df.end_fov) & (df.vel_init_norot_err < 2) & (df.begin_height > df.end_height) & (df.vel_init_norot > df.vel_avg_norot) &
    (df.elevation_norot >=0) & (df.elevation_norot <= 90) & (df.begin_height < 180) & (df.F > 0) & (df.F < 1) & (df.begin_height > 80) & (df.vel_init_norot < 75)]
    # delete the rows with NaN values
    df_shower_EMCCD = df_shower_EMCCD.dropna()


    # trail_len in km
    df_shower_EMCCD['trail_len'] = (df_shower_EMCCD['begin_height'] - df_shower_EMCCD['end_height'])/np.sin(np.radians(df_shower_EMCCD['elevation_norot']))
    # acceleration in km/s^2
    df_shower_EMCCD['acceleration'] = (df_shower_EMCCD['vel_init_norot'] - df_shower_EMCCD['vel_avg_norot'])/(df_shower_EMCCD['duration'])

    df_shower_EMCCD['zenith_angle'] = (90 - df_shower_EMCCD['elevation_norot'])

    acceleration=[]
    vel_init_norot=[]
    vel_avg_norot=[]
    abs_mag_pickl=[]
    height_pickl=[]

    begin_height=[]
    end_height=[]
    peak_mag_height=[]

    peak_abs_mag=[]
    beg_abs_mag=[]
    end_abs_mag=[]

    kurtosisness=[]
    skewness=[]

    jj=0
    for ii in range(len(df_shower_EMCCD)):

        # pick the ii element of the solution_id column 
        namefile=df_shower_EMCCD.iloc[ii]['solution_id']

        # split the namefile base on the '_' character and pick the first element
        folder=namefile.split('_')[0]

        traj = wmpl.Utils.Pickling.loadPickle("/home/mvovk/PCA/PER_pk/", namefile+".pylig.pickle")
        vel_pickl=[]
        time_pickl=[]
        abs_mag_pickl=[]
        height_pickl=[]
        for obs in traj.observations:
            # put it at the end obs.velocities[1:] at the end of vel_pickl list
            vel_pickl.extend(obs.velocities[1:])
            time_pickl.extend(obs.time_data[1:])
            abs_mag_pickl.extend(obs.absolute_magnitudes[1:])
            height_pickl.extend(obs.model_ht[1:])


        # compute the linear regression
        vel_pickl = [i/1000 for i in vel_pickl] # convert m/s to km/s
        time_pickl = [i for i in time_pickl]
        height_pickl = [i/1000 for i in height_pickl]
        abs_mag_pickl = [i for i in abs_mag_pickl]

        # fit a line to the throught the vel_sim and ht_sim
        a, b = np.polyfit(time_pickl,vel_pickl, 1)

        vel_sim_line=[a*x+b for x in time_pickl]

        # append the values to the list
        acceleration.append((-1)*a)
        vel_init_norot.append(vel_sim_line[0])
        vel_avg_norot.append(np.mean(vel_sim_line))

        begin_height.append(np.max(height_pickl))
        end_height.append(np.min(height_pickl))
        # find the peak of the absolute magnitude in the height_pickl list
        peak_mag_height.append(height_pickl[abs_mag_pickl.index(min(abs_mag_pickl))])

        peak_abs_mag.append(np.min(abs_mag_pickl))
        beg_abs_mag.append(abs_mag_pickl[0])
        end_abs_mag.append(abs_mag_pickl[-1])

        #####order the list by time
        vel_pickl = [x for _,x in sorted(zip(time_pickl,vel_pickl))]
        abs_mag_pickl = [x for _,x in sorted(zip(time_pickl,abs_mag_pickl))]
        height_pickl = [x for _,x in sorted(zip(time_pickl,height_pickl))]
        time_pickl = sorted(time_pickl)
        
        # # # find the index that that is a multiples of 0.031 s in time_pickl
        index = [i for i in range(len(time_pickl)) if time_pickl[i] % 0.031 < 0.01]
        # only use those index to create a new list
        time_pickl = [time_pickl[i] for i in index]
        abs_mag_pickl = [abs_mag_pickl[i] for i in index]
        height_pickl = [height_pickl[i] for i in index]
        vel_pickl = [vel_pickl[i] for i in index]

        # create a new array with the same values as time_pickl
        index=[]
        # if the distance between two index is smalle than 0.05 delete the second one
        for i in range(len(time_pickl)-1):
            if time_pickl[i+1]-time_pickl[i] < 0.01:
                # save the index as an array
                index.append(i+1)
        # delete the index from the list
        time_pickl = np.delete(time_pickl, index)
        abs_mag_pickl = np.delete(abs_mag_pickl, index)
        height_pickl = np.delete(height_pickl, index)
        vel_pickl = np.delete(vel_pickl, index)

        ############ KURT AND SKEW

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

        kurtosisness.append(kurtosis(mag_sampled_distr))
        skewness.append(skew(mag_sampled_distr))

        jj=jj+1
        print('Loading pickle file: ', namefile, ' n.', jj, ' of ', len(df_shower_EMCCD), ' done.')
        
    df_shower_EMCCD['acceleration'] = acceleration
    df_shower_EMCCD['vel_init_norot'] = vel_init_norot
    df_shower_EMCCD['vel_avg_norot'] = vel_avg_norot 

    df_shower_EMCCD['begin_height']=begin_height
    df_shower_EMCCD['end_height']=end_height
    df_shower_EMCCD['peak_mag_height']=peak_mag_height

    df_shower_EMCCD['kc'] = df_shower_EMCCD['begin_height'] + (2.86 - 2*np.log(df_shower_EMCCD['vel_init_norot']))/0.0612
    df_shower_EMCCD['kurtosis'] = kurtosisness
    df_shower_EMCCD['skew'] = skewness 

    df_shower_EMCCD_no_outliers = df_shower_EMCCD.loc[
    (df_shower_EMCCD.elevation_norot>25) &
    (df_shower_EMCCD.acceleration>0) &
    (df_shower_EMCCD.acceleration<100) &
    # (df_shower_EMCCD.vel_init_norot<np.percentile(df_shower_EMCCD['vel_init_norot'], 99)) & (df_shower_EMCCD.vel_init_norot>np.percentile(df_shower_EMCCD['vel_init_norot'], 1)) &
    # (df_shower_EMCCD.vel_avg_norot<np.percentile(df_shower_EMCCD['vel_avg_norot'], 99)) & (df_shower_EMCCD.vel_avg_norot>np.percentile(df_shower_EMCCD['vel_avg_norot'], 1)) &
    (df_shower_EMCCD.vel_init_norot<72) &
    (df_shower_EMCCD.trail_len<50)
    ]
    # print the number of droped observation
    print(len(df_shower_EMCCD)-len(df_shower_EMCCD_no_outliers),'number of droped ',Shower,' observation')

    # trail_len in km
    df_shower_EMCCD_no_outliers['trail_len'] = (df_shower_EMCCD_no_outliers['begin_height'] - df_shower_EMCCD_no_outliers['end_height'])/np.sin(np.radians(df_shower_EMCCD_no_outliers['elevation_norot']))
    # Zenith angle in radians
    df_shower_EMCCD_no_outliers['zenith_angle'] = (90 - df_shower_EMCCD_no_outliers['elevation_norot'])

    # delete the columns that are not needed
    df_shower_EMCCD_no_outliers = df_shower_EMCCD_no_outliers.drop(['vel_init_norot_err','beg_fov','end_fov','elevation_norot'], axis=1)

    # save the dataframe in a csv file in the same folder of the code withouth the index
    df_shower_EMCCD_no_outliers.to_csv(os.getcwd()+r'\\'+Shower+'.csv', index=False)

    f.close()

    return df_shower_EMCCD_no_outliers






# CODE ####################################################################################

def PCASim(OUT_PUT_PATH, Shower=['PER'], N_sho_sel=10000, No_var_PCA=[], INPUT_PATH=os.getcwd()):
    '''
    This function generate the simulated shower from the erosion model and apply PCA.
    The function read the json file in the folder and create a csv file with the simulated shower and take the data from GenerateSimulation.py folder.
    The function return the dataframe of the selected simulated shower.
    '''

    # the variable used in PCA are all = 'vel_init_norot','vel_avg_norot','duration','mass','begin_height','end_height','peak_abs_mag','beg_abs_mag','end_abs_mag','F','trail_len','acceleration','zenith_angle'
    variable_PCA=[] #'vel_init_norot','peak_abs_mag','begin_height','end_height','F','acceleration','duration'
    # variable_PCA=['vel_init_norot','peak_abs_mag','zenith_angle','peak_mag_height','acceleration','duration','vel_avg_norot','begin_height','end_height','beg_abs_mag','end_abs_mag','F','Dynamic_pressure_peak_abs_mag']
    # variable_PCA=['vel_init_norot','peak_abs_mag','zenith_angle','peak_mag_height','acceleration','duration','Dynamic_pressure_peak_abs_mag','kurtosis','skew','trail_len'] # perfect!
    # decel_after_knee_vel and height_knee_vel create errors in the PCA space  decel_after_knee_vel,height_knee_vel

    #No_var_PCA=['decel_after_knee_vel','height_knee_vel']
    No_var_PCA=['decel_after_knee_vel','height_knee_vel','acceleration_lin','a1_acc_jac','a2_acc_jac','a_acc','b_acc','c_acc','c_mag_init','c_mag_end','kc','acceleration_parab'] #,'acc_jacchia',acceleration_parab
    # if PC below 7 wrong
    
    # if variable_PCA is not empty
    if variable_PCA != []:
        # add to variable_PCA array 'shower_code','solution_id'
        variable_PCA = ['solution_id','shower_code'] + variable_PCA
        if No_var_PCA != []:
            # remove from variable_PCA the variables in No_var_PCA
            for var in No_var_PCA:
                variable_PCA.remove(var)

    else:
        # put in variable_PCA all the variables except mass
        variable_PCA = list(df_obs_shower.columns)
        variable_PCA.remove('mass')
        # if No_var_PCA is not empty
        if No_var_PCA != []:
            # remove from variable_PCA the variables in No_var_PCA
            for var in No_var_PCA:
                variable_PCA.remove(var)

    # keep only the variable_PCA variables
    df_all = pd.concat([df_sim_shower[variable_PCA],df_obs_shower[variable_PCA]], axis=0, ignore_index=True)


    # delete nan
    df_all = df_all.dropna()

    # Now we have all the data and we apply PCA to the dataframe

    df_all_nameless=df_all.drop(['shower_code','solution_id'], axis=1)
    # print the data columns names
    df_all_columns_names=(df_all_nameless.columns)
    # print the name of the variables used in PCA
    print('Variables used in PCA: ',df_all_nameless.columns)

    # Separating out the features
    scaled_df_all = df_all_nameless[df_all_columns_names].values

    # Standardizing the features
    scaled_df_all = StandardScaler().fit_transform(scaled_df_all)

    # print(scaled_df_all)

    pca = PCA()

    all_PCA = pca.fit_transform(scaled_df_all)


    # PLOT explained variance ratio #########################################

    # compute the explained variance ratio
    percent_variance = np.round(pca.explained_variance_ratio_* 100, decimals =2)
    print("explained variance ratio: \n",percent_variance)

    # name of the principal components
    columns_PC = ['PC' + str(x) for x in range(1, len(percent_variance)+1)]

    # plot the explained variance ratio of each principal componenets base on the number of column of the original dimension
    plt.bar(x= range(1,len(percent_variance)+1), height=percent_variance, tick_label=columns_PC, color='black')
    plt.ylabel('Percentance of Variance Explained')
    plt.xlabel('Principal Component')
    # plt.show()

    # PLOT the correlation coefficients #########################################

    # Compute the correlation coefficients
    cov_data = pca.components_.T

    # Plot the correlation matrix
    img = plt.matshow(cov_data.T, cmap=plt.cm.coolwarm, vmin=-1, vmax=1)
    plt.colorbar(img)
    rows=variable_PCA

    # Add the variable names as labels on the x-axis and y-axis
    plt.xticks(range(len(rows)-2), rows[2:], rotation=90)
    plt.yticks(range(len(columns_PC)), columns_PC)

    # plot the influence of each component on the original dimension
    for i in range(cov_data.shape[0]):
        for j in range(cov_data.shape[1]):
            plt.text(i, j, "{:.2f}".format(cov_data[i, j]), size=12, color='black', ha="center", va="center")   
    # plt.show()

    # PLOT the shorter PCA space ########################################

    # find the number of PC that explain 95% of the variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

    # recomute PCA with the number of PC that explain 95% of the variance
    pca= PCA(n_components=np.argmax(cumulative_variance >= PCA_percent/100) + 1)
    all_PCA = pca.fit_transform(scaled_df_all)

    # # select only the column with in columns_PC with the same number of n_components
    columns_PC = ['PC' + str(x) for x in range(1, pca.n_components_+1)]

    # create a dataframe with the PCA space
    df_all_PCA = pd.DataFrame(data = all_PCA, columns = columns_PC)
    
    print(str(len(percent_variance))+' PC = '+str(PCA_percent)+' of the variance explained by ',pca.n_components_,' PC')

    # check if can be refined
    number_of_deleted_PC=len(percent_variance)-pca.n_components_

    # print('Number of deleted PC: ', number_of_deleted_PC)



    # repeat the define_PCA_space in order to delete the PC that are not needed and stop when the number of PC is equal to the number of variable_PCA
    while number_of_deleted_PC>0:
        df_all_PCA, number_of_deleted_PC = refine_PCA_space(df_all_PCA)




    
    # add the shower code to the dataframe
    df_all_PCA['shower_code'] = df_all['shower_code'].values

    # delete the lines after len(df_sim_shower) to have only the simulated shower
    df_sim_PCA = df_all_PCA.drop(df_all_PCA.index[len(df_sim_shower):])
    df_obs_PCA = df_all_PCA.drop(df_all_PCA.index[:len(df_sim_shower)])

    # print('number of PC that explain 95% of the variance: ',pca.n_components_)

    # # plot all the data in the PCA space
    # sns.pairplot(df_obs_PCA, hue='shower_code', plot_kws={'alpha': 0.6, 's': 5, 'edgecolor': 'k'},corner=True)
    # plt.show()
    # # # sns.pairplot(df_obs_PCA, plot_kws={'alpha': 0.6, 's': 5, 'edgecolor': 'k', 'c':'palegreen'},corner=True)

    # sns.pairplot(df_sim_PCA, hue='shower_code', plot_kws={'alpha': 0.6, 's': 5, 'edgecolor': 'k'},corner=True)
    # plt.show()

    ##########################################
    # define the mean position and extract the n_selected meteor closest to the mean

    # find the mean base on the shower_code in the PCA space
    meanPCA = df_all_PCA.groupby('shower_code').mean()
    df_all_PCA['solution_id']=df_all['solution_id']
    # create a list with the selected meteor properties and PCA space
    df_sel_shower=[]
    df_sel_PCA=[]
    # i_shower_preced=0
    jj=0

    for current_shower in Shower:
        # find the mean of the simulated shower
        meanPCA_current = meanPCA.loc[(meanPCA.index == current_shower)]
        # take only the value of the mean of the first row
        meanPCA_current = meanPCA_current.values

        shower_current = df_obs_shower[df_obs_shower['shower_code']==current_shower]
        shower_current_PCA = df_obs_PCA[df_obs_PCA['shower_code']==current_shower]
        # trasform the dataframe in an array
        shower_current_PCA = shower_current_PCA.drop(['shower_code'], axis=1)
        shower_current_PCA = shower_current_PCA.values 

        df_sim_PCA_for_now = df_sim_PCA
        df_sim_PCA_for_now = df_sim_PCA_for_now.drop(['shower_code'], axis=1)
        df_sim_PCA_val = df_sim_PCA_for_now.values 

        for i_shower in range(len(shower_current)):
            distance_current = []
            for i_sim in range(len(df_sim_PCA_val)):                
                distance_current.append(scipy.spatial.distance.euclidean(df_sim_PCA_val[i_sim], shower_current_PCA[i_shower]))
            

            ############ Value ###############
            # print the ['solution_id'] of the element [i_shower]
            # print(shower_current['solution_id'][i_shower])
            # create an array with lenght equal to the number of simulations and set it to shower_current_PCA['solution_id'][i_shower]
            solution_id_dist = [shower_current['solution_id'][i_shower]]*len(df_sim_PCA_val)

            # give the same solution_id of the shower_current_PCA['solution_id'] of at the i_shower row in a new column solution_id_dist of the df_sim_shower create a slice long as the simulations
            df_sim_shower['solution_id_dist']=solution_id_dist

            df_sim_shower['distance_meteor']=distance_current

            # sort the distance and select the n_selected closest to the mean
            df_sim_shower_dis = df_sim_shower.sort_values(by=['distance_meteor'])
            # drop the index
            df_sim_shower_dis = df_sim_shower_dis.reset_index(drop=True)
        
            # create a dataframe with the selected simulated shower characteristics
            df_sim_selected = df_sim_shower_dis[:N_sho_sel]
            # delete the shower code
            df_sim_selected = df_sim_selected.drop(['shower_code'], axis=1)
            # add the shower code
            df_sim_selected['shower_code']= current_shower+'_sel'
            df_sel_shower.append(df_sim_selected)

            # sort the distance and select the n_selected closest to the mean
            df_sim_shower_dis = df_sim_shower.sort_values(by=['distance_meteor'])
            # drop the index
            df_sim_shower_dis = df_sim_shower_dis.reset_index(drop=True)

            ################ PCA ################

            df_sim_PCA_dist=df_sim_PCA
            df_sim_PCA_dist['distance_meteor']=distance_current

            # sort the distance and select the n_selected closest to the mean
            df_sim_PCA_dist = df_sim_PCA_dist.sort_values(by=['distance_meteor'])
            # drop the index
            df_sim_PCA_dist = df_sim_PCA_dist.reset_index(drop=True)
        
            # create a dataframe with the selected simulated shower characteristics
            df_sim_selected_PCA = df_sim_PCA_dist[:N_sho_sel]
            # delete the shower code
            df_sim_selected_PCA = df_sim_selected_PCA.drop(['shower_code'], axis=1)
            # add the shower code
            df_sim_selected_PCA['shower_code']= current_shower+'_sel'

            df_sel_PCA.append(df_sim_selected_PCA)

            # print the progress bar in percent that refresh every 10 iteration and delete the previous one
            if i_shower%10==0:
                print('Processing ',current_shower,' shower: ', round(i_shower/len(shower_current)*100,2),'%', end="\r")


            # print('.', end='', flush=True)


        print('Processing ',current_shower,' shower:  100  %      ', end="\r")

        # concatenate the list of the PC components to a dataframe
        df_sel_PCA = pd.concat(df_sel_PCA)

        # delete the distace column from df_sel_PCA
        df_sel_PCA = df_sel_PCA.drop(['distance_meteor'], axis=1)

        # delete the shower code column
        df_sim_PCA = df_sim_PCA.drop(['distance_meteor'], axis=1)

        # save the dataframe to a csv file withouth the index
        df_sel_PCA.to_csv(OUT_PUT_PATH+r'/Simulated_'+current_shower+'_select_PCA.csv', index=False)

        # concatenate the list of the properties to a dataframe
        df_sel_shower = pd.concat(df_sel_shower)

        df_sel_PCA_NEW = df_sel_PCA.drop(['shower_code'], axis=1)
        # create a list of the selected shower
        df_sel_PCA_NEW = df_sel_PCA_NEW.values
        distance_current = []
        #print(meanPCA_current[0])
        #print('and the df_sel_PCA_NEW')
        #print(df_sel_PCA_NEW[i_shower])
        for i_shower in range(len(df_sel_shower)):
            #distance_current.append(scipy.spatial.distance.euclidean(meanPCA_current, df_sel_PCA_NEW[i_shower]))
            #distance_current.append(scipy.spatial.distance.euclidean(meanPCA_current[0], df_sel_PCA_NEW[i_shower]))
            distance_current.append(scipy.spatial.distance.euclidean(meanPCA_current.ravel(), df_sel_PCA_NEW[i_shower].ravel()))
        df_sel_shower['distance']=distance_current # from the mean of the selected shower
        # save the dataframe to a csv file withouth the index
        df_sel_shower.to_csv(OUT_PUT_PATH+r'/Simulated_'+current_shower+'_select.csv', index=False)

        # save dist also on selected shower
        distance_current = []
        for i_shower in range(len(shower_current)):
            distance_current.append(scipy.spatial.distance.euclidean(meanPCA_current.ravel(), shower_current_PCA[i_shower].ravel()))
        shower_current['distance']=distance_current # from the mean of the selected shower
        shower_current.to_csv(OUT_PUT_PATH+r'/'+current_shower+'_and_dist.csv', index=False)

    # copy Simulated_PER.csv in OUT_PUT_PATH
    shutil.copyfile(INPUT_PATH+r'/Simulated_PER.csv', OUT_PUT_PATH+r'/Simulated_PER.csv')


    # PLOT the selected simulated shower ########################################

    # dataframe with the simulated and the selected meteors in the PCA space
    # df_sim_sel_PCA = pd.concat([df_sim_PCA,df_sel_PCA], axis=0)

    df_sim_sel_PCA = pd.concat([df_sim_PCA,df_sel_PCA,df_obs_PCA], axis=0)


    # sns.pairplot(df_sim_sel_PCA, hue='shower_code', plot_kws={'alpha': 0.6, 's': 5, 'edgecolor': 'k'},corner=True)
    # plt.show()

    # sns.pairplot(df_sel_PCA, hue='shower_code', plot_kws={'alpha': 0.6, 's': 5, 'edgecolor': 'k'},corner=True)
    # plt.show()


    # dataframe with the simulated and the selected meteors physical characteristics
    df_sim_sel_shower = pd.concat([df_sim_shower,df_sel_shower], axis=0)

    # sns.pairplot(df_sel_shower[['shower_code','rho','sigma','erosion_height_start','erosion_coeff','erosion_mass_index','erosion_mass_min','erosion_mass_max']], hue='shower_code', plot_kws={'alpha': 0.6, 's': 5, 'edgecolor': 'k'},corner=True)
    # plt.show()

    print('\nSUCCESS: the simulated shower have been selected')





def refine_PCA_space(df_all):
    '''
    from the simulated and observed shower dataframe it create a dataframe with the PCA space
    for the given variable_PCA and the one that are not in No_var_PCA
    if variable_PCA is empty it takes all except for mass
    '''

    pca = PCA()

    all_PCA = pca.fit_transform(df_all)

    # compute the explained variance ratio
    percent_variance = np.round(pca.explained_variance_ratio_* 100, decimals =2)
    print("explained variance ratio: \n",percent_variance)

    # name of the principal components
    columns_PC = ['PC' + str(x) for x in range(1, len(percent_variance)+1)]

    # find the number of PC that explain 95% of the variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

    # recomute PCA with the number of PC that explain 95% of the variance
    pca= PCA(n_components=np.argmax(cumulative_variance >= PCA_percent/100) + 1)
    all_PCA = pca.fit_transform(df_all)

    # # select only the column with in columns_PC with the same number of n_components
    columns_PC = ['PC' + str(x) for x in range(1, pca.n_components_+1)]

    # create a dataframe with the PCA space
    df_all_PCA = pd.DataFrame(data = all_PCA, columns = columns_PC)

    print(str(len(percent_variance))+' PC = '+str(PCA_percent)+' of the variance explained by ',pca.n_components_,' PC')

    number_of_deleted_PC=len(percent_variance)-pca.n_components_

    # print('Number of deleted PC: ', number_of_deleted_PC)


    return df_all_PCA, number_of_deleted_PC





if __name__ == "__main__":

    import argparse


    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Fom Observation and simulated data weselect the most likely through PCA, run it, and store results to disk.")

    arg_parser.add_argument('output_dir', metavar='OUTPUT_PATH', type=str, \
        help="Path to the output directory.")

    arg_parser.add_argument('shower', metavar='SHOWER', type=str, \
        help="Use specific shower from the given simulation.")
    
    arg_parser.add_argument('input_dir', metavar='INPUT_PATH', type=str, \
        help="Path were are store both simulated and observed shower .csv file.")

    arg_parser.add_argument('nsel', metavar='SEL_NUM', type=int, \
        help="Number of selected simulations to consider.")

    arg_parser.add_argument('--NoPCA', metavar='NOPCA', type=str, default=[], \
        help="Use specific variable not considered in PCA.")

    arg_parser.add_argument('--cores', metavar='CORES', type=int, default=None, \
        help="Number of cores to use. All by default.")

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################

    # Make the output directory
    mkdirP(cml_args.output_dir)

    # make only one shower
    Shower=[cml_args.shower]

    # # Init simulation parameters with the given class name
    # erosion_sim_params = SIM_CLASSES[SIM_CLASSES_NAMES.index(cml_args.simclass)]()

    # Set the folder where are the GenerateSimulations.py output json files e.g. "Simulations_"+Shower+""
    # the numbr of showers and folder must be the same
    folder_GenerateSimulations_json = ["Simulations_"+f+"" for f in Shower]

    # save all the simulated showers in a list
    df_sim_shower = []
    df_obs_shower = []
    # search for the simulated showers in the folder
    for current_shower in Shower:
        # check in the current folder there is a csv file with the name of the simulated shower
        if os.path.isfile(cml_args.input_dir+r'/Simulated_'+current_shower+'.csv'):
            # if yes read the csv file
            df_sim = pd.read_csv(cml_args.input_dir+r'/Simulated_'+current_shower+'.csv')
        else:
            # open the folder and extract all the json files
            os.chdir(folder_GenerateSimulations_json[Shower.index(current_shower)])
            directory=cml_args.input_dir
            extension = 'json'
            # all_jsonfiles = [i for i in glob.glob('*.{}'.format(extension))]
            
            all_jsonfiles = [i for i in glob.glob('**/*.{}'.format(extension), recursive=True)]
            print('Number of simulated files: ',len(all_jsonfiles))
            os.chdir('..')
                                    

            # # append the path to the json files
            # all_jsonfiles = [directory+'\\'+i for i in all_jsonfiles]
            # Generate simulations using multiprocessing
            input_list = [[folder_GenerateSimulations_json[Shower.index(current_shower)], current_shower, [all_jsonfiles[ii]]] for ii in range(len(all_jsonfiles))]
            results_list = domainParallelizer(input_list, read_GenerateSimulations_folder_output, cores=cml_args.cores)

            # if no read the json files in the folder and create a new csv file
            df_sim = pd.concat(results_list)

            # print(df_sim_shower)
            # reindex the dataframe
            df_sim.reset_index(drop=True, inplace=True)


            df_sim.to_csv(cml_args.input_dir+r'/Simulated_'+current_shower+'.csv', index=False)



        if os.path.isfile(cml_args.input_dir+r'/'+current_shower+'.csv'):
            # if yes read the csv file
            df_obs = pd.read_csv(cml_args.input_dir+r'/'+current_shower+'.csv')
        else:
            # if no read the solution_table.json file
            df_obs = read_solution_table_json(current_shower)

        # limit the simulation to the maximum abs magnitude of the observed shower

        # df_sim=df_sim[df_sim.peak_abs_mag>np.min(np.percentile(df_obs['peak_abs_mag'], 10) )]
        # df_sim=df_sim[df_sim.peak_abs_mag<np.max(np.percentile(df_obs['peak_abs_mag'], 90) )]

        # append the simulated shower to the list
        df_sim_shower.append(df_sim)

        # append the observed shower to the list
        df_obs_shower.append(df_obs)
        

    # concatenate all the simulated shower in a single dataframe
    df_sim_shower = pd.concat(df_sim_shower)
    # concatenate all the EMCCD observed showers in a single dataframe
    df_obs_shower = pd.concat(df_obs_shower)

    PCASim(cml_args.output_dir, Shower, cml_args.nsel, cml_args.NoPCA, cml_args.input_dir)