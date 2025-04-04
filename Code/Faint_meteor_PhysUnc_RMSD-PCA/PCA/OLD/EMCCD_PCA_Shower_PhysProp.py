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
import copy
# import multiprocessing
from numpy.linalg import inv
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
from wmpl.MetSim.GUI import loadConstants
import shutil
from scipy.stats import kurtosis, skew
from wmpl.Utils.OSTools import mkdirP
import math
from wmpl.Utils.PyDomainParallelizer import domainParallelizer
from scipy.linalg import svd
# from scipy.optimize import curve_fit # faster 
# from scipy.optimize import basinhopping # slower but more accurate
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
import scipy.stats as stats
import statsmodels.api as sm
from sklearn.preprocessing import PowerTransformer

add_json_noise = True

PCA_percent = 99

PCA_pairplot=True

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
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\
        0, 0,\
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,\
        0, 0, 0, 0, 0,\
        0, 0, 0,\
        0, 0]]

    # create a dataframe to store the data
    df_json = pd.DataFrame(dataList, columns=['solution_id','shower_code','vel_init_norot','vel_avg_norot','duration',\
    'mass','peak_mag_height','begin_height','end_height','t0','peak_abs_mag','beg_abs_mag','end_abs_mag',\
    'F','trail_len','deceleration_lin','deceleration_parab','decel_parab_t0','decel_t0','decel_jacchia','zenith_angle', 'kurtosis','skew',\
    'kc','Dynamic_pressure_peak_abs_mag',\
    'a_acc','b_acc','c_acc','a_t0', 'b_t0', 'c_t0','a1_acc_jac','a2_acc_jac','a_mag_init','b_mag_init','c_mag_init','a_mag_end','b_mag_end','c_mag_end',\
    'rho','sigma','erosion_height_start','erosion_coeff', 'erosion_mass_index',\
    'erosion_mass_min','erosion_mass_max','erosion_range',\
    'erosion_energy_per_unit_cross_section', 'erosion_energy_per_unit_mass'])

    There_is_data=False
    
    pickle_flag=False
    # open the all file and extract the data
    for i in range(len(all_jsonfiles)):
    # for i in range(len(all_jsonfiles[1:100])):
        f = open(all_jsonfiles[i],"r")
        data = json.loads(f.read())

        # show the current processed file and the number of files left to process
        # print(all_jsonfiles[i]+' - '+str(len(all_jsonfiles)-i)+' left')
        print(all_jsonfiles[i])
        # check if the json file name has _sim_fit.json
        if '_sim_fit.json' in all_jsonfiles[i]:
            # from 'params' extract the observable parameters and save them in a list
            pickle_df=read_manual_pikle(all_jsonfiles[i],directory,Shower)
            pickle_flag=True

        elif data['ht_sampled']!= None: 
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
                
                # # spline and smooth the data
                # spline_mag = UnivariateSpline(obs_time, abs_mag_obs, s=0.1)  # 's' is the smoothing factor
                # abs_mag_obs=spline_mag(obs_time)

                # obs_vel=[v0]
                obs_length=[x/1000 for x in obs_length]

                vel_sim=[vel_sim[jj_index_cut] for jj_index_cut in closest_indices]
                # create a list of the same length of obs_time with the value of the first element of vel_sim
                obs_vel=vel_sim

                for vel_ii in range(1,len(obs_time)):
                    if obs_time[vel_ii]-obs_time[vel_ii-1]<0.03125:
                    # if obs_time[vel_ii] % 0.03125 < 0.000000001:
                        if vel_ii+1<len(obs_length):
                            obs_vel[vel_ii+1]=(obs_length[vel_ii+1]-obs_length[vel_ii-1])/(obs_time[vel_ii+1]-obs_time[vel_ii-1])
                    else:
                        obs_vel[vel_ii]=(obs_length[vel_ii]-obs_length[vel_ii-1])/(obs_time[vel_ii]-obs_time[vel_ii-1])

                # # append from vel_sampled the rest by the difference of the first element of obs_length divided by the first element of obs_time
                # rest_vel_sampled=[(obs_length[vel_ii]-obs_length[vel_ii-1])/(obs_time[vel_ii]-obs_time[vel_ii-1]) for vel_ii in range(1,len(obs_length))]
                # # append the rest_vel_sampled to vel_sampled
                # obs_vel.extend(rest_vel_sampled)

                
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
            p0 = [a, 0, 0, t0]

            opt_res = opt.minimize(lag_residual, p0, args=(np.array(obs_time), np.array(obs_lag)), method='Nelder-Mead')

            # sample the fit for the velocity and acceleration
            a_t0, b_t0, c_t0, t0 = opt_res.x

            # compute reference decelearation
            t_decel_ref = (t0 + np.max(obs_time))/2
            decel_t0 = cubic_acceleration(t_decel_ref, a_t0, b_t0, t0)[0]

            a_t0=-abs(a_t0)
            b_t0=-abs(b_t0)

            acceleration_parab_t0=a_t0*6 + b_t0*2

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

            acc_jacchia = abs(jac_a1)*abs(jac_a2)**2

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


            ######## SKEW KURT ################ 
            # create a new array with the same values as time_pickl
            index=[]
            # if the distance between two index is smalle than 0.05 delete the second one
            for i in range(len(obs_time)-1):
                if obs_time[i+1]-obs_time[i] < 0.01:
                    # save the index as an array
                    index.append(i+1)
            # delete the index from the list
            time_pickl = np.delete(obs_time, index)
            abs_mag_pickl = np.delete(abs_mag_obs, index)

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

            # # # find the index of the first element of the simulation that is equal to the first element of the observation
            # mag_sampled_norm = [0 if math.isnan(x) else x for x in abs_mag_obs]
            # # normalize the fuction with x data['time_sampled'] and y abs_mag_obs and center it at the origin
            # time_sampled_norm= data['time_sampled'] - np.mean(data['time_sampled'])
            # # subrtract the max value of the mag to center it at the origin
            # mag_sampled_norm = (-1)*(mag_sampled_norm - np.max(mag_sampled_norm))
            # # normalize the mag so that the sum is 1
            # # mag_sampled_norm = mag_sampled_norm/np.sum(mag_sampled_norm)
            # mag_sampled_norm = mag_sampled_norm/np.max(mag_sampled_norm)

            # # trasform abs_mag_obs[i] value 'numpy.float64' to int
            # # abs_mag_obs = abs_mag_obs.astype(int)

            # # create an array with the number the ammount of same number equal to the value of the mag
            # mag_sampled_distr = []
            # mag_sampled_array=np.asarray(mag_sampled_norm*1000, dtype = 'int')
            # # i_pos=(-1)*np.round(len(abs_mag_obs)/2)
            # for ii in range(len(abs_mag_obs)):
            #     # create an integer form the array mag_sampled_array[i] and round of the given value
            #     numbs=mag_sampled_array[ii]
            #     # invcrease the array number by the mag_sampled_distr numbs 
            #     array_nu=(np.ones(numbs+1)*time_sampled_norm[ii])#.astype(int)
            #     mag_sampled_distr=np.concatenate((mag_sampled_distr, array_nu))
            #     # i_pos=i_pos+1

            # kurtosyness=kurtosis(mag_sampled_distr)
            # skewness=skew(mag_sampled_distr)


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
            F, trail_len, acceleration_lin, acceleration_parab, acceleration_parab_t0, decel_t0, acc_jacchia, zenith_angle, kurtosyness,skewness,\
            kc_par, Dynamic_pressure_peak_abs_mag,\
            a3, b3, c3, a_t0, b_t0, c_t0, jac_a1, jac_a2, a3_Inabs, b3_Inabs, c3_Inabs, a3_Outabs, b3_Outabs, c3_Outabs, rho, sigma,\
            erosion_height_start, erosion_coeff, erosion_mass_index,\
            erosion_mass_min, erosion_mass_max, erosion_range,\
            erosion_energy_per_unit_cross_section, erosion_energy_per_unit_mass]

            There_is_data=True

    os.chdir('..')

    # delete the first line of the dataframe that is empty
    df_json = df_json.drop([0])

    if pickle_flag==True:
        # merge the two dataframes
        print('Merging the two dataframes')
        df_json = pd.concat([pickle_df,df_json], axis=0, ignore_index=True)
        # df_json = df_json.append(pickle_df, ignore_index=True)
        # print(df_json)
        
        There_is_data=True

    f.close()

    if There_is_data==True:
        return df_json


def read_manual_pikle(ID,INPUT_PATH,Shower=''):

    Shower=Shower
    # keep the first 14 characters of the ID
    name=ID[:15]
    name_file_json=ID
    name_file=name+'_trajectory.pickle'
    print('Loading pickle file: ', INPUT_PATH+name)
    # check if there are any pickle files in the 
    find_flag=False
    if os.path.isfile(os.path.join(INPUT_PATH, name_file)):
        find_flag=True
        # load the pickle file
        traj = wmpl.Utils.Pickling.loadPickle(INPUT_PATH,name_file)    
        jd_dat=traj.jdt_ref
        # print(os.path.join(root, name_file))

        vel_pickl=[]
        time_pickl=[]
        time_total=[]
        abs_mag_pickl=[]
        abs_total=[]
        height_pickl=[]
        height_total=[]
        lag=[]
        lag_total=[]
        elev_angle_pickl=[]
        elg_pickl=[]
        tav_pickl=[]
        
        lat_dat=[]
        lon_dat=[]
        # create a list to store the values of the pickle file
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

            elif jj==2:
                elg_pickl=obs.velocities[1:int(len(obs.velocities)/4)]
                if len(elg_pickl)==0:
                    elg_pickl=obs.velocities[1:2]

            # put it at the end obs.velocities[1:] at the end of vel_pickl list
            vel_pickl.extend(obs.velocities[1:])
            time_pickl.extend(obs.time_data[1:])
            time_total.extend(obs.time_data)
            abs_mag_pickl.extend(obs.absolute_magnitudes[1:])
            abs_total.extend(obs.absolute_magnitudes)
            height_pickl.extend(obs.model_ht[1:])
            height_total.extend(obs.model_ht)
            lag.extend(obs.lag[1:])
            lag_total.extend(obs.lag)
            elev_angle_pickl.extend(obs.elev_data)
            # length_pickl=len(obs.state_vect_dist[1:])
            
            lat_dat=obs.lat
            lon_dat=obs.lon

        # compute the linear regression
        vel_pickl = [i/1000 for i in vel_pickl] # convert m/s to km/s
        time_pickl = [i for i in time_pickl]
        time_total = [i for i in time_total]
        height_pickl = [i/1000 for i in height_pickl]
        height_total = [i/1000 for i in height_total]
        abs_mag_pickl = [i for i in abs_mag_pickl]
        abs_total = [i for i in abs_total]
        lag=[i/1000 for i in lag]
        lag_total=[i/1000 for i in lag_total]

        # print('length_pickl', length_pickl)
        # length_pickl = [i/1000 for i in length_pickl]


        # find the height when the velocity start dropping from the initial value 
        vel_init_mean = (np.mean(elg_pickl)+np.mean(tav_pickl))/2/1000
        v0=vel_init_mean
        vel_init_norot=vel_init_mean
        # print('mean_vel_init', vel_init_mean)

        # fit a line to the throught the vel_sim and ht_sim
        a, b = np.polyfit(time_pickl,vel_pickl, 1)
        trendLAG, bLAG = np.polyfit(time_pickl,lag, 1)

        vel_sim_line=[a*x+b for x in time_pickl]

        lag_line = [trendLAG*x+bLAG for x in time_pickl] 

        #####order the list by time
        vel_pickl = [x for _,x in sorted(zip(time_pickl,vel_pickl))]
        abs_mag_pickl = [x for _,x in sorted(zip(time_pickl,abs_mag_pickl))]
        abs_total = [x for _,x in sorted(zip(time_total,abs_total))]
        height_pickl = [x for _,x in sorted(zip(time_pickl,height_pickl))]
        height_total = [x for _,x in sorted(zip(time_total,height_total))]
        lag = [x for _,x in sorted(zip(time_pickl,lag))]
        lag_total = [x for _,x in sorted(zip(time_total,lag_total))]
        # length_pickl = [x for _,x in sorted(zip(time_pickl,length_pickl))]
        time_pickl = sorted(time_pickl)
        time_total = sorted(time_total)

        #######################################################
        # fit a line to the throught the vel_sim and ht_sim
        a3, b3, c3 = np.polyfit(time_pickl,vel_pickl, 2)

        t0 = np.mean(time_pickl)

        # initial guess of deceleration decel equal to linear fit of velocity
        p0 = [a, 0, 0, t0]
        # print(lag)
        # lag_calc = length_pickl-(v0*np.array(time_pickl)+length_pickl[0])
        opt_res = opt.minimize(lag_residual, p0, args=(np.array(time_pickl), np.array(lag)), method='Nelder-Mead')

        # sample the fit for the velocity and acceleration
        a_t0, b_t0, c_t0, t0 = opt_res.x

        # compute reference decelearation
        t_decel_ref = (t0 + np.max(time_pickl))/2
        decel_t0 = cubic_acceleration(t_decel_ref, a_t0, b_t0, t0)[0]

        a_t0=-abs(a_t0)
        b_t0=-abs(b_t0)

        acceleration_parab_t0=a_t0*6 + b_t0*2

        # Assuming the jacchiaVel function is defined as:
        def jacchiaVel(t, a1, a2, v_init):
            return v_init - np.abs(a1) * np.abs(a2) * np.exp(np.abs(a2) * t)

        # Generating synthetic observed data for demonstration
        t_observed = np.array(time_pickl)  # Observed times
        vel_observed = np.array(vel_pickl)  # Observed velocities

        # Residuals function for optimization
        def residuals(params):
            a1, a2 = params
            predicted_velocity = jacchiaVel(t_observed, a1, a2, v0)
            return np.sum((vel_observed - predicted_velocity)**2)

        # Initial guess for a1 and a2
        initial_guess = [0.005,	10]

        # Apply minimize to the residuals
        result = minimize(residuals, initial_guess)

        # Results
        jac_a1, jac_a2 = abs(result.x)

        acc_jacchia = abs(jac_a1)*abs(jac_a2)**2

        # only use first index to pick the height
        a3_Inabs, b3_Inabs, c3_Inabs = np.polyfit(height_total[:np.argmin(abs_total)],abs_total[:np.argmin(abs_total)], 2)
        #
        a3_Outabs, b3_Outabs, c3_Outabs = np.polyfit(height_total[np.argmin(abs_total):],abs_total[np.argmin(abs_total):], 2)

        # append the values to the list
        acceleration_parab=(a3*2+b3)
        acceleration_lin=(a)
        # v0=(vel_sim_line[0])
        # v0.append(vel_sim_line[0])
        vel_init_noro=vel_init_mean
        # print('mean_vel_init', vel_sim_line[0])
        vel_avg_norot=np.mean(vel_pickl) #trail_len / duration
        peak_mag_vel=(vel_pickl[np.argmin(abs_mag_pickl)])   

        begin_height=(height_total[0])
        end_height=(height_total[-1])
        peak_mag_height=(height_total[np.argmin(abs_total)])

        peak_abs_mag=(np.min(abs_total))
        beg_abs_mag=(abs_total[0])
        end_abs_mag=(abs_total[-1])

        lag_fin=(lag_line[-1])
        lag_init=(lag_line[0])
        lag_avg=(np.mean(lag_line))

        duration=(time_total[-1]-time_total[0])

        kc_par=(height_total[0] + (2.86 - 2*np.log(vel_sim_line[0]))/0.0612)

        F=((height_total[0] - height_total[np.argmin(abs_total)]) / (height_total[0] - height_total[-1]))

        zenith_angle=(90 - elev_angle_pickl[0]*180/np.pi)
        trail_len=((height_total[0] - height_total[-1])/(np.sin(np.radians(elev_angle_pickl[0]*180/np.pi))))
        

        Dynamic_pressure_peak_abs_mag=(wmpl.Utils.Physics.dynamicPressure(lat_dat, lon_dat, height_total[np.argmin(abs_total)]*1000, jd_dat, vel_pickl[np.argmin(abs_mag_pickl)]*1000))


        # check if in os.path.join(root, name_file) present and then open the .json file with the same name as the pickle file with in stead of _trajectory.pickle it has _sim_fit_latest.json
        if os.path.isfile(os.path.join(INPUT_PATH, name_file_json)):
            with open(os.path.join(INPUT_PATH, name_file_json),'r') as json_file: # 20210813_061453_sim_fit.json
                print('Loading json file: ', os.path.join(INPUT_PATH, name_file_json))
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

                cost_path = os.path.join(INPUT_PATH, name_file_json)

                # Load the constants
                const, _ = loadConstants(cost_path)
                const.dens_co = np.array(const.dens_co)

                # Compute the erosion energies
                erosion_energy_per_unit_cross_section, erosion_energy_per_unit_mass = wmpl.MetSim.MetSimErosion.energyReceivedBeforeErosion(const)
                erosion_energy_per_unit_cross_section_arr=(erosion_energy_per_unit_cross_section)
                erosion_energy_per_unit_mass_arr=(erosion_energy_per_unit_mass)


        else:
            print('No json file'+os.path.join(INPUT_PATH, name_file_json))

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




        shower_code_sim=('sim_'+Shower)
        

        ########################################## SKEWNESS AND kurtosyness ##########################################
        
        # # # find the index that that is a multiples of 0.031 s in time_pickl
        index = [i for i in range(len(time_pickl)) if time_pickl[i] % 0.031 < 0.01]
        # only use those index to create a new list
        time_pickl = [time_total[i] for i in index]
        abs_mag_pickl = [abs_total[i] for i in index]
        height_pickl = [height_total[i] for i in index]
        # vel_pickl = [vel_pickl[i] for i in index]

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
        # vel_pickl = np.delete(vel_pickl, index)


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

        kurtosyness=(kurtosis(mag_sampled_distr))
        skewness=(skew(mag_sampled_distr))

        # Data to populate the dataframe
        data_picklefile_pd = {
            'solution_id': [name],
            'shower_code': [shower_code_sim],
            'vel_init_norot': [vel_init_norot],
            'vel_avg_norot': [vel_avg_norot],
            'duration': [duration],
            'mass': [mass],
            'peak_mag_height': [peak_mag_height],
            'begin_height': [begin_height],
            'end_height': [end_height],
            't0': [t0],
            'peak_abs_mag': [peak_abs_mag],
            'beg_abs_mag': [beg_abs_mag],
            'end_abs_mag': [end_abs_mag],
            'F': [F],
            'trail_len': [trail_len],
            'deceleration_lin': [acceleration_lin],
            'deceleration_parab': [acceleration_parab],
            'decel_parab_t0': [acceleration_parab_t0],
            'decel_t0': [decel_t0],
            'decel_jacchia': [acc_jacchia],
            'zenith_angle': [zenith_angle],
            'kurtosis': [kurtosyness],
            'skew': [skewness],
            'kc': [kc_par],
            'Dynamic_pressure_peak_abs_mag': [Dynamic_pressure_peak_abs_mag],
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
            'rho': [rho],
            'sigma': [sigma],
            'erosion_height_start': [erosion_height_start],
            'erosion_coeff': [erosion_coeff],
            'erosion_mass_index': [erosion_mass_index],
            'erosion_mass_min': [erosion_mass_min],
            'erosion_mass_max': [erosion_mass_max],
            'erosion_range': [erosion_range],
            'erosion_energy_per_unit_cross_section': [erosion_energy_per_unit_cross_section_arr],
            'erosion_energy_per_unit_mass': [erosion_energy_per_unit_mass_arr]
        }

        # Create the dataframe
        infov_sim = pd.DataFrame(data_picklefile_pd)

    if find_flag==True:
        # print(infov_sim)
        return infov_sim
    else:
        # raise an error if the pickle file is not found
        print('No pickle file found')
        # raise an error
        raise ValueError('No pickle file found')
        return None


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

        traj = wmpl.Utils.Pickling.loadPickle("M:\\emccd\\pylig\\trajectory\\"+folder+"\\", namefile+".pylig.pickle")
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
    df_shower_EMCCD_no_outliers.to_csv(os.getcwd()+os.sep+Shower+'.csv', index=False)

    f.close()

    return df_shower_EMCCD_no_outliers

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

def mahalanobis_distance(x, mean, cov_inv):
    diff = x - mean
    return np.sqrt(np.dot(np.dot(diff, cov_inv), diff.T))

# # Function to compute modified Mahalanobis distance
# def modified_mahalanobis_distance(x, y, cov_inv):
#     diff = x - y
#     return np.sqrt(np.dot(diff, np.dot(cov_inv, diff)))

def check_normality(data, title):
    print(f"\nNormality check for {title}:")

    # # Q-Q Plot
    # sm.qqplot(data, line='45')
    # plt.title(f'Q-Q Plot for {title}')
    # plt.show()

    # Histogram and Density Plot
    # plt.hist(data, bins=30, density=True, alpha=0.6, color='g')
    # xmin, xmax = plt.xlim()
    # x = np.linspace(xmin, xmax, 100)
    # p = stats.norm.pdf(x, np.mean(data), np.std(data))
    # plt.plot(x, p, 'k', linewidth=2)
    # plt.title(f'Histogram with Density Plot for {title}')
    # plt.show()

    # Shapiro-Wilk Test
    shapiro_test = stats.shapiro(data)
    print("Shapiro-Wilk Test:", shapiro_test.statistic, shapiro_test.pvalue)

    # Anderson-Darling Test
    # ad_test = stats.anderson(data, dist='norm')
    # print("Anderson-Darling Test:", ad_test)

    # # Skewness and Kurtosis
    # skewness = stats.skew(data)
    # kurtosis = stats.kurtosis(data)
    # print("Skewness:", skewness)
    # print("Kurtosis:", kurtosis)

    return shapiro_test.pvalue

def transform_to_gaussian(data):
    # Compute the ECDF of the original data
    sorted_data = np.sort(data)
    ecdf = np.arange(1, len(sorted_data)+1) / len(sorted_data)
    
    # Transform to standard normal space
    standard_normal = np.random.normal(0, 1, size=len(data))
    standard_normal_sorted = np.sort(standard_normal)
    
    # Interpolate to match the ECDF
    resampled_data = np.interp(ecdf, np.linspace(0, 1, len(data)), standard_normal_sorted)
    
    # Scale and shift to match original mean and standard deviation
    mean = np.mean(data)
    std_dev = np.std(data)
    resampled_data = resampled_data * std_dev + mean
    return resampled_data
# CODE ####################################################################################

def PCASim(df_sim_shower,df_obs_shower, OUT_PUT_PATH, Shower=['PER'], N_sho_sel=10000, No_var_PCA=[], INPUT_PATH=os.getcwd()):
    '''
    This function generate the simulated shower from the erosion model and apply PCA.
    The function read the json file in the folder and create a csv file with the simulated shower and take the data from GenerateSimulation.py folder.
    The function return the dataframe of the selected simulated shower.
    '''
    # the variable used in PCA are all = 'vel_init_norot','vel_avg_norot','duration','mass','begin_height','end_height','peak_abs_mag','beg_abs_mag','end_abs_mag','F','trail_len','acceleration','zenith_angle'
    # vel_init_norot	vel_avg_norot	duration	mass	peak_mag_height	begin_height	end_height	height_knee_vel	peak_abs_mag	beg_abs_mag	end_abs_mag	F	trail_len	acceleration	decel_after_knee_vel	zenith_angle	kurtosis	skew	kc	Dynamic_pressure_peak_abs_mag
    variable_PCA=[] #'vel_init_norot','peak_abs_mag','begin_height','end_height','F','acceleration','duration'
    # variable_PCA=['vel_init_norot','vel_avg_norot','duration','begin_height','peak_mag_height','end_height','beg_abs_mag','peak_abs_mag','end_abs_mag','F','zenith_angle','t0'] #'vel_init_norot','peak_abs_mag','begin_height','end_height','F','acceleration','duration'
    # variable_PCA=['vel_init_norot','vel_avg_norot','peak_abs_mag','begin_height','peak_mag_height','F','duration','decel_parab_t0'] # perfect!
    # variable_PCA=['vel_init_norot','peak_abs_mag','zenith_angle','peak_mag_height','acceleration','duration','vel_avg_norot','begin_height','end_height','beg_abs_mag','end_abs_mag','F','Dynamic_pressure_peak_abs_mag']
    # variable_PCA=['vel_init_norot','peak_abs_mag','zenith_angle','peak_mag_height','acceleration','duration','Dynamic_pressure_peak_abs_mag'] # perfect!
    # variable_PCA=['vel_init_norot','peak_abs_mag','zenith_angle','peak_mag_height','acceleration','duration','Dynamic_pressure_peak_abs_mag','kurtosis','skew','trail_len']
    # decel_after_knee_vel and height_knee_vel create errors in the PCA space  decel_after_knee_vel,height_knee_vel

# ['solution_id','shower_code','vel_init_norot','vel_avg_norot','duration',\
#     'mass','peak_mag_height','begin_height','end_height','t0','peak_abs_mag','beg_abs_mag','end_abs_mag',\
#     'F','trail_len','deceleration_lin','deceleration_parab','decel_jacchia','decel_t0','zenith_angle', 'kurtosis','skew',\
#     'kc','Dynamic_pressure_peak_abs_mag',\
#     'a_acc','b_acc','c_acc','a1_acc_jac','a2_acc_jac','a_mag_init','b_mag_init','c_mag_init','a_mag_end','b_mag_end','c_mag_end',\
#     'rho','sigma','erosion_height_start','erosion_coeff', 'erosion_mass_index',\
#     'erosion_mass_min','erosion_mass_max','erosion_range',\
#     'erosion_energy_per_unit_cross_section', 'erosion_energy_per_unit_mass']
    No_var_PCA=[]
    No_var_PCA=['skew','a1_acc_jac','a2_acc_jac','a_acc','b_acc','c_acc','c_mag_init','c_mag_end','a_t0', 'b_t0', 'c_t0'] #,deceleration_lin','deceleration_parab','decel_jacchia','decel_t0' decel_parab_t0
    # No_var_PCA=['t0','deceleration_lin','kc','decel_jacchia','deceleration_parab','a1_acc_jac','a2_acc_jac','a_acc','b_acc','c_acc','a_t0', 'b_t0', 'c_t0', 'kurtosis','skew','beg_abs_mag', 'a_mag_init','b_mag_init','c_mag_init','a_mag_end','b_mag_end','c_mag_end'] #,deceleration_lin','deceleration_parab','decel_jacchia','decel_t0' decel_parab_t0

    # # if PC below 7 wrong
    # No_var_PCA=['duration','vel_avg_norot','t0','peak_abs_mag','deceleration_lin','decel_jacchia','deceleration_parab','a1_acc_jac','a2_acc_jac','a_acc','b_acc','c_acc', 'decel_t0','trail_len','vel_init_norot','zenith_angle','F','Dynamic_pressure_peak_abs_mag','beg_abs_mag','end_abs_mag'] #,deceleration_lin','deceleration_parab','decel_jacchia','decel_t0' decel_parab_t0
    # No_var_PCA=['t0','deceleration_lin','kc','decel_jacchia','deceleration_parab','a1_acc_jac','a2_acc_jac','a_acc','b_acc','c_acc','c_mag_init','c_mag_end','a_t0', 'b_t0', 'c_t0', 'decel_t0','peak_mag_height','F','Dynamic_pressure_peak_abs_mag','beg_abs_mag','end_abs_mag'] #,deceleration_lin','deceleration_parab','decel_jacchia','decel_t0' decel_parab_t0

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

    scaled_sim=df_sim_shower[variable_PCA].copy()
    scaled_sim=scaled_sim.drop(['shower_code','solution_id'], axis=1)
    # Standardize each column separately
    scaler = StandardScaler()
    df_sim_var_sel_standardized = scaler.fit_transform(scaled_sim)
    df_sim_var_sel_standardized = pd.DataFrame(df_sim_var_sel_standardized, columns=scaled_sim.columns)

    # Identify outliers using Z-score method on standardized data
    z_scores = np.abs(zscore(df_sim_var_sel_standardized))
    threshold = 3
    outliers = (z_scores > threshold).any(axis=1)

    # Assign df_sim_shower to the version without outliers
    df_sim_shower = df_sim_shower[~outliers].copy()

    if PCA_pairplot:
        # scale the data so to be easily plot against each other with the same scale
        df_sim_var_sel = df_sim_shower[variable_PCA].copy()
        df_sim_var_sel = df_sim_var_sel.drop(['shower_code','solution_id'], axis=1)

        if len(df_sim_var_sel)>10000:
            # pick randomly 10000 events
            print('Number of events in the simulated shower:',len(df_sim_var_sel))
            df_sim_var_sel=df_sim_var_sel.sample(n=10000)

        df_sim_var_sel_standardized = df_sim_var_sel
        # # # Standardize each column separately
        # # scaler = StandardScaler()
        # # df_sim_var_sel_standardized = scaler.fit_transform(df_sim_var_sel)
        # # # put in a dataframe
        # # df_sim_var_sel_standardized = pd.DataFrame(df_sim_var_sel_standardized, columns=df_sim_var_sel.columns)
        # print('Number of events in the simulated shower after the selection:',len(df_sim_var_sel))
        # # sns plot of the df_sim_var_sel and df_sim_var_sel_no_outliers hue='shower_code'
        # sns.pairplot(df_sim_var_sel_standardized)
        # print('Pairplot of the simulated shower')
        # # save the figure
        # plt.savefig(OUT_PUT_PATH+os.sep+'var_sns_'+Shower[0]+'_select_PCA.png')
        # # close the figure
        # plt.close()




        # # loop all pphysical variables
        # physical_vars = ['mass','rho','sigma','erosion_height_start','erosion_coeff','erosion_mass_index','erosion_mass_min','erosion_mass_max']
        # for var_phys in physical_vars:
        #     # make a subplot of the rho againist each variable_PCA as a scatter plot
        #     fig, axs = plt.subplots(3, 4, figsize=(20, 15))
        #     # flatten the axs array
        #     axs = axs.flatten()
        #     for i, var in enumerate(variable_PCA[2:]):
        #         # plot the rho againist the variable with black borders
        #         axs[i].scatter(df_sim_shower[var], df_sim_shower[var_phys], c='b') #, edgecolors='k', alpha=0.5
        #         # put a green vertical line for the df_obs_shower[var] value
        #         axs[i].axvline(df_obs_shower[var].values[0], color='limegreen', linestyle='--', linewidth=5)
        #         # put a horizontal line for the rho of the first df_sim_shower
        #         axs[i].axhline(df_sim_shower[var_phys].values[0], color='k', linestyle='-', linewidth=2)
        #         axs[i].set_title(var)
        #         # as a suptitle put the variable_PCA
        #         fig.suptitle(var_phys)
        #         # grid
        #         axs[i].grid()
        #         # make y axis log if the variable is 'erosion_mass_min' 'erosion_mass_max'
        #         if var_phys == 'erosion_mass_min' or var_phys == 'erosion_mass_max':
        #             axs[i].set_yscale('log')

        #     # save the figure
        #     plt.savefig(OUT_PUT_PATH+os.sep+var_phys+'_vs_var_'+Shower[0]+'_select_PCA.png')
        #     # close the figure
        #     plt.close()


        # # make a subplot of the distribution of the variables
        # fig, axs = plt.subplots(3, 4, figsize=(20, 15))
        # for i, var in enumerate(variable_PCA[2:]):
        #     # plot the distribution of the variable
        #     sns.histplot(df_sim_var_sel[var], kde=True, ax=axs[i//4, i%4], color='b', alpha=0.5)
        #     # axs[i//4, i%4].set_title('Distribution of '+var)
        #     # put a vertical line for the df_obs_shower[var] value
        #     axs[i//4, i%4].axvline(df_obs_shower[var].values[0], color='limegreen', linestyle='--', linewidth=5)
        #     # grid
        #     axs[i//4, i%4].grid()
            
        # # save the figure
        # plt.savefig(OUT_PUT_PATH+os.sep+'var_hist_'+Shower[0]+'_select_PCA.png')
        # # close the figure
        # plt.close()
        



    ##################################### delete var that are not in the 5 and 95 percentile of the simulated shower #####################################

    df_all = pd.concat([df_sim_shower[variable_PCA],df_obs_shower[variable_PCA]], axis=0, ignore_index=True)
    # delete nan
    df_all = df_all.dropna()

    # create a copy of df_sim_shower for the resampling
    df_sim_shower_resample=df_sim_shower.copy()
    df_obs_shower_resample=df_obs_shower.copy()
    No_var_PCA_perc=[]
    # check that all the df_obs_shower for variable_PCA is within th 5 and 95 percentie of df_sim_shower of variable_PCA
    for var in variable_PCA:
        if var != 'shower_code' and var != 'solution_id':
            # check if the variable is in the df_obs_shower
            if var in df_obs_shower.columns:
                # check if the variable is in the df_sim_shower
                if var in df_sim_shower.columns:

                    ii_all=0
                    for i_var in range(len(df_obs_shower[var])):
                        # # check if all the values are outside the 5 and 95 percentile of the df_sim_shower if so delete the variable from the variable_PCA
                        # if df_obs_shower[var][i_var] < np.percentile(df_sim_shower[var], 1) or df_obs_shower[var][i_var] > np.percentile(df_sim_shower[var], 99):
                        #     ii_all=ii_all+
                        print(var)

                    if ii_all==len(df_obs_shower[var]):
                        print('All variable',var,'are not within the 1 and 99 percentile of the simulated meteors')
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

                        # elif var=='decel_parab_t0':
                        #     # do the cosine of the zenith angle
                        #     df_all[var]=np.log10(abs(df_all[var]))
                        #     # df_all[var]=transform_to_gaussian(df_all[var])
                        #     df_sim_shower_resample[var]=np.log10(abs(df_sim_shower_resample[var]))

                        else:
                            # if var=='vel_avg_norot':
                            # do the log10 of the variable
                            # df_all[var]=np.log10(abs(df_all[var]))

                            # var_transformed_boxcox, lam = stats.boxcox(abs(df_all[var]))
                            # df_all[var]=var_transformed_boxcox
                            # var_transformed_boxcox_sim, lam = stats.boxcox(abs(df_sim_shower_resample[var]))
                            # df_sim_shower_resample[var]=var_transformed_boxcox_sim

                            pt = PowerTransformer(method='yeo-johnson')
                            df_all[var]=pt.fit_transform(df_all[[var]])
                            df_sim_shower_resample[var]=pt.fit_transform(df_sim_shower_resample[[var]])

                            # pt = PowerTransformer(method='yeo-johnson')
                            # df_all[var] = pt.fit_transform(df_all[[var]])

                        shapiro_test = stats.shapiro(df_all[var])
                        print("NEW Shapiro-Wilk Test:", shapiro_test.statistic,"p-val", shapiro_test.pvalue)
                        
                        
                        # # 
                        # check_normality(df_sim_shower[var], var)
                        # # do the log of the variable and see the normality
                        # check_normality(np.log10(abs(df_sim_shower[var])), 'log_'+var)
                        # var_transformed_boxcox, lam = stats.boxcox(abs(df_sim_shower[var]))
                        # # use the boxcox transformation
                        # check_normality(var_transformed_boxcox, 'boxcox_'+var)
                        # df_sim_shower_resample[var]=var_transformed_boxcox
                else:
                    print('Variable ',var,' is not in the simulated shower')
            else:
                print('Variable ',var,' is not in the observed shower')

    #### PCR test #######################################################################################################

    # exclude_columns = ['shower_code', 'solution_id']
    # physical_vars = ['mass','rho','sigma','erosion_height_start','erosion_coeff','erosion_mass_index','erosion_mass_min','erosion_mass_max'] #, 'erosion_range', 'erosion_energy_per_unit_cross_section', 'erosion_energy_per_unit_mass'

    # # Define the observable variables for PCA
    # variable_PCA_no_info = [col for col in variable_PCA if col not in exclude_columns]

    # # Define X (observable variables) and y (physical variables)
    # X = df_sim_shower_resample[variable_PCA_no_info]
    # y = df_sim_shower_resample[physical_vars]

    # # Split the data into training and testing sets
    # X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    # X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.4, random_state=42)

    # # Save the n_pc and the r-squared in a variable
    # n_pc_r2 = []
    # n_pc_r2_err = []
    # n_pc_r2_val_err = []
    # n_pc_r2_plot = []
    # n_pc_r2_val = []
    # n_pc_r2_val_plot = []

    # # Loop over the number of principal components
    # for n_pc in range(1, len(variable_PCA_no_info) + 1):
    #     print("PCR Predictions with ", n_pc, "PC :")

    #     # PCR: Principal Component Regression
    #     pcr = make_pipeline(StandardScaler(), PCA(n_components=n_pc), LinearRegression())

    #     pcr.fit(X_train, y_train)
    #     # Predict using the models
    #     y_pred_train = pcr.predict(X_train)
    #     y_pred_val = pcr.predict(X_val)

    #     # Evaluate the model on the training and validation set
    #     r2_score_train = pcr.score(X_train, y_train)
    #     r2_score_val = pcr.score(X_val, y_val)
    #     mse_train = mean_squared_error(y_train, y_pred_train)
    #     mse_val = mean_squared_error(y_val, y_pred_val)

    #     # print(f"PCR training r-squared: {r2_score_train:.3f}")
    #     # print(f"PCR validation r-squared: {r2_score_val:.3f}")
    #     # print(f"PCR training MSE: {mse_train:.3f}")
    #     # print(f"PCR validation MSE: {mse_val:.3f}")
    #     # real vs predicted
    #     y_pred_pcr = pcr.predict(df_sim_shower_resample[variable_PCA_no_info])
    #     to_plot_unit=['mass [kg]','rho [kg/m^3]','sigma [s^2/km^2]','erosion height start [km]','erosion coeff [s^2/km^2]','erosion mass index [-]','eros. mass min [kg]','eros. mass max [kg]']
    #     # multiply y_pred_pcr that has the 'erosion_coeff'*1000000 and 'sigma'*1000000
    #     y_pred_pcr[:,4]=y_pred_pcr[:,4]*1000000
    #     y_pred_pcr[:,2]=y_pred_pcr[:,2]*1000000
    #     # Get the real values
    #     real_values = df_sim_shower_resample[physical_vars].iloc[0].values
    #     # multiply the real_values
    #     real_values[4]=real_values[4]*1000000
    #     real_values[2]=real_values[2]*1000000


    #     # Print the predictions alongside the real values
    #     print("Predicted vs Real Values:")
    #     for i, unit in enumerate(to_plot_unit):
    #         print(f'{unit}: Predicted: {y_pred_pcr[0, i]:.4g}, Real: {real_values[i]:.4g}')
    #     print('--------------------------')


    #     # Save the results for plotting and analysis
    #     n_pc_r2.append((n_pc, r2_score_train, r2_score_val))
    #     n_pc_r2_plot.append((n_pc, r2_score_train))
    #     n_pc_r2_val_plot.append((n_pc, r2_score_val))
    #     if r2_score_val < 0 or np.isnan(r2_score_val):
    #         n_pc_r2_err.append((n_pc, r2_score_train))
    #         n_pc_r2_val_err.append((n_pc, r2_score_val))

    # # Convert lists to arrays for plotting
    # n_pc_r2_plot = np.array(n_pc_r2_plot)
    # n_pc_r2_val_plot = np.array(n_pc_r2_val_plot)

    # plt.figure(figsize=(10, 6))
    # # put the values of different variance explained by PCA
    # plt.plot(n_pc_r2_plot[:, 0], n_pc_r2_plot[:, 1], label='Training R-squared',color='k')
    # plt.plot(n_pc_r2_val_plot[:, 0], n_pc_r2_val_plot[:, 1], label='Validation R-squared', linestyle='--',color='k')
    # plt.xlabel('Number of Principal Components')
    # plt.ylabel('R-squared')
    # plt.title('PCR R-squared vs. Number of Principal Components')
    # plt.legend()
    # num_components = len(n_pc_r2_val_plot[:, 1])
    # plt.xticks(ticks=np.arange(1, num_components + 1))
    # plt.grid()
    # plt.savefig(OUT_PUT_PATH+os.sep+'R-sq_'+Shower[0]+'_PCR.png')
    # plt.close()

    # # Find the best number of PCs based on the highest validation R-squared
    # n_pc_r2_plot = np.array(n_pc_r2_plot)
    # best_n_pc = int(n_pc_r2_plot[np.argmax(n_pc_r2_plot[:, 1]), 0])
    # print(f'The best number of PCs is: {best_n_pc}')

    ####################################################################################################################

    # keep only the variable_PCA variables
    # df_all = pd.concat([df_sim_shower[variable_PCA],df_obs_shower[variable_PCA]], axis=0, ignore_index=True)
    # df_all = pd.concat([df_sim_shower_resample[variable_PCA],df_obs_shower_resample[variable_PCA]], axis=0, ignore_index=True)

    # # delete nan
    # df_all = df_all.dropna()

    # Now we have all the data and we apply PCA to the dataframe
    df_all_nameless=df_all.drop(['shower_code','solution_id'], axis=1)

    if PCA_pairplot:
        df_all_nameless_plot=df_all_nameless.copy()

        if len(df_all_nameless_plot)>10000:
            # pick randomly 10000 events
            print('Number of events in the simulated shower:',len(df_all_nameless_plot))
            df_all_nameless_plot=df_all_nameless_plot.sample(n=10000)

        # # sns plot of the df_sim_var_sel and df_sim_var_sel_no_outliers hue='shower_code'
        # sns.pairplot(df_all_nameless_plot)
        # print('Pairplot of the all shower to go to PCA:')
        # # save the figure
        # plt.savefig(OUT_PUT_PATH+os.sep+'var_sns_after_norm_'+Shower[0]+'_select_PCA.png')
        # # close the figure
        # plt.close()





        # # make a subplot of the distribution of the variables
        # fig, axs = plt.subplots(3, 4, figsize=(20, 15))
        # for i, var in enumerate(variable_PCA[2:]):
        #     # plot the distribution of the variable
        #     sns.histplot(df_all_nameless_plot[var], kde=True, ax=axs[i//4, i%4], color='b', alpha=0.5)
        #     # axs[i//4, i%4].set_title('Distribution of '+var)
        #     # put a vertical line for the df_obs_shower[var] value
        #     axs[i//4, i%4].axvline(df_all_nameless_plot[var].values[0], color='limegreen', linestyle='--', linewidth=5)
        #     # grid
        #     axs[i//4, i%4].grid()
        # # save the figure
        # plt.savefig(OUT_PUT_PATH+os.sep+'var_hist_after_norm_'+Shower[0]+'_select_PCA.png')
        # # close the figure
        # plt.close()

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
    
    # # show the explained variance ratio of all th pc
    # # check if any in the explained variance ratio is below 1%
    # if np.any(pca.explained_variance_ratio_ < 0.01):
    #     print(pca.explained_variance_ratio_)
    #     # print the number of elements below 1%
    #     print('PC below 1%:',np.sum(pca.explained_variance_ratio_ < 0.01))
    #     print('PC above 1%:',np.sum(pca.explained_variance_ratio_ > 0.01))
    #     # delete the np.sum(pca.explained_variance_ratio_ < 0.01) PC whith an explained variance ratio below 1%
    #     pca = PCA(n_components=np.sum(pca.explained_variance_ratio_ > 0.01))
    #     all_PCA = pca.fit_transform(scaled_df_all)
    #     print(pca.explained_variance_ratio_)

    # check if a file with the name "log"+n_PC_in_PCA+"_"+str(len(df_sel))+"ev.txt" already exist
    if os.path.exists(OUT_PUT_PATH+os.sep+"log_"+str(len(variable_PCA))+"var_"+str(PCA_percent)+"%_"+str(pca.n_components_)+"PC.txt"):
        # remove the file
        os.remove(OUT_PUT_PATH+os.sep+os.sep+"log_"+str(len(variable_PCA))+"var_"+str(PCA_percent)+"%_"+str(pca.n_components_)+"PC.txt")
    sys.stdout = Logger(OUT_PUT_PATH,"log_"+str(len(variable_PCA))+"var_"+str(PCA_percent)+"%_"+str(pca.n_components_)+"PC.txt") # _30var_99%_13PC

    ################################# Apply Varimax rotation ####################################
    loadings = pca.components_.T

    rotated_loadings = varimax(loadings)

    # # chage the loadings to the rotated loadings in the pca components
    pca.components_ = rotated_loadings.T

    # Transform the original PCA scores with the rotated loadings ugly PC space but same results
    # all_PCA = np.dot(all_PCA, rotated_loadings.T[:pca.n_components_, :pca.n_components_])

    ############### PCR ########################################################################################


    exclude_columns = ['shower_code', 'solution_id']
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
    for i, unit in enumerate(to_plot_unit):
        print(f'{unit}: Predicted: {y_pred_pcr[0, i]:.4g}, Real: {real_values[i]:.4g}')
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
    plt.savefig(OUT_PUT_PATH+os.sep+'PCAexplained_variance_ratio_'+str(len(variable_PCA)-2)+'var_'+str(PCA_percent)+'%_'+str(pca.n_components_)+'PC.png')
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
    plt.savefig(OUT_PUT_PATH+os.sep+'PCAcovariance_matrix_'+str(len(variable_PCA)-2)+'var_'+str(PCA_percent)+'%_'+str(pca.n_components_)+'PC.png')
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
    df_all_PCA['shower_code'] = df_all['shower_code'].values

    # delete the lines after len(df_sim_shower) to have only the simulated shower
    df_sim_PCA = df_all_PCA.drop(df_all_PCA.index[len(df_sim_shower):])
    df_obs_PCA = df_all_PCA.drop(df_all_PCA.index[:len(df_sim_shower)])

    # # # plot all the data in the PCA space
    # sns.pairplot(df_obs_PCA, hue='shower_code', plot_kws={'alpha': 0.6, 's': 5, 'edgecolor': 'k'},corner=True)
    # # plt.show()
    # # # sns.pairplot(df_obs_PCA, plot_kws={'alpha': 0.6, 's': 5, 'edgecolor': 'k', 'c':'palegreen'},corner=True)

    # sns.pairplot(df_sim_PCA, hue='shower_code', plot_kws={'alpha': 0.6, 's': 5, 'edgecolor': 'k'},corner=True)
    # plt.show()


    ##############
    # create a new dataframe with the selected simulated shower
    # df_obs_PCA_rho=df_obs_PCA
    # df_obs_PCA_rho['rho_n']='REAL'

    # df_sim_PCA_rho=df_sim_PCA
    # for ii in range(9):
    #     # find index in df_sim_shower that have rho btween 100*ii and 100*(ii+1) and label it as str(ii*100)
    #     df_sim_PCA_rho.loc[(df_sim_shower['rho'] > ii*100) & (df_sim_shower['rho'] <= (ii+1)*100), 'rho_n'] = str(ii*100)
    

    # Test=pd.concat([df_sim_PCA_rho,df_obs_PCA_rho], axis=0)

    # sns.pairplot(Test, hue='rho_n', plot_kws={'alpha': 0.6, 's': 5, 'edgecolor': 'k'},corner=True)
    # # plt.show()
    # # delete the rho_n column
    # df_obs_PCA_rho = df_obs_PCA_rho.drop(['rho_n'], axis=1)
    # df_sim_PCA_rho = df_sim_PCA_rho.drop(['rho_n'], axis=1)
    # plt.savefig(OUT_PUT_PATH+os.sep+'PCA_pairplot_'+str(len(variable_PCA)-2)+'var_'+str(PCA_percent)+'%_'+str(pca.n_components_)+'PC.png')
    # plt.close()
    ##############
    
    # # save the figure
    # plt.savefig(OUT_PUT_PATH+os.sep+'PCA_pairplot_'+str(len(variable_PCA)-2)+'var_'+str(PCA_percent)+'%_'+str(pca.n_components_)+'PC.png')
    # # close the figure
    # plt.close()

    # ##########################################
    # # define the mean position and extract the n_selected meteor closest to the mean

    # # find the mean base on the shower_code in the PCA space
    # meanPCA = df_all_PCA.groupby('shower_code').mean()
    # df_all_PCA['solution_id']=df_all['solution_id']
    # # create a list with the selected meteor properties and PCA space
    # df_sel_shower=[]
    # df_sel_PCA=[]
    # # i_shower_preced=0
    # jj=0

    # for current_shower in Shower:
    #     # find the mean of the simulated shower
    #     meanPCA_current = meanPCA.loc[(meanPCA.index == current_shower)]
    #     # take only the value of the mean of the first row
    #     meanPCA_current = meanPCA_current.values

    #     shower_current = df_obs_shower[df_obs_shower['shower_code']==current_shower]
    #     shower_current_PCA = df_obs_PCA[df_obs_PCA['shower_code']==current_shower]
    #     # trasform the dataframe in an array
    #     shower_current_PCA = shower_current_PCA.drop(['shower_code'], axis=1)
    #     shower_current_PCA = shower_current_PCA.values 

    #     df_sim_PCA_for_now = df_sim_PCA
    #     df_sim_PCA_for_now = df_sim_PCA_for_now.drop(['shower_code'], axis=1)
    #     df_sim_PCA_val = df_sim_PCA_for_now.values 

    #     for i_shower in range(len(shower_current)):
    #         distance_current = []
    #         for i_sim in range(len(df_sim_PCA_val)):
    #             distance_current.append(scipy.spatial.distance.euclidean(df_sim_PCA_val[i_sim], shower_current_PCA[i_shower]))
            

    #         ############ Value ###############
    #         # print the ['solution_id'] of the element [i_shower]
    #         # print(shower_current['solution_id'][i_shower])
    #         # create an array with lenght equal to the number of simulations and set it to shower_current_PCA['solution_id'][i_shower]
    #         solution_id_dist = [shower_current['solution_id'][i_shower]]*len(df_sim_PCA_val)

    #         # give the same solution_id of the shower_current_PCA['solution_id'] of at the i_shower row in a new column solution_id_dist of the df_sim_shower create a slice long as the simulations
    #         df_sim_shower['solution_id_dist']=solution_id_dist

    #         df_sim_shower['distance_meteor']=distance_current

    #         # sort the distance and select the n_selected closest to the mean
    #         df_sim_shower_dis = df_sim_shower.sort_values(by=['distance_meteor'])
    #         # drop the index
    #         df_sim_shower_dis = df_sim_shower_dis.reset_index(drop=True)
        
    #         # create a dataframe with the selected simulated shower characteristics
    #         df_sim_selected = df_sim_shower_dis[:N_sho_sel]
    #         # delete the shower code
    #         df_sim_selected = df_sim_selected.drop(['shower_code'], axis=1)
    #         # add the shower code
    #         df_sim_selected['shower_code']= current_shower+'_sel'
    #         df_sel_shower.append(df_sim_selected)

    #         ################ PCA ################

    #         df_sim_PCA_dist=df_sim_PCA
    #         df_sim_PCA_dist['distance_meteor']=distance_current

    #         # sort the distance and select the n_selected closest to the mean
    #         df_sim_PCA_dist = df_sim_PCA_dist.sort_values(by=['distance_meteor'])
    #         # drop the index
    #         df_sim_PCA_dist = df_sim_PCA_dist.reset_index(drop=True)
        
    #         # create a dataframe with the selected simulated shower characteristics
    #         df_sim_selected_PCA = df_sim_PCA_dist[:N_sho_sel]
    #         # delete the shower code
    #         df_sim_selected_PCA = df_sim_selected_PCA.drop(['shower_code'], axis=1)
    #         # add the shower code
    #         df_sim_selected_PCA['shower_code']= current_shower+'_sel'

    #         df_sel_PCA.append(df_sim_selected_PCA)

    #         # print the progress bar in percent that refresh every 10 iteration and delete the previous one
    #         if i_shower%10==0:
    #             print('Processing ',current_shower,' shower: ', round(i_shower/len(shower_current)*100,2),'%', end="\r")


    #         # print('.', end='', flush=True)


    #     print('Processing ',current_shower,' shower:  100  %      ', end="\r")

    #     # concatenate the list of the PC components to a dataframe
    #     df_sel_PCA = pd.concat(df_sel_PCA)

    #     # delete the distace column from df_sel_PCA
    #     df_sel_PCA = df_sel_PCA.drop(['distance_meteor'], axis=1)

    #     # delete the shower code column
    #     df_sim_PCA = df_sim_PCA.drop(['distance_meteor'], axis=1)

    #     # save the dataframe to a csv file withouth the index
    #     df_sel_PCA.to_csv(OUT_PUT_PATH+os.sep+'Simulated_'+current_shower+'_select_PCA.csv', index=False)

    #     # concatenate the list of the properties to a dataframe
    #     df_sel_shower = pd.concat(df_sel_shower)

    #     df_sel_PCA_NEW = df_sel_PCA.drop(['shower_code'], axis=1)
    #     # create a list of the selected shower
    #     df_sel_PCA_NEW = df_sel_PCA_NEW.values
    #     distance_current = []
    #     # Flatten meanPCA_current to make it a 1-D array
    #     meanPCA_current = meanPCA_current.flatten()
    #     for i_shower in range(len(df_sel_shower)):
    #         distance_current.append(scipy.spatial.distance.euclidean(meanPCA_current, df_sel_PCA_NEW[i_shower]))
    #     df_sel_shower['distance']=distance_current # from the mean of the selected shower
    #     # save the dataframe to a csv file withouth the index
    #     df_sel_shower.to_csv(OUT_PUT_PATH+os.sep+'Simulated_'+current_shower+'_select.csv', index=False)

    #     # save dist also on selected shower
    #     distance_current = []
    #     for i_shower in range(len(shower_current)):
    #         distance_current.append(scipy.spatial.distance.euclidean(meanPCA_current, shower_current_PCA[i_shower]))
    #     shower_current['distance']=distance_current # from the mean of the selected shower
    #     shower_current.to_csv(OUT_PUT_PATH+os.sep+current_shower+'_and_dist.csv', index=False)

    # # copy Simulated_PER.csv in OUT_PUT_PATH
    # shutil.copyfile(INPUT_PATH+os.sep+'Simulated_PER.csv', OUT_PUT_PATH+os.sep+'Simulated_PER.csv')

    ####################distance with mahalanobis distance############################################

    # for i_shower in range(len(df_sel_shower)):
    #     distance_current.append(mahalanobis_distance(meanPCA_current, df_sel_PCA_NEW[i_shower], cov_inv))
    # df_sel_shower['distance'] = distance_current
    # df_sel_shower.to_csv(os.path.join(OUT_PUT_PATH, f'Simulated_{current_shower}_select.csv'), index=False)
    # distance_current = []

    # for i_shower in range(len(shower_current)):
    #     distance_current.append(mahalanobis_distance(meanPCA_current, shower_current_PCA[i_shower], cov_inv))
    # shower_current['distance'] = distance_current
    # shower_current.to_csv(os.path.join(OUT_PUT_PATH, f'{current_shower}_and_dist.csv'), index=False)

    ##################################################################################################

    # Calculate the mean and inverse covariance matrix for Mahalanobis distance
    meanPCA = df_all_PCA.groupby('shower_code').mean()

    # cov_matrix = df_all_PCA.cov()
    # cov_inv = inv(cov_matrix)

    df_all_PCA['solution_id'] = df_all['solution_id']
    df_sel_shower = []
    df_sel_PCA = []
    jj = 0

    # Get explained variances of principal components
    explained_variance = pca.explained_variance_ratio_

    # Calculate mean and inverse covariance matrix for Mahalanobis distance
    cov_matrix = df_all_PCA.drop(['shower_code'], axis=1).cov()

    # Modify covariance matrix based on explained variances
    for i in range(len(explained_variance)):
        cov_matrix.iloc[i, :] /= explained_variance[i]

    # # Modify covariance matrix to positively reflect variance explained
    # for i in range(len(explained_variance)):
    #     cov_matrix.iloc[i, :] *= explained_variance[i]

    cov_inv = inv(cov_matrix)

    for current_shower in Shower:
        # find the mean of the simulated shower
        meanPCA_current = meanPCA.loc[(meanPCA.index == current_shower)].values.flatten()
        # take only the value of the mean of the first row
        shower_current = df_obs_shower[df_obs_shower['shower_code'] == current_shower]
        shower_current_PCA = df_obs_PCA[df_obs_PCA['shower_code'] == current_shower]
        # trasform the dataframe in an array
        shower_current_PCA = shower_current_PCA.drop(['shower_code'], axis=1).values
        df_sim_PCA_for_now = df_sim_PCA.drop(['shower_code'], axis=1).values
        
        for i_shower in range(len(shower_current)):
            distance_current = []

            for i_sim in range(len(df_sim_PCA_for_now)):
                distance_current.append(mahalanobis_distance(df_sim_PCA_for_now[i_sim], shower_current_PCA[i_shower], cov_inv))
        
            # # Calculate distances between current observed shower and all simulated showers
            # for i_sim in range(len(df_sim_PCA)):
            #     distance_current.append(modified_mahalanobis_distance(df_sim_PCA_for_now[i_sim], shower_current_PCA[i_shower], cov_inv))
            
            # for i_sim in range(len(df_sim_PCA_for_now)):
            #     distance = mahalanobis_distance(df_sim_PCA_for_now[i_sim], shower_current_PCA[i_shower], cov_inv)
            #     weighted_distance = distance * np.sqrt(explained_variance)  # Weighting by explained variance
            #     distance_current.append(np.sum(weighted_distance))  # Sum the weighted distances
            
            # create an array with lenght equal to the number of simulations and set it to shower_current_PCA['solution_id'][i_shower]
            solution_id_dist = [shower_current['solution_id'][i_shower]] * len(df_sim_PCA_for_now)
            df_sim_shower['solution_id_dist'] = solution_id_dist
            df_sim_shower['distance_meteor'] = distance_current
            # sort the distance and select the n_selected closest to the meteor
            df_sim_shower_dis = df_sim_shower.sort_values(by=['distance_meteor']).reset_index(drop=True)
            df_sim_selected = df_sim_shower_dis[:N_sho_sel].drop(['shower_code'], axis=1)
            df_sim_selected['shower_code'] = current_shower + '_sel'
            df_sel_shower.append(df_sim_selected)
            
            # create a dataframe with the selected simulated shower characteristics
            df_sim_PCA_dist = df_sim_PCA
            df_sim_PCA_dist['distance_meteor'] = distance_current
            df_sim_PCA_dist = df_sim_PCA_dist.sort_values(by=['distance_meteor']).reset_index(drop=True)
            # delete the shower code
            df_sim_selected_PCA = df_sim_PCA_dist[:N_sho_sel].drop(['shower_code'], axis=1)
            # add the shower code
            df_sim_selected_PCA['shower_code'] = current_shower + '_sel'
            df_sel_PCA.append(df_sim_selected_PCA)
            
            if i_shower % 10 == 0:
                print(f'Processing {current_shower} shower: {round(i_shower/len(shower_current)*100, 2)}%', end="\r")
        
        print(f'Processing {current_shower} shower: 100%', end="\r")
        
    df_sel_PCA = pd.concat(df_sel_PCA).drop(['distance_meteor'], axis=1)
    df_sim_PCA = df_sim_PCA.drop(['distance_meteor'], axis=1)
    df_sel_PCA.to_csv(os.path.join(OUT_PUT_PATH, f'Simulated_{current_shower}_select_PCA.csv'), index=False)
    df_sel_shower = pd.concat(df_sel_shower)
    df_sel_PCA_NEW = df_sel_PCA.drop(['shower_code'], axis=1).values
    distance_current = []

    # Flatten meanPCA_current to make it a 1-D array
    meanPCA_current = meanPCA_current.flatten()

    for i_shower in range(len(df_sel_shower)):
        distance_current.append(scipy.spatial.distance.euclidean(meanPCA_current, df_sel_PCA_NEW[i_shower]))
    df_sel_shower['distance']=distance_current # from the mean of the selected shower
    # for the name of observation shower check if it has ben selected one with the same name unique for the solution_id_dist ['solution_id_dist'].unique()
    for len_shower in range(len(shower_current)):
        sol_id_dist_search_OG = shower_current['solution_id'][0]
        sol_id_dist_search = shower_current['solution_id'][len_shower]
        # get all the data with the same solution_id_dist
        all_sol_id_dist = df_sel_shower[df_sel_shower['solution_id_dist'] == sol_id_dist_search]
        # check if among all_sol_id_dist there is one with the same solution_id as the sol_id_dist_search
        if len(all_sol_id_dist[all_sol_id_dist['solution_id'] == sol_id_dist_search_OG]) == 0:
            # add a row with the same solution_id_dist and the same solution_id with the values of df_sim_shower
            df_sel_shower_real = df_sim_shower[df_sim_shower['solution_id'] == sol_id_dist_search_OG].copy()
            # add to the df_sel_shower_real the solution_id_dist, distance_meteor, shower_code, distance
            df_sel_shower_real.loc[:, 'solution_id_dist'] = sol_id_dist_search
            df_sel_shower_real.loc[:, 'distance_meteor'] = 9999
            df_sel_shower_real.loc[:, 'shower_code'] = current_shower + '_sel'
            df_sel_shower_real.loc[:, 'distance'] = 9999
            df_sel_shower_real.loc[:, 'solution_id'] = sol_id_dist_search
            # add the row to the df_sel_shower
            df_sel_shower = pd.concat([df_sel_shower, df_sel_shower_real])
        else:
            # change the solution_id of the selected shower to the solution_id of the observation shower
            df_sel_shower.loc[(df_sel_shower['solution_id_dist'] == sol_id_dist_search) & (df_sel_shower['solution_id'] == sol_id_dist_search_OG), 'solution_id'] = sol_id_dist_search
    # save the dataframe to a csv file withouth the index
    df_sel_shower.to_csv(OUT_PUT_PATH+os.sep+'Simulated_'+current_shower+'_select.csv', index=False)

    # save dist also on selected shower
    distance_current = []
    for i_shower in range(len(shower_current)):
        distance_current.append(scipy.spatial.distance.euclidean(meanPCA_current, shower_current_PCA[i_shower]))
    shower_current['distance']=distance_current # from the mean of the selected shower
    shower_current.to_csv(OUT_PUT_PATH+os.sep+current_shower+'_and_dist.csv', index=False)

    shutil.copyfile(os.path.join(INPUT_PATH, 'Simulated_PER.csv'), os.path.join(OUT_PUT_PATH, 'Simulated_PER.csv'))

    # PLOT the selected simulated shower ########################################

    # dataframe with the simulated and the selected meteors in the PCA space
    # df_sim_sel_PCA = pd.concat([df_sim_PCA,df_sel_PCA], axis=0)

    if PCA_pairplot:
        if len(df_sim_PCA)>10000:
            # pick randomly 10000 events
            df_sim_PCA=df_sim_PCA.sample(n=10000)

        df_sim_sel_PCA = pd.concat([df_sim_PCA,df_sel_PCA,df_obs_PCA], axis=0)

        # Select only the numeric columns for percentile calculations
        numeric_columns = df_sim_sel_PCA.select_dtypes(include=[np.number]).columns

        # Calculate the 1st and 99th percentiles for each numeric variable
        percentiles_1 = df_sim_sel_PCA[numeric_columns].quantile(0.01)
        percentiles_99 = df_sim_sel_PCA[numeric_columns].quantile(0.99)

        # Create a new column for point sizes
        df_sim_sel_PCA['point_size'] = df_sim_sel_PCA['shower_code'].map({
            'sim_PER': 5,
            'PER_sel': 5,
            'PER': 40
        })

        # open a new figure to plot the pairplot
        fig = plt.figure(figsize=(10, 10), dpi=300)

        # # fig = sns.pairplot(df_sim_sel_PCA, hue='shower_code', plot_kws={'alpha': 0.6, 's': 5, 'edgecolor': 'k'},corner=True)
        # fig = sns.pairplot(df_sim_sel_PCA, hue='shower_code',corner=True, palette='bright', diag_kind='kde', plot_kws={'s': 5, 'edgecolor': 'k'})
        # # plt.show()

        # Create the pair plot without points initially
        fig = sns.pairplot(df_sim_sel_PCA[numeric_columns.append(pd.Index(['shower_code']))], hue='shower_code', corner=True, palette='bright', diag_kind='kde', plot_kws={'s': 5, 'edgecolor': 'k'})

        # Overlay scatter plots with custom point sizes
        for i in range(len(fig.axes)):
            for j in range(len(fig.axes)):
                if i > j:
                    # check if the variable is in the list of the numeric_columns and set the axis limit
                    if df_sim_sel_PCA.columns[j] in numeric_columns and df_sim_sel_PCA.columns[i] in numeric_columns:

                        ax = fig.axes[i, j]
                        sns.scatterplot(data=df_sim_sel_PCA, x=df_sim_sel_PCA.columns[j], y=df_sim_sel_PCA.columns[i], hue='shower_code', size='point_size', sizes=(5, 40), ax=ax, legend=False, edgecolor='k', palette='bright')

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
        fig.savefig(OUT_PUT_PATH+os.sep+'PCAspace_sim_sel_real_'+str(len(variable_PCA)-2)+'var_'+str(PCA_percent)+'%_'+str(pca.n_components_)+'PC.png')
        # close the figure
        plt.close()

        # # df_sim_PCA,df_sel_PCA,df_obs_PCA
        # # print(df_sim_shower)
        # # loop all pphysical variables
        # physical_vars = ['mass','rho','sigma','erosion_height_start','erosion_coeff','erosion_mass_index','erosion_mass_min','erosion_mass_max']
        # for var_phys in physical_vars:
        #     # make a subplot of the rho againist each variable_PCA as a scatter plot
        #     fig, axs = plt.subplots(4, 5, figsize=(20, 15))
        #     # flatten the axs array
        #     axs = axs.flatten()
        #     for i, var in enumerate(variable_PCA[2:]):
        #         # plot the rho againist the variable with black borders
        #         axs[i].scatter(df_sim_shower[var], df_sim_shower[var_phys], c='b') #, edgecolors='k', alpha=0.5

        #         axs[i].scatter(df_sel_shower[var], df_sel_shower[var_phys], c='orange') #, edgecolors='k', alpha=0.5
        #         # put a green vertical line for the df_obs_shower[var] value
        #         axs[i].axvline(shower_current[var].values[0], color='limegreen', linestyle='--', linewidth=5)
        #         # put a horizontal line for the rho of the first df_sim_shower
        #         axs[i].axhline(df_sim_shower[var_phys].values[0], color='k', linestyle='-', linewidth=2)
        #         axs[i].set_title(var)
        #         # as a suptitle put the variable_PCA
        #         fig.suptitle(var_phys)
        #         # grid
        #         axs[i].grid()
        #         # make y axis log if the variable is 'erosion_mass_min' 'erosion_mass_max'
        #         if var_phys == 'erosion_mass_min' or var_phys == 'erosion_mass_max':
        #             axs[i].set_yscale('log')

        #     # save the figure
        #     plt.savefig(OUT_PUT_PATH+os.sep+var_phys+'_vs_var_'+Shower[0]+'_select_PCA.png')
        #     # close the figure
        #     plt.close()


        # loop all pphysical variables
        physical_vars = ['mass','rho','sigma','erosion_height_start','erosion_coeff','erosion_mass_index','erosion_mass_min','erosion_mass_max']
        for var_phys in physical_vars:
            # make a subplot of the rho againist each variable_PCA as a scatter plot
            fig, axs = plt.subplots(3, 4, figsize=(20, 15))
            # flatten the axs array
            axs = axs.flatten()
            for i, var in enumerate(columns_PC):
                # plot the rho againist the variable with black borders
                axs[i].scatter(df_sim_PCA[var], df_sim_shower[var_phys], c='b') #, edgecolors='k', alpha=0.5

                axs[i].scatter(df_sel_PCA[var], df_sel_shower[var_phys], c='orange') #, edgecolors='k', alpha=0.5
                # put a green vertical line for the df_obs_shower[var] value
                axs[i].axvline(df_obs_PCA[var].values[0], color='limegreen', linestyle='--', linewidth=5)
                # put a horizontal line for the rho of the first df_sim_shower
                axs[i].axhline(df_sim_shower[var_phys].values[0], color='k', linestyle='-', linewidth=2)
                axs[i].set_title(var)
                # as a suptitle put the variable_PCA
                fig.suptitle(var_phys)
                # grid
                axs[i].grid()
                # make y axis log if the variable is 'erosion_mass_min' 'erosion_mass_max'
                if var_phys == 'erosion_mass_min' or var_phys == 'erosion_mass_max':
                    axs[i].set_yscale('log')

            # save the figure
            plt.savefig(OUT_PUT_PATH+os.sep+var_phys+'_vs_var_'+Shower[0]+'_select_PC_space.png')
            # close the figure
            plt.close()

    # sns.pairplot(df_sel_PCA, hue='shower_code', plot_kws={'alpha': 0.6, 's': 5, 'edgecolor': 'k'},corner=True)
    # plt.show()


    # # dataframe with the simulated and the selected meteors physical characteristics
    # df_sim_sel_shower = pd.concat([df_sim_shower,df_sel_shower], axis=0)

    # sns.pairplot(df_sel_shower[['shower_code','rho','sigma','erosion_height_start','erosion_coeff','erosion_mass_index','erosion_mass_min','erosion_mass_max']], hue='shower_code', plot_kws={'alpha': 0.6, 's': 5, 'edgecolor': 'k'},corner=True)
    # plt.show()

    print('\nSUCCESS: the simulated shower have been selected')

    # Close the Logger to ensure everything is written to the file STOP COPY in TXT file
    sys.stdout.close()

    # Reset sys.stdout to its original value if needed
    sys.stdout = sys.__stdout__




# Assuming df_sim_shower and df_obs_shower are defined and populated somewhere
# Example usage:
# PCASim(df_sim_shower, df_obs_shower, '/path/to/output', Shower=['PER'], N_sho_sel=10000, No_var_PCA=['var1', 'var2'], INPUT_PATH='/path/to/input')




if __name__ == "__main__":

    import argparse


    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Fom Observation and simulated data weselect the most likely through PCA, run it, and store results to disk.")

    arg_parser.add_argument('--output_dir', metavar='OUTPUT_PATH', type=str, default=r'C:\Users\maxiv\Documents\UWO\Papers\1)PCA\Reproces_2cam\SimFolder\TEST', \
        help="Path to the output directory.")

    arg_parser.add_argument('--shower', metavar='SHOWER', type=str, default='PER', \
        help="Use specific shower from the given simulation.")
    
    arg_parser.add_argument('--input_dir', metavar='INPUT_PATH', type=str, default=r'C:\Users\maxiv\Documents\UWO\Papers\1)PCA\Reproces_2cam\SimFolder', \
        help="Path were are store both simulated and observed shower .csv file.")

    arg_parser.add_argument('--nsel', metavar='SEL_NUM', type=int, default=1000, \
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
        if os.path.isfile(cml_args.input_dir+os.sep+'Simulated_'+current_shower+'.csv'):
            # if yes read the csv file
            df_sim = pd.read_csv(cml_args.input_dir+os.sep+'Simulated_'+current_shower+'.csv')
        else:
            # open the folder and extract all the json files
            os.chdir(cml_args.input_dir+os.sep+folder_GenerateSimulations_json[Shower.index(current_shower)])
            # print the current directory in
            directory=cml_args.input_dir
            extension = 'json'
            # walk thorought the directories and find all the json files inside each folder inside the directory
            all_jsonfiles = [i for i in glob.glob('**/*.{}'.format(extension), recursive=True)]
            #

            print('Number of simulated files: ',len(all_jsonfiles))
            
            # all_jsonfiles = [i for i in glob.glob('*.{}'.format(extension))]
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


            df_sim.to_csv(cml_args.input_dir+os.sep+'Simulated_'+current_shower+'.csv', index=False)



        if os.path.isfile(cml_args.input_dir+os.sep+current_shower+'.csv'):
            # if yes read the csv file
            df_obs = pd.read_csv(cml_args.input_dir+os.sep+current_shower+'.csv')
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

    PCASim(df_sim_shower,df_obs_shower, cml_args.output_dir, Shower, cml_args.nsel, cml_args.NoPCA, cml_args.input_dir)

    # @cython.cdivision(True) 
    # cpdef double decelerationRK4(double dt, double K, double m, double rho_atm, double v):
    #     """ Computes the deceleration using the 4th order Runge-Kutta method. """

    #     cdef double vk1, vk2, vk3, vk4

    #     # Compute change in velocity
    #     vk1 = dt*deceleration(K, m, rho_atm, v)
    #     vk2 = dt*deceleration(K, m, rho_atm, v + vk1/2.0)
    #     vk3 = dt*deceleration(K, m, rho_atm, v + vk2/2.0)
    #     vk4 = dt*deceleration(K, m, rho_atm, v + vk3)
        
    #     return (vk1/6.0 + vk2/3.0 + vk3/3.0 + vk4/6.0)/dt

    # cdef double deceleration(double K, double m, double rho_atm, double v):
    # """ Computes the deceleration derivative.     

    # Arguments:
    #     K: [double] Shape-density coefficient (m^2/kg^(2/3)).
    #     m: [double] Mass (kg).
    #     rho_atm: [double] Atmosphere density (kg/m^3).
    #     v: [double] Velocity (m/S).

    # Return:
    #     dv/dt: [double] Deceleration.
    # """

    # return -K*m**(-1/3.0)*rho_atm*v**2

    #    # Compute change in velocity
    #     deceleration_total = decelerationRK4(const.dt, frag.K, frag.m, rho_atm, frag.v)


            #     # Vertical component of a
            # av = -deceleration_total*frag.vv/frag.v + frag.vh*frag.v/(const.r_earth + frag.h)

            # # Horizontal component of a
            # ah = -deceleration_total*frag.vh/frag.v - frag.vv*frag.v/(const.r_earth + frag.h)

            # # Update the velocity
            # frag.vv -= av*const.dt
            # frag.vh -= ah*const.dt
            # frag.v = math.sqrt(frag.vh**2 + frag.vv**2)
