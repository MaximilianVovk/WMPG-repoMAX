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
import multiprocessing
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import seaborn as sns
import scipy.spatial.distance
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from heapq import nsmallest
import wmpl
from wmpl.MetSim.GUI import loadConstants
import shutil
from scipy.stats import kurtosis, skew
from wmpl.Utils.OSTools import mkdirP
import math
from wmpl.Utils.PyDomainParallelizer import domainParallelizer


# FUNCTIONS ###########################################################################################


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
        0, 0, 0, 0, 0, 0, 0, 0,\
        0, 0,\
        0, 0, 0, 0, 0, 0, 0, 0,\
        0, 0, 0,\
        0, 0, 0,\
        0, 0]]

    # create a dataframe to store the data
    df_json = pd.DataFrame(dataList, columns=['solution_id','shower_code','vel_init_norot','vel_avg_norot','duration',\
    'mass','peak_mag_height','begin_height','end_height','height_knee_vel','peak_abs_mag','beg_abs_mag','end_abs_mag',\
    'F','trail_len','acceleration','acceleration_lin','decel_after_knee_vel','zenith_angle', 'kurtosis','skew',\
    'kc','Dynamic_pressure_peak_abs_mag',\
    'a_acc','b_acc','a_mag_init','b_mag_init','a_mag_end','b_mag_end','rho','sigma',\
    'erosion_height_start','erosion_coeff', 'erosion_mass_index',\
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
            mass = data['params']['m_init']['val']
            vel_init_norot = data['params']['v_init']['val']/1000
            zenith_angle= data['params']['zenith_angle']['val']*180/np.pi

            # Physical parameters
            rho = data['params']['rho']['val']
            sigma = data['params']['sigma']['val']
            erosion_height_start = data['params']['erosion_height_start']['val']/1000
            erosion_coeff = data['params']['erosion_coeff']['val']
            erosion_mass_index = data['params']['erosion_mass_index']['val']
            erosion_mass_min = data['params']['erosion_mass_min']['val']
            erosion_mass_max = data['params']['erosion_mass_max']['val']

            # from 'time_sampled' extract the last element and save it in a list
            duration = data['time_sampled'][-1]
            begin_height = data['ht_sampled'][0] / 1000
            end_height = data['ht_sampled'][-1] / 1000
            peak_abs_mag = data['mag_sampled'][np.argmin(data['mag_sampled'])]
            F = (begin_height - (data['ht_sampled'][np.argmin(data['mag_sampled'])] / 1000)) / (begin_height - end_height)
            peak_mag_height = data['ht_sampled'][np.argmin(data['mag_sampled'])] / 1000
            beg_abs_mag	= data['mag_sampled'][0]
            end_abs_mag	= data['mag_sampled'][-1]
            trail_len = data['len_sampled'][-1] / 1000
            shower_code = 'sim_'+Shower
            vel_avg_norot = trail_len / duration

            vel_sim=data['simulation_results']['leading_frag_vel_arr']#['brightest_vel_arr']#['leading_frag_vel_arr']#['main_vel_arr']
            ht_sim=data['simulation_results']['leading_frag_height_arr']#['brightest_height_arr']['leading_frag_height_arr']['main_height_arr']
            time_sim=data['simulation_results']['time_arr']#['main_time_arr']

            kc_par = begin_height + (2.86 - 2*np.log(vel_init_norot))/0.0612

            obs_height=data['ht_sampled']
            
            # delete the nan term in vel_sim and ht_sim
            vel_sim=[x for x in vel_sim if str(x) != 'nan']
            ht_sim=[x for x in ht_sim if str(x) != 'nan']

            # # find the index of the first element of the simulation that is equal to the first element of the observation
            index_ht_sim=next(x for x, val in enumerate(ht_sim) if val <= obs_height[0])
            # find the index of the last element of the simulation that is equal to the last element of the observation
            index_ht_sim_end=next(x for x, val in enumerate(ht_sim) if val <= obs_height[-1])

            # time_sim=time_sim[index_ht_sim:index_ht_sim_end]

            ht_sim = [i/1000 for i in ht_sim]
            # find the index_mag_peak in ht_sim that has a value smaller than the peak_mag_height
            # index_mag_peak = next(x for x, val in enumerate(ht_sim) if val <= peak_mag_height)
            index_mag_peak = [i for i in range(len(ht_sim)) if ht_sim[i] < peak_mag_height]
            Dynamic_pressure_peak_abs_mag = data['simulation_results']['leading_frag_dyn_press_arr'][index_mag_peak[0]]

            # pick from the end of vel_sim the same number of element of time_sim
            # vel_sim=vel_sim[-len(time_sim):]

            vel_sim=vel_sim[index_ht_sim:index_ht_sim_end]
            time_sim=time_sim[index_ht_sim:index_ht_sim_end]
            ht_sim=ht_sim[index_ht_sim:index_ht_sim_end]

            # divide the vel_sim by 1000 considering is a list
            vel_sim = [i/1000 for i in vel_sim]
            time_sim = [i-time_sim[0] for i in time_sim]

            # find the sigle index of the height when the velocity start dropping from the vel_init_norot of 0.2 km/s
            # index_knee = next(x for x, val in enumerate(vel_sim) if val <= vel_sim[0]-10)
            index_knee = [i for i in range(len(vel_sim)) if vel_sim[i] < vel_sim[0]-0.2]
            jj_index_knee=2
            # if index_knee == empty start a loop to find one
            while index_knee == []:
                index_knee = [i for i in range(len(vel_sim)) if vel_sim[i] < vel_sim[0]-0.2/jj_index_knee]
                print('index_knee is None so ',0.2/jj_index_knee)
                jj_index_knee=jj_index_knee+1
            index_knee=index_knee[0]
            # only use first index to pick the height
            height_knee_vel = ht_sim[index_knee]
            # find the height of the height_knee_vel in data['ht_sampled']
            index_ht_knee = next(x for x, val in enumerate(data['ht_sampled']) if val/1000 <= height_knee_vel)
            height_knee_vel=data['ht_sampled'][index_ht_knee]/1000
            
            # define thelinear deceleration from that index to the end of the simulation
            a2, b2 = np.polyfit(time_sim[index_knee:],vel_sim[index_knee:], 1)
            decel_after_knee_vel=((-1)*a2)

            # fit a line to the throught the vel_sim and ht_sim
            acceleration, b = np.polyfit(time_sim,vel_sim, 1)
            acceleration_lin = (-1)*acceleration

            a3, b3, c3 = np.polyfit(time_sim,vel_sim, 2)
            acceleration=a3*2+b3

            # fit a line to the throught the vel_sim and ht_sim
            index_ht_peak = next(x for x, val in enumerate(data['ht_sampled']) if val/1000 <= peak_mag_height)
            #print('index_ht_peak',index_ht_peak)
            # only use first index to pick the height
            height_pickl = [i/1000 for i in data['ht_sampled']]

            # check if the height_pickl[:index_ht_peak] and data['mag_sampled'][:index_ht_peak] are empty
            if height_pickl[:index_ht_peak] == [] or data['mag_sampled'][:index_ht_peak] == []:
                a3_Inabs, b3_Inabs, c3_Inabs = 0, 0, 0
            else:
                a3_Inabs, b3_Inabs, c3_Inabs = np.polyfit(height_pickl[:index_ht_peak], data['mag_sampled'][:index_ht_peak], 2)

            # check if the height_pickl[index_ht_peak:] and data['mag_sampled'][index_ht_peak:] are empty
            if height_pickl[index_ht_peak:] == [] or data['mag_sampled'][index_ht_peak:] == []:
                a3_Outabs, b3_Outabs, c3_Outabs = 0, 0, 0
            else:
                a3_Outabs, b3_Outabs, c3_Outabs = np.polyfit(height_pickl[index_ht_peak:], data['mag_sampled'][index_ht_peak:], 2)
            
            # from 'params' extract the physical parameters and save them in a list
            rho = data['params']['rho']['val']
            sigma = data['params']['sigma']['val']
            erosion_height_start = data['params']['erosion_height_start']['val']/1000
            erosion_coeff = data['params']['erosion_coeff']['val']
            erosion_mass_index = data['params']['erosion_mass_index']['val']
            erosion_mass_min = data['params']['erosion_mass_min']['val']
            erosion_mass_max = data['params']['erosion_mass_max']['val']
            erosion_range = np.log10(erosion_mass_max) - np.log10(erosion_mass_min)

            # erosion energy
            const_path = os.path.join(directory, all_jsonfiles[i])

            # Load the constants
            const, _ = loadConstants(const_path)
            const.dens_co = np.array(const.dens_co)
            # Compute the erosion energies
            erosion_energy_per_unit_cross_section, erosion_energy_per_unit_mass = wmpl.MetSim.MetSimErosion.energyReceivedBeforeErosion(const)



            # # find the index of the first element of the simulation that is equal to the first element of the observation
            mag_sampled_norm = [0 if math.isnan(x) else x for x in data['mag_sampled']]
            # normalize the fuction with x data['time_sampled'] and y data['mag_sampled'] and center it at the origin
            time_sampled_norm= data['time_sampled'] - np.mean(data['time_sampled'])
            # subrtract the max value of the mag to center it at the origin
            mag_sampled_norm = (-1)*(mag_sampled_norm - np.max(mag_sampled_norm))
            # normalize the mag so that the sum is 1
            # mag_sampled_norm = mag_sampled_norm/np.sum(mag_sampled_norm)
            mag_sampled_norm = mag_sampled_norm/np.max(mag_sampled_norm)

            # trasform data['mag_sampled'][i] value 'numpy.float64' to int
            # data['mag_sampled'] = data['mag_sampled'].astype(int)

            # create an array with the number the ammount of same number equal to the value of the mag
            mag_sampled_distr = []
            mag_sampled_array=np.asarray(mag_sampled_norm*1000, dtype = 'int')
            # i_pos=(-1)*np.round(len(data['mag_sampled'])/2)
            for i in range(len(data['mag_sampled'])):
                # create an integer form the array mag_sampled_array[i] and round of the given value
                numbs=mag_sampled_array[i]
                # invcrease the array number by the mag_sampled_distr numbs 
                array_nu=(np.ones(numbs+1)*time_sampled_norm[i])#.astype(int)
                mag_sampled_distr=np.concatenate((mag_sampled_distr, array_nu))
                # i_pos=i_pos+1

            kurtosyness=kurtosis(mag_sampled_distr)
            skewness=skew(mag_sampled_distr)


            # add a new line in dataframe
            df_json.loc[len(df_json)] = [name,shower_code, vel_init_norot, vel_avg_norot, duration,\
            mass, peak_mag_height,begin_height, end_height, height_knee_vel, peak_abs_mag, beg_abs_mag, end_abs_mag,\
            F, trail_len, acceleration, acceleration_lin, decel_after_knee_vel, zenith_angle, kurtosyness,skewness,\
            kc_par, Dynamic_pressure_peak_abs_mag,\
            a3, b3, a3_Inabs, b3_Inabs, a3_Outabs, b3_Outabs, rho, sigma,\
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
    # vel_init_norot	vel_avg_norot	duration	mass	peak_mag_height	begin_height	end_height	height_knee_vel	peak_abs_mag	beg_abs_mag	end_abs_mag	F	trail_len	acceleration	decel_after_knee_vel	zenith_angle	kurtosis	skew	kc	Dynamic_pressure_peak_abs_mag
    variable_PCA=[] #'vel_init_norot','peak_abs_mag','begin_height','end_height','F','acceleration','duration'
    # variable_PCA=['vel_init_norot','peak_abs_mag','zenith_angle','peak_mag_height','acceleration','duration','vel_avg_norot','begin_height','end_height','beg_abs_mag','end_abs_mag','F','Dynamic_pressure_peak_abs_mag']
    # variable_PCA=['vel_init_norot','peak_abs_mag','zenith_angle','peak_mag_height','acceleration','duration','Dynamic_pressure_peak_abs_mag'] # perfect!
    # variable_PCA=['vel_init_norot','peak_abs_mag','zenith_angle','peak_mag_height','acceleration','duration','Dynamic_pressure_peak_abs_mag','kurtosis','skew','trail_len']
    # decel_after_knee_vel and height_knee_vel create errors in the PCA space  decel_after_knee_vel,height_knee_vel

    No_var_PCA=['decel_after_knee_vel','height_knee_vel','acceleration_lin']
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
    pca= PCA(n_components=np.argmax(cumulative_variance >= 0.97) + 1)
    all_PCA = pca.fit_transform(scaled_df_all)

    # # select only the column with in columns_PC with the same number of n_components
    columns_PC = ['PC' + str(x) for x in range(1, pca.n_components_+1)]

    # create a dataframe with the PCA space
    df_all_PCA = pd.DataFrame(data = all_PCA, columns = columns_PC)

    print(str(len(percent_variance))+' PC = 95% of the variance explained by ',pca.n_components_,' PC')

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

        # # find the average distance of the closest df_sim_PCA_val to each df_sim_PCA_val
        # distance_current_sim = []
        # # but only select a random number of simulation equal to reduce the computational time
        # df_sim_PCA_val_rand_dist = df_sim_PCA_val[np.random.choice(len(df_sim_PCA_val), 10, replace=False)]
        # print('sim considered:',len(df_sim_PCA_val_rand_dist))
        # for i_sim_dist_base in range(len(df_sim_PCA_val_rand_dist)):
        #     distance_current_sim_curr = []
        #     for i_sim_dist_curr in range(len(df_sim_PCA_val)):
        #         distance_current_sim_curr.append(scipy.spatial.distance.euclidean(df_sim_PCA_val[i_sim_dist_curr], df_sim_PCA_val_rand_dist[i_sim_dist_base]))
        #     # order the distance from the closest to the farest
        #     distance_current_sim_curr = sorted(distance_current_sim_curr)
        #     # take the mean of the first 10 closest and do the mean
        #     distance_current_sim.append(np.mean(distance_current_sim_curr[1]))
        #     print('Processing ',i_sim_dist_base,' the dist: ', np.round(distance_current_sim[i_sim_dist_base],2), end="\r")
        # print('Of ',len(df_sim_PCA_val_rand_dist),' the mean distance of the closest neightbor :', np.round(np.mean(distance_current_sim),2))

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

            # # sort the distance and select the n_selected closest to the mean
            # df_sim_shower_dis = df_sim_shower.sort_values(by=['distance_meteor'])
            # # drop the index
            # df_sim_shower_dis = df_sim_shower_dis.reset_index(drop=True)

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
        df_sel_PCA.to_csv(OUT_PUT_PATH+r'\\Simulated_'+current_shower+'_select_PCA.csv', index=False)

        # concatenate the list of the properties to a dataframe
        df_sel_shower = pd.concat(df_sel_shower)

        df_sel_PCA_NEW = df_sel_PCA.drop(['shower_code'], axis=1)
        # create a list of the selected shower
        df_sel_PCA_NEW = df_sel_PCA_NEW.values
        distance_current = []
        for i_shower in range(len(df_sel_shower)):
            distance_current.append(scipy.spatial.distance.euclidean(meanPCA_current, df_sel_PCA_NEW[i_shower]))
        df_sel_shower['distance']=distance_current # from the mean of the selected shower
        # save the dataframe to a csv file withouth the index
        df_sel_shower.to_csv(OUT_PUT_PATH+r'\\Simulated_'+current_shower+'_select.csv', index=False)

        # save dist also on selected shower
        distance_current = []
        for i_shower in range(len(shower_current)):
            distance_current.append(scipy.spatial.distance.euclidean(meanPCA_current, shower_current_PCA[i_shower]))
        shower_current['distance']=distance_current # from the mean of the selected shower
        shower_current.to_csv(OUT_PUT_PATH+r'\\'+current_shower+'_and_dist.csv', index=False)

    # copy Simulated_PER.csv in OUT_PUT_PATH
    shutil.copyfile(INPUT_PATH+r'\\Simulated_PER.csv', OUT_PUT_PATH+r'\\Simulated_PER.csv')


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
    pca= PCA(n_components=np.argmax(cumulative_variance >= 0.97) + 1)
    all_PCA = pca.fit_transform(df_all)

    # # select only the column with in columns_PC with the same number of n_components
    columns_PC = ['PC' + str(x) for x in range(1, pca.n_components_+1)]

    # create a dataframe with the PCA space
    df_all_PCA = pd.DataFrame(data = all_PCA, columns = columns_PC)

    print(str(len(percent_variance))+' PC = 95% of the variance explained by ',pca.n_components_,' PC')

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
        if os.path.isfile(cml_args.input_dir+r'\\Simulated_'+current_shower+'.csv'):
            # if yes read the csv file
            df_sim = pd.read_csv(cml_args.input_dir+r'\\Simulated_'+current_shower+'.csv')
        else:
            # open the folder and extract all the json files
            os.chdir(folder_GenerateSimulations_json[Shower.index(current_shower)])
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


            df_sim.to_csv(cml_args.input_dir+r'\\Simulated_'+current_shower+'.csv', index=False)



        if os.path.isfile(cml_args.input_dir+r'\\'+current_shower+'.csv'):
            # if yes read the csv file
            df_obs = pd.read_csv(cml_args.input_dir+r'\\'+current_shower+'.csv')
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