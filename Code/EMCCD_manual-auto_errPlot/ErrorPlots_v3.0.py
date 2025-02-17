import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import wmpl
import seaborn as sns
from heapq import nsmallest
from heapq import nlargest
import math
from scipy.stats import kurtosis, skew
from scipy.stats import norm
from scipy import stats
import os
from wmpl.Utils.Math import vectMag, findClosestPoints
from wmpl.Utils.Pickling import loadPickle
from dataclasses import dataclass
from os import walk

##############################################################################################################

path_init='C:\\Users\\maxiv\\Documents\\UWO\\Papers\\1)PCA\\Error_acceleration\\'

##############################################################################################################




# # list of the .txt files in the directory
# files_pickle = [f for f in os.listdir(path) if f.endswith('.pickle')]
@dataclass
class obs_cxcy:
    Station: str
    seconds: np.array
    x_image: np.array
    y_image: np.array
    mag: np.array
    distance: np.array
    diff_distance: np.array
    distance_point: np.array 
    distance_init: np.array     

def load_ecsv(file):
    # open the first file
    f=open(file)
    contents = f.read()
    # extract camera_id: 01G
    Station1 = contents[contents.find('# - {camera_id: ')+1:contents.find('#')-1]
    # delete everithing after #
    Station1 = Station1[:Station1.find('}')-1]
    # keep everthing after :
    Station1 = Station1[Station1.find(': \'')+3:]

    Station1 = ''.join(filter(str.isdigit, Station1))

    # data with # is a comment and delimiter is a comma in the file
    ecsv = pd.read_csv(file, comment='#', delimiter=',')
    # keep only the columns 'datetime','x_image', 'y_image'
    ecsv = ecsv[['datetime','x_image', 'y_image','mag_data']]
    # rename the columns
    ecsv.columns = ['datetime','x_image', 'y_image','mag']
    # # split the sacond column into two columns at the . and keep the . in the second column
    # ecsv[['datetime','seconds']] = ecsv.datetime.str.split(".",expand=True)
    # split the date time by : and keep the last one
    ecsv['seconds'] = ecsv['datetime'].str.split(':').str[-1]
    # # add to seconds2 a . in the front
    # ecsv['seconds'] = '.' + ecsv['seconds']
    # convert the seconds column to a float
    ecsv['seconds'] = ecsv['seconds'].astype(float)
    # delete datetime column 
    del ecsv['datetime']
    # add the seconds column to the first column
    ecsv.insert(0, 'seconds', ecsv.pop('seconds'))
    

    # plot the data in a plot with the distance in sqrt(x^2 + y^2) image
    ecsv['distance_init'] = np.sqrt((ecsv['x_image']-ecsv['x_image'][0])**2 + (ecsv['y_image']-ecsv['y_image'][0])**2)
    # add a column to the dataframe with the distance
    ecsv['distance'] = np.sqrt(ecsv['x_image']**2 + ecsv['y_image']**2)
    # print(ecsv['distance'].diff())
    # print(ecsv['seconds'].diff())
    # difference of distance to the previous point
    ecsv['diff_distance'] = ecsv['distance'].diff()/ecsv['seconds'].diff()
    # print(np.sqrt((ecsv['x_image'].diff())**2 + (ecsv['y_image'].diff())**2))
    # distance between points
    ecsv['distance_point'] = np.sqrt((ecsv['x_image'].diff())**2 + (ecsv['y_image'].diff())**2)/ecsv['seconds'].diff()

    obs_man=obs_cxcy(Station=Station1, seconds=ecsv['seconds'], x_image=ecsv['x_image'], y_image=ecsv['y_image'], 
             mag=ecsv['mag'], distance=ecsv['distance'], diff_distance=ecsv['diff_distance'],  
             distance_point=ecsv['distance_point'], distance_init=ecsv['distance_init'])

    return obs_man

def load_txt(file):
    f=open(file)
    contents = f.read()
    # take the data time : 20180813 07:19:44.282 UTC
    # and convert it to a datetime object
    Station2 = contents[contents.find('text :')+1:contents.find('#')-1]
    # keep everthing after :
    Station2 = Station2[Station2.find(':')+1:]
    # delete everithing after #
    Station2 = Station2[:Station2.find('i')-0]
    if Station2 == ' tav' or Station2 == ' Tav':
        Station2 = '01'
    elif Station2 == ' Elg' or Station2 == ' elg':
        Station2 = '02'


    time = contents[contents.find('time :')+1:contents.find('UTC')-1]
    # split by : and keep the last one
    time = time.split(':')[-1]
    # to number
    time = float(time)
    # save the data after the # in a pd dataframe
    dftxt = pd.read_csv(file, comment='#', delim_whitespace=True, header=None)
    # hold only column 1 4 and 5
    dftxt = dftxt[[1,4,5,9]]
    # rename the columns
    dftxt.columns = ['seconds','x_image', 'y_image','mag']
    # sum the time from the time column
    dftxt['seconds'] = time + dftxt['seconds']

    dftxt['distance_init'] = np.sqrt((dftxt['x_image']-dftxt['x_image'][0])**2 + (dftxt['y_image']-dftxt['y_image'][0])**2)

    # add a column to the dataframe with the distance
    dftxt['distance'] = np.sqrt(dftxt['x_image']**2 + dftxt['y_image']**2)
    # difference of distance to the previous point
    # print(dftxt['distance'].diff())
    # print(dftxt['seconds'].diff())
    dftxt['diff_distance'] = dftxt['distance'].diff()/dftxt['seconds'].diff()
    # distance between points
    # print(np.sqrt((dftxt['x_image'].diff())**2 + (dftxt['y_image'].diff())**2))
    dftxt['distance_point'] = np.sqrt((dftxt['x_image'].diff())**2 + (dftxt['y_image'].diff())**2)/dftxt['seconds'].diff()

    obs_auto=obs_cxcy(Station=Station2, seconds=dftxt['seconds'], x_image=dftxt['x_image'], y_image=dftxt['y_image'], 
             mag=dftxt['mag'], distance=dftxt['distance'], diff_distance=dftxt['diff_distance'],  
             distance_point=dftxt['distance_point'], distance_init=dftxt['distance_init'])

    return obs_auto

# def deceleration_pickle(file_pickle):
#     traj = loadPickle(*os.path.split(file_pickle))
#     vel_pickl=[]
#     time_pickl=[]
#     for obs in traj.observations:
#         # put it at the end obs.velocities[1:] at the end of vel_pickl list
#         vel_pickl.extend(obs.velocities[1:])
#         time_pickl.extend(obs.time_data[1:])

#     # compute the linear regression
#     vel_pickl = [i/1000 for i in vel_pickl] # convert m/s to km/s
#     time_pickl = [i for i in time_pickl]

#     # fit a line to the throught the vel_sim and ht_sim
#     a, b = np.polyfit(time_pickl,vel_pickl, 1)
#     deceleration=((-1)*a)
#     return deceleration, vel_pickl[0], vel_pickl[-1]

def plot_reg_line(data_clas):
    m, b = np.polyfit(data_clas.seconds[1:],data_clas.distance_point[1:], 1)
    #add linear regression line to scatterplot 
    plt.plot(m*data_clas.seconds+b, data_clas.seconds, color='black', linewidth=0.5, linestyle='dashed')


def plot_diff_position(data_man, data_auto, Station):
    # create an empty panda data frame called a
    ecsv = pd.DataFrame()
    dftxt = pd.DataFrame()

    # add data_man.seconds and data_man.distance to the data frame
    ecsv['seconds']= data_man.seconds
    ecsv['distance']= data_man.distance
    dftxt['seconds']= data_auto.seconds
    dftxt['distance']= data_auto.distance

    ecsv['seconds'] = ecsv['seconds'].round(2)

    dftxt['seconds'] = dftxt['seconds'].round(2)

    # find the index of ecsv['seconds'] that is the same of dftxt['seconds']
    index_ecsv = ecsv[ecsv['seconds'].isin(dftxt['seconds'])].index.tolist()

    index_txt = dftxt[dftxt['seconds'].isin(ecsv['seconds'])].index.tolist()

    # use the index to find the difference of the distance
    dist_ecsv=ecsv['distance'][index_ecsv]

    dist_ecsv=dist_ecsv.reset_index(drop=True)

    dist_dftxt=dftxt['distance'][index_txt]

    dist_dftxt=dist_dftxt.reset_index(drop=True)

    # delete nan
    dist_diff=dist_ecsv-dist_dftxt

    # time index of the difference
    time_diff=dftxt['seconds'][index_txt]

    # delete index
    time_diff=time_diff.reset_index(drop=True)

    plt.plot(dist_diff, time_diff, label='auto-man '+Station, linestyle='dashed', marker='x')


def deceleration_pickle(file_pickle):
    traj = loadPickle(*os.path.split(file_pickle))
    vel_pickl=[]
    time_pickl=[]
    for obs in traj.observations:
        # put it at the end obs.velocities[1:] at the end of vel_pickl list
        vel_pickl.extend(obs.velocities[1:])
        time_pickl.extend(obs.time_data[1:])

    # compute the linear regression
    vel_pickl = [i/1000 for i in vel_pickl] # convert m/s to km/s
    time_pickl = [i for i in time_pickl]

    # fit a line to the throught the vel_sim and ht_sim
    a, b = np.polyfit(time_pickl,vel_pickl, 1)
    deceleration=((-1)*a)
    # plot the line with the data
    # plt.plot(time_pickl, a*np.array(time_pickl)+b, color='black', linewidth=0.5, linestyle='dashed')
    # sort from the smallest to the biggest time_pickl
    time_pickl, vel_pickl = zip(*sorted(zip(time_pickl, vel_pickl)))


    plt.subplot(2, 3, 5)
    vel2o=a*np.array(time_pickl)+b
    plt.plot(vel2o, time_pickl, color='black', linewidth=0.5, linestyle='dashed')

    return deceleration, vel2o[0], vel2o[-1]    





def plot_ErrorPlots(files_manual, files_auto, path, path_init):

    plt.figure(figsize=(12, 8))

    # add to the files_manual the path
    files_manual = [path+'\\'+i for i in files_manual]
    # add to the files_auto the path
    files_auto = [path+'\\'+i for i in files_auto]

    for ii in range(len(files_manual)):
        obs_man = load_ecsv(files_manual[ii])
        if obs_man.Station == '01':
            obs_man_01=obs_man
        elif obs_man.Station == '02':
            obs_man_02=obs_man

    for ii in range(len(files_auto)):
        obs_auto = load_txt(files_auto[ii])
        if obs_auto.Station == '01':
            obs_auto_01=obs_auto
        elif obs_auto.Station == '02':
            obs_auto_02=obs_auto
    

    plt.subplot(2, 3, 1) # position on CCD camera
    plt.scatter(obs_man_01.x_image[0], obs_man_01.y_image[0], marker='x')
    # plt.text(obs_man_01.x_image[0], obs_man_01.y_image[0], np.round(obs_man_01.x_image[0],2)+';'+np.round(obs_man_01.y_image[0],2))
    plt.scatter(obs_man_02.x_image[0], obs_man_02.y_image[0], marker='x')
    # plt.text(obs_man_02.x_image[0], obs_man_02.y_image[0], np.round(obs_man_02.x_image[0],2)+';'+np.round(obs_man_02.y_image[0],2))
    plt.scatter(obs_auto_01.x_image[0], obs_auto_01.y_image[0], marker='x')
    # plt.text(obs_auto_01.x_image[0], obs_auto_01.y_image[0], str(np.round(obs_auto_01.x_image[0],2))+';'+str(np.round(obs_auto_01.y_image[0],2)),color = 'gray')
    # plt.text(obs_auto_01.x_image[len(obs_auto_01.x_image)-1], obs_auto_01.y_image[len(obs_auto_01.y_image)-1], str(np.round(obs_auto_01.x_image[len(obs_auto_01.x_image)-1],2))+';'+str(np.round(obs_auto_01.y_image[len(obs_auto_01.y_image)-1],2)),color = 'gray')
    plt.scatter(obs_auto_02.x_image[0], obs_auto_02.y_image[0], marker='x')
    # plt.text(obs_auto_02.x_image[0], obs_auto_02.y_image[0], str(np.round(obs_auto_02.x_image[0],2))+';'+str(np.round(obs_auto_02.y_image[0],2)),color = 'gray')
    # plt.text(obs_auto_02.x_image[len(obs_auto_02.x_image)-1], obs_auto_02.y_image[len(obs_auto_02.y_image)-1], str(np.round(obs_auto_02.x_image[len(obs_auto_02.x_image)-1],2))+';'+str(np.round(obs_auto_02.y_image[len(obs_auto_02.y_image)-1],2)),color = 'gray')
    plt.gca().set_prop_cycle(None)
    plt.scatter(obs_man_01.x_image, obs_man_01.y_image, s=100000*(0.1**obs_man_01.mag), label='manual '+obs_man_01.Station, marker='d')
    plt.scatter(obs_man_02.x_image, obs_man_02.y_image, s=100000*(0.1**obs_man_02.mag), label='manual '+obs_man_02.Station, marker='d')
    plt.scatter(obs_auto_01.x_image, obs_auto_01.y_image, s=100000*(0.1**obs_auto_01.mag), label='auto '+obs_auto_01.Station, marker='o')
    plt.scatter(obs_auto_02.x_image, obs_auto_02.y_image, s=100000*(0.1**obs_auto_02.mag), label='auto '+obs_auto_02.Station, marker='o')


    plt.subplot(2, 3, 2) # dist in pixel
    plt.plot(obs_man_01.distance_init, obs_man_01.seconds, label='manual '+obs_man_01.Station, marker='d')
    plt.plot(obs_man_02.distance_init, obs_man_02.seconds, label='manual '+obs_man_02.Station, marker='d')
    plt.plot(obs_auto_01.distance_init, obs_auto_01.seconds, label='auto '+obs_auto_01.Station, marker='o',linestyle='dashed')
    plt.plot(obs_auto_02.distance_init, obs_auto_02.seconds, label='auto '+obs_auto_02.Station, marker='o',linestyle='dashed')

    plt.subplot(2, 3, 4) # derivative of dist in pixel
    plt.plot(obs_man_01.distance_point, obs_man_01.seconds, label='manual '+obs_man_01.Station, marker='d')
    plot_reg_line(obs_man_01)
    plt.plot(obs_man_02.distance_point, obs_man_02.seconds, label='manual '+obs_man_02.Station, marker='d')
    plot_reg_line(obs_man_02)
    plt.plot(obs_auto_01.distance_point, obs_auto_01.seconds, label='auto '+obs_auto_01.Station, marker='o',linestyle='dashed')
    plot_reg_line(obs_auto_01)
    plt.plot(obs_auto_02.distance_point, obs_auto_02.seconds, label='auto '+obs_auto_02.Station, marker='o',linestyle='dashed')
    plot_reg_line(obs_auto_02)


    plt.subplot(2, 3, 3)
    plot_diff_position(obs_man_01,obs_auto_01,'01')
    plot_diff_position(obs_man_02,obs_auto_02,'02')



    # list of the .txt files in the directory
    files_pickle = [f for f in os.listdir(path) if f.endswith('.pickle')]

    # add to the files_auto the path
    files_pickle = [path+'\\'+i for i in files_pickle]

    # find the pickle file with _sim in the name
    for ii in range(len(files_pickle)):
        if '_sim' in files_pickle[ii]:
            pkl_auto_path=files_pickle[ii]
        else:
            pkl_man_path = files_pickle[ii]

    # Define the path to the pickle files
    # pkl_man_path = "20180813_071944_trajectory.pickle"
    # pkl_auto_path = "20180813_071944_trajectory_sim.pickle"



    # Load the pickle files
    traj_man = loadPickle(*os.path.split(pkl_man_path))
    traj_auto = loadPickle(*os.path.split(pkl_auto_path))




##########################################################################################################################################################
# DENIS code
##########################################################################################################################################################

    # print('pickle manual test :',files_pickle_manual)

    # print('pickle auto test :',files_pickle_auto)

    # # split base on / and take the last one
    # name_man_pik = files_pickle_manual.split('/')[-1]
    # # add the rest of path.split('/') withouth considering the last one
    # path_man_pik = '/'.join(files_pickle_manual.split('/')[:-1])

    # # split base on / and take the last one
    # name_auto_pik = files_pickle_auto.split('/')[-1]
    # # add the rest of path.split('/') withouth considering the last one
    # path_auto_pik = '/'.join(files_pickle_auto.split('/')[:-1])


    # traj_man = loadPickle(dir_path=path_man_pik, file_name=name_man_pik)
    # traj_auto = loadPickle(dir_path=path_auto_pik, file_name=name_auto_pik)

# IDK they are opposite but it is ok, or the lag will be opposite manual and auto
    # traj_auto = loadPickle(dir_path=path_man_pik, file_name=name_man_pik)
    # traj_man = loadPickle(dir_path=path_auto_pik, file_name=name_auto_pik)




    # Choose the trajectory with the highest state vector as reference
    if vectMag(traj_man.state_vect_mini) > vectMag(traj_auto.state_vect_mini):
        traj_ref = traj_man
        print("Reference trajectory: manual")
    else:
        traj_ref = traj_auto
        print("Reference trajectory: auto")

    # Find common stations between the two trajectories
    traj_man_stations = set([obs.station_id for obs in traj_man.observations])
    traj_auto_stations = set([obs.station_id for obs in traj_auto.observations])
    common_stations = traj_man_stations.intersection(traj_auto_stations)


    # Recompute the time and state vector distances as reference to the reference trajectory
    traj_list = [
        [traj_man, 'manual'], 
        [traj_auto, 'auto']
        ]
    for traj, traj_type in traj_list:

        print("Processing {} trajectory...".format(traj_type))
        
        # NOTE: Simply synching the time by comparing the reference JDs doesn't work for some reason

        ## Synchronize the time by finding the time offset between the heights of the middle points on the trajectory
        ## and the reference trajectory
        

        # Find the longest observation in the trajectory, only if the station is common to both trajectories
        # obs_longest = None
        # for obs in traj.observations:
        #     if obs.station_id in common_stations:
        #         if obs_longest is None or len(obs.time_data) > len(obs_longest.time_data):
        #             obs_longest = obs
        #             print(obs_longest)
        obs_longest = None
        obs_heights = 0
        for obs in traj_ref.observations:
            if obs.rbeg_ele > obs_heights:
                obs_heights = obs.rbeg_ele
                obs_longest = obs
                print(obs_heights, obs_longest.station_id)


        # Find the matching observation from the same station in the reference trajectory
        obs_ref = None
        for obs in traj_ref.observations:
            print(obs.station_id, obs_longest.station_id)
            if obs.station_id == obs_longest.station_id:
                obs_ref = obs
                break

        obs_ref = None
        obs_heights = 0
        for obs in traj_ref.observations:
            if obs.rbeg_ele > obs_heights:
                obs_heights = obs.rbeg_ele
                obs_ref = obs
                print(obs_heights, obs_ref.station_id)



        # Find the middle point of the longest observation
        middle_idx = int(len(obs_longest.time_data)/2)
        middle_time = obs_longest.time_data[middle_idx]
        middle_ht = obs_longest.meas_ht[middle_idx]

        # Find the height in the reference trajectory closest to the middle height of the longest observation
        ref_ht_diff = np.abs(obs_ref.meas_ht - middle_ht)
        ref_ht_idx = np.argmin(ref_ht_diff)
        ref_time = obs_ref.time_data[ref_ht_idx]

        # Compute the time offset from the reference trajectory
        dt_s = ref_time - middle_time

        ##
        



        # # Compute the time offset from the reference trajectory
        # #dt_s = (traj.jdt_ref - traj_ref.jdt_ref)*86400
        # # print("delta T for {}: {:.2f} s".format(traj_type, dt_s))
        # if traj_type == 'manual':
        #     #dt_s = -0.03125*3
        #     dt_s = 0.03125*3
        # else:
        #     dt_s = 0.0

        print("delta T for {}: {:.2f} s".format(traj_type, dt_s))

        for obs in traj.observations:

            # Compute the normalized time
            obs.time_data = obs.time_data + dt_s

            # Compute the state vector distance from the reference state vector
            sv_dist = []
            
            # Go through all individual position measurement from each site
            for i, (stat, meas) in enumerate(zip(obs.stat_eci_los, obs.meas_eci_los)):

                # Calculate closest points of approach (observed line of sight to radiant line)
                obs_cpa, rad_cpa, d = findClosestPoints(stat, meas, 
                                                        traj_ref.state_vect_mini, traj_ref.radiant_eci_mini)
                
                # Calculate the distance between the point on the trajectory and the reference state vector
                dist = vectMag(obs_cpa - traj_ref.state_vect_mini)
                sv_dist.append(dist)

            sv_dist = np.array(sv_dist)

            # Save the normalized state vector distance
            obs.state_vect_dist = sv_dist


    ## Find the constant offset between the state vector distances of the two trajectories and subtract it
    ## from the manual trajectory
    time_auto = np.concatenate([obs.time_data for obs in traj_auto.observations])
    sv_dist_auto = np.concatenate([obs.state_vect_dist for obs in traj_auto.observations])
    time_man = np.concatenate([obs.time_data for obs in traj_man.observations])
    sv_dist_man = np.concatenate([obs.state_vect_dist for obs in traj_man.observations])

    # Fit a line to the auto trajectory
    fit = np.polyfit(time_auto, sv_dist_auto, 1)
    fit_fn = np.poly1d(fit)

    # Subtract the auto trajectory from the manual trajectory
    sv_dist_diff = sv_dist_man - fit_fn(time_man)

    # Compute the mean offset
    mean_offset = np.mean(sv_dist_diff)

    # Apply the offset to the state vector distance of the manual trajectory
    for obs in traj_man.observations:
        obs.state_vect_dist -= mean_offset

    ##




    # Init a plot with two subplots side by side
    # fig, (ax_svd, ax_lag) = plt.subplots(ncols=2, figsize=(12, 6), sharey=True)

    # plt.subplot(2, 3, 5)
    # plt.title('State vector distance')

    # # Plot the state vector distance as a function of time
    # for traj, traj_type in traj_list:
    #     for obs in traj.observations:
    #         plt.plot(obs.state_vect_dist, obs.time_data, 'x', label=traj_type + ' ' + obs.station_id)
    #         # ax_svd.plot(obs.state_vect_dist, obs.time_data, 'x', label=traj_type + ' ' + obs.station_id)

    # # ax_svd.set_ylabel('Time (s)')
    # # ax_svd.set_xlabel('State vector distance (m)')
    # plt.ylabel('Time (s)')
    # plt.xlabel('State vector distance (m)')
    # plt.grid(True)

    plt.subplot(2, 3, 5)
    plt.title('velocity vs time')

    # ax_svd.set_ylabel('Time (s)')
    # ax_svd.set_xlabel('State vector distance (m)')
    plt.ylabel('Time (s)')
    plt.xlabel('Velocity (km/s)')
    plt.grid(True)


    # Concatenate all time and state vector distance data into single arrays
    time_concat = np.concatenate([obs.time_data for obs in traj.observations for traj, _ in traj_list])
    sv_dist_concat = np.concatenate([obs.state_vect_dist for obs in traj.observations for traj, _ in traj_list])

    # Sort the points by time
    sort_idx = np.argsort(time_concat)
    time_concat = time_concat[sort_idx]
    sv_dist_concat = sv_dist_concat[sort_idx]



    # Fit a line to 50% of the first data points
    fit_idx = int(len(time_concat)*0.5)
    fit_idx = np.arange(fit_idx)

    # Fit a line to the data
    fit = np.polyfit(time_concat[fit_idx], sv_dist_concat[fit_idx], 1)
    fit_fn = np.poly1d(fit)

    # Plot the fit line
    time_arr = np.linspace(time_concat[0], time_concat[-1], 100)
    # ax_svd.plot(fit_fn(time_arr), time_arr, color='k', label='Fit', alpha=0.5)
    # ax_svd.legend()
    # plt.plot(fit_fn(time_arr), time_arr, color='k', label='Fit', alpha=0.5)
    plt.gca().invert_yaxis()
    plt.legend()



    plt.subplot(2, 3, 6)
    plt.title('Lag')
    # Compute the lag - subtract the line from the state vector distance per each observation
    for traj, _ in traj_list:
        for obs in traj.observations:
            obs.lag = obs.state_vect_dist - fit_fn(obs.time_data)


    # Plot the lag as a function of time for each observation

    for traj, traj_type in traj_list:

        if traj_type == 'manual':
            marker = 'd'
            linestyle = 'solid'
        elif traj_type == 'auto':
            marker = 'o'
            linestyle = '--' 
        else:
            marker = 'x'
            linestyle = '-.'

        # Reset the matplotlib color cycle to start from the beginning
        # ax_lag.set_prop_cycle(None)
        # plt.gca().set_prop_cycle(None)


        for obs in traj.observations:
            # ax_lag.plot(obs.lag, obs.time_data, label=traj_type + ' ' + obs.station_id, marker=marker, 
            #             linestyle=linestyle)
# find if inside obs.station_id there is a 1 or 2 
            if traj_type == 'manual':
                plt.subplot(2, 3, 6)
                if '1' in obs.station_id:
                    plt.plot(obs.lag, obs.time_data, label=traj_type + ' ' + obs.station_id, marker=marker, 
                            linestyle=linestyle, color='tab:blue')
                elif '2' in obs.station_id:
                    plt.plot(obs.lag, obs.time_data, label=traj_type + ' ' + obs.station_id, marker=marker, 
                            linestyle=linestyle, color='tab:orange')
                plt.subplot(2, 3, 5)
                obs.velocities[1:]=[i/1000 for i in obs.velocities[1:]]
                if '1' in obs.station_id:
                    plt.plot(obs.velocities[1:], obs.time_data[1:], label=traj_type + ' ' + obs.station_id, marker=marker, 
                            linestyle=linestyle, color='tab:blue')
                elif '2' in obs.station_id:
                    plt.plot(obs.velocities[1:], obs.time_data[1:], label=traj_type + ' ' + obs.station_id, marker=marker, 
                            linestyle=linestyle, color='tab:orange')

                    
            if traj_type == 'auto':
                plt.subplot(2, 3, 6)
                if '1' in obs.station_id:
                    plt.plot(obs.lag, obs.time_data, label=traj_type + ' ' + obs.station_id, marker=marker, 
                            linestyle=linestyle, color='tab:green')
                elif '2' in obs.station_id:
                    plt.plot(obs.lag, obs.time_data, label=traj_type + ' ' + obs.station_id, marker=marker, 
                            linestyle=linestyle, color='tab:red')
                plt.subplot(2, 3, 5)
                obs.velocities[1:]=[i/1000 for i in obs.velocities[1:]]
                if '1' in obs.station_id:
                    plt.plot(obs.velocities[1:], obs.time_data[1:], label=traj_type + ' ' + obs.station_id, marker=marker, 
                            linestyle=linestyle, color='tab:green')
                elif '2' in obs.station_id:
                    plt.plot(obs.velocities[1:], obs.time_data[1:], label=traj_type + ' ' + obs.station_id, marker=marker, 
                            linestyle=linestyle, color='tab:red')
                            

            # plt.plot(obs.lag, obs.time_data, label=traj_type + ' ' + obs.station_id, marker=marker, 
            #             linestyle=linestyle)

    plt.subplot(2, 3, 5)
    plt.legend()
    
    

    plt.subplot(2, 3, 6)
    # ax_lag.set_ylabel('Time (s)')
    # ax_lag.set_xlabel('Lag (m)')
    plt.ylabel('Time (s)')
    plt.xlabel('Lag (m)')

    # Invert y axis
    # ax_lag.invert_yaxis()
    plt.gca().invert_yaxis()

    # ax_lag.legend()
    # ax_lag.grid()
    plt.legend()
    plt.grid()

    # plt.tight_layout()
    # plt.show()

    # plt.subplot(2, 3, 3)
    # # plt.title('Lag')
    # # plt.ylabel('time [s]')
    # # plt.xlabel('lag [m]')
    # # plt.grid(True)
    # # plt.legend()

    # plt.subplot(2, 3, 6)
    # plt.title('Lag residuals')
    # plt.ylabel('time [s]')
    # plt.xlabel('lag [m]')
    # plt.grid(True)
    # plt.legend()


##################################################################################################################################################


    man_decel,man_Ivel,man_Fvel=deceleration_pickle(pkl_man_path)
    auto_decel,auto_Ivel,auto_Fvel=deceleration_pickle(pkl_auto_path)

    print('auto   :\tinit.vel=\t',np.round(auto_Ivel,2),'\tkm/s | fin.vel=\t',np.round(auto_Fvel,2),'\tkm/s | decel.=\t', np.round(auto_decel,2),'\tkm/s^2')
    print('manual :\tinit.vel=\t',np.round(man_Ivel,2),'\tkm/s | fin.vel=\t',np.round(man_Fvel,2),'\tkm/s | decel.=\t', np.round(man_decel,2),'\tkm/s^2')

    textstr = 'auto : ',np.round(auto_Ivel,1),' - ',np.round(auto_Fvel,1),' km/s ', np.round(auto_decel,1),' km/s^2 | manual : ',np.round(man_Ivel,1),' - ',np.round(man_Fvel,1),' km/s ', np.round(man_decel,1),' km/s^2'
    # join the text
    textstr = ''.join(str(e) for e in textstr)
    plt.suptitle(textstr)


    plt.subplot(2, 3, 1)
    # make a title of the whole plot split by \
    plt.title(path.split('\\')[-1])
    plt.ylabel('y')
    plt.xlabel('x')
    # x limits
    plt.xlim(0, 512)
    # y limits
    plt.ylim(0, 512)
    # background black
    plt.gca().set_facecolor('black')
    # must be square
    plt.gca().set_aspect('equal', adjustable='box')
    plt.gca().invert_yaxis()
    plt.grid(True)
    plt.legend()


    plt.subplot(2, 3, 2)
    plt.title('distance from the origin')
    plt.ylabel('seconds')
    plt.xlabel('pixel')
    plt.grid(True)
    plt.gca().invert_yaxis()
    plt.legend()

    plt.subplot(2, 3, 4)
    plt.title('first derivative')
    plt.ylabel('seconds')
    plt.xlabel('pixel per second')
    plt.grid(True)
    plt.gca().invert_yaxis()
    plt.legend()

    plt.subplot(2, 3, 3)
    plt.title('difference automatic - manual')
    plt.ylabel('seconds')
    plt.xlabel('pixel')
    plt.grid(True)
    plt.legend()
    # find the max between dftxt['seconds'] and ecsv['seconds']
    max_time = max(obs_man_01.seconds.max(), obs_man_02.seconds.max(), obs_auto_01.seconds.max(), obs_auto_02.seconds.max())
    # find the min between dftxt['seconds'] and ecsv['seconds']
    min_time = min(obs_man_01.seconds.min(), obs_man_02.seconds.min(), obs_auto_01.seconds.min(), obs_auto_02.seconds.min())
    # set y limit
    plt.ylim(min_time, max_time)
    plt.gca().invert_yaxis()
    # make it visible the labels
    plt.tight_layout()

    # # full screen
    # mng = plt.get_current_fig_manager()
    # mng.full_screen_toggle()

    # figManager = plt.get_current_fig_manager()
    # figManager.window.Maximized()


    # save the plot
    plt.savefig(path_init+'\\'+path.split('\\')[-1]+'.png')

    # plt.show()
    # plt.close()



################################################################################################################

# def __main__():
# set the working directory
# path_init=os.getcwd()

# list of the .ecsv files in the directory
files_manual = [f for f in os.listdir(path_init) if f.endswith('.ecsv')]

# list of the .txt files in the directory
files_auto = [f for f in os.listdir(path_init) if f.endswith('.txt')]

exclude = ['compare_manual_vs_detapp_lag','GOOD','TEST']
for (dir_path, dir_names, file_names) in os.walk(path_init, topdown=True):
    # res = []
    [dir_names.remove(d) for d in list(dir_names) if d in exclude]
    # res.extend(dir_names)
    # list of the .ecsv files in the directory
    files_manual = [f for f in os.listdir(dir_path) if f.endswith('.ecsv')]

    # list of the .txt files in the directory
    files_auto = [f for f in os.listdir(dir_path) if f.endswith('.txt')]

    if not files_auto or not files_manual:
        print('no files in ',dir_path)
    else:
        print('files in ',dir_path)
        print(files_manual)
        print(files_auto)
        plot_ErrorPlots(files_manual, files_auto, dir_path, path_init)