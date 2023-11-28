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

Shower='PER' # ORI ETA SDA CAP GEM PER
# # Perseids = PER about 1000 rho 100 to 1000 sigma 0.1 to 0.01 2 and 2
# # Leonids = LEON
# # Geminids = GEM about 200 rho 300 t0 3000 sigma 0.005 to 0.05 2 and 2
# # Ursids = URS
# # Orionid = ORI
""" 
    'solution_id','shower_code','vel_init_norot','vel_avg_norot','duration',\
    'mass','peak_mag_height','begin_height','end_height','peak_abs_mag','beg_abs_mag','end_abs_mag',\
    'F','trail_len','acceleration','zenith_angle', 'kurtosis','skew',\
    'kc'
"""
# dataList = [['','', 0, 0, 0,\
#     0, 0, 0, 0, 0, 0, 0,\
#     0, 0, 0, 0, 0, 0,\
#     0, 0, 0, 0, 0]]

# infov = pd.DataFrame(dataList, columns=['solution_id','shower_code','vel_init_norot','vel_avg_norot','duration',\
# 'mass','peak_mag_height','begin_height','end_height','peak_abs_mag','beg_abs_mag','end_abs_mag',\
# 'F','trail_len','acceleration','zenith_angle', 'kurtosis','skew',\
# 'kc', 'lag_init', 'lag_avg','lag_final', 'lag_trend'])

dataList = [['','', 0, 0, 0,\
    0, 0, 0, 0, 0, 0, 0,\
    0, 0, 0, 0, 0, 0,\
    0]]

infov = pd.DataFrame(dataList, columns=['solution_id','shower_code','vel_init_norot','vel_avg_norot','duration',\
'mass','peak_mag_height','begin_height','end_height','peak_abs_mag','beg_abs_mag','end_abs_mag',\
'F','trail_len','acceleration','zenith_angle', 'kurtosis','skew',\
'kc'])


directory=os.getcwd()
extension = 'pickle'
all_picklefiles = [i for i in glob.glob('*.{}'.format(extension))]


name=[]
shower_code=[]

acceleration=[]
vel_init_norot=[]
vel_avg_norot=[]
duration=[]
mass=[]

begin_height=[]
end_height=[]
peak_mag_height=[]

peak_abs_mag=[]
beg_abs_mag=[]
end_abs_mag=[]

lag_data=[]

F_data=[]
trail_len=[]
zenith_angle=[]

kurtosisness=[]
skewness=[]

inclin_m=[]

lag_init=[]
lag_fin=[]
lag_avg=[]

lag_trend=[]


jj=0
for ii in range(len(all_picklefiles)):
    
    # pick the ii element of the solution_id column 
    namefile=all_picklefiles[ii]
    print('Loading pickle file: ', namefile, ' n.', jj, ' of ', len(all_picklefiles), ' done.')
    jj=jj+1
    # vel_init=infov_percentile.iloc[ii]['solution_id']
    # split the namefile base on the '_' character and pick the first element
    # create the path to the folder so that is the directory and namefile
    path=directory+'\\'+namefile
    print(path)
    traj = wmpl.Utils.Pickling.loadPickle(directory,namefile)

    # save a copy of the file in the folder 
    # namefile2=namefile.split('A')[0]
    # create a folder with the name of the shower
    # wmpl.Utils.Pickling.savePickle(traj, "C:\\Users\\maxiv\\Documents\\UWO\\Papers\\1)PCA\\PER_trajectory_pk\\"+namefile2+"\\", namefile2+"_trajectory.pickle")
    # delete A from namefile with split

    vel_pickl=[]
    time_pickl=[]
    abs_mag_pickl=[]
    height_pickl=[]
    lag=[]
    elev_angle_pickl=[]
    trail_len_pickl=[]
    mass_pickl=[]
    for obs in traj.observations:
        # put it at the end obs.velocities[1:] at the end of vel_pickl list
        vel_pickl.extend(obs.velocities[1:])
        time_pickl.extend(obs.time_data[1:])
        abs_mag_pickl.extend(obs.absolute_magnitudes[1:])
        height_pickl.extend(obs.model_ht[1:])
        lag.extend(obs.lag[1:])
        elev_angle_pickl.extend(obs.elev_data)

    # compute the linear regression
    vel_pickl = [i/1000 for i in vel_pickl] # convert m/s to km/s
    time_pickl = [i for i in time_pickl]
    height_pickl = [i/1000 for i in height_pickl]
    abs_mag_pickl = [i for i in abs_mag_pickl]
    lag=[i for i in lag]

    # fit a line to the throught the vel_sim and ht_sim
    a, b = np.polyfit(time_pickl,vel_pickl, 1)

    trendLAG, bLAG = np.polyfit(time_pickl,lag, 1)

    vel_sim_line=[a*x+b for x in time_pickl]

    lag_line = [trendLAG*x+bLAG for x in time_pickl] 

    hcoef1, hcoef2 = np.polyfit(time_pickl,height_pickl, 1)

    height_line=[hcoef1*x+hcoef2 for x in time_pickl]

    # infov_percentile.acceleration[ii]=(-1)*a
    # infov_percentile.vel_init_norot[ii]=vel_sim_line[0]
    # infov_percentile.vel_avg_norot[ii]=np.mean(vel_sim_line)

    #####order the list by time
    vel_pickl = [x for _,x in sorted(zip(time_pickl,vel_pickl))]
    abs_mag_pickl = [x for _,x in sorted(zip(time_pickl,abs_mag_pickl))]
    height_pickl = [x for _,x in sorted(zip(time_pickl,height_pickl))]
    time_pickl = sorted(time_pickl)

    # append the values to the list
    acceleration=((-1)*a)
    lag_trend=(trendLAG)
    # vel_init_norot=(vel_sim_line[0])
    vel_init_norot=(vel_sim_line[0])
    vel_avg_norot=(np.mean(vel_sim_line))

    begin_height=(height_pickl[0])
    end_height=(height_pickl[-1])
    peak_mag_height=(height_pickl[np.argmin(abs_mag_pickl)])

    peak_abs_mag=(np.min(abs_mag_pickl))
    beg_abs_mag=(abs_mag_pickl[0])
    end_abs_mag=(abs_mag_pickl[-1])

    # lag_fin=(lag[-1])
    # lag_init=(lag[0])
    # lag_avg=(np.mean(lag))

    lag_fin=(lag_line[-1])
    lag_init=(lag_line[0])
    lag_avg=(np.mean(lag_line))

    duration=(time_pickl[-1]-time_pickl[0])

    kc_par = begin_height + (2.86 - 2*np.log(vel_init_norot))/0.0612

    F_data=((height_pickl[0] - height_pickl[np.argmin(abs_mag_pickl)]) / (height_pickl[0] - height_pickl[-1]))

    zenith_angle=(90 - elev_angle_pickl[0]*180/np.pi)
    trail_len=((height_pickl[0] - height_pickl[-1])/(np.sin(np.radians(elev_angle_pickl[0]*180/np.pi))))


    name=(namefile.split('_trajectory')[0])

    # open a .json file with a part of name in the name of the file
    with open('C:\\Users\\maxiv\\Documents\\UWO\\Papers\\1)PCA\\PCA_code\\manual\\'+name+'_sim_fit.json','r') as json_file:
        data = json.load(json_file)

    mass_keys = ['m_init','rho','sigma',\
    'erosion_height_start','erosion_coeff', 'erosion_mass_index',\
    'erosion_mass_min','erosion_mass_max']

    mass=(data['m_init'])

    # add an 'A' at the end of the name
    name=name+'A'
    shower_code=(Shower)
    

    ########################################## SKEWNESS AND KURTOSISNESS ##########################################
    
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

    kurtosisness=(kurtosis(mag_sampled_distr))
    skewness=(skew(mag_sampled_distr))

    infov.loc[ii] = [name,shower_code, vel_init_norot, vel_avg_norot, duration,\
    mass, peak_mag_height,begin_height, end_height, peak_abs_mag, beg_abs_mag, end_abs_mag,\
    F_data, trail_len, acceleration, zenith_angle, kurtosisness,skewness,\
    kc_par]




# infov['begin_height']=begin_height
# infov['end_height']=end_height

# # infov['peak_abs_mag']=peak_abs_mag
# # infov['beg_abs_mag']=beg_abs_mag
# # infov['end_abs_mag']=end_abs_mag
    
# infov['acceleration'] = acceleration
# infov['vel_init_norot'] = vel_init_norot
# infov['vel_avg_norot'] = vel_avg_norot

# infov['kc'] = infov['begin_height'] + (2.86 - 2*np.log(infov['vel_init_norot']))/0.0612
# infov['kurtosis'] = kurtosisness
# infov['skew'] = skewness 

# infov['lag'] = lag_data 




# save the dataframe to a csv file
infov.to_csv('C:\\Users\\maxiv\\Documents\\UWO\\Papers\\1)PCA\\PCA_code\\manual\\'+Shower+'_manual.csv', index=False)