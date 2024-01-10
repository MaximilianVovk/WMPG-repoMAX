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
from wmpl.Utils.Physics import dynamicPressure
import shutil
from scipy.stats import kurtosis, skew
from wmpl.Utils.OSTools import mkdirP
import math
from wmpl.Utils.PyDomainParallelizer import domainParallelizer

Shower='PER' # ORI ETA SDA CAP GEM PER

def PCAmanuallyReduced(OUT_PUT_PATH=os.getcwd(), Shower='PER', INPUT_PATH=['C:\\Users\\maxiv\\Documents\\UWO\\Papers\\1)PCA\\Reductions\\manual_reductions']):
    
    Shower=Shower[0]

    all_picklefiles=[]

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

    kc_par=[]

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

    # knee of the velocity caracteristics
    height_knee_vel=[]
    decel_after_knee_vel=[]
    peak_mag_vel=[]
    # the dynamic pressure at the knee at the peak of absolute mag
    Dynamic_pressure_peak_abs_mag=[]

    # directory='C:\\Users\\maxiv\\Documents\\UWO\\Papers\\1)PCA\\Reductions\\manual_reductions'
    # open the directory and walk through all the files and subfolders and open only the files with the extension .pickle
    for root, dirs, files in os.walk(INPUT_PATH):
        for name_file in files:
            if name_file.endswith(".pickle"):
                # print(os.path.join(root, name_file))

                print('Loading pickle file: ', name_file)

                traj = wmpl.Utils.Pickling.loadPickle(root,name_file)
                jd_dat=traj.jdt_ref

                vel_pickl=[]
                time_pickl=[]
                abs_mag_pickl=[]
                height_pickl=[]
                lag=[]
                elev_angle_pickl=[]
                elg_pickl=[]
                tav_pickl=[]
                
                lat_dat=[]
                lon_dat=[]

                jj=0
                for obs in traj.observations:
                    jj+=1
                    if jj==1:
                        elg_pickl=obs.velocities[1:int(len(obs.velocities)/4)]
                        if len(elg_pickl)==0:
                            elg_pickl=obs.velocities[1:2]
                    elif jj==2:
                        tav_pickl=obs.velocities[1:int(len(obs.velocities)/4)]
                        # if tav_pickl is empty append the first value of obs.velocities
                        if len(tav_pickl)==0:
                            tav_pickl=obs.velocities[1:2]
                    # put it at the end obs.velocities[1:] at the end of vel_pickl list
                    vel_pickl.extend(obs.velocities[1:])
                    time_pickl.extend(obs.time_data[1:])
                    abs_mag_pickl.extend(obs.absolute_magnitudes[1:])
                    height_pickl.extend(obs.model_ht[1:])
                    lag.extend(obs.lag[1:])
                    elev_angle_pickl.extend(obs.elev_data)
                    
                    lat_dat=obs.lat
                    lon_dat=obs.lon

                # compute the linear regression
                vel_pickl = [i/1000 for i in vel_pickl] # convert m/s to km/s
                time_pickl = [i for i in time_pickl]
                height_pickl = [i/1000 for i in height_pickl]
                abs_mag_pickl = [i for i in abs_mag_pickl]
                lag=[i for i in lag]

                # find the height when the velocity start dropping from the initial value 
                vel_init_mean = (np.mean(elg_pickl)+np.mean(tav_pickl))/2/1000
                # print('mean_vel_init', vel_init_mean)

                # find the smallest among all elg_pickl and all tav_pickl
                vel_small=min([min(elg_pickl),min(tav_pickl)])/1000
                # print('vel_small:', vel_small)
                

                # fit a line to the throught the vel_sim and ht_sim
                a, b = np.polyfit(time_pickl,vel_pickl, 1)

                trendLAG, bLAG = np.polyfit(time_pickl,lag, 1)

                vel_sim_line=[a*x+b for x in time_pickl]

                lag_line = [trendLAG*x+bLAG for x in time_pickl] 

                # hcoef1, hcoef2 = np.polyfit(time_pickl,height_pickl, 1)

                # height_line=[hcoef1*x+hcoef2 for x in time_pickl]

                # infov_percentile.acceleration[ii]=(-1)*a
                # infov_percentile.vel_init_norot[ii]=vel_sim_line[0]
                # infov_percentile.vel_avg_norot[ii]=np.mean(vel_sim_line)

                #####order the list by time
                vel_pickl = [x for _,x in sorted(zip(time_pickl,vel_pickl))]
                abs_mag_pickl = [x for _,x in sorted(zip(time_pickl,abs_mag_pickl))]
                height_pickl = [x for _,x in sorted(zip(time_pickl,height_pickl))]
                time_pickl = sorted(time_pickl)

                # find the sigle index of the height when the velocity start dropping from the vel_init_mean of about 0.5 km/s
                index = [i for i in range(len(vel_pickl)) if vel_pickl[i] < vel_small-0.2]
                # only use first index to pick the height
                height_knee_vel.append(height_pickl[index[0]-1])
                # print('height_knee_vel', height_pickl[index[0]-1])

                a2, b2 = np.polyfit(time_pickl[index[0]-1:],vel_pickl[index[0]-1:], 1)

                # print('decel_after_knee_vel', (-1)*a2)
                # print(vel_pickl[index[0]-1:])

                # append the values to the list
                acceleration.append((-1)*a)
                decel_after_knee_vel.append((-1)*a2)
                lag_trend.append(trendLAG)
                # vel_init_norot=(vel_sim_line[0])
                # vel_init_norot.append(vel_sim_line[0])
                vel_init_norot.append(vel_init_mean)
                # print('mean_vel_init', vel_sim_line[0])
                vel_avg_norot.append(np.mean(vel_pickl)) #trail_len / duration
                peak_mag_vel.append(vel_pickl[np.argmin(abs_mag_pickl)])   

                begin_height.append(height_pickl[0])
                end_height.append(height_pickl[-1])
                peak_mag_height.append(height_pickl[np.argmin(abs_mag_pickl)])

                peak_abs_mag.append(np.min(abs_mag_pickl))
                beg_abs_mag.append(abs_mag_pickl[0])
                end_abs_mag.append(abs_mag_pickl[-1])

                lag_fin.append(lag_line[-1])
                lag_init.append(lag_line[0])
                lag_avg.append(np.mean(lag_line))

                duration.append(time_pickl[-1]-time_pickl[0])

                kc_par.append(height_pickl[0] + (2.86 - 2*np.log(vel_sim_line[0]))/0.0612)

                F_data.append((height_pickl[0] - height_pickl[np.argmin(abs_mag_pickl)]) / (height_pickl[0] - height_pickl[-1]))

                zenith_angle.append(90 - elev_angle_pickl[0]*180/np.pi)
                trail_len.append((height_pickl[0] - height_pickl[-1])/(np.sin(np.radians(elev_angle_pickl[0]*180/np.pi))))
                
                # vel_avg_norot.append( ((height_pickl[0] - height_pickl[-1])/(np.sin(np.radians(elev_angle_pickl[0]*180/np.pi))))/(time_pickl[-1]-time_pickl[0]) ) #trail_len / duration

                name.append(name_file.split('_trajectory')[0]+'A')

                Dynamic_pressure_peak_abs_mag.append(wmpl.Utils.Physics.dynamicPressure(lat_dat, lon_dat, height_pickl[np.argmin(abs_mag_pickl)]*1000, jd_dat, vel_pickl[np.argmin(abs_mag_pickl)]*1000))


                # check if in os.path.join(root, name_file) present and then open the .json file with the same name as the pickle file with in stead of _trajectory.pickle it has _sim_fit_latest.json
                if os.path.isfile(os.path.join(root, name_file.split('_trajectory')[0]+'_sim_fit.json')):
                    with open(os.path.join(root, name_file.split('_trajectory')[0]+'_sim_fit.json'),'r') as json_file: # 20210813_061453_sim_fit.json
                        data = json.load(json_file)
                        mass.append(data['m_init'])
                else:
                    # if no data on weight is 0
                    mass.append(0)

                # mass.append(0)
            
                # # add an 'A' at the end of the name
                # name=name+'A'
                shower_code.append(Shower)
                

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

                kurtosisness.append(kurtosis(mag_sampled_distr))
                skewness.append(skew(mag_sampled_distr))

    # dataList = [['','', 0, 0, 0,\
    #     0, 0, 0, 0, 0, 0, 0,\
    #     0, 0, 0, 0, 0, 0,\
    #     0]]

    # infov = pd.DataFrame(dataList, columns=['solution_id','shower_code','vel_init_norot','vel_avg_norot','duration',\
    # 'mass','peak_mag_height','begin_height','end_height','peak_abs_mag','beg_abs_mag','end_abs_mag',\
    # 'F','trail_len','acceleration','zenith_angle', 'kurtosis','skew',\
    # 'kc'])

    dataList = [['','', 0, 0, 0,\
        0, 0, 0, 0, 0, 0, 0, 0,\
        0, 0, 0, 0, 0, 0, 0,\
        0, 0]]

    infov = pd.DataFrame(dataList, columns=['solution_id','shower_code','vel_init_norot','vel_avg_norot','duration',\
    'mass','peak_mag_height','begin_height','end_height','height_knee_vel','peak_abs_mag','beg_abs_mag','end_abs_mag',\
    'F','trail_len','acceleration','decel_after_knee_vel','zenith_angle', 'kurtosis','skew',\
    'kc','Dynamic_pressure_peak_abs_mag'])

    # create a loop to populate the dataframe
    for ii in range(len(name)):
        # print(name[ii], shower_code[ii], vel_init_norot[ii], vel_avg_norot[ii], duration[ii],\
        #       mass[ii], peak_mag_height[ii], begin_height[ii], end_height[ii], peak_abs_mag[ii], beg_abs_mag[ii], end_abs_mag[ii],\
        #         F_data[ii], trail_len[ii], acceleration[ii], zenith_angle[ii], kurtosisness[ii], skewness[ii],\
        #             kc_par[ii])
        # infov.loc[ii] = [name[ii], shower_code[ii], vel_init_norot[ii], vel_avg_norot[ii], duration[ii],\
        # mass[ii], peak_mag_height[ii], begin_height[ii], end_height[ii], peak_abs_mag[ii], beg_abs_mag[ii], end_abs_mag[ii],\
        # F_data[ii], trail_len[ii], acceleration[ii], zenith_angle[ii], kurtosisness[ii], skewness[ii],\
        # kc_par[ii]]
        infov.loc[ii] = [name[ii], shower_code[ii], vel_init_norot[ii], vel_avg_norot[ii], duration[ii],\
        mass[ii], peak_mag_height[ii], begin_height[ii], end_height[ii], height_knee_vel[ii], peak_abs_mag[ii], beg_abs_mag[ii], end_abs_mag[ii],\
        F_data[ii], trail_len[ii], acceleration[ii], decel_after_knee_vel[ii], zenith_angle[ii], kurtosisness[ii], skewness[ii],\
        kc_par[ii], Dynamic_pressure_peak_abs_mag[ii]]

    
    # save the dataframe to a csv file
    infov.to_csv(OUT_PUT_PATH+r'\\'+Shower+'_manual.csv', index=False)





if __name__ == "__main__":

    import argparse


    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Get the manual data from the pickle files of the manually reduced events for the PCA.")

    arg_parser.add_argument('output_dir', metavar='OUTPUT_PATH', type=str, \
        help="Path to the output directory where is saved .csv file in the end.")

    arg_parser.add_argument('shower', metavar='SHOWER', type=str, \
        help="Use specific shower from the given simulation.")
    
    arg_parser.add_argument('input_dir', metavar='INPUT_PATH', type=str, \
        help="Path were are store all manual reduced events, it use walk from the given diretory and looks for puickle and json files.")

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################

    # Make the output directory
    mkdirP(cml_args.output_dir)

    # make only one shower
    Shower=[cml_args.shower]

    PCAmanuallyReduced(cml_args.output_dir, Shower, cml_args.input_dir)