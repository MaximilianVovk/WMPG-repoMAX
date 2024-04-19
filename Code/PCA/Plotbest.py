import wmpl
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import os
import json
import numpy as np
import math
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
import glob


def find_closest_index(time_arr, time_sampled):
    closest_indices = []
    for sample in time_sampled:
        closest_index = min(range(len(time_arr)), key=lambda i: abs(time_arr[i] - sample))
        closest_indices.append(closest_index)
    return closest_indices

# put the first plot in 2 sublots
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

# List all JSON files in the current directory
json_files = glob.glob('./*.json')

print(json_files)

ii=0
# loop over the json files in the directory
for namefile_sel in json_files:
    # open the json file with the name namefile_sel
    f = open(namefile_sel,"r")
    data = json.loads(f.read())

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

    if ii==0:

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
        vel_sim=[v0]
        obs_length=[x/1000 for x in obs_length]
        # append from vel_sampled the rest by the difference of the first element of obs_length divided by the first element of obs_time
        rest_vel_sampled=[(obs_length[vel_ii]-obs_length[vel_ii-1])/(obs_time[vel_ii]-obs_time[vel_ii-1]) for vel_ii in range(1,len(obs_length))]
        # append the rest_vel_sampled to vel_sampled
        vel_sim.extend(rest_vel_sampled)

############ add noise to the simulation

        # plot a line plot in the first subplot the magnitude vs height dashed with x markers
        ax[0].plot(abs_mag_sim, ht_sim, linestyle='dashed', marker='x', label='1')

        # add the erosion_height_start as a horizontal line in the first subplot grey dashed
        ax[0].axhline(y=erosion_height_start, color='grey', linestyle='dashed')
        # add the name on the orizontal height line
        ax[0].text(max(abs_mag_sim)+1, erosion_height_start, 'Erosion heig', color='grey')

        # plot a scatter plot in the second subplot the velocity vs height
        ax[1].scatter(vel_sim, ht_sim, marker='.', label='1')

        # set the xlim and ylim of the first subplot
        ax[0].set_xlim([min(abs_mag_sim)-1, max(abs_mag_sim)+1])
        # check if the max(ht_sim) is greater than the erosion_height_start and set the ylim of the first subplot
        if max(ht_sim)>erosion_height_start:
            ax[0].set_ylim([min(ht_sim)-1, max(ht_sim)+1])
            ax[1].set_ylim([min(ht_sim)-1, max(ht_sim)+1])
        else:
            ax[0].set_ylim([min(ht_sim)-1, erosion_height_start+2])
            ax[1].set_ylim([min(ht_sim)-1, erosion_height_start+2])

        # set the xlim and ylim of the second subplot
        ax[1].set_xlim([min(vel_sim)-1, max(vel_sim)+1])


    else:
        
        # divide the vel_sim by 1000 considering is a list
        time_sim = [i-time_sim[0] for i in time_sim]
        vel_sim = [i/1000 for i in vel_sim]
        vel_sim_brigh = [i/1000 for i in vel_sim_brigh]
        len_sim = [(i-len_sim[0])/1000 for i in len_sim]
        ht_sim = [i/1000 for i in ht_sim]

        # plot a line plot in the first subplot the magnitude vs height dark black line
        ax[0].plot(abs_mag_sim, ht_sim, color='black', label='Simulated')
        # plot a scatter plot in the second subplot the velocity vs height dark black line
        # ax[1].plot(vel_sim_brigh, ht_sim, color='black', linestyle='dashed', label='Simulated - leading')
        # ax[1].plot(vel_sim, ht_sim, color='black', label='Simulated - brightest')
        ax[1].plot(vel_sim, ht_sim, color='black', label='Simulated')

    ii+=1




# put the grid in the subplots and make it dashed
ax[0].grid(linestyle='dashed')
ax[1].grid(linestyle='dashed')
# add the legend
ax[0].legend()
ax[1].legend()

# add the labels
ax[0].set_ylabel('Height [km]')
ax[0].set_xlabel('Absolute Magnitude')
# invert the x axis
ax[0].invert_xaxis()

ax[1].set_ylabel('Height [km]')
ax[1].set_xlabel('Velocity [km/s]')

# make the plot visible
plt.show()

# save the plot
plt.savefig(namefile_sel+'.png')