from __future__ import print_function, division, absolute_import, unicode_literals

import os
import json
import copy
import matplotlib.pyplot as plt

import numpy as np

# number of realizations
numb_all=99

# Standard deviation of the magnitude Gaussian noise 1 sigma
mag_noise = 0.1

# SD of noise in length (m) 1 sigma
len_noise = 20.0

# velocity noise 1 sigma
vel_noise = (20*np.sqrt(2)/0.03125)

# open the current directory
name_directory = os.getcwd()
name_file = 'TRUEerosion_sim_v59.84_m1.33e-02g_rho0209_z39.8_abl0.014_eh117.3_er0.636_s1.61.json'

namefile_sel=name_directory+'\\'+name_file

def find_closest_index(time_arr, time_sampled):
    closest_indices = []
    for sample in time_sampled:
        closest_index = min(range(len(time_arr)), key=lambda i: abs(time_arr[i] - sample))
        closest_indices.append(closest_index)
    return closest_indices


# chec if the file exist
if not os.path.isfile(namefile_sel):
    print('file '+namefile_sel+' not found')
else:
    # open the json file with the name namefile_sel
    f = open(namefile_sel,"r")
    data = json.loads(f.read())
    # data['main_vel_arr']
    # ht_sim=data['simulation_results']['main_height_arr']
    # absmag_sim=data['simulation_results']['abs_magnitude']
    # cut out absmag_sim above 7 considering that absmag_sim is a list
    
    abs_mag_sim=data['simulation_results']['abs_magnitude']#['brightest_vel_arr']#['leading_frag_vel_arr']#['main_vel_arr']
    ht_sim=data['simulation_results']['leading_frag_height_arr']#['brightest_height_arr']['leading_frag_height_arr']['main_height_arr']
    time_sim=data['simulation_results']['time_arr']#['brightest_time_arr']#['leading_frag_time_arr']#['main_time_arr']
    len_sim=data['simulation_results']['leading_frag_length_arr']#['brightest_len_arr']['leading_frag_len_arr']['main_len_arr']
    vel_sim=data['simulation_results']['leading_frag_vel_arr']#['brightest_vel_arr']#['leading_frag_vel_arr']#['main_vel_arr']

    obs_abs_mag= data['mag_sampled']
    obs_height= data['ht_sampled']
    obs_time= data['time_sampled']
    obs_length= data['len_sampled']




    # delete the nan term in abs_mag_sim and ht_sim
    abs_mag_sim=[x for x in abs_mag_sim if str(x) != 'nan']
    ht_sim=[x for x in ht_sim if str(x) != 'nan']

    # find the index of the first element of the simulation that is equal to the first element of the observation
    index_ht_sim=next(x for x, val in enumerate(ht_sim) if val <= obs_height[0])
    # find the index of the last element of the simulation that is equal to the last element of the observation
    index_ht_sim_end=next(x for x, val in enumerate(ht_sim) if val <= obs_height[-1])

    abs_mag_sim=abs_mag_sim[index_ht_sim:index_ht_sim_end]
    ht_sim=ht_sim[index_ht_sim:index_ht_sim_end]
    time_sim=time_sim[index_ht_sim:index_ht_sim_end]
    len_sim=len_sim[index_ht_sim:index_ht_sim_end]
    vel_sim=vel_sim[index_ht_sim:index_ht_sim_end]

    time_sim=[x-time_sim[0] for x in time_sim]
    len_sim=[x-len_sim[0] for x in len_sim]

    # Find and print the closest indices
    closest_indices = find_closest_index(time_sim, obs_time)

    abs_mag_sim=[abs_mag_sim[i] for i in closest_indices]
    len_sim=[len_sim[i] for i in closest_indices]
    vel_sim=[vel_sim[i] for i in closest_indices]
    time_sim=[time_sim[i] for i in closest_indices]


    # subplot 2 plots
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    # plot noisy area around vel_kms for vel_noise for the fix height_km
    ax[1].fill_between(obs_time, np.array(vel_sim-vel_noise)/1000, np.array(vel_sim+vel_noise)/1000, color='lightgray', alpha=0.5)

    # from data params v_init val
    vel_sampled_sim=[data['params']['v_init']['val']]
    # append from vel_sampled the rest by the difference of the first element of obs_length divided by the first element of obs_time
    rest_vel_sampled_sim=[(obs_length[i]-obs_length[i-1])/(obs_time[i]-obs_time[i-1]) for i in range(1,len(obs_length))]
    # append the rest_vel_sampled to vel_sampled
    vel_sampled_sim.extend(rest_vel_sampled_sim)

    ax[1].plot(obs_time,np.array(vel_sampled_sim)/1000, label='Noisy No.1',linestyle='None', marker='x')

    # plot noisy area around vel_kms for vel_noise for the fix height_km
    ax[0].fill_betweenx(np.array(data['ht_sampled'])/1000, np.array(abs_mag_sim)-mag_noise, np.array(abs_mag_sim)+mag_noise, color='lightgray', alpha=0.5)

    ax[0].plot(obs_abs_mag,np.array(obs_height)/1000, label='Noisy No.1', linestyle='None', marker='x')

# create a for loop from the biggest to the smallest number
    for numb in range(2,numb_all+1):
        print('Realization No.', numb)
        # Add noise to magnitude data (Gaussian noise) for each realization
        abs_mag_sim_chose = np.array(abs_mag_sim)
        # add to each magnitude a random number from a normal distribution with mean 0 and standard deviation mag_noise
        # abs_mag_sim_chose = abs_mag_sim_chose + np.random.normal(loc=0, scale=mag_noise, size=abs_mag_sim_chose.shape)
        abs_mag_sim_chose += np.random.normal(loc=0.0, scale=mag_noise, size=len(abs_mag_sim_chose))
        # mag_sampled[mag_sampled <= lim_mag] += np.random.normal(loc=0.0, scale=params.mag_noise, \
        #     size=len(mag_sampled[mag_sampled <= lim_mag]))


        # Add noise to length data (Gaussian noise) for each realization
        len_sim_chose = np.array(len_sim)
        # len_sim_chose = len_sim_chose + np.random.normal(loc=0, scale=len_noise, size=len_sim_chose.shape)
        len_sim_chose += np.random.normal(loc=0.0, scale=len_noise, size=len(len_sim_chose))

        # len_sim = (len_sim - np.min(len_sim))/(np.max(len_sim) - np.min(len_sim))
        # abs_mag_sim = (abs_mag_sim - np.min(abs_mag_sim))/(np.max(abs_mag_sim) - np.min(abs_mag_sim))

        # Create a new json file with the noisy data
        new_data = copy.deepcopy(data)
        new_data['mag_sampled']= list(abs_mag_sim_chose)
        new_data['len_sampled']= list(len_sim_chose)
        new_data['time_sampled']= list(time_sim)

        # Save the new json file
        new_namefile = namefile_sel.replace('.json', '_'+str(numb)+'_noisy.json')
        with open(new_namefile, 'w') as f:
            json.dump(new_data, f, indent=4)

        print('New json file saved as', new_namefile)

    ###########################################################################

        ax[0].plot(abs_mag_sim_chose,np.array(obs_height)/1000, label='Noisy No.'+str(numb), linestyle='None', marker='x')

        # from data params v_init val
        vel_sampled=[data['params']['v_init']['val']]
        # append from vel_sampled the rest by the difference of the first element of obs_length divided by the first element of obs_time
        rest_vel_sampled=[(len_sim_chose[i]-len_sim_chose[i-1])/(time_sim[i]-time_sim[i-1]) for i in range(1,len(len_sim_chose))]
        # append the rest_vel_sampled to vel_sampled
        vel_sampled.extend(rest_vel_sampled)

        # plot the velocity
        ax[1].plot(obs_time,np.array(vel_sampled)/1000, label='Noisy No.'+str(numb), linestyle='None', marker='x')# , markersize=5)

 
    ax[1].plot(obs_time,np.array(vel_sim)/1000, label='Original', color='k')
    ax[0].plot(abs_mag_sim,np.array(data['ht_sampled'])/1000, label='Original', color='k')
    # grid on
    ax[0].grid()
    ax[1].grid()

    # divid the y axis values by 1000
    # ax[1].set_yticklabels([str(int(x)) for x in ax[1].get_yticks()/1000])
    ax[1].set_ylabel('Velocity (km/s)')
    ax[1].set_xlabel('Time (s)')

    # ax[0].set_yticklabels([str(int(x)) for x in ax[1].get_yticks()/1000])
    ax[0].set_ylabel('Height (km)')
    ax[0].set_xlabel('Absolute Magnitude')

    #plot legend
    # ax[0].legend()

    # save the plot as a png file in the same directory as the json file
    plt.savefig(name_directory+'\\'+str(numb_all)+'all_noisyPlot.png')


    # plot the observation
    plt.show()
    ###########################################################################
