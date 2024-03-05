import wmpl
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import os
import json
import numpy as np
import math
from sklearn.linear_model import LinearRegression

#matplotlib.use('Agg')
#matplotlib.use("Qt5Agg")

# MODIFY HERE THE PARAMETERS ###############################################################################
# Set the shower name (can be multiple) e.g. 'GEM' or ['GEM','PER', 'ORI', 'ETA', 'SDA', 'CAP']
Shower=['PER']#['PER']

# number of selected events selected
n_select=1000

# min distance over which are selected
min_dist_obs=0
min_dist_sel=0

# number to confront
n_confront_obs=1
n_confront_sel=4

only_select_meteors_from=''

# no legend for a lot of simulations
with_legend=True

# add the lenght of the simulation
with_LEN=True

# FUNCTIONS ###########################################################################################

if with_LEN==True:
    # put the first plot in 3 sublots
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
else:
    # put the first plot in 2 sublots
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

# save all the simulated showers in a list
df_obs_shower = []
df_sel_shower = []
# search for the simulated showers in the folder
for current_shower in Shower:
    print(current_shower)

    df_obs = pd.read_csv(os.getcwd()+r'\\'+current_shower+'_and_dist.csv')

    # if only_select_meteors_from!='':
    #     df_obs=df_obs[df_obs['solution_id']==only_select_meteors_from]
    
    print('observed: '+str(len(df_obs)))
    # append the observed shower to the list

    df_obs_shower.append(df_obs)

    # check in the current folder there is a csv file with the name of the simulated shower
    df_sel = pd.read_csv(os.getcwd()+r'\\Simulated_'+current_shower+'_select.csv')

    if len(df_sel)>n_select:
        df_sel=df_sel.head(n_select)

    # append the simulated shower to the list
    df_sel_shower.append(df_sel)

# # select the one below 1 in distance and order for distance
df_sel_shower = pd.concat(df_sel_shower)
# df_sel_shower = df_sel_shower.sort_values(by=['distance'])
# df_sel_shower = df_sel_shower[df_sel_shower['distance']<set_dist]
# print('selected: '+str(len(df_sel_shower)))
# # same for the observed
df_obs_shower = pd.concat(df_obs_shower)
# df_obs_shower = df_obs_shower.sort_values(by=['distance'])
# df_obs_shower = df_obs_shower[df_obs_shower['distance']<set_dist]
# print('observed: '+str(len(df_obs_shower)))

df_sel_shower['erosion_coeff']=df_sel_shower['erosion_coeff']*1000000
df_sel_shower['sigma']=df_sel_shower['sigma']*1000000














if df_obs_shower is None:
    print('no observed shower found')
    exit()

for current_shower in Shower:
    curr_obs_og=df_obs_shower[df_obs_shower['shower_code']==current_shower]
    curr_sel_og=df_sel_shower[df_sel_shower['shower_code']==current_shower+'_sel']

    curr_sel = curr_sel_og.sort_values(by=['distance'])
    curr_sel = curr_sel[curr_sel['distance']>=min_dist_sel]

    curr_obs = curr_obs_og.sort_values(by=['distance'])
    curr_obs = curr_obs[curr_obs['distance']>=min_dist_obs]

    if n_confront_obs<len(df_obs_shower):
        curr_obs=curr_obs.head(n_confront_obs)
    
    if n_confront_sel<len(df_sel_shower):
        curr_sel=curr_sel.head(n_confront_sel)



initial_folder=os.getcwd()

# go back one folder
os.chdir('..')

# find the directory where the script is running
current_folder=os.getcwd()

# in current_folder entern in the folder current_folder+'\\Simulation_'current_shower
os.chdir(current_folder+'\\Simulations_'+current_shower+'\\')
# os.chdir('Simulations_'+current_shower)
for ii in range(len(curr_sel)):
    # pick the ii element of the solution_id column 
    namefile_sel=curr_sel.iloc[ii]['solution_id']
    # find the index of curr_obs_og with the same distance
    index_sel=curr_sel_og[curr_sel_og['solution_id']==namefile_sel].index
    index_sel=index_sel[0]



    # chec if the file exist
    if not os.path.isfile(namefile_sel):
        print('file '+namefile_sel+' not found')
        continue
    else:
        # open the json file with the name namefile_sel
        f = open(namefile_sel,"r")
        data = json.loads(f.read())
        # data['main_vel_arr']
        # ht_sim=data['simulation_results']['main_height_arr']
        # absmag_sim=data['simulation_results']['abs_magnitude']
        # cut out absmag_sim above 7 considering that absmag_sim is a list
        height_km=[x/1000 for x in data['ht_sampled']]
        if with_legend:
            # put it in the first subplot
            ax[0].plot(data['mag_sampled'],height_km,label='sel_'+current_shower+'('+str(index_sel)+') MEANdist:'+str(round(curr_sel.iloc[ii]['distance'],2)))
        else:
            ax[0].plot(data['mag_sampled'],height_km,label='sel_'+current_shower+'('+str(index_sel)+') MEANdist:'+str(round(curr_sel.iloc[ii]['distance'],2)),color='coral')



# title with the current_shower and written also absolute mag vs height
ax[0].set_title(current_shower+' abs.mag vs height')
# grid on
ax[0].grid(linestyle='--',color='lightgray')
# if with_legend:
    # plt.legend()
# plt.show()




for ii in range(len(curr_sel)):

    # pick the ii element of the solution_id column 
    namefile_sel=curr_sel.iloc[ii]['solution_id']
    # find the index of curr_obs_og with the same distance
    index_sel=curr_sel_og[curr_sel_og['solution_id']==namefile_sel].index
    index_sel=index_sel[0]

    # check if the file exist
    if os.path.isfile(namefile_sel):
        # open the json file with the name namefile_sel
        f = open(namefile_sel,"r")
        data = json.loads(f.read())
        # vel_sim=data['simulation_results']['leading_frag_vel_arr']#['brightest_vel_arr']#['leading_frag_vel_arr']#['main_vel_arr']
        # ht_sim=data['simulation_results']['leading_frag_height_arr']#['main_height_arr']
        vel_sim=data['simulation_results']['leading_frag_vel_arr']#['brightest_vel_arr']#['leading_frag_vel_arr']#['main_vel_arr']
        ht_sim=data['simulation_results']['leading_frag_height_arr']#['brightest_height_arr']['leading_frag_height_arr']['main_height_arr']
        # absmag_sim=data['simulation_results']['abs_magnitude']

        obs_height=data['ht_sampled']
        obs_time=data['time_sampled']
        obs_length=data['len_sampled']

        # delete the nan term in vel_sim and ht_sim
        vel_sim=[x for x in vel_sim if str(x) != 'nan']
        ht_sim=[x for x in ht_sim if str(x) != 'nan']

        # find the index of the first element of the simulation that is equal to the first element of the observation
        index_ht_sim=next(x for x, val in enumerate(ht_sim) if val <= obs_height[0])
        # find the index of the last element of the simulation that is equal to the last element of the observation
        index_ht_sim_end=next(x for x, val in enumerate(ht_sim) if val <= obs_height[-1])

        vel_sim=vel_sim[index_ht_sim:index_ht_sim_end]
        ht_sim=ht_sim[index_ht_sim:index_ht_sim_end]

        # pick from the end of vel_sim the same number of element of time_sim
        # vel_sim=vel_sim[-len(ht_sim):]
        # Function to find the index of the closest value

        def find_closest_index(time_arr, time_sampled):
            closest_indices = []
            for sample in time_sampled:
                closest_index = min(range(len(time_arr)), key=lambda i: abs(time_arr[i] - sample))
                closest_indices.append(closest_index)
            return closest_indices
        
        
        # Find and print the closest indices
        closest_indices = find_closest_index(ht_sim, obs_height)

        vel_sim=[vel_sim[i] for i in closest_indices]
        ht_sim=obs_height
        
        height_km=[x/1000 for x in ht_sim]
        vel_kms=[x/1000 for x in vel_sim]
        

        # from data params v_init val
        vel_sampled=[data['params']['v_init']['val']]
        # append from vel_sampled the rest by the difference of the first element of obs_length divided by the first element of obs_time
        rest_vel_sampled=[(obs_length[i]-obs_length[i-1])/(obs_time[i]-obs_time[i-1]) for i in range(1,len(obs_length))]
        # append the rest_vel_sampled to vel_sampled
        vel_sampled.extend(rest_vel_sampled)
        
        vel_sampled=[x/1000 for x in vel_sampled]

        vel_kms=vel_sampled

        obs_length=[x/1000 for x in obs_length]

        
        # fit a line to the throught the vel_sim and ht_sim
        a, b = np.polyfit(ht_sim,vel_sim, 1)
        # create a list of the same length of vel_sim with the value of the line
        vel_sim_line=[a*x+b for x in ht_sim]


        if with_legend:

            if with_LEN==True:
                # difference obs_length between obs_length[i-1] and obs_length[i]
                # diff_obs_length=[obs_length[i]-obs_length[i-1] for i in range(1,len(obs_length))]
                # # the first element of diff_obs_length is the same as the first element of obs_length
                # diff_obs_length.insert(0,diff_obs_length[0])
                # ax[1].plot(diff_obs_length,height_km)

                ax[1].plot(obs_length,height_km)

                ax[2].plot(vel_kms,height_km,label='sel_'+current_shower+'('+str(index_sel)+') dist:'+str(round(curr_sel.iloc[ii]['distance'],2))+'\n\
        m:'+str('{:.2e}'.format(curr_sel.iloc[ii]['mass'],1))+' F:'+str(round(curr_sel.iloc[ii]['F'],2))+'\n\
        rho:'+str(round(curr_sel.iloc[ii]['rho']))+' sigma:'+str(round(curr_sel.iloc[ii]['sigma'],4))+'\n\
        er.height:'+str(round(curr_sel.iloc[ii]['erosion_height_start'],2))+' er.log:'+str(round(curr_sel.iloc[ii]['erosion_range'],1))+'\n\
        er.coeff:'+str(round(curr_sel.iloc[ii]['erosion_coeff'],3))+' er.index:'+str(round(curr_sel.iloc[ii]['erosion_mass_index'],2)))
                

            else:

                ax[1].plot(vel_kms,height_km,label='sel_'+current_shower+'('+str(index_sel)+') dist:'+str(round(curr_sel.iloc[ii]['distance'],2))+'\n\
        m:'+str('{:.2e}'.format(curr_sel.iloc[ii]['mass'],1))+' F:'+str(round(curr_sel.iloc[ii]['F'],2))+'\n\
        rho:'+str(round(curr_sel.iloc[ii]['rho']))+' sigma:'+str(round(curr_sel.iloc[ii]['sigma'],4))+'\n\
        er.height:'+str(round(curr_sel.iloc[ii]['erosion_height_start'],2))+' er.log:'+str(round(curr_sel.iloc[ii]['erosion_range'],1))+'\n\
        er.coeff:'+str(round(curr_sel.iloc[ii]['erosion_coeff'],3))+' er.index:'+str(round(curr_sel.iloc[ii]['erosion_mass_index'],2)))

   
        else:
            # plt.plot(vel_sim,ht_sim,label='sel_'+current_shower+'('+str(index_sel)+') dist:'+str(round(curr_sel.iloc[ii]['distance'],2)),color='coral')
            ax[1].plot(height_km,vel_kms,label='sel_'+current_shower+'('+str(index_sel)+') dist:'+str(round(curr_sel.iloc[ii]['distance'],2)),color='coral')

                

# go back one folder
os.chdir('..')


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

if with_LEN==True:
    # change the first plotted line style to be a dashed line
    ax[2].lines[0].set_linestyle("None")
    # change the first plotted marker to be a x
    ax[2].lines[0].set_marker("x")
    # change first line color
    ax[2].lines[0].set_color('black')
    # change the zorder=-1 of the first line
    ax[2].lines[0].set_zorder(n_confront_sel)
            

# grid on on both subplot with -- as linestyle and light gray color
ax[1].grid(linestyle='--',color='lightgray')

if with_legend:
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

        # plt.legend()

# add the label to the x and y axis
ax[0].set_ylabel('height [km]')
ax[0].set_xlabel('abs.mag [-]')

if with_LEN==True:
    ax[1].set_ylabel('height [km]')
    ax[1].set_xlabel('length [km]')

    ax[2].set_ylabel('height [km]')
    ax[2].set_xlabel('velocity [km/s]')

    ax[2].grid(linestyle='--',color='lightgray')
    # title with the current_shower and written also vel vs height in the second subplot
    ax[1].set_title(current_shower+' len vs height')
    # title with the current_shower and written also vel vs height in the second subplot
    ax[2].set_title(current_shower+' vel vs height')
else:
    # title with the current_shower and written also vel vs height in the second subplot
    ax[1].set_title(current_shower+' vel vs height')
    ax[1].set_ylabel('height [km]')
    ax[1].set_xlabel('velocity [km/s]')

#figManager = plt.get_current_fig_manager()
#figManager.window.showMaximized()
#figManager.resize(*figManager.window.maxsize())
#plt.show()
# plt.figure(figsize=(13,6))

# save inintial_folder+'\\'+current_shower+'_Heigh_MagVel.png'

plt.savefig(initial_folder+'\\'+only_select_meteors_from+'Heigh_MagVel.png')



















if with_LEN==True:
    # put the first plot in 3 sublots
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
else:
    # put the first plot in 2 sublots
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
                

for current_shower in Shower:
    curr_obs_og=df_obs_shower[df_obs_shower['shower_code']==current_shower]
    curr_sel_og=df_sel_shower[df_sel_shower['shower_code']==current_shower+'_sel']

    curr_sel = curr_sel_og.sort_values(by=['distance'])
    curr_sel = curr_sel[curr_sel['distance']>=min_dist_sel]

    curr_obs = curr_obs_og.sort_values(by=['distance'])
    curr_obs = curr_obs[curr_obs['distance']>=min_dist_obs]

    if n_confront_obs<len(df_obs_shower):
        curr_obs=curr_obs.head(n_confront_obs)
    
    if n_confront_sel<len(df_sel_shower):
        curr_sel=curr_sel.head(n_confront_sel)



# find the directory where the script is running
current_folder=os.getcwd()

# in current_folder entern in the folder current_folder+'\\Simulation_'current_shower
os.chdir(current_folder+'\\Simulations_'+current_shower+'\\')
# os.chdir('Simulations_'+current_shower)
for ii in range(len(curr_sel)):
    # pick the ii element of the solution_id column 
    namefile_sel=curr_sel.iloc[ii]['solution_id']
    # find the index of curr_obs_og with the same distance
    index_sel=curr_sel_og[curr_sel_og['solution_id']==namefile_sel].index
    index_sel=index_sel[0]

    # chec if the file exist
    if not os.path.isfile(namefile_sel):
        print('file '+namefile_sel+' not found')
        continue
    else:
        # open the json file with the name namefile_sel
        f = open(namefile_sel,"r")
        data = json.loads(f.read())
        # data['main_vel_arr']
        # ht_sim=data['simulation_results']['main_height_arr']
        # absmag_sim=data['simulation_results']['abs_magnitude']
        # cut out absmag_sim above 7 considering that absmag_sim is a list
        if with_legend:
            # put it in the first subplot
            ax[0].plot(data['time_sampled'],data['mag_sampled'],label='sel_'+current_shower+'('+str(index_sel)+') dist:'+str(round(curr_sel.iloc[ii]['distance'],2)))
        else:
            ax[0].plot(data['time_sampled'],data['mag_sampled'],label='sel_'+current_shower+'('+str(index_sel)+') dist:'+str(round(curr_sel.iloc[ii]['distance'],2)),color='coral')

# if len(df_obs_shower) == 1:
#     namefile_manual=df_obs_shower.iloc[0]['solution_id']
#     # check in the "Manual Reduction" folder if there is a folder name "namefile_manual"
#     # if not, exit
#     if os.path.exists(os.getcwd()+r'/Manual Reduction/'+namefile_manual):
#         folder=namefile_manual.split('_')[0]
#         # put the naeme as a title
#         plt.title(namefile_manual+' abs.mag vs time')
    

# y limit
# plt.ylim(90500, 117500)

# title with the current_shower and written also absolute mag vs height
ax[0].set_title(current_shower+' abs.mag vs time')
# grid on
ax[0].grid(linestyle='--',color='lightgray')
# invert the y axis
ax[0].invert_yaxis()
# if with_legend:
#     plt.legend()
# plt.show()




            

for ii in range(len(curr_sel)):

    # pick the ii element of the solution_id column 
    namefile_sel=curr_sel.iloc[ii]['solution_id']
    # find the index of curr_obs_og with the same distance
    index_sel=curr_sel_og[curr_sel_og['solution_id']==namefile_sel].index
    index_sel=index_sel[0]

    # check if the file exist
    if os.path.isfile(namefile_sel):
        # open the json file with the name namefile_sel
        f = open(namefile_sel,"r")
        data = json.loads(f.read())
        # vel_sim=data['simulation_results']['leading_frag_vel_arr']#['brightest_vel_arr']#['leading_frag_vel_arr']#['main_vel_arr']
        # ht_sim=data['simulation_results']['leading_frag_height_arr']#['main_height_arr']
        vel_sim=data['simulation_results']['leading_frag_vel_arr']#['brightest_vel_arr']#['leading_frag_vel_arr']#['main_vel_arr']
        ht_sim=data['simulation_results']['leading_frag_height_arr']#['brightest_height_arr']['leading_frag_height_arr']['main_height_arr']
        time_sim=data['simulation_results']['time_arr']#['brightest_time_arr']#['leading_frag_time_arr']#['main_time_arr']
        # absmag_sim=data['simulation_results']['abs_magnitude']
        obs_time=data['time_sampled']
        obs_height=data['ht_sampled']
        obs_length=data['len_sampled']

        # delete the nan term in vel_sim and ht_sim
        vel_sim=[x for x in vel_sim if str(x) != 'nan']
        ht_sim=[x for x in ht_sim if str(x) != 'nan']

        # # find the index of the first element of the simulation that is equal to the first element of the observation
        index_ht_sim=next(x for x, val in enumerate(ht_sim) if val <= obs_height[0])
        # find the index of the last element of the simulation that is equal to the last element of the observation
        index_ht_sim_end=next(x for x, val in enumerate(ht_sim) if val <= obs_height[-1])


        # # # delete term with velocity equal 0
        # for jj in range(len(vel_sim)):
        #     if vel_sim[jj]==0:
        #         vel_sim=vel_sim[:jj]
        #         ht_sim=ht_sim[:jj]
        #         time_sim=time_sim[:jj]
        #         break

        # # delete the nan term
        # for jj in range(len(vel_sim)):
        #     if np.isnan(vel_sim[jj]):
        #         vel_sim=vel_sim[:jj]
        #         ht_sim=ht_sim[:jj]
        #         time_sim=time_sim[:jj]
        #         break

        # for jj in range(len(ht_sim)):
        #     if ht_sim[jj]<data['ht_sampled'][0]:
        #         vel_sim=vel_sim[jj:]
        #         ht_sim=ht_sim[jj:]
        #         time_sim=time_sim[jj:]
        #         break
        
        # # create an array of vel_sim and ht_sim that start from index_ht_sim and end at index_ht_sim_end
        vel_sim=vel_sim[index_ht_sim:index_ht_sim_end]
        time_sim=time_sim[index_ht_sim:index_ht_sim_end]

        # pick from the end of vel_sim the same number of element of time_sim
        # vel_sim=vel_sim[-len(time_sim):]

        time_sim=[x-time_sim[0] for x in time_sim]

        vel_kms=[x/1000 for x in vel_sim]

        # Find and print the closest indices
        closest_indices = find_closest_index(time_sim, obs_time)

        vel_kms=[vel_kms[i] for i in closest_indices]
        time_sim=obs_time

        # from data params v_init val
        vel_sampled=[data['params']['v_init']['val']]
        # append from vel_sampled the rest by the difference of the first element of obs_length divided by the first element of obs_time
        rest_vel_sampled=[(obs_length[i]-obs_length[i-1])/(obs_time[i]-obs_time[i-1]) for i in range(1,len(obs_length))]
        # append the rest_vel_sampled to vel_sampled
        vel_sampled.extend(rest_vel_sampled)
        
        vel_sampled=[x/1000 for x in vel_sampled]

        vel_kms=vel_sampled

        obs_length=[x/1000 for x in obs_length]



        
        # fit a line to the throught the vel_sim and ht_sim
        a, b = np.polyfit(time_sim,vel_kms, 1)
        # create a list of the same length of vel_sim with the value of the line
        vel_sim_line=[a*x+b for x in time_sim]

        time_in=[x-time_sim[0] for x in time_sim]

        

        if with_legend:

            if with_LEN==True:
                ax[1].plot(time_in,obs_length)

                ax[2].plot(time_in,vel_kms,label='sel_'+current_shower+'('+str(index_sel)+') dist:'+str(round(curr_sel.iloc[ii]['distance'],2))+'\n\
    m:'+str('{:.2e}'.format(curr_sel.iloc[ii]['mass'],1))+' F:'+str(round(curr_sel.iloc[ii]['F'],2))+'\n\
    rho:'+str(round(curr_sel.iloc[ii]['rho']))+' sigma:'+str(round(curr_sel.iloc[ii]['sigma'],4))+'\n\
    er.height:'+str(round(curr_sel.iloc[ii]['erosion_height_start'],2))+' er.log:'+str(round(curr_sel.iloc[ii]['erosion_range'],1))+'\n\
    er.coeff:'+str(round(curr_sel.iloc[ii]['erosion_coeff'],3))+' er.index:'+str(round(curr_sel.iloc[ii]['erosion_mass_index'],2)))


            else:
            
                ax[1].plot(time_in,vel_kms,label='sel_'+current_shower+'('+str(index_sel)+') dist:'+str(round(curr_sel.iloc[ii]['distance'],2))+'\n\
        m:'+str('{:.2e}'.format(curr_sel.iloc[ii]['mass'],1))+' F:'+str(round(curr_sel.iloc[ii]['F'],2))+'\n\
        rho:'+str(round(curr_sel.iloc[ii]['rho']))+' sigma:'+str(round(curr_sel.iloc[ii]['sigma'],4))+'\n\
        er.height:'+str(round(curr_sel.iloc[ii]['erosion_height_start'],2))+' er.log:'+str(round(curr_sel.iloc[ii]['erosion_range'],1))+'\n\
        er.coeff:'+str(round(curr_sel.iloc[ii]['erosion_coeff'],3))+' er.index:'+str(round(curr_sel.iloc[ii]['erosion_mass_index'],2)))
            
        else:
            # plt.plot(vel_sim,ht_sim,label='sel_'+current_shower+'('+str(index_sel)+') dist:'+str(round(curr_sel.iloc[ii]['distance'],2)),color='coral')
            ax[1].plot(time_in,vel_kms,label='sel_'+current_shower+'('+str(index_sel)+') dist:'+str(round(curr_sel.iloc[ii]['distance'],2)),color='coral')

            

# go back one folder
os.chdir('..')

# y limit
# plt.ylim(90500, 117500)
# plt.xlim(14000, 70500)

if len(df_obs_shower) == 1:
    namefile_manual=df_obs_shower.iloc[0]['solution_id']
    # check in the "Manual Reduction" folder if there is a folder name "namefile_manual"
    # if not, exit
    if os.path.exists(os.getcwd()+r'\\Manual Reduction\\'+namefile_manual):
        folder=namefile_manual.split('_')[0]
        # put the naeme as a super title of the plot
        plt.suptitle(namefile_manual)



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

if with_LEN==True:
    # change the first plotted line style to be a dashed line
    ax[2].lines[0].set_linestyle("None")
    # change the first plotted marker to be a x
    ax[2].lines[0].set_marker("x")
    # change first line color
    ax[2].lines[0].set_color('black')
    # change the zorder=-1 of the first line
    ax[2].lines[0].set_zorder(n_confront_sel)

# title with the current_shower and written also vel vs height in the second subplot
if with_legend:
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



    # plt.legend()


# add the label to the x and y axis
ax[0].set_xlabel('time [s]')
ax[0].set_ylabel('abs.mag [-]')

# grid on on both subplot with -- as linestyle and light gray color
ax[1].grid(linestyle='--',color='lightgray')

if with_LEN==True:
    ax[1].set_ylabel('length [km]')
    ax[1].set_xlabel('time [s]')
    # invert the y axis
    ax[1].invert_yaxis()

    ax[2].set_xlabel('time [s]')
    ax[2].set_ylabel('velocity [km/s]')

    ax[2].grid(linestyle='--',color='lightgray')
    # title with the current_shower and written also vel vs height in the second subplot
    ax[1].set_title(current_shower+' len vs time')
    # title with the current_shower and written also vel vs height in the second subplot
    ax[2].set_title(current_shower+' vel vs time')
else:
    # title with the current_shower and written also vel vs height in the second subplot
    ax[1].set_title(current_shower+' vel vs time')
    ax[1].set_xlabel('time [s]')
    ax[1].set_ylabel('velocity [km/s]')




#figManager = plt.get_current_fig_manager()
# figManager.window.showMaximized()
#figManager.resize(*figManager.window.maxsize())
# plt.show()
# plt.figure(figsize=(13,6))
# fig.set_size_inches(18.5, 10.5, forward=True)
plt.savefig(initial_folder+'\\'+only_select_meteors_from+'Time_MagVel.png')

plt.close('all')
