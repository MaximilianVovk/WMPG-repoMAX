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
from scipy.interpolate import UnivariateSpline
from wmpl.Utils.Pickling import loadPickle

#matplotlib.use('Agg')
#matplotlib.use("Qt5Agg")

# MODIFY HERE THE PARAMETERS ###############################################################################

# Set the shower name (can be multiple) e.g. 'GEM' or ['GEM','PER', 'ORI', 'ETA', 'SDA', 'CAP']
#PCA_confrontPLOT(output_dir, Shower, input_dir, true_file='', true_path=''):
def PCA_LightCurveCoefPLOT(output_dir, Shower, input_dir, true_file='', true_path=''):
    Shower=['PER']#['PER']

    # number of selected events selected
    n_select=1000

    # number to confront
    n_confront_obs=1
    n_confront_sel=10

    if true_file.endswith('.pickle'):
        # delete the extension
        only_select_meteors_from=true_file.replace('_trajectory.pickle','')
        # add the noise to the reduced simulated event
        with_noise=False

    else:
        only_select_meteors_from=true_file
        # add the noise to the reduced simulated event
        with_noise=True

    # no legend for a lot of simulations
    with_legend=True

    # add the lenght of the simulation
    with_LEN=False

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
    mag_noise = 0.1

    # SD of noise in length (m) 1 sigma
    len_noise = 20.0/1000

    # velocity noise 1 sigma
    vel_noise = (len_noise*np.sqrt(2)/0.03125)

    # # 5 sigma confidence interval
    # if five_sigma==True:
    #     # 5 sigma confidence interval
    #     conf_int = 0.999999426696856
    #     # 5 sigma confidence interval
    #     mag_noise = 5*mag_noise
    #     # 5 sigma confidence interval
    #     len_noise = 5*len_noise
    #     # 5 sigma confidence interval
    #     vel_noise = 5*vel_noise

    # min distance over which are selected
    min_dist_obs=0
    min_dist_sel=0

    if with_LEN==True:
        # put the first plot in 3 sublots
        fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    else:
        # put the first plot in 2 sublots
        fig, ax = plt.subplots(1, 2, figsize=(17, 5))

    # save all the simulated showers in a list
    df_obs_shower = []
    df_sel_shower = []
    # search for the simulated showers in the folder
    for current_shower in Shower:
        print(current_shower)

        # check in the current folder there is a csv file with the name of the simulated shower
        df_sel = pd.read_csv(input_dir+os.sep+'Simulated_'+current_shower+'_select.csv')

        # check if there is only_select_meteors_from in any of the solution_id_dist
        if only_select_meteors_from in df_sel['solution_id_dist'].values:
            # keep only the selected meteor wioth the name only_select_meteors_from
            df_sel=df_sel[df_sel['solution_id_dist']==only_select_meteors_from]

        if len(df_sel)>n_select:
            df_sel=df_sel.head(n_select)

        df_obs = pd.read_csv(input_dir+os.sep+current_shower+'_and_dist.csv')

        print('observed: '+str(len(df_obs)))
        # append the observed shower to the list

        df_obs_shower.append(df_obs)

        if only_select_meteors_from!='':
            df_obs=df_obs[df_obs['solution_id']==only_select_meteors_from]

            # reset index df_sel
            df_sel=df_sel.reset_index(drop=True)

            # check if present the selected meteor in df_sel
            if only_select_meteors_from in df_sel['solution_id'].values:
                
                
                # place that meteor in the top row shifiting all the other down
                # find the index of the selected meteor
                index_sel=df_sel[df_sel['solution_id']==only_select_meteors_from].index

                print('selected meteor : '+only_select_meteors_from+'\n'+'at '+str(index_sel))
                
                row_to_move = df_sel.iloc[index_sel]
                df_dropped = df_sel.drop(df_sel.index[index_sel])
                # drop the selected meteor
                df_sel=pd.concat([row_to_move, df_dropped]).reset_index(drop=True)

            else:
                print('NOT found selected meteor : '+only_select_meteors_from)
                # add the selected meteor to the df_sel in the first row
                df_sel=pd.concat([df_obs, df_sel]).reset_index(drop=True)

        # append the simulated shower to the list
        df_sel_shower.append(df_sel)

        df_PCA_columns = pd.read_csv(input_dir+os.sep+'Simulated_'+current_shower+'_select_PCA.csv')
        # fid the numbr of columns
        n_PC_in_PCA=str(len(df_PCA_columns.columns)-1)+'PC'
        # print the number of selected events
        print('The PCA space has '+str(n_PC_in_PCA))


    # print(df_obs)

    # # select the one below 1 in distance and order for distance
    df_sel_shower = pd.concat(df_sel_shower)
    # df_sel_shower = df_sel_shower.sort_values(by=['distance_meteor'])
    # df_sel_shower = df_sel_shower[df_sel_shower['distance_meteor']<set_dist]
    # print('selected: '+str(len(df_sel_shower)))
    # # same for the observed
    df_obs_shower = pd.concat(df_obs_shower)
    # df_obs_shower = df_obs_shower.sort_values(by=['distance_meteor'])
    # df_obs_shower = df_obs_shower[df_obs_shower['distance_meteor']<set_dist]
    # print('observed: '+str(len(df_obs_shower)))

    df_sel_shower['erosion_coeff']=df_sel_shower['erosion_coeff']*1000000
    df_sel_shower['sigma']=df_sel_shower['sigma']*1000000


    def find_closest_index(time_arr, time_sampled):
        closest_indices = []
        for sample in time_sampled:
            closest_index = min(range(len(time_arr)), key=lambda i: abs(time_arr[i] - sample))
            closest_indices.append(closest_index)
        return closest_indices











    if df_obs_shower is None:
        print('no observed shower found')
        exit()

    for current_shower in Shower:
        curr_obs_og=df_obs_shower[df_obs_shower['shower_code']==current_shower]
        curr_sel_og=df_sel_shower[df_sel_shower['shower_code']==current_shower+'_sel']

        curr_sel = curr_sel_og
        # curr_sel = curr_sel_og.sort_values(by=['distance_meteor'])
        # curr_sel = curr_sel[curr_sel['distance_meteor']>=min_dist_sel]
        
        curr_obs = curr_obs_og
        # curr_obs = curr_obs_og.sort_values(by=['distance_meteor'])
        # curr_obs = curr_obs[curr_obs['distance_meteor']>=min_dist_obs]

        if n_confront_obs<len(df_obs_shower):
            curr_obs=curr_obs.head(n_confront_obs)
        
        if n_confront_sel<len(df_sel_shower):
            curr_sel=curr_sel.head(n_confront_sel)



    # go back one folder
    # os.chdir('..')

    # find the directory where the script is running

    # in current_folder entern in the folder current_folder+os.sep+'Simulation_'current_shower
    os.chdir(true_path)
    # os.chdir('Simulations_'+current_shower)
    for ii in range(len(curr_sel)):
        # pick the ii element of the solution_id column 
        namefile_sel=curr_sel.iloc[ii]['solution_id']
        # find the index of curr_obs_og with the same distance
        index_sel=curr_sel_og[curr_sel_og['solution_id']==namefile_sel].index
        index_sel=index_sel[0]
        
        flag_pickl=False
        # if there is _trajectory.pickle in true_file
        if '_trajectory.pickle' in true_file:
            if namefile_sel==true_file.replace('_trajectory.pickle',''):
                # add the name of the file to the path
                namefile_sel=namefile_sel+'_sim_fit.json'
                # add the true path to the path
                namefile_sel=true_path+os.sep+namefile_sel
                flag_pickl=True

        # chec if the file exist
        if not os.path.isfile(namefile_sel):
            print('file '+namefile_sel+' not found')
            continue
        else:
            # open the json file with the name namefile_sel
            f = open(namefile_sel,"r")
            data = json.loads(f.read())
            if flag_pickl==True:

                traj = loadPickle(true_path,true_file)

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
                height_km=ht_obs
                obs_abs_mag=abs_mag_sim
                height_pickl = ht_obs


                erosion_height_start = data['erosion_height_start']/1000
                data_index_2cam = pd.DataFrame(list(zip(obs_time, ht_obs, obs_vel, abs_mag_sim)), columns =['time_sampled', 'ht_sampled', 'vel_sampled', 'mag_sampled'])

                peak_mag_height = curr_sel.iloc[ii]['peak_mag_height']
                # fit a line to the throught the vel_sim and ht_sim
                index_ht_peak = next(x for x, val in enumerate(ht_obs) if val <= peak_mag_height)


            else:
                # data['main_vel_arr']
                # ht_sim=data['simulation_results']['main_height_arr']
                # absmag_sim=data['simulation_results']['abs_magnitude']
                # cut out absmag_sim above 7 considering that absmag_sim is a list
                
                abs_mag_sim=data['simulation_results']['abs_magnitude']#['brightest_vel_arr']#['leading_frag_vel_arr']#['main_vel_arr']
                ht_sim=data['simulation_results']['leading_frag_height_arr']#['brightest_height_arr']['leading_frag_height_arr']['main_height_arr']
                time_sim=data['simulation_results']['time_arr']#['brightest_time_arr']#['leading_frag_time_arr']#['main_time_arr']

                obs_abs_mag= data['mag_sampled']
                obs_height= data['ht_sampled']
                obs_time= data['time_sampled']


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

                time_sim=[x-time_sim[0] for x in time_sim]

                # Find and print the closest indices
                closest_indices = find_closest_index(time_sim, obs_time)

                abs_mag_sim=[abs_mag_sim[i] for i in closest_indices]

                height_km=[x/1000 for x in obs_height]

                peak_mag_height = curr_sel.iloc[ii]['peak_mag_height']
                # fit a line to the throught the vel_sim and ht_sim
                index_ht_peak = next(x for x, val in enumerate(data['ht_sampled']) if val/1000 <= peak_mag_height)
                #print('index_ht_peak',index_ht_peak)
                # only use first index to pick the height
                height_pickl = [i/1000 for i in data['ht_sampled']]

            # ht_sim=height_km 

            if with_legend:
                
                if ii==0:
                    
                    if with_noise==True:
                        # from list to array
                        height_km_err=np.array(height_km)
                        abs_mag_sim_err=np.array(abs_mag_sim)

                        # plot noisy area around vel_kms for vel_noise for the fix height_km
                        ax[0].fill_betweenx(height_km_err, abs_mag_sim_err-mag_noise, abs_mag_sim_err+mag_noise, color='lightgray', alpha=0.5)
                        # spline and smooth the data
                        # spline_mag = UnivariateSpline(obs_time, abs_mag_sim_err, s=0.1)  # 's' is the smoothing factor
                        # ax[0].plot(spline_mag(obs_time),height_km)
                        # plot the line
                        abs_mag_sim=obs_abs_mag
                
                if noise_data_input==True:
                    abs_mag_sim=obs_abs_mag

                # put it in the first subplot
                ax[0].plot(abs_mag_sim,height_km,label='sel_'+current_shower+'('+str(index_sel)+') MEANdist:'+str(round(curr_sel.iloc[ii]['distance_meteor'],2)))
                # plot the parabolic curve before the peak_mag_height 


    #############ADD COEF#############################################
                if mag_fit==True:

                    ax[0].plot(curr_sel.iloc[ii]['a_mag_init']*np.array(height_pickl[:index_ht_peak])**2+curr_sel.iloc[ii]['b_mag_init']*np.array(height_pickl[:index_ht_peak])+curr_sel.iloc[ii]['c_mag_init'],height_pickl[:index_ht_peak], color=ax[0].lines[-1].get_color(), linestyle='None', marker='<')# , markersize=5

                    ax[0].plot(curr_sel.iloc[ii]['a_mag_end']*np.array(height_pickl[index_ht_peak:])**2+curr_sel.iloc[ii]['b_mag_end']*np.array(height_pickl[index_ht_peak:])+curr_sel.iloc[ii]['c_mag_end'],height_pickl[index_ht_peak:], color=ax[0].lines[-1].get_color(), linestyle='None', marker='>')# , markersize=5
                    
                    # a3_Inabs, b3_Inabs, c3_Inabs = np.polyfit(height_pickl[:index_ht_peak], abs_mag_sim[:index_ht_peak], 2)
                    # # plot the parabolic curve before the peak_mag_height with the same color of the line
                    # ax[0].plot(a3_Inabs*np.array(height_pickl[:index_ht_peak])**2+b3_Inabs*np.array(height_pickl[:index_ht_peak])+c3_Inabs,height_pickl[:index_ht_peak], color=ax[0].lines[-1].get_color(), linestyle='None', marker='<')# , markersize=5
                    
                    # if ii==0:
                    #     print(str(curr_sel.iloc[ii]['solution_id']))
                    #     print('Abs.Mag before peak')
                    #     print('a:',a3_Inabs)
                    #     print('b:',b3_Inabs)

                    # # the other side of the peak
                    # a3_Inabs, b3_Inabs, c3_Inabs = np.polyfit(height_pickl[index_ht_peak:], abs_mag_sim[index_ht_peak:], 2)
                    # # plot the parabolic curve before the peak_mag_height with the same color of the line
                    # ax[0].plot(a3_Inabs*np.array(height_pickl[index_ht_peak:])**2+b3_Inabs*np.array(height_pickl[index_ht_peak:])+c3_Inabs,height_pickl[index_ht_peak:], color=ax[0].lines[-1].get_color(), linestyle='None', marker='>')# , markersize=5
                    
                    # if ii==0:
                    #     print('Abs.Mag after peak')
                    #     print('a:',a3_Inabs)
                    #     print('b:',b3_Inabs)

            else:
                ax[0].plot(abs_mag_sim,height_km,label='sel_'+current_shower+'('+str(index_sel)+') MEANdist:'+str(round(curr_sel.iloc[ii]['distance_meteor'],2)),color='coral')



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

        flag_pickl=False
        # if there is _trajectory.pickle in true_file
        if '_trajectory.pickle' in true_file:
            if namefile_sel==true_file.replace('_trajectory.pickle',''):
                # add the name of the file to the path
                namefile_sel=namefile_sel+'_sim_fit.json'
                # add the true path to the path
                namefile_sel=true_path+os.sep+namefile_sel
                flag_pickl=True

        # check if the file exist
        if os.path.isfile(namefile_sel):
            # open the json file with the name namefile_sel
            f = open(namefile_sel,"r")
            data = json.loads(f.read())

            if flag_pickl==True:

                traj = loadPickle(true_path,true_file)

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
                height_km=ht_obs
                vel_kms=obs_vel
                vel_sampled=obs_vel

                erosion_height_start = data['erosion_height_start']/1000
                data_index_2cam = pd.DataFrame(list(zip(obs_time, ht_obs, obs_vel, abs_mag_sim)), columns =['time_sampled', 'ht_sampled', 'vel_sampled', 'mag_sampled'])

            else:
            
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


                
                
                # Find and print the closest indices
                closest_indices = find_closest_index(ht_sim, obs_height)

                vel_sim=[vel_sim[i] for i in closest_indices]
                ht_sim=obs_height
                
                height_km=[x/1000 for x in ht_sim]
                vel_kms=[x/1000 for x in vel_sim]
                
                # create a list of the same length of obs_time with the value of the first element of vel_sim
                vel_sampled=vel_sim
                # from data params v_init val
                vel_sampled[0]=data['params']['v_init']['val']
                # # append from vel_sampled the rest by the difference of the first element of obs_length divided by the first element of obs_time
                # rest_vel_sampled=[(obs_length[i]-obs_length[i-1])/(obs_time[i]-obs_time[i-1]) for i in range(1,len(obs_length))]
                # # append the rest_vel_sampled to vel_sampled
                # vel_sampled.extend(rest_vel_sampled)

                for vel_ii in range(1,len(obs_time)):
                    if obs_time[vel_ii]-obs_time[vel_ii-1]<0.03125:
                    # if obs_time[vel_ii] % 0.03125 < 0.000000001:
                        # vel_sampled[vel_ii]=data['params']['v_init']['val']
                        if vel_ii+1<len(obs_length):
                            vel_sampled[vel_ii+1]=(obs_length[vel_ii+1]-obs_length[vel_ii-1])/(obs_time[vel_ii+1]-obs_time[vel_ii-1])
                    else:
                        vel_sampled[vel_ii]=(obs_length[vel_ii]-obs_length[vel_ii-1])/(obs_time[vel_ii]-obs_time[vel_ii-1])

                vel_sampled=[x/1000 for x in vel_sampled]

                # vel_kms=vel_sampled

                obs_length=[x/1000 for x in obs_length]

                
                # fit a line to the throught the vel_sim and ht_sim
                a, b = np.polyfit(ht_sim,vel_sim, 1)

                # create a list of the same length of vel_sim with the value of the line
                vel_sim_line=[a*x+b for x in ht_sim]


            if with_legend:

                if with_LEN==True:

                    if ii==0:
                        
                        if with_noise==True:
                            height_km_err=np.array(height_km)
                            vel_kms_err=np.array(vel_kms)
                            obs_length_err=np.array(obs_length)

                            ax[1].fill_betweenx(height_km_err, obs_length_err-len_noise, obs_length_err+len_noise, color='lightgray', alpha=0.5)
                            # plot noisy area around vel_kms for vel_noise for the fix height_km
                            ax[2].fill_betweenx(height_km_err, vel_kms_err-vel_noise, vel_kms_err+vel_noise, color='lightgray', alpha=0.5)
                            # plot the line
                            vel_kms=vel_sampled

                    ax[1].plot(obs_length,height_km)

                    ax[2].plot(vel_kms,height_km,label='sel_'+current_shower+'('+str(index_sel)+') dist:'+str(round(curr_sel.iloc[ii]['distance_meteor'],2))+'\n\
            m:'+str('{:.2e}'.format(curr_sel.iloc[ii]['mass'],1))+' F:'+str(round(curr_sel.iloc[ii]['F'],2))+'\n\
            rho:'+str(round(curr_sel.iloc[ii]['rho']))+' sigma:'+str(round(curr_sel.iloc[ii]['sigma'],4))+'\n\
            er.height:'+str(round(curr_sel.iloc[ii]['erosion_height_start'],2))+' er.log:'+str(round(curr_sel.iloc[ii]['erosion_range'],1))+'\n\
            er.coeff:'+str(round(curr_sel.iloc[ii]['erosion_coeff'],3))+' er.index:'+str(round(curr_sel.iloc[ii]['erosion_mass_index'],2)))

                        
                else:
                    if ii==0:

                        if with_noise==True:
                            # from list to array
                            height_km_err=np.array(height_km)
                            vel_kms_err=np.array(vel_kms)

                            # plot noisy area around vel_kms for vel_noise for the fix height_km
                            ax[1].fill_between(obs_time, vel_kms_err-vel_noise, vel_kms_err+vel_noise, color='lightgray', alpha=0.5)
                            # plot the line
                            vel_kms=vel_sampled

                    if noise_data_input==True:
                        vel_kms=vel_sampled

                    ax[1].plot(obs_time, vel_kms,label='sel_'+current_shower+'('+str(index_sel)+') dist:'+str(round(curr_sel.iloc[ii]['distance_meteor'],2))+'\n\
            m:'+str('{:.2e}'.format(curr_sel.iloc[ii]['mass'],1))+' F:'+str(round(curr_sel.iloc[ii]['F'],2))+'\n\
            rho:'+str(round(curr_sel.iloc[ii]['rho']))+' sigma:'+str(round(curr_sel.iloc[ii]['sigma'],4))+'\n\
            er.height:'+str(round(curr_sel.iloc[ii]['erosion_height_start'],2))+' er.log:'+str(round(curr_sel.iloc[ii]['erosion_range'],1))+'\n\
            er.coeff:'+str(round(curr_sel.iloc[ii]['erosion_coeff'],3))+' er.index:'+str(round(curr_sel.iloc[ii]['erosion_mass_index'],2)))


        ############### add coefficient to the plot ############################################  
                    if parabolic_fit==True:
                        ax[1].plot(obs_time,curr_sel.iloc[ii]['a_acc']*np.array(obs_time)**2+curr_sel.iloc[ii]['b_acc']*np.array(obs_time)+curr_sel.iloc[ii]['c_acc'], color=ax[1].lines[-1].get_color(), linestyle='None', marker='o')# , markersize=5
                    
                    # Assuming the jacchiaVel function is defined as:
                    def jacchiaVel(t, a1, a2, v_init):
                        return v_init - np.abs(a1) * np.abs(a2) * np.exp(np.abs(a2) * t)
                    if jacchia_fit==True:
                        ax[1].plot(obs_time, jacchiaVel(np.array(obs_time), curr_sel.iloc[ii]['a1_acc_jac'], curr_sel.iloc[ii]['a2_acc_jac'],vel_kms[0]), color=ax[1].lines[-1].get_color(), linestyle='None', marker='d') 

                    if t0_fit==True: # quadratic_velocity(t, a, v0, t0)
                        ax[1].plot(obs_time, quadratic_velocity(np.array(obs_time), curr_sel.iloc[ii]['a_t0'], curr_sel.iloc[ii]['b_t0'], curr_sel.iloc[ii]['vel_init_norot'], curr_sel.iloc[ii]['t0']), color=ax[1].lines[-1].get_color(), linestyle='None', marker='s') 


            else:

                if ii==0:
                    vel_kms=vel_sampled
                # plt.plot(vel_sim,ht_sim,label='sel_'+current_shower+'('+str(index_sel)+') dist:'+str(round(curr_sel.iloc[ii]['distance_meteor'],2)),color='coral')
                ax[1].plot(height_km,vel_kms,label='sel_'+current_shower+'('+str(index_sel)+') dist:'+str(round(curr_sel.iloc[ii]['distance_meteor'],2)),color='coral')

                    

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
    # invert the x axis
    ax[0].invert_xaxis()

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
        ax[1].set_ylabel('velocity [km/s]')
        ax[1].set_xlabel('time [s]')

    #figManager = plt.get_current_fig_manager()
    #figManager.window.showMaximized()
    #figManager.resize(*figManager.window.maxsize())
    #plt.show()
    # plt.figure(figsize=(13,6))

    # save inintial_folder+os.sep+''+current_shower+'_Heigh_MagVel.png'

    plt.savefig(output_dir+os.sep+'Heigh_MagVelCoef'+str(n_PC_in_PCA)+'.png')

    # close the plot
    plt.close()


def quadratic_velocity(t, a, b, v0, t0):
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


if __name__ == "__main__":

    import argparse


    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Fom Observation and simulated data weselect the most likely through PCA, run it, and store results to disk.")

    # arg_parser.add_argument('--output_dir', metavar='OUTPUT_PATH', type=str, default=r"C:\Users\maxiv\Documents\UWO\Papers\1)PCA\PCA_Error_propagation\TEST", \
    #     help="Path to the output directory.")

    # arg_parser.add_argument('--shower', metavar='SHOWER', type=str, default='PER', \
    #     help="Use specific shower from the given simulation.")
    
    # arg_parser.add_argument('--input_dir', metavar='INPUT_PATH', type=str, default=r"C:\Users\maxiv\Documents\UWO\Papers\1)PCA\PCA_Error_propagation\TEST", \
    #     help="Path were are store both simulated and observed shower .csv file.")
    
    # arg_parser.add_argument('--true_file', metavar='TRUE_PICKLE', type=str, default='TRUEerosion_sim_v60.05_m1.05e-04g_rho0588_z39.3_abl0.009_eh108.3_er0.763_s2.08.json', \
    #     help="the real .pickle file name.")
    
    # arg_parser.add_argument('--true_path', metavar='TRUE_PATH', type=str, default=r"C:\Users\maxiv\Documents\UWO\Papers\1)PCA\PCA_Error_propagation\Simulations_PER", \
    #     help="Path were are store all the .pickle file.")
    
    # arg_parser.add_argument('--output_dir', metavar='OUTPUT_PATH', type=str, default=r"C:\Users\maxiv\Documents\UWO\Papers\1)PCA\PCA_Error_propagation\TEST", \
    arg_parser.add_argument('--output_dir', metavar='OUTPUT_PATH', type=str, default=r"C:\Users\maxiv\Documents\UWO\Papers\1)PCA\Reproces_2cam\SimFolder\TEST", \
        help="Path to the output directory.")

    arg_parser.add_argument('--shower', metavar='SHOWER', type=str, default='PER', \
        help="Use specific shower from the given simulation.")
    
    # arg_parser.add_argument('--input_dir', metavar='INPUT_PATH', type=str, default=r"C:\Users\maxiv\Documents\UWO\Papers\1)PCA\PCA_Error_propagation\TEST", \
    arg_parser.add_argument('--input_dir', metavar='INPUT_PATH', type=str, default=r"C:\Users\maxiv\Documents\UWO\Papers\1)PCA\Reproces_2cam\SimFolder\TEST", \
        help="Path were are store both simulated and observed shower .csv file.")

    # arg_parser.add_argument('--true_file', metavar='TRUE_FILE', type=str, default='TRUEerosion_sim_v65.00_m7.01e-04g_rho0709_z51.7_abl0.015_eh115.2_er0.483_s2.46.json', \ TRUEerosion_sim_v59.84_m1.33e-02g_rho0209_z39.8_abl0.014_eh117.3_er0.636_s1.61.json
    arg_parser.add_argument('--true_file', metavar='TRUE_FILE', type=str, default='20230811_082648_trajectory.pickle', \
        help="The real json file the ground truth for the PCA simulation results.") 

    # arg_parser.add_argument('--input_dir_true', metavar='INPUT_PATH_TRUE', type=str, default=r"C:\Users\maxiv\Documents\UWO\Papers\1)PCA\PCA_Error_propagation\Simulations_PER", \
    arg_parser.add_argument('--input_dir_true', metavar='INPUT_PATH_TRUE', type=str, default=r"C:\Users\maxiv\Documents\UWO\Papers\1)PCA\Reproces_2cam\SimFolder\Simulations_PER", \
        help="Path to the real file the ground truth for the PCA simulation results.") 

    # arg_parser.add_argument('--input_dir', metavar='INPUT_PATH', type=str, default='/home/mvovk/Documents/PCA_Error_propagation/TEST', \
    #     help="Path were are store both simulated and observed shower .csv file.")
    
    # arg_parser.add_argument('--input_dir', metavar='INPUT_PATH', type=str, default='/home/mvovk/Documents/PCA_Error_propagation/TEST', \
    #     help="Path were are store both simulated and observed shower .csv file.")

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    # make only one shower
    Shower=[cml_args.shower]

    #########################

    PCA_LightCurveCoefPLOT(cml_args.output_dir, Shower, cml_args.input_dir, cml_args.true_file, cml_args.input_dir_true)