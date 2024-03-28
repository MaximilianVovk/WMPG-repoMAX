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

#matplotlib.use('Agg')
#matplotlib.use("Qt5Agg")

# MODIFY HERE THE PARAMETERS ###############################################################################


def quadratic_velocity(t, a, v0, t0):
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
    v_after = 3*a*(t_after - t0)**2 + v0

    return np.concatenate((v_before, v_after))

# Set the shower name (can be multiple) e.g. 'GEM' or ['GEM','PER', 'ORI', 'ETA', 'SDA', 'CAP']
def PCA_LightCurveCoefPLOT(output_dir, Shower, input_dir, input_dir_pickle):
    Shower=['PER']#['PER']

    # number of selected events selected
    n_select=1000

    # min distance over which are selected
    min_dist_obs=0
    min_dist_sel=0

    # number to confront
    n_confront_obs=1
    n_confront_sel=4

    only_select_meteors_from='TRUEerosion_sim_v59.84_m1.33e-02g_rho0209_z39.8_abl0.014_eh117.3_er0.636_s1.61.json'

    # no legend for a lot of simulations
    with_legend=True

    # add the lenght of the simulation
    with_LEN=False

    # add the noise to the reduced simulated event
    with_noise=True

    # is the input data noisy
    noise_data_input=False

    # activate jachia
    jacchia_fit=True

    # activate parabolic fit
    parabolic_fit=True

    t0_fit=True

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
    os.chdir(input_dir_pickle+os.sep+'Simulations_'+current_shower+os.sep)
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

            # ht_sim=height_km 

            if with_legend:
                
                if ii==0:
                    
                    if with_noise==True:
                        # from list to array
                        height_km_err=np.array(height_km)
                        abs_mag_sim_err=np.array(abs_mag_sim)

                        # plot noisy area around vel_kms for vel_noise for the fix height_km
                        ax[0].fill_betweenx(height_km_err, abs_mag_sim_err-mag_noise, abs_mag_sim_err+mag_noise, color='lightgray', alpha=0.5)
                        # plot the line
                        abs_mag_sim=obs_abs_mag
                
                if noise_data_input==True:
                    abs_mag_sim=obs_abs_mag

                # put it in the first subplot
                ax[0].plot(abs_mag_sim,height_km,label='sel_'+current_shower+'('+str(index_sel)+') MEANdist:'+str(round(curr_sel.iloc[ii]['distance_meteor'],2)))
                # plot the parabolic curve before the peak_mag_height 


    #############ADD COEF#############################################
                peak_mag_height = curr_sel.iloc[ii]['peak_mag_height']
                # fit a line to the throught the vel_sim and ht_sim
                index_ht_peak = next(x for x, val in enumerate(data['ht_sampled']) if val/1000 <= peak_mag_height)
                #print('index_ht_peak',index_ht_peak)
                # only use first index to pick the height
                height_pickl = [i/1000 for i in data['ht_sampled']]

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
                        ax[1].plot(obs_time, quadratic_velocity(np.array(obs_time), curr_sel.iloc[ii]['decel_t0'], curr_sel.iloc[ii]['vel_init_norot'], curr_sel.iloc[ii]['t0']), color=ax[1].lines[-1].get_color(), linestyle='None', marker='s') 

                    # # Generating synthetic observed data for demonstration
                    # t_observed = np.array(obs_time)  # Observed times
                    # v_init=vel_kms[0]  # Initial velocity
                    # velocity_observed = vel_kms

                    # # Residuals function for optimization
                    # def residuals(params):
                    #     a1, a2 = params
                    #     predicted_velocity = jacchiaVel(t_observed, a1, a2, v_init)
                    #     return np.sum((velocity_observed - predicted_velocity)**2)

                    # # Initial guess for a1 and a2
                    # initial_guess = [0.005,	10]

                    # from scipy.optimize import basinhopping
                    # minimizer_kwargsss = {
                    #     "method": "L-BFGS-B",
                    #     "args": (t_observed, velocity_observed)
                    # }
                    # # Apply basinhopping to minimize the residuals
                    # result = basinhopping(residuals, initial_guess, minimizer_kwargs={"method": "L-BFGS-B"}, niter=100)

                    # # Results
                    # optimized_a1, optimized_a2 = result.x
                    # print(f"Optimized a1: {optimized_a1}, Optimized a2: {optimized_a2}")

                    # ax[1].plot(obs_time, jacchiaVel(t_observed, optimized_a1, optimized_a2, v_init), color=ax[1].lines[-1].get_color(), linestyle='None', marker='d')
                    
                
                    # ax[1].plot(obs_time, jacchiaVel(np.array(obs_time), curr_sel.iloc[ii]['a1_acc_jac'], curr_sel.iloc[ii]['a2_acc_jac']), color=ax[1].lines[-1].get_color(), linestyle='None', marker='d') 

                    # # fit the to a parabolic curve
                    # a3, b3, c3 = np.polyfit(obs_time,vel_kms, 2)

                    # # plot the parabolic curve before the peak_mag_height with the same color of the line
                    # ax[1].plot(obs_time,a3*np.array(obs_time)**2+b3*np.array(obs_time)+c3, color=ax[1].lines[-1].get_color(), linestyle='None', marker='o')# , markersize=5

                    # v_init=vel_kms[0]

                    # def jacchiaVel(t, a1, a2):
                    #     return v_init - np.abs(a1)*np.abs(a2)*np.exp(np.abs(a2)*t)

                    # # Perform the curve fitting
                    # popt, pcov = curve_fit(jacchiaVel, np.array(obs_time), np.array(vel_kms))

                    # # Extract the optimal coefficients
                    # a1_opt, a2_opt = popt

                    # ax[1].plot(obs_time, jacchiaVel(np.array(obs_time), a1_opt, a2_opt), color=ax[1].lines[-1].get_color(), linestyle='None', marker='d')    

                    # if ii==0:
                    #     print(str(curr_sel.iloc[ii]['solution_id']))
                    #     print('Velocity')
                    #     print('a:',a3)
                    #     print('b:',b3)

                    #     print('Jacchia Velocity')
                    #     print('ja:',a1_opt)
                    #     print('jb:',a2_opt)


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


if __name__ == "__main__":

    import argparse


    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Fom Observation and simulated data weselect the most likely through PCA, run it, and store results to disk.")

    arg_parser.add_argument('--output_dir', metavar='OUTPUT_PATH', type=str, default='/home/mvovk/Documents/PCA_Error_propagation/TEST', \
        help="Path to the output directory.")

    arg_parser.add_argument('--shower', metavar='SHOWER', type=str, default='PER', \
        help="Use specific shower from the given simulation.")
    
    arg_parser.add_argument('--input_dir', metavar='INPUT_PATH', type=str, default='/home/mvovk/Documents/PCA_Error_propagation/TEST', \
        help="Path were are store both simulated and observed shower .csv file.")
    
    arg_parser.add_argument('--input_dir_pickle', metavar='INPUT_PATH_PICKLE', type=str, default='/home/mvovk/Documents/PCA_Error_propagation/', \
        help="Path were are store all the .pickle file.")
    
    # arg_parser.add_argument('--input_dir', metavar='INPUT_PATH', type=str, default='/home/mvovk/Documents/PCA_Error_propagation/TEST', \
    #     help="Path were are store both simulated and observed shower .csv file.")
    
    # arg_parser.add_argument('--input_dir', metavar='INPUT_PATH', type=str, default='/home/mvovk/Documents/PCA_Error_propagation/TEST', \
    #     help="Path were are store both simulated and observed shower .csv file.")

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    # make only one shower
    Shower=[cml_args.shower]

    #########################

    PCA_LightCurveCoefPLOT(cml_args.output_dir, Shower, cml_args.input_dir, cml_args.input_dir_pickle)