"""
The code is used to extract the physical properties of the simulated showers from EMCCD observations
by selecting the most similar simulated shower.
The code is divided in three parts:
    1. from GenerateSimulations.py output folder extract the simulated showers observable and physiscal characteristics
    2. extract from the EMCCD solution_table.json file the observable property of the shower
    3. select the simulated meteors similar to the EMCCD meteor observations and extract their physical properties
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import seaborn as sns
import scipy.spatial.distance
import scipy.stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import gaussian_kde
from scipy.optimize import minimize
from wmpl.MetSim.GUI import loadConstants, SimulationResults
from wmpl.MetSim.MetSimErosion import runSimulation, Constants
from wmpl.Utils.Pickling import loadPickle
from wmpl.Utils.OSTools import mkdirP
from sklearn.cluster import KMeans
from matplotlib.colors import Normalize
from matplotlib.ticker import ScalarFormatter,MaxNLocator
import copy
import sys
from scipy.integrate import simps  # For numerical integration
from Plots_LightCurves_onlysim_coef import PCA_LightCurveCoefPLOT
from sklearn.preprocessing import StandardScaler
import itertools

# MODIFY HERE THE PARAMETERS ###############################################################################


def PCA_confrontPLOT(output_dir, Shower, input_dir, true_file='', true_path=''):
    # Set the shower name (can be multiple) e.g. 'GEM' or ['GEM','PER', 'ORI', 'ETA', 'SDA', 'CAP']
    # Shower=['GEM', 'PER', 'ORI', 'ETA', 'SDA', 'CAP']
    # Shower=['PER']#['CAP']

    # number of selected events selected
    n_select=10
    dist_select=np.array([10000000000000])
    # dist_select=np.ones(9)*10000000000000

    # weight factor for the distance
    distance_weight_fact=0
    # if true_fil has a pickle extension
    if true_file.endswith('.pickle'):
        # delete the extension
        only_select_meteors_from=true_file.replace('_trajectory.pickle','')
        do_not_select_meteor=[true_file.replace('_trajectory.pickle','')]
    else:
        only_select_meteors_from=true_file
        do_not_select_meteor=[true_file]

    # do_not_select_meteor=[]

    Sim_data_distribution=True

    curvature_selection_diff=True
    window_of_smothing_avg=3
    std_multip_threshold=1

    plot_dist=True

    plot_var=True
    
    # Standard deviation of the magnitude Gaussian noise 1 sigma
    mag_noise = 0.1

    # SD of noise in length (m) 1 sigma
    len_noise = 20.0/1000

    # velocity noise 1 sigma
    vel_noise = (len_noise*np.sqrt(2)/0.03125)

    # dist_select=[1,\
    #                    1,\
    #                    1.72,\
    #                    2.2,\
    #                    1.35,\
    #                    1,\
    #                    1.1,\
    #                    0.85,\
    #                    1.16]

    # dist_select=[1000000000000,\
    #                 1000000000000,\
    #                 1000000000000,\
    #                 1000000000000,\
    #                 1000000000000,\
    #                 1000000000000,\
    #                 1000000000000,\
    #                 1000000000000,\
    #                 1000000000000]

    # FUNCTIONS ###########################################################################################


    # save all the simulated showers in a list
    df_sim_shower = []
    df_obs_shower = []
    df_sel_shower = []
    # search for the simulated showers in the folder
    for current_shower in Shower:
        print('\n'+current_shower)

        df_obs = pd.read_csv(input_dir+os.sep+current_shower+'_and_dist.csv')

        # if there only_select_meteors_from is equal to any solution_id_dist
        if only_select_meteors_from in df_obs['solution_id'].values:
            # select only the one with the similar name as only_select_meteors_from in solution_id_dist for df_sel
            df_obs=df_obs[df_obs['solution_id']==only_select_meteors_from]
            print('Observed event : '+only_select_meteors_from)
        else:
            print('observed: '+str(len(df_obs)))
        # weight=((1/df_obs['distance']))**distance_weight_fact
        # df_obs['weight']=weight/weight.sum()
        df_obs['weight']=1/len(df_obs)
        # append the observed shower to the list
        df_obs_shower.append(df_obs)
        

        # check in the current folder there is a csv file with the name of the simulated shower
        df_sim = pd.read_csv(input_dir+os.sep+'Simulated_'+current_shower+'.csv')
        print('simulation: '+str(len(df_sim)))
        # simulation with acc positive
        df_sim['weight']=1/len(df_sim)
        # df_sim['weight']=0.00000000000000000000001
        df_PCA_columns = pd.read_csv(input_dir+os.sep+'Simulated_'+current_shower+'_select_PCA.csv')
        # fid the numbr of columns
        n_PC_in_PCA=str(len(df_PCA_columns.columns)-1)+'PC'
        # print the number of selected events
        print('The PCA space has '+str(n_PC_in_PCA))

        # append the simulated shower to the list
        df_sim_shower.append(df_sim)

        # check in the current folder there is a csv file with the name of the simulated shower
        df_sel = pd.read_csv(input_dir+os.sep+'Simulated_'+current_shower+'_select.csv')
        df_sel_save = pd.read_csv(input_dir+os.sep+'Simulated_'+current_shower+'_select.csv')
        df_sel_save_dist = df_sel_save

        n_sample_noise=len(df_sel['solution_id_dist'].unique())

        flag_remove=False
        # check if the do_not_select_meteor any of the array value is in the solution_id of the df_sel if yes remove it
        for i in range(len(do_not_select_meteor)):
            if do_not_select_meteor[i] in df_sel['solution_id'].values:
                df_sel=df_sel[df_sel['solution_id']!=do_not_select_meteor[i]]
                df_sel_save_dist=df_sel_save[df_sel_save['solution_id']!=do_not_select_meteor[i]]
                print('removed: '+do_not_select_meteor[i])
                flag_remove=True

        if curvature_selection_diff==True:

            if Sim_data_distribution==True or Sim_data_distribution==False and n_sample_noise==1:
                # if there only_select_meteors_from is equal to any solution_id_dist
                if only_select_meteors_from in df_sel['solution_id_dist'].values:
                    # select only the one with the similar name as only_select_meteors_from in solution_id_dist for df_sel
                    df_sel=df_sel[df_sel['solution_id_dist']==only_select_meteors_from]
                    df_sel_save=df_sel_save[df_sel_save['solution_id_dist']==only_select_meteors_from]
                    df_sel_save_dist=df_sel_save_dist[df_sel_save_dist['solution_id_dist']==only_select_meteors_from]
                #     print('selected events for : '+only_select_meteors_from)
                # print(len(df_sel))
                dist_to_cut=diff_dist_index(df_sel,window_of_smothing_avg,std_multip_threshold)

                # change of curvature print  
                # print('Change of curvature at:'+str(dist_to_cut))

                # get the data from df_sel upto the dist_to_cut
                df_sel=df_sel.iloc[:dist_to_cut+1]  

            elif Sim_data_distribution==False:
                # create a for loop for each different solution_id_dist in df_sel
                df_app=[]
                for around_meteor in df_sel['solution_id_dist'].unique():
                    # select the data with distance less than dist_select and check if there are more than n_select
                    df_curr_sel_curv = df_sel[df_sel['solution_id_dist']==around_meteor]
                    dist_to_cut=diff_dist_index(df_curr_sel_curv,window_of_smothing_avg,std_multip_threshold)
                    # # change of curvature print
                    # print(around_meteor)
                    # print('- Curvature change in the first '+str(dist_to_cut)+' at a distance of: '+str(df_curr_sel_curv['distance_meteor'].iloc[dist_to_cut]))
                    # get the data from df_sel upto the dist_to_cut
                    dist_to_cut=df_curr_sel_curv.iloc[:dist_to_cut+1]

                    # print(dist_to_cut)
                    df_app.append(dist_to_cut)
                df_sel=pd.concat(df_app)
                # print(df_sel['solution_id_dist'])
                # print(df_sel["solution_id_dist"].unique())
                # print(df_sel_save["solution_id_dist"].unique())


        else:
            if Sim_data_distribution==True:
                n_sample_noise=1
                # if there only_select_meteors_from is equal to any solution_id_dist
                if only_select_meteors_from in df_sel['solution_id_dist'].values:
                    # select only the one with the similar name as only_select_meteors_from in solution_id_dist for df_sel
                    df_sel=df_sel[df_sel['solution_id_dist']==only_select_meteors_from]
                    df_sel_save=df_sel_save[df_sel_save['solution_id_dist']==only_select_meteors_from]
                    print('selected events for : '+only_select_meteors_from)

                if len(df_sel)>n_select:
                    df_sel=df_sel.head(n_select)
            elif Sim_data_distribution==False:
                # pick the first n_select for each set of solution_id_dist selected event
                df_sel=df_sel.groupby('solution_id_dist').head(n_select)
                # print the number of selected events
                print('selected events for each case below the value: '+str(len(df_sel)))


            
            # # find the one with the same solution_id_dist and select the first n_select in df_sel
            # for i in range(len(df_sel)):
            #     df_sel['solution_id_dist'].iloc[i]=df_sel['solution_id_dist'].iloc[i].split('_')[0]

            # if dist_select has more than one element
            if n_sample_noise==1:
                dist_select_1=dist_select[0]
                # select the data with distance less than dist_select and check if there are more than n_select
                if np.min(df_sel['distance_meteor'])>dist_select_1:
                    print('No selected event below the given minimum distance :'+str(dist_select))
                    print('SEL = MAX dist: '+str(np.round(np.max(df_sel['distance_meteor']),2)) +' min dist:'+str(np.round(np.min(df_sel['distance_meteor']),2)))
                    print('OBS = MAX mean dist: '+str(np.round(np.max(df_obs['distance']),2)) +' min mean dist:'+str(np.round(np.min(df_obs['distance']),2)))
                else:
                    df_sel=df_sel[df_sel['distance_meteor']<dist_select_1]
                    # print the number of selected events
                    print('selected events below the value: '+str(len(df_sel)))
                    print('SEL = MAX dist: '+str(np.round(np.max(df_sel['distance_meteor']),2)) +' min dist:'+str(np.round(np.min(df_sel['distance_meteor']),2)))
                    print('OBS = MAX mean dist: '+str(np.round(np.max(df_obs['distance']),2)) +' min mean dist:'+str(np.round(np.min(df_obs['distance']),2)))

      
        # weight=((1/df_sel['distance']))**distance_weight_fact
        weight=((1/(df_sel['distance_meteor']+0.001)))**distance_weight_fact 
        df_sel['weight']=weight/weight.sum()
        # df_sel['weight']=0.00000000000000000000001
        # df_sel=df_sel[df_sel['acceleration']>0]
        # df_sel=df_sel[df_sel['acceleration']<100]
        # df_sel=df_sel[df_sel['trail_len']<50]
        # append the simulated shower to the list
        df_sel_shower.append(df_sel)

    ########## txt file for the print ############################################################

    # check if a file with the name "log"+n_PC_in_PCA+"_"+str(len(df_sel))+"ev.txt" already exist
    if os.path.exists(output_dir+os.sep+"log"+n_PC_in_PCA+"_"+str(len(df_sel))+"ev.txt"):
        # remove the file
        os.remove(output_dir+os.sep+"log"+n_PC_in_PCA+"_"+str(len(df_sel))+"ev.txt")
    sys.stdout = Logger(output_dir, "log"+n_PC_in_PCA+"_"+str(len(df_sel))+"ev.txt")

    ########## txt file for the print ############################################################

    if flag_remove==True:
        print('TOT simulations : '+str(len(df_sim)))            
        print('removed: '+do_not_select_meteor[i])
    else:
        print('TOT simulations : '+str(len(df_sim)))
        print('removed nothing')
    
    for around_meteor in df_sel['solution_id_dist'].unique():
        # select the data with distance less than dist_select and check if there are more than n_select
        df_curr_sel_curv = df_sel[df_sel['solution_id_dist']==around_meteor]
        # change of curvature print
        print()
        print(around_meteor)
        print('- Curvature change in the first '+str(len(df_curr_sel_curv['distance_meteor']))+' at a distance of: '+str(df_curr_sel_curv['distance_meteor'].iloc[-1]))

    # concatenate all the simulated shower in a single dataframe
    df_sim_shower = pd.concat(df_sim_shower)

    # concatenate all the EMCCD observed showers in a single dataframe
    df_obs_shower = pd.concat(df_obs_shower)

    # concatenate all the selected simulated showers in a single dataframe
    df_sel_shower = pd.concat(df_sel_shower)

    # # correlation matrix observed and fit parameters
    # corr = df_sel_shower.drop(['weight','distance_meteor'], axis=1).corr() # need solar longitude
    # sns.heatmap(corr,
    #             xticklabels=corr.columns.values,
    #             yticklabels=corr.columns.values,
    #             cmap="coolwarm")
    # plt.title('Correlation Matrix')
    # # shift the plot to the right
    # plt.show()


    for current_shower in Shower:
        curr_sim=df_sim_shower[df_sim_shower['shower_code']=='sim_'+current_shower]
        curr_obs=df_obs_shower[df_obs_shower['shower_code']==current_shower]
        curr_obs['shower_code']=current_shower+'_obs'
        curr_sel=df_sel_shower[df_sel_shower['shower_code']==current_shower+'_sel']
        curr_sel_save=df_sel_save[df_sel_save['shower_code']==current_shower+'_sel']
        curr_sel_save_dist=df_sel_save_dist[df_sel_save_dist['shower_code']==current_shower+'_sel']

        if curvature_selection_diff==False:
            if n_sample_noise>1:

                if len(dist_select)<n_sample_noise:
                    dist_select=np.ones(n_sample_noise)*10000000000000

                # Extract unique locations
                meteors_IDs = curr_sel_save["solution_id_dist"].unique()

                # # split the pd for the different solution_id_dist
                # curr_sel_split = [curr_sel[curr_sel["solution_id_dist"] == around_meteor] for around_meteor in meteors_IDs]
                sel_split_curr=[]
                # for each meteors_IDs consider the dist_select to cut the data
                for i, around_meteor in enumerate(meteors_IDs):

                    curr_sel_for_meteor = curr_sel[curr_sel["solution_id_dist"] == around_meteor]

                    # for each meteors_IDs consider the dist_select to cut the data
                    if np.min(curr_sel_for_meteor['distance_meteor'])>dist_select[i]:
                        forprint=curr_sel_for_meteor
                        sel_split_curr.append(curr_sel_for_meteor)
                        print(around_meteor)
                        print(str(i+1)+') No selected event below the given minimum distance :'+str(len(forprint)))
                        print('SEL = MAX dist: '+str(np.round(np.max(forprint['distance_meteor']),2)) +' min dist:'+str(np.round(np.min(forprint['distance_meteor']),2)))
                    else:
                        forprint=curr_sel_for_meteor[curr_sel_for_meteor['distance_meteor']<dist_select[i]]
                        # delete the data with the same around_meteor ["solution_id_dist"] that have distance_meteor bigger than dist_select[i]
                        sel_split_curr.append(curr_sel_for_meteor[curr_sel_for_meteor['distance_meteor']<dist_select[i]])
                        print(around_meteor)
                        # print the number of selected events
                        print(str(i+1)+') selected events below the value: '+str(len(forprint)))
                        print('SEL = MAX dist: '+str(np.round(np.max(forprint['distance_meteor']),2)) +' min dist:'+str(np.round(np.min(forprint['distance_meteor']),2)))
                        
                curr_sel=pd.concat(sel_split_curr)
                print('selected events below the distances give : '+str(len(curr_sel)))
            
        curr_df=pd.concat([curr_sim.drop(['rho','sigma','erosion_height_start','erosion_coeff','erosion_mass_index','erosion_mass_min','erosion_mass_max','erosion_energy_per_unit_cross_section',  'erosion_energy_per_unit_mass', 'erosion_range'], axis=1),curr_sel.drop(['distance_meteor','solution_id_dist','rho','sigma','erosion_height_start','erosion_coeff','erosion_mass_index','erosion_mass_min','erosion_mass_max','erosion_energy_per_unit_cross_section',  'erosion_energy_per_unit_mass','distance','erosion_range'], axis=1)], axis=0, ignore_index=True)
        curr_df=pd.concat([curr_df,curr_obs.drop(['distance'], axis=1)], axis=0, ignore_index=True)
        curr_df=curr_df.dropna()
        if Sim_data_distribution==True:
            if len(curr_sim)>10000:
                # pick randomly 10000 events
                curr_sim=curr_sim.sample(n=10000)
                curr_sim['weight']=1/len(curr_sim)
            curr_df_sim_sel=pd.concat([curr_sim,curr_sel.drop(['distance'], axis=1)], axis=0, ignore_index=True)
            
            curr_sel['erosion_coeff']=curr_sel['erosion_coeff']*1000000
            curr_sel['sigma']=curr_sel['sigma']*1000000
            curr_sel['erosion_energy_per_unit_cross_section']=curr_sel['erosion_energy_per_unit_cross_section']/1000000
            curr_sel['erosion_energy_per_unit_mass']=curr_sel['erosion_energy_per_unit_mass']/1000000

        elif Sim_data_distribution==False:
            curr_df_sim_sel=curr_sel
        

        if plot_var==True:
            fig, axs = plt.subplots(4, 3)
            # flatten the axs
            axs = axs.flatten()
            # fig.suptitle('Data between the 5 and 95 percentile of the variable values')
            # with color based on the shower but skip the first 2 columns (shower_code, shower_id)
            ii=0

            to_plot_unit=['init vel [km/s]','avg vel [km/s]','duration [s]','begin height [km]','peak height [km]','end height [km]','begin abs mag [-]','peak abs mag [-]','end abs mag [-]','F parameter [-]','zenith angle [deg]','deceleration [km/s^2]','trail lenght [km]','kurtosis','skew']

            to_plot=['vel_init_norot','vel_avg_norot','duration','begin_height','peak_mag_height','end_height','beg_abs_mag','peak_abs_mag','end_abs_mag','F','zenith_angle','decel_parab_t0','trail_len','kurtosis','skew']

            # deleter form curr_df the mass
            #curr_df=curr_df.drop(['mass'], axis=1)
            for ii in range(len(axs)):
                plotvar=to_plot[ii]
                # plot x within the 5 and 95 percentile of curr_df[plotvar] 
                x_plot=curr_df[plotvar]

                if plotvar in ['decel_parab_t0','decel_t0']:
                    sns.histplot(curr_df, x=x_plot[x_plot>-500], weights=curr_df['weight'][x_plot>-500],hue='shower_code', ax=axs[ii], kde=True, palette='bright', bins=20)
                    axs[ii].set_xticks([np.round(np.min(x_plot[x_plot>-500]),2),np.round(np.max(x_plot[x_plot>-500]),2)])
                
                else:
                    # cut the x axis after the 95 percentile and 5 percentile for the one that have a shower_code Shower+'_sim'
                    # x_plot= x_plot[(x_plot > np.percentile(curr_df[plotvar], 5)) & (x_plot < np.percentile(curr_df[plotvar], 95))]
                    sns.histplot(curr_df, x=x_plot, weights=curr_df['weight'],hue='shower_code', ax=axs[ii], kde=True, palette='bright', bins=20)
                    axs[ii].set_xticks([np.round(np.min(x_plot),2),np.round(np.max(x_plot),2)])

                # if beg_abs_mag','peak_abs_mag','end_abs_mag inver the x axis
                if plotvar in ['beg_abs_mag','peak_abs_mag','end_abs_mag']:
                    axs[ii].invert_xaxis()

                # Set the x-axis formatter to ScalarFormatter
                axs[ii].xaxis.set_major_formatter(ScalarFormatter())
                axs[ii].ticklabel_format(useOffset=False, style='plain', axis='x')
                # Set the number of x-axis ticks to 3
                # axs[ii].xaxis.set_major_locator(MaxNLocator(nbins=3))

                axs[ii].set_ylabel('probability')
                axs[ii].set_xlabel(to_plot_unit[ii])
                axs[ii].get_legend().remove()
                # check if there are more than 3 ticks and if yes only use the first and the last

                # put y axis in log scale
                axs[ii].set_yscale('log')
                axs[ii].set_ylim(0.01,1)

                

                    
            # more space between the subplots
            plt.tight_layout()
            # # full screen
            # figManager = plt.get_current_fig_manager()
            # figManager.window.showMaximized()

            print(output_dir+os.sep+'HistogramsVar_'+n_PC_in_PCA+'_'+str(len(curr_sel))+'ev_dist'+str(np.round(np.min(curr_sel['distance_meteor']),2))+'-'+str(np.round(np.max(curr_sel['distance_meteor']),2))+'.png')
            # save the figure
            fig.savefig(output_dir+os.sep+'Histograms'+str(len(axs))+'Var_'+n_PC_in_PCA+'_'+str(len(curr_sel))+'ev_MAXdist-'+str(np.round(np.max(curr_sel['distance_meteor']),2))+'.png', dpi=300)
            plt.close()

    ######### DISTANCE PLOT ##################################################
        if plot_dist==True:
            if n_sample_noise>1 and Sim_data_distribution==False:

                # Extract unique locations
                meteors_IDs = curr_sel_save["solution_id_dist"].unique()
                
                # save the distance_meteor from df_sel_save
                distance_meteor_sel_save=curr_sel_save['distance_meteor']
                # save the distance_meteor from df_sel_save
                distance_meteor_sel=curr_sel['distance_meteor']

                # Plotting
                fig, axes = plt.subplots(nrows=3, ncols=3)
                axes = axes.flatten()  # Flatten the array for easier iteration

                # use the default matpotlib default color cycle for the plots
                # print(plt.rcParams['axes.prop_cycle'].by_key()['color'])
                colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

                # if above meteors_IDs is bigger than 9 cut it to 9
                if len(meteors_IDs)>9:
                    meteors_IDs=meteors_IDs[:9]

                for i, around_meteor in enumerate(meteors_IDs):
                        
                    # Filter data for the current location
                    data_for_meteor = curr_sel_save_dist[curr_sel_save_dist["solution_id_dist"] == around_meteor]
                    data_for_meteor_sel = curr_sel[curr_sel["solution_id_dist"] == around_meteor]
                    # use the default matpotlib default color cycle
                    sns.histplot(data_for_meteor, x=data_for_meteor["distance_meteor"], kde=True, cumulative=True, bins=len(data_for_meteor["distance_meteor"]), color=colors[i], ax=axes[i])
                    
                    data_range = [data_for_meteor["distance_meteor"].min(), data_for_meteor["distance_meteor"].max()]

                    # sns.histplot(data_for_meteor, x=data_for_meteor["distance_meteor"], kde=True, cumulative=True, bins=len(data_for_meteor["distance_meteor"]))
                    # sns.kdeplot(data_for_meteor, x=data_for_meteor["distance_meteor"], cumulative=True, bw_adjust=sensib, clip=data_range, color='k',ax=axes[i])
                    
                    # axes[i].set_title(around_meteor[-12:-8])
                    axes[i].set_xlabel('Dist. PCA space')  # Remove x label for clarity
                    axes[i].set_ylabel('No.events')
                    # axes[i].tick_params(labelrotation=45)  # Rotate labels for better readability
                    # check if distance_meteor_sel have any value
                    if len(data_for_meteor_sel)>0:
                        axes[i].axvline(x=np.max(data_for_meteor_sel["distance_meteor"]), color=colors[i], linestyle='--')
                    else:
                        axes[i].axvline(x=distance_meteor_sel, color=colors[i], linestyle='--')
                    # plot a dasced line with the max distance_meteor_sel
                    #axes[i].axvline(x=np.max(distance_meteor_sel), color='k', linestyle='--')
                    # pu a y lim .ylim(0,100) 
                    axes[i].set_ylim(0,100)
                    # axes[i].set_ylim(0,0.01)
                    # if len(distance_meteor_sel)<1000:
                    #     axes[i].set_ylim(0,100)

                # Hide unused subplots if there are any
                for ax in axes[len(meteors_IDs):]:
                    ax.set_visible(False)

                plt.tight_layout()
                # plt.show()
                # save the figure maximized and with the right name
                plt.savefig(output_dir+os.sep+'DistributionDist'+n_PC_in_PCA+'_'+str(len(df_sel))+'ev_MAXdist'+str(np.round(np.max(distance_meteor_sel),2))+'.png', dpi=300)

                # close the figure
                plt.close()
            else:

                # Extract unique locations
                meteors_IDs = curr_sel_save["solution_id_dist"].unique()

                curr_sel_save_dist=curr_sel_save_dist[curr_sel_save_dist["solution_id_dist"] == meteors_IDs[0]]

                # save the distance_meteor from df_sel_save
                distance_meteor_sel_save=curr_sel_save_dist['distance_meteor']
                # save the distance_meteor from df_sel_save
                distance_meteor_sel=curr_sel['distance_meteor']
                # delete the index
                distance_meteor_sel_save=distance_meteor_sel_save.reset_index(drop=True)

                ###################################################
                # # check if distance_meteor_sel_save index is bigger than the index distance_meteor_sel+50
                # sns.histplot(distance_meteor_sel_save, kde=True, cumulative=True, bins=len(distance_meteor_sel_save)) # , stat='density' to have probability
                # # plt.ylim(0,len(distance_meteor_sel_save))
                # if len(distance_meteor_sel)<100:
                #     plt.ylim(0,100) 
                # # axis label
                # plt.xlabel('Distance in PCA space')
                # plt.ylabel('Number of events')

                # colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

                # # plot a dasced line with the max distance_meteor_sel
                # plt.axvline(x=np.max(distance_meteor_sel), color=colors[0], linestyle='--')

                # # make the y axis logarithmic
                # # plt.xscale('log')
                
                # # show
                # # plt.show()
                ####################################################

                #make subtraction of the next element and the previous element of distance_meteor_sel_save
                diff_distance_meteor = np.diff(distance_meteor_sel_save)
                # histogram plot of the difference with the count on the x axis and diff_distance_meteor on the y axis 
                indices = np.arange(len(diff_distance_meteor))
                # create the cumulative sum of the diff_distance_meteor
                cumsum_diff_distance_meteor = np.cumsum(diff_distance_meteor)
                # normalize the diff_distance_meteor xnormalized = (x - xminimum) / range of x
                diff_distance_meteor_normalized = (diff_distance_meteor - np.min(diff_distance_meteor)) / (np.max(diff_distance_meteor) - np.min(diff_distance_meteor))

                # find the index equal to 1 in diff_distance_meteor_normalized
                # index1 = np.where(diff_distance_meteor_normalized == 1)[0]
                # check when the diff_distance_meteor is two nxt to eac other are smaller than 0.1 starting from the first element
            
                # dimension of the plot 15,5
                plt.figure(figsize=(15,5))

                plt.subplot(1,2,2)
                sns.histplot(distance_meteor_sel_save, kde=True, cumulative=True, bins=len(distance_meteor_sel_save)) # , stat='density' to have probability
                colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
                plt.xlabel('Distance in PCA space')
                plt.ylabel('Number of events')
                plt.title('Cumulative distance in PCA space')
                plt.axvline(x=np.max(distance_meteor_sel), color=colors[0], linestyle='--')
                if len(distance_meteor_sel)<100:
                    plt.ylim(0,100) 
                

                plt.subplot(1,2,1)
                # sns.histplot(diff_distance_meteor_normalized, kde=True, bins=len(distance_meteor_sel_save))
                #make the bar plot 0.5 transparency
                
                plt.bar(indices, diff_distance_meteor_normalized,color=colors[0], alpha=0.5, edgecolor='black')
                plt.xlabel('Number of events')
                plt.ylabel('Normalized difference')
                plt.title('Distance difference Normalized')
                # put a horizontal line at len(curr_sel['distance_meteor'])
                plt.axvline(x=len(distance_meteor_sel)-1, color=colors[0], linestyle='--') 
                if len(distance_meteor_sel)<100:
                    plt.xlim(-1,100) 

                # find the mean of the first 100 elements of diff_distance_meteor_normalized and put a horizontal line
                # plt.axhline(y=np.std(moving_average_smoothing(diff_distance_meteor_normalized, window_size=3)), color=colors[0], linestyle='--')


                # plt.subplot(1,3,3)
                # # sns.histplot(diff_distance_meteor_normalized, kde=True, cumulative=True, bins=len(distance_meteor_sel_save))
                # plt.bar(indices, cumsum_diff_distance_meteor,color=colors[0], alpha=0.5, edgecolor='black')
                # # sns.histplot(data_for_meteor, x=data_for_meteor["distance_meteor"][:100], kde=True, cumulative=True, bins=len(data_for_meteor["distance_meteor"]))
                # plt.ylabel('Cumulative sum')
                # plt.xlabel('Number of events')
                # plt.title('Cumulative sum diff Normalized') 
                # # put a horizontal line at len(curr_sel['distance_meteor'])
                # plt.axvline(x=len(distance_meteor_sel), color=colors[0], linestyle='--')  
                # if len(distance_meteor_sel)<100:
                #     plt.xlim(-1,101) 
                # give more space
                plt.tight_layout()  
                # plt.show()

                # save the figure maximized and with the right name
                plt.savefig(output_dir+os.sep+'DistributionDist'+n_PC_in_PCA+'_'+str(len(df_sel))+'ev_MAXdist'+str(np.round(np.max(distance_meteor_sel),2))+'.png', dpi=300)

                # close the figure
                plt.close()

        ############################################################################

        # multiply the erosion coeff by 1000000 to have it in km/s
        curr_df_sim_sel['erosion_coeff']=curr_df_sim_sel['erosion_coeff']*1000000
        curr_df_sim_sel['sigma']=curr_df_sim_sel['sigma']*1000000
        df_sel_save['erosion_coeff']=df_sel_save['erosion_coeff']*1000000
        df_sel_save['sigma']=df_sel_save['sigma']*1000000
        curr_df_sim_sel['erosion_energy_per_unit_cross_section']=curr_df_sim_sel['erosion_energy_per_unit_cross_section']/1000000
        curr_df_sim_sel['erosion_energy_per_unit_mass']=curr_df_sim_sel['erosion_energy_per_unit_mass']/1000000
        df_sel_save['erosion_energy_per_unit_cross_section']=df_sel_save['erosion_energy_per_unit_cross_section']/1000000
        df_sel_save['erosion_energy_per_unit_mass']=df_sel_save['erosion_energy_per_unit_mass']/1000000
        # # pick the one with shower_code==current_shower+'_sel'
        # Acurr_df_sel=curr_df_sim_sel[curr_df_sim_sel['shower_code']==current_shower+'_sel']
        # Acurr_df_sim=curr_df_sim_sel[curr_df_sim_sel['shower_code']=='sim_'+current_shower]

        var_kde=['mass','rho','sigma','erosion_height_start','erosion_coeff','erosion_mass_index','erosion_mass_min','erosion_mass_max']

        # create the dataframe with the selected variable
        curr_sel_data = curr_sel[var_kde].values

        # check if leng of curr_sel_data is bigger than 1
        if len(curr_sel_data)>8:
            kde = gaussian_kde(dataset=curr_sel_data.T)  # Note the transpose to match the expected input shape

            # Negative of the KDE function for optimization
            def neg_density(x):
                return -kde(x)

            # Bounds for optimization within all the sim space
            # data_sim = df_sim[var_kde].values
            bounds = [(np.min(curr_sel_data[:,i]), np.max(curr_sel_data[:,i])) for i in range(curr_sel_data.shape[1])]

            # Initial guesses: curr_sel_data mean, curr_sel_data median, and KMeans centroids
            mean_guess = np.mean(curr_sel_data, axis=0)
            median_guess = np.median(curr_sel_data, axis=0)

            # KMeans centroids as additional guesses
            kmeans = KMeans(n_clusters=5, n_init='auto').fit(curr_sel_data)  # Adjust n_clusters based on your understanding of the curr_sel_data
            centroids = kmeans.cluster_centers_

            # Combine all initial guesses
            initial_guesses = [mean_guess, median_guess] + centroids.tolist()

            # Perform optimization from each initial guess
            results = [minimize(neg_density, x0, method='L-BFGS-B', bounds=bounds) for x0 in initial_guesses]

            # Filter out unsuccessful optimizations and find the best result
            successful_results = [res for res in results if res.success]
            if successful_results:
                best_result = min(successful_results, key=lambda x: x.fun)
                densest_point = best_result.x
                print("Densest point using KMeans centroid:", densest_point)
            else:
                # raise ValueError('Optimization was unsuccessful. Consider revising the strategy.')
                print('Optimization was unsuccessful. Consider revising the strategy.')
                # revise the optimization strategy
                print('Primary optimization strategies were unsuccessful. Trying fallback strategy (Grid Search).')
                # Fallback strategy: Grid Search
                grid_size = 5  # Define the grid size for the search
                grid_points = [np.linspace(bound[0], bound[1], grid_size) for bound in bounds]
                grid_combinations = list(itertools.product(*grid_points))

                best_grid_point = None
                best_grid_density = -np.inf

                for point in grid_combinations:
                    density = kde(point)
                    if density > best_grid_density:
                        best_grid_density = density
                        best_grid_point = point

                if best_grid_point is not None:
                    print("Densest point found using Grid Search:", best_grid_point)
                    return best_grid_point
                else:
                    raise ValueError('Grid Search was unsuccessful. None of the strategy worked.')

        else:
            print('Not enough data to perform the KDE need more than 8 meteors')
            # raise ValueError('The data is ill-conditioned. Consider a bigger number of elements.')




# if pickle change the extension and the code ##################################################################################################
        
        # check if the file is a pickle
        if true_file.endswith('.pickle'):
            # Load the trajectory file
            traj = loadPickle(true_path, true_file)

            sim_fit_json_nominal = os.path.join(true_path, true_file)
            # look for the 20230811_082648_sim_fit.json in the same folder
            sim_fit_json_nominal = sim_fit_json_nominal.replace('.pickle', '_sim_fit.json')
            # check if the file exist
            if not os.path.exists(sim_fit_json_nominal):
                # open any file with the json extension
                sim_fit_json_nominal = os.path.join(true_path, [f for f in os.listdir(true_path) if f.endswith('.json')][0])
                # print the file name
                print(sim_fit_json_nominal)

        else:
            # Load the nominal simulation
            sim_fit_json_nominal = os.path.join(true_path, true_file)
            traj = None

        # Load the nominal simulation parameters
        const_nominal, _ = loadConstants(sim_fit_json_nominal)
        const_nominal.dens_co = np.array(const_nominal.dens_co)

        dens_co=np.array(const_nominal.dens_co)

        # print(const_nominal.__dict__)

        ### Calculate atmosphere density coeffs (down to the bottom observed height, limit to 15 km) ###

        # Determine the height range for fitting the density
        dens_fit_ht_beg = const_nominal.h_init
        # dens_fit_ht_end = const_nominal.h_final

        # Assign the density coefficients
        const_nominal.dens_co = dens_co

        # Turn on plotting of LCs of individual fragments 
        const_nominal.fragmentation_show_individual_lcs = True

        # # change the sigma of the fragmentation
        # const_nominal.sigma = 1.0

        # 'rho': 209.27575861617834, 'm_init': 1.3339843905562902e-05, 'v_init': 59836.848805126894, 'shape_factor': 1.21, 'sigma': 1.387556841276162e-08, 'zenith_angle': 0.6944268835985749, 'gamma': 1.0, 'rho_grain': 3000, 'lum_eff_type': 5, 'lum_eff': 0.7, 'mu': 3.8180000000000003e-26, 'erosion_on': True, 'erosion_bins_per_10mass': 10, 'erosion_height_start': 117311.48011974395, 'erosion_coeff': 6.356639734390828e-07, 'erosion_height_change': 0, 'erosion_coeff_change': 3.3e-07, 'erosion_rho_change': 3700, 'erosion_sigma_change': 2.3e-08, 'erosion_mass_index': 1.614450928834309, 'erosion_mass_min': 4.773894502090459e-11, 'erosion_mass_max': 7.485333377052805e-10, 'disruption_on': False, 'compressive_strength': 2000, 

    # create a copy of the const_nominal
        const_nominal_1D_KDE = copy.deepcopy(const_nominal)
        const_nominal_allD_KDE = copy.deepcopy(const_nominal)

        var_cost=['m_init','rho','sigma','erosion_height_start','erosion_coeff','erosion_mass_index','erosion_mass_min','erosion_mass_max']
        # print for each variable the kde
        percent_diff_1D=[]
        percent_diff_allD=[]
        for i in range(len(var_kde)):

            x=curr_sel[var_kde[i]]

            # Compute KDE
            kde = gaussian_kde(x)
            
            # Define the range for which you want to compute KDE values, with more points for higher accuracy
            kde_x = np.linspace(x.min(), x.max(), 1000)
            kde_values = kde(kde_x)
            
            # Find the mode (x-value where the KDE curve is at its maximum)
            mode_index = np.argmax(kde_values)
            mode = kde_x[mode_index]
            
            real_val=df_sel_save[df_sel_save['solution_id']==only_select_meteors_from][var_kde[i]]
            print(real_val)
            # put it from Series.__format__ to double format
            real_val=real_val.values[0]

            print()
            #     var_kde=['mass','rho','sigma','erosion_height_start','erosion_coeff','erosion_mass_index','erosion_mass_min','erosion_mass_max']
            # Print the mode
            print(f"Real value {var_kde[i]}: {'{:.4g}'.format(real_val)}")
            print(f"1D Mode of KDE for {var_kde[i]}: {'{:.4g}'.format(mode)} percent diff: {'{:.4g}'.format(abs((real_val-mode)/(real_val+mode))/2*100)}%")
            percent_diff_1D.append(abs((real_val-mode)/(real_val+mode))/2*100)
            if len(curr_sel_data)>8:
                print(f"Mult.dim. KDE densest {var_kde[i]}:  {'{:.4g}'.format(densest_point[i])} percent diff: {'{:.4g}'.format(abs((real_val-densest_point[i])/(real_val+densest_point[i]))/2*100)}%")
                percent_diff_allD.append(abs((real_val-densest_point[i])/(real_val+densest_point[i]))/2*100)
            # print the value of const_nominal
            # print(f"const_nominal {var_cost[i]}:  {'{:.4g}'.format(const_nominal.__dict__[var_cost[i]])}")

            if var_cost[i] == 'sigma' or var_cost[i] == 'erosion_coeff':
                # put it back as it was
                const_nominal_1D_KDE.__dict__[var_cost[i]]=mode/1000000
                if len(curr_sel_data)>8:
                    const_nominal_allD_KDE.__dict__[var_cost[i]]=densest_point[i]/1000000
            elif var_cost[i] == 'erosion_height_start':
                # put it back as it was
                const_nominal_1D_KDE.__dict__[var_cost[i]]=mode*1000
                if len(curr_sel_data)>8:
                    const_nominal_allD_KDE.__dict__[var_cost[i]]=densest_point[i]*1000
            else:
                # add each to const_nominal_1D_KDE and const_nominal_allD_KDE
                const_nominal_1D_KDE.__dict__[var_cost[i]]=mode
                if len(curr_sel_data)>8:
                    const_nominal_allD_KDE.__dict__[var_cost[i]]=densest_point[i]

        # # Close the Logger to ensure everything is written to the file STOP COPY in TXT file
        # sys.stdout.close()

        # # Reset sys.stdout to its original value if needed
        # sys.stdout = sys.__stdout__

        # Run the simulation
        frag_main, results_list, wake_results = runSimulation(const_nominal, \
            compute_wake=False)

        sr_nominal = SimulationResults(const_nominal, frag_main, results_list, wake_results)

        # Run the simulation
        frag_main, results_list, wake_results = runSimulation(const_nominal_1D_KDE, \
            compute_wake=False)

        sr_nominal_1D_KDE = SimulationResults(const_nominal_1D_KDE, frag_main, results_list, wake_results)

        if len(curr_sel_data)>8:
            # Run the simulation
            frag_main, results_list, wake_results = runSimulation(const_nominal_allD_KDE, \
                compute_wake=False)

            sr_nominal_allD_KDE = SimulationResults(const_nominal_allD_KDE, frag_main, results_list, wake_results)

        # const_nominal = sr_nominal.const

        # open the json file with the name namefile_sel
        f = open(sim_fit_json_nominal,"r")
        data = json.loads(f.read())

        if traj == None:
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
            obs_length=[x/1000 for x in obs_length]

            # vel_sim=[v0]
            # # append from vel_sampled the rest by the difference of the first element of obs_length divided by the first element of obs_time
            # rest_vel_sampled=[(obs_length[vel_ii]-obs_length[vel_ii-1])/(obs_time[vel_ii]-obs_time[vel_ii-1]) for vel_ii in range(1,len(obs_length))]
            # # append the rest_vel_sampled to vel_sampled
            # vel_sim.extend(rest_vel_sampled)


            # create a list of the same length of obs_time with the value of the first element of vel_sim
            obs_vel=vel_sim

            for vel_ii in range(1,len(obs_time)):
                if obs_time[vel_ii]-obs_time[vel_ii-1]<0.03125:
                # if obs_time[vel_ii] % 0.03125 < 0.000000001:
                    if vel_ii+1<len(obs_length):
                        obs_vel[vel_ii+1]=(obs_length[vel_ii+1]-obs_length[vel_ii-1])/(obs_time[vel_ii+1]-obs_time[vel_ii-1])
                else:
                    obs_vel[vel_ii]=(obs_length[vel_ii]-obs_length[vel_ii-1])/(obs_time[vel_ii]-obs_time[vel_ii-1])

            vel_sim=obs_vel
            
            data_index_2cam = pd.DataFrame(list(zip(obs_time, ht_obs, obs_vel, abs_mag_sim)), columns =['time_sampled', 'ht_sampled', 'vel_sampled', 'mag_sampled'])

            # find in the index of camera 1 and camera 2 base if time_sampled % 0.03125 < 0.000000001 ==cam1 and the rest cam2
            time_cam1= [i for i in obs_time if i % 0.03125 < 0.000000001]
            time_cam2= [i for i in obs_time if i % 0.03125 > 0.000000001]


        else:

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

            erosion_height_start = data['erosion_height_start']/1000

            data_index_2cam = pd.DataFrame(list(zip(obs_time, ht_obs, obs_vel, abs_mag_sim)), columns =['time_sampled', 'ht_sampled', 'vel_sampled', 'mag_sampled'])



        # find the index of the camera 1 and camera 2 in the dataframe
        index_cam1_df= data_index_2cam[data_index_2cam['time_sampled'].isin(time_cam1)].index
        index_cam2_df= data_index_2cam[data_index_2cam['time_sampled'].isin(time_cam2)].index



        ############ plot the simulation
        # multiply ht_sim by 1000 to have it in m
        ht_sim_meters=[x*1000 for x in ht_sim]

        # find for the index of sr_nominal.leading_frag_height_arr with the same values as sr_nominal_1D_KDE.leading_frag_height_arr
        closest_indices_1D = find_closest_index(sr_nominal_1D_KDE.leading_frag_height_arr, ht_sim_meters )
        # make the subtraction of the closest_indices between sr_nominal.abs_magnitude and sr_nominal_1D_KDE.abs_magnitude
        diff_mag_1D=[(sr_nominal.abs_magnitude[jj_index_cut]-sr_nominal_1D_KDE.abs_magnitude[jj_index_cut]) for jj_index_cut in closest_indices_1D]
        diff_vel_1D=[(sr_nominal.leading_frag_vel_arr[jj_index_cut]-sr_nominal_1D_KDE.leading_frag_vel_arr[jj_index_cut])/1000 for jj_index_cut in closest_indices_1D]
        # do the same for the sr_nominal_allD_KDE
        if len(curr_sel_data)>8:
            closest_indices_allD = find_closest_index(sr_nominal_allD_KDE.leading_frag_height_arr, ht_sim_meters )
            diff_mag_allD=[(sr_nominal.abs_magnitude[jj_index_cut]-sr_nominal_allD_KDE.abs_magnitude[jj_index_cut]) for jj_index_cut in closest_indices_allD]
            diff_vel_allD=[(sr_nominal.leading_frag_vel_arr[jj_index_cut]-sr_nominal_allD_KDE.leading_frag_vel_arr[jj_index_cut])/1000 for jj_index_cut in closest_indices_allD]

        # Plot the simulation results
        fig, ax = plt.subplots(2, 2, figsize=(8, 10),gridspec_kw={'width_ratios': [ 3, 1]}, dpi=300) #  figsize=(10, 5), dpi=300 0.5, 3, 3, 0.5

        # flat the ax
        ax = ax.flatten()

        # plot a line plot in the first subplot the magnitude vs height dashed with x markers
        ax[0].plot(data_index_2cam['mag_sampled'][index_cam1_df], data_index_2cam['ht_sampled'][index_cam1_df], linestyle='dashed', marker='x', label='1')
        ax[0].plot(data_index_2cam['mag_sampled'][index_cam2_df], data_index_2cam['ht_sampled'][index_cam2_df], linestyle='dashed', marker='x', label='2')
        

        # add the erosion_height_start as a horizontal line in the first subplot grey dashed
        ax[0].axhline(y=erosion_height_start, color='grey', linestyle='dashed')
        # add the name on the orizontal height line
        ax[0].text(max(abs_mag_sim)+1, erosion_height_start, 'Erosion height', color='grey')

        # plot a scatter plot in the second subplot the velocity vs height
        # ax[2].scatter(vel_sim, ht_sim, marker='.', label='1')
        # use the . maker and none linestyle
        ax[2].plot(data_index_2cam['vel_sampled'][index_cam1_df], data_index_2cam['ht_sampled'][index_cam1_df], marker='.', linestyle='none', label='1')
        ax[2].plot(data_index_2cam['vel_sampled'][index_cam2_df], data_index_2cam['ht_sampled'][index_cam2_df], marker='.', linestyle='none', label='2')

        # set the xlim and ylim of the first subplot
        ax[0].set_xlim([min(abs_mag_sim)-1, max(abs_mag_sim)+1])
        # check if the max(ht_sim) is greater than the erosion_height_start and set the ylim of the first subplot

        # set the xlim and ylim of the second subplot
        ax[2].set_xlim([min(vel_sim)-1, max(vel_sim)+1])
    
        # Plot the height vs magnitude
        ax[0].plot(sr_nominal.abs_magnitude, sr_nominal.leading_frag_height_arr/1000, label="Simulated", \
            color='k')
        
        ax[0].plot(sr_nominal_1D_KDE.abs_magnitude, sr_nominal_1D_KDE.leading_frag_height_arr/1000, label="Mode", color='r')
        
        if len(curr_sel_data)>8:
            ax[0].plot(sr_nominal_allD_KDE.abs_magnitude, sr_nominal_allD_KDE.leading_frag_height_arr/1000, label="Min KDE", color='b')

        # velocity vs height

        # # height vs velocity
        # ax[2].plot(sr_nominal.brightest_vel_arr/1000, sr_nominal.brightest_height_arr/1000, label="Simulated - brightest", \
        #     color='k', alpha=0.75)  
        
        # # Plot the velocity of the main mass
        # ax[2].plot(sr_nominal.leading_frag_vel_arr/1000, sr_nominal.leading_frag_height_arr/1000, color='k', \
        #     linestyle='dashed', label="Simulated - leading")        
        ax[2].plot(sr_nominal.leading_frag_vel_arr/1000, sr_nominal.leading_frag_height_arr/1000, color='k', \
            label="Simulated")
        
        # ax[2].plot(sr_nominal_1D_KDE.brightest_vel_arr/1000, sr_nominal_1D_KDE.brightest_height_arr/1000, \
        #             label="Mode - brightest", alpha=0.75)

        # # keep the same color and use a dashed line
        # ax[2].plot(sr_nominal_1D_KDE.leading_frag_vel_arr/1000, sr_nominal_1D_KDE.leading_frag_height_arr/1000, \
        #     linestyle='dashed', label="Mode - leading", color=ax[2].lines[-1].get_color())
        ax[2].plot(sr_nominal_1D_KDE.leading_frag_vel_arr/1000, sr_nominal_1D_KDE.leading_frag_height_arr/1000, \
            label="Mode", color='r')
        

        ax[1].scatter(diff_mag_1D,sr_nominal.leading_frag_height_arr[closest_indices_1D]/1000, color=ax[2].lines[-1].get_color(), marker='.')
        ax[3].scatter(diff_vel_1D,sr_nominal.leading_frag_height_arr[closest_indices_1D]/1000, color=ax[2].lines[-1].get_color(), marker='.')

        # ax[2].plot(sr_nominal_allD_KDE.brightest_vel_arr/1000, sr_nominal_allD_KDE.brightest_height_arr/1000, \
        #             label="Min KDE - brightest")

        # # keep the same color and use a dashed line
        # ax[2].plot(sr_nominal_allD_KDE.leading_frag_vel_arr/1000, sr_nominal_allD_KDE.leading_frag_height_arr/1000, \
        #     linestyle='dashed', label="Min KDE - leading", color=ax[2].lines[-1].get_color())
        if len(curr_sel_data)>8:
            ax[2].plot(sr_nominal_allD_KDE.leading_frag_vel_arr/1000, sr_nominal_allD_KDE.leading_frag_height_arr/1000, \
                label="Min KDE", color='b')
        
            ax[1].scatter(diff_mag_allD,sr_nominal.leading_frag_height_arr[closest_indices_allD]/1000, color=ax[2].lines[-1].get_color(), marker='.')
            ax[3].scatter(diff_vel_allD,sr_nominal.leading_frag_height_arr[closest_indices_allD]/1000, color=ax[2].lines[-1].get_color(), marker='.')

        if max(ht_sim)>erosion_height_start:
            ax[1].set_ylim([min(ht_sim)-1, max(ht_sim)+1])
            ax[0].set_ylim([min(ht_sim)-1, max(ht_sim)+1])
            ax[2].set_ylim([min(ht_sim)-1, max(ht_sim)+1])
            ax[3].set_ylim([min(ht_sim)-1, max(ht_sim)+1])
        else:
            ax[1].set_ylim([min(ht_sim)-1, erosion_height_start+2])
            ax[0].set_ylim([min(ht_sim)-1, erosion_height_start+2])
            ax[2].set_ylim([min(ht_sim)-1, erosion_height_start+2])
            ax[3].set_ylim([min(ht_sim)-1, erosion_height_start+2])
        
        # set the xlabel and ylabel of the subplots

        # on ax[1] the sides of the plot put the error in the magnitude as a value with one axis
        ax[1].set_xlabel('abs.mag.err')
        # set the same y axis as the plot above
        # ax[1].set_ylim(ax[0].get_ylim())
        # place the y axis along the zero
        ax[1].spines['left'].set_position(('data', 0))
        # place the ticks along the zero
        ax[1].yaxis.set_ticks_position('left')
        # delete the numbers from the y axis
        ax[1].yaxis.set_tick_params(labelleft=False)
        # invert the y axis
        ax[1].invert_xaxis()
        # delte the border of the plot
        ax[1].spines['right'].set_color('none')
        ax[1].spines['top'].set_color('none')
        
        
        if len(curr_sel_data)>8:
            # append diff_vel_allD to diff_vel_1D
            diff_mag_1D.extend(diff_mag_allD)
        
        # delete any nan or inf from the list
        diff_mag_1D = [x for x in diff_mag_1D if str(x) != 'nan' and str(x) != 'inf']
        # put the ticks in the x axis to -1*max(abs(np.array(diff_mag_allD))), max(abs(np.array(diff_mag_allD)) with only 2 significant digits
        # ax[1].set_xticks([-1*max(abs(np.array(diff_mag_1D))), max(abs(np.array(diff_mag_1D)))])

        ax[1].axvspan(-mag_noise, mag_noise, color='lightgray', alpha=0.5)
        # Rotate tick labels
        ax[1].tick_params(axis='x', rotation=45)
        # rotate that by 45 degrees
        ax[1].set_xlim([-1*max(abs(np.array(diff_mag_1D)))-max(abs(np.array(diff_mag_1D)))/4, max(abs(np.array(diff_mag_1D)))+max(abs(np.array(diff_mag_1D)))/4])
        # add more ticks
        ax[1].xaxis.set_major_locator(plt.MaxNLocator(5))
        # add grid to the plot
        ax[1].grid(linestyle='dashed')
        
        # on ax[3] the sides of the plot put the error in the velocity as a value with one axis
        # ax[3].set_xlabel('vel.lead.err [km/s]')
        ax[3].set_xlabel('vel.err [km/s]')
        # set the same y axis as the plot above
        # ax[3].set_ylim(ax[2].get_ylim())
        # place the y axis along the zero
        ax[3].spines['right'].set_position(('data', 0))
        # place the ticks along the zero
        ax[3].yaxis.set_ticks_position('right')
        # delete the numbers from the y axis
        ax[3].yaxis.set_tick_params(labelright=False)
        # delte the border of the plot
        ax[3].spines['left'].set_color('none')
        ax[3].spines['top'].set_color('none')
        if len(curr_sel_data)>8:
            # append diff_vel_allD to diff_vel_1D
            diff_vel_1D.extend(diff_vel_allD)

        # delete any nan or inf from the list
        diff_vel_1D = [x for x in diff_vel_1D if str(x) != 'nan' and str(x) != 'inf']
        # x limit of the plot equal to max of the absolute magnitude
        # ax[3].set_xticks([-1*max(abs(np.array(diff_vel_1D))), max(abs(np.array(diff_vel_1D)))])

        ax[3].axvspan(-vel_noise, vel_noise, color='lightgray', alpha=0.5)
        # Rotate tick labels
        ax[3].tick_params(axis='x', rotation=45)
        ax[3].set_xlim([-1*max(abs(np.array(diff_vel_1D)))-max(abs(np.array(diff_vel_1D)))/4, max(abs(np.array(diff_vel_1D)))+max(abs(np.array(diff_vel_1D)))/4])
        # add more ticks
        ax[3].xaxis.set_major_locator(plt.MaxNLocator(5))
        # add grid to the plot
        ax[3].grid(linestyle='dashed')
        
        # put the grid in the subplots and make it dashed
        ax[0].grid(linestyle='dashed')
        ax[2].grid(linestyle='dashed')
        # add the legend
        ax[0].legend()
        ax[2].legend()

        # add the labels
        ax[0].set_ylabel('Height [km]')
        ax[0].set_xlabel('Absolute Magnitude')
        # invert the x axis
        ax[0].invert_xaxis()

        # put the ticks on the right
        # ax[2].yaxis.tick_right()
        ax[2].set_ylabel('Height [km]')
        ax[2].set_xlabel('Velocity [km/s]')
        # put the labels on the right
        # ax[2].yaxis.set_label_position("right")


        # make more space between the subplots
        plt.tight_layout()

        # make the plot visible
        # plt.show()
        print(output_dir+os.sep+'BestFitKDE'+n_PC_in_PCA+'_'+str(len(curr_sel))+'ev_dist'+str(np.round(np.min(curr_sel['distance_meteor']),2))+'-'+str(np.round(np.max(curr_sel['distance_meteor']),2))+'.png')
        # save the figure maximized and with the right name
        fig.savefig(output_dir+os.sep+'BestFitKDE'+n_PC_in_PCA+'_'+str(len(curr_sel))+'ev_dist'+str(np.round(np.min(curr_sel['distance_meteor']),2))+'-'+str(np.round(np.max(curr_sel['distance_meteor']),2))+'.png', dpi=300)
        
        # close the figure
        plt.close()
        
    ##########################################################################

        # open a new figure to plot the pairplot
        fig = plt.figure(figsize=(10, 10), dpi=300)

        # Define your label mappings
        label_mappings = {
            'mass': 'mass [kg]',
            'rho': 'rho [kg/m^3]',
            'sigma': 'sigma [s^2/km^2]',
            'erosion_height_start': 'erosion height start [km]',
            'erosion_coeff': 'erosion coeff [s^2/km^2]',
            'erosion_mass_index': 'erosion mass index [-]',
            'erosion_mass_min': 'log eros. mass min [kg]',
            'erosion_mass_max': 'log eros. mass max [kg]'
        }

        if Sim_data_distribution==True:
            to_plot8=['shower_code','mass','rho','sigma','erosion_height_start','erosion_coeff','erosion_mass_index','erosion_mass_min','erosion_mass_max']
            # crete the triangular plot to of the selected variables with seaborn with the simulated and the selected events and plot it in the figure
            fig = sns.pairplot(curr_df_sim_sel[to_plot8], hue='shower_code', palette='bright', diag_kind='kde', corner=True, plot_kws={'edgecolor': 'k'}) 
            fig._legend.remove()
        elif Sim_data_distribution==False:
            to_plot8=['solution_id_dist','mass','rho','sigma','erosion_height_start','erosion_coeff','erosion_mass_index','erosion_mass_min','erosion_mass_max']
            # crete the triangular plot to of the selected variables with seaborn with the simulated and the selected events
            fig = sns.pairplot(curr_df_sim_sel[to_plot8], hue='solution_id_dist', palette='bright', diag_kind='kde', corner=True, plot_kws={'edgecolor': 'k'})
            fig._legend.remove()
        
        label_plot=['mass [kg]','rho [kg/m^3]','sigma [s^2/km^2]','erosion height start [km]','erosion coeff [s^2/km^2]','erosion mass index [-]','log eros. mass min [kg]','log eros. mass max [kg]']
        # change the x and y labels of the plot
        # Update the labels
        for ax in fig.axes.flatten():
            if ax is not None:  # Check if the axis exists
                xlabel = ax.get_xlabel()
                ylabel = ax.get_ylabel()
                if ylabel=='erosion_mass_min' or ylabel=='erosion_mass_max': # ylabel == 'mass' or xlabel == 'sigma'
                    # set it to log scale
                    ax.set_yscale('log')
                if xlabel=='erosion_mass_min' or xlabel=='erosion_mass_max': # xlabel=='erosion_coeff'
                    # set it to log sca
                    ax.set_xscale('log')
                if xlabel in label_mappings:
                    ax.set_xlabel(label_mappings[xlabel])
                if ylabel in label_mappings:
                    ax.set_ylabel(label_mappings[ylabel])
        # plt.tight_layout()
        # plt.show()

        print(output_dir+os.sep+'PhysicPropPairPlot'+n_PC_in_PCA+'_'+str(len(curr_sel))+'ev_dist'+str(np.round(np.min(curr_sel['distance_meteor']),2))+'-'+str(np.round(np.max(curr_sel['distance_meteor']),2))+'.png')
        # save the figure maximized and with the right name
        fig.savefig(output_dir+os.sep+'PhysicPropPairPlot'+n_PC_in_PCA+'_'+str(len(curr_sel))+'ev_dist'+str(np.round(np.min(curr_sel['distance_meteor']),2))+'-'+str(np.round(np.max(curr_sel['distance_meteor']),2))+'.png', dpi=300)
        # close the figure
        plt.close()

##########################################################################

        # Define your label mappings
        label_mappings = {
            'mass': 'mass [kg]',
            'rho': 'rho [kg/m^3]',
            'sigma': 'sigma [s^2/km^2]',
            'erosion_height_start': 'erosion height start [km]',
            'erosion_coeff': 'erosion coeff [s^2/km^2]',
            'erosion_mass_index': 'erosion mass index [-]',
            'erosion_mass_min': 'log eros. mass min [kg]',
            'erosion_mass_max': 'log eros. mass max [kg]'
        }

        # Choose which columns to plot based on condition
        if Sim_data_distribution:
            to_plot8 = ['shower_code', 'mass', 'rho', 'sigma', 'erosion_height_start', 'erosion_coeff', 'erosion_mass_index', 'erosion_mass_min', 'erosion_mass_max']
            hue_column = 'shower_code'
        else:
            to_plot8 = ['solution_id_dist', 'mass', 'rho', 'sigma', 'erosion_height_start', 'erosion_coeff', 'erosion_mass_index', 'erosion_mass_min', 'erosion_mass_max']
            hue_column = 'solution_id_dist'

        # Create a PairGrid
        pairgrid = sns.PairGrid(curr_df_sim_sel[to_plot8], hue=hue_column, palette='bright')

        # Map the plots
        pairgrid.map_lower(sns.scatterplot, edgecolor='k', palette='bright')
        # for the upper triangle delete x and y axis
        # pairgrid.map_diag(sns.kdeplot)
        # pairgrid.map_diag(sns.histplot, kde=True, color='k', edgecolor='k')
        # pairgrid.add_legend()

        # Update the labels
        for ax in pairgrid.axes.flatten():
            if ax is not None:  # Check if the axis exists
                xlabel = ax.get_xlabel()
                ylabel = ax.get_ylabel()
                if ylabel in label_mappings:
                    ax.set_ylabel(label_mappings[ylabel])
                if xlabel in label_mappings:
                    ax.set_xlabel(label_mappings[xlabel])
                if ylabel in ['erosion_mass_min', 'erosion_mass_max']:#'sigma', 
                    ax.set_yscale('log')
                if xlabel in ['erosion_mass_min', 'erosion_mass_max']: #'sigma', 
                    ax.set_xscale('log')

        # # Calculate the correlation matrix
        # corr = curr_df_sim_sel[to_plot8[1:]].corr()

        if Sim_data_distribution==True:
            corr = curr_sel[to_plot8[1:]].corr()
        if Sim_data_distribution==False:
            corr = curr_df_sim_sel[to_plot8[1:]].corr()

        # Find the min and max correlation values
        vmin = corr.values.min()
        vmax = corr.values.max()
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap = sns.color_palette('coolwarm', as_cmap=True)

        # Fill the upper triangle plots with the correlation matrix values and color it with the coolwarm cmap
        for i, row in enumerate(to_plot8[1:]):
            for j, col in enumerate(to_plot8[1:]):
                if i < j:
                    ax = pairgrid.axes[i, j]  # Adjust index to fit the upper triangle
                    corr_value = corr.loc[row, col]
                    ax.text(0.5, 0.5, f'{corr_value:.2f}', horizontalalignment='center', verticalalignment='center', fontsize=12, color='black', transform=ax.transAxes)
                    ax.set_facecolor(cmap(norm(corr_value)))
                    # cmap = sns.color_palette('coolwarm', as_cmap=True)
                    # ax.set_facecolor(cmap(corr_value))

                    # Remove the axis labels
                    ax.xaxis.set_visible(False)
                    ax.yaxis.set_visible(False)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['left'].set_visible(False)
                    ax.spines['bottom'].set_visible(False)
                if i == j:
                    ax = pairgrid.axes[i, j]
                    ax.set_axis_off()

        # Adjust layout
        plt.tight_layout()

        fig_name = (output_dir+os.sep+'MixPhysicPropPairPlot'+n_PC_in_PCA+'_'+str(len(curr_sel))+'ev_dist'+str(np.round(np.min(curr_sel['distance_meteor']),2))+'-'+str(np.round(np.max(curr_sel['distance_meteor']),2))+'.png')
        plt.savefig(fig_name, dpi=300)

        # Close the figure
        plt.close()

    ##########################################################################

        to_plot8=['mass','rho','sigma','erosion_height_start','erosion_coeff','erosion_mass_index','erosion_mass_min','erosion_mass_max']
        # labe for the variables
        label_plot=['mass','rho','sigma','er.height','er.coeff','er.mass index','er.mass min','er.mass max']

        # # create a covarariance matrix plot of the selected variables
        # scaler = StandardScaler()
        # scaled_data = scaler.fit_transform(curr_df_sim_sel[to_plot8])  # Assuming 'df' and 'selected_columns' from your context
        # scaled_df = pd.DataFrame(scaled_data, columns=to_plot8)

        # # Now compute covariance on scaled data
        # cov_matrix = scaled_df.cov()

        fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
        # Compute the correlation coefficients
        # corr_matrix = curr_df_sim_sel[to_plot8].corr()

        # create covariance matrix plot of the selected variables
        # sns.heatmap(curr_df_sim_sel[to_plot8].cov(), annot=True, cmap='coolwarm', ax=ax)
        # create a heatmap of the selected variables base on the covariance matrix corr()
        if Sim_data_distribution==True:
            sns.heatmap(curr_sel[to_plot8].corr(), annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
        if Sim_data_distribution==False:
            sns.heatmap(curr_df_sim_sel[to_plot8].corr(), annot=True, fmt=".2f", cmap='coolwarm', ax=ax)

        # sns.heatmap(curr_df_sim_sel[to_plot8].cov(), annot=True, fmt=".2f", cmap='coolwarm', ax=ax, vmin=-1, vmax=1)
        # sns.heatmap(cov_matrix, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
        # and put the values of the correlation in the heatmap rounded to 2 decimals in the center of the square
        for t in ax.texts: t.set_text(np.round(float(t.get_text()), 2))
        # use the label_plot as the xticks and yticks
        ax.set_xticklabels(label_plot, rotation=45)
        ax.set_yticklabels(label_plot, rotation=0)

        # plt.show()
        # save the figure maximized and with the right name
        fig.savefig(output_dir+os.sep+'PhysicPropCovar'+n_PC_in_PCA+'_'+str(len(curr_sel))+'ev_dist'+str(np.round(np.min(curr_sel['distance_meteor']),2))+'-'+str(np.round(np.max(curr_sel['distance_meteor']),2))+'.png', dpi=300)
    
        # close the figure
        plt.close()

    ##########################################################################

        # with color based on the shower but skip the first 2 columns (shower_code, shower_id)
        to_plot=['mass','rho','sigma','erosion_height_start','erosion_coeff','erosion_mass_index','erosion_mass_min','erosion_mass_max','erosion_range','erosion_energy_per_unit_mass','erosion_energy_per_unit_cross_section','erosion_energy_per_unit_cross_section']
        to_plot_unit=['mass [kg]','rho [kg/m^3]','sigma [s^2/km^2]','erosion height start [km]','erosion coeff [s^2/km^2]','erosion mass index [-]','log eros. mass min [kg]','log eros. mass max [kg]','log eros. mass range [-]','erosion energy per unit mass [MJ/kg]','erosion energy per unit cross section [MJ/m^2]','erosion energy per unit cross section [MJ/m^2]']
        
        # # multiply the erosion coeff by 1000000 to have it in km/s
        # curr_df_sim_sel['erosion_coeff']=curr_df_sim_sel['erosion_coeff']*1000000
        # curr_df_sim_sel['sigma']=curr_df_sim_sel['sigma']*1000000
        # df_sel_save['erosion_coeff']=df_sel_save['erosion_coeff']*1000000
        # df_sel_save['sigma']=df_sel_save['sigma']*1000000
        # curr_df_sim_sel['erosion_energy_per_unit_cross_section']=curr_df_sim_sel['erosion_energy_per_unit_cross_section']/1000000
        # curr_df_sim_sel['erosion_energy_per_unit_mass']=curr_df_sim_sel['erosion_energy_per_unit_mass']/1000000
        # df_sel_save['erosion_energy_per_unit_cross_section']=df_sel_save['erosion_energy_per_unit_cross_section']/1000000
        # df_sel_save['erosion_energy_per_unit_mass']=df_sel_save['erosion_energy_per_unit_mass']/1000000
        # # pick the one with shower_code==current_shower+'_sel'
        # Acurr_df_sel=curr_df_sim_sel[curr_df_sim_sel['shower_code']==current_shower+'_sel']
        # Acurr_df_sim=curr_df_sim_sel[curr_df_sim_sel['shower_code']=='sim_'+current_shower]
        
        fig, axs = plt.subplots(3, 3)
        # from 2 numbers to one numbr for the subplot axs
        axs = axs.flatten()

        print('\\hline')
        # print('var & $real$ & $1D_{KDE}$ & $1D_{KDE}\\%_{dif}$ & $allD_{KDE}$ & $allD_{KDE}\\%_{dif}$\\\\')
        # print('var & real & mode & min$_{KDE}$ & -1\\sigma/+1\\sigma & -2\\sigma/+2\\sigma \\\\')
        print('Variables & Real & Mode & Min$_{KDE}$ & 95\\%CIlow & 95\\%CIup \\\\')

        ii_densest=0        
        for i in range(9):
            # put legendoutside north
            plotvar=to_plot[i]


            if Sim_data_distribution==True:
                if plotvar == 'erosion_mass_min' or plotvar == 'erosion_mass_max':
                    # take the log of the erosion_mass_min and erosion_mass_max
                    curr_df_sim_sel[plotvar]=np.log10(curr_df_sim_sel[plotvar])
                    df_sel_save[plotvar]=np.log10(df_sel_save[plotvar])
                    curr_sel[plotvar]=np.log10(curr_sel[plotvar])
                    if len(curr_sel_data)>8:
                        densest_point[ii_densest]=np.log10(densest_point[ii_densest])
                        # densest_point[ii_densest-1]=np.log10(densest_point[ii_densest-1])
                # sns.histplot(curr_df_sim_sel, x=curr_df_sim_sel[plotvar], weights=curr_df_sim_sel['weight'],hue='shower_code', ax=axs[i], kde=True, palette='bright', bins=20)
                sns.histplot(curr_df_sim_sel, x=curr_df_sim_sel[plotvar], weights=curr_df_sim_sel['weight'],hue='shower_code', ax=axs[i], palette='bright', bins=20)
                # # add the kde to the plot probability density function
                sns.histplot(curr_sel, x=curr_sel[plotvar], weights=curr_sel['weight'], bins=20, ax=axs[i], fill=False, edgecolor=False, color='r', kde=True, binrange=[np.min(curr_df_sim_sel[plotvar]),np.max(curr_df_sim_sel[plotvar])])
                kde_line = axs[i].lines[-1]

                # if the only_select_meteors_from is equal to any curr_df_sim_sel plot the observed event value as a vertical red line
                if only_select_meteors_from in df_sel_save['solution_id'].values:
                    axs[i].axvline(x=df_sel_save[df_sel_save['solution_id']==only_select_meteors_from][plotvar].values[0], color='k', linewidth=2)

                if plotvar == 'erosion_mass_min' or plotvar == 'erosion_mass_max':
                    # put it back as it was
                    curr_df_sim_sel[plotvar]=10**curr_df_sim_sel[plotvar]
                    df_sel_save[plotvar]=10**df_sel_save[plotvar]
                    curr_sel[plotvar]=10**curr_sel[plotvar]
                
            elif Sim_data_distribution==False:

                    if plotvar == 'erosion_mass_min' or plotvar == 'erosion_mass_max':
                        # sns.histplot(curr_df_sim_sel, x=curr_df_sim_sel[plotvar], weights=curr_df_sim_sel['weight'],hue='solution_id_dist', ax=axs[i], multiple="stack", kde=True, bins=20, binrange=[np.min(df_sel_save[plotvar]),np.max(df_sel_save[plotvar])])
                        sns.histplot(curr_df_sim_sel, x=np.log10(curr_df_sim_sel[plotvar]), weights=curr_df_sim_sel['weight'],hue='solution_id_dist', ax=axs[i], multiple="stack", bins=20, binrange=[np.log10(np.min(df_sel_save[plotvar])),np.log10(np.max(df_sel_save[plotvar]))])
                        # # add the kde to the plot as a probability density function
                        sns.histplot(curr_df_sim_sel, x=np.log10(curr_df_sim_sel[plotvar]), weights=curr_df_sim_sel['weight'], bins=20, ax=axs[i],  multiple="stack", fill=False, edgecolor=False, color='r', kde=True, binrange=[np.log10(np.min(df_sel_save[plotvar])),np.log10(np.max(df_sel_save[plotvar]))])
                        
                        kde_line = axs[i].lines[-1]
                        
                        # if the only_select_meteors_from is equal to any curr_df_sim_sel plot the observed event value as a vertical red line
                        if only_select_meteors_from in df_sel_save['solution_id'].values:
                            axs[i].axvline(x=np.log10(df_sel_save[df_sel_save['solution_id']==only_select_meteors_from][plotvar].values[0]), color='k', linewidth=2)
                        if len(curr_sel_data)>8:
                            densest_point[ii_densest]=np.log10(densest_point[ii_densest])
                            # densest_point[ii_densest-1]=np.log10(densest_point[ii_densest-1])
                    
                    else:
                        # sns.histplot(curr_df_sim_sel, x=curr_df_sim_sel[plotvar], weights=curr_df_sim_sel['weight'],hue='solution_id_dist', ax=axs[i], multiple="stack", kde=True, bins=20, binrange=[np.min(df_sel_save[plotvar]),np.max(df_sel_save[plotvar])])
                        sns.histplot(curr_df_sim_sel, x=curr_df_sim_sel[plotvar], weights=curr_df_sim_sel['weight'],hue='solution_id_dist', ax=axs[i], multiple="stack", bins=20, binrange=[np.min(df_sel_save[plotvar]),np.max(df_sel_save[plotvar])])
                        # # add the kde to the plot as a probability density function
                        sns.histplot(curr_df_sim_sel, x=curr_df_sim_sel[plotvar], weights=curr_df_sim_sel['weight'], bins=20, ax=axs[i],  multiple="stack", fill=False, edgecolor=False, color='r', kde=True, binrange=[np.min(df_sel_save[plotvar]),np.max(df_sel_save[plotvar])])
                        
                        kde_line = axs[i].lines[-1]

                        # if the only_select_meteors_from is equal to any curr_df_sim_sel plot the observed event value as a vertical red line
                        if only_select_meteors_from in df_sel_save['solution_id'].values:
                            axs[i].axvline(x=df_sel_save[df_sel_save['solution_id']==only_select_meteors_from][plotvar].values[0], color='k', linewidth=2)
                            # put the value of diff_percent_1d at th upper left of the line
                            

            # Get the x and y data from the KDE line
            x = kde_line.get_xdata()
            y = kde_line.get_ydata()

            # \hline
            # $h_{init}$ & $v_{init}$ & $\vartheta_z$ &size  & $\tau$ & Peak Mag\\
            # \hline
            # 130 km & 7 to 9 km/s & 88° to 85° &5 to 20 cm & 0.001 to 0.1 & 0 to -5 \\
            # \hline

            # Find the index of the maximum y value
            max_index = np.argmax(y)
            if i!=8:
                # Plot a dot at the maximum point
                axs[i].plot(x[max_index], y[max_index], 'ro')  # 'ro' for red dot

            if len(curr_sel_data)>8:        
                if len(densest_point)>ii_densest:                    
                
                    # print(densest_point[ii_densest])
                    # Find the index with the closest value to densest_point[ii_dense] to all y values
                    densest_index = find_closest_index(x, [densest_point[ii_densest]])

                    # add also the densest_point[i] as a blue dot
                    axs[i].plot(densest_point[ii_densest], y[densest_index[0]], 'bo')
                    # get te 97.72nd percentile and the 2.28th percentile of curr_sel[plotvar] and call them sigma_97 and sigma_2
                    sigma_97=np.percentile(curr_sel[plotvar], 95)
                    sigma_84=np.percentile(curr_sel[plotvar], 84.13)
                    sigma_15=np.percentile(curr_sel[plotvar], 15.87)
                    sigma_2=np.percentile(curr_sel[plotvar], 5)
                    
                    x_10mode=x[max_index]
                    if plotvar == 'erosion_mass_min' or plotvar == 'erosion_mass_max':
                        densest_point[ii_densest]=10**(densest_point[ii_densest])
                        x_10mode=10**x[max_index]

                    if i<9:
                        print('\\hline') #df_sel_save[df_sel_save['solution_id']==only_select_meteors_from][plotvar].values[0]
                        # print(f"{to_plot_unit[i]} & ${'{:.4g}'.format(df_sel_save[df_sel_save['solution_id']==only_select_meteors_from][plotvar].values[0])}$ & ${'{:.4g}'.format(x_10mode)}$ & $ {'{:.2g}'.format(percent_diff_1D[i])}$\\% & $ {'{:.4g}'.format(densest_point[i])}$ & $ {'{:.2g}'.format(percent_diff_allD[i])}$\\% \\\\")
                        # print(to_plot_unit[i]+'& $'+str(x[max_index])+'$ & $'+str(percent_diff_1D[i])+'$\\% & $'+str(densest_point[ii_densest])+'$ & $'+str(percent_diff_allD[i])+'\\% \\\\')
                        # print(f"{to_plot_unit[i]} & ${'{:.4g}'.format(df_sel_save[df_sel_save['solution_id']==only_select_meteors_from][plotvar].values[0])}$ & ${'{:.4g}'.format(x_10mode)}$ & $ {'{:.2g}'.format(percent_diff_1D[i])}$\\% & $ {'{:.4g}'.format(densest_point[i])}$ & $ {'{:.2g}'.format(percent_diff_allD[i])}$\\% \\\\")
                        # print(f"{to_plot_unit[i]} & {'{:.4g}'.format(df_sel_save[df_sel_save['solution_id']==only_select_meteors_from][plotvar].values[0])} & {'{:.4g}'.format(x_10mode)} & {'{:.4g}'.format(densest_point[i])} & {'{:.4g}'.format(sigma_15)} / {'{:.4g}'.format(sigma_84)} & {'{:.4g}'.format(sigma_2)} / {'{:.4g}'.format(sigma_97)} \\\\")
                        print(f"{to_plot_unit[i]} & {'{:.4g}'.format(df_sel_save[df_sel_save['solution_id']==only_select_meteors_from][plotvar].values[0])} & {'{:.4g}'.format(x_10mode)} & {'{:.4g}'.format(densest_point[i])} & {'{:.4g}'.format(sigma_2)} & {'{:.4g}'.format(sigma_97)} \\\\")
                    ii_densest=ii_densest+1

            axs[i].set_ylabel('probability')
            axs[i].set_xlabel(to_plot_unit[i])
            
            # # plot the legend outside the plot
            # axs[i].legend()
            axs[i].get_legend().remove()
                

            if i==0:
                # place the xaxis exponent in the bottom right corner
                axs[i].xaxis.get_offset_text().set_x(1.10)

        # # more space between the subplots erosion_coeff sigma
        plt.tight_layout()

        print('\\hline')
        
        # plt.show()
        print(output_dir+os.sep+'PhysicProp'+n_PC_in_PCA+'_'+str(len(curr_sel))+'ev_dist'+str(np.round(np.min(curr_sel['distance_meteor']),2))+'-'+str(np.round(np.max(curr_sel['distance_meteor']),2))+'.png')
        # save the figure maximized and with the right name
        fig.savefig(output_dir+os.sep+'PhysicProp'+n_PC_in_PCA+'_'+str(len(curr_sel))+'ev_dist'+str(np.round(np.min(curr_sel['distance_meteor']),2))+'-'+str(np.round(np.max(curr_sel['distance_meteor']),2))+'.png', dpi=300)

        # close the figure
        plt.close()

        # Close the Logger to ensure everything is written to the file STOP COPY in TXT file
        sys.stdout.close()

        # Reset sys.stdout to its original value if needed
        sys.stdout = sys.__stdout__

                    # # cumulative distribution histogram of the distance wihouth considering the first two elements
                    # sns.histplot(curr_df_sim_sel, x=curr_df_sim_sel[plotvar][2:], weights=curr_df_sim_sel['weight'][2:],hue='shower_code', ax=axs[i], kde=True, palette='bright', bins=20, cumulative=True, stat='density')
        # # cumulative distribution histogram of the distance wihouth considering the first two elements
        # sns.histplot(curr_df_sim_sel, x=curr_df_sim_sel[plotvar][2:], weights=curr_df_sim_sel['weight'][2:],hue='shower_code', ax=axs[i], kde=True, palette='bright', bins=20, cumulative=True, stat='density')

    ##########################################################################################################

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


def diff_dist_index(data_for_meteor, window_of_smothing_avg=3, std_multip_threshold=1):
    #make subtraction of the next element and the previous element of data_for_meteor["distance_meteor"]
    diff_distance_meteor = np.diff(data_for_meteor["distance_meteor"][:int(len(data_for_meteor["distance_meteor"])/10)])
    # histogram plot of the difference with the count on the x axis and diff_distance_meteor on the y axis 
    indices = np.arange(len(diff_distance_meteor))
    # create the cumulative sum of the diff_distance_meteor
    cumsum_diff_distance_meteor = np.cumsum(diff_distance_meteor)
    # normalize the diff_distance_meteor xnormalized = (x - xminimum) / range of x
    diff_distance_meteor_normalized = (diff_distance_meteor - np.min(diff_distance_meteor)) / (np.max(diff_distance_meteor) - np.min(diff_distance_meteor))

    # # find the index equal to 1 in diff_distance_meteor_normalized
    # index1 = np.where(diff_distance_meteor_normalized == 1)[0]
    # # check when the diff_distance_meteor is two nxt to eac other are smaller than 0.1 starting from the first element
    # select=0
    # for ii in range(len(diff_distance_meteor_normalized)):
    #     if index1<numbr_10:
    #         ii+=index1[0]
    #     if diff_distance_meteor_normalized[ii] < 0.1:
    #         index10percent = ii-1
    #         select+=1
    #         if ii<numbr_10:
    #             if select==numbr_10percent:
    #                 break
    #         else:
    #             if select==1:
    #                 break
    #     else:
    #         select=0

    def moving_average_smoothing(data, window_size):
        smoothed_data = np.convolve(data, np.ones(window_size)/window_size, mode='same')
        return smoothed_data

    # apply the smoothing finction
    smoothed_diff_distance_meteor = moving_average_smoothing(diff_distance_meteor_normalized, window_of_smothing_avg)
    
    # fid the first value of the smoothed_diff_distance_meteor that is smaller than the std of the smoothed_diff_distance_meteor
    index10percent = np.where(smoothed_diff_distance_meteor < np.std(smoothed_diff_distance_meteor)*std_multip_threshold)[0][0]-2
    if index10percent<0:
        index10percent=0

   
    # # dimension of the plot 15,5
    # plt.figure(figsize=(15,5))
    # plt.subplot(1,3,1)
    # plt.bar(indices, diff_distance_meteor_normalized,color='blue', edgecolor='black')
    # plt.xlabel('Index')
    # plt.ylabel('Difference')
    # plt.title('Diff normalized')

    # plt.subplot(1,3,2)
    # plt.bar(indices, cumsum_diff_distance_meteor,color='blue', edgecolor='black')
    # plt.xlabel('Index')
    # plt.ylabel('Cumulative sum')
    # plt.title('Cumulative sum diff')

    # plt.subplot(1,3,3)
    # sns.histplot(data_for_meteor, x=data_for_meteor["distance_meteor"][:100], kde=True, cumulative=True, bins=len(data_for_meteor["distance_meteor"]))
    # plt.ylabel('Index')
    # plt.xlabel('distance')
    # plt.title('Dist')  
    # # give more space
    # plt.tight_layout()  
    # plt.show()

    return index10percent

if __name__ == "__main__":

    import argparse


    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Fom Observation and simulated data weselect the most likely through PCA, run it, and store results to disk.")

    # arg_parser.add_argument('--output_dir', metavar='OUTPUT_PATH', type=str, default=r"C:\Users\maxiv\Documents\UWO\Papers\1)PCA\PCA_Error_propagation\TEST", \
    arg_parser.add_argument('--output_dir', metavar='OUTPUT_PATH', type=str, default=r"C:\Users\maxiv\Documents\UWO\Papers\1)PCA\Reproces_2cam\SimFolder\TEST", \
        help="Path to the output directory.")

    arg_parser.add_argument('--shower', metavar='SHOWER', type=str, default='PER', \
        help="Use specific shower from the given simulation.")
    
    # arg_parser.add_argument('--input_dir', metavar='INPUT_PATH', type=str, default=r"C:\Users\maxiv\Documents\UWO\Papers\1)PCA\PCA_Error_propagation\TEST", \
    arg_parser.add_argument('--input_dir', metavar='INPUT_PATH', type=str, default=r"C:\Users\maxiv\Documents\UWO\Papers\1)PCA\Reproces_2cam\SimFolder\TEST", \
        help="Path were are store both simulated and observed shower .csv file.")

    # arg_parser.add_argument('--true_file', metavar='TRUE_FILE', type=str, default='TRUEerosion_sim_v65.00_m7.01e-04g_rho0709_z51.7_abl0.015_eh115.2_er0.483_s2.46.json', \ TRUEerosion_sim_v59.84_m1.33e-02g_rho0209_z39.8_abl0.014_eh117.3_er0.636_s1.61.json
    arg_parser.add_argument('--true_file', metavar='TRUE_FILE', type=str, default='TRUEerosion_sim_v59.84_m1.33e-02g_rho0209_z39.8_abl0.014_eh117.3_er0.636_s1.61.json', \
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

    # check if the file is in the folder
    if os.path.exists(os.path.join(cml_args.input_dir_true, cml_args.true_file)):
        # Load the nominal simulation
        print('Loaded: ', cml_args.true_file)   
        mkdirP(cml_args.output_dir)
    else:
        print(cml_args.true_file,' file in not is the folder')   
        # try and load the first json file in the folder
        for file in os.listdir(cml_args.input_dir_true):
            if file.endswith(".json"):
                sim_fit_json_nominal = os.path.join(cml_args.input_dir_true, file)
                print('Loaded: ', sim_fit_json_nominal)
                cml_args.true_file=file
                mkdirP(cml_args.output_dir)
                break
                

    # #########################

    # file_handler = ['TRUEerosion_sim_v59.84_m1.33e-02g_rho0209_z39.8_abl0.014_eh117.3_er0.636_s1.61.json',
    #     'TRUEerosion_sim_v59.84_m1.33e-02g_rho0209_z39.8_abl0.014_eh117.3_er0.636_s1.61_2_noisy.json',
    #     'TRUEerosion_sim_v59.84_m1.33e-02g_rho0209_z39.8_abl0.014_eh117.3_er0.636_s1.61_3_noisy.json',
    #     'TRUEerosion_sim_v59.84_m1.33e-02g_rho0209_z39.8_abl0.014_eh117.3_er0.636_s1.61_4_noisy.json',
    #     'TRUEerosion_sim_v59.84_m1.33e-02g_rho0209_z39.8_abl0.014_eh117.3_er0.636_s1.61_5_noisy.json',
    #     'TRUEerosion_sim_v59.84_m1.33e-02g_rho0209_z39.8_abl0.014_eh117.3_er0.636_s1.61_6_noisy.json',
    #     'TRUEerosion_sim_v59.84_m1.33e-02g_rho0209_z39.8_abl0.014_eh117.3_er0.636_s1.61_7_noisy.json',
    #     'TRUEerosion_sim_v59.84_m1.33e-02g_rho0209_z39.8_abl0.014_eh117.3_er0.636_s1.61_8_noisy.json',
    #     'TRUEerosion_sim_v59.84_m1.33e-02g_rho0209_z39.8_abl0.014_eh117.3_er0.636_s1.61_9_noisy.json']
     
    # # create an output directory 1_noise, 2_noise, 3_noise, 4_noise, 5_noise, 6_noise, 7_noise, 8_noise, 9_noise
    # for ii in range(len(file_handler)):
    #     mkdirP(cml_args.output_dir+os.sep+str(ii+1)+'_noise')
    #     folder_output=cml_args.output_dir+os.sep+str(ii+1)+'_noise'
    #     print(folder_output)
    #     print(file_handler)

    #     PCA_confrontPLOT(folder_output, Shower, cml_args.input_dir, file_handler[ii], cml_args.input_dir_true)

    #     PCA_LightCurveCoefPLOT(folder_output, Shower, cml_args.input_dir, file_handler[ii], cml_args.input_dir_true)

    #########################

    PCA_confrontPLOT(cml_args.output_dir, Shower, cml_args.input_dir, cml_args.true_file, cml_args.input_dir_true)

    PCA_LightCurveCoefPLOT(cml_args.output_dir, Shower, cml_args.input_dir, cml_args.true_file, cml_args.input_dir_true)
