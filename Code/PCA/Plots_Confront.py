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

# MODIFY HERE THE PARAMETERS ###############################################################################


# Set the shower name (can be multiple) e.g. 'GEM' or ['GEM','PER', 'ORI', 'ETA', 'SDA', 'CAP']
# Shower=['GEM', 'PER', 'ORI', 'ETA', 'SDA', 'CAP']
Shower=['PER']#['CAP']

# number of selected events selected
n_select=4
dist_select=10000000000

# weight factor for the distance
distance_weight_fact=0

only_select_meteors_from='TRUEerosion_sim_v59.84_m1.33e-02g_rho0209_z39.8_abl0.014_eh117.3_er0.636_s1.61.json'

# FUNCTIONS ###########################################################################################



# save all the simulated showers in a list
df_sim_shower = []
df_obs_shower = []
df_sel_shower = []
# search for the simulated showers in the folder
for current_shower in Shower:
    print('\n'+current_shower)

    df_obs = pd.read_csv(os.getcwd()+r'\\'+current_shower+'_and_dist.csv')

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
    df_sim = pd.read_csv(os.getcwd()+r'\Simulated_'+current_shower+'.csv')
    print('simulation: '+str(len(df_sim)))
    # simulation with acc positive
    df_sim['weight']=1/len(df_sim)
    # df_sim['weight']=0.00000000000000000000001

    # append the simulated shower to the list
    df_sim_shower.append(df_sim)

    # check in the current folder there is a csv file with the name of the simulated shower
    df_sel = pd.read_csv(os.getcwd()+r'\Simulated_'+current_shower+'_select.csv')
    df_sel_save = pd.read_csv(os.getcwd()+r'\Simulated_'+current_shower+'_select.csv')

    # if there only_select_meteors_from is equal to any solution_id_dist
    if only_select_meteors_from in df_sel['solution_id_dist'].values:
        # select only the one with the similar name as only_select_meteors_from in solution_id_dist for df_sel
        df_sel=df_sel[df_sel['solution_id_dist']==only_select_meteors_from]
        df_sel_save=df_sel_save[df_sel_save['solution_id_dist']==only_select_meteors_from]
        print('selected events for : '+only_select_meteors_from)

    if len(df_sel)>n_select:
        df_sel=df_sel.head(n_select)
    
    # # find the one with the same solution_id_dist and select the first n_select in df_sel
    # for i in range(len(df_sel)):
    #     df_sel['solution_id_dist'].iloc[i]=df_sel['solution_id_dist'].iloc[i].split('_')[0]


    # select the data with distance less than dist_select and check if there are more than n_select
    if np.min(df_sel['distance_meteor'])>dist_select:
        print('No selected event below the given minimum distance :'+str(dist_select))
        print('SEL = MAX dist: '+str(np.round(np.max(df_sel['distance_meteor']),2)) +' min dist:'+str(np.round(np.min(df_sel['distance_meteor']),2)))
        print('OBS = MAX mean dist: '+str(np.round(np.max(df_obs['distance']),2)) +' min mean dist:'+str(np.round(np.min(df_obs['distance']),2)))
    else:
        df_sel=df_sel[df_sel['distance_meteor']<dist_select]
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
    curr_df=pd.concat([curr_sim.drop(['rho','sigma','erosion_height_start','erosion_coeff','erosion_mass_index','erosion_mass_min','erosion_mass_max','erosion_energy_per_unit_cross_section',  'erosion_energy_per_unit_mass', 'erosion_range'], axis=1),curr_sel.drop(['distance_meteor','solution_id_dist','rho','sigma','erosion_height_start','erosion_coeff','erosion_mass_index','erosion_mass_min','erosion_mass_max','erosion_energy_per_unit_cross_section',  'erosion_energy_per_unit_mass','distance','erosion_range'], axis=1)], axis=0, ignore_index=True)
    curr_df=pd.concat([curr_df,curr_obs.drop(['distance'], axis=1)], axis=0, ignore_index=True)
    curr_df=curr_df.dropna()
    curr_df_sim_sel=pd.concat([curr_sim,curr_sel.drop(['distance'], axis=1)], axis=0, ignore_index=True)
    
    # sns plot of duration trail lenght and F and acceleration parameters
    # sns.pairplot(curr_df, hue='shower_code', diag_kind='kde', plot_kws={'alpha':0.6, 's':80, 'edgecolor':'k'}, height=3,corner=True)
    # subplot with all the parameters histograms

    # fig, axs = plt.subplots(4, 3)
    # fig.suptitle(current_shower)
    # # with color based on the shower but skip the first 2 columns (shower_code, shower_id)
    # ii=0

    # # to_plot_unit=['init vel [km/s]','avg vel [km/s]','duration [s]','','mass [kg]','begin height [km]','end height [km]','','peak abs mag [-]','begin abs mag [-]','end abs mag [-]','','F parameter [-]','trail lenght [km]','acceleration [km/s^2]','','zenith angle [deg]','kurtosis','kc']
    # # to_plot_unit=['init vel [km/s]','avg vel [km/s]','duration [s]','','begin height [km]','end height [km]','peak abs mag [-]','','begin abs mag [-]','end abs mag [-]','F parameter [-]','','trail lenght [km]','deceleration [km/s^2]','zenith angle [deg]','','kc','kurtosis','skew']
    # # to_plot_unit=['init vel [km/s]','avg vel [km/s]','acceleration [km/s^2]','','begin height [km]','end height [km]','peak abs mag [-]','','begin abs mag [-]','end abs mag [-]','','F parameter [-]','trail lenght [km]','acceleration [km/s^2]','','zenith angle [deg]','kurtosis','kc']
    # to_plot_unit=['init vel [km/s]','avg vel [km/s]','duration [s]','','begin height [km]','peak height [km]','end height [km]','','begin abs mag [-]','peak abs mag [-]','end abs mag [-]','','F parameter [-]','trail lenght [km]','deceleration [km/s^2]','','zenith angle [deg]','kurtosis','skew']


    # # to_plot=['vel_init_norot','vel_avg_norot','duration','','mass','begin_height','end_height','','peak_abs_mag','beg_abs_mag','end_abs_mag','','F','trail_len','acceleration','','zenith_angle','kurtosis','skew']
    # # to_plot=['vel_init_norot','vel_avg_norot','duration','','begin_height','end_height','peak_abs_mag','','beg_abs_mag','end_abs_mag','F','','trail_len','acceleration','zenith_angle','','kc','kurtosis','skew']
    # # to_plot=['vel_init_norot','vel_avg_norot','acceleration','','begin_height','end_height','peak_abs_mag','','peak_abs_mag','beg_abs_mag','end_abs_mag','','F','trail_len','acceleration','','zenith_angle','kurtosis','skew']
    # to_plot=['vel_init_norot','vel_avg_norot','duration','','begin_height','peak_mag_height','end_height','','beg_abs_mag','peak_abs_mag','end_abs_mag','','F','trail_len','acceleration','','zenith_angle','kurtosis','skew']

    # # deleter form curr_df the mass
    # #curr_df=curr_df.drop(['mass'], axis=1)
    # for i in range(4):
    #     for j in range(3):
    #         plotvar=to_plot[ii]
    #         if plotvar=='mass':
    #                         # put legendoutside north curr_df.columns[i*3+j+2]
    #             sns.histplot(curr_df, x=curr_df[plotvar], weights=curr_df['weight'],hue='shower_code', ax=axs[i,j], kde=True, palette='bright', bins=20, log_scale=True)
    #         elif plotvar=='kurtosis':
    #             sns.histplot(curr_df, x=curr_df[plotvar], weights=curr_df['weight'],hue='shower_code', ax=axs[i,j], kde=True, palette='bright', bins=2000)
    #             # x limits
    #             axs[i,j].set_xlim(-1.5,0)
    #         elif plotvar=='skew':
    #             sns.histplot(curr_df, x=curr_df[plotvar], weights=curr_df['weight'],hue='shower_code', ax=axs[i,j], kde=True, palette='bright', bins=200)
    #             # x limits
    #             axs[i,j].set_xlim(-1,1)
    #         else:
    #             # put legendoutside north
    #             sns.histplot(curr_df, x=curr_df[plotvar], weights=curr_df['weight'],hue='shower_code', ax=axs[i,j], kde=True, palette='bright', bins=20)
    #         axs[i,j].set_ylabel('percentage')
    #         axs[i,j].set_xlabel(to_plot_unit[ii])
    #         if ii!=0:
    #             axs[i,j].get_legend().remove()
    #         ii=ii+1
    #     ii=ii+1
            
    # # more space between the subplots
    # plt.tight_layout()
    # # full screen
    # figManager = plt.get_current_fig_manager()
    # figManager.window.showMaximized()
    # plt.show()

    
    # # save the figure
    # fig.savefig(os.getcwd()+r'\Plots\Histograms_'+current_shower+'.png', dpi=300)

######### DISTANCE PLOT ##################################################
    # save the distance_meteor from df_sel_save
    distance_meteor_sel_save=curr_sel_save['distance_meteor']
    # save the distance_meteor from df_sel_save
    distance_meteor_sel=curr_sel['distance_meteor']

    # check if distance_meteor_sel_save index is bigger than the index distance_meteor_sel+50
    if len(distance_meteor_sel)<100:
        sns.histplot(distance_meteor_sel_save[1:], kde=True, cumulative=True, bins=len(distance_meteor_sel_save), color='r') # , stat='density' to have probability
        plt.ylim(0,100) 
    else:
    # plot the cumulative distribution histogram of distance_meteor_sel_save with the number of elements on the y axis
        sns.histplot(distance_meteor_sel_save[1:], kde=True, cumulative=True, bins=len(distance_meteor_sel_save), color='r') # , stat='density' to have probability
        # plt.ylim(0,len(distance_meteor_sel_save))
    # axis label
    plt.xlabel('Distance in PCA space')
    plt.ylabel('Number of events')

    # plot a dasced line with the max distance_meteor_sel
    plt.axvline(x=np.max(distance_meteor_sel), color='k', linestyle='--')

    # make the y axis logarithmic
    # plt.xscale('log')
    
    # show
    plt.show()
##########################################################################

    # with color based on the shower but skip the first 2 columns (shower_code, shower_id)
    to_plot=['rho','sigma','erosion_height_start','','erosion_coeff','erosion_mass_index','erosion_mass_min','','erosion_mass_max','erosion_energy_per_unit_cross_section','erosion_energy_per_unit_mass']
    ii=0
    to_plot=['mass','rho','sigma','','erosion_height_start','erosion_coeff','erosion_energy_per_unit_mass','','erosion_mass_index','erosion_mass_min','erosion_mass_max','','erosion_range','erosion_energy_per_unit_cross_section','erosion_energy_per_unit_cross_section']
    to_plot_unit=['mass [kg]','rho [kg/m^3]','sigma [s^2/km^2]','','erosion height start [km]','erosion coeff [s^2/km^2]','erosion energy per unit mass [MJ/kg]','','erosion mass index [-]','erosion mass min [kg]','erosion mass max [kg]','','log erosion mass range [-]','erosion energy per unit cross section [MJ/m^2]','erosion energy per unit cross section [MJ/m^2]']
    
    to_plot=['rho','sigma','erosion_height_start','','erosion_coeff','erosion_mass_index','erosion_mass_min','','erosion_mass_max','erosion_energy_per_unit_cross_section','erosion_energy_per_unit_mass']
    ii=0
    to_plot=['mass','rho','sigma','','erosion_height_start','erosion_coeff','erosion_mass_index','','erosion_mass_min','erosion_mass_max','erosion_range','','erosion_energy_per_unit_mass','erosion_energy_per_unit_cross_section','erosion_energy_per_unit_cross_section']
    to_plot_unit=['mass [kg]','rho [kg/m^3]','sigma [s^2/km^2]','','erosion height start [km]','erosion coeff [s^2/km^2]','erosion mass index [-]','','erosion mass min [kg]','erosion mass max [kg]','log erosion mass range [-]','','erosion energy per unit mass [MJ/kg]','erosion energy per unit cross section [MJ/m^2]','erosion energy per unit cross section [MJ/m^2]']
    
    # multiply the erosion coeff by 1000000 to have it in km/s
    curr_df_sim_sel['erosion_coeff']=curr_df_sim_sel['erosion_coeff']*1000000
    curr_df_sim_sel['sigma']=curr_df_sim_sel['sigma']*1000000
    curr_df_sim_sel['erosion_energy_per_unit_cross_section']=curr_df_sim_sel['erosion_energy_per_unit_cross_section']/1000000
    curr_df_sim_sel['erosion_energy_per_unit_mass']=curr_df_sim_sel['erosion_energy_per_unit_mass']/1000000
    # pick the one with shower_code==current_shower+'_sel'
    Acurr_df_sel=curr_df_sim_sel[curr_df_sim_sel['shower_code']==current_shower+'_sel']
    Acurr_df_sim=curr_df_sim_sel[curr_df_sim_sel['shower_code']=='sim_'+current_shower]
    
    fig, axs = plt.subplots(3, 3)
    fig.suptitle(current_shower)
    
    for i in range(3):
        for j in range(3):
            # put legendoutside north
            plotvar=to_plot[ii]
            if plotvar == 'mass' or plotvar == 'erosion_mass_min' or plotvar == 'erosion_mass_max' or plotvar == 'erosion_energy_per_unit_cross_section'or plotvar == 'erosion_energy_per_unit_mass':
                sns.histplot(curr_df_sim_sel, x=curr_df_sim_sel[plotvar], weights=curr_df_sim_sel['weight'],hue='shower_code', ax=axs[i,j], kde=True, palette='bright', bins=20, log_scale=True)
                axs[i,j].set_xlabel(to_plot_unit[ii])
                axs[i,j].set_ylabel('percentage')
                # gaussian_kde_sel=scipy.stats.gaussian_kde(Acurr_df_sel[plotvar], bw_method=None, weights=None)
                # gaussian_kde_sim=scipy.stats.gaussian_kde(Acurr_df_sim[plotvar], bw_method=None, weights=None)
                # # plot the difference between the two gaussian kde
                # diff_gaussian_kde=gaussian_kde_sel(Acurr_df_sel[plotvar])-gaussian_kde_sim(Acurr_df_sim[plotvar])
                # if the only_select_meteors_from is equal to any curr_df_sim_sel plot the observed event value as a vertical red line

                if only_select_meteors_from in curr_df_sim_sel['solution_id_dist'].values:
                    axs[i,j].axvline(x=curr_df_sim_sel[curr_df_sim_sel['solution_id_dist']==only_select_meteors_from][plotvar].values[0], color='r', linewidth=2)
                            
                

                if ii!=0:
                    axs[i,j].get_legend().remove()
                # else:
                    # put the legend outside the plot and in the north position with two columns
                    # axs[i,j].legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)
            elif plotvar == 'erosion_range':
                sns.histplot(curr_df_sim_sel, x=curr_df_sim_sel[plotvar], weights=curr_df_sim_sel['weight'],hue='shower_code', ax=axs[i,j], kde=True, palette='bright', bins=20)
                axs[i,j].set_xlabel(to_plot_unit[ii])
                axs[i,j].set_ylabel('percentage')
                # if the only_select_meteors_from is equal to any curr_df_sim_sel plot the observed event value as a vertical red line
                if only_select_meteors_from in df_sel_shower['solution_id_dist'].values:
                    axs[i,j].axvline(x=curr_df_sim_sel[curr_df_sim_sel['solution_id_dist']==only_select_meteors_from][plotvar].values[0], color='r', linewidth=2)

                if ii!=0:
                    axs[i,j].get_legend().remove()

            else:
                sns.histplot(curr_df_sim_sel, x=curr_df_sim_sel[plotvar], weights=curr_df_sim_sel['weight'],hue='shower_code', ax=axs[i,j], kde=True, palette='bright', bins=20)
                axs[i,j].set_ylabel('percentage')
                axs[i,j].set_xlabel(to_plot_unit[ii])
                axs[i,j].get_legend().remove()
                # if the only_select_meteors_from is equal to any curr_df_sim_sel plot the observed event value as a vertical red line
                if only_select_meteors_from in df_sel_shower['solution_id_dist'].values:
                    axs[i,j].axvline(x=curr_df_sim_sel[curr_df_sim_sel['solution_id_dist']==only_select_meteors_from][plotvar].values[0], color='r', linewidth=2)

            ii=ii+1
        ii=ii+1


    # compute the kernel density estimate curve of curr_df_sim_sel['rho'] with curr_df_sim_sel['shower_code']=='PER'
    

    

    # more space between the subplots erosion_coeff sigma
    plt.tight_layout()

    # full screen
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    # plt.show()
    print(os.getcwd()+r'\\PhysicProp'+str(len(df_sel))+'_dist'+str(np.round(np.min(df_sel['distance_meteor']),2))+'-'+str(np.round(np.max(df_sel['distance_meteor']),2))+'.png')
    # save the figure maximized and with the right name
    fig.savefig(os.getcwd()+r'\\PhysicProp'+str(len(df_sel))+'_dist'+str(np.round(np.min(df_sel['distance_meteor']),2))+'-'+str(np.round(np.max(df_sel['distance_meteor']),2))+'.png', dpi=300)



                # # cumulative distribution histogram of the distance wihouth considering the first two elements
                # sns.histplot(curr_df_sim_sel, x=curr_df_sim_sel[plotvar][2:], weights=curr_df_sim_sel['weight'][2:],hue='shower_code', ax=axs[i,j], kde=True, palette='bright', bins=20, cumulative=True, stat='density')

