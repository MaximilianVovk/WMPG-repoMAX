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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# MODIFY HERE THE PARAMETERS ###############################################################################


# Set the shower name (can be multiple) e.g. 'GEM' or ['GEM','PER', 'ORI', 'ETA', 'SDA', 'CAP']
# Shower=['GEM', 'PER', 'ORI', 'ETA', 'SDA', 'CAP']
Shower=['PER']#['CAP']

# number of selected events selected
n_select=100000000
dist_select=10000000000

# weight factor for the distance
distance_weight_fact=0

only_select_meteors_from=''

# FUNCTIONS ###########################################################################################



# save all the simulated showers in a list
df_sim_shower = []
df_obs_shower = []
df_sel_shower = []
# search for the simulated showers in the folder
for current_shower in Shower:
    print('\n'+current_shower)

    df_obs = pd.read_csv(os.getcwd()+r'\\'+current_shower+'.csv')

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
    # df_obs=df_obs[df_obs['acceleration']>0]
    # append the observed shower to the list
    df_obs_shower.append(df_obs)
    

    # check in the current folder there is a csv file with the name of the simulated shower
    df_sim = pd.read_csv(os.getcwd()+r'\Simulated_'+current_shower+'.csv')
    print('simulation: '+str(len(df_sim)))
    # simulation with acc positive
    df_sim=df_sim[df_sim['acceleration']>0]
    df_sim=df_sim[df_sim['acceleration']<100]
    df_sim=df_sim[df_sim['trail_len']<50]
    df_sim['weight']=1/len(df_sim)
    # df_sim['weight']=0.00000000000000000000001

    # append the simulated shower to the list
    df_sim_shower.append(df_sim)





    



# concatenate all the EMCCD observed showers in a single dataframe
df_obs_shower = pd.concat(df_obs_shower)

# concatenate all the simulated shower in a single dataframe
df_sim_shower = pd.concat(df_sim_shower)

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

    curr_df=pd.concat([curr_sim.drop(['rho','sigma','erosion_height_start','erosion_coeff','erosion_mass_index','erosion_mass_min','erosion_mass_max','erosion_energy_per_unit_cross_section',  'erosion_energy_per_unit_mass', 'erosion_range'], axis=1)], axis=0, ignore_index=True)
    curr_df=pd.concat([curr_df,curr_obs], axis=0, ignore_index=True)
    curr_df=curr_df.dropna()

    
    # sns plot of duration trail lenght and F and acceleration parameters
    # sns.pairplot(curr_df, hue='shower_code', diag_kind='kde', plot_kws={'alpha':0.6, 's':80, 'edgecolor':'k'}, height=3,corner=True)
    # subplot with all the parameters histograms

    fig, axs = plt.subplots(4, 3)
    fig.suptitle(current_shower)
    # with color based on the shower but skip the first 2 columns (shower_code, shower_id)
    ii=0

    # to_plot_unit=['init vel [km/s]','avg vel [km/s]','duration [s]','','mass [kg]','begin height [km]','end height [km]','','peak abs mag [-]','begin abs mag [-]','end abs mag [-]','','F parameter [-]','trail lenght [km]','acceleration [km/s^2]','','zenith angle [deg]','kurtosis','kc']
    # to_plot_unit=['init vel [km/s]','avg vel [km/s]','duration [s]','','begin height [km]','end height [km]','peak abs mag [-]','','begin abs mag [-]','end abs mag [-]','F parameter [-]','','trail lenght [km]','deceleration [km/s^2]','zenith angle [deg]','','kc','kurtosis','skew']
    # to_plot_unit=['init vel [km/s]','avg vel [km/s]','acceleration [km/s^2]','','begin height [km]','end height [km]','peak abs mag [-]','','begin abs mag [-]','end abs mag [-]','','F parameter [-]','trail lenght [km]','acceleration [km/s^2]','','zenith angle [deg]','kurtosis','kc']
    to_plot_unit=['init vel [km/s]','avg vel [km/s]','duration [s]','','begin height [km]','peak height [km]','end height [km]','','begin abs mag [-]','peak abs mag [-]','end abs mag [-]','','F parameter [-]','trail lenght [km]','deceleration [km/s^2]','','zenith angle [deg]','kurtosis','skew']


    # to_plot=['vel_init_norot','vel_avg_norot','duration','','mass','begin_height','end_height','','peak_abs_mag','beg_abs_mag','end_abs_mag','','F','trail_len','acceleration','','zenith_angle','kurtosis','skew']
    # to_plot=['vel_init_norot','vel_avg_norot','duration','','begin_height','end_height','peak_abs_mag','','beg_abs_mag','end_abs_mag','F','','trail_len','acceleration','zenith_angle','','kc','kurtosis','skew']
    # to_plot=['vel_init_norot','vel_avg_norot','acceleration','','begin_height','end_height','peak_abs_mag','','peak_abs_mag','beg_abs_mag','end_abs_mag','','F','trail_len','acceleration','','zenith_angle','kurtosis','skew']
    to_plot=['vel_init_norot','vel_avg_norot','duration','','begin_height','peak_mag_height','end_height','','beg_abs_mag','peak_abs_mag','end_abs_mag','','F','trail_len','acceleration','','zenith_angle','kurtosis','skew']

    # deleter form curr_df the mass
    #curr_df=curr_df.drop(['mass'], axis=1)
    for i in range(4):
        for j in range(3):
            plotvar=to_plot[ii]
            if plotvar=='mass':
                # put legendoutside north curr_df.columns[i*3+j+2]
                sns.histplot(curr_df, x=curr_df[plotvar], weights=curr_df['weight'],hue='shower_code', ax=axs[i,j], kde=True, palette='bright', bins=20, log_scale=True)
            elif plotvar=='kurtosis':
                sns.histplot(curr_df, x=curr_df[plotvar], weights=curr_df['weight'],hue='shower_code', ax=axs[i,j], kde=True, palette='bright', bins=2000)
                # x limits
                axs[i,j].set_xlim(-1.5,0)
            elif plotvar=='skew':
                sns.histplot(curr_df, x=curr_df[plotvar], weights=curr_df['weight'],hue='shower_code', ax=axs[i,j], kde=True, palette='bright', bins=200)
                # x limits
                axs[i,j].set_xlim(-1,1)
            else:
                # put legendoutside north
                sns.histplot(curr_df, x=curr_df[plotvar], weights=curr_df['weight'],hue='shower_code', ax=axs[i,j], kde=True, palette='bright', bins=20)
            axs[i,j].set_ylabel('percentage')
            axs[i,j].set_xlabel(to_plot_unit[ii])
            if ii!=0:
                axs[i,j].get_legend().remove()
            ii=ii+1
        ii=ii+1
            
    # more space between the subplots
    plt.tight_layout()
    # full screen
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()
    
    # # save the figure
    # fig.savefig(os.getcwd()+r'\Plots\Histograms_'+current_shower+'.png', dpi=300)

