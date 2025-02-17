import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import wmpl
import seaborn as sns
from heapq import nsmallest
from heapq import nlargest
import math
from scipy.stats import kurtosis, skew
import os

Shower='PER' # ORI ETA SDA CAP GEM PER
# # Perseids = PER about 1000 rho 100 to 1000 sigma 0.1 to 0.01 2 and 2
# # Leonids = LEON
# # Geminids = GEM about 200 rho 300 t0 3000 sigma 0.005 to 0.05 2 and 2
# # Ursids = URS
# # Orionid = ORI

f = open('solution_table.json',"r")

data = json.loads(f.read())
# FOV begin, FOV end, Duration, mass, f-parameter, trail length, begin height and end height, velocity, acceleration (based on difference with initial and final velocity) 
df = pd.DataFrame(data, columns=['solution_id','shower_code','vel_init_norot','vel_avg_norot','vel_init_norot_err','beg_fov','end_fov','elevation_norot','duration','mass','begin_height','end_height','peak_abs_mag','beg_abs_mag','end_abs_mag','F'])
# df = pd.DataFrame(data, columns=['shower_code','vel_init_norot','vel_init_norot_err','beg_fov','end_fov','elevation_norot','elevation_norot_err','mass','begin_height','begin_height_err','end_height','end_height_err'])


# df = pd.DataFrame(data)
# # # show all the type of data
# print(df.dtypes)

# infov = df.loc[(df.shower_code == Shower) & (df.beg_fov==True) & (df.end_fov==True) & (df.vel_init_norot_err < 2) & (df.begin_height > df.end_height) & (df.vel_init_norot > df.vel_avg_norot) &
# (df.elevation_norot >=0) & (df.elevation_norot <= 90) & (df.begin_height < 180) & (df.F > 0) & (df.F < 1) & (df.begin_height > 80) & (df.vel_init_norot < 75)]
# # delete the rows with NaN values
# infov = infov.dropna()

infov = df.loc[(df.shower_code == Shower) & (df.beg_fov) & (df.end_fov) & (df.vel_init_norot_err < 2) & (df.begin_height > df.end_height) & (df.vel_init_norot > df.vel_avg_norot) &
(df.elevation_norot >=0) & (df.elevation_norot <= 90) & (df.begin_height < 180) & (df.F > 0) & (df.F < 1) & (df.begin_height > 80) & (df.vel_init_norot < 75)]
# delete the rows with NaN values
infov = infov.dropna()


# pick the columns accel_model jacchia and gural-linear
# print(infov['accel_model'].value_counts())

# print(df['accel_model'])
# print(df['accel_coeff'])

# trail_len in km
infov['trail_len'] = (infov['begin_height'] - infov['end_height'])/np.sin(np.radians(infov['elevation_norot']))
# acceleration in km/s^2
infov['acceleration'] = (infov['vel_init_norot'] - infov['vel_avg_norot'])/(infov['duration'])

infov['zenith_angle'] = (90 - infov['elevation_norot'])






# only the first 30
# infov = infov.iloc[0:30]










acceleration=[]
vel_init_norot=[]
vel_avg_norot=[]

begin_height=[]
end_height=[]

peak_abs_mag=[]
beg_abs_mag=[]
end_abs_mag=[]

lag_data=[]

F_data=[]

kurtosisness=[]
skewness=[]

inclin_m=[]

jj=0
for ii in range(len(infov)):
    
    # pick the ii element of the solution_id column 
    namefile=infov.iloc[ii]['solution_id']
    print('Loading pickle file: ', namefile, ' n.', jj, ' of ', len(infov), ' done.')
    jj=jj+1
    # vel_init=infov_percentile.iloc[ii]['solution_id']
    # split the namefile base on the '_' character and pick the first element
    folder=namefile.split('_')[0]
    traj = wmpl.Utils.Pickling.loadPickle("M:\\emccd\\pylig\\trajectory\\"+folder+"\\", namefile+".pylig.pickle")
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
    for obs in traj.observations:
        # put it at the end obs.velocities[1:] at the end of vel_pickl list
        vel_pickl.extend(obs.velocities[1:])
        time_pickl.extend(obs.time_data[1:])
        abs_mag_pickl.extend(obs.absolute_magnitudes[1:])
        height_pickl.extend(obs.model_ht[1:])
        lag.extend(obs.lag[1:])

    # compute the linear regression
    vel_pickl = [i/1000 for i in vel_pickl] # convert m/s to km/s
    time_pickl = [i for i in time_pickl]
    height_pickl = [i/1000 for i in height_pickl]
    abs_mag_pickl = [i for i in abs_mag_pickl]

    # fit a line to the throught the vel_sim and ht_sim
    a, b = np.polyfit(time_pickl,vel_pickl, 1)

    vel_sim_line=[a*x+b for x in time_pickl]

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
    acceleration.append((-1)*a)
    # vel_init_norot.append(vel_sim_line[0])
    vel_init_norot.append(vel_sim_line[0])
    vel_avg_norot.append(np.mean(vel_sim_line))

    begin_height.append(height_pickl[0])
    end_height.append(height_pickl[-1])

    peak_abs_mag.append(np.min(abs_mag_pickl))
    beg_abs_mag.append(abs_mag_pickl[0])
    end_abs_mag.append(abs_mag_pickl[-1])

    lag_data.append(lag[-1])

    # F_data.append((height_pickl[0] - height_pickl[np.argmin(abs_mag_pickl)]) / (height_pickl[0] - height_pickl[-1]))
    

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

    # directrory_xy = "M:\\emccd\\events\\"+folder+"\\"
    # files = [f for f in os.listdir(directrory_xy) if f.endswith('.txt')]

    # # find in file the one that "ev_"+namefile+"_" is in the name
    # files = [f for f in files if namefile in f]
    # print(files)
    # inclin_m2=[]
    # for ii in range(len(files)):
    #     with open(directrory_xy+files[ii]) as f:
    #         contents = f.read()
    #         # take the data time : 20180813 07:19:44.282 UTC
    #         # and convert it to a datetime object
    #         Station = contents[contents.find('text :')+1:contents.find('#')-1]
    #         # keep everthing after :
    #         Station = Station[Station.find(':')+1:]
    #         # delete everithing after #
    #         Station = Station[:Station.find('i')-0]

    #         time = contents[contents.find('time :')+1:contents.find('UTC')-1]
    #         # split by : and keep the last one
    #         time = time.split(':')[-1]
    #         # to number
    #         time = float(time)
    #         # save the data after the # in a pd dataframe
    #         dftxt = pd.read_csv(directrory_xy+files[ii], comment='#', delim_whitespace=True, header=None)
    #         # hold only column 1 4 and 5
    #         dftxt = dftxt[[1,4,5,9]]
    #         # rename the columns
    #         dftxt.columns = ['seconds','x_image', 'y_image','mag']
    #         # sum the time from the time column
    #         dftxt['seconds'] = time + dftxt['seconds']

    #     # add a column to the dataframe with the distance
    #     dftxt['distance'] = np.sqrt(dftxt['x_image']**2 + dftxt['y_image']**2)
    #     # difference of distance to the previous point
    #     dftxt['diff_distance'] = dftxt['distance'].diff()/dftxt['seconds'].diff()
    #     # distance between points
    #     dftxt['distance_point'] = np.sqrt((dftxt['x_image'].diff())**2 + (dftxt['y_image'].diff())**2)/dftxt['seconds'].diff()

    #     m, b = np.polyfit(dftxt['seconds'][1:],dftxt['distance_point'][1:], 1)
    #     # save m and do a mean of all the m
    #     inclin_m2.append(m)
    # inclin_m.append(np.mean(inclin_m2))




infov['begin_height']=begin_height
infov['end_height']=end_height

# infov['peak_abs_mag']=peak_abs_mag
# infov['beg_abs_mag']=beg_abs_mag
# infov['end_abs_mag']=end_abs_mag
    
infov['acceleration'] = acceleration
infov['vel_init_norot'] = vel_init_norot
infov['vel_avg_norot'] = vel_avg_norot

infov['kc'] = infov['begin_height'] + (2.86 - 2*np.log(infov['vel_init_norot']))/0.0612
infov['kurtosis'] = kurtosisness
infov['skew'] = skewness 

infov['lag'] = lag_data 

# infov['inclin_m'] = inclin_m





############################################################################################################
# NEW DATAFRAME WITH THE PERCENTILES

# infov_percentile = infov.loc[
# (infov.mass<np.percentile(infov['mass'], 95)) & (infov.mass>np.percentile(infov['mass'], 5)) & 
# (infov.duration<np.percentile(infov['duration'], 95)) &
# (infov.elevation_norot<np.percentile(infov['zenith_angle'], 95)) &
# (infov.beg_abs_mag<np.percentile(infov['beg_abs_mag'], 95)) & (infov.beg_abs_mag>np.percentile(infov['beg_abs_mag'], 5)) &
# (infov.vel_init_norot<np.percentile(infov['vel_init_norot'], 95)) & (infov.vel_init_norot>np.percentile(infov['vel_init_norot'], 5)) 
# ]

#######################################

# infov_percentile = infov

infov_percentile = infov.loc[
(infov.elevation_norot>25) &
# (infov.begin_height>105) &
(infov.acceleration>0) &
(infov.acceleration<100) &
# (infov.vel_init_norot<np.percentile(infov['vel_init_norot'], 99)) & (infov.vel_init_norot>np.percentile(infov['vel_init_norot'], 1)) &
# (infov.vel_avg_norot<np.percentile(infov['vel_avg_norot'], 99)) & (infov.vel_avg_norot>np.percentile(infov['vel_avg_norot'], 1)) &
(infov.vel_init_norot<72) &
(infov.trail_len<50)
]

#######################################

# # only the first 20 events
# infov_percentile = infov_percentile.iloc[0:20]

infov = infov.loc[
    (infov.elevation_norot>25) &
    (infov.acceleration>0) &
    (infov.acceleration<100) &
    # (df_shower_EMCCD.vel_init_norot<np.percentile(df_shower_EMCCD['vel_init_norot'], 99)) & (df_shower_EMCCD.vel_init_norot>np.percentile(df_shower_EMCCD['vel_init_norot'], 1)) &
    # (df_shower_EMCCD.vel_avg_norot<np.percentile(df_shower_EMCCD['vel_avg_norot'], 99)) & (df_shower_EMCCD.vel_avg_norot>np.percentile(df_shower_EMCCD['vel_avg_norot'], 1)) &
    (infov.vel_init_norot<72) &
    (infov.trail_len<50)
    ]

# (infov.acceleration<np.percentile(infov['acceleration'], 95)) & (infov.acceleration>np.percentile(infov['acceleration'], 5))


# delete the rows with NaN values
infov_percentile = infov_percentile.dropna()
# delete the entry with accel_coeff list for each term <0
# infov_percentile = infov_percentile[infov_percentile.accel_coeff.apply(lambda x: all(i >= 0 for i in x))]


# put in infov['acceleration'] the second df['accel_coeff'] value if the df['accel_model'] is jacchia
# infov_percentile.loc[infov_percentile['accel_model'] == 'jacchia', 'acceleration'] = infov_percentile['accel_coeff'].apply(lambda x: x[1])
# infov_percentile.loc[infov_percentile['accel_model'] == 'gural-linear', 'acceleration'] = infov_percentile['accel_coeff'].apply(lambda x: x[0])



# infov_percentile['begin_height']=begin_height
# infov_percentile['end_height']=end_height

# infov_percentile['peak_abs_mag']=peak_abs_mag
# infov_percentile['beg_abs_mag']=beg_abs_mag
# infov_percentile['end_abs_mag']=end_abs_mag

# infov_percentile['F']=F_data

# trail_len in km
infov_percentile['trail_len'] = (infov_percentile['begin_height'] - infov_percentile['end_height'])/np.sin(np.radians(infov_percentile['elevation_norot']))
# Zenith angle in radians
infov_percentile['zenith_angle'] = (90 - infov_percentile['elevation_norot'])

# print the minimum initia magnitude with also written 'min magnitude' in the same line
# round it to the biggest integer
print('\nmin init mag [-] ', np.round(nlargest(5, infov_percentile['beg_abs_mag']),1))
print('Max init mag [-] ', np.round(nsmallest(5, infov_percentile['beg_abs_mag']),1))

# print the minimum and maximum mass
print('\nmin mass [kg] ', np.round(nsmallest(5,infov_percentile['mass']),9))
print('Max mass [kg] ', np.round(nlargest(5,infov_percentile['mass']),6))

# print the minimum and maximum velocity
print('\nmin velocity [km/s] ', np.round(nsmallest(5,infov_percentile['vel_init_norot']),1))
print('Max velocity [km/s] ', np.round(nlargest(5,infov_percentile['vel_init_norot']),1))

# print the minimum and maximum Zenith angle in degrees
print('\nmin Zenith [deg] ', np.round(nsmallest(5,infov_percentile['zenith_angle']),1))
print('Max Zenith [deg] ', np.round(nlargest(5,infov_percentile['zenith_angle']),1))

# print the minimum duration
print('\nmin duration [s] ', np.round(nsmallest(5,infov_percentile['duration']),3))
print('Max duration [s] ', np.round(nlargest(5,infov_percentile['duration']),3))

# print the minimum and max of the begin height
print('\nmin begin height [km] ', np.round(nsmallest(5,infov_percentile['begin_height']),1))
print('Max begin height [km] ', np.round(nlargest(5,infov_percentile['begin_height']),1))

# print the minimum and max of the begin height
print('\nmin begin acceleration [km/s^2] ', np.round(nsmallest(5,infov_percentile['acceleration']),1))
print('Max begin acceleration [km/s^2] ', np.round(nlargest(5,infov_percentile['acceleration']),1))

print(len(infov))
print(len(infov_percentile))

# drop the columns that are not needed ,'accel_model','accel_coeff'
infov_percentile = infov_percentile.drop(['vel_init_norot_err','beg_fov','end_fov','elevation_norot'], axis=1)

#################################################
# infov['shower_code']=Shower+'_base'
# # add infov
# infov = pd.concat([infov_percentile,infov.drop(['accel_model','accel_coeff','vel_init_norot_err','beg_fov','end_fov','elevation_norot'], axis=1)], axis=0)

# to_plot=['vel_avg_norot','duration','mass','begin_height','end_height','peak_abs_mag','beg_abs_mag','end_abs_mag','F','trail_len','acceleration','zenith_angle']
# fig, axs = plt.subplots(4, 3)
# fig.suptitle(Shower)
# jj=0
# # with color based on the shower but skip the first 2 columns (shower_code, shower_id)
# for i in range(4):
#     for j in range(3):
#         if to_plot[jj]=='mass':
#                         # put legendoutside north
#             sns.histplot(infov, x=to_plot[jj], hue='shower_code',ax=axs[i,j], kde=True, palette="dark:#5A9_r", bins=20, log_scale=True)
#             # delete the legend
#             axs[i,j].get_legend().remove()
#         else:
#             # put legendoutside north
#             sns.histplot(infov, x=to_plot[jj], hue='shower_code',ax=axs[i,j], kde=True, palette="dark:#5A9_r", bins=20)
#             if jj!=11:
#                 axs[i,j].get_legend().remove()
            
            
#         jj=jj+1
        
# # more space between the subplots
# plt.tight_layout()
# # full screen
# figManager = plt.get_current_fig_manager()
# figManager.window.showMaximized()
# plt.show()
###################################################


# save the dataframe to a csv file
infov_percentile.to_csv('C:\\Users\\maxiv\\Documents\\UWO\\Papers\\1)PCA\\'+Shower+'_test.csv', index=False)

# dfSelected.to_csv(r'C:\Users\maxiv\Documents\UWO\Courses\ASTRO_9506S-Astro_Machine_Lerning\Project\Simulated_'+Shower+'_select.csv', index=False)

f.close()