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
from scipy.optimize import minimize
import scipy.optimize as opt

Shower='PER' # ORI ETA SDA CAP GEM PER

def find_closest_index(time_arr, time_sampled):
    closest_indices = []
    for sample in time_sampled:
        closest_index = min(range(len(time_arr)), key=lambda i: abs(time_arr[i] - sample))
        closest_indices.append(closest_index)
    return closest_indices

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

def cubic_lag(t, a, b, c, t0):
    """
    Quadratic lag function.
    """

    # Only take times <= t0
    t_before = t[t <= t0]

    # Only take times > t0
    t_after = t[t > t0]

    # Compute the lag linearly before t0
    l_before = np.zeros_like(t_before)+c

    # Compute the lag quadratically after t0
    l_after = -abs(a)*(t_after - t0)**3 - abs(b)*(t_after - t0)**2 + c

    return np.concatenate((l_before, l_after))


def cubic_acceleration(t, a, b, t0):
    """
    Quadratic acceleration function.
    """

    # Only take times <= t0
    t_before = t[t <= t0]

    # Only take times > t0
    t_after = t[t > t0]

    # No deceleration before t0
    a_before = np.zeros_like(t_before)

    # Compute the acceleration quadratically after t0
    a_after = -6*abs(a)*(t_after - t0) - 2*abs(b)

    return np.concatenate((a_before, a_after))

def lag_residual(params, t_time, l_data):
    """
    Residual function for the optimization.
    """

    return np.sum((l_data - cubic_lag(t_time, *params))**2)

def PCAmanuallyReduced(OUT_PUT_PATH=os.getcwd(), Shower='PER', INPUT_PATH=['C:\\Users\\maxiv\\Documents\\UWO\\Papers\\1)PCA\\Reductions\\manual_reductions']):
    
    Shower=Shower[0]

    all_picklefiles=[]

    name=[]
    shower_code=[]
    shower_code_sim=[]

    acceleration_parab=[]
    acceleration_lin=[]
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

    F=[]
    trail_len=[]
    zenith_angle=[]

    kurtosyness=[]
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

    a_acc=[]
    b_acc=[]
    c_acc=[]

    a_mag_init=[]
    b_mag_init=[]
    c_mag_init=[]

    a_mag_end=[]
    b_mag_end=[]
    c_mag_end=[]

    acceleration_parab_t0_arr=[]
    decel_t0_arr=[]
    acc_jacchia_arr=[]
    
    a_t0_arr=[]
    b_t0_arr=[] 
    c_t0_arr=[] 
    
    jac_a1_arr=[]
    jac_a2_arr=[]

    t0_arr=[]

    rho=[]
    sigma=[]
    erosion_height_start=[]
    erosion_coeff=[]
    erosion_mass_index=[]
    erosion_mass_min=[]
    erosion_mass_max=[]
    erosion_energy_per_unit_cross_section_arr=[]
    erosion_energy_per_unit_mass_arr=[]
    erosion_range=[]

    # directory='C:\\Users\\maxiv\\Documents\\UWO\\Papers\\1)PCA\\Reductions\\manual_reductions'
    # open the directory and walk through all the files and subfolders and open only the files with the extension .pickle
    for root, dirs, files in os.walk(INPUT_PATH):
        for name_file in files:
            if name_file.endswith("_sim.pickle"):
                print('Not Load pickle file: ', name_file)
            elif name_file.endswith(".pickle"):
                # print(os.path.join(root, name_file))

                print('Loading pickle file: ', name_file)

                traj = wmpl.Utils.Pickling.loadPickle(root,name_file)
                # save the pickle file in a folder PER_pk
                shutil.copy(os.path.join(root, name_file), OUT_PUT_PATH+r'\\'+Shower+'_pk')
                
                jd_dat=traj.jdt_ref

                vel_pickl=[]
                time_pickl=[]
                time_total=[]
                abs_mag_pickl=[]
                abs_total=[]
                height_pickl=[]
                height_total=[]
                lag=[]
                lag_total=[]
                elev_angle_pickl=[]
                elg_pickl=[]
                tav_pickl=[]
                
                lat_dat=[]
                lon_dat=[]


                # print('len(traj.observations)', len(traj.observations))

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
                        lag_01=obs.lag
                        elev_angle_01=obs.elev_data

                    elif jj==2:
                        elg_pickl=obs.velocities[1:int(len(obs.velocities)/4)]
                        if len(elg_pickl)==0:
                            elg_pickl=obs.velocities[1:2]
                        
                        vel_02=obs.velocities
                        time_02=obs.time_data
                        abs_mag_02=obs.absolute_magnitudes
                        height_02=obs.model_ht
                        lag_02=obs.lag
                        elev_angle_02=obs.elev_data

                    # put it at the end obs.velocities[1:] at the end of vel_pickl list
                    vel_pickl.extend(obs.velocities[1:])
                    time_pickl.extend(obs.time_data[1:])
                    time_total.extend(obs.time_data)
                    abs_mag_pickl.extend(obs.absolute_magnitudes[1:])
                    abs_total.extend(obs.absolute_magnitudes)
                    height_pickl.extend(obs.model_ht[1:])
                    height_total.extend(obs.model_ht)
                    lag.extend(obs.lag[1:])
                    lag_total.extend(obs.lag)
                    elev_angle_pickl.extend(obs.elev_data)
                    # length_pickl=len(obs.state_vect_dist[1:])
                    
                    lat_dat=obs.lat
                    lon_dat=obs.lon

                # compute the linear regression
                vel_pickl = [i/1000 for i in vel_pickl] # convert m/s to km/s
                time_pickl = [i for i in time_pickl]
                time_total = [i for i in time_total]
                height_pickl = [i/1000 for i in height_pickl]
                height_total = [i/1000 for i in height_total]
                abs_mag_pickl = [i for i in abs_mag_pickl]
                abs_total = [i for i in abs_total]
                lag=[i/1000 for i in lag]
                lag_total=[i/1000 for i in lag_total]

                # print('length_pickl', length_pickl)
                # length_pickl = [i/1000 for i in length_pickl]


                # find the height when the velocity start dropping from the initial value 
                vel_init_mean = (np.mean(elg_pickl)+np.mean(tav_pickl))/2/1000
                v0=vel_init_mean
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

                # infov_percentile.acceleration_parab[ii]=(-1)*a
                # infov_percentile.v0[ii]=vel_sim_line[0]
                # infov_percentile.vel_avg_norot[ii]=np.mean(vel_sim_line)

                #####order the list by time
                vel_pickl = [x for _,x in sorted(zip(time_pickl,vel_pickl))]
                abs_mag_pickl = [x for _,x in sorted(zip(time_pickl,abs_mag_pickl))]
                abs_total = [x for _,x in sorted(zip(time_total,abs_total))]
                height_pickl = [x for _,x in sorted(zip(time_pickl,height_pickl))]
                height_total = [x for _,x in sorted(zip(time_total,height_total))]
                lag = [x for _,x in sorted(zip(time_pickl,lag))]
                lag_total = [x for _,x in sorted(zip(time_total,lag_total))]
                # length_pickl = [x for _,x in sorted(zip(time_pickl,length_pickl))]
                time_pickl = sorted(time_pickl)
                time_total = sorted(time_total)

                # find the sigle index of the height when the velocity start dropping from the vel_init_mean of about 0.5 km/s
                index = [i for i in range(len(vel_pickl)) if vel_pickl[i] < vel_small-0.2]
                # only use first index to pick the height
                height_knee_vel.append(height_pickl[index[0]-1])
                # print('height_knee_vel', height_pickl[index[0]-1])

                a2, b2 = np.polyfit(time_pickl[index[0]-1:],vel_pickl[index[0]-1:], 1)

                #######################################################
                # fit a line to the throught the vel_sim and ht_sim
                a3, b3, c3 = np.polyfit(time_pickl,vel_pickl, 2)
                curve_fit=[a3*x**2+x*b3+c3 for x in time_total]

                t0 = np.mean(time_pickl)

                # initial guess of deceleration decel equal to linear fit of velocity
                p0 = [a, 0, 0, t0]
                # print(lag)
                # lag_calc = length_pickl-(v0*np.array(time_pickl)+length_pickl[0])
                opt_res = opt.minimize(lag_residual, p0, args=(np.array(time_pickl), np.array(lag)), method='Nelder-Mead')

                # sample the fit for the velocity and acceleration
                a_t0, b_t0, c_t0, t0 = opt_res.x

                # compute reference decelearation
                t_decel_ref = (t0 + np.max(time_pickl))/2
                decel_t0 = cubic_acceleration(t_decel_ref, a_t0, b_t0, t0)[0]

                a_t0=-abs(a_t0)
                b_t0=-abs(b_t0)

                acceleration_parab_t0=a_t0*6 + b_t0*2

                # Assuming the jacchiaVel function is defined as:
                def jacchiaVel(t, a1, a2, v_init):
                    return v_init - np.abs(a1) * np.abs(a2) * np.exp(np.abs(a2) * t)

                # Generating synthetic observed data for demonstration
                t_observed = np.array(time_pickl)  # Observed times
                vel_observed = np.array(vel_pickl)  # Observed velocities

                # Residuals function for optimization
                def residuals(params):
                    a1, a2 = params
                    predicted_velocity = jacchiaVel(t_observed, a1, a2, v0)
                    return np.sum((vel_observed - predicted_velocity)**2)

                # Initial guess for a1 and a2
                initial_guess = [0.005,	10]

                # Apply minimize to the residuals
                result = minimize(residuals, initial_guess)

                # Results
                jac_a1, jac_a2 = abs(result.x)

                acc_jacchia = abs(jac_a1)*abs(jac_a2)**2


                # only use first index to pick the height
                a3_Inabs, b3_Inabs, c3_Inabs = np.polyfit(height_total[:np.argmin(abs_total)],abs_total[:np.argmin(abs_total)], 2)
                curve_fit_absBEGIN=[a3_Inabs*x**2+x*b3_Inabs+c3_Inabs for x in height_total[:np.argmin(abs_total)]]
                #
                a3_Outabs, b3_Outabs, c3_Outabs = np.polyfit(height_total[np.argmin(abs_total):],abs_total[np.argmin(abs_total):], 2)
                curve_fit_absEND=[a3_Outabs*x**2+x*b3_Outabs+c3_Outabs for x in height_total[np.argmin(abs_total):]]

                #############################################################
                        # vel_01=obs.velocities
                        # time_01=obs.time_data
                        # abs_mag_01=obs.absolute_magnitudes
                        # height_01=obs.model_ht
                        # lag_01=obs.lag
                        # elev_angle_01=obs.elev_data
                fig, axs = plt.subplots(1, 2)
                # fig.suptitle(name_file.split('_trajectory')[0]+'A')
                fig.suptitle(name_file.split('_trajectory')[0])
                # plot the velocity and the time and the curve_fit in the first subplot
                # axs[0].plot(time_pickl,vel_pickl, 'd', color='black', label=name_file.split('_trajectory')[0]+'A')

                # plot the velocity and the time and the curve_fit in the first subplot color=ax[0].lines[-1].get_color(), linestyle='None', marker='<'
                # axs[1].plot(abs_mag_pickl,height_pickl, 'd', color='black', label='1')
                # divide height_01/1000 to convert it to km
                height_01 = [i/1000 for i in height_01]
                height_02 = [i/1000 for i in height_02]
                # divide vel_01/1000 to convert it to km
                vel_01 = [i/1000 for i in vel_01]
                vel_02 = [i/1000 for i in vel_02]
                
                axs[0].plot(abs_mag_01,height_01,linestyle='--',marker='x',label='1')
                axs[0].plot(abs_mag_02,height_02,linestyle='--',marker='x',label='2')
                axs[0].plot(curve_fit_absBEGIN,height_total[:np.argmin(abs_total)], linestyle='None', color='black', marker='<', markersize=3)#, color='black', label='Parab Fit Bf.Peak')
                axs[0].plot(curve_fit_absEND,height_total[np.argmin(abs_total):], linestyle='None', color='black', marker='>', markersize=3)#, color='black', label='Parab Fit Af.Peak')
                # ax[0].plot(curr_sel.iloc[ii]['a_mag_init']*np.array(height_pickl[:index_ht_peak])**2+curr_sel.iloc[ii]['b_mag_init']*np.array(height_pickl[:index_ht_peak])+curr_sel.iloc[ii]['c_mag_init'],height_pickl[:index_ht_peak], color=ax[0].lines[-1].get_color(), linestyle='None', marker='<')# , markersize=5
                axs[0].set_xlabel('abs.mag [-]')
                axs[0].set_ylabel('height [km]')
                # invert the x axis
                axs[0].invert_xaxis()
                axs[0].legend()
                axs[0].grid(True)

                # vel_01.insert(0,v0)
                # Swap the first element with the new value
                # vel_01[0], v0 = v0, vel_01[0]
                vel_01[0] = v0
                vel_02[0] = v0
                axs[1].plot(time_01,vel_01,linestyle='None', marker='.',label='1')
                axs[1].plot(time_02,vel_02,linestyle='None', marker='.',label='2')
                # axs[1].plot(0,vel_init_mean, 'x', color='black', label='V0')
                # axs[0].plot(time_pickl,curve_fit, 's', color='red', label='Fitted Parabula')
                # axs[1].plot(time_total,curve_fit, 'o', color='black', label='Fit Parabula')
                # axs[1].plot(time_total, jacchiaVel(np.array(time_total), jac_a1, jac_a2, v0), color='black', linestyle='None', marker='d', label='Jacchia') 
                # make it smaller in size
                axs[1].plot(time_total, quadratic_velocity(np.array(time_total), a_t0, b_t0, v0, t0), color='black', linestyle='None', marker='s', label='Fit t0', markersize=3) 
                axs[1].set_xlabel('time [s]')
                axs[1].set_ylabel('velocity [km/s]')
                axs[1].legend()
                axs[1].grid(True)

                # give space between the subplots
                fig.tight_layout(pad=2.0)
                fig.savefig(os.getcwd()+r'\\fit'+name_file.split('_trajectory')[0]+'A'+'.png', dpi=300)
                plt.close(fig)

                #############################################################

                #plot the velocity and the time and the 
                # print('decel_after_knee_vel', (-1)*a2)
                # print(vel_pickl[index[0]-1:])
                
                acceleration_parab_t0_arr.append(acceleration_parab_t0)
                decel_t0_arr.append(decel_t0)
                acc_jacchia_arr.append(acc_jacchia)

                a_t0_arr.append(a_t0)
                b_t0_arr.append(b_t0)
                c_t0_arr.append(c_t0)

                jac_a1_arr.append(jac_a1)
                jac_a2_arr.append(jac_a2)

                t0_arr.append(t0)


                a_acc.append(a3)
                b_acc.append(b3)
                c_acc.append(c3)

                a_mag_init.append(a3_Inabs)
                b_mag_init.append(b3_Inabs)
                c_mag_init.append(c3_Inabs)

                a_mag_end.append(a3_Outabs)
                b_mag_end.append(b3_Outabs)
                c_mag_end.append(c3_Outabs)

                # append the values to the list
                acceleration_parab.append(a3*2+b3)
                acceleration_lin.append(a)
                # decel_after_knee_vel.append((-1)*a2)
                lag_trend.append(trendLAG)
                # v0=(vel_sim_line[0])
                # v0.append(vel_sim_line[0])
                vel_init_norot.append(vel_init_mean)
                # print('mean_vel_init', vel_sim_line[0])
                vel_avg_norot.append(np.mean(vel_pickl)) #trail_len / duration
                peak_mag_vel.append(vel_pickl[np.argmin(abs_mag_pickl)])   

                begin_height.append(height_total[0])
                end_height.append(height_total[-1])
                peak_mag_height.append(height_total[np.argmin(abs_total)])

                peak_abs_mag.append(np.min(abs_total))
                beg_abs_mag.append(abs_total[0])
                end_abs_mag.append(abs_total[-1])

                lag_fin.append(lag_line[-1])
                lag_init.append(lag_line[0])
                lag_avg.append(np.mean(lag_line))

                duration.append(time_total[-1]-time_total[0])

                kc_par.append(height_total[0] + (2.86 - 2*np.log(vel_sim_line[0]))/0.0612)

                F.append((height_total[0] - height_total[np.argmin(abs_total)]) / (height_total[0] - height_total[-1]))

                zenith_angle.append(90 - elev_angle_pickl[0]*180/np.pi)
                trail_len.append((height_total[0] - height_total[-1])/(np.sin(np.radians(elev_angle_pickl[0]*180/np.pi))))
                
                # vel_avg_norot.append( ((height_pickl[0] - height_pickl[-1])/(np.sin(np.radians(elev_angle_pickl[0]*180/np.pi))))/(time_pickl[-1]-time_pickl[0]) ) #trail_len / duration

                # name.append(name_file.split('_trajectory')[0]+'A')
                name.append(name_file.split('_trajectory')[0])

                Dynamic_pressure_peak_abs_mag.append(wmpl.Utils.Physics.dynamicPressure(lat_dat, lon_dat, height_total[np.argmin(abs_total)]*1000, jd_dat, vel_pickl[np.argmin(abs_mag_pickl)]*1000))


                # check if in os.path.join(root, name_file) present and then open the .json file with the same name as the pickle file with in stead of _trajectory.pickle it has _sim_fit_latest.json
                if os.path.isfile(os.path.join(root, name_file.split('_trajectory')[0]+'_sim_fit.json')):
                    with open(os.path.join(root, name_file.split('_trajectory')[0]+'_sim_fit.json'),'r') as json_file: # 20210813_061453_sim_fit.json
                        data = json.load(json_file)
                        mass.append(data['m_init'])
                        # add also rho	sigma	erosion_height_start	erosion_coeff	erosion_mass_index	erosion_mass_min	erosion_mass_max	erosion_range	erosion_energy_per_unit_cross_section	erosion_energy_per_unit_mass
                        # mass.append(data['m_init'])
                        rho.append(data['rho'])
                        sigma.append(data['sigma'])
                        erosion_height_start.append(data['erosion_height_start']/1000)
                        erosion_coeff.append(data['erosion_coeff'])
                        erosion_mass_index.append(data['erosion_mass_index'])
                        erosion_mass_min.append(data['erosion_mass_min'])
                        erosion_mass_max.append(data['erosion_mass_max'])

                        # Compute the erosion range
                        erosion_range.append(np.log10(data['erosion_mass_max']) - np.log10(data['erosion_mass_min']))

                        cost_path = os.path.join(root, name_file.split('_trajectory')[0]+'_sim_fit.json')

                        # Load the constants
                        const, _ = loadConstants(cost_path)
                        const.dens_co = np.array(const.dens_co)

                        # Compute the erosion energies
                        erosion_energy_per_unit_cross_section, erosion_energy_per_unit_mass = wmpl.MetSim.MetSimErosion.energyReceivedBeforeErosion(const)
                        erosion_energy_per_unit_cross_section_arr.append(erosion_energy_per_unit_cross_section)
                        erosion_energy_per_unit_mass_arr.append(erosion_energy_per_unit_mass)

                else:
                    # if no data on weight is 0
                    mass.append(0)
                    rho.append(0)
                    sigma.append(0)
                    erosion_height_start.append(0)
                    erosion_coeff.append(0)
                    erosion_mass_index.append(0)
                    erosion_mass_min.append(0)
                    erosion_mass_max.append(0)
                    erosion_range.append(0)
                    erosion_energy_per_unit_cross_section_arr.append(0)
                    erosion_energy_per_unit_mass_arr.append(0)



                # mass.append(0)
            
                # # add an 'A' at the end of the name
                # name=name+'A'
                shower_code.append(Shower)

                shower_code_sim.append('sim_'+Shower)
                

                ########################################## SKEWNESS AND kurtosyness ##########################################
                
                # # # find the index that that is a multiples of 0.031 s in time_pickl
                index = [i for i in range(len(time_pickl)) if time_pickl[i] % 0.031 < 0.01]
                # only use those index to create a new list
                time_pickl = [time_total[i] for i in index]
                abs_mag_pickl = [abs_total[i] for i in index]
                height_pickl = [height_total[i] for i in index]
                # vel_pickl = [vel_pickl[i] for i in index]

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
                # vel_pickl = np.delete(vel_pickl, index)


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
                
                # # # plot the mag_sampled_distr as an histogram
                # plt.hist(mag_sampled_distr)
                # plt.show()

                kurtosyness.append(kurtosis(mag_sampled_distr))
                skewness.append(skew(mag_sampled_distr))

    # dataList = [['','', 0, 0, 0,\
    #     0, 0, 0, 0, 0, 0, 0,\
    #     0, 0, 0, 0, 0, 0,\
    #     0]]

    # infov = pd.DataFrame(dataList, columns=['solution_id','shower_code','v0','vel_avg_norot','duration',\
    # 'mass','peak_mag_height','begin_height','end_height','peak_abs_mag','beg_abs_mag','end_abs_mag',\
    # 'F','trail_len','acceleration_parab','zenith_angle', 'kurtosis','skew',\
    # 'kc'])

    # Create a dataframe to store the data
    infov = pd.DataFrame(columns=['solution_id', 'shower_code', 'vel_init_norot', 'vel_avg_norot', 'duration',
                                'mass', 'peak_mag_height', 'begin_height', 'end_height', 't0', 'peak_abs_mag',
                                'beg_abs_mag', 'end_abs_mag', 'F', 'trail_len', 'deceleration_lin', 'deceleration_parab',
                                'decel_parab_t0', 'decel_t0', 'decel_jacchia', 'zenith_angle', 'kurtosis', 'skew',
                                'kc', 'Dynamic_pressure_peak_abs_mag', 'a_acc', 'b_acc', 'c_acc', 'a_t0', 'b_t0', 'c_t0',
                                'a1_acc_jac', 'a2_acc_jac', 'a_mag_init', 'b_mag_init', 'c_mag_init', 'a_mag_end', 'b_mag_end', 'c_mag_end'])

    # Populate the dataframe
    for ii in range(len(name)):
        infov.loc[ii] = [name[ii], shower_code[ii], vel_init_norot[ii], vel_avg_norot[ii], duration[ii],
                        mass[ii], peak_mag_height[ii], begin_height[ii], end_height[ii], t0_arr[ii],
                        peak_abs_mag[ii], beg_abs_mag[ii], end_abs_mag[ii], F[ii], trail_len[ii],
                        acceleration_lin[ii], acceleration_parab[ii], acceleration_parab_t0_arr[ii],
                        decel_t0_arr[ii], acc_jacchia_arr[ii], zenith_angle[ii], kurtosyness[ii],
                        skewness[ii], kc_par[ii], Dynamic_pressure_peak_abs_mag[ii], a_acc[ii], b_acc[ii],
                        c_acc[ii], a_t0_arr[ii], b_t0_arr[ii], c_t0_arr[ii], jac_a1_arr[ii], jac_a2_arr[ii],
                        a_mag_init[ii], b_mag_init[ii], c_mag_init[ii], a_mag_end[ii], b_mag_end[ii], c_mag_end[ii]]


    # Create a dataframe to store the data
    infov_sim = pd.DataFrame(columns=['solution_id','shower_code','vel_init_norot','vel_avg_norot','duration',\
                        'mass','peak_mag_height','begin_height','end_height','t0','peak_abs_mag','beg_abs_mag','end_abs_mag',\
                        'F','trail_len','deceleration_lin','deceleration_parab','decel_parab_t0','decel_t0','decel_jacchia','zenith_angle', 'kurtosis','skew',\
                        'kc','Dynamic_pressure_peak_abs_mag',\
                        'a_acc','b_acc','c_acc','a_t0', 'b_t0', 'c_t0','a1_acc_jac','a2_acc_jac','a_mag_init','b_mag_init','c_mag_init','a_mag_end','b_mag_end','c_mag_end',\
                        'rho','sigma','erosion_height_start','erosion_coeff', 'erosion_mass_index',\
                        'erosion_mass_min','erosion_mass_max','erosion_range',\
                        'erosion_energy_per_unit_cross_section', 'erosion_energy_per_unit_mass'])

    # Populate the dataframe
    for ii in range(len(name)):
        infov_sim.loc[ii] = [name[ii], shower_code_sim[ii], vel_init_norot[ii], vel_avg_norot[ii], duration[ii],
                        mass[ii], peak_mag_height[ii], begin_height[ii], end_height[ii], t0_arr[ii],
                        peak_abs_mag[ii], beg_abs_mag[ii], end_abs_mag[ii], F[ii], trail_len[ii],
                        acceleration_lin[ii], acceleration_parab[ii], acceleration_parab_t0_arr[ii],
                        decel_t0_arr[ii], acc_jacchia_arr[ii], zenith_angle[ii], kurtosyness[ii],
                        skewness[ii], kc_par[ii], Dynamic_pressure_peak_abs_mag[ii], a_acc[ii], b_acc[ii],
                        c_acc[ii], a_t0_arr[ii], b_t0_arr[ii], c_t0_arr[ii], jac_a1_arr[ii], jac_a2_arr[ii],
                        a_mag_init[ii], b_mag_init[ii], c_mag_init[ii], a_mag_end[ii], b_mag_end[ii], c_mag_end[ii], rho[ii], sigma[ii],\
                        erosion_height_start[ii], erosion_coeff[ii], erosion_mass_index[ii],\
                        erosion_mass_min[ii], erosion_mass_max[ii], erosion_range[ii],\
                        erosion_energy_per_unit_cross_section_arr[ii], erosion_energy_per_unit_mass_arr[ii]]

    # Save the dataframe to a CSV file
    infov.to_csv(os.path.join(OUT_PUT_PATH, Shower + '_manual.csv'), index=False)
    infov_sim.to_csv(os.path.join(OUT_PUT_PATH, Shower + '_sim_manual.csv'), index=False)






if __name__ == "__main__":

    import argparse


    ### COMMAND LINE ARGUMENTS

    # Init the command line arguments parser
    arg_parser = argparse.ArgumentParser(description="Get the manual data from the pickle files of the manually reduced events for the PCA.")

    # C:\Users\maxiv\WMPG-repoMAX\Code\PCA\manual_reduce # os.getcwd()
    arg_parser.add_argument('--output_dir', metavar='OUTPUT_PATH', type=str, default=os.getcwd(), \
        help="Path to the output directory where is saved .csv file in the end.")

    arg_parser.add_argument('--shower', metavar='SHOWER', type=str, default="PER", \
        help="Use specific shower from the given simulation.")
    
    arg_parser.add_argument('--input_dir', metavar='INPUT_PATH', type=str, default=r"C:\Users\maxiv\Documents\UWO\Papers\1)PCA\Reductions\manual_reductions", \
        help="Path were are store all manual reduced events, it use walk from the given diretory and looks for puickle and json files.")

    # Parse the command line arguments
    cml_args = arg_parser.parse_args()

    #########################

    # Make the output directory
    mkdirP(cml_args.output_dir)

    # make only one shower
    Shower=[cml_args.shower]

    PCAmanuallyReduced(cml_args.output_dir, Shower, cml_args.input_dir)