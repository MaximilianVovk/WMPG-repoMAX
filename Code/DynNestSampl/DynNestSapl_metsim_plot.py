import numpy as np
import pandas as pd
import pickle
import sys
import json
import os
import copy
import matplotlib.pyplot as plt
import dynesty
from dynesty import plotting as dyplot
import time
from matplotlib.ticker import ScalarFormatter
from scipy.stats import norm
from scipy.stats import multivariate_normal

from wmpl.MetSim.GUI import loadConstants, saveConstants,SimulationResults
from wmpl.MetSim.MetSimErosion import runSimulation, Constants, zenithAngleAtSimulationBegin
from wmpl.MetSim.ML.GenerateSimulations import generateErosionSim,saveProcessedList,MetParam
from wmpl.Utils.TrajConversions import J2000_JD, date2JD
from wmpl.Utils.Math import meanAngle
from wmpl.Utils.AtmosphereDensity import fitAtmPoly
from wmpl.Utils.Physics import dynamicPressure

ht_sampled = [
    116536.30672989249,
    115110.05178727282,
    113684.16779440964,
    113083.52176646683,
    112258.70769182382,
    111658.26293157891,
    110833.7560201216,
    110233.56447703335,
    109409.46328620492,
    108809.62866081292,
    107986.16022881713,
    107386.98063665455,
    106565.21195410144,
    105968.56935228375,
    105150.10100517314,
    104551.73106847175,
    103734.61524641037,
    103140.58540786116,
    102326.43004031561,
    101735.07664182257,
    100925.22671750661,
    100337.56278479846,
    99533.73715875004,
    98951.32781515278,
    98156.27352170387,
    97581.7215316846,
    96800.25882608269,
    96238.45679599466,
    95480.62157380355,
    94240.66042891663
]

mag_sampled = [
    5.007660533177523,
    3.3265873118503335,
    2.5464444953563428,
    2.145252609826516,
    1.7679457288795726,
    1.741187657219813,
    1.40484226116799,
    1.1238999999255155,
    1.054036156142602,
    0.7899871355675105,
    0.7519377204984318,
    0.6329126310337716,
    0.4518418596153948,
    0.3490237440687807,
    0.2684538603082568,
    0.3345305534532859,
    0.24680534587066463,
    0.31191759237233396,
    0.1699651543419376,
    0.28149072143285303,
    0.3970816649970808,
    0.28046356530150807,
    0.3974088113018212,
    0.6907812280796398,
    0.8369301875849462,
    1.1838672037832072,
    1.6649283756822753,
    2.1730332105068406,
    3.0583341692267147,
    6.308512445827362
]

time_sampled=[
    0.0,
    0.03125,
    0.0625,
    0.07566659137611853,
    0.09375,
    0.10691659137611853,
    0.125,
    0.13816659137611853,
    0.15625,
    0.16941659137611853,
    0.1875,
    0.20066659137611853,
    0.21875,
    0.23191659137611853,
    0.25,
    0.26316659137611853,
    0.28125,
    0.29441659137611853,
    0.3125,
    0.32566659137611853,
    0.34375,
    0.35691659137611853,
    0.375,
    0.38816659137611853,
    0.40625,
    0.41941659137611853,
    0.4375,
    0.45066659137611853,
    0.46875,
    0.5
]

lag_sampled = [
    0.0,
    -4.48762471,
    20.22486645,
    47.87077673,
    -8.36099409,
    -8.69188961,
    48.39385267,
    22.03765375,
    -18.2415974,
    13.06091513,
    -20.53832741,
    -22.21994643,
    -2.78280529,
    -74.2648876,
    -73.42361563,
    -41.15836802,
    -63.42730629,
    -28.25484624,
    -78.82771967,
    -105.06174599,
    -30.32777112,
    -98.71462315,
    -113.29739184,
    -181.65335499,
    -188.03966759,
    -197.33292507,
    -287.942161,
    -284.30179231,
    -392.49153447,
    -587.79484216
]
    
dens_coef = [
            34.067090334166345,
            -2257.137635650747,
            56522.21296907875,
            -734951.556330054,
            5081875.733489659,
            -17804424.227462955,
            24913101.34607125
        ]


simulated_lc_noise = [
    7.76564283,   # 0
    45.9003461,   # 1
    95.51941976,  # 2
    130.7400714,  # 3
    171.5345507,  # 4
    209.23247285, # 5
    267.17677614, # 6
    314.15710598, # 7
    375.82018905, # 8
    433.60141327, # 9
    499.82748122, # 10
    543.0895856,  # 11
    595.75361642, # 12
    630.60615462, # 13
    676.25862748, # 14
    707.84084628, # 15
    738.33267711, # 16
    754.09393909, # 17
    769.10148559, # 18
    759.80631885, # 19
    725.65497342, # 20
    685.73731365, # 21
    608.66967324, # 22
    525.00972133, # 23
    409.14024966, # 24
    314.77655079, # 25
    204.28789954, # 26
    137.17950206, # 27
    58.10412326,  # 28
    8.08138952    # 29
]

v_inti = 59811.87059507625
# make it so can be readed like this metsim_obj.absolute_magnitudes and metsim_obj.leading_frag_height_arr
metsim_obj = type('metsim_obj', (object,), {})()
metsim_obj.absolute_magnitudes = np.array(mag_sampled)
metsim_obj.leading_frag_height_arr = np.array(ht_sampled)
metsim_obj.time = np.array(time_sampled)
metsim_obj.lag = np.array(lag_sampled)
metsim_obj.v_init = v_inti
metsim_obj.luminosity_arr = np.array(simulated_lc_noise)
metsim_obj.const = type('const', (object,), {})()
metsim_obj.const.dens_co = np.array(dens_coef)

noise_lc = 2.754133642710149



def run_simulation(parameter_guess, real_event, var_names):
    '''
        path_and_file = must be a json file generated file from the generate_simulationsm function or from Metsim file
    '''

    # Load the nominal simulation parameters
    const_nominal = Constants()
    const_nominal.dens_co = np.array(const_nominal.dens_co)

    dens_co=real_event.const.dens_co

    ### Calculate atmosphere density coeffs (down to the bottom observed height, limit to 15 km) ###

    # Assign the density coefficients
    const_nominal.dens_co = dens_co

    # Turn on plotting of LCs of individual fragments 
    const_nominal.fragmentation_show_individual_lcs = True

    # var_cost=['v_init','zenith_angle','m_init','rho','sigma','erosion_height_start','erosion_coeff','erosion_mass_index','erosion_mass_min','erosion_mass_max']
    # for loop for the var_cost that also give a number from 0 to the length of the var_cost
    for i, var in enumerate(var_names):
        const_nominal.__dict__[var] = parameter_guess[i]

    const_nominal.P_0m = 935

    const_nominal.disruption_on = False

    const_nominal.lum_eff_type = 5

    # # Minimum height [m]
    # const_nominal.h_kill = 60000

    # # Initial meteoroid height [m]
    # const_nominal.h_init = 180000

    try:
        # Run the simulation
        frag_main, results_list, wake_results = runSimulation(const_nominal, compute_wake=False)
        simulation_MetSim_object = SimulationResults(const_nominal, frag_main, results_list, wake_results)
    except ZeroDivisionError as e:
        print(f"Error during simulation: {e}")
        const_nominal = Constants()
        # Run the simulation
        frag_main, results_list, wake_results = runSimulation(const_nominal, compute_wake=False)
        simulation_MetSim_object = SimulationResults(const_nominal, frag_main, results_list, wake_results)

    return simulation_MetSim_object




# Data to populate the dataframe
bounds = {
    'v_init': [59841.68503,59841.68505],
    'zenith_angle': [0.69443467,0.69443469],
    'm_init': [1.3e-5,1.88e-5],
    'rho': [100,1000],
    'sigma': [0.008/1e6, 0.03/1e6],
    'erosion_height_start': [116000, 118000],
    'erosion_coeff': [0.0, 1/1e6],
    'erosion_mass_index': [1.5, 2.5],
    'erosion_mass_min': [5e-12, 1e-10],
    'erosion_mass_max': [1e-10, 5e-8]
}

variable_map = {
        'v_init':  r"$v_0$",
        'zenith_angle': r"$z_c$",
        'm_init': r'$m_0$ [kg]', 
        'rho': r'$\rho$ [kg/m$^3$]', 
        'sigma': r'$\sigma$ [kg/J]', 
        'erosion_height_start': r'$h_{e}$ [m]', 
        'erosion_coeff': r'$\eta$ [kg/J]', 
        'erosion_mass_index': r'$s$',
        'erosion_mass_min': r'log($m_{l}$)',
        'erosion_mass_max': r'log($m_{u}$)'
    }

# Create the dataframe
bounds = pd.DataFrame(bounds)

var_names = bounds.columns # get the variable names

path_and_file_MetSim = r"C:\Users\maxiv\Documents\UWO\Papers\1)PCA\Results\json_test_CAMO_EMCCD\Simulations_PER_v59_heavy\PER_v59_heavy_sim_fit_latest.json"

ndim = len(bounds.columns)

# filename = '/home/mvovk/Downloads/1994_02_01_restricted_heights.save'  # CHANGE THIS
# filename = r"C:\Users\maxiv\Documents\UWO\Papers\Besian\PER_v59_heavy_dynesty_lum.save"
# filename = r"C:\Users\maxiv\Documents\UWO\Papers\Besian\PER_v59_heavy_dynesty_lum_log.save"
# filename = r"C:\Users\maxiv\Documents\UWO\Papers\Besian\PER_v59_heavy_dynesty_mag.save"
# filename = r"C:\Users\maxiv\Documents\UWO\Papers\Besian\PER_v59_heavy_dynesty_mag_log.save"
# filename = r"C:\Users\maxiv\Documents\UWO\Papers\Besian\PER_v59_heavy_dynesty_lum_gaus.save"
# filename = r"C:\Users\maxiv\Documents\UWO\Papers\Besian\PER_v59_heavy_dynesty_mag_gaus.save"
# filename = r"C:\Users\maxiv\Documents\UWO\Papers\Besian\PER_v59_heavy_dynesty_lum_gaus_log.save"
# filename = r"C:\Users\maxiv\Documents\UWO\Papers\Besian\PER_v59_heavy_dynesty_mag_gaus_log.save"
# filename =r"C:\Users\maxiv\Documents\UWO\Papers\Besian\PER_v59_heavy_dynesty_lum_gauss_rho_log_new.save"
# filename =r"C:\Users\maxiv\Documents\UWO\Papers\Besian\PER_v59_heavy_dynesty_lum_new.save"
# filename =r"C:\Users\maxiv\Documents\UWO\Papers\Besian\PER_v59_heavy_dynesty_lum_new_weight.save"
filename = r"C:\Users\maxiv\Documents\UWO\Papers\Besian\PER_v59_heavy_dynesty_onlylum.save"
filename = r"C:\Users\maxiv\Documents\UWO\Papers\Besian\PER_v59_heavy_dynesty_onlylum_log.save"
# filename = r"C:\Users\maxiv\Documents\UWO\Papers\Besian\PER_v59_heavy_dynesty_onlylag_log.save"
filename = r"C:\Users\maxiv\Documents\UWO\Papers\Besian\PER_v59_heavy_dynesty_multi.save"
filename = r"C:\Users\maxiv\Documents\UWO\Papers\Besian\PER_v59_heavy_dynesty_multi_log.save"
filename = r"C:\Users\maxiv\Documents\UWO\Papers\Besian\PER_v59_heavy_dynesty_multi0corr.save"
# filename = r"C:\Users\maxiv\Documents\UWO\Papers\Besian\PER_v59_heavy_dynesty_multi0corr_log.save"
# filename = r"C:\Users\maxiv\Documents\UWO\Papers\Besian\PER_v59_heavy_dynesty_multi0corr_log_N1000.save"
# filename = r"C:\Users\maxiv\Documents\UWO\Papers\Besian\PER_v59_heavy_dynesty_multi0corr_log_N5000.save"
# take from the file name the characters after dynesty
file_img_sub = filename.split('dynesty')[1]
file_img_sub = file_img_sub.split('.')[0]

print('file_img_sub:', file_img_sub)

# with dynesty.pool.Pool(n_core, log_likelihood, prior, logl_args=(metsim_obj, var_names, 10), ptform_args=(bounds_np,)) as pool:
# sampler = dynesty.NestedSampler(log_likelihood, prior, ndim)

dsampler = dynesty.DynamicNestedSampler.restore(filename)


# res = dsampler.results

# # look inside the results
# for key in res.keys():
#     print(key)
print('sol')

sim_num = -1
best_guess = dsampler.results.samples[sim_num]

# check if any of the best_guess values are negative is so do 10**value
for i in range(len(best_guess)):
    if best_guess[i] < 0:
        best_guess[i] = 10**best_guess[i]
    if i==3 and best_guess[i] < 10:
        # print('rho:', best_guess[i])
        best_guess[i] = 10**best_guess[i]
        # print('rho:', best_guess[i])

print('num of samples:', len(dsampler.results.samples))
print('Best fit')
# write the best fit variable names and then the best guess values
for i in range(len(best_guess)):
    print(var_names[i],':\t', best_guess[i])

print()
# print every res value separated by a coma
for i in range(len(best_guess)):
    if i == len(best_guess) - 1:
        print(best_guess[i])
    else:
        print(best_guess[i], end=', ')
print()

# print('log likelihood:', dsampler.results.logl[sim_num])

# rho_samples =dsampler.results.samples[:,-1]

# # Compute the posterior mean and standard deviation
# rho_mean = np.mean(rho_samples)
# rho_std = np.std(rho_samples)

# print(f"Posterior Mean of ρ: {rho_mean}")
# print(f"Posterior Standard Deviation of ρ: {rho_std}")

# # Decide based on the spread
# if rho_std < 0.05:  # If variance is small, fixing ρ is reasonable
#     print("ρ is well-constrained; consider fixing it.")
# elif rho_std < 0.1:  # Moderate spread, reducing prior range is a good idea
#     print("ρ is somewhat constrained; consider reducing the prior range.")
# else:
#     print("ρ has a wide posterior; keep it as a free parameter.")

obs_lc_intensity = 935*(10 ** (metsim_obj.absolute_magnitudes/(-2.5))) # const.P_0m
obs_lc_intensity_sigma = 935*(10 ** (0.1/(-2.5))) # 10% error

simulation_results = run_simulation(best_guess, metsim_obj, var_names)
const_nominal, _ = loadConstants(path_and_file_MetSim)
const_nominal.dens_co = np.array(const_nominal.dens_co)
dens_co=metsim_obj.const.dens_co
const_nominal.dens_co = dens_co
const_nominal.fragmentation_show_individual_lcs = True
frag_main, results_list, wake_results = runSimulation(const_nominal, compute_wake=False)
simulation_MetSim_object = SimulationResults(const_nominal, frag_main, results_list, wake_results)


# interpolate to make sure they are the same length and discard points after height starts increasing if it does at any point metsim_obj.traj.observations[0].model_ht
simulated_lc_intensity = np.interp(metsim_obj.leading_frag_height_arr, 
                                    np.flip(simulation_results.leading_frag_height_arr), 
                                    np.flip(simulation_results.luminosity_arr))

simulated_lc_mag = np.interp(metsim_obj.leading_frag_height_arr, 
                                    np.flip(simulation_results.leading_frag_height_arr), 
                                    np.flip(simulation_results.abs_magnitude))

# find index_abs_heigh_sim_start base on the first height that is smaller than the first height of the observations
index_abs_heigh_sim_start = np.where(simulation_results.leading_frag_height_arr <= metsim_obj.leading_frag_height_arr[0])[0][0] 

time_sim = simulation_results.time_arr-simulation_results.time_arr[index_abs_heigh_sim_start]
len_sim = simulation_results.leading_frag_length_arr-simulation_results.leading_frag_length_arr[index_abs_heigh_sim_start]

len_sim = np.interp(metsim_obj.leading_frag_height_arr, 
                                    np.flip(simulation_results.leading_frag_height_arr), 
                                    np.flip(len_sim))

time_sim = np.interp(metsim_obj.leading_frag_height_arr, 
                                    np.flip(simulation_results.leading_frag_height_arr), 
                                    np.flip(time_sim))

lag_sim = len_sim - (metsim_obj.v_init * time_sim)

# # simulated_lc_len - simulated_lc_len[0] # to make it start at 0
lag_sim = lag_sim - lag_sim[0]
# lag_sim = simulated_lag

# lag_sim = simulated_lc_len - (metsim_obj.v_init * metsim_obj.time)


# No Noise
mag_sim_no_noise = np.interp(metsim_obj.leading_frag_height_arr,
                                    np.flip(simulation_MetSim_object.leading_frag_height_arr),
                                    np.flip(simulation_MetSim_object.abs_magnitude))

simulated_lc_intensity_no_noise = np.interp(metsim_obj.leading_frag_height_arr,
                                    np.flip(simulation_MetSim_object.leading_frag_height_arr),
                                    np.flip(simulation_MetSim_object.luminosity_arr))                              

# F = 935 * 10**(-mag_sim_no_noise/2.5)
noise_3SNR = simulated_lc_intensity_no_noise[0]/3 # np.min(simulated_lc_intensity_no_noise)/3

simulated_lc_noise = simulated_lc_intensity_no_noise + np.random.normal(loc=0.0, scale=noise_3SNR, size=len(simulated_lc_intensity_no_noise))

# print('noise_3SNR:', noise_3SNR)
# print('noise:', simulated_lc_noise)

# optional: get noisy magnitudes back
mag_noisy = -2.5*np.log10(simulated_lc_noise/935)

lag_sim_no_noise = simulation_MetSim_object.leading_frag_length_arr - (metsim_obj.v_init * simulation_MetSim_object.time_arr )

simulated_lag_no_noise = np.interp(metsim_obj.leading_frag_height_arr, 
                                    np.flip(simulation_MetSim_object.leading_frag_height_arr), 
                                    np.flip(lag_sim_no_noise))

# simulated_lc_len - simulated_lc_len[0] # to make it start at 0
lag_sim_no_noise = simulated_lag_no_noise - simulated_lag_no_noise[0]

log_likelihood_lum = - 0.5/(2.754133642710149**2) * np.nansum((metsim_obj.luminosity_arr - simulated_lc_intensity) ** 2)  # add the error
log_likelihood_lag = - 0.5/(40**2) * np.nansum((metsim_obj.lag - lag_sim) ** 2)  # add the error
# log_likelihood_mag = - 1/(0.1**2) * np.nansum((metsim_obj.absolute_magnitudes - mag_sim_no_noise) ** 2)  # add the error

log_likelihood_lum = - 0.5* np.log(2*np.pi*2.754133642710149**2) + log_likelihood_lum
log_likelihood_lag = - 0.5* np.log(2*np.pi*40**2) + log_likelihood_lag

log_likelihood_lum = np.nansum(-0.5 * np.log(2*np.pi*2.754133642710149**2) - 0.5 / (2.754133642710149**2) * (metsim_obj.luminosity_arr - simulated_lc_intensity) ** 2)
log_likelihood_lag = np.nansum(-0.5 * np.log(2*np.pi*40**2) - 0.5 / (40**2) * (metsim_obj.lag - lag_sim) ** 2)

# log_likelihood_mag_norm = log_likelihood_mag / len(metsim_obj.lag) # / 18.757609257597053
# log_likelihood_lag_norm = log_likelihood_lag / len(metsim_obj.lag) # / 15.781542381940243
# log_likelihood_lum_norm = log_likelihood_lum / len(metsim_obj.lag) # / 15.781542381940243
# log_likelihood_lum = log_likelihood_mag / 18.757609257597053
# log_likelihood_lag = log_likelihood_lag / 15.781542381940243

# Compute difference only where values are not NaN
mask_lag = ~np.isnan(lag_sim)  # Mask for valid values (non-NaN)
mask_mag = ~np.isnan(simulated_lc_mag)  # Mask for valid values (non-NaN)
mask_lum = ~np.isnan(simulated_lc_intensity)  # Mask for valid values (non-NaN)
# combine the 3 masks to get the total mask
mask = mask_lag & mask_mag & mask_lum

# logl_baes_lag_bayes = np.sum(norm.logpdf(metsim_obj.lag[mask], loc=lag_sim_no_noise[mask], scale=40))
# # logl_baes_mag_bayes = np.sum(norm.logpdf(obs_metsim_obj.absolute_magnitudes[mask], loc=simulated_lc_mag[mask], scale=0.1))
# logl_baes_lum_bayes = np.sum(norm.logpdf(metsim_obj.luminosity_arr[mask], loc=simulated_lc_intensity_no_noise[mask], scale=2.754133642710149))

# log_likelihood_bayes = logl_baes_lag_bayes + logl_baes_lum_bayes

logl_baes_lag_bayes_sim = np.sum(norm.logpdf(metsim_obj.lag[mask], loc=lag_sim[mask], scale=40))
# logl_baes_mag_bayes = np.sum(norm.logpdf(obs_metsim_obj.absolute_magnitudes[mask], loc=simulated_lc_mag[mask], scale=0.1))
logl_baes_lum_bayes_sim = np.sum(norm.logpdf(metsim_obj.luminosity_arr[mask], loc=simulated_lc_intensity[mask], scale=2.754133642710149))

log_likelihood_bayes_sim = logl_baes_lag_bayes_sim + logl_baes_lum_bayes_sim

# print(f'logL norm mag: {log_likelihood_mag_norm} logL norm lag: {log_likelihood_lag_norm} sum: {log_likelihood_mag_norm + log_likelihood_lag_norm}')
print(f'chi^2 REAL lum: {log_likelihood_lum} logL lag: {log_likelihood_lag} sum: {log_likelihood_lum + log_likelihood_lag}')
# print(f'logL REAL lum: {logl_baes_lum_bayes} logL lag: {logl_baes_lag_bayes} sum: {log_likelihood_bayes}')
print(f'logL New lum: {logl_baes_lum_bayes_sim} logL lag: {logl_baes_lag_bayes_sim} sum: {log_likelihood_bayes_sim}')
print(f'logL best guess sum: {dsampler.results.logl[sim_num]}')
# log_likelihood_tot_lag_mag = log_likelihood_mag_norm + log_likelihood_lag_norm
log_likelihood_tot_lag_lum = log_likelihood_lum + log_likelihood_lag

# make a subplot of the height and the lag and 8,15 in size
fig, ax = plt.subplots(1,2)
fig.set_size_inches(12,5)
# flatten the array so instead of calling [0,0] and [0,1] you can call [0] and [1]
ax = ax.flatten()
# ax[0].plot(obs_lc_intensity, metsim_obj.leading_frag_height_arr/1000,'go', label='0.1 mag Real')
ax[0].plot(metsim_obj.luminosity_arr, metsim_obj.leading_frag_height_arr/1000,'go', label='lum noise Real')
# # # fill between the along the x axis +- obs_lc_intensity_sigma from simulation_MetSim_object.luminosity_arr
# # ax[0].fill_betweenx(simulation_MetSim_object.leading_frag_height_arr, magminus_o_one,magplus_o_one, color='lightgray', alpha=0.2, label='0.1 mag noise')
ax[0].plot(simulation_MetSim_object.luminosity_arr, simulation_MetSim_object.leading_frag_height_arr/1000,'k', label='No Noise')
ax[0].plot(simulated_lc_intensity, metsim_obj.leading_frag_height_arr/1000, color='darkorange', label='Best Guess')
# ax[0].plot(metsim_obj.absolute_magnitudes, metsim_obj.leading_frag_height_arr/1000,'go', label='0.1 mag Real')
# ax[0].plot(mag_noisy, metsim_obj.leading_frag_height_arr/1000,'ro', label='lum noise Real')
# # fill between the along the x axis +- obs_lc_intensity_sigma from simulation_MetSim_object.luminosity_arr
# # ax[0].fill_betweenx(simulation_MetSim_object.leading_frag_height_arr, magminus_o_one,magplus_o_one, color='lightgray', alpha=0.2, label='0.1 mag noise')
# ax[0].plot(simulation_MetSim_object.abs_magnitude, simulation_MetSim_object.leading_frag_height_arr/1000,'k', label='No Noise')
# ax[0].plot(simulated_lc_mag, metsim_obj.leading_frag_height_arr/1000, color='darkorange', label='Best Guess')
# ax[0].set_title('Height')
ax[0].set_xlabel('Luminosity [J/s]')
# ax[0].set_xlabel('Abs.Mag')
ax[0].set_ylabel('Height [km]')
ax[0].set_ylim([np.min(metsim_obj.leading_frag_height_arr)/1000-1, np.max(metsim_obj.leading_frag_height_arr)/1000+1])
# ax[0].set_xlim([np.min(metsim_obj.absolute_magnitudes)-0.1, np.max(metsim_obj.absolute_magnitudes)+0.1])
# ax[0].set_xlim([np.min(obs_lc_intensity)-0.1, np.max(obs_lc_intensity)+0.1])
# flip x axis
ax[0].invert_xaxis()
# ax[0].set_xlim([np.min(obs_lc_intensity), np.max(obs_lc_intensity)])
ax[0].grid()
# ax[0].legend()
ax[1].plot(metsim_obj.time, metsim_obj.lag,'go', label='Real')
# ax[1].fill_between(metsim_obj.time, lag_sim_no_noise - 40, lag_sim_no_noise + 40, color='lightgray', alpha=0.2, label='40 m noise')
ax[1].plot(metsim_obj.time, lag_sim_no_noise,'k', label='No Noise')
ax[1].plot(time_sim, lag_sim, color='darkorange', label='Best Guess')
# ax[1].set_title('Lag')
ax[1].set_xlabel('Time [s]')
ax[1].set_ylabel('Lag [m]')
ax[1].grid()
ax[1].legend()

# # make a super title and write the dsampler.results.logl[-1] but in e notation with 2 decimal points
# fig.suptitle('loglikelihood sim'+f'{dsampler.results.logl[sim_num]:.2e}'+f' No Noise lum: {log_likelihood_bayes:.2e}')

# tight layout
plt.tight_layout()
# plt.show()
plt.savefig(r'C:\Users\maxiv\Documents\UWO\Papers\Besian\bestsol'+file_img_sub+'.png')
plt.close()


print()

res = dsampler.results

print(res.summary())

print('information gain:', res.information[-1]) # high value means very peaked distribution (good)
# print('log evidence:', res.logz[-1]) # log evidence +/- log evidence error (stoping criteria)
# print('log evidence error:', res.logzerr[-1])
print()

logwt = res.logwt

# Subtract the maximum logwt for numerical stability
logwt_shifted = logwt - np.max(logwt)
weights = np.exp(logwt_shifted)

# Normalize so that sum(weights) = 1
weights /= np.sum(weights)

samples_equal = dynesty.utils.resample_equal(res.samples, weights)

if len(best_guess) == 11:
    labels = [
        r"$v_0 [m/s]$",
        r"$z_c [deg]$",
        r'$m_0$ [kg]',
        r'$\rho$ [kg/m$^3$]',
        r'$\sigma$ [kg/J]',
        r'$h_{e}$ [m]',
        r'$\eta$ [kg/J]',
        r'$s$',
        r'$m_{l}$ [kg]',
        r'$m_{u}$ [kg]',
        r'corr'
    ]
else:
    labels = [
        r"$v_0 [m/s]$",
        r"$z_c [deg]$",
        r'$m_0$ [kg]',
        r'$\rho$ [kg/m$^3$]',
        r'$\sigma$ [kg/J]',
        r'$h_{e}$ [m]',
        r'$\eta$ [kg/J]',
        r'$s$',
        r'$m_{l}$ [kg]',
        r'$m_{u}$ [kg]'
    ]

ndim = len(labels)

truth_values = {
    r"$v_0 [m/s]$": 59841.68504171332,
    r"$z_c [deg]$": 0.694434680449545,
    r'$m_0$ [kg]': 1.378892563434121e-05,
    r'$\rho$ [kg/m$^3$]': 229.92305807400527,
    r'$\sigma$ [kg/J]': 1.4007074382971546e-08,
    r'$h_{e}$ [m]': 117072.99838688939,
    r'$\eta$ [kg/J]': 6.560346848869713e-07,
    r'$s$': 1.6901049357328148,
    r'$m_{l}$ [kg]': 4.464614802830274e-11,
    r'$m_{u}$ [kg]': 6.876069079152311e-10,
    r'corr': 0
}
# make a copy of the truth values
truth_values_plot = truth_values.copy()

best_guess = dsampler.results.samples[sim_num]
for i in range(len(best_guess)):
    if best_guess[i] < 0:
        truth_values_plot[labels[i]] = np.log10(truth_values[labels[i]])
        samples_equal[:, i] = 10**(samples_equal[:, i])
    if i == 3 and best_guess[i] < 10:
        truth_values_plot[labels[i]] = np.log10(truth_values[labels[i]])
        samples_equal[:, i] = 10**(samples_equal[:, i])

# Posterior mean (per dimension)
posterior_mean = np.mean(samples_equal, axis=0)      # shape (ndim,)

# Posterior median (per dimension)
posterior_median = np.median(samples_equal, axis=0)  # shape (ndim,)

# 95% credible intervals (2.5th and 97.5th percentiles)
lower_95 = np.percentile(samples_equal, 2.5, axis=0)   # shape (ndim,)
upper_95 = np.percentile(samples_equal, 97.5, axis=0)  # shape (ndim,)


# Convert to an array in the same order as the parameter labels
truths = np.array([truth_values[label] for label in labels])

truth_plot = np.array([truth_values_plot[label] for label in labels])

# Compare to true theta
bias = posterior_mean - truths
abs_error = np.abs(bias)
rel_error = abs_error / np.abs(truths)

# print("Posterior mean:", posterior_mean)
# print("Posterior median:", posterior_median)
# print("95% CI lower:", lower_95)
# print("95% CI upper:", upper_95)
# print("Bias:", bias)
# print("Absolute error:", abs_error)
# print("Relative error:", rel_error)

# Coverage check
coverage_mask = (truths >= lower_95) & (truths <= upper_95)
print("Coverage mask per dimension:", coverage_mask)
print("Fraction of dimensions covered:", coverage_mask.mean())

# Function to approximate mode using histogram binning
def approximate_mode_1d(samples):
    hist, bin_edges = np.histogram(samples, bins='auto', density=True)
    idx_max = np.argmax(hist)
    return 0.5 * (bin_edges[idx_max] + bin_edges[idx_max + 1])

approx_modes = [approximate_mode_1d(samples_equal[:, d]) for d in range(ndim)]

# Generate LaTeX table
latex_str = r"""\begin{table}[htbp]
    \centering
    \renewcommand{\arraystretch}{1.2} % Increase row height for readability
    \setlength{\tabcolsep}{4pt} % Adjust column spacing
    \resizebox{\textwidth}{!}{ % Resizing table to fit page width
    \begin{tabular}{|l|c|c|c|c|c|c||c|c||c|}
    \hline
    Parameter & 2.5\% & True Value & Mean & Median & Mode & 97.5\% & Abs. Error & Rel. Error & Cover \\
    \hline
"""
# & Mode
# {approx_modes[i]:.4g} &
for i, label in enumerate(labels):
    coverage_val = "\ding{51}" if coverage_mask[i] else "\ding{55}"  # Use checkmark/x for coverage
    latex_str += (f"    {label} & {lower_95[i]:.4g} & {truths[i]:.4g} & {posterior_mean[i]:.4g} "
                  f"& {posterior_median[i]:.4g} & {approx_modes[i]:.4g} & {upper_95[i]:.4g} "
                  f"& {abs_error[i]:.4g} & {rel_error[i]:.4g} & {coverage_val} \\\\\n    \hline\n")

latex_str += r"""
    \end{tabular}}
    \caption{Posterior summary statistics comparing estimated values with the true values. The cover column indicates whether the true value is within the 95\% confidence interval.}
    \label{tab:posterior_summary}
\end{table}"""

# Save to a .tex file
with open("results_table.tex", "w") as f:
    f.write(latex_str)

# Print LaTeX code for quick copy-pasting
print(latex_str)

print()

# 25310it [5:59:39,  1.32s/it, batch: 0 | bound: 10 | nc: 30 | ncall: 395112 | eff(%):  6.326 | loglstar:   -inf < -16256.467 <    inf | logz: -16269.475 +/-  0.049 | dlogz: 15670.753 >  0.010]

print('saving trace plot...')
fig, axes = dyplot.traceplot(res, truths=truth_plot, labels=labels,
                             label_kwargs={"fontsize": 10},  # Reduce axis label size
                             title_kwargs={"fontsize": 10},  # Reduce title font size
                             title_fmt='.2e',  # Scientific notation for titles
                             truth_color='black', show_titles=True,
                             trace_cmap='viridis', connect=True,
                             connect_highlight=range(5))

# Adjust spacing and tick label size
fig.subplots_adjust(hspace=0.5)  # Increase spacing between plots

# save the figure
plt.savefig(r'C:\Users\maxiv\Documents\UWO\Papers\Besian\trace_plot'+file_img_sub+'.png', dpi=300)

# show the trace plot
# plt.show()

print('saving corner plot...')

# Trace Plots
fig, axes = plt.subplots(ndim, ndim, figsize=(35, 15))
axes = axes.reshape((ndim, ndim))  # reshape axes

# Increase spacing between subplots
fg, ax = dyplot.cornerplot(
    res, 
    color='blue', 
    truths=truth_plot,  # Use the defined truth values
    truth_color='black', 
    show_titles=True, 
    max_n_ticks=3, 
    quantiles=None, 
    labels=labels,  # Update axis labels
    label_kwargs={"fontsize": 15},  # Reduce axis label size
    title_kwargs={"fontsize": 10},  # Reduce title font size
    title_fmt='.2e',  # Scientific notation for titles
    fig=(fig, axes[:, :ndim])
)

# Adjust spacing and tick label size
fg.subplots_adjust(wspace=0.3, hspace=0.3)  # Increase spacing between plots

# # Reduce tick size
# for ax_row in ax:
#     for ax_ in ax_row:
#         ax_.tick_params(axis='both', labelsize=6)  # Reduce tick number size

# Apply scientific notation and horizontal tick labels
for ax_row in ax:
    for ax_ in ax_row:
        ax_.tick_params(axis='both', labelsize=10, direction='in')
        
        # # Apply scientific notation to tick labels
        # ax_.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        # ax_.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        # ax_.xaxis.get_major_formatter().set_scientific(True)
        # ax_.yaxis.get_major_formatter().set_scientific(True)
        # ax_.xaxis.get_major_formatter().set_powerlimits((-2, 2))
        # ax_.yaxis.get_major_formatter().set_powerlimits((-2, 2))

        # Set tick labels to be horizontal
        for label in ax_.get_xticklabels():
            label.set_rotation(0)
        for label in ax_.get_yticklabels():
            label.set_rotation(0)

# save the figure
plt.savefig(r'C:\Users\maxiv\Documents\UWO\Papers\Besian\corner_plot'+file_img_sub+'.png', dpi=300)

# plt.show()



#############################################################################

# import numpy as np
# import pandas as pd
# import pickle
# import sys
# import json
# import os
# import copy
# import matplotlib.pyplot as plt
# import dynesty
# from dynesty import plotting as dyplot
# import time
# from matplotlib.ticker import ScalarFormatter
# import scipy
# from scipy.stats import norm
# from scipy.stats import multivariate_normal

# from wmpl.MetSim.GUI import loadConstants, saveConstants,SimulationResults
# from wmpl.MetSim.MetSimErosion import runSimulation, Constants, zenithAngleAtSimulationBegin
# from wmpl.MetSim.ML.GenerateSimulations import generateErosionSim,saveProcessedList,MetParam
# from wmpl.Utils.TrajConversions import J2000_JD, date2JD
# from wmpl.Utils.Math import meanAngle
# from wmpl.Utils.AtmosphereDensity import fitAtmPoly
# from wmpl.Utils.Physics import dynamicPressure

# # print(bounds)
# # custom code to catch timeouts
# import signal

# class TimeoutException(Exception):
#     """Custom exception for timeouts."""
#     pass

# def timeout_handler(signum, frame):
#     raise TimeoutException("Function execution timed out")


# def read_pickle_reduction_file(file_path):


#     with open(file_path, 'rb') as f:
#         traj = pickle.load(f, encoding='latin1')

#     v_avg = traj.v_avg
#     v_init=traj.orbit.v_init
#     obs_data = []
#     # obs_init_vel = []
#     for obs in traj.observations:
#         if obs.station_id == "01G" or obs.station_id == "02G" or obs.station_id == "01F" or obs.station_id == "02F" or obs.station_id == "1G" or obs.station_id == "2G" or obs.station_id == "1F" or obs.station_id == "2F":
#             obs_dict = {
#                 'v_init': obs.v_init, # m/s
#                 'velocities': np.array(obs.velocities), # m/s
#                 # 'velocities': np.array(obs.velocities)[1:], # m/s
#                 'height': np.array(obs.model_ht), # m
#                 # pick all except the first element
#                 # 'height' : np.array(obs.model_ht)[1:],
#                 'absolute_magnitudes': np.array(obs.absolute_magnitudes),
#                 # 'absolute_magnitudes': np.array(obs.absolute_magnitudes)[1:],
#                 'lag': np.array(obs.lag), # m
#                 # 'lag': np.array(obs.lag)[1:],
#                 'length': np.array(obs.state_vect_dist), # m
#                 # 'length': np.array(obs.state_vect_dist)[1:],
#                 'time': np.array(obs.time_data) # s
#                 # 'time': np.array(obs.time_data)[1:]
#                 # 'station_id': obs.station_id
#                 # 'elev_data':  np.array(obs.elev_data)
#             }
            
#             obs_dict['velocities'][0] = obs_dict['v_init']
#             obs_data.append(obs_dict)

#         else:
#             print(obs.station_id,'Station not in the list of stations')
#             continue
    
    
#     # Save distinct values for the two observations
#     obs1, obs2 = obs_data[0], obs_data[1]

#     # # do the average of the two obs_init_vel
#     # v_init_vel = np.mean(obs_init_vel)

#     # save time of each observation
#     obs1_time = np.array(obs1['time'])
#     obs2_time = np.array(obs2['time'])
#     obs1_length = np.array(obs1['length'])
#     obs2_length = np.array(obs2['length'])
#     obs1_height = np.array(obs1['height'])
#     obs2_height = np.array(obs2['height'])
#     obs1_velocities = np.array(obs1['velocities'])
#     obs2_velocities = np.array(obs2['velocities'])
#     obs1_absolute_magnitudes = np.array(obs1['absolute_magnitudes'])
#     obs2_absolute_magnitudes = np.array(obs2['absolute_magnitudes'])
#     obs1_lag = np.array(obs1['lag'])
#     obs2_lag = np.array(obs2['lag'])
    
#     # Combine obs1 and obs2
#     combined_obs = {}
#     for key in ['velocities', 'height', 'absolute_magnitudes', 'lag', 'length', 'time']: #, 'elev_data']:
#         combined_obs[key] = np.concatenate((obs1[key], obs2[key]))

#     # Order the combined observations based on time
#     sorted_indices = np.argsort(combined_obs['time'])
#     for key in ['time', 'velocities', 'height', 'absolute_magnitudes', 'lag', 'length']: #, 'elev_data']:
#         combined_obs[key] = combined_obs[key][sorted_indices]

#     # check if any value is below 10 absolute_magnitudes and print find values below 8 absolute_magnitudes
#     if np.any(combined_obs['absolute_magnitudes'] > 8):
#         print('Found values below 8 absolute magnitudes:', combined_obs['absolute_magnitudes'][combined_obs['absolute_magnitudes'] > 8])
    
#     # delete any values above 10 absolute_magnitudes and delete the corresponding values in the other arrays
#     combined_obs = {key: combined_obs[key][combined_obs['absolute_magnitudes'] < 8] for key in combined_obs.keys()}

#     dens_fit_ht_beg = 180000
#     dens_fit_ht_end = traj.rend_ele - 5000
#     if dens_fit_ht_end < 14000:
#         dens_fit_ht_end = 14000

#     lat_mean = np.mean([traj.rbeg_lat, traj.rend_lat])
#     lon_mean = meanAngle([traj.rbeg_lon, traj.rend_lon])
#     jd_dat=traj.jdt_ref

#     # Fit the polynomail describing the density
#     dens_co = fitAtmPoly(lat_mean, lon_mean, dens_fit_ht_end, dens_fit_ht_beg, jd_dat)

#     Dynamic_pressure_peak_abs_mag=(dynamicPressure(lat_mean, lon_mean, combined_obs['height'][np.argmin(combined_obs['absolute_magnitudes'])], jd_dat, combined_obs['velocities'][np.argmin(combined_obs['absolute_magnitudes'])])) # , gamma=traj.const.gamma
#     const=Constants()
#     zenith_angle=zenithAngleAtSimulationBegin(const.h_init, traj.rbeg_ele, traj.orbit.zc, const.r_earth)

#     # # delete the elev_data from the combined_obs
#     # del combined_obs['elev_data']

#     # add to combined_obs the avg velocity and the peak dynamic pressure and all the physical parameters
#     combined_obs['name'] = file_path    
#     combined_obs['v_init'] = v_init
#     combined_obs['dens_co'] = dens_co
#     combined_obs['obs1_time'] = obs1_time
#     combined_obs['obs2_time'] = obs2_time
#     combined_obs['obs1_length'] = obs1_length   
#     combined_obs['obs2_length'] = obs2_length
#     combined_obs['obs1_height'] = obs1_height
#     combined_obs['obs2_height'] = obs2_height
#     combined_obs['obs1_velocities'] = obs1_velocities
#     combined_obs['obs2_velocities'] = obs2_velocities
#     combined_obs['obs1_absolute_magnitudes'] = obs1_absolute_magnitudes
#     combined_obs['obs2_absolute_magnitudes'] = obs2_absolute_magnitudes
#     combined_obs['obs1_lag'] = obs1_lag
#     combined_obs['obs2_lag'] = obs2_lag
#     combined_obs['v_avg'] = v_avg
#     combined_obs['Dynamic_pressure_peak_abs_mag'] = Dynamic_pressure_peak_abs_mag
#     combined_obs['zenith_angle'] = zenith_angle*180/np.pi

#     return combined_obs



# def run_simulation(parameter_guess, real_event, var_names):
#     '''
#         path_and_file = must be a json file generated file from the generate_simulationsm function or from Metsim file
#     '''

#     # Load the nominal simulation parameters
#     const_nominal = Constants()
#     const_nominal.dens_co = np.array(const_nominal.dens_co)

#     dens_co=real_event.const.dens_co

#     ### Calculate atmosphere density coeffs (down to the bottom observed height, limit to 15 km) ###

#     # Assign the density coefficients
#     const_nominal.dens_co = dens_co

#     # Turn on plotting of LCs of individual fragments 
#     const_nominal.fragmentation_show_individual_lcs = True

#     # var_cost=['v_init','zenith_angle','m_init','rho','sigma','erosion_height_start','erosion_coeff','erosion_mass_index','erosion_mass_min','erosion_mass_max']
#     # for loop for the var_cost that also give a number from 0 to the length of the var_cost
#     for i, var in enumerate(var_names):
#         const_nominal.__dict__[var] = parameter_guess[i]

#     const_nominal.P_0m = 935

#     const_nominal.disruption_on = False

#     const_nominal.lum_eff_type = 5

#     # # Minimum height [m]
#     # const_nominal.h_kill = 60000

#     # # Initial meteoroid height [m]
#     # const_nominal.h_init = 180000

#     try:
#         # Run the simulation
#         frag_main, results_list, wake_results = runSimulation(const_nominal, compute_wake=False)
#         simulation_MetSim_object = SimulationResults(const_nominal, frag_main, results_list, wake_results)
#     except ZeroDivisionError as e:
#         print(f"Error during simulation: {e}")
#         const_nominal = Constants()
#         # Run the simulation
#         frag_main, results_list, wake_results = runSimulation(const_nominal, compute_wake=False)
#         simulation_MetSim_object = SimulationResults(const_nominal, frag_main, results_list, wake_results)

#     return simulation_MetSim_object






# ht_sampled = [
#     116536.30672989249,
#     115110.05178727282,
#     113684.16779440964,
#     113083.52176646683,
#     112258.70769182382,
#     111658.26293157891,
#     110833.7560201216,
#     110233.56447703335,
#     109409.46328620492,
#     108809.62866081292,
#     107986.16022881713,
#     107386.98063665455,
#     106565.21195410144,
#     105968.56935228375,
#     105150.10100517314,
#     104551.73106847175,
#     103734.61524641037,
#     103140.58540786116,
#     102326.43004031561,
#     101735.07664182257,
#     100925.22671750661,
#     100337.56278479846,
#     99533.73715875004,
#     98951.32781515278,
#     98156.27352170387,
#     97581.7215316846,
#     96800.25882608269,
#     96238.45679599466,
#     95480.62157380355,
#     94240.66042891663
# ]

# mag_sampled = [
#     5.007660533177523,
#     3.3265873118503335,
#     2.5464444953563428,
#     2.145252609826516,
#     1.7679457288795726,
#     1.741187657219813,
#     1.40484226116799,
#     1.1238999999255155,
#     1.054036156142602,
#     0.7899871355675105,
#     0.7519377204984318,
#     0.6329126310337716,
#     0.4518418596153948,
#     0.3490237440687807,
#     0.2684538603082568,
#     0.3345305534532859,
#     0.24680534587066463,
#     0.31191759237233396,
#     0.1699651543419376,
#     0.28149072143285303,
#     0.3970816649970808,
#     0.28046356530150807,
#     0.3974088113018212,
#     0.6907812280796398,
#     0.8369301875849462,
#     1.1838672037832072,
#     1.6649283756822753,
#     2.1730332105068406,
#     3.0583341692267147,
#     6.308512445827362
# ]

# time_sampled=[
#     0.0,
#     0.03125,
#     0.0625,
#     0.07566659137611853,
#     0.09375,
#     0.10691659137611853,
#     0.125,
#     0.13816659137611853,
#     0.15625,
#     0.16941659137611853,
#     0.1875,
#     0.20066659137611853,
#     0.21875,
#     0.23191659137611853,
#     0.25,
#     0.26316659137611853,
#     0.28125,
#     0.29441659137611853,
#     0.3125,
#     0.32566659137611853,
#     0.34375,
#     0.35691659137611853,
#     0.375,
#     0.38816659137611853,
#     0.40625,
#     0.41941659137611853,
#     0.4375,
#     0.45066659137611853,
#     0.46875,
#     0.5
# ]

# lag_sampled = [
#     0.0,
#     -4.48762471,
#     20.22486645,
#     47.87077673,
#     -8.36099409,
#     -8.69188961,
#     48.39385267,
#     22.03765375,
#     -18.2415974,
#     13.06091513,
#     -20.53832741,
#     -22.21994643,
#     -2.78280529,
#     -74.2648876,
#     -73.42361563,
#     -41.15836802,
#     -63.42730629,
#     -28.25484624,
#     -78.82771967,
#     -105.06174599,
#     -30.32777112,
#     -98.71462315,
#     -113.29739184,
#     -181.65335499,
#     -188.03966759,
#     -197.33292507,
#     -287.942161,
#     -284.30179231,
#     -392.49153447,
#     -587.79484216
# ]
    
# dens_coef = [
#             34.067090334166345,
#             -2257.137635650747,
#             56522.21296907875,
#             -734951.556330054,
#             5081875.733489659,
#             -17804424.227462955,
#             24913101.34607125
#         ]

# simulated_lc_noise = [
#     7.76564283,   # 0
#     45.9003461,   # 1
#     95.51941976,  # 2
#     130.7400714,  # 3
#     171.5345507,  # 4
#     209.23247285, # 5
#     267.17677614, # 6
#     314.15710598, # 7
#     375.82018905, # 8
#     433.60141327, # 9
#     499.82748122, # 10
#     543.0895856,  # 11
#     595.75361642, # 12
#     630.60615462, # 13
#     676.25862748, # 14
#     707.84084628, # 15
#     738.33267711, # 16
#     754.09393909, # 17
#     769.10148559, # 18
#     759.80631885, # 19
#     725.65497342, # 20
#     685.73731365, # 21
#     608.66967324, # 22
#     525.00972133, # 23
#     409.14024966, # 24
#     314.77655079, # 25
#     204.28789954, # 26
#     137.17950206, # 27
#     58.10412326,  # 28
#     8.08138952    # 29
# ]

# v_inti = 59811.87059507625
# # make it so can be readed like this metsim_obj.absolute_magnitudes and metsim_obj.leading_frag_height_arr
# metsim_obj = type('metsim_obj', (object,), {})()
# metsim_obj.absolute_magnitudes = np.array(mag_sampled)
# metsim_obj.leading_frag_height_arr = np.array(ht_sampled)
# metsim_obj.time = np.array(time_sampled)
# metsim_obj.lag = np.array(lag_sampled)
# metsim_obj.v_init = v_inti
# metsim_obj.luminosity_arr = np.array(simulated_lc_noise)
# metsim_obj.const = type('const', (object,), {})()
# metsim_obj.const.dens_co = np.array(dens_coef)

# noise_lc = 2.754133642710149

# # Data to populate the dataframe
# # bounds = {
# #     'v_init': [59841.68503,59841.68505],
# #     'zenith_angle': [0.69443467,0.69443469],
# #     'm_init': [0.88e-5,1.88e-5],
# #     'rho': [100,1000],
# #     'sigma': [0.008/1e6, 0.03/1e6],
# #     'erosion_height_start': [116000, 118000],
# #     'erosion_coeff': [0.0, 1/1e6],
# #     'erosion_mass_index': [1.5, 2.5],
# #     'erosion_mass_min': [5e-12, 1e-10],
# #     'erosion_mass_max': [1e-10, 5e-8]
# # }

# # Data to populate the dataframe
# bounds = {
#     'v_init': [59841.68503,59841.68505],
#     'zenith_angle': [0.69443467,0.69443469],
#     'm_init': [1.3e-5,1.88e-5],
#     'rho': [100,1000],
#     'sigma': [0.008/1e6, 0.03/1e6],
#     'erosion_height_start': [116000, 118000],
#     'erosion_coeff': [0.0, 1/1e6],
#     'erosion_mass_index': [1.5, 2.5],
#     'erosion_mass_min': [5e-12, 1e-10],
#     'erosion_mass_max': [1e-10, 5e-8]
# }

# # Create the dataframe
# bounds = pd.DataFrame(bounds)

# print(len(bounds))
# print(len(bounds.columns))

# var_names = bounds.columns # get the variable names

# # # create a array with np.max(np.size(bounds)) number of rows and 0 on the first column and 1 on the second column
# # cube = np.array([np.zeros(len(bounds.columns)), np.ones(len(bounds.columns))]).T

# # print(cube)
# # print(bounds)

# # new_cube = prior(cube, bounds)
# # print(new_cube)

# ndim = len(bounds.columns)

# path_and_file_MetSim = '/home/mvovk/Documents/json_test/Simulations_PER_v59_heavy/PER_v59_heavy_sim_fit_latest.json'

# # guess_var = [59841.68504, 0.69443468, 1.38e-5, 550, 0.019/1e6, 117000, 0.5/1e6, 2.0, 7.5e-12, 3e-8]
# guess_var = [59841.68504171332, 0.694434680449545, 1.378892563434121e-05, 229.92305807400527, 1.4007074382971546e-08, 117072.99838688939, 6.560346848869713e-07, 1.6901049357328148, 4.464614802830274e-11, 6.876069079152311e-10]
# print(type(guess_var))
# print(guess_var)
# # loglike_val=log_likelihood(guess_var, metsim_obj, var_names, timeout=10)

# # print(loglike_val)

# start_time = time.time()

# ### LUM ONLY

# # LOG LIKELIHOOD = function to determine how far off the guess is from the real event
# def log_likelihood_onlylum(guess_var, obs_metsim_obj=metsim_obj, var_names=var_names, timeout=10):
#     """
#     similar to the get_lc_cost_function function but with positive instead of negative log-likelihood
#     """
#     # # # observed LC intensity, this doesn't change

#     # Set timeout handler
#     signal.signal(signal.SIGALRM, timeout_handler)
#     signal.alarm(timeout)  # Start the timer for timeout
#     # get simulated LC intensity onthe object
#     try: 
#         # simulation_results = sim_lc(guess_phys_var, obs_metsim_obj)
#         simulation_results = run_simulation(guess_var, obs_metsim_obj, var_names)
#     except TimeoutException:
#         print('timeout')
#         return -np.inf  # immediately return -np.inf if times out
#     finally:
#         signal.alarm(0)  # Cancel alarm


#     simulated_lc_mag = np.interp(obs_metsim_obj.leading_frag_height_arr, 
#                                        np.flip(simulation_results.leading_frag_height_arr), 
#                                        np.flip(simulation_results.abs_magnitude))

#     # interpolate to make sure they are the same length and discard points after height starts increasing if it does at any point obs_metsim_obj.traj.observations[0].model_ht
#     simulated_lc_intensity = np.interp(obs_metsim_obj.leading_frag_height_arr, 
#                                        np.flip(simulation_results.leading_frag_height_arr), 
#                                        np.flip(simulation_results.luminosity_arr))

#     lag_sim = simulation_results.leading_frag_length_arr - (obs_metsim_obj.v_init * simulation_results.time_arr )

#     simulated_lag = np.interp(obs_metsim_obj.leading_frag_height_arr, 
#                                        np.flip(simulation_results.leading_frag_height_arr), 
#                                        np.flip(lag_sim))
    
#     # simulated_lc_len - simulated_lc_len[0] # to make it start at 0
#     lag_sim = simulated_lag - simulated_lag[0]

#     # lag_sim = simulated_lc_len - (obs_metsim_obj.v_init * obs_metsim_obj.time)
    
#     # log_likelihood = (-1/2 * np.nansum((obs_lc_intensity - simulated_lc_intensity) ** 2))#/1.e24  # scale this
#     log_likelihood_lum = - 1/(2.754133642710149**2) * np.nansum((obs_metsim_obj.luminosity_arr - simulated_lc_intensity) ** 2)  # add the error
#     log_likelihood_lag = - 1/(40**2) * np.nansum((obs_metsim_obj.lag - lag_sim) ** 2)  # add the error
#     # log_likelihood_mag = - 1/(0.1**2) * np.nansum((obs_metsim_obj.absolute_magnitudes - simulated_lc_mag) ** 2) / len(obs_metsim_obj.lag) # add the error


#     # log_likelihood_mag_norm = log_likelihood_mag / len(obs_metsim_obj.lag)  # / 18.757609257597053
#     log_likelihood_lum_norm = log_likelihood_lum / len(obs_metsim_obj.lag)  # 0.03680101768116091
#     log_likelihood_lag_norm = log_likelihood_lag / len(obs_metsim_obj.lag)  # / 15.781542381940243

#     # log_likelihood_lag_norm = log_likelihood_lag / len(obs_metsim_obj.lag) # / 15.781542381940243
#     # log_likelihood_lum_norm = log_likelihood_lum / len(obs_metsim_obj.lag) # / 15.781542381940243

#     # print(f'logL norm mag: {log_likelihood_lum_norm} logL norm lag: {log_likelihood_lag_norm} sum: {log_likelihood_lum_norm + log_likelihood_lag_norm}')
#     log_likelihood_tot = log_likelihood_lum_norm + log_likelihood_lum_norm

#     # positive log lieklihood unlike get_lc_cost_function
#     return log_likelihood_tot

# # LOG LIKELIHOOD = function to determine how far off the guess is from the real event
# def log_likelihood_onlylum_log(guess_var, obs_metsim_obj=metsim_obj, var_names=var_names, timeout=10):
#     """
#     similar to the get_lc_cost_function function but with positive instead of negative log-likelihood
#     """
#     # # # observed LC intensity, this doesn't change
#     for i, var in enumerate(var_names):
#         if var == 'erosion_coeff' or var == 'erosion_mass_min' or var == 'erosion_mass_max' or var == 'rho':
#             guess_var[i] = 10**(guess_var[i])

#     # Set timeout handler
#     signal.signal(signal.SIGALRM, timeout_handler)
#     signal.alarm(timeout)  # Start the timer for timeout
#     # get simulated LC intensity onthe object
#     try: 
#         # simulation_results = sim_lc(guess_phys_var, obs_metsim_obj)
#         simulation_results = run_simulation(guess_var, obs_metsim_obj, var_names)
#     except TimeoutException:
#         print('timeout')
#         return -np.inf  # immediately return -np.inf if times out
#     finally:
#         signal.alarm(0)  # Cancel alarm


#     simulated_lc_mag = np.interp(obs_metsim_obj.leading_frag_height_arr, 
#                                        np.flip(simulation_results.leading_frag_height_arr), 
#                                        np.flip(simulation_results.abs_magnitude))

#     # interpolate to make sure they are the same length and discard points after height starts increasing if it does at any point obs_metsim_obj.traj.observations[0].model_ht
#     simulated_lc_intensity = np.interp(obs_metsim_obj.leading_frag_height_arr, 
#                                        np.flip(simulation_results.leading_frag_height_arr), 
#                                        np.flip(simulation_results.luminosity_arr))

#     lag_sim = simulation_results.leading_frag_length_arr - (obs_metsim_obj.v_init * simulation_results.time_arr )

#     simulated_lag = np.interp(obs_metsim_obj.leading_frag_height_arr, 
#                                        np.flip(simulation_results.leading_frag_height_arr), 
#                                        np.flip(lag_sim))
    
#     # simulated_lc_len - simulated_lc_len[0] # to make it start at 0
#     lag_sim = simulated_lag - simulated_lag[0]

#     # lag_sim = simulated_lc_len - (obs_metsim_obj.v_init * obs_metsim_obj.time)
    
#     # log_likelihood = (-1/2 * np.nansum((obs_lc_intensity - simulated_lc_intensity) ** 2))#/1.e24  # scale this
#     log_likelihood_lum = - 1/(2.754133642710149**2) * np.nansum((obs_metsim_obj.luminosity_arr - simulated_lc_intensity) ** 2)  # add the error
#     # log_likelihood_lag = - 1/(40**2) * np.nansum((obs_metsim_obj.lag - lag_sim) ** 2)  # add the error
#     # log_likelihood_mag = - 1/(0.1**2) * np.nansum((obs_metsim_obj.absolute_magnitudes - simulated_lc_mag) ** 2) / len(obs_metsim_obj.lag) # add the error


#     # log_likelihood_mag_norm = log_likelihood_mag / len(obs_metsim_obj.lag)  # / 18.757609257597053
#     log_likelihood_lum_norm = log_likelihood_lum / len(obs_metsim_obj.lag)  # 0.03680101768116091
#     # log_likelihood_lag_norm = log_likelihood_lag / len(obs_metsim_obj.lag)  # / 15.781542381940243

#     # log_likelihood_lag_norm = log_likelihood_lag / len(obs_metsim_obj.lag) # / 15.781542381940243
#     # log_likelihood_lum_norm = log_likelihood_lum / len(obs_metsim_obj.lag) # / 15.781542381940243

#     # print(f'logL norm mag: {log_likelihood_lum_norm} logL norm lag: {log_likelihood_lag_norm} sum: {log_likelihood_lum_norm + log_likelihood_lag_norm}')
#     log_likelihood_tot = log_likelihood_lum_norm # + log_likelihood_lum_norm

#     # positive log lieklihood unlike get_lc_cost_function
#     return log_likelihood_tot


# ### Multi-likelihood

# # LOG LIKELIHOOD = function to determine how far off the guess is from the real event
# def log_likelihood_multi(guess_var, obs_metsim_obj=metsim_obj, var_names=var_names, timeout=10):
#     """
#     similar to the get_lc_cost_function function but with positive instead of negative log-likelihood
#     """
#     # # # observed LC intensity, this doesn't change

#     # Set timeout handler
#     signal.signal(signal.SIGALRM, timeout_handler)
#     signal.alarm(timeout)  # Start the timer for timeout

#     # get simulated LC intensity onthe object
#     try: 
#         # simulation_results = sim_lc(guess_phys_var, obs_metsim_obj)
#         simulation_results = run_simulation(guess_var[:-1], obs_metsim_obj, var_names[:-1])
#     except TimeoutException:
#         print('timeout')
#         return -np.inf  # immediately return -np.inf if times out
#     finally:
#         signal.alarm(0)  # Cancel alarm


#     simulated_lc_mag = np.interp(obs_metsim_obj.leading_frag_height_arr, 
#                                        np.flip(simulation_results.leading_frag_height_arr), 
#                                        np.flip(simulation_results.abs_magnitude))

#     # interpolate to make sure they are the same length and discard points after height starts increasing if it does at any point obs_metsim_obj.traj.observations[0].model_ht
#     simulated_lc_intensity = np.interp(obs_metsim_obj.leading_frag_height_arr, 
#                                        np.flip(simulation_results.leading_frag_height_arr), 
#                                        np.flip(simulation_results.luminosity_arr))

#     lag_sim = simulation_results.leading_frag_length_arr - (obs_metsim_obj.v_init * simulation_results.time_arr )

#     simulated_lag = np.interp(obs_metsim_obj.leading_frag_height_arr, 
#                                        np.flip(simulation_results.leading_frag_height_arr), 
#                                        np.flip(lag_sim))
    
#     # simulated_lc_len - simulated_lc_len[0] # to make it start at 0
#     lag_sim = simulated_lag - simulated_lag[0]

#     # Compute difference only where values are not NaN
#     mask_lag = ~np.isnan(lag_sim)  # Mask for valid values (non-NaN)
#     mask_mag = ~np.isnan(simulated_lc_mag)  # Mask for valid values (non-NaN)
#     mask_lum = ~np.isnan(simulated_lc_intensity)  # Mask for valid values (non-NaN)
#     # combine the 3 masks to get the total mask
#     mask = mask_lag & mask_mag & mask_lum

#     # if all values are NaN, return -inf
#     if not np.any(mask):
#         return -np.inf

#     # Uncertainties (assumed known)
#     sigma_lag = 40  # Replace with actual uncertainty
#     sigma_lum = 2.754133642710149  # Replace with actual uncertainty
#     sigma_mag = 0.1  # Replace with actual uncertainty

#     # Correlation coefficient (to be inferred)
#     corr_coef = guess_var[-1] # Assume last parameter is the correlation coefficient

#     # Build the covariance matrix
#     covariance_matrix = np.array([
#         [sigma_lag**2, corr_coef * sigma_lag * sigma_lum],
#         [corr_coef * sigma_lag * sigma_lum, sigma_lum**2]
#     ])

#     # Compute residuals
#     residuals_lag = obs_metsim_obj.lag[mask] - lag_sim[mask]
#     residuals_lum = obs_metsim_obj.luminosity_arr[mask] - simulated_lc_intensity[mask]
#     residuals = np.vstack((residuals_lag, residuals_lum)).T

#     # Compute multivariate log-likelihood
#     multivariate_logl= np.sum(multivariate_normal.logpdf(residuals, mean=[0, 0], cov=covariance_matrix))

#     # positive log lieklihood unlike get_lc_cost_function
#     return multivariate_logl

# # LOG LIKELIHOOD = function to determine how far off the guess is from the real event
# def log_likelihood_multi_log(guess_var, obs_metsim_obj=metsim_obj, var_names=var_names, timeout=10):
#     """
#     similar to the get_lc_cost_function function but with positive instead of negative log-likelihood
#     """
#     # # # observed LC intensity, this doesn't change
#     for i, var in enumerate(var_names):
#         if var == 'erosion_coeff' or var == 'erosion_mass_min' or var == 'erosion_mass_max' or var == 'rho':
#             guess_var[i] = 10**(guess_var[i])

#     # Set timeout handler
#     signal.signal(signal.SIGALRM, timeout_handler)
#     signal.alarm(timeout)  # Start the timer for timeout
#     # get simulated LC intensity onthe object
#     try: 
#         # simulation_results = sim_lc(guess_phys_var, obs_metsim_obj)
#         simulation_results = run_simulation(guess_var[:-1], obs_metsim_obj, var_names[:-1])
#     except TimeoutException:
#         print('timeout')
#         return -np.inf  # immediately return -np.inf if times out
#     finally:
#         signal.alarm(0)  # Cancel alarm


#     simulated_lc_mag = np.interp(obs_metsim_obj.leading_frag_height_arr, 
#                                        np.flip(simulation_results.leading_frag_height_arr), 
#                                        np.flip(simulation_results.abs_magnitude))

#     # interpolate to make sure they are the same length and discard points after height starts increasing if it does at any point obs_metsim_obj.traj.observations[0].model_ht
#     simulated_lc_intensity = np.interp(obs_metsim_obj.leading_frag_height_arr, 
#                                        np.flip(simulation_results.leading_frag_height_arr), 
#                                        np.flip(simulation_results.luminosity_arr))

#     lag_sim = simulation_results.leading_frag_length_arr - (obs_metsim_obj.v_init * simulation_results.time_arr )

#     simulated_lag = np.interp(obs_metsim_obj.leading_frag_height_arr, 
#                                        np.flip(simulation_results.leading_frag_height_arr), 
#                                        np.flip(lag_sim))
    
#     # simulated_lc_len - simulated_lc_len[0] # to make it start at 0
#     lag_sim = simulated_lag - simulated_lag[0]

#     # Compute difference only where values are not NaN
#     mask_lag = ~np.isnan(lag_sim)  # Mask for valid values (non-NaN)
#     mask_mag = ~np.isnan(simulated_lc_mag)  # Mask for valid values (non-NaN)
#     mask_lum = ~np.isnan(simulated_lc_intensity)  # Mask for valid values (non-NaN)
#     # combine the 3 masks to get the total mask
#     mask = mask_lag & mask_mag & mask_lum

#     # if all values are NaN, return -inf
#     if not np.any(mask):
#         return -np.inf

#     # Uncertainties (assumed known)
#     sigma_lag = 40  # Replace with actual uncertainty
#     sigma_lum = 2.754133642710149  # Replace with actual uncertainty
#     sigma_mag = 0.1  # Replace with actual uncertainty

#     # Correlation coefficient (to be inferred)
#     corr_coef = guess_var[-1]  # Assume last parameter is the correlation coefficient

#     # Build the covariance matrix
#     covariance_matrix = np.array([
#         [sigma_lag**2, corr_coef * sigma_lag * sigma_lum],
#         [corr_coef * sigma_lag * sigma_lum, sigma_lum**2]
#     ])

#     # Compute residuals
#     residuals_lag = obs_metsim_obj.lag[mask] - lag_sim[mask]
#     residuals_lum = obs_metsim_obj.luminosity_arr[mask] - simulated_lc_intensity[mask]
#     residuals = np.vstack((residuals_lag, residuals_lum)).T

#     # Compute multivariate log-likelihood
#     multivariate_logl= np.sum(multivariate_normal.logpdf(residuals, mean=[0, 0], cov=covariance_matrix))

#     # positive log lieklihood unlike get_lc_cost_function
#     return multivariate_logl


# ### Multi-likelihood 0 correlation

# # LOG LIKELIHOOD = function to determine how far off the guess is from the real event
# def log_likelihood_multi0corr(guess_var, obs_metsim_obj=metsim_obj, var_names=var_names, timeout=10):
#     """
#     similar to the get_lc_cost_function function but with positive instead of negative log-likelihood
#     """
#     # # # observed LC intensity, this doesn't change

#     # Set timeout handler
#     signal.signal(signal.SIGALRM, timeout_handler)
#     signal.alarm(timeout)  # Start the timer for timeout

#     # get simulated LC intensity onthe object
#     try: 
#         # simulation_results = sim_lc(guess_phys_var, obs_metsim_obj)
#         simulation_results = run_simulation(guess_var, obs_metsim_obj, var_names)
#     except TimeoutException:
#         print('timeout')
#         return -np.inf  # immediately return -np.inf if times out
#     finally:
#         signal.alarm(0)  # Cancel alarm


#     simulated_lc_mag = np.interp(obs_metsim_obj.leading_frag_height_arr, 
#                                        np.flip(simulation_results.leading_frag_height_arr), 
#                                        np.flip(simulation_results.abs_magnitude))

#     # interpolate to make sure they are the same length and discard points after height starts increasing if it does at any point obs_metsim_obj.traj.observations[0].model_ht
#     simulated_lc_intensity = np.interp(obs_metsim_obj.leading_frag_height_arr, 
#                                        np.flip(simulation_results.leading_frag_height_arr), 
#                                        np.flip(simulation_results.luminosity_arr))

#     lag_sim = simulation_results.leading_frag_length_arr - (obs_metsim_obj.v_init * simulation_results.time_arr )

#     simulated_lag = np.interp(obs_metsim_obj.leading_frag_height_arr, 
#                                        np.flip(simulation_results.leading_frag_height_arr), 
#                                        np.flip(lag_sim))
    
#     # simulated_lc_len - simulated_lc_len[0] # to make it start at 0
#     lag_sim = simulated_lag - simulated_lag[0]

#     log_likelihood_lum = - 0.5/(2.754133642710149**2) * np.nansum((obs_metsim_obj.luminosity_arr - simulated_lc_intensity) ** 2)  # add the error
#     log_likelihood_lag = - 0.5/(40**2) * np.nansum((obs_metsim_obj.lag - lag_sim) ** 2)  # add the error

#     multivariate_logl = log_likelihood_lag + log_likelihood_lum

#     # # Compute difference only where values are not NaN
#     # mask_lag = ~np.isnan(lag_sim)  # Mask for valid values (non-NaN)
#     # mask_mag = ~np.isnan(simulated_lc_mag)  # Mask for valid values (non-NaN)
#     # mask_lum = ~np.isnan(simulated_lc_intensity)  # Mask for valid values (non-NaN)
#     # # combine the 3 masks to get the total mask
#     # mask = mask_lag & mask_mag & mask_lum

#     # # if all values are NaN, return -inf
#     # if not np.any(mask):
#     #     return -np.inf

#     # # Define the log-likelihood for dataset 1 (Lag vs. Time)
#     # logl_baes_lag = np.sum(norm.logpdf(obs_metsim_obj.lag[mask], loc=lag_sim[mask], scale=40))
#     # # logl_baes_mag = np.sum(norm.logpdf(obs_metsim_obj.absolute_magnitudes[mask], loc=simulated_lc_mag[mask], scale=0.1))
#     # logl_baes_lum = np.sum(norm.logpdf(obs_metsim_obj.luminosity_arr[mask], loc=simulated_lc_intensity[mask], scale=2.754133642710149))

#     # multivariate_logl = logl_baes_lag + logl_baes_lum

#     return multivariate_logl

# # LOG LIKELIHOOD = function to determine how far off the guess is from the real event
# def log_likelihood_multi0corr_log(guess_var, obs_metsim_obj=metsim_obj, var_names=var_names, timeout=10):
#     """
#     similar to the get_lc_cost_function function but with positive instead of negative log-likelihood
#     """
#     # # # observed LC intensity, this doesn't change
#     for i, var in enumerate(var_names):
#         if var == 'erosion_coeff' or var == 'erosion_mass_min' or var == 'erosion_mass_max' or var == 'rho':
#             guess_var[i] = 10**(guess_var[i])

#     # Set timeout handler
#     signal.signal(signal.SIGALRM, timeout_handler)
#     signal.alarm(timeout)  # Start the timer for timeout
#     # get simulated LC intensity onthe object
#     try: 
#         # simulation_results = sim_lc(guess_phys_var, obs_metsim_obj)
#         simulation_results = run_simulation(guess_var, obs_metsim_obj, var_names)
#     except TimeoutException:
#         print('timeout')
#         return -np.inf  # immediately return -np.inf if times out
#     finally:
#         signal.alarm(0)  # Cancel alarm


#     simulated_lc_mag = np.interp(obs_metsim_obj.leading_frag_height_arr, 
#                                        np.flip(simulation_results.leading_frag_height_arr), 
#                                        np.flip(simulation_results.abs_magnitude))

#     # interpolate to make sure they are the same length and discard points after height starts increasing if it does at any point obs_metsim_obj.traj.observations[0].model_ht
#     simulated_lc_intensity = np.interp(obs_metsim_obj.leading_frag_height_arr, 
#                                        np.flip(simulation_results.leading_frag_height_arr), 
#                                        np.flip(simulation_results.luminosity_arr))

#     lag_sim = simulation_results.leading_frag_length_arr - (obs_metsim_obj.v_init * simulation_results.time_arr )

#     simulated_lag = np.interp(obs_metsim_obj.leading_frag_height_arr, 
#                                        np.flip(simulation_results.leading_frag_height_arr), 
#                                        np.flip(lag_sim))
    
#     # simulated_lc_len - simulated_lc_len[0] # to make it start at 0
#     lag_sim = simulated_lag - simulated_lag[0]

#     log_likelihood_lum = - 0.5/(2.754133642710149**2) * np.nansum((obs_metsim_obj.luminosity_arr - simulated_lc_intensity) ** 2)  # add the error
#     log_likelihood_lag = - 0.5/(40**2) * np.nansum((obs_metsim_obj.lag - lag_sim) ** 2)  # add the error

#     multivariate_logl = log_likelihood_lag + log_likelihood_lum

#     # # Compute difference only where values are not NaN
#     # mask_lag = ~np.isnan(lag_sim)  # Mask for valid values (non-NaN)
#     # mask_mag = ~np.isnan(simulated_lc_mag)  # Mask for valid values (non-NaN)
#     # mask_lum = ~np.isnan(simulated_lc_intensity)  # Mask for valid values (non-NaN)
#     # # combine the 3 masks to get the total mask
#     # mask = mask_lag & mask_mag & mask_lum

#     # # if all values are NaN, return -inf
#     # if not np.any(mask):
#     #     return -np.inf

#     # # Define the log-likelihood for dataset 1 (Lag vs. Time)
#     # logl_baes_lag = np.sum(norm.logpdf(obs_metsim_obj.lag[mask], loc=lag_sim[mask], scale=40))
#     # # logl_baes_mag = np.sum(norm.logpdf(obs_metsim_obj.absolute_magnitudes[mask], loc=simulated_lc_mag[mask], scale=0.1))
#     # logl_baes_lum = np.sum(norm.logpdf(obs_metsim_obj.luminosity_arr[mask], loc=simulated_lc_intensity[mask], scale=2.754133642710149))

#     # multivariate_logl = logl_baes_lag + logl_baes_lum

#     return multivariate_logl





# #### PRIORS ####

# def prior_log(cube):
#     """
#     Transform the unit cube to a uniform prior
#     """
#     x = np.array(cube)  # Copy u to avoid modifying it directly

#     # Define the parameter bounds as a list of (min, max) tuples
#     bounds = [
#         (58841.68503, 60841.68505),   # v_init
#         (0.39443467, 0.99443469),     # zenith_angle
#         (0.3e-5, 2.3e-5),             # m_init
#         (10, 2000),                  # rho
#         (0.008 / 1e6, 0.03 / 1e6),    # sigma
#         (110000, 120000),             # erosion_height_start
#         (np.log10(1 / 1e12), np.log10(1 / 1e5)),               # erosion_coeff
#         (1, 2.5),                   # erosion_mass_index
#         (np.log10(5e-12), np.log10(1e-10)),               # erosion_mass_min
#         (np.log10(1e-10), np.log10(5e-8))                 # erosion_mass_max
#     ]

#     # Transform each u[i] to the corresponding parameter space using the bounds
#     for i, (low, high) in enumerate(bounds):
#         x[i] = cube[i] * (high - low) + low  # Scale and shift

#     return x


# def prior(cube):
#     """
#     Transform the unit cube to a uniform prior
#     """
#     x = np.array(cube)  # Copy u to avoid modifying it directly

#     # Define the parameter bounds as a list of (min, max) tuples
#     bounds = [
#         (58841.68503, 60841.68505),   # v_init
#         (0.59443467, 0.79443469),     # zenith_angle
#         (0.3e-5, 2.3e-5),             # m_init
#         (100, 1000),                  # rho
#         (0.008 / 1e6, 0.03 / 1e6),    # sigma
#         (116000, 118000),             # erosion_height_start
#         (0.0, 1 / 1e6),               # erosion_coeff
#         (1.5, 2.5),                   # erosion_mass_index
#         (5e-12, 1e-10),               # erosion_mass_min
#         (1e-10, 5e-8)                 # erosion_mass_max
#     ]

#     # Transform each u[i] to the corresponding parameter space using the bounds
#     for i, (low, high) in enumerate(bounds):
#         x[i] = cube[i] * (high - low) + low  # Scale and shift

#     return x


# def prior_gaus_lum_rho_log(cube):
#     """
#     Transform the unit cube to a uniform prior
#     """
#     x = np.array(cube)  # Copy u to avoid modifying it directly

#     # Define the parameter bounds as a list of (min, max) tuples
#     bounds = [
#         (58841.68503, 60841.68505),   # v_init
#         (0.59443467, 0.79443469),     # zenith_angle
#         (0.3e-5, 2.3e-5),             # m_init
#         (np.log10(10), np.log10(8000)),# rho
#         (0.008 / 1e6, 0.03 / 1e6),    # sigma
#         (116000, 118000),             # erosion_height_start
#         (0.0, 1 / 1e6),               # erosion_coeff
#         (1.5, 2.5),                   # erosion_mass_index
#         (5e-12, 1e-10),               # erosion_mass_min
#         (1e-10, 5e-8)                 # erosion_mass_max
#     ]

#     # Transform each u[i] to the corresponding parameter space using the bounds
#     for i, (low, high) in enumerate(bounds):
#         # Mean hyper-prior 59841.68504171332, 0.694434680449545
#         if i == 0:
#             mu, sigma = 59841.68504171332, 500  # mean, standard deviation
#             x[i] = scipy.stats.norm.ppf(cube[i], loc=mu, scale=sigma)
#         elif i == 1:
#             mu, sigma = 0.694434680449545, 0.01  # mean, standard deviation
#             x[i] = scipy.stats.norm.ppf(cube[i], loc=mu, scale=sigma)
#         else:
#             x[i] = cube[i] * (high - low) + low  # Scale and shift

#     return x


# def prior_gaus(cube):
#     """
#     Transform the unit cube to a uniform prior
#     """
#     x = np.array(cube)  # Copy u to avoid modifying it directly

#     # Define the parameter bounds as a list of (min, max) tuples
#     bounds = [
#         (58841.68503, 60841.68505),   # v_init
#         (0.59443467, 0.79443469),     # zenith_angle
#         (0.3e-5, 2.3e-5),             # m_init
#         (100, 1000),                  # rho
#         (0.008 / 1e6, 0.03 / 1e6),    # sigma
#         (116000, 118000),             # erosion_height_start
#         (0.0, 1 / 1e6),               # erosion_coeff
#         (1.5, 2.5),                   # erosion_mass_index
#         (5e-12, 1e-10),               # erosion_mass_min
#         (1e-10, 5e-8)                 # erosion_mass_max
#     ]

#     # Transform each u[i] to the corresponding parameter space using the bounds
#     for i, (low, high) in enumerate(bounds):
#         # Mean hyper-prior 59841.68504171332, 0.694434680449545
#         if i == 0:
#             mu, sigma = 59841.68504171332, 500  # mean, standard deviation
#             x[i] = scipy.stats.norm.ppf(cube[i], loc=mu, scale=sigma)
#         elif i == 1:
#             mu, sigma = 0.694434680449545, 0.01  # mean, standard deviation
#             x[i] = scipy.stats.norm.ppf(cube[i], loc=mu, scale=sigma)
#         else:
#             x[i] = cube[i] * (high - low) + low  # Scale and shift

#     return x


# def prior_gaus_log(cube):
#     """
#     Transform the unit cube to a uniform prior
#     """
#     x = np.array(cube)  # Copy u to avoid modifying it directly

#     # Define the parameter bounds as a list of (min, max) tuples
#     bounds = [
#         (58841.68503, 60841.68505),   # v_init
#         (0.59443467, 0.79443469),     # zenith_angle
#         (0.3e-5, 2.3e-5),             # m_init
#         (10, 2000),                  # rho
#         (0.008 / 1e6, 0.03 / 1e6),    # sigma
#         (110000, 120000),             # erosion_height_start
#         (np.log10(1 / 1e12), np.log10(1 / 1e5)),               # erosion_coeff
#         (1, 2.5),                   # erosion_mass_index
#         (np.log10(5e-12), np.log10(1e-10)),               # erosion_mass_min
#         (np.log10(1e-10), np.log10(5e-8))                 # erosion_mass_max
#     ]

#     # Transform each u[i] to the corresponding parameter space using the bounds
#     for i, (low, high) in enumerate(bounds):
#         # Mean hyper-prior 59841.68504171332, 0.694434680449545
#         if i == 0:
#             mu, sigma = 59841.68504171332, 500  # mean, standard deviation
#             x[i] = scipy.stats.norm.ppf(cube[i], loc=mu, scale=sigma)
#         elif i == 1:
#             mu, sigma = 0.694434680449545, 0.01  # mean, standard deviation
#             x[i] = scipy.stats.norm.ppf(cube[i], loc=mu, scale=sigma)
#         else:
#             x[i] = cube[i] * (high - low) + low  # Scale and shift

#     return x


# def prior_gaus_log_new(cube):
#     """
#     Transform the unit cube to a uniform prior
#     """
#     x = np.array(cube)  # Copy u to avoid modifying it directly

#     # Define the parameter bounds as a list of (min, max) tuples
#     bounds = [
#         (58841.68503, 60841.68505),   # v_init
#         (0.59443467, 0.79443469),     # zenith_angle
#         (0.3e-5, 2.3e-5),             # m_init
#         (np.log10(10), np.log10(8000)), # rho
#         (0.008 / 1e6, 0.03 / 1e6),    # sigma
#         (116000, 120000),             # erosion_height_start
#         (np.log10(1 / 1e12), np.log10(1 / 1e5)), # erosion_coeff
#         (1, 2.5),                   # erosion_mass_index
#         (np.log10(5e-12), np.log10(1e-10)), # erosion_mass_min
#         (np.log10(1e-10), np.log10(5e-8))   # erosion_mass_max
#     ]

#     # Transform each u[i] to the corresponding parameter space using the bounds
#     for i, (low, high) in enumerate(bounds):
#         # Mean hyper-prior 59841.68504171332, 0.694434680449545
#         if i == 0:
#             mu, sigma = 59841.68504171332, 500  # mean, standard deviation
#             x[i] = scipy.stats.norm.ppf(cube[i], loc=mu, scale=sigma)
#         elif i == 1:
#             mu, sigma = 0.694434680449545, 0.01  # mean, standard deviation
#             x[i] = scipy.stats.norm.ppf(cube[i], loc=mu, scale=sigma)
#         else:
#             x[i] = cube[i] * (high - low) + low  # Scale and shift

#     return x


# def prior_gaus_multiv(cube):
#     """
#     Transform the unit cube to a uniform prior
#     """
#     x = np.array(cube)  # Copy u to avoid modifying it directly

#     # Define the parameter bounds as a list of (min, max) tuples
#     bounds = [
#         (58841.68503, 60841.68505),   # v_init
#         (0.59443467, 0.79443469),     # zenith_angle
#         (0.3e-5, 2.3e-5),             # m_init
#         (100, 1000),                  # rho
#         (0.008 / 1e6, 0.03 / 1e6),    # sigma
#         (116000, 118000),             # erosion_height_start
#         (0.0, 1 / 1e6),               # erosion_coeff
#         (1.5, 2.5),                   # erosion_mass_index
#         (5e-12, 1e-10),               # erosion_mass_min
#         (1e-10, 5e-8),                # erosion_mass_max
#         (-1, 1)                       # datasets correlation
#     ]

#     # Transform each u[i] to the corresponding parameter space using the bounds
#     for i, (low, high) in enumerate(bounds):
#         # Mean hyper-prior 59841.68504171332, 0.694434680449545
#         if i == 0:
#             mu, sigma = 59841.68504171332, 500  # mean, standard deviation
#             x[i] = scipy.stats.norm.ppf(cube[i], loc=mu, scale=sigma)
#         elif i == 1:
#             mu, sigma = 0.694434680449545, 0.01  # mean, standard deviation
#             x[i] = scipy.stats.norm.ppf(cube[i], loc=mu, scale=sigma)
#         else:
#             x[i] = cube[i] * (high - low) + low  # Scale and shift

#     return x


# def prior_gaus_multiv_log(cube):
#     """
#     Transform the unit cube to a uniform prior
#     """
#     x = np.array(cube)  # Copy u to avoid modifying it directly

#     # Define the parameter bounds as a list of (min, max) tuples
#     bounds = [
#         (58841.68503, 60841.68505),   # v_init
#         (0.59443467, 0.79443469),     # zenith_angle
#         (0.3e-5, 2.3e-5),             # m_init
#         (np.log10(10), np.log10(8000)),# rho
#         (0.008 / 1e6, 0.03 / 1e6),    # sigma
#         (116000, 120000),             # erosion_height_start
#         (np.log10(1 / 1e12), np.log10(1 / 1e5)), # erosion_coeff
#         (1, 2.5),                   # erosion_mass_index
#         (np.log10(5e-12), np.log10(1e-10)), # erosion_mass_min
#         (np.log10(1e-10), np.log10(5e-8)),  # erosion_mass_max
#         (-1, 1)                      # datasets correlation
#     ]

#     # Transform each u[i] to the corresponding parameter space using the bounds
#     for i, (low, high) in enumerate(bounds):
#         # Mean hyper-prior 59841.68504171332, 0.694434680449545
#         if i == 0:
#             mu, sigma = 59841.68504171332, 500  # mean, standard deviation
#             x[i] = scipy.stats.norm.ppf(cube[i], loc=mu, scale=sigma)
#         elif i == 1:
#             mu, sigma = 0.694434680449545, 0.01  # mean, standard deviation
#             x[i] = scipy.stats.norm.ppf(cube[i], loc=mu, scale=sigma)
#         else:
#             x[i] = cube[i] * (high - low) + low  # Scale and shift

#     return x



# filename = '/home/mvovk/Documents/PER_v59_heavy_dynesty_lum_log.save'  # CHANGE THIS

# n_core=os.cpu_count()

# # with dynesty.pool.Pool(n_core, log_likelihood_lum_log, prior_log, logl_args=(metsim_obj, var_names, 10)) as pool:
# # # sampler = dynesty.NestedSampler(log_likelihood, prior, ndim)

# # # # with dynesty.pool.Pool(n_core, log_likelihood, prior) as pool:
# #     # NEW RUN
# #     dsampler = dynesty.DynamicNestedSampler(pool.loglike, pool.prior_transform, ndim, pool = pool)
# #     dsampler.run_nested(print_progress=True, checkpoint_file=filename)

# #     # # # RESUME:
# #     # dsampler = dynesty.DynamicNestedSampler.restore(filename, pool = pool)
# #     # dsampler.run_nested(resume=True, print_progress=True, checkpoint_file=filename)

# # print('done')

# # Data to populate the dataframe
# bounds = {
#     'v_init': [59841.68503,59841.68505],
#     'zenith_angle': [0.69443467,0.69443469],
#     'm_init': [1.3e-5,1.88e-5],
#     'rho': [100,1000],
#     'sigma': [0.008/1e6, 0.03/1e6],
#     'erosion_height_start': [116000, 118000],
#     'erosion_coeff': [0.0, 1/1e6],
#     'erosion_mass_index': [1.5, 2.5],
#     'erosion_mass_min': [5e-12, 1e-10],
#     'erosion_mass_max': [1e-10, 5e-8]
# }

# # Create the dataframe
# bounds = pd.DataFrame(bounds)

# var_names = bounds.columns # get the variable names

# ndim = len(bounds.columns)

# filename = '/home/mvovk/Documents/PER_v59_heavy_dynesty_multi0corr_chi.save'  # CHANGE THIS

# with dynesty.pool.Pool(n_core-1, log_likelihood_multi0corr, prior_gaus, logl_args=(metsim_obj, var_names, 10)) as pool:
#     # sampler = dynesty.NestedSampler(log_likelihood, prior, ndim)

#     # # # with dynesty.pool.Pool(n_core, log_likelihood, prior) as pool:
#     # NEW RUN
#     dsampler = dynesty.DynamicNestedSampler(pool.loglike, pool.prior_transform, ndim, pool = pool)
#     dsampler.run_nested(print_progress=True, checkpoint_file=filename)

#     # # # RESUME:
#     # dsampler = dynesty.DynamicNestedSampler.restore(filename, pool = pool)
#     # dsampler.run_nested(resume=True, print_progress=True, checkpoint_file=filename)

# print('done')

# filename = '/home/mvovk/Documents/PER_v59_heavy_dynesty_multi0corr_log_chi.save'  # CHANGE THIS

# with dynesty.pool.Pool(n_core-1, log_likelihood_multi0corr_log, prior_gaus_log_new, logl_args=(metsim_obj, var_names, 10)) as pool:
#     # sampler = dynesty.NestedSampler(log_likelihood, prior, ndim)

#     # # with dynesty.pool.Pool(n_core, log_likelihood, prior) as pool:
#     # NEW RUN
#     dsampler = dynesty.DynamicNestedSampler(pool.loglike, pool.prior_transform, ndim, pool = pool)
#     dsampler.run_nested(print_progress=True, checkpoint_file=filename)

#     # # # RESUME:
#     # dsampler = dynesty.DynamicNestedSampler.restore(filename, pool = pool)
#     # dsampler.run_nested(resume=True, print_progress=True, checkpoint_file=filename)

# print('done')





