import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
import os
from wmpl.MetSim.GUI import FragmentationEntry, SimulationResults, loadConstants, saveConstants, loadWakeFile, plotWakeOverview, WakeContainter
from wmpl.MetSim.MetSimErosion import energyReceivedBeforeErosion
from wmpl.MetSim.MetSimErosion import Constants, runSimulation, zenithAngleAtSimulationBegin
from wmpl.Utils.Physics import calcMass, dynamicPressure, calcRadiatedEnergy

# the sript generate a 
constjson_bestfit = Constants()
constjson_bestfit.dens_co = np.array(constjson_bestfit.dens_co)

for sigma in [0.1/1e6, 0.01/1e6, 0.001/1e6]:
    constjson_bestfit.__dict__['sigma'] = sigma

    constjson_bestfit.__dict__['P_0m'] = 840 # 935 # 840
    constjson_bestfit.__dict__['disruption_on'] = False
    constjson_bestfit.__dict__['h_kill'] = 1
    constjson_bestfit.__dict__['v_kill'] = 2500 # final velocity of 2.5 km/s, which is the lowest velocity for which we have data, so we can be sure to capture the full wake
    constjson_bestfit.__dict__['h_init'] = 70000
    constjson_bestfit.__dict__['v_init'] = 8000 # m_init
    constjson_bestfit.__dict__['m_init'] = 1e-6
    constjson_bestfit.__dict__['rho'] = 5700
    constjson_bestfit.__dict__['zenith_angle'] = np.radians(45)
    constjson_bestfit.__dict__['erosion_height_start'] = 1
    # constjson_bestfit.__dict__['sigma'] = 0.1/1e6 # 0.001/1e6 # 0.001/1e6,0.1/1e6 
    constjson_bestfit.__dict__['lum_eff_type'] = 0
    constjson_bestfit.__dict__['lum_eff'] = 1

    frag_main, results_list, wake_results = runSimulation(constjson_bestfit, compute_wake=False)
    best_guess_obj_plot_global = SimulationResults(constjson_bestfit, frag_main, results_list, wake_results)

    photom_mass10 = calcMass(np.array(best_guess_obj_plot_global.time_arr[:-1]), np.array(best_guess_obj_plot_global.abs_magnitude[:-1]), constjson_bestfit.__dict__['v_init'], tau=10/100, P_0m=constjson_bestfit.__dict__['P_0m'])
    # photom_mass10 = photom_mass10*10
    photom_mass01 = calcMass(np.array(best_guess_obj_plot_global.time_arr[:-1]), np.array(best_guess_obj_plot_global.abs_magnitude[:-1]), constjson_bestfit.__dict__['v_init'], tau=0.1/100, P_0m=constjson_bestfit.__dict__['P_0m'])
    # photom_mass01 = photom_mass01/10
    photom_mass001 = calcMass(np.array(best_guess_obj_plot_global.time_arr[:-1]), np.array(best_guess_obj_plot_global.abs_magnitude[:-1]), constjson_bestfit.__dict__['v_init'], tau=0.01/100, P_0m=constjson_bestfit.__dict__['P_0m'])
    # photom_mass001 = photom_mass001/100
    print(f"photometric mass with tau=10%: {photom_mass10:.2g} kg, photometric mass with tau=1%: {best_guess_obj_plot_global.mass_total_active_arr[0]:.2g} kg, photometric mass with tau=0.1%: {photom_mass01:.2g} kg, photometric mass with tau=0.01%: {photom_mass001:.2g} kg")


    # create a plot that shows how the mass [:-1] varies against height [:-1]
    plt.subplots(1, 3, figsize=(15, 5))

    for tau, photom_mass in zip([10, 1, 0.1, 0.01], [photom_mass10, best_guess_obj_plot_global.mass_total_active_arr[0], photom_mass01, photom_mass001]):
        print(f"tau: {tau}%, photometric mass: {photom_mass:.2g}")

        # make a copy of constjson_bestfit and change the mass intial to the photometric mass with tau=0.1% and run the simulation again to see how it compares to the best guess object
        constjson_bestfit_copy = copy.deepcopy(constjson_bestfit)
        constjson_bestfit_copy.__dict__['m_init'] = photom_mass
        constjson_bestfit_copy.__dict__['lum_eff'] = tau

        frag_main_copy, results_list_copy, wake_results_copy = runSimulation(constjson_bestfit_copy, compute_wake=False)
        best_guess_obj_plot_kill = SimulationResults(constjson_bestfit_copy, frag_main_copy, results_list_copy, wake_results_copy)

        final_mass = best_guess_obj_plot_kill.mass_total_active_arr[-2]
        final_mass_percent = (final_mass/best_guess_obj_plot_kill.mass_total_active_arr[0])*100
        size_final = ((final_mass/(4/3*np.pi*best_guess_obj_plot_kill.const.rho))**(1/3))*2 * 1e6 # diameter in microns
        print(f"Stop at 2.5 km/s: initial mass: {best_guess_obj_plot_kill.mass_total_active_arr[0]:.2g} kg, final mass: {final_mass:.2g} kg, final mass in percent {(final_mass_percent):.2g} %, final size: {size_final:.2f} microns")

        plt.subplot(1, 3, 2)
        # plt.plot(best_guess_obj_plot_kill.leading_frag_height_arr[:-1], best_guess_obj_plot_kill.mass_total_active_arr[:-1], label='Best Fit')
        plt.plot(best_guess_obj_plot_kill.mass_total_active_arr[:-1],best_guess_obj_plot_kill.leading_frag_height_arr[:-1]/1000)#, label=f'tau={tau}%')
        # if the final velocity is higher than 1.71 then the stop was because the mass was below 10^-14 kg 
        # if best_guess_obj_plot_kill.leading_frag_vel_arr[-2] > 2510:
        #     # plt.text(final_mass, best_guess_obj_plot_kill.leading_frag_height_arr[-2]/1000, 'Stop at mass limit', color='red', fontsize=8, ha='center', va='bottom')
        #     plt.plot(final_mass, best_guess_obj_plot_kill.leading_frag_height_arr[-2]/1000, 'rx')#, label=f"Fully ablated (stop at mass limit)")
        # # else:
        #     # put dot where it reac 1.7 km/s i.e. at the end of the plot
        #     plt.plot(final_mass, best_guess_obj_plot_kill.leading_frag_height_arr[-2]/1000, 'ro', label=f"Stop at {best_guess_obj_plot_kill.leading_frag_vel_arr[-2]/1000:.2f} km/s")


        # display the leggen left up
        # plt.legend(loc='upper left', fontsize=10)
        # add the label
        # plt.ylabel('Height [km]')
        plt.xlabel('Mass [kg]')
        # take the current y axis value
        y_axis = plt.gca().get_ylim()
        # make x log
        plt.xscale('log')
        # plt.title(f"Mass vs Height for {file_name}")
        plt.grid(True, linestyle='--', color='lightgray')

        plt.subplot(1, 3, 1)
        # put the absolute magnitude vs height plot with the obs_data and the best_guess_obj_plot
        # plt.plot(obs_data.absolute_magnitudes, obs_data.height_lum/1000, color='k', label='Observed data')
        plt.plot(best_guess_obj_plot_kill.abs_magnitude[:-1], best_guess_obj_plot_kill.leading_frag_height_arr[:-1]/1000, label=f'tau={tau}% final size: {size_final:.2f} μm')
        plt.xlabel('Absolute Magnitude [-]')
        plt.ylabel('Height [km]')
        plt.grid(True, linestyle='--', color='lightgray')
        if photom_mass == photom_mass01:
            # invert the x axis
            plt.gca().invert_xaxis()
        plt.legend(loc='upper left', fontsize=10)
        plt.subplot(1, 3, 3)
        # put the velocity data
        plt.plot(best_guess_obj_plot_kill.leading_frag_vel_arr[:-1]/1000, best_guess_obj_plot_kill.leading_frag_height_arr[:-1]/1000) #, label=f'tau={tau}%')

        # if best_guess_obj_plot_kill.leading_frag_vel_arr[-2] < 2510:
        #     #put dot where it reac 1.7 km/s i.e. at the end of the plot
        #     plt.plot(best_guess_obj_plot_kill.leading_frag_vel_arr[-2]/1000, best_guess_obj_plot_kill.leading_frag_height_arr[-2]/1000, 'ro')#, label=f"Stop at {best_guess_obj_plot_kill.leading_frag_vel_arr[-2]/1000:.2f} km/s")

        # plt.legend(loc='upper left', fontsize=10)
        plt.grid(True, linestyle='--', color='lightgray')
        plt.xlabel('Velocity [km/s]')
        # plt.ylabel('Height [km]')

        # as a super title put the value of $\sigma$
    plt.suptitle(f"sigma={sigma*1e6:.3f} kg/MJ", fontsize=12)

    plt.tight_layout()
    # # show the plot
    # plt.show()

    # the current folder the code is in
    output_folder = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(output_folder +os.sep+ f'final_mass_sigma{sigma*1e6:.3f}.png', dpi=300, bbox_inches='tight')
    plt.close()

