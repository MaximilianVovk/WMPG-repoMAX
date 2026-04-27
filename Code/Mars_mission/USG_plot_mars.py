import numpy as np
import sys
import os
import multiprocessing
import dynesty
import signal
import matplotlib.pyplot as plt
import copy
# Makes WMPL directory visible, change this depending on where the WMPL library is located relative to your script
# sys.path.append('../../')
# import wmpl.MetSim.GUI as gui
from wmpl.MetSim.GUI import FragmentationEntry, SimulationResults, loadConstants, loadUSGInputFile, FragmentationContainer, MetSimGUI, saveConstants, loadWakeFile, plotWakeOverview, WakeContainter
# import wmpl.MetSim.MetSimErosion as erosion
from wmpl.MetSim.MetSimErosion import Constants, runSimulation, energyReceivedBeforeErosion, zenithAngleAtSimulationBegin
from wmpl.Utils.AtmosphereDensity import fitAtmPoly, atmDensPoly, getAtmDensity
# import from Mars_AtmDens.py
from Mars_AtmDens import fitAtmPoly_mars

#### EVENT
event_path = r'C:\Users\maxiv\WMPG-repoMAX\Code\Mars_mission\usg_input_jul_2010' # file path to the USG light curve, WITHOUT the .txt extension
#### DYNESTY SAVE:
save_to = r'C:\Users\maxiv\WMPG-repoMAX\Code\Mars_mission'  # file path to the dynesty save to save to/continue running from
new_run = True  # boolean, set True if this is a new run (i.e. the dynesty save does not already exist) and set False if continuing an existing run from the specified dynesty save

# CONSTANTS (change if necessary):
############################################################################
P0m = 3030  # bolometric power of zero magnitude meteor in watts (3030 for silicon bandpass CNEOS at 6000 K blackbody)
obs_sigma = 1.2e10 # Observational uncertainty in the data (sigma), in units of W/ster
# Prior bounds for the log-scatter:
log_scatter_bounds = (22., 28.)  # log-(W/ster) 22, 28 for 2010-07-06
# Prior bounds for the initial mass
m_init_bounds = (1.e2, 1.e8)  # kg  1.e2, 1.e8
# Prior bounds for the fragmentation parameters
mass_pct_bounds = (0., 100.)  # % of initial mass
er_coeff_bounds = (0.01 * 1.e-6, 10.0 * 1.e-6)  # s^2/km^2 0.01, 10
grain_min_bounds = (1.e-5, 1.e2)  # kg 1.e-5, 1.e2
grain_max_bounds = (1.e-4, 1.e3)  # kg. Note that the bounds for each run are actually set to max(grain_min, 1.e-4) since the grain maximum cannot be less than the grain minimum. 1.e-4, 1.e3
max_height_diff = 3000.  # maximum vertical distance up or down to be used for the bounds of each manually set fragmentation point 5000
# dynesty:
n_threads = multiprocessing.cpu_count() - 1  # number of threads to use for the dynamic nested sampling
timeout = 15  # time out a metsim run after this many seconds
nlive = 500  # number of live points to use (500 is dynesty's default)
sample = 'rslice'  # sampling strategy to use (see https://dynesty.readthedocs.io/en/stable/quickstart.html#sampling-options)


# SUN constants
MU_SUN = 1.32712440018e11 # km^3/s^2
# Mars constants
MU_MARS   = 4.282837e4    # km^3/s^2
A_MARS_AU = 1.523679      # mean heliocentric distance (assume circular)
R_MARS    = 3389.5        # km
# Earth constants
MU_EARTH = 3.986004418e5  # km^3/s^2
A_EARTH_AU = 1.00000261    # mean heliocentric distance (assume circular)
R_EARTH = 6371.0          # km

AU_KM = 149_597_870.7     # km

G0_mars = 3.75  # m/s^2

############################################################################


def earth_to_mars_empirical(v_earth=None, zenith_earth=None, sigma_level=1.0):
    """
    Empirical Earth-to-Mars estimator based on the fitted Mars-log results.

    Inputs
    ------
    v_earth : float or None
        Corrected Earth Vinf in km/s.

    zenith_earth : float or None
        Earth zenith angle in degrees.

    sigma_level : float
        Uncertainty multiplier.
        Use 1.0 for ~1 sigma.
        Use 2.0 for a wider rough range.

    Returns
    -------
    dict
        Estimated Mars velocity and/or Mars zenith angle.
    """

    if v_earth is None and zenith_earth is None:
        raise ValueError("Give at least v_earth or zenith_earth.")

    result = {}

    # ------------------------------------------------------------
    # Mars velocity estimate
    # ------------------------------------------------------------
    if v_earth is not None:

        if zenith_earth is not None:
            # Best empirical fit when both Earth velocity and zenith angle are known:
            #
            # V_mars = a + b * V_earth + c * zenith_earth
            #
            a_v = 2.0871627049571764
            b_v = 0.72678226
            c_v = 0.02263327

            v_sigma = 1.7638140418516142  # km/s, 1-sigma residual scatter

            v_mars = a_v + b_v * v_earth + c_v * zenith_earth
            velocity_model = "V_mars = 2.08716 + 0.726782*V_earth + 0.0226333*zenith_earth"

        else:
            # Fallback if only Earth velocity is known:
            #
            # V_mars = a + b * V_earth
            #
            a_v = 2.9287194366029965
            b_v = 0.72787983

            v_sigma = 1.7948163153831902  # km/s, 1-sigma residual scatter

            v_mars = a_v + b_v * v_earth
            velocity_model = "V_mars = 2.92872 + 0.727880*V_earth"

        result["v_mars"] = v_mars
        result["v_mars_sigma"] = v_sigma
        result["v_mars_min"] = v_mars - sigma_level * v_sigma
        result["v_mars_max"] = v_mars + sigma_level * v_sigma
        result["velocity_model"] = velocity_model

    # ------------------------------------------------------------
    # Mars zenith-angle estimate
    # ------------------------------------------------------------
    if zenith_earth is not None:

        # Empirical fit:
        #
        # zenith_mars = a + b * zenith_earth
        #
        a_z = 0.0035683925115321813
        b_z = 0.99930066

        z_sigma = 0.07390989836020422  # deg, 1-sigma residual scatter

        zenith_mars = a_z + b_z * zenith_earth

        result["zenith_mars"] = zenith_mars
        result["zenith_mars_sigma"] = z_sigma
        result["zenith_mars_min"] = zenith_mars - sigma_level * z_sigma
        result["zenith_mars_max"] = zenith_mars + sigma_level * z_sigma
        result["zenith_model"] = "zenith_mars = 0.00356839 + 0.999301*zenith_earth"

    return result


# custom code to catch timeouts
class TimeoutException(Exception):
    """Custom exception for timeouts."""
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Function execution timed out")

class MetSimObj():
    def __init__(self, traj_path, const_json_file):
        # Init an axis for the electron line density
        # self.electronDensityPlot = self.magnitudePlot.canvas.axes.twiny()
        self.electron_density_plot_show = False
        ### Wake parameters ###
        self.wake_on = False
        self.wake_show_mass_bins = False
        self.wake_ht_current_index = 0
        self.current_wake_container = None
        # if self.wake_heights is not None:
        #     self.wake_plot_ht, self.current_wake_container = self.wake_heights[self.wake_ht_current_index]
        # else:
        #     self.wake_plot_ht = self.traj.rbeg_ele # m
        self.wake_normalization_method = 'area'
        self.wake_align_method = 'none'
        self.magnitudePlotWakeLines = None
        self.magnitudePlotWakeLineLabels = None
        self.velocityPlotWakeLines = None
        self.lagPlotWakeLines = None
        self.usg_data, self.traj = loadUSGInputFile(*os.path.split(traj_path))
        self.dir_path = os.path.dirname(traj_path)
        # Disable different density after erosion change
        self.erosion_different_rho = False
        # Disable different ablation coeff after erosion change
        self.erosion_different_sigma = False
        # Disable different erosion coeff after disruption at the beginning
        self.disruption_different_erosion_coeff = False
        # Fragmentation object
        self.fragmentation = None
        self.simulation_results = None
        self.const_prev = None
        self.simulation_results_prev = None
        self.const = Constants()  # initialize this, these will be replaced later
        self.const.P_0m = self.usg_data.P_0m_bolo
        # If a JSON file with constant was given, load them instead of initing from scratch
        if const_json_file is not None:
            # Load the constants from the JSON files
            self.const, const_json = loadConstants(const_json_file)
            # Init the fragmentation container for the GUI
            if len(self.const.fragmentation_entries):
                self.fragmentation = FragmentationContainer(self, \
                    os.path.join(self.dir_path, self.const.fragmentation_file_name))
                self.fragmentation.fragmentation_entries = self.const.fragmentation_entries
                # Overwrite the existing fragmentatinon file
                # self.fragmentation.writeFragmentationFile()
            # Check if the disruption erosion coefficient is different than the main erosion coeff
            if const_json['disruption_erosion_coeff'] != const_json['erosion_coeff']:
                self.disruption_different_erosion_coeff = True
            # Check if the density is changed after Hchange
            if 'erosion_rho_change' in const_json:
                if const_json['erosion_rho_change'] != const_json['rho']:
                    self.erosion_different_rho = True
            # Check if the ablation coeff is changed after Hchange
            if 'erosion_sigma_change' in const_json:
                if const_json['erosion_sigma_change'] != const_json['sigma']:
                    self.erosion_different_sigma = True
        else:
            raise('no json file!')

        ### Calculate atmosphere density coeffs (down to the bottom observed height, limit to 15 km) ###

        # Determine the height range for fitting the density
        self.dens_fit_ht_beg = self.const.h_init
        self.dens_fit_ht_end = self.traj.rend_ele - 5000
        if self.dens_fit_ht_end < 14000:
            self.dens_fit_ht_end = 14000

        # Fit the polynomail describing the density
        dens_co = MetSimGUI.fitAtmosphereDensity(self, self.dens_fit_ht_beg, self.dens_fit_ht_end)
        self.const.dens_co = dens_co

        # get global parameters from json file, everything other than params marked "free" are fixed
        dt = const_json.get('dt')
        P_0m = const_json.get('P_0m')
        h_init = const_json.get('h_init')
        m_kill = const_json.get('m_kill')
        v_kill = const_json.get('v_kill')
        h_kill = const_json.get('h_kill')
        len_kill = const_json.get('len_kill') 
        rho = const_json.get('rho')  # free
        rho_grain = const_json.get('rho_grain')  # free
        m_init = const_json.get('m_init')  # free
        sigma = const_json.get('sigma')  # free
        v_init = const_json.get('v_init')
        shape_factor = const_json.get('shape_factor')
        gamma = const_json.get('gamma')
        zenith_angle = const_json.get('zenith_angle')
        lum_eff = const_json.get('lum_eff')
        lum_eff_type = const_json.get('lum_eff_type')
        erosion_height_start = const_json.get('erosion_height_start')
        erosion_bins_per_10mass = const_json.get('erosion_bins_per_10mass')
        erosion_coeff = const_json.get('erosion_coeff')
        erosion_height_change = const_json.get('erosion_height_change')
        erosion_coeff_change = const_json.get('erosion_coeff_change')
        erosion_mass_index = const_json.get('erosion_mass_index')
        erosion_mass_min = const_json.get('erosion_mass_min')
        erosion_mass_max = const_json.get('erosion_mass_max')
        erosion_rho_change = const_json.get('rho')
        erosion_sigma_change = const_json.get('sigma')
        compressive_strength = const_json.get('compressive_strength')
        disruption_erosion_coeff = const_json.get('erosion_coeff')
        disruption_mass_grain_ratio = const_json.get('disruption_mass_grain_ratio')
        disruption_mass_index = const_json.get('disruption_mass_index')
        disruption_mass_min_ratio = const_json.get('disruption_mass_min_ratio')
        disruption_mass_max_ratio = const_json.get('disruption_mass_max_ratio')

        # get fragmentation parameters from json file
        # type, height, number, gamma, mass index are fixed, 
        # mass, ablation coefficient, erosion coefficient, grain min, grain max are not fixed
        num_frags = len(const_json.get('fragmentation_entries'))
        # fixed parameters
        frag_types = []
        frag_numbers = []
        frag_ab_coeffs = []
        frag_gammas = []
        frag_mis = []
        # free parameters
        frag_masses = []
        frag_er_coeffs = []
        frag_grain_mins = []
        frag_grain_maxs = []
        frag_heights = []

        # get indices of all free fragments and all eroding fragments (i.e. EF rather than D)
        self.fixed_frag_indices = []
        self.free_frag_indices = []
        self.er_frag_indices = []
        
        for i, frag in enumerate(const_json.get('fragmentation_entries')):
            # Check to see what type of fragment it is. Currently, only EF and D are supported
            self.free_frag_indices.append(i)
            # if eroding, add erosion coefficient and add fragment index to eroding fragment indices list
            if frag['frag_type'] == 'EF':
                self.er_frag_indices.append(i)
                frag_er_coeffs.append(frag['erosion_coeff'])
            # if dust, add zero erosion coefficient (ignored by MetSim) and do NOT add fragment index to eroding fragment indices list
            elif frag['frag_type'] == 'D':
                frag_er_coeffs.append(0.)
            else:
                raise ValueError('Only eroding fragments (EF) and dust (D) are currently supported by this script')
            ### If sigma, gamma are none, convert to whatever the default value is 
            frag_types.append(frag['frag_type'])
            frag_numbers.append(frag['number'])
            frag_mis.append(frag['mass_index'])
            frag_masses.append(frag['mass_percent'])
            frag_grain_mins.append(frag['grain_mass_min'])
            frag_grain_maxs.append(frag['grain_mass_max'])
            frag_heights.append(frag['height'])
            if frag['gamma'] == None:
                frag_gammas.append(gamma)
            else:
                frag_gammas.append(frag['gamma'])
            if frag['sigma'] == None:
                frag_ab_coeffs.append(sigma)
            else:
                frag_ab_coeffs.append(frag['sigma'])

        # set up masks for free and fixed indices
        # free
        free_frag_mask = np.zeros(num_frags, bool)
        free_frag_mask[self.free_frag_indices] = True  # only free frags
        # fixed
        fixed_frag_mask = np.ones(num_frags, bool)
        fixed_frag_mask[self.free_frag_indices] = False  # everything EXCEPT free frags (i.e. fixed frags)
        # erosion
        er_frag_mask = np.zeros(num_frags, bool)
        er_frag_mask[self.er_frag_indices] = True  # only frags that have erosion coefficients (i.e. all free fragments excluding dust)
        # assign free fixed parameters to object
        self.free_params = [m_init, 
                            list(np.array(frag_masses)[free_frag_mask]), 
                            list(np.array(frag_er_coeffs)[er_frag_mask]),  # use er frag mask for this one! 
                            list(np.array(frag_grain_mins)[free_frag_mask]), 
                            list(np.array(frag_grain_maxs)[free_frag_mask]),
                            list(np.array(frag_heights)[free_frag_mask])  # heights
                           ]
        self.fixed_params = [dt, P_0m, h_init, m_kill, v_kill, h_kill, len_kill, rho, rho_grain, sigma, 
                             v_init, shape_factor, 
                            gamma, zenith_angle, lum_eff, lum_eff_type, erosion_height_start, 
                            erosion_bins_per_10mass, erosion_coeff, erosion_height_change, 
                            erosion_coeff_change, erosion_mass_index, erosion_mass_min, erosion_mass_max, 
                            erosion_rho_change, erosion_sigma_change, compressive_strength, 
                            disruption_erosion_coeff, disruption_mass_grain_ratio, disruption_mass_index, 
                            disruption_mass_min_ratio, disruption_mass_max_ratio, 
                            frag_types, frag_numbers, 
                            frag_ab_coeffs, frag_gammas, frag_mis,
                            list(np.array(frag_masses)[fixed_frag_mask]), 
                            list(np.array(frag_er_coeffs)[~er_frag_mask]),  # and inverse of er frag mask 
                            list(np.array(frag_grain_mins)[fixed_frag_mask]), 
                            list(np.array(frag_grain_maxs)[fixed_frag_mask]),
                            list(np.array(frag_heights)[fixed_frag_mask])
                            ]
        
        # load all the global parameters into the object
        consts = dt, P_0m, h_init, m_kill, v_kill, h_kill, len_kill, rho, rho_grain, m_init, sigma, v_init, shape_factor, gamma, zenith_angle, lum_eff, lum_eff_type, erosion_height_start, erosion_bins_per_10mass, erosion_coeff, erosion_height_change, erosion_coeff_change, erosion_mass_index, erosion_mass_min, erosion_mass_max, erosion_rho_change, erosion_sigma_change, compressive_strength, disruption_erosion_coeff, disruption_mass_grain_ratio, disruption_mass_index, disruption_mass_min_ratio, disruption_mass_max_ratio
        self.loadGlobalParameters(consts)

        # get all parameters
        self.all_params = (self.free_params, self.fixed_params)
        
        # self.initializeSimulation(all_params)
        # self.initializeSimulation(const_json)
    
    def loadGlobalParameters(self, consts):
            """
            Loads the global parameters (constants) into the object
            """
            dt, P_0m, h_init, m_kill, v_kill, h_kill, len_kill, rho, rho_grain, m_init, sigma, v_init, shape_factor, gamma, zenith_angle, lum_eff, lum_eff_type, erosion_height_start, erosion_bins_per_10mass, erosion_coeff, erosion_height_change, erosion_coeff_change, erosion_mass_index, erosion_mass_min, erosion_mass_max, erosion_rho_change, erosion_sigma_change, compressive_strength, disruption_erosion_coeff, disruption_mass_grain_ratio, disruption_mass_index, disruption_mass_min_ratio, disruption_mass_max_ratio = consts
            # load all the non-fragmentation parameters into the object
            # 33 parameters
            self.const.dt = dt
            self.const.P_0m = P_0m
            self.const.h_init = h_init
            self.const.m_kill = m_kill
            self.const.v_kill = v_kill
            self.const.h_kill = h_kill
            self.const.len_kill = len_kill
            self.const.rho = rho
            self.const.rho_grain = rho_grain
            self.const.m_init = m_init
            self.const.sigma = sigma
            self.const.v_init = v_init
            self.const.shape_factor = shape_factor
            self.const.gamma = gamma
            self.const.zenith_angle = zenith_angle
            self.const.lum_eff = lum_eff
            self.const.lum_eff_type = lum_eff_type
            self.const.erosion_height_start = erosion_height_start
            self.const.erosion_bins_per_10mass = erosion_bins_per_10mass
            self.const.erosion_coeff = erosion_coeff
            self.const.erosion_height_change = erosion_height_change
            self.const.erosion_coeff_change = erosion_coeff_change
            self.const.erosion_mass_index = erosion_mass_index
            self.const.erosion_mass_min = erosion_mass_min
            self.const.erosion_mass_max = erosion_mass_max
            self.const.erosion_rho_change = erosion_rho_change
            self.const.erosion_sigma_change = erosion_sigma_change
            self.const.compressive_strength = compressive_strength
            self.const.disruption_erosion_coeff = disruption_erosion_coeff
            self.const.disruption_mass_grain_ratio = disruption_mass_grain_ratio
            self.const.disruption_mass_index = disruption_mass_index
            self.const.disruption_mass_min_ratio = disruption_mass_min_ratio
            self.const.disruption_mass_max_ratio = disruption_mass_max_ratio

    def initializeSimulation(self, all_params):
            """ Run the simulation and show the results. """
            # unpack all params
            free_params, fixed_params = all_params
            # unpack again
            m_init, frag_masses_free, frag_er_coeffs_free, frag_grain_mins_free, frag_grain_maxs_free, frag_heights_free = free_params 
            dt, P_0m, h_init, m_kill, v_kill, h_kill, len_kill, rho, rho_grain, sigma, v_init, shape_factor, gamma, zenith_angle, lum_eff, lum_eff_type, erosion_height_start, erosion_bins_per_10mass, erosion_coeff, erosion_height_change, erosion_coeff_change, erosion_mass_index, erosion_mass_min, erosion_mass_max, erosion_rho_change, erosion_sigma_change, compressive_strength, disruption_erosion_coeff, disruption_mass_grain_ratio, disruption_mass_index, disruption_mass_min_ratio, disruption_mass_max_ratio, frag_types, frag_numbers, frag_ab_coeffs, frag_gammas, frag_mis, frag_masses_fixed, frag_er_coeffs_fixed, frag_grain_mins_fixed, frag_grain_maxs_fixed, frag_heights_fixed = fixed_params
            # load all the global parameters into the object
            consts = dt, P_0m, h_init, m_kill, v_kill, h_kill, len_kill, rho, rho_grain, m_init, sigma, v_init, shape_factor, gamma, zenith_angle, lum_eff, lum_eff_type, erosion_height_start, erosion_bins_per_10mass, erosion_coeff, erosion_height_change, erosion_coeff_change, erosion_mass_index, erosion_mass_min, erosion_mass_max, erosion_rho_change, erosion_sigma_change, compressive_strength, disruption_erosion_coeff, disruption_mass_grain_ratio, disruption_mass_index, disruption_mass_min_ratio, disruption_mass_max_ratio
            self.loadGlobalParameters(consts)
            # combine free fixed fragmentation parameters
            frag_order = np.argsort(self.fixed_frag_indices + self.free_frag_indices)
            frag_masses = np.concatenate((frag_masses_fixed, frag_masses_free))[frag_order]
            frag_er_coeffs = np.zeros(len(frag_masses))  # number of fragments
            frag_er_coeffs[self.er_frag_indices] = frag_er_coeffs_free  # set all erosion fragments to their values, everything else is zero
            frag_grain_mins = np.concatenate((frag_grain_mins_fixed, frag_grain_mins_free))[frag_order]
            frag_grain_maxs = np.concatenate((frag_grain_maxs_fixed, frag_grain_maxs_free))[frag_order]
            frag_heights = np.concatenate((frag_heights_fixed, frag_heights_free))[frag_order]

            # print(frag_order, np.concatenate((frag_masses_fixed, frag_masses_free)), frag_masses)
            
            # Load fragmentation entries
            self.fragmentation_entries = []
            for i in range(0, len(frag_masses)):  # pick any frag entry to iterate over
                frag_entry = FragmentationEntry(frag_types[i], frag_heights[i], frag_numbers[i], frag_masses[i], 
                                                    frag_ab_coeffs[i], frag_gammas[i], frag_er_coeffs[i], frag_grain_mins[i], frag_grain_maxs[i], frag_mis[i])
                self.fragmentation_entries.append(frag_entry)
            # set the fragmentation entries to constants
            self.const.fragmentation_entries = self.fragmentation_entries
    
            # Sort entries by height
            self.fragmentation.sortByHeight()

            # Reset the status of all fragmentations
            self.fragmentation.resetAll()
        
            # fragmentation
            self.const.fragmentation_on = True

            # print(self.const.fragmentation_on)
    
            # Run the simulation
            frag_main, results_list, wake_results = runSimulation(self.const, compute_wake=self.wake_on)
            # print(results_list)

            # Store simulation results
            self.simulation_results = SimulationResults(self.const, frag_main, results_list, wake_results)

# initialize MetSim object
metsim_obj = MetSimObj(traj_path=event_path + '.txt', const_json_file=event_path + '_sim_fit_latest.json')
# initialize simulation, run with all parameters
metsim_obj.initializeSimulation(metsim_obj.all_params)

# plot 
print('plotting...')
resuls = earth_to_mars_empirical(v_earth=metsim_obj.const.v_init, zenith_earth=metsim_obj.const.zenith_angle*180/np.pi, sigma_level=2.0)
print(f"Velocity at Earth: {metsim_obj.const.v_init:.2f} km/s, Zenith angle at Earth: {metsim_obj.const.zenith_angle:.2f} deg")
print(f"Estimated Mars velocity: {resuls['v_mars']:.2f} km/s (model: {resuls['velocity_model']})")
Vinf_val_mars = resuls['v_mars']
print(f"Estimated Mars zenith angle: {resuls['zenith_mars']:.2f} deg (model: {resuls['zenith_model']})")
zenith_angle_mars = resuls['zenith_mars'] * np.pi / 180  # convert to radians

best_guess_obj_plot = metsim_obj.simulation_results

obs_data = metsim_obj.usg_data
obs_traj = metsim_obj.traj

# # print all the keys in obs_data
# print("USG data keys:", obs_data.__dict__.keys())
# print("USG trajectory keys:", obs_traj.__dict__.keys())

dens_co_mars = fitAtmPoly_mars(1*1000, 180*1000)
dens_co_earth = np.array(best_guess_obj_plot.const.dens_co)
rho_poly_earth = []
rho_poly_mars = []
altitude = np.arange(40, 181, 0.1)*1000
for alt in altitude:
    rho_poly_earth.append((atmDensPoly(alt, dens_co_earth)))
    rho_poly_mars.append((atmDensPoly(alt, dens_co_mars)))

# start simulations at the same bulk density point
dens_start_earth = atmDensPoly(best_guess_obj_plot.const.h_init, dens_co_earth)
start_height_mars = altitude[np.argmin(np.abs(np.array(rho_poly_mars) - dens_start_earth))]

dens_erosion_earth = atmDensPoly(best_guess_obj_plot.const.erosion_height_start, dens_co_earth)
erosion_height_start_mars = altitude[np.argmin(np.abs(np.array(rho_poly_mars) - dens_erosion_earth))]

# create a deep copy of best_guess_cost
best_guess_cost_mars = copy.deepcopy(best_guess_obj_plot.const)
best_guess_cost_mars.h_init = start_height_mars
print(f"Starting height Mars: {best_guess_cost_mars.h_init/1000:.2f} km")
print(f"Starting erosion_height_start_mars Mars: {erosion_height_start_mars/1000:.2f} km")

print("Fragmentation entries Earth:")
for i, frag in enumerate(best_guess_obj_plot.const.fragmentation_entries):
    print(f"Fragmentation {i}: Type: {frag.frag_type}, Height: {frag.height/1000:.2f} km, Number: {frag.number}, Mass Percent: {frag.mass_percent}%, Ablation Coeff: {frag.sigma}, Gamma: {frag.gamma}, Grain Mass Min: {frag.grain_mass_min} kg, Grain Mass Max: {frag.grain_mass_max} kg, Mass Index: {frag.mass_index}")
# print best_guess_cost_mars.fragmentation_entries
print("Fragmentation entries Mars:")
for i, frag in enumerate(best_guess_cost_mars.fragmentation_entries):
    print(f"Fragmentation {i}: Type: {frag.frag_type}, Height: {frag.height/1000:.2f} km, Number: {frag.number}, Mass Percent: {frag.mass_percent}%, Ablation Coeff: {frag.sigma}, Gamma: {frag.gamma}, Grain Mass Min: {frag.grain_mass_min} kg, Grain Mass Max: {frag.grain_mass_max} kg, Mass Index: {frag.mass_index}")


best_guess_cost_mars.erosion_height_start = erosion_height_start_mars
best_guess_cost_mars.v_init = Vinf_val_mars # is m/s
# PLANET PARAMETERS
best_guess_cost_mars.G0 = G0_mars  # m/s^2
best_guess_cost_mars.r_earth = R_MARS * 1000  # in m
best_guess_cost_mars.dens_co = np.array(dens_co_mars)
best_guess_cost_mars.v_kill = Vinf_val_mars - 10000  # to m/s
if best_guess_cost_mars.v_kill < 2500:
    best_guess_cost_mars.v_kill = 2500
    print(f"v_kill is smaller than 2.5 km/s, setting it to 2.5 km/s")

best_guess_cost_mars.zenith_angle = zenith_angle_mars
print(f"Zenith angle Mars: {180/np.pi*best_guess_cost_mars.zenith_angle:.6f}°")

# Minimum height (m) for simulation termination
best_guess_cost_mars.h_kill = 1000

frag_main, results_list, wake_results = runSimulation(best_guess_cost_mars, compute_wake=False)
best_guess_obj_plot_mars = SimulationResults(best_guess_cost_mars, frag_main, results_list, wake_results)

# Load fragmentation entries
fragmentation_entries = []
for i, frag in enumerate(best_guess_obj_plot.const.fragmentation_entries):

    # now I want to know when it reaches the dyn_press on mars that matches the erosion_beg_dyn_press on earth
    heightsame_dynpress_mars = best_guess_obj_plot_mars.leading_frag_height_arr[np.argmin(np.abs(best_guess_obj_plot_mars.leading_frag_dyn_press_arr[:-1] - frag.dyn_pressure))]
    # heightsame_dynpress_mars_single = best_guess_obj_plot_single_mars.leading_frag_height_arr[np.argmin(np.abs(best_guess_obj_plot_single_mars.leading_frag_dyn_press_arr[:-1] - best_guess_obj_plot.const.erosion_beg_dyn_press))]
    print(f"Erosion onset dynamic pressure on Mars : {frag.dyn_pressure} Pa at height {frag.height/1000:.2f} km instead of {heightsame_dynpress_mars/1000:.2f} km")

    # # for i in range(0, len(frag_masses)):  # pick any frag entry to iterate over
    # frag_entry = FragmentationEntry(frag.frag_type, frag.height, frag.number, frag.mass_percent,
    #                                     frag.sigma, frag.gamma, frag.erosion_coeff, frag.grain_mass_min, frag.grain_mass_max, frag.mass_index)
    # for i in range(0, len(frag_masses)):  # pick any frag entry to iterate over

    frag_entry = FragmentationEntry(frag.frag_type, heightsame_dynpress_mars, frag.number, frag.mass_percent,
                                        frag.sigma, frag.gamma, frag.erosion_coeff, frag.grain_mass_min, frag.grain_mass_max, frag.mass_index)

    fragmentation_entries.append(frag_entry)

# set the fragmentation entries to constants
best_guess_cost_mars.fragmentation_entries = fragmentation_entries


frag_main, results_list, wake_results = runSimulation(best_guess_cost_mars, compute_wake=False)
best_guess_obj_plot_mars = SimulationResults(best_guess_cost_mars, frag_main, results_list, wake_results)

# frag_main, results_list, wake_results = runSimulation(best_guess_obj_plot.const, compute_wake=False)
# best_guess_cost_mars_test = SimulationResults(best_guess_obj_plot.const, frag_main, results_list, wake_results)

# plot y axis the unique_heights_massvar vs Tot_energy_arr
# fig, ax = plt.subplots(1,2, figsize=(12, 6))
fig, ax = plt.subplots(figsize=(6, 6))
station_colors = {}
cmap = plt.get_cmap("tab10")
# # ABS MAGNITUDE
ax.plot(obs_data.absolute_magnitudes, obs_traj.observations[0].model_ht / 1000, 'x--', color='tab:blue', label='USG ATLAS fireball data')
# for station in np.unique(obs_data.stations_lum):
#     mask = obs_data.stations_lum == station
#     if station not in station_colors:
#         station_colors[station] = cmap(len(station_colors) % 10)
#     ax.plot(obs_data.absolute_magnitudes[mask], obs_data.height_lum[mask] / 1000, 'x--', color=station_colors[station], label=station)
# # max_mag= np.max(obs_data.absolute_magnitudes)+1
# # take the y axis limits from the obs_data
y_min = ax.get_ylim()[0]
y_max = ax.get_ylim()[1]
x_max = ax.get_xlim()[1]

# make a first subplot with the lightcurve against height
ax.plot(best_guess_obj_plot.abs_magnitude,best_guess_obj_plot.leading_frag_height_arr/1000, color='k', label='Best Fit Simulation')

ax.set_ylabel('Height [km]', fontsize=15)
ax.set_xlabel('Abs.Mag [-]', fontsize=15)

# add a hrizontal line at y=total_energy_before_erosion
# ax.axhline(y=metsim_obj.const.erosion_height_start/1000, color='gray', linestyle='--', label='Erosion Height Start $h_{e}$')
# if flag_total_rho:
#     ax.axhline(y=metsim_obj.const.erosion_height_change/1000, color='gray', linestyle='-.', label='Erosion Height Change $h_{e2}$')
# ax.legend(fontsize=10)
# call the Earth plot
# ax[0].set_title('Earth', fontsize=16)


# # make a second subplot with the lightcurve against height for mars
ax.plot(best_guess_obj_plot_mars.abs_magnitude,best_guess_obj_plot_mars.leading_frag_height_arr/1000, color='red', label='Best Fit at Mars')# color='tab:purple',  (Mars same $\\rho$)
# print(f"magnitude at mars: {best_guess_obj_plot_mars.abs_magnitude} ")
# print(f"height at mars: {best_guess_obj_plot_mars.leading_frag_height_arr/1000} ")
# ax.axhline(y=best_guess_obj_plot_mars.const.erosion_height_start/1000, color='tab:purple', linestyle='--')
# ax.plot(best_guess_obj_plot_mars_dyn_press.abs_magnitude,best_guess_obj_plot_mars_dyn_press.leading_frag_height_arr/1000, color='tab:brown', label='Best Fit Simulation (Mars same $p_{dyn}$)')
# ax.axhline(y=heightsame_dynpress_mars/1000, color='tab:brown', linestyle='--')
# ax.plot(best_guess_obj_plot_mars_energy.abs_magnitude,best_guess_obj_plot_mars_energy.leading_frag_height_arr/1000, color='tab:pink', label='Best Fit Simulation (Mars same $E_{e}$)')
# ax.axhline(y=height_energy_erosion_start_mars/1000, color='tab:pink', linestyle='--')
# if flag_total_rho:
#     ax.axhline(y=best_guess_obj_plot_mars.const.erosion_height_change/1000, color='tab:purple', linestyle='-.')
#     ax.axhline(y=heightsame_dynpress_change_mars/1000, color='tab:brown', linestyle='-.')
#     ax.axhline(y=height_energy_erosion_change_mars/1000, color='tab:pink', linestyle='-.')



# ####### plot the single body ablation model on Mars ######
# if found_single_body:
#     if (1 / obs_data.fps_lum) > best_guess_obj_plot_single_mars.const.dt:
#         best_guess_obj_plot_single_mars.luminosity_arr, best_guess_obj_plot_single_mars.abs_magnitude = luminosity_integration(
#             best_guess_obj_plot_single_mars.time_arr, best_guess_obj_plot_single_mars.time_arr, best_guess_obj_plot_single_mars.luminosity_arr,
#             best_guess_obj_plot_single_mars.const.dt, obs_data.fps_lum, obs_data.P_0m   
#         )
#     ax.plot(best_guess_obj_plot_single_mars.abs_magnitude,best_guess_obj_plot_single_mars.leading_frag_height_arr/1000, color='blue', linestyle=':', label='Single Body Ablation (Mars)')


# ax[0].invert_xaxis()
# ax[1].legend(fontsize=10)
# ax.set_title('Mars', fontsize=16)
# ax[1].set_ylabel('Height [km]', fontsize=15)
ax.set_xlabel('Abs.Mag [-]', fontsize=15)
# activate grid
ax.grid()
x_min = ax.get_xlim()[0]
x_max = -18
# put the x axis from the x_max to x_min
ax.set_xlim(x_max, x_min)
ax.set_ylim(0, 60)

# for the y axis
new_ax_min = np.min([y_min, ax.get_ylim()[0]])
# # closest index to the x_max in best_guess_obj_plot_mars.abs_magnitude after the erosion start
# index_min_abs_mag = np.argmin(np.abs(best_guess_obj_plot_mars.abs_magnitude[np.argmin(np.abs(best_guess_obj_plot_mars.leading_frag_height_arr - best_guess_obj_plot_mars.const.erosion_height_start)):] - x_max))
# # closest index to the x_max in best_guess_obj_plot_mars.leading_frag_height_arr/1000
# new_ax_min = np.max([best_guess_obj_plot_mars.leading_frag_height_arr[index_min_abs_mag]/1000, new_ax_min])
new_ax_max = np.min([y_max, ax.get_ylim()[1]])
new_ax_max = np.max([metsim_obj.const.erosion_height_start/1000+2, new_ax_max])
ax.set_ylim(0, new_ax_max)
# ax.invert_xaxis()
# put legend outside the plot
# ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
# put it outside to the top right
# ax.legend(fontsize=10, loc='upper left', bbox_to_anchor=(1.05, 1))
# put thelegend inside i the lower right corner
ax.legend(fontsize=8, loc='lower right')
# ax[1].grid()
# plt.suptitle(f'Lightcurve Comparison for {base_name}', fontsize=18)
plt.tight_layout()
# show the plot
plt.show()
# plt.savefig(output_dir + os.sep + base_name + "_Lightcurve_Earth_vs_Mars.png")
# plt.close()


