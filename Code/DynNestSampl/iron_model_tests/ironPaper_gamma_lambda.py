import numpy as np
from scipy.constants import k as k_B  # Boltzmann constant
from math import pi, exp
import matplotlib.pyplot as plt
from scipy.special import gamma
from wmpl.MetSim.MetSimErosionCyTools import atmDensityPoly
from dynesty.utils import quantile as _quantile
import os
import sys
from scipy.integrate import quad

# Add the parent directory to the sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from DynNestSapl_metsim import *

# --- Settings ---
D_dr_microns = 50           # Mean diameter in micrometers
rho = 7800                   # Density in kg/m^3
total_droplets = 10000       # Total droplet count to normalize both distributions
mass_index = 1               # Power-law mass index (slope)
bins_per_decade = 10         # Power-law: 10 bins per order of magnitude

input_dirfile = r"C:\Users\maxiv\Documents\UWO\Papers\3)Sporadics\iron_erosion_coeff\iron_sigma0\20190704_072615_iron_2fr-4000-8000"
save_dir = r"C:\Users\maxiv\WMPG-repoMAX\Code\DynNestSampl\iron_model_tests"

def get_air_properties(alt_km):
    """Returns temperature (K), pressure (Pa), and density (kg/m^3) for 80-120 km using exponential fit."""
    # Approximate fits (you can replace with NRLMSISE-00 or table lookup for accuracy)
    T = 200  # K (constant approximation in mesosphere/lower thermosphere)
    
    # Pressure exponential decrease (rough estimate)
    P0 = 101325  # Sea-level pressure (Pa)
    H = 7.64  # Scale height (km)
    P = P0 * np.exp(-alt_km / H)
    
    # Density approximation: rho = P / (R_specific * T)
    R_specific = 287.05  # J/(kg·K) for dry air
    rho = P / (R_specific * T)
    
    return T, P, rho

def compute_gamma_lambda(mass_loss_rate, radius, velocity, altitude, dens_co=[]):
    """
    Parameters:
    - mass_loss_rate: dm/dt in kg/s (positive number)
    - radius: meteoroid radius in meters
    - velocity: in m/s
    - altitude_km: altitude in meters

    Returns:
    - Gamma (Γ)
    - Lambda (Λ)
    """
    T, P, rho_air = get_air_properties(altitude / 1000)
    # print(f"Temperature: {T} K, Pressure: {P} Pa, Density: {rho_air} kg/m^3 at altitude {altitude} km")
    if len(dens_co) > 0:
        rho_air = atmDensityPoly(altitude, dens_co)
        # print(f"Using custom density Metsim: {rho_air}")

    # Mean free path λ
    d_air = 3.7e-10  # m, effective diameter of air molecule
    lambda_mfp = k_B * T / (np.sqrt(2) * pi * d_air**2 * P)

    # Knudsen number
    Kn = lambda_mfp / radius

    # Blowing ratio
    Br = -mass_loss_rate / (pi * radius**2 * rho_air * velocity)

    # Coefficients
    f = 0.87864 * Kn**(-0.68044)
    g = 0.91999 * Kn**(-0.62867)

    # Gamma and Lambda
    Gamma = (f * g * (exp(g * Br) - 1)) / (exp(f * Br) - 1)
    Lambda = (Br * f) / (exp(f * Br) - 1)

    return Gamma, Lambda

# Use the class to find .dynesty, load prior, and decide output folders
finder = find_dynestyfile_and_priors(input_dir_or_file=input_dirfile, prior_file="",resume=True,output_dir=input_dirfile,use_all_cameras=False,pick_position=0)

for i, (base_name, dynesty_info, prior_path, out_folder) in enumerate(zip(finder.base_names, finder.input_folder_file, finder.priors, finder.output_folders)):
    dynesty_file, pickle_file, bounds, flags_dict, fixed_values = dynesty_info
    print(base_name)
    obs_data = finder.observation_instance(base_name)
    obs_data.file_name = pickle_file  # update the file name in the observation data object
    dsampler = dynesty.DynamicNestedSampler.restore(dynesty_file)
    # dsampler = load_dynesty_file(dynesty_file)

    # set up the observation data object
    obs_data = finder.observation_instance(base_name)
    obs_data.file_name = pickle_file # update teh file name in the observation data object

    # if the real_event has an initial velocity lower than 30000 set "dt": 0.005 to "dt": 0.01
    if obs_data.v_init < 30000:
        obs_data.dt = 0.01
        # const_nominal.erosion_bins_per_10mass = 5
    else:
        obs_data.dt = 0.005
        # const_nominal.erosion_bins_per_10mass = 10

    obs_data.disruption_on = False

    obs_data.lum_eff_type = 5

    obs_data.h_kill = np.min([obs_data.height_lum[-1],obs_data.height_lag[-1]])-1000
    # check if the h_kill is smaller than 0
    if obs_data.h_kill < 0:
        obs_data.h_kill = 1
    # check if np.min(obs_data.velocity[-1]) is smaller than v_init-10000
    if np.min(obs_data.velocities) < obs_data.v_init-10000:
        obs_data.v_kill = obs_data.v_init-10000
    else:
        obs_data.v_kill = np.min(obs_data.velocities)-5000
    # check if the v_kill is smaller than 0
    if obs_data.v_kill < 0:
        obs_data.v_kill = 1

    dynesty_run_results = dsampler.results

    sim_num = np.argmax(dynesty_run_results.logl) # best simulation 
    guess = dynesty_run_results.samples[sim_num]
    samples = dynesty_run_results.samples
    weights = dynesty_run_results.importance_weights()
    w = weights / np.sum(weights)

    variables_sing = list(flags_dict.keys())
    flag_total_rho = False
    # for variable in variables: for 
    for i, variable in enumerate(variables_sing):
        if 'log' in flags_dict[variable]:  
            guess[i] = 10**(guess[i])
        if 'noise_lag' == variable:
            obs_data.noise_lag = guess[i]
            obs_data.noise_vel = guess[i]*np.sqrt(2)/(1.0/32)
        if 'noise_lum' == variable:
            obs_data.noise_lum = guess[i]
        if variable == 'erosion_rho_change':
            flag_total_rho = True
    # find erosion change height
    if 'erosion_height_change' in variables_sing:
        erosion_height_change = guess[variables_sing.index('erosion_height_change')]
    if 'm_init' in variables_sing:
        m_init = guess[variables_sing.index('m_init')]

    best_guess_obj_plot = run_simulation(guess, obs_data, variables_sing, fixed_values)

    dt = best_guess_obj_plot.const.dt
    time = np.array(best_guess_obj_plot.time_arr, dtype=np.float64)[:-1]  # remove last element to match heights and velocities
    heights = np.array(best_guess_obj_plot.leading_frag_height_arr, dtype=np.float64)[:-1]
    mass_best = np.array(best_guess_obj_plot.mass_total_active_arr, dtype=np.float64)
    mass_loss_rate = (-1)*np.diff(mass_best) / dt # positive mass loss rate in kg/s
    vel_best = np.array(best_guess_obj_plot.leading_frag_vel_arr, dtype=np.float64)[:-1]
    dens_co = best_guess_obj_plot.const.dens_co
    mass_best = mass_best[:-1]  # remove last element to match heights and velocities
    mass_before = mass_best[np.argmin(np.abs(heights - erosion_height_change))]
    rho = best_guess_obj_plot.const.rho
    erosion_height = best_guess_obj_plot.const.erosion_height_start

    if flag_total_rho:
        rho = best_guess_obj_plot.const.rho*(abs(m_init-mass_before) / m_init) + best_guess_obj_plot.const.erosion_rho_change * (mass_before / m_init)

    # compute radius from mass
    radius_best = (3 * mass_best / (4 * pi * rho))**(1/3)

    # # Example usage:
    # mass_loss_rate = 0.000001  # kg/s
    # radius = 0.001  # meters (1 mm)
    # velocity = 20000  # m/s (20 km/s)
    # altitude_m = 75*1000  # example
    # Gamma, Lambda = compute_gamma_lambda(mass_loss_rate, radius, velocity, altitude_m)
    # print(f"Gamma (Γ): {Gamma:.4f}")
    # print(f"Lambda (Λ): {Lambda:.4f}")
    Gamma_array = []
    Lambda_array = []
    # find the index where mass_loss_rate is not zero
    non_zero_indices = np.where(mass_loss_rate > 0)[0]
    # find the Gamma and Lambda for each timestep in non_zero_indices
    for i in non_zero_indices:
        Gamma, Lambda = compute_gamma_lambda(mass_loss_rate[i], radius_best[i], vel_best[i], heights[i],dens_co)
        Gamma_array.append(Gamma)
        Lambda_array.append(Lambda)

    params_index = 0
    for i in [0, 1]:

        # now plot
        if flag_total_rho and i == 1:
            # only plot from the erosion_height_change onwards
            erosion_height_change_index = np.argmin(np.abs(heights - erosion_height_change))
            # change the non_zero_indices to only include the ones after the erosion_height_change
            lenght_nonzero= len(non_zero_indices)
            non_zero_indices = non_zero_indices[non_zero_indices >= erosion_height_change_index]
            lenght_nonzero_change = len(non_zero_indices)
            params_index = abs(lenght_nonzero-lenght_nonzero_change)

        # plotting Gamma and Lambda in 2 subplots against height
        plt.figure(figsize=(10, 6)) 
        plt.subplot(2, 2, 1)
        plt.plot(Gamma_array[params_index:], heights[non_zero_indices]/1000, '.', label='Gamma (Γ)', color='blue')
        # add a horizontal line for erosion_height and erosion_height_change
        plt.axhline(erosion_height/1000, color='gray', linestyle='--', label='Erosion Height')
        if 'erosion_height_change' in locals():
            plt.axhline(erosion_height_change/1000, color='gray', linestyle='-.', label='Erosion Height Change')
        plt.ylabel('Height (km)')
        plt.xlabel('Gamma (Γ)')
        plt.grid(True)
        plt.legend()
        plt.subplot(2, 2, 2)
        plt.plot(Lambda_array[params_index:], heights[non_zero_indices]/1000, '.', label='Lambda (Λ)', color='orange')
        # add a horizontal line for erosion_height and erosion_height_change
        plt.axhline(erosion_height/1000, color='gray', linestyle='--')
        if 'erosion_height_change' in locals():
            plt.axhline(erosion_height_change/1000, color='gray', linestyle='-.')
        plt.ylabel('Height (km)')
        plt.xlabel('Lambda (Λ)')
        plt.xscale('log')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        # add the mass loss rate to the plot
        plt.subplot(2, 2, 3)
        plt.plot(mass_loss_rate[non_zero_indices], heights[non_zero_indices]/1000, '.', label='Mass Loss Rate (kg/s)', color='green')
        # add a horizontal line for erosion_height and erosion_height_change
        plt.axhline(erosion_height/1000, color='gray', linestyle='--')
        if 'erosion_height_change' in locals():
            plt.axhline(erosion_height_change/1000, color='gray', linestyle='-.')
        plt.ylabel('Height (km)')
        plt.xlabel('Mass Loss Rate (kg/s)')
        plt.grid(True)
        plt.legend()
        plt.subplot(2, 2, 4)
        plt.plot(np.array(Lambda_array[params_index:])/(2*np.array(Gamma_array[params_index:])*7115134)*1e6, heights[non_zero_indices]/1000, '.', label='Ablation Coeficient (kg/MJ)', color='indigo')
        # add a horizontal line for erosion_height and erosion_height_change
        plt.axhline(erosion_height/1000, color='gray', linestyle='--')
        if 'erosion_height_change' in locals():
            plt.axhline(erosion_height_change/1000, color='gray', linestyle='-.')
        # xscale to logarithmic
        plt.xscale('log')
        plt.ylabel('Height (km)')
        plt.xlabel('Ablation Coeficient (kg/MJ)')
        plt.grid(True)
        plt.legend()
        # save the figure
        if flag_total_rho and i == 1:
            # save the figure
            plt.savefig(os.path.join(save_dir, f"{base_name}_Gamma_Lambda_mass_loss_rate_afterheight.png"), 
                    bbox_inches='tight',
                    pad_inches=0.1,       # a little padding around the edge
                    dpi=300)
        else:
            # save the figure
            plt.savefig(os.path.join(save_dir, f"{base_name}_Gamma_Lambda_mass_loss_rate.png"), 
                    bbox_inches='tight',
                    pad_inches=0.1,       # a little padding around the edge
                    dpi=300)
        # plt.savefig(os.path.join(save_dir, f"{base_name}_Gamma_Lambda_mass_loss_rate.png"), dpi=300)
        plt.close()





# --- Gamma Distributions ---

gamma_5_3 = gamma(5/3)
denominator = (D_dr_microns * gamma_5_3) ** 3
total_droplets = 10000

# PDF function
def dN_dD(D):
    return (3 * D**2 / denominator) * np.exp(-D**3 / denominator)

# Bin edges (20 bins centered around D_dr, ±100 µm)
# bin_width = 10
bin_edges = np.linspace(D_dr_microns - D_dr_microns/2, D_dr_microns + D_dr_microns/2, 21)
bin_width = bin_edges[1] - bin_edges[0]  # Width of each bin in micrometers
bin_centers_um = (bin_edges[:-1] + bin_edges[1:]) / 2
bin_counts = []

# Integrate over each bin
for D_min, D_max in zip(bin_edges[:-1], bin_edges[1:]):
    N_bin, _ = quad(dN_dD, D_min, D_max)
    bin_counts.append(N_bin * total_droplets)

# # Convert bin centers to mass in grams (D in µm → m)
# bin_centers_m = bin_centers_um * 1e-6
# bin_masses_kg = (np.pi / 6) * rho * bin_centers_m**3
# mass_bin_widths = np.diff(bin_masses_kg)

# Convert diameter edges to mass edges
bin_edges_m = bin_edges * 1e-6
mass_edges_kg = (np.pi / 6) * rho * bin_edges_m**3
mass_centers_kg = (mass_edges_kg[:-1] + mass_edges_kg[1:]) / 2
mass_widths_kg = np.diff(mass_edges_kg)


# Plot
plt.figure(figsize=(8, 5))
plt.bar(mass_centers_kg, bin_counts, width=mass_widths_kg, align='center', edgecolor='black', color='salmon')
# make the x-axis logarithmic
plt.xscale('log')
plt.xlabel("Mass (kg)")
plt.ylabel("Expected droplet count per bin")
plt.title(r"Size Distribution in Mass Space Centered at $D_{dr}$ = "+f"{D_dr_microns}"+r" $\mu$m (20 bins, "+f"{total_droplets} droplets)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
# save the figure
plt.savefig(os.path.join(save_dir,"gamma_distribution_mass_bins.png"), dpi=300)

# Plot
plt.figure(figsize=(8, 5))
plt.bar(bin_centers_um, bin_counts, width=bin_width, 
        align='center', edgecolor='black', color='salmon', label="Gamma-based dN")
plt.xlabel("Diameter $D$ (µm)")
plt.ylabel("Expected droplet count per bin")
plt.title(r"Size Distribution Centered at $D_{dr}$ = "+f"{D_dr_microns}"+r" $\mu$m (20 bins, "+f"{total_droplets} droplets)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
# save the figure
plt.savefig(os.path.join(save_dir,"gamma_distribution_diameter_bins.png"), dpi=300)
plt.close()

# --- Change Size Distribution Function ---

# Compute mass range from Gamma model
mass_min_plot = 10**(-9)#min(mass_centers_min)
mass_max_plot = 10**(-7)#max(mass_centers_max)

# create a list 
D_dr_list = [D_dr_microns-49, D_dr_microns, D_dr_microns+50, D_dr_microns+100]  # values of D_dr in µm

plt.figure(figsize=(9, 5))

mass_centers_min = None
mass_centers_max = None
for D_dr_microns_list in D_dr_list:
    gamma_5_3 = gamma(5/3)
    denominator = (D_dr_microns_list * gamma_5_3) ** 3

    def dN_dD(D):
        return (3 * D**2 / denominator) * np.exp(-D**3 / denominator)

    # Bin settings
    bin_edges_um = np.linspace(D_dr_microns_list - D_dr_microns_list/2, D_dr_microns_list + D_dr_microns_list/2, 21)
    bin_centers_um = (bin_edges_um[:-1] + bin_edges_um[1:]) / 2

    # Integrate over each diameter bin
    bin_counts = []
    for D_min, D_max in zip(bin_edges_um[:-1], bin_edges_um[1:]):
        N_bin, _ = quad(dN_dD, D_min, D_max)
        bin_counts.append(N_bin * total_droplets)

    # Convert to mass space (kg)
    bin_edges_m = bin_edges_um * 1e-6
    mass_edges_kg = (np.pi / 6) * rho * bin_edges_m**3
    mass_centers_kg = (mass_edges_kg[:-1] + mass_edges_kg[1:]) / 2

    # save the max and the min of the mass centers
    if mass_centers_min is None or min(mass_centers_kg) < mass_centers_min:
        mass_centers_min = min(mass_centers_kg)
    if mass_centers_max is None or max(mass_centers_kg) > mass_centers_max:
        mass_centers_max = max(mass_centers_kg)

    # Plot each distribution as a line
    plt.plot(mass_centers_kg, bin_counts, label=f"$D_{{dr}}$ = {D_dr_microns_list} µm")

# Finalize plot
plt.xscale('log')
plt.xlabel("Mass (kg)")
plt.ylabel("Expected droplet count per bin")
plt.title(f"Gamma Distributions for 10,000 Droplets")
plt.grid(True, linestyle='--', alpha=0.6)
# set xlim to the min and max of the mass centers
plt.xlim(mass_min_plot, mass_max_plot)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "gamma_distributions_overlay.png"), dpi=300)
plt.close()

# --- Cumulative Size Distribution Function ---

# Constants
C = 1  # Integration constant (set to 1 for normalized cumulative distribution)

# Convert to meters for consistency if needed (but we keep microns since units cancel out)
D = np.linspace(1, 500, 1000)  # micrometers

# Compute the function
N = -np.exp(- (D**3) / (D_dr_microns * gamma_5_3)**3 ) + C

# Plotting
plt.figure(figsize=(8, 5))
plt.plot(D, N, label=r'$N(D) = -\exp\left(-\frac{D^3}{(D_{dr}\Gamma(5/3))^3}\right) + C$', color='salmon')
plt.xlabel("Particle Diameter $D$ (µm)")
plt.ylabel("$N(D)$")
plt.title(r"Cumulative Size Distribution Function for $D_{dr}$ = "+f"{D_dr_microns}"+r" $\mu$m")
plt.grid(True)
plt.legend()
plt.tight_layout()
# plt.show()
# Save the plot
plt.savefig(os.path.join(save_dir,"cumulative_distribution_function.png"), dpi=300)
plt.close()



# --- Gamma-based Distribution Function ---
def compute_gamma_distribution(D_dr_um, rho, total_droplets, bin_count=20):
    gamma_5_3 = gamma(5/3)
    denom = (D_dr_um * gamma_5_3) ** 3

    def dN_dD(D):
        return (3 * D**2 / denom) * np.exp(-D**3 / denom)

    # Create bin edges ±100 µm around D_dr
    bin_edges_um = np.linspace(D_dr_um - D_dr_um/2, D_dr_um + D_dr_um/2, bin_count + 1)
    bin_centers_um = (bin_edges_um[:-1] + bin_edges_um[1:]) / 2

    bin_edges_m = bin_edges_um * 1e-6
    bin_centers_m = bin_centers_um * 1e-6

    # Integrate dN over each bin
    bin_counts = []
    for D_min, D_max in zip(bin_edges_um[:-1], bin_edges_um[1:]):
        N_bin, _ = quad(dN_dD, D_min, D_max)
        bin_counts.append(N_bin * total_droplets)

    # Convert diameter bins to mass bins
    mass_edges_kg = (np.pi / 6) * rho * bin_edges_m**3
    mass_centers_kg = (np.pi / 6) * rho * bin_centers_m**3
    mass_widths_kg = np.diff(mass_edges_kg)

    return mass_centers_kg, mass_widths_kg, bin_counts

# --- Power-law Distribution Function with Log Binning ---
def compute_log_binned_powerlaw(mass_min, mass_max, bins_per_decade, mass_index, total_droplets):
    decades = np.log10(mass_max) - np.log10(mass_min)
    n_bins = int(bins_per_decade * decades)

    # Logarithmically spaced mass bins
    mass_edges_kg = np.logspace(np.log10(mass_min), np.log10(mass_max), n_bins + 1)
    mass_centers_kg = (mass_edges_kg[:-1] + mass_edges_kg[1:]) / 2
    mass_widths_kg = np.diff(mass_edges_kg)

    # Power-law distribution over bin centers
    n_grains = mass_centers_kg**(-mass_index)
    n_grains /= np.sum(n_grains)
    n_grains *= total_droplets

    return mass_centers_kg, mass_widths_kg, n_grains

# --- Generate Distributions ---
print(f"Computing Gamma-based distribution for D_dr = {D_dr_microns} µm, rho = {rho} kg/m^3, total droplets = {total_droplets}")
mass_centers_gamma, mass_widths_gamma, gamma_counts = compute_gamma_distribution(
    D_dr_microns, rho, total_droplets, bin_count=20)

mass_min = min(mass_centers_gamma)
mass_max = max(mass_centers_gamma)
mass_centers_powerlaw, mass_widths_powerlaw, powerlaw_counts = compute_log_binned_powerlaw(
    mass_min, mass_max, bins_per_decade, mass_index, total_droplets)

# --- Plot side-by-side ---
plt.figure(figsize=(10, 5))
bar1 = plt.bar(mass_centers_gamma, gamma_counts, width=mass_widths_gamma,
               align='center', edgecolor='black', color='salmon', label=f"Gamma-based dN (Ddr {D_dr_microns} µm)")
# line1, = plt.plot(mass_centers_kg, powerlaw_counts, 'o-', color='blue',
#                   label=f"Power-law (s = {mass_index})")
plt.plot(mass_centers_powerlaw, powerlaw_counts, 'o-', color='blue',
                  label=f"Power-law (s = {mass_index})")
        #  label=f"Power-law (s = {mass_index}, {bins_per_decade} bins/decade)")

plt.xlabel("Mass (kg)")
plt.xscale('log')
plt.ylabel("Expected droplet count per bin")
plt.title("Gamma-based Distribution vs. Power-law Fragmentation for "+str(total_droplets)+" droplets")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(save_dir,'Ddr_'+str(D_dr_microns)+'_mass_index_'+str(mass_index)+"_gamma_vs_powerlaw_distribution.png"), dpi=300)
plt.close()

# --- Overlay Power-law Distributions ---

plt.figure(figsize=(10, 5))

mass_indices = [1, 1.5, 2, 2.5, 3]  # Power-law mass indices to compare

def compute_log_binned_powerlaw(mass_min, mass_max, bins_per_decade, mass_index, total_droplets):
    decades = np.log10(mass_max) - np.log10(mass_min)
    n_bins = max(int(bins_per_decade * decades), 5)
    mass_edges_kg = np.logspace(np.log10(mass_min), np.log10(mass_max), n_bins + 1)
    mass_centers_kg = (mass_edges_kg[:-1] + mass_edges_kg[1:]) / 2
    n_grains = mass_centers_kg**(-mass_index)
    n_grains /= np.sum(n_grains)
    n_grains *= total_droplets
    return mass_centers_kg, n_grains


for s in mass_indices:
    m_centers, n_grains = compute_log_binned_powerlaw(mass_min_plot, mass_max_plot, bins_per_decade, s, total_droplets)
    plt.plot(m_centers, n_grains, label=f"Power-law (s = {s})")

# Finalize plot
plt.xscale('log')
plt.xlabel("Mass (kg)")
plt.ylabel("Expected droplet count per bin")
plt.title(f"Power-law Distributions")
plt.grid(True, linestyle='--', alpha=0.6)
# set xlim to the min and max of the mass centers
plt.xlim(mass_min_plot, mass_max_plot)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "pow_distributions_overlay.png"), dpi=300)
plt.close()
