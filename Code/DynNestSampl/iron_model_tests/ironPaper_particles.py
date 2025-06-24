import numpy as np
import time
import math
import matplotlib.pyplot as plt
import os


erosion_bins_per_10mass = 10  # Number of erosion bins per decade of mass
mass_min = 1e-11  # Minimum mass in kg
mass_max = 1e-8  # Maximum mass in kg
mass_index = 1.6  # Mass index for the distribution (2 for a power law)
eroded_mass = 2.945956677208338e-08  # Total mass eroded in kg
rho = 7000  # kg/m^3 (example density for rock/iron)
gamma_5_3 = 0.90274529295093375313996375552960671484470367431640625 # gamma(5/3)
m_dr = mass_min + (mass_max - mass_min) * mass_index

# folder where code is running
output_path = r'C:\Users\maxiv\WMPG-repoMAX\Code\DynNestSampl\iron_model_tests'
array_timegamma = []
### exponential distribution ###
start_time = time.time()

# Compute the mass bin coefficient
mass_bin_coeff = 10**(-1.0/erosion_bins_per_10mass)

# Compute the number of mass bins
k = int(1 + math.log10(mass_min/mass_max)/math.log10(mass_bin_coeff))

# Compute the number of the largest grains
if mass_index == 2:
    n0 = eroded_mass/(mass_max*k)
else:
    n0 = abs((eroded_mass/mass_max)*(1 - mass_bin_coeff**(2 - mass_index))/(1 - mass_bin_coeff**((2 - mass_index)*k)))


# Go though every mass bin
frag_children = []
leftover_mass = 0
for i in range(0, k):

    # Compute the mass of all grains in the bin (per grain)
    m_grain = mass_max*mass_bin_coeff**i

    # Compute the number of grains in the bin
    n_grains_bin = n0*(mass_max/m_grain)**(mass_index - 1) + leftover_mass/m_grain
    n_grains_bin_round = int(math.floor(n_grains_bin))

    # Compute the leftover mass
    leftover_mass = (n_grains_bin - n_grains_bin_round)*m_grain
    # Store result
    if n_grains_bin_round > 0:
        # plot the number of grains in each bin and the mass of the grains
        frag_children.append((m_grain, n_grains_bin_round))
        print(f"Mass bin {i}: m_grain = {m_grain:.2e} kg, n_grains_bin = {n_grains_bin:.2f}, n_grains_bin_round = {n_grains_bin_round}")

# stop the timer
end_time = time.time()
print("Power-law distribution:")
print(f"Time taken: {end_time - start_time:.4e} seconds")

# Final stats
total_fragments = sum(n for _, n in frag_children)
total_mass = sum(m * n for m, n in frag_children)
print(f"Total fragments: {total_fragments}")
print(f"Total mass (kg): {total_mass:.4e}")
print(f"Relative error: {abs(eroded_mass - total_mass)/eroded_mass:.2%}\n")


### fast gamma distribution ###
# reset the timer
start_time = time.time()

# Bin setup
mass_bin_coeff = 10 ** (-1.0 / erosion_bins_per_10mass)
k = int(1 + np.log10(mass_min / mass_max) / np.log10(mass_bin_coeff))
mass_bins = np.array([mass_max * (mass_bin_coeff ** i) for i in range(k)])
bin_widths = mass_bins * (1 - mass_bin_coeff)

# Mean mass from power-law
if mass_index == 1.0:
    m_mean = (mass_max - mass_min) / np.log(mass_max / mass_min)
elif mass_index == 2.0:
    m_mean = np.log(mass_max / mass_min) / (1 / mass_min - 1 / mass_max)
else:
    numerator = (mass_max**(2 - mass_index) - mass_min**(2 - mass_index)) / (2 - mass_index)
    denominator = (mass_max**(1 - mass_index) - mass_min**(1 - mass_index)) / (1 - mass_index)
    m_mean = numerator / denominator

# mass_index_dr = 0.05  # Mass index for the gamma distribution
# m_mean = mass_min + (mass_max - mass_min) * mass_index_dr

# Convert mean mass to mean diameter
D_mean = (6 * m_mean / (np.pi * rho))**(1/3)
s = (D_mean * gamma_5_3) ** 3
# print(f"Mean mass (kg): {m_mean:.4e}")
# print(f"Mean diameter (µm): { D_mean * 1e6:.4f}\n")

D = (6 * mass_bins / (np.pi * rho)) ** (1 / 3)
n_D = (3 * D**2 / s) * np.exp(-D**3 / s)
dD_dm = (1/3) * (6 / (np.pi * rho))**(1/3) * mass_bins**(-2/3)
n_m_raw = n_D * np.abs(dD_dm)

mass_per_bin_raw = n_m_raw * bin_widths * mass_bins
scaling = eroded_mass / np.sum(mass_per_bin_raw)
n_m_scaled = n_m_raw * scaling

# Integer grain assignment with leftover mass tracking
# int_counts = np.zeros_like(mass_bins, dtype=int)
leftover_mass = 0
frag_children_dr = []
for i in range(k):
    m_grain = mass_bins[i]
    expected_count = n_m_scaled[i] * bin_widths[i] + leftover_mass / m_grain
    n_grains_bin_round = int(math.floor(expected_count)) # int(expected_count)
    leftover_mass = (expected_count - n_grains_bin_round) * m_grain
    # Store results for fast gamma distribution
    if n_grains_bin_round > 0:
        frag_children_dr.append((m_grain, n_grains_bin_round))
        print(f"Mass bin {i}: m_grain = {m_grain:.2e} kg, n_grains_bin = {expected_count:.2f}, n_grains_bin_round = {n_grains_bin_round}")

# stop the timer
end_time = time.time()
print("Gamma-like distribution:")
print(f"Time taken: {end_time - start_time:.4e} seconds")

# Final stats
total_fragments = sum(n for _, n in frag_children_dr)
total_mass = sum(m * n for m, n in frag_children_dr)

print(f"Total fragments: {total_fragments}")
print(f"Total mass (kg): {total_mass:.4e}")
print(f"Relative error: {abs(eroded_mass - total_mass)/eroded_mass:.2%}\n")



# Plot the number of grains in each bin
fig, ax = plt.subplots()
mass_range = np.logspace(np.log10(mass_min), np.log10(mass_max), k)
# Compute unscaled theoretical gamma-like counts
n_gamma_bins = []
for m in mass_range:
    D = (6 * m / (np.pi * rho))**(1/3)
    n_D = (3 * D**2 / s) * np.exp(-D**3 / s)
    dD_dm = (1/3) * (6 / (np.pi * rho))**(1/3) * m**(-2/3)
    n_m = n_D * abs(dD_dm)
    bin_width = m * (1 - mass_bin_coeff)
    n_gamma_bins.append(n_m * bin_width)  # still unnormalized

# Normalize to match total number of grains from gamma fragment count
total_grains = sum(n for _, n in frag_children_dr)
scaling = total_grains / sum(n_gamma_bins)
n_gamma_bins_scaled = [n * scaling for n in n_gamma_bins]
ax.plot(mass_range, n_gamma_bins_scaled, linestyle='-.', color='bisque', label='Gamma-like expected number of grains')
ax.scatter([m[0] for m in frag_children_dr], [m[1] for m in frag_children_dr],  s=10, color='darkorange', label='Number of grains Gamma-like (with leftover mass)')# marker='o', linestyle='-', color='blue', label='Number of grains gamma distribution')
ax.plot(mass_range, [n0*(mass_max/m)**(mass_index - 1) for m in mass_range], color='deepskyblue', label='Powerlaw Expected number of grains', linestyle='--')
# plot the number of grains in each bin as a scatter plot
ax.scatter([m[0] for m in frag_children], [m[1] for m in frag_children], s=10, color='blue', label='Number of grains Powerlaw (with leftover mass)')  # marker='o', linestyle='-', color='blue', label='Number of grains Powerlaw (with leftover mass)')
ax.set_xscale('log')
# ax.set_yscale('log')
# show laegend
ax.legend()

# take the y axis limits from the scatter plot
ymin, ymax = ax.get_ylim()
# limit the y axis to the range of the scatter plot
ax.set_ylim(0.9, ymax)


# grid on
ax.grid(True)#, which='both', linestyle='--', linewidth=0.5)
ax.set_xlabel('Grain mass (kg)')
ax.set_ylabel('Number of grains')
ax.set_title('Powerlaw vs Gamma-like for mass index '+str(mass_index))
# plt.show()\

# Save the figure
fig.savefig(output_path+os.sep+'number_of_grains_in_each_mass_bin_s'+str(mass_index)+'.png', dpi=300, bbox_inches='tight')

