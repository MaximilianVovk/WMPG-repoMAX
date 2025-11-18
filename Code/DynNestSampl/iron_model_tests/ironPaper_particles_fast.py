import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma as gamma_func
from concurrent.futures import ThreadPoolExecutor
import os

output_path = r'C:\Users\maxiv\WMPG-repoMAX\Code\DynNestSampl\iron_model_tests'

def compute_gamma_distribution_with_leftover(
    erosion_bins_per_10mass=10,
    mass_min=1e-11,
    mass_max=1e-8,
    mass_index=1.6,
    eroded_mass=2.945956677208338e-08,
    rho=7000,
    gamma_5_3=0.9027452929509338
):
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

    D_mean = (6 * m_mean / (np.pi * rho))**(1/3)
    s = (D_mean * gamma_5_3) ** 3

    D = (6 * mass_bins / (np.pi * rho)) ** (1 / 3)
    n_D = (3 * D**2 / s) * np.exp(-D**3 / s)
    dD_dm = (1/3) * (6 / (np.pi * rho))**(1/3) * mass_bins**(-2/3)
    n_m_raw = n_D * np.abs(dD_dm)

    mass_per_bin_raw = n_m_raw * bin_widths * mass_bins
    scaling = eroded_mass / np.sum(mass_per_bin_raw)
    n_m_scaled = n_m_raw * scaling

    # Integer grain assignment with leftover mass tracking
    int_counts = np.zeros_like(mass_bins, dtype=int)
    leftover_mass = 0.0

    for i in range(k):
        expected_count = n_m_scaled[i] * bin_widths[i] + leftover_mass / mass_bins[i]
        int_counts[i] = int(expected_count)
        leftover_mass = (expected_count - int_counts[i]) * mass_bins[i]

    return mass_bins, int_counts, m_mean

def compute_pwerlaw_distribution_with_leftover(
    erosion_bins_per_10mass=10,
    mass_min=1e-11,
    mass_max=1e-8,
    mass_index=1.6,
    eroded_mass=2.945956677208338e-08,
):

    # Compute the mass bin coefficient
    mass_bin_coeff = 10**(-1.0/erosion_bins_per_10mass)

    # Compute the number of mass bins
    k = int(1 + np.log10(mass_min/mass_max)/np.log10(mass_bin_coeff))

    # Compute the number of the largest grains
    if mass_index == 2:
        n0 = eroded_mass/(mass_max*k)
    else:
        n0 = abs((eroded_mass/mass_max)*(1 - mass_bin_coeff**(2 - mass_index))/(1 - mass_bin_coeff**((2 - mass_index)*k)))


    # Go though every mass bin
    m_grains = []
    int_counts = []
    leftover_mass = 0
    for i in range(0, k):

        # Compute the mass of all grains in the bin (per grain)
        m_grain = mass_max*mass_bin_coeff**i

        # Compute the number of grains in the bin
        n_grains_bin = n0*(mass_max/m_grain)**(mass_index - 1) + leftover_mass/m_grain
        n_grains_bin_round = int(np.floor(n_grains_bin))

        # Compute the leftover mass
        leftover_mass = (n_grains_bin - n_grains_bin_round)*m_grain
        int_counts.append(n_grains_bin_round)
        m_grains.append(m_grain)
    return m_grains, int_counts

# Parallel execution for multiple mass indices
# mass_indices = [1, 1.25, 1.5, 1.75]
mass_indices = [2, 2.25, 2.5, 2.75, 3]
# mass_indices = [2]
results = {}
results_pow = {}
with ThreadPoolExecutor() as executor:
    futures = {
        idx: executor.submit(compute_gamma_distribution_with_leftover, mass_index=idx)
        for idx in mass_indices
    }

    for idx, future in futures.items():
        mass_bins, int_counts, m_mean = future.result()
        results[idx] = (mass_bins, int_counts, m_mean)

    futures_powerlaw = {
        idx: executor.submit(compute_pwerlaw_distribution_with_leftover, mass_index=idx)
        for idx in mass_indices
    }

    for idx, future in futures_powerlaw.items():
        mass_bins_pow, int_counts_pow = future.result()
        results_pow[idx] = (mass_bins_pow, int_counts_pow)

# Plotting
fig, ax = plt.subplots(figsize=(8, 6))


for idx in mass_indices:
    mass_bins, int_counts, m_mean = results[idx]
    # ax.axvline(m_mean, color='r', linestyle='--', linewidth=2, label=r'$\langle m \rangle$', zorder=0)
    ax.plot(mass_bins, int_counts, marker='o', linestyle='-', label=f'Gamma s = {idx}')

# reset the color cycle for the next plot
ax.set_prop_cycle(None)
for idx in mass_indices:
    mass_bins_pow, int_counts_pow = results_pow[idx]
    ax.plot(mass_bins_pow, int_counts_pow, marker='x', linestyle='--', label=f'Power-law s = {idx}')

# make the y axis up to the biggest number of grain of gamma distribution mass_bins
max_y = max(max(int_counts) for _, int_counts, _ in results.values())
ax.set_ylim(0, max_y * 1.1)

ax.set_xscale('log')
ax.set_xlabel('Mass (kg)', fontsize=15)
ax.set_ylabel('Number of grains', fontsize=15)
# increase te size of the number in x and y axis
ax.tick_params(axis='both', which='major', labelsize=13)
# ax.set_title('Gamma Distribution Fragment Counts with Leftover Mass Handling')
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
# make the legend 2 columns
# ax.legend(title='Distribution Type', loc='upper right', ncol=2)
# put it top right
ax.legend(ncol=2, fontsize=13, loc='upper right')
# ax.legend(fontsize=15)
# save the figure
plt.tight_layout()
plt.savefig(output_path + os.sep + 'gamma_distribution_fragment_counts_s'+str(np.min(mass_indices))+'-'+str(np.max(mass_indices))+'.png', dpi=300)