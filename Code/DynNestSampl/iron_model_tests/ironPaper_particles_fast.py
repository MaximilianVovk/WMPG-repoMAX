import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma as gamma_func
from concurrent.futures import ThreadPoolExecutor

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

    return mass_bins, int_counts

# Parallel execution for multiple mass indices
mass_indices = [1.5, 1.6, 1.8, 2.0]
results = {}

with ThreadPoolExecutor() as executor:
    futures = {
        idx: executor.submit(compute_gamma_distribution_with_leftover, mass_index=idx)
        for idx in mass_indices
    }

    for idx, future in futures.items():
        mass_bins, int_counts = future.result()
        results[idx] = (mass_bins, int_counts)

# Plotting
fig, ax = plt.subplots(figsize=(8, 6))

for idx in mass_indices:
    mass_bins, int_counts = results[idx]
    ax.plot(mass_bins, int_counts, marker='o', linestyle='-', label=f's = {idx}')

ax.set_xscale('log')
ax.set_xlabel('Mass (kg)')
ax.set_ylabel('Number of grains')
ax.set_title('Gamma Distribution Fragment Counts with Leftover Mass Handling')
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
ax.legend()
plt.tight_layout()

plt.show()
