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

print(f"total eroded mass: {eroded_mass} kg")
for mass_index in [1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3]:
    # folder where code is running
    array_time_power = []
    array_time_gamma = []
    for time_index in range(1000):
        ##########################################################################################################################

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
                # print(f"Mass bin {i}: m_grain = {m_grain:.2e} kg, n_grains_bin = {n_grains_bin:.2f}, n_grains_bin_round = {n_grains_bin_round}")

        # stop the timer
        end_time = time.time()
        array_time_power.append(end_time - start_time)

        ##########################################################################################################################
        # reset the timer
        start_time = time.time()

        # Bin setup
        mass_bin_coeff = 10 ** (-1.0 / erosion_bins_per_10mass)
        # k = int(1 + np.log10(mass_min / mass_max) / np.log10(mass_bin_coeff))
        k = int(1 + math.log10(mass_min/mass_max)/math.log10(mass_bin_coeff))
        mass_bins = np.array([mass_max * (mass_bin_coeff ** i) for i in range(k)])
        bin_widths = mass_bins * (1 - mass_bin_coeff)

        # log_range = np.log(mass_max / mass_min)
        log_range = math.log(mass_max / mass_min)

        if mass_index == 1.0:
            m_mean = (mass_max - mass_min) / log_range
        elif mass_index == 2.0:
            m_mean = log_range / (1.0 / mass_min - 1.0 / mass_max)
        else:
            a = 2 - mass_index
            b = 1 - mass_index
            m_max_a = mass_max**a
            m_min_a = mass_min**a
            m_max_b = mass_max**b
            m_min_b = mass_min**b

            num = (m_max_a - m_min_a) / a
            den = (m_max_b - m_min_b) / b
            m_mean = num / den


        # mass_index_dr = 0.05  # Mass index for the gamma distribution
        # m_mean = mass_min + (mass_max - mass_min) * mass_index_dr

        # Convert mean mass to mean diameter
        D_mean = (6 * m_mean / (math.pi * rho))**(1/3)
        s = (D_mean * gamma_5_3) ** 3
        # print(f"Mean mass (kg): {m_mean:.4e}")
        # print(f"Mean diameter (µm): { D_mean * 1e6:.4f}\n")

        Diameter = (6 * mass_bins / (math.pi * rho)) ** (1 / 3)
        n_D = (3 * Diameter **2 / s) * np.exp(-Diameter **3 / s)
        dD_dm = (1/3) * (6 / (math.pi * rho))**(1/3) * mass_bins**(-2/3)
        n_m_raw = n_D * np.abs(dD_dm)

        mass_per_bin_raw = n_m_raw * bin_widths * mass_bins
        scaling = eroded_mass / np.sum(mass_per_bin_raw)
        n_m_scaled = n_m_raw * scaling

        # Integer grain assignment with leftover mass tracking
        # int_counts = np.zeros_like(mass_bins, dtype=int)
        leftover_mass = 0
        frag_children_dr = []
        for i in range(k):
            expected_count = n_m_scaled[i] * bin_widths[i] + leftover_mass / mass_bins[i]
            n_grains_bin_round = int(math.floor(expected_count)) # int(expected_count)
            leftover_mass = (expected_count - n_grains_bin_round) * mass_bins[i]
            # Store results for fast gamma distribution
            if n_grains_bin_round > 0:
                frag_children_dr.append((mass_bins[i], n_grains_bin_round))
                # print(f"Mass bin {i}: m_grain = {mass_bins[i]:.2e} kg, n_grains_bin = {expected_count:.2f}, n_grains_bin_round = {int_counts[i]}")

        # stop the timer
        end_time = time.time()
        array_time_gamma.append(end_time - start_time)
        ##########################################################################################################################

    # Print the times results
    print(f"mass index: {mass_index}")
    # print(f"mass index: {mass_index} for {len(array_time_power)} total number of iterations")
    # print(f"   Average time for power law distribution: {np.mean(array_time_power):.6e} seconds")
    # print(f"   Average time for gamma distribution: {np.mean(array_time_gamma):.6e} seconds")
    # print(f"   Total time for power law distribution: {np.sum(array_time_power):.6e} seconds")
    # print(f"   Total time for gamma distribution: {np.sum(array_time_gamma):.6e} seconds")
    # Final stats
    total_fragments = sum(n for _, n in frag_children)
    total_mass = sum(m * n for m, n in frag_children)
    print(f"Power-law distribution:")
    print(f"   Total fragments: {total_fragments}")
    print(f"   Total mass (kg): {total_mass}")
    print(f"   Difference in mass (kg): {abs(total_mass - eroded_mass):.4e}")
    print(f"   Relative error: {abs(eroded_mass - total_mass)/eroded_mass:.4%}")
    total_fragments = sum(n for _, n in frag_children_dr)
    total_mass = sum(m * n for m, n in frag_children_dr)
    print(f"Gamma-like distribution:")
    print(f"   Total fragments: {total_fragments}")
    print(f"   Total mass (kg): {total_mass}")
    print(f"   Difference in mass (kg): {abs(total_mass - eroded_mass):.4e}")
    print(f"   Relative error: {abs(eroded_mass - total_mass)/eroded_mass:.4%}\n")
    print(f"Mean mass (kg): {m_mean:.4e}")
    print(f"Mean diameter (µm): { D_mean * 1e6:.4f}\n")

