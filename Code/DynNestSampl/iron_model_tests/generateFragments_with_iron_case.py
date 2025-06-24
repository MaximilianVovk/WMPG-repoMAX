import math
import copy
import numpy as np
from DynNestSampl.iron_model_tests.ironPaper_particles import getErosionCoeff


def generateFragments(const, frag_parent, eroded_mass, mass_index, mass_min, mass_max, keep_eroding=False,
    disruption=False, iron_model=False):
    """ Given the parent fragment, fragment it into daughter fragments using a power law mass distribution.

    Masses are binned and one daughter fragment may represent several fragments/grains, which is specified 
    with the n_grains atribute.

    Arguments:
        const: [object] Constants instance.
        frag_parent: [object] Fragment instance, the parent fragment.
        eroded_mass: [float] Mass to be distributed into daughter fragments. 
        mass_index: [float] Mass index to use to distribute the mass.
        mass_min: [float] Minimum mass bin (kg).
        mass_max: [float] Maximum mass bin (kg).

    Keyword arguments:
        keep_eroding: [bool] Whether the daughter fragments should keep eroding.
        disruption: [bool] Indicates that the disruption occured, uses a separate erosion parameter for
            disrupted daughter fragments.
        iron_model: [bool] Whether to use the iron model for mass distribution.

    Return:
        frag_children: [list] A list of Fragment instances - these are the generated daughter fragments.

    """

    # Compute the mass bin coefficient
    mass_bin_coeff = 10**(-1.0/const.erosion_bins_per_10mass)

    # Compute the number of mass bins
    k = int(1 + math.log10(mass_min/mass_max)/math.log10(mass_bin_coeff))

    if iron_model:
        # Compute the number of needed bins for the gamma distribution
        mass_bins = np.array([mass_max * (mass_bin_coeff ** i) for i in range(k)])
        bin_widths = mass_bins * (1 - mass_bin_coeff)        

        # Compute the expected value from the mass power-law distribution 
        log_range = math.log(mass_max / mass_min)
        if mass_index == 1.0:
            # For mass_index = 1, the mean is the arithmetic mean
            m_mean = (mass_max - mass_min) / log_range
        elif mass_index == 2.0:
            # For mass_index = 2, the mean is the harmonic mean
            m_mean = log_range / (1.0 / mass_min - 1.0 / mass_max)
        else:
            # For other mass indices, compute the mean using the formula, computing each step separatelly to save time
            a = 2 - mass_index
            b = 1 - mass_index
            m_max_a = mass_max**a
            m_min_a = mass_min**a
            m_max_b = mass_max**b
            m_min_b = mass_min**b

            num = (m_max_a - m_min_a) / a
            den = (m_max_b - m_min_b) / b
            m_mean = num / den

        # Convert mean mass to mean diameter
        D_mean = (6 * m_mean / (math.pi * const.rho_grain))**(1/3)
        # print(f"Mean mass (kg): {m_mean:.4g}")
        # print(f"Mean diameter (µm): { D_mean * 1e6:.4g}")

        # The gamma function value for 5/3, used in the gamma distribution
        gamma_5_3 = 0.90274529295093375313996375552960671484470367431640625  # gamma(5/3)
        s = (D_mean * gamma_5_3) ** 3

        # covert the diameter to mass
        Diameter = (6 * mass_bins / (math.pi * const.rho_grain)) ** (1 / 3)

        # Compute the number of grains in the bin for diameter distribution
        n_D = (3 * Diameter **2 / s) * np.exp(-Diameter **3 / s)

        # Compute the derivative of the diameter with respect to mass
        dD_dm = (1/3) * (6 / (math.pi * const.rho_grain))**(1/3) * mass_bins**(-2/3)

        # Compute the number of grains in the bin for unit mass distribution
        n_m_raw = n_D * np.abs(dD_dm)

        # Compute the mass per bin from the unit mass distribution
        mass_per_bin_raw = n_m_raw * bin_widths * mass_bins

        # Scale the number of grains in the bin to match the eroded mass
        scaling = eroded_mass / np.sum(mass_per_bin_raw)

        # Compute the mass of all grains in the bin (per grain)
        n_m_scaled = n_m_raw * scaling

    else:
        # Compute the number of the largest grains
        if mass_index == 2:
            n0 = eroded_mass/(mass_max*k)
        else:
            n0 = abs((eroded_mass/mass_max)*(1 - mass_bin_coeff**(2 - mass_index))/(1 - mass_bin_coeff**((2 - mass_index)*k)))


    # Go though every mass bin
    frag_children = []
    leftover_mass = 0
    for i in range(0, k):

        if iron_model:
            # extract the mass of the grain in the bin
            m_grain = mass_bins[i]

            # Compute the number of grains in the bin
            n_grains_bin = n_m_scaled[i] * bin_widths[i] + leftover_mass/m_grain
            n_grains_bin_round = int(math.floor(n_grains_bin)) # int(expected_count)

            # Compute the leftover mass
            leftover_mass = (n_grains_bin - n_grains_bin_round)*m_grain

        else:
            # Compute the mass of all grains in the bin (per grain)
            m_grain = mass_max*mass_bin_coeff**i

            # Compute the number of grains in the bin
            n_grains_bin = n0*(mass_max/m_grain)**(mass_index - 1) + leftover_mass/m_grain
            n_grains_bin_round = int(math.floor(n_grains_bin))

            # Compute the leftover mass
            leftover_mass = (n_grains_bin - n_grains_bin_round)*m_grain

        # If there are any grains to erode, erode them
        if n_grains_bin_round > 0:

            # Init the new fragment with params of the parent
            frag_child = copy.deepcopy(frag_parent)

            # Assign the number of grains this fragment stands for (make sure to preserve the previous value
            #   if erosion is done for more fragments)
            frag_child.n_grains *= n_grains_bin_round

            # Assign the grain mass
            frag_child.m = m_grain
            frag_child.m_init = m_grain

            frag_child.active = True
            frag_child.main = False
            frag_child.disruption_enabled = False

            # Indicate that the fragment is a grain
            if (not keep_eroding) and (not disruption):
                frag_child.grain = True

            # Set the erosion coefficient value (disable in grain, only larger fragments)
            if keep_eroding:
                frag_child.erosion_enabled = True

                # If the disruption occured, use a different erosion coefficient for daguhter fragments
                if disruption:
                    frag_child.erosion_coeff = const.disruption_erosion_coeff
                else:
                    frag_child.erosion_coeff = getErosionCoeff(const, frag_parent.h)

            else:
                # Compute the grain density and shape-density coeff
                frag_child.rho = const.rho_grain
                frag_child.updateShapeDensityCoeff()

                frag_child.erosion_enabled = False
                frag_child.erosion_coeff = 0


            # Give every fragment a unique ID
            frag_child.id = const.total_fragments
            const.total_fragments += 1

            frag_children.append(frag_child)


    return frag_children, const
