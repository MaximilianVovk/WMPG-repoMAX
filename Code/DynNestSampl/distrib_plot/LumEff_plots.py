import numpy as np
import matplotlib.pyplot as plt

def luminous_efficiency_tauI_Hill2005(v_kms: float) -> float:
    """
    Luminous efficiency τ_I(v) from Hill et al. (2005):
    - ζ(v) piecewise from Eqs. (11)-(14)
    - τ_I from Eq. (10) using epsilon/mu = 7.668e6 J/kg

    Input:
        v_kms : scalar speed in km/s

    Returns:
        tau_I : luminous efficiency (dimensionless, visual band)
    """

    zeta = np.full_like(v_kms, np.nan, dtype=float)

    m1 = (v_kms <= 20.0)
    m2 = (v_kms > 20.0) & (v_kms <= 60.0)
    m3 = (v_kms > 60.0) & (v_kms <= 100.0)
    m4 = (v_kms > 100.0)

    # v <= 20
    zeta[m1] = (
        -0.0021887 * v_kms[m1]**2
        + 0.00042903 * v_kms[m1]**3
        - 1.2447e-05 * v_kms[m1]**4
    )

    # 20 < v <= 60
    zeta[m2] = 0.01333 * v_kms[m2]**1.25

    # 60 < v <= 100
    zeta[m3] = (
        -12.835
        + 0.67672 * v_kms[m3]
        - 0.01163076 * v_kms[m3]**2
        + 9.191681e-05 * v_kms[m3]**3
        - 2.7465805e-07 * v_kms[m3]**4
    )

    # v > 100
    zeta[m4] = 1.615 + 0.013725 * v_kms[m4]
    # ---- τ_I from Eq. (10): use v in m/s here ----
    eps_over_mu = 7.668e6  # J/kg (mean value used in the paper)
    v_mps = v_kms * 1000.0

    tau_I = 2.0 * eps_over_mu * zeta / (v_mps**2)
    return tau_I * 100.0  # convert to percent


def luminous_efficiency_tau_PecinaCeplecha1983(v_kms, P0m_W):
    """
    Return luminous efficiency from the 1983 fit:
      - tau_percent: in %

    Inputs
    ------
    v_kms : float or array
        Speed in km/s.
    P0m_W : float or array
        Zero-magnitude meteor radiant power for camera/bandpass (Watts),
        defined at 100 km.
    """
    v = np.asarray(v_kms, dtype=float)
    if np.any(v < 3.0):
        raise ValueError("Fit is stated for v >= 3 km/s.")

    logv = np.log10(v)

    # Eq. (40) for 3 <= v < 25.372 km/s; and high-velocity assumption for v >= 25.372
    logtau_poly = (-12.834
                   - 10.307*logv
                   + 22.522*(logv**2)
                   - 16.125*(logv**3)
                   +  3.922*(logv**4))
    logtau_hi = logv - 13.70

    logtau = np.where(v < 25.372, logtau_poly, logtau_hi)
    tau_paper_cgs = 10.0**logtau  # ~ s/erg (because I is dimensionless)

    # Convert using P0m in erg/s (1 W = 1e7 erg/s)
    tau_dimless = tau_paper_cgs * (P0m_W * 1e7)
    tau_percent = 100.0 * tau_dimless

    return tau_percent


def tau_borovicka2020(v_kms, m_kg):
    """
    Borovička et al. (2020) luminous efficiency τ(v,m).
    - Input: v in km/s, m in kg
    - Output:
        tau_percent  : τ in percent (%)
        tau_fraction : τ as dimensionless fraction (SI)
    Notes:
      Uses Eqs. (3) and (4) (ln = natural log, tanh = hyperbolic tangent).
    """
    v = np.asarray(v_kms, dtype=float)
    m = np.asarray(m_kg, dtype=float)

    if np.any(v <= 0) or np.any(m <= 0):
        raise ValueError("v_kms and m_kg must be > 0.")
    if np.any(v < 3.0):
        raise ValueError("Model is stated for v >= 3 km/s.")

    lnv = np.log(v)
    lnm = np.log(m)

    # Eq. (3): v < 25.372 km/s
    ln_tau_pct_poly = (0.567
                       - 10.307*lnv
                       + 9.781*(lnv**2)
                       - 3.0414*(lnv**3)
                       + 0.3213*(lnv**4)
                       + 0.347*np.tanh(0.38*lnm))

    # Eq. (4): v >= 25.372 km/s
    ln_tau_pct_hi = (-1.4286
                     + lnv
                     + 0.347*np.tanh(0.38*lnm))

    ln_tau_pct = np.where(v < 25.372, ln_tau_pct_poly, ln_tau_pct_hi)

    tau_percent = np.exp(ln_tau_pct)      # τ in %
    # tau_fraction = tau_percent / 100.0    # dimensionless (SI)

    return tau_percent

import numpy as np

def tau_vida_icarus2024(v_kms, m_kg):
    """
    Vida et al. (Icarus 2024) luminous efficiency model (Eq. 3, Section 3.1).

    Inputs
    ------
    v_kms : float or array
        Meteoroid speed in km/s.
    m_kg : float or array
        Meteoroid mass in kg.

    Returns
    -------
    tau_percent : ndarray
        Luminous efficiency in percent (%).
    tau_fraction : ndarray
        Luminous efficiency as a dimensionless fraction (SI), i.e. %/100.

    Notes
    -----
    Uses natural log (ln). τ is defined in the paper as "dimensionless luminous
    efficiency in percent". v must be > 0, m must be > 0.
    """
    v = np.asarray(v_kms, dtype=float)
    m = np.asarray(m_kg, dtype=float)

    if np.any(v <= 0) or np.any(m <= 0):
        raise ValueError("v_kms and m_kg must be > 0.")

    lnv = np.log(v)
    # tanh(0.2 * ln(m * 1e6))
    mass_term = np.tanh(0.2 * np.log(m * 1e6))

    A, B, C, D = -12.59, 5.58, -0.17, -1.21

    ln_tau_percent = A + B * lnv + C * (lnv ** 3) + D * mass_term

    tau_percent = np.exp(ln_tau_percent)
    # tau_fraction = tau_percent / 100.0

    return tau_percent

# create a plot for spedd ranging from 11 km/s to 72 km/s and with mass ranging from 0.01 to 0.000001 kg with order of magnitude steps and a P_0m of 935 W
if __name__ == "__main__":
    v_vals = np.linspace(11, 72, 100)
    m_vals = [10**-6, 10**-7, 10**-8, 10**-9, 10**-10]  # kg
    P_0m = [840,935.0,1000,1500,2500]  # W

    plt.figure(figsize=(12, 6))

    # cycle though the C0 colors
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for i, m in enumerate(m_vals):
        tau_vals = tau_vida_icarus2024(v_vals, m)
        plt.plot(v_vals, tau_vals, label=f'Vida et al. (2024) m = $10^{{{int(np.log10(m))}}}$ kg', color=colors[i % len(colors)], ls='-')
    
    for i, m in enumerate(m_vals):
        tau_vals = tau_borovicka2020(v_vals, m)
        plt.plot(v_vals, tau_vals, label=f'Borovička et al. (2020) m = $10^{{{int(np.log10(m))}}}$ kg', color=colors[i % len(colors)], ls='--')
    
    tau_hill_vals = luminous_efficiency_tauI_Hill2005(v_vals)
    plt.plot(v_vals, tau_hill_vals, label='Hill et al. (2005)', color='black', ls=':')
    
    # cycle though a different set of colors for the Pecina & Ceplecha lines
    colors = plt.cm.viridis(np.linspace(0, 1, len(P_0m)))
    for i, P_0m_s in enumerate(P_0m):
        tau_pecina_vals = luminous_efficiency_tau_PecinaCeplecha1983(v_vals, P_0m_s)
        plt.plot(v_vals, tau_pecina_vals, label=f'Pecina & Ceplecha (1983) $P_{{0M}} = {P_0m_s}$ W', color=colors[i % len(colors)], ls='-.')

    # plt.title('Luminous Efficiency τ(v,m) from Vida et al. (Icarus 2024)')
    plt.xlabel('Velocity (km/s)')
    plt.ylabel('Luminous Efficiency τ (%)')
    plt.yscale('log')
    plt.grid(True, which='both', ls='--', lw=0.5)
    # put the legend outside the plot on the right side
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()
