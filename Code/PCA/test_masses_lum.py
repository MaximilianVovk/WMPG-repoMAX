import numpy as np
import matplotlib.pyplot as plt
import numpy as np

# Given parameters
A = -12.59
B = 5.58
C = -0.17
D = -1.21

def compute_mass(v, tau):
    """
    Compute the mass of the meteoroid given velocity and luminous efficiency.
    """
    # Calculate the intermediate variable E
    E = np.log(tau) - A - B * np.log(v) - C * (np.log(v))**3
    E_over_D = E / D
    # Ensure E_over_D is within the valid range for arctanh
    E_over_D = np.clip(E_over_D, -0.999999, 0.999999)
    # Calculate ln(m × 10^6)
    ln_m_times_1e6 = np.arctanh(E_over_D) / 0.2
    # Calculate the mass m
    m = np.exp(ln_m_times_1e6) / 1e6
    return m

def compute_tau(v, m):
    """
    Compute the luminous efficiency given velocity and mass.
    """
    ln_tau = A + B * np.log(v) + C * (np.log(v))**3 + D * np.tanh(0.2 * np.log(m * 1e6))
    tau = np.exp(ln_tau)
    return tau

def assess_mass(v, tau_old, tol=1e-6, max_iter=100, iteration=0):
    """
    Recursively assess the meteoroid mass until the luminous efficiency converges.
    """
    if iteration >= max_iter:
        print("Maximum iterations reached without convergence.")
        return None
    m = compute_mass(v, tau_old)
    tau_new = compute_tau(v, m)
    print(f"Iteration {iteration}: τ = {tau_new:.6f}, m = {m:.6e} kg")
    if abs(tau_new - tau_old) < tol:
        return m
    else:
        return assess_mass(v, tau_new, tol, max_iter, iteration + 1)



# Example usage
if __name__ == "__main__":
    # Luminous efficiencies
    tau_values = [0.6, 0.7, 0.8, 0.9, 1.0]
    # Velocities from 5 km/s to 70 km/s
    velocities = np.linspace(5, 70, 500)
    masses_dict = {tau: [] for tau in tau_values}
    for tau in tau_values:
        for v in velocities:
            m = compute_mass(v, tau)
            print(f"τ = {tau}, v = {v:.2f} km/s, m = {m:.6e} kg")
            masses_dict[tau].append(m)



    # Plotting
    plt.figure(figsize=(10, 6))
    for tau in tau_values:
        plt.plot(velocities, masses_dict[tau], label=f'τ = {tau}')
    plt.title('Meteoroid Mass m vs Velocity v')
    plt.xlabel('Velocity (km/s)')
    plt.ylabel('Mass (kg)')
    # make y axis log scale
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.show()