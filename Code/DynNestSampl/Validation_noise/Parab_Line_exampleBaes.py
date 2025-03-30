"""
Example script demonstrating how to use dynamic nested sampling (via `dynesty`)
to compare a linear model, a second-order polynomial model, and a third-order
polynomial model on synthetic data. We also illustrate how to visualize
the posterior-derived uncertainty in the model fits.

Author: Maximilian Vovk
Modified: 2025-02-18
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

import dynesty
from dynesty import DynamicNestedSampler

# -------------------------------------------------------------------------
# 1. Generate synthetic data from a true second-order polynomial with noise
# -------------------------------------------------------------------------
np.random.seed(42)

N = 50                     # Number of data points
x_data = np.linspace(0, 10, N)
true_a0 = 1.0              # True constant term
true_a1 = 0.5              # True linear coefficient
true_a2 = 0.1              # True quadratic coefficient
true_sigma = 1.0           # Gaussian noise standard deviation

# True model: y = a0 + a1*x + a2*x^2
y_true = true_a0 + true_a1 * x_data + true_a2 * x_data**2
noise = np.random.normal(0.0, true_sigma, size=N)
y_data = y_true + noise

# -------------------------------------------------------------------------
# 2. Define the log-likelihood functions
# -------------------------------------------------------------------------
def loglike_linear(params):
    """
    Log-likelihood for a linear model: y = m*x + b
    params = [m, b]
    """
    m, b = params
    y_model = m * x_data + b
    resid = y_data - y_model
    return -0.5 * np.sum(
        np.log(2.0 * np.pi * true_sigma**2) + (resid**2 / true_sigma**2)
    )

def loglike_parabola(params):
    """
    Log-likelihood for a second-order polynomial: y = a*x^2 + b*x + c
    params = [a, b, c]
    """
    a, b, c = params
    y_model = a * x_data**2 + b * x_data + c
    resid = y_data - y_model
    return -0.5 * np.sum(
        np.log(2.0 * np.pi * true_sigma**2) + (resid**2 / true_sigma**2)
    )

def loglike_cubic(params):
    """
    Log-likelihood for a third-order polynomial: y = d*x^3 + a*x^2 + b*x + c
    params = [d, a, b, c]
    """
    d, a, b, c = params
    y_model = d * x_data**3 + a * x_data**2 + b * x_data + c
    resid = y_data - y_model
    return -0.5 * np.sum(
        np.log(2.0 * np.pi * true_sigma**2) + (resid**2 / true_sigma**2)
    )

# -------------------------------------------------------------------------
# 3. Define the prior transforms for each model
# -------------------------------------------------------------------------
def prior_transform_linear(uparams):
    """
    Convert samples from unit cube to parameters:
    (m, b) in some broad range, e.g. m in [-10, 10], b in [-10, 10]
    """
    um, ub = uparams  # each in [0,1]
    m = -10.0 + 20.0 * um
    b = -10.0 + 20.0 * ub
    return np.array([m, b])

def prior_transform_parabola(uparams):
    """
    Convert samples from unit cube to parameters:
    (a, b, c) in some broad range, e.g. each in [-10, 10]
    """
    ua, ub, uc = uparams
    a = -10.0 + 20.0 * ua
    b = -10.0 + 20.0 * ub
    c = -10.0 + 20.0 * uc
    return np.array([a, b, c])

def prior_transform_cubic(uparams):
    """
    Convert samples from unit cube to parameters:
    (d, a, b, c) each in [-10, 10]
    """
    ud, ua, ub, uc = uparams
    d = -10.0 + 20.0 * ud
    a = -10.0 + 20.0 * ua
    b = -10.0 + 20.0 * ub
    c = -10.0 + 20.0 * uc
    return np.array([d, a, b, c])

# -------------------------------------------------------------------------
# 4. Run dynamic nested sampling for each model
# -------------------------------------------------------------------------
# --- Linear model
dsampler_linear = DynamicNestedSampler(
    loglike_linear,
    prior_transform_linear,
    ndim=2,
    bound='multi',
    sample='rwalk'
)
dsampler_linear.run_nested(print_progress=True)
results_linear = dsampler_linear.results

# --- Parabolic model
dsampler_parabola = DynamicNestedSampler(
    loglike_parabola,
    prior_transform_parabola,
    ndim=3,
    bound='multi',
    sample='rwalk'
)
dsampler_parabola.run_nested(print_progress=True)
results_parabola = dsampler_parabola.results

# --- Cubic model
dsampler_cubic = DynamicNestedSampler(
    loglike_cubic,
    prior_transform_cubic,
    ndim=4,
    bound='multi',
    sample='rwalk'
)
dsampler_cubic.run_nested(print_progress=True)
results_cubic = dsampler_cubic.results

# -------------------------------------------------------------------------
# 5. Extract the Bayesian evidence and best-fit parameters
# -------------------------------------------------------------------------
logZ_linear = results_linear.logz[-1]     
logZerr_linear = results_linear.logzerr[-1]
logZ_parabola = results_parabola.logz[-1]
logZerr_parabola = results_parabola.logzerr[-1]
logZ_cubic = results_cubic.logz[-1]
logZerr_cubic = results_cubic.logzerr[-1]

print("=== Linear Model (m*x + b) ===")
print(f"Estimated ln(Evidence): {logZ_linear:.2f} +/- {logZerr_linear:.2f}")
print(f"Evidence: {np.exp(logZ_linear):.2g}")

print("\n=== Parabolic Model (a*x^2 + b*x + c) ===")
print(f"Estimated ln(Evidence): {logZ_parabola:.2f} +/- {logZerr_parabola:.2f}")
print(f"Evidence: {np.exp(logZ_parabola):.2g}")

print("\n=== Cubic Model (d*x^3 + a*x^2 + b*x + c) ===")
print(f"Estimated ln(Evidence): {logZ_cubic:.2f} +/- {logZerr_cubic:.2f}")
print(f"Evidence: {np.exp(logZ_cubic):.2g}")

# Compare them via pairwise Bayes factors
delta_logZ_parabola_linear = logZ_parabola - logZ_linear
delta_logZ_cubic_linear = logZ_cubic - logZ_linear
delta_logZ_cubic_parabola = logZ_cubic - logZ_parabola

print(f"\nBayes factor (parabola vs. linear) ~ exp({delta_logZ_parabola_linear:.2f})"
      f" = {np.exp(delta_logZ_parabola_linear):.2g}")
print(f"Bayes factor (cubic vs. linear)    ~ exp({delta_logZ_cubic_linear:.2f})"
      f" = {np.exp(delta_logZ_cubic_linear):.2g}")
print(f"Bayes factor (cubic vs. parabola)  ~ exp({delta_logZ_cubic_parabola:.2f})"
      f" = {np.exp(delta_logZ_cubic_parabola):.2g}")

# -------------------------------------------------------------------------
# 6. Posterior Sampling: Generate model predictions for plotting uncertainties
# -------------------------------------------------------------------------
n_samples_band = 200

# Random draws from each posterior
idx_linear = np.random.choice(len(results_linear.samples), n_samples_band, replace=False)
idx_parabola = np.random.choice(len(results_parabola.samples), n_samples_band, replace=False)
idx_cubic = np.random.choice(len(results_cubic.samples), n_samples_band, replace=False)

samples_linear = results_linear.samples[idx_linear]
samples_parabola = results_parabola.samples[idx_parabola]
samples_cubic = results_cubic.samples[idx_cubic]

def eval_linear(params, x):
    return params[0]*x + params[1]

def eval_parabola(params, x):
    return params[0]*x**2 + params[1]*x + params[2]

def eval_cubic(params, x):
    return params[0]*x**3 + params[1]*x**2 + params[2]*x + params[3]

# Compute model predictions for each posterior sample
y_linear_preds = np.zeros((n_samples_band, len(x_data)))
y_parabola_preds = np.zeros((n_samples_band, len(x_data)))
y_cubic_preds = np.zeros((n_samples_band, len(x_data)))

for i in range(n_samples_band):
    y_linear_preds[i,:] = eval_linear(samples_linear[i], x_data)
    y_parabola_preds[i,:] = eval_parabola(samples_parabola[i], x_data)
    y_cubic_preds[i,:]   = eval_cubic(samples_cubic[i], x_data)

# Median and 68% intervals
linear_median = np.median(y_linear_preds, axis=0)
linear_upper = np.percentile(y_linear_preds, 84, axis=0)
linear_lower = np.percentile(y_linear_preds, 16, axis=0)

parabola_median = np.median(y_parabola_preds, axis=0)
parabola_upper = np.percentile(y_parabola_preds, 84, axis=0)
parabola_lower = np.percentile(y_parabola_preds, 16, axis=0)

cubic_median = np.median(y_cubic_preds, axis=0)
cubic_upper = np.percentile(y_cubic_preds, 84, axis=0)
cubic_lower = np.percentile(y_cubic_preds, 16, axis=0)

# -------------------------------------------------------------------------
# 7. Plot the results
# -------------------------------------------------------------------------
plt.figure(figsize=(8,6))
plt.scatter(x_data, y_data, c='k', label='Noisy Data', alpha=0.6, zorder=3)
plt.plot(x_data, y_true, 'k--', label='True 2nd-Order Model', zorder=1, linewidth=0.5)

# Plot linear fit + uncertainty band
plt.plot(x_data, linear_median, 'r-', label='Linear (median)', zorder=2)
plt.fill_between(x_data, linear_lower, linear_upper, color='r', alpha=0.2,
                 label='Linear ~1$\sigma$ interval', zorder=2)

# Plot parabolic fit + uncertainty band
plt.plot(x_data, parabola_median, 'b-', label='Parabola (median)', zorder=2)
plt.fill_between(x_data, parabola_lower, parabola_upper, color='b', alpha=0.2,
                 label='Parabola ~1$\sigma$ interval', zorder=2)

# Plot cubic fit + uncertainty band
plt.plot(x_data, cubic_median, 'm-', label='Cubic (median)', zorder=2)
plt.fill_between(x_data, cubic_lower, cubic_upper, color='m', alpha=0.2,
                 label='Cubic ~1$\sigma$ interval', zorder=2)

# plt.title('Comparison of Linear, Parabolic, and Cubic Fits with dynesty')
plt.xlabel('x', fontsize=14)
plt.ylabel('y', fontsize=14)
# make the ticks larger
plt.tick_params(axis='both', which='major', labelsize=12)
plt.tick_params(axis='both', which='minor', labelsize=10)
# make the labels larger
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend()
plt.tight_layout()
# plt.show()
# find python file directory where this script is located
python_executable_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
# save the figure in the python executable directory
plt.savefig(os.path.join(python_executable_dir, 'dynesty_fit_comparison.png'), dpi=300)
plt.close()
