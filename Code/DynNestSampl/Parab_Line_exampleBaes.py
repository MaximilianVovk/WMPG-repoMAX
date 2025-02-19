#!/usr/bin/env python3
"""
Example script demonstrating how to use dynamic nested sampling (via `dynesty`)
to compare a linear model vs. a second-order polynomial model on synthetic data.
We also illustrate how to visualize the posterior-derived uncertainty in the model fits.

Author: Your Name
Date: YYYY-MM-DD
"""

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
    # Predicted values
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

# -------------------------------------------------------------------------
# 3. Define the prior transforms for both models
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
dsampler_linear.run_nested(print_progress=False)
results_linear = dsampler_linear.results

# --- Parabolic model
dsampler_parabola = DynamicNestedSampler(
    loglike_parabola,
    prior_transform_parabola,
    ndim=3,
    bound='multi',
    sample='rwalk'
)
dsampler_parabola.run_nested(print_progress=False)
results_parabola = dsampler_parabola.results

# -------------------------------------------------------------------------
# 5. Extract the Bayesian evidence and best-fit parameters
# -------------------------------------------------------------------------
logZ_linear = results_linear.logz[-1]     # log-evidence
logZerr_linear = results_linear.logzerr[-1]
logZ_parabola = results_parabola.logz[-1]
logZerr_parabola = results_parabola.logzerr[-1]

print("=== Linear Model (m*x + b) ===")
print(f"Estimated logZ: {logZ_linear:.2f} +/- {logZerr_linear:.2f}")

print("\n=== Parabolic Model (a*x^2 + b*x + c) ===")
print(f"Estimated logZ: {logZ_parabola:.2f} +/- {logZerr_parabola:.2f}")

# Bayes factor
delta_logZ = logZ_parabola - logZ_linear
print(f"\nBayes factor (parabola vs. linear) ~ exp({delta_logZ:.2f}) "
      f"= {np.exp(delta_logZ):.2f}")

# -------------------------------------------------------------------------
# 6. Posterior Sampling: Generate model predictions for plotting uncertainties
#    We'll take random samples from each model's posterior to compute
#    mean + credible bands.
# -------------------------------------------------------------------------

# How many posterior samples to draw for banding
n_samples_band = 200

# Generate random draws from each posterior
idx_linear = np.random.choice(len(results_linear.samples), n_samples_band, replace=False)
idx_parabola = np.random.choice(len(results_parabola.samples), n_samples_band, replace=False)

samples_linear = results_linear.samples[idx_linear]
samples_parabola = results_parabola.samples[idx_parabola]

# Evaluate model predictions for each sample
def eval_linear(params, x):
    return params[0]*x + params[1]

def eval_parabola(params, x):
    return params[0]*x**2 + params[1]*x + params[2]

# Arrays to hold y-model predictions for each posterior sample
y_linear_preds = np.zeros((n_samples_band, len(x_data)))
y_parabola_preds = np.zeros((n_samples_band, len(x_data)))

for i in range(n_samples_band):
    # Linear
    y_linear_preds[i,:] = eval_linear(samples_linear[i], x_data)
    # Parabolic
    y_parabola_preds[i,:] = eval_parabola(samples_parabola[i], x_data)

# Compute median and percentile ranges (e.g., 68% or 95%)
linear_median = np.median(y_linear_preds, axis=0)
linear_upper = np.percentile(y_linear_preds, 84, axis=0)  # ~ +1 sigma
linear_lower = np.percentile(y_linear_preds, 16, axis=0)  # ~ -1 sigma

parabola_median = np.median(y_parabola_preds, axis=0)
parabola_upper = np.percentile(y_parabola_preds, 84, axis=0) 
parabola_lower = np.percentile(y_parabola_preds, 16, axis=0)

# -------------------------------------------------------------------------
# 7. Plot the results
# -------------------------------------------------------------------------
plt.figure(figsize=(8,6))
plt.scatter(x_data, y_data, c='k', label='Noisy Data', alpha=0.6, zorder=3)
plt.plot(x_data, y_true, 'g--', label='True 2nd-Order Model', zorder=1)

# Plot linear fit + uncertainty band
plt.plot(x_data, linear_median, 'r-', label='Linear (median)', zorder=2)
plt.fill_between(x_data, linear_lower, linear_upper, color='r', alpha=0.2, 
                 label='Linear ~1$\sigma$ interval', zorder=2)

# Plot parabolic fit + uncertainty band
plt.plot(x_data, parabola_median, 'b-', label='Parabola (median)', zorder=2)
plt.fill_between(x_data, parabola_lower, parabola_upper, color='b', alpha=0.2,
                 label='Parabola ~1$\sigma$ interval', zorder=2)

plt.title('Comparison of Linear vs. Parabolic Fits with dynesty')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.tight_layout()
plt.show()
