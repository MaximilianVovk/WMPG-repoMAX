import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from scipy.stats import zscore

# Load the CSV file with the training data
data = pd.read_csv(r'C:\Users\maxiv\Documents\UWO\Papers\1)PCA\Real_event_results\1good\20210813_061453_emccd_skyfit2_CAMO\20210813-061452.941123\20210813_061452_GenSim\20210813_061452_sim.csv')

# Define the columns corresponding to the simple function parameters
simple_params = [
    'vel_init_norot', 'vel_avg_norot', 'duration', 'peak_mag_height',
    'begin_height', 'end_height', 'peak_abs_mag', 'beg_abs_mag', 'end_abs_mag',
    'F', 'trail_len', 't0', 'deceleration_lin', 'deceleration_parab', 'decel_parab_t0',
    'decel_t0', 'decel_jacchia', 'kurtosis', 'skew', 'avg_lag', 'kc',
    'Dynamic_pressure_peak_abs_mag', 'a_acc', 'b_acc', 'c_acc', 'a_t0', 'b_t0',
    'c_t0', 'a1_acc_jac', 'a2_acc_jac', 'a_mag_init', 'b_mag_init', 'c_mag_init',
    'a_mag_end', 'b_mag_end', 'c_mag_end'
]

# Define the columns corresponding to the complex variables
complex_vars = [
    'v_init_180km', 'zenith_angle', 'mass', 'rho', 'sigma',
    'erosion_height_start', 'erosion_coeff', 'erosion_mass_index', 'erosion_mass_min', 'erosion_mass_max'
]

# Separate features and target variables
X = data[simple_params]
y = data[complex_vars]

# Combine X and y to filter together
combined_data = pd.concat([X, y], axis=1)

# Calculate the Z-scores
z_scores = zscore(combined_data)

# Set a threshold for Z-score filtering
z_threshold = 4  # Increased threshold to prevent filtering out all data

# Filter out the outliers based on Z-scores (keeping rows with all Z-scores < threshold)
filtered_data = combined_data[(np.abs(z_scores) < z_threshold).all(axis=1)]

# Check the number of samples after filtering
print(f"Number of samples after filtering: {len(filtered_data)}")

# If no samples remain after filtering, consider relaxing the threshold or bypass filtering
if len(filtered_data) == 0:
    print("No data left after filtering. Reverting to original data.")
    filtered_data = combined_data  # Revert to original data if filtering is too strict

# Separate the filtered data back into X and y
X_filtered = filtered_data[simple_params]
y_filtered = filtered_data[complex_vars]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_filtered, y_filtered, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the parameter grid for GridSearchCV
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto'],
    'epsilon': [0.01, 0.1, 0.5, 1],
    'kernel': ['rbf']  # RBF kernel is often effective in nonlinear problems
}

# Initialize the list to hold the best models for each target variable
models = []
best_params = {}

# Train and tune an SVM model for each target variable (complex variable)
for i, target in enumerate(complex_vars):
    svr = SVR()
    grid_search = GridSearchCV(svr, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=1)
    grid_search.fit(X_train_scaled, y_train[target])
    
    # Store the best parameters
    best_params[target] = grid_search.best_params_
    print(f"Best parameters for {target}: {best_params[target]}")
    
    # Train the final model using the best parameters
    best_model = grid_search.best_estimator_
    models.append(best_model)
    
    # Predict on the test set using the best model
    y_pred = best_model.predict(X_test_scaled)
    
    # Clamp predictions to the min and max values observed in the training data
    y_pred_clamped = np.clip(y_pred, y_filtered[target].min(), y_filtered[target].max())

    # Evaluate the model performance with clamped predictions
    mse = mean_squared_error(y_test[target], y_pred_clamped)
    r2 = r2_score(y_test[target], y_pred_clamped)
    
    print(f"Target: {target}")
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")
    print("-" * 30)

# Load the new data CSV file for prediction
new_data = pd.read_csv(r'C:\Users\maxiv\Documents\UWO\Papers\1)PCA\Real_event_results\1good\20210813_061453_emccd_skyfit2_CAMO\20210813-061452.941123\20210813_061452_GenSim\20210813_061452_obs_real.csv')

# Ensure that the new data contains the same simple_params columns
new_data = new_data[simple_params]

# Standardize the new data
new_data_scaled = scaler.transform(new_data)

# Predict the complex variables using the trained SVM models
svm_predictions = np.array([model.predict(new_data_scaled) for model in models])

# Clamp predictions to the min and max values observed in the training data
for i, target in enumerate(complex_vars):
    svm_predictions[i] = np.clip(svm_predictions[i], y_filtered[target].min(), y_filtered[target].max())

# Ensure the arrays are correctly shaped
svm_predictions = svm_predictions.T  # Transpose to have the correct shape

# Calculate the mean prediction, and 95% confidence interval
mean_prediction = np.mean(svm_predictions, axis=0)
percentile_95_low = np.percentile(svm_predictions, 2.5, axis=0)
percentile_95_high = np.percentile(svm_predictions, 97.5, axis=0)

# Multiply erosion_coeff and sigma by 1,000,000
for i, var in enumerate(complex_vars):
    if var in ['erosion_coeff', 'sigma']:
        mean_prediction[i] *= 1e6
        percentile_95_low[i] *= 1e6
        percentile_95_high[i] *= 1e6

# Prepare the LaTeX table content with .4g formatting
latex_table = "\\hline\n"
for i, var in enumerate(complex_vars):
    latex_table += f"{var.replace('_', ' ')} & {mean_prediction[i]:.4g} & {np.median(svm_predictions[:, i]):.4g} & {percentile_95_low[i]:.4g} & {percentile_95_high[i]:.4g} \\\\ \n\\hline\n"

# Print the LaTeX table
print("LaTeX Table:")
print(latex_table)

# Optionally, save the LaTeX table to a .txt file
with open('complex_vars_latex_table.txt', 'w') as f:
    f.write(latex_table)
