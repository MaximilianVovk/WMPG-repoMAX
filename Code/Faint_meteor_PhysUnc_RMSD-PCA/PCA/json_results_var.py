import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_variable_importance(df, plot_dir):
    """
    Creates a bar plot of variables sorted by average importance.

    Parameters:
        df (pd.DataFrame): DataFrame containing Variable, Color, Average Importance, Max Importance, and Min Importance.
        plot_dir (str): Directory to save the plot.
    """
    # Make sure the plot directory exists
    os.makedirs(plot_dir, exist_ok=True)
    
    # Sort by average importance
    df_sorted = df.sort_values(by='Average Importance', ascending=False)

    # Convert columns to float for arithmetic operations
    df_sorted['Average Importance'] = df_sorted['Average Importance'].astype(float)
    df_sorted['Max Importance'] = df_sorted['Max Importance'].astype(float)
    df_sorted['Min Importance'] = df_sorted['Min Importance'].astype(float)

    # Extract data for the plot
    variables = df_sorted['Variable']
    average_importance = df_sorted['Average Importance']
    max_importance = df_sorted['Max Importance']
    min_importance = df_sorted['Min Importance']
    colors = df_sorted['Color']

    # Calculate error bars (ensure non-negative values)
    lower_error = average_importance - min_importance
    upper_error = max_importance - average_importance
    lower_error[lower_error < 0] = 0
    upper_error[upper_error < 0] = 0
    yerr = [lower_error, upper_error]

    # Create the bar plot
    plt.figure(figsize=(12, 6))
    plt.bar(variables, average_importance, yerr=yerr, color=colors, capsize=5, alpha=0.7) # , edgecolor='black'

    # Customize the plot
    plt.xticks(rotation=90, ha='right')
    plt.xlabel('Variables')
    plt.ylabel('Average Importance')
    # plt.title('Variable Importance with Error Bars')
    plt.tight_layout()

    # Save the plot
    plot_file = os.path.join(plot_dir, 'variable_importance_plot.png')
    plt.savefig(plot_file, dpi=300)
    plt.close()

    print(f"Plot saved to: {plot_file}")


# Directory to search for the CSV files
directory = r'N:\mvovk\1stPaper\Results_12-07'
# Directory where you want to save the final CSV and plot
output_dir = r'C:\Users\maxiv\Documents\UWO\Papers\1)PCA\jsonsim_results'
os.makedirs(output_dir, exist_ok=True)

# Initialize variables
result_df = None
num_file = 0
for root, dirs, files in os.walk(directory):
    for file in files:
        if file.endswith('PCA_sorted_variable_importance_percent.csv'):
            file_path = os.path.join(root, file)
            df = pd.read_csv(file_path)

            if 'Variable' in df.columns and 'Importance' in df.columns and 'Color' in df.columns:
                num_file += 1
                # Initialize the result_df with float columns
                if result_df is None:
                    result_df = df[['Variable', 'Color']].copy()
                    result_df['Average Importance'] = 0.0
                    result_df['Max Importance'] = 0.0
                    # Set Min Importance to infinity so that the first comparison sets it properly
                    result_df['Min Importance'] = float('inf')
                    result_df['File with Max Importance'] = ""
                    result_df['File with Min Importance'] = ""

                for idx, row in df.iterrows():
                    variable = row['Variable']
                    importance = float(row['Importance'])  # Ensure importance is float
                    # Match the variable in the result_df
                    result_row = result_df[result_df['Variable'] == variable]

                    if not result_row.empty:
                        index = result_row.index[0]
                        # Update average importance
                        result_df.at[index, 'Average Importance'] += abs(importance)
                        # Update max importance
                        if importance > result_df.at[index, 'Max Importance']:
                            result_df.at[index, 'Max Importance'] = importance
                            result_df.at[index, 'File with Max Importance'] = file
                        # Update min importance
                        if importance < result_df.at[index, 'Min Importance']:
                            result_df.at[index, 'Min Importance'] = importance
                            result_df.at[index, 'File with Min Importance'] = file

# Normalize the average importance by the number of files found
if result_df is not None:
    result_df['Average Importance'] /= num_file

    # Replace infinite Min Importance (in case a variable never got updated) with 0 or NaN
    # result_df['Min Importance'].replace([float('inf')], 0.0, inplace=True)
    result_df['Min Importance'] = result_df['Min Importance'].replace([float('inf')], 0.0)


    # Save the result to a new CSV file in the chosen output directory
    csv_output_file = os.path.join(output_dir, 'PCA_aggregated_results.csv')
    result_df.to_csv(csv_output_file, index=False)
    print(f"Number of files found: {num_file}")
    print(f"Aggregated results saved to: {csv_output_file}")

    # Plot the variable importance
    plot_variable_importance(result_df, output_dir)
else:
    print("No matching CSV files found.")
