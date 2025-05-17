import os
import pandas as pd
import numpy as np

def load_and_sum_correlations(directory):
    # Initialize a dictionary to store correlation sums, max, min, and count of correlations
    correlation_sum = {}
    max_correlation = {}
    min_correlation = {}
    max_correlation_file = {}
    min_correlation_file = {}
    count_correlation = {}

    # Walk through all subdirectories in the given directory
    for root, dirs, files in os.walk(directory):
        for file_name in files:
            if 'weighted_correlation_matrix' in file_name and file_name.endswith('.csv'):
                # Get the base name of the file (e.g., YYYYMMDD_HHMMSS)
                base_name = os.path.splitext(file_name)[0]
                
                # Load the CSV file, skipping the first empty cell (first row and first column)
                file_path = os.path.join(root, file_name)
                df = pd.read_csv(file_path, index_col=0)  # Set the first column as index

                # Ensure the dataframe has more than 1 row and column
                if df.shape[0] > 1 and df.shape[1] > 1:
                    # Process each pair of variables (excluding the first row and column)
                    correlation_matrix = df.values[0:, 0:]  # Skip first row and column (empty first cell)
                    # Get the labels (variables) for the rows and columns
                    row_labels = df.index[0:]  # Skip the first label in the index
                    column_labels = df.columns[0:]  # Skip the first label in the columns

                    # Process the correlations
                    for i in range(correlation_matrix.shape[0]):
                        for j in range(i + 1, correlation_matrix.shape[1]):  # Avoid duplicates (i.e., the diagonal)
                            # Get the correlation value
                            correlation_value = correlation_matrix[i, j]
                            variable_1 = row_labels[i]  # Corresponding variable for the row
                            variable_2 = column_labels[j]  # Corresponding variable for the column
                            
                            # Combine the pair of variables as a unique identifier
                            pair_key = f'{variable_1} - {variable_2}'

                            if pair_key not in correlation_sum:
                                correlation_sum[pair_key] = 0
                                max_correlation[pair_key] = correlation_value
                                min_correlation[pair_key] = correlation_value
                                max_correlation_file[pair_key] = base_name
                                min_correlation_file[pair_key] = base_name
                                count_correlation[pair_key] = 0  # Initialize count of correlations

                            if not np.isnan(correlation_value):
                                correlation_sum[pair_key] += correlation_value
                                count_correlation[pair_key] += 1  # Increment the count
                                
                                # Update the max and min correlations for this pair
                                if correlation_value > max_correlation[pair_key]:
                                    max_correlation[pair_key] = correlation_value
                                    max_correlation_file[pair_key] = base_name
                                if correlation_value < min_correlation[pair_key]:
                                    min_correlation[pair_key] = correlation_value
                                    min_correlation_file[pair_key] = base_name

    # Convert the correlation sum dictionary to a DataFrame
    correlation_df = pd.DataFrame(list(correlation_sum.items()), columns=['pair', 'total_correlation'])
    
    # Add max, min, and average correlation columns, including the file names
    correlation_df['max_correlation'] = correlation_df['pair'].map(max_correlation)
    correlation_df['min_correlation'] = correlation_df['pair'].map(min_correlation)
    correlation_df['max_correlation_file'] = correlation_df['pair'].map(max_correlation_file)
    correlation_df['min_correlation_file'] = correlation_df['pair'].map(min_correlation_file)
    correlation_df['average_correlation'] = correlation_df['total_correlation'] / correlation_df['pair'].map(count_correlation)
    
    # Sort by absolute correlation value
    correlation_df['abs_correlation'] = correlation_df['total_correlation'].abs()
    sorted_df = correlation_df.sort_values(by='abs_correlation', ascending=False)
    
    # Get the top 5 and bottom 5 correlations (considering sign)
    top_5 = sorted_df.head(5)
    bottom_5 = sorted_df.tail(5)
    
    # Save the results in a new CSV file
    output_file = os.path.join(directory, 'sum_correlations.csv')
    sorted_df[['pair', 'total_correlation', 'max_correlation', 'max_correlation_file', 
               'min_correlation', 'min_correlation_file', 'average_correlation']].to_csv(output_file, index=False)
    
    return top_5, bottom_5, output_file


# Specify the directory where your files are stored
directory = r'C:\Users\maxiv\Documents\UWO\Papers\2)ORI-CAP-PER-DRA\Results\CAMO+EMCCD\CAP'  # Adjust if needed based on your working directory

top_5, bottom_5, output_file = load_and_sum_correlations(directory)

print(top_5,'\n', bottom_5,'\n', output_file)
