import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

def plot_upper_triangle_corr_squares(corr_df: pd.DataFrame,
                                     outpath: str,
                                     annotate: bool = True,
                                     fmt: str = "{:.2f}",
                                     figsize=(10, 10),
                                     dpi=300):
    """
    Plot an upper-triangle correlation matrix as colored squares.
    - labels only on left (y) and bottom (x)
    - upper triangle (k=1) only, diagonal and lower are blank
    - colormap: blue (negative) -> red (positive)
    """
    labels = list(corr_df.index)
    n = len(labels)

    # Ensure same order for rows/cols
    corr_df = corr_df.loc[labels, labels]

    cmap = plt.colormaps["coolwarm"]
    norm = Normalize(vmin=-1, vmax=1)

    fig, ax = plt.subplots(figsize=figsize)

    # Draw only LOWER triangle squares
    for i in range(n):
        for j in range(n):
            if j >= i:
                continue  # skip diagonal and upper triangle

            val = corr_df.iat[i, j]
            if np.isnan(val):
                continue

            color = cmap(norm(val))
            rect = plt.Rectangle((j, i), 1, 1, facecolor=color, edgecolor="white", linewidth=1.0)
            ax.add_patch(rect)

            if annotate:
                ax.text(j + 0.5, i + 0.5, fmt.format(val),
                        ha="center", va="center", fontsize=10, color="black")


    # Axes formatting: make it look like a matrix
    ax.set_xlim(0, n)
    ax.set_ylim(n, 0)  # invert y so first label is at top

    ax.set_aspect("equal")

    # ticks at cell centers
    ax.set_xticks(np.arange(n) + 0.5)
    ax.set_yticks(np.arange(n) + 0.5)

    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels, rotation=0)

    # Only show bottom and left labels; hide top/right ticks
    ax.tick_params(top=False, bottom=True, left=True, right=False, labeltop=False)

    # Hide x tick labels for non-bottom? (we only have one axis here, so keep bottom)
    # Draw a clean frame
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Correlation", rotation=90)

    # ax.set_title("Summed correlation matrix (lower triangle)")

    plt.tight_layout()
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def load_and_sum_correlations(directory):
    correlation_sum = {}
    max_correlation = {}
    min_correlation = {}
    max_correlation_file = {}
    min_correlation_file = {}
    count_correlation = {}

    master_labels = None
    master_size = -1   # <-- track the largest matrix dimension


    # Walk through all subdirectories in the given directory
    for root, dirs, files in os.walk(directory):
        for file_name in files:
            if 'weighted_correlation_matrix' in file_name and file_name.endswith('.csv'):
                # Get the base name of the file (e.g., YYYYMMDD_HHMMSS)
                base_name = os.path.splitext(file_name)[0]
                
                # Load the CSV file, skipping the first empty cell (first row and first column)
                file_path = os.path.join(root, file_name)

                df = pd.read_csv(file_path, index_col=0)

                # Strip whitespace just in case
                df.index = df.index.astype(str).str.strip()
                df.columns = df.columns.astype(str).str.strip()

                # Keep the label order from the BIGGEST matrix we find
                # (use number of columns; you can also use df.shape[0]*df.shape[1] if you prefer)
                this_size = df.shape[1]

                if this_size > master_size:
                    master_size = this_size
                    master_labels = list(df.columns)

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

    # Use the CSV's native order (from the first file)
    labels = [s.strip() for s in master_labels]
    n = len(labels)

    sum_mat = np.full((n, n), np.nan, dtype=float)

    # Fill upper triangle using your summed totals (and mirror if you want symmetry)
    idx = {lab: k for k, lab in enumerate(labels)}

    for pair, total in correlation_sum.items():
        v1, v2 = [s.strip() for s in pair.split(" - ")]
        if v1 not in idx or v2 not in idx:
            continue
        i = idx[v1]; j = idx[v2]
        if i == j:
            continue
        ii, jj = (i, j) if i > j else (j, i)  # lower triangle
        avg = total / count_correlation[pair]
        sum_mat[ii, jj] = avg



    # Wrap in DataFrame (same labels for rows/cols)
    sum_corr_df = pd.DataFrame(sum_mat, index=labels, columns=labels)

    # Plot it
    out_png = os.path.join(directory, "sum_correlation_each_meteor_single.png")
    plot_upper_triangle_corr_squares(sum_corr_df, out_png, annotate=True, figsize=(0.6*n + 4, 0.6*n + 4))

    
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
directory = r'C:\Users\maxiv\Documents\UWO\Papers\2)ORI-CAP-PER-DRA\Results\CAMO+EMCCD\CAP_radiance_new'  # Adjust if needed based on your working directory

top_5, bottom_5, output_file = load_and_sum_correlations(directory)

print(top_5,'\n', bottom_5,'\n', output_file)
