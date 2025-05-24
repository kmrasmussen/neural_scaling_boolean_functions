import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker # Import the ticker module
import numpy as np

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)

def plot_scatter(df, x_column, y_column, color_column, title, x_label, y_label, filename_prefix="plot"):
    plt.figure(figsize=(12, 8))
    
    # Create a normalizer for the color scale
    norm = mcolors.Normalize(vmin=df[color_column].min(), vmax=df[color_column].max())
    # Get a colormap
    cmap = cm.viridis # Or cm.plasma, cm.inferno, etc.

    scatter = plt.scatter(
        df[x_column],
        df[y_column],
        c=df[color_column],
        cmap=cmap,
        norm=norm,
        alpha=0.7,
        edgecolors='w',
        linewidth=0.5
    )
    
    plt.title(title, fontsize=16)
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    
    # Add a colorbar
    cbar = plt.colorbar(scatter, label=f'{color_column} (Gradient)')
    
    # Format colorbar ticks to scientific notation
    formatter = mticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1,1)) # Adjust powerlimits as needed for when to switch to sci notation
    cbar.ax.yaxis.set_major_formatter(formatter)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    # plt.show() # Replace plt.show() with plt.savefig()
    save_filename = f"{filename_prefix}_{x_column}_vs_{y_column}.png"
    plt.savefig(save_filename)
    print(f"Plot saved to {save_filename}")
    plt.close() # Close the figure to free memory

if __name__ == '__main__':
    jsonl_file_path = 'experiment_runs.jsonl'  # Make sure this path is correct
    
    try:
        df = load_jsonl(jsonl_file_path)
    except FileNotFoundError:
        print(f"Error: The file {jsonl_file_path} was not found.")
        exit()
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {jsonl_file_path}. Check file format.")
        exit()
    except Exception as e:
        print(f"An unexpected error occurred while loading the file: {e}")
        exit()

    if df.empty:
        print("The DataFrame is empty. No data to plot.")
        exit()

    # Ensure required columns exist
    required_columns = ['emaloss', 'cumulbits', 'batch_forward_backward_flops', 'total_params']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns in the JSONL file: {', '.join(missing_cols)}")
        exit()

    # Convert columns to numeric if they aren't already, coercing errors
    for col in ['emaloss', 'cumulbits', 'batch_forward_backward_flops', 'total_params']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows where conversion to numeric might have failed (NaNs)
    df.dropna(subset=required_columns, inplace=True)

    if df.empty:
        print("The DataFrame is empty after cleaning. No data to plot.")
        exit()

    # Create the new x-axis column
    df['cumulbits_times_flops'] = df['cumulbits'] * df['batch_forward_backward_flops']

    # Plot 1: Loss vs. Cumulative Bits
    plot_scatter(
        df,
        x_column='cumulbits',
        y_column='emaloss',
        color_column='total_params',
        title='EMA Loss vs. Cumulative Bits (Colored by Total Params)',
        x_label='Cumulative samples (bits)',
        y_label='Cross-entropy (bits)',
        filename_prefix="loss_vs_cumulbits"
    )

    # Plot 2: Loss vs. Cumulative Bits * Batch Forward-Backward FLOPs
    plot_scatter(
        df,
        x_column='cumulbits_times_flops',
        y_column='emaloss',
        color_column='total_params',
        title='EMA Loss vs. Cumulative Compute (Colored by Total Params)',
        x_label='Cumulative compute (FLOPs)',
        y_label='Cross-entropy (bits)',
        filename_prefix="loss_vs_compute"
    )

    print("Plots generated and saved.")