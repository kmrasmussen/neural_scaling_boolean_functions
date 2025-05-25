# %%
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.colors as mcolors
# %%
df = pd.read_json('experiment_runs.jsonl', lines=True)
# %%
df_filtered = df[df['cumulbits'] >= 1e6]
# %%
def scatter_plotter_function(
  df,
  x_column, 
  y_column,
  total_params_column,
  xlabel,
  ylabel,
  title,
  x_scale_log=True,
  output_filepath=None
):
  plt.figure()
  plt.scatter(
    df[x_column],
    df[y_column],
    c=df[total_params_column],
    cmap='viridis',
    norm=mcolors.LogNorm())
  if x_scale_log:
    plt.xscale('log')
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.title(title)
  cbar = plt.colorbar()
  cbar.set_label('model parameters')
  if output_filepath:
    plt.savefig(output_filepath)
    print('Saved plot')
  plt.close()
  
# %%
scatter_plotter_function(
  df_filtered,
  x_column='cumulbits',
  y_column='emaloss',
  total_params_column='total_params',
  xlabel='data samples seen',
  ylabel='cross-entropy (bits)',
  title='Loss vs data',
  output_filepath='plots/loss_vs_data.png'
)
# %%
scatter_plotter_function(
  df_filtered,
  x_column='cumulsurp',
  y_column='emaloss',
  total_params_column='total_params',
  xlabel='cumulative surprisal',
  ylabel='cross-entropy (bits)',
  title='Loss vs surprise',
  output_filepath='plots/loss_vs_cumulsurp.png'
)
# %%
df_filtered['cumulflops'] = df_filtered['cumulbits']*df_filtered['forward_backward_flops']
scatter_plotter_function(
  df_filtered,
  x_column='cumulflops',
  y_column='emaloss',
  total_params_column='total_params',
  xlabel='data samples seen',
  ylabel='cross-entropy (bits)',
  title='Loss vs flops',
  output_filepath='plots/loss_vs_flops.png'
)