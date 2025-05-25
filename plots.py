# %%
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.colors as mcolors
# %%
df = pd.read_json('jsonls/hejsa4.jsonl', lines=True)
# %%
df_filtered = df[df['cumulbits'] >= 1e6]
# %%
plt.scatter(df_filtered['cumulbits'], df_filtered['emaloss'], c=df_filtered['total_params'], cmap='viridis', norm=mcolors.LogNorm())
plt.xscale('log')
plt.xlabel('cumulbits (log scale)')
plt.ylabel('cross-entropy loss (bits)')
plt.title('Training snapshots for loss vs cumulative number of samples seen (cumulbits >= 1e6)')
cbar = plt.colorbar()
cbar.set_label('model parameters')
plt.show()
# %%

plt.scatter(df_filtered['cumulbits']*df_filtered['batch_flops']/df_filtered['batch_size'], df_filtered['emaloss'], c=df_filtered['total_params'], cmap='viridis', norm=mcolors.LogNorm())
plt.xscale('log')
plt.xlabel('cumultative flops (log scale)')
plt.ylabel('cross-entropy loss (bits)')
plt.title('Training snapshots for loss vs cumulative flops (cumulbits >= 1e6)')
cbar = plt.colorbar()
cbar.set_label('model parameters')
plt.show()
# %%
df
# %%
plt.scatter(df_filtered['cumulbits']*df_filtered['batch_flops']/df_filtered['batch_size'], df_filtered['cumulsurp'], c=df_filtered['total_params'], cmap='viridis', norm=mcolors.LogNorm())
plt.xscale('log')
plt.xlabel('cumulflops')
plt.ylabel('cumulsurp')
plt.yscale('log')
plt.title('Training snapshots for loss vs cumulative flops (cumulbits >= 1e6)')
cbar = plt.colorbar()
cbar.set_label('model parameters')
plt.show()
# %%
plt.scatter(df_filtered['cumulbits']*df_filtered['batch_flops']/df_filtered['batch_size'], df_filtered['emaloss'], c=df_filtered['total_params'], cmap='viridis', norm=mcolors.LogNorm())
plt.xscale('log')
plt.xlabel('cumulflops')
plt.ylabel('cumulsurp')
plt.yscale('log')
plt.title('Training snapshots for loss vs cumulative flops (cumulbits >= 1e6)')
cbar = plt.colorbar()
cbar.set_label('model parameters')
plt.show()
# %%
plt.scatter(df_filtered['cumulbits'], df_filtered['cumulsurp'], c=df_filtered['total_params'], cmap='viridis', norm=mcolors.LogNorm())
plt.xscale('log')
plt.yscale('log')
# %%
plt.scatter(df_filtered['cumulbits'], df_filtered['cumulsurp']/df_filtered['cumulbits'], c=df_filtered['total_params'], cmap='viridis', norm=mcolors.LogNorm())
plt.xscale('log')
plt.yscale('log')
# %%
plt.scatter(df_filtered['cumulbits'], df_filtered['emaloss'], c=df_filtered['total_params'], cmap='viridis', norm=mcolors.LogNorm())
plt.xscale('log')
plt.yscale('log')
# %%
