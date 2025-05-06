import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (14, 8)

def plot_metric(dataframe:pd.DataFrame, metric:str, limit:int=5, asc=True, title:str=None, y_label:str=None, start:int=None, end:int=None):
    df = dataframe[metric].copy()
    df.columns = ["_".join(a) for a in df.columns.to_flat_index()]
    fig, ax = plt.subplots(1, 1, figsize=(20, 8))
    xstart = df.index[0]
    xend = df.index[-1]
    if start:
        xstart = df.index[start]
    if end:
        xend = df.index[end]

    if metric == 'TMPTR':
        # diff_df = df.copy()
        mean = df.mean(axis=0)
        df['mean'] = df.mean(axis=1)
        df['std'] = df.std(axis=1)
        # diff_df.sub(df['mean']+2*df['std'],axis=0) >= 0
        mean.sort_values(ascending=asc, inplace=True)

        ax.plot(df['mean'], label='mean', color=(0.5,0.5,1), linewidth=1)
        ax.fill_between(df.index, df['mean'] + 2 * df['std'], df['mean'] - 2 * df['std'],color=(0.1,0.1,0.1),alpha=0.2)

        for x in mean.index[:limit]:
            ax.plot(df.index, df[x],
                    alpha=0.5, linewidth=0.5, label=f"{x.split('_')[0]}-GPU{x.split('_')[1]}")

    if metric == 'GRACT':
        diff=(df-1).sum()
        diff.sort_values(ascending=asc, inplace=True)

        for x in diff.index[:limit]:
            ax.plot(df.index, df[x],
                     alpha=0.5, linewidth=0.5, label=f"{x.split('_')[0]}-GPU{x.split('_')[1]}")

    if metric == 'POWER':
        # diff_df = df.copy()
        mean = df.mean(axis=0)
        df['mean'] = df.mean(axis=1)
        df['std'] = df.std(axis=1)
        # diff_df.sub(df['mean']+2*df['std'],axis=0) >= 0
        mean.sort_values(ascending=asc, inplace=True)

        ax.plot(df['mean'], label='mean', color=(0.5,0.5,1), linewidth=1)
        ax.fill_between(df.index, df['mean'] + 2 * df['std'], df['mean'] - 2 * df['std'],color=(0.1,0.1,0.1),alpha=0.2)

        for x in mean.index[:limit]:
            ax.plot(df.index, df[x],
                    alpha=0.5, linewidth=0.5, label=f"{x.split('_')[0]}-GPU{x.split('_')[1]}")

    if metric == 'SMACT':
        sum_df=df.sum().sort_values(ascending=asc)
        # mean = df.mean(axis=1)
        # std = df.std(axis=1)
        #
        # ax.plot(mean, label='mean', color=(0.5,0.5,1), linewidth=1)
        # ax.fill_between(df.index, mean + 2 * std, mean - 2 * std,color=(0.1,0.1,0.1),alpha=0.2)

        for x in sum_df.index[:limit]:
            ax.plot(df.index, df[x],alpha=0.5, linewidth=0.5, label=f"{x.split('_')[0]}-GPU{x.split('_')[1]}")

    # Set titles
    if title:
        fig.suptitle(title, fontsize=16)
    # ax.set_title(f"Fast Run")

    # Set labels
    ax.set_xlabel("Time")
    ax.set_ylabel(y_label if y_label else metric)

    # Show the completed plots
    plt.tight_layout()
    plt.xlim(xstart, xend)
    plt.show()

# Function to compute statistics per node
def compute_node_stats(df:pd.DataFrame,metric):
    """Compute statistics for each node across all its GPUs."""
    return df.groupby('node_id')[metric].agg(['mean', 'median', 'std', 'min', 'max'])

def plot_box(df:pd.DataFrame,metric:str,title:str=None,y_label:str=None):
    plt.figure(figsize=(14, 8))
    sns.boxplot(y='mean', data=compute_node_stats(df, metric))
    if title:
        plt.title(title)
    if y_label:
        plt.ylabel('Mean GPU Utilization (%)')

    plt.show()

# Function to create a comparison plot for a metric
def plot_box_and_violin(df:pd.DataFrame,metric:str, title=None, y_label=None):
    """Create side-by-side boxplots comparing a metric across fast and slow runs."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Boxplot comparison
    sns.boxplot(y='mean', data=compute_node_stats(df, metric), ax=ax1)
    ax1.set_title(f'Mean {title or metric} Distribution')
    ax1.set_ylabel(y_label or f"Mean {metric}")
    ax1.set_xlabel('Box plot')

    # Violinplot for a different view
    sns.violinplot(y='mean', data=compute_node_stats(df, metric), ax=ax2)
    ax2.set_title(f'Mean {title or metric} Distribution (Violin)')
    ax2.set_ylabel(y_label or f"Mean {metric}")
    ax2.set_xlabel('Violin plot')

    plt.tight_layout()
    plt.show()

def identify_stragglers(df:pd.DataFrame, metric:str, percentile=10,top:bool=False):
    """Identify potential straggler nodes based on a metric.

    Args:
        df_run: Dataframe for a specific run
        metric: Metric to analyze (e.g., 'GRACT' for GPU utilization)
        percentile: Bottom percentile threshold to consider a node a straggler

    Returns:
        DataFrame with straggler node information
    """
    # Compute stats per node
    node_stats = compute_node_stats(df, metric)

    # Determine threshold for stragglers
    if top:
        threshold = np.percentile(node_stats['mean'], 100-percentile)
    else:
        threshold = np.percentile(node_stats['mean'], percentile)

    # Identify stragglers
    if top:
        stragglers = node_stats[node_stats['mean'] >= threshold].sort_values('mean',ascending=False)
    else:
        stragglers = node_stats[node_stats['mean'] <= threshold].sort_values('mean')

    if top:
        print(f"Identified {len(stragglers)} potential straggler nodes (top {percentile}% in {metric})")
    else:
        print(f"Identified {len(stragglers)} potential straggler nodes (bottom {percentile}% in {metric})")
    print(f"Threshold: {threshold:.2f}")

    return stragglers

def plot_correlations(df:pd.DataFrame, metrics:list[str]):
    # Calculate correlation between metrics

    # Create correlation matrices
    df_corr = df[metrics].corr()

    # Plot heatmaps side by side
    fig, ax = plt.subplots(1, 1, figsize=(20, 8))

    sns.heatmap(df_corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    ax.set_title('Metric Correlations')

    plt.tight_layout()
    plt.show()

# Function to plot all NVLink error metrics
def plot_nvlink_error_metrics(df:pd.DataFrame,sample_rate=1,nodes:list[str]=None):
    """Plot all NVLink error metrics for both fast and slow runs side by side."""
    # NVLink error metrics
    error_metrics = ['NVL0T', 'NVL0R', 'NVL1T', 'NVL1R', 'NVL2T', 'NVL2R', 'NVL3T', 'NVL3R']

    # Create a figure with two columns (fast vs slow) and 4 rows (each error type)
    fig, axes = plt.subplots(4, 1, figsize=(20, 24))

    # Sample the data for faster plotting
    if sample_rate > 1:
        df = df.iloc[::sample_rate]

    if nodes:
        nodes_to_plot = nodes
    else:
    # Select some nodes to visualize (limit to 5 for clarity)
        nodes_to_plot = sorted(df['node_id'].unique())[:5]

    # Plot each error metric
    for i, metric_pair in enumerate([('NVL0T', 'NVL0R'), ('NVL1T', 'NVL1R'),
                                    ('NVL2T', 'NVL2R'), ('NVL3T', 'NVL3R')]):
        # TX metric
        tx_metric = metric_pair[0]
        rx_metric = metric_pair[1]

        # Plot fast run TX errors
        for node in nodes_to_plot:
            node_data = df[df['node_id'] == node]
            for gpu in sorted(node_data['gpu_id'].unique())[:1]:  # Just one GPU per node
                gpu_data = node_data[node_data['gpu_id'] == gpu]
                axes[i].plot(gpu_data['timestamp'], gpu_data[tx_metric],
                         alpha=0.7, linewidth=1.0, label=f"{node}-GPU{gpu} TX")
                axes[i].plot(gpu_data['timestamp'], gpu_data[rx_metric],
                         alpha=0.7, linewidth=1.0, linestyle='--', label=f"{node}-GPU{gpu} RX")

        # Set titles and labels
        axes[i].set_title(f"NVLink {i} TX/RX Errors")

        # Y-axis in scientific notation
        axes[i].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        axes[i].set_xlabel("Time")
        axes[i].set_ylabel("Error Count")

    # Add legend to first plot only
    axes[0].legend(loc='upper left')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    df = pd.read_parquet('data/128_node_run/outputs/397112/data_397112.parquet')
    df.reset_index(inplace=True)
    df.drop_duplicates(['timestamp','node_id','gpu_id'], inplace=True)

    pivot_df = df.pivot(index='timestamp',columns=['node_id','gpu_id'], values=['GRACT', 'SMACT', 'TENSO', 'TMPTR',
           'POWER', 'PCITX', 'PCIRX', 'NVLTX', 'NVLRX', 'NVL0T', 'NVL0R', 'NVL1T',
           'NVL1R', 'NVL2T', 'NVL2R', 'NVL3T', 'NVL3R'])
    pivot_df.dropna(axis=1,how='all',inplace=True)

    plot_metric(pivot_df, 'GRACT',title="GPU Utilization Over Time",y_label= "GPU Utilization (%)")
    plot_metric(pivot_df, 'TMPTR',title="GPU Temperature Over Time", y_label="Temperature (°C)")
    plot_metric(pivot_df, 'POWER',title="Power Draw Over Time", y_label="Power (Watts)")
    plot_metric(pivot_df, 'SMACT',title="SM Utilization Over Time", y_label="SM Utilization (%)")
    # plot_box(df,'GRACT',title='Distribution of Average GPU Utilization per Node',y_label='Mean GPU Utilization (%)')
    plot_box_and_violin(df, 'GRACT', 'GPU Utilization', 'GPU Utilization (%)')
    plot_box_and_violin(df,'POWER', 'Power Draw', 'Power (Watts)')
    plot_box_and_violin(df,'TMPTR', 'GPU Temperature', 'Temperature (°C)')

    #not a plot
    stragglers = identify_stragglers(df, 'GRACT', 5)
    stragglers.head(10)

    plot_correlations(df,metrics = ['GRACT', 'SMACT', 'POWER', 'TMPTR', 'PCITX', 'PCIRX', 'NVLTX', 'NVLRX'])
    plot_nvlink_error_metrics(df)

