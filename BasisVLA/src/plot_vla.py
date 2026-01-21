import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import textwrap
from matplotlib.patches import Patch

# Design Constants matching the OpenVLA PDF style
COLORS_MAP = {
    'DP': '#7B8AF1',               # Blue (Diffusion Policy / Baseline)
    'DP + basis': '#B0B0B0',       # Gray
    'DP + plucker': '#D98A4B',     # Orange
    'DP + rescale basis': '#E7B1D9',# Pink (or Light Purple)
    'vanilla': '#7B8AF1',          # Blue
    'ours': '#B85450'              # Red
}

DEFAULT_COLORS = ['#7B8AF1', '#B0B0B0', '#D98A4B', '#E7B1D9', '#B85450', '#82A171', '#4D4D4D']

def parse_single_task_csv(file_path):
    """
    Parses a single-task CSV where rows are seeds/average and columns are models.
    Task name is derived from filename.
    """
    # line 27: Deriving task name from filename
    task_name = os.path.splitext(os.path.basename(file_path))[0]
    df = pd.read_csv(file_path)

    # line 31: Extracting model names from columns
    models = df.columns[1:].tolist()

    # line 34: Separating seeds from average row
    seeds_df = df[df.iloc[:, 0].str.contains('seed', case=False, na=False)]
    avg_row = df[df.iloc[:, 0].str.lower() == 'average']

    averages = avg_row.iloc[0, 1:].values.astype(float)

    # line 40: Calculating standard deviation if seeds exist
    if not seeds_df.empty:
        stds = seeds_df.iloc[:, 1:].astype(float).std().values
    else:
        stds = np.zeros_like(averages)

    return task_name, models, averages, stds

def parse_multi_task_csv(file_path):
    """
    Parses a multi-task CSV where rows are tasks and columns are models.
    Returns a list of task data.
    """
    # line 53: Reading the multi-task CSV
    df = pd.read_csv(file_path)
    models = df.columns[1:].tolist()

    tasks_data = []
    # line 58: Extracting each task row (excluding the 'average' row)
    for index, row in df.iterrows():
        label = str(row.iloc[0]).lower()
        if label == 'average' or label == 'model':
            continue

        task_name = row.iloc[0]
        averages = row.iloc[1:].values.astype(float)
        stds = np.zeros_like(averages) # Multi-task CSV might not have seeds
        tasks_data.append((task_name, models, averages, stds))

    return tasks_data

def format_task_name(name, max_width=20):
    name = name.replace('_', ' ')
    return "\n".join(textwrap.wrap(name, width=max_width))

def create_plot(tasks, models, averages, stds, output_name, title=None, output_dir='../outputs'):
    """
    Generic plotting function.
    tasks: list of task names
    models: list of model names
    averages: 2D array [n_tasks, n_models]
    stds: 2D array [n_tasks, n_models]
    """
    n_tasks = len(tasks)
    n_models = len(models)

    # Calculate group average
    group_avg = np.mean(averages, axis=0)
    # Average STD is shown as the variation across tasks
    group_std = np.std(averages, axis=0)

    plot_tasks = ['Average'] + [format_task_name(t) for t in tasks]
    plot_averages = np.vstack([group_avg, averages])
    plot_stds = np.vstack([group_std, stds])

    fig_width = max(10, len(plot_tasks) * 2.5)
    fig, ax = plt.subplots(figsize=(fig_width, 7))

    bar_width = 0.14
    task_indices = np.arange(len(plot_tasks))

    for i in range(n_models):
        model_name = models[i]
        positions = task_indices + (i - (n_models - 1) / 2) * (bar_width + 0.02)
        color = COLORS_MAP.get(model_name, DEFAULT_COLORS[i % len(DEFAULT_COLORS)])

        # line 105: Drawing bars
        bars = ax.bar(positions, plot_averages[:, i], bar_width, label=model_name, color=color, zorder=3, alpha=0.9)
        # line 107: Adding error bars
        ax.errorbar(positions, plot_averages[:, i], yerr=plot_stds[:, i], fmt='none', ecolor='#333333', alpha=0.7, capsize=4, zorder=4)

        # lines 110-114: Value labels
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, height + 1.5, f'{height:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold', color='#444444')

    ax.set_ylabel('Success Rate (%)', fontsize=14, fontweight='bold', labelpad=15)
    ax.set_xticks(task_indices)
    ax.set_xticklabels(plot_tasks, fontsize=11, fontweight='medium')
    ax.set_ylim(0, 115)
    ax.yaxis.grid(True, linestyle='--', color='gray', alpha=0.2, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axvspan(-0.5, 0.5, color='#F0F0F0', zorder=0)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=min(n_models, 4), frameon=True, fontsize=12, edgecolor='#CCCCCC', facecolor='#FFFFFF')

    if title:
        plt.title(title, pad=30, fontsize=18, fontweight='bold')

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, output_name)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    # line 140: Setting up paths relative to BasisVLA structure
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    single_task_dir = os.path.join(base_dir, 'dataset', 'tasks', 'single-task')
    multi_task_dir = os.path.join(base_dir, 'dataset', 'tasks', 'multi-task')
    output_dir = os.path.join(base_dir, 'outputs')

    # Example: Plot Single Tasks
    # line 147: Processing all single-task CSV files
    single_task_files = glob.glob(os.path.join(single_task_dir, '*.csv'))
    if single_task_files:
        tasks = []
        all_averages = []
        all_stds = []
        models = []
        for f in sorted(single_task_files):
            t_name, m_names, avgs, stds = parse_single_task_csv(f)
            tasks.append(t_name)
            all_averages.append(avgs)
            all_stds.append(stds)
            models = m_names # assume same models

        create_plot(tasks, models, np.array(all_averages), np.array(all_stds), 'single_task_comparison.png', "Single-Task Model Performance", output_dir)

    # Example: Plot Multi Task groups
    # line 164: Processing multi-task files individually
    multi_task_files = glob.glob(os.path.join(multi_task_dir, '*.csv'))
    for f in multi_task_files:
        tasks_data = parse_multi_task_csv(f)
        if tasks_data:
            t_names = [d[0] for d in tasks_data]
            m_names = tasks_data[0][1]
            avgs = np.array([d[2] for d in tasks_data])
            stds = np.array([d[3] for d in tasks_data])
            file_name = os.path.splitext(os.path.basename(f))[0]
            create_plot(t_names, m_names, avgs, stds, f'{file_name}_comparison.png', f"{file_name} Evaluation", output_dir)
