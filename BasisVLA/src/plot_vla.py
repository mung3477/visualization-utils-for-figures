import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import textwrap
import sys
from matplotlib.patches import Patch

# Robust import logic to handle sibling files regardless of execution path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from constants import COLOR_EMPH, COLOR_MAP_MODEL, COLOR_GROUP_CAPTION
except ImportError:
    # Fallback if constants.py is missing
    COLOR_EMPH = "#ee5b00"
    COLOR_MAP_MODEL = {}
    COLOR_GROUP_CAPTION = '#000000BF'

DEFAULT_COLORS = ['#7B8AF1', '#B0B0B0', '#D98A4B', '#E7B1D9', '#B85450', '#82A171', '#4D4D4D']

def parse_single_task_csv(file_path):
    """Parses single-task CSV (rows: seeds/average, cols: models)."""
    task_name = os.path.splitext(os.path.basename(file_path))[0]
    df = pd.read_csv(file_path)
    # Standardize column names (remove whitespace)
    df.columns = [c.strip() for c in df.columns]
    models = df.columns[1:].tolist()

    seeds_df = df[df.iloc[:, 0].str.contains('seed', case=False, na=False)]
    avg_row = df[df.iloc[:, 0].str.lower() == 'average']

    if avg_row.empty:
        averages = df.iloc[:, 1:].astype(float).mean().values
    else:
        averages = avg_row.iloc[0, 1:].values.astype(float)

    if not seeds_df.empty:
        stds = seeds_df.iloc[:, 1:].astype(float).std().values
    else:
        stds = np.zeros_like(averages)

    return task_name, models, averages, stds

def parse_multi_task_csv(file_path):
    """Parses multi-task CSV (rows: tasks/average, cols: models)."""
    df = pd.read_csv(file_path)
    df.columns = [c.strip() for c in df.columns]
    models = df.columns[1:].tolist()
    dataset_name = os.path.splitext(os.path.basename(file_path))[0]

    avg_row = df[df.iloc[:, 0].str.lower() == 'average']
    task_rows = df[~df.iloc[:, 0].str.lower().isin(['average', 'model'])]

    if not avg_row.empty:
        averages = avg_row.iloc[0, 1:].values.astype(float)
    else:
        averages = task_rows.iloc[:, 1:].astype(float).mean().values

    if not task_rows.empty:
        stds = task_rows.iloc[:, 1:].astype(float).std().values
    else:
        stds = np.zeros_like(averages)

    return dataset_name, models, averages, stds

def format_name(name):
    """Formats task/dataset names for display."""
    name = name.replace('_', ' ').replace('-', ' ')
    if 'LIBERO' in name:
        name = name.replace('LIBERO', '')
        prefix = "LIBERO"
    else:
        prefix = ""
    # Increasing wrap width to 20 to use more horizontal space
    return "\n".join(textwrap.wrap(prefix + name.title(), width=20))

def create_plot(tasks, models, averages, stds, output_name, title=None, output_dir='../outputs'):
    """
    Generic plotting function.
    tasks: list of task names
    models: list of model names
    averages: 2D array [n_tasks, n_models]
    stds: 2D array [n_tasks, n_models]
    """
    # Setting the global font to Times New Roman
    plt.rcParams['font.family'] = 'Times New Roman'

    n_tasks = len(tasks)
    n_models = len(models)

    # Calculate group average
    group_avg = np.mean(averages, axis=0)
    # Average STD is shown as the variation across tasks
    group_std = np.std(averages, axis=0)

    plot_tasks = ['Average'] + [format_name(t) for t in tasks]
    plot_averages = np.vstack([group_avg, averages])
    plot_stds = np.vstack([group_std, stds])

    fig_width = max(10, len(plot_tasks) * 3)
    fig, ax = plt.subplots(figsize=(fig_width, 7))

    bar_width = 0.18
    task_indices = np.arange(len(plot_tasks))

    for i in range(n_models):
        model_name = models[i]
        positions = task_indices + (i - (n_models - 1) / 2) * bar_width

        # Use COLOR_MAP_MODEL with lower-case matching
        color = COLOR_MAP_MODEL.get(model_name.lower(), DEFAULT_COLORS[i % len(DEFAULT_COLORS)])

        # Drawing bars with black edges
        bars = ax.bar(positions, plot_averages[:, i], bar_width, label=model_name,
                      color=color, edgecolor="black", linewidth=0.75, zorder=3, alpha=0.9)

        # Error bars are optional - commenting out to match plot_vla_combined style
        # ax.errorbar(positions, plot_averages[:, i], yerr=plot_stds[:, i], fmt='none',
        #             ecolor='#333333', alpha=0.7, capsize=4, zorder=4)

        # Success rate labels
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, height + 0.5, f'{height:.1f}',
                        ha='center', va='bottom', fontsize=14)

    ax.set_ylabel('Success Rate (%)', fontsize=18, fontweight='bold', labelpad=15)
    ax.set_xticks(task_indices)
    ax.set_xticklabels(plot_tasks, fontsize=15, fontweight='medium')
    ax.set_ylim(50, 90)
    ax.set_yticks([50, 70, 90])
    ax.yaxis.grid(True, linestyle='--', color='gray', alpha=0.2, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Highlight the Average column
    ax.axvspan(-0.5, 0.5, color='#F0F0F0', zorder=0)

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=min(n_models, 4),
              frameon=True, fontsize=16, edgecolor='#CCCCCC', facecolor='#FFFFFF')

    if title:
        plt.title(title, pad=30, fontsize=24, fontweight='bold')

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, output_name)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    single_task_dir = os.path.join(base_dir, 'dataset', 'tasks', 'single-task')
    multi_task_dir = os.path.join(base_dir, 'dataset', 'tasks', 'multi-task')
    output_dir = os.path.join(base_dir, 'outputs')

    # 1. Process Single Tasks
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
            models = m_names

        create_plot(tasks, models, np.array(all_averages), np.array(all_stds),
                    'single_task_comparison.png', "Single-Task Model Performance", output_dir)

    # 2. Process Multi-Task groups (Aggregated View)
    multi_task_files = glob.glob(os.path.join(multi_task_dir, '*.csv'))
    if multi_task_files:
        m_tasks = []
        m_averages = []
        m_stds = []
        m_models = []
        for f in sorted(multi_task_files):
            t_name, models, avgs, stds = parse_multi_task_csv(f)
            m_tasks.append(t_name)
            m_averages.append(avgs)
            m_stds.append(stds)
            m_models = models

        create_plot(m_tasks, m_models, np.array(m_averages), np.array(m_stds),
                    'multi_task_benchmarks_comparison.png', "Multi-Task Benchmark Performance", output_dir)
