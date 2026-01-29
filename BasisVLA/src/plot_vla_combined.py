import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import textwrap
from matplotlib.gridspec import GridSpec

# line 8: Robust import logic to handle sibling files regardless of execution path
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from constants import COLOR_EMPH, COLOR_MAP_MODEL, COLOR_GROUP_CAPTION

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
    # line 66: Increasing wrap width to 20 to use more horizontal space
    return "\n".join(textwrap.wrap(prefix + name.title(), width=20))

def create_combined_plot(single_task_files, multi_task_files, output_path):
    # line 70: Setting the global font to Times New Roman
    plt.rcParams['font.family'] = 'Times New Roman'

    # 1. Collect and Sort Data
    single_data = [parse_single_task_csv(f) for f in sorted(single_task_files)]

    # line 74-76: Sorting multi-task files in the requested order: Spatial -> Object -> Goal -> Long
    order_map = {'spatial': 0, 'object': 1, 'goal': 2, 'long': 3}
    def multi_sort_key(f):
        name = os.path.splitext(os.path.basename(f))[0].lower().replace('libero-', '')
        return order_map.get(name, 99)

    multi_task_files = sorted(multi_task_files, key=multi_sort_key)
    multi_data = [parse_multi_task_csv(f) for f in multi_task_files]

    n_single = len(single_data)
    n_multi = len(multi_data)
    total_plots = n_single + n_multi

    # 2. Setup Figure
    fig = plt.figure(figsize=(max(15, total_plots * 2.75), 8))
    gs = GridSpec(1, total_plots, figure=fig, wspace=0.15)
    axes = [fig.add_subplot(gs[0, i]) for i in range(total_plots)]

    all_handles = []
    all_labels = []
    seen_labels = set()

    single_handles = []
    single_labels = []
    multi_handles = []
    multi_labels = []

    # Plotting loop for all datasets
    for i in range(total_plots):
        ax = axes[i]
        is_single = (i < n_single)
        if is_single:
            name, models, avgs, stds = single_data[i]
            bar_width = 0.42
        else:
            name, models, avgs, stds = multi_data[i - n_single]
            bar_width = 0.33

        x = np.arange(1)

        for j, model in enumerate(models):
            # line 116: Offset calculation for very skinny touching bars
            offset = (j - (len(models) - 1) / 2) * bar_width
            pos = x + offset

            color = COLOR_MAP_MODEL.get(model.lower(), '#4D4D4D')

            # line 98: Set a balanced bar width (skinny but readable)
            # line 124: Adding a border to the bars using edgecolor and linewidth
            bar = ax.bar(pos, avgs[j], bar_width, color=color, edgecolor="black", linewidth=0.75, zorder=3, alpha=0.9, label=model)
            # ax.errorbar(pos, avgs[j], yerr=stds[j], fmt='none', ecolor=COLOR_EMPH, capsize=4, zorder=4)

            val_text = f'{avgs[j]:.1f}'
            if 'ours' in model.lower() and j > 0:
                advantage = avgs[j] - avgs[0]
                # line 128: Improvement metric in COLOR_EMPH (enlarged to 12)
                # ax.text(pos, avgs[j] + 6, f'(+{advantage:.1f})', ha='center', va='bottom',
                #         fontsize=12, fontweight='bold', color=COLOR_EMPH)

            # line 131: Success rate label (enlarged to 14)
            ax.text(pos, avgs[j] + 0.5, val_text, ha='center', va='bottom', fontsize=14)

            # line 135-144: Splitting legend collectors by group
            if is_single:
                if model not in single_labels:
                    single_handles.append(bar[0])
                    single_labels.append(model)
            else:
                if model not in multi_labels:
                    multi_handles.append(bar[0])
                    multi_labels.append(model)

        # Aesthetics
        ax.set_xticks([])
        # line 147: Enlarged task name font to 14
        ax.set_xlabel(format_name(name), fontsize=14, labelpad=15, fontweight='medium')

        # line 151-155: Independent Y-axis limits for each group
        is_single_group = (i < n_single)
        if is_single_group:
            # Single-Task Limit (0-115 to show up to 100 with margin)
            ax.set_ylim(60, 115)
            ax.set_yticks([60, 80, 100])
            ax.set_yticklabels(['60', '80', '100'])
        else:
            # Multi-Task Limit (Can be different, e.g., 0-100 to show up to 80/90)
            # Adjusting here to 0-105 as an example of group-specific limits
            ax.set_ylim(40, 105)
            ax.set_yticks([40, 60, 80, 100])

        # line 164: Enforcing a wider fixed horizontal range
        ax.set_xlim(-0.7, 0.7)

        # line 166-174: Showing Y-axis scale for the start of EACH group
        is_group_head = (i == 0 or i == n_single)
        if not is_group_head:
            # Hide Y-axis for interior plots
            ax.set_yticklabels([])
            ax.tick_params(axis='y', which='both', left=False)
            ax.spines['left'].set_visible(False)
        else:
            # Show Y-axis and label for group heads
            if i == 0:
                ax.set_ylabel('Success Rate (%)', fontsize=18, fontweight='bold')
            ax.tick_params(axis='y', which='both', left=True)
            ax.spines['left'].set_visible(True)

        ax.yaxis.grid(True, linestyle='--', color='gray', alpha=0.2, zorder=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # 3. Group Captions and Dividers
    fig.subplots_adjust(bottom=0.25) # Increase bottom space for legends

    s_start = axes[0].get_position().x0
    s_end = axes[n_single-1].get_position().x1
    m_start = axes[n_single].get_position().x0
    m_end = axes[-1].get_position().x1

    # lines 170-172: Group captions
    # lines 175-176: Enlarged group captions to 18
    fig.text((s_start + s_end)/2, 0.125, 'Single-Task Evaluation', ha='center', fontsize=18, fontweight="bold", color=COLOR_GROUP_CAPTION)
    fig.text((m_start + m_end)/2, 0.125, 'Multi-Task Evaluation (LIBERO)', ha='center', fontsize=18, fontweight="bold", color=COLOR_GROUP_CAPTION)

    # lines 178-181: Brackets with curly (tick) ends
    line_y = 0.16
    tick_h = 0.01  # Height of the ends

    # Single-Task Bracket
    fig.add_artist(plt.Line2D([s_start, s_end], [line_y, line_y], transform=fig.transFigure, color=COLOR_GROUP_CAPTION, lw=1.5))
    fig.add_artist(plt.Line2D([s_start, s_start], [line_y, line_y + tick_h], transform=fig.transFigure, color=COLOR_GROUP_CAPTION, lw=1.5))
    fig.add_artist(plt.Line2D([s_end, s_end], [line_y, line_y + tick_h], transform=fig.transFigure, color=COLOR_GROUP_CAPTION, lw=1.5))

    # Multi-Task Bracket
    fig.add_artist(plt.Line2D([m_start, m_end], [line_y, line_y], transform=fig.transFigure, color=COLOR_GROUP_CAPTION, lw=1.5))
    fig.add_artist(plt.Line2D([m_start, m_start], [line_y, line_y + tick_h], transform=fig.transFigure, color=COLOR_GROUP_CAPTION, lw=1.5))
    fig.add_artist(plt.Line2D([m_end, m_end], [line_y, line_y + tick_h], transform=fig.transFigure, color=COLOR_GROUP_CAPTION, lw=1.5))

    # 4. Split Legends
    # line 195, 200: Positioning split legends at the top right of each group area
    fig.legend(single_handles, single_labels, loc='upper right',
               bbox_to_anchor=(s_end, 0.88), ncol=1,
               frameon=True, fontsize=14, edgecolor='#CCCCCC')

    fig.legend(multi_handles, multi_labels, loc='upper right',
               bbox_to_anchor=(m_end, 0.88), ncol=1,
               frameon=True, fontsize=14, edgecolor='#CCCCCC')

    # line 202: Enlarged title to 28
    # plt.suptitle('Performance Advantage of Visual Basis in Robotics Policies', fontsize=28, fontweight='bold', y=0.95)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Combined plot with advantage metrics saved to {output_path}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    single_dir = os.path.join(base_dir, 'dataset', 'tasks', 'single-task')
    multi_dir = os.path.join(base_dir, 'dataset', 'tasks', 'multi-task')
    output_dir = os.path.join(base_dir, 'outputs')

    s_files = glob.glob(os.path.join(single_dir, '*.csv'))
    m_files = glob.glob(os.path.join(multi_dir, '*.csv'))

    if s_files or m_files:
        # line 222: Saving the finalized figure as a vector PDF
        create_combined_plot(s_files, m_files, os.path.join(output_dir, 'combined_vla_advantage_figure.pdf'))
    else:
        print("No CSV files found.")
