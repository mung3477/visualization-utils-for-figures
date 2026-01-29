import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# sibling import (네 코드와 동일)
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from constants import COLOR_EMPH, COLOR_MAP_MODEL, COLOR_GROUP_CAPTION


def format_env(name: str) -> str:
    # Real-world / Simulation 정도는 wrap 필요 없지만, 통일감 위해 둠
    return name.replace("_", " ").replace("-", " ")


def _get_model_styles(name: str):
    """Helper to resolve model-specific keys, labels, and colors."""
    key = (name or "baseline")

    label_base = key
    label_ours = f"{label_base} + Ours"

    c_key = key.lower()
    color_base = COLOR_MAP_MODEL.get(c_key, "#4D4D4D")
    color_ours = COLOR_MAP_MODEL.get(f"{c_key} + ours", COLOR_MAP_MODEL.get("ours", "#7A7A7A"))

    return label_base, label_ours, color_base, color_ours


def plot_new_object_location_barh(csv_path: str, output_path: str):
    plt.rcParams["font.family"] = "Times New Roman"

    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]

    # Extract data (line 31-39)
    envs = df["environment"].astype(str).tolist()
    baseline = df["baseline"].astype(float).to_numpy()
    ours = df["ours"].astype(float).to_numpy()
    baseline_names = df["baseline_name"].tolist() if "baseline_name" in df.columns else [None] * len(envs)

    fig, ax = plt.subplots(figsize=(7.4, 3.2))
    y = np.arange(len(envs)) * 1.5 + 0.95  # Consistent spacing
    h = 0.42
    seen_labels = set()

    for i, (env, base_val, our_val, b_name) in enumerate(zip(envs, baseline, ours, baseline_names)):
        l_base, l_ours, c_base, c_ours = _get_model_styles(b_name)

        # Environment Label (moved back to the TOP-LEFT of the bars)
        env_text = format_env(env)
        ax.text(0.8, y[i] - 0.45, env_text, va="bottom", ha="left",
                fontsize=12, fontweight="bold", color="#222222", zorder=4)

        # Plot bars (lines 50-67)
        for val, offset, color, label in [(base_val, -h/2, c_base, l_base), (our_val, h/2, c_ours, l_ours)]:
            lbl = label if label not in seen_labels else None
            ax.barh(y[i] + offset, val, height=h, color=color, edgecolor="black",
                    linewidth=0.75, alpha=0.9, zorder=3, label=lbl)
            if lbl: seen_labels.add(lbl)

        # Value texts (lines 82-91)
        dv = our_val - base_val
        ax.text(base_val + 1.0, y[i] - h/2, f"{base_val:.2f}", va="center", ha="left", fontsize=12)
        ax.text(our_val + 1.0, y[i] + h/2, f"{our_val:.2f}", va="center", ha="left",
                fontsize=12, fontweight="bold")
        ax.text(our_val + 1.0, y[i] + h/2 - 0.30, f"(+{dv:.2f})",
                va="center", ha="left", fontsize=12,
                fontweight="bold", color=COLOR_EMPH)

    # Axes and labeling (lines 69-79)
    ax.set_yticks(y)
    ax.set_yticklabels([])  # Hide Y-axis labels (now moved into plot)
    ax.tick_params(axis='y', which='both', length=0)
    # Match top and bottom margins (top content at 0.35, so bottom edge at max(y)+0.69)
    ax.set_ylim(max(y) + 0.69, 0)


    ax.set_xlabel("Success Rate (%)", fontsize=15, fontweight="bold")
    ax.set_xlim(0, 80)
    ax.margins(x=0) # Remove internal horizontal padding
    ax.xaxis.grid(True, linestyle="--", color="gray", alpha=0.2, zorder=0)
    # The loop hiding top/right spines has been removed to create a full outline.
    # for spine in ["top", "right"]: ax.spines[spine].set_visible(False)

    # Legend (line 82-83) - Stacked vertically, top right
    ax.legend(
        loc='upper right',
        ncol=1,
        frameon=True,
        fontsize=10,
        edgecolor="#CCCCCC",
        handlelength=0.7,
        handleheight=0.7,
        labelspacing=0.4
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.0)
    print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, "dataset", "tasks", "new-location", "new_object_location.csv")
    out_path = os.path.join(base_dir, "outputs", "new_object_location_barh.png")
    plot_new_object_location_barh(csv_path, out_path)
