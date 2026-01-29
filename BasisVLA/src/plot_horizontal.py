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


def plot_new_object_location_barh(csv_path: str, output_path: str):
    plt.rcParams["font.family"] = "Times New Roman"

    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]

    # required cols
    assert "environment" in df.columns, "CSV must have 'environment' column"
    assert "baseline" in df.columns, "CSV must have 'baseline' column"
    assert "ours" in df.columns, "CSV must have 'ours' column"

    envs = df["environment"].astype(str).tolist()
    baseline = df["baseline"].astype(float).to_numpy()
    ours = df["ours"].astype(float).to_numpy()
    delta = ours - baseline

    # optional: baseline_name (DP / SmolVLA)
    baseline_name = None
    if "baseline_name" in df.columns:
        baseline_name = df["baseline_name"].astype(str).tolist()

    # colors: constants에서 가져오되, 없으면 fallback
    c_base = COLOR_MAP_MODEL.get("baseline", "#4D4D4D")
    c_ours = COLOR_MAP_MODEL.get("ours", COLOR_MAP_MODEL.get("axisguide", "#7A7A7A"))

    fig, ax = plt.subplots(figsize=(7.4, 2.8))

    y = np.arange(len(envs))
    h = 0.34

    # baseline/ours barh (paired)
    ax.barh(y - h/2, baseline, height=h, color=c_base, edgecolor="black",
            linewidth=0.75, alpha=0.9, zorder=3, label="Baseline")
    ax.barh(y + h/2, ours, height=h, color=c_ours, edgecolor="black",
            linewidth=0.75, alpha=0.9, zorder=3, label="Baseline + AxisGuide (Ours)")

    # y labels
    ylabels = [format_env(e) for e in envs]
    # baseline 이름도 같이 보여주고 싶으면 (선택)
    if baseline_name is not None:
        ylabels = [f"{format_env(e)}\n(Baseline: {bn})" for e, bn in zip(envs, baseline_name)]

    ax.set_yticks(y)
    ax.set_yticklabels(ylabels, fontsize=13)
    ax.invert_yaxis()

    # x axis
    ax.set_xlabel("Success Rate (%)", fontsize=15, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.xaxis.grid(True, linestyle="--", color="gray", alpha=0.2, zorder=0)

    # spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # title (네 그룹 캡션 톤)
    ax.set_title(
        "Generalization to New Object Locations (Pick up)",
        fontsize=16, fontweight="bold", color=COLOR_GROUP_CAPTION, pad=10
    )

    # value texts + delta texts
    for i in range(len(envs)):
        bv, ov, dv = baseline[i], ours[i], delta[i]

        ax.text(bv + 1.0, y[i] - h/2, f"{bv:.2f}", va="center", ha="left", fontsize=12)
        ax.text(ov + 1.0, y[i] + h/2, f"{ov:.2f}", va="center", ha="left",
                fontsize=12, fontweight="bold")

        ax.text(ov + 1.0, y[i] + h/2 - 0.20, f"(+{dv:.2f}↑)",
                va="center", ha="left", fontsize=12,
                fontweight="bold", color=COLOR_EMPH)

    # legend
    ax.legend(loc="lower right", frameon=True, fontsize=11, edgecolor="#CCCCCC")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, "dataset", "tasks", "new-location", "new_object_location.csv")
    out_path = os.path.join(base_dir, "outputs", "new_object_location_barh.pdf")
    plot_new_object_location_barh(csv_path, out_path)
