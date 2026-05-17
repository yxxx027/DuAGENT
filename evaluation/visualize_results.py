import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
import os


def plot_score_comparison_bar(baseline_scores: list[float],
                               improved_scores: list[float],
                               output_path: str,
                               labels: list[str] = None):
    if labels is None:
        labels = [f"Prompt {i}" for i in range(len(baseline_scores))]
    x = range(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar([i - width / 2 for i in x], baseline_scores, width, label="Baseline", color="#4C72B0")
    ax.bar([i + width / 2 for i in x], improved_scores, width, label="Improved", color="#DD8452")
    ax.set_xlabel("Prompt")
    ax.set_ylabel("CLIP Score")
    ax.set_title("CLIP Score Comparison: Baseline vs Improved")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, max(max(baseline_scores), max(improved_scores)) * 1.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_attribute_heatmap(attribute_scores: dict, output_path: str):
    objects = list(attribute_scores.keys())
    attr_names = set()
    for obj_data in attribute_scores.values():
        attr_names.update(obj_data["attributes"].keys())
    attr_names = sorted(attr_names)
    data = []
    for obj in objects:
        row = []
        for attr in attr_names:
            row.append(attribute_scores[obj]["attributes"].get(attr, 0.0))
        data.append(row)
    fig, ax = plt.subplots(figsize=(max(8, len(attr_names)), max(4, len(objects))))
    im = ax.imshow(data, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(attr_names)))
    ax.set_xticklabels([a.split()[-1] for a in attr_names], rotation=45, ha="right")
    ax.set_yticks(range(len(objects)))
    ax.set_yticklabels(objects)
    ax.set_title("Per-Attribute CLIP Score Heatmap")
    for i in range(len(objects)):
        for j in range(len(attr_names)):
            ax.text(j, i, f"{data[i][j]:.1f}", ha="center", va="center", fontsize=8)
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def generate_report_table(baseline_scores: list[float],
                           improved_scores: list[float],
                           output_path: str):
    lines = []
    lines.append("| Prompt | Baseline CLIP | Improved CLIP | Delta |")
    lines.append("|--------|--------------|--------------|-------|")
    for i, (b, imp) in enumerate(zip(baseline_scores, improved_scores)):
        delta = imp - b
        sign = "+" if delta > 0 else ""
        lines.append(f"| Prompt {i} | {b:.4f} | {imp:.4f} | {sign}{delta:.4f} |")
    avg_b = sum(baseline_scores) / len(baseline_scores) if baseline_scores else 0
    avg_i = sum(improved_scores) / len(improved_scores) if improved_scores else 0
    avg_d = avg_i - avg_b
    sign = "+" if avg_d > 0 else ""
    lines.append(f"| **Average** | **{avg_b:.4f}** | **{avg_i:.4f}** | **{sign}{avg_d:.4f}** |")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
