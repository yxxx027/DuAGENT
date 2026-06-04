import argparse
import json
import os
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _ensure_dir(output_path: str):
    directory = os.path.dirname(output_path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def _default_labels(n: int) -> list[str]:
    return [f"P{i + 1}" for i in range(n)]


def plot_score_comparison_bar(
    baseline_scores: list[float],
    improved_scores: list[float],
    output_path: str,
    labels: list[str] | None = None,
):
    if len(baseline_scores) != len(improved_scores):
        raise ValueError("baseline_scores and improved_scores must have the same length.")

    if labels is None:
        labels = _default_labels(len(baseline_scores))

    x = list(range(len(labels)))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar([i - width / 2 for i in x], baseline_scores, width, label="Baseline")
    ax.bar([i + width / 2 for i in x], improved_scores, width, label="Improved")

    ax.set_xlabel("Prompt")
    ax.set_ylabel("Normalized CLIP Score (0-10)")
    ax.set_title("CLIP Score Comparison: Baseline vs Improved")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylim(0, 10)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    plt.tight_layout()
    _ensure_dir(output_path)
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_win_rate_pie(results: list[dict[str, Any]], output_path: str):
    baseline_wins = sum(1 for r in results if r.get("winner") == "Baseline")
    improved_wins = sum(1 for r in results if r.get("winner") == "Improved")
    ties = sum(1 for r in results if r.get("winner") == "Tie")

    labels = ["Baseline Win", "Improved Win", "Tie"]
    values = [baseline_wins, improved_wins, ties]

    if sum(values) == 0:
        values = [0, 0, 1]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=90)
    ax.set_title("Win Rate: Baseline vs Improved")
    ax.axis("equal")

    _ensure_dir(output_path)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_delta_distribution(
    baseline_scores: list[float],
    improved_scores: list[float],
    output_path: str,
):
    if len(baseline_scores) != len(improved_scores):
        raise ValueError("baseline_scores and improved_scores must have the same length.")

    deltas = [i - b for b, i in zip(baseline_scores, improved_scores)]
    labels = _default_labels(len(deltas))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(labels, deltas)
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_xlabel("Prompt")
    ax.set_ylabel("Delta Score")
    ax.set_title("Improvement Delta Distribution")
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    plt.tight_layout()
    _ensure_dir(output_path)
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_attribute_heatmap(attribute_scores: dict, output_path: str):
    objects = list(attribute_scores.keys())
    attr_names = set()

    for obj_data in attribute_scores.values():
        attr_names.update(obj_data.get("attributes", {}).keys())

    attr_names = sorted(attr_names)

    if not objects or not attr_names:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No attribute scores available", ha="center", va="center")
        ax.axis("off")
        _ensure_dir(output_path)
        plt.savefig(output_path, dpi=200)
        plt.close()
        return

    data = []
    for obj in objects:
        row = []
        for attr in attr_names:
            row.append(attribute_scores[obj].get("attributes", {}).get(attr, 0.0))
        data.append(row)

    fig, ax = plt.subplots(figsize=(max(8, len(attr_names)), max(4, len(objects))))
    im = ax.imshow(data, aspect="auto")

    ax.set_xticks(range(len(attr_names)))
    ax.set_xticklabels(attr_names, rotation=45, ha="right")
    ax.set_yticks(range(len(objects)))
    ax.set_yticklabels(objects)
    ax.set_title("Per-Attribute CLIP Score Heatmap")

    for i in range(len(objects)):
        for j in range(len(attr_names)):
            ax.text(j, i, f"{data[i][j]:.1f}", ha="center", va="center", fontsize=8)

    plt.colorbar(im)
    plt.tight_layout()
    _ensure_dir(output_path)
    plt.savefig(output_path, dpi=200)
    plt.close()


def generate_report_table(
    baseline_scores: list[float],
    improved_scores: list[float],
    output_path: str,
    prompts: list[str] | None = None,
):
    if len(baseline_scores) != len(improved_scores):
        raise ValueError("baseline_scores and improved_scores must have the same length.")

    if prompts is None:
        prompts = [f"Prompt {i + 1}" for i in range(len(baseline_scores))]

    lines = []
    lines.append("| Prompt | Baseline CLIP | Improved CLIP | Delta | Winner |")
    lines.append("|--------|---------------|---------------|-------|--------|")

    for i, (b, imp) in enumerate(zip(baseline_scores, improved_scores)):
        delta = imp - b
        sign = "+" if delta > 0 else ""

        if abs(delta) <= 0.15:
            winner = "Tie"
        elif delta > 0:
            winner = "Improved"
        else:
            winner = "Baseline"

        prompt_label = prompts[i]
        if len(prompt_label) > 60:
            prompt_label = prompt_label[:57] + "..."

        lines.append(
            f"| {prompt_label} | {b:.4f} | {imp:.4f} | {sign}{delta:.4f} | {winner} |"
        )

    avg_b = sum(baseline_scores) / len(baseline_scores) if baseline_scores else 0
    avg_i = sum(improved_scores) / len(improved_scores) if improved_scores else 0
    avg_d = avg_i - avg_b
    sign = "+" if avg_d > 0 else ""

    lines.append(
        f"| **Average** | **{avg_b:.4f}** | **{avg_i:.4f}** | **{sign}{avg_d:.4f}** | - |"
    )

    _ensure_dir(output_path)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def generate_all_visualizations_from_results(results_json: str, output_dir: str | None = None):
    with open(results_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    baseline_scores = data.get("baseline_scores", [])
    improved_scores = data.get("improved_scores", [])
    prompts = data.get("prompts", [])

    if not baseline_scores or not improved_scores:
        raise ValueError("results.json does not contain baseline_scores/improved_scores.")

    if output_dir is None:
        output_dir = os.path.dirname(results_json)

    os.makedirs(output_dir, exist_ok=True)

    labels = _default_labels(len(baseline_scores))

    plot_score_comparison_bar(
        baseline_scores,
        improved_scores,
        os.path.join(output_dir, "score_comparison.png"),
        labels=labels,
    )

    comparison_results = []
    for b, imp in zip(baseline_scores, improved_scores):
        delta = imp - b
        if abs(delta) <= 0.15:
            winner = "Tie"
        elif delta > 0:
            winner = "Improved"
        else:
            winner = "Baseline"

        comparison_results.append(
            {
                "baseline_score": b,
                "improved_score": imp,
                "delta": delta,
                "winner": winner,
            }
        )

    plot_win_rate_pie(
        comparison_results,
        os.path.join(output_dir, "win_rate.png"),
    )

    plot_delta_distribution(
        baseline_scores,
        improved_scores,
        os.path.join(output_dir, "delta_distribution.png"),
    )

    generate_report_table(
        baseline_scores,
        improved_scores,
        os.path.join(output_dir, "report_table.md"),
        prompts=prompts,
    )

    summary = {
        "num_prompts": len(baseline_scores),
        "baseline_avg": round(sum(baseline_scores) / len(baseline_scores), 4),
        "improved_avg": round(sum(improved_scores) / len(improved_scores), 4),
        "avg_delta": round(
            sum(i - b for b, i in zip(baseline_scores, improved_scores))
            / len(baseline_scores),
            4,
        ),
        "baseline_wins": sum(1 for r in comparison_results if r["winner"] == "Baseline"),
        "improved_wins": sum(1 for r in comparison_results if r["winner"] == "Improved"),
        "ties": sum(1 for r in comparison_results if r["winner"] == "Tie"),
    }

    with open(os.path.join(output_dir, "evaluation_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Visualization files generated:")
    print(os.path.join(output_dir, "score_comparison.png"))
    print(os.path.join(output_dir, "win_rate.png"))
    print(os.path.join(output_dir, "delta_distribution.png"))
    print(os.path.join(output_dir, "report_table.md"))
    print(os.path.join(output_dir, "evaluation_summary.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate evaluation visualizations.")
    parser.add_argument("--results", type=str, default="", help="Path to results.json")
    parser.add_argument("--output_dir", type=str, default="", help="Output directory")
    parser.add_argument("--demo", action="store_true", help="Run demo with fake scores")
    args = parser.parse_args()

    if args.demo:
        baseline = [5.8, 6.1, 4.9, 7.0, 5.5, 6.2, 6.8, 5.9, 6.0, 5.4]
        improved = [6.5, 6.3, 5.6, 6.8, 6.7, 6.6, 7.2, 6.4, 5.8, 6.1]
        results = []

        for b, imp in zip(baseline, improved):
            delta = imp - b
            if abs(delta) <= 0.15:
                winner = "Tie"
            elif delta > 0:
                winner = "Improved"
            else:
                winner = "Baseline"

            results.append(
                {
                    "baseline_score": b,
                    "improved_score": imp,
                    "delta": delta,
                    "winner": winner,
                }
            )

        os.makedirs("output/member3_demo", exist_ok=True)
        plot_score_comparison_bar(baseline, improved, "output/member3_demo/score_comparison.png")
        plot_win_rate_pie(results, "output/member3_demo/win_rate.png")
        plot_delta_distribution(baseline, improved, "output/member3_demo/delta_distribution.png")
        generate_report_table(baseline, improved, "output/member3_demo/report_table.md")

        print("Demo visualizations generated in output/member3_demo/")
    elif args.results:
        generate_all_visualizations_from_results(
            args.results,
            args.output_dir if args.output_dir else None,
        )
    else:
        baseline = [5.8, 6.1, 4.9, 7.0, 5.5, 6.2, 6.8, 5.9, 6.0, 5.4]
        improved = [6.5, 6.3, 5.6, 6.8, 6.7, 6.6, 7.2, 6.4, 5.8, 6.1]
        results = []

        for b, imp in zip(baseline, improved):
            delta = imp - b
            if abs(delta) <= 0.15:
                winner = "Tie"
            elif delta > 0:
                winner = "Improved"
            else:
                winner = "Baseline"

            results.append(
                {
                    "baseline_score": b,
                    "improved_score": imp,
                    "delta": delta,
                    "winner": winner,
                }
            )

        os.makedirs("output/member3_demo", exist_ok=True)
        plot_score_comparison_bar(baseline, improved, "output/member3_demo/score_comparison.png")
        plot_win_rate_pie(results, "output/member3_demo/win_rate.png")
        plot_delta_distribution(baseline, improved, "output/member3_demo/delta_distribution.png")
        generate_report_table(baseline, improved, "output/member3_demo/report_table.md")

        print("Demo visualizations generated in output/member3_demo/")
