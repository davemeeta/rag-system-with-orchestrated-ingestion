"""Render eval/results.json into a comparison chart for the README.

Not a core project dependency — matplotlib is only needed to regenerate this
chart, so it's intentionally left out of requirements.txt:

    pip install matplotlib
    python -m eval.make_chart
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt

RESULTS_PATH = Path(__file__).resolve().parent / "results.json"
OUT_PATH = Path(__file__).resolve().parent / "comparison_chart.png"


def main() -> None:
    results = json.loads(RESULTS_PATH.read_text())
    labels = [f"{r['chunk_strategy']} /\n{r['retrieval_strategy']}" for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    x = range(len(labels))
    width = 0.35

    ax = axes[0]
    ax.bar([i - width / 2 for i in x], [r["avg_precision_at_k"] for r in results], width, label="precision@5", color="#4C72B0")
    ax.bar([i + width / 2 for i in x], [r["avg_mrr"] for r in results], width, label="MRR", color="#DD8452")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_title("Retrieval quality (higher is better)")
    ax.legend(loc="lower right")

    ax = axes[1]
    ax.bar([i - width / 2 for i in x], [r["avg_faithfulness"] for r in results], width, label="faithfulness", color="#4C72B0")
    ax.bar([i + width / 2 for i in x], [r["avg_relevance"] for r in results], width, label="relevance", color="#DD8452")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 5.3)
    ax.set_title("Generation quality, LLM-judged 1-5 (higher is better)")
    ax.legend(loc="lower right")

    fig.suptitle("Chunking x Retrieval comparison — 24-question eval set")
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
