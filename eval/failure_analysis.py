"""Phase 4: deliberately probe the system with questions it should struggle
with, and record what actually happens -- no-answer (out-of-corpus),
ambiguous, and multi-document-synthesis questions, run through the
confidence-gated query path in ragpipeline.rag.

    PYTHONPATH=. ./.venv/bin/python -m eval.failure_analysis
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ragpipeline.rag import answer_question  # noqa: E402

CASES = [
    # -- no-answer: genuinely off-topic (no company in the corpus is relevant) --
    {"category": "no_answer_off_topic", "question": "What is the capital of France?"},
    {"category": "no_answer_off_topic", "question": "How do I bake a chocolate cake?"},
    # -- no-answer: topically on-target, but about a company NOT in the corpus --
    {"category": "no_answer_wrong_company", "question": "What are Netflix's main risk factors?"},
    {"category": "no_answer_wrong_company", "question": "What did Amazon report for AWS cloud revenue?"},
    # -- ambiguous: no company specified, or an underspecified referent --
    {"category": "ambiguous", "question": "What are the risks?"},
    {"category": "ambiguous", "question": "How much revenue did the company make last year?"},
    # -- multi-document synthesis: requires combining >1 company's filing --
    {"category": "multi_doc", "question": "Compare the cybersecurity risk factors between Apple and Microsoft."},
    {"category": "multi_doc", "question": "Which of these companies has the largest litigation risk?"},
]

RESULTS_PATH = Path(__file__).resolve().parent / "failure_analysis_results.json"


def main() -> None:
    results = []
    for case in CASES:
        result = answer_question(case["question"])
        record = {
            "category": case["category"],
            "question": case["question"],
            "refused": result.refused,
            "top_score": round(result.top_score, 3),
            "retrieved_tickers": [c.ticker for c in result.chunks],
            "answer": result.answer,
        }
        results.append(record)
        print(f"[{case['category']}] {case['question']}")
        print(f"  top_score={record['top_score']} tickers={record['retrieved_tickers']} refused={record['refused']}")
        print(f"  answer: {result.answer[:300]}\n")

    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"Wrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
