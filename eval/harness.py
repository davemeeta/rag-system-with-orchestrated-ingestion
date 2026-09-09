"""Phase 3 evaluation harness: hand-built retrieval + generation metrics,
run across chunking-strategy x retrieval-strategy configurations.

Retrieval metrics (deterministic, no LLM needed) stand in for RAGAS's
"context precision" / "context recall":
  - precision@k: fraction of top-k retrieved chunks from the expected ticker
  - hit@k ("context recall"): 1 if ANY top-k chunk is from the expected ticker
  - MRR: 1 / rank of the first correctly-sourced chunk (0 if none)

Generation metrics use llama3.2:3b as an LLM judge — RAGAS's own approach,
just without RAGAS itself. See README for why: RAGAS defaults to OpenAI
internally and needs a bespoke LangChain LLM-wrapper override per metric,
which is a lot of surface area to get reliable structured output out of a 3B
local judge model. A small hand-built judge prompt is more robust here, and
is the fallback the project spec itself calls out as acceptable:
  - faithfulness (1-5): is the answer supported by the retrieved context?
  - relevance (1-5): does the answer address the question asked?

    python -m eval.harness             # full 24-question x 4-config run
    python -m eval.harness --limit 2   # smoke test on the first N questions
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ollama import Client  # noqa: E402

from ragpipeline import config  # noqa: E402
from ragpipeline.generation.ollama_client import generate_answer  # noqa: E402
from ragpipeline.retrieval.dense import dense_search  # noqa: E402
from ragpipeline.retrieval.hybrid import hybrid_search  # noqa: E402

QUESTIONS_PATH = Path(__file__).resolve().parent / "questions.json"
RESULTS_PATH = Path(__file__).resolve().parent / "results.json"

TOP_K = 5

CONFIGS = [
    ("fixed_size", "dense"),
    ("fixed_size", "hybrid"),
    ("semantic", "dense"),
    ("semantic", "hybrid"),
]

_judge_client = Client(host=config.OLLAMA_HOST)

_JUDGE_PROMPT = """You are grading a RAG (retrieval-augmented generation) system's answer.

Question: {question}

Retrieved context:
{context}

Generated answer:
{answer}

Score the answer on two dimensions, each from 1 to 5:
- faithfulness: does the answer only make claims supported by the retrieved context (no fabrication)? 5 = fully supported, 1 = mostly fabricated/unsupported.
- relevance: does the answer actually address the question asked? 5 = directly and completely addresses it, 1 = off-topic or a non-answer.

Respond with ONLY a JSON object, no other text, in exactly this form:
{{"faithfulness": <int 1-5>, "relevance": <int 1-5>}}
"""


def _judge(question: str, context: str, answer: str) -> dict:
    prompt = _JUDGE_PROMPT.format(question=question, context=context[:4000], answer=answer)
    resp = _judge_client.chat(
        model=config.GEN_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0},
    )
    raw = resp["message"]["content"]
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {"faithfulness": None, "relevance": None}
    try:
        parsed = json.loads(match.group(0))
        return {
            "faithfulness": int(parsed["faithfulness"]),
            "relevance": int(parsed["relevance"]),
        }
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return {"faithfulness": None, "relevance": None}


def _retrieval_metrics(chunks, expected_ticker: str) -> dict:
    correct_flags = [c.ticker == expected_ticker for c in chunks]
    precision = sum(correct_flags) / len(correct_flags) if correct_flags else 0.0
    hit = 1.0 if any(correct_flags) else 0.0
    mrr = 0.0
    for rank, correct in enumerate(correct_flags, start=1):
        if correct:
            mrr = 1.0 / rank
            break
    return {"precision_at_k": precision, "hit_at_k": hit, "mrr": mrr}


def run_config(questions: list[dict], chunk_strategy: str, retrieval_strategy: str) -> dict:
    search_fn = dense_search if retrieval_strategy == "dense" else hybrid_search
    per_question = []

    for q in questions:
        t0 = time.time()
        chunks = search_fn(q["question"], top_k=TOP_K, strategy=chunk_strategy)
        retrieval_time = time.time() - t0

        ret_metrics = _retrieval_metrics(chunks, q["ticker"])
        context = "\n\n".join(c.text for c in chunks)
        answer = generate_answer(q["question"], chunks)
        judge = _judge(q["question"], context, answer)

        per_question.append(
            {
                "id": q["id"],
                "question": q["question"],
                "expected_ticker": q["ticker"],
                "retrieved_tickers": [c.ticker for c in chunks],
                "answer": answer,
                "retrieval_time_s": round(retrieval_time, 2),
                **ret_metrics,
                **judge,
            }
        )
        print(
            f"  [{chunk_strategy}/{retrieval_strategy}] {q['id']}: hit={ret_metrics['hit_at_k']} "
            f"prec={ret_metrics['precision_at_k']:.2f} faith={judge['faithfulness']} rel={judge['relevance']}"
        )

    def _avg(key):
        vals = [r[key] for r in per_question if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    return {
        "chunk_strategy": chunk_strategy,
        "retrieval_strategy": retrieval_strategy,
        "num_questions": len(questions),
        "avg_precision_at_k": _avg("precision_at_k"),
        "avg_hit_at_k": _avg("hit_at_k"),
        "avg_mrr": _avg("mrr"),
        "avg_faithfulness": _avg("faithfulness"),
        "avg_relevance": _avg("relevance"),
        "per_question": per_question,
    }


def print_table(all_results: list[dict]) -> None:
    print("\n\n=== Comparison table ===")
    header = (
        f"{'chunking':<12} {'retrieval':<8} {'precision@5':>12} {'hit@5':>8} "
        f"{'MRR':>6} {'faithfulness':>13} {'relevance':>10}"
    )
    print(header)
    print("-" * len(header))
    for r in all_results:
        print(
            f"{r['chunk_strategy']:<12} {r['retrieval_strategy']:<8} "
            f"{r['avg_precision_at_k']:>12.2f} {r['avg_hit_at_k']:>8.2f} {r['avg_mrr']:>6.2f} "
            f"{r['avg_faithfulness']:>13.2f} {r['avg_relevance']:>10.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="only run the first N questions (smoke test)")
    args = parser.parse_args()

    questions = json.loads(QUESTIONS_PATH.read_text())
    if args.limit:
        questions = questions[: args.limit]

    all_results = []
    for chunk_strategy, retrieval_strategy in CONFIGS:
        print(f"\n=== {chunk_strategy} chunking + {retrieval_strategy} retrieval ===")
        all_results.append(run_config(questions, chunk_strategy, retrieval_strategy))

    RESULTS_PATH.write_text(json.dumps(all_results, indent=2))
    print_table(all_results)


if __name__ == "__main__":
    main()
