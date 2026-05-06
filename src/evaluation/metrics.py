"""
Ragas metric computation.

Ragas measures four things — understand each before looking at the scores:

┌─────────────────────┬────────────────────────────────────────────────────────┐
│ Metric              │ Question it answers                                    │
├─────────────────────┼────────────────────────────────────────────────────────┤
│ faithfulness        │ Is every claim in the answer supported by the          │
│                     │ retrieved contexts? (anti-hallucination score)         │
│                     │ 1.0 = no hallucinations, 0.0 = entirely hallucinated  │
├─────────────────────┼────────────────────────────────────────────────────────┤
│ answer_relevancy    │ Does the answer actually address the question asked?   │
│                     │ 1.0 = perfectly on-topic, 0.0 = completely off-topic  │
├─────────────────────┼────────────────────────────────────────────────────────┤
│ context_precision   │ Are the retrieved chunks relevant to the question?     │
│                     │ 1.0 = all chunks are useful, 0.0 = all noise          │
├─────────────────────┼────────────────────────────────────────────────────────┤
│ context_recall      │ Did retrieval find all the chunks needed to answer?    │
│                     │ 1.0 = nothing missed, 0.0 = all evidence missing       │
└─────────────────────┴────────────────────────────────────────────────────────┘

faithfulness     — needs: answer + contexts
answer_relevancy — needs: question + answer
context_precision  — needs: question + contexts + ground_truth
context_recall     — needs: contexts + ground_truth
"""

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from src.config import settings
from src.observability.logger import get_logger

log = get_logger(__name__)

# Thresholds — below these scores the system is not production-ready.
# These are common starting points; tune them for your use case.
THRESHOLDS = {
    "faithfulness":      0.80,   # < 80% means unacceptable hallucination rate
    "answer_relevancy":  0.75,   # < 75% means answers are often off-topic
    "context_precision": 0.70,   # < 70% means retrieval returns too much noise
    "context_recall":    0.70,   # < 70% means retrieval misses key evidence
}


def compute(results: list[dict]) -> dict:
    """
    Convert pipeline results into a Ragas Dataset and compute all four metrics.

    Why Ragas needs an LLM to score?
    Metrics like faithfulness work by asking an LLM to verify each claim in
    the answer against the contexts. It is LLM-as-a-judge — faster and cheaper
    than human evaluation, but still requires an LLM API call per sample.

    Returns a dict of metric_name → score (float, 0.0-1.0).
    """
    log.info("metrics_start", samples=len(results))

    # Ragas requires the dataset in HuggingFace Dataset format
    dataset = Dataset.from_list([
        {
            "question":     r["question"],
            "answer":       r["answer"],
            "contexts":     r["contexts"],
            "ground_truth": r["ground_truth"],
        }
        for r in results
        # Skip error entries — they would skew scores unfairly
        if not r["answer"].startswith("ERROR:")
    ])

    # Ragas uses its own LLM internally for judging — wire it to our key
    llm        = ChatOpenAI(model="gpt-4o-mini", api_key=settings.openai_api_key)
    embeddings = OpenAIEmbeddings(api_key=settings.openai_api_key)

    scores = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=embeddings,
    )

    result = {
        "faithfulness":      float(scores["faithfulness"]),
        "answer_relevancy":  float(scores["answer_relevancy"]),
        "context_precision": float(scores["context_precision"]),
        "context_recall":    float(scores["context_recall"]),
    }

    log.info("metrics_done", **result)
    return result


def print_report(scores: dict, results: list[dict]) -> None:
    """
    Print a human-readable evaluation report to stdout.
    Shows per-metric scores, pass/fail against thresholds, and
    per-question breakdown.
    """
    passed = all(scores[m] >= THRESHOLDS[m] for m in THRESHOLDS)

    print("\n" + "=" * 60)
    print("  EVALUATION REPORT")
    print("=" * 60)

    print(f"\n{'Metric':<22} {'Score':>7}  {'Threshold':>10}  {'Status':>6}")
    print("-" * 52)

    for metric, threshold in THRESHOLDS.items():
        score  = scores.get(metric, 0.0)
        status = "PASS" if score >= threshold else "FAIL"
        print(f"{metric:<22} {score:>7.2%}  {threshold:>10.0%}  {status:>6}")

    print("-" * 52)
    overall = "ALL PASS" if passed else "NEEDS WORK"
    print(f"\nOverall: {overall}")

    # Per-question breakdown
    print("\n" + "=" * 60)
    print("  PER-QUESTION RESULTS")
    print("=" * 60)

    for i, r in enumerate(results, start=1):
        status = "ERROR" if r["answer"].startswith("ERROR:") else "OK"
        chunks = len(r.get("contexts", []))
        q      = r["question"][:55] + "..." if len(r["question"]) > 55 else r["question"]
        print(f"\n[{i}] {q}")
        print(f"     chunks_retrieved={chunks}  status={status}")

    print("\n" + "=" * 60)

    # Improvement hints
    if not passed:
        print("\nIMPROVEMENT HINTS:")
        if scores.get("faithfulness", 1.0) < THRESHOLDS["faithfulness"]:
            print("  faithfulness LOW  → tighten the Reasoning Agent's system prompt")
            print("                      add: 'Do not state anything not in the chunks'")
        if scores.get("context_recall", 1.0) < THRESHOLDS["context_recall"]:
            print("  context_recall LOW → increase TOP_K_RETRIEVE in .env (try 30)")
            print("                       or add more sub_questions in the Planner")
        if scores.get("context_precision", 1.0) < THRESHOLDS["context_precision"]:
            print("  context_precision LOW → decrease TOP_K_RERANK in .env (try 3)")
            print("                          the reranker may be letting noise through")
        if scores.get("answer_relevancy", 1.0) < THRESHOLDS["answer_relevancy"]:
            print("  answer_relevancy LOW → review Citation Agent prompt")
            print("                         answers may be drifting off-topic")
        print()
