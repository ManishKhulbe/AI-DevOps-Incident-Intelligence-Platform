"""
Evaluation CLI.

Run this script to measure the quality of the full agent pipeline against
the hand-written ground truth dataset.

Usage:
    python scripts/evaluate.py              # run all 5 scenarios
    python scripts/evaluate.py --question 1 # run only scenario 1 (1-indexed)
    python scripts/evaluate.py --dry-run    # print ground truth without running agents

Prerequisites:
    1. docker compose up -d            (Elasticsearch + Qdrant must be running)
    2. python scripts/seed_sample_logs.py  (logs must be ingested)
    3. OPENAI_API_KEY set in .env      (agents + Ragas both need it)

What the scores mean:
    faithfulness     >= 0.80  →  hallucination rate acceptable
    answer_relevancy >= 0.75  →  answers are on-topic
    context_precision >= 0.70 →  retrieval is not returning noise
    context_recall   >= 0.70  →  retrieval finds the needed evidence

Run this script before every significant change to:
    - Agent prompts
    - Retrieval parameters (TOP_K_RETRIEVE, TOP_K_RERANK)
    - Chunking strategy
    - Embedding model

If a score drops after your change, you know exactly what regressed and why.
"""

import sys
import argparse
from pathlib import Path

# Allow running from project root: python scripts/evaluate.py
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.ground_truth import GROUND_TRUTH
from src.evaluation.pipeline import run_all, run_single
from src.evaluation.metrics import compute, print_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the agent pipeline with Ragas")
    parser.add_argument(
        "--question", type=int, default=None,
        help="Run only this scenario (1-indexed). Omit to run all."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the ground truth dataset without running agents."
    )
    args = parser.parse_args()

    # ── Dry run — inspect ground truth ────────────────────────────────────
    if args.dry_run:
        print(f"\n{len(GROUND_TRUTH)} scenarios in ground truth dataset:\n")
        for i, entry in enumerate(GROUND_TRUTH, start=1):
            print(f"[{i}] {entry['question']}")
            print(f"     Ground truth: {entry['ground_truth'][:100]}...")
            print(f"     Expected contexts ({len(entry['expected_contexts'])}): "
                  f"{', '.join(entry['expected_contexts'][:3])}")
            print()
        return

    # ── Select scenarios ───────────────────────────────────────────────────
    if args.question is not None:
        idx = args.question - 1
        if idx < 0 or idx >= len(GROUND_TRUTH):
            print(f"Error: --question must be between 1 and {len(GROUND_TRUTH)}")
            sys.exit(1)
        dataset = [GROUND_TRUTH[idx]]
        print(f"\nRunning scenario {args.question} of {len(GROUND_TRUTH)}...")
    else:
        dataset = GROUND_TRUTH
        print(f"\nRunning all {len(dataset)} evaluation scenarios...")
        print("This will make LLM API calls for each scenario + Ragas scoring.")
        print("Estimated cost: ~$0.05-0.15 for the full dataset with gpt-4o-mini.\n")

    # ── Run pipeline ───────────────────────────────────────────────────────
    results = run_all(dataset)

    errors = [r for r in results if r["answer"].startswith("ERROR:")]
    if errors:
        print(f"\nWarning: {len(errors)} scenario(s) failed to run:")
        for r in errors:
            print(f"  - {r['question']}: {r['answer']}")

    successful = [r for r in results if not r["answer"].startswith("ERROR:")]
    if not successful:
        print("\nNo successful runs to score. Check that Docker services are running.")
        sys.exit(1)

    # ── Compute Ragas scores ───────────────────────────────────────────────
    print(f"\nScoring {len(successful)} result(s) with Ragas...")
    scores = compute(results)

    # ── Print report ───────────────────────────────────────────────────────
    print_report(scores, results)


if __name__ == "__main__":
    main()
