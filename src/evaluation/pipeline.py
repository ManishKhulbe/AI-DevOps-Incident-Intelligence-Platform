"""
Evaluation pipeline.

For each ground truth entry, this module:
1. Runs the full agent graph (same as a real user query)
2. Collects the four inputs Ragas needs to score the response:
       question         — the original question
       answer           — what our system actually said
       contexts         — the raw text of the retrieved chunks
       ground_truth     — the correct answer we wrote by hand

Why collect contexts from the agent state?
Ragas scores retrieval quality (context_precision, context_recall) separately
from answer quality (faithfulness, answer_relevancy). To compute these, Ragas
needs to see the exact text that was fed to the LLM — not just the final answer.
We pull retrieved_chunks directly from the agent state for this reason.
"""

from src.agents.graph import agent_graph
from src.agents.state import AgentState
from src.observability.logger import get_logger

log = get_logger(__name__)


def _build_initial_state(question: str) -> AgentState:
    return AgentState(
        user_query=question,
        query_plan={},
        retrieved_chunks=[],
        reasoning_output="",
        critic_feedback={},
        retry_count=0,
        additional_queries=[],
        citations=[],
        final_response="",
    )


def run_single(entry: dict) -> dict:
    """
    Run the agent pipeline for one ground truth entry.

    Returns a dict in the shape Ragas expects:
        {
            "question":     str,
            "answer":       str,
            "contexts":     list[str],   # text of each retrieved chunk
            "ground_truth": str,
        }
    """
    question     = entry["question"]
    ground_truth = entry["ground_truth"]

    log.info("eval_run_start", question=question)

    try:
        state: AgentState = agent_graph.invoke(_build_initial_state(question))
    except Exception as exc:
        log.error("eval_run_failed", question=question, error=str(exc))
        return {
            "question":     question,
            "answer":       f"ERROR: {exc}",
            "contexts":     [],
            "ground_truth": ground_truth,
        }

    answer   = state.get("final_response", "")
    contexts = [c["content"] for c in state.get("retrieved_chunks", [])]

    log.info(
        "eval_run_done",
        question=question,
        answer_length=len(answer),
        contexts_count=len(contexts),
        retry_count=state.get("retry_count", 0),
        critic_valid=state.get("critic_feedback", {}).get("valid"),
    )

    return {
        "question":     question,
        "answer":       answer,
        "contexts":     contexts,
        "ground_truth": ground_truth,
    }


def run_all(ground_truth_dataset: list[dict]) -> list[dict]:
    """
    Run the pipeline for every entry in the ground truth dataset.
    Returns a list of Ragas-ready result dicts.
    """
    results = []
    total = len(ground_truth_dataset)

    for i, entry in enumerate(ground_truth_dataset, start=1):
        print(f"\n[{i}/{total}] Running: {entry['question'][:60]}...")
        result = run_single(entry)
        results.append(result)

    return results
