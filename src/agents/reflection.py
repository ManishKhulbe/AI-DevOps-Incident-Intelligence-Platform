from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.agents.state import AgentState
from src.config import settings

MAX_RETRIES = 3


# ── Structured output schema ───────────────────────────────────────────────────

class ReflectionOutput(BaseModel):
    should_retry: bool = Field(
        description="True if new retrieval queries should be attempted to find missing evidence"
    )
    new_queries: list[str] = Field(
        default_factory=list,
        description="Specific retrieval queries targeting the unsupported claims"
    )
    reasoning: str = Field(
        description="Brief explanation of the decision to retry or proceed"
    )


# ── Prompt ─────────────────────────────────────────────────────────────────────
_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a reflection agent in a DevOps incident analysis pipeline.

The Critic Agent found factual claims in the RCA that could not be verified
against the retrieved log chunks. Your job is to decide what to do next.

You have two options:
1. RETRY — generate new, targeted retrieval queries that might find the missing evidence
2. PROCEED — accept that the evidence is incomplete and let the Citation Agent
             flag low-confidence claims

Choose RETRY when:
- The unsupported claims are specific and searchable (contain timestamps, error codes, service names)
- The missing evidence likely exists in the logs but wasn't retrieved yet

Choose PROCEED when:
- The unsupported claims are vague interpretations that no log line would prove
- The RCA is still useful even without the missing claims
- Note: the graph enforces a hard limit of {max_retries} retries regardless of your choice

If retrying, generate 1-2 precise search queries per unsupported claim.
Make queries specific: include service names, error keywords, and timestamps from the claim.

Unsupported claims from the Critic:
{issues}

Original user question: {user_query}
"""
    ),
    ("human", "Decide whether to retry retrieval or proceed to citation."),
])


def reflection_node(state: AgentState) -> dict:
    """
    Reflection Agent — the retry loop controller.

    Reads the Critic's issues and decides whether to:
    A) Generate new targeted queries and send the graph back to the Retriever
    B) Accept the current evidence and proceed to Citation

    The routing decision (A vs B) is made by the conditional edge function
    in graph.py, which reads should_retry from the returned state. This node
    only updates state — it does not control routing directly.

    Why separate routing from the node?
    LangGraph's design principle: nodes transform state, edges route.
    Keeping routing logic in edge functions makes the graph readable —
    you can see the full flow in graph.py without reading each agent's code.
    """
    feedback = state["critic_feedback"]
    retry_count = state.get("retry_count", 0)

    # Already at max retries — force proceed regardless of critic result
    if retry_count >= MAX_RETRIES or feedback.get("valid", True):
        return {
            "retry_count": retry_count,
            "additional_queries": [],
        }

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=settings.openai_api_key,
    )

    issues_text = "\n".join(
        f"- Claim: \"{issue['claim']}\"\n  Reason: {issue['reason']}"
        for issue in feedback.get("issues", [])
    )

    chain = _PROMPT | llm.with_structured_output(ReflectionOutput)
    result: ReflectionOutput = chain.invoke({
        "issues": issues_text,
        "user_query": state["user_query"],
        "max_retries": MAX_RETRIES,
    })

    if result.should_retry:
        return {
            "retry_count": retry_count + 1,
            "additional_queries": result.new_queries,
        }

    # Reflection decided not to retry even though critic flagged issues
    return {
        "retry_count": retry_count,
        "additional_queries": [],
    }


def should_retry(state: AgentState) -> str:
    """
    Conditional edge function — called by LangGraph to decide the next node.

    Returns the NAME of the next node as a string.
    LangGraph uses this string to look up the node in its registry.

    Routing logic:
    - Critic valid AND no retry needed  → "citation"
    - Issues found AND retries remain   → "retriever"  (retry loop)
    - Issues found AND retries exhausted → "citation"  (graceful degradation)
    """
    feedback = state.get("critic_feedback", {})
    retry_count = state.get("retry_count", 0)
    additional_queries = state.get("additional_queries", [])

    if feedback.get("valid", True):
        return "citation"

    if additional_queries and retry_count < MAX_RETRIES:
        return "retriever"

    return "citation"
