from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.agents.state import AgentState
from src.config import settings
from src.observability.logger import get_logger

log = get_logger(__name__)

MAX_RETRIES = 3


# ── Structured output schema ───────────────────────────────────────────────────

class ReflectionOutput(BaseModel):
    should_retry: bool       = Field(description="True if new retrieval queries should be attempted")
    new_queries:  list[str]  = Field(default_factory=list, description="Targeted queries for missing evidence")
    reasoning:    str        = Field(description="Brief explanation of the decision")


# ── Prompt ─────────────────────────────────────────────────────────────────────

_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a reflection agent in a DevOps incident analysis pipeline.

The Critic Agent found factual claims in the RCA that could not be verified.
Your job is to decide what to do next.

Choose RETRY when:
- The unsupported claims are specific and searchable (contain timestamps, error codes, service names)
- The missing evidence likely exists in the logs but wasn't retrieved yet

Choose PROCEED when:
- The unsupported claims are vague interpretations that no log line would prove
- The RCA is still useful even without the missing claims
- Note: the graph enforces a hard limit of {max_retries} retries

If retrying, generate 1-2 precise search queries per unsupported claim.
Include service names, error keywords, and timestamps from the claim.

Unsupported claims:
{issues}

Original question: {user_query}
""",
    ),
    ("human", "Decide whether to retry retrieval or proceed to citation."),
])


def reflection_node(state: AgentState) -> dict:
    """
    Reflection Agent — reads Critic issues, decides retry or proceed.
    Returns updated retry_count and additional_queries for the Retriever.
    The actual routing decision lives in should_retry() below.
    """
    feedback    = state["critic_feedback"]
    retry_count = state.get("retry_count", 0)

    log.info(
        "agent_start",
        agent="reflection",
        critic_valid=feedback.get("valid"),
        retry_count=retry_count,
        issues_count=len(feedback.get("issues", [])),
    )

    # Short-circuit: no issues or retries exhausted
    if retry_count >= MAX_RETRIES or feedback.get("valid", True):
        log.info(
            "agent_done",
            agent="reflection",
            decision="proceed",
            reason="valid or retries_exhausted",
        )
        return {"retry_count": retry_count, "additional_queries": []}

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=settings.openai_api_key,
    )

    issues_text = "\n".join(
        f"- Claim: \"{i['claim']}\"\n  Reason: {i['reason']}"
        for i in feedback.get("issues", [])
    )

    chain = _PROMPT | llm.with_structured_output(ReflectionOutput)
    result: ReflectionOutput = chain.invoke({
        "issues":      issues_text,
        "user_query":  state["user_query"],
        "max_retries": MAX_RETRIES,
    })

    if result.should_retry:
        log.info(
            "agent_done",
            agent="reflection",
            decision="retry",
            new_queries=result.new_queries,
            retry_count=retry_count + 1,
        )
        return {
            "retry_count":       retry_count + 1,
            "additional_queries": result.new_queries,
        }

    log.info("agent_done", agent="reflection", decision="proceed", reason=result.reasoning)
    return {"retry_count": retry_count, "additional_queries": []}


def should_retry(state: AgentState) -> str:
    """
    Conditional edge function — LangGraph calls this after reflection_node
    to determine the next node name.

    Returns "retriever" to loop back, or "citation" to proceed.
    """
    feedback          = state.get("critic_feedback", {})
    retry_count       = state.get("retry_count", 0)
    additional_queries = state.get("additional_queries", [])

    if feedback.get("valid", True):
        return "citation"

    if additional_queries and retry_count < MAX_RETRIES:
        return "retriever"

    return "citation"
