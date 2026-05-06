from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.agents.state import AgentState
from src.config import settings
from src.observability.logger import get_logger

log = get_logger(__name__)


# ── Structured output schema ───────────────────────────────────────────────────

class CriticIssue(BaseModel):
    claim:  str = Field(description="The exact claim from the RCA that is unsupported")
    reason: str = Field(description="Why this claim cannot be verified from the log chunks")


class CriticOutput(BaseModel):
    valid:      bool         = Field(description="True if every factual claim is supported by at least one chunk")
    issues:     list[CriticIssue] = Field(default_factory=list)
    confidence: float        = Field(description="Overall confidence 0.0-1.0")


# ── Prompt ─────────────────────────────────────────────────────────────────────

_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a strict fact-checker for DevOps incident reports.

Your job is to verify that every factual claim in the RCA below can be directly
traced to at least one of the provided log chunks.

WHAT COUNTS AS A FACTUAL CLAIM:
- Specific timestamps or times ("failed at 14:32")
- Specific error messages or codes ("NullPointerException in CartSerializer")
- Specific numeric values ("error rate reached 78%")
- Causal statements ("the deployment caused the failure")
- Status changes ("circuit breaker opened")

WHAT IS NOT A CLAIM (do not flag these):
- Interpretive summaries ("the service became unstable")
- Structural words ("Timeline:", "Root Cause:")
- Recommendations or next steps

INSTRUCTIONS:
1. Read each factual claim in the RCA
2. Search the log chunks for direct evidence
3. If a claim is supported by at least one chunk, it passes
4. If a claim cannot be found in any chunk, add it to issues
5. Set valid=True only if issues list is EMPTY
6. Set confidence: 0.9+ = strong evidence, 0.5-0.9 = partial, <0.5 = weak

Log chunks (ground truth):
{chunks}

RCA to validate:
{reasoning_output}
""",
    ),
    ("human", "Validate the RCA against the log chunks."),
])


def critic_node(state: AgentState) -> dict:
    """
    Critic Agent — the hallucination firewall.
    Checks every factual claim in the RCA against source chunks.
    """
    log.info(
        "agent_start",
        agent="critic",
        rca_length=len(state["reasoning_output"]),
        chunks_count=len(state["retrieved_chunks"]),
    )

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=settings.openai_api_key,
    )

    chunks_text = "\n\n---\n\n".join(
        f"[Chunk {i}]\n{c['content']}"
        for i, c in enumerate(state["retrieved_chunks"], start=1)
    )

    chain = _PROMPT | llm.with_structured_output(CriticOutput)
    result: CriticOutput = chain.invoke({
        "chunks":           chunks_text,
        "reasoning_output": state["reasoning_output"],
    })

    log.info(
        "agent_done",
        agent="critic",
        valid=result.valid,
        issues_count=len(result.issues),
        confidence=result.confidence,
    )

    if not result.valid:
        for issue in result.issues:
            log.warning(
                "hallucination_detected",
                agent="critic",
                claim=issue.claim,
                reason=issue.reason,
            )

    return {"critic_feedback": result.model_dump()}
