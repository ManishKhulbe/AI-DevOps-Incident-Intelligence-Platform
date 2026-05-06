from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.agents.state import AgentState
from src.config import settings
from src.observability.logger import get_logger

log = get_logger(__name__)


# ── Structured output schema ───────────────────────────────────────────────────

class Citation(BaseModel):
    claim:      str   = Field(description="The exact claim from the RCA being cited")
    evidence:   str   = Field(description="The exact log line that supports this claim")
    chunk_id:   str   = Field(description="The chunk_id of the source chunk")
    confidence: float = Field(description="Confidence score 0.0-1.0 for this citation")


class CitationOutput(BaseModel):
    citations:      list[Citation] = Field(description="One citation per supported claim")
    final_response: str            = Field(description="Complete answer with inline [N] citation markers")


# ── Prompt ─────────────────────────────────────────────────────────────────────

_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a technical writer producing a final incident report.

Your job:
1. Rewrite the RCA into a clean, professional incident report
2. Add inline citation markers [1], [2], ... to every factual claim
3. Build a citation list where each [N] maps to the exact log line that proves the claim
4. For claims the Critic flagged as low-confidence, add "(low confidence)" after the marker
5. End with a "Citations" section listing each [N] with the source log text

TONE: Direct, factual, technical.
FORMAT: One-line summary → Timeline → Root Cause → Impact → Citations

Critic confidence score: {confidence}
Low-confidence claims: {unsupported_claims}

RCA:
{reasoning_output}

Log chunks for citation matching:
{chunks}
""",
    ),
    ("human", "Produce the final cited incident report."),
])


def citation_node(state: AgentState) -> dict:
    """
    Citation Agent — final node before END.
    Attaches inline [N] markers to every claim and builds the citation list.
    """
    feedback       = state.get("critic_feedback", {})
    confidence     = feedback.get("confidence", 1.0)
    unsupported    = [i["claim"] for i in feedback.get("issues", [])]
    chunks         = state["retrieved_chunks"]

    log.info(
        "agent_start",
        agent="citation",
        confidence=confidence,
        unsupported_claims=len(unsupported),
        chunks_count=len(chunks),
    )

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=settings.openai_api_key,
    )

    chunks_text = "\n\n---\n\n".join(
        f"[chunk_id={c['chunk_id']}]\n{c['content']}"
        for c in chunks
    )

    chain = _PROMPT | llm.with_structured_output(CitationOutput)
    result: CitationOutput = chain.invoke({
        "confidence":         confidence,
        "unsupported_claims": "\n".join(f"- {c}" for c in unsupported) or "None",
        "reasoning_output":   state["reasoning_output"],
        "chunks":             chunks_text,
    })

    log.info(
        "agent_done",
        agent="citation",
        citations_count=len(result.citations),
        response_length=len(result.final_response),
    )

    return {
        "citations":      [c.model_dump() for c in result.citations],
        "final_response": result.final_response,
    }
