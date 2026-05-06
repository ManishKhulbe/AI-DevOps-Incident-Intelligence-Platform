from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.agents.state import AgentState
from src.config import settings


# ── Structured output schema ───────────────────────────────────────────────────

class Citation(BaseModel):
    claim:      str   = Field(description="The exact claim from the RCA being cited")
    evidence:   str   = Field(description="The exact log line that supports this claim")
    chunk_id:   str   = Field(description="The chunk_id of the source chunk")
    confidence: float = Field(description="Confidence score 0.0-1.0 for this citation")


class CitationOutput(BaseModel):
    citations:      list[Citation] = Field(description="One citation per supported claim")
    final_response: str            = Field(
        description="The complete human-readable answer with inline [N] citation markers"
    )


# ── Prompt ─────────────────────────────────────────────────────────────────────
_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a technical writer producing a final incident report.

Your inputs are:
1. An RCA with a timeline, root cause, impact, and evidence summary
2. The retrieved log chunks that were used as evidence
3. The Critic's confidence score and any flagged low-confidence areas

Your job:
1. Rewrite the RCA into a clean, professional incident report
2. Add inline citation markers [1], [2], ... to every factual claim
3. Build a citation list where each [N] maps to the exact log line that proves the claim
4. For claims the Critic flagged as low-confidence, add "(low confidence)" after the marker
5. End with a "Citations" section listing each [N] with the source log text

TONE: Direct, factual, technical. No hedging except where confidence is low.
FORMAT:
- Start with a one-line summary of the incident
- Then Timeline, Root Cause, Impact sections
- End with numbered Citations list

Critic confidence score: {confidence}
Unsupported claims (mark as low confidence): {unsupported_claims}

RCA:
{reasoning_output}

Log chunks for citation matching:
{chunks}
"""
    ),
    ("human", "Produce the final cited incident report."),
])


def citation_node(state: AgentState) -> dict:
    """
    Citation Agent — the final node before END.

    Takes the validated RCA and attaches inline citation markers to every
    factual claim, then builds a citation list mapping each marker to the
    exact log line that supports it.

    Why citations matter (from the BRD):
    "Create explainable AI responses with citations" is a core objective.
    An RCA without citations is an assertion. An RCA with citations is
    verifiable — the engineer can click to the source log line and confirm
    the system's reasoning themselves. This is what separates a trustworthy
    AI tool from a hallucination machine.

    The confidence score from the Critic is passed through so low-confidence
    claims are visibly flagged rather than silently included.
    """
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=settings.openai_api_key,
    )

    feedback = state.get("critic_feedback", {})
    confidence = feedback.get("confidence", 1.0)
    unsupported = [i["claim"] for i in feedback.get("issues", [])]

    chunks_text = "\n\n---\n\n".join(
        f"[chunk_id={c['chunk_id']}]\n{c['content']}"
        for c in state["retrieved_chunks"]
    )

    chain = _PROMPT | llm.with_structured_output(CitationOutput)
    result: CitationOutput = chain.invoke({
        "confidence":        confidence,
        "unsupported_claims": "\n".join(f"- {c}" for c in unsupported) or "None",
        "reasoning_output":   state["reasoning_output"],
        "chunks":             chunks_text,
    })

    return {
        "citations":      [c.model_dump() for c in result.citations],
        "final_response": result.final_response,
    }
