from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from src.agents.state import AgentState
from src.config import settings


_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a senior DevOps incident analyst.

Your task is to analyze the log chunks below and produce a structured root cause analysis (RCA).

STRICT RULES — violating these will cause downstream validation to fail:
1. Base every claim ONLY on the log chunks provided. Do not invent facts.
2. If a log chunk contains a timestamp, use it exactly — do not estimate times.
3. If the logs do not contain enough information to determine a root cause, say so explicitly.
4. Do not mention "the logs show" or "according to the logs" — write the RCA as facts.

OUTPUT FORMAT — follow this structure exactly:

## Timeline
List events in chronological order. One bullet per event.
Format: `HH:MM:SS — [service] event description`

## Root Cause
One paragraph. State the root cause directly. Reference the specific error or condition that triggered the incident.

## Impact
What failed, for how long, and which services were affected.

## Evidence Summary
List the 3-5 most important log lines that prove the root cause.
Format: `- "exact log line text"`

Log chunks to analyze:
{chunks}

User question: {user_query}
"""
    ),
    ("human", "Produce the RCA based on the log chunks above."),
])


def _format_chunks(chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a numbered list for the prompt.

    Why numbered? The Critic Agent references chunk numbers when flagging
    unsupported claims, making it easier to cross-reference evidence.
    """
    if not chunks:
        return "No log chunks retrieved."

    parts = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(
            f"[Chunk {i}] service={chunk.get('service_name', '?')} "
            f"timestamp={chunk.get('timestamp', '?')} "
            f"severity={chunk.get('severity', '?')}\n"
            f"{chunk['content']}"
        )
    return "\n\n---\n\n".join(parts)


def reasoning_node(state: AgentState) -> dict:
    """
    Reasoning Agent — builds the timeline and draft RCA.

    This is the most prompt-sensitive node. The "ONLY on the log chunks provided"
    constraint is the primary defence against hallucination. The Critic Agent
    will check every factual claim against the chunks — any claim that cannot
    be traced back here will trigger a retry.

    temperature=0 is critical here. Higher temperatures make the model more
    creative, which means it will invent plausible-sounding causes not in the
    logs. For evidence-grounded analysis, creativity is the enemy.
    """
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=settings.openai_api_key,
    )

    chunks_text = _format_chunks(state["retrieved_chunks"])

    chain = _PROMPT | llm
    response = chain.invoke({
        "chunks": chunks_text,
        "user_query": state["user_query"],
    })

    return {"reasoning_output": response.content}
