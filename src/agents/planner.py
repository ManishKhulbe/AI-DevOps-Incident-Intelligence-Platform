from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.agents.state import AgentState
from src.config import settings
from src.observability.logger import get_logger

log = get_logger(__name__)


# ── Structured output schema ───────────────────────────────────────────────────

class TimeRange(BaseModel):
    start: str | None = Field(None, description="ISO 8601 start time, e.g. 2024-01-15T14:00:00")
    end:   str | None = Field(None, description="ISO 8601 end time,   e.g. 2024-01-15T16:00:00")


class QueryPlan(BaseModel):
    intent: str = Field(
        description="One of: root_cause_analysis, log_search, deployment_correlation, summary"
    )
    services: list[str] = Field(
        default_factory=list,
        description="Service names to focus on, e.g. ['payment-service', 'checkout-service']"
    )
    time_range: TimeRange = Field(
        default_factory=TimeRange,
        description="Time window extracted from the question"
    )
    severity_filter: list[str] = Field(
        default_factory=list,
        description="Severity levels to filter, e.g. ['ERROR', 'CRITICAL']"
    )
    sub_questions: list[str] = Field(
        description="2-4 specific retrieval queries that together answer the user question"
    )


# ── Prompt ─────────────────────────────────────────────────────────────────────

_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a DevOps incident analysis planner.

Your job is to decompose a natural language incident question into a structured
query plan that will guide the downstream retrieval and reasoning agents.

Rules:
- Extract service names exactly as they might appear in logs (use hyphens, not spaces)
- If no time range is mentioned, leave time_range fields as null
- severity_filter should be ["ERROR", "CRITICAL"] for incident questions unless specified
- sub_questions must be specific and retrieval-friendly — each will become a search query
- Generate 2-4 sub_questions that together cover all aspects of the user question
- If the question mentions "why", always include a sub_question about the root cause
- If the question mentions a time, always include it in the sub_questions
""",
    ),
    ("human", "{user_query}"),
])


# ── Node function ──────────────────────────────────────────────────────────────

def planner_node(state: AgentState) -> dict:
    """
    Planner Agent — first node in the graph.

    Converts a free-form user question into a structured QueryPlan.
    with_structured_output() uses OpenAI function calling to guarantee valid JSON.
    """
    log.info("agent_start", agent="planner", question=state["user_query"])

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=settings.openai_api_key,
    )

    chain = _PROMPT | llm.with_structured_output(QueryPlan)
    plan: QueryPlan = chain.invoke({"user_query": state["user_query"]})

    log.info(
        "agent_done",
        agent="planner",
        intent=plan.intent,
        services=plan.services,
        sub_questions_count=len(plan.sub_questions),
    )

    return {
        "query_plan": plan.model_dump(),
        "retry_count": 0,
        "additional_queries": [],
    }
