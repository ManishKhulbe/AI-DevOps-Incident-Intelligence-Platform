from langgraph.graph import StateGraph, END

from src.agents.state import AgentState
from src.agents.planner    import planner_node
from src.agents.retriever  import retriever_node
from src.agents.reasoning  import reasoning_node
from src.agents.critic     import critic_node
from src.agents.reflection import reflection_node, should_retry
from src.agents.citation   import citation_node


def build_graph():
    """
    Assemble the LangGraph state machine and return a compiled runnable.

    ┌─────────────────────────────────────────────────────────────────┐
    │                        GRAPH TOPOLOGY                          │
    │                                                                 │
    │  START ──► planner ──► retriever ──► reasoning ──► critic      │
    │                            ▲                          │        │
    │                            │                          ▼        │
    │                        reflection ◄──────────────────┘        │
    │                            │                                    │
    │               ┌────────────┴────────────┐                      │
    │               │ should_retry()           │                      │
    │               ▼                          ▼                      │
    │           retriever                  citation ──► END          │
    │    (retry with new queries)                                     │
    └─────────────────────────────────────────────────────────────────┘

    Key LangGraph concepts used here:

    add_node(name, fn)
        Register a node. fn must be: (AgentState) -> dict.
        The dict is MERGED into state (not a full replacement).

    add_edge(a, b)
        Unconditional edge: after node a always go to node b.

    add_conditional_edges(node, fn, mapping)
        After node completes, call fn(state) which returns a string.
        Look that string up in mapping to find the next node name.
        This is how the retry loop works: should_retry() returns
        "retriever" or "citation" depending on state.

    set_entry_point(name)
        Which node receives the initial state when graph.invoke() is called.

    compile()
        Validates the graph (no orphan nodes, no missing edges),
        then returns a Runnable that can be called like a function.
    """
    graph = StateGraph(AgentState)

    # ── Register nodes ────────────────────────────────────────────────────────
    graph.add_node("planner",    planner_node)
    graph.add_node("retriever",  retriever_node)
    graph.add_node("reasoning",  reasoning_node)
    graph.add_node("critic",     critic_node)
    graph.add_node("reflection", reflection_node)
    graph.add_node("citation",   citation_node)

    # ── Entry point ───────────────────────────────────────────────────────────
    graph.set_entry_point("planner")

    # ── Unconditional edges (linear flow) ─────────────────────────────────────
    graph.add_edge("planner",   "retriever")
    graph.add_edge("retriever", "reasoning")
    graph.add_edge("reasoning", "critic")
    graph.add_edge("critic",    "reflection")

    # ── Conditional edge (the retry loop) ────────────────────────────────────
    # After reflection_node runs, call should_retry(state).
    # It returns "retriever" or "citation" as a string.
    # The mapping below translates that string to the actual node.
    graph.add_conditional_edges(
        "reflection",
        should_retry,
        {
            "retriever": "retriever",   # loop back — retry with new queries
            "citation":  "citation",    # proceed — evidence is sufficient (or retries exhausted)
        },
    )

    # ── Terminal edge ─────────────────────────────────────────────────────────
    graph.add_edge("citation", END)

    return graph.compile()


# Module-level compiled graph — import this in the FastAPI route handler.
# Compiling once at import time is important: compile() validates the graph
# and pre-computes the execution plan. Re-compiling per request would add
# ~50ms overhead on every query.
agent_graph = build_graph()
