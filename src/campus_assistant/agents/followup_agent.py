"""
Follow-up Agent.

create_agent + a checkpointer (SQLite by default, see memory/checkpointer.py)
so a student can ask a follow-up question in the same session ("how many
days do I have left") without repeating context.

`governed_ask()` is the governed entry point: unlike the earlier `ask()`
(kept only for the demo block below), it runs any kb_lookup-backed answer
through the same reflection_agent + governance_agent checks the first
pipeline turn gets, so a follow-up turn can't produce a confident-looking
but ungrounded answer with no escalation path. date_calculator-only answers
skip the groundedness check since they're deterministic arithmetic, not a
factual claim that needs verifying against retrieved context.
"""

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, ToolMessage

from campus_assistant.agents.governance_agent import governance_agent
from campus_assistant.agents.reflection_agent import reflection_agent
from campus_assistant.llm.factory import get_llm
from campus_assistant.memory.checkpointer import get_checkpointer
from campus_assistant.tools import ALL_TOOLS

SYSTEM_PROMPT = """
You are a campus helpdesk follow-up assistant. Use the kb_lookup tool to
answer factual questions about academic, hostel, examcell, placement, or
fee matters — always specify which desk to search. Use the
date_calculator tool for any question about deadlines or days remaining.
Resolve references like "that deadline" or "the due date" using the
conversation history before calling a tool.
"""


def build_followup_agent(mock: bool | None = None):
    llm = get_llm(mock=mock)
    checkpointer = get_checkpointer()
    agent = create_agent(
        model=llm,
        tools=ALL_TOOLS,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )
    return agent


def ask(agent, session_id: str, question: str) -> str:
    """Ungoverned follow-up — kept for the demo block below. Prefer
    governed_ask() in the UI/CLI: this bypasses reflection/governance
    entirely, so a kb_lookup-backed answer here gets no groundedness
    check before being returned."""
    config = {"configurable": {"thread_id": session_id}}
    result = agent.invoke({"messages": [HumanMessage(content=question)]}, config=config)
    return result["messages"][-1].content


def governed_ask(agent, session_id: str, question: str) -> dict:
    """Governed follow-up turn. Returns:
        {"answer", "draft_answer", "escalated", "escalation_contact", "confidence"}
    """
    config = {"configurable": {"thread_id": session_id}}

    # The checkpointer persists the full message history across turns, and
    # agent.invoke() returns that whole accumulated list — not just this
    # turn's new messages. Record how many messages existed before this
    # call so only genuinely new tool calls are inspected below; otherwise
    # a kb_lookup result from an earlier turn in the same session would be
    # mistaken for evidence backing a completely unrelated later answer
    # (e.g. a later date_calculator-only turn).
    prior_state = agent.get_state(config)
    prior_count = len(prior_state.values.get("messages", [])) if prior_state and prior_state.values else 0

    result = agent.invoke({"messages": [HumanMessage(content=question)]}, config=config)
    messages = result["messages"]
    new_messages = messages[prior_count:]
    answer = messages[-1].content

    kb_tool_messages = [
        m for m in new_messages
        if isinstance(m, ToolMessage) and getattr(m, "name", None) == "kb_lookup"
    ]

    if not kb_tool_messages:
        # No factual KB claim was made — either no tool was used, or only
        # date_calculator (deterministic arithmetic, not a groundedness
        # claim) — so there's nothing to reflect/govern against.
        return {
            "answer": answer, "draft_answer": answer,
            "escalated": False, "escalation_contact": None, "confidence": "high",
        }

    desk_hint = "academic"
    for m in new_messages:
        for call in getattr(m, "tool_calls", None) or []:
            if call.get("name") == "kb_lookup":
                desk_hint = call.get("args", {}).get("desk", desk_hint)

    context_text = "\n\n".join(m.content for m in kb_tool_messages)
    mini_state = {
        "draft_answer": answer,
        "retrieved_chunks": [{"text": context_text, "source": "followup_kb_lookup"}],
        "desk": desk_hint,
    }
    mini_state.update(reflection_agent(mini_state))
    mini_state.update(governance_agent(mini_state))

    return {
        "answer": mini_state["final_answer"],
        "draft_answer": answer,
        "escalated": mini_state["escalated"],
        "escalation_contact": mini_state.get("escalation_contact"),
        "confidence": mini_state.get("confidence"),
    }


if __name__ == "__main__":
    agent = build_followup_agent()
    session_id = "demo-session-1"

    q1 = "How many days are left until 30 September 2026?"
    print(f"Q1: {q1}\nA1: {governed_ask(agent, session_id, q1)}\n")

    q2 = "What does the hostel desk say about mess timings?"
    print(f"Q2: {q2}\nA2: {governed_ask(agent, session_id, q2)}\n")
