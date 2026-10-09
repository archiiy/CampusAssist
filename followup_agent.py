"""
Phase 7 — Follow-up Agent.

Same pattern as the trip concierge's follow-up agent: create_agent +
MemorySaver checkpointer, so a student can ask a follow-up question in
the same session ("how many days do I have left") without repeating
context.
"""

from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent

from llm import get_llm
from tools import ALL_TOOLS

SYSTEM_PROMPT = """
You are a campus helpdesk follow-up assistant. Use the kb_lookup tool to
answer factual questions about academic, hostel, examcell, placement, or
fee matters — always specify which desk to search. Use the
date_calculator tool for any question about deadlines or days remaining.
Resolve references like "that deadline" or "the due date" using the
conversation history before calling a tool.
"""


def build_followup_agent():
    llm = get_llm()
    checkpointer = MemorySaver()
    agent = create_agent(
        model=llm,
        tools=ALL_TOOLS,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )
    return agent


def ask(agent, session_id: str, question: str) -> str:
    config = {"configurable": {"thread_id": session_id}}
    result = agent.invoke({"messages": [HumanMessage(content=question)]}, config=config)
    return result["messages"][-1].content


if __name__ == "__main__":
    agent = build_followup_agent()
    session_id = "demo-session-1"

    q1 = "How many days are left until 30 September 2026?"
    print(f"Q1: {q1}\nA1: {ask(agent, session_id, q1)}\n")

    q2 = "What does the hostel desk say about mess timings?"
    print(f"Q2: {q2}\nA2: {ask(agent, session_id, q2)}\n")
