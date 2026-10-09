"""
Streamlit UI for the Multi-Agent Campus Assistant.

Wraps the pipeline (router -> desk RAG -> reflection -> governance) and
the follow-up agent in a chat interface. Every turn — the first message
and all follow-ups — goes through reflection/governance before being
shown, via governed_ask() for follow-ups.

Run with:
    streamlit run app.py
"""

import json
import uuid

import streamlit as st

st.set_page_config(page_title="Campus Assistant", page_icon="🎓", layout="centered")

from campus_assistant.pipeline import build_pipeline_graph
from campus_assistant.agents.followup_agent import build_followup_agent, governed_ask
from campus_assistant.report import build_report, save_report

DESK_LABELS = {
    "academic": "Academic",
    "hostel": "Hostel",
    "examcell": "Exam Cell",
    "placement": "Placement",
    "fee": "Fee",
}


@st.cache_resource
def get_pipeline(mock: bool):
    return build_pipeline_graph(mock=mock)


@st.cache_resource
def get_followup_agent(mock: bool):
    return build_followup_agent(mock=mock)


def reset_session():
    st.session_state.session_id = str(uuid.uuid4())[:8]
    st.session_state.messages = []
    st.session_state.primary_result = None
    st.session_state.qna = []


def escalation_badge(desk_label: str | None, escalated: bool, contact: str | None, confidence: str) -> str:
    prefix = f"`{desk_label}` · " if desk_label else ""
    if escalated:
        return f"{prefix}:orange[⚠ Escalated — {contact or 'relevant department'}]"
    return f"{prefix}:green[✓ Grounded, {confidence} confidence]"


if "session_id" not in st.session_state:
    reset_session()

with st.sidebar:
    st.header("Settings")
    mock_mode = st.checkbox(
        "Use offline mock LLM (no Ollama needed)",
        value=False,
        help="For demos without a running Ollama server. Uses keyword-matching stand-ins, not real understanding.",
    )

pipeline_app = get_pipeline(mock_mode)
followup_agent = get_followup_agent(mock_mode)

with st.sidebar:
    st.caption(f"Session ID: `{st.session_state.session_id}`")
    if st.button("Start new session"):
        reset_session()
        st.rerun()

    if st.session_state.primary_result:
        report = build_report(st.session_state.primary_result, st.session_state.qna)
        st.download_button(
            "Download session report (JSON)",
            data=json.dumps(report, indent=2, ensure_ascii=False),
            file_name=f"session_report_{st.session_state.session_id}.json",
            mime="application/json",
        )

st.title("CampusAssist")
st.caption("Ask about academics, hostel, exams, placements, or fees.")

# --- render conversation so far ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("badges"):
            st.markdown(msg["badges"])
        if msg.get("sources"):
            with st.expander("Sources used"):
                for src in msg["sources"]:
                    st.markdown(f"- **{src['source']}** (score: {src['score']:.2f})")
        if msg.get("draft_answer"):
            with st.expander("Show the system's draft answer (unverified)"):
                st.markdown(msg["draft_answer"])

# --- handle new input ---
query = st.chat_input("Ask your question...")
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if st.session_state.primary_result is None:
                # First message in the session -> full pipeline: route, retrieve, reflect, govern
                result = pipeline_app.invoke({"raw_query": query, "session_id": st.session_state.session_id})
                st.session_state.primary_result = result

                desk_label = DESK_LABELS.get(result["desk"], result["desk"])
                badge = escalation_badge(
                    desk_label, result["escalated"], result.get("escalation_contact"), result.get("confidence", "high"),
                )

                answer = result["final_answer"]
                sources = result.get("retrieved_chunks", [])
                draft_answer = result.get("draft_answer") if result["escalated"] else None

                st.markdown(answer)
                st.markdown(badge)
                if sources:
                    with st.expander("Sources used"):
                        for src in sources:
                            st.markdown(f"- **{src['source']}** (score: {src['score']:.2f})")
                if draft_answer:
                    with st.expander("Show the system's draft answer (unverified)"):
                        st.markdown(draft_answer)

                st.session_state.messages.append({
                    "role": "assistant", "content": answer, "badges": badge,
                    "sources": sources, "draft_answer": draft_answer,
                })
            else:
                # Follow-up turn -> tool-using agent with memory, still governed
                followup = governed_ask(followup_agent, st.session_state.session_id, query)
                answer = followup["answer"]
                badge = escalation_badge(
                    None, followup["escalated"], followup.get("escalation_contact"), followup.get("confidence", "high"),
                )
                draft_answer = followup.get("draft_answer") if followup["escalated"] else None

                st.markdown(answer)
                st.markdown(badge)
                if draft_answer and draft_answer != answer:
                    with st.expander("Show the system's draft answer (unverified)"):
                        st.markdown(draft_answer)

                st.session_state.qna.append({"question": query, "answer": answer})
                st.session_state.messages.append({
                    "role": "assistant", "content": answer, "badges": badge, "draft_answer": draft_answer,
                })

if not st.session_state.messages:
    st.info(
        "Try asking something like:\n\n"
        "- *When is the last date to pay hostel fees?*\n"
        "- *How many days do I have to apply for revaluation?*\n"
        "- *What CGPA do I need for placements?*"
    )
