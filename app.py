"""
Streamlit UI for the Multi-Agent Campus Assistant.

Wraps the existing pipeline (router -> desk RAG -> reflection ->
governance) and the follow-up agent in a chat interface. No changes to
any of the agent code — this file only calls what already exists in
pipeline.py, followup_agent.py, and report.py, exactly the way main.py
does.

Run with:
    streamlit run app.py
"""

import os
import uuid
import json
import streamlit as st

st.set_page_config(page_title="Campus Assistant", page_icon=" ", layout="centered")

# --- Mock-mode toggle must be set BEFORE any agent modules are imported,
# since llm.py reads the env var at call time inside get_llm()/get_embeddings() ---
with st.sidebar:
    st.header("Settings")
    mock_mode = st.checkbox(
        "Use offline mock LLM (no Ollama needed)",
        value=False,
        help="For demos without a running Ollama server. Uses keyword-matching stand-ins, not real understanding.",
    )
    if mock_mode:
        os.environ["CAMPUS_ASSISTANT_MOCK"] = "1"
    else:
        os.environ.pop("CAMPUS_ASSISTANT_MOCK", None)

from pipeline import build_pipeline_graph
from followup_agent import build_followup_agent, ask
from report import build_report, save_report

DESK_LABELS = {
    "academic": "Academic",
    "hostel": " Hostel",
    "examcell": "Exam Cell",
    "placement": "Placement",
    "fee": "Fee",
}


@st.cache_resource
def get_pipeline(mock: bool):
    return build_pipeline_graph()


@st.cache_resource
def get_followup_agent(mock: bool):
    return build_followup_agent()


def reset_session():
    st.session_state.session_id = str(uuid.uuid4())[:8]
    st.session_state.messages = []
    st.session_state.primary_result = None
    st.session_state.qna = []


if "session_id" not in st.session_state:
    reset_session()

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
                if result["escalated"]:
                    badge = f"`{desk_label}` · :orange[⚠ Escalated — {result.get('escalation_contact', 'relevant department')}]"
                else:
                    conf = result.get("confidence", "high")
                    badge = f"`{desk_label}` · :green[✓ Grounded, {conf} confidence]"

                answer = result["final_answer"]
                sources = result.get("retrieved_chunks", [])

                st.markdown(answer)
                st.markdown(badge)
                if sources:
                    with st.expander("Sources used"):
                        for src in sources:
                            st.markdown(f"- **{src['source']}** (score: {src['score']:.2f})")

                st.session_state.messages.append({
                    "role": "assistant", "content": answer, "badges": badge, "sources": sources,
                })
            else:
                # Follow-up turn -> tool-using agent with memory
                answer = ask(followup_agent, st.session_state.session_id, query)
                st.markdown(answer)
                st.session_state.qna.append({"question": query, "answer": answer})
                st.session_state.messages.append({"role": "assistant", "content": answer})

if not st.session_state.messages:
    st.info(
        "Try asking something like:\n\n"
        "- *When is the last date to pay hostel fees?*\n"
        "- *How many days do I have to apply for revaluation?*\n"
        "- *What CGPA do I need for placements?*"
    )
    
    
