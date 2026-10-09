"""
Streamlit UI for the Multi-Agent Campus Assistant.

Wraps the pipeline (router -> desk RAG -> reflection -> governance) and
the follow-up agent in a branded chat interface. Every turn — the first
message and all follow-ups — goes through reflection/governance before
being shown, via governed_ask() for follow-ups.

Run with:
    streamlit run app.py
"""

import json
import uuid

import streamlit as st

st.set_page_config(page_title="CampusAssist", page_icon="🎓", layout="wide")

from campus_assistant.pipeline import build_pipeline_graph
from campus_assistant.agents.followup_agent import build_followup_agent, governed_ask
from campus_assistant.report import build_report, save_report

DESK_META = {
    "academic":  {"label": "Academic",  "icon": "📚", "color": "#2563EB"},
    "hostel":    {"label": "Hostel",    "icon": "🏠", "color": "#D97706"},
    "examcell":  {"label": "Exam Cell", "icon": "📝", "color": "#7C3AED"},
    "placement": {"label": "Placement", "icon": "💼", "color": "#059669"},
    "fee":       {"label": "Fee",       "icon": "💳", "color": "#DC2626"},
}

SUGGESTED_QUESTIONS = [
    "When is the last date to pay hostel fees?",
    "How many days do I have to apply for revaluation?",
    "What CGPA do I need for placements?",
    "What is the minimum attendance required for exams?",
]

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --ca-navy: #122A4D;
    --ca-navy-light: #1E3A5F;
    --ca-blue: #2563EB;
    --ca-gold: #D9A62E;
    --ca-bg: #F4F6FA;
    --ca-muted: #64748B;
    --ca-green-bg: #DCFCE7;
    --ca-green-fg: #15803D;
    --ca-amber-bg: #FEF3C7;
    --ca-amber-fg: #B45309;
}

html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', sans-serif;
    background: var(--ca-bg);
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stMainBlockContainer"] { padding-top: 1.5rem; max-width: 920px; }

/* ---- Sidebar ---- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--ca-navy) 0%, var(--ca-navy-light) 100%);
}
[data-testid="stSidebar"] * { color: #E7EDF6 !important; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.15); }
[data-testid="stSidebar"] [data-testid^="stBaseButton"] {
    background: rgba(255,255,255,0.1) !important;
    border: 1px solid rgba(255,255,255,0.25) !important;
    color: #fff !important;
    border-radius: 10px !important;
}
.ca-sidebar-brand {
    font-size: 1.5rem; font-weight: 800; display: flex; align-items: center; gap: .4rem;
}
.ca-sidebar-tagline { font-size: .85rem; opacity: .75; margin-bottom: .75rem; }
.ca-session-card {
    background: rgba(255,255,255,0.08); border-radius: 10px; padding: .6rem .8rem;
    font-size: .85rem; margin-bottom: .6rem;
}
.ca-session-card code { color: var(--ca-gold) !important; background: transparent !important; }
.ca-desk-row { margin-bottom: .45rem; }

/* ---- Buttons (main area) ---- */
.stButton button, .stDownloadButton button { border-radius: 10px; }

/* ---- Hero banner ---- */
.ca-hero {
    background: linear-gradient(135deg, var(--ca-navy) 0%, var(--ca-blue) 100%);
    border-radius: 18px; padding: 2rem 2.2rem; color: #fff; margin-bottom: 1.4rem;
    box-shadow: 0 8px 24px rgba(18,42,77,0.25);
}
.ca-hero-title { font-size: 2rem; font-weight: 800; margin-bottom: .35rem; }
.ca-hero-sub { font-size: 1rem; opacity: .92; max-width: 680px; line-height: 1.5; }

/* ---- Chat bubbles ---- */
[data-testid="stChatMessage"] { padding: .25rem 0; }
[data-testid="stChatMessageContent"] {
    border-radius: 14px !important; padding: .8rem 1.05rem !important; line-height: 1.55;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stChatMessageContent"] {
    background: #EEF3FC;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
    background: var(--ca-navy); color: #fff;
}
[data-testid="stChatMessageAvatarUser"] { background: var(--ca-gold) !important; }
[data-testid="stChatMessageAvatarAssistant"] { background: var(--ca-navy) !important; }

/* ---- Status pills ---- */
.ca-status-row { margin-top: .55rem; display: flex; gap: .5rem; flex-wrap: wrap; }
.ca-pill {
    display: inline-flex; align-items: center; gap: .3rem; padding: .25rem .7rem;
    border-radius: 999px; font-size: .78rem; font-weight: 600; white-space: nowrap;
}
.ca-pill-ok { background: var(--ca-green-bg); color: var(--ca-green-fg); }
.ca-pill-warn { background: var(--ca-amber-bg); color: var(--ca-amber-fg); }

/* ---- Welcome card ---- */
.ca-welcome-title { font-weight: 700; font-size: 1.05rem; margin-bottom: .6rem; }

/* ---- Footer ---- */
.ca-footer { text-align: center; color: var(--ca-muted); font-size: .8rem; margin-top: 1.6rem; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


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
    st.session_state.pending_query = None


def desk_pill(desk_key: str) -> str:
    meta = DESK_META.get(desk_key, {"label": desk_key.title(), "icon": "🏢", "color": "#334155"})
    return (
        f'<span class="ca-pill" style="background:{meta["color"]}1A;color:{meta["color"]};'
        f'border:1px solid {meta["color"]}40;">{meta["icon"]} {meta["label"]}</span>'
    )


def status_pill(escalated: bool, contact: str | None, confidence: str) -> str:
    if escalated:
        return f'<span class="ca-pill ca-pill-warn">⚠ Escalated → {contact or "relevant department"}</span>'
    return f'<span class="ca-pill ca-pill-ok">✓ Grounded · {confidence} confidence</span>'


def status_row_html(desk_key: str | None, escalated: bool, contact: str | None, confidence: str) -> str:
    parts = [desk_pill(desk_key)] if desk_key else []
    parts.append(status_pill(escalated, contact, confidence))
    return f'<div class="ca-status-row">{"".join(parts)}</div>'


if "session_id" not in st.session_state:
    reset_session()
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown('<div class="ca-sidebar-brand">🎓 CampusAssist</div>', unsafe_allow_html=True)
    st.markdown('<div class="ca-sidebar-tagline">Your AI-powered college helpdesk</div>', unsafe_allow_html=True)

    st.markdown("**Settings**")
    mock_mode = st.checkbox(
        "Offline demo mode (no Ollama)",
        value=False,
        help="Uses keyword-matching stand-ins instead of the real model — for demos without Ollama running.",
    )

pipeline_app = get_pipeline(mock_mode)
followup_agent = get_followup_agent(mock_mode)

with st.sidebar:
    st.markdown("---")
    st.markdown(
        f'<div class="ca-session-card">Session<br><code>{st.session_state.session_id}</code></div>',
        unsafe_allow_html=True,
    )
    if st.button("🔄 Start new session", use_container_width=True):
        reset_session()
        st.rerun()

    if st.session_state.primary_result:
        report = build_report(st.session_state.primary_result, st.session_state.qna)
        st.download_button(
            "⬇️ Download session report",
            data=json.dumps(report, indent=2, ensure_ascii=False),
            file_name=f"session_report_{st.session_state.session_id}.json",
            mime="application/json",
            use_container_width=True,
        )

    st.markdown("---")
    st.markdown("**Desks we cover**")
    for key in DESK_META:
        st.markdown(f'<div class="ca-desk-row">{desk_pill(key)}</div>', unsafe_allow_html=True)

# ---------------- Hero ----------------
st.markdown(
    """
    <div class="ca-hero">
      <div class="ca-hero-title">🎓 CampusAssist</div>
      <div class="ca-hero-sub">
        Ask about academics, hostel, exams, placements, or fees. Answers are grounded
        in official circulars — and escalated to the right office whenever we're not confident.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

chat_card = st.container(border=True)

with chat_card:
    if not st.session_state.messages:
        st.markdown('<div class="ca-welcome-title">👋 Try one of these:</div>', unsafe_allow_html=True)
        cols = st.columns(2)
        for i, q in enumerate(SUGGESTED_QUESTIONS):
            if cols[i % 2].button(q, use_container_width=True, key=f"suggest_{i}"):
                st.session_state.pending_query = q
                st.rerun()

    for msg in st.session_state.messages:
        avatar = "🧑‍🎓" if msg["role"] == "user" else "🏛️"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            if msg.get("status_html"):
                st.markdown(msg["status_html"], unsafe_allow_html=True)
            if msg.get("sources"):
                with st.expander("📎 Sources used"):
                    for src in msg["sources"]:
                        st.markdown(f"- **{src['source']}** (score: {src['score']:.2f})")
            if msg.get("draft_answer"):
                with st.expander("🔍 Show the system's draft answer (unverified)"):
                    st.markdown(msg["draft_answer"])

    query = st.chat_input("Ask your question...")
    if not query and st.session_state.pending_query:
        query = st.session_state.pending_query
    st.session_state.pending_query = None

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user", avatar="🧑‍🎓"):
            st.markdown(query)

        with st.chat_message("assistant", avatar="🏛️"):
            with st.spinner("Thinking..."):
                if st.session_state.primary_result is None:
                    # First message in the session -> full pipeline: route, retrieve, reflect, govern
                    result = pipeline_app.invoke({"raw_query": query, "session_id": st.session_state.session_id})
                    st.session_state.primary_result = result

                    answer = result["final_answer"]
                    sources = result.get("retrieved_chunks", [])
                    draft_answer = result.get("draft_answer") if result["escalated"] else None
                    status_html = status_row_html(
                        result["desk"], result["escalated"], result.get("escalation_contact"),
                        result.get("confidence", "high"),
                    )

                    st.markdown(answer)
                    st.markdown(status_html, unsafe_allow_html=True)
                    if sources:
                        with st.expander("📎 Sources used"):
                            for src in sources:
                                st.markdown(f"- **{src['source']}** (score: {src['score']:.2f})")
                    if draft_answer:
                        with st.expander("🔍 Show the system's draft answer (unverified)"):
                            st.markdown(draft_answer)

                    st.session_state.messages.append({
                        "role": "assistant", "content": answer, "status_html": status_html,
                        "sources": sources, "draft_answer": draft_answer,
                    })
                else:
                    # Follow-up turn -> tool-using agent with memory, still governed
                    followup = governed_ask(followup_agent, st.session_state.session_id, query)
                    answer = followup["answer"]
                    draft_answer = followup.get("draft_answer") if followup["escalated"] else None
                    status_html = status_row_html(
                        None, followup["escalated"], followup.get("escalation_contact"),
                        followup.get("confidence", "high"),
                    )

                    st.markdown(answer)
                    st.markdown(status_html, unsafe_allow_html=True)
                    if draft_answer and draft_answer != answer:
                        with st.expander("🔍 Show the system's draft answer (unverified)"):
                            st.markdown(draft_answer)

                    st.session_state.qna.append({"question": query, "answer": answer})
                    st.session_state.messages.append({
                        "role": "assistant", "content": answer, "status_html": status_html, "draft_answer": draft_answer,
                    })

st.markdown(
    '<div class="ca-footer">Answers are AI-generated from official circulars and may be escalated to a '
    'human office when confidence is low. Built with LangGraph + Ollama.</div>',
    unsafe_allow_html=True,
)
