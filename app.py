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
    "academic":  {"label": "Academic",  "icon": ":material/menu_book:"},
    "hostel":    {"label": "Hostel",    "icon": ":material/home:"},
    "examcell":  {"label": "Exam Cell", "icon": ":material/edit_note:"},
    "placement": {"label": "Placement", "icon": ":material/work:"},
    "fee":       {"label": "Fee",       "icon": ":material/payments:"},
}

SUGGESTED_QUESTIONS = [
    "When is the last date to pay hostel fees?",
    "How many days do I have to apply for revaluation?",
    "What CGPA do I need for placements?",
    "What is the minimum attendance required for exams?",
]

USER_AVATAR = ":material/person:"
ASSISTANT_AVATAR = ":material/account_balance:"

CUSTOM_CSS = """
<style>
/*
  Design tokens. Native Streamlit chrome (buttons, chat input, Deploy menu,
  sidebar toggle, container borders) is themed separately via
  .streamlit/config.toml using the SAME color values, so both systems move
  together instead of fighting each other. Light values below are the
  defaults; the prefers-color-scheme block overrides them for dark, mirroring
  config.toml's [theme.dark] (which Streamlit's own "Use system setting"
  theme option already keys off of).

  Contrast ratios (WCAG 2.2), computed against the exact paired background:
    LIGHT  text on bg            #1B2430 / #F7F8FA   14.73:1  (AA body 4.5 req)
    LIGHT  text on surface       #1B2430 / #FFFFFF   15.65:1
    LIGHT  text-muted on bg      #5B6472 / #F7F8FA    5.63:1
    LIGHT  text-muted on surface #5B6472 / #FFFFFF    5.98:1
    LIGHT  accent-text on accent #FFFFFF / #2F5D8A     6.88:1
    LIGHT  accent on surface     #2F5D8A / #FFFFFF     6.88:1
    LIGHT  accent on accent-soft #2F5D8A / #E7EEF5     5.88:1
    LIGHT  text on accent-soft   #1B2430 / #E7EEF5    13.38:1
    LIGHT  danger on surface     #B42318 / #FFFFFF     6.57:1
    LIGHT  border on surface     #7E8A99 / #FFFFFF     3.51:1  (AA UI 3:1 req)
    LIGHT  focus on bg           #2F5D8A / #F7F8FA     6.47:1
    DARK   text on bg            #E7EBF0 / #11161D    15.17:1
    DARK   text on surface       #E7EBF0 / #1A212B    13.53:1
    DARK   text-muted on bg      #9AA5B1 / #11161D     7.26:1
    DARK   text-muted on surface #9AA5B1 / #1A212B     6.47:1
    DARK   accent-text on accent #0B1620 / #7FB2E0     8.13:1
    DARK   accent on surface     #7FB2E0 / #1A212B     7.21:1
    DARK   accent on accent-soft #7FB2E0 / #1E3A52     5.24:1
    DARK   text on accent-soft   #E7EBF0 / #1E3A52     9.84:1
    DARK   danger on bg          #FF6B6B / #11161D     6.54:1
    DARK   border on surface     #687489 / #1A212B     3.43:1
    DARK   focus on bg           #7FB2E0 / #11161D     8.08:1
*/
:root {
    /* semantic colors */
    --bg: #F7F8FA;
    --surface: #FFFFFF;
    --surface-raised: #FFFFFF;
    --text: #1B2430;
    --text-muted: #5B6472;
    --border: #7E8A99;
    --accent: #2F5D8A;
    --accent-text: #FFFFFF;
    --accent-soft: #E7EEF5;
    --focus: #2F5D8A;
    --danger: #B42318;
    --success-text: #12753A;
    --success-soft: #DCFCE7;

    /* type scale */
    --fs-xs: 0.75rem;
    --fs-sm: 0.875rem;
    --fs-base: 1rem;
    --fs-lg: 1.125rem;
    --fs-xl: 1.375rem;
    --fs-2xl: 1.75rem;
    --fs-display: 2.1rem;
    --font-display: 'Lora', Georgia, 'Times New Roman', serif;
    --font-body: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;

    /* spacing scale (4px base) */
    --space-1: 4px;  --space-2: 8px;  --space-3: 12px; --space-4: 16px;
    --space-5: 20px; --space-6: 24px; --space-8: 32px; --space-10: 40px;

    /* radii + one shadow style */
    --radius-sm: 6px;
    --radius-md: 10px;
    --radius-lg: 16px;
    --radius-full: 999px;
    --shadow: 0 1px 2px rgba(15,23,42,0.06), 0 2px 8px rgba(15,23,42,0.08);
}

@media (prefers-color-scheme: dark) {
    :root {
        --bg: #11161D;
        --surface: #1A212B;
        --surface-raised: #212A36;
        --text: #E7EBF0;
        --text-muted: #9AA5B1;
        --border: #687489;
        --accent: #7FB2E0;
        --accent-text: #0B1620;
        --accent-soft: #1E3A52;
        --focus: #7FB2E0;
        --danger: #FF6B6B;
        --success-text: #4ADE80;
        --success-soft: #14532D;
    }
}

@media (prefers-reduced-motion: reduce) {
    * { transition: none !important; animation-duration: 0.001ms !important; }
}

body, [data-testid="stAppViewContainer"] { font-family: var(--font-body); }
[data-testid="stMainBlockContainer"] { padding-top: var(--space-6); max-width: 820px; }

/* ---- Sidebar brand lockup ---- */
.ca-sidebar-brand {
    font-family: var(--font-display); font-size: var(--fs-xl); font-weight: 600;
    display: flex; align-items: center; gap: var(--space-2); color: var(--text);
}
.ca-sidebar-tagline { font-size: var(--fs-sm); color: var(--text-muted); margin-bottom: var(--space-3); }
.ca-session-card {
    background: var(--accent-soft); border-radius: var(--radius-md); padding: var(--space-3) var(--space-4);
    font-size: var(--fs-sm); color: var(--text); margin-bottom: var(--space-3);
}
.ca-session-card code { color: var(--accent); background: transparent; font-weight: 600; }
.ca-desk-row { margin-bottom: var(--space-2); }

/* ---- Hero banner ---- */
.ca-hero {
    background: var(--accent-soft); border: 1px solid var(--border); border-radius: var(--radius-lg);
    padding: var(--space-6) var(--space-8); margin-bottom: var(--space-6);
}
.ca-hero-title {
    font-family: var(--font-display); font-size: var(--fs-display); font-weight: 600;
    color: var(--text); margin: 0 0 var(--space-2) 0; display: flex; align-items: center; gap: var(--space-3);
}
.ca-hero-title [data-testid="stIconMaterial"] { color: var(--accent); font-size: 1.6rem; }
.ca-hero-sub { font-size: var(--fs-base); color: var(--text-muted); max-width: 640px; line-height: 1.55; margin: 0; }

/* ---- "Try one of these" section heading ---- */
.ca-welcome-title {
    font-family: var(--font-display); font-weight: 600; font-size: var(--fs-lg);
    color: var(--text); margin: var(--space-2) 0 var(--space-4) 0;
}

/* ---- Buttons: lightweight chips, not heavy slabs ---- */
[data-testid="stButton"] button, [data-testid="stDownloadButton"] button {
    background: var(--surface) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    font-weight: 500;
    min-height: 44px;
    box-shadow: none !important;
}
[data-testid="stButton"] button:hover, [data-testid="stDownloadButton"] button:hover {
    background: var(--accent-soft) !important;
    border-color: var(--accent) !important;
    color: var(--accent) !important;
}
[data-testid="stButton"] button:focus-visible, [data-testid="stDownloadButton"] button:focus-visible {
    outline: 2px solid var(--focus); outline-offset: 2px;
}

/* ---- Chat bubbles: understated, not heavy ---- */
[data-testid="stChatMessage"] { padding: var(--space-1) 0; }
[data-testid="stChatMessageContent"] {
    border-radius: var(--radius-md) !important;
    padding: var(--space-3) var(--space-4) !important;
    line-height: 1.6;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stChatMessageContent"] {
    background: var(--surface); border: 1px solid var(--border);
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
    background: var(--accent-soft);
}
[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"] {
    background: var(--accent-soft) !important; color: var(--accent) !important;
}

/* ---- Status pills: text + icon convey meaning, not color alone ---- */
.ca-status-row { margin-top: var(--space-3); display: flex; gap: var(--space-2); flex-wrap: wrap; }
.ca-pill {
    display: inline-flex; align-items: center; gap: var(--space-1); padding: var(--space-1) var(--space-3);
    border-radius: var(--radius-full); font-size: var(--fs-xs); font-weight: 600; white-space: nowrap;
    border: 1px solid transparent;
}
.ca-pill-desk { background: var(--accent-soft); color: var(--accent); border-color: var(--border); }
.ca-pill-ok { background: var(--success-soft); color: var(--success-text); }
.ca-pill-warn { background: transparent; color: var(--danger); border-color: var(--danger); }

/* ---- Footer ---- */
.ca-footer {
    text-align: center; color: var(--text-muted); font-size: var(--fs-sm);
    margin-top: var(--space-8); padding: var(--space-4) 0; line-height: 1.5;
}
</style>
"""

st.html(CUSTOM_CSS)


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
    meta = DESK_META.get(desk_key, {"label": desk_key.title(), "icon": ":material/apartment:"})
    return f'<span class="ca-pill ca-pill-desk">{meta["icon"]} {meta["label"]}</span>'


def status_pill(escalated: bool, contact: str | None, confidence: str) -> str:
    if escalated:
        return f'<span class="ca-pill ca-pill-warn">:material/error: Escalated → {contact or "relevant department"}</span>'
    return f'<span class="ca-pill ca-pill-ok">:material/check_circle: Grounded · {confidence} confidence</span>'


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
    st.markdown('<h2 class="ca-sidebar-brand">:material/school: CampusAssist</h2>', unsafe_allow_html=True)
    st.markdown('<p class="ca-sidebar-tagline">Your AI-powered college helpdesk</p>', unsafe_allow_html=True)

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
    if st.button("Start new session", icon=":material/refresh:", use_container_width=True):
        reset_session()
        st.rerun()

    if st.session_state.primary_result:
        report = build_report(st.session_state.primary_result, st.session_state.qna)
        st.download_button(
            "Download session report",
            icon=":material/download:",
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
      <h1 class="ca-hero-title">:material/school: CampusAssist</h1>
      <p class="ca-hero-sub">
        Ask about academics, hostel, exams, placements, or fees. Answers are grounded
        in official circulars — and escalated to the right office whenever we're not confident.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

chat_card = st.container(border=True)

with chat_card:
    if not st.session_state.messages:
        st.markdown('<h2 class="ca-welcome-title">Try one of these</h2>', unsafe_allow_html=True)
        cols = st.columns(2)
        for i, q in enumerate(SUGGESTED_QUESTIONS):
            if cols[i % 2].button(q, use_container_width=True, key=f"suggest_{i}"):
                st.session_state.pending_query = q
                st.rerun()

    for msg in st.session_state.messages:
        avatar = USER_AVATAR if msg["role"] == "user" else ASSISTANT_AVATAR
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            if msg.get("status_html"):
                st.markdown(msg["status_html"], unsafe_allow_html=True)
            if msg.get("sources"):
                with st.expander("Sources used", icon=":material/link:"):
                    for src in msg["sources"]:
                        st.markdown(f"- **{src['source']}** (score: {src['score']:.2f})")
            if msg.get("draft_answer"):
                with st.expander("Show the system's draft answer (unverified)", icon=":material/visibility:"):
                    st.markdown(msg["draft_answer"])

    query = st.chat_input("Ask your question...")
    if not query and st.session_state.pending_query:
        query = st.session_state.pending_query
    st.session_state.pending_query = None

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(query)

        with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
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
                        with st.expander("Sources used", icon=":material/link:"):
                            for src in sources:
                                st.markdown(f"- **{src['source']}** (score: {src['score']:.2f})")
                    if draft_answer:
                        with st.expander("Show the system's draft answer (unverified)", icon=":material/visibility:"):
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
                        with st.expander("Show the system's draft answer (unverified)", icon=":material/visibility:"):
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
