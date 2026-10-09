# Multi-Agent Campus Assistant and Query Resolution System

An agentic AI college helpdesk built on LangChain + LangGraph. A Router Agent
classifies a student's question into one of five departments, a RAG-based
Desk Agent retrieves grounded answers from real department documents, a
Reflection Agent checks whether the answer is actually supported by what was
retrieved, and a Governance Agent decides whether to answer or escalate to a
human — before a follow-up agent (with persistent memory and tools) handles
any conversational follow-ups, itself subject to the same reflection/
governance checks, and everything is logged to a JSON session report.

See `IMPROVEMENT_PLAN.md` for the structural rationale behind the current
layout (package restructure, governance-gap fix, PDF ingestion, tests, etc.).

## Architecture

```
Student Query
      |
Router Agent            -- classifies: academic / hostel / examcell / placement / fee
      |
Desk Agent (RAG)         -- retrieves from that desk's FAISS vector store, drafts an answer
      |
Reflection Agent         -- checks the draft answer against retrieved context
      |
Governance Agent         -- grounded+high-confidence -> answer; else -> escalate
      |
Follow-Up Agent          -- ReAct agent with tools + persistent memory, same session,
                             governed by the same reflection/governance checks
      |
Session Report           -- JSON log: query, desk, sources, confidence, escalation
```

## Project layout

```
main.py                     CLI entry point
app.py                       Streamlit web UI
scripts/ingest.py            CLI wrapper to (re)build the vector stores
src/campus_assistant/
    config.py                 Central settings (paths, model names, mock flag, TODAY override, checkpointer backend)
    state.py                  Shared CampusAssistState TypedDict
    prompts.py                Shared prompt templates + mock-detection markers
    tools.py                  date_calculator / kb_lookup tools for the follow-up agent
    pipeline.py                Compiles Router + Desk + Reflection + Governance into one LangGraph
    report.py                  Builds/saves the JSON session report
    agents/
        router_agent.py         Classifies the query into one of five desks
        desk_agent.py            Retrieves context and drafts an answer using ONLY that context
        reflection_agent.py      Checks whether the draft answer is grounded in retrieved context
        governance_agent.py      Rule-based: escalate unless grounded AND high-confidence
        followup_agent.py        Tool-using agent with governed_ask() for governed follow-up turns
    rag/
        ingest.py                Loads .txt/.pdf docs, chunks, embeds, builds a FAISS index per desk
        chunking.py              RecursiveCharacterTextSplitter-based chunker (pure, unit-tested)
        retriever.py             Queries a desk's FAISS index for top-k matching chunks
    llm/
        factory.py               get_llm() / get_embeddings() — local Ollama, cached, mock-aware
        mock_backends.py         Offline mock LLM for testing without Ollama
    memory/
        checkpointer.py          SQLite (default) or in-memory checkpointer for follow-up sessions
knowledge_base/<desk>/*.txt,*.pdf   Sample department documents — replace with your college's real ones
vector_store/<desk>.index/.meta.pkl  Pre-built FAISS indexes (run scripts/ingest.py to regenerate)
tests/                         pytest suite — runs fully offline (mock mode), no Ollama required
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -e .                 # installs the campus_assistant package (editable)
pip install -r requirements.txt  # pinned runtime deps
# for running tests too:
pip install -r requirements-dev.txt

ollama pull qwen2.5:3b
ollama pull nomic-embed-text     # embedding model for the vector store
ollama serve
```

## Configuration

All settings live in `src/campus_assistant/config.py` and can be overridden
via environment variables (or a `.env` file — see `.env.example`):

| Env var | Default | Purpose |
|---|---|---|
| `CAMPUS_ASSISTANT_MOCK` | `0` | `1` to use the offline mock LLM/embeddings instead of Ollama |
| `CAMPUS_ASSISTANT_TODAY` | `2026-09-19` | Overrides "today" for `date_calculator`, matching the sample knowledge base's dates |
| `CAMPUS_ASSISTANT_CHECKPOINTER_BACKEND` | `sqlite` | `sqlite` (persistent, default) or `memory` (non-persistent, used by tests) |
| `CAMPUS_ASSISTANT_LLM_MODEL` | `qwen2.5:3b` | Ollama chat model |
| `CAMPUS_ASSISTANT_EMBEDDING_MODEL` | `nomic-embed-text` | Ollama embedding model |

Paths (`kb_dir`, `vector_store_dir`, `reports_dir`, `checkpointer_sqlite_path`)
are resolved relative to the project root, not the current working
directory, so every entry point behaves the same regardless of where it's
invoked from.

## Running it

**1. Build the vector stores first** (only needed once, or when documents change):
```bash
python scripts/ingest.py
```
Note: the index is built against whichever embedding model is active
(real Ollama vs. mock) — if you switch `CAMPUS_ASSISTANT_MOCK`, re-run
ingestion so the index dimension matches.

**2a. Web UI (recommended for demos):**
```bash
streamlit run app.py
```
Opens a chat interface in your browser. The sidebar has a "use offline mock
LLM" checkbox for demoing without Ollama running, a "Start new session"
button, and a "Download session report" button once a conversation exists.
Every turn — the first message and all follow-ups — goes through
reflection/governance, with an escalation badge and an optional "unverified
draft answer" expander when the system wasn't confident.

**2b. Command line:**
```bash
python main.py                 # built-in sample queries
python main.py --mock          # offline demo, no Ollama needed (run scripts/ingest.py with CAMPUS_ASSISTANT_MOCK=1 first)
python main.py --interactive   # type your own query + follow-ups
```

Reports are saved to `reports/session_report_<id>.json` (CLI) or offered as a
download in the sidebar (web UI). Follow-up session memory persists across
restarts in `followup_sessions.sqlite` (gitignored) unless
`CAMPUS_ASSISTANT_CHECKPOINTER_BACKEND=memory` is set.

## Running the tests

```bash
pip install -r requirements-dev.txt
pytest
```

The suite forces offline mock mode and an isolated temp vector store (see
`tests/conftest.py`), so it runs fully offline with no live Ollama server
and without touching the real `vector_store/` or leaving `.sqlite` files
behind.

## Replacing the sample documents with real ones

The `knowledge_base/` folder currently has small, illustrative sample circulars
for each desk. For a real deployment or a stronger demo:

1. Drop real `.txt` or `.pdf` files into the matching desk folder —
   `knowledge_base/hostel/`, `knowledge_base/fee/`, etc.
2. Re-run `python scripts/ingest.py` to rebuild the vector stores.
3. Nothing else changes — the router, desk agent, reflection, and governance
   logic are all document-agnostic.

## Design notes

- **Governance is deliberately rule-based, not another LLM call.** The
  decision to trust or escalate an AI-generated answer shouldn't itself
  depend on an unaccountable AI judgment — a simple, auditable threshold
  (grounded AND high-confidence) is the point. On escalation, the original
  draft answer is kept (not discarded) so the UI can optionally show it,
  clearly labeled as unverified.
- **Follow-up turns are governed too.** `followup_agent.governed_ask()` runs
  any `kb_lookup`-backed follow-up answer through the same reflection and
  governance checks as the first turn, rather than trusting the tool-using
  agent's output directly. `date_calculator`-only answers skip the
  groundedness check since they're deterministic arithmetic, not a factual
  claim against retrieved context.
- **The mock backends** (`CAMPUS_ASSISTANT_MOCK=1`) are for testing graph
  wiring, routing, and the report pipeline without needing Ollama running.
  They use keyword overlap, not real understanding — don't use them to
  judge actual classification, RAG, or reflection quality.
- **Chunking** (`rag/chunking.py`) uses LangChain's `RecursiveCharacterTextSplitter`
  for paragraph/sentence-aware splitting of both `.txt` and `.pdf` sources.
