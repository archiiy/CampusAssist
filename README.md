# Multi-Agent Campus Assistant and Query Resolution System

An agentic AI college helpdesk built on LangChain + LangGraph. A Router Agent
classifies a student's question into one of five departments, a RAG-based
Desk Agent retrieves grounded answers from real department documents, a
Reflection Agent checks whether the answer is actually supported by what was
retrieved, and a Governance Agent decides whether to answer or escalate to a
human — before a follow-up agent (with memory and tools) handles any
conversational follow-ups and everything is logged to a JSON session report.

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
Follow-Up Agent          -- ReAct agent with tools + memory, same session
      |
Session Report           -- JSON log: query, desk, sources, confidence, escalation
```

## Files

| File | What it does |
|---|---|
| `state.py` | Shared `CampusAssistState` TypedDict |
| `llm.py` | `get_llm()` / `get_embeddings()` — connects to local Ollama (`qwen2.5:3b` + `nomic-embed-text`). Mock backends available for offline testing |
| `mock_backends.py` | Rule-based stand-ins for offline testing only (`CAMPUS_ASSISTANT_MOCK=1`) |
| `knowledge_base/` | Sample department documents (academic, hostel, examcell, placement, fee) — **replace with your actual college's handbook/circulars** |
| `ingest.py` | Chunks the knowledge base and builds a FAISS index per desk. Run this once, or whenever documents change |
| `retriever.py` | Queries a desk's FAISS index for the top matching chunks |
| `router_agent.py` | Classifies the query into one of five desks |
| `desk_agent.py` | Retrieves context and drafts an answer using ONLY that context |
| `reflection_agent.py` | Checks whether the draft answer is actually grounded in the retrieved context |
| `governance_agent.py` | Rule-based: escalate to a human unless the answer is grounded AND high-confidence |
| `tools.py` | `date_calculator` and `kb_lookup` tools for the follow-up agent |
| `followup_agent.py` | Tool-using agent with `MemorySaver` for multi-turn conversation |
| `pipeline.py` | Combines Router + Desk + Reflection + Governance into one compiled graph |
| `report.py` | Builds and saves the JSON session report |
| `main.py` | CLI entry point — runs the full pipeline + follow-up + report for one or more queries |
| `app.py` | Streamlit web UI — chat interface over the same pipeline and follow-up agent |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

ollama pull qwen2.5:3b
ollama pull nomic-embed-text     # embedding model for the vector store
ollama serve
```

## Running it

**1. Build the vector stores first** (only needed once, or when documents change):
```bash
python ingest.py
```

**2a. Web UI (recommended for demos):**
```bash
streamlit run app.py
```
Opens a chat interface in your browser. The sidebar has a "use offline mock
LLM" checkbox for demoing without Ollama running, a "Start new session"
button, and a "Download session report" button once a conversation exists.
The first message in a session goes through the full pipeline (router → RAG
→ reflection → governance); every message after that goes to the follow-up
agent, exactly like the CLI's `--interactive` mode.

**2b. Command line:**
```bash
python main.py                 # built-in sample queries
python main.py --mock          # offline demo, no Ollama needed (also run: python ingest.py with CAMPUS_ASSISTANT_MOCK=1 set first)
python main.py --interactive   # type your own query + follow-ups
```

Reports are saved to `reports/session_report_<id>.json` (CLI) or offered as a
download in the sidebar (web UI).

## Replacing the sample documents with real ones

The `knowledge_base/` folder currently has small, illustrative sample circulars
for each desk. For a real deployment or a stronger demo:

1. Drop real `.txt` files (or extend `ingest.py` to read `.pdf`/`.docx`) into
   the matching desk folder — `knowledge_base/hostel/`, `knowledge_base/fee/`, etc.
2. Re-run `python ingest.py` to rebuild the vector stores.
3. Nothing else changes — the router, desk agent, reflection, and governance
   logic are all document-agnostic.

## Design notes

- **Governance is deliberately rule-based, not another LLM call.** The
  decision to trust or escalate an AI-generated answer shouldn't itself
  depend on an unaccountable AI judgment — a simple, auditable threshold
  (grounded AND high-confidence) is the point.
- **The mock backends** (`CAMPUS_ASSISTANT_MOCK=1`) are for testing graph
  wiring, routing, and the report pipeline without needing Ollama running.
  They use keyword overlap, not real understanding — don't use them to
  judge actual classification, RAG, or reflection quality.
- **Chunking in `ingest.py`** is a simple fixed-size sliding window, fine for
  short circulars. For longer real documents, swap in LangChain's
  `RecursiveCharacterTextSplitter` for smarter paragraph-aware chunking.
