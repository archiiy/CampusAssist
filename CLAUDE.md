# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

CampusAssist — an agentic AI college helpdesk (for Sathyabama Institute students) built on LangChain + LangGraph. A Router Agent classifies a student's question into one of five departments, a RAG-based Desk Agent retrieves grounded answers from department documents, a Reflection Agent checks whether the answer is actually supported by what was retrieved, and a Governance Agent decides whether to answer or escalate to a human. A follow-up agent (with persistent memory and tools) handles conversational follow-ups under the same governance checks. Everything is local: Ollama (`qwen2.5:3b` + `nomic-embed-text`), FAISS, no cloud APIs.

## Commands

```bash
# Setup (once)
pip install -e .                    # installs the campus_assistant package (editable)
pip install -r requirements.txt     # pinned runtime deps
pip install -r requirements-dev.txt # + pytest, for running tests

ollama pull qwen2.5:3b
ollama pull nomic-embed-text
ollama serve

# Build/rebuild the vector store (run once, or whenever knowledge_base/ changes)
python scripts/ingest.py

# Run
streamlit run app.py          # web UI
python main.py                # CLI, runs built-in sample queries
python main.py --interactive  # CLI, type your own query + follow-ups
python main.py --mock         # offline demo, no Ollama (rebuild index with CAMPUS_ASSISTANT_MOCK=1 first — see Mock mode below)

# Tests (fully offline, no Ollama required — conftest.py forces mock mode)
pytest                                          # full suite
pytest tests/test_governance_agent.py           # single file
pytest tests/test_governance_agent.py::test_unknown_desk_falls_back_to_generic_department  # single test
```

There is no linter/formatter configured in this repo (no ruff/black config) and no CI.

## Architecture

### Pipeline flow

```
query → router_agent → desk_agent (RAG) → reflection_agent → governance_agent → final_answer
                                                                      ↘ escalation (if unconfident)
```

`src/campus_assistant/pipeline.py` wires this as a LangGraph `StateGraph` over `CampusAssistState` (`state.py`), a single `TypedDict` threaded through every node. All five desk labels route to the same generic `desk_agent` node (one function parameterized by `state["desk"]`, not five separate agents).

**Follow-up turns are a second code path**, not graph nodes: `agents/followup_agent.py`'s `build_followup_agent()` wraps `langchain.agents.create_agent` with `kb_lookup`/`date_calculator` tools (`tools.py`) and a checkpointer. The important detail: `governed_ask()` (not the older `ask()`, which is ungoverned and kept only for its own demo block) re-runs any `kb_lookup`-backed answer through `reflection_agent`/`governance_agent` manually, building a mini-state dict — this is what keeps follow-up turns subject to the same grounding/escalation checks as the first turn. **When editing `governed_ask()`**: the checkpointer returns the *entire* conversation history on every `agent.invoke()` call, not just the new turn — it must diff against `agent.get_state(config)`'s prior message count before inspecting tool calls, or a `kb_lookup` result from an earlier turn in the same session gets mistaken for evidence backing a later, unrelated answer (this was a real bug found during development).

### Mock mode

`src/campus_assistant/llm/factory.py`'s `get_llm()`/`get_embeddings()` take an explicit `mock` parameter (default: `config.settings.mock_mode`) and swap in `llm/mock_backends.py`'s `MockChatModel`/`_MockEmbeddings` instead of `ChatOllama`/`OllamaEmbeddings`. The mock detects which behavior to emulate (classify / draft answer / reflect / tool-call) by sniffing marker substrings that are shared constants in `prompts.py` — the real prompts and the mock import the *same* `ROUTER_MARKER`/`ANSWER_MARKER`/`REFLECTION_MARKER` constants, so editing a real prompt can't silently desync the mock.

**Embedding dimension gotcha**: real Ollama embeddings (`nomic-embed-text`, 768-dim) and mock embeddings (256-dim) are not interchangeable against the same FAISS index. Switching `CAMPUS_ASSISTANT_MOCK` requires re-running `python scripts/ingest.py` to rebuild `vector_store/` at the matching dimension, or retrieval raises a FAISS dimension-mismatch assertion. The test suite avoids this entirely by building its own disposable temp vector store with mock embeddings in `tests/conftest.py`, rather than touching the committed `vector_store/` (which is built with real embeddings, for normal app use).

### Config (`src/campus_assistant/config.py`)

Single `Settings` object (pydantic-settings) — paths (`kb_dir`, `vector_store_dir`, `reports_dir`, `checkpointer_sqlite_path`) resolve from the project root via `Path(__file__)`, not CWD, so every entry point behaves the same regardless of invocation directory. `mock_mode` and `today_override` use explicit `validation_alias` (`CAMPUS_ASSISTANT_MOCK`, `CAMPUS_ASSISTANT_TODAY`) because pydantic-settings' default env-var naming would otherwise require the longer `CAMPUS_ASSISTANT_MOCK_MODE`/`CAMPUS_ASSISTANT_TODAY_OVERRIDE` — worth remembering if adding new settings fields that want a short env var name. `today_override` defaults to `2026-09-19` to match the hardcoded dates in the sample knowledge base and `main.py`'s `SAMPLE_QUERIES`.

### Follow-up memory

`memory/checkpointer.py`'s `get_checkpointer()` returns a `SqliteSaver` (persists to `followup_sessions.sqlite` at the project root) by default, or `MemorySaver` when `settings.checkpointer_backend == "memory"` (used by tests, so pytest runs don't leave `.sqlite` files behind). The sqlite connection is opened directly (not via `SqliteSaver.from_conn_string`'s context-manager form) since the checkpointer must outlive a single call across the app/CLI process lifetime.

### UI (`app.py`)

Single dark theme only — `.streamlit/config.toml` (`base = "dark"`) and `app.py`'s injected CSS `:root` tokens use the *same* hex values deliberately, since they theme different things: config.toml controls native Streamlit widget chrome (buttons, chat input, Deploy menu, sidebar toggle), while the CSS tokens style custom HTML (hero banner, pills, chat bubble backgrounds). If one is changed, the other needs the matching update or they'll visually clash. The CSS `<style>` block is injected via `st.html()`, not `st.markdown(unsafe_allow_html=True)` — Streamlit's documented pattern for pure CSS/HTML. No icons/emoji are used anywhere in the UI (removed deliberately — keep it that way unless asked).

**Known gap, intentionally not fixed**: `app.py` has no try/except around `pipeline_app.invoke(...)` or `governed_ask(...)` — if Ollama is down, the student sees a raw Python traceback instead of a friendly message. This is a logic change, not a styling one, so it was explicitly left alone during the UI-only redesign work; see `IMPROVEMENT_PLAN.md` for the fuller fix list.

### Reports

`report.py`'s `build_report()`/`save_report()` build a JSON summary of a session (query, desk, retrieval sources, reflection/confidence, escalation, follow-up Q&A) — used by both `main.py` (writes to `reports/`) and `app.py` (offered as a sidebar download).

## Other notes

- `IMPROVEMENT_PLAN.md` documents the full restructuring rationale (why the flat-file layout became `src/campus_assistant/`, the governance-gap fix, PDF ingestion, etc.) — read it for historical context on *why* things are shaped this way, not just *what* they are.
- `ui_backup/app.py` is a manual, one-off backup of `app.py` from before a UI redesign round (kept at the user's request) — not an automated mechanism, don't rely on it being kept up to date.
- `knowledge_base/` contains small illustrative sample circulars per desk — real deployment should replace these with the institute's actual documents, then re-run `python scripts/ingest.py`.
