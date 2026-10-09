# CampusAssist — Structural Improvement Plan

> **This is a planning document only.** No code in this repository has been changed as part of producing this file. It documents a reviewed, verified plan for restructuring and hardening the project; execution is expected to happen incrementally, phase by phase, in future sessions.

## 1. Overview

CampusAssist is a local, Ollama-based agentic college helpdesk:

```
student query → Router Agent → Desk RAG Agent (FAISS) → Reflection Agent → Governance Agent → final answer
                                                                                   ↘ escalation (if unconfident)
```

It also has a follow-up agent (LangGraph `create_agent` + `MemorySaver`) for conversational follow-ups, a Streamlit chat UI (`app.py`), and a CLI (`main.py`).

**Current state** (verified by reading every source file):

- Flat layout: 15 `.py` files directly at the project root, no package, no `__init__.py`, no `src/`, no `tests/`.
- Tech stack: LangGraph `StateGraph` orchestration, `langchain-ollama` (`qwen2.5:3b` chat, `nomic-embed-text` embeddings), FAISS (`faiss-cpu`, `IndexFlatIP`) for retrieval, Streamlit UI, offline mock backend behind `CAMPUS_ASSISTANT_MOCK=1`.
- Data: `knowledge_base/<desk>/*.txt` (sample circulars) → pre-built `vector_store/<desk>.index` + `.meta.pkl`.
- No git repository, no tests, no pinned dependencies, no central config module.

**Goals of this plan**: make the project more maintainable (proper package structure), close its single biggest reliability gap (follow-up turns currently skip the reflection/governance checks entirely), fix a set of smaller robustness and correctness issues, and add tests/persistence — all while staying local-only (Ollama) and not over-engineering a ~1,050-line project.

---

## 2. Target package layout

```
campus_assistant/
├── .env.example
├── .gitignore
├── pyproject.toml                     # package-dir = {"" = "src"}; pip install -e .
├── requirements.txt                   # pinned
├── requirements-dev.txt               # pytest, etc.
├── README.md
├── main.py                            # thin CLI entry point
├── app.py                             # thin Streamlit entry point
├── knowledge_base/<desk>/*.txt,*.pdf  # unchanged location, PDF support added
├── vector_store/<desk>.index/.meta.pkl
├── reports/                           # gitignored, runtime output
├── scripts/
│   └── ingest.py                      # CLI wrapper calling rag/ingest.py
├── src/
│   └── campus_assistant/
│       ├── __init__.py
│       ├── config.py                  # NEW — central Settings
│       ├── state.py                   # trimmed (dead fields removed)
│       ├── prompts.py                 # NEW — shared prompt text + marker constants
│       ├── tools.py                   # TODAY sourced from config
│       ├── pipeline.py
│       ├── report.py
│       ├── agents/
│       │   ├── __init__.py
│       │   ├── router_agent.py
│       │   ├── desk_agent.py
│       │   ├── reflection_agent.py
│       │   ├── governance_agent.py
│       │   └── followup_agent.py
│       ├── rag/
│       │   ├── __init__.py
│       │   ├── ingest.py              # + PDF support, delegates to chunking.py
│       │   ├── retriever.py
│       │   └── chunking.py            # NEW — RecursiveCharacterTextSplitter wrapper
│       ├── llm/
│       │   ├── __init__.py
│       │   ├── factory.py             # was llm.py; cached, explicit mock param
│       │   └── mock_backends.py
│       └── memory/
│           ├── __init__.py
│           └── checkpointer.py        # NEW — SqliteSaver / MemorySaver switch
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── test_router_agent.py
    ├── test_reflection_agent.py
    ├── test_governance_agent.py
    ├── test_chunking.py
    ├── test_report.py
    └── test_config.py
```

Not added, deliberately: no `domain/`/`services/` layering, no DI framework — `agents/`, `rag/`, `llm/`, `memory/` map 1:1 onto the architecture that already exists; the move is mechanical, not a redesign.

**Migration mechanics**: intra-project imports change from flat (`from state import CampusAssistState`) to package-qualified (`from campus_assistant.state import CampusAssistState`). Add a minimal `pyproject.toml` with `package-dir = {"" = "src"}` and use `pip install -e .` rather than `sys.path` hacks, so `main.py`/`app.py` can import the package normally.

---

## 3. Issue-by-issue fix list

1. **Dead state fields** (`state.py`) — `conversation` and `report` are declared in `CampusAssistState` but never written/read anywhere. Remove them. Add a new, actually-used field: `router_fallback: bool` (see #2).

2. **Router silent fallback to "academic"** (`router_agent.py`) — when the LLM returns a label outside `VALID_DESKS`, it's silently rewritten to `"academic"`, indistinguishable from a genuine classification. Fix: log a warning and set `router_fallback=True` in state so the UI/report can surface "routed by fallback" distinctly.

3. **Brittle reflection parsing** (`reflection_agent.py`) — parses LLM output via exact substring checks (`"grounded: yes" in text.lower()`); any format deviation (likely with a 3B model) silently resolves to not-grounded/low-confidence with no signal that parsing failed. Fix: extract a pure `parse_reflection_output(text)` function using independent regexes for `GROUNDED:`/`CONFIDENCE:`/`REASON:`; when a field doesn't match *at all*, set `reflection_notes="PARSE_FAILURE: ..."` and log a warning, so a parse failure is distinguishable from a genuine "not grounded" verdict.

4. **Mock backend tightly coupled to literal prompt text** (`mock_backends.py`) — detects which behavior to emulate via substrings duplicated from the real prompts in `router_agent.py`/`desk_agent.py`/`reflection_agent.py`, with no shared constant. Fix: add `prompts.py` holding both the real prompt templates and short marker constants (`ROUTER_MARKER`, `ANSWER_MARKER`, `REFLECTION_MARKER`); both the real agents and the mock backend import from this one source.

5. **Hardcoded "today" scattered across 3 files** (`tools.py`'s `TODAY = datetime(2026, 9, 19)`, `main.py`'s `SAMPLE_QUERIES`, the hostel KB text) — implicitly synchronized, no shared constant. Fix: add `Settings.today_override` to `config.py` (env var `CAMPUS_ASSISTANT_TODAY`, defaults to `date.today()`); `tools.py` reads from there instead of a local hardcoded constant.

6. **CWD-dependent relative paths** (`ingest.py`'s `KB_DIR`/`INDEX_DIR`, `retriever.py`'s `INDEX_DIR`, `report.py`'s `out_dir="reports"`) — break if invoked from a different working directory. Fix: `config.py` computes `PROJECT_ROOT = Path(__file__).resolve().parents[2]` and derives `KB_DIR`, `VECTOR_STORE_DIR`, `REPORTS_DIR` from it; all modules import these instead of defining their own relative constants.

7. **No PDF/DOCX ingestion** despite handbooks/circulars realistically being PDFs — `ingest.py` only globs `*.txt`. See §4.

8. **No persistent follow-up memory** — `followup_agent.py` uses in-process `MemorySaver`, lost on restart. See §6.

9. **Follow-up turns bypass governance entirely — highest priority fix.** In `app.py` (lines ~109-139) and `main.py`, only the *first* message in a session goes through the full pipeline (router → desk → reflection → governance). Every subsequent message goes straight to `followup_agent.ask()`, which calls `kb_lookup`/`date_calculator` tools with **no reflection or governance check on the output** — so a confident-looking but ungrounded follow-up answer can reach the student with no escalation. Fix:
   - Add `governed_ask(agent, session_id, question) -> dict` in `agents/followup_agent.py` that calls `agent.invoke(...)`, inspects the returned messages for a `ToolMessage` from `kb_lookup`, and if one is present, builds a mini-state dict (`{"draft_answer": ..., "retrieved_chunks": [{"text": tool_content, "source": "followup_kb_lookup"}]}`) and runs it through the *existing* `reflection_agent()` then `governance_agent()` — no graph coupling needed, they already operate on plain dicts.
   - If only `date_calculator` was used (no factual claim beyond arithmetic), skip the grounding check and set `is_grounded=True, confidence="high"` directly.
   - Returns `{"answer": ..., "escalated": bool, "escalation_contact": ..., "confidence": ...}`.
   - `app.py`'s follow-up branch and `main.py`'s follow-up loop call `governed_ask` instead of `ask`, and render the same escalation badge UI already built for turn 1.
   - Keep `ask()` for the `__main__` demo block, with a docstring noting it bypasses governance.

10. **No caching of `get_llm()`/`get_embeddings()`** (`llm.py`) — a fresh `ChatOllama`/`OllamaEmbeddings`/`MockChatModel` instance is constructed on every node call. Fix: wrap both factories with `functools.lru_cache`, keyed to include the mock flag explicitly (see #11) so toggling mock mode doesn't return a stale cached instance.

11. **Streamlit `@st.cache_resource` keyed on an ignored parameter** (`app.py`) — `get_pipeline(mock)`/`get_followup_agent(mock)` take `mock` only to vary the cache key; actual behavior is driven by the global `CAMPUS_ASSISTANT_MOCK` env var read deep inside `llm.py`. Fix: thread `mock` explicitly through `build_pipeline_graph(mock=...)` → `get_llm(mock=...)`/`get_embeddings(mock=...)` as real parameters (defaulting to `config.settings.mock_mode` when omitted), removing the implicit env-var reach-through.

12. **Governance fully discards a possibly-correct draft on escalation** (`governance_agent.py`) — `final_answer` is replaced with generic boilerplate, and the original `draft_answer` is never shown. Fix: keep `draft_answer` in state (already there, just not surfaced) and add an "Show the system's draft answer (unverified)" expander in `app.py`'s escalation badge rendering — preserves the "safe" boilerplate as the default while not hiding potentially useful information.

13. **No automated tests.** See §7.

14. **Cosmetic issues in `app.py`** — `page_icon=" "` (a literal space), `"hostel": " Hostel"` (stray leading space in `DESK_LABELS`), trailing whitespace/blank lines at EOF. Fix directly; consider adding `ruff format`/`black` to dev tooling so this class of issue doesn't recur silently.

15. **Not a git repo** — `.venv/` and `__pycache__/` sit directly in the working tree with no exclusion mechanism. See §8.

16. **Unpinned dependencies** (`requirements.txt`: `langchain>=1.0`, `langgraph>=1.0`, and fully unpinned `langchain-community`, `langchain-ollama`, `pydantic>=2.0`, `faiss-cpu`, `numpy`, `streamlit`). Fix: `pip freeze` the working `.venv`, pin everything with `==`, add new pinned deps (`pypdf`, `langchain-text-splitters`, `langgraph-checkpoint-sqlite`, `pytest`). Split into `requirements.txt` (runtime) + `requirements-dev.txt` (test tooling).

---

## 4. PDF ingestion + chunking plan

- Extract a `load_documents(desk: str) -> list[dict]` function in `rag/ingest.py` that globs both `*.txt` and `*.pdf` under `knowledge_base/<desk>/`. `.txt` stays as-is; `.pdf` uses `pypdf.PdfReader` to extract per-page text, recording `source` as `"<filename>.pdf (page N)"`.
- Move chunking into a standalone `rag/chunking.py` module: `chunk_text(text: str, source: str, chunk_size=500, chunk_overlap=80) -> list[dict]`, internally using `RecursiveCharacterTextSplitter(chunk_size=..., chunk_overlap=..., separators=["\n\n", "\n", ". ", " ", ""])` from `langchain_text_splitters`, returning the same `{"text", "source"}` dict shape already consumed downstream.
- **No change to the FAISS index format or `.meta.pkl` schema** — the chunk dict shape is preserved, so `retriever.py` needs no changes.
- `chunk_size`/`chunk_overlap` move into `config.py` as settings instead of module-level constants, so they're centrally tunable.
- This module has zero FAISS/Ollama imports, making it directly unit-testable (§7).

---

## 5. Config module plan

Add `src/campus_assistant/config.py` with a `Settings` object (dataclass or `pydantic-settings`, the latter preferred since `pydantic>=2.0` is already a transitive dependency via LangChain):

- `mock_mode: bool` — replaces ad hoc `os.environ.get("CAMPUS_ASSISTANT_MOCK")` reads scattered across `llm.py`/`app.py`.
- `llm_model`, `embedding_model`, `llm_temperature` — centralizes the `qwen2.5:3b`/`nomic-embed-text` defaults.
- `project_root`, `kb_dir`, `vector_store_dir`, `reports_dir` — all derived from `Path(__file__).resolve().parents[2]`, not CWD.
- `chunk_size`, `chunk_overlap`.
- `today_override: date | None` — `CAMPUS_ASSISTANT_TODAY` env var.
- `checkpointer_backend: str` (`"memory"` | `"sqlite"`), `checkpointer_sqlite_path`.

Every module currently reading `os.environ` directly for config switches to importing `settings` from `config.py`.

---

## 6. Persistent follow-up memory plan

**Recommendation (approved): add `SqliteSaver` from `langgraph-checkpoint-sqlite`.**

- Add `src/campus_assistant/memory/checkpointer.py` with `get_checkpointer()` returning `SqliteSaver.from_conn_string(str(settings.checkpointer_sqlite_path))` when `settings.checkpointer_backend == "sqlite"`, else `MemorySaver()`.
- `agents/followup_agent.py`'s `build_followup_agent()` calls `get_checkpointer()` instead of instantiating `MemorySaver()` directly.
- SQLite path comes from `config.py` (project-root-relative, not CWD-relative — ties into fix #6).
- Default to `"sqlite"` in normal use; tests force `"memory"` via `conftest.py` so `pytest` runs don't leave `.sqlite` files behind.
- Add the `.sqlite` file to `.gitignore` (generated runtime state).
- This is a drop-in swap — `SqliteSaver` implements the same `BaseCheckpointSaver` interface as `MemorySaver`, so `create_agent(...)`'s call shape in `followup_agent.py` doesn't change. No new schema design needed; LangGraph owns the checkpoint schema internally.

---

## 7. Testing plan (pytest)

Add `tests/` with `conftest.py` forcing `CAMPUS_ASSISTANT_MOCK=1` (and `checkpointer_backend="memory"`) at session scope, so no test requires a live Ollama server:

- **`test_router_agent.py`** — monkeypatch `get_llm()` to return a stub producing an invalid label; assert fallback to `"academic"` **and** `router_fallback=True` (post-fix #2).
- **`test_reflection_agent.py`** — test the extracted `parse_reflection_output()` directly against well-formed output, missing `GROUNDED:` line (expect parse-failure flag per fix #3), case variations, extra whitespace.
- **`test_governance_agent.py`** — table-driven over `governance_agent()`: grounded+high → not escalated; any other combination → escalated with correct `escalation_contact` per desk; unknown desk → `"the relevant department"`.
- **`test_chunking.py`** — `chunk_text()` on empty string, short string, multi-chunk string with verified overlap, no empty/whitespace-only chunks.
- **`test_report.py`** — `build_report()` shape on a fake state dict; `save_report()` round-trips through a `tmp_path` fixture.
- **`test_config.py`** — `Settings` resolves `project_root`/`kb_dir`/`vector_store_dir` to existing absolute paths; `CAMPUS_ASSISTANT_TODAY` override is honored.
- Optional: **`test_pipeline_mock.py`** — full `build_pipeline_graph()` run under mock mode against existing sample queries (reusing the manual test cases already in `pipeline.py`'s `__main__` block), asserting escalation triggers for out-of-scope queries.

Add `pytest` (pinned) to `requirements-dev.txt`. No CI system yet — premature before the repo is even a git repo.

---

## 8. Git init + .gitignore plan

1. `git init` at the project root.
2. `.gitignore`: `.venv/`, `__pycache__/`, `*.pyc`, `reports/`, `*.sqlite`, `.env`.
3. **Track** `vector_store/` and `knowledge_base/` — the pre-built sample indexes are small, and committing them means a fresh clone works immediately without requiring Ollama + a manual `ingest.py` run first. Revisit (gitignore + document regeneration) if the KB grows to real handbook-sized PDFs.
4. Commit the current flat-file state first (baseline), then do the `src/` restructuring as a second, reviewable commit.

---

## 9. Phased rollout

Each phase is independently shippable; later phases depend on earlier ones only where noted.

| Phase | Scope | Risk/Value |
|---|---|---|
| **0** | Git init, `.gitignore`, commit current flat state as baseline | Prerequisite, ~15 min |
| **1** | Full `src/campus_assistant/` restructure; `config.py`; path fixes (#6); LLM/embedding caching (#10); Streamlit cache-key fix (#11); dependency pinning (#16); `app.py` cosmetic cleanup (#14); remove dead state fields (#1) | Low risk, high value — regression-gated by `python main.py --mock` and `streamlit run app.py` (mock mode) both working post-move |
| **2** | Close the follow-up governance gap: `governed_ask()` (#9); retain draft answer on escalation + UI expander (#12) | **Highest-priority reliability fix** |
| **3** | Router fallback flagging + logging (#2); reflection parse-failure detection (#3); shared `prompts.py` markers (#4); centralize `TODAY` (#5) | Robustness + observability |
| **4** | PDF ingestion via `pypdf` (#7); extract `rag/chunking.py` with `RecursiveCharacterTextSplitter`; re-run ingestion, verify existing sample queries still work | Ingestion upgrade, no vector-store format change |
| **5** | pytest suite (#13); SQLite-backed follow-up memory (#8); final `.gitignore` additions; README updates | Tests + persistence, lowest risk since fully additive |

---

## 10. Critical files (most central to future changes)

- `followup_agent.py` → `agents/followup_agent.py` — site of fix #9 (`governed_ask()`), the most important change
- `app.py` — mock-flag threading (#11), follow-up call-site swap (#9), escalation draft-answer UI (#12), cosmetic cleanup (#14)
- `llm.py` → `llm/factory.py` — explicit mock parameter + caching (#10, #11), dependency root for `config.py`
- `ingest.py` → `rag/ingest.py` — PDF loading + delegation to new `chunking.py` (#7)
- `reflection_agent.py` → `agents/reflection_agent.py` — robust parsing with parse-failure detection (#3), prompt moves to `prompts.py` (#4)
- `router_agent.py` → `agents/router_agent.py` — fallback flagging/logging (#2)
- `state.py` — dead-field removal (#1), new `router_fallback` field
- `requirements.txt` — full pinning (#16) + new deps (`pypdf`, `langgraph-checkpoint-sqlite`, `langchain-text-splitters`, `pytest`)
