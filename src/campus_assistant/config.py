"""
Central settings for CampusAssist.

Replaces scattered `os.environ.get(...)` reads and hardcoded relative
paths/constants (previously duplicated across llm.py, ingest.py,
retriever.py, report.py, tools.py, app.py) with one source of truth.

All paths are derived from this file's own location, not the process's
current working directory, so `python main.py`, `streamlit run app.py`,
and `python scripts/ingest.py` behave identically regardless of the
shell's CWD at invocation time.
"""

from datetime import date
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# src/campus_assistant/config.py -> parents[2] is the project root
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CAMPUS_ASSISTANT_", env_file=".env", extra="ignore")

    # Explicit validation_alias on this field and today_override below:
    # pydantic-settings derives env var names from the field name itself
    # (env_prefix + FIELD_NAME.upper()), which would otherwise require
    # CAMPUS_ASSISTANT_MOCK_MODE / CAMPUS_ASSISTANT_TODAY_OVERRIDE instead
    # of the shorter CAMPUS_ASSISTANT_MOCK / CAMPUS_ASSISTANT_TODAY.
    mock_mode: bool = Field(default=False, validation_alias="CAMPUS_ASSISTANT_MOCK")

    llm_model: str = "qwen2.5:3b"
    embedding_model: str = "nomic-embed-text"
    llm_temperature: float = 0.0

    project_root: Path = _PROJECT_ROOT
    kb_dir: Path = _PROJECT_ROOT / "knowledge_base"
    vector_store_dir: Path = _PROJECT_ROOT / "vector_store"
    reports_dir: Path = _PROJECT_ROOT / "reports"

    chunk_size: int = 500
    chunk_overlap: int = 80

    # Overrides "today" for date_calculator (CAMPUS_ASSISTANT_TODAY=YYYY-MM-DD).
    # Useful for demos/tests whose sample data references a fixed date
    # (see knowledge_base/hostel/hostel_circular_2026.txt and
    # main.py's SAMPLE_QUERIES, both of which assume 2026-09-19).
    today_override: date | None = Field(default=date(2026, 9, 19), validation_alias="CAMPUS_ASSISTANT_TODAY")

    # "memory" (non-persistent, used by tests) or "sqlite" (default for real use)
    checkpointer_backend: str = "sqlite"
    checkpointer_sqlite_path: Path = _PROJECT_ROOT / "followup_sessions.sqlite"

    @property
    def today(self) -> date:
        return self.today_override or date.today()


settings = Settings()
