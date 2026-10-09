from datetime import date

from campus_assistant.config import Settings, settings


def test_paths_resolve_under_project_root_and_exist():
    # vector_store_dir is overridden by conftest.py to an isolated temp
    # directory for tests (see its docstring), so only kb_dir is checked
    # against project_root here; vector_store_dir's existence (built by
    # conftest's build_all_indexes() call) is still verified.
    assert settings.kb_dir.exists()
    assert settings.vector_store_dir.exists()
    assert settings.project_root in settings.kb_dir.parents


def test_mock_mode_enabled_by_conftest():
    assert settings.mock_mode is True


def test_today_override_env_var(monkeypatch):
    monkeypatch.setenv("CAMPUS_ASSISTANT_TODAY", "2027-01-15")
    overridden = Settings()
    assert overridden.today == date(2027, 1, 15)


def test_today_defaults_to_override_when_set():
    assert settings.today == settings.today_override
