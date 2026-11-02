from pathlib import Path

from boardwatch.constants import _resolve_project_root


def test_project_root_uses_source_checkout_root(tmp_path: Path):
    project = tmp_path / "source-project"
    (project / "src" / "boardwatch").mkdir(parents=True)
    (project / "pyproject.toml").write_text("[project]\nname='boardwatch'\n", encoding="utf-8")
    module = project / "src" / "boardwatch" / "constants.py"

    assert _resolve_project_root(module, tmp_path / "elsewhere") == project.resolve()


def test_project_root_uses_working_directory_for_installed_wheel(tmp_path: Path):
    module = tmp_path / "venv" / "lib" / "python3.12" / "site-packages" / "boardwatch" / "constants.py"
    working_directory = tmp_path / "run"
    working_directory.mkdir()

    assert _resolve_project_root(module, working_directory) == working_directory.resolve()


def test_explicit_boardwatch_home_overrides_other_roots(tmp_path: Path):
    module = tmp_path / "source" / "src" / "boardwatch" / "constants.py"
    override = tmp_path / "configured-home"

    assert _resolve_project_root(module, tmp_path, str(override)) == override.resolve()
