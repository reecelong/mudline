"""Shared pytest fixtures for all Mudline tests."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

# Use a writable temp directory for fixtures since the workspace mount
# may not support all filesystem operations (e.g., SQLite journaling).
_WRITABLE_FIXTURE_DIR = Path(tempfile.gettempdir()) / "mudline_test_fixtures" / "backup"
_FIXTURE_SCRIPT = Path(__file__).parent / "fixtures" / "create_fixture.py"


@pytest.fixture(scope="session", autouse=True)
def ensure_test_fixture() -> Path:
    """Ensure the synthetic backup fixture exists before running tests."""
    manifest_db = _WRITABLE_FIXTURE_DIR / "Manifest.db"

    if not manifest_db.exists() or manifest_db.stat().st_size == 0:
        # Generate fixture in writable temp dir by patching FIXTURE_DIR
        # via a wrapper script to avoid mount filesystem limitations.
        wrapper = (
            f"import sys; sys.path.insert(0, '.')\n"
            f"from pathlib import Path\n"
            f"import tests.fixtures.create_fixture as cf\n"
            f"cf.FIXTURE_DIR = Path('{_WRITABLE_FIXTURE_DIR}')\n"
            f"cf.main()\n"
        )
        subprocess.run(
            [sys.executable, "-c", wrapper],
            check=True,
            cwd=str(Path(__file__).parent.parent),
        )

    return _WRITABLE_FIXTURE_DIR


@pytest.fixture
def backup_path(ensure_test_fixture: Path) -> Path:
    """Path to the synthetic test backup."""
    return ensure_test_fixture
