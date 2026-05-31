"""Engine-isolation smoke test.

Proves the domain-agnostic engine (`models` + `index` + `intelligence.llm`)
imports cleanly without the iOS or media extras installed. This is the guarantee
that lets a downstream application depend on the engine subset alone. The
companion static guarantee is the ``import-linter`` contract in ``pyproject.toml``
(run ``lint-imports``).
"""

from __future__ import annotations

import subprocess
import sys
import textwrap


def test_engine_imports_without_ios_or_media_extras() -> None:
    """Importing the engine must not require iOSbackup / Pillow / open_clip.

    Runs in a subprocess so the import-blocking does not pollute the test
    process's module cache.
    """
    code = textwrap.dedent(
        """
        import sys

        # Simulate the iOS/media-only deps being absent: any attempt to import
        # one raises ImportError, so this run fails if the engine needs them.
        for name in ("iOSbackup", "PIL", "PIL.Image", "open_clip"):
            sys.modules[name] = None

        import mudline.models  # noqa: F401
        import mudline.index.structured  # noqa: F401
        import mudline.index.vector  # noqa: F401
        import mudline.index.retriever  # noqa: F401
        import mudline.index.ingest  # noqa: F401
        import mudline.index.contacts  # noqa: F401
        import mudline.index.media  # noqa: F401
        from mudline.intelligence.llm import create_provider  # noqa: F401

        # Confirm the engine did not pull in any iOS/media-only module.
        leaked = [n for n in ("iOSbackup", "open_clip") if sys.modules.get(n) is not None]
        assert not leaked, f"engine imported iOS/media deps: {leaked}"
        print("ENGINE_ISOLATION_OK")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"engine failed to import without iOS/media extras:\n{result.stderr}"
    )
    assert "ENGINE_ISOLATION_OK" in result.stdout
