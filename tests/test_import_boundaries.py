"""The reliability and reporting layers must import with no heavy dependencies.

This is an architectural constraint, not a nicety. The whole selective-prediction and
reporting stack is specified to run on a laptop with no GPU so that routing analysis,
metric computation and probe verification stay reproducible without Colab. A stray
top-level ``import torch`` anywhere in those packages would silently break that, and the
break would only surface on a machine without torch - which is to say, on a reviewer's.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"

HEAVY = ("torch", "transformers", "rank_bm25", "chromadb", "sentence_transformers")

LIGHT_PACKAGES = (
    "aaoifi_rag.reliability",
    "aaoifi_rag.reporting",
    "aaoifi_rag.orchestration",
)


@pytest.mark.parametrize("package", LIGHT_PACKAGES)
def test_package_imports_without_heavy_dependencies(package: str) -> None:
    """Import in a fresh interpreter and assert no heavy module was pulled in.

    A subprocess is used deliberately: by the time this test runs, the pytest session may
    already have imported torch for some other reason, so checking ``sys.modules`` in
    process would be vacuous.
    """
    code = (
        "import sys, json;"
        f"import {package};"
        f"print(json.dumps([m for m in sys.modules if m.split('.')[0] in {HEAVY!r}]))"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        env=env,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr
    leaked = json.loads(result.stdout.strip().splitlines()[-1])
    assert leaked == [], f"{package} pulled in heavy modules: {leaked}"


def test_all_exports_resolve() -> None:
    """Every name in ``__all__`` must exist, and must appear once.

    A stale export is an ``ImportError`` waiting for the first ``from ... import *`` or
    documentation build; a duplicated one is the usual residue of a merge. Ordering is
    deliberately not asserted - ``reliability`` sorts plainly while ``reporting`` groups
    constants before classes before functions, and forcing one convention on the other
    would be a cosmetic rewrite with no defect behind it.
    """
    import importlib

    for package in LIGHT_PACKAGES:
        module = importlib.import_module(package)
        exported = list(getattr(module, "__all__", []))
        assert exported, f"{package} declares no __all__"
        missing = [name for name in exported if not hasattr(module, name)]
        assert missing == [], f"{package}.__all__ names missing attributes: {missing}"
        duplicates = sorted({n for n in exported if exported.count(n) > 1})
        assert duplicates == [], f"{package}.__all__ lists duplicates: {duplicates}"
