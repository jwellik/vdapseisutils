"""Pytest configuration: Agg backend and stable import path for the real package."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
import pytest

matplotlib.use("Agg")

# Project root: directory that contains the inner ``vdapseisutils/`` package tree.
_ROOT = Path(__file__).resolve().parents[1]
_ROOT_S = str(_ROOT)
_INNER_INIT = (_ROOT / "vdapseisutils" / "__init__.py").resolve()


def _ensure_project_root_first_on_path() -> None:
    """Pytest may prepend ``pythonpath`` parent dirs later; keep the repo root first."""
    while _ROOT_S in sys.path:
        sys.path.remove(_ROOT_S)
    sys.path.insert(0, _ROOT_S)


def _purge_stale_vdapseisutils_modules() -> None:
    """If the repo-root legacy ``__init__.py`` was imported, drop cached submodules."""
    mod = sys.modules.get("vdapseisutils")
    if mod is None or not getattr(mod, "__file__", None):
        return
    try:
        loaded = Path(mod.__file__).resolve()
    except OSError:
        return
    if loaded == _INNER_INIT:
        return
    for key in list(sys.modules):
        if key == "vdapseisutils" or key.startswith("vdapseisutils."):
            del sys.modules[key]


def _fix_import_context() -> None:
    _ensure_project_root_first_on_path()
    _purge_stale_vdapseisutils_modules()


_fix_import_context()


@pytest.fixture(autouse=True)
def _vdapseisutils_import_path() -> None:
    """Re-apply path + module cache fix after pytest adjusts ``sys.path`` for the test item."""
    _fix_import_context()
    yield
