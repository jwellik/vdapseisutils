"""Batch 4 Part C: maps defaults_constants has no matplotlib; rc applies on register."""

import ast
from pathlib import Path

import pytest


def test_defaults_constants_has_no_matplotlib_imports():
    import vdapseisutils.core.maps.defaults_constants as dc

    tree = ast.parse(Path(dc.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("matplotlib")
        elif isinstance(node, ast.ImportFrom) and node.module:
            assert not node.module.startswith("matplotlib")
    assert dc.default_volcano["name"]


def test_register_maps_mpl_style_updates_rcparams():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import vdapseisutils.core.maps.defaults as md

    md.register_maps_mpl_style()
    assert matplotlib.rcParams["svg.fonttype"] == "none"
    assert float(matplotlib.rcParams["font.size"]) == pytest.approx(8.0)


def test_ensure_maps_mpl_style_runs_once():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import vdapseisutils.core.maps.defaults as md

    md.reset_maps_mpl_style_registration_for_tests()
    assert md._maps_mpl_style_registered is False
    md.ensure_maps_mpl_style()
    assert md._maps_mpl_style_registered is True
    md.ensure_maps_mpl_style()
    assert md._maps_mpl_style_registered is True
