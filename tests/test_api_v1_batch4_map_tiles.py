"""Batch 4 Part D: map tiles must not call urllib.request.install_opener."""

import urllib.request

import pytest


def test_create_tile_with_ssl_context_does_not_install_global_opener():
    pytest.importorskip("cartopy")
    pytest.importorskip("PIL")

    from vdapseisutils.core.maps import map_tiles

    opener_before = urllib.request._opener
    tile = map_tiles._create_tile_with_ssl_context(
        "https://example.invalid/tiles/{z}/{x}/{y}.png",
        cache=False,
        ssl_verify=True,
    )
    assert tile is not None
    assert urllib.request._opener is opener_before
