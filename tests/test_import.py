"""ISSUE-0001: package scaffold import / version sanity checks."""
from __future__ import annotations


def test_import_top_level():
    import mahjong_agent  # noqa: F401


def test_version_attribute():
    import mahjong_agent

    assert isinstance(mahjong_agent.__version__, str)
    assert mahjong_agent.__version__


def test_subpackages_importable():
    """全 subpackage が import できる (skeleton であることの確認)。"""
    import importlib

    subpackages = (
        "mahjong_agent.envs",
        "mahjong_agent.actions",
        "mahjong_agent.encoders",
        "mahjong_agent.models",
        "mahjong_agent.training",
        "mahjong_agent.evaluation",
        "mahjong_agent.agents",
        "mahjong_agent.bot",
        "mahjong_agent.diagnostics",
        "mahjong_agent.utils",
    )
    for name in subpackages:
        importlib.import_module(name)
