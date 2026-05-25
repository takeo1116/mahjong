"""Build glue for the optional ``mahjong_agent._mahjong_fast`` C++ extension.

C++ fast path (shanten / ukeire / discard analysis) を pybind11 でビルドする。
コンパイラや pybind11 が無い環境でも編集インストールが壊れないよう、ext
ビルドに失敗しても package 本体は import 可能 (Python fallback が動く)。

`pyproject.toml` の build-system requires に pybind11 を入れてあるため、
通常の ``pip install -e .`` でこの ext がビルドされる。失敗時の挙動は
``optional_build_ext`` 参照。
"""
from __future__ import annotations

from setuptools import setup
from setuptools.command.build_ext import build_ext as _build_ext

try:
    from pybind11.setup_helpers import Pybind11Extension

    ext_modules = [
        Pybind11Extension(
            "mahjong_agent._mahjong_fast",
            ["src/mahjong_agent/_mahjong_fast.cpp"],
            cxx_std=17,
        ),
    ]
except Exception:  # pragma: no cover - pybind11 が無い環境
    ext_modules = []


class optional_build_ext(_build_ext):
    """C++ ext のビルド失敗を fatal にしない (Python fallback で動くため)。"""

    def run(self) -> None:
        try:
            super().run()
        except Exception as exc:  # pragma: no cover
            import warnings

            warnings.warn(
                f"mahjong_agent._mahjong_fast の C++ ビルドに失敗しました "
                f"(Python fallback を使用します): {exc}",
                stacklevel=2,
            )

    def build_extension(self, ext) -> None:
        try:
            super().build_extension(ext)
        except Exception as exc:  # pragma: no cover
            import warnings

            warnings.warn(
                f"extension {ext.name} のビルドに失敗しました "
                f"(Python fallback を使用します): {exc}",
                stacklevel=2,
            )


setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": optional_build_ext},
)
