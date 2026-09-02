"""Tests for utility decorators."""

# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#
# License: BSD 3-Clause License

import pytest
import torch

from torchdr.utils import compile_if_requested


def test_compile_if_requested_caches_per_instance(monkeypatch):
    """Each live instance should reuse its compiled callable."""
    compiled_functions = []

    def fake_compile(func):
        compiled_functions.append(func)
        return func

    monkeypatch.setattr(torch, "compile", fake_compile)

    class Example:
        compile = True

        @compile_if_requested
        def run(self, value):
            return value + 1

    first = Example()
    assert first.run(1) == 2
    assert first.run(2) == 3
    assert len(compiled_functions) == 1

    second = Example()
    assert second.run(3) == 4
    assert len(compiled_functions) == 2


def test_compile_if_requested_falls_back_on_first_execution(monkeypatch):
    """Lazy compilation failures should fall back and stay eager."""
    compile_calls = 0

    def fake_compile(func):
        nonlocal compile_calls
        compile_calls += 1

        def compiled(*args, **kwargs):
            raise RuntimeError("backend compilation failed")

        return compiled

    monkeypatch.setattr(torch, "compile", fake_compile)

    class Example:
        compile = True

        @compile_if_requested
        def run(self, value):
            return value + 1

    example = Example()
    with pytest.warns(UserWarning, match="Falling back to eager execution"):
        assert example.run(1) == 2

    assert example.run(2) == 3
    assert compile_calls == 1
