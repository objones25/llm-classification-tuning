from __future__ import annotations

import random
import socket
import sys
from pathlib import Path

import pytest

from llm_reward.data.pairwise import PairwiseExample

SEED = 20260907


class NetworkBlockedError(RuntimeError):
    """Raised when a unit test tries to open a socket."""


@pytest.fixture(autouse=True)
def _deterministic_rngs() -> None:
    """Reseed every RNG before each test — per-test, not per-session, so a single test run in
    isolation gives the same numbers as a full suite run."""
    random.seed(SEED)
    import numpy as np

    np.random.seed(SEED)
    import torch

    torch.manual_seed(SEED)


@pytest.fixture(autouse=True)
def _block_network(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    if request.node.get_closest_marker("integration"):
        return

    def guard(*args: object, **kwargs: object):
        raise NetworkBlockedError(
            "This test attempted a network connection. Stub the boundary with "
            "monkeypatch, or mark the test @pytest.mark.integration."
        )

    monkeypatch.setattr(socket.socket, "connect", guard)
    monkeypatch.setattr(socket, "create_connection", guard)
    monkeypatch.setattr(socket, "getaddrinfo", guard)


@pytest.fixture(autouse=True)
def _isolated_cwd(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTHONHASHSEED", str(SEED))


@pytest.fixture(autouse=True)
def _restore_model_registry():
    """Undo any `@register(...)` a test performed. Several tests register throwaway fake
    variants; without this they leak into every later test in the session and break outright
    under a re-run/re-collection plugin (`register` rejects a duplicate name)."""
    from llm_reward.models.registry import _REGISTRY

    snapshot = dict(_REGISTRY)
    yield
    _REGISTRY.clear()
    _REGISTRY.update(snapshot)


@pytest.fixture
def tiny_pairwise_examples() -> list[PairwiseExample]:
    """Six hand-built records, two per label — the one shared fixture every track's tests build
    from, per CLAUDE.md's testing convention. Never hand-roll a second copy of this."""
    return [
        PairwiseExample(id="1", prompt="What is 2+2?", response_a="4", response_b="Four", label=0),
        PairwiseExample(
            id="2", prompt="Capital of France?",
            response_a="Paris is the capital.", response_b="I don't know.", label=0,
        ),
        PairwiseExample(
            id="3", prompt="Say hi", response_a="hi", response_b="Hello there!", label=1,
        ),
        PairwiseExample(
            id="4", prompt="Explain gravity",
            response_a="Things fall down.", response_b="Mass attracts mass.", label=1,
        ),
        PairwiseExample(id="5", prompt="Pick a number", response_a="7", response_b="7", label=2),
        PairwiseExample(
            id="6", prompt="Tell a joke",
            response_a="Why did the chicken cross the road?",
            response_b="Why did the chicken cross the road?", label=2,
        ),
    ]


def pytest_configure(config: pytest.Config) -> None:
    for line in (
        "slow: takes more than ~1s; deselected by default",
        "integration: crosses a process, service, or network boundary",
        "gpu: requires CUDA; skipped when unavailable",
    ):
        config.addinivalue_line("markers", line)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for item in items:
        parts = {p.name for p in Path(str(item.fspath)).parents}
        if "integration" in parts:
            item.add_marker(pytest.mark.integration)


def pytest_report_header(config: pytest.Config) -> list[str]:
    return [f"llm-reward tests: seed={SEED} python={sys.version.split()[0]}"]
