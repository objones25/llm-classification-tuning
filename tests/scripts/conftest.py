"""Puts the repo root on sys.path so `scripts/` (a top-level directory of standalone,
non-packaged scripts — see CLAUDE.md — not part of `src/llm_reward`) is importable as
`scripts.push_to_hub` from tests. `pyproject.toml`'s `pythonpath` only lists `src`.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
