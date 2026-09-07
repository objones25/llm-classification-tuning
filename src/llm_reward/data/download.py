from __future__ import annotations

from pathlib import Path

import kagglehub

from ..negative_space import bounded


def download_competition_data(
    competition: str = "llm-classification-finetuning", max_retries: int = 3
) -> Path:
    """Download the competition archive, retrying a bounded number of times.

    A download is an operating error when it fails for good (network down, bad credentials), so
    the final failure is a typed exception rather than an assertion. The retry count is bounded
    per CLAUDE.md's "max retries on a Kaggle/HF download" rule — there is no unbounded loop here.
    """
    last_error: Exception | None = None
    for _ in bounded(range(max_retries), limit=max_retries, name="kagglehub download attempts"):
        try:
            return Path(kagglehub.competition_download(competition))
        # Deliberately broad: kagglehub's failure modes (BackendError, KaggleApiHTTPError,
        # DataCorruptionError, ...) share no base class narrower than Exception. Nothing is
        # swallowed -- the last error is re-raised as the cause below if every attempt fails.
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"kagglehub download failed after {max_retries} attempts") from last_error
