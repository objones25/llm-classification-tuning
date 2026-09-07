from __future__ import annotations

from pathlib import Path

import kagglehub


def download_competition_data(competition: str = "llm-classification-finetuning") -> Path:
    return Path(kagglehub.competition_download(competition))
