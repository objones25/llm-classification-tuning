import pytest

from llm_reward.data.download import download_competition_data


@pytest.mark.integration
def test_real_download_returns_an_existing_directory():
    """Requires KAGGLE_API_TOKEN in the environment (loaded from .env) and accepted competition
    rules on Kaggle. Run explicitly:
    uv run pytest -m integration tests/integration/test_download_real.py"""
    path = download_competition_data()
    assert path.exists()
