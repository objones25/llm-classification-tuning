import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import pytest
import torch
from torch import nn

from llm_reward.models.checkpoint import Checkpoint
from llm_reward.models.config import TrainConfig
from llm_reward.models.registry import ModelBundle, register
from llm_reward.negative_space import CheckFailed
from scripts.push_to_hub import push_to_hub


class _FakeApi:
    def __init__(self) -> None:
        self.created_repos = []
        self.uploaded_folders = []

    def create_repo(self, repo_id, private=False, exist_ok=True):
        self.created_repos.append((repo_id, private))

    def upload_folder(self, repo_id, folder_path):
        # Real HfApi.upload_folder reads and uploads the file contents synchronously during
        # this call, before push_to_hub's `with tempfile.TemporaryDirectory()` block exits and
        # deletes the directory. This fake must likewise snapshot the contents now — the
        # original folder_path won't exist once push_to_hub returns.
        snapshot = Path(tempfile.mkdtemp()) / "snapshot"
        shutil.copytree(folder_path, snapshot)
        self.uploaded_folders.append((repo_id, snapshot))
        return f"https://huggingface.co/{repo_id}"


@dataclass(frozen=True, kw_only=True)
class _PushTestConfig(TrainConfig):
    # Defined at module scope (not nested in the test function) so torch.save can pickle it by
    # qualified name — a locally-defined class raises AttributeError when pickled.
    variant: ClassVar[str] = "test_push_variant"


def test_push_to_hub_uploads_a_plain_module_via_upload_folder(tmp_path, monkeypatch):
    @register("test_push_variant")
    def _build(config):
        return ModelBundle(model=nn.Linear(4, 3), collate_fn=lambda batch: {})

    config = _PushTestConfig(seed=1, batch_size=2, epochs=1, lr=1e-2, output_dir=tmp_path, run_name="r")
    checkpoint = Checkpoint(
        epoch=0, global_step=0, model_state=nn.Linear(4, 3).state_dict(), optimizer_state={},
        scheduler_state=None, best_val_metric=0.9, config=config, wandb_run_id="r",
    )
    checkpoint_path = tmp_path / "best.pt"
    torch.save(checkpoint, checkpoint_path)

    fake_api = _FakeApi()
    import scripts.push_to_hub as push_module

    monkeypatch.setattr(push_module, "HfApi", lambda: fake_api)

    result = push_to_hub(checkpoint_path, repo_id="me/test-model", private=True)

    assert fake_api.created_repos == [("me/test-model", True)]
    assert len(fake_api.uploaded_folders) == 1
    repo_id, folder_path = fake_api.uploaded_folders[0]
    assert repo_id == "me/test-model"
    assert (folder_path / "model_state_dict.pt").exists()
    assert (folder_path / "README.md").exists()
    assert "0.9" in (folder_path / "README.md").read_text()
    assert result == "https://huggingface.co/me/test-model"


def test_push_to_hub_raises_on_missing_checkpoint(tmp_path):
    with pytest.raises(CheckFailed, match="not found"):
        push_to_hub(tmp_path / "does_not_exist.pt", repo_id="me/x")
