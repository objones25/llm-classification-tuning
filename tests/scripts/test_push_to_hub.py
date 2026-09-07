import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import pytest
import torch
from huggingface_hub import CommitInfo
from peft import LoraConfig as PeftLoraConfig
from peft import PeftModel, get_peft_model
from torch import nn
from transformers import AutoModelForSequenceClassification, PreTrainedModel, Qwen2Config

from llm_reward.models._hf_common import LogitsOnly
from llm_reward.models.checkpoint import Checkpoint
from llm_reward.models.config import TrainConfig
from llm_reward.models.registry import ModelBundle, register
from llm_reward.negative_space import CheckFailed
from scripts.push_to_hub import push_to_hub


def _commit_info(repo_id: str) -> CommitInfo:
    return CommitInfo(
        commit_url=f"https://huggingface.co/{repo_id}/commit/deadbeef",
        commit_message="upload",
        commit_description="",
        oid="deadbeef",
    )


class _FakeApi:
    def __init__(self) -> None:
        self.created_repos = []
        self.uploaded_folders = []
        self.uploaded_files = []

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
        return _commit_info(repo_id)

    def upload_file(self, path_or_fileobj, path_in_repo, repo_id):
        self.uploaded_files.append((repo_id, path_in_repo, bytes(path_or_fileobj).decode()))
        return _commit_info(repo_id)


def _tiny_qwen2_model():
    cfg = Qwen2Config(
        vocab_size=1000, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=64, pad_token_id=0,
    )
    cfg.num_labels = 3
    return AutoModelForSequenceClassification.from_config(cfg)


def _tiny_peft_model():
    return get_peft_model(
        _tiny_qwen2_model(),
        PeftLoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"], task_type="SEQ_CLS"),
    )


@dataclass(frozen=True, kw_only=True)
class _PushTestConfig(TrainConfig):
    # Defined at module scope (not nested in the test function) so torch.save can pickle it by
    # qualified name — a locally-defined class raises AttributeError when pickled.
    variant: ClassVar[str] = "test_push_variant"


@dataclass(frozen=True, kw_only=True)
class _PushHFTestConfig(TrainConfig):
    """A stand-in for the SFT/LoRA configs that exercises the HF dispatch branch without the
    real `SFTHeadConfig`/`LoRAConfig` — which would make `push_to_hub` reach for a real
    tokenizer over the network."""

    variant: ClassVar[str] = "test_push_hf_variant"


def _write_checkpoint(tmp_path: Path, config: TrainConfig, model_state: dict) -> Path:
    checkpoint = Checkpoint(
        epoch=0, global_step=0, model_state=model_state, optimizer_state={},
        scheduler_state=None, best_val_metric=0.9, config=config, wandb_run_id="r",
    )
    checkpoint_path = tmp_path / "best.pt"
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def test_push_to_hub_uploads_a_plain_module_via_upload_folder(tmp_path, monkeypatch):
    @register("test_push_variant")
    def _build(config):
        return ModelBundle(model=nn.Linear(4, 3), collate_fn=lambda batch: {})

    config = _PushTestConfig(
        seed=1, batch_size=2, epochs=1, lr=1e-2, output_dir=tmp_path, run_name="r"
    )
    checkpoint_path = _write_checkpoint(tmp_path, config, nn.Linear(4, 3).state_dict())

    fake_api = _FakeApi()
    import scripts.push_to_hub as push_module

    monkeypatch.setattr(push_module, "HfApi", lambda: fake_api)

    result = push_to_hub(checkpoint_path, repo_id="me/test-model", private=True)

    assert fake_api.created_repos == [("me/test-model", True)]
    assert len(fake_api.uploaded_folders) == 1
    repo_id, folder_path = fake_api.uploaded_folders[0]
    assert repo_id == "me/test-model"
    assert (folder_path / "model_state_dict.pt").exists()

    assert len(fake_api.uploaded_files) == 1
    card_repo_id, path_in_repo, card = fake_api.uploaded_files[0]
    assert (card_repo_id, path_in_repo) == ("me/test-model", "README.md")
    assert "0.9000" in card

    assert result == "https://huggingface.co/me/test-model/commit/deadbeef"


@pytest.mark.parametrize(
    ("make_model", "push_owner"),
    [(_tiny_qwen2_model, PreTrainedModel), (_tiny_peft_model, PeftModel)],
    ids=["pretrained_model", "peft_model"],
)
def test_push_to_hub_uploads_a_model_card_for_every_branch(
    make_model, push_owner, tmp_path, monkeypatch
):
    """The HF branches used to `return` before any README was written, leaving those two
    variants with HF's generic auto-card: no hyperparameters, no best val metric."""
    pushed = []

    def _fake_push(self, repo_id, **kwargs):
        pushed.append(repo_id)
        return _commit_info(repo_id)

    monkeypatch.setattr(push_owner, "push_to_hub", _fake_push)

    @register("test_push_hf_variant")
    def _build(config):
        return ModelBundle(model=LogitsOnly(make_model()), collate_fn=lambda batch: {})

    config = _PushHFTestConfig(
        seed=1, batch_size=2, epochs=1, lr=1e-2, output_dir=tmp_path, run_name="hf-run"
    )
    checkpoint_path = _write_checkpoint(
        tmp_path, config, LogitsOnly(make_model()).state_dict()
    )

    fake_api = _FakeApi()
    import scripts.push_to_hub as push_module

    monkeypatch.setattr(push_module, "HfApi", lambda: fake_api)

    result = push_to_hub(checkpoint_path, repo_id="me/hf-model", private=True)

    assert pushed == ["me/hf-model"]
    assert fake_api.uploaded_folders == []
    assert len(fake_api.uploaded_files) == 1
    repo_id, path_in_repo, card = fake_api.uploaded_files[0]
    assert (repo_id, path_in_repo) == ("me/hf-model", "README.md")
    assert "hf-run" in card
    assert "0.9000" in card
    assert "test_push_hf_variant" in card
    assert result == "https://huggingface.co/me/hf-model/commit/deadbeef"


def test_push_to_hub_raises_on_missing_checkpoint(tmp_path):
    with pytest.raises(CheckFailed, match="not found"):
        push_to_hub(tmp_path / "does_not_exist.pt", repo_id="me/x")
