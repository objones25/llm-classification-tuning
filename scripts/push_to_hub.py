from __future__ import annotations

import tempfile
from pathlib import Path

import torch
from huggingface_hub import HfApi
from transformers import AutoTokenizer, PreTrainedModel

from llm_reward.models.checkpoint import Checkpoint
from llm_reward.models.config import LoRAConfig, SFTHeadConfig
from llm_reward.models.registry import build_model
from llm_reward.negative_space import require

try:
    from peft import PeftModel
except ImportError:  # pragma: no cover — peft is a project dependency, always installed
    PeftModel = ()  # type: ignore[assignment]


def _model_card_text(checkpoint: Checkpoint) -> str:
    config_lines = "\n".join(
        f"- **{key}**: {value}" for key, value in vars(checkpoint.config).items()
    )
    return (
        f"# {checkpoint.config.run_name}\n\n"
        f"Variant: `{checkpoint.config.variant}`\n\n"
        f"Best validation accuracy: {checkpoint.best_val_metric:.4f}\n\n"
        f"## Training configuration\n\n{config_lines}\n"
    )


def push_to_hub(checkpoint_path: Path, repo_id: str, private: bool = True) -> str:
    require(checkpoint_path.exists(), f"checkpoint not found: {checkpoint_path}")
    checkpoint: Checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")

    bundle = build_model(checkpoint.config)
    bundle.model.load_state_dict(checkpoint.model_state)
    real_model = getattr(bundle.model, "hf_model", bundle.model)

    api = HfApi()
    api.create_repo(repo_id, private=private, exist_ok=True)

    if isinstance(real_model, (PeftModel, PreTrainedModel)):
        result = real_model.push_to_hub(repo_id, private=private)
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            torch.save(real_model.state_dict(), tmp_path / "model_state_dict.pt")
            (tmp_path / "README.md").write_text(_model_card_text(checkpoint))
            result = api.upload_folder(repo_id=repo_id, folder_path=str(tmp_path))

    if isinstance(checkpoint.config, (SFTHeadConfig, LoRAConfig)):
        tokenizer = AutoTokenizer.from_pretrained(checkpoint.config.hf_model_name)
        tokenizer.push_to_hub(repo_id, private=private)

    return str(result)
