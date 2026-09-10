from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import torch
from huggingface_hub import HfApi
from transformers import AutoTokenizer, PreTrainedModel

from llm_reward.models.checkpoint import Checkpoint, ConfigMismatchError
from llm_reward.models.config import ConfigError, LoRAConfig, SFTHeadConfig
from llm_reward.models.registry import build_model, load_model_state_dict
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
        f"Best validation loss: {checkpoint.best_val_metric:.4f}\n\n"
        f"## Training configuration\n\n{config_lines}\n"
    )


def push_to_hub(checkpoint_path: Path, repo_id: str, private: bool = True) -> str:
    """Push a reviewed checkpoint to the Hub. Returns the commit URL of the model upload."""
    require(checkpoint_path.exists(), f"checkpoint not found: {checkpoint_path}")
    checkpoint: Checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")

    bundle = build_model(checkpoint.config)
    load_model_state_dict(bundle, checkpoint.model_state)
    real_model = getattr(bundle.model, "hf_model", bundle.model)

    api = HfApi()
    api.create_repo(repo_id, private=private, exist_ok=True)

    if isinstance(real_model, (PeftModel, PreTrainedModel)):
        # transformers/peft both return a huggingface_hub CommitInfo here, despite the `-> str`
        # annotation on PushToHubMixin.push_to_hub (CommitInfo subclasses str for backwards
        # compatibility, but reading it as a string is deprecated — go through .commit_url).
        # `real_model`'s static type collapses to nn.Module here (the `getattr(..., default)`
        # fallback and the PeftModel-or-() try/except import both defeat Pyright's isinstance
        # narrowing), so it misreads push_to_hub via nn.Module.__getattr__'s stubbed `Tensor |
        # Module` return type instead of the real PushToHubMixin method.
        result = real_model.push_to_hub(repo_id, private=private)  # pyright: ignore[reportCallIssue]
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            torch.save(real_model.state_dict(), tmp_path / "model_state_dict.pt")
            result = api.upload_folder(repo_id=repo_id, folder_path=str(tmp_path))

    if isinstance(checkpoint.config, (SFTHeadConfig, LoRAConfig)):
        tokenizer = AutoTokenizer.from_pretrained(checkpoint.config.hf_model_name)
        tokenizer.push_to_hub(repo_id, private=private)

    # Unconditional and LAST: every variant gets the same generated card, and both
    # `push_to_hub` calls above write their own auto-generated README.md that this must
    # overwrite. Without it the HF-backed variants would ship a generic card carrying neither
    # the hyperparameters nor the best val metric (spec Section E).
    api.upload_file(
        path_or_fileobj=_model_card_text(checkpoint).encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
    )

    return result.commit_url


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Push a reviewed checkpoint to the Hugging Face Hub"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--repo-id", type=str, required=True)
    parser.add_argument(
        "--public", action="store_true", help="Push to a public repo (default: private)"
    )
    args = parser.parse_args()
    try:
        url = push_to_hub(args.checkpoint, args.repo_id, private=not args.public)
    except (ConfigError, ConfigMismatchError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    print(url)


if __name__ == "__main__":
    main()
