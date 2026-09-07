from pathlib import Path

import torch

from llm_reward.models.checkpoint import Checkpoint, ConfigMismatchError
from llm_reward.models.config import LSTMConfig


def _config():
    return LSTMConfig(seed=1, batch_size=8, epochs=2, lr=1e-3, output_dir=Path("out"), run_name="r")


def test_checkpoint_round_trips_through_torch_save(tmp_path):
    checkpoint = Checkpoint(
        epoch=0,
        global_step=42,
        model_state={"weight": torch.zeros(2)},
        optimizer_state={},
        scheduler_state=None,
        best_val_metric=0.5,
        epochs_without_improvement=0,
        config=_config(),
        wandb_run_id="run-123",
    )
    path = tmp_path / "last.pt"
    torch.save(checkpoint, path)
    loaded: Checkpoint = torch.load(path, weights_only=False, map_location="cpu")

    assert loaded.epoch == 0
    assert loaded.global_step == 42
    assert loaded.config == _config()
    assert loaded.wandb_run_id == "run-123"
    torch.testing.assert_close(loaded.model_state["weight"], torch.zeros(2))


def test_config_mismatch_error_is_a_plain_exception():
    assert issubclass(ConfigMismatchError, Exception)
    assert not issubclass(ConfigMismatchError, AssertionError)
