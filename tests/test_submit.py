from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import pytest
import torch
from torch import nn

from llm_reward.data.pairwise import PairwiseExample
from llm_reward.models.checkpoint import Checkpoint
from llm_reward.models.config import TrainConfig
from llm_reward.models.registry import ModelBundle, register
from llm_reward.negative_space import CheckFailed
from llm_reward.submit import generate_submission, main

FIXTURES = Path(__file__).parent / "fixtures"
TEST_CSV = FIXTURES / "tiny_test.csv"
SAMPLE_SUBMISSION_CSV = FIXTURES / "tiny_sample_submission.csv"


class _TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 3)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features)


def _collate_fn(batch: list[PairwiseExample]):
    features = torch.randn(len(batch), 4)
    labels = torch.tensor([ex.label for ex in batch], dtype=torch.long)
    return {"features": features, "labels": labels}


@dataclass(frozen=True, kw_only=True)
class _SubmitTestConfig(TrainConfig):
    # Module-scope, not nested in a test function, so torch.save can pickle it by qualified
    # name -- a locally-defined class raises AttributeError when pickled (same reason
    # scripts/test_push_to_hub.py's stand-in configs are module-scope).
    variant: ClassVar[str] = "test_submit_variant"


def _write_checkpoint(tmp_path: Path) -> Path:
    model = _TinyClassifier()
    config = _SubmitTestConfig(
        seed=1, batch_size=2, epochs=1, lr=1e-2, output_dir=tmp_path, run_name="r"
    )
    checkpoint = Checkpoint(
        epoch=0, global_step=0, model_state=model.state_dict(), optimizer_state={},
        scheduler_state=None, best_val_metric=0.9, config=config, wandb_run_id="r",
    )
    checkpoint_path = tmp_path / "best.pt"
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


@pytest.fixture(autouse=True)
def _register_test_variant():
    @register("test_submit_variant")
    def _build(config):
        return ModelBundle(model=_TinyClassifier(), collate_fn=_collate_fn)


def test_generate_submission_writes_header_and_valid_probabilities(tmp_path):
    checkpoint_path = _write_checkpoint(tmp_path)
    output_csv = tmp_path / "submission.csv"

    result = generate_submission(checkpoint_path, TEST_CSV, SAMPLE_SUBMISSION_CSV, output_csv)

    assert result == output_csv
    lines = output_csv.read_text().splitlines()
    assert lines[0] == "id,winner_model_a,winner_model_b,winner_tie"
    assert len(lines) == 4  # header + 3 rows
    for line in lines[1:]:
        probs = [float(v) for v in line.split(",")[1:]]
        assert len(probs) == 3
        assert sum(probs) == pytest.approx(1.0, abs=1e-5)


def test_generate_submission_preserves_sample_submission_id_order(tmp_path):
    checkpoint_path = _write_checkpoint(tmp_path)
    output_csv = tmp_path / "submission.csv"

    generate_submission(checkpoint_path, TEST_CSV, SAMPLE_SUBMISSION_CSV, output_csv)

    lines = output_csv.read_text().splitlines()
    written_ids = [line.split(",")[0] for line in lines[1:]]
    # tiny_sample_submission.csv orders ids 103,101,102 -- different from tiny_test.csv's
    # 101,102,103 -- so this only passes if generate_submission actually reorders.
    assert written_ids == ["103", "101", "102"]


def test_generate_submission_raises_on_id_mismatch(tmp_path):
    checkpoint_path = _write_checkpoint(tmp_path)
    bad_sample_submission = tmp_path / "sample_submission.csv"
    bad_sample_submission.write_text(
        "id,winner_model_a,winner_model_b,winner_tie\n999,0.3,0.3,0.4\n"
    )

    with pytest.raises(CheckFailed, match="ids do not match"):
        generate_submission(
            checkpoint_path, TEST_CSV, bad_sample_submission, tmp_path / "submission.csv"
        )


def test_generate_submission_raises_on_bad_sample_submission_header(tmp_path):
    checkpoint_path = _write_checkpoint(tmp_path)
    bad_sample_submission = tmp_path / "sample_submission.csv"
    bad_sample_submission.write_text("id,winner_model_a\n101,1.0\n")

    with pytest.raises(CheckFailed, match="expected columns"):
        generate_submission(
            checkpoint_path, TEST_CSV, bad_sample_submission, tmp_path / "submission.csv"
        )


def test_generate_submission_raises_on_missing_checkpoint(tmp_path):
    with pytest.raises(CheckFailed, match="not found"):
        generate_submission(
            tmp_path / "does_not_exist.pt",
            TEST_CSV,
            SAMPLE_SUBMISSION_CSV,
            tmp_path / "submission.csv",
        )


def test_main_writes_submission_csv(tmp_path, monkeypatch, capsys):
    checkpoint_path = _write_checkpoint(tmp_path)
    output_csv = tmp_path / "out" / "submission.csv"

    monkeypatch.setattr(
        "sys.argv",
        [
            "submit",
            "--checkpoint", str(checkpoint_path),
            "--test-csv", str(TEST_CSV),
            "--sample-submission", str(SAMPLE_SUBMISSION_CSV),
            "--output", str(output_csv),
        ],
    )
    main()

    assert output_csv.exists()
    assert str(output_csv) in capsys.readouterr().out
