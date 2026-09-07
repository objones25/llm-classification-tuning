from pathlib import Path

from llm_reward.train import main


class _FakeRun:
    def __init__(self, logged: list[dict]) -> None:
        self.id = "fake"
        self._logged = logged

    def log(self, data: dict) -> None:
        self._logged.append(data)

    def define_metric(self, *args, **kwargs) -> None:
        pass

    def __enter__(self) -> "_FakeRun":
        return self

    def __exit__(self, *exc_info: object) -> None:
        pass


class _FakeWandb:
    def __init__(self) -> None:
        self.logged: list[dict] = []

    def init(self, **kwargs):  # noqa: ARG002
        self.run = _FakeRun(self.logged)
        return self.run

    def Table(self, *, columns, data):
        return {"columns": columns, "data": data}


def _write_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "config.yaml"
    output_dir = tmp_path / "output"
    config_path.write_text(
        "variant: lstm_baseline\n"
        "seed: 1\nbatch_size: 2\nepochs: 1\nlr: 0.01\n"
        f"output_dir: {output_dir}\nrun_name: cli-test\n"
        "max_seq_len: 8\nvocab_size: 100\n"
    )
    return config_path


def _write_train_csv(tmp_path: Path) -> Path:
    csv_path = tmp_path / "train.csv"
    lines = [
        "id,model_a,model_b,prompt,response_a,response_b,"
        "winner_model_a,winner_model_b,winner_model_tie"
    ]
    for i in range(10):
        label_cols = ["0", "0", "0"]
        label_cols[i % 3] = "1"
        lines.append(f'{i},m,n,"prompt {i}","resp a {i}","resp b {i}",{",".join(label_cols)}')
    csv_path.write_text("\n".join(lines) + "\n")
    return csv_path


def test_main_runs_end_to_end_and_writes_checkpoints(tmp_path, monkeypatch):
    import llm_reward.train as train_module

    monkeypatch.setattr(train_module, "wandb", _FakeWandb())
    config_path = _write_config(tmp_path)
    csv_path = _write_train_csv(tmp_path)

    monkeypatch.setattr(
        "sys.argv",
        ["train", "--config", str(config_path), "--train-csv", str(csv_path)],
    )
    main()

    output_dir = tmp_path / "output"
    assert (output_dir / "last.pt").exists()
    assert (output_dir / "best.pt").exists()
