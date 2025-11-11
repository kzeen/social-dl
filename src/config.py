from dataclasses import dataclass

@dataclass
class TrainingConfig:
    model_name: str = "logistic"
    data_dir: str = "data"
    output_dir: str = "outputs"
    max_epochs: int = 5
    batch_size: int = 32
    lr: float = 1e-3
    seed: int = 42
