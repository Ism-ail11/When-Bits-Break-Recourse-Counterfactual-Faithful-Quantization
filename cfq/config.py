from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ModelConfig:
    backbone: Literal["logreg", "mlp", "deep_mlp"] = "mlp"
    hidden_dims: tuple[int, ...] = (128, 64)
    dropout: float = 0.0
    num_classes: int = 2


@dataclass
class RecourseConfig:
    train_steps: int = 3
    eval_steps: int = 80
    train_step_size: float = 0.08
    eval_step_size: float = 0.04
    train_restarts: int = 1
    eval_restarts: int = 3
    cost_kind: Literal["l1", "l2", "mixed"] = "l1"
    cost_weight: float = 0.02
    mixed_l1: float = 0.5
    mixed_l2: float = 0.5
    margin: float = 0.0
    support_threshold: float = 1e-4


@dataclass
class QuantConfig:
    bits: tuple[int, ...] = (2, 3, 4, 8)
    uniform_bit: int = 4
    mixed_precision: bool = True
    temperature_start: float = 5.0
    temperature_end: float = 0.25
    target_avg_bits: float = 4.0
    budget_tolerance: float = 0.05
    quantize_activations: bool = True


@dataclass
class TrainConfig:
    epochs_fp: int = 30
    epochs_qat: int = 20
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    eta: float = 1.0
    bit_lambda: float = 1e-2
    hinge_beta: float = 0.0
    hinge_gamma: float = 0.25
    match_alpha1: float = 0.0
    match_alpha2: float = 0.0
    early_stopping_patience: int = 8
    seed: int = 42


@dataclass
class ExperimentConfig:
    dataset: str = "synthetic"
    method: str = "cfq"
    model: ModelConfig = field(default_factory=ModelConfig)
    recourse: RecourseConfig = field(default_factory=RecourseConfig)
    quant: QuantConfig = field(default_factory=QuantConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    output_dir: str = "results/run"
    device: str | None = None
