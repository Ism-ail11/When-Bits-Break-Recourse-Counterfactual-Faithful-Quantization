from __future__ import annotations

from pathlib import Path

import torch

from cfq.config import ExperimentConfig
from cfq.data import load_tabular_dataset
from cfq.models import make_quant_tabular_model, make_tabular_model


def load_models(config: ExperimentConfig, run_dir: str | Path, max_samples: int | None = None, cache_dir="data"):
    bundle = load_tabular_dataset(config.dataset, cache_dir=cache_dir, seed=config.train.seed, max_samples=max_samples)
    fp_model = make_tabular_model(
        config.model.backbone,
        bundle.input_dim,
        config.model.hidden_dims,
        config.model.num_classes,
    )
    q_model = make_quant_tabular_model(
        config.model.backbone,
        bundle.input_dim,
        config.model.hidden_dims,
        config.model.num_classes,
        bits=config.quant.bits,
        init_bit=config.quant.uniform_bit,
        quantize_activations=config.quant.quantize_activations,
    )
    run_dir = Path(run_dir)
    fp_model.load_state_dict(torch.load(run_dir / "fp_model.pt", map_location="cpu"))
    q_model.load_state_dict(torch.load(run_dir / "quantized_model.pt", map_location="cpu"))
    q_model.temperature = config.quant.temperature_end
    q_model.hard = True
    q_model.stochastic = False
    q_model.fixed_bit = config.quant.uniform_bit if config.method in {"lsq", "pact", "cfq_uniform", "ptq4"} else (8 if config.method == "ptq8" else None)
    return bundle, fp_model, q_model
