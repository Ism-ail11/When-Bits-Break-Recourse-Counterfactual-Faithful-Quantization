from __future__ import annotations

import copy
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from ..config import ExperimentConfig
from ..data import load_tabular_dataset
from ..metrics import accuracy, evaluate_recourse
from ..models import copy_fp_to_quantized, make_quant_tabular_model, make_tabular_model
from ..ptq import calibrate_ptq
from ..quantization import bit_cost, hard_bit_allocation
from ..recourse import PGDRecourseSolver
from ..training import (
    distill_model,
    magnitude_prune,
    train_quantized,
    train_recourse_margin_model,
    train_recourse_consistency_model,
    train_supervised,
)
from ..utils import resolve_device, save_json, seed_everything


def _eval_solver(config: ExperimentConfig) -> PGDRecourseSolver:
    rc = config.recourse
    return PGDRecourseSolver(
        steps=rc.eval_steps,
        step_size=rc.eval_step_size,
        restarts=rc.eval_restarts,
        cost_kind=rc.cost_kind,
        cost_weight=rc.cost_weight,
        mixed_l1=rc.mixed_l1,
        mixed_l2=rc.mixed_l2,
        margin=rc.margin,
    )


def _build_quantized(config: ExperimentConfig, input_dim: int):
    return make_quant_tabular_model(
        config.model.backbone,
        input_dim,
        config.model.hidden_dims,
        config.model.num_classes,
        bits=config.quant.bits,
        init_bit=config.quant.uniform_bit,
        quantize_activations=config.quant.quantize_activations,
    )


def run_tabular_experiment(
    config: ExperimentConfig,
    max_samples: int | None = None,
    max_eval_examples: int | None = 512,
    cache_dir: str | Path = "data",
    teacher_noise: float = 0.0,
    recourse_batch_fraction: float = 1.0,
) -> dict[str, Any]:
    seed_everything(config.train.seed)
    device = resolve_device(config.device)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_tabular_dataset(
        config.dataset,
        cache_dir=cache_dir,
        seed=config.train.seed,
        max_samples=max_samples,
    )
    fp_model = make_tabular_model(
        config.model.backbone,
        bundle.input_dim,
        config.model.hidden_dims,
        config.model.num_classes,
    )

    method = config.method.lower()
    fp_history = None
    if method == "r_margin":
        fp_history = train_recourse_margin_model(
            fp_model,
            bundle.x_train,
            bundle.y_train,
            bundle.x_val,
            bundle.y_val,
            bundle.action_set,
            bundle.feature_weights,
            bundle.target_label,
            config.train,
            config.recourse,
            device,
        )
    elif method == "r_consistency":
        peer = make_tabular_model(
            config.model.backbone,
            bundle.input_dim,
            config.model.hidden_dims,
            config.model.num_classes,
        )
        fp_history = train_recourse_consistency_model(
            fp_model,
            peer,
            bundle.x_train,
            bundle.y_train,
            bundle.x_val,
            bundle.y_val,
            bundle.action_set,
            bundle.feature_weights,
            bundle.target_label,
            config.train,
            config.recourse,
            device,
        )
    else:
        fp_history = train_supervised(
            fp_model,
            bundle.x_train,
            bundle.y_train,
            bundle.x_val,
            bundle.y_val,
            config.train,
            device,
        )

    fp_model.eval()
    fp_accuracy = accuracy(fp_model, bundle.x_test.to(device), bundle.y_test.to(device))

    if method == "fp32":
        q_model = copy.deepcopy(fp_model)
        quant_history: Any = None
    elif method == "kd_quant":
        student = make_tabular_model(
            config.model.backbone,
            bundle.input_dim,
            config.model.hidden_dims,
            config.model.num_classes,
        )
        distill_model(
            student,
            fp_model,
            bundle.x_train,
            bundle.y_train,
            bundle.x_val,
            bundle.y_val,
            config.train,
            device,
        )
        q_model = _build_quantized(config, bundle.input_dim)
        copy_fp_to_quantized(student, q_model)
        quant_history = train_quantized(
            q_model,
            student,
            bundle.x_train,
            bundle.y_train,
            bundle.x_val,
            bundle.y_val,
            bundle.action_set,
            bundle.feature_weights,
            bundle.target_label,
            "mixedprec",
            config.train,
            config.recourse,
            config.quant,
            device,
            teacher_noise=teacher_noise,
            recourse_batch_fraction=recourse_batch_fraction,
        )
        fp_model = student
    elif method == "prune_quant":
        magnitude_prune(fp_model, fraction=0.30)
        q_model = _build_quantized(config, bundle.input_dim)
        copy_fp_to_quantized(fp_model, q_model)
        quant_history = train_quantized(
            q_model,
            fp_model,
            bundle.x_train,
            bundle.y_train,
            bundle.x_val,
            bundle.y_val,
            bundle.action_set,
            bundle.feature_weights,
            bundle.target_label,
            "mixedprec",
            config.train,
            config.recourse,
            config.quant,
            device,
            teacher_noise=teacher_noise,
            recourse_batch_fraction=recourse_batch_fraction,
        )
    else:
        q_model = _build_quantized(config, bundle.input_dim)
        copy_fp_to_quantized(fp_model, q_model)
        if method in {"ptq4", "ptq8", "mixedptq", "cfptq", "cfptq_sensitivity"}:
            if method in {"ptq4", "ptq8"}:
                target_bit = 4 if method == "ptq4" else 8
                setattr(q_model, "fixed_bit", target_bit)
                ptq_quant = copy.deepcopy(config.quant)
                ptq_quant.target_avg_bits = float(target_bit)
                quant_history = calibrate_ptq(
                    q_model,
                    fp_model,
                    bundle.x_val,
                    bundle.y_val,
                    bundle.action_set,
                    bundle.feature_weights,
                    bundle.target_label,
                    candidate_bits=(target_bit,),
                    target_avg_bits=float(target_bit),
                    device=device,
                    cf_aware=False,
                    sensitivity_allocation=False,
                )
            else:
                quant_history = calibrate_ptq(
                    q_model,
                    fp_model,
                    bundle.x_val,
                    bundle.y_val,
                    bundle.action_set,
                    bundle.feature_weights,
                    bundle.target_label,
                    candidate_bits=config.quant.bits,
                    target_avg_bits=config.quant.target_avg_bits,
                    device=device,
                    cf_aware=method in {"cfptq", "cfptq_sensitivity"},
                    sensitivity_allocation=method == "cfptq_sensitivity",
                    teacher_steps=config.recourse.train_steps,
                )
        else:
            training_method = "mixedprec" if method in {"r_margin", "r_consistency"} else method
            quant_history = train_quantized(
                q_model,
                fp_model,
                bundle.x_train,
                bundle.y_train,
                bundle.x_val,
                bundle.y_val,
                bundle.action_set,
                bundle.feature_weights,
                bundle.target_label,
                training_method,
                config.train,
                config.recourse,
                config.quant,
                device,
                teacher_noise=teacher_noise,
                recourse_batch_fraction=recourse_batch_fraction,
            )

    q_model.to(device).eval()
    fp_model.to(device).eval()
    x_test = bundle.x_test.to(device)
    y_test = bundle.y_test.to(device)
    metrics, fp_recourse, q_recourse = evaluate_recourse(
        fp_model,
        q_model,
        x_test,
        y_test,
        bundle.target_label,
        bundle.action_set,
        _eval_solver(config),
        bundle.feature_weights.to(device),
        config.recourse.support_threshold,
        max_examples=max_eval_examples,
    )

    allocation = hard_bit_allocation(q_model) if method != "fp32" else [32]
    average_bits = float(bit_cost(q_model).detach().cpu().item()) if method != "fp32" else 32.0
    result = {
        "dataset": bundle.name,
        "method": method,
        "seed": config.train.seed,
        "fp_accuracy": fp_accuracy,
        "quantized_average_bits": average_bits,
        "bit_allocation": allocation,
        "metrics": metrics.to_dict(),
        "n_train": int(bundle.x_train.shape[0]),
        "n_val": int(bundle.x_val.shape[0]),
        "n_test": int(bundle.x_test.shape[0]),
        "input_dim": bundle.input_dim,
        "assumptions": {
            "crg_conditioning": "computed on examples with feasible FP and quantized recourse",
            "vd_conditioning": "target failure among examples with feasible FP recourse",
            "target": bundle.target_label,
        },
        "history": {
            "fp": asdict(fp_history) if fp_history is not None else None,
            "quantized": asdict(quant_history) if hasattr(quant_history, "__dataclass_fields__") else None,
        },
    }
    save_json(result, output_dir / "metrics.json")
    save_json(asdict(config), output_dir / "config.json")
    torch.save(fp_model.state_dict(), output_dir / "fp_model.pt")
    torch.save(q_model.state_dict(), output_dir / "quantized_model.pt")
    torch.save(
        {
            "fp_delta": fp_recourse.delta.detach().cpu(),
            "fp_success": fp_recourse.success.detach().cpu(),
            "q_delta": q_recourse.delta.detach().cpu(),
            "q_success": q_recourse.success.detach().cpu(),
        },
        output_dir / "recourse.pt",
    )
    return result
