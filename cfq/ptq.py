from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .quantization import iter_quant_layers
from .recourse import PGDRecourseSolver
from .training import make_loader


@dataclass
class PTQHistory:
    losses: list[float]
    allocation: list[int]


def _set_layer_bit(layer, bit: int) -> None:
    bits = tuple(int(value) for value in layer.quantizer.policy.bits.tolist())
    if bit not in bits:
        raise ValueError(f"Bit {bit} not in candidates {bits}")
    with torch.no_grad():
        layer.quantizer.policy.logits.fill_(-8.0)
        layer.quantizer.policy.logits[bits.index(bit)] = 8.0


def set_allocation(model: nn.Module, allocation: list[int]) -> None:
    layers = list(iter_quant_layers(model))
    if len(layers) != len(allocation):
        raise ValueError(f"Expected {len(layers)} bits, got {len(allocation)}")
    for layer, bit in zip(layers, allocation):
        _set_layer_bit(layer, int(bit))
    setattr(model, "temperature", 0.1)
    setattr(model, "hard", True)
    setattr(model, "stochastic", False)
    setattr(model, "fixed_bit", None)


def greedy_bit_allocation(
    model: nn.Module,
    sensitivities: list[float],
    candidate_bits: tuple[int, ...],
    target_avg_bits: float,
) -> list[int]:
    layers = list(iter_quant_layers(model))
    if len(layers) != len(sensitivities):
        raise ValueError("Sensitivity length must match quantized layers")
    candidates = tuple(sorted(candidate_bits))
    allocation = [candidates[0]] * len(layers)
    counts = [layer.parameter_count_for_budget for layer in layers]
    total_params = sum(counts)
    budget = target_avg_bits * total_params
    current = sum(bit * count for bit, count in zip(allocation, counts))
    while True:
        best = None
        best_ratio = -float("inf")
        for index, (bit, count, score) in enumerate(zip(allocation, counts, sensitivities)):
            position = candidates.index(bit)
            if position + 1 >= len(candidates):
                continue
            next_bit = candidates[position + 1]
            extra = (next_bit - bit) * count
            if current + extra > budget + 1e-6:
                continue
            ratio = float(score) * (next_bit - bit) / max(extra, 1)
            if ratio > best_ratio:
                best_ratio = ratio
                best = (index, next_bit, extra)
        if best is None:
            break
        index, next_bit, extra = best
        allocation[index] = next_bit
        current += extra
    return allocation


def layer_sensitivity(
    fp_model: nn.Module,
    x: torch.Tensor,
    target: torch.Tensor,
    batch_size: int = 256,
) -> list[float]:
    layers = [module for module in fp_model.modules() if isinstance(module, (nn.Linear, nn.Conv2d))]
    scores = [0.0 for _ in layers]
    counts = 0
    fp_model.train(False)
    for start in range(0, x.shape[0], batch_size):
        bx = x[start : start + batch_size]
        bt = target[start : start + batch_size]
        fp_model.zero_grad(set_to_none=True)
        loss = F.cross_entropy(fp_model(bx), bt)
        loss.backward()
        for index, layer in enumerate(layers):
            if layer.weight.grad is not None:
                scores[index] += float(layer.weight.grad.detach().abs().mean().item()) * bx.shape[0]
        counts += bx.shape[0]
    return [score / max(counts, 1) for score in scores]


def calibrate_ptq(
    q_model: nn.Module,
    fp_model: nn.Module,
    x_calibration: torch.Tensor,
    y_calibration: torch.Tensor,
    action_set,
    feature_weights: torch.Tensor,
    target_label: int,
    candidate_bits: tuple[int, ...],
    target_avg_bits: float,
    device: torch.device,
    cf_aware: bool = False,
    sensitivity_allocation: bool = False,
    calibration_epochs: int = 8,
    batch_size: int = 256,
    learning_rate: float = 5e-3,
    teacher_steps: int = 3,
) -> PTQHistory:
    q_model.to(device)
    fp_model.to(device).eval()
    for parameter in fp_model.parameters():
        parameter.requires_grad_(False)
    x_calibration = x_calibration.to(device)
    y_calibration = y_calibration.to(device)
    target = torch.full_like(y_calibration, target_label)

    calibration_inputs = x_calibration
    sensitivity_inputs = x_calibration
    if cf_aware:
        solver = PGDRecourseSolver(steps=teacher_steps, step_size=0.08, restarts=1, cost_kind="l1", cost_weight=0.02)
        with torch.enable_grad():
            result = solver.solve(
                fp_model,
                x_calibration,
                target,
                action_set.to(device),
                feature_weights.to(device),
            )
        calibration_inputs = torch.cat([x_calibration, x_calibration + result.delta], dim=0)
        sensitivity_inputs = x_calibration + result.delta

    if sensitivity_allocation:
        sensitivities = layer_sensitivity(fp_model, sensitivity_inputs, target)
    else:
        factual_targets = y_calibration if not cf_aware else target
        sensitivities = layer_sensitivity(fp_model, sensitivity_inputs, factual_targets)
    allocation = greedy_bit_allocation(q_model, sensitivities, candidate_bits, target_avg_bits)
    set_allocation(q_model, allocation)

    # Freeze backbone values; optimize only quantizer scales/clips and policy logits.
    trainable = []
    for name, parameter in q_model.named_parameters():
        is_quantizer = any(token in name for token in ("raw_steps", "raw_clips", "policy.logits"))
        parameter.requires_grad_(is_quantizer)
        if is_quantizer:
            trainable.append(parameter)
    optimizer = torch.optim.Adam(trainable, lr=learning_rate)
    loader = make_loader(
        calibration_inputs.detach().cpu(),
        torch.zeros(calibration_inputs.shape[0], dtype=torch.long),
        batch_size,
        True,
        42,
    )
    losses = []
    for _ in range(calibration_epochs):
        q_model.train()
        epoch_loss, count = 0.0, 0
        for bx, _ in loader:
            bx = bx.to(device)
            with torch.no_grad():
                teacher_logits = fp_model(bx)
            student_logits = q_model(bx)
            loss = F.mse_loss(student_logits, teacher_logits)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * bx.shape[0]
            count += bx.shape[0]
        losses.append(epoch_loss / max(count, 1))
    q_model.eval()
    for parameter in q_model.parameters():
        parameter.requires_grad_(True)
    return PTQHistory(losses=losses, allocation=allocation)
