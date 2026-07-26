from __future__ import annotations

import torch


def feature_weights_from_std(x_train: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return 1.0 / (x_train.std(dim=0, unbiased=False).clamp_min(eps))


def feature_weights_from_mad(x_train: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    median = x_train.median(dim=0).values
    mad = (x_train - median).abs().median(dim=0).values
    return 1.0 / mad.clamp_min(eps)


def recourse_cost(
    delta: torch.Tensor,
    weights: torch.Tensor | None = None,
    kind: str = "l1",
    mixed_l1: float = 0.5,
    mixed_l2: float = 0.5,
) -> torch.Tensor:
    weighted = delta if weights is None else delta * weights
    if kind == "l1":
        return weighted.abs().sum(dim=-1)
    if kind == "l2":
        return torch.linalg.vector_norm(weighted, ord=2, dim=-1)
    if kind == "mixed":
        return mixed_l1 * weighted.abs().sum(dim=-1) + mixed_l2 * torch.linalg.vector_norm(
            weighted, ord=2, dim=-1
        )
    raise ValueError(f"Unknown cost kind: {kind}")
