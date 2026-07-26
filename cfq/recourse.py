from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .constraints import ActionSet
from .costs import recourse_cost


@dataclass
class RecourseResult:
    delta: torch.Tensor
    success: torch.Tensor
    cost: torch.Tensor
    target_loss: torch.Tensor


def target_margin(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target_logit = logits.gather(1, target.unsqueeze(1)).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, target.unsqueeze(1), float("-inf"))
    competitor = masked.max(dim=1).values
    return target_logit - competitor


class PGDRecourseSolver:
    def __init__(
        self,
        steps: int = 80,
        step_size: float = 0.04,
        restarts: int = 3,
        cost_kind: str = "l1",
        cost_weight: float = 0.02,
        mixed_l1: float = 0.5,
        mixed_l2: float = 0.5,
        margin: float = 0.0,
    ) -> None:
        self.steps = int(steps)
        self.step_size = float(step_size)
        self.restarts = int(restarts)
        self.cost_kind = cost_kind
        self.cost_weight = float(cost_weight)
        self.mixed_l1 = float(mixed_l1)
        self.mixed_l2 = float(mixed_l2)
        self.margin = float(margin)

    def solve(
        self,
        model: nn.Module,
        x: torch.Tensor,
        target: torch.Tensor | int,
        action_set: ActionSet,
        weights: torch.Tensor | None = None,
        create_graph: bool = False,
        detach_result: bool = True,
        initial_noise: float = 0.02,
    ) -> RecourseResult:
        if isinstance(target, int):
            target = torch.full((x.shape[0],), target, device=x.device, dtype=torch.long)
        else:
            target = target.to(device=x.device, dtype=torch.long)
        local_action_set = action_set.to(x.device)
        if weights is not None:
            weights = weights.to(x.device)

        best_delta = torch.zeros_like(x)
        best_success = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
        best_cost = torch.full((x.shape[0],), float("inf"), device=x.device)
        best_loss = torch.full((x.shape[0],), float("inf"), device=x.device)

        for restart in range(max(self.restarts, 1)):
            if restart == 0:
                delta = torch.zeros_like(x)
            else:
                delta = torch.randn_like(x) * initial_noise
                delta = local_action_set.project(x, delta, ste=create_graph)
            delta.requires_grad_(True)

            for _ in range(self.steps):
                logits = model(x + delta)
                ce = F.cross_entropy(logits, target, reduction="none")
                cost = recourse_cost(
                    delta,
                    weights,
                    self.cost_kind,
                    mixed_l1=self.mixed_l1,
                    mixed_l2=self.mixed_l2,
                )
                if self.margin > 0:
                    margin_penalty = F.relu(self.margin - target_margin(logits, target))
                else:
                    margin_penalty = torch.zeros_like(ce)
                objective = ce + self.cost_weight * cost + margin_penalty
                gradient = torch.autograd.grad(
                    objective.sum(),
                    delta,
                    create_graph=create_graph,
                    retain_graph=create_graph,
                )[0]
                delta = delta - self.step_size * gradient
                delta = local_action_set.project(x, delta, ste=create_graph)
                if not create_graph:
                    delta = delta.detach().requires_grad_(True)

            logits = model(x + delta)
            loss = F.cross_entropy(logits, target, reduction="none")
            success = logits.argmax(dim=1).eq(target)
            cost = recourse_cost(
                delta,
                weights,
                self.cost_kind,
                mixed_l1=self.mixed_l1,
                mixed_l2=self.mixed_l2,
            )
            better_success = success & (~best_success | (cost < best_cost))
            better_failure = ~success & ~best_success & (loss < best_loss)
            better = better_success | better_failure
            best_delta = torch.where(better.unsqueeze(1), delta, best_delta)
            best_success = torch.where(better, success, best_success)
            best_cost = torch.where(better, cost, best_cost)
            best_loss = torch.where(better, loss, best_loss)

        if detach_result:
            best_delta = best_delta.detach()
            best_success = best_success.detach()
            best_cost = best_cost.detach()
            best_loss = best_loss.detach()
        return RecourseResult(best_delta, best_success, best_cost, best_loss)


def solve_linear_l2(
    weight: torch.Tensor,
    bias: torch.Tensor,
    x: torch.Tensor,
    target_class: int = 1,
    feature_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Closed-form unconstrained binary affine recourse under weighted L2 cost.

    This is a sanity-check solver. Box, categorical, immutable, and sparsity
    constraints should be enforced afterwards with ``ActionSet.project`` or by
    the PGD solver.
    """
    if weight.ndim == 2 and weight.shape[0] == 2:
        direction = weight[target_class] - weight[1 - target_class]
        offset = bias[target_class] - bias[1 - target_class]
    elif weight.ndim == 1:
        direction = weight if target_class == 1 else -weight
        offset = bias if target_class == 1 else -bias
    else:
        raise ValueError("Expected a binary linear classifier")
    current_margin = x @ direction + offset
    needed = (-current_margin).clamp_min(0.0)
    if feature_weights is None:
        inverse_metric = torch.ones_like(direction)
    else:
        inverse_metric = 1.0 / feature_weights.square().clamp_min(1e-12)
    denominator = (direction.square() * inverse_metric).sum().clamp_min(1e-12)
    delta = needed.unsqueeze(1) * (direction * inverse_metric).unsqueeze(0) / denominator
    return delta

class RobustPGDRecourseSolver(PGDRecourseSolver):
    """Recourse solver robust to a finite uncertainty set of deployment models."""

    def solve_ensemble(
        self,
        models: list[nn.Module],
        x: torch.Tensor,
        target: torch.Tensor | int,
        action_set: ActionSet,
        weights: torch.Tensor | None = None,
        worst_case: bool = True,
    ) -> RecourseResult:
        if not models:
            raise ValueError("At least one model is required")
        if isinstance(target, int):
            target = torch.full((x.shape[0],), target, device=x.device, dtype=torch.long)
        else:
            target = target.to(x.device)
        local_action = action_set.to(x.device)
        local_weights = None if weights is None else weights.to(x.device)
        best_delta = torch.zeros_like(x)
        best_success = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
        best_cost = torch.full((x.shape[0],), float("inf"), device=x.device)
        best_loss = torch.full((x.shape[0],), float("inf"), device=x.device)

        for restart in range(max(self.restarts, 1)):
            delta = torch.zeros_like(x) if restart == 0 else local_action.project(x, torch.randn_like(x) * 0.02)
            delta.requires_grad_(True)
            for _ in range(self.steps):
                losses = torch.stack(
                    [F.cross_entropy(model(x + delta), target, reduction="none") for model in models], dim=0
                )
                aggregate = losses.max(dim=0).values if worst_case else losses.mean(dim=0)
                cost = recourse_cost(
                    delta,
                    local_weights,
                    self.cost_kind,
                    mixed_l1=self.mixed_l1,
                    mixed_l2=self.mixed_l2,
                )
                objective = aggregate + self.cost_weight * cost
                gradient = torch.autograd.grad(objective.sum(), delta)[0]
                delta = local_action.project(x, delta - self.step_size * gradient).detach().requires_grad_(True)

            predictions = torch.stack([model(x + delta).argmax(dim=1) for model in models], dim=0)
            success = predictions.eq(target.unsqueeze(0)).all(dim=0)
            losses = torch.stack(
                [F.cross_entropy(model(x + delta), target, reduction="none") for model in models], dim=0
            ).max(dim=0).values
            cost = recourse_cost(delta, local_weights, self.cost_kind)
            better = (success & (~best_success | (cost < best_cost))) | (~success & ~best_success & (losses < best_loss))
            best_delta = torch.where(better.unsqueeze(1), delta, best_delta)
            best_success = torch.where(better, success, best_success)
            best_cost = torch.where(better, cost, best_cost)
            best_loss = torch.where(better, losses, best_loss)
        return RecourseResult(best_delta.detach(), best_success.detach(), best_cost.detach(), best_loss.detach())
