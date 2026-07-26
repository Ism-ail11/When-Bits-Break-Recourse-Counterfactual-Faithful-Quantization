from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import torch


@dataclass
class ActionSet:
    """Actionability constraints in preprocessed feature space.

    Bounds apply to the post-action point ``x + delta``. Categorical groups are
    lists of one-hot coordinates. Ordinal domains map a feature index to its
    allowed scalar values. ``immutable`` always overrides other constraints.
    """

    lower: torch.Tensor
    upper: torch.Tensor
    immutable: tuple[int, ...] = ()
    sparsity: int | None = None
    categorical_groups: tuple[tuple[int, ...], ...] = ()
    ordinal_domains: dict[int, tuple[float, ...]] = field(default_factory=dict)

    def to(self, device: torch.device | str) -> "ActionSet":
        return ActionSet(
            lower=self.lower.to(device),
            upper=self.upper.to(device),
            immutable=self.immutable,
            sparsity=self.sparsity,
            categorical_groups=self.categorical_groups,
            ordinal_domains=self.ordinal_domains,
        )

    @property
    def dimension(self) -> int:
        return int(self.lower.numel())

    def actionable_mask(self, device: torch.device | str | None = None) -> torch.Tensor:
        target_device = device or self.lower.device
        mask = torch.ones(self.dimension, dtype=torch.bool, device=target_device)
        if self.immutable:
            mask[list(self.immutable)] = False
        return mask

    def project(self, x: torch.Tensor, delta: torch.Tensor, ste: bool = False) -> torch.Tensor:
        if x.shape != delta.shape:
            raise ValueError(f"x and delta must have the same shape, got {x.shape} and {delta.shape}")
        projected = delta
        if self.immutable:
            mask = self.actionable_mask(delta.device).to(delta.dtype)
            projected = projected * mask

        x_new = torch.maximum(torch.minimum(x + projected, self.upper), self.lower)

        for group in self.categorical_groups:
            if not group:
                continue
            index = torch.as_tensor(group, device=x.device, dtype=torch.long)
            values = x_new.index_select(-1, index)
            hard = torch.nn.functional.one_hot(values.argmax(dim=-1), num_classes=len(group)).to(values.dtype)
            if ste:
                hard = values + (hard - values).detach()
            x_new = x_new.clone()
            x_new[..., index] = hard

        for feature_index, domain in self.ordinal_domains.items():
            values = torch.as_tensor(domain, device=x.device, dtype=x.dtype)
            current = x_new[..., feature_index].unsqueeze(-1)
            nearest = values[(current - values).abs().argmin(dim=-1)]
            if ste:
                nearest = x_new[..., feature_index] + (nearest - x_new[..., feature_index]).detach()
            x_new = x_new.clone()
            x_new[..., feature_index] = nearest

        projected = x_new - x
        if self.immutable:
            projected = projected * self.actionable_mask(delta.device).to(delta.dtype)

        if self.sparsity is not None:
            actionable = self.actionable_mask(delta.device)
            grouped_indices = {index for group in self.categorical_groups for index in group}
            units: list[tuple[int, ...]] = []
            for group in self.categorical_groups:
                active_group = tuple(index for index in group if bool(actionable[index].item()))
                if active_group:
                    units.append(active_group)
            for index in range(self.dimension):
                if index not in grouped_indices and bool(actionable[index].item()):
                    units.append((index,))
            k = max(0, min(int(self.sparsity), len(units)))
            if k == 0:
                projected = torch.zeros_like(projected)
            elif k < len(units):
                unit_scores = torch.stack(
                    [projected[..., list(unit)].abs().sum(dim=-1) for unit in units], dim=-1
                )
                selected_units = unit_scores.topk(k, dim=-1).indices
                unit_mask = torch.zeros_like(unit_scores).scatter_(-1, selected_units, 1.0)
                sparse_mask = torch.zeros_like(projected)
                for unit_index, unit in enumerate(units):
                    sparse_mask[..., list(unit)] = unit_mask[..., unit_index].unsqueeze(-1)
                projected = projected * sparse_mask
        return projected

    @classmethod
    def unconstrained(cls, dimension: int, bound: float = 5.0) -> "ActionSet":
        return cls(
            lower=torch.full((dimension,), -bound),
            upper=torch.full((dimension,), bound),
            immutable=(),
            sparsity=None,
        )


def groups_to_tuples(groups: Iterable[Iterable[int]]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(index) for index in group) for group in groups)
