"""Contrastive objectives used by the DM-GAN+CL extension."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def _validate_views(view_a: torch.Tensor, view_b: torch.Tensor) -> None:
    if view_a.ndim != 2 or view_b.ndim != 2:
        raise ValueError("view_a and view_b must have shape [batch, feature_dim]")
    if view_a.shape != view_b.shape:
        raise ValueError("view_a and view_b must have the same shape")
    if view_a.size(0) < 2:
        raise ValueError("NT-Xent requires batch_size >= 2")
    if view_a.size(1) < 1:
        raise ValueError("feature_dim must be at least 1")
    if view_a.device != view_b.device:
        raise ValueError("view_a and view_b must be on the same device")
    if view_a.dtype != view_b.dtype:
        raise ValueError("view_a and view_b must have the same dtype")
    if not torch.is_floating_point(view_a):
        raise TypeError("view_a and view_b must be floating-point tensors")


def nt_xent_loss(
    view_a: torch.Tensor,
    view_b: torch.Tensor,
    *,
    temperature: float = 0.5,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return the symmetric normalized temperature-scaled cross-entropy loss.

    ``view_a[i]`` and ``view_b[i]`` form a positive pair.  For each of the
    ``2 * batch`` anchors, every other representation except the anchor itself
    participates in the softmax denominator; the remaining samples are
    in-batch negatives.

    Args:
        view_a: First set of representations with shape ``[batch, feature_dim]``.
        view_b: Paired representations with the same shape, device, and dtype.
        temperature: Positive temperature used to scale cosine similarities.
        eps: Positive lower bound used during L2 normalization.

    Returns:
        A scalar loss averaged over all ``2 * batch`` anchors.
    """
    _validate_views(view_a, view_b)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and greater than zero")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and greater than zero")

    # Compute low-precision inputs in float32 so normalization and softmax stay
    # stable under mixed precision.  The cast remains differentiable.
    compute_dtype = torch.float32 if view_a.dtype in (torch.float16, torch.bfloat16) else view_a.dtype
    representations = torch.cat((view_a, view_b), dim=0).to(dtype=compute_dtype)
    normalization_eps = max(eps, torch.finfo(compute_dtype).tiny)
    representations = F.normalize(representations, dim=1, eps=normalization_eps)

    logits = representations @ representations.transpose(0, 1)
    logits = logits / temperature
    sample_count = logits.size(0)
    self_mask = torch.eye(sample_count, dtype=torch.bool, device=logits.device)
    logits = logits.masked_fill(self_mask, torch.finfo(logits.dtype).min)

    batch = view_a.size(0)
    anchors = torch.arange(sample_count, device=logits.device)
    positive_indices = (anchors + batch) % sample_count
    return F.cross_entropy(logits, positive_indices)


class NTXentLoss(nn.Module):
    """Module wrapper around :func:`nt_xent_loss`."""

    def __init__(self, temperature: float = 0.5, eps: float = 1e-8) -> None:
        super().__init__()
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and greater than zero")
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be finite and greater than zero")
        self.temperature = float(temperature)
        self.eps = float(eps)

    def forward(self, view_a: torch.Tensor, view_b: torch.Tensor) -> torch.Tensor:
        return nt_xent_loss(view_a, view_b, temperature=self.temperature, eps=self.eps)
