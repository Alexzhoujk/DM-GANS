"""Experimental part-aware alignment utilities; not enabled in the baseline."""

from __future__ import annotations

import torch


def gaussian_part_heatmaps(
    coordinates: torch.Tensor,
    visible: torch.Tensor,
    height: int,
    width: int,
    sigma: float = 2.0,
) -> torch.Tensor:
    """Convert normalized CUB part coordinates into spatial probability maps.

    Args:
        coordinates: [batch, parts, 2] normalized x/y coordinates in [0, 1].
        visible: [batch, parts] visibility mask.
    """
    if coordinates.ndim != 3 or coordinates.size(-1) != 2:
        raise ValueError("coordinates must have shape [batch, parts, 2]")
    y_grid, x_grid = torch.meshgrid(
        torch.arange(height, device=coordinates.device, dtype=coordinates.dtype),
        torch.arange(width, device=coordinates.device, dtype=coordinates.dtype),
        indexing="ij",
    )
    x = coordinates[..., 0, None, None] * (width - 1)
    y = coordinates[..., 1, None, None] * (height - 1)
    squared_distance = (x_grid - x).square() + (y_grid - y).square()
    heatmaps = torch.exp(-squared_distance / (2.0 * sigma**2))
    heatmaps = heatmaps * visible[..., None, None].to(heatmaps.dtype)
    normalizer = heatmaps.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    return heatmaps / normalizer


def part_aware_alignment_loss(
    attention: torch.Tensor,
    part_heatmaps: torch.Tensor,
    token_part_targets: torch.Tensor,
    word_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cross-entropy between word attention and the corresponding CUB part map.

    `token_part_targets[b, t, p]` expresses which part(s) token `t` describes.
    Tokens without a part target do not contribute to the loss.
    """
    if attention.ndim != 4:
        raise ValueError("attention must have shape [batch, words, height, width]")
    if part_heatmaps.ndim != 4 or token_part_targets.ndim != 3:
        raise ValueError("part_heatmaps and token_part_targets must be rank 4 and 3")
    if attention.shape[0] != part_heatmaps.shape[0] or attention.shape[:2] != token_part_targets.shape[:2]:
        raise ValueError("batch/word dimensions do not match")
    if part_heatmaps.size(1) != token_part_targets.size(2):
        raise ValueError("part target dimension does not match heatmaps")
    if part_heatmaps.shape[-2:] != attention.shape[-2:]:
        part_heatmaps = torch.nn.functional.interpolate(
            part_heatmaps, size=attention.shape[-2:], mode="bilinear", align_corners=False
        )
        part_heatmaps = part_heatmaps / part_heatmaps.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-8)

    target = torch.einsum("btp,bphw->bthw", token_part_targets, part_heatmaps)
    target_mass = target.sum(dim=(-2, -1), keepdim=True)
    target = target / target_mass.clamp_min(1e-8)
    spatial_attention = attention.clamp_min(0)
    spatial_attention = spatial_attention / spatial_attention.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    per_token = -(target * spatial_attention.clamp_min(1e-8).log()).sum(dim=(-2, -1))
    active = target_mass.squeeze(-1).squeeze(-1) > 0
    if word_mask is not None:
        active = active & ~word_mask.bool()
    if not active.any():
        return attention.new_zeros(())
    return per_token[active].mean()
