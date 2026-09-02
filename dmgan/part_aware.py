"""Part-aware alignment utilities for the controlled Session 7 ablation.

The CUB annotations contain 15 landmark locations.  The helpers in this module
map high-confidence caption tokens to those landmarks and measure whether a
DM-GAN word-attention map puts probability near the annotated location.
"""

from __future__ import annotations

import torch

CUB_PART_NAMES = (
    "back",
    "beak",
    "belly",
    "breast",
    "crown",
    "forehead",
    "left_eye",
    "left_leg",
    "left_wing",
    "nape",
    "right_eye",
    "right_leg",
    "right_wing",
    "tail",
    "throat",
)

# Zero-based indices into CUB_PART_NAMES.  Left/right landmarks are grouped for
# generic words such as "wing" because captions normally do not specify side.
PART_ALIASES: dict[str, tuple[int, ...]] = {
    "back": (0,),
    "beak": (1,),
    "beaked": (1,),
    "bill": (1,),
    "mandible": (1,),
    "belly": (2,),
    "abdomen": (2,),
    "stomach": (2,),
    "breast": (3,),
    "chest": (3,),
    "crown": (4,),
    "forehead": (5,),
    "eye": (6, 10),
    "eyes": (6, 10),
    "eyering": (6, 10),
    "leg": (7, 11),
    "legs": (7, 11),
    "feet": (7, 11),
    "foot": (7, 11),
    "tarsus": (7, 11),
    "wing": (8, 12),
    "wings": (8, 12),
    "wingbar": (8, 12),
    "wingbars": (8, 12),
    "primaries": (8, 12),
    "secondaries": (8, 12),
    "coverts": (8, 12),
    "nape": (9,),
    "neck": (9, 14),
    "tail": (13,),
    "rectrices": (13,),
    "throat": (14,),
    # CUB has no single head landmark.  This fixed composite deliberately
    # excludes beak and throat, which have their own explicit annotations.
    "head": (4, 5, 6, 9, 10),
    "face": (1, 5, 6, 10),
}

ATTRIBUTE_WORDS = {
    "black",
    "blue",
    "brown",
    "buff",
    "cream",
    "gray",
    "green",
    "grey",
    "iridescent",
    "olive",
    "orange",
    "pink",
    "purple",
    "red",
    "rufous",
    "tan",
    "white",
    "yellow",
    "barred",
    "mottled",
    "plain",
    "spotted",
    "striped",
    "streaked",
}


def token_part_targets(
    token_ids: torch.Tensor,
    index_to_word: dict[int, str],
    *,
    attribute_window: int = 4,
) -> torch.Tensor:
    """Build a [words, 15] token-to-part target matrix.

    Part nouns receive direct targets.  A conservative list of colour/pattern
    words receives the target of the closest part noun within four tokens.  A
    tie is resolved in favour of the later noun, matching common phrases such
    as "yellow belly".
    """
    if token_ids.ndim != 1:
        raise ValueError("token_ids must have shape [words]")
    words = [str(index_to_word.get(int(token), "")).lower() for token in token_ids]
    result = torch.zeros((len(words), len(CUB_PART_NAMES)), dtype=torch.float32)
    direct: dict[int, tuple[int, ...]] = {}
    for index, word in enumerate(words):
        parts = PART_ALIASES.get(word)
        if parts is None:
            continue
        direct[index] = parts
        result[index, list(parts)] = 1.0

    for index, word in enumerate(words):
        if word not in ATTRIBUTE_WORDS:
            continue
        candidates = [
            (abs(index - noun_index), 0 if noun_index >= index else 1, noun_index, parts)
            for noun_index, parts in direct.items()
            if abs(index - noun_index) <= attribute_window
        ]
        if not candidates:
            continue
        _, _, _, parts = min(candidates)
        result[index, list(parts)] = 1.0
    return result


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


def part_alignment_statistics(
    attention: torch.Tensor,
    part_heatmaps: torch.Tensor,
    token_part_targets: torch.Tensor,
    word_mask: torch.Tensor | None = None,
    *,
    support_fraction: float = 0.10,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return per-sample CE, target-support mass, and active-token counts.

    ``support mass`` is the attention probability inside the most likely
    ``support_fraction`` of pixels under the target heatmap.  Uniform spatial
    attention therefore scores approximately ``support_fraction``.
    """
    if not 0.0 < support_fraction <= 1.0:
        raise ValueError("support_fraction must be in (0, 1]")
    if attention.ndim != 4 or part_heatmaps.ndim != 4 or token_part_targets.ndim != 3:
        raise ValueError("attention/heatmaps/targets must have ranks 4/4/3")
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
    per_token_ce = -(target * spatial_attention.clamp_min(1e-8).log()).sum(dim=(-2, -1))
    active = target_mass.squeeze(-1).squeeze(-1) > 0
    if word_mask is not None:
        active = active & ~word_mask.bool()

    flat_target = target.flatten(2)
    flat_attention = spatial_attention.flatten(2)
    support_pixels = max(1, round(flat_target.size(-1) * support_fraction))
    support_indices = flat_target.topk(support_pixels, dim=-1).indices
    support_mass = flat_attention.gather(-1, support_indices).sum(dim=-1)

    counts = active.sum(dim=1)
    denominator = counts.clamp_min(1).to(attention.dtype)
    sample_ce = (per_token_ce * active).sum(dim=1) / denominator
    sample_support = (support_mass * active).sum(dim=1) / denominator
    nan = torch.full_like(sample_ce, float("nan"))
    return torch.where(counts > 0, sample_ce, nan), torch.where(counts > 0, sample_support, nan), counts


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
