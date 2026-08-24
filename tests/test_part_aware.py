import torch

from dmgan.part_aware import gaussian_part_heatmaps, part_aware_alignment_loss


def test_part_aware_loss_prefers_aligned_attention() -> None:
    coordinates = torch.tensor([[[0.25, 0.5], [0.75, 0.5]]])
    heatmaps = gaussian_part_heatmaps(coordinates, torch.ones(1, 2, dtype=torch.bool), 16, 16, sigma=1.5)
    targets = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    aligned = heatmaps.clone()
    swapped = heatmaps.flip(1)
    assert part_aware_alignment_loss(aligned, heatmaps, targets) < part_aware_alignment_loss(
        swapped, heatmaps, targets
    )


def test_masked_or_unassigned_tokens_are_ignored() -> None:
    attention = torch.ones(1, 2, 4, 4)
    heatmaps = torch.ones(1, 1, 4, 4) / 16
    targets = torch.tensor([[[1.0], [0.0]]])
    loss = part_aware_alignment_loss(attention, heatmaps, targets, torch.tensor([[False, True]]))
    assert torch.isfinite(loss)
