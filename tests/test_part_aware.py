import torch

from dmgan.part_aware import (
    gaussian_part_heatmaps,
    part_alignment_statistics,
    part_aware_alignment_loss,
    token_part_targets,
)


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


def test_token_targets_attach_colour_to_nearest_part() -> None:
    index_to_word = {1: "yellow", 2: "belly", 3: "black", 4: "wings", 0: "<end>"}
    targets = token_part_targets(torch.tensor([1, 2, 3, 4, 0]), index_to_word)
    assert targets[0, 2] == 1  # yellow -> belly
    assert targets[1, 2] == 1
    assert targets[2, 8] == targets[2, 12] == 1  # black -> both wings
    assert targets[4].sum() == 0


def test_alignment_statistics_reward_target_support() -> None:
    coordinates = torch.tensor([[[0.25, 0.5], [0.75, 0.5]]])
    heatmaps = gaussian_part_heatmaps(coordinates, torch.ones(1, 2, dtype=torch.bool), 16, 16, sigma=1.5)
    targets = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    aligned_ce, aligned_mass, counts = part_alignment_statistics(heatmaps, heatmaps, targets)
    swapped_ce, swapped_mass, _ = part_alignment_statistics(heatmaps.flip(1), heatmaps, targets)
    assert counts.tolist() == [2]
    assert aligned_ce.item() < swapped_ce.item()
    assert aligned_mass.item() > swapped_mass.item()
