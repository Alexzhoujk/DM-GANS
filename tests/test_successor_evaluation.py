from __future__ import annotations

import pytest
import torch

from scripts.evaluate_successor_checkpoint import (
    conditioning_pairing_description,
    paired_generator_forward,
)


class _RngProbeGenerator:
    def __call__(
        self,
        noise: torch.Tensor,
        sentence: torch.Tensor,
        words: torch.Tensor,
        mask: torch.Tensor,
        *,
        sample_conditioning: bool,
    ):
        del sentence, words, mask
        condition = torch.randn_like(noise) if sample_conditioning else torch.zeros_like(noise)
        return [condition], {}, condition, condition


def _paired_probe(
    baseline_sample_conditioning: bool,
    candidate_sample_conditioning: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    noise = torch.zeros(2, 3)
    sentence = torch.zeros(2, 3)
    words = torch.zeros(2, 3, 1)
    mask = torch.zeros(2, 1, dtype=torch.bool)
    return paired_generator_forward(
        _RngProbeGenerator(),
        _RngProbeGenerator(),
        noise,
        sentence,
        words,
        sentence,
        words,
        mask,
        torch.device("cpu"),
        baseline_sample_conditioning,
        candidate_sample_conditioning,
    )


def test_both_sample_modes_share_conditioning_epsilon() -> None:
    torch.manual_seed(123)
    baseline, candidate = _paired_probe(True, True)
    assert torch.equal(baseline, candidate)


def test_sample_vs_mean_advances_rng_between_batches() -> None:
    torch.manual_seed(123)
    first_baseline, first_candidate = _paired_probe(True, False)
    second_baseline, second_candidate = _paired_probe(True, False)
    assert not torch.equal(first_baseline, second_baseline)
    assert torch.count_nonzero(first_candidate) == 0
    assert torch.count_nonzero(second_candidate) == 0


def test_pairing_description_does_not_claim_shared_epsilon_for_mixed_modes() -> None:
    description = conditioning_pairing_description("sample", "mean")
    assert "same CA epsilon" not in description
    assert "mean-mode side uses no CA epsilon" in description


@pytest.mark.parametrize(
    ("baseline_samples", "candidate_samples", "draw_count"),
    [(True, True, 1), (True, False, 1), (False, True, 1), (False, False, 0)],
)
def test_pairing_advances_one_shared_ca_stream(
    baseline_samples: bool,
    candidate_samples: bool,
    draw_count: int,
) -> None:
    torch.manual_seed(987)
    _paired_probe(baseline_samples, candidate_samples)
    observed_next = torch.randn(2, 3)

    torch.manual_seed(987)
    for _ in range(draw_count):
        torch.randn(2, 3)
    expected_next = torch.randn(2, 3)
    assert torch.equal(observed_next, expected_next)
