from __future__ import annotations

import pytest
import torch

from dmgan.contrastive import NTXentLoss, nt_xent_loss


def test_correct_pairs_have_lower_loss_than_shuffled_pairs() -> None:
    view_a = torch.eye(4)
    view_b = view_a.clone()
    shuffled_view_b = view_b[torch.tensor([1, 0, 3, 2])]

    paired_loss = nt_xent_loss(view_a, view_b, temperature=0.5)
    shuffled_loss = nt_xent_loss(view_a, shuffled_view_b, temperature=0.5)

    assert paired_loss < shuffled_loss


def test_loss_is_finite_and_backpropagates_to_both_views() -> None:
    torch.manual_seed(23)
    view_a = torch.randn(5, 8, requires_grad=True)
    view_b = (view_a.detach() + 0.1 * torch.randn(5, 8)).requires_grad_()

    loss = NTXentLoss(temperature=0.5)(view_a, view_b)
    loss.backward()

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert view_a.grad is not None and torch.isfinite(view_a.grad).all()
    assert view_b.grad is not None and torch.isfinite(view_b.grad).all()
    assert view_a.grad.abs().sum() > 0
    assert view_b.grad.abs().sum() > 0


def test_zero_representations_remain_numerically_stable() -> None:
    loss = nt_xent_loss(torch.zeros(2, 3), torch.zeros(2, 3))
    assert torch.isfinite(loss)


@pytest.mark.parametrize(
    ("view_a", "view_b", "message"),
    [
        (torch.randn(1, 4), torch.randn(1, 4), "batch_size"),
        (torch.randn(2, 4), torch.randn(2, 5), "same shape"),
        (torch.randn(2, 4, 1), torch.randn(2, 4, 1), "shape"),
    ],
)
def test_invalid_view_shapes_are_rejected(
    view_a: torch.Tensor,
    view_b: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        nt_xent_loss(view_a, view_b)


@pytest.mark.parametrize("temperature", [0.0, -0.5, float("inf"), float("nan")])
def test_invalid_temperature_is_rejected(temperature: float) -> None:
    with pytest.raises(ValueError, match="temperature"):
        nt_xent_loss(torch.randn(2, 4), torch.randn(2, 4), temperature=temperature)
