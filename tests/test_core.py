from __future__ import annotations

import pytest
import torch

from dmgan.data import build_word_mask
from dmgan.losses import discriminator_loss, generator_loss
from dmgan.models import DMGenerator, DynamicMemory, build_discriminators


def test_generator_shapes_attention_and_gates() -> None:
    torch.manual_seed(7)
    generator = DMGenerator(channels=4, memory_dim=8, residual_blocks=1)
    noise = torch.randn(2, 100)
    sentence = torch.randn(2, 256)
    words = torch.randn(2, 256, 6)
    lengths = torch.tensor([6, 4])
    mask = build_word_mask(lengths, 6)
    images, diagnostics, mu, logvar = generator(noise, sentence, words, mask)
    assert [image.shape for image in images] == [
        (2, 3, 64, 64),
        (2, 3, 128, 128),
        (2, 3, 256, 256),
    ]
    assert mu.shape == logvar.shape == (2, 100)
    attention = diagnostics["attention_128"]
    assert attention.shape == (2, 6, 64, 64)
    assert torch.allclose(attention.sum(dim=1), torch.ones_like(attention[:, 0]), atol=1e-5)
    assert attention[1, 4:].abs().max() < 1e-6
    for name in ("writing_gate_128", "writing_gate_256", "response_gate_128", "response_gate_256"):
        gate = diagnostics[name]
        assert torch.all((gate >= 0) & (gate <= 1))


def test_all_masked_caption_is_rejected() -> None:
    memory = DynamicMemory(image_dim=4, word_dim=8, memory_dim=8)
    with pytest.raises(ValueError, match="unmasked"):
        memory(torch.randn(2, 4, 8, 8), torch.randn(2, 8, 3), torch.ones(2, 3, dtype=torch.bool))


def test_adversarial_and_kl_backward() -> None:
    torch.manual_seed(11)
    generator = DMGenerator(channels=4, memory_dim=8, residual_blocks=1)
    discriminators = build_discriminators(channels=4)
    noise = torch.randn(2, 100)
    sentence = torch.randn(2, 256)
    words = torch.randn(2, 256, 5)
    images, _, mu, logvar = generator(noise, sentence, words)
    real = [torch.randn_like(image) for image in images]
    for discriminator, real_image, fake_image in zip(discriminators, real, images, strict=True):
        loss, _ = discriminator_loss(discriminator, real_image, fake_image, sentence)
        assert torch.isfinite(loss)
    loss, metrics = generator_loss(
        list(discriminators), images, sentence, mu, logvar, matching_lambda=0.0
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert "kl" in metrics
    assert any(parameter.grad is not None for parameter in generator.parameters())


def test_discriminator_requires_two_examples() -> None:
    discriminator = build_discriminators(channels=4)[0]
    with pytest.raises(ValueError, match="batch_size"):
        discriminator_loss(
            discriminator, torch.randn(1, 3, 64, 64), torch.randn(1, 3, 64, 64), torch.randn(1, 256)
        )
