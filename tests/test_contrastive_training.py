from __future__ import annotations

import torch

from dmgan.config import DMGANConfig
from dmgan.damsm import DAMSMTextEncoder, TinyMatchingImageEncoder
from dmgan.data import CaptionSample, collate_caption_samples
from dmgan.models import DMGenerator, build_discriminators
from dmgan.training import DMGANTrainer


def _sample(primary: list[int], paired: list[int], key: str) -> CaptionSample:
    words_num = 6
    caption = torch.zeros(words_num, dtype=torch.long)
    paired_caption = torch.zeros(words_num, dtype=torch.long)
    caption[: len(primary)] = torch.tensor(primary)
    paired_caption[: len(paired)] = torch.tensor(paired)
    return CaptionSample(
        images=[torch.randn(3, size, size) for size in (64, 128, 256)],
        caption=caption,
        caption_length=len(primary),
        class_id=int(key),
        key=key,
        paired_caption=paired_caption,
        paired_caption_length=len(paired),
    )


def test_collate_keeps_second_caption_on_the_same_image_row() -> None:
    batch = collate_caption_samples(
        [
            _sample([1, 2, 3], [8], "1"),
            _sample([4, 5, 6, 7, 8], [9, 10, 11, 12], "2"),
        ]
    )

    assert batch["keys"] == ["2", "1"]
    assert batch["caption_lengths"].tolist() == [5, 3]
    assert batch["paired_caption_lengths"].tolist() == [4, 1]
    assert batch["paired_captions"][:, 0].tolist() == [9, 8]


def test_dual_caption_step_backpropagates_through_frozen_image_encoder() -> None:
    torch.manual_seed(31)
    config = DMGANConfig(
        generator_channels=4,
        discriminator_channels=4,
        memory_dim=8,
        residual_blocks=1,
        batch_size=2,
        matching_lambda=0.0,
        contrastive_lambda=0.2,
        contrastive_temperature=0.5,
    )
    generator = DMGenerator(channels=4, memory_dim=8, residual_blocks=1)
    image_encoder = TinyMatchingImageEncoder(embedding_dim=256)
    trainer = DMGANTrainer(
        config,
        generator,
        build_discriminators(channels=4),
        DAMSMTextEncoder(vocabulary_size=32),
        image_encoder,
        torch.device("cpu"),
    )
    # View 2 is deliberately not length-sorted after view 1 is sorted.  The
    # trainer must sort it for pack_padded_sequence and restore pair order.
    batch = collate_caption_samples(
        [
            _sample([1, 2, 3, 4, 5], [6, 7], "1"),
            _sample([8, 9, 10], [11, 12, 13, 14], "2"),
        ]
    )

    metrics, images = trainer.train_step(batch)

    assert len(images) == 3
    assert metrics["g_contrastive"] > 0
    assert metrics["g_contrastive_weighted"] > 0
    assert any(parameter.grad is not None for parameter in trainer.generator.parameters())
    assert all(parameter.grad is None for parameter in trainer.image_encoder.parameters())
