"""Adversarial, DAMSM matching, and KL losses for DM-GAN."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .models import MultiscaleDiscriminator


def _class_collision_mask(class_ids: torch.Tensor | None) -> torch.Tensor | None:
    if class_ids is None:
        return None
    class_ids = class_ids.reshape(-1)
    mask = class_ids[:, None].eq(class_ids[None, :])
    mask.fill_diagonal_(False)
    return mask


def sentence_matching_loss(
    image_code: torch.Tensor,
    sentence_code: torch.Tensor,
    labels: torch.Tensor,
    class_ids: torch.Tensor | None = None,
    gamma3: float = 10.0,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    image_code = F.normalize(image_code, dim=1, eps=eps)
    sentence_code = F.normalize(sentence_code, dim=1, eps=eps)
    scores = image_code @ sentence_code.transpose(0, 1) * gamma3
    class_mask = _class_collision_mask(class_ids)
    if class_mask is not None:
        scores = scores.masked_fill(class_mask, torch.finfo(scores.dtype).min)
    return F.cross_entropy(scores, labels), F.cross_entropy(scores.transpose(0, 1), labels)


def word_region_attention(
    words: torch.Tensor,
    image_regions: torch.Tensor,
    gamma1: float = 4.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """AttnGAN/DAMSM word-to-region attention.

    Args:
        words: [batch, embedding_dim, word_count]
        image_regions: [batch, embedding_dim, height, width]
    """
    batch, _, word_count = words.shape
    height, width = image_regions.shape[-2:]
    source_count = height * width
    context = image_regions.flatten(2)
    attention = torch.bmm(context.transpose(1, 2), words)
    attention = torch.softmax(attention, dim=2)
    attention = attention.transpose(1, 2).reshape(batch * word_count, source_count)
    attention = torch.softmax(attention * gamma1, dim=1)
    attention = attention.view(batch, word_count, source_count)
    weighted_context = torch.bmm(context, attention.transpose(1, 2))
    return weighted_context, attention.view(batch, word_count, height, width)


def word_matching_loss(
    image_regions: torch.Tensor,
    word_embeddings: torch.Tensor,
    caption_lengths: torch.Tensor,
    labels: torch.Tensor,
    class_ids: torch.Tensor | None = None,
    gamma1: float = 4.0,
    gamma2: float = 5.0,
    gamma3: float = 10.0,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    batch = image_regions.size(0)
    similarities: list[torch.Tensor] = []
    attention_maps: list[torch.Tensor] = []
    for caption_index, length_tensor in enumerate(caption_lengths):
        word_count = int(length_tensor.item())
        if word_count < 1:
            raise ValueError("Every caption must contain at least one token")
        words = word_embeddings[caption_index : caption_index + 1, :, :word_count]
        words = words.expand(batch, -1, -1).contiguous()
        weighted_context, attention = word_region_attention(words, image_regions, gamma1)
        attention_maps.append(attention[caption_index : caption_index + 1])
        word_vectors = words.transpose(1, 2)
        context_vectors = weighted_context.transpose(1, 2)
        row_similarity = F.cosine_similarity(word_vectors, context_vectors, dim=2)
        row_similarity = torch.logsumexp(row_similarity * gamma2, dim=1, keepdim=True)
        similarities.append(row_similarity)

    scores = torch.cat(similarities, dim=1) * gamma3
    class_mask = _class_collision_mask(class_ids)
    if class_mask is not None:
        scores = scores.masked_fill(class_mask, torch.finfo(scores.dtype).min)
    return (
        F.cross_entropy(scores, labels),
        F.cross_entropy(scores.transpose(0, 1), labels),
        attention_maps,
    )


def discriminator_loss(
    discriminator: MultiscaleDiscriminator,
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    sentence_features: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    batch = real_images.size(0)
    if batch < 2:
        raise ValueError("Matching-aware discriminator loss requires batch_size >= 2")
    real_features = discriminator(real_images)
    fake_features = discriminator(fake_images.detach())

    real_cond = discriminator.conditional_logits(real_features, sentence_features)
    fake_cond = discriminator.conditional_logits(fake_features, sentence_features)
    wrong_cond = discriminator.conditional_logits(real_features[:-1], sentence_features[1:])
    real_cond_loss = F.binary_cross_entropy_with_logits(real_cond, torch.ones_like(real_cond))
    fake_cond_loss = F.binary_cross_entropy_with_logits(fake_cond, torch.zeros_like(fake_cond))
    wrong_cond_loss = F.binary_cross_entropy_with_logits(wrong_cond, torch.zeros_like(wrong_cond))

    real_uncond = discriminator.unconditional_logits(real_features)
    fake_uncond = discriminator.unconditional_logits(fake_features)
    if real_uncond is not None and fake_uncond is not None:
        real_uncond_loss = F.binary_cross_entropy_with_logits(real_uncond, torch.ones_like(real_uncond))
        fake_uncond_loss = F.binary_cross_entropy_with_logits(fake_uncond, torch.zeros_like(fake_uncond))
        total = (real_uncond_loss + real_cond_loss) / 2.0
        total = total + (fake_uncond_loss + fake_cond_loss + wrong_cond_loss) / 3.0
    else:
        real_uncond_loss = real_cond_loss.new_zeros(())
        fake_uncond_loss = fake_cond_loss.new_zeros(())
        total = real_cond_loss + (fake_cond_loss + wrong_cond_loss) / 2.0
    return total, {
        "real_cond": real_cond_loss.detach(),
        "fake_cond": fake_cond_loss.detach(),
        "wrong_cond": wrong_cond_loss.detach(),
        "real_uncond": real_uncond_loss.detach(),
        "fake_uncond": fake_uncond_loss.detach(),
    }


def generator_loss(
    discriminators: Sequence[MultiscaleDiscriminator],
    images: Sequence[torch.Tensor],
    sentence_features: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    *,
    image_encoder: nn.Module | None = None,
    image_features: tuple[torch.Tensor, torch.Tensor] | None = None,
    word_embeddings: torch.Tensor | None = None,
    caption_lengths: torch.Tensor | None = None,
    class_ids: torch.Tensor | None = None,
    matching_lambda: float = 5.0,
    kl_lambda: float = 1.0,
    gamma1: float = 4.0,
    gamma2: float = 5.0,
    gamma3: float = 10.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if len(discriminators) != len(images):
        raise ValueError("Each generated resolution requires one discriminator")
    total = images[0].new_zeros(())
    metrics: dict[str, torch.Tensor] = {}
    for index, (discriminator, image) in enumerate(zip(discriminators, images, strict=True)):
        features = discriminator(image)
        cond_logits = discriminator.conditional_logits(features, sentence_features)
        scale_loss = F.binary_cross_entropy_with_logits(cond_logits, torch.ones_like(cond_logits))
        uncond_logits = discriminator.unconditional_logits(features)
        if uncond_logits is not None:
            scale_loss = scale_loss + F.binary_cross_entropy_with_logits(
                uncond_logits, torch.ones_like(uncond_logits)
            )
        total = total + scale_loss
        metrics[f"adversarial_{image.shape[-1]}"] = scale_loss.detach()

    if matching_lambda > 0:
        if image_encoder is None and image_features is None:
            raise ValueError("DAMSM matching requires image_encoder or precomputed image_features")
        if word_embeddings is None or caption_lengths is None:
            raise ValueError("DAMSM matching requires image_encoder, word_embeddings, and caption_lengths")
        labels = torch.arange(images[-1].size(0), device=images[-1].device)
        if image_features is None:
            assert image_encoder is not None
            region_features, image_code = image_encoder(images[-1])
        else:
            region_features, image_code = image_features
        word_loss_0, word_loss_1, _ = word_matching_loss(
            region_features,
            word_embeddings,
            caption_lengths,
            labels,
            class_ids,
            gamma1,
            gamma2,
            gamma3,
        )
        sentence_loss_0, sentence_loss_1 = sentence_matching_loss(
            image_code, sentence_features, labels, class_ids, gamma3
        )
        word_loss = (word_loss_0 + word_loss_1) * matching_lambda
        sentence_loss = (sentence_loss_0 + sentence_loss_1) * matching_lambda
        total = total + word_loss + sentence_loss
        metrics["word_matching"] = word_loss.detach()
        metrics["sentence_matching"] = sentence_loss.detach()

    kl = kl_loss(mu, logvar) * kl_lambda
    total = total + kl
    metrics["kl"] = kl.detach()
    return total, metrics


def kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(1.0 + logvar - mu.square() - logvar.exp())
