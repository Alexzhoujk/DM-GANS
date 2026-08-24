"""Evaluation metrics used by the fixed CUB baseline protocol."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class InceptionEvaluator(nn.Module):
    """One Inception-v3 trunk for repository-compatible FID and PyTorch IS.

    The FID preprocessing intentionally reproduces the transformation in the
    DM-GAN repository's bundled PyTorch-FID evaluator. The ImageNet IS path uses
    standard torchvision normalization and is therefore not comparable with
    the paper's legacy 50-class TensorFlow bird classifier.
    """

    def __init__(self) -> None:
        super().__init__()
        from torchvision import models

        weights = models.Inception_V3_Weights.IMAGENET1K_V1
        model = models.inception_v3(weights=weights, aux_logits=True, transform_input=False)
        self.Conv2d_1a_3x3 = model.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = model.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = model.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = model.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = model.Conv2d_4a_3x3
        self.Mixed_5b = model.Mixed_5b
        self.Mixed_5c = model.Mixed_5c
        self.Mixed_5d = model.Mixed_5d
        self.Mixed_6a = model.Mixed_6a
        self.Mixed_6b = model.Mixed_6b
        self.Mixed_6c = model.Mixed_6c
        self.Mixed_6d = model.Mixed_6d
        self.Mixed_6e = model.Mixed_6e
        self.Mixed_7a = model.Mixed_7a
        self.Mixed_7b = model.Mixed_7b
        self.Mixed_7c = model.Mixed_7c
        self.dropout = model.dropout
        self.fc = model.fc
        self.register_buffer("imagenet_mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer("imagenet_std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def _trunk(self, image: torch.Tensor) -> torch.Tensor:
        image = self.Conv2d_1a_3x3(image)
        image = self.Conv2d_2a_3x3(image)
        image = self.Conv2d_2b_3x3(image)
        image = F.max_pool2d(image, kernel_size=3, stride=2)
        image = self.Conv2d_3b_1x1(image)
        image = self.Conv2d_4a_3x3(image)
        image = F.max_pool2d(image, kernel_size=3, stride=2)
        image = self.Mixed_5b(image)
        image = self.Mixed_5c(image)
        image = self.Mixed_5d(image)
        image = self.Mixed_6a(image)
        image = self.Mixed_6b(image)
        image = self.Mixed_6c(image)
        image = self.Mixed_6d(image)
        image = self.Mixed_6e(image)
        image = self.Mixed_7a(image)
        image = self.Mixed_7b(image)
        return self.Mixed_7c(image)

    def fid_features(self, image_01: torch.Tensor) -> torch.Tensor:
        """Return 2048-D features matching the repository PyTorch-FID path."""

        image = F.interpolate(image_01, size=(299, 299), mode="bilinear", align_corners=True)
        image = image.clone()
        image[:, 0] = image[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
        image[:, 1] = image[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
        image[:, 2] = image[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        return F.adaptive_avg_pool2d(self._trunk(image), 1).flatten(1)

    def imagenet_probabilities(self, image_01: torch.Tensor) -> torch.Tensor:
        """Return ImageNet probabilities for a modern, non-paper-comparable IS."""

        image = F.interpolate(image_01, size=(299, 299), mode="bilinear", align_corners=False)
        image = (image - self.imagenet_mean) / self.imagenet_std
        pooled = F.adaptive_avg_pool2d(self._trunk(image), 1).flatten(1)
        return torch.softmax(self.fc(self.dropout(pooled)), dim=1)


def activation_statistics(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or features.shape[0] < 2:
        raise ValueError("features must have shape [N,D] with N >= 2")
    return features.mean(axis=0), np.cov(features, rowvar=False)


def frechet_distance(
    mean_a: np.ndarray,
    covariance_a: np.ndarray,
    mean_b: np.ndarray,
    covariance_b: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """Compute the Fréchet distance used by FID."""

    from scipy import linalg

    mean_a = np.atleast_1d(mean_a).astype(np.float64, copy=False)
    mean_b = np.atleast_1d(mean_b).astype(np.float64, copy=False)
    covariance_a = np.atleast_2d(covariance_a).astype(np.float64, copy=False)
    covariance_b = np.atleast_2d(covariance_b).astype(np.float64, copy=False)
    if mean_a.shape != mean_b.shape:
        raise ValueError("FID mean vectors have different shapes")
    if covariance_a.shape != covariance_b.shape:
        raise ValueError("FID covariance matrices have different shapes")
    difference = mean_a - mean_b
    product = covariance_a.dot(covariance_b)
    try:
        result = linalg.sqrtm(product, disp=False)
    except TypeError:
        result = linalg.sqrtm(product)
    covariance_mean = result[0] if isinstance(result, tuple) else result
    if not np.isfinite(covariance_mean).all():
        offset = np.eye(covariance_a.shape[0]) * eps
        product = (covariance_a + offset).dot(covariance_b + offset)
        try:
            result = linalg.sqrtm(product, disp=False)
        except TypeError:
            result = linalg.sqrtm(product)
        covariance_mean = result[0] if isinstance(result, tuple) else result
    if np.iscomplexobj(covariance_mean):
        if not np.allclose(np.diagonal(covariance_mean).imag, 0, atol=1e-3):
            raise ValueError(f"FID covariance square root has imaginary magnitude {np.abs(covariance_mean.imag).max()}")
        covariance_mean = covariance_mean.real
    value = (
        difference.dot(difference)
        + np.trace(covariance_a)
        + np.trace(covariance_b)
        - 2.0 * np.trace(covariance_mean)
    )
    return float(value)


def fid_from_features(
    generated_features: np.ndarray,
    reference_mean: np.ndarray,
    reference_covariance: np.ndarray,
) -> float:
    generated_mean, generated_covariance = activation_statistics(generated_features)
    return frechet_distance(generated_mean, generated_covariance, reference_mean, reference_covariance)


def inception_score(
    probabilities: np.ndarray,
    splits: int = 10,
    seed: int = 0,
) -> tuple[float, float]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[0] < splits:
        raise ValueError("probabilities must have shape [N,C] with N >= splits")
    if np.any(probabilities < 0):
        raise ValueError("probabilities must be non-negative")
    row_sums = probabilities.sum(axis=1, keepdims=True)
    probabilities = probabilities / np.clip(row_sums, 1e-12, None)
    order = np.random.default_rng(seed).permutation(probabilities.shape[0])
    scores: list[float] = []
    for part in np.array_split(probabilities[order], splits):
        marginal = part.mean(axis=0, keepdims=True)
        kl = part * (np.log(np.clip(part, 1e-12, None)) - np.log(np.clip(marginal, 1e-12, None)))
        scores.append(float(np.exp(kl.sum(axis=1).mean())))
    return float(np.mean(scores)), float(np.std(scores))


def r_precision(
    image_embeddings: torch.Tensor,
    correct_embeddings: torch.Tensor,
    negative_embeddings: torch.Tensor,
) -> torch.Tensor:
    """Return per-sample 1-vs-N top-1 matches for DAMSM R-precision."""

    if image_embeddings.ndim != 2 or correct_embeddings.shape != image_embeddings.shape:
        raise ValueError("image and correct embeddings must have matching [B,D] shapes")
    if negative_embeddings.ndim != 3 or negative_embeddings.shape[0] != image_embeddings.shape[0]:
        raise ValueError("negative embeddings must have shape [B,N,D]")
    candidates = torch.cat([correct_embeddings[:, None, :], negative_embeddings], dim=1)
    image_embeddings = F.normalize(image_embeddings, dim=1)
    candidates = F.normalize(candidates, dim=2)
    scores = torch.einsum("bd,bnd->bn", image_embeddings, candidates)
    return scores.argmax(dim=1).eq(0)


def split_mean_std(values: np.ndarray, splits: int = 10, seed: int = 0) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size < splits:
        raise ValueError("values must be one-dimensional with N >= splits")
    order = np.random.default_rng(seed).permutation(values.size)
    means = [float(part.mean()) for part in np.array_split(values[order], splits)]
    return float(np.mean(means)), float(np.std(means))
