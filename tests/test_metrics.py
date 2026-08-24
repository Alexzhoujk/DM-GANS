import numpy as np
import pytest
import torch

from dmgan.metrics import frechet_distance, inception_score, r_precision, split_mean_std


def test_frechet_distance_is_zero_for_identical_statistics() -> None:
    mean = np.array([0.2, -0.1])
    covariance = np.array([[1.0, 0.1], [0.1, 0.5]])
    assert frechet_distance(mean, covariance, mean, covariance) == pytest.approx(0.0, abs=1e-8)


def test_uniform_inception_predictions_have_score_one() -> None:
    probabilities = np.full((20, 4), 0.25)
    mean, std = inception_score(probabilities, splits=5, seed=7)
    assert mean == pytest.approx(1.0)
    assert std == pytest.approx(0.0)


def test_r_precision_selects_correct_caption() -> None:
    images = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    correct = images.clone()
    negatives = torch.tensor([[[0.0, 1.0]], [[1.0, 0.0]]])
    assert r_precision(images, correct, negatives).tolist() == [True, True]


def test_split_mean_std_reports_fraction() -> None:
    mean, std = split_mean_std(np.array([1, 1, 0, 0], dtype=np.float64), splits=2, seed=0)
    assert mean == pytest.approx(0.5)
    assert std >= 0.0
