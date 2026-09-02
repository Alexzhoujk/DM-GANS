from __future__ import annotations

import pickle

import numpy as np
import torch

from dmgan.dae import (
    DAEGenerator,
    DAEGlobalAttention,
    load_dae_attribute_bank,
    load_dae_generator_checkpoint,
)


def test_dae_mask_broadcast_matches_legacy_for_batch_one() -> None:
    mask = torch.tensor([[False, False, True]])
    corrected = DAEGlobalAttention.flattened_mask(mask, 4, legacy=False)
    legacy = DAEGlobalAttention.flattened_mask(mask, 4, legacy=True)
    assert torch.equal(corrected, legacy)


def test_dae_mask_broadcast_fixes_mixed_length_batch_order() -> None:
    mask = torch.tensor([[False, False, False], [False, True, True]])
    corrected = DAEGlobalAttention.flattened_mask(mask, 2, legacy=False)
    legacy = DAEGlobalAttention.flattened_mask(mask, 2, legacy=True)
    expected = torch.tensor(
        [
            [False, False, False],
            [False, False, False],
            [False, True, True],
            [False, True, True],
        ]
    )
    assert torch.equal(corrected, expected)
    assert not torch.equal(corrected, legacy)


def test_dae_attention_legacy_switch_changes_only_mixed_batch() -> None:
    torch.manual_seed(4)
    attention = DAEGlobalAttention(2, 3).eval()

    one_image = torch.randn(1, 2, 2, 2)
    one_words = torch.randn(1, 3, 3)
    one_aspect = torch.randn(1, 3)
    one_mask = torch.tensor([[False, False, True]])
    corrected_one = attention(one_image, one_words, one_aspect, one_mask)[0]
    legacy_one = attention(
        one_image,
        one_words,
        one_aspect,
        one_mask,
        legacy_mask_repeat=True,
    )[0]
    torch.testing.assert_close(corrected_one, legacy_one)

    images = torch.randn(2, 2, 2, 2)
    words = torch.randn(2, 3, 3)
    aspects = torch.randn(2, 3)
    mixed_mask = torch.tensor([[False, False, False], [False, True, True]])
    corrected = attention(images, words, aspects, mixed_mask)[0]
    legacy = attention(
        images,
        words,
        aspects,
        mixed_mask,
        legacy_mask_repeat=True,
    )[0]
    assert not torch.allclose(corrected, legacy)


def test_dae_generator_checkpoint_roundtrip_and_output_sizes(tmp_path) -> None:
    generator = DAEGenerator(
        noise_dim=5,
        text_dim=8,
        condition_dim=4,
        channels=4,
        residual_blocks=1,
    ).eval()
    checkpoint = tmp_path / "netG.pth"
    torch.save(generator.state_dict(), checkpoint)
    restored = DAEGenerator(
        noise_dim=5,
        text_dim=8,
        condition_dim=4,
        channels=4,
        residual_blocks=1,
    ).eval()
    load_dae_generator_checkpoint(restored, checkpoint, strict=True)

    noise = torch.randn(1, 5)
    sentence = torch.randn(1, 8)
    words = torch.randn(1, 8, 3)
    aspects = torch.randn(1, 8, 3)
    mask = torch.tensor([[False, False, True]])
    with torch.inference_mode():
        images, diagnostics, mu, logvar = restored(
            noise,
            sentence,
            words,
            aspects,
            mask,
            sample_conditioning=False,
        )

    assert [tuple(image.shape) for image in images] == [
        (1, 3, 64, 64),
        (1, 3, 128, 128),
        (1, 3, 128, 128),
        (1, 3, 256, 256),
    ]
    assert set(diagnostics) == {
        "attention_128_global",
        "attention_128_local",
        "attention_256",
    }
    assert mu.shape == logvar.shape == (1, 4)
    assert torch.isfinite(images[-1]).all()


def test_load_dae_attribute_bank_preserves_caption_payload(tmp_path) -> None:
    caption_payload = [[1], [2], {0: "<end>"}, {"<end>": 0}]
    train_attributes = [[[1, 2]]]
    test_attributes = [
        [[4, 5, 6], [7], [8, 9, 10, 11, 12, 13], [99]],
        [],
    ]
    metadata = tmp_path / "captions.pickle"
    with metadata.open("wb") as stream:
        pickle.dump([*caption_payload, train_attributes, test_attributes], stream)

    bank, returned_payload = load_dae_attribute_bank(metadata)
    assert returned_payload == tuple(caption_payload)
    assert bank.shape == (2, 3, 5)
    np.testing.assert_array_equal(bank[0, 0], [4, 5, 6, 0, 0])
    np.testing.assert_array_equal(bank[0, 1], [7, 0, 0, 0, 0])
    np.testing.assert_array_equal(bank[0, 2], [8, 9, 10, 11, 12])
    np.testing.assert_array_equal(bank[1], np.zeros((3, 5), dtype=np.int64))
