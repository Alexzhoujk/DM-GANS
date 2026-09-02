from pathlib import Path

import pytest
import torch

from dmgan.checkpoints import _generator_key, load_generator_checkpoint
from dmgan.models import DMGenerator


def test_official_generator_key_mapping() -> None:
    assert _generator_key("ca_net.fc.weight") == "ca.projection.0.weight"
    assert _generator_key("h_net1.upsample4.2.running_mean") == "initial.upsample.3.2.running_mean"
    assert _generator_key("h_net2.A.weight") == "refine_128.memory.word_gate.weight"
    assert _generator_key("h_net3.response_gate.0.bias") == "refine_256.memory.response.0.bias"
    assert _generator_key("img_net3.img.0.weight") == "to_image_256.head.0.weight"


def _modern_checkpoint(path: Path, *, include_ema: bool = True) -> tuple[str, str]:
    generator = DMGenerator(channels=4, memory_dim=8, residual_blocks=1)
    raw = {key: value.detach().clone() for key, value in generator.state_dict().items()}
    shadow = {
        key: value.detach().clone() for key, value in generator.named_parameters()
    }
    parameter_name = next(iter(shadow))
    buffer_name = next(key for key in raw if key not in shadow)
    raw[parameter_name].fill_(1.0)
    raw[buffer_name].fill_(3.0)
    shadow[parameter_name].fill_(2.0)
    checkpoint: dict[str, object] = {"generator": raw}
    if include_ema:
        checkpoint["ema"] = {"decay": 0.999, "shadow": shadow}
    torch.save(checkpoint, path)
    return parameter_name, buffer_name


def test_modern_checkpoint_auto_prefers_ema_and_preserves_raw_buffers(tmp_path: Path) -> None:
    checkpoint = tmp_path / "trainer.pt"
    parameter_name, buffer_name = _modern_checkpoint(checkpoint)
    generator, resolved = load_generator_checkpoint(
        DMGenerator(channels=4, memory_dim=8, residual_blocks=1), checkpoint
    )

    assert resolved == "modern-ema"
    state = generator.state_dict()
    assert torch.all(state[parameter_name] == 2.0)
    assert torch.all(state[buffer_name] == 3.0)


def test_modern_checkpoint_can_select_raw_state(tmp_path: Path) -> None:
    checkpoint = tmp_path / "trainer.pt"
    parameter_name, _ = _modern_checkpoint(checkpoint)
    generator, resolved = load_generator_checkpoint(
        DMGenerator(channels=4, memory_dim=8, residual_blocks=1),
        checkpoint,
        checkpoint_format="modern-raw",
    )

    assert resolved == "modern-raw"
    assert torch.all(generator.state_dict()[parameter_name] == 1.0)


def test_modern_checkpoint_auto_falls_back_to_raw_without_ema(tmp_path: Path) -> None:
    checkpoint = tmp_path / "trainer.pt"
    parameter_name, _ = _modern_checkpoint(checkpoint, include_ema=False)
    generator, resolved = load_generator_checkpoint(
        DMGenerator(channels=4, memory_dim=8, residual_blocks=1), checkpoint
    )

    assert resolved == "modern-raw"
    assert torch.all(generator.state_dict()[parameter_name] == 1.0)


def test_explicit_modern_ema_requires_shadow(tmp_path: Path) -> None:
    checkpoint = tmp_path / "trainer.pt"
    _modern_checkpoint(checkpoint, include_ema=False)

    with pytest.raises(ValueError, match="ema.shadow"):
        load_generator_checkpoint(
            DMGenerator(channels=4, memory_dim=8, residual_blocks=1),
            checkpoint,
            checkpoint_format="modern-ema",
        )


def test_modern_ema_rejects_partial_shadow(tmp_path: Path) -> None:
    checkpoint = tmp_path / "trainer.pt"
    _modern_checkpoint(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload["ema"]["shadow"].pop(next(iter(payload["ema"]["shadow"])))
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError, match="missing generator parameters"):
        load_generator_checkpoint(
            DMGenerator(channels=4, memory_dim=8, residual_blocks=1),
            checkpoint,
            checkpoint_format="modern-ema",
        )
