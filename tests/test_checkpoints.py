from dmgan.checkpoints import _generator_key


def test_official_generator_key_mapping() -> None:
    assert _generator_key("ca_net.fc.weight") == "ca.projection.0.weight"
    assert _generator_key("h_net1.upsample4.2.running_mean") == "initial.upsample.3.2.running_mean"
    assert _generator_key("h_net2.A.weight") == "refine_128.memory.word_gate.weight"
    assert _generator_key("h_net3.response_gate.0.bias") == "refine_256.memory.response.0.bias"
    assert _generator_key("img_net3.img.0.weight") == "to_image_256.head.0.weight"
