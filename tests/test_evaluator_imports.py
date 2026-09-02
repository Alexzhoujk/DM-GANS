from __future__ import annotations

import importlib


def test_checkpoint_evaluators_are_importable_as_modules() -> None:
    for module_name in (
        "scripts.evaluate_contrastive_checkpoint",
        "scripts.evaluate_dae_checkpoint",
        "scripts.evaluate_successor_checkpoint",
    ):
        module = importlib.import_module(module_name)
        assert callable(module.main)
