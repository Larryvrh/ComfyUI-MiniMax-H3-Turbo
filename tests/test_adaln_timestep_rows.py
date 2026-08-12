"""Check that pruned-base LoRA timesteps match H3's packed layout."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


PACKAGE_DIR = Path(__file__).resolve().parents[1]
COMFY_DIR = PACKAGE_DIR.parents[1]
sys.path.insert(0, str(COMFY_DIR))

import comfy.cli_args

comfy.cli_args.args.cpu = True

spec = importlib.util.spec_from_file_location(
    "h3turbo_timestep_test", PACKAGE_DIR / "__init__.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def layout(*kinds):
    return SimpleNamespace(segments=[(0, 1, kind) for kind in kinds])


def main():
    timestep = torch.tensor([1000.0])

    assert len(module._unique_t(
        timestep, 12.0, 3.0, {"layout": layout("audio", "video")})) == 1
    assert len(module._unique_t(
        timestep, 12.0, 3.0, {"layout": layout("cond", "audio", "video")})) == 2
    assert len(module._unique_t(
        timestep, 12.0, 3.0,
        {"layout": layout("ref_audio", "audio", "video")})) == 2
    assert len(module._unique_t(
        timestep, 12.0, 3.0,
        {"layout": layout("cond", "ref_audio", "audio", "video")})) == 3
    assert len(module._unique_t(
        timestep, 12.0, 3.0,
        {"keyframes": [{}], "refs": [{"kind": "audio", "ref_audio_t": 1}]})) == 3
    assert len(module._unique_t(
        timestep, 12.0, 3.0,
        {"refs": [{"kind": "audio", "ref_audio_t": 0}]})) == 1
    print("AdaLN timestep rows match visual and audio conditioning")


if __name__ == "__main__":
    main()
