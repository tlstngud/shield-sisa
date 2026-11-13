#!/usr/bin/env python3
"""Export the trusted SHIELD-A checkpoint to the native engine's FP32 format.

Run with python3. No Python is needed by the
engine after this one-time conversion. This utility never changes reference/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True
sys.path[:0] = [str(ROOT / "reference"), str(ROOT / "reference/third_party/sisa/src")]

import numpy as np
import torch
from shield_a.model import ShieldA, ShieldConfig


def load_model(checkpoint: Path):
    model = ShieldA(ShieldConfig(gdn_impl="torch", sisa_impl="torch"))
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["model"]
    model.load_state_dict(state, strict=True)
    return model.float().eval()


def export(model, output: Path):
    tensors = dict(model.state_dict())
    tensors.update({"frontend.window": model.frontend.window, "frontend.fb": model.frontend.fb})
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary=output.with_name(output.name+f'.tmp.{os.getpid()}')
    with temporary.open("wb") as f:
        f.write(b"SHLDA001")
        f.write(struct.pack("<I", len(tensors)))
        for name, tensor in tensors.items():
            encoded = name.encode("utf-8")
            values = tensor.detach().cpu().numpy().astype("<f4", copy=False)
            f.write(struct.pack("<I", len(encoded)))
            f.write(encoded)
            f.write(struct.pack("<I", values.ndim))
            f.write(struct.pack("<" + "I" * values.ndim, *values.shape))
            f.write(struct.pack("<Q", values.size))
            f.write(values.tobytes(order="C"))
    temporary.replace(output)
    return {"tensor_count": len(tensors), "bytes": output.stat().st_size,
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}


@torch.inference_mode()
def fixture(model, directory: Path, input_path: Path | None, index: int, seed: int):
    directory.mkdir(parents=True, exist_ok=True)
    if input_path:
        if input_path.suffix == ".npy":
            array = np.load(input_path, mmap_mode="r")
            pcm = np.array(array if array.ndim == 1 else array[index], copy=True)
        else:
            pcm = np.fromfile(input_path, dtype="<f4")
        if pcm.dtype == np.int16:
            pcm = pcm.astype(np.float32) / 32768.0
        pcm = np.asarray(pcm, dtype=np.float32)
    else:
        pcm = np.random.default_rng(seed).normal(0, 0.1, 160000).astype(np.float32)
    if pcm.shape != (160000,):
        raise ValueError(f"Expected one 160000-sample window, got {pcm.shape}")
    values = {"input": torch.from_numpy(pcm[None])}
    values["mel"] = model.frontend(values["input"])
    values["cnn"] = model.cnn(values["mel"])
    values["tokens"] = torch.cat([values["cnn"], model.readout], dim=1)
    values["gdn_a"] = model.gdn_a(values["tokens"])
    values["gdn_b"] = model.gdn_b(values["gdn_a"])
    values["sisa_last"] = model.sisa(values["gdn_b"])[:, -1]
    values["logit"] = model.head(model.head_norm(values["sisa_last"])).squeeze(-1)
    for name, value in values.items():
        value.detach().numpy().astype("<f4").tofile(directory / f"{name}.f32")
    meta = {"shapes": {name: list(t.shape) for name, t in values.items()},
            "logit": values["logit"].item(), "source": str(input_path) if input_path else "seeded Gaussian",
            "index": index, "seed": seed}
    (directory / "fixture.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=Path(os.environ.get('SHIELD_CHECKPOINT',ROOT / "reference/ckpt/best.pt")))
    p.add_argument("--output", type=Path, default=ROOT / "build/shield.weights")
    p.add_argument("--fixture", type=Path, help="Optional directory for raw intermediate reference tensors")
    p.add_argument("--input", type=Path, help="NPY int16 windows or one raw float32 window")
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=1)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    model = load_model(args.checkpoint)
    meta = export(model, args.output)
    meta.update(checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(), output=str(args.output.resolve()))
    if args.fixture:
        meta["fixture"] = fixture(model, args.fixture, args.input, args.index, args.seed)
    args.output.with_suffix(args.output.suffix + ".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
