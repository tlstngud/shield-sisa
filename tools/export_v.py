#!/usr/bin/env python3
"""Export a trusted SHIELD-V checkpoint to the native engine's FP32 weight file.

Format "SHLDV001" (same layout as the audio engine's "SHLDA001"):
  magic[8] | u32 count | count x { u32 name_len | name utf-8 | u32 rank | u32 dims[rank] | u64 elements | f32 data[elements] }
All little-endian, C order. Every parameter and every buffer (persistent or not) of ShieldV is written as float32,
except BatchNorm num_batches_tracked. Nothing is folded or fused here; the engine does that at load time.
A sidecar <output>.json records the checkpoint SHA-256, threshold, model id, tensor table and output SHA-256.

Run: python tools/export_v.py --model-src PATH_TO_SHIELD_V_PACKAGE_PARENT --checkpoint best.pt --output build/shield_v.weights
The SHIELD-V model code (package `shield_v`) and checkpoint come from the SHIELD-V model release; they are not vendored here.
Most users do not need this: the exported weights are published on Hugging Face.
"""
import argparse
import hashlib
import json
import os
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
_src = next((sys.argv[i + 1] for i, a in enumerate(sys.argv[:-1]) if a == "--model-src"), os.environ.get("SHIELD_V_MODEL_SRC", ""))
if _src:
    sys.path.insert(0, _src)

import torch  # noqa: E402
from shield_v.model import ShieldV  # noqa: E402

DEFAULT_CKPT = os.path.join(os.environ.get("SHIELD_V_MODEL_SRC", "."), "best.pt")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-src", default=None, help="directory containing the shield_v package (or $SHIELD_V_MODEL_SRC)")
    ap.add_argument("--checkpoint", default=os.environ.get("SHIELD_V_CHECKPOINT", DEFAULT_CKPT))
    ap.add_argument("--output", default=str(ROOT / "build/shield_v.weights"))
    a = ap.parse_args()
    ckpt, out = Path(a.checkpoint).resolve(), Path(a.output).resolve()
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    model = ShieldV(pretrained=False)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    tensors = {}
    for name, p in model.named_parameters():
        tensors[name] = p
    for name, b in model.named_buffers():
        if not name.endswith("num_batches_tracked"):
            tensors[name] = b
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + f".tmp.{os.getpid()}")
    table = []
    with tmp.open("wb") as f:
        f.write(b"SHLDV001")
        f.write(struct.pack("<I", len(tensors)))
        for name, t in tensors.items():
            v = t.detach().cpu().float().contiguous().numpy().astype("<f4", copy=False)
            e = name.encode()
            f.write(struct.pack("<I", len(e))); f.write(e)
            f.write(struct.pack("<I", v.ndim)); f.write(struct.pack("<" + "I" * v.ndim, *v.shape))
            f.write(struct.pack("<Q", v.size)); f.write(v.tobytes(order="C"))
            table.append({"name": name, "shape": list(v.shape)})
    tmp.replace(out)
    meta = {
        "format": "SHLDV001", "checkpoint": str(ckpt), "checkpoint_sha256": sha256(ckpt),
        "threshold": float(state["threshold"]), "manifest_sha256": state["config"].get("manifest_sha256"),
        "parameters": model.parameter_counts()["total"], "tensor_count": len(tensors),
        "output": str(out), "output_bytes": out.stat().st_size, "output_sha256": sha256(out), "tensors": table,
    }
    out.with_name(out.name + ".json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: meta[k] for k in ("checkpoint_sha256", "threshold", "parameters", "tensor_count",
                                             "output_bytes", "output_sha256")}))


if __name__ == "__main__":
    main()
