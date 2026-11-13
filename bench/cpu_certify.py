#!/usr/bin/env python3
"""Certify approximate CPU modes and write "<weights>.cpu.cert" (owned by cpu_model).

For each mode: score all 840 reference crops with the experimental library (shv_cpu_load_experimental), evaluate with
the oracle's metric code (bench/metrics.py, baseline = oracle cpu_f32 logits, strict per-method checks), and, if it
passes, record "<weights sha256> <mode> <shv_cpu_revision()> threshold=<repr>" (SHIELD_V_CPU_CERT_V2; the threshold the
metrics were computed at) in the certificate file.  Modes that fail are listed
in the JSON summary and are NOT written, so the production loader keeps refusing them.  Afterwards the production
library is loaded in every certified mode and must reproduce the experimental logits bit for bit (every 10th sample).

  python bench/cpu_certify.py --modes bf16-cnn,bf16-seq,bf16
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench.cpu_engine import CpuModel  # noqa: E402
from bench import metrics as M  # noqa: E402
from oracle import shieldv_ref as R  # noqa: E402


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def score(model, rows, batch=4):
    out = {}
    for i in range(0, len(rows), batch):
        ch = rows[i:i + batch]
        lg = model.predict(np.stack([R.load_crop(r["sample_id"]) for r in ch]))
        for r, v in zip(ch, lg):
            out[r["sample_id"]] = float(v) if math.isfinite(float(v)) else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default="bf16-cnn,bf16-seq,bf16")
    ap.add_argument("--weights", default=str(ROOT / "build/shield_v.weights"))
    ap.add_argument("--exp-lib", default=str(ROOT / "build/cpu_exp/libshield_v_cpu.so"))
    ap.add_argument("--lib", default=str(ROOT / "build/cpu/libshield_v_cpu.so"))
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out-dir", default=str(ROOT / "bench/artifacts/cpu"))
    a = ap.parse_args()
    rows = R.read_map()
    base = M.read_values(ROOT / "bench/artifacts/oracle_logits.jsonl", "logits.cpu_f32")
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"weights": a.weights, "weights_sha256": sha256(a.weights), "exp_lib_sha256": sha256(a.exp_lib),
               "lib_sha256": sha256(a.lib), "time": time.strftime("%F %T"), "modes": {}}
    lines = []
    revision = None
    for mode in [m for m in a.modes.split(",") if m]:
        m = CpuModel(threads=a.threads, mode=mode, lib=a.exp_lib, weights=a.weights, experimental=True)
        revision = m.lib.shv_cpu_revision().decode()
        assert m.fingerprint == summary["weights_sha256"]
        vals = score(m, rows)
        m.close()
        (out_dir / f"{mode}_logits.jsonl").write_text("".join(json.dumps({"sample_id": k, "logit": v}) + "\n"
                                                               for k, v in vals.items()))
        res = M.evaluate(vals, baseline=base, strict_methods=True)
        res["weights_sha256"] = summary["weights_sha256"]
        res["mode"], res["revision"], res["lib_sha256"] = mode, revision, summary["exp_lib_sha256"]
        (out_dir / f"cert_{mode}.json").write_text(json.dumps(res, indent=1))
        (out_dir / f"cert_{mode}.md").write_text(M.markdown(res, f"cpu {mode} vs oracle cpu_f32 (strict methods)") + "\n")
        summary["modes"][mode] = {"verdict": res["verdict"], "revision": revision}
        print(mode, res["verdict"], flush=True)
        if res["verdict"] == "PASS":
            # V2: the line also binds the threshold the metrics were computed at (exact double, repr round-trips)
            lines.append(f"{summary['weights_sha256']} {mode} {revision} threshold={float(res['threshold'])!r}")
    cert = Path(a.weights + ".cpu.cert")
    tmp = cert.with_name(cert.name + f".tmp.{os.getpid()}")
    tmp.write_text("SHIELD_V_CPU_CERT_V2\n" + "".join(l + "\n" for l in lines))
    tmp.replace(cert)
    summary["cert_file"] = str(cert)
    summary["cert_lines"] = lines
    # production loader: certified modes load and reproduce the experimental logits bitwise
    for mode in summary["modes"]:
        try:
            pm = CpuModel(threads=a.threads, mode=mode, lib=a.lib, weights=a.weights)
        except RuntimeError as e:
            summary["modes"][mode]["production_load"] = f"refused: {e}"
            continue
        exp = M.read_values(out_dir / f"{mode}_logits.jsonl", "logit")
        sub = rows[::10]
        got = score(pm, sub)
        pm.close()
        same = sum(1 for r in sub if got[r["sample_id"]] == exp[r["sample_id"]])
        summary["modes"][mode]["production_load"] = "ok"
        summary["modes"][mode]["production_bitwise_equal"] = f"{same}/{len(sub)}"
    (out_dir / "cert_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
