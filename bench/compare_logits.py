#!/usr/bin/env python3
"""Compare two logit sets sample by sample: max / p99.9 / mean |diff|, and decision flips at the threshold.

Each side is FILE or FILE:FIELD (FIELD is a dot path, default "logit"; e.g.
bench/artifacts/oracle_logits.jsonl:logits.cpu_f64). Rows are joined by sample_id over the map; a sample missing
or non-finite on either side is reported separately and excluded from the statistics.
Decision: sigmoid(logit) in double >= threshold of build/shield_v.weights.json (use KIND=score via --kind-a/-b
for score files).

Python API: from bench.compare_logits import compare; compare(a: dict, b: dict, ...) -> dict
CLI: python bench/compare_logits.py A B [--kind-a logit|score] [--kind-b ...] [--out-json X] [--top 10]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.metrics import WEIGHTS_JSON, read_map, read_values, sigmoid64  # noqa: E402


def parse_spec(spec: str):
    if ":" in spec and not Path(spec).exists():
        path, field = spec.rsplit(":", 1)
    else:
        path, field = spec, "logit"
    return path, field


def _dist(d: np.ndarray) -> dict:
    if d.size == 0:
        return {"n": 0}
    ad = np.abs(d)
    return {"n": int(d.size), "max_abs": float(ad.max()), "p99.9_abs": float(np.percentile(ad, 99.9)),
            "p99_abs": float(np.percentile(ad, 99)), "mean_abs": float(ad.mean()), "mean_signed": float(d.mean()),
            "rms": float(np.sqrt((d ** 2).mean()))}


def compare(a: dict, b: dict, kind_a="logit", kind_b="logit", map_rows=None, threshold=None, top=10) -> dict:
    map_rows = map_rows or read_map()
    thr = float(threshold if threshold is not None else json.loads(WEIGHTS_JSON.read_text())["threshold"])

    def ok(v):
        return v is not None and math.isfinite(v)

    def dec(v, kind):
        return (v >= thr) if kind == "score" else (sigmoid64(v) >= thr)

    res = {"threshold": thr, "kind_a": kind_a, "kind_b": kind_b, "groups": {}}
    missing = [r["sample_id"] for r in map_rows if not ok(a.get(r["sample_id"])) or not ok(b.get(r["sample_id"]))]
    res["n_missing_or_invalid"] = len(missing)
    res["missing_or_invalid"] = missing[:50]
    rows = [r for r in map_rows if r["sample_id"] not in set(missing)]
    groups = {"all": rows}
    for s in ("val", "test"):
        groups[s] = [r for r in rows if r["split"] == s]
    for name, sub in groups.items():
        d = np.array([a[r["sample_id"]] - b[r["sample_id"]] for r in sub], dtype=np.float64)
        flips = [r["sample_id"] for r in sub if dec(a[r["sample_id"]], kind_a) != dec(b[r["sample_id"]], kind_b)]
        g = _dist(d)
        g["flips"] = len(flips)
        g["flip_ids"] = flips
        g["flips_real"] = sum(1 for r in sub if r["sample_id"] in set(flips) and r["label"] == 0)
        g["flips_fake"] = g["flips"] - g["flips_real"]
        res["groups"][name] = g
    order = sorted(rows, key=lambda r: -abs(a[r["sample_id"]] - b[r["sample_id"]]))[:top]
    res["largest"] = [{"sample_id": r["sample_id"], "split": r["split"], "label": r["label"], "a": a[r["sample_id"]],
                       "b": b[r["sample_id"]], "diff": a[r["sample_id"]] - b[r["sample_id"]]} for r in order]
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--kind-a", choices=["logit", "score"], default="logit")
    ap.add_argument("--kind-b", choices=["logit", "score"], default="logit")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--out-json")
    x = ap.parse_args(argv)
    pa, fa = parse_spec(x.a)
    pb, fb = parse_spec(x.b)
    res = compare(read_values(pa, fa), read_values(pb, fb), x.kind_a, x.kind_b, top=x.top)
    res["a"], res["b"] = f"{pa}:{fa}", f"{pb}:{fb}"
    if x.out_json:
        Path(x.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(x.out_json).write_text(json.dumps(res, indent=1))
    print(f"A = {res['a']}\nB = {res['b']}  (invalid/missing: {res['n_missing_or_invalid']})")
    print("| group | n | max abs | p99.9 abs | mean abs | mean (A-B) | flips (Real/Fake) |")
    print("|---|---|---|---|---|---|---|")
    for name, g in res["groups"].items():
        if g["n"]:
            print(f"| {name} | {g['n']} | {g['max_abs']:.3e} | {g['p99.9_abs']:.3e} | {g['mean_abs']:.3e} | "
                  f"{g['mean_signed']:+.3e} | {g['flips']} ({g['flips_real']}/{g['flips_fake']}) |")
    if res["groups"]["all"]["flip_ids"]:
        print("flipped:", ", ".join(res["groups"]["all"]["flip_ids"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
