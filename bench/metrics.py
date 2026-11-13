#!/usr/bin/env python3
"""SHIELD-V metrics and certification (the oracle's metric code; every other agent uses this).

Input: a JSONL file with one row per sample: "sample_id" plus a numeric value (a logit by default) at --field
(a dot path, e.g. "logit" or "logits.cpu_f32"; null/NaN/missing = not valid). Labels, split, method and
source_group are taken from the map ($SHIELD_DATA_DIR/shieldv_codec/maps/orig.jsonl), never from the file.

Per split (val, test) and per fake method (inswapper, uniface: all Reals of the split vs that method's Fakes):
  AUROC (Mann-Whitney, ties = 1/2) with a source_group cluster-bootstrap 95% CI (1000 resamples, seed 12345:
  groups of the subset drawn with replacement, every sample weighted by its group's multiplicity; percentile CI);
  Real FPR and Fake TPR at the threshold of build/shield_v.weights.json (decision: sigmoid(logit) in double >= thr;
  with --kind score the value is compared directly).
With --baseline (exact baseline, e.g. bench/artifacts/oracle_logits.jsonl --baseline-field logits.cpu_f32):
  deltas, paired bootstrap CI of the AUROC difference (same resamples), decision flips, and the BRIEF.md
  certification verdict: per split AUROC drop <= 0.002, Real FPR increase <= 1 pp, Fake TPR drop <= 1 pp, and all 840
  samples valid. Per-method checks are evaluated and reported (field per_method_pass) but the verdict is per split
  unless --strict-methods.

Python API:  from bench.metrics import evaluate;  evaluate(values: dict sample_id -> float|None, kind="logit",
             baseline=None, baseline_kind="logit", ...) -> dict
CLI:  python bench/metrics.py --logits FILE [--field logit] [--kind logit|score] [--baseline FILE
      --baseline-field F --baseline-kind K] [--name NAME] [--out-json X] [--out-md Y]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MAP = Path(os.environ.get("SHIELD_DATA_DIR", "data")) / "shieldv_codec/maps/orig.jsonl"
WEIGHTS_JSON = ROOT / "build/shield_v.weights.json"
SPLITS = ("val", "test")
METHODS = ("inswapper", "uniface")
N_BOOT, SEED = 1000, 12345
LIMITS = {"auroc_drop": 0.002, "fpr_increase": 0.01, "tpr_drop": 0.01}
TOL = 1e-12


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_map(path=MAP) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def get_path(obj, path):
    for part in path.split("."):
        if not isinstance(obj, dict) or part not in obj:
            return None
        obj = obj[part]
    return obj


def read_values(path, field="logit") -> dict:
    out = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        v = get_path(row, field)
        out[row["sample_id"]] = None if v is None else float(v)
    return out


def sigmoid64(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def decisions(vals: np.ndarray, kind: str, thr: float) -> np.ndarray:
    if kind == "score":
        return vals >= thr
    return np.array([sigmoid64(float(v)) >= thr for v in vals], dtype=bool)


def pair_matrix(fake: np.ndarray, real: np.ndarray) -> np.ndarray:
    """C[i, j] = 1 if fake_i > real_j, 1/2 if equal, else 0."""
    f, r = fake[:, None], real[None, :]
    return (f > r).astype(np.float64) + 0.5 * (f == r)


def subset_metrics(rows, vals, kind, thr, base_vals=None, base_kind="logit", n_boot=N_BOOT, seed=SEED):
    """rows: map rows of the subset (Reals + Fakes); vals/base_vals: arrays aligned with rows."""
    lab = np.array([r["label"] for r in rows], dtype=np.int64)
    groups = sorted({r["source_group"] for r in rows})
    gi = {g: i for i, g in enumerate(groups)}
    grp = np.array([gi[r["source_group"]] for r in rows], dtype=np.int64)
    rm, fm = lab == 0, lab == 1
    if rm.sum() == 0 or fm.sum() == 0:   # e.g. an incomplete file: nothing to rank
        out = {"n_real": int(rm.sum()), "n_fake": int(fm.sum()), "n_groups": len(groups), "auroc": None,
               "error": "subset lacks Reals or Fakes"}
        if base_vals is not None:
            out["pass"] = False
        return out
    C = pair_matrix(vals[fm], vals[rm])
    dec = decisions(vals, kind, thr)
    out = {"n_real": int(rm.sum()), "n_fake": int(fm.sum()), "n_groups": len(groups),
           "auroc": float(C.mean()),
           "fp": int(dec[rm].sum()), "fpr": float(dec[rm].mean()),
           "tp": int(dec[fm].sum()), "tpr": float(dec[fm].mean())}
    if base_vals is not None:
        Cb = pair_matrix(base_vals[fm], base_vals[rm])
        decb = decisions(base_vals, base_kind, thr)
        out["baseline"] = {"auroc": float(Cb.mean()), "fp": int(decb[rm].sum()), "fpr": float(decb[rm].mean()),
                           "tp": int(decb[fm].sum()), "tpr": float(decb[fm].mean())}
        out["flips"] = int((dec != decb).sum())
        out["flips_real"] = int((dec != decb)[rm].sum())
        out["flips_fake"] = int((dec != decb)[fm].sum())
    rng = np.random.default_rng(seed)
    gr, gf = grp[rm], grp[fm]
    boots, boots_b, boots_fpr, boots_tpr = [], [], [], []
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, len(groups), len(groups)), minlength=len(groups)).astype(np.float64)
        wr, wf = w[gr], w[gf]
        if wr.sum() == 0 or wf.sum() == 0:
            continue
        den = wf.sum() * wr.sum()
        boots.append(wf @ C @ wr / den)
        boots_fpr.append(wr @ dec[rm] / wr.sum())
        boots_tpr.append(wf @ dec[fm] / wf.sum())
        if base_vals is not None:
            boots_b.append(wf @ Cb @ wr / den)
    pct = lambda a: [float(x) for x in np.percentile(a, [2.5, 97.5])]
    out["auroc_ci95"] = pct(boots)
    out["fpr_ci95"] = pct(boots_fpr)
    out["tpr_ci95"] = pct(boots_tpr)
    out["boot_used"] = len(boots)
    if base_vals is not None:
        out["auroc_delta_ci95"] = pct(np.asarray(boots) - np.asarray(boots_b))
        b = out["baseline"]
        out["auroc_drop"] = b["auroc"] - out["auroc"]
        out["fpr_increase"] = out["fpr"] - b["fpr"]
        out["tpr_drop"] = b["tpr"] - out["tpr"]
        out["pass"] = bool(out["auroc_drop"] <= LIMITS["auroc_drop"] + TOL
                           and out["fpr_increase"] <= LIMITS["fpr_increase"] + TOL
                           and out["tpr_drop"] <= LIMITS["tpr_drop"] + TOL)
    return out


def evaluate(values: dict, kind="logit", baseline: dict | None = None, baseline_kind="logit",
             map_rows=None, threshold=None, n_boot=N_BOOT, seed=SEED, strict_methods=False) -> dict:
    map_rows = map_rows or read_map()
    thr = float(threshold if threshold is not None else json.loads(WEIGHTS_JSON.read_text())["threshold"])
    invalid = [r["sample_id"] for r in map_rows
               if values.get(r["sample_id"]) is None or not math.isfinite(values[r["sample_id"]])]
    base_invalid = []
    if baseline is not None:
        base_invalid = [r["sample_id"] for r in map_rows
                        if baseline.get(r["sample_id"]) is None or not math.isfinite(baseline[r["sample_id"]])]
    bad = set(invalid) | set(base_invalid)
    res = {"threshold": thr, "kind": kind, "n_map": len(map_rows), "n_valid": len(map_rows) - len(invalid),
           "invalid": invalid[:50], "n_invalid": len(invalid), "baseline_invalid": base_invalid[:50],
           "n_boot": n_boot, "seed": seed, "splits": {}}
    for split in SPLITS:
        rows = [r for r in map_rows if r["split"] == split and r["sample_id"] not in bad]
        subsets = {"all": rows}
        for m in METHODS:
            subsets[m] = [r for r in rows if r["label"] == 0 or r["method"] == m]
        res["splits"][split] = {}
        for name, sub in subsets.items():
            v = np.array([values[r["sample_id"]] for r in sub], dtype=np.float64)
            bv = None if baseline is None else np.array([baseline[r["sample_id"]] for r in sub], dtype=np.float64)
            res["splits"][split][name] = subset_metrics(sub, v, kind, thr, bv, baseline_kind, n_boot, seed)
    if baseline is not None:
        split_pass = all(res["splits"][s]["all"]["pass"] for s in SPLITS)
        method_pass = all(res["splits"][s][m]["pass"] for s in SPLITS for m in METHODS)
        complete = not invalid and not base_invalid
        res["per_split_pass"] = split_pass
        res["per_method_pass"] = method_pass
        res["complete"] = complete
        ok = split_pass and complete and (method_pass or not strict_methods)
        res["verdict"] = "PASS" if ok else "FAIL"
        res["criteria"] = dict(LIMITS, all_samples_valid=True, strict_methods=strict_methods)
    return res


def markdown(res: dict, name: str = "") -> str:
    has_b = "verdict" in res
    lines = []
    if name:
        lines.append(f"**{name}**  (threshold {res['threshold']:.16g}, {res['n_valid']}/{res['n_map']} valid)")
        lines.append("")
    head = "| split | subset | Real | Fake | AUROC [95% CI] | Real FPR (FP) | Fake TPR (TP) |"
    sep = "|---|---|---|---|---|---|---|"
    if has_b:
        head += " AUROC drop | FPR +pp | TPR -pp | flips | pass |"
        sep += "---|---|---|---|---|"
    lines += [head, sep]
    for split, subs in res["splits"].items():
        for sub, m in subs.items():
            if m.get("auroc") is None:
                lines.append(f"| {split} | {sub} | {m['n_real']} | {m['n_fake']} | n/a ({m.get('error')}) |")
                continue
            row = (f"| {split} | {sub} | {m['n_real']} | {m['n_fake']} | {m['auroc']:.4f} "
                   f"[{m['auroc_ci95'][0]:.4f}, {m['auroc_ci95'][1]:.4f}] | {100 * m['fpr']:.2f}% ({m['fp']}) | "
                   f"{100 * m['tpr']:.2f}% ({m['tp']}) |")
            if has_b:
                row += (f" {m['auroc_drop']:+.5f} | {100 * m['fpr_increase']:+.2f} | {100 * m['tpr_drop']:+.2f} | "
                        f"{m['flips']} | {'yes' if m['pass'] else 'NO'} |")
            lines.append(row)
    if has_b:
        lines.append("")
        lines.append(f"Verdict: **{res['verdict']}** (per split {res['per_split_pass']}, per method "
                     f"{res['per_method_pass']}, complete {res['complete']})")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logits", required=True)
    ap.add_argument("--field", default="logit")
    ap.add_argument("--kind", choices=["logit", "score"], default="logit")
    ap.add_argument("--baseline")
    ap.add_argument("--baseline-field", default="logit")
    ap.add_argument("--baseline-kind", choices=["logit", "score"], default="logit")
    ap.add_argument("--map", default=str(MAP))
    ap.add_argument("--weights-json", default=str(WEIGHTS_JSON))
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--strict-methods", action="store_true")
    ap.add_argument("--name", default="")
    ap.add_argument("--out-json")
    ap.add_argument("--out-md")
    a = ap.parse_args(argv)
    wj = json.loads(Path(a.weights_json).read_text())
    vals = read_values(a.logits, a.field)
    base = read_values(a.baseline, a.baseline_field) if a.baseline else None
    res = evaluate(vals, a.kind, base, a.baseline_kind, read_map(a.map), wj["threshold"], a.n_boot, a.seed,
                   a.strict_methods)
    res["inputs"] = {"logits": str(a.logits), "field": a.field, "logits_sha256": sha256_file(a.logits),
                     "baseline": a.baseline, "baseline_field": a.baseline_field if a.baseline else None,
                     "baseline_sha256": sha256_file(a.baseline) if a.baseline else None,
                     "weights_sha256": wj.get("output_sha256"), "checkpoint_sha256": wj.get("checkpoint_sha256"),
                     "map": str(a.map)}
    md = markdown(res, a.name)
    if a.out_json:
        Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out_json).write_text(json.dumps(res, indent=1))
    if a.out_md:
        Path(a.out_md).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out_md).write_text(md + "\n")
    print(md)
    return 0 if ("verdict" not in res or res["verdict"] == "PASS") else 1


if __name__ == "__main__":
    sys.exit(main())
