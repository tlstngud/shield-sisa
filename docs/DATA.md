# Test and benchmark data

The parity, contract, edge-case and load tests read their inputs from `$SHIELD_DATA_DIR` (default `./data`).
**This repository ships no data.**

| path under `$SHIELD_DATA_DIR` | used by | content |
|---|---|---|
| `win_v2/test.npy` | service_contracts, verify_service_edge, service_load, demo_av (`--audio win_v2:ROW`) | float32 `[N, 160000]` 16 kHz audio windows |
| `shieldv_codec/maps/orig.jsonl` | parity, gpu_cli, gpu_bench, metrics.py | one JSON object per video: `sample_id`, `path`, `split`, `label`, `method`, `source_group` |
| `shieldv_codec/crops/orig_bt601/<sha256(sample_id)[:24]>.npz` | parity, certification | reference 20×180×320×3 uint8 crops from the Python preprocessing |
| `shieldv_codec/results/orig_bt601.jsonl` | parity | published reference scores |

Certification also needs `bench/artifacts/oracle_logits.jsonl`, which holds the exact-FP32 baseline logits. To create it, run `oracle/shieldv_ref.py` over your crops.

`demo_av` needs no data: `demo_av --video FILE --audio FILE|silence`.
