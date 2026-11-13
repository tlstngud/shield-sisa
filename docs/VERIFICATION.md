# Verification

The engine was built by separate agents for the oracle, preprocessing, CPU, CUDA and service parts. A separate verifier then checked it. The verifier rebuilt every library from source and re-measured each claim independently.

| claim | result |
|---|---|
| Preprocessing parity | Crops are byte-identical to the Python reference on 840/840 videos (RGB24 and I420). The native Haar detector matches OpenCV on 1,740/1,740 images. |
| CPU exact vs float64 oracle | max \|Δlogit\| 2.835e-4, p99.9 2.336e-4, 0 decision flips |
| CUDA FP32 vs float64 oracle | max 3.98e-4, 0 flips |
| CUDA FP16 / BF16 | max 1.56 / 2.35 logit, 0 flips. Certified: AUROC drop ≤ 0.002, FPR +≤ 1 pp, TPR −≤ 1 pp, per split and per method. |
| Independent float64 rebuild of the model | agrees with the oracle to 2.3e-13 |
| Row independence | outputs are bitwise identical across batch size, order, subsets, thread count and ISA, on CPU and GPU |
| Live service parity | 3,998/3,998 windows across all 840 videos are bitwise equal to batch predict, and 156/156 equal the Python reference |
| Audio parity | 202/202 hops equal the engine's `shield_predict` on low-passed windows |
| Contracts | 14/14 contract tests and 12/12 edge-case tests pass on the CPU and CUDA backends |
| Report numbers | 85/85 quoted numbers match the raw benchmark files |

Exact-tier metrics, which are equal to the oracle's:

| split | AUROC | false alarms (Real) | detected (Fake) |
|---|---|---|---|
| val | 0.97648 | 21 / 211 | 206 / 216 |
| test | 0.99081 | 21 / 232 | 180 / 181 |

AUROC is reported with 95% bootstrap CIs, computed with 1,000 source-group resamples. The test CI is [0.9840, 0.9958].

The sanitizer findings and their status are in [`KNOWN_ISSUES.md`](../KNOWN_ISSUES.md).
