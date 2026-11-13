# SHIELD-A native engine

Linux x86-64 C++17 inference for same-architecture SHIELD-A checkpoints. Runtime dependencies are the C/C++ standard libraries, libm and glibc's libmvec (used for vector exponentials). Python and PyTorch are used only for exporting, benchmarks and verification. No GPU, OpenMP, BLAS installation or network download is required.

```bash
mkdir -p build/compiler-tmp
export TMPDIR="$PWD/build/compiler-tmp"
python3 -B engine/export.py \
  --checkpoint reference/ckpt/best.pt --output build/shield.weights
cmake -S engine -B build/native -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build/native -j 1
```

`--checkpoint` accepts another path to a trusted PyTorch checkpoint containing a compatible `model` state dictionary. `SHIELD_CHECKPOINT` supplies its default when the flag is omitted. Export uses strict state-dictionary loading; changing the architecture is unsupported. The export format contains the state dictionary plus the exact nonpersistent Hann window and Mel filterbank. The adjacent `.weights.json` records absolute checkpoint/output paths and their SHA-256 hashes. The native loader validates tensor shapes and computes the exported file's SHA-256; it does not require the original checkpoint at runtime.

The default mode uses FP32 inference parameters and operations. Normal floating-point rounding differences from torch are measured in `REPORT.md`; this is not a bitwise torch implementation. The service sigmoid and alert threshold are double precision, matching the Python contract.

Use [include/shield.h](include/shield.h). Link `build/native/libshield.so`. `shield_model_load(path, workers)` creates immutable shared weights and 1–8 persistent workers. `shield_predict` evaluates one contiguous 160,000-sample window. `shield_predict_batch` schedules independent rows with an explicit row stride; it never performs normalization or attention across participants. A model handle is safe for concurrent calls.

`shield_model_load_mode(path, workers, mode)` accepts these names. Every mode retains FP32 linear outputs/biases, FFT/Mel, depthwise convolutions, normalization, nonlinearities, recurrent state/updates, and the readout head.

| Mode | Linear weights and inputs | Affected layers |
|---|---|---|
| `exact` | FP32 | Default everywhere |
| `fp16-linear` | FP16 nearest-even; FP32 FMA accumulation | CNN, GDN and SISA projections |
| `bf16-linear` | BF16 nearest-even; FP32 dot accumulation | CNN, GDN and SISA projections |
| `fp16-cnn` | FP16 nearest-even; FP32 FMA accumulation | CNN projections |
| `bf16-cnn` | BF16 nearest-even; FP32 dot accumulation | CNN projections |
| `int8-cnn` | Per-input-row activation and per-output-channel weight scales; INT32 accumulation | CNN linears with input/output width at least 32 |

Nonexact production loads require `weights_path + ".cert"`. A passing entry must match the exact exported-file SHA-256, mode name, and `shield_compute_revision()`. Moving an unchanged file preserves its identity; changing any weight requires fresh certification. `shield_model_mode()` and `shield_model_fingerprint()` expose the loaded identities. Certification is produced by [bench/certify.py](../bench/certify.py) against the full Torch oracle; a successful load or a synthetic fixture is not a detector-quality certificate. The certificate is a local trusted artifact, not a signed authorization file.

An ordinary build has `SHIELD_ENABLE_EXPERIMENTAL=OFF`. For offline certification only, configure a separate build with `-DSHIELD_ENABLE_EXPERIMENTAL=ON` and use `shield_model_load_experimental(...)`; that entrypoint bypasses the certificate requirement. It returns an error in ordinary builds. The normal loader checks certificates in either build. Experimental mode availability does not mean a mode passed the quality contract. Precision, cache and input-row identities are kept separate between model jobs; no participant shares activation scales. See [precision notes](../bench/precision.md) for arithmetic and fallback details.

Create a `shield_stream` per participant, with the threshold, model version and alert k/n policy. Push any number of finite mono float32 samples in [-1,1] at 16 kHz. The ring emits one result per completed 32,000 samples. Reserve enough output results before calling; insufficient capacity fails without consuming audio. The caller can push different participants concurrently: ready windows enter the shared FIFO worker queue, while each participant serializes its own pushes. Caller buffers remain borrowed only until the function returns. Destroy handles only after outstanding calls return. Streams retain the model when its public model handle is freed.

For asynchronous scheduling, call `shield_stream_submit_hop(stream, pcm_32000, t_end)` at a completed hop boundary. It copies one complete 32,000-sample hop before returning a ticket. Collecting or VAD-skipped results can be ready immediately; valid inference enters the shared worker queue. A stream permits one job in flight: another submission for that participant waits for it, providing backpressure. Other participants can submit without creating a caller thread for each inference.

`shield_ticket_ready()` returns 1 when ready, 0 while pending, or -1 for a null ticket. `shield_ticket_wait()` returns the completed `shield_result`; a result with status `unavailable` is still a completed service result. `shield_ticket_timing()` returns 0 ready, 1 pending, or -1 for bad arguments, with queue/compute milliseconds and completion time in `CLOCK_MONOTONIC` seconds. Freeing a pending ticket waits for completion; tickets retain model ownership. Reset/free of a stream also waits for its in-flight job. Use one caller to resolve/free each ticket. `shield_stream_phase_samples()` returns a stable participant-ID phase in `[0,32000)` for a capture scheduler to stagger hop boundaries; it does not move boundaries or discard samples itself.

`shield_stream_set_lowpass(stream, 1)` makes the model read a low-passed copy of the ring, for checkpoints trained on SHIELD's 7.2 kHz low-passed windows (every in-the-wild fine-tune from `train_wild.py`). Training uses zero-phase `scipy.signal.sosfiltfilt` per window; a stream instead runs the same order-8 Butterworth sections forward twice (equal magnitude response, causal phase) with filter state carried across pushes and hops, so the 2-second frontend reuse stays valid. VAD and `valid_speech_sec` still read the raw PCM. It is off by default (every earlier contract compares stream logits with `shield_predict` on raw windows), changes only on an empty stream (after create or reset), and adds a 640,000-byte ring per participant. `bench/lowpass_contracts.py` checks that stream logits equal `shield_predict` on scipy `sosfilt(sos, sosfilt(sos, x))` exactly, across chunked, asynchronous, reset and cache-level paths. On 2,000 ML-ITW dev clips the causal filter changed AUROC from 0.8057 to 0.8059 (logit correlation 0.9998, median |Δlogit| 0.023).

The four statuses are collecting, insufficient_speech, valid and unavailable. Nonvalid scores and logits are NaN and alert is -1, representing null; applications must serialize these as null, never as zero. Participant ID and version remain on the stream handle and are available through accessor functions. VAD is the reference double-precision 20 ms energy/percentile rule. Only valid windows enter the last-n alert history. Silence skips the entire model. A null model permits status-only operation and yields unavailable for speech.

`t_end` is the timestamp of the final sample in a push. Every emitted result receives the timestamp of its own hop boundary. This deliberately fixes the reference implementation's stale timestamp assignment (it updates `t_end` after emitting results). The 10-second window, 2-second hop, speech gate and alert policy are unchanged.

Service caches rely on the ring's guaranteed 2-second shift and are invalidated after skipped or failed inference. The frontend reuses Mel frames and 90 interior CNN tokens; a general internal frontend cache instead verifies PCM overlap byte-for-byte. `shield_stream_set_cache_level(stream, level)` controls the additional GDN-A cache:

| Level | GDN-A preprocessing retained per participant |
|---|---|
| 0 | Off; frontend reuse remains enabled |
| 1 (default) | Six projections for 90 unchanged tokens: 833,760 payload bytes |
| 2 | Level 1 plus 87 eligible Q/K/V short-convolution/normalization rows: another 534,528 payload bytes |

Changing level invalidates sequence eligibility; reducing it releases the unused GDN arrays. Reset clears eligibility and alert history while retaining allocated buffer capacity. Caches check model and execution mode; they never retain recurrent GDN state. The learned READOUT's initial GDN-A projections are shared per model/mode. Large CNN intermediates and sequence workspace belong to workers; immutable original and packed weights are shared by model workers. Worker-local pools reuse CNN storage after warmup, but first-use/growth and bookkeeping allocations remain. Payload cache counts are not whole-stream RSS; `shield_stream_bytes()` reports retained stream storage.

Compile targets are baseline x86-64. Function-level feature dispatch selects AVX-512/FMA, AVX2/FMA or scalar fallback, checking CPU/OS support before calling a specialized kernel. `SHIELD_KERNEL=avx2` exercises fallback on this machine; `scalar` forces the portable path. Unknown or unsupported forced modes fall back safely. Avoid `-march=native` for deployment. `SHIELD_CNN_TILE=0` and `SHIELD_DELTA_KERNEL=scalar` are benchmark ablations. Production should leave these unset.

Pin the process to the intended cores with `taskset` and size the worker pool to that set. Capacity under contention is not guaranteed by the measured percentiles; use the report's measured cohort results and retest the actual demo host. The queue is FIFO, batch submission uses chunks of at most 64, and the core has no network transport or overload admission controller.

Checkpoint portability and stale-certificate/oracle checks can run against an existing frozen library:

```bash
python3 -B bench/checkpoint_contracts.py \
  --checkpoint reference/ckpt/best.pt --weights build/shield.weights \
  --library build/phase2/libshield.so
```

This creates a labeled synthetic head-bias fixture inside `bench/artifacts/checkpoint_contracts`; it does not retrain or certify a candidate. The full-reference/certification tools bind resumable artifacts to checkpoint/source/environment identity and exported-weight/library hashes. New checkpoint contents must use fresh matching oracle/candidate artifacts; copying a previous certificate or oracle does not authorize them.

Regenerate the complete Phase 2 report and every current measurement for the supplied checkpoint or another compatible checkpoint:

```bash
bash bench/reproduce.sh --checkpoint reference/ckpt/best.pt
# A positional path or SHIELD_CHECKPOINT can also select the checkpoint.
```

The command preserves the historical `bench/phase1` report, rebuilds its frozen source for matched comparisons, builds development with experimental loading enabled and production with it disabled, verifies production exact logits on 2,000 real windows, and generates/resumes all 19,112-window oracle and five-candidate quality tables. It then checks mode/cache/async/checkpoint and cross-build contracts before atomically promoting production to `build/native`. Final paired, deadline, memory and optimization measurements feed `bench/report_phase2.py`; the old Phase 1 report generator is not invoked. A failed candidate remains unavailable through the normal production loader.

`SHIELD_CPUS=0,1,2,3,4,5,6,7` selects eight distinct CPUs; all sustained jobs are pinned and staged within that set. Four one-thread Torch workers run together, then the five candidate sweeps share eight CPUs as 2+2+2+1+1 workers. Subsequent timing suites run separately. `SHIELD_BENCH_CPU` chooses one selected CPU for paired runs, `SHIELD_BENCH_RUNS` defaults to 30, and `SHIELD_MATRIX_ROUNDS` defaults to five. `SHIELD_WEIGHTS` can select another output inside the workspace; checkpoint/export/library identities follow it through report generation. Compiler and Python temporary files stay inside the workspace. Full reference generation may take substantial time; resumable artifacts require matching checkpoint/source/environment and binary/weight identities. Exact remains the default even when an approximation passes.
