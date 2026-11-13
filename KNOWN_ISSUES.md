# Known issues

## Fixed and re-verified (2026-11-13)

An internal sanitizer review found the issues below. They are fixed, and the fixes were re-verified independently on a clean build of this tree.

| Issue | Re-verification |
|---|---|
| CUDA service path could read a video job after it was freed (job completed before its handle was published) | ASan and TSan: 0 reports in 66 runs. The pre-fix build reproduces it as a positive control. |
| CUDA library: a job's wait object could be destroyed while the GPU worker was still signalling it | TSan: 0 reports, 0 logit mismatches. The pre-fix build reproduces it. |
| A stalled audio input (e.g. a muted microphone) blocked video results. New rule: `audio_wait_s` | Passes under ASan and TSan on both backends. A stress test with 11,880 stalled-mic segments had 0 ordering violations. |
| Decision threshold was not bound to the weights. Now: certificates `*_CERT_V2` carry `threshold=`, the sidecar `output_sha256` must match, and missing or out-of-range thresholds are rejected | 22 fixtures × 11 load paths |
| Participant ids with control characters are now JSON-escaped | All of 0x01–0x1f round-trip through `json.loads` |

Other checks on the fixed tree:

- **Stress without sanitizers (CUDA):**
  - 64 sessions for 20 min: 19,833 sessions, 7,975 resets and 4,758 frees while jobs were in flight.
  - 96 sessions for 15 min.
  - 0 crashes, hangs or ordering violations.
- **Parity:** 840/840 reference windows and 3,998/3,998 live windows are bit-identical to the pre-fix code. This holds for RGB24 and I420 input, on CPU exact and CUDA FP16.
- **Audio parity:** 202/202 and 240/240.

The certificate mismatch below was fixed in a later commit. Builds of the first public commit compute CUDA revision `cuda-247beadf531769d3`. The published `.cuda.cert` covers `cuda-a2bc0624de64263f`. As a result, FP16/BF16 refused to load, and the service silently ran FP32 (about 2× GPU time).

- The cause was comment-only edits to CUDA sources that are part of the revision hash.
- The current tree builds `cuda-a2bc0624de64263f` again.
- Any edit to a file listed in `NUMERIC_SRCS` (`engine/cuda/CMakeLists.txt`), **comments included**, changes the revision and requires re-certification.

## Open

- **The CUDA toolkit is required to build the service.** `libshield_av` links `libshield_v_cuda` even for CPU-only use. The CUDA runtime is linked statically, so no GPU or driver is needed at run time.
- **x86-64 only.** The CPU engines use AVX2/AVX-512 runtime dispatch. ARM (for example AWS Graviton) is not supported.
- **Compression sensitivity (video).** Real videos re-encoded like a low WebRTC simulcast layer score markedly higher. Use the highest available layer (720p, ≥1.5 Mbps) for analysis, and see the README deployment notes.
- **Capacity on 16 CPU threads.** 512 participants with both modalities did not meet the 2 s deadline. Video alone passes at 512.
- **Host memory is about 50 MB per participant.** The frame store holds one 10 s window at the 1280×720 canvas.
- **No face.** Without any detected face, the whole frame is used, as the reference does. There is no separate `no_face` status yet.
- **Certificates are lookup lines, not signatures.**
- **Minor:**
  - A failed `shav_engine_create` leaks about 330 KB (the audio model and cascade).
  - A threshold with trailing characters (`0.26abc`) is accepted.
  - A participant id that is not valid UTF-8 is copied into the JSON unchanged.
