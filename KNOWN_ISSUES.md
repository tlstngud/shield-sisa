# Known issues

This is an early release. These issues are known and tracked.

## Fixed in this release, but only partly re-verified

These fixes come from an internal sanitizer review. The contract and edge-case suites pass on both backends with the fixes. The independent sanitizer re-run (ASan/TSan) and the full 840-video parity re-run have **not** been repeated on the fixed code yet.

- The CUDA service path could use a video job after it was freed, when a job completed before its handle was published.
- In the CUDA library, a job's wait object could be destroyed while the GPU worker thread was still signalling it.
- When a participant's audio stalled (for example a muted microphone), no video results were emitted. A symmetric audio lag rule (`audio_wait_s`) now releases these hops, with the audio part reported as `unavailable/lag`.
- The decision threshold is now bound to the weights and certificates:
  - certificates are `*_CERT_V2` with a `threshold=` field;
  - the sidecar's `output_sha256` must match the weight file;
  - a missing or out-of-range threshold is rejected on both backends.
- Participant ids that contain control characters are now JSON-escaped.

Until the re-verification is complete, **the CPU backend is the recommended deployment**. The first two issues were CUDA-only.

## Open

- **The CUDA toolkit is required to build the service.** `libshield_av` links `libshield_v_cuda` even when only the CPU backend is used. The CUDA runtime is linked statically, so no GPU or driver is needed at run time. Making CUDA fully optional is planned.
- **x86-64 only.** The CPU engines use AVX2/AVX-512 runtime dispatch. A scalar fallback exists but is much slower. ARM (for example AWS Graviton) is not supported.
- **Capacity on 16 CPU threads.** 512 participants with both modalities did not meet the 2 s deadline, because the CPU is the bottleneck (video preprocessing plus audio). Video alone passes at 512 participants.
- **Host memory is about 50 MB per participant.** All 20 sampled frames of a 10 s window are kept at the 1280×720 canvas until the window's face anchors are known.
- **No face.** When no face is found in any anchor frame, the model falls back to the whole frame, as the reference does. There is no separate `no_face` status yet.
- **Certificates are lookup lines, not signatures.** They protect against accidental mismatch, not against tampering.
- Some debug and ablation environment variables are honoured by the production libraries. Measurement showed they do not change outputs.
