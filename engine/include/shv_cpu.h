/*
 * shv_cpu.h -- SHIELD-V native CPU model (C API), build/cpu/libshield_v_cpu.so
 *
 * The SHIELD-V video deepfake detector (reference/hf_model/shield_v/model.py) in C++17 without Python at run time:
 *   uint8 RGB crops [20][180][320][3] -> (/255, ImageNet mean/std, fused into the stem) -> MobileNetV3-Large features
 *   -> adaptive average pool 6x10 -> 4x4 (overlapping bins) -> Linear 960->384 = 16 tokens per frame
 *   -> + (time[t] + row[r]) + col[c] embeddings, readout token -> GDN-A -> GDN-B (mixer twice) -> SISA (last row)
 *   -> final RMSNorm -> Linear(384,128) -> SiLU -> Linear(128,1) = logit.
 * The caller converts logit -> score with a double-precision sigmoid and compares it with shv_cpu_threshold()
 * (read from "<weights>.json"; decision: score >= threshold).
 *
 * Weights: the SHLDV001 export of tools/export_v.py (any checkpoint of the same architecture).  Every tensor
 * name and shape is validated; BatchNorm (eps 1e-3) is folded into the convolutions at load.  The SHA-256 of the
 * weight file is the model fingerprint and binds quality certificates.
 *
 * Modes ("mode" argument; NULL or "" = "exact"):
 *   "exact"    float32 everywhere (the default and the reference tier).  Not bitwise equal to PyTorch (different
 *              summation orders: GEMMs accumulate K in blocks of 64); its error vs the float64 oracle is reported in
 *              docs/PERFORMANCE.md.
 *   "bf16-cnn" the CNN 1x1 convolutions (expand, project, final 160->960, projection 960->384) take BF16 inputs and
 *              weights (round to nearest even) with FP32 accumulation (VDPBF16PS); stem, depthwise, SE, pooling FP32.
 *   "bf16-seq" the same for the sequence linears (GDN in/out/FFN, SISA in/Q/C/O/FFN); norms, short conv,
 *              recurrence, attention, gates and the head stay FP32.
 *   "bf16"     both.
 *   "int8-cnn", "int8-cnn-bf16-seq"  CNN 1x1 convolutions in INT8 (per-pixel dynamic activation scale, per-channel
 *              weight scale, VPDPBUSD): implemented for evaluation only; they FAILED certification (see
 *              docs/PERFORMANCE.md), have no certificate and are refused by the production loader.
 *   The BF16 modes need the avx512 dispatch level on a CPU with AVX-512 BF16, and are accepted only when
 *   "<weights>.cpu.cert" holds a passing certificate line "<weights sha256> <mode> <shv_cpu_revision()>
 *   threshold=<%.17g>" whose threshold equals the loaded threshold exactly (first line "SHIELD_V_CPU_CERT_V2";
 *   written by bench/cpu_certify.py from bench/metrics.py on all 840 val+test reference crops).  Builds configured
 *   with -DSHV_CPU_EXPERIMENTAL=ON additionally export shv_cpu_load_experimental(), which skips the certificate check
 *   (offline certification only).
 *   Sidecar "<weights>.json" (every mode, also experimental loads): "output_sha256" must equal the weight file's
 *   SHA-256 and "threshold" must be present, finite and in (0, 1), else the load fails (same rule as the CUDA
 *   engine).  An edited threshold with the matching SHA still loads "exact" but no certified mode.
 *
 * Determinism / row independence: every output value is computed by the same sequence of float operations
 * regardless of batch size, position in the batch, the number of threads, how frames/windows are distributed
 * over threads, and the dispatch level (AVX-512 / AVX2 / scalar).  Results are bit-identical across all of these.
 *
 * Threads: "threads" (1..64) is the size of the model's compute team; the calling thread is one of them, the
 * other threads-1 are persistent workers created at load.  Frames (encode) and windows (classify) are spread
 * over the team: whole items per thread while there are at least "threads" items left, then the remaining items
 * are each processed by a sub-team of threads/remaining threads (row / column partitioning inside the item).
 *   - threads == 1: calls may be made concurrently from any number of threads on one handle; each concurrent
 *     call uses its own workspace (created on first use, then reused).
 *   - threads > 1: calls on one handle are serialised (one team).
 * Workspaces are allocated on first use and reused; steady-state calls do not allocate (measured: 0 heap
 * allocations over all timed calls of bench/cpu_bench).  Call shv_cpu_free only after every call on the handle
 * has returned.
 *
 * Dispatch level: chosen at load from the CPU (AVX-512F+BW+DQ+VL -> "avx512", AVX2+FMA -> "avx2", else "scalar").
 * The environment variable SHV_CPU_ISA=avx512|avx2|scalar (or shv_cpu_load_ex) forces a level; a level the CPU
 * does not support is an error.  In exact mode all three levels give bit-identical outputs (tested on the 840
 * crops for avx2, on 42 for scalar).  SHV_CPU_PIN=1 pins worker i to the i-th CPU of the process affinity mask.
 * Debug: SHV_CPU_DUMP_DIR=<dir> (read at load) writes float32 intermediates of every frame / window
 * (bench/cpu_fixture_check.py uses it).  Never set it in production.
 *
 * Memory (this checkpoint, measured): packed weights 40.05 MB (exact), 43.5 / 53.0 / 56.4 MB (bf16-cnn /
 * bf16-seq / bf16: the FP32 panels are kept next to the BF16 ones); workspace per compute slot (one thread's frame +
 * window buffers) 11.3 MB, allocated on first use; a predict() call also keeps its tokens (24.6 KB per frame).
 *
 * All functions return 0 on success and -1 on failure (shv_cpu_last_error(), thread-local, explains), unless
 * documented otherwise.  Outputs are left unspecified on failure.
 */
#ifndef SHV_CPU_H
#define SHV_CPU_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif
#if defined(__GNUC__)
#pragma GCC visibility push(default)
#endif

#define SHV_CPU_API_VERSION 1
#define SHV_CPU_FRAME_H 180
#define SHV_CPU_FRAME_W 320
#define SHV_CPU_FRAME_BYTES (180 * 320 * 3)   /* one RGB24 crop */
#define SHV_CPU_FRAMES 20                     /* frames per window */
#define SHV_CPU_TOKENS 16                     /* tokens per frame */
#define SHV_CPU_DIM 384                       /* token width */
#define SHV_CPU_FRAME_FLOATS (16 * 384)       /* tokens of one frame */

typedef struct shv_cpu_model shv_cpu_model;

/* Load a SHLDV001 weight file.  mode: NULL/"" = "exact".  threads: 1..64.  Returns NULL on failure. */
shv_cpu_model* shv_cpu_load(const char* weights, const char* mode, int threads);
/* Same, with an explicit dispatch level: isa NULL/"" = automatic (or SHV_CPU_ISA), "avx512", "avx2", "scalar". */
shv_cpu_model* shv_cpu_load_ex(const char* weights, const char* mode, int threads, const char* isa);
/* Experimental builds only (otherwise returns NULL with an error): load an uncertified approximate mode. */
shv_cpu_model* shv_cpu_load_experimental(const char* weights, const char* mode, int threads, const char* isa);
void shv_cpu_free(shv_cpu_model* m);

/* rgb: n contiguous crops [180][320][3] uint8 (RGB, the 320x180 model input).  tokens: n x [16][384] float32
 * (projection output, before position embeddings). */
int shv_cpu_encode_frames(shv_cpu_model* m, const uint8_t* rgb, size_t n, float* tokens);
/* Gather / scatter form: frames[i] -> one crop, tokens[i] -> 16*384 floats.  Lets a caller encode only the frames
 * whose crop changed and keep a per-participant token cache keyed by (frame id, crop rectangle). */
int shv_cpu_encode_frame_ptrs(shv_cpu_model* m, const uint8_t* const* frames, size_t n, float* const* tokens);

/* tokens: b windows x [20][16][384] (frame order = window order).  logits: b floats. */
int shv_cpu_classify(shv_cpu_model* m, const float* tokens, size_t b, float* logits);
/* frame_tokens: b*20 pointers, window w frame t at index w*20+t, each to 16*384 floats. */
int shv_cpu_classify_ptrs(shv_cpu_model* m, const float* const* frame_tokens, size_t b, float* logits);

/* clips: b x [20][180][320][3] uint8.  logits: b floats.  = encode all 20*b frames, then classify. */
int shv_cpu_predict(shv_cpu_model* m, const uint8_t* clips, size_t b, float* logits);

const char* shv_cpu_last_error(void);
const char* shv_cpu_mode(const shv_cpu_model* m);         /* "exact", ... */
const char* shv_cpu_fingerprint(const shv_cpu_model* m);  /* SHA-256 (hex) of the weight file */
const char* shv_cpu_isa(const shv_cpu_model* m);          /* "avx512" | "avx2" | "scalar" */
int shv_cpu_threads(const shv_cpu_model* m);
/* Decision threshold from "<weights>.json" ("threshold"; validated at load: finite, in (0, 1)); NaN for a NULL model. */
double shv_cpu_threshold(const shv_cpu_model* m);
/* Bytes: packed weights; allocated workspace so far (all workspaces); workspace of one compute slot
 * (one encode frame + one classify window).  Any pointer may be NULL. */
void shv_cpu_memory(const shv_cpu_model* m, size_t* weight_bytes, size_t* workspace_bytes, size_t* slot_bytes);
const char* shv_cpu_revision(void);  /* numeric implementation revision (part of every certificate) */
/* Profile builds (-DSHV_CPU_PROFILE=ON) only: per-stage nanoseconds and counts accumulated by rank 0 since the last
 * call, then reset (stage ids: docs/PERFORMANCE.md).  Returns the number of stages, or 0 in normal builds. */
int shv_cpu_profile_get(const shv_cpu_model* m, double* ns, long* counts, int len);
const char* shv_cpu_version(void);

#if defined(__GNUC__)
#pragma GCC visibility pop
#endif
#ifdef __cplusplus
}
#endif
#endif /* SHV_CPU_H */
