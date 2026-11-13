/*
 * shv_cuda.h -- SHIELD-V video detector, CUDA inference engine (C API), build/cuda/libshield_v_cuda.so
 *
 * Same function as the PyTorch reference (reference/hf_model/shield_v/model.py), native CUDA C++, no Python:
 *   uint8 RGB crops [20,180,320,3] --(/255, ImageNet mean/std, fused into the stem)--> MobileNetV3-Large features
 *   (folded BatchNorm) -> adaptive avg pool 4x4 -> Linear 960->384 = 16 tokens per frame ("tokens", before positions)
 *   -> + (time + row) + col, readout token -> GDN-A -> GDN-B x2 -> SISA (last row) -> RMSNorm -> head -> logit.
 * Decision (caller side): score = sigmoid(logit) in double, alert iff score >= shv_cuda_threshold(m).
 * A logit is NaN when the window could not be scored (failed job, or a non-finite result such as an fp16 overflow);
 * treat NaN as "unavailable", never as a score.
 *
 * MODES (precision of tensor operands; accumulation, norms, recurrences and the residual stream are fp32 in all
 * modes, the SISA cumulative sums are double):
 *   "fp32"  IEEE fp32 everywhere (no TF32).  Exact tier: same function as the float32 oracle, error reported in
 *           docs/PERFORMANCE.md.  Always loadable.
 *   "bf16"  bf16 activations/weights for every convolution and all-row GEMM (tensor cores), fp32 elsewhere.
 *   "fp16"  same with fp16.
 *   Approximate modes load only with a passing quality certificate for these exact weight bytes AND this threshold:
 *   "<weights>.cuda.cert" (first line "SHIELDV_CUDA_CERT_V2") must contain the line
 *   "<weights sha256> <mode> <shv_cuda_compute_revision()> PASS <cert json> <cert json sha256> threshold=<%.17g>"
 *   whose threshold equals the loaded threshold exactly.  Certificates are produced by bench/gpu_certify.sh with
 *   bench/metrics.py on all 840 val+test samples.
 * SIDECAR "<weights>.json" (every mode, fp32 included): "output_sha256" must equal the weight file's SHA-256 (the
 *   sidecar belongs to this export) and "threshold" must be present, finite and in (0, 1); otherwise the load fails.
 *   An edited threshold with the matching SHA therefore still loads fp32 (the threshold is reported by
 *   shv_cuda_threshold) but no certified mode: the certificate covers the FPR/TPR of one threshold only.
 *   GPU DEFAULT for the service: "fp16" (certified, 0 decision flips vs exact on 840 samples, 3x smaller logit error
 *   than bf16 at the same speed, 2x the throughput of fp32; docs/PERFORMANCE.md).  If its certificate is missing
 *   or stale the load fails: fall back to "fp32" (mode NULL), which is always loadable and is the exact tier.
 *
 * ROW INDEPENDENCE: the logit of a window and the tokens of a frame are bitwise identical whatever else is in the
 * batch (batch size, order, padding), in every mode.  GEMMs never split K and accumulate k in the same order for
 * every tile shape; all reductions are per row in a fixed order; there are no atomics.  Tested by
 * bench/gpu_rowind.sh (batch 1 vs N vs shuffled vs async-coalesced).
 *
 * THREADING AND LIFETIMES
 *   - A model owns one GPU worker thread, its CUDA streams, device weights and buffers.  Every function here is
 *     safe to call concurrently from any number of threads on the same model, except shv_cuda_free, which must
 *     not race any other call on that model.  Jobs from all threads are executed in submission (FIFO) order;
 *     consecutive queued jobs are coalesced into one GPU batch up to the limits (see shv_cuda_limits).
 *   - Synchronous functions (encode_frames, classify, predict) submit jobs and wait; they may be called from many
 *     threads at once.  They must not be called from inside a job callback (deadlock).
 *   - Asynchronous jobs: every buffer a request points to (frames, window_tokens blocks, tokens_out, logits_out)
 *     must stay valid and unchanged/unread until the job is complete (status != 0).  The small index arrays
 *     (window_frames, window_slots, window_tokens pointer array, store_slots) are copied at submit.
 *   - Each job returned by shv_cuda_submit must be released exactly once with shv_cuda_job_free, which waits for
 *     completion (and for its callback to return) if needed.  Jobs may outlive the model (status stays final).
 *   - Callbacks run on the worker thread after the job's outputs are written and its status is final.  Keep them
 *     short; they may call shv_cuda_job_status/wait/timing/error on any job and shv_cuda_submit, but not the
 *     synchronous functions, shv_cuda_job_free of their own job, or shv_cuda_free.
 *   - shv_cuda_free completes every queued job, then releases the GPU.
 *   - Error strings (shv_cuda_last_error) are thread-local.  All int functions return 0 on success, -1 on error.
 *
 * DEVICE TOKEN SLOTS (optional, options.token_slots > 0): a pool of 16x384 fp32 token blocks resident on the GPU,
 * indexed 0..token_slots-1, owned by the caller (e.g. a ring of slots per participant).  A request may store the
 * tokens of the frames it encodes into slots (store_slots) and windows may read slots (window_slots), so a live
 * 2 s update uploads only its new frames.  Within a batch, stores happen after encoding and before classification;
 * the scheduler never coalesces a job that stores into a slot read by an earlier job of the same batch, so every
 * window sees exactly the slot contents implied by submission order.
 */
#ifndef SHV_CUDA_H
#define SHV_CUDA_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif
#if defined(__GNUC__)
#pragma GCC visibility push(default)
#endif

#define SHV_CUDA_API_VERSION 1
#define SHV_CUDA_FRAME_BYTES 172800    /* 180 x 320 x 3 uint8 RGB */
#define SHV_CUDA_TOKEN_FLOATS 6144     /* 16 tokens x 384, one frame */
#define SHV_CUDA_WINDOW_FRAMES 20

typedef struct shv_cuda_model shv_cuda_model;
typedef struct shv_cuda_job shv_cuda_job;

typedef struct shv_cuda_options {
    int max_frames;       /* frames per GPU batch (CNN); default 640 */
    int max_windows;      /* windows per GPU batch (sequence); default 512 */
    int max_host_tokens;  /* host-provided token blocks per GPU batch; default 2048 */
    int token_slots;      /* device token slots (0 = none); default 0 */
    int use_graphs;       /* CUDA graphs per batch bucket; default 1 */
    int max_delay_us;     /* scheduler: wait up to this long for more jobs before launching; default 0 */
} shv_cuda_options;

void shv_cuda_default_options(shv_cuda_options* opt);
/* weights: SHLDV001 file from tools/export_v.py (+ "<weights>.json" with the threshold).  mode: "fp32" | "bf16" |
 * "fp16" (NULL = "fp32").  device: CUDA device ordinal (after CUDA_VISIBLE_DEVICES).  NULL on error. */
shv_cuda_model* shv_cuda_load(const char* weights, const char* mode, int device);
shv_cuda_model* shv_cuda_load_ex(const char* weights, const char* mode, int device, const shv_cuda_options* opt);
void shv_cuda_free(shv_cuda_model* m);

const char* shv_cuda_last_error(void);
const char* shv_cuda_mode(const shv_cuda_model* m);
const char* shv_cuda_fingerprint(const shv_cuda_model* m);   /* SHA-256 hex of the weight file */
double shv_cuda_threshold(const shv_cuda_model* m);          /* from "<weights>.json" */
const char* shv_cuda_compute_revision(void);                 /* hash of the engine sources (certificate binding) */
int shv_cuda_limits(const shv_cuda_model* m, int* max_frames, int* max_windows, int* max_host_tokens, int* token_slots);
size_t shv_cuda_weight_bytes(const shv_cuda_model* m);       /* device bytes of weights */
size_t shv_cuda_device_bytes(const shv_cuda_model* m);       /* device bytes allocated by the model in total */

/* ---------------------------------------------------------------- synchronous (split into GPU batches internally)
 * rgb: n x 180x320x3 uint8 crops; tokens: n x 16 x 384 fp32 (before positions).
 * tokens (classify): b x 20 x 16 x 384 fp32; clips (predict): b x 20 x 180x320x3 uint8; logits: b fp32.
 * On failure the outputs are NaN (logits) and -1 is returned. */
int shv_cuda_encode_frames(shv_cuda_model* m, const uint8_t* rgb, size_t n, float* tokens);
int shv_cuda_classify(shv_cuda_model* m, const float* tokens, size_t b, float* logits);
int shv_cuda_predict(shv_cuda_model* m, const uint8_t* clips, size_t b, float* logits);

/* ---------------------------------------------------------------- asynchronous */
typedef void (*shv_cuda_callback)(shv_cuda_job* job, int status /* 1 done, -1 failed */, void* user);

typedef struct shv_cuda_request {
    /* frames to encode (n_frames <= max_frames) */
    const uint8_t* frames;            /* n_frames x 180x320x3 uint8 */
    size_t n_frames;
    float* tokens_out;                /* optional: n_frames x 6144 fp32 (host) */
    const int32_t* store_slots;       /* optional: n_frames device token slots (-1 = do not store) */
    /* windows to classify (n_windows <= max_windows); entry e = window*20 + frame position:
     *   window_frames[e] >= 0       -> frame window_frames[e] of this request
     *   else window_slots[e] >= 0   -> device token slot
     *   else window_tokens[e]       -> host pointer to 6144 fp32 (<= max_host_tokens such entries per request) */
    size_t n_windows;
    const int32_t* window_frames;     /* optional (NULL = all -1) */
    const int32_t* window_slots;      /* optional */
    const float* const* window_tokens;/* optional */
    float* logits_out;                /* n_windows fp32 (NaN for every window of a failed job) */
    shv_cuda_callback callback;       /* optional */
    void* user;
} shv_cuda_request;

/* Validates the request (NULL + last_error on invalid input or a stopped model), copies its index arrays, queues it.
 * Never blocks on the GPU. */
shv_cuda_job* shv_cuda_submit(shv_cuda_model* m, const shv_cuda_request* req);
int shv_cuda_job_status(const shv_cuda_job* job);   /* 0 pending, 1 done, -1 failed */
int shv_cuda_job_wait(shv_cuda_job* job);           /* blocks; 0 done, -1 failed */
const char* shv_cuda_job_error(const shv_cuda_job* job);  /* "" unless failed; valid until job_free */
/* After completion: queue_ms = submit -> batch start, run_ms = batch start -> outputs written, gpu_ms = GPU time of
 * the batch (events), batch_frames/windows = size of the coalesced batch.  Returns 1 if still pending. */
int shv_cuda_job_timing(const shv_cuda_job* job, double* queue_ms, double* run_ms, double* gpu_ms, int* batch_frames,
                        int* batch_windows);
void shv_cuda_job_free(shv_cuda_job* job);

/* Pinned host memory: frames / outputs in pinned memory are copied by DMA directly (no staging memcpy). */
void* shv_cuda_host_alloc(size_t bytes);
void shv_cuda_host_free(void* p);

#if defined(__GNUC__)
#pragma GCC visibility pop
#endif
#ifdef __cplusplus
}
#endif
#endif /* SHV_CUDA_H */
